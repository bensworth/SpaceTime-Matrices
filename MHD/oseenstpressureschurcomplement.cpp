#include "oseenstpressureschurcomplement.hpp"


namespace mfem{



//##############################################################################
//
// SPACE-TIME BLOCK PRECONDITIONING
//
//##############################################################################

// - For information on the components of the block preconditioner, see E/S/W:
//    H. Elman, D. Silvester, and A. Wathen. Finite elements and fast
//    iterative solvers: with applications in incompressible fluid dynamics.

//******************************************************************************
// Pressure block
//******************************************************************************
// These functions are used to aid in the application of the pressure part of
//  the block preconditioner (ie, approximating the inverse of pressure schur
//  complement)
// This is defined as: - XX^-1 = - D(Mp)^-1 * FFp * D(Ap)^-1, where:
//  - D(*) represents the block-diagonal matrix with (*) as blocks
//  - FFp is the space-time matrix representing time-stepping on pressure: a
//     block bi-diagonal matrix with operator Fp = Mp + dt*Wp on the main 
//     diagonal, and -Mp on the lower diagonal
//  - Mp is the pressure mass matrix
//  - Ap is the pressure "laplacian" (or its stabilised/approximated version)
//  - Wp is the pressure (convection)-diffusion operator
//
// Inversion of the pressure mass and stiffness matrices is implemented using
//  PETSc solvers. Their options can be prescribed by pre-appending the flag
//  - PSolverMass_      (for the pressure mass matrix)
//  - PSolverLaplacian_ (for the pressure stiffness matrix)
//  to the PETSc options file.
//
// If the "pressure laplacian" Ap coincides with mu*Wp then, after some
//  algebra, XX^-1 can be simplified to the block bi-diagonal:
//          ⌈ Ap^-1 + dt*mu*Mp^-1                          ⌉
// XX^-1 =  |      -Ap^-1          Ap^-1 + dt*mu*Mp^-1     |,
//          |                           -Ap^-1          \\ |
//          ⌊                                           \\ ⌋
//  which boils down to two parallel solves: one for Mp and one for Ap.
// If not, then we have:
//          ⌈ (I + dt*mu*Mp^-1*Ap)*Ap^-1                                ⌉
// XX^-1 =  |           -Ap^-1            (I + dt*mu*Mp^-1*Ap)*Ap^-1    |
//          |                                       -Ap^-1           \\ |
//          ⌊                                                        \\ ⌋
//  which boils down to two parallel solves and a matrix multiplication
//
// {NB: 
// If Ap is assembled with Dirichlet BC in outflow (as we do) and Fp is
//  assembled the same way as Ap (as we do, since it seems to give better
//  results-even though this goes against the advice in E/S/W), we need extra
//  care to apply the BC to the space-time matrix FFp, too. It's easy for the
//  main diagonal: if Mp and Ap are assembled with Dirichlet, then it's taken
//  care of automatically; but for the subdiagonal, we need to kill the
//  contribution from the Dirichlet nodes (much like we do in the assembly of
//  FFu).}
//
// {NB2: 
// The formulation above starts from the assumption that the PCD and the
//  gradient operator commute. In Elman/Silvester/Wathen, instead, they
//  propose a derivation where the *divergence* operator is considered.
// The big change is in the order of the operators: in that case, we have
//  XX^-1 = - D(Ap)^-1 * FFp * D(Mp)^-1,
//  that is, Ap and Mp are swapped. In that case, XX has the form
//          ⌈ Ap^-1 + dt*Ap^-1*Wp*Mp^-1                              ⌉
// XX^-1 =  |           -Ap^-1          Ap^-1 + dt*Ap^-1*Wp*Mp^-1    |
//          |                                     -Ap^-1          \\ |
//          ⌊                                                     \\ ⌋
//  and requires one extra application of Ap^-1 in general. On top of this,
//  it seems like it provides worse results, hence we'll go for the one above
//  (but this second implementation is provided below nonetheless)}

OseenSTPressureSchurComplement::OseenSTPressureSchurComplement( const MPI_Comm& comm, double dt, double mu,
                                                                const SparseMatrix* Ap, const SparseMatrix* Mp, const SparseMatrix* Wp,
                                                                const Array<int>& essQhTDOF, int verbose ):
  _comm(comm), _dt(dt), _mu(mu), _Ap(NULL), _Mp(NULL), _WpEqualsAp(false), _Asolve(NULL), _Msolve(NULL), _essQhTDOF(essQhTDOF), _verbose(verbose){

  if( Ap != NULL ) SetAp(Ap);
  if( Mp != NULL ) SetMp(Ap);
  if( Wp != NULL ) SetWp(Wp, false);

  // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
  //  to actually make use of the IG or not
  // TODO: delete this. Never use initial guess in preconditioners!
  iterative_mode = true;

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



OseenSTPressureSchurComplement::~OseenSTPressureSchurComplement(){
  delete _Ap;
  delete _Mp;
  delete _Asolve;
  delete _Msolve;
}



// initialise info on pressure 'laplacian'
void OseenSTPressureSchurComplement::SetAp( const SparseMatrix* Ap ){
  delete _Ap;
  _Ap = new PetscParMatrix( Ap );

  // CholeskySolver C( Ap );

  // int uga = 0;
  // std::cin>>uga;


  // using namespace petsc;
  // Mat myA = Mat(*_Ap), myC;
  // PetscErrorCode ierr;
  // IS perm;
  // MatFactorInfo info;

  // MatGetFactor( myA, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &myC );
  // // MatCholeskyFactorSymbolic( myC, myA, perm, &info);
  // MatCholeskyFactorNumeric(  myC, myA, &info);

  // PetscObjectSetName((PetscObject)myC,"ApChol");
  // PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
  // MatView(myC,PETSC_VIEWER_STDOUT_WORLD);
  // PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);

  // std::cout<<"Exported matrix"<<std::endl;
  // int uga;
  // std::cin>>uga;

  // Flag non-trivial null space (constant funcs) if there are no essnodes
  // - NB: this option can be passed at runtime with -ksp_constant_null_space TRUE
  // if( _essQhTDOF.Size() == 0 ){
  //   if( _myRank == 0 ){
  //     // std::cout<<"Assuming that pressure 'laplacian' has non-trivial kernel (constant functions)"<<std::endl;
  //     std::cout<<"Warning: the pressure 'laplacian' has non-trivial kernel (constant functions)."<<std::endl
  //              <<"         Make sure to flag that in the petsc options prescribing:"<<std::endl
  //              <<"         -for iterative solver: -PSolverLaplacian_ksp_constant_null_space TRUE"<<std::endl
  //              <<"         -for direct solver: -PSolverLaplacian_pc_factor_shift_type NONZERO"<<std::endl
  //              <<"                         and -PSolverLaplacian_pc_factor_shift_amount 1e-10"<<std::endl
  //              <<"                         (this will hopefully save us from 0 pivots in the singular mat)"<<std::endl;
  //     // TODO: or maybe just fix one unknown?
  //   }
  //   // TODO: for some reason, the following causes memory leak
  //   // // extract the underlying petsc object
  //   // PetscErrorCode ierr;
  //   // Mat petscA = Mat( *_Ap );
  //   // // initialise null space
  //   // MatNullSpace nsp = NULL;
  //   // MatNullSpaceCreate( PETSC_COMM_SELF, PETSC_TRUE, 0, NULL, &nsp); CHKERRV(ierr);
  //   // // // attach null space to matrix
  //   // MatSetNullSpace( petscA, nsp ); CHKERRV(ierr);
  //   // MatNullSpaceDestroy( &nsp ); CHKERRV(ierr);      // hopefully all info is stored
  // }

  height = Ap->Height();
  width  = Ap->Width();
  SetApSolve();
}



// initialise solver for pressure 'laplacian'
void OseenSTPressureSchurComplement::SetApSolve(){
  delete _Asolve;

  _Asolve = new PetscLinearSolver( *_Ap, "PSolverLaplacian_" );
  

  // PetscErrorCode ierr;
  // PetscBool set;
  // char optName[PETSC_MAX_PATH_LEN];

  // // TODO: delete this. Never use initial guess in preconditioners!
  // _Asolve->iterative_mode = true;  // trigger iterative mode...
  // ierr = PetscOptionsGetString( NULL ,"PSolverLaplacian_", "-ksp_type", optName, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
  // if( !strcmp( optName, "preonly" ) ){
  //   char optName1[PETSC_MAX_PATH_LEN];
  //   ierr = PetscOptionsGetString( NULL ,"PSolverLaplacian_", "-pc_type", optName1, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
  //   if(!( strcmp( optName1, "ilu" ) ) || !( strcmp( optName1, "lu" ) ) ){
  //     _Asolve->iterative_mode = false;  // ...unless you're using ilu or lu
  //   }
  // }
  
  // if ( _verbose && _Asolve->iterative_mode && _myRank==0 ){
  //   std::cout<<"Selected iterative solver for pressure 'laplacian'"<<std::endl;
  // }

}



// initialise info on pressure mass matrix
void OseenSTPressureSchurComplement::SetMp( const SparseMatrix* Mp ){
  delete _Mp;
  _Mp = new PetscParMatrix( Mp );

  height = Mp->Height();
  width  = Mp->Width();
  SetMpSolve();
}

// initialise info on pressure time-stepping operator
void OseenSTPressureSchurComplement::SetWp( const SparseMatrix* Wp, bool WpEqualsAp ){
  _Wp = *Wp;
  _WpEqualsAp = WpEqualsAp;

  height = Wp->Height();
  width  = Wp->Width();
}



// initialise solver for pressure mass matrix
void OseenSTPressureSchurComplement::SetMpSolve(){
  delete _Msolve;

  _Msolve = new PetscLinearSolver( *_Mp, "PSolverMass_" );
  
  // // TODO: delete this. Never use initial guess in preconditioners!
  // PetscErrorCode ierr;
  // PetscBool set;
  // char optName[PETSC_MAX_PATH_LEN];

  // _Msolve->iterative_mode = true;  // trigger iterative mode...
  // ierr = PetscOptionsGetString( NULL ,"PSolverMass_", "-ksp_type", optName, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
  // if( !strcmp( optName, "preonly" ) ){
  //   char optName1[PETSC_MAX_PATH_LEN];
  //   ierr = PetscOptionsGetString( NULL ,"PSolverMass_", "-pc_type", optName1, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
  //   if(!( strcmp( optName1, "ilu" ) ) || !( strcmp( optName1, "lu" ) ) ){
  //     _Msolve->iterative_mode = false;  // ...unless you're using ilu or lu
  //   }
  // }


  // if ( _verbose && _Msolve->iterative_mode && _myRank==0 ){
  //   std::cout<<"Selected iterative solver for pressure mass matrix"<<std::endl;
  // }

}



// Define multiplication by preconditioner - the most relevant function here
// This implementation refers to XX assembled starting from the commutation of gradient and PCD operators:
// XX^-1 = - D(Mp)^-1 * FFp * D(Ap)^-1
void OseenSTPressureSchurComplement::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT( _Asolve!=NULL, "Solver for press 'laplacian' not initialised" );
  MFEM_ASSERT( _Msolve!=NULL, "Solver for press mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  // Initialise
  const int     lclSize = x.Size();
  const double* lclData = x.GetData();

  if ( _verbose>20 ){
    if ( _myRank==0 ){
      std::cout<<"Applying pressure block preconditioner\n";
    }
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", rhs for p: "; x.Print(mfem::out, x.Size());
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", IG  for p: "; y.Print(mfem::out, y.Size());
    }
  }

  // TODO: do I really need to copy this? Isn't there a way to force lclx to point at lclData and still be const?
  Vector lclx( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    lclx(i) = lclData[i];
  }

  Vector invAxMine( lclSize ), lcly( lclSize ), invAxPrev( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    invAxMine(i) = y(i);
    lcly(i)      = y(i);
  }


  // Have each processor solve for the "laplacian"
  // _Asolve->Mult( lclx, invAxMine );
  // Have each processor solve for the "laplacian"
  if ( _Asolve->Height() == this->height ){
    _Asolve->Mult( lclx, invAxMine );
  }else{
    // I'm considering an "augmented" version of the laplacian, which takes also the lagrangian multiplies for int(p)=0;
    Vector xAug( lclSize+1 );
    Vector invAxMineAug( lclSize+1 );
    for ( int i = 0; i < lclSize; ++i ){
      xAug(i) = lclx(i);
      invAxMineAug(i) = invAxMine(i);
    }
    xAug(lclSize) = 0.;
    invAxMineAug(lclSize) = 0.;
    _Asolve->Mult( xAug, invAxMineAug );
    for ( int i = 0; i < lclSize; ++i ){
      invAxMine(i) = invAxMineAug(i);
    }
  }



  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure stiffness matrix\n";
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after inverting stiffness matrix: ";
      invAxMine.Print(mfem::out, invAxMine.Size());
    }
  }

  // Eventually include contribution from pressure (convection)-diffusion operator
  if ( !_WpEqualsAp ){
    // - if Wp is not the same as Ap, then Mp^-1 will have to be applied to Wp*Ap^-1 * x
    // - NB: make sure Wp is defined with the viscosity coefficient included in it!
    // - NB: if imposing some Dirichlet BC on the PCD operator, we need to kill their contribution here
    Vector temp = invAxMine;
    temp.SetSubVector( _essQhTDOF, 0.0 );
    _Wp.Mult( temp, lclx );

  }else{
    // - otherwise, simply apply Mp^-1 to x, and then multiply by mu (or rather, the other way around)
    lclx *= _mu;
  }
  // either case, kill dirichlet contributions there
  lclx.SetSubVector( _essQhTDOF, 0.0 );

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" included contribution from pressure (convection)-diffusion operator"<<std::endl;
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after spatial contribution: ";
      lclx.Print(mfem::out, lclx.Size());
    }
    MPI_Barrier(_comm);
  }  



// #ifndef MULT_BY_DT
//   // - divide "laplacian" solve by dt if you didn't rescale system
//   invAxMine *= (1./_dt);
//   // -- careful not to dirty dirichlet nodes (multiply and divide is ugly as hell, but oh well...)
//   for ( int i = 0; i < _essQhTDOF.Size(); ++i ){
//     invAxMine(_essQhTDOF[i]) *= _dt;
//   }
// #endif

  // Send this partial result to the following processor  
  MPI_Request reqSend, reqRecv;

  if( _myRank < _numProcs && _numProcs>1 ){
    MPI_Isend( invAxMine.GetData(), lclSize, MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
  }
  if( _myRank > 0 ){
    MPI_Irecv( invAxPrev.GetData(), lclSize, MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
  }


  // Have each processor solve for the mass matrix
  _Msolve->Mult( lclx, lcly );
  lcly.SetSubVector( _essQhTDOF, 0.0 ); // eventually kill contribution from Dirichlet nodes on the spatial part of the operator

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure mass matrix\n";
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after inverting mass matrix: ";
      lcly.Print(mfem::out, lcly.Size());
    }
    MPI_Barrier(_comm);
  }


  // Combine all partial results together locally (once received required data, if necessary)
// #ifdef MULT_BY_DT
  lcly *= _dt;    //eventually rescale mass solve by dt
// #endif

  // - if you want to ignore the "time-stepping" structure in the preconditioner (that is,
  //    the contribution from the Mp/dt terms, just comment out the next few lines
  lcly += invAxMine;
  if( _myRank > 0 ){
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );

    // - kill contributions from Dirichlet BC
    invAxPrev.SetSubVector( _essQhTDOF, 0.0 );
    
    lcly -= invAxPrev;
  }


  // Assemble global vector
  y = lcly;
  // - remember to flip sign! Notice the minus in front of XX^-1
  y.Neg();


  // TODO:
  // This shouldn't be necessary: the dirichlet nodes should already be equal to invAxMine
  //  Actually, not true: I'm flipping the sign on the dirichlet nodes with y.Neg()!
  // THIS IS A NEW ADDITION:
  // - only consider action of Ap^-1 on Dirichlet BC
  for ( int i = 0; i < _essQhTDOF.Size(); ++i ){
    y(_essQhTDOF[i]) = invAxMine(_essQhTDOF[i]);
  }



  if(_verbose>100){
    std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result for p: ";
    y.Print(mfem::out, y.Size());
  }

  if(_myRank==0 && _verbose>200){
    int uga;
    std::cin>>uga;
  }


  // lest we risk destroying invAxMine before it's been sent (probably unnecessary)
  // if( _myRank < _numProcs ){
  //   MPI_Wait( &reqSend, MPI_STATUS_IGNORE ); // this triggers a memory error on reqSend, for a reason...
  // }
  MPI_Barrier( _comm );                         // ...but the barrier should do the same trick, and this seems to work

}





/* This implementation refers to XX assembled starting from the commutation of divergence and PCD operators
// XX^-1 = - D(Ap)^-1 * FFp * D(Mp)^-1
// Define multiplication by preconditioner - the most relevant function here
void OseenSTPressureSchurComplement::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT( _Asolve!=NULL, "Solver for press 'laplacian' not initialised" );
  MFEM_ASSERT( _Msolve!=NULL, "Solver for press mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  // Initialise
  const int     lclSize = x.Size();
  const double* lclData = x.GetData();

  if ( _verbose ){
    if ( _myRank==0 ){
      std::cout<<"Applying pressure block preconditioner"<<std::endl;
    }
    MPI_Barrier(_comm);
    std::cout<<"Inside P-block: Rank: "<<_myRank<< ", rhs for p: "; x.Print(std::cout, x.Size());
    std::cout<<"Inside P-block: Rank: "<<_myRank<< ", IG  for p: "; y.Print(std::cout, y.Size());
  }

  // TODO: do I really need to copy this? Isn't there a way to force lclx to point at lclData and still be const?
  Vector lclx( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    lclx.GetData()[i] = lclData[i];
  }

  Vector invAxMine( lclSize ), lcly( lclSize ), invAxPrev( lclSize ), temp( lclSize );
  temp = 0.;
  for ( int i = 0; i < lclSize; ++i ){
    invAxMine.GetData()[i] = y.GetData()[i];
    lcly.GetData()[i]      = y.GetData()[i];
  }

  // Have each processor solve for the "laplacian"
  // // - kill every contribution on outflow boundary
  // invAxMine.SetSubVector( _essQhTDOF, 0.0 );  // before, to improve initial guess
  // lclx.SetSubVector(      _essQhTDOF, 0.0 );  // to the rhs, to make sure it won't affect solution (not even afterwards)
  // // TODO: if the system is singular, set the first unknown to zero to stabilise it
  // //       NB: this must match the definition of Ap!!
  // if( _essQhTDOF.Size() == 0 ){
  //   lclx.GetData()[0] = 0.;
  // }
  _Asolve->Mult( lclx, invAxMine );
  // invAxMine.SetSubVector( _essQhTDOF, 0.0 );  // and even afterwards, to really make sure it's 0

  if (_verbose ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure 'laplacian'"<<std::endl;
    MPI_Barrier(_comm);
    std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after inverting 'laplacian': "; invAxMine.Print(std::cout, invAxMine.Size());
  }

#ifndef MULT_BY_DT
  invAxMine *= (1./_dt);   //divide "laplacian" solve by dt if you didn't rescale system
#endif

  // Send this partial result to the following processor  
  MPI_Request reqSend, reqRecv;

  if( _myRank < _numProcs ){
    MPI_Isend( invAxMine.GetData(), lclSize, MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
  }
  if( _myRank > 0 ){
    MPI_Irecv( invAxPrev.GetData(), lclSize, MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
  }


  // Have each processor solve for the mass matrix
  _Msolve->Mult( lclx, lcly );

  if (_verbose ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure mass matrix"<<std::endl;
    MPI_Barrier(_comm);
  }
  
  // Eventually include contribution from pressure (convection)-diffusion operator
  if ( !_WpEqualsAp ){
    // if Wp is not the same as Ap, then Ap^-1 will have to be applied to Wp*Mp^-1 * x
    // - NB: make sure Wp is defined with the viscosity coefficient included in it!
    
    // - apply Wp
    _Wp.Mult( lcly, temp );
    // - invert Ap again
    _Asolve->Mult( temp, lcly );

    if (_verbose ){
      std::cout<<"Rank "<<_myRank<<" included contribution from pressure (convection)-diffusion operator"<<std::endl;
      MPI_Barrier(_comm);
    }  
  }else{
    // - otherwise, simply apply Mp^-1 to x, and then multiply by mu (or rather, the other way around)
    lcly *= _mu;
  }


  // Combine all partial results together locally (once received required data, if necessary)
#ifdef MULT_BY_DT
  lcly *= _dt;    //eventually rescale mass solve by dt
#endif
  lcly += invAxMine;

  // - if you want to ignore the "time-stepping" structure in the preconditioner (that is,
  //    the contribution from the subdiagonal Mp/dt terms, just comment out the next few lines
  if( _myRank > 0 ){
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
    lcly -= invAxPrev;
  }

  // Assemble global vector
  for ( int i = 0; i < lclSize; ++i ){
    // remember to flip sign! Notice the minus in front of XX^-1
    y.GetData()[i] = - lcly.GetData()[i];
  }

  if(_verbose){
    std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result for p: "; y.Print(std::cout, y.Size());
  }


  // lest we risk destroying invAxMine before it's been sent (probably unnecessary)
  // if( _myRank < _numProcs ){
  //   MPI_Wait( &reqSend, MPI_STATUS_IGNORE ); // this triggers a memory error on reqSend, for a reason...
  // }
  MPI_Barrier( _comm );                         // ...but the barrier should do the same trick, and this seems to work

}
*/


} // namespace mfem