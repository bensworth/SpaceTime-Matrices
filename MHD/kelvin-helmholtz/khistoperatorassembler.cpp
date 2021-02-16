#include "khistoperatorassembler.hpp"
#include "vectorconvectionintegrator.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include "spacetimesolver.hpp"
#include "pararealsolver.hpp"

#include <mpi.h>
#include <string>
#include <cstring>
#include <iostream>
#include "HYPRE.h"
#include "petsc.h"
#include "mfem.hpp"
#include <experimental/filesystem>

using namespace mfem;

// Seems like multiplying every operator by dt gives slightly better results.
#define MULT_BY_DT

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
StokesSTPreconditioner::StokesSTPreconditioner( const MPI_Comm& comm, double dt, double mu,
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



StokesSTPreconditioner::~StokesSTPreconditioner(){
  delete _Ap;
  delete _Mp;
  delete _Asolve;
  delete _Msolve;
}



// initialise info on pressure 'laplacian'
void StokesSTPreconditioner::SetAp( const SparseMatrix* Ap ){
  delete _Ap;
  _Ap = new PetscParMatrix( Ap );


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
void StokesSTPreconditioner::SetApSolve(){
  delete _Asolve;

  _Asolve = new PetscLinearSolver( *_Ap, "PSolverLaplacian_" );
  

  PetscErrorCode ierr;
  PetscBool set;
  char optName[PETSC_MAX_PATH_LEN];

  // TODO: delete this. Never use initial guess in preconditioners!
  _Asolve->iterative_mode = true;  // trigger iterative mode...
  ierr = PetscOptionsGetString( NULL ,"PSolverLaplacian_", "-ksp_type", optName, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
  if( !strcmp( optName, "preonly" ) ){
    char optName1[PETSC_MAX_PATH_LEN];
    ierr = PetscOptionsGetString( NULL ,"PSolverLaplacian_", "-pc_type", optName1, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
    if(!( strcmp( optName1, "ilu" ) ) || !( strcmp( optName1, "lu" ) ) ){
      _Asolve->iterative_mode = false;  // ...unless you're using ilu or lu
    }
  }
  
  if ( _verbose && _Asolve->iterative_mode && _myRank==0 ){
    std::cout<<"Selected iterative solver for pressure 'laplacian'"<<std::endl;
  }

}



// initialise info on pressure mass matrix
void StokesSTPreconditioner::SetMp( const SparseMatrix* Mp ){
  delete _Mp;
  _Mp = new PetscParMatrix( Mp );

  height = Mp->Height();
  width  = Mp->Width();
  SetMpSolve();
}

// initialise info on pressure time-stepping operator
void StokesSTPreconditioner::SetWp( const SparseMatrix* Wp, bool WpEqualsAp ){
  _Wp.MakeRef( *Wp );
  _WpEqualsAp = WpEqualsAp;

  height = Wp->Height();
  width  = Wp->Width();
}



// initialise solver for pressure mass matrix
void StokesSTPreconditioner::SetMpSolve(){
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
void StokesSTPreconditioner::Mult( const Vector &x, Vector &y ) const{
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
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", rhs for p: "; x.Print(std::cout, x.Size());
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", IG  for p: "; y.Print(std::cout, y.Size());
    }
  }

  // TODO: do I really need to copy this? Isn't there a way to force lclx to point at lclData and still be const?
  Vector lclx( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    lclx.GetData()[i] = lclData[i];
  }

  Vector invAxMine( lclSize ), lcly( lclSize ), invAxPrev( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    invAxMine.GetData()[i] = y.GetData()[i];
    lcly.GetData()[i]      = y.GetData()[i];
  }


  // Have each processor solve for the "laplacian"
  _Asolve->Mult( lclx, invAxMine );

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure stiffness matrix\n";
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after inverting stiffness matrix: ";
      invAxMine.Print(std::cout, invAxMine.Size());
    }
  }

  // Eventually include contribution from pressure (convection)-diffusion operator
  if ( !_WpEqualsAp ){
    // - if Wp is not the same as Ap, then Mp^-1 will have to be applied to Wp*Ap^-1 * x
    // - NB: make sure Wp is defined with the viscosity coefficient included in it!
    _Wp.Mult( invAxMine, lclx );

    if (_verbose>50 ){
      std::cout<<"Rank "<<_myRank<<" included contribution from pressure (convection)-diffusion operator"<<std::endl;
      MPI_Barrier(_comm);
    }  
  }else{
    // - otherwise, simply apply Mp^-1 to x, and then multiply by mu (or rather, the other way around)
    lclx *= _mu;
  }


#ifndef MULT_BY_DT
  invAxMine *= (1./_dt);   //divide "laplacian" solve by dt if you didn't rescale system
#endif

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

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure mass matrix\n";
    MPI_Barrier(_comm);
  }


  // Combine all partial results together locally (once received required data, if necessary)
#ifdef MULT_BY_DT
  lcly *= _dt;    //eventually rescale mass solve by dt
#endif

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
  for ( int i = 0; i < lclSize; ++i ){
    // remember to flip sign! Notice the minus in front of XX^-1
    y.GetData()[i] = - lcly.GetData()[i];
  }

  if(_verbose>100){
    std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result for p: ";
    y.Print(std::cout, y.Size());
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
void StokesSTPreconditioner::Mult( const Vector &x, Vector &y ) const{
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

















//******************************************************************************
// Space-time block assembly
//******************************************************************************
// These functions are useful for assembling all the necessary space-time operators

// constructor (uses analytical function for advection field)
KHISTOperatorAssembler::KHISTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
                                                const int refLvl, const int ordU, const int ordP,
                                                const double dt, const double mu, const double Pe,
                                                void(  *f)(const Vector &, double, Vector &),
                                                double(*g)(const Vector &, double ),
                                                void(  *n)(const Vector &, double, Vector &),
                                                void(  *w)(const Vector &, double, Vector &),
		                         							      void(  *u)(const Vector &, double, Vector &),
		                         							      double(*p)(const Vector &, double ),
                                                int verbose ):
	_comm(comm), _dt(dt), _mu(mu), _Pe(Pe), _fFunc(f), _gFunc(g), _nFunc(n), _wFunc(w), _wFuncCoeff(), _uFunc(u), _pFunc(p), _ordU(ordU), _ordP(ordP),
  _MuAssembled(false), _FuAssembled(false), _MpAssembled(false), _ApAssembled(false), _WpAssembled(false), _BAssembled(false),
  _FFinvPrec(NULL), _FFAssembled(false), _BBAssembled(false), _pSAssembled(false), _FFinvAssembled(false),
  _verbose(verbose){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

	// For each processor:
	//- generate mesh
	_mesh = new Mesh( meshName.c_str(), 1, 1 );
  _dim = _mesh->Dimension();
  
  if ( _dim != 2 && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: KHI only works for 2D domains\n";
  }

  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  // - initialise FE info
  _WhFEColl = new H1_FECollection( ordW, _dim );
  _PhFEColl = new H1_FECollection( ordP, _dim );
  _VhFEColl = new H1_FECollection( ordV, _dim );
  _WhFESpace = new FiniteElementSpace( _mesh, _WhFEColl );
  _PhFESpace = new FiniteElementSpace( _mesh, _PhFEColl );
  _VhFESpace = new FiniteElementSpace( _mesh, _VhFEColl, _dim );


  if ( _mesh->bdr_attributes.Size() > 0 ) {
    Array<int> essBdrW( _mesh->bdr_attributes.Max() ), essBdrP( _mesh->bdr_attributes.Max() ), essBdrV( _mesh->bdr_attributes.Max() );
    essBdrW = 0; essBdrP = 0; essBdrV = 0;
    for ( int i = 0; i < _mesh->bdr_attributes.Max(); ++i ){
      if( _mesh->bdr_attributes[i] == 1 )
        essBdrW[i] = 1;
      if( _mesh->bdr_attributes[i] == 2 )
        essBdrP[i] = 1;
      if( _mesh->bdr_attributes[i] == 3 )
        essBdrV[i] = 1;
    }

    _WhFESpace->GetEssentialTrueDofs( essBdrW, _essWhTDOF );
    _PhFESpace->GetEssentialTrueDofs( essBdrP, _essPhTDOF );
    _VhFESpace->GetEssentialTrueDofs( essBdrV, _essVhTDOF );
  }


  if (_myRank == 0 ){
    std::cout << "***********************************************************\n";
    std::cout << "dim(Wh) = " << _WhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Vh) = " << _VhFESpace->GetTrueVSize() << "\n";
    std::cout << "***********************************************************\n";
  }

}








//-----------------------------------------------------------------------------
// Assemble operators for single time-steps
//-----------------------------------------------------------------------------
// Assemble operator on main diagonal of space-time matrix for w block:
//  Fw = M + mu*dt K + dt*W(v)
void KHISTOperatorAssembler::AssembleFw( ){
  if( _FwAssembled ){
    return;
  }

  BilinearForm fwVarf(_WhFESpace);

  VectorCoefficient* vCoeff = NULL;   // need to define them here otherwise they go out of scope for VectorConvectionIntegrator
  GridFunction vGridFun( _WhFESpace );
  vGridFun = _vFuncCoeff;
  if ( _vFunc == NULL ){
    vCoeff = new VectorGridFunctionCoefficient( &vGridFun );
  }else{
    vCoeff = new VectorFunctionCoefficient( _dim, _vFunc );
    vCoeff->SetTime( _dt*(_myRank+1) );
  }

#ifdef MULT_BY_DT
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
  fwVarf.AddDomainIntegrator(new MassIntegrator( one ));
  fwVarf.AddDomainIntegrator(new DiffusionIntegrator( muDt ));
  fwVarf.AddDomainIntegrator(new ConvectionIntegrator( vCoeff, _dt ));
#else
  ConstantCoefficient mu( _mu );
  ConstantCoefficient dtinv( 1./_dt );
  fwVarf.AddDomainIntegrator(new MassIntegrator( dtinv ));
  fwVarf.AddDomainIntegrator(new DiffusionIntegrator( mu ));
  fwVarf.AddDomainIntegrator(new ConvectionIntegrator( vCoeff, 1.0 ));
#endif


  fwVarf.Assemble();
  fwVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  _Fw = fwVarf.SpMat();
  _Fw.SetGraphOwner(true);
  _Fw.SetDataOwner(true);
  fwVarf.LoseMat();

  // still need BC!
  // _FwAssembled = true;

  delete vCoeff;


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial operator Fw assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}








// Assemble operator on subdiagonal of space-time matrix for w block:
//  Mw = -M
void KHISTOperatorAssembler::AssembleMwVarf( ){
  if( _MwAssembled ){
    return;
  }

  BilinearForm mwVarf(_WhFESpace);
#ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  mwVarf.AddDomainIntegrator(new MassIntegrator( mone ));
#else
  ConstantCoefficient mdtinv( -1./_dt );
  mwVarf.AddDomainIntegrator(new MassIntegrator( mdtinv ));
#endif
  mwVarf.Assemble();
  mwVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Mw = mwVarf.SpMat();
  _Mw.SetGraphOwner(true);
  _Mw.SetDataOwner(true);
  mwVarf.LoseMat();

  // still need BC!
  // _MwAssembled = true;




  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass-matrix for w (negative) Mw assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}




// Assemble mass matrix for v block:
//  Mv = dt*u*v
void KHISTOperatorAssembler::AssembleMvVarf( ){
  if( _MvAssembled ){
    return;
  }

  BilinearForm mvVarf(_VhFESpace);
#ifdef MULT_BY_DT
  ConstantCoefficient one( 1.0 );
  mvVarf.AddDomainIntegrator(new VectorMassIntegrator( one ));
#else
  ConstantCoefficient Dt( _dt );
  mvVarf.AddDomainIntegrator(new VectorMassIntegrator( Dt ));
#endif
  mvVarf.Assemble();
  mvVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Mv = mvVarf.SpMat();
  _Mv.SetGraphOwner(true);
  _Mv.SetDataOwner(true);
  mvVarf.LoseMat();

  // still need BC!
  // _MvAssembled = true;



  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass-matrix for v Mv assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}






// Assemble laplacian for potential block:
//  Ap = dt * div(grad(\phi))
void KHISTOperatorAssembler::AssembleAp( ){

  if( _ApAssembled ){
    return;
  }

  BilinearForm aVarf(_PhFESpace);
#ifdef MULT_BY_DT  
  ConstantCoefficient Dt( _dt );
  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( Dt ));
#else
  ConstantCoefficient one( 1.0 );
  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
#endif

  // Impose homogeneous dirichlet BC by simply removing corresponding equations?
  TODO
  aVarf.Assemble();
  aVarf.Finalize();
  
  aVarf.FormSystemMatrix( _essPhTDOF, _Ap );
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf.LoseMat();


  // still need BC!
  // _ApAssembled = true;


  // if ( _verbose>50 && _myRank == 0 ){
  //   std::ofstream myfile;
  //   std::string myfilename = "./results/out_final_Ap.dat";
  //   myfile.open( myfilename );
  //   _Ap.PrintMatlab(myfile);
  //   myfile.close( );
  // }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Laplacian operator Ap for potential assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}






// Assemble divergence operator coupling v to w:
//  B(w) = dt * div(w * u), v
void KHISTOperatorAssembler::AssembleBVarf( ){

  if( _BAssembled ){
    return;
  }

  MixedBilinearForm bVarf( _VhFESpace, _WhFESpace );

  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for VectorConvectionIntegrator
  GridFunction wGridFun( _WhFESpace );
  wGridFun = _wFuncCoeff;
  if ( _wFunc == NULL ){
    wCoeff = new GridFunctionCoefficient( &wGridFun );
  }else{
    wCoeff = new FunctionCoefficient( _wFunc );
    wCoeff->SetTime( _dt*(_myRank+1) );
  }

  // TODO: how to include multiplication by dt?

#ifdef MULT_BY_DT  
  ConstantCoefficient Dt( _dt );
  bVarf.AddDomainIntegrator(new VectorDivergenceIntegrator(Dt*wCoeff) );
#else
  bVarf.AddDomainIntegrator(new VectorDivergenceIntegrator(wCoeff) );
#endif

  bVarf.Assemble();
  bVarf.Finalize();

  bVarf.SpMat();
  _B.SetGraphOwner(true);
  _B.SetDataOwner(true);
  bVarf.LoseMat();

  delete wCoeff;

  // still need BC!
  // _BAssembled = true;


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Divergence operator B(w) assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}









// Assemble mass matrix coupling w to potential:
//  Mwp = dt * w * \phi
void KHISTOperatorAssembler::AssembleMwp( ){

  if( _MwpAssembled ){
    return;
  }

  MixedBilinearForm mVarf( _WhFESpace, _PhFESpace );

#ifdef MULT_BY_DT  
  ConstantCoefficient Dt( _dt );
  mVarf.AddDomainIntegrator(new MixedScalarMassIntegrator( Dt ));
#else
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new MixedScalarMassIntegrator( one ));
#endif

  mVarf.Assemble();
  mVarf.Finalize();


  // - once the matrix is generated, we can get rid of the operator
  _Mwp = mVarf.SpMat();
  _Mwp.SetGraphOwner(true);
  _Mwp.SetDataOwner(true);
  mVarf.LoseMat();

  // still need BC!
  // _MwpAssembled = true;


  // if ( _verbose>50 && _myRank == 0 ){
  //   std::ofstream myfile;
  //   std::string myfilename = "./results/out_final_Mwp.dat";
  //   myfile.open( myfilename );
  //   _Mwp.PrintMatlab(myfile);
  //   myfile.close( );
  // }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mixed w-phi mass matrix Mwp assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}






// Assemble scalar curl operator coupling potential to v:
//  C = dt * curl(k*phi) v = dt * (v, J grad(phi) ), with J = [0,1;-1,0]
//  where v is trial and phi is test function
void KHISTOperatorAssembler::AssembleC( ){

  if( _CAssembled ){
    return;
  }

  MixedBilinearForm cVarf( _PhFESpace, _VhFESpace );

  // ok, this is tricky. MixedVectorWeakDivergenceIntegrator assembles
  // a(v,phi) = - (Q v, grad(phi))
  // so in order to assemble (v, J grad(phi)), I need Q = - J^T = J :)
  DenseMatrix mJT(2);
#ifdef MULT_BY_DT
  mJT.GetData()[0] = 0.;
  mJT.GetData()[1] =  _dt;
  mJT.GetData()[2] = -_dt;
  mJT.GetData()[3] = 0.;
#else
  mJT.GetData()[0] = 0.;
  mJT.GetData()[1] =  1.;
  mJT.GetData()[2] = -1.;
  mJT.GetData()[3] = 0.;
#endif
  MatrixConstantCoefficient Q( mJT );
  cVarf.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator ( Q ));

  cVarf.Assemble();
  cVarf.Finalize();


  // - once the matrix is generated, we can get rid of the operator
  _C = cVarf.SpMat();
  _C.SetGraphOwner(true);
  _C.SetDataOwner(true);
  cVarf.LoseMat();

  // still need BC!
  // _CAssembled = true;


  // if ( _verbose>50 && _myRank == 0 ){
  //   std::ofstream myfile;
  //   std::string myfilename = "./results/out_final_C.dat";
  //   myfile.open( myfilename );
  //   _C.PrintMatlab(myfile);
  //   myfile.close( );
  // }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Scalar curl operator C assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}














//-----------------------------------------------------------------------------
// Assemble space-time operators
//-----------------------------------------------------------------------------
// Assemble space-time matrix FF (w block)
void KHISTOperatorAssembler::AssembleFF( ){ 
  
  if( _FFAssembled ){
    return;
  }

  if( !( _FwAssembled && _MwAssembled ) && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble Mw and Fw before assembling FF\n";
    return;
  }


  // Create FF block ********************************************************
  // Initialize HYPRE matrix
  // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
  //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
  // - get info on matrix structure
  const int blockSizeFF = _Fw.NumRows();
 
  Array<int> nnzPerRowD( blockSizeFF );   // num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<int> nnzPerRowO( blockSizeFF );   // ..and in off-diagonal block
  const int  *offIdxsD = _Fw.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  const int  *offIdxsO = _Mw.GetI();
  for ( int i = 0; i < blockSizeFF; ++i ){
    nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
    if ( _myRank > 0 ){
      nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
    }else{
      nnzPerRowO[i] = 0;  // first block only has elements on block-diag
    }
  }


  // - initialise matrix
  HYPRE_IJMatrixCreate( _comm, blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1,
                               blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1, &_FF );
  HYPRE_IJMatrixSetObjectType( _FF, HYPRE_PARCSR );
  HYPRE_IJMatrixSetDiagOffdSizes( _FF, nnzPerRowD.GetData(), nnzPerRowO.GetData() );    // this gives issues :/
  HYPRE_IJMatrixInitialize( _FF );


  // - fill it with matrices assembled above
  // -- diagonal block
  Array<int> rowsGlbIdxD( blockSizeFF );
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxD[i] = i + blockSizeFF*_myRank;
  }
  Array<int> colsGlbIdxD( _Fw.NumNonZeroElems() );
  for ( int i=0; i<_Fw.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = _Fw.GetJ()[i] + blockSizeFF*_myRank;
  }
  HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowD.GetData(),
                           rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), _Fw.GetData() );     // setvalues *copies* the data

  // -- off-diagonal block
  Array<int> rowsGlbIdxO( blockSizeFF );      // TODO: just use rowsGlbIdx once for both matrices?
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxO[i] = i + blockSizeFF*_myRank;
  }
  if ( _myRank > 0 ){
    Array<int> colsGlbIdxO( _Mw.NumNonZeroElems() );
    for ( int i=0; i<_Mw.NumNonZeroElems(); i++ ) {
      colsGlbIdxO[i] = _Mw.GetJ()[i] + blockSizeFF*(_myRank-1);
    }
    HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowO.GetData(),
                             rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), _Mw.GetData() );
  }


  // - assemble
  HYPRE_IJMatrixAssemble( _FF );
  _FFAssembled = true;

  // - convert to a MFEM operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( _FF, (void **) &FFref);
  _FFF = new HypreParMatrix( FFref, false ); //"false" doesn't take ownership of data


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time velocity operator FF assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}





// code duplication at its worst: assemble a bunch of block-diagonal space-time matrices
inline void KHISTOperatorAssembler::AssembleAA( ){ 
  AssembleSTBlockDiagonal( _Ap, _AA, "AA", _ApAssembled, _AAAssembled );
}
inline void KHISTOperatorAssembler::AssembleMMv( ){ 
  AssembleSTBlockDiagonal( _Mv, _MMv, "MMv", _MvAssembled, _MMvAssembled );
}
inline void KHISTOperatorAssembler::AssembleBB( ){ 
  AssembleSTBlockDiagonal( _B, _BB, "BB", _BAssembled, _BBAssembled );
}
inline void KHISTOperatorAssembler::AssembleCC( ){ 
  AssembleSTBlockDiagonal( _C, _CC, "CC", _CAssembled, _CCAssembled );
}
inline void KHISTOperatorAssembler::AssembleMMwp( ){ 
  AssembleSTBlockDiagonal( _Mwp, _MMwp, "MMwp", _MwpAssembled, _MMwpAssembled );
}





// Handy function to assemble a monolithic space-time block-diagonal matrix starting from its block(s)
void KHISTOperatorAssembler::AssembleSTBlockDiagonal( const SparseMatrix& D, HYPRE_IJMatrix& DD,
                                                      const std::string& STMatName, const bool blockAssembled, bool& STMatAssembled ){ 
  if( STMatAssembled )
    return;

  if( !blockAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble block before assembling "<< STMatName <<std::endl;
    return;
  }


  // recover info on matrix structure
  const int numRowsPerBlock = D.NumRows();
  const int numColsPerBlock = D.NumCols();

  Array<int> nnzPerRow( numRowsPerBlock );    // num of non-zero els per row in main (diagonal) block (for preallocation)
  const int  *offIdxs = D.GetI();             // has size numrows, contains offsets for data in J for each row
  for ( int i = 0; i < numRowsPerBlock; ++i ){
    nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
  }


  // initialise matrix
  HYPRE_IJMatrixCreate( _comm, numRowsPerBlock*_myRank, numRowsPerBlock*(_myRank+1)-1,
                               numColsPerBlock*_myRank, numColsPerBlock*(_myRank+1)-1, &DD );
  HYPRE_IJMatrixSetObjectType( DD, HYPRE_PARCSR );
  HYPRE_IJMatrixSetRowSizes( DD, nnzPerRow.GetData() );
  HYPRE_IJMatrixInitialize( DD );


  // fill it with matrices assembled above
  Array<int> rowsGlbIdxBB( numRowsPerBlockBB );
  for ( int i = 0; i < numRowsPerBlockBB; ++i ){
    rowsGlbIdxBB[i] = i + numRowsPerBlockBB*_myRank;
  }
  Array<int> colsGlbIdx( D.NumNonZeroElems() );
  for ( int i=0; i<D.NumNonZeroElems(); i++ ) {
    colsGlbIdx[i] = D.GetJ()[i] + numColsPerBlock*_myRank;
  }
  HYPRE_IJMatrixSetValues( DD, numRowsPerBlock, nnzPerRow.GetData(),
                           rowsGlbIdx.GetData(), colsGlbIdx.GetData(), D.GetData() );


  // assemble
  HYPRE_IJMatrixAssemble( DD );

  STMatAssembled = true;


  if(_verbose>1 && _myRank == 0 ){
    std::cout<<" - Space-time operator "<<STMatName<<" assembled\n";
  }  


}







//-----------------------------------------------------------------------------
// Assemble operators for preconditioner
//-----------------------------------------------------------------------------

// Assemble inverse of AA
void KHISTOperatorAssembler::AssembleAinv(){
  if ( _AinvAssembled ){
    return;
  }

  if( !_ApAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble A before assembling Ainv\n";
    return;
  }

  _Ainv = new PetscLinearSolver( _Ap, "ApSolver_" );
  
  _AinvAssembled = true;

}



// Assemble inverse of Mv
void KHISTOperatorAssembler::AssembleMvinv(){
  if ( _MvinvAssembled ){
    return;
  }

  if( !_MvAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble Mv before assembling Mvinv\n";
    return;
  }

  _Mvinv = new PetscLinearSolver( _Mv, "MvSolver_" );
  
  _MvinvAssembled = true;

}



----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------




// Assemble FF^-1 (used in preconditioner)
void KHISTOperatorAssembler::AssembleFFinv( const int spaceTimeSolverType = 0 ){
  if ( _FFinvAssembled ){
    return;
  }

  switch (spaceTimeSolverType){
    // Use sequential time-stepping to solve for space-time block
    case 0:{
      if(!( _MwAssembled && _FwAssembled ) && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble Mw and Fw before assembling Finv\n";
        return;
      }
      //                                             flag as time-dependent regardless of anything
      SpaceTimeSolver *temp  = new SpaceTimeSolver( _comm, NULL, NULL, _essVhTDOF, true, _verbose );

      temp->SetF( &_Fw );
      temp->SetM( &_Mw );
      _FFinv = temp;

      _FFinvAssembled = true;
      
      break;
    }

    // Use BoomerAMG with AIR set-up
    case 1:{
      if(! _FFAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble FF before assembling Finv\n";
        return;
      }

      // Initialise MFEM wrapper for BoomerAMG solver
      HypreBoomerAMG *temp = new HypreBoomerAMG( *_FFF );

      // Cast as HYPRE_Solver to get the underlying hypre object
      HYPRE_Solver FFinv( *temp );

      // Set it up
      SetUpBoomerAMG( FFinv );

      _FFinv = temp;
  
      _FFinvAssembled = true;

      break;
    }



    // Use GMRES with BoomerAMG precon
    case 2:{
      if(! _FFAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble FF before assembling Finv\n";
        return;
      }
      if( _myRank == 0 ){
        std::cout<<"WARNING: Since you're using GMRES to solve the space-time block inside the preconditioner"<<std::endl
                 <<"         make sure that flexible GMRES is used as the outer solver!"<<std::endl;
      }


      // Initialise MFEM wrappers for GMRES solver and preconditioner
      HypreGMRES     *temp  = new HypreGMRES(     *_FFF );
      HypreBoomerAMG *temp2 = new HypreBoomerAMG( *_FFF );

      // Cast preconditioner as HYPRE_Solver to get the underlying hypre object
      HYPRE_Solver FFinvPrecon( *temp2 );
      // Set it up
      SetUpBoomerAMG( FFinvPrecon, 1 );   // with just one iteration this time around

      // Attach preconditioner to solver
      temp->SetPreconditioner( *temp2 );

      // adjust gmres options
      temp->SetKDim( 50 );
      temp->SetTol( 0.0 );   // to ensure fixed number of iterations
      temp->SetMaxIter( 15 );

      _FFinv     = temp;
      _FFinvPrec = temp2;
  
      _FFinvAssembled = true;

      break;
    }

    // // Use Parareal with coarse/fine solver of different accuracies
    // case 3:{

    //   const int maxIt = 2;

    //   if( _numProcs <= maxIt ){
    //     if( _myRank == 0 ){
    //       std::cerr<<"ERROR: AssembleFFinv: Trying to set solver as "<<maxIt<<" iterations of Parareal, but the fine discretisation only has "<<_numProcs<<" nodes. "
    //                <<"This is equivalent to time-stepping, so I'm picking that as a solver instead."<<std::endl;
    //     }
        
    //     AssembleFFinv( 0 );
    //     return;
    //   }


    //   PararealSolver *temp  = new PararealSolver( _comm, NULL, NULL, NULL, maxIt, _verbose );

    //   temp->SetF( &_Fw );
    //   temp->SetC( &_Fw ); // same operator is used for both! it's the solver that changes, eventually...
    //   temp->SetM( &_Mw );
    //   _FFinv = temp;

    //   _FFinvAssembled = true;
      
    //   break;
    // }
    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<spaceTimeSolverType<<" not recognised."<<std::endl;
      }
    }
  }


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time block inverse (approximation) FF^-1 assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}








void KHISTOperatorAssembler::SetUpBoomerAMG( HYPRE_Solver& FFinv, const int maxiter ){
  int printLevel = 0;

  // AIR parameters for diffusion equation:
  double distance_R = 1.5;
  std::string prerelax = "A";
  std::string postrelax = "FFC";
  int interp_type = 0;
  int relax_type = 8;
  int coarsen_type = 6;
  double strength_tolC = 0.005;
  double strength_tolR = 0.005;
  double filter_tolR = 0.0;
  double filter_tolA = 0.0;
  int cycle_type = 1;


  // // AIR parameters:
  // double distance_R = 1.5;
  // std::string prerelax = "A";
  // std::string postrelax = "FFC";
  // int interp_type = 100;
  // int relax_type = 3;
  // int coarsen_type = 6;
  // double strength_tolC = 0.005;
  // double strength_tolR = 0.005;
  // double filter_tolR = 0.0;
  // double filter_tolA = 0.0;
  // int cycle_type = 1;

  // AMG parameters
  // double distance_R = -1;
  // std::string prerelax = "AA";
  // std::string postrelax = "AA";
  // int interp_type = 6;
  // int relax_type = 3;
  // int coarsen_type = 6;
  // double strength_tolC = 0.1;
  // double strength_tolR = -1;
  // double filter_tolR = 0.0;
  // double filter_tolA = 0.0;
  // int cycle_type = 1;

  // // AIR hyperbolic parameters
  // double distance_R = 1.5;
  // std::string prerelax = "A";
  // std::string postrelax = "F";
  // int interp_type = 100;
  // int relax_type = 10;
  // int coarsen_type = 6;
  // double strength_tolC = 0.005;
  // double strength_tolR = 0.005;
  // double filter_tolR = 0.0;
  // double filter_tolA = 0.0001;
  // int cycle_type = 1;


  // Create preconditioner
  HYPRE_BoomerAMGSetTol( FFinv, 0 );    // set tolerance to 0 so to have a fixed number of iterations
  HYPRE_BoomerAMGSetMaxIter( FFinv, maxiter );
  HYPRE_BoomerAMGSetPrintLevel( FFinv, printLevel );

  unsigned int ns_down = prerelax.length();
  unsigned int ns_up   = postrelax.length();
  int ns_coarse = 1;
  std::string Fr("F");
  std::string Cr("C");
  std::string Ar("A");
  int* *grid_relax_points = new int* [4];
  grid_relax_points[0] = NULL;
  grid_relax_points[1] = new int[ns_down];
  grid_relax_points[2] = new int [ns_up];
  grid_relax_points[3] = new int[1];
  grid_relax_points[3][0] = 0;

  // set down relax scheme 
  for(unsigned int i = 0; i<ns_down; i++) {
    if (prerelax.compare(i,1,Fr) == 0) {
      grid_relax_points[1][i] = -1;
    }
    else if (prerelax.compare(i,1,Cr) == 0) {
      grid_relax_points[1][i] = 1;
    }
    else if (prerelax.compare(i,1,Ar) == 0) {
      grid_relax_points[1][i] = 0;
    }
  }

  // set up relax scheme 
  for(unsigned int i = 0; i<ns_up; i++) {
    if (postrelax.compare(i,1,Fr) == 0) {
      grid_relax_points[2][i] = -1;
    }
    else if (postrelax.compare(i,1,Cr) == 0) {
      grid_relax_points[2][i] = 1;
    }
    else if (postrelax.compare(i,1,Ar) == 0) {
      grid_relax_points[2][i] = 0;
    }
  }


  if (distance_R > 0) {
    HYPRE_BoomerAMGSetRestriction( FFinv, distance_R );
    HYPRE_BoomerAMGSetStrongThresholdR( FFinv, strength_tolR );
    HYPRE_BoomerAMGSetFilterThresholdR( FFinv, filter_tolR );
  }
  HYPRE_BoomerAMGSetInterpType( FFinv, interp_type );
  HYPRE_BoomerAMGSetCoarsenType( FFinv, coarsen_type );
  HYPRE_BoomerAMGSetAggNumLevels( FFinv, 0 );
  HYPRE_BoomerAMGSetStrongThreshold( FFinv, strength_tolC );
  HYPRE_BoomerAMGSetGridRelaxPoints( FFinv, grid_relax_points );     // TODO: THIS FUNCTION IS DEPRECATED!! nobody knows whose responsibility it is to free grid_relax_points
  if (relax_type > -1) {
    HYPRE_BoomerAMGSetRelaxType( FFinv, relax_type );
  }
  HYPRE_BoomerAMGSetCycleNumSweeps( FFinv, ns_coarse, 3 );
  HYPRE_BoomerAMGSetCycleNumSweeps( FFinv, ns_down,   1 );
  HYPRE_BoomerAMGSetCycleNumSweeps( FFinv, ns_up,     2 );
  if (filter_tolA > 0) {
    HYPRE_BoomerAMGSetADropTol( FFinv, filter_tolA );
  }
  // type = -1: drop based on row inf-norm
  else if (filter_tolA == -1) {
    HYPRE_BoomerAMGSetADropType( FFinv, -1 );
  }

  // Set cycle type for solve 
  HYPRE_BoomerAMGSetCycleType( FFinv, cycle_type );


}








// Assembles space-time KHI block system
//              ⌈ FFF BBB     ⌉⌈w⌉ ⌈f⌉
//   Ax = b <-> | Mwp AAA     ||p|=|g|,
//              ⌊     CCC MMv ⌋⌊v⌋ ⌊h⌋
// where:
//  - FF is space-time matrix for w
//  - AA is block-diagonal with potential laplacian
//  - Mv is block-diagonal with v mass matrix
//  - BB is block-diagonal with div(w * )
//  - Mw is block-diagonal with mixed w-p mass matrix
//  - CC is block-diagonal with curl(k * )
//  - f  is the rhs for w
//  - g  is the rhs for p
//  - h  is the rhs for v
// Function also provides suitable initial guess for system (initialised with dirichlet BC)
void KHISTOperatorAssembler::AssembleSystem( HypreParMatrix*& FFF,  HypreParMatrix*& AAA,  HypreParMatrix*& MMv,
                                             HypreParMatrix*& BBB,  HypreParMatrix*& Mwp,  HypreParMatrix*& CCC,
                                             HypreParVector*& frhs, HypreParVector*& grhs, HypreParVector*& hrhs,
                                             HypreParVector*& IGw,  HypreParVector*& IGp,  HypreParVector*& IGv ){

  // - initialise relevant matrices
  AssembleFw();
  AssembleMw();
  AssembleAp();
  AssembleMv();
  AssembleMwp();
  AssembleB();
  AssembleC();

  if ( _verbose>50 ){
    if ( _myRank == 0 ){
      std::ofstream myfile;
      std::string myfilename;

      myfilename = "./results/out_original_Fw.dat";
      myfile.open( myfilename );
      _Fw.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mw.dat";
      myfile.open( myfilename );
      _Mw.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mv.dat";
      myfile.open( myfilename );
      _Mv.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Ap.dat";
      myfile.open( myfilename );
      _Ap.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_B.dat";
      myfile.open( myfilename );
      _B.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mwp.dat";
      myfile.open( myfilename );
      _Mwp.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_C.dat";
      myfile.open( myfilename );
      _C.PrintMatlab(myfile);
      myfile.close( );  
    }
    MPI_Barrier(_comm);
  }






  // ASSEMBLE RHS -----------------------------------------------------------
  // Initialise handy functions for rhs
  FunctionCoefficient       wFuncCoeff( _wFunc );
  FunctionCoefficient       pFuncCoeff( _pFunc );
  FunctionCoefficient       fFuncCoeff( _fFunc );
  FunctionCoefficient       gFuncCoeff( _gFunc );
  VectorFunctionCoefficient vFuncCoeff( _dim, _vFunc );
  VectorFunctionCoefficient hFuncCoeff( _dim, _hFunc );

  // - specify evaluation time
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  fFuncCoeff.SetTime( _dt*(_myRank+1) );
  gFuncCoeff.SetTime( _dt*(_myRank+1) );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );
  hFuncCoeff.SetTime( _dt*(_myRank+1) );


  // Assemble local part of rhs
  // - for w
  LinearForm *fform( new LinearForm );
  fform->Update( _WhFESpace );
  fform->AddDomainIntegrator(   new DomainLFIntegrator(       fFuncCoeff       ) );  //int_\Omega f*v
  TODO
  // fform->AddBoundaryIntegrator( new BoundaryLFIntegrator(     nFuncCoeff       ) );  //int_d\Omega \mu * du/dn *v
  // fform->AddBoundaryIntegrator( new BoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n

  fform->Assemble();

#ifdef MULT_BY_DT
  fform->operator*=( _dt );
#endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for W assembled\n";
    }
    MPI_Barrier(_comm);
  }  

  // -- include initial conditions
  if( _myRank == 0 ){
    wFuncCoeff.SetTime( 0.0 );
    LinearForm *w0form( new LinearForm );
    w0form->Update( _WhFESpace );
    w0form->AddDomainIntegrator( new DomainLFIntegrator( wFuncCoeff ) );  //int_\Omega w0*v
    w0form->Assemble();

#ifndef MULT_BY_DT
    w0form->operator*=(1./_dt);
#endif
    fform->operator+=( *w0form );


    if ( _verbose>100 ){
      std::cout<<"Contribution from IC on w: "; w0form->Print(std::cout, w0form->Size());
    }

    // remember to reset function evaluation for w to the current time
    wFuncCoeff.SetTime( _dt*(_myRank+1) );


    delete w0form;

    if(_verbose>10){
      std::cout<<"Contribution from initial condition included\n"<<std::endl;
    }
  }



  // - for p
  LinearForm *gform( new LinearForm );
  gform->Update( _PhFESpace );
  gform->AddDomainIntegrator( new DomainLFIntegrator( gFuncCoeff ) );  //int_\Omega g*q
  gform->Assemble();

#ifdef MULT_BY_DT
  gform->operator*=( _dt );
#endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for phi assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // - for v
  LinearForm *hform( new LinearForm );
  hform->Update( _VhFESpace );
  hform->AddDomainIntegrator( new VectorDomainLFIntegrator( hFuncCoeff ) );  //int_\Omega h*v
  hform->Assemble();

#ifdef MULT_BY_DT
  hform->operator*=( _dt );
#endif


  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for v assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  TODO
  // - adjust rhs to take dirichlet BC for current time-step into account
  // -- initialise function with BC
  GridFunction wBC(_WhFESpace);//, pBC(_QhFESpace);
  wBC.ProjectCoefficient(wFuncCoeff);
  // pBC.ProjectCoefficient(pFuncCoeff);
  // -- initialise local rhs
  Vector fRhsLoc(  fform->Size() );
  Vector gRhsLoc(  gform->Size() );
  Vector hRhsLoc(  hform->Size() );
  // -- initialise local initial guess to exact solution
  Vector igwLoc( fform->Size() );
  Vector igpLoc( gform->Size() );
  Vector igvLoc( hform->Size() );
  Vector empty2;
  igwLoc = wBC;
  igwLoc.SetSubVectorComplement( _essWhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  TODO
  igpLoc = 0.;                                      // dirichlet BC are not actually imposed on p
  igvLoc = 0.;                                      // dirichlet BC are not actually imposed on v
  // igpLoc.SetSubVectorComplement( _essQhTDOF, 0.0);
  Array<int> empty;



  TODO
  // ASSEMBLE LOCAL LINEAR SYSTEMS (PARTICULARLY, CONSTRAINED MATRICES) -----
  // - Assemble _Fw (and modify rhs to take dirichlet on u into account)
  fRhsLoc = *fform;
  mfem::Array<int> cols(_Fw.Height());
  cols = 0;
  for (int i = 0; i < _essWhTDOF.Size(); ++i){
    cols[_essWhTDOF[i]] = 1;
  }
  _Fw.EliminateCols( cols, &wBC, &fRhsLoc );
  for (int i = 0; i < _essWhTDOF.Size(); ++i){
    _Fw.EliminateRow( _essWhTDOF[i], mfem::Matrix::DIAG_ONE );
    fRhsLoc(_essWhTDOF[i]) = wBC(_essWhTDOF[i]);
  }
  // - Assemble _B (and modify rhs to take dirichlet on u into account)
  _bVarf->FormRectangularLinearSystem( _essVhTDOF, empty, wBC, *gform, _B,  empty2, gRhsLoc );  // iguloc should still be initialised to uBC
  // - Assemble _Mu (and modify rhs to take dirichlet on u into account)
  wFuncCoeff.SetTime( _dt*_myRank );                // set uFunc to previous time-step
  GridFunction wm1BC(_VhFESpace);
  wm1BC.ProjectCoefficient(wFuncCoeff);

  Vector wm1Rel( fRhsLoc.Size() );
  wm1Rel = 0.0;
  _muVarf->EliminateVDofs( _essVhTDOF, wm1BC, wm1Rel, Matrix::DiagonalPolicy::DIAG_ZERO ); 


  if( _myRank > 0 ){
    // add to rhs (wm1Rel should already take minus sign on _Mw into account)
    // NB: - no need to rescale by dt, as _Mw will be already scaled accordingly.
    //     - no need to flip sign, as _Mw carries with it already
    fRhsLoc += wm1Rel;
  }

  // remember to reset function evaluation for u to the current time
  wFuncCoeff.SetTime( _dt*(_myRank+1) );

  // store velocity mass matrix - without contribution from dirichlet nodes
  _Mu = _muVarf->SpMat();


  _FwAssembled = true;
  _MwAssembled = true;
  _BAssembled  = true;

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Effect from Dirichlet BC (if prescribed) included in assembled blocks\n";
    }
    MPI_Barrier(_comm);
  }  



  // - Assemble _Ap (and modify rhs to take dirichlet on p into account?)
  TODO

  // - Assemble _Mv (and modify rhs to take dirichlet on v into account?)
  TODO



  TODO
  // ASSEMBLE GLOBAL (PARALLEL) RHS -----------------------------------------
  // - for velocity
  int colPartV[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPartV );
  frhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for velocity block f assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for pressure
  int colPartP[2] = {_myRank*gRhsLoc.Size(), (_myRank+1)*gRhsLoc.Size()};
  grhs = new HypreParVector( _comm, gRhsLoc.Size()*_numProcs, gRhsLoc.StealData(), colPartP );
  grhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for pressure block g assembled\n";
    }
    MPI_Barrier(_comm);
  }  





  // ASSEMBLE INITIAL GUESS -------------------------------------------------
  // Assemble global vectors
  IGu = new HypreParVector( _comm, iguLoc.Size()*_numProcs, iguLoc.StealData(), colPartV );
  IGp = new HypreParVector( _comm, igpLoc.Size()*_numProcs, igpLoc.StealData(), colPartP );
  IGu->SetOwnership( 1 );
  IGp->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time initial guess assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  // ASSEMBLE SPACE-TIME OPERATOR -------------------------------------------
  //Assemble top-left block
  AssembleFF();

  // - pass handle to mfem matrix
  FFF = new HypreParMatrix();
  FFF->MakeRef( *_FFF );



  //Assemble bottom-left block
  AssembleBB();

  // - convert to mfem operator
  HYPRE_ParCSRMatrix  BBref;
  HYPRE_IJMatrixGetObject( _BB, (void **) &BBref);
  BBB = new HypreParMatrix( BBref, false ); //"false" doesn't take ownership of data


  

  if ( _verbose>50 ){
    std::string myfilename = std::string("./results/IGu.dat");
    IGu->Print( myfilename.c_str() );
    myfilename = std::string("./results/RHSu.dat");
    frhs->Print( myfilename.c_str() );
    myfilename = std::string("./results/IGp.dat");
    IGp->Print( myfilename.c_str() );
    myfilename = std::string("./results/RHSp.dat");
    grhs->Print( myfilename.c_str() );

    if ( _myRank == 0 ){
      std::ofstream myfile;
      std::string myfilename = "./results/out_final_B.dat";
      myfile.open( myfilename );
      _B.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_F.dat";
      myfile.open( myfilename );
      _Fu.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_M.dat";
      myfile.open( myfilename );
      _Mu.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_essV.dat";
      myfile.open( myfilename );
      _essVhTDOF.Print(myfile,1);
      myfile.close( );

      myfilename = "./results/out_essQ.dat";
      myfile.open( myfilename );
      _essQhTDOF.Print(myfile,1);
      myfile.close( );

      std::cout<<"U essential nodes: ";_essVhTDOF.Print(std::cout, _essVhTDOF.Size());
      std::cout<<"P essential nodes: ";_essQhTDOF.Print(std::cout, _essQhTDOF.Size());

    }

  }

}












// Assemble operators appearing in diagonal of preconditioner
//          ⌈ FF^-1 \\\\  \\\\\  ⌉
//   P^-1 = |       AA^-1 \\\\\  |
//          ⌊             MMv^-1 ⌋
// (not necessarily in this order)
// where FF contains space-time matrix for w
void KHISTOperatorAssembler::AssemblePreconditioner( Operator*& Finv, Operator*& Ainv, Operator*& Mvinv, const int spaceTimeSolverType=0 ){

  //Assemble inverses
  AssembleFFinv( spaceTimeSolverType );
  AssembleAAinv( );
  AssembleMvinv( );

  Finv  = _FFinv;
  Ainv  = _Ainv;
  Mvinv = _Mvinv;

}










//-----------------------------------------------------------------------------
// Utils functions
//-----------------------------------------------------------------------------

// Returns vector containing the space-time exact solution (if available)
void KHISTOperatorAssembler::ExactSolution( HypreParVector*& w, HypreParVector*& p, HypreParVector*& v ) const{
  // Initialise handy functions
  FunctionCoefficient       wFuncCoeff(_wFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
  // - specify evaluation time
  // -- notice first processor actually refers to instant dt
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );

  GridFunction wFun( _WhFESpace );
  GridFunction pFun( _PhFESpace );
  GridFunction vFun( _VhFESpace );

  wFun.ProjectCoefficient( wFuncCoeff );
  pFun.ProjectCoefficient( pFuncCoeff );
  vFun.ProjectCoefficient( vFuncCoeff );
  

  Array<int> rowStartsW(2), rowStartsP(2), rowStartsV(2);
  rowStartsW[0] = ( wFun.Size() )*_myRank;
  rowStartsW[1] = ( wFun.Size() )*(_myRank+1);
  rowStartsP[0] = ( pFun.Size() )*_myRank;
  rowStartsP[1] = ( pFun.Size() )*(_myRank+1);
  rowStartsV[0] = ( vFun.Size() )*_myRank;
  rowStartsV[1] = ( vFun.Size() )*(_myRank+1);

  w = new HypreParVector( _comm, (wFun.Size())*_numProcs, wFun.StealData(), rowStartsW.GetData() );
  p = new HypreParVector( _comm, (pFun.Size())*_numProcs, pFun.StealData(), rowStartsP.GetData() );
  v = new HypreParVector( _comm, (vFun.Size())*_numProcs, vFun.StealData(), rowStartsV.GetData() );

  w->SetOwnership( 1 );
  p->SetOwnership( 1 );
  v->SetOwnership( 1 );

}



// Each processor computes L2 error of solution at its time-step
void KHISTOperatorAssembler::ComputeL2Error( const Vector& wh, const Vector& ph, const Vector& vh,
                                             double& err_w, double& err_p, double& err_v ) const{

  const GridFunction w( _WhFESpace, wh.GetData() );
  const GridFunction p( _PhFESpace, ph.GetData() );
  const GridFunction v( _VhFESpace, vh.GetData() );

  int order_quad = 5;
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i=0; i < Geometry::NumGeom; ++i){
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  FunctionCoefficient       wFuncCoeff(_wFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );


  err_w  = w.ComputeL2Error(wFuncCoeff, irs);
  err_p  = p.ComputeL2Error(pFuncCoeff, irs);
  err_v  = v.ComputeL2Error(vFuncCoeff, irs);

  // for ( int i = 0; i < _numProcs; ++i ){
  //   if ( _myRank == i ){
  //     std::cout << "Instant t="       << _dt*(_myRank+1) << std::endl;
  //     std::cout << "|| wh - wEx ||_L2= " << err_w << "\n";
  //     std::cout << "|| ph - pEx ||_L2= " << err_p << "\n";
  //     std::cout << "|| vh - vEx ||_L2= " << err_v << "\n";
  //   }
  //   MPI_Barrier( _comm );
  // }
}




void KHISTOperatorAssembler::SaveExactSolution( const std::string& path="ParaView",
                                                const std::string& filename="STKHI_Ex" ) const{
  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set wpv paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w",   wFun );
    paraviewDC.RegisterField( "phi", pFun );
    paraviewDC.RegisterField( "v",   vFun );

    // main time loop
    for ( int t = 0; t < _numProcs+1; ++t ){
      FunctionCoefficient       wFuncCoeff(_wFunc);
      FunctionCoefficient       pFuncCoeff(_pFunc);
      VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
      wFuncCoeff.SetTime( t*_dt );
      pFuncCoeff.SetTime( t*_dt );
      vFuncCoeff.SetTime( t*_dt );

      wFun->ProjectCoefficient( wFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      vFun->ProjectCoefficient( vFuncCoeff );

      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}

// store given approximate solution in paraview format
void KHISTOperatorAssembler::SaveSolution( const HypreParVector& wh, const HypreParVector& ph, const HypreParVector& vh,
                                           const std::string& path="ParaView", const std::string& filename="STKHI" ) const{
  
  // gather parallel vector
  Vector *wGlb = wh.GlobalVector();
  Vector *pGlb = ph.GlobalVector();
  Vector *vGlb = vh.GlobalVector();


  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w",   wFun );
    paraviewDC.RegisterField( "phi", pFun );
    paraviewDC.RegisterField( "v",   vFun );


    // store initial conditions
    FunctionCoefficient       wFuncCoeff(_wFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
    wFuncCoeff.SetTime( 0. );
    pFuncCoeff.SetTime( 0. );
    vFuncCoeff.SetTime( 0. );

    wFun->ProjectCoefficient( wFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    vFun->ProjectCoefficient( vFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // handy variables for time loop
    const int blockSizeW = wh.Size();
    const int blockSizeP = ph.Size();
    const int blockSizeV = vh.Size();
    Vector wLcl, pLcl, vLcl;
    Array<int> idxW(blockSizeW), idxP(blockSizeP), idxV(blockSizeV);

    // main time loop
    for ( int t = 1; t < _numProcs+1; ++t ){
      // - identify correct sub-vector idx in global vectors
      for ( int i = 0; i < blockSizeW; ++i ){
        idxW[i] = blockSizeW*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeP; ++i ){
        idxP[i] = blockSizeP*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeV; ++i ){
        idxV[i] = blockSizeV*(t-1) + i;
      }

      // - extract subvector
      wGlb->GetSubVector( idxW, wLcl );
      pGlb->GetSubVector( idxP, pLcl );
      vGlb->GetSubVector( idxV, vLcl );
      
      // - assign to linked variables
      *wFun = wLcl;
      *pFun = pLcl;
      *vFun = vLcl;
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}

// This function is the same as above, but it doesn't rely on HypreParVector's
void KHISTOperatorAssembler::SaveSolution( const Vector& wh, const Vector& ph, const Vector& vh,
                                           const std::string& path="ParaView",
                                           const std::string& filename="STKHI" ) const{
  const int blockSizeW = wh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeV = vh.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( wh.GetData(), blockSizeW, MPI_DOUBLE, 0, 3*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 3*_myRank+1, _comm );
    MPI_Send( vh.GetData(), blockSizeV, MPI_DOUBLE, 0, 3*_myRank+2, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w",   wFun );
    paraviewDC.RegisterField( "phi", pFun );
    paraviewDC.RegisterField( "v",   vFun );


    // store initial conditions
    FunctionCoefficient       wFuncCoeff(_wFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
    wFuncCoeff.SetTime( 0.0 );
    pFuncCoeff.SetTime( 0.0 );
    vFuncCoeff.SetTime( 0.0 );
    
    wFun->ProjectCoefficient( wFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    vFun->ProjectCoefficient( vFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // this will store the approximate solution at current time-step
    Vector wLcl(blockSizeW), pLcl(blockSizeP), vLcl(blockSizeV);

    // handle first time-step separately
    *wFun = wh;
    *pFun = ph;
    *vFun = vh;
    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( wLcl.GetData(), blockSizeW, MPI_DOUBLE, t-1, 3*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 3*(t-1)+1, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( vLcl.GetData(), blockSizeV, MPI_DOUBLE, t-1, 3*(t-1)+2, _comm, MPI_STATUS_IGNORE );

      // - assign to linked variables
      *wFun = wLcl;
      *pFun = pLcl;
      *vFun = vLcl;
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}





// Saves a plot of the error
void KHISTOperatorAssembler::SaveError( const Vector& wh, const Vector& ph, const Vector& vh,
                                        const std::string& path="ParaView",
                                        const std::string& filename="STKHI_err" ) const{
  const int blockSizeW = wh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeV = vh.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( wh.GetData(), blockSizeW, MPI_DOUBLE, 0, 3*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 3*_myRank+1, _comm );
    MPI_Send( vh.GetData(), blockSizeV, MPI_DOUBLE, 0, 3*_myRank+2, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w-wh",   wFun );
    paraviewDC.RegisterField( "p-ph", pFun );
    paraviewDC.RegisterField( "v-vh",   vFun );

    // this will store the approximate solution at current time-step
    Vector wLcl(blockSizeW), pLcl(blockSizeP), vLcl(blockSizeV);

    // these will provide exact solution
    VectorFunctionCoefficient wFuncCoeff(_dim,_wFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    FunctionCoefficient       vFuncCoeff(_vFunc);

    // error at instant 0 is 0 (IC)
    *wFun = 0.;
    *pFun = 0.;
    *vFun = 0.;
    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();

    // handle first time-step separately
    wFuncCoeff.SetTime( _dt );
    pFuncCoeff.SetTime( _dt );
    vFuncCoeff.SetTime( _dt );
    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    vFun->ProjectCoefficient( vFuncCoeff );

    wFun->operator-=( wh );
    pFun->operator-=( ph );
    vFun->operator-=( vh );

    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( wLcl.GetData(), blockSizeW, MPI_DOUBLE, t-1, 3*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 3*(t-1)+1, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( vLcl.GetData(), blockSizeV, MPI_DOUBLE, t-1, 3*(t-1)+2, _comm, MPI_STATUS_IGNORE );
      
      // - assign to linked variables
      wFuncCoeff.SetTime( _dt*t );
      pFuncCoeff.SetTime( _dt*t );
      vFuncCoeff.SetTime( _dt*t );
      wFun->ProjectCoefficient( wFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      vFun->ProjectCoefficient( vFuncCoeff );
      wFun->operator-=( wLcl );
      pFun->operator-=( pLcl );
      vFun->operator-=( vLcl );
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}










void KHISTOperatorAssembler::GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const{
  if(_mesh == NULL)
    std::cerr<<"Mesh not yet set"<<std::endl;
  else
    _mesh->GetCharacteristics( h_min, h_max, k_min, k_max );
}




void KHISTOperatorAssembler::PrintMatrices( const std::string& filename ) const{

  if( ! ( _FuAssembled && _MuAssembled && _MpAssembled && _ApAssembled && _BAssembled && (_Pe==0.0 || _WpAssembled) ) ){
    if( _myRank == 0){
        std::cerr<<"Make sure all matrices have been initialised, otherwise they can't be printed"<<std::endl;
    }
    return;
  }

  std::string myfilename;
  std::ofstream myfile;


  if ( _myRank == 0 ){
    myfilename = filename + "_Fu_" + std::to_string(_myRank) +".dat";
    myfile.open( myfilename );
    _Fu.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Mu.dat";
    myfile.open( myfilename );
    _Mu.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Mp.dat";
    myfile.open( myfilename );
    _Mp.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Ap.dat";
    myfile.open( myfilename );
    _Ap.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_B.dat";
    myfile.open( myfilename );
    _B.PrintMatlab(myfile);
    myfile.close( );
  }

  if ( _Pe!=0. ){
    myfilename = filename + "_Wp_" + std::to_string(_myRank) +".dat";
    myfile.open( myfilename );
    _Wp.PrintMatlab(myfile);
    myfile.close( );
    if ( _myRank!=0 ){
      myfilename = filename + "_Fu_" + std::to_string(_myRank) +".dat";
      myfile.open( myfilename );
      _Fu.PrintMatlab(myfile);
      myfile.close( );
    }
  }


}




// Actual Time-stepper
void KHISTOperatorAssembler::TimeStep( const BlockVector& x, BlockVector& y,
                                          const std::string &fname1, const std::string &path2, int refLvl ){
  TODO
  // Define operator
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = _Fu.NumRows();
  offsets[2] =  _B.NumRows();
  offsets.PartialSum();
  PetscParMatrix myB( &_B );

  BlockOperator stokesOp( offsets );
  stokesOp.SetBlock(0, 0, &_Fu );
  stokesOp.SetBlock(0, 1, myB.Transpose() );
  stokesOp.SetBlock(1, 0, &_B  );


  // Define preconditioner
  // - inverse of pressure Schur complement (bottom-right block in precon) - reuse code from ST case, but with single processor
  StokesSTPreconditioner myPSchur( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essQhTDOF, _verbose );
  AssembleAp();
  AssembleMp();
  AssembleWp();
  myPSchur.SetAp( &_Ap );
  myPSchur.SetMp( &_Mp );
  if( _Pe != 0. ){
    myPSchur.SetWp( &_Wp, false );   // if there is convection, then clearly Wp differs from Ap (must include pressure convection)
  }else if( _essQhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
    myPSchur.SetWp( &_Wp, true );    
  }else{
    // _pSchur->SetWp( &_Wp, false );
    myPSchur.SetWp( &_Wp, true );     // should be false, according to E/S/W!
    if( _myRank == 0 ){
      std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
               <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
               <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
    }
  }

  // - inverse of velocity operator (top-left block in precon)
  const PetscParMatrix myFu( &_Fu );
  PetscLinearSolver Fsolve( myFu, "VSolver_" );

  // - assemble precon
  BlockUpperTriangularPreconditioner stokesPr(offsets);
  stokesPr.iterative_mode = false;
  stokesPr.SetDiagonalBlock( 0, &Fsolve );
  stokesPr.SetDiagonalBlock( 1, &myPSchur );
  stokesPr.SetBlock( 0, 1, myB.Transpose() );


  // Define solver
  PetscLinearSolver solver( MPI_COMM_SELF, "solver_" );
  bool isIterative = true;
  solver.iterative_mode = isIterative;
  solver.SetPreconditioner(stokesPr);
  solver.SetOperator(stokesOp);

  double tol = 1e-10 / sqrt( _numProcs );
  if( _myRank == 0 ){
    std::cout<<"Warning: Considering a fixed overall tolerance of 1e-10, scaled by the number of time steps."<<std::endl
             <<"          This option gets overwritten if a tolerance is prescribed in the petsc option file,"<<std::endl    
             <<"          so make sure to delete it from there!"<<std::endl;    
  }
  solver.SetTol(tol);
  solver.SetRelTol(tol);
  solver.SetAbsTol(tol);


  // Main "time-stepping" routine
  const int totDofs = x.Size();
  // - for each time-step, this will contain rhs
  BlockVector b( offsets );
  b = 0.0;
  
  // - receive solution from previous processor (unless initial time-step)
  if ( _myRank > 0 ){
    // - use it as initial guess for the solver (so store in y), unless initial step!
    MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
    // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
    _Mu.Mult( y.GetBlock(0), b.GetBlock(0) );
    // - M is stored with negative sign for velocity, so flip it
    b.GetBlock(0).Neg();
  }

  // - define rhs for this step (includes contribution from sol at previous time-step
  b += x;

  // - solve for current time-step
  //  --y acts as an initial guess! So use prev sol, unless first time step
  solver.Mult( b, y );

  int GMRESit    = solver.GetNumIterations();
  double resnorm = solver.GetFinalNorm();
  std::cout<<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

  if (!std::experimental::filesystem::exists( path2 )){
    std::experimental::filesystem::create_directories( path2 );
  }
  std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
  std::ofstream myfile;
  myfile.open( fname2, std::ios::app );
  myfile <<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
  myfile.close();

  //TODO: I should probably add a check that all the print to file are done in order, but cmon...



  // - send solution to following processor (unless last time-step)
  if ( _myRank < _numProcs-1 ){
    MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
  }

  // sum residuals at each time step
  resnorm*= resnorm;
  MPI_Allreduce( MPI_IN_PLACE, &resnorm, 1, MPI_DOUBLE, MPI_SUM, _comm );
  double finalResNorm = sqrt(resnorm);

  // average out iterations at each time step
  MPI_Allreduce( MPI_IN_PLACE, &GMRESit, 1, MPI_INT,    MPI_SUM, _comm );
  double avgGMRESit = double(GMRESit) / _numProcs;




  // OUTPUT -----------------------------------------------------------------
  // - save #it to convergence to file
  if (_myRank == 0){
    std::cout<<"Solver converged in "    <<avgGMRESit<<" average GMRES it per time-step.";
    std::cout<<" Final residual norm is "<<finalResNorm   <<".\n";
  
    double hmin, hmax, kmin, kmax;
    this->GetMeshSize( hmin, hmax, kmin, kmax );

    std::ofstream myfile;
    myfile.open( fname1, std::ios::app );
    myfile << _dt*_numProcs << ",\t" << _dt  << ",\t" << _numProcs   << ",\t"
           << hmax << ",\t" << hmin << ",\t" << refLvl << ",\t"
           << avgGMRESit << ",\t"  << finalResNorm  << std::endl;
    myfile.close();
  }

  // wait for write to be completed..possibly non-necessary
  MPI_Barrier(_comm);

}
















// Optimised Time-stepper (only triggers one solve per time-step if necessary, otherwise master takes care of solving everything)
void KHISTOperatorAssembler::TimeStep( const BlockVector& x, BlockVector& y,
                                          const std::string &fname1, const std::string &path2, int refLvl, int pbType ){

  // Define operator
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = _Fu.NumRows();
  offsets[2] =  _B.NumRows();
  offsets.PartialSum();
  PetscParMatrix myB( &_B );

  BlockOperator stokesOp( offsets );
  stokesOp.SetBlock(0, 0, &_Fu );
  stokesOp.SetBlock(0, 1, myB.Transpose() );
  stokesOp.SetBlock(1, 0, &_B  );

  BlockUpperTriangularPreconditioner* stokesPr = NULL;
  StokesSTPreconditioner* myPSchur = NULL;

  PetscParMatrix* myFu = NULL;
  PetscLinearSolver* Fsolve = NULL;
  PetscLinearSolver* solver = NULL;

  const int totDofs = x.Size();
  double finalResNorm = 0.0;
  double avgGMRESit   = 0.0;


  // Define operators
  if ( pbType == 4 || _myRank == 0 ){
    // Define preconditioner
    // - inverse of pressure Schur complement (bottom-right block in precon) - reuse code from ST case, but with single processor
    myPSchur = new StokesSTPreconditioner( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essQhTDOF, _verbose );
    AssembleAp();
    AssembleMp();
    AssembleWp();
    myPSchur->SetAp( &_Ap );
    myPSchur->SetMp( &_Mp );
    if( _Pe != 0. ){
      myPSchur->SetWp( &_Wp, false );   // if there is convection, then clearly Wp differs from Ap (must include pressure convection)
    }else if( _essQhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
      myPSchur->SetWp( &_Wp, true );    
    }else{
      // _pSchur->SetWp( &_Wp, false );
      myPSchur->SetWp( &_Wp, true );     // should be false, according to E/S/W!
      if( _myRank == 0 ){
        std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
                 <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
                 <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
      }
    }

    // - inverse of velocity operator (top-left block in precon)
    myFu   = new PetscParMatrix( &_Fu );
    Fsolve = new PetscLinearSolver( *myFu, "VSolver_" );

    // - assemble precon
    stokesPr = new BlockUpperTriangularPreconditioner(offsets);
    stokesPr->iterative_mode = false;
    stokesPr->SetDiagonalBlock( 0, Fsolve );
    stokesPr->SetDiagonalBlock( 1, myPSchur );
    stokesPr->SetBlock( 0, 1, myB.Transpose() );


    // Define solver
    solver = new PetscLinearSolver( MPI_COMM_SELF, "solver_" );
    bool isIterative = true;
    solver->iterative_mode = isIterative;
    solver->SetPreconditioner(*stokesPr);
    solver->SetOperator(stokesOp);

    double tol = 1e-10 / sqrt( _numProcs );
    if( _myRank == 0 ){
      std::cout<<"Warning: Considering a fixed overall tolerance of 1e-10, scaled by the number of time steps."<<std::endl
               <<"          This option gets overwritten if a tolerance is prescribed in the petsc option file,"<<std::endl    
               <<"          so make sure to delete it from there!"<<std::endl;    
    }
    solver->SetTol(tol);
    solver->SetRelTol(tol);
    solver->SetAbsTol(tol);
  }



  // Main "time-stepping" routine
  // - if each operator is different, each time step needs to solve its own
  if ( pbType == 4 ){
    // - for each time-step, this will contain rhs
    BlockVector b( offsets );
    b = 0.0;
    
    // - receive solution from previous processor (unless initial time-step)
    if ( _myRank > 0 ){
      // - use it as initial guess for the solver (so store in y), unless initial step!
      MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
      // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
      _Mu.Mult( y.GetBlock(0), b.GetBlock(0) );
      // - M is stored with negative sign for velocity, so flip it
      b.GetBlock(0).Neg();
    }

    // - define rhs for this step (includes contribution from sol at previous time-step
    b += x;

    // - solve for current time-step
    //  --y acts as an initial guess! So use prev sol, unless first time step
    solver->Mult( b, y );

    int GMRESit    = solver->GetNumIterations();
    double resnorm = solver->GetFinalNorm();
    std::cout<<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

    if (!std::experimental::filesystem::exists( path2 )){
      std::experimental::filesystem::create_directories( path2 );
    }
    std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
    std::ofstream myfile;
    myfile.open( fname2, std::ios::app );
    myfile <<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
    myfile.close();

    // - send solution to following processor (unless last time-step)
    if ( _myRank < _numProcs-1 ){
      MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
    }

    // sum residuals at each time step
    resnorm*= resnorm;
    MPI_Allreduce( MPI_IN_PLACE, &resnorm, 1, MPI_DOUBLE, MPI_SUM, _comm );
    finalResNorm = sqrt(resnorm);

    // average out iterations at each time step
    MPI_Allreduce( MPI_IN_PLACE, &GMRESit, 1, MPI_INT,    MPI_SUM, _comm );
    avgGMRESit = double(GMRESit) / _numProcs;


  // - otherwise, master takes care of it all
  }else{
    if ( _myRank == 0 ){

      // first time-step is a bit special
      solver->Mult( x, y );
      int GMRESit    = solver->GetNumIterations();
      double resnorm = solver->GetFinalNorm();

      avgGMRESit   += GMRESit;
      finalResNorm += resnorm*resnorm;

      std::cout<<"Solved for time-step "<<1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

      if (!std::experimental::filesystem::exists( path2 )){
        std::experimental::filesystem::create_directories( path2 );
      }
      std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
      std::ofstream myfile;
      myfile.open( fname2, std::ios::app );
      myfile <<"Solved for time-step "<<1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
      myfile.close();

      // this will keep track of sol at current time-step
      BlockVector y0 = y;

      // after first time-step, you need to
      for ( int t = 1; t < _numProcs; ++t ){
        // - this will contain rhs
        BlockVector b( offsets ), temp( offsets );
        b = 0.0; temp = 0.0;

        MPI_Recv( b.GetData(), totDofs, MPI_DOUBLE, t, 2*t,   _comm, MPI_STATUS_IGNORE );
        // nah, use prev sol as IG
        // MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, t, 3*t+1, _comm, MPI_STATUS_IGNORE );

        // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
        _Mu.Mult( y0.GetBlock(0), temp.GetBlock(0) );
        // - M is stored with negative sign for velocity, so flip it
        b -= temp;
    
        // - solve for current time-step
        //  --y acts as an initial guess! So use prev sol
        solver->Mult( b, y0 );

        int GMRESit    = solver->GetNumIterations();
        double resnorm = solver->GetFinalNorm();

        avgGMRESit   += GMRESit;
        finalResNorm += resnorm*resnorm;

        std::cout<<"Solved for time-step "<<t+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

        std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
        std::ofstream myfile;
        myfile.open( fname2, std::ios::app );
        myfile <<"Solved for time-step "<<t+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
        myfile.close();

        // - send solution to right processor
        MPI_Send( y0.GetData(), totDofs, MPI_DOUBLE, t, 2*t+1, _comm );
      }

      avgGMRESit = double(avgGMRESit)/_numProcs;
      finalResNorm = sqrt(finalResNorm);

    }else{
      // - send rhs to master
      MPI_Send( x.GetData(), totDofs, MPI_DOUBLE, 0, 2*_myRank,   _comm );
      // - send IG to master
      // MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, 0, 3*_myRank+1, _comm );
      // - receive solution from master
      MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, 0, 2*_myRank+1, _comm, MPI_STATUS_IGNORE );
    }
  }





  // OUTPUT -----------------------------------------------------------------
  // - save #it to convergence to file
  if (_myRank == 0){
    std::cout<<"Solver converged in "    <<avgGMRESit<<" average GMRES it per time-step.";
    std::cout<<" Final residual norm is "<<finalResNorm   <<".\n";
  
    double hmin, hmax, kmin, kmax;
    this->GetMeshSize( hmin, hmax, kmin, kmax );

    std::ofstream myfile;
    myfile.open( fname1, std::ios::app );
    myfile << _dt*_numProcs << ",\t" << _dt  << ",\t" << _numProcs   << ",\t"
           << hmax << ",\t" << hmin << ",\t" << refLvl << ",\t"
           << avgGMRESit << ",\t"  << finalResNorm  << std::endl;
    myfile.close();
  }

  // wait for write to be completed..possibly non-necessary
  MPI_Barrier(_comm);


  // clean up
  delete stokesPr;
  delete myPSchur;
  delete myFu;
  delete Fsolve;
  delete solver;

}


















KHISTOperatorAssembler::~KHISTOperatorAssembler(){
  delete _Ainv;
  delete _MvFinv;
  delete _FFinv;
  delete _FFinvPrec;
  delete _FFF;
  if( _FFAssembled )
    HYPRE_IJMatrixDestroy( _FF );
  if( _AAAssembled )
    HYPRE_IJMatrixDestroy( _AA );
  if( _MMvAssembled )
    HYPRE_IJMatrixDestroy( _MMv );
  if( _BBAssembled )
    HYPRE_IJMatrixDestroy( _BB );
  if( _CCAssembled )
    HYPRE_IJMatrixDestroy( _CC );
  if( _MMwpAssembled )
    HYPRE_IJMatrixDestroy( _MMwp );

  // delete _fwVarf;
  // delete _mwVarf;
  // delete _apVarf;
  // delete _mvVarf;
  // delete _bVarf;
  // delete _cVarf;
  // delete _mwpVarf;

  delete _WhFESpace;
  delete _PhFESpace;
  delete _VhFESpace;
  delete _WhFEColl;
  delete _PhFEColl;
  delete _VhFEColl;

  delete _mesh;
}


























