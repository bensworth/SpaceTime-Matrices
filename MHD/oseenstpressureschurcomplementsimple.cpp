#include "oseenstpressureschurcomplementsimple.hpp"



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
OseenSTPressureSchurComplementSimple::OseenSTPressureSchurComplementSimple( const MPI_Comm& comm, double dt, double mu,
                                                                            const SparseMatrix* Ap, const SparseMatrix* Mp, const SparseMatrix* Fp,
                                                                            const Array<int>& essQhTDOF, int verbose ):
  _comm(comm), _Ap(NULL), _Mp(NULL), _Asolve(NULL), _Msolve(NULL), _essQhTDOF(essQhTDOF), _verbose(verbose){

  if( Ap != NULL ) SetAp(Ap);
  if( Mp != NULL ) SetMp(Ap);
  if( Fp != NULL ) SetFp(Fp);

  // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
  //  to actually make use of the IG or not
  // TODO: delete this. Never use initial guess in preconditioners!
  iterative_mode = true;

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



OseenSTPressureSchurComplementSimple::~OseenSTPressureSchurComplementSimple(){
  delete _Ap;
  delete _Mp;
  delete _Asolve;
  delete _Msolve;
}



// initialise info on pressure 'laplacian'
void OseenSTPressureSchurComplementSimple::SetAp( const SparseMatrix* Ap ){
  delete _Ap;
  _Ap = new PetscParMatrix( Ap );

  height = Ap->Height();
  width  = Ap->Width();
  SetApSolve();
}



// initialise solver for pressure 'laplacian'
void OseenSTPressureSchurComplementSimple::SetApSolve(){
  delete _Asolve;

  _Asolve = new PetscLinearSolver( *_Ap, "PSolverLaplacian_" );
  
}



// initialise info on pressure mass matrix
void OseenSTPressureSchurComplementSimple::SetMp( const SparseMatrix* Mp ){
  delete _Mp;
  _Mp = new PetscParMatrix( Mp );

  height = Mp->Height();
  width  = Mp->Width();
  SetMpSolve();
}

// initialise info on pressure time-stepping operator
void OseenSTPressureSchurComplementSimple::SetFp( const SparseMatrix* Fp ){
  _Fp.MakeRef( *Fp );

  height = Fp->Height();
  width  = Fp->Width();
}



// initialise solver for pressure mass matrix
void OseenSTPressureSchurComplementSimple::SetMpSolve(){
  delete _Msolve;

  _Msolve = new PetscLinearSolver( *_Mp, "PSolverMass_" );
  
}



// Define multiplication by preconditioner - the most relevant function here
// This implementation refers to XX assembled starting from the commutation of gradient and PCD operators:
// XX^-1 = - D(Mp)^-1 * FFp * D(Ap)^-1
void OseenSTPressureSchurComplementSimple::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT( _Asolve!=NULL, "Solver for press 'laplacian' not initialised" );
  MFEM_ASSERT( _Msolve!=NULL, "Solver for press mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

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

  // Have each processor solve for the "laplacian"
  Vector invAxMine( y ), invAxPrev( x.Size() ); invAxPrev=0.;
  if ( _Asolve->Height() == this->height ){
    _Asolve->Mult( x, invAxMine );
  }else{
    // I'm considering an "augmented" version of the laplacian, which takes also the lagrangian multiplies for int(p)=0;
    Vector xAug( x.Size()+1 );
    Vector invAxMineAug( y.Size()+1 );
    for ( int i = 0; i < x.Size(); ++i ){
      xAug(i) = x(i);
      invAxMineAug(i) = invAxMine(i);
    }
    xAug(x.Size()) = 0.;
    invAxMineAug(x.Size()) = 0.;
    _Asolve->Mult( xAug, invAxMineAug );
    for ( int i = 0; i < x.Size(); ++i ){
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

  // Include contribution from pressure (convection)-diffusion operator
  Vector FpinvAx( invAxMine );
  _Fp.Mult( invAxMine, FpinvAx );


  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" included contribution from pressure (convection)-diffusion operator"<<std::endl;
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after spatial contribution: ";
      FpinvAx.Print(mfem::out, FpinvAx.Size());
    }
    MPI_Barrier(_comm);
  }  


  // Send this partial result to the following processor  
  MPI_Request reqSend, reqRecv;

  if( _myRank < _numProcs && _numProcs>1 ){
    MPI_Isend( invAxMine.GetData(), invAxMine.Size(), MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
  }
  if( _myRank > 0 ){
    MPI_Irecv( invAxPrev.GetData(), invAxMine.Size(), MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
  }


  // Have each processor solve for the mass matrix
  _Msolve->Mult( FpinvAx, y );

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure mass matrix\n";
    if ( _verbose>100 ){
      std::cout<<"Inside P-block: Rank: "<<_myRank<< ", result after inverting mass matrix: ";
      y.Print(mfem::out, y.Size());
    }
    MPI_Barrier(_comm);
  }

  // - if you want to ignore the "time-stepping" structure in the preconditioner (that is,
  //    the contribution from the Mp/dt terms, just comment out the next few lines
  if( _myRank > 0 ){
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );

    // - kill contributions from Dirichlet BC
    invAxPrev.SetSubVector( _essQhTDOF, 0.0 );
    
    y -= invAxPrev;
  }


  // Remember to flip sign! Notice the minus in front of XX^-1
  y.Neg();


  // TODO:
  // This shouldn't be necessary: the dirichlet nodes should already be equal to invAxMine
  //  Actually, not true: I'm flipping the sign on the dirichlet nodes with y.Neg()!
  // THIS IS A NEW ADDITION:
  // - only consider action of Ap^-1 on Dirichlet BC
  for ( int i = 0; i < _essQhTDOF.Size(); ++i ){
    y(_essQhTDOF[i]) *= -1.;
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



} // namespace mfem