#include "stokesstoperatorassembler.hpp"

#include <mpi.h>
#include <string>
#include <mpi.h>
#include "HYPRE.h"
#include "petsc.h"
#include "mfem.hpp"

using namespace mfem;


//******************************************************************************
// Space-time block preconditioning
//******************************************************************************
StokesSTPreconditioner::StokesSTPreconditioner( const MPI_Comm& comm, const double dt, const double mu,
                                                const SparseMatrix* Ap = NULL, const SparseMatrix* Mp = NULL, const double tol ):
  _comm(comm), _dt(dt), _mu(mu), _tol(tol),
  _Ap(Ap), _Mp(Mp){

  if( Ap != NULL ){
    height = Ap->Height();
    width  = Ap->Width();
  }else if( Mp != NULL ){
    height = Mp->Height();
    width  = Mp->Width();
  }

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



SpaceTimeSolver::SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F = NULL, const SparseMatrix* M = NULL, const double tol ):
  _comm(comm), _tol(tol), _F(F), _M(M), _X(NULL), _Y(NULL){

  if( F != NULL ){
    height = F->Height();
    width  = F->Width();
  }else if( M != NULL ){
    height = M->Height();
    width  = M->Width();
  }

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}




// Define multiplication by preconditioner (ie, approx inverse of schur complement)
// This is defined as XX^-1 = D(Mp)^-1 * FFp * D(Ap)^-1, where
// - D(*) represents the block-diagonal matrix with (*) as blocks
// - Mp is the pressure mass matrix
// - Ap is the pressure "laplacian" (or stabilised/lumped version thereof)
// - FFp is the space-time matrix associated to time-stepping for pressure
// After some algebra, it can be simplified to the block bi-diagonal
// XX^-1 = ⌈ Ap^-1/dt + mu*Mp^-1                            ⌉
//         |      -Ap^-1/dt      Ap^-1/dt + mu*Mp^-1        |
//         |                          -Ap^-1/dt          \\ |
//         ⌊                                             \\ ⌋
// which boils down to a couple of parallel solves with Mp and Ap
void StokesSTPreconditioner::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT(_Ap != NULL, "Pressure 'laplacian' not initialised" );
  MFEM_ASSERT(_Mp != NULL, "Pressure mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  const int     lclSize = x.Size();
  const double* lclData = x.GetData();

  // TODO: do I really need to copy this? Isn't there a way to force lclx to point at lxlData and still be const?
  Vector lclx( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    lclx.GetData()[i] = lclData[i];
  }

  Vector invAxMine( lclSize ), invAxPrev( lclSize ), lcly( lclSize );

  for ( int i = 0; i < lclSize; ++i ){
    invAxMine.GetData()[i] = 0.0;   // shouldn't be necessary, but vargrind complains if data is not set S_S
    invAxPrev.GetData()[i] = 0.0;
    lcly.GetData()[i]      = 0.0;
  }

  // have each processor solve for the "laplacian"
  CG( *_Ap, lclx, invAxMine, 0, lclSize, _tol, _tol );
  invAxMine *= 1./_dt;   //scale by dt


  // send this partial result to the following processor  
  MPI_Request reqSend, reqRecv;

  if( _myRank < _numProcs ){
    MPI_Isend( invAxMine.GetData(), lclSize, MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
  }
  if( _myRank > 0 ){
    MPI_Irecv( invAxPrev.GetData(), lclSize, MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
  }


  // have each processor solve for the mass matrix
  CG( *_Mp, lclx, lcly, 0, lclSize, _tol, _tol );

  // combine all partial results together locally (once received necessary data, if necessary)
  lcly *= _mu;
  lcly += invAxMine;
  if( _myRank > 0 ){
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
    lcly -= invAxPrev;
  }

  // assemble global vector
  for ( int i = 0; i < lclSize; ++i ){
    y.GetData()[i] = lcly.GetData()[i];
  }

  // lest we risk destroying invAxMine before it's been sent (probably unnecessary)
  // if( _myRank < _numProcs ){
  //   MPI_Wait( &reqSend, MPI_STATUS_IGNORE ); // this triggers a memory error on reqSend, for a reason...
  // }
  MPI_Barrier( _comm );                         // but the barrier should do the same trick, and this seems to work

}






// Time-stepping on velocity block
void SpaceTimeSolver::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT( _F != NULL, "Velocity spatial operator not initialised" );
  MFEM_ASSERT( _M != NULL, "Velocity mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  const int spaceDofs = x.Size();

  // convert data to internal HypreParVector for ease of use
  auto x_data = x.HostRead();
  auto y_data = y.HostReadWrite();
  if ( _X == NULL){
    int starts[2] = { spaceDofs*_myRank, spaceDofs*(_myRank+1) };
    _X = new HypreParVector( _comm, spaceDofs * _numProcs, const_cast<double*>(x_data), starts );
    _Y = new HypreParVector( _comm, spaceDofs * _numProcs, y_data,                      starts );
  }else{
    _X->SetData(const_cast<double*>(x_data));
    _Y->SetData(y_data);
  }

  // Broadcast rhs to each proc (overkill, as only proc 0 will play with it, but whatever)
  const Vector *glbRhs = _X->GlobalVector();

  // Initialise local vector containing solution at single time-step
  Vector lclSol( spaceDofs );
  for ( int i = 0; i < spaceDofs; ++i ){
    lclSol.GetData()[i] = 0.0;   // shouldn't be necessary, but vargrind complains if data is not set S_S
  }

  // Mster performs time-stepping and sends solution to other processors
  if ( _myRank == 0 ){

    // Thise will contain rhs for each time-step
    Vector b( spaceDofs );
    for ( int i = 0; i < spaceDofs; ++i ){
      b.GetData()[i]      = 0.0;
    }

    // Main time-stepping routine
    for ( int t = 0; t < _numProcs; ++t ){

      // - define rhs for this step (including contribution from sol at previous time-step - see below)
      for ( int i = 0; i < spaceDofs; ++i ){
        b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
      }

      // - solve for current time-step
      CG( *_F, b, lclSol, 0, spaceDofs, _tol, _tol );

      // - send local solution to corresponding processor
      if( t==0 ){
        for ( int j = 0; j < spaceDofs; ++j ){
          _Y->GetData()[j] = lclSol.GetData()[j];
        }
      }else{
        MPI_Send( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t,  _comm );   // TODO: non-blocking + wait before CG solve?
      }

      // - include solution as rhs for next time-step
      if( t < _numProcs-1 ){
        _M->Mult( lclSol, b );
        b.Neg();    //M has negative sign, so flip it
      }

    }

  }else{
    // slaves receive data
    MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank, _comm, MPI_STATUS_IGNORE );
  }

  // make sure we're all done
  MPI_Barrier( _comm );

}



SpaceTimeSolver::~SpaceTimeSolver( ){
  delete _X;
  delete _Y;
}




















// // These functions are useful for the Petsc-defined block preconditioner

// // Context definition
// typedef struct{
//   SparseMatrix *_Ap;
//   SparseMatrix *_Mp;
//   double _dt;
//   double _mu;
//   double _tol;

// }BlockSTPCCtx;


// // Create a context
// PetscErrorCode BlockSTPCCreate( BlockSTPCCtx **ctxout ){
//   BlockSTPCCtx *ctx;
//   PetscErrorCode ierr;

//   ierr = PetscNew( &ctx ); CHKERRQ(ierr);

//   *ctxout = ctx;

//   return 0;
// }


// // Initialise context
// PetscErrorCode BlockSTPCSetUp( PC pc, SparseMatrix& Ap, SparseMatrix& Mp, double dt, double mu, double tol ){
//   BlockSTPCCtx *ctx;
//   PetscErrorCode ierr;

//   ierr = PCShellGetContext( pc, (void**)&ctx ); CHKERRQ(ierr);

//   ctx->_Ap  = &Ap;
//   ctx->_Mp  = &Mp;
//   ctx->_dt  = dt;
//   ctx->_mu  = mu;
//   ctx->_tol = tol;

//   return 0;
// }


// // Destroy context
// PetscErrorCode BlockSTPCDestroy( PC pc ){
//   BlockSTPCCtx *ctx;
//   PetscErrorCode ierr;

//   ierr = PCShellGetContext( pc, (void**)&ctx ); CHKERRQ(ierr);

//   // not much to do here: just free the context allocated with PetscNew
//   ierr = PetscFree( ctx ); CHKERRQ(ierr);

//   return 0;
// }



// // Apply preconditioner
// PetscErrorCode BlockSTPCApply( PC pc, Vec x, Vec y ){
//   BlockSTPCCtx *ctx;
//   PetscErrorCode ierr;

//   ierr = PCShellGetContext( pc, (void**)&ctx ); CHKERRQ(ierr);

//   return 0;  
// }



// // Define multiplication by preconditioner (ie, approx inverse of schur complement)
// // This is defined as XX^-1 = D(Mp)^-1 * FFp * D(Ap)^-1, where
// // - D(*) represents the block-diagonal matrix with (*) as blocks
// // - Mp is the pressure mass matrix
// // - Ap is the pressure "laplacian" (or stabilised/lumped version thereof)
// // - FFp is the space-time matrix associated to time-stepping for pressure
// // After some algebra, it can be simplified to the block bi-diagonal
// // XX^-1 = ⌈ Ap^-1/dt + mu*Mp^-1                            ⌉
// //         |      -Ap^-1/dt      Ap^-1/dt + mu*Mp^-1        |
// //         |                          -Ap^-1/dt          \\ |
// //         ⌊                                             \\ ⌋
// // which boils down to a couple of parallel solves with Mp and Ap
// PetscErrorCode BlockSTPMult( Mat XX, Vec x, Vec y ){
//   BlockSTPCCtx *ctx;
//   PetscErrorCode ierr;

//   ierr = MatShellGetContext( XX, (void**)&ctx ); CHKERRQ(ierr);

//   PetscScalar *lclData;
//   PetscInt lclSize;

//   ierr = VecGetLocalSize( x, &lclSize );
//   ierr = VecGetArray( x, &lclData ); CHKERRQ(ierr);

//   Vector lclx( lclData, lclSize ), invAxMine( lclSize ), invAxPrev( lclSize ), lcly( lclSize );

//   for ( int i = 0; i < lclSize; ++i ){
//     invAxMine.GetData()[i] = 0.0;   // shouldn't be necessary, but vargrind complains if data is not set S_S
//     invAxPrev.GetData()[i] = 0.0;
//   }

//   // have each processor solve for the "laplacian"
//   CG( ctx->_Ap, lclx, invAxMine, 0, lclSize, ctx->_tol, ctx->_tol );
//   invAxMine *= 1./(ctx->_dt);   //scale by dt


//   // send this partial result to the following processor
//   int myRank, numProcs;
//   MPI_Comm comm;
//   PetscObjectGetComm( (PetscObject)x, &comm );
//   MPI_Comm_size( comm, &myRank);
//   MPI_Comm_rank( comm, &numProcs);
  
//   MPI_Request reqSend, reqRecv;

//   if( myRank < numProcs ){
//     MPI_Isend( invAxMine.GetData(), lclSize, MPI_DOUBLE, myRank+1, myRank,   comm, &reqSend );
//   }
//   if( myRank > 0 ){
//     MPI_Irecv( invAxPrev.GetData(), lclSize, MPI_DOUBLE, myRank-1, myRank-1, comm, &reqRecv );
//   }


//   // have each processor solve for the mass matrix
//   CG( ctx->_Mp, lclx, lcly, 0, lclSize, ctx->_tol, ctx->_tol );

//   // combine all partial results together locally (once received necessary data)
//   MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
//   lcly = invAxMine - invAxPrev + (ctx->_mu) * lcly;


//   // asemble global vector
//   Array<int> idx(lclSize);
//   for ( int i = 0; i < lclSize; ++i ){
//     lclSize[i] = i + lclSize * myRank;
//   }
//   ierr = VecSetValuesLocal( y, lclSize.GetData(), const PetscInt ix[], lcly.GetData(), INSERT_VALUES );
//   VecAssemblyBegin( y );
//   VecAssemblyEnd( y ); 


//   // lest we risk destroying invAxMine before it's been sent (probably unnecessary)
//   MPI_Wait( &reqSend, MPI_STATUS_IGNORE );

//   return 0;
// }









//******************************************************************************
// Space-time block assembly
//******************************************************************************
// These functions are useful for assembling all the necessary space-time operators

StokesSTOperatorAssembler::StokesSTOperatorAssembler( const MPI_Comm& comm, const char* meshName, const int refLvl,
                                                      const int ordV, const int ordP, const double dt, const double mu,
		                         							            void(  *f)(const Vector &, double, Vector &),
		                         							            void(  *u)(const Vector &, double, Vector &),
		                         							            double(*p)(const Vector &, double ),
                                                      const double tol ):
	_comm(comm), _dt(dt), _mu(mu), _fFunc(f), _uFunc(u), _pFunc(p), _tol(tol),
  _MuAssembled(false), _FuAssembled(false), _MpAssembled(false), _ApAssembled(false), _BAssembled(false),
  _pSchur( comm, dt, mu, NULL, NULL, tol ),
  _FFinv( comm, NULL, NULL, tol ),
  _FFAssembled(false), _BBAssembled(false), _pSAssembled(false), _FFinvAssembled(false){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

	// For each processor:
	//- generate mesh
	_mesh = new Mesh( meshName, 1, 1 );
  
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  _dim = _mesh->Dimension();

  // - initialise FE info
  _VhFEColl  = new H1_FECollection( ordV, _dim );
  _QhFEColl  = new H1_FECollection( ordP, _dim );
  _VhFESpace = new FiniteElementSpace( _mesh, _VhFEColl, _dim );
  _QhFESpace = new FiniteElementSpace( _mesh, _QhFEColl );

   if (_myRank == 0 ){
      std::cout << "***********************************************************\n";
      std::cout << "dim(Vh) = " << _VhFESpace->GetTrueVSize() << "\n";
      std::cout << "dim(Qh) = " << _QhFESpace->GetTrueVSize() << "\n";
      std::cout << "***********************************************************\n";
   }

}



// Assemble operator on main diagonal of space-time matrix for velocity block:
//  Fu = M + mu*dt K
void StokesSTOperatorAssembler::AssembleFu( ){
  if( _FuAssembled ){
    return;
  }

  Array<int> essVhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
  }

  BilinearForm *fVarf( new BilinearForm(_VhFESpace) );
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
	fVarf->AddDomainIntegrator(new VectorMassIntegrator( one ));
	fVarf->AddDomainIntegrator(new VectorDiffusionIntegrator( muDt ));
  fVarf->Assemble();
  fVarf->Finalize();
  
  fVarf->FormSystemMatrix( essVhTDOF, _Fu );
  // - once the matrix is generated, we can get rid of the operator
  _Fu.SetGraphOwner(true);
  _Fu.SetDataOwner(true);
  fVarf->LoseMat();
  delete fVarf;

  _FuAssembled = true;

}



// Assemble operator on subdiagonal of space-time matrix for velocity block:
//  Mu = -M
void StokesSTOperatorAssembler::AssembleMu( ){
  if( _MuAssembled ){
    return;
  }


  Array<int> essVhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
  }

	BilinearForm *mVarf( new BilinearForm(_VhFESpace) );
  ConstantCoefficient mone( -1.0 );
	mVarf->AddDomainIntegrator(new VectorMassIntegrator( mone ));
  mVarf->Assemble();
  mVarf->Finalize();
  mVarf->FormSystemMatrix( essVhTDOF, _Mu );
  // - once the matrix is generated, we can get rid of the operator
  _Mu.SetGraphOwner(true);
  _Mu.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

  _MuAssembled = true;
}



// Assemble -divergence operator:
//  B = -dt * div
void StokesSTOperatorAssembler::AssembleB( ){

  if( _BAssembled ){
    return;
  }

  Array<int> essVhTDOF;
  Array<int> essQhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
    _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  }

	MixedBilinearForm *bVarf(new MixedBilinearForm( _VhFESpace, _QhFESpace ));
  ConstantCoefficient minusDt( -_dt );
  bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(minusDt) );
  bVarf->Assemble();
  bVarf->Finalize();

  Array<int> emptyVhTDOF;
  Array<int> emptyQhTDOF;

  // bVarf->FormRectangularSystemMatrix( essQhTDOF, essVhTDOF, B );	//TODO: imposing essential nodes causes memory corruption??
  bVarf->FormRectangularSystemMatrix( emptyQhTDOF, emptyVhTDOF, _B );
	// - once the matrix is generated, we can get rid of the operator
  _B.SetGraphOwner(true);
  _B.SetDataOwner(true);
  bVarf->LoseMat();
  delete bVarf;

  _BAssembled = true;
}





// Assemble "laplacian" operator for pressure block:
void StokesSTOperatorAssembler::AssembleAp( ){

  if( _ApAssembled ){
    return;
  }

  Array<int> essQhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    // _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
    _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  }

  BilinearForm *aVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient one( 1.0 );
	aVarf->AddDomainIntegrator(new DiffusionIntegrator( one ));
  aVarf->Assemble();
  aVarf->Finalize();
  
  aVarf->FormSystemMatrix( essQhTDOF, _Ap );
  // - once the matrix is generated, we can get rid of the operator
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf->LoseMat();
  delete aVarf;

  _ApAssembled = true;
}



// Assemble operator on subdiagonal of space-time matrix for pressure block:
//  Mp = M
void StokesSTOperatorAssembler::AssembleMp( ){

  if( _MpAssembled ){
    return;
  }

  Array<int> essQhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  }

	BilinearForm *mVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient one( 1.0 );
	mVarf->AddDomainIntegrator(new MassIntegrator( one ));
  mVarf->Assemble();
  mVarf->Finalize();
  mVarf->FormSystemMatrix( essQhTDOF, _Mp );
  // - once the matrix is generated, we can get rid of the operator
  _Mp.SetGraphOwner(true);
  _Mp.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

  _MpAssembled = true;

}










// Solves Space-time system for velocity via SEQUENTIAL time-stepping
void StokesSTOperatorAssembler::TimeStepVelocity( const HypreParVector& rhs, HypreParVector*& sol ){

  AssembleFu();
  AssembleMu();

  TimeStep( _Fu, _Mu, rhs, sol );
}



// Solves Space-time system for pressure via SEQUENTIAL time-stepping
void StokesSTOperatorAssembler::TimeStepPressure( const HypreParVector& rhs, HypreParVector*& sol ){

  AssembleAp();
  AssembleMp();

  SparseMatrix Fp = _Ap;
  Fp *= _mu * _dt;
  Fp.Add( 1.0, _Mp );     // TODO: check that Mp falls into the sparsity pattern of A
  SparseMatrix Mp = _Mp;
  Mp *= -1.0;

  TimeStep( Fp, Mp, rhs, sol );
}





// Actual Time-stepper. Reuses code for both pressure and velocity solve
void StokesSTOperatorAssembler::TimeStep( const SparseMatrix &F, const SparseMatrix &M, const HypreParVector& rhs, HypreParVector*& sol ){

  const int spaceDofs = rhs.Size();

  // Broadcast rhs to each proc (overkill, as only proc 0 will play with it, but whatever)
  const Vector *glbRhs = rhs.GlobalVector();

  // Initialise local vector containing solution at single time-step
  Vector lclSol( spaceDofs );


  // Mster performs time-stepping and sends solution to other processors
  if ( _myRank == 0 ){

    // These will contain rhs and sol for each time-step
    Vector b( spaceDofs ), x( spaceDofs );
    for ( int i = 0; i < spaceDofs; ++i ){
      b.GetData()[i] = 0.0;
      x.GetData()[i] = 0.0;   // shouldn't be necessary, but vargrind complains if data is not set S_S
    }

    // Main time-stepping routine
    for ( int t = 0; t < _numProcs; ++t ){

      // - define rhs for this step
      for ( int i = 0; i < spaceDofs; ++i ){
        b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
      }

      // - solve for current time-step
      CG( F, b, x, 0, spaceDofs, _tol, _tol );

      // - send local solution to corresponding processor
      if( t==0 ){
        lclSol = x;
      }else{
        MPI_Send( x.GetData(), spaceDofs, MPI_DOUBLE, t, t,  _comm );   // TODO: non-blocking + wait before CG solve?
      }

      // - include solution as rhs for next time-step
      if( t < _numProcs-1 ){
        M.Mult( x, b );
        b.Neg();    //M has negative sign, so flip it
      }

    }

  }else{
    // slaves receive data
    MPI_Recv( lclSol.GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank, _comm, MPI_STATUS_IGNORE );
  }

  // make sure we're all done
  MPI_Barrier( _comm );

  // collect space-time solution
  int col[2] = { spaceDofs * _myRank, spaceDofs * (_myRank+1) };
  sol = new HypreParVector( _comm, spaceDofs * _numProcs, lclSol.StealData(), col );


}











// Assemble FF (top-left)
void StokesSTOperatorAssembler::AssembleFF( ){ 
  
  if( _FFAssembled ){
    return;
  }


  // For each processor, define main operators
  // - main diagonal = M + mu*dt K
  AssembleFu();

  // - subidagonal = -M
  AssembleMu();


  // Create FF block ********************************************************
  // Initialize HYPRE matrix
  // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
  //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
  // - get info on matrix structure
  const int blockSizeFF = _Fu.NumRows();
 
  Array<int> nnzPerRowD( blockSizeFF );   // num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<int> nnzPerRowO( blockSizeFF );   // ..and in off-diagonal block
  const int  *offIdxsD = _Fu.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  const int  *offIdxsO = _Mu.GetI();
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
  Array<int> colsGlbIdxD( _Fu.NumNonZeroElems() );
  for ( int i=0; i<_Fu.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = _Fu.GetJ()[i] + blockSizeFF*_myRank;
  }
  HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowD.GetData(),
                           rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), _Fu.GetData() );     // setvalues *copies* the data

  // -- off-diagonal block
  Array<int> rowsGlbIdxO( blockSizeFF );      // TODO: just use rowsGlbIdx once for both matrices?
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxO[i] = i + blockSizeFF*_myRank;
  }
  if ( _myRank > 0 ){
    Array<int> colsGlbIdxO( _Mu.NumNonZeroElems() );
    for ( int i=0; i<_Mu.NumNonZeroElems(); i++ ) {
      colsGlbIdxO[i] = _Mu.GetJ()[i] + blockSizeFF*(_myRank-1);
    }
    HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowO.GetData(),
                             rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), _Mu.GetData() );
  }


  // - assemble
  HYPRE_IJMatrixAssemble( _FF );
  _FFAssembled = true;

}












// Assemble BB (bottom-left)
void StokesSTOperatorAssembler::AssembleBB( ){ 

  if(_BBAssembled){
    return;
  }

  // For each processor, define -div operator
  AssembleB();


  // Assemble BB and BB^T blocks
  // - recover info on matrix structure
  const int numRowsPerBlockBB = _QhFESpace->GetTrueVSize();
  const int numColsPerBlockBB = _VhFESpace->GetTrueVSize();

  Array<int> nnzPerRow( numRowsPerBlockBB );    // num of non-zero els per row in main (diagonal) block (for preallocation)
  const int  *offIdxs = _B.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  for ( int i = 0; i < numRowsPerBlockBB; ++i ){
    nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
  }


  // - initialise matrix
  HYPRE_IJMatrixCreate( _comm, numRowsPerBlockBB*_myRank, numRowsPerBlockBB*(_myRank+1)-1,
                               numColsPerBlockBB*_myRank, numColsPerBlockBB*(_myRank+1)-1, &_BB );
  HYPRE_IJMatrixSetObjectType( _BB, HYPRE_PARCSR );
  HYPRE_IJMatrixSetRowSizes( _BB, nnzPerRow.GetData() );
  HYPRE_IJMatrixInitialize( _BB );


  // - fill it with matrices assembled above
  Array<int> rowsGlbIdxBB( numRowsPerBlockBB );
  for ( int i = 0; i < numRowsPerBlockBB; ++i ){
    rowsGlbIdxBB[i] = i + numRowsPerBlockBB*_myRank;
  }
  Array<int> colsGlbIdx( _B.NumNonZeroElems() );
  for ( int i=0; i<_B.NumNonZeroElems(); i++ ) {
    colsGlbIdx[i] = _B.GetJ()[i] + numColsPerBlockBB*_myRank;
  }
  HYPRE_IJMatrixSetValues( _BB, numRowsPerBlockBB, nnzPerRow.GetData(),
                           rowsGlbIdxBB.GetData(), colsGlbIdx.GetData(), _B.GetData() );



  // - assemble
  HYPRE_IJMatrixAssemble( _BB );
  _BBAssembled = true;

}












// Assemble XX (bottom-right in preconditioner)
void StokesSTOperatorAssembler::AssemblePS(){
  if ( _pSAssembled ){
    return;
  }

  AssembleAp();
  AssembleMp();

  _pSchur.SetAp( &_Ap );
  _pSchur.SetMp( &_Mp );

}




// Assemble FF^-1 (top-left in preconditioner)
void StokesSTOperatorAssembler::AssembleFFinv(){
  if ( _FFinvAssembled ){
    return;
  }

  AssembleFu();
  AssembleMu();

  _FFinv.SetF( &_Fu );
  _FFinv.SetM( &_Mu );

}












void StokesSTOperatorAssembler::AssembleRhs( HypreParVector*& frhs ){
  // Initialise handy functions
  VectorFunctionCoefficient fFuncCoeff(_dim,_fFunc);
  // - specify evaluation time
  fFuncCoeff.SetTime( _dt*(_myRank+1) );

  // Assemble local part of rhs
  LinearForm *fform( new LinearForm );
  fform->Update( _VhFESpace );
  fform->AddDomainIntegrator( new VectorDomainLFIntegrator( fFuncCoeff ) );                //int_\Omega f*v
  // fform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n + \mu*grad u*v *n (remember to put a minus in the pFuncCoeff definition above)
  fform->Assemble();
  Vector fRhsLoc( fform->Size() );
  fRhsLoc.SetData( fform->StealData() );
  fRhsLoc *= _dt;
  delete fform;     // once data is stolen, we can delete the linear form


  // - include initial conditions
  if( _myRank == 0 ){
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    uFuncCoeff.SetTime( 0.0 );
    LinearForm *u0form( new LinearForm );
    u0form->Update( _VhFESpace );
    u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) ); //int_\Omega u0*v
    u0form->Assemble();

    fRhsLoc += *u0form;
    delete u0form;
  }


  // Assemble global (parallel) rhs
  // Array<HYPRE_Int> rowStarts(2);
 //  rowStarts[0] = ( fRhsLoc.Size() )*_myRank;
 //  rowStarts[1] = ( fRhsLoc.Size() )*(_myRank+1);
  // HypreParVector *frhs = new HypreParVector( _comm, (fRhsLoc.Size())*_numProcs, fRhsLoc.GetData(), rowStarts.GetData() );
  // HypreParVector *frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
  // frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
  int colPart[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPart );
  frhs->SetOwnership( 1 );

}









// Assemble space-time Stokes operator
//   A = [ FF  BB^T ]
//       [ BB  0    ],
// where FF contains space-time matrix for velocity,
// while BB is block-diagonal with -div operator in it  
void StokesSTOperatorAssembler::AssembleOperator( HypreParMatrix*& FFF, HypreParMatrix*& BBB ){

  //Assemble top-left block
  AssembleFF();

  // - convert to mfem operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( _FF, (void **) &FFref);
  FFF = new HypreParMatrix( FFref, false ); //"false" doesn't take ownership of data



  //Assemble bottom-left block
  AssembleBB();

  // - convert to mfem operator
  HYPRE_ParCSRMatrix  BBref;
  HYPRE_IJMatrixGetObject( _BB, (void **) &BBref);
  BBB = new HypreParMatrix( BBref, false ); //"false" doesn't takes ownership of data

}












// // Assemble space-time Stokes operator
// //   A = [ FF  BB^T ]
// //       [ BB  0    ],
// // where FF contains space-time matrix for velocity,
// // while BB is block-diagonal with -div operator in it	
// // void StokesSTOperatorAssembler::AssembleSystem( BlockOperator*& stokesOp, BlockVector*& rhs ){
// void StokesSTOperatorAssembler::AssembleSystem( HypreParMatrix*& FFF, HypreParMatrix*& BBB, 
//                                                 HypreParVector*& frhs ){

// 	// Define general structure of time-dep Stokes operator
// 	// Array<int> block_offsets(3); // number of variables + 1
//  //  block_offsets[0] = 0;
//  //  block_offsets[1] = _VhFESpace->GetTrueVSize(); // * _numProcs; TODO: yeah, I know the actual size is different, but seems like it wants size on single proc.
//  //  block_offsets[2] = _QhFESpace->GetTrueVSize(); // * _numProcs;
//  //  block_offsets.PartialSum();

// 	// stokesOp = new BlockOperator( block_offsets );
//  //  rhs      = new BlockVector(   block_offsets );


//  //  //*************************************************************************
// 	// // Fill FF (top-left)
// 	// //*************************************************************************
//  //  if( !_FFAssembled ){
//  //  	// For each processor, define main operators
//  //  	// - main diagonal = M + mu*dt K
//  //  	AssembleFu();

//  //  	// - subidagonal = -M
//  //  	AssembleMu();


//  //    // Create FF block ********************************************************
//  //    // Initialize HYPRE matrix
//  //    // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
//  //    //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
//  //    // - get info on matrix structure
//  //  	const int blockSizeFF = _Fu.NumRows();
   
//  //    Array<int> nnzPerRowD( blockSizeFF );  	// num of non-zero els per row in main (diagonal) block (for preallocation)
//  //    Array<int> nnzPerRowO( blockSizeFF );  	// ..and in off-diagonal block
//  //    const int  *offIdxsD = _Fu.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
//  //    const int  *offIdxsO = _Mu.GetI();
//  //    for ( int i = 0; i < blockSizeFF; ++i ){
//  //    	nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
//  //      if ( _myRank > 0 ){
//  //      	nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
//  //      }else{
//  //        nnzPerRowO[i] = 0;  // first block only has elements on block-diag
//  //      }
//  //    }


//  //    // - initialise matrix
//  //    HYPRE_IJMatrixCreate( _comm, blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1,
//  //                                 blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1, &_FF );
//  //    HYPRE_IJMatrixSetObjectType( _FF, HYPRE_PARCSR );
//  //    HYPRE_IJMatrixSetDiagOffdSizes( _FF, nnzPerRowD.GetData(), nnzPerRowO.GetData() );    // this gives issues :/
//  //    HYPRE_IJMatrixInitialize( _FF );


//  //    // - fill it with matrices assembled above
//  //    // -- diagonal block
//  //    Array<int> rowsGlbIdxD( blockSizeFF );
//  //    for ( int i = 0; i < blockSizeFF; ++i ){
//  //      rowsGlbIdxD[i] = i + blockSizeFF*_myRank;
//  //    }
//  //    Array<int> colsGlbIdxD( _Fu.NumNonZeroElems() );
//  //    for ( int i=0; i<_Fu.NumNonZeroElems(); i++ ) {
//  //      colsGlbIdxD[i] = _Fu.GetJ()[i] + blockSizeFF*_myRank;
//  //    }
//  //    HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowD.GetData(),
//  //    	                       rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), _Fu.GetData() );     // setvalues *copies* the data

//  //    // -- off-diagonal block
//  //    Array<int> rowsGlbIdxO( blockSizeFF );      // TODO: just use rowsGlbIdx once for both matrices?
//  //    for ( int i = 0; i < blockSizeFF; ++i ){
//  //      rowsGlbIdxO[i] = i + blockSizeFF*_myRank;
//  //    }
//  //    if ( _myRank > 0 ){
//  //      Array<int> colsGlbIdxO( _Mu.NumNonZeroElems() );
//  //      for ( int i=0; i<_Mu.NumNonZeroElems(); i++ ) {
//  //        colsGlbIdxO[i] = _Mu.GetJ()[i] + blockSizeFF*(_myRank-1);
//  //      }
//  //      HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowO.GetData(),
//  //      	                       rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), _Mu.GetData() );
//  //    }


//  //    // - assemble
//  //    HYPRE_IJMatrixAssemble( _FF );
//  //    _FFAssembled = true;

//  //  }


//   AssembleFF();

// 	// - convert to mfem operator
//   // HypreParMatrix *FFF = new HypreParMatrix( FFref, true ); //"true" takes ownership of data
//   HYPRE_ParCSRMatrix  FFref;
//   HYPRE_IJMatrixGetObject( _FF, (void **) &FFref);
//   FFF = new HypreParMatrix( FFref, false ); //"false" doesn't take ownership of data




//  //  //*************************************************************************
// 	// // Fill BB and BB^T (bottom-left / top-right)
// 	// //*************************************************************************

//  //  if(!_BBAssembled){
//  //  	// For each processor, define -div operator
//  //  	AssembleB();


//  //    // Assemble BB and BB^T blocks
//  //    // - recover info on matrix structure
//  //    const int numRowsPerBlockBB = _QhFESpace->GetTrueVSize();
//  //    const int numColsPerBlockBB = _VhFESpace->GetTrueVSize();

//  //    Array<int> nnzPerRow( numRowsPerBlockBB );  	// num of non-zero els per row in main (diagonal) block (for preallocation)
//  //    const int  *offIdxs = _B.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
//  //    for ( int i = 0; i < numRowsPerBlockBB; ++i ){
//  //    	nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
//  //    }


//  //    // - initialise matrix
//  //    HYPRE_IJMatrixCreate( _comm, numRowsPerBlockBB*_myRank, numRowsPerBlockBB*(_myRank+1)-1,
//  //                                 numColsPerBlockBB*_myRank, numColsPerBlockBB*(_myRank+1)-1, &_BB );
//  //    HYPRE_IJMatrixSetObjectType( _BB, HYPRE_PARCSR );
//  //    HYPRE_IJMatrixSetRowSizes( _BB, nnzPerRow.GetData() );
//  //    HYPRE_IJMatrixInitialize( _BB );


//  //    // - fill it with matrices assembled above
//  //    Array<int> rowsGlbIdxBB( numRowsPerBlockBB );
//  //    for ( int i = 0; i < numRowsPerBlockBB; ++i ){
//  //    	rowsGlbIdxBB[i] = i + numRowsPerBlockBB*_myRank;
//  //    }
//  //    Array<int> colsGlbIdx( _B.NumNonZeroElems() );
//  //    for ( int i=0; i<_B.NumNonZeroElems(); i++ ) {
//  //      colsGlbIdx[i] = _B.GetJ()[i] + numColsPerBlockBB*_myRank;
//  //    }
//  //    HYPRE_IJMatrixSetValues( _BB, numRowsPerBlockBB, nnzPerRow.GetData(),
//  //    	                       rowsGlbIdxBB.GetData(), colsGlbIdx.GetData(), _B.GetData() );



//  //    // - assemble
//  //    HYPRE_IJMatrixAssemble( _BB );
//  //    _BBAssembled = true;

//  //  }

//   AssembleBB();

// 	// - convert to mfem operator
// 	// HypreParMatrix *BBB = new HypreParMatrix( BBref, true ); //"true" takes ownership of data
//   HYPRE_ParCSRMatrix  BBref;
//   HYPRE_IJMatrixGetObject( _BB, (void **) &BBref);
//   BBB = new HypreParMatrix( BBref, false ); //"false" doesn't takes ownership of data
//   // HypreParMatrix *BBt = BBB->Transpose( );                 //TODO: does it reference the same data as in BBB?




// 	// // store in the stokes space-time operator
//  //  stokesOp->SetBlock( 0, 1, BBt );
//  //  stokesOp->SetBlock( 1, 0, BBB );



//  //  // Clean up
//  //  // HYPRE_IJMatrixDestroy( BB );
//  //  // - set stokeOp as the owner of its own blocks
//  //  stokesOp->owns_blocks = true;
//  //  BBB->SetOwnerFlags( false, false, false );
//  //  BBt->SetOwnerFlags( false, false, false );
//  //  FFF->SetOwnerFlags( false, false, false );

//  // //  // - clean up
//  // //  HYPRE_IJMatrixDestroy( BB );
//  //  // delete FFF;
// 	// // delete BBB;
// 	// // delete BBt;







//   //*************************************************************************
// 	// Assemble rhs
// 	//*************************************************************************
//   // Initialise handy functions
//   // FunctionCoefficient       pFuncCoeff(     _pFunc);
//   // VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
// 	VectorFunctionCoefficient fFuncCoeff(_dim,_fFunc);
// 	// - specify evaluation time
// 	// pFuncCoeff.SetTime( _dt*(_myRank+1) );
// 	// uFuncCoeff.SetTime( _dt*(_myRank+1) );
// 	fFuncCoeff.SetTime( _dt*(_myRank+1) );

// 	// Assemble local part of rhs
//   LinearForm *fform( new LinearForm );
//   fform->Update( _VhFESpace );
//   fform->AddDomainIntegrator( new VectorDomainLFIntegrator( fFuncCoeff ) );					       //int_\Omega f*v
//   // fform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n + \mu*grad u*v *n (remember to put a minus in the pFuncCoeff definition above)
//   fform->Assemble();
//   Vector fRhsLoc( fform->Size() );        // should be blockSizeFF
//   fRhsLoc.SetData( fform->StealData() );
//   fRhsLoc *= _dt;
//   delete fform;			// once data is stolen, we can delete the linear form


// 	// - include initial conditions
// 	if( _myRank == 0 ){
//     VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
// 		uFuncCoeff.SetTime( 0.0 );
// 	  LinearForm *u0form( new LinearForm );
// 	  u0form->Update( _VhFESpace );
// 	  u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) ); //int_\Omega u0*v
// 	  u0form->Assemble();

// 	  fRhsLoc += *u0form;
//  	  delete u0form;
// 	}


// 	// Assemble global (parallel) rhs
// 	// Array<HYPRE_Int> rowStarts(2);
//  //  rowStarts[0] = ( fRhsLoc.Size() )*_myRank;
//  //  rowStarts[1] = ( fRhsLoc.Size() )*(_myRank+1);
//   // HypreParVector *frhs = new HypreParVector( _comm, (fRhsLoc.Size())*_numProcs, fRhsLoc.GetData(), rowStarts.GetData() );
//   // HypreParVector *frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
//   // frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
//   frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.StealData(), FFF->ColPart() );
//   frhs->SetOwnership( 1 );



//   // - store in rhs
//   // rhs->GetBlock( 0 ).SetData( frhs->StealData() );




// // hypre_ParCSRMatrixOwnsData(     FFref) = false;
// // hypre_ParCSRMatrixOwnsRowStarts(FFref) = false;
// // hypre_ParCSRMatrixOwnsColStarts(FFref) = false;
// // hypre_ParCSRMatrixOwnsData(     BBref) = false;
// // hypre_ParCSRMatrixOwnsRowStarts(BBref) = false;
// // hypre_ParCSRMatrixOwnsColStarts(BBref) = false;
// // FFF->SetOwnerFlags(true, true, true);
// // BBB->SetOwnerFlags(true, true, true);
// // HYPRE_IJMatrixDestroy( FF );
// // HYPRE_IJMatrixDestroy( BB );

// // HYPRE_ParCSRMatrixDestroy ( FFref );


// // TODO: this is me trying to figure out what the hell is going on...
// // {  SparseMatrix diag;
// //   FFF->GetDiag( diag );
// //   if ( _myRank == 0 ){
// //     for ( int i = 0; i < diag.NumRows(); ++i ){
// //       std::cout<<"Row: "<<i<<"-";
// //       for ( int j = diag.GetI()[i]; j < diag.GetI()[i+1]; ++j ){
// //         std::cout<<" Col "<<diag.GetJ()[j]<<": "<<diag.GetData()[j];
// //       }
// //       std::cout<<std::endl;
// //     }
// //   }
// // }
// //     {int uga;
// //     std::cin>>uga;
// //     MPI_Barrier( MPI_COMM_WORLD );}


// // {  HypreParVector buga( *FFF, 1 );
// //   FFF->Mult( *frhs, buga );
// //   if ( _myRank==0 ){
// //     for ( int i = 0; i < buga.Partitioning()[1] - buga.Partitioning()[0]; ++i ){
// //       std::cout<<"Rank "<<_myRank<<": "<<buga.GetData()[i]<<std::endl;
// //     }
// //   }
// // }

// //     {int uga;
// //     std::cin>>uga;
// //     MPI_Barrier( MPI_COMM_WORLD );}








//   /* 
//   // Initialise Petsc matrix
//   // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
//   //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
//   // - get info on matrix structure
//   const PetscInt blockSizeFF = F.NumRows();
//   const PetscInt glbSizeFF   = F.NumRows() * _numProcs;
  

//   Array<PetscInt> nnzPerRowD( blockSizeFF );  	// num of non-zero els per row in main (diagonal) block (for preallocation)
//   Array<PetscInt> nnzPerRowO( blockSizeFF );  	// ..and in off-diagonal block
//   int  *offIdxsD = F.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
//   int  *offIdxsO = M.GetI();
//   for ( int i = 0; i < blockSizeFF; ++i ){
//   	nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
//   	nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
//   }



//   // - initialise matrix
// 	PetscErrorCode petscErr;
//   Mat FF;
//   petscErr = MatCreateAIJ( _comm, blockSizeFF, blockSizeFF, glbSizeFF, glbSizeFF, 
//   	                       nnzPerRowD.Max(), nnzPerRowD.GetData(),
//   	                       nnzPerRowO.Max(), nnzPerRowO.GetData(), &FF ); CHKERRV(petscErr);
  
//   // - fill it with data collected above - one row at a time
//   // -- diagonal block
//   for ( PetscInt i = 0; i < blockSizeFF; ++i ){
//   	const PetscInt rowGlbIdx =  i + blockSizeFF * _myRank;
//   	Array<PetscInt> colGlbIdx( nnzPerRowD[i] );
//   	for ( int j = 0; j < nnzPerRowD[i]; ++j ){
//   		colGlbIdx[j] = (F.GetJ())[ offIdxsD[i] +j ] + blockSizeFF * _myRank; 
//   	}
// 	  petscErr = MatSetValues( FF,         1, &rowGlbIdx,
// 	  	                       nnzPerRowD[i], colGlbIdx.GetData(),
// 	  	                       &((F.GetData())[offIdxsD[i]]), INSERT_VALUES ); CHKERRV(petscErr);
//   }
//   // -- off-diagonal block
//   for ( PetscInt i = 0; i < blockSizeFF; ++i ){
//   	const PetscInt rowGlbIdx =  i + blockSizeFF * _myRank;
//   	Array<PetscInt> colGlbIdx( nnzPerRowO[i] );
//   	for ( int j = 0; j < nnzPerRowO[i]; ++j ){
//   		colGlbIdx[j] = (M.GetJ())[ offIdxsO[i] +j ] + blockSizeFF * (_myRank-1); //will be skipped for _myRank==0
//   	}
// 	  petscErr = MatSetValues( FF,         1, &rowGlbIdx,
// 	  	                       nnzPerRowO[i], colGlbIdx.GetData(),
// 	  	                       &((M.GetData())[offIdxsO[i]]), INSERT_VALUES ); CHKERRV(petscErr);
//   }

//   // - assemble
// 	petscErr = MatAssemblyBegin( FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);
// 	petscErr = MatAssemblyEnd(   FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);


// 	// convert to mfem operator
// 	PetscParMatrix *FFF = new PetscParMatrix( FF, true ); //"true" increases ref counts: now even if FF dies, there should be no memory loss


// 	// // finally store in the stokes space-time operator
// 	stokesOp->SetBlock( 0, 0, FFF );

//   petscErr = MatDestroy( &FF ); CHKERRV(petscErr);*/












//   /* Yeah, it'd be nice to use block matrices, but it seems like building them is a pain: delete this code
//   PetscErrorCode petscErr;
//   Mat FF;
//   const int maxNZBlocksPerRow = 2;
//   const int glbSizeFF   = _VhFESpace->GetVSize()*_numProcs;
//   const int blockSizeFF = _VhFESpace->GetVSize();
//   petscErr = MatCreateBAIJ( _comm, blockSizeFF, blockSizeFF, blockSizeFF, glbSizeFF, glbSizeFF,
//                             1, NULL, 1, NULL, &FF ); CHKERRV(petscErr);

//   // petscErr = MatCreateBlockMat( _comm, glbSizeFF, glbSizeFF, blockSizeFF,
//   // 	                            maxNZBlocksPerRow, maxNZPerBlockRow.GetData(), &FF ); CHKERRV(petscErr); // maxNZPerBlockRow is actually nnz blocks, rather than elems?

//   petscErr = MatSetUp( FF ); CHKERRV(petscErr);
  
//   // for each proc, build a map for local 2 global rows and col indeces (block to gl matrix),
//   //  for both the block on the Diagonal and that on the SubDiagonal
// 	ISLocalToGlobalMapping l2gColMapD, l2gRowMapD, l2gColMapSD, l2gRowMapSD;
//   PetscInt *l2gColDIdx, *l2gRowDIdx, *l2gColSDIdx, *l2gRowSDIdx;
//   petscErr = PetscMalloc( sizeof(PetscInt), &l2gRowDIdx);  CHKERRV(petscErr);	// shouldn't need to petscfree if using PETSC_OWN_POINTER in ISLocalToGlobalMappingCreate()
//   petscErr = PetscMalloc( sizeof(PetscInt), &l2gColDIdx);  CHKERRV(petscErr); //  otherwise, just use PETSC_COPY_VALUES and whatever
//   petscErr = PetscMalloc( sizeof(PetscInt), &l2gRowSDIdx); CHKERRV(petscErr);
//   petscErr = PetscMalloc( sizeof(PetscInt), &l2gColSDIdx); CHKERRV(petscErr);

//   *l2gRowDIdx  = _myRank;
//   *l2gColDIdx  = _myRank;
// 	*l2gRowSDIdx = _myRank;
//   *l2gColSDIdx = _myRank-1; // should be invalid for myRank = 1 	

//   petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gRowDIdx,  PETSC_COPY_VALUES, &l2gRowMapD  ); CHKERRV(petscErr);
//   petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gColDIdx,  PETSC_COPY_VALUES, &l2gColMapD  ); CHKERRV(petscErr);
//   petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gRowSDIdx, PETSC_COPY_VALUES, &l2gRowMapSD ); CHKERRV(petscErr);
//   petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gColSDIdx, PETSC_COPY_VALUES, &l2gColMapSD ); CHKERRV(petscErr);
 

// 	petscErr = PetscFree( l2gRowDIdx  ); CHKERRV(petscErr);
// 	petscErr = PetscFree( l2gColDIdx  ); CHKERRV(petscErr);
// 	petscErr = PetscFree( l2gRowSDIdx ); CHKERRV(petscErr);
// 	petscErr = PetscFree( l2gColSDIdx ); CHKERRV(petscErr);


//   // fill each block: main diagonal
// 	petscErr = MatSetLocalToGlobalMapping( FF, l2gRowMapD, l2gColMapD); CHKERRV(petscErr);
//   int  *rowIdxD = F.GetI();
//   int  *colIdxD = F.GetJ();
//   double *dataD = F.GetData();
//   // F.LoseData();	// we can get rid of the matrix now
//   // delete F;
// 	// petscErr = MatSetValuesBlockedLocal( FF, blockSizeFF, rowIdxD, blockSizeFF, colIdxD, dataD, INSERT_VALUES ); CHKERRV(petscErr);

//   petscErr = ISLocalToGlobalMappingDestroy( &l2gRowMapD  ); CHKERRV(petscErr);
//   petscErr = ISLocalToGlobalMappingDestroy( &l2gColMapD  ); CHKERRV(petscErr);
//   petscErr = ISLocalToGlobalMappingDestroy( &l2gRowMapSD ); CHKERRV(petscErr);
//   petscErr = ISLocalToGlobalMappingDestroy( &l2gColMapSD ); CHKERRV(petscErr);



// 	petscErr = MatAssemblyBegin( FF, MAT_FLUSH_ASSEMBLY ); CHKERRV(petscErr);
// 	petscErr = MatAssemblyEnd(   FF, MAT_FLUSH_ASSEMBLY ); CHKERRV(petscErr);

//   // fill each block: sub-diagonal
// 	petscErr = MatSetLocalToGlobalMapping( FF, l2gRowMapSD, l2gColMapSD); CHKERRV(petscErr);
//   int  *rowIdxSD = M.GetI();
//   int  *colIdxSD = M.GetJ();
//   double *dataSD = M.GetData();
//   M.LoseData();	// we can get rid of the matrix now
//   // delete M;
// 	petscErr = MatSetValuesBlockedLocal( FF, blockSizeFF, rowIdxSD, blockSizeFF, colIdxSD, dataSD, INSERT_VALUES ); CHKERRV(petscErr);

// 	// assemble the whole thing
// 	petscErr = MatAssemblyBegin( FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);
// 	petscErr = MatAssemblyEnd(   FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);

//   petscErr = MatDestroy( &FF ); CHKERRV(petscErr); */


// 	// // convert to mfem operator
// 	// PetscParMatrix *FFF = new PetscParMatrix( FF, true ); //"true" increases ref counts: now even if FF dies, there should be no memory loss

// 	// // finally store in the stokes space-time operator
// 	// stokesOp->SetBlock( 0, 0, FFF );



// 	// petscErr = MatView( FF, 	PETSC_VIEWER_STDOUT_(_comm) ); CHKERRV(petscErr);








//   /* Again, Petsc is a pain
//  //  // Assemble BB and BB^T blocks
//  //  // - recover info on matrix structure
//  //  const int glbNumRows = _QhFESpace->GetVSize() * _numProcs;
//  //  const int glbNumCols = _VhFESpace->GetVSize() * _numProcs;
// 	// Array<PetscInt> rowStarts(_numProcs+1);
// 	// Array<PetscInt> colStarts(_numProcs+1);
//  //  rowStarts[0] = 0;  colStarts[0] = 0;
//  //  for ( int i = 1; i < _numProcs+1; ++i ){
//  //  	rowStarts[i] = _QhFESpace->GetVSize();
//  //  	colStarts[i] = _VhFESpace->GetVSize();
//  //  }
//  //  rowStarts.PartialSum();  colStarts.PartialSum();
//  //  // - assemble actual matrix
//   // TODO: seems like this function doesn't build a block-diagonal matrix, with blocks specified in B, but
//   //  rather assumes B is already block diagonal, and somehow it breaks it into a parallel matrix??
// 	// PetscParMatrix *BB = new PetscParMatrix( _comm, glbNumRows, glbNumCols,
// 	// 	                  										 rowStarts.GetData(), colStarts.GetData(),
// 	//                     										 &B, mfem::Operator::PETSC_MATAIJ ); //PETSC_MATNEST is unsupported?
//  //  PetscParMatrix *BBt = BB->Transpose( true );

// 	// if (_myRank == 0 ){
// 	//   std::cout << "***********************************************************\n";
// 	//   std::cout << "B  is a  " << B.NumRows()   << "x" << B.NumCols()   << " matrix\n";
// 	//   std::cout << "BB is a  " << BB->NumRows()  << "x" << BB->NumCols()  << " matrix\n";
// 	//   std::cout << "BBt is a " << BBt->NumRows() << "x" << BBt->NumCols() << " matrix\n";
// 	//   std::cout << "A is a "   << stokesOp->NumRows() << "x" << stokesOp->NumCols() << " matrix\n";
// 	//   // std::cout << "F is a " << F.NumRows() << "x" << F.NumCols() << " matrix\n";
// 	//   // std::cout << "M is a " << M.NumRows() << "x" << M.NumCols() << " matrix\n";
// 	//   std::cout << "***********************************************************\n";
// 	// }

// 	// // finally store in the stokes space-time operator
//  //  stokesOp->SetBlock( 0, 1, BBt);
//  //  stokesOp->SetBlock( 1, 0, BB );

//   // TODO: think about how to deal with data ownership
// 	// BB->ReleaseMat(true);

// 	// stokesOp.owns_block = true;
// 	// // B.LoseData();	// we can get rid of the matrix now
//  //  // delete B;
//  //  PetscParMatrix *BBt = BB->Transpose( true );

// 	// // finally store in the stokes space-time operator
//  //  stokesOp->SetBlock( 0, 1, BBt);
//  //  stokesOp->SetBlock( 1, 0, BB );

//  */



// //  //  clean up
// //  //  delete FF;
// //  //  delete FFF; // should I delete the underlying FFF operator?
// // 	// delete BB;
// // 	// delete BBt;
// }






// Assemble preconditioner
//   P^-1 = [ FF^-1  0     ]
//          [ 0      XX^-1 ],
// where FF contains space-time matrix for velocity,
void StokesSTOperatorAssembler::AssemblePreconditioner( Operator*& FFi, Operator*& XXi ){

  //Assemble top-left block
  AssembleFFinv();
  AssemblePS();

  FFi = &_FFinv;
  XXi = &_pSchur;

}













void StokesSTOperatorAssembler::ExactSolution( HypreParVector*& u, HypreParVector*& p ){
  // Initialise handy functions
  VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  // - specify evaluation time
  // -- notice first processor actually refers to instant dt
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );

  GridFunction uFun( _VhFESpace );
  GridFunction pFun( _QhFESpace );

  uFun.ProjectCoefficient( uFuncCoeff );
  pFun.ProjectCoefficient( pFuncCoeff );
  

  Array<HYPRE_Int> rowStartsV(2), rowStartsQ(2);
  rowStartsV[0] = ( uFun.Size() )*_myRank;
  rowStartsV[1] = ( uFun.Size() )*(_myRank+1);
  rowStartsQ[0] = ( pFun.Size() )*_myRank;
  rowStartsQ[1] = ( pFun.Size() )*(_myRank+1);

  u = new HypreParVector( _comm, (uFun.Size())*_numProcs, uFun.StealData(), rowStartsV.GetData() );
  p = new HypreParVector( _comm, (pFun.Size())*_numProcs, pFun.StealData(), rowStartsQ.GetData() );

}




void StokesSTOperatorAssembler::ComputeL2Error( const HypreParVector& uh, const HypreParVector& ph ){

  const GridFunction u( _VhFESpace, uh.GetData() );
  const GridFunction p( _QhFESpace, ph.GetData() );

  int order_quad = 4;
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i=0; i < Geometry::NumGeom; ++i){
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);

  double err_u  = u.ComputeL2Error(uFuncCoeff, irs);
  double err_p  = p.ComputeL2Error(pFuncCoeff, irs);

  for ( int i = 0; i < _numProcs; ++i ){
    if ( _myRank == i ){
      std::cout << "Instant t="       << _dt*(_myRank+1) << std::endl;
      std::cout << "|| uh - uEx ||_L2= " << err_u << "\n";
      std::cout << "|| ph - pEx ||_L2= " << err_p << "\n";
    }
    MPI_Barrier( _comm );
  }
}




StokesSTOperatorAssembler::~StokesSTOperatorAssembler(){
	delete _VhFESpace;
	delete _QhFESpace;
	delete _VhFEColl;
	delete _QhFEColl;
	delete _mesh;
}







