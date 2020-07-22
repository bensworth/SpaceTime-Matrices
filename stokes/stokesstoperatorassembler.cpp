#include "stokesstoperatorassembler.hpp"

#include <mpi.h>
#include <string>
#include <iostream>
#include <mpi.h>
#include "HYPRE.h"
#include "petsc.h"
#include "mfem.hpp"

using namespace mfem;

// Seems like multiplying every operator by dt gives slightly better results.
#define MULT_BY_DT

//##############################################################################
//
// SPACE-TIME BLOCK PRECONDITIONING
//
//##############################################################################


//******************************************************************************
// Pressure block
//******************************************************************************
// These functions are used to ais in the application of the pressure part of
//  the block preconditioner (ie, approximating the inverse of pressure schur
//  complement)
// This is defined as: - XX^-1 = - D(Mp)^-1 * FFp * D(Ap)^-1, where:
//  - D(*) represents the block-diagonal matrix with (*) as blocks
//  - FFp is the space-time matrix representing time-stepping on pressure
//  - Mp is the pressure mass matrix
//  - Ap is the pressure "laplacian" (or its stabilised/approximated version)
// After some algebra, it can be simplified to the block bi-diagonal
//          ⌈ Ap^-1 + dt*mu*Mp^-1                          ⌉
// XX^-1 =  |      -Ap^-1          Ap^-1 + dt*mu*Mp^-1     |,
//          |                           -Ap^-1          \\ |
//          ⌊                                           \\ ⌋
//  which boils down to a couple of parallel solves involving Mp and Ap

StokesSTPreconditioner::StokesSTPreconditioner( const MPI_Comm& comm, double dt, double mu,
                                                const SparseMatrix* Ap, const SparseMatrix* Mp,
                                                const Array<int>& essQhTDOF, bool verbose ):
  _comm(comm), _dt(dt), _mu(mu), _Asolve(NULL), _Msolve(NULL), _essQhTDOF(essQhTDOF), _verbose(verbose){

  if( Ap != NULL ) SetAp(Ap);
  if( Mp != NULL ) SetMp(Mp);

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



StokesSTPreconditioner::~StokesSTPreconditioner(){
  delete _Asolve;
  delete _Msolve;
}



// initialise info on pressure 'laplacian'
void StokesSTPreconditioner::SetAp( const SparseMatrix* Ap ){
  _Ap = new PetscParMatrix( Ap );

  // // This option can be passed at runtime with -ksp_constant_null_space TRUE
  // if( _myRank == 0 ){
  //   std::cout<<"Warning: assuming that pressure 'laplacian' has non-trivial kernel (constant functions)"<<std::endl;
  // }
  // Mat petscA = Mat( *_Ap );
  // MatNullSpace nsp;
  // MatNullSpaceCreate( MPI_COMM_SELF, PETSC_TRUE, 0, NULL, &nsp);
  // MatSetNullSpace( petscA, nsp );
  // MatNullSpaceDestroy( &nsp );        // hopefully all info is stored

  height = Ap->Height();
  width  = Ap->Width();
  SetApSolve();
}



// initialise solver for pressure 'laplacian'
void StokesSTPreconditioner::SetApSolve(){
  delete _Asolve;

  _Asolve = new PetscLinearSolver( *_Ap, "PSolverLaplacian_" );

}



// initialise info on pressure mass matrix
void StokesSTPreconditioner::SetMp( const SparseMatrix* Mp ){
  _Mp = new PetscParMatrix( Mp );

  height = Mp->Height();
  width  = Mp->Width();
  SetMpSolve();
}




// initialise solver for pressure mass matrix
void StokesSTPreconditioner::SetMpSolve(){
  delete _Msolve;

  _Msolve = new PetscLinearSolver( *_Mp, "PSolverMass_" );

}



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

  if ( _verbose && _myRank==0 ){
    std::cout<<"Applying pressure block preconditioner"<<std::endl;
  }

  // TODO: do I really need to copy this? Isn't there a way to force lclx to point at lclData and still be const?
  Vector lclx( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    lclx.GetData()[i] = lclData[i];
  }

  Vector invAxMine( lclSize ), invAxPrev( lclSize ), lcly( lclSize );

  for ( int i = 0; i < lclSize; ++i ){
    // TODO: if iterative solvers are set, these act as initial guesses. Improve them!
    invAxMine.GetData()[i] = 0.0;
    lcly.GetData()[i]      = 0.0;
    // invAxPrev.GetData()[i] = 0.0;
  }


  // Have each processor solve for the "laplacian"
  _Asolve->Mult( lclx, invAxMine );

#ifndef MULT_BY_DT
  invAxMine *= (1./_dt);   //divide by dt
#endif

  if (_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
    //   if ( rank==_myRank ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure 'laplacian'"<<std::endl;
        // // print extra info if solver is iterative (not the best programming practice)
        // if (_ASolveType == 0){
        //   const IterativeSolver *temp = dynamic_cast<const IterativeSolver*>( _Asolve );
        //   if (temp->GetConverged()){
        //     std::cout <<": Solver converged in "        << temp->GetNumIterations();
        //   }else{
        //     std::cout <<": Solver did not converge in " << temp->GetNumIterations();
        //   }
        //   std::cout << " iterations. Residual norm is " << temp->GetFinalNorm() << ".\n";
        // }
      // }
    MPI_Barrier(_comm);
    // }
  }


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
    // for ( int rank = 0; rank < _numProcs; ++rank ){
    //   if ( rank==_myRank ){
    std::cout<<"Rank "<<_myRank<<" inverted pressure mass matrix"<<std::endl;
        // // print extra info if solver is iterative (not the best programming practice)
        // if (_MSolveType == 0){
        //   const IterativeSolver *temp = dynamic_cast<const IterativeSolver*>( _Msolve );
        //   if (temp->GetConverged()){
        //     std::cout <<": Solver converged in "         << temp->GetNumIterations();
        //   }else{
        //     std::cout <<": Solver did not converge in "  << temp->GetNumIterations();
        //   }
        //   std::cout << " iterations. Residual norm is " << temp->GetFinalNorm() << ".\n";
        // }
      // }
    MPI_Barrier(_comm);
    // }
  }


  // Combine all partial results together locally (once received necessary data, if necessary)
#ifndef MULT_BY_DT
  lcly *= _mu*_dt;    //remember to include factor mu*dt
#else
  lcly *= _mu;        //remember to include factor mu
#endif

  lcly += invAxMine;
  if( _myRank > 0 ){
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
    // if you want to ignore the "time-stepping" structure in the preconditioner, just comment out this line
    lcly -= invAxPrev;
  }


  // Assemble global vector
  for ( int i = 0; i < lclSize; ++i ){
    // remember to flip sign! Notice the minus in front of XX^-1
    y.GetData()[i] = - lcly.GetData()[i];
  }

  // lest we risk destroying invAxMine before it's been sent (probably unnecessary)
  // if( _myRank < _numProcs ){
  //   MPI_Wait( &reqSend, MPI_STATUS_IGNORE ); // this triggers a memory error on reqSend, for a reason...
  // }
  MPI_Barrier( _comm );                         // ...but the barrier should do the same trick, and this seems to work

}















//******************************************************************************
// Velocity block
//******************************************************************************
// These functions are used to ais in the application of the velocity part of
//  the block preconditioner. That is, it approximates the inverse of the
//  space-time velocity matrix FFu:
//       ⌈ Fu*dt          ⌉
// FFu = |  -M   Fu*dt    |,
//       |        -M   \\ |
//       ⌊             \\ ⌋
//  where Fu = Mu/dt + mu Au is the spatial operator for the velocity

SpaceTimeSolver::SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F, const SparseMatrix* M,
                                  const Array<int>& essVhTDOF, bool verbose ):
  _comm(comm), _Fsolve(NULL), _essVhTDOF(essVhTDOF), _X(NULL), _Y(NULL), _verbose(verbose){

  if( F != NULL ) SetF(F);
  if( M != NULL ) SetM(M);

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



SpaceTimeSolver::~SpaceTimeSolver(){
  delete _Fsolve;
  delete _X;
  delete _Y;
}



// initialise info on spatial operator for the velocity.
void SpaceTimeSolver::SetF( const SparseMatrix* F ){
  _F = new PetscParMatrix( F );

  height = F->Height();
  width  = F->Width();
  SetFSolve();
}

// initialise info on mass-matrix for the velocity.
void SpaceTimeSolver::SetM( const SparseMatrix* M ){
  _M = M;

  height = M->Height();
  width  = M->Width();
}



// initialise solver of spatial operator for the velocity.
void SpaceTimeSolver::SetFSolve(){
  delete _Fsolve;

  // Only master will take care of solving the system: this means that the same
  //  spatial operator is considered throughout the whole time-stepping routine
  //TODO: generalise this
  if (_myRank==0){
    std::cout<<"SpaceTimeSolver: Warning: spatial operator is assumed to be time-independent!"<<std::endl;
    
    _Fsolve = new PetscLinearSolver( *_F, "VSolver_" );

  }

}



// Define multiplication by preconditioner (that is, time-stepping on the
//  velocity space-time block)- the most relevant function here
void SpaceTimeSolver::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT( _Fsolve != NULL, "Solver for velocity spatial operator not initialised" );
  MFEM_ASSERT( _M      != NULL, "Velocity mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  if ( _verbose && _myRank==0 ){
    std::cout<<"Applying velocity block preconditioner (time-stepping)"<<std::endl;
  }
  
  // Initialise
  const int spaceDofs = x.Size();

  // - convert data to internal HypreParVector for ease of use
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

  // - broadcast IG and rhs to each proc (overkill, as only proc 0 will play with it, but whatever)
  const Vector *glbRhs = _X->GlobalVector();
  const Vector *glbIG  = _Y->GlobalVector();

  // - initialise local vector containing solution at single time-step
  Vector lclSol( spaceDofs );
  for ( int i = 0; i < spaceDofs; ++i ){
    lclSol.GetData()[i] = glbIG->GetData()[i];
  }

  // Master performs time-stepping and sends solution to other processors
  if ( _myRank == 0 ){

    // - these will contain rhs for each time-step
    Vector b( spaceDofs );
    b = 0.;


    // Main time-stepping routine
    for ( int t = 0; t < _numProcs; ++t ){

      // - define rhs for this step (including contribution from sol at previous time-step - see below)
      for ( int i = 0; i < spaceDofs; ++i ){
        b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
      }


      // - solve for current time-step
      //  --if an iterative solver is set, lclSol acts as an initial guess
      //  --if no changes are made to lclSol, it simply picks the solution at
      //     the previous time-step
      //  --here we copy from the IG of the global system for the dirichlet
      //     nodes in the velocity solution
      lclSol.SetSubVector( _essVhTDOF, &( glbIG->GetData()[spaceDofs * t] ) );
      _Fsolve->Mult( b, lclSol );

      if (_verbose ){
        if ( t==0 ){
          std::cout<<"Rank "<<_myRank<<" solved for time-step ";
        }
        if ( t<_numProcs-1){
          std::cout<<t<<", ";
        }else{
          std::cout<<t<<std::endl;
        }

      }


      // - send local solution to corresponding processor
      if( t==0 ){
        // that is, myself if first time step
        for ( int j = 0; j < spaceDofs; ++j ){
          _Y->GetData()[j] = lclSol.GetData()[j];
        }
      }else{
        // or the right slave it if solution is later in time
        MPI_Send( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t,  _comm );   // TODO: non-blocking + wait before solve?
      }


      // - include solution as rhs for next time-step
      if( t < _numProcs-1 ){
        _M->Mult( lclSol, b );
        b.Neg();    //M has negative sign for velocity, so flip it
        b.SetSubVector( _essVhTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution
      }

    }

  }else{
    // Slaves receive data
    MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank, _comm, MPI_STATUS_IGNORE );
  }


  // Make sure we're all done
  MPI_Barrier( _comm );

}















//******************************************************************************
// Space-time block assembly
//******************************************************************************
// These functions are useful for assembling all the necessary space-time operators

StokesSTOperatorAssembler::StokesSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName, const int refLvl,
                                                      const int ordU, const int ordP, const double dt, const double mu,
                                                      void(  *f)(const Vector &, double, Vector &),
                                                      double(*g)(const Vector &, double ),
                                                      void(  *n)(const Vector &, double, Vector &),
		                         							            void(  *u)(const Vector &, double, Vector &),
		                         							            double(*p)(const Vector &, double ),
                                                      bool verbose ):
	_comm(comm), _dt(dt), _mu(mu), _fFunc(f), _gFunc(g), _nFunc(n), _uFunc(u), _pFunc(p), _ordU(ordU), _ordP(ordP),
  _MuAssembled(false), _FuAssembled(false), _MpAssembled(false), _ApAssembled(false), _BAssembled(false),
  _FFAssembled(false), _BBAssembled(false), _pSAssembled(false), _FFinvAssembled(false),
  _verbose(verbose){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

	// For each processor:
	//- generate mesh
	_mesh = new Mesh( meshName.c_str(), 1, 1 );
  
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  _dim = _mesh->Dimension();

  // - initialise FE info
  _VhFEColl  = new H1_FECollection( ordU, _dim );  

  if( ordP > 0 )
    _QhFEColl  = new H1_FECollection( ordP, _dim );
  else
    _QhFEColl  = new L2_FECollection(    0, _dim );


  _VhFESpace = new FiniteElementSpace( _mesh, _VhFEColl, _dim );
  _QhFESpace = new FiniteElementSpace( _mesh, _QhFEColl );

  if ( _mesh->bdr_attributes.Size() > 0 ) {
    Array<int> essBdrV( _mesh->bdr_attributes.Max() ), essBdrQ( _mesh->bdr_attributes.Max() );
    essBdrV = 0; essBdrQ = 0;
    for ( int i = 0; i < _mesh->bdr_attributes.Max(); ++i ){
      if( _mesh->bdr_attributes[i] == 1 )
        essBdrV[i] = 1;
      if( _mesh->bdr_attributes[i] == 2 )
        essBdrQ[i] = 1;
    }

    _VhFESpace->GetEssentialTrueDofs( essBdrV, _essVhTDOF );
    _QhFESpace->GetEssentialTrueDofs( essBdrQ, _essQhTDOF );
  }


  _pSchur = new StokesSTPreconditioner( comm, dt, mu, NULL, NULL, _essQhTDOF, verbose );
  _FFinv  = new SpaceTimeSolver(        comm,         NULL, NULL, _essVhTDOF, verbose );


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

  _fuVarf =  new BilinearForm(_VhFESpace);

#ifdef MULT_BY_DT
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
	_fuVarf->AddDomainIntegrator(new VectorMassIntegrator( one ));
	_fuVarf->AddDomainIntegrator(new VectorDiffusionIntegrator( muDt ));
#else
  ConstantCoefficient mu( _mu );
  ConstantCoefficient dtinv( 1./_dt );
  _fuVarf->AddDomainIntegrator(new VectorMassIntegrator( dtinv ));
  _fuVarf->AddDomainIntegrator(new VectorDiffusionIntegrator( mu ));
#endif

  _fuVarf->Assemble();
  _fuVarf->Finalize();
  


  _fuVarf->FormSystemMatrix( _essVhTDOF, _Fu );


  _FuAssembled = true;


  // - once the matrix is generated, we can get rid of the operator
  // _Fu = fVarf->SpMat();
  // _Fu.SetGraphOwner(true);
  // _Fu.SetDataOwner(true);
  // fVarf->LoseMat();
  // delete fVarf;

  if(_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
    //   if ( rank==_myRank ){        
    std::cout<<"Rank "<<_myRank<<" - Space-operator matrix Fu assembled"<<std::endl;
        // for ( int i = 0; i < _Fu.NumRows(); ++i ){
        //   std::cout<<"Row "<<i<<" - Cols: ";
        //   for ( int j = _Fu.GetI()[i]; j < _Fu.GetI()[i+1]; ++j ){
        //       std::cout<<_Fu.GetJ()[j]<<": "<< _Fu.GetData()[j]<<" - ";
        //   }
        //   std::cout<<std::endl;
        // }
      // }
    MPI_Barrier(_comm);
    // }
  }

}



// Assemble operator on subdiagonal of space-time matrix for velocity block:
//  Mu = -M
void StokesSTOperatorAssembler::AssembleMu( ){
  if( _MuAssembled ){
    return;
  }

	BilinearForm *mVarf( new BilinearForm(_VhFESpace) );
#ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  mVarf->AddDomainIntegrator(new VectorMassIntegrator( mone ));
#else
  ConstantCoefficient mdtinv( -1./_dt );
  mVarf->AddDomainIntegrator(new VectorMassIntegrator( mdtinv ));
#endif
  mVarf->Assemble();
  mVarf->Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Mu = mVarf->SpMat();
  _Mu.SetGraphOwner(true);
  _Mu.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

  _MuAssembled = true;

  if(_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
    //   if ( rank==_myRank ){        
    std::cout<<"Rank "<<_myRank<<" - Velocity mass-matrix Mu assembled"<<std::endl;
        // for ( int i = 0; i < _Mu.NumRows(); ++i ){
        //   std::cout<<"Row "<<i<<" - Cols: ";
        //   for ( int j = _Mu.GetI()[i]; j < _Mu.GetI()[i+1]; ++j ){
        //       std::cout<<_Mu.GetJ()[j]<<": "<< _Mu.GetData()[j]<<" - ";
        //   }
        //   std::cout<<std::endl;
        // }
      // }
    MPI_Barrier(_comm);
    // }
  }  
}



// Assemble -divergence operator:
//  B = -dt * div
// TODO: it really bothers me that I cannot just use FormRectangularSystemMatrix here
//  to recover the actual SparseMatrix representing B, and then reuse FormRectangularLinearSystem
//  to include BC / initialise the system properly. It seems to work for Fu, but here it throws
//  weird errors.
void StokesSTOperatorAssembler::AssembleBvarf( ){

  if( _BAssembled ){
    return;
  }

	_bVarf = new MixedBilinearForm( _VhFESpace, _QhFESpace );

#ifdef MULT_BY_DT  
  ConstantCoefficient minusDt( -_dt );
  _bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(minusDt) );
#else
  ConstantCoefficient mone( -1.0 );
  _bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(mone) );
#endif

  _bVarf->Assemble();
  _bVarf->Finalize();


	// // - once the matrix is generated, we can get rid of the operator
 //  _B = bVarf->SpMat();
 //  _B.SetGraphOwner(true);
 //  _B.SetDataOwner(true);
 //  bVarf->LoseMat();
 //  delete bVarf;

  if(_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
    //   if ( rank==_myRank ){        
    std::cout<<"Rank "<<_myRank<<" - Bilinear for for -divergence operator B assembled"<<std::endl;
      //   for ( int i = 0; i < _B.NumRows(); ++i ){
      //     std::cout<<"Row "<<i<<" - Cols: ";
      //     for ( int j = _B.GetI()[i]; j < _B.GetI()[i+1]; ++j ){
      //         std::cout<<_B.GetJ()[j]<<": "<< _B.GetData()[j]<<" - ";
      //     }
      //     std::cout<<std::endl;
      //   }
      // }
    MPI_Barrier(_comm);
    // }
  }  
}





// Assemble "laplacian" operator for pressure block:
void StokesSTOperatorAssembler::AssembleAp( ){

  if( _ApAssembled ){
    return;
  }

  BilinearForm *aVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient one( 1.0 );
	aVarf->AddDomainIntegrator(new DiffusionIntegrator( one ));
  if ( _ordP == 0 ){  // DG
    double sigma = -1.0;
    double kappa =  1.0;
    aVarf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
    // a->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));  // to weakly impose Dirichlet BC - don't bother for now
  }
  aVarf->Assemble();
  aVarf->Finalize();
  
  // TODO: extract boundary tagged to "1" and impose dirichlet there...but forget about this for now
  // Array<int> essQhTDOF;
  // if ( _mesh->bdr_attributes.Size() ) {
  //   Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
  //   ess_bdr = 1;
  //   // _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
  //   _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  // }
  // aVarf->FormSystemMatrix( essQhTDOF, _Ap );
  // - once the matrix is generated, we can get rid of the operator

  _Ap = aVarf->SpMat();
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf->LoseMat();

  // Vector diag( aVarf->NumRows() );
  // ( aVarf->SpMat() ).GetDiag(diag); // TODO: not really the most efficient way..use pointers? in any case, this deep-copies
  // SparseMatrix temp(diag);
  // _Ap = temp;


  delete aVarf;

  _ApAssembled = true;
}



// Assemble mass operator for pressure block:
//  Mp = M
void StokesSTOperatorAssembler::AssembleMp( ){

  if( _MpAssembled ){
    return;
  }

	BilinearForm *mVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient one( 1.0 );
	mVarf->AddDomainIntegrator(new MassIntegrator( one ));
  mVarf->Assemble();
  mVarf->Finalize();

  // // TODO: extract boundary tagged to "1" and impose dirichlet there...but forget about this for now
  // Array<int> essQhTDOF;
  // if ( _mesh->bdr_attributes.Size() ) {
  //   Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
  //   ess_bdr = 1;
  //   _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  // }
  // mVarf->FormSystemMatrix( essQhTDOF, _Mp );

  // - once the matrix is generated, we can get rid of the operator
  _Mp = mVarf->SpMat();
  _Mp.SetGraphOwner(true);
  _Mp.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

  _MpAssembled = true;

}










// <<  Deprecated >>>
// Solve Space-time system for velocity via SEQUENTIAL time-stepping
void StokesSTOperatorAssembler::TimeStepVelocity( const HypreParVector& rhs, HypreParVector*& sol ){

  AssembleFu();
  AssembleMu();

  TimeStep( _Fu, _Mu, rhs, sol );
}



// <<  Deprecated >>>
// Solve Space-time system for pressure via SEQUENTIAL time-stepping
void StokesSTOperatorAssembler::TimeStepPressure( const HypreParVector& rhs, HypreParVector*& sol ){

  AssembleAp();
  AssembleMp();

  SparseMatrix Fp = _Ap;

#ifdef MULT_BY_DT
  Fp *= _mu * _dt;
  Fp.Add( 1.0, _Mp );     // TODO: check that Mp falls into the sparsity pattern of A
  SparseMatrix Mp = _Mp;
  Mp *= -1.0;
#else
  Fp *= _mu;
  Fp.Add( 1./_dt, _Mp );     // TODO: check that Mp falls into the sparsity pattern of A
  SparseMatrix Mp = _Mp;
  Mp *= -1./_dt;
#endif

  TimeStep( Fp, Mp, rhs, sol );
}




// <<  Deprecated >>>
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

    // Initialise solver
    std::cout<<"Warning: TimeStep: spatial operator is assumed to be time-independent!"<<std::endl;
    CGSolver solver;
    solver.SetOperator( F );
    solver.SetRelTol( 1e-12 );
    solver.SetMaxIter( spaceDofs );

    // Main time-stepping routine
    for ( int t = 0; t < _numProcs; ++t ){

      // - define rhs for this step
      for ( int i = 0; i < spaceDofs; ++i ){
        b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
      }

      // - solve for current time-step
      // CG( F, b, x, 0, spaceDofs, _tol, _tol );
      solver.Mult( b, x );
      std::cout<<"Time step: PCG converged in "<<solver.GetNumIterations()<<", final norm: "<<solver.GetFinalNorm()<<std::endl;

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


  if(_verbose ){
    std::cout<<"Rank "<<_myRank<<" - Space-time velocity operator FF assembled"<<std::endl;
    MPI_Barrier(_comm);
  }  

}












// Assemble BB (bottom-left)
void StokesSTOperatorAssembler::AssembleBB( ){ 

  if(_BBAssembled){
    return;
  }

  if(!_BAssembled){
    std::cerr<<"Divergence operator matrix not initialised"<<std::endl;
    return;
  }
  // For each processor, define -div operator
  // AssembleB();


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

  if(_verbose ){
    std::cout<<"Rank "<<_myRank<<" - Space-time divergence operator BB assembled"<<std::endl;
    MPI_Barrier(_comm);
  }  

}












// Assemble XX (bottom-right in preconditioner)
void StokesSTOperatorAssembler::AssemblePS(){
  if ( _pSAssembled ){
    return;
  }

  AssembleAp();
  AssembleMp();

  _pSchur->SetAp( &_Ap );
  _pSchur->SetMp( &_Mp );

  if(_verbose ){
    std::cout<<"Rank "<<_myRank<<" - Approximate space-time pressure Schur complement XX assembled"<<std::endl;
    MPI_Barrier(_comm);
  }  

}




// Assemble FF^-1 (top-left in preconditioner)
void StokesSTOperatorAssembler::AssembleFFinv(){
  if ( _FFinvAssembled ){
    return;
  }

  AssembleFu();
  AssembleMu();

  _FFinv->SetF( &_Fu );
  _FFinv->SetM( &_Mu );

  if(_verbose ){
    std::cout<<"Rank "<<_myRank<<" - (Approximate) inverse of space-time velocity operator FF^-1 assembled"<<std::endl;
    MPI_Barrier(_comm);
  }  

}




// Assembles space-time Stokes block system
//   Ax = b <-> ⌈ FF  BB^T ⌉⌈u⌉_⌈f⌉
//              ⌊ BB  0    ⌋⌊p⌋‾⌊g⌋,
// where:
//  - FF contains space-time matrix for velocity,
//  - BB is block-diagonal with -div operator in it
//  - f  is the velocity rhs
//  - g  is the pressure rhs
// Function also provides suitable initial guess for system (initialised with dirichlet BC)
void StokesSTOperatorAssembler::AssembleSystem( HypreParMatrix*& FFF,  HypreParMatrix*& BBB,
                                                HypreParVector*& frhs, HypreParVector*& grhs,
                                                HypreParVector*& IGu,  HypreParVector*& IGp ){

  // ASSEMBLE RHS -----------------------------------------------------------
  // Initialise handy functions for rhs
  VectorFunctionCoefficient uFuncCoeff( _dim, _uFunc );
  VectorFunctionCoefficient fFuncCoeff( _dim, _fFunc );
  VectorFunctionCoefficient nFuncCoeff( _dim, _nFunc );
  FunctionCoefficient       pFuncCoeff( _pFunc );
  FunctionCoefficient       gFuncCoeff( _gFunc );
  // - specify evaluation time
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  fFuncCoeff.SetTime( _dt*(_myRank+1) );
  nFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  gFuncCoeff.SetTime( _dt*(_myRank+1) );

  // Assemble local part of rhs
  // - for velocity
  LinearForm *fform( new LinearForm );
  fform->Update( _VhFESpace );
  fform->AddDomainIntegrator(   new VectorDomainLFIntegrator(       fFuncCoeff       ) );  //int_\Omega f*v
  fform->AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     nFuncCoeff       ) );  //int_d\Omega \mu * du/dn *v
  fform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n

  fform->Assemble();

#ifdef MULT_BY_DT
  fform->operator*=( _dt );
#endif

  // - include initial conditions
  if( _myRank == 0 ){
    uFuncCoeff.SetTime( 0.0 );
    LinearForm *u0form( new LinearForm );
    u0form->Update( _VhFESpace );
    u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) );  //int_\Omega u0*v
    u0form->Assemble();

#ifndef MULT_BY_DT
    u0form->operator*=(1./_dt);
#endif
    fform->operator+=( *u0form );

    // remember to reset function evaluation for u to the current time
    uFuncCoeff.SetTime( _dt*(_myRank+1) );


    delete u0form;

    if(_verbose){
      std::cout<<"Initial condition included "<<std::endl;
    }
  }



  // - for pressure
  LinearForm *gform( new LinearForm );
  gform->Update( _QhFESpace );
  gform->AddDomainIntegrator( new DomainLFIntegrator( gFuncCoeff ) );  //int_\Omega g*q
  gform->Assemble();

#ifdef MULT_BY_DT
  gform->operator*=( _dt );
#endif




  // - adjust rhs to take dirichlet BC into account
  // -- initialise relevant bilinear forms
  AssembleFu();
  AssembleBvarf();



  // if ( _myRank == 0 ){        
  //   std::cout<<"Rank "<<_myRank<<" - Space-operator matrix Fu assembled"<<std::endl;
  //   for ( int i = 0; i < _Fu.NumRows(); ++i ){
  //     std::cout<<"Row "<<i<<" - Cols: ";
  //     for ( int j = _Fu.GetI()[i]; j < _Fu.GetI()[i+1]; ++j ){
  //         std::cout<<_Fu.GetJ()[j]<<": "<< _Fu.GetData()[j]<<" - ";
  //     }
  //     std::cout<<std::endl;
  //   }
  // }


  // -- initialise function with BC
  GridFunction uBC(_VhFESpace), pBC(_QhFESpace);
  uBC.ProjectCoefficient(uFuncCoeff);
  pBC.ProjectCoefficient(pFuncCoeff);
  // -- initialise local rhs
  Vector fRhsLoc(  fform->Size() );
  Vector gRhsLoc(  gform->Size() );
  // -- initialise local initial guess to exact solution
  Vector iguLoc( uBC ), igpLoc( pBC );
  iguLoc.SetSubVectorComplement( _essVhTDOF, 0.0); // set to zero on interior (non-essential) nodes
  igpLoc.SetSubVectorComplement( _essQhTDOF, 0.0);



  // ASSEMBLE LOCAL LINEAR SYSTEMS ------------------------------------------
  _fuVarf->FormLinearSystem(           _essVhTDOF,             uBC, *fform, _Fu, iguLoc, fRhsLoc );
  _bVarf->FormRectangularLinearSystem( _essVhTDOF, _essQhTDOF, uBC, *gform, _B,  iguLoc, gRhsLoc );


  _FuAssembled = true;
  _BAssembled  = true;





  if(_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
      // if ( rank==_myRank ){        
    std::cout<<"Rank "<<_myRank<<" - RHS assembled: "<<std::endl;
        // std::cout<<"\t\tf,\t\tn,\t\t-p:"<<std::endl;
        // for ( int i = 0; i < fform->Size(); ++i ){
        //   std::cout<<"Row: "<<i<<",\t\t"<<fform->GetData()[i]<<",\t\t"<<nform->GetData()[i]<<",\t\t"<<pform->GetData()[i]<<std::endl;
        // }
      // }
    MPI_Barrier(_comm);
    // } 
  }





  // ASSEMBLE GLOBAL (PARALLEL) RHS -----------------------------------------
  // - for velocity
  int colPartV[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPartV );
  frhs->SetOwnership( 1 );

  // - for pressure
  int colPartP[2] = {_myRank*gRhsLoc.Size(), (_myRank+1)*gRhsLoc.Size()};
  grhs = new HypreParVector( _comm, gRhsLoc.Size()*_numProcs, gRhsLoc.StealData(), colPartP );
  grhs->SetOwnership( 1 );





  // ASSEMBLE INITIAL GUESS -------------------------------------------------
  // Assemble global vectors
  IGu = new HypreParVector( _comm, iguLoc.Size()*_numProcs, iguLoc.StealData(), colPartV );
  IGp = new HypreParVector( _comm, igpLoc.Size()*_numProcs, igpLoc.StealData(), colPartP );
  IGu->SetOwnership( 1 );
  IGp->SetOwnership( 1 );






  // ASSEMBLE OPERATOR ------------------------------------------------------
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

























/*
void StokesSTOperatorAssembler::AssembleRhs( HypreParVector*& frhs ){
  // Initialise handy functions
  VectorFunctionCoefficient fFuncCoeff(_dim,_fFunc);
  VectorFunctionCoefficient nFuncCoeff(_dim,_nFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  FunctionCoefficient       gFuncCoeff(_gFunc);
  // - specify evaluation time
  fFuncCoeff.SetTime( _dt*(_myRank+1) );
  nFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  gFuncCoeff.SetTime( _dt*(_myRank+1) );

  // Assemble local part of rhs
  LinearForm *fform( new LinearForm ), *nform( new LinearForm ), *pform( new LinearForm );
  fform->Update( _VhFESpace ); nform->Update( _VhFESpace ); pform->Update( _VhFESpace );
  fform->AddDomainIntegrator(   new VectorDomainLFIntegrator(       fFuncCoeff       ) );  //int_\Omega f*v
  nform->AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     nFuncCoeff       ) );  //int_d\Omega \mu * du/dn *v
  pform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n
  fform->Assemble(); nform->Assemble(); pform->Assemble();
  if(_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
      // if ( rank==_myRank ){        
    std::cout<<"Rank "<<_myRank<<" - RHS assembled: "<<std::endl;
        // std::cout<<"\t\tf,\t\tn,\t\t-p:"<<std::endl;
        // for ( int i = 0; i < fform->Size(); ++i ){
        //   std::cout<<"Row: "<<i<<",\t\t"<<fform->GetData()[i]<<",\t\t"<<nform->GetData()[i]<<",\t\t"<<pform->GetData()[i]<<std::endl;
        // }
      // }
    MPI_Barrier(_comm);
    // } 
  }

  Vector fRhsLoc( fform->Size() );
  fRhsLoc.SetData( fform->StealData() );
  fRhsLoc += *nform;
  fRhsLoc += *pform;

#ifdef MULT_BY_DT
  fRhsLoc *= _dt;
#endif

  delete fform;     // once data is stolen, we can delete the linear form
  delete nform;     // once data is stolen, we can delete the linear form
  delete pform;     // once data is stolen, we can delete the linear form


  // - include initial conditions
  if( _myRank == 0 ){
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    uFuncCoeff.SetTime( 0.0 );
    LinearForm *u0form( new LinearForm );
    u0form->Update( _VhFESpace );
    u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) ); //int_\Omega u0*v
    u0form->Assemble();

#ifdef MULT_BY_DT
    fRhsLoc += *u0form;
#else
    Vector temp = *u0form;
    temp *= (1./_dt);
    fRhsLoc += temp;
#endif

    if(_verbose){
      std::cout<<"Initial condition included "<<std::endl;
      // std::cout<<"(Row, value) = ";
      // for ( int i = 0; i < u0form->Size(); ++i ){
      //   std::cout<<"("<<i<<","<<u0form->GetData()[i]<<"), ";
      // }
      // std::cout<<std::endl;
    }

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
*/













// Assemble preconditioner
//   P^-1 = [ FF^-1  0     ]
//          [ 0      XX^-1 ],
// where FF contains space-time matrix for velocity,
void StokesSTOperatorAssembler::AssemblePreconditioner( Operator*& FFi, Operator*& XXi ){

  //Assemble top-left block
  AssembleFFinv( );
  AssemblePS( );

  FFi = _FFinv;
  XXi = _pSchur;

}







//<< deprecated >>
void StokesSTOperatorAssembler::ApplySTOperatorVelocity( const HypreParVector*& u, HypreParVector*& res ){
  // Initialise handy functions
  const int     lclSize = u->Size();
  const double* lclData = u->GetData();

  Vector Umine( lclSize ), Uprev( lclSize ), lclres( lclSize ), temp( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    Umine.GetData()[i]   = lclData[i];
    // lclres.GetData()[i] = 0.0;
  }

  // send my local solution to the following processor  
  MPI_Request reqSend, reqRecv;

  if( _myRank < _numProcs ){
    MPI_Isend( Umine.GetData(), lclSize, MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
  }
  if( _myRank > 0 ){
    MPI_Irecv( Uprev.GetData(), lclSize, MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
  }

  // assemble relevant matrices
  AssembleFu();
  AssembleMu();

  // diagonal part
  _Fu.Mult( Umine, lclres );

  if( _myRank > 0 ){
    // sub-diagonal part
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
    _Mu.Mult( Uprev, temp );    //Mu is minus mass matrix
    lclres += temp;
  }


  // assemble resulting vector
  Array<int> rowStarts(2);
  rowStarts[0] = ( lclSize )*_myRank;
  rowStarts[1] = ( lclSize )*(_myRank+1);

  res = new HypreParVector( _comm, lclSize*_numProcs, lclres.StealData(), rowStarts.GetData() );

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
  

  Array<int> rowStartsV(2), rowStartsQ(2);
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

  int order_quad = 5;
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i=0; i < Geometry::NumGeom; ++i){
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );


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






void StokesSTOperatorAssembler::SaveExactSolution(){
  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _VhFESpace );
    GridFunction *pFun = new GridFunction( _QhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( "STstokesEx", _mesh );
    paraviewDC.SetPrefixPath("ParaView");
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link uFun and pFun
    paraviewDC.RegisterField( "velocity", uFun );
    paraviewDC.RegisterField( "pressure", pFun );

    // main time loop
    for ( int t = 0; t < _numProcs+1; ++t ){
      VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
      FunctionCoefficient       pFuncCoeff(_pFunc);
      uFuncCoeff.SetTime( t*_dt );
      pFuncCoeff.SetTime( t*_dt );

      uFun->ProjectCoefficient( uFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );

      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete uFun;
    delete pFun;

  }
}


void StokesSTOperatorAssembler::SaveSolution( const HypreParVector& uh, const HypreParVector& ph ){
  
  // gather parallel vector
  Vector *uGlb = uh.GlobalVector();
  Vector *pGlb = ph.GlobalVector();


  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _VhFESpace );
    GridFunction *pFun = new GridFunction( _QhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( "STstokes", _mesh );
    paraviewDC.SetPrefixPath("ParaView");
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link uFun and pFun
    paraviewDC.RegisterField( "velocity", uFun );
    paraviewDC.RegisterField( "pressure", pFun );


    // store initial conditions
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    uFuncCoeff.SetTime( 0.0 );
    pFuncCoeff.SetTime( 0.0 );

    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // handy variables for time loop
    const int blockSizeU = uh.Size();
    const int blockSizeP = ph.Size();
    Vector uLcl, pLcl;
    Array<int> idxU(blockSizeU), idxP(blockSizeP);

    // main time loop
    for ( int t = 1; t < _numProcs+1; ++t ){
      // - identify correct sub-vector idx in global vectors
      for ( int i = 0; i < blockSizeU; ++i ){
        idxU[i] = blockSizeU*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeP; ++i ){
        idxP[i] = blockSizeP*(t-1) + i;
      }

      // - extract subvector
      uGlb->GetSubVector( idxU, uLcl );
      pGlb->GetSubVector( idxP, pLcl );
      
      // - assign to linked variables
      *uFun = uLcl;
      *pFun = pLcl;
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete uFun;
    delete pFun;

  }
}

void StokesSTOperatorAssembler::GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const{
  if(_mesh == NULL)
    std::cerr<<"Mesh not yet set"<<std::endl;
  else
    _mesh->GetCharacteristics( h_min, h_max, k_min, k_max );
}




void StokesSTOperatorAssembler::PrintMatrices( const std::string& filename ) const{

  if( _myRank == 0){
    if( ! ( _FuAssembled && _MuAssembled && _MpAssembled && _ApAssembled && _BAssembled ) ){
      std::cerr<<"Make sure all matrices have been initialised, otherwise they can't be printed"<<std::endl;
      return;
    }

    std::string myfilename;
    std::ofstream myfile;

    myfilename = filename + "_Fu.dat";
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
}




StokesSTOperatorAssembler::~StokesSTOperatorAssembler(){
  delete _pSchur;
  delete _FFinv;
  delete _fuVarf;
  delete _bVarf;
  delete _VhFESpace;
  delete _QhFESpace;
  delete _VhFEColl;
  delete _QhFEColl;
  delete _mesh;
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










