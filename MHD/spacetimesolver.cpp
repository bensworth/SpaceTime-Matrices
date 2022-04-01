#include "spacetimesolver.hpp"

#include <iostream>
#include "petsc.h"

using namespace mfem;



SpaceTimeSolver::SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F, const SparseMatrix* M,
                                  const Array<int>& essVhTDOF, const std::string& solverOpt, bool timeDep, int verbose ):
  _comm(comm), _timeDep(timeDep), _F(NULL), _M(NULL), _Fsolve(NULL), _essVhTDOF(essVhTDOF), _solverOpt(solverOpt), _X(NULL), _Y(NULL), _verbose(verbose){

  if( F != NULL ) SetF(F);
  if( M != NULL ) SetM(M);

  // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
  //  to actually make use of the IG or not
  iterative_mode = true;

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



SpaceTimeSolver::~SpaceTimeSolver(){
  delete _Fsolve;
  delete _F;
  delete _X;
  delete _Y;
}



// initialise info on spatial operator
void SpaceTimeSolver::SetF( const SparseMatrix* F ){
  delete _F;
  _F = new PetscParMatrix( F );

  height = F->Height();
  width  = F->Width();
  SetFSolve();
}

// initialise info on mass-matrix
void SpaceTimeSolver::SetM( const SparseMatrix* M ){
  _M = M;

  height = M->Height();
  width  = M->Width();
}



// initialise solver of spatial operator
void SpaceTimeSolver::SetFSolve(){
  delete _Fsolve;

  if ( _timeDep || _myRank == 0 ){
    _Fsolve = new PetscLinearSolver( *_F, _solverOpt );

    // PetscBool set;
    // PetscErrorCode ierr;
    // char optName[PETSC_MAX_PATH_LEN];
    // _Fsolve->iterative_mode = true;  // trigger iterative mode...
    // ierr = PetscOptionsGetString( NULL ,"VSolver_", "-ksp_type", optName, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
    // if( !strcmp( optName, "preonly" ) ){
    //   char optName1[PETSC_MAX_PATH_LEN];
    //   ierr = PetscOptionsGetString( NULL ,"VSolver_", "-pc_type", optName1, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
    //   if(!( strcmp( optName1, "ilu" ) ) || !( strcmp( optName1, "lu" ) ) ){
    //     _Fsolve->iterative_mode = false;  // ...unless you're using ilu or lu
    //   }
    // }
  }
  
  // if ( _verbose && _myRank==0 && _Fsolve->iterative_mode ){
  //   std::cout<<"Selected iterative solver for velocity time-stepping"<<std::endl;
  // }

}





// Define multiplication by preconditioner (that is, time-stepping)
//  - the most relevant function here
void SpaceTimeSolver::Mult( const Vector &x, Vector &y ) const{
  if ( _timeDep || _myRank == 0 ){
    MFEM_ASSERT( _Fsolve != NULL, "Solver for diagonal block of space-time operator not initialised" );
  }
  MFEM_ASSERT( _M      != NULL, "Mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());


  if( _verbose>20 ){
    if ( _myRank==0 ){
      std::cout<<"Applying exact space-time solver (time-stepping)\n";
    }
    MPI_Barrier(_comm);
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


  if ( _verbose>100 ){
    std::cout<<"Inside V-block, rank "<<_myRank<<", rhs for u: "; x.Print(std::cout, x.Size());
    std::cout<<"Inside V-block, rank "<<_myRank<<", IG  for u: "; y.Print(std::cout, y.Size());

    std::ofstream myfile;
    std::string myfilename = std::string("./results/rhsu.dat");
    myfile.open( myfilename, std::ios_base::app );
    _X->Print_HYPRE(myfile);
    myfile.close( );

    myfilename = std::string("./results/IGu.dat");
    myfile.open( myfilename, std::ios_base::app );
    _Y->Print_HYPRE(myfile);
    myfile.close( );
  }

  
  // If the spatial operator is time-dependent, then each processor will have to solve for its own time-step
  if ( _timeDep ){

    // these will contain rhs for each time-step
    Vector b( spaceDofs );
    b = 0.;



    // Main "time-stepping" routine
    
    // - receive solution from previous processor (unless initial time-step)
    if ( _myRank > 0 ){
      MPI_Recv( b.GetData(), spaceDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
    }

    // - define rhs for this step (including contribution from sol at previous time-step
    for ( int i = 0; i < spaceDofs; ++i ){
      b.GetData()[i] += _X->GetData()[i];
    }

    // - solve for current time-step
    //  --if an iterative solver is set, _Y acts as an initial guess
    _Fsolve->Mult( b, *_Y );



    if (_verbose>100 ){
      if ( _myRank==0 ){
        std::cout<<"Solved for time-step: ";
      }
      if ( _myRank<_numProcs-1){
        std::cout<<_myRank<<", ";
      }else{
        std::cout<<_myRank<<std::endl;
      }
    }

    // - send solution to following processor (unless last time-step)
    if ( _myRank < _numProcs-1 ){
      // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
      _M->Mult( *_Y, b );
      // - M is stored with negative sign for velocity, so flip it
      b.Neg();
      // NB: _M should be defined so that essential BC are not dirtied! So the next command is useless
      // b.SetSubVector( _essVhTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution

      MPI_Send( b.GetData(), spaceDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
    }




  // If the spatial operator is constant, however, we can have rank 0 take care of all the solutions
  }else{
    // - broadcast IG and rhs to each proc (overkill, as only proc 0 will play with it, but whatever)
    // Man, valgrind really *hates* GlobalVector(). Better if I handle this myself
    // Vector *glbRhs = _X->GlobalVector();
    // Vector *glbIG  = _Y->GlobalVector();

  
    // Master performs time-stepping and sends solution to other processors
    if ( _myRank == 0 ){

      // - this will contain rhs for each time-step
      Vector b( spaceDofs );
      b = 0.;


      // Main time-stepping routine
      for ( int t = 0; t < _numProcs; ++t ){

        // - define rhs for this step (including contribution from sol at previous time-step - see below)
        Vector lclRhs( spaceDofs );
        if ( t==0 ){
          b = *_X;
        }else{
          MPI_Recv( lclRhs.GetData(), spaceDofs, MPI_DOUBLE, t, t, _comm, MPI_STATUS_IGNORE );
          b += lclRhs;
        }
        // for ( int i = 0; i < spaceDofs; ++i ){
        //   b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
        // }

        // - initialise local vector containing solution at single time-step
        Vector lclSol( spaceDofs );
        if ( t==0 ){
          lclSol = *_Y;
        }else{
          MPI_Recv( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t + _numProcs, _comm, MPI_STATUS_IGNORE );
        }
        // for ( int i = 0; i < spaceDofs; ++i ){
        //   lclSol.GetData()[i] = glbIG->GetData()[spaceDofs*t + i];
        // }

        _Fsolve->Mult( b, lclSol );

        //  TODO: Maybe set to 0 the solution on the dirichlet nodes?
        // lclSol.SetSubVector( _essVhTDOF, 0 );


        if (_verbose>100 ){
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
          MPI_Send( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t+2*_numProcs, _comm );   // TODO: non-blocking + wait before solve?
        }


        // - include solution as rhs for next time-step
        if( t < _numProcs-1 ){
          _M->Mult( lclSol, b );
          b.Neg();    //M has negative sign for velocity, so flip it
          // NB: _M should be defined so that essential BC are not dirtied! So the next command is useless
          // b.SetSubVector( _essVhTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution
        }

      }

    }else{
      // Slaves sends data on rhs and ig
      MPI_Send( _X->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank            , _comm );   // TODO: non-blocking + wait before solve?
      MPI_Send( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank+  _numProcs, _comm );   // TODO: non-blocking + wait before solve?
      // and receive data on solution
      MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank+2*_numProcs, _comm, MPI_STATUS_IGNORE );
    }

    // cleanup
    // delete glbIG;
    // delete glbRhs;

  }

  // Make sure we're all done
  MPI_Barrier( _comm );


  if ( _verbose>100 ){
    std::cout<<"Inside space-time block: Rank: "<<_myRank<< ", result for u: "; y.Print(std::cout, y.Size());
  }

}
