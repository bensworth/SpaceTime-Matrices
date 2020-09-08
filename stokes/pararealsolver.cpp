#include "pararealsolver.hpp"


//******************************************************************************
// Parareal
//******************************************************************************
// These functions apply the parareal algorithm to solve for a space-time matrix
//       ⌈ F*dt           ⌉
// FF  = |  -M   F*dt     |,
//       |        -M   \\ |
//       ⌊             \\ ⌋
//  where F = M/dt + S with S the spatial operator
// The discretisation of coarse and fine solvers doesn't change! But C can (and 
//  should) be prescribed as being lower-resolution
PararealSolver::PararealSolver( const MPI_Comm& comm, const SparseMatrix* F, const SparseMatrix* C, const SparseMatrix* M,
                                int maxIT, int verbose ):
  _comm(comm),  _Fsolve(NULL), _Csolve(NULL), _maxIT(maxIT), _X(NULL), _Y(NULL), _verbose(verbose){

  if( F != NULL ) SetF(F);
  if( C != NULL ) SetC(C);
  if( M != NULL ) SetM(M);

  // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
  //  to actually make use of the IG or not
  iterative_mode = true;

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );

}



PararealSolver::~PararealSolver(){
  delete _Fsolve;
  delete _Csolve;
  delete _F;
  delete _C;
  delete _X;
  delete _Y;
}



// initialise info on fine spatial operator
void PararealSolver::SetF( const SparseMatrix* F ){
  _F = new PetscParMatrix( F );

  height = F->Height();
  width  = F->Width();
  SetFSolve();
}

// initialise info on coarse spatial operator
void PararealSolver::SetC( const SparseMatrix* C ){
  _C = new PetscParMatrix( C );

  height = C->Height();
  width  = C->Width();
  SetCSolve();
}

// initialise info on mass-matrix for the velocity.
void PararealSolver::SetM( const SparseMatrix* M ){
  _M = M;

  height = M->Height();
  width  = M->Width();
}



// initialise solver of fine spatial operator
void PararealSolver::SetFSolve(){
  delete _Fsolve;

 _Fsolve = new PetscLinearSolver( *_F, "VSolver_" );
}

// initialise solver of coarse spatial operator
void PararealSolver::SetCSolve(){
  delete _Csolve;
  _Csolve = new PetscLinearSolver( *_C, "VCoarseSolver_" );
}





// Define multiplication by solver (that is, implementation of parareal)
// - the most relevant function here
void PararealSolver::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT( _Fsolve != NULL, "Fine solver for spatial operator not initialised" );
  MFEM_ASSERT( _Csolve != NULL, "Coarse solver for spatial operator not initialised" );
  MFEM_ASSERT( _M      != NULL, "Mass matrix not initialised" );
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  if ( _verbose && _myRank==0 ){
    std::cout<<"Applying velocity block preconditioner (parareal)"<<std::endl;
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


  if ( _verbose ){
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


  // initialise auxiliary solution vectors for parareal
  Vector     u0G( spaceDofs ),     u0F( spaceDofs ),
          u0Gnew( spaceDofs ),      u0( spaceDofs ),
         lclSolG( spaceDofs ), lclSolF( spaceDofs ),
               b(spaceDofs),    bAndIC(spaceDofs);
  u0G=0.; u0F = 0.; u0Gnew=0.; u0 = 0.; lclSolG=0.; lclSolF = 0.; b=0.; bAndIC = 0.;
  for ( int j = 0; j < spaceDofs; ++j ){
    b.GetData()[j] = _X->GetData()[j];          // initialise rhs
  }


  //-------------------------------------------------------------------------
  // Main parareal iteration
  //-------------------------------------------------------------------------
  for ( int iii = 0; iii < _maxIT; ++iii ){  

    // Only processors corresponding to non-converged chunks participate in this iteration
    if( _myRank >= iii ){
    
      // All time chunks (except the one that completes at this iteration) need to wait to get their (coarse) initial conditions
      if( _myRank > iii ){

        MPI_Recv( u0Gnew.GetData(), spaceDofs, MPI_DOUBLE, _myRank-1, 2*(_myRank-1) + 2*_numProcs*iii, _comm, MPI_STATUS_IGNORE );

        // if( _verbose ){
        //   std::cout<<"Rank "<<_myRank<<" of "<<_numProcs<<" at it "<< iii<<" receives coarse from "<<_myRank-1<<" with tag : "<<2*(_myRank-1) + 2*_numProcs*iii<<std::endl;
        // }

        // Parareal update
        for ( int j = 0; j < spaceDofs; ++j ){
          u0.GetData()[j]  = u0Gnew.GetData()[j] - u0G.GetData()[j] + u0F.GetData()[j];
        }
        u0G = u0Gnew;
      }

      // Include the new IC in the rhs, so that the time-stepping is done right
      _M->Mult(u0, bAndIC);     //NB for the very first proc and it, u0 is zero, so I'm not including the IC twice
      
      // actual rhs is given by b - (-M)*u0
      //NB M already incorporates negative sign for velocity!
      bAndIC.Neg();
      bAndIC += b;
      

      // All time chunks (except the very last one) need to perform a coarse integration and send the result to the following time chunk
      if( _myRank < _numProcs - 1 ){
        // coarse integration
        // - this is to make sure that the same IG is used all the time
        for ( int j = 0; j < spaceDofs; ++j ){
          lclSolG.GetData()[j] = _Y->GetData()[j];
        }
        _Csolve->Mult( bAndIC, lclSolG );


        MPI_Send( lclSolG.GetData(), spaceDofs, MPI_DOUBLE, _myRank + 1, 2*_myRank + 2*_numProcs*iii, _comm );

        // if( _verbose ){
        //   std::cout<<"Rank "<<_myRank<<" of "<<_numProcs<<" at it "<< iii<<" sends coarse to "<<_myRank+1<<" with tag : "<<2*_myRank + 2*_numProcs*iii<<std::endl;
        // }
      }


      // All time chunks need to perform a fine integration
      // - this is to make sure that the same IG is used all the time
      for ( int j = 0; j < spaceDofs; ++j ){
        lclSolF.GetData()[j] = _Y->GetData()[j];
      }
      _Fsolve->Mult( bAndIC, lclSolF );


      // All time chunks (except the one that completes at this iteration) need to wait to get their (fine) initial conditions
      if( _myRank > iii ){
        MPI_Recv( u0F.GetData(), spaceDofs, MPI_DOUBLE, _myRank-1, 2*(_myRank-1)+1 + 2*_numProcs*iii, _comm, MPI_STATUS_IGNORE );

        // if( _verbose ){
        //   std::cout<<"Rank "<<_myRank<<" of "<<_numProcs<<" at it "<< iii<<" receives fine from "<<_myRank-1<<" with tag : "<<2*(_myRank-1)+1 + 2*_numProcs*iii<<std::endl;
        // }
      }

      // All time chunks (except the very last one) need to send the result of the fine integration to the following time chunk
      if( _myRank < _numProcs-1 ){
        MPI_Send( lclSolF.GetData(), spaceDofs, MPI_DOUBLE, _myRank + 1, 2*_myRank+1 + 2*_numProcs*iii, _comm );

        // if( _verbose ){
        //   std::cout<<"Rank "<<_myRank<<" of "<<_numProcs<<" at it "<< iii<<" sends fine to "<<_myRank+1<<" with tag : "<<2*_myRank+1 + 2*_numProcs*iii<<std::endl;
        // }
      }

      // The time chunk right next the process that just completed, does not need to update: u0F *IS* the real u0
      if ( _myRank == iii+1 ){
        u0 = u0F;
      }
    
    }

  }

  // store everything in the output
  for ( int j = 0; j < spaceDofs; ++j ){
    _Y->GetData()[j] = lclSolF.GetData()[j];
  }

  // Make sure we're all done
  MPI_Barrier( _comm );


  if ( _verbose ){
    std::cout<<"Inside V-block: Rank: "<<_myRank<< ", result for V: "; y.Print(std::cout, y.Size());
  }
}










