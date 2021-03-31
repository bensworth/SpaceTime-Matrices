#include "spacetimewavesolver.hpp"

#include <iostream>
#include "petsc.h"

using namespace mfem;



SpaceTimeWaveSolver::SpaceTimeWaveSolver( const MPI_Comm& comm, const SparseMatrix* Cp, const SparseMatrix* C0, const SparseMatrix* Cm,
                                          const Array<int>& essTDOF, bool timeDep, bool symmetric, int verbose):
  _comm(comm), _timeDep(timeDep), _symmetric(symmetric), _essTDOF(essTDOF),
  _Cp(NULL), _C0(NULL), _Cm(NULL), _Cpsolve(NULL),
  _X(NULL), _Y(NULL), _verbose(verbose){

  // if ( timeDep || !symmetric ){
  //   std::cerr<<"Error: only supports symmetric, non-time-dep schemes for now"<<std::endl;
  // }

  if( Cp != NULL ) SetDiag( Cp, 0 );
  if( C0 != NULL ) SetDiag( C0, 1 );
  if( Cm != NULL ) SetDiag( Cm, 2 );

  // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
  //  to actually make use of the IG or not
  iterative_mode = true;

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



SpaceTimeWaveSolver::~SpaceTimeWaveSolver(){
  delete _Cpsolve;
  delete _Cp;
  delete _X;
  delete _Y;
}



// Initialise info on operators on diagonal diag.
// NB: Each processor should own the three operators evaluated at the *same* instant!
//  That is, if the matrix is 
//       ⌈\\      \\      \\                    ⌉
//       |Cm{i-2} C0{i-1} Cp{i}                 |
//  CC = |        Cm{i-1} C0{i} Cp{i+1}         |,
//       |                Cm{i} C0{i+1} Cp{i+2} |
//       ⌊                   \\      \\     \\  ⌋
//  then processor i should own Cp{i} C0{i} Cm{i}.
void SpaceTimeWaveSolver::SetDiag( const SparseMatrix* Op, int diag ){

  height = Op->Height();
  width  = Op->Width();

  switch (diag){
    case 0:{
      delete _Cp;
      _Cp = new PetscParMatrix( Op );
      SetCpSolve();
      break;
    }
    case 1:{
      _C0 = Op;
      break;
    }
    case 2:{
      _Cm = Op;
      break;
    }
    default:
      std::cerr<<"ERROR: trying to set diagonal" << diag <<", while only first three subdiagonals can be set"<<std::endl;

  }
}




// initialise solver of main diagonal operator
void SpaceTimeWaveSolver::SetCpSolve(){
  delete _Cpsolve;

  // 0th processor must always initialise the solver
  // - if the spatial operator is time-dependent, then every processor must initialise a solver
  // - if the spatial operator is not time-dependent, and the stencil is symmetric, then Cp==Cm and
  //    the 0th processor can take care of solving each time-step
  if ( _timeDep || _myRank == 0 ){
    _Cpsolve = new PetscLinearSolver( *_Cp, "AWaveSolver_" );
  // if it's not time-dependent, but the stencil is non-symmetric, then:
  //  - the 0th processor must solve for (Cm+Cp)
  //  - the 1st processor can solve for Cp for all time-steps (constant)
  }else if ( !_symmetric || _myRank == 1 ){
    _Cpsolve = new PetscLinearSolver( *_Cp, "AWaveSolver_" );
  }
  
}







// Define multiplication by operator (that is, time-stepping)
//  - the most relevant function here
void SpaceTimeWaveSolver::Mult( const Vector &x, Vector &y ) const{
  // - If operators are time-dependent, all must initialise their respective operators (bar the last two)
  if ( _timeDep ){
    MFEM_ASSERT( _Cpsolve != NULL, "Solver for diagonal block of space-time operator not initialised" );
    if( _myRank < _numProcs-1 ){
      MFEM_ASSERT( _C0    != NULL, "First subdiagonal not initialised" );
    }
    if( _myRank < _numProcs-2 ){
        MFEM_ASSERT( _Cm  != NULL, "Second subdiagonal not initialised" );
    }
  // - otherwise we can apply some simplifications and have only rank 0/1 take care of everything
  // -- if the scheme is symmetric, Cp==Cm, and rank 0 can take care of everything (divide by 2 to solve for Cp+Cm)
  // -- otherwise, rank 0 solves for Cp+Cm, and rank 1 can take care of everything else
  }else if( _myRank == 0 || ( _myRank == 1 && !_symmetric ) ){
    MFEM_ASSERT( _Cpsolve != NULL, "Solver for diagonal block of space-time operator not initialised" );
    MFEM_ASSERT( _C0      != NULL, "First subdiagonal not initialised" );
    MFEM_ASSERT( _Cm      != NULL, "Second subdiagonal not initialised" );
  }

  MFEM_ASSERT(x.Size() == Width(),  "invalid x.Size() = " << x.Size() << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size() << ", expected size = " << Height());


  if( _verbose>20 ){
    if ( _myRank==0 ){
      std::cout<<"Applying exact space-time solver (time-stepping) for wave equation\n";
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
    std::cout<<"Inside wave solver, rank "<<_myRank<<", rhsA: "; x.Print(std::cout, x.Size());
    std::cout<<"Inside wave solver, rank "<<_myRank<<", IGA : "; y.Print(std::cout, y.Size());

    std::ofstream myfile;
    std::string myfilename = std::string("./results/rhsA.dat");
    myfile.open( myfilename, std::ios_base::app );
    _X->Print_HYPRE(myfile);
    myfile.close( );

    myfilename = std::string("./results/IGA.dat");
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
    
    // - receive contribution to rhs from previous processors (careful if starting up)
    if ( _myRank > 0 ){
      MPI_Recv(      b.GetData(), spaceDofs, MPI_DOUBLE, _myRank-1, 2*_myRank,  _comm, MPI_STATUS_IGNORE );
      if ( _myRank > 1 ){
        Vector temp(spaceDofs);
        MPI_Recv( temp.GetData(), spaceDofs, MPI_DOUBLE, _myRank-2, 2*_myRank+1, _comm, MPI_STATUS_IGNORE );
        b += temp;
      }
    }

    // - define rhs for this step (including contribution from sol at previous time-step
    for ( int i = 0; i < spaceDofs; ++i ){
      b.GetData()[i] += _X->GetData()[i];
    }

    // - solve for current time-step
    //  --if an iterative solver is set, _Y acts as an initial guess
    _Cpsolve->Mult( b, *_Y );



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

    // - send solution to following processors (careful if at the end)
    if ( _myRank < _numProcs-1 ){
      // - I own C0 evaluated at my instant, so I need to do the mult
      _C0->Mult( *_Y, b );
      // - this gives a contribution to the rhs, so flip its sign
      b.Neg();
      // NB: _C0 should be defined so that essential BC are not dirtied! So the next command is useless
      // b.SetSubVector( _essTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution

      MPI_Send(   b.GetData(), spaceDofs, MPI_DOUBLE, _myRank+1, 2*(_myRank+1), _comm );
      
      if ( _myRank < _numProcs-2 ){
        // - I own Cm evaluated at my instant, so I need to do the mult
        _Cm->Mult( *_Y, b );
        // - this gives a contribution to the rhs, so flip its sign
        b.Neg();
        // NB: _Cm should be defined so that essential BC are not dirtied! So the next command is useless
        // b.SetSubVector( _essTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution

        MPI_Send( b.GetData(), spaceDofs, MPI_DOUBLE, _myRank+2, 2*(_myRank+2)+1, _comm );
      }

    }





  // If the spatial operator is constant, however, we can have rank 0 / rank 1 take care of all the solutions
  }else{

    // decide which proc will take care of most of the computations
    int master;
    if ( _symmetric ){
      master = 0;
    }else{
      master = 1;
    }

    Vector solm1, solm2;

    // Whatever happens, rank 0 must solve for instant 0
    if ( _myRank == 0 ){
      // - solve for 0th time-step
      //  --if an iterative solver is set, _Y acts as an initial guess
      _Cpsolve->Mult( *_X, *_Y );

      if (_verbose>100 ){
        std::cout<<"Solved for time-step: 0";
      }

      // - if rank 0 is master
      if ( _myRank == master ){
        // -- rescale the solution accurately (remember you solved for Cp, but need to solve for Cp+Cm = 2*Cp)
        _Y->operator*=( 0.5 );     
        // --- but careful not to dirty Dirichlet nodes
        for ( int i = 0; i < _essTDOF.Size(); ++i ){
          _Y->operator()(_essTDOF[i]) = _X->operator()(_essTDOF[i]);
        }
        
        // -- then it needs to solve for instant 1, too
        Vector b(spaceDofs), temp(spaceDofs), ig(spaceDofs);
        // --- include result from instant 0 in the rhs
        _C0->Mult( *_Y, temp );
        // --- receive IG and rhs from rank 1
        MPI_Recv(  b.GetData(), spaceDofs, MPI_DOUBLE, 1, 0, _comm, MPI_STATUS_IGNORE );
        MPI_Recv( ig.GetData(), spaceDofs, MPI_DOUBLE, 1, 1, _comm, MPI_STATUS_IGNORE );
        // --- update rhs
        b -= temp;
        
        // --- solve for instant 1
        _Cpsolve->Mult( b, ig );

        if (_verbose>100 ){
          std::cout<<", 1";
        }
        
        // --- send sol back to rank 1
        MPI_Send( ig.GetData(), spaceDofs, MPI_DOUBLE, 1, 2, _comm );

        // ---store previous solutions
        solm1 = ig;
        solm2 = *_Y;

      // - otherwise, 0 needs to send solution to 1
      }else{
        // -- no need to rescale, since in this case at 0 I should've set the solver for Cp+Cm
        MPI_Send( _Y->GetData(), spaceDofs, MPI_DOUBLE, 1, 0, _comm );
      }

    }



    // Things are a bit messier for rank 1
    if ( _myRank == 1 ){
      // if 1 is master
      if ( _myRank == master ){
        // - receive solution at previous time-step from rank 0
        solm2.SetSize(spaceDofs);
        MPI_Recv( solm2.GetData(), spaceDofs, MPI_DOUBLE, 0, 0, _comm, MPI_STATUS_IGNORE );
        
        // - solve for instant 1
        Vector b = *_X;
        Vector temp(spaceDofs);
        // -- include result from instant 0 in the rhs
        _C0->Mult( solm2, temp );
        // -- update rhs
        b -= temp;
        
        // -- solve for instant 1
        // --- if an iterative solver is set, _Y acts as an initial guess
        _Cpsolve->Mult( b, *_Y );

        if (_verbose>100 ){
          std::cout<<", 1";
        }

        // ---store solution
        solm1 = *_Y;

      // - otherwise, 1 needs to send info to 0 and receive back solution
      }else{
        MPI_Send( _X->GetData(), spaceDofs, MPI_DOUBLE, 0, 0, _comm );
        MPI_Send( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, 1, _comm );

        MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, 2, _comm, MPI_STATUS_IGNORE );

      }

    }





    // Now we can finally start the main time-stepping routine
    if ( _myRank == master ){
      for ( int t = 2; t < _numProcs; ++t ){

        // - solve for current instant
        Vector b(spaceDofs), temp(spaceDofs), ig(spaceDofs);
        // -- receive IG and rhs from rank associated with current instant
        MPI_Recv(  b.GetData(), spaceDofs, MPI_DOUBLE, t, 3*t  , _comm, MPI_STATUS_IGNORE );
        MPI_Recv( ig.GetData(), spaceDofs, MPI_DOUBLE, t, 3*t+1, _comm, MPI_STATUS_IGNORE );
        // -- include result from prev instants in the rhs
        _C0->Mult( solm1, temp );
        b -= temp;
        _Cm->Mult( solm2, temp );
        b -= temp;
        
        // -- solve for instant t
        _Cpsolve->Mult( b, ig );

        if (_verbose>100 ){
          std::cout<<", " << t;
        }
        
        // -- send sol back to rank 1
        MPI_Send( ig.GetData(), spaceDofs, MPI_DOUBLE, t, 3*t+2, _comm );

        // -- store previous solutions
        solm2 = solm1;
        solm1 = ig;
        
      }

    }else if( _myRank > 1 ){
      // Slaves sends data on rhs and ig
      MPI_Send( _X->GetData(), spaceDofs, MPI_DOUBLE, master, 3*_myRank  , _comm );   // TODO: non-blocking + wait before solve?
      MPI_Send( _Y->GetData(), spaceDofs, MPI_DOUBLE, master, 3*_myRank+1, _comm );   // TODO: non-blocking + wait before solve?
      // and receive data on solution
      MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, master, 3*_myRank+2, _comm, MPI_STATUS_IGNORE );
    }

  }

  // Make sure we're all done
  MPI_Barrier( _comm );


  // if ( _verbose>100 ){
  //   std::cout<<"Inside wave solver, rank "<<_myRank<<", result for A: "; _Y->Print(std::cout, _Y->Size());
  // }

}




