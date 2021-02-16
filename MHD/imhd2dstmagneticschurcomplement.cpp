#include "imhd2dstmagneticschurcomplement.hpp"



namespace mfem{



//##############################################################################
//
// SPACE-TIME BLOCK PRECONDITIONING
//
//##############################################################################

// - For information on the components of the block preconditioner, see [1]
//    E.C.Cyr, J.N.Shadid, R.S.Tuminaro, R.P.Pawlowski, L. Chacon: "A new
//    approximate block factorization preconditioner for two-dimensional"


//******************************************************************************
// Magnetic block
//******************************************************************************
// These functions are used to aid in the application of the magnetic part of
//  the block preconditioner (ie, approximating the inverse of the magnetic
//  Schur complement)
// This is defined as: XX^-1 = CC^-1 * FFa * D(M)^-1, where:
//  - D(*) represents the block-diagonal matrix with (*) as blocks
//  - FFa is the space-time matrix representing time-stepping on the vector
//     potential: a block bi-diagonal matrix with operators F = M + dt*W on the
//     main diagonal, and -M on the lower diagonal
//  - M  is the vector potential mass matrix
//  - W  is the spatial operator for the vector potential
//  - CC is the space-time discretisation of a wave equation propagating the
//     vector potential
//
// Inversion of the vector potential mass matrix is implemented using a PETSc
//  solver. Its options can be prescribed by pre-appending the flag
//  - ASolverMass_
//  to the PETSc options file.
//
// The factors FFa * D(M)^-1 can be combined together to apply some tiny
//  simplifications: the space-time Schur complement approximation becomes
//                 ⌈ I + dt*W*M^-1                               ⌉
// XX^-1 = CC^-1 * |      -I       I + dt*W*M^-1                 |.
//                 |                    -I       I + dt*W*M^-1   |
//                 ⌊                                  \\       \\⌋


IMHD2DSTMagneticSchurComplement::IMHD2DSTMagneticSchurComplement( const MPI_Comm& comm, double dt,
                                                                  const SparseMatrix* M, const SparseMatrix* W, const SparseMatrix* CCinv,
                                                                  const Array<int>& essTDOF, int verbose ):
  _comm(comm), _dt(dt), _eta(eta), _M(NULL), _W(NULL), _CCinv(NULL), _Msolve(NULL), _essTDOF(essTDOF), _verbose(verbose){

  if( M     != NULL ) SetM(     M     );
  if( W     != NULL ) SetW(     W     );
  if( CCinv != NULL ) SetCCinv( CCinv );

  // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
  //  to actually make use of the IG or not
  // TODO: delete this. Never use initial guess in preconditioners!
  iterative_mode = true;

  MPI_Comm_size( comm, &_numProcs );
  MPI_Comm_rank( comm, &_myRank );
}



IMHD2DSTMagneticSchurComplement::~IMHD2DSTMagneticSchurComplement(){
  delete _M;
  delete _Msolve;
}


// initialise info on mass matrix
void IMHD2DSTMagneticSchurComplement::SetM( const SparseMatrix* M ){
  delete _M;
  _M = new PetscParMatrix( M );

  height = M->Height();
  width  = M->Width();
  SetMSolve();
}

// initialise info on spatial operator
void IMHD2DSTMagneticSchurComplement::SetW( const SparseMatrix* W ){
  _W.MakeRef( *W );

  height = W->Height();
  width  = W->Width();
}



// initialise solver for mass matrix
void IMHD2DSTMagneticSchurComplement::SetMSolve(){
  delete _Msolve;

  _Msolve = new PetscLinearSolver( *_M, "ASolverMass_" );
  
}

// initialise solver for space-time wave equation
void IMHD2DSTMagneticSchurComplement::SetCCinv( const Solver* CCinv ){
  delete _CCinv;

  _CCinv = CCinv;
  
}






// Define multiplication by preconditioner - the most relevant function here
// XX^-1 = CC^-1 * FFa * D(M)^-1
void IMHD2DSTMagneticSchurComplement::Mult( const Vector &x, Vector &y ) const{

  MFEM_ASSERT( _Msolve!=NULL, "Solver for vector potential mass matrix not initialised" );
  MFEM_ASSERT( _CCinv !=NULL, "Solver for space-time wave equation not initialised" );
  MFEM_ASSERT( _W     !=NULL, "Spatial operator for space-time vector potential discretisation not initialised" );
  MFEM_ASSERT(x.Size() == Width(),  "invalid x.Size() = " << x.Size() << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size() << ", expected size = " << Height());

  // Initialise
  const int     lclSize = x.Size();
  const double* lclData = x.GetData();

  if ( _verbose>20 ){
    if ( _myRank==0 ){
      std::cout<<"Applying magnetic Schur complement approximation\n";
    }
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Rank: "<<_myRank<< ", rhs for A: "; x.Print(std::cout, x.Size());
      std::cout<<"Rank: "<<_myRank<< ", IG  for A: "; y.Print(std::cout, y.Size());
    }
  }

  // TODO: do I really need to copy this? Isn't there a way to force lclx to point at lclData and still be const?
  Vector lclx( lclSize ), prevx( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    lclx.GetData()[i] = lclData[i];
  }

  // Send my part of rhs to next proc
  MPI_Request reqSend, reqRecv;
  if( _myRank < _numProcs ){
    MPI_Isend( lclx.GetData(),  lclSize, MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
  }
  if( _myRank > 0 ){
    MPI_Irecv( prevx.GetData(), lclSize, MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
  }



  // Have each processor solve for the mass matrix
  Vector invMx( lclSize ), temp( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    invMx.GetData()[i] = y.GetData()[i];
  }
  _Msolve->Mult( lclx, invMx );

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" inverted vector potential mass matrix\n";
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Rank: "<<_myRank<< ", result after inverting mass matrix: ";
      invMx.Print(std::cout, invMx.Size());
    }
  }



  // Include contribution from spatial operator
  _W.Mult( invMx, temp );
  // - kill contributions from Dirichlet BC
  temp.SetSubVector( _essTDOF, 0.0 );

  if (_verbose>50 ){
    std::cout<<"Rank "<<_myRank<<" included contribution from spatial operator"<<std::endl;
    MPI_Barrier(_comm);
  }  



  // Include contribution from subdiagonal
  // - if you want to ignore the "time-stepping" structure in the preconditioner (that is,
  //    the contribution from the subdiag M/dt terms, just comment out the next few lines
  if( _myRank > 0 ){
    MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
    // - kill contributions from Dirichlet BC
    prevx.SetSubVector( _essTDOF, 0.0 );
    lclx -= prevx;
  }

#ifdef MULT_BY_DT
  temp *= _dt;        //multiply spatial part by dt
#else
  lclx *= (1./_dt);   //otherwise rescale the temporal part - careful not to dirty dirichlet BC
  for ( int i = 0; i < _essTDOF.Size(); ++i ){
    lclx.GetData()[_essTDOF(i)] *= _dt;
  }
#endif


  // Combine temporal and spatial parts
  temp += lclx;

  if (_verbose>50 ){
    if ( _myRank==0 ){
      std::cout<<"Applied FFa*D(M)^-1\n";
    }
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Rank: "<<_myRank<< ", result after applying FFa*D(M)^-1: ";
      temp.Print(std::cout, temp.Size());
    }
  }


  // TODO:
  // THIS IS A NEW ADDITION:
  // - Ignore action of solver on Dirichlet BC
  for ( int i = 0; i < _essTDOF.Size(); ++i ){
    temp.GetData()[_essTDOF(i)] = lclData[i];
  }




  // Apply space-time wave solver
  y = 0.;                   // kill initial guess
  _CCinv->Mult( temp, y );

  if (_verbose>50 ){
    if ( _myRank==0 ){
      std::cout<<"Applied CC^-1\n";
    }
    MPI_Barrier(_comm);
    if ( _verbose>100 ){
      std::cout<<"Rank: "<<_myRank<< ", result after applying CC^-1: ";
      y.Print(std::cout, y.Size());
    }
  }



  MPI_Barrier( _comm );

}






} // namespace mfem