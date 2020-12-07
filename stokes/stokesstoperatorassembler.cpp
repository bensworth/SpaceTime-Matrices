#include "stokesstoperatorassembler.hpp"
#include "vectorconvectionintegrator.hpp"
#include "spacetimesolver.hpp"
#include "pararealsolver.hpp"

#include <mpi.h>
#include <string>
#include <cstring>
#include <iostream>
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
// {NB: As a side note
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
  if( _essQhTDOF.Size() == 0 ){
    if( _myRank == 0 ){
      // std::cout<<"Assuming that pressure 'laplacian' has non-trivial kernel (constant functions)"<<std::endl;
      std::cout<<"Warning: the pressure 'laplacian' has non-trivial kernel (constant functions)."<<std::endl
               <<"         Make sure to flag that in the petsc options prescribing:"<<std::endl
               <<"         -for iterative solver: -PSolverLaplacian_ksp_constant_null_space TRUE"<<std::endl
               <<"         -for direct solver: -PSolverLaplacian_pc_factor_shift_type NONZERO"<<std::endl
               <<"                         and -PSolverLaplacian_pc_factor_shift_amount 1e-10"<<std::endl
               <<"                         (this will hopefully save us from 0 pivots in the singular mat)"<<std::endl;
      // TODO: or maybe just fix one unknown?
    }
    // TODO: for some reason, the following causes memory leak
    // // extract the underlying petsc object
    // PetscErrorCode ierr;
    // Mat petscA = Mat( *_Ap );
    // // initialise null space
    // MatNullSpace nsp = NULL;
    // MatNullSpaceCreate( PETSC_COMM_SELF, PETSC_TRUE, 0, NULL, &nsp); CHKERRV(ierr);
    // // // attach null space to matrix
    // MatSetNullSpace( petscA, nsp ); CHKERRV(ierr);
    // MatNullSpaceDestroy( &nsp ); CHKERRV(ierr);      // hopefully all info is stored
  }

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
  // // - again, kill every contribution on outflow boundary
  // lcly.SetSubVector( _essQhTDOF, 0.0 );
  // lclx.SetSubVector( _essQhTDOF, 0.0 );
  _Msolve->Mult( lclx, lcly );
  // lcly.SetSubVector( _essQhTDOF, 0.0 );

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










// // Now defined in its own file
// //******************************************************************************
// // Velocity block
// //******************************************************************************
// // These functions are used to aid in the application of the velocity part of
// //  the block preconditioner. That is, it approximates the inverse of the
// //  space-time velocity matrix FFu:
// //       ⌈ Fu*dt          ⌉
// // FFu = |  -M   Fu*dt    |,
// //       |        -M   \\ |
// //       ⌊             \\ ⌋
// //  where Fu = Mu/dt + mu Au is the spatial operator for the velocity

// SpaceTimeSolver::SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F, const SparseMatrix* M,
//                                   const Array<int>& essVhTDOF, bool timeDep, int verbose ):
//   _comm(comm), _timeDep(timeDep), _F(NULL), _M(NULL), _Fsolve(NULL), _essVhTDOF(essVhTDOF), _X(NULL), _Y(NULL), _verbose(verbose){

//   if( F != NULL ) SetF(F);
//   if( M != NULL ) SetM(M);

//   // Flag it as an iterative method so to reuse IG in any case. It's up to the setup of its internal solvers
//   //  to actually make use of the IG or not
//   iterative_mode = true;

//   MPI_Comm_size( comm, &_numProcs );
//   MPI_Comm_rank( comm, &_myRank );
// }



// SpaceTimeSolver::~SpaceTimeSolver(){
//   delete _Fsolve;
//   delete _F;
//   delete _X;
//   delete _Y;
// }



// // initialise info on spatial operator for the velocity.
// void SpaceTimeSolver::SetF( const SparseMatrix* F ){
//   _F = new PetscParMatrix( F );

//   height = F->Height();
//   width  = F->Width();
//   SetFSolve();
// }

// // initialise info on mass-matrix for the velocity.
// void SpaceTimeSolver::SetM( const SparseMatrix* M ){
//   _M = M;

//   height = M->Height();
//   width  = M->Width();
// }



// // initialise solver of spatial operator for the velocity.
// void SpaceTimeSolver::SetFSolve(){
//   delete _Fsolve;

//   if ( _timeDep || _myRank == 0 ){
//     _Fsolve = new PetscLinearSolver( *_F, "VSolver_" );

//     // PetscBool set;
//     // PetscErrorCode ierr;
//     // char optName[PETSC_MAX_PATH_LEN];
//     // _Fsolve->iterative_mode = true;  // trigger iterative mode...
//     // ierr = PetscOptionsGetString( NULL ,"VSolver_", "-ksp_type", optName, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
//     // if( !strcmp( optName, "preonly" ) ){
//     //   char optName1[PETSC_MAX_PATH_LEN];
//     //   ierr = PetscOptionsGetString( NULL ,"VSolver_", "-pc_type", optName1, PETSC_MAX_PATH_LEN, &set ); CHKERRV(ierr);
//     //   if(!( strcmp( optName1, "ilu" ) ) || !( strcmp( optName1, "lu" ) ) ){
//     //     _Fsolve->iterative_mode = false;  // ...unless you're using ilu or lu
//     //   }
//     // }
//   }
  
//   // if ( _verbose && _myRank==0 && _Fsolve->iterative_mode ){
//   //   std::cout<<"Selected iterative solver for velocity time-stepping"<<std::endl;
//   // }

// }





// // Define multiplication by preconditioner (that is, time-stepping on the
// //  velocity space-time block)- the most relevant function here
// void SpaceTimeSolver::Mult( const Vector &x, Vector &y ) const{
//   if ( _timeDep || _myRank == 0 ){
//     MFEM_ASSERT( _Fsolve != NULL, "Solver for velocity spatial operator not initialised" );
//   }
//   MFEM_ASSERT( _M      != NULL, "Velocity mass matrix not initialised" );
//   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
//               << ", expected size = " << Width());
//   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
//               << ", expected size = " << Height());


//   if( _verbose>20 ){
//     if ( _myRank==0 ){
//       std::cout<<"Applying exact velocity solver (time-stepping)\n";
//     }
//     MPI_Barrier(_comm);
//   }  

  

//   // Initialise
//   const int spaceDofs = x.Size();

//   // - convert data to internal HypreParVector for ease of use
//   auto x_data = x.HostRead();
//   auto y_data = y.HostReadWrite();
//   if ( _X == NULL){
//     int starts[2] = { spaceDofs*_myRank, spaceDofs*(_myRank+1) };
//     _X = new HypreParVector( _comm, spaceDofs * _numProcs, const_cast<double*>(x_data), starts );
//     _Y = new HypreParVector( _comm, spaceDofs * _numProcs, y_data,                      starts );
//   }else{
//     _X->SetData(const_cast<double*>(x_data));
//     _Y->SetData(y_data);
//   }


//   if ( _verbose>100 ){
//     std::cout<<"Inside V-block, rank "<<_myRank<<", rhs for u: "; x.Print(std::cout, x.Size());
//     std::cout<<"Inside V-block, rank "<<_myRank<<", IG  for u: "; y.Print(std::cout, y.Size());

//     std::ofstream myfile;
//     std::string myfilename = std::string("./results/rhsu.dat");
//     myfile.open( myfilename, std::ios_base::app );
//     _X->Print_HYPRE(myfile);
//     myfile.close( );

//     myfilename = std::string("./results/IGu.dat");
//     myfile.open( myfilename, std::ios_base::app );
//     _Y->Print_HYPRE(myfile);
//     myfile.close( );
//   }

  
//   // If the spatial operator is time-dependent, then each processor will have to solve for its own time-step
//   if ( _timeDep ){

//     // these will contain rhs for each time-step
//     Vector b( spaceDofs );
//     b = 0.;



//     // Main "time-stepping" routine
    
//     // - receive solution from previous processor (unless initial time-step)
//     if ( _myRank > 0 ){
//       MPI_Recv( b.GetData(), spaceDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
//     }

//     // - define rhs for this step (including contribution from sol at previous time-step
//     for ( int i = 0; i < spaceDofs; ++i ){
//       b.GetData()[i] += _X->GetData()[i];
//     }

//     // - solve for current time-step
//     //  --if an iterative solver is set, _Y acts as an initial guess
//     _Fsolve->Mult( b, *_Y );



//     if (_verbose>100 ){
//       if ( _myRank==0 ){
//         std::cout<<"Solved for time-step: ";
//       }
//       if ( _myRank<_numProcs-1){
//         std::cout<<_myRank<<", ";
//       }else{
//         std::cout<<_myRank<<std::endl;
//       }
//     }

//     // - send solution to following processor (unless last time-step)
//     if ( _myRank < _numProcs-1 ){
//       // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
//       _M->Mult( *_Y, b );
//       // - M is stored with negative sign for velocity, so flip it
//       b.Neg();
//       // NB: _M should be defined so that essential BC are not dirtied! So the next command is useless
//       // b.SetSubVector( _essVhTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution

//       MPI_Send( b.GetData(), spaceDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
//     }




//   // If the spatial operator is constant, however, we can have rank 0 take care of all the solutions
//   }else{
//     // - broadcast IG and rhs to each proc (overkill, as only proc 0 will play with it, but whatever)
//     // Man, valgrind really *hates* GlobalVector(). Better if I handle this myself
//     // Vector *glbRhs = _X->GlobalVector();
//     // Vector *glbIG  = _Y->GlobalVector();

  
//     // Master performs time-stepping and sends solution to other processors
//     if ( _myRank == 0 ){

//       // - this will contain rhs for each time-step
//       Vector b( spaceDofs );
//       b = 0.;


//       // Main time-stepping routine
//       for ( int t = 0; t < _numProcs; ++t ){

//         // - define rhs for this step (including contribution from sol at previous time-step - see below)
//         Vector lclRhs( spaceDofs );
//         if ( t==0 ){
//           b = *_X;
//         }else{
//           MPI_Recv( lclRhs.GetData(), spaceDofs, MPI_DOUBLE, t, t, _comm, MPI_STATUS_IGNORE );
//           b += lclRhs;
//         }
//         // for ( int i = 0; i < spaceDofs; ++i ){
//         //   b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
//         // }

//         // - initialise local vector containing solution at single time-step
//         Vector lclSol( spaceDofs );
//         if ( t==0 ){
//           lclSol = *_Y;
//         }else{
//           MPI_Recv( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t + _numProcs, _comm, MPI_STATUS_IGNORE );
//         }
//         // for ( int i = 0; i < spaceDofs; ++i ){
//         //   lclSol.GetData()[i] = glbIG->GetData()[spaceDofs*t + i];
//         // }

//         _Fsolve->Mult( b, lclSol );

//         //  TODO: Maybe set to 0 the velocity solution on the dirichlet nodes?
//         // lclSol.SetSubVector( _essVhTDOF, 0 );


//         if (_verbose>100 ){
//           if ( t==0 ){
//             std::cout<<"Rank "<<_myRank<<" solved for time-step ";
//           }
//           if ( t<_numProcs-1){
//             std::cout<<t<<", ";
//           }else{
//             std::cout<<t<<std::endl;
//           }

//         }

//         // - send local solution to corresponding processor
//         if( t==0 ){
//           // that is, myself if first time step
//           for ( int j = 0; j < spaceDofs; ++j ){
//             _Y->GetData()[j] = lclSol.GetData()[j];
//           }
//         }else{
//           // or the right slave it if solution is later in time
//           MPI_Send( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t+2*_numProcs, _comm );   // TODO: non-blocking + wait before solve?
//         }


//         // - include solution as rhs for next time-step
//         if( t < _numProcs-1 ){
//           _M->Mult( lclSol, b );
//           b.Neg();    //M has negative sign for velocity, so flip it
//           // NB: _M should be defined so that essential BC are not dirtied! So the next command is useless
//           // b.SetSubVector( _essVhTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution
//         }

//       }

//     }else{
//       // Slaves sends data on rhs and ig
//       MPI_Send( _X->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank            , _comm );   // TODO: non-blocking + wait before solve?
//       MPI_Send( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank+  _numProcs, _comm );   // TODO: non-blocking + wait before solve?
//       // and receive data on solution
//       MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank+2*_numProcs, _comm, MPI_STATUS_IGNORE );
//     }

//     // cleanup
//     // delete glbIG;
//     // delete glbRhs;

//   }

//   // Make sure we're all done
//   MPI_Barrier( _comm );


//   if ( _verbose>100 ){
//     std::cout<<"Inside V-block: Rank: "<<_myRank<< ", result for V: "; y.Print(std::cout, y.Size());
//   }

// }








// // Define multiplication by preconditioner (that is, time-stepping on the
// //  velocity space-time block)- the most relevant function here
// // NB: in this version, the spatial operator is supposed to be se same for
// //      every time-step (that is, the parameters of the PDE are not
// //      time-dependent). To (slightly) increase performance, then, it is
// //      the master which takes care of solving each time-step
// void SpaceTimeSolver::Mult( const Vector &x, Vector &y ) const{
//   MFEM_ASSERT( _Fsolve != NULL, "Solver for velocity spatial operator not initialised" );
//   MFEM_ASSERT( _M      != NULL, "Velocity mass matrix not initialised" );
//   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
//               << ", expected size = " << Width());
//   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
//               << ", expected size = " << Height());

//   if ( _verbose && _myRank==0 ){
//     std::cout<<"Applying velocity block preconditioner (time-stepping)"<<std::endl;
//   }
  

//   // Initialise
//   const int spaceDofs = x.Size();

//   // - convert data to internal HypreParVector for ease of use
//   auto x_data = x.HostRead();
//   auto y_data = y.HostReadWrite();
//   if ( _X == NULL){
//     int starts[2] = { spaceDofs*_myRank, spaceDofs*(_myRank+1) };
//     _X = new HypreParVector( _comm, spaceDofs * _numProcs, const_cast<double*>(x_data), starts );
//     _Y = new HypreParVector( _comm, spaceDofs * _numProcs, y_data,                      starts );
//   }else{
//     _X->SetData(const_cast<double*>(x_data));
//     _Y->SetData(y_data);
//   }

//   // - broadcast IG and rhs to each proc (overkill, as only proc 0 will play with it, but whatever)
//   const Vector *glbRhs = _X->GlobalVector();
//   const Vector *glbIG  = _Y->GlobalVector();


//   if ( _verbose && _myRank == 0 ){
//     std::cout<<"Inside V-block, rhs for u: "; glbRhs->Print(std::cout, glbRhs->Size());
//     std::cout<<"Inside V-block, IG  for u: ";  glbIG->Print(std::cout,  glbIG->Size());

//     std::ofstream myfile;
//     std::string myfilename = std::string("./results/rhsu") + std::to_string(_nCalls) + ".dat";
//     myfile.open( myfilename, std::ios_base::app );
//     glbRhs->Print(myfile,1);
//     myfile.close( );

//     myfilename = std::string("./results/IGu") + std::to_string(_nCalls) + ".dat";
//     myfile.open( myfilename, std::ios_base::app );
//     glbIG->Print(myfile,1);
//     myfile.close( );
//   }



//   // - initialise local vector containing solution at single time-step
//   Vector lclSol( spaceDofs );
//   for ( int i = 0; i < spaceDofs; ++i ){
//     lclSol.GetData()[i] = glbIG->GetData()[i];
//   }

//   // Master performs time-stepping and sends solution to other processors
//   if ( _myRank == 0 ){

//     // - these will contain rhs for each time-step
//     Vector b( spaceDofs );
//     b = 0.;


//     // Main time-stepping routine
//     for ( int t = 0; t < _numProcs; ++t ){

//       // - define rhs for this step (including contribution from sol at previous time-step - see below)
//       for ( int i = 0; i < spaceDofs; ++i ){
//         b.GetData()[i] += glbRhs->GetData()[ i + spaceDofs * t ];
//       }


//       // - solve for current time-step
//       //  --if an iterative solver is set, lclSol acts as an initial guess
//       //  --if no changes are made to lclSol, it simply picks the solution at
//       //     the previous time-step
//       // TODO: I feel like there's something terribly wrong here: if we change the IG
//       //        to be influenced by the sol at previous time-step, and pick an iterative
//       //        method as _Fsolve, then we are changing operator at each outer GMRES
//       //        iteration, which is no good!
//       // //  --here we copy from the IG of the global system for the dirichlet
//       // //     nodes in the velocity solution
//       // lclSol.SetSubVector( _essVhTDOF, &( glbIG->GetData()[spaceDofs * t] ) );
//       _Fsolve->Mult( b, lclSol );

//       //  TODO: Maybe set to 0 the velocity solution on the dirichlet nodes?
//       // lclSol.SetSubVector( _essVhTDOF, &( glbIG->GetData()[spaceDofs * t] ) );


//       if (_verbose ){
//         if ( t==0 ){
//           std::cout<<"Rank "<<_myRank<<" solved for time-step ";
//         }
//         if ( t<_numProcs-1){
//           std::cout<<t<<", ";
//         }else{
//           std::cout<<t<<std::endl;
//         }

//       }

//       // - send local solution to corresponding processor
//       if( t==0 ){
//         // that is, myself if first time step
//         for ( int j = 0; j < spaceDofs; ++j ){
//           _Y->GetData()[j] = lclSol.GetData()[j];
//         }
//       }else{
//         // or the right slave it if solution is later in time
//         MPI_Send( lclSol.GetData(), spaceDofs, MPI_DOUBLE, t, t, _comm );   // TODO: non-blocking + wait before solve?
//       }


//       // - include solution as rhs for next time-step
//       if( t < _numProcs-1 ){
//         _M->Mult( lclSol, b );
//         b.Neg();    //M has negative sign for velocity, so flip it
//         // NB: _M should be defined so that essential BC are not dirtied! So the next command is useless
//         // b.SetSubVector( _essVhTDOF, 0.0 );  // kill every contribution to the dirichlet-part of the solution
//       }

//     }

//   }else{
//     // Slaves receive data
//     MPI_Recv( _Y->GetData(), spaceDofs, MPI_DOUBLE, 0, _myRank, _comm, MPI_STATUS_IGNORE );
//   }

//   // Make sure we're all done
//   MPI_Barrier( _comm );


//   // delete glbRhs;
//   // delete glbIG;


//   if ( _verbose ){
//     std::cout<<"Inside V-block: Rank: "<<_myRank<< ", result for V: "; y.Print(std::cout, y.Size());
//   }
//   // std::string myfilename = std::string("./results/usol") + std::to_string(_nCalls) + ".dat";
//   // _Y->Print( myfilename.c_str() );
//   // _nCalls++;

// }















//******************************************************************************
// Space-time block assembly
//******************************************************************************
// These functions are useful for assembling all the necessary space-time operators

StokesSTOperatorAssembler::StokesSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
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
  
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  _dim = _mesh->Dimension();

  // - initialise FE info
  _VhFEColl  = new H1_FECollection( ordU, _dim );  

  if( ordP > 0 )
    _QhFEColl  = new H1_FECollection( ordP, _dim );
  else{
    _QhFEColl  = new L2_FECollection(    0, _dim );
    
    if ( _myRank == 0 ){
      std::cerr<<"WARNING: since you're using L2 pressure, you should definitely double-check the implementation of DG"<<std::endl;
    }
  }


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


  // _pSchur = new StokesSTPreconditioner( comm, dt, mu, NULL, NULL, NULL, _essQhTDOF, verbose );
  // _FFinv  = new SpaceTimeSolver(        comm,         NULL, NULL,       _essVhTDOF, verbose );


  if (_myRank == 0 ){
    std::cout << "***********************************************************\n";
    std::cout << "dim(Vh) = " << _VhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Qh) = " << _QhFESpace->GetTrueVSize() << "\n";
    std::cout << "***********************************************************\n";
  }

}



StokesSTOperatorAssembler::StokesSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
                                                      const int refLvl, const int ordU, const int ordP,
                                                      const double dt, const double mu, const double Pe,
                                                      void(  *f)(const Vector &, double, Vector &),
                                                      double(*g)(const Vector &, double ),
                                                      void(  *n)(const Vector &, double, Vector &),
                                                      const Vector &w,
                                                      void(  *u)(const Vector &, double, Vector &),
                                                      double(*p)(const Vector &, double ),
                                                      int verbose ):
  _comm(comm), _dt(dt), _mu(mu), _Pe(Pe), _fFunc(f), _gFunc(g), _nFunc(n), _wFunc(NULL), _wFuncCoeff(w), _uFunc(u), _pFunc(p), _ordU(ordU), _ordP(ordP),
  _MuAssembled(false), _FuAssembled(false), _MpAssembled(false), _ApAssembled(false), _WpAssembled(false), _BAssembled(false),
  _FFinvPrec(NULL), _FFAssembled(false), _BBAssembled(false), _pSAssembled(false), _FFinvAssembled(false),
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
  else{
    _QhFEColl  = new L2_FECollection(    0, _dim );
    
    if ( _myRank == 0 ){
      std::cerr<<"WARNING: since you're using L2 pressure, you should definitely double-check the implementation of DG"<<std::endl;
    }
  }


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


  if (_myRank == 0 ){
    std::cout << "***********************************************************\n";
    std::cout << "dim(Vh) = " << _VhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Qh) = " << _QhFESpace->GetTrueVSize() << "\n";
    std::cout << "***********************************************************\n";
  }

}









// Assemble operator on main diagonal of space-time matrix for velocity block:
//  Fu = M + mu*dt K + mu*Pe*dt*W
void StokesSTOperatorAssembler::AssembleFuVarf( ){
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


  // add convection integrator if necessary
  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for VectorConvectionIntegrator
  GridFunction wGridFun( _VhFESpace );
  wGridFun = _wFuncCoeff;

  if ( _Pe != 0. ){
    if ( _wFunc == NULL ){
      wCoeff = new VectorGridFunctionCoefficient( &wGridFun );
    }else{
      wCoeff = new VectorFunctionCoefficient( _dim, _wFunc );
      wCoeff->SetTime( _dt*(_myRank+1) );
    }
#ifdef MULT_BY_DT
    double muPeDt = _mu*_Pe*_dt;  
    _fuVarf->AddDomainIntegrator(new VectorConvectionIntegrator( *wCoeff, muPeDt ));
#else
    double muPe = _mu*_Pe;  
    _fuVarf->AddDomainIntegrator(new VectorConvectionIntegrator( *wCoeff, muPe ));
#endif
  }


  _fuVarf->Assemble();
  _fuVarf->Finalize();
  
  delete wCoeff;

  // _fuVarf->FormSystemMatrix( _essVhTDOF, _Fu );


  // _FuAssembled = true;


  // - once the matrix is generated, we can get rid of the operator
  // // NOT really! We impose dirichlet BC later
  // _Fu = fVarf->SpMat();
  // _Fu.SetGraphOwner(true);
  // _Fu.SetDataOwner(true);
  // fVarf->LoseMat();
  // delete fVarf;

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Velocity spatial operator Fu assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}



// Assemble operator on subdiagonal of space-time matrix for velocity block:
//  Mu = -M
void StokesSTOperatorAssembler::AssembleMuVarf( ){
  if( _MuAssembled ){
    return;
  }

	_muVarf = new BilinearForm(_VhFESpace);
#ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  _muVarf->AddDomainIntegrator(new VectorMassIntegrator( mone ));
#else
  ConstantCoefficient mdtinv( -1./_dt );
  _muVarf->AddDomainIntegrator(new VectorMassIntegrator( mdtinv ));
#endif
  _muVarf->Assemble();
  // mVarf->EliminateVDofs( _essVhTDOF, Matrix::DiagonalPolicy::DIAG_ZERO );   // eliminate essential nodes from matrix (to avoid dirtying dirichlet BC)
  _muVarf->Finalize();

  // // - once the matrix is generated, we can get rid of the operator
  // // NOT really! We impose dirichlet BC later
  // _Mu = mVarf->SpMat();
  // _Mu.SetGraphOwner(true);
  // _Mu.SetDataOwner(true);
  // mVarf->LoseMat();
  // delete mVarf;

  // _MuAssembled = true;


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Velocity mass-matrix (negative) Mu assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}



// Assemble -divergence operator:
//  B = -dt * div
// TODO: it really bothers me that I cannot just use FormRectangularSystemMatrix here
//  to recover the actual SparseMatrix representing B, and then reuse FormRectangularLinearSystem
//  to include BC / initialise the system properly. It seems to work for Fu, but here it throws
//  weird errors.
void StokesSTOperatorAssembler::AssembleBVarf( ){

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
  // // NOT really! We impose dirichlet BC later
 //  _B = bVarf->SpMat();
 //  _B.SetGraphOwner(true);
 //  _B.SetDataOwner(true);
 //  bVarf->LoseMat();
 //  delete bVarf;


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Divergence operator (negative) B assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}





// Assemble "laplacian" operator for pressure block:
// - This should be assembled as if it had homogeneous dirichlet BC on the outflow boundary
//    and homogeneous Neumann BC on the inflow boundary (dirichlet for u)
void StokesSTOperatorAssembler::AssembleAp( ){

  if( _ApAssembled ){
    return;
  }

  BilinearForm *aVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient one( 1.0 );     // diffusion
  // ConstantCoefficient beta( 1e6 );    // penalty term for weakly imposing dirichlet BC

  if ( _ordP == 0 ){  // DG
    double sigma = -1.0;
    double kappa =  1.0;
    aVarf->AddDomainIntegrator(      new DiffusionIntegrator( one ));                 // classical grad-grad inside each element
    aVarf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));   // contribution to jump across elements
    // aVarf->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));   // TODO: includes boundary contributions (otherwise you'd be imposing neumann?)
  }else{
    aVarf->AddDomainIntegrator(  new DiffusionIntegrator( one ));
    // Impose homogeneous dirichlet BC weakly via penalty method  -> Andy says it's not a good idea (and indeed results are baaad)
    // if( _essQhTDOF.Size()>0 ){
    //   aVarf->AddBoundaryIntegrator(new BoundaryMassIntegrator( beta ), _essQhTDOF );
    // }
  }
  // Impose homogeneous dirichlet BC by simply removing corresponding equations
  aVarf->Assemble();
  aVarf->Finalize();
  
  aVarf->FormSystemMatrix( _essQhTDOF, _Ap );
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf->LoseMat();

  // // TODO: FormSystemMatrix doesn't set diagonal to 1. Alternatively, one can do the following:
  // _Ap = aVarf->SpMat();
  // _Ap.SetGraphOwner(true);
  // _Ap.SetDataOwner(true);
  // aVarf->LoseMat();
  // for ( int i = 0; i < _essQhTDOF.Size(); ++i ){
  //   _Ap.EliminateRowCol( _essQhTDOF.GetData()[i] );
  // }

  // // TODO: if the system is singular, one can stabilise it by fixing randomly to 0 the value of pressure somewhere
  // if( _essQhTDOF.Size() == 0 ){
  //   _Ap.EliminateRowCol( 0 );
  // }



  delete aVarf;

  _ApAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Ap.dat";
    myfile.open( myfilename );
    _Ap.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Pressure stiffness matrix Ap assembled\n";
    }
    MPI_Barrier(_comm);
  }  

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


  // - once the matrix is generated, we can get rid of the operator
  _Mp = mVarf->SpMat();
  _Mp.SetGraphOwner(true);
  _Mp.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

  _MpAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Mp.dat";
    myfile.open( myfilename );
    _Mp.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Pressure mass matrix Mp assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}



// Assemble spatial part of pressure (convection) diffusion operator
// According Elman/Silvester/Wathen:
// - If convection is included, this should be assembled as if it had Robin BC: dp/dn + (w*n)*p = 0 (w is convective velocity)
// - If convection is not included, this should be assembled as if it had Neumann BC: dp/dn = 0
// However, it seems like it's best to just leave dirichlet BC in outflow, just like how Ap is assembled :/
// NB: This bit of code is hence intended to be used *only* for the double-glazing problem, where the velocity
//     field has Dirichlet BC everywhere!
void StokesSTOperatorAssembler::AssembleWp( ){

  if( _WpAssembled ){
    return;
  }

  if ( _myRank == 0 ){
    std::cout<<"Warning: the assembly of the spatial part of the PCD considers only Neumann BC on pressure."<<std::endl
             <<"         For this to make sense, you need to make sure that the test problem we're trying"  <<std::endl
             <<"         to solve has Dirichlet BC on velocity everywhere, and that the prescribed"         <<std::endl
             <<"         advection field is tangential to the boundary (w*n=0) if solving Oseen."<<std::endl;
  }


  BilinearForm *wVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient mu( _mu );
  wVarf->AddDomainIntegrator(new DiffusionIntegrator( mu ));
  if ( _ordP == 0 ){  // DG
    double sigma = -1.0;
    double kappa =  1.0;
    wVarf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
    // wVarf->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));  // to weakly impose Dirichlet BC - don't bother for now
  }

  // include convection if necessary
  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for ConvectionIntegrator
  GridFunction wGridFun( _VhFESpace );
  wGridFun = _wFuncCoeff;
  if( _Pe!= 0. ){
    if ( _wFunc == NULL ){
      wCoeff = new VectorGridFunctionCoefficient( &wGridFun );
    }else{
      wCoeff = new VectorFunctionCoefficient( _dim, _wFunc );
      wCoeff->SetTime( _dt*(_myRank+1) );
    }
    // TODO: should I impose Robin, then? Like this I'm still applying Neumann
    wVarf->AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, _mu*_Pe ));  // if used for NS, make sure both _mu*_Pe=1.0!!

    // // This includes Robin -> can't be bothered to implement it / test it: just pick a w: w*n = 0 on the bdr in your tests
    // if( _ordP == 0 ){
    //   // Counteract the n.Grad(u) term on the Dirichlet portion of the boundary
    //   double sigma = -1.0;
    //   double kappa =  1.0;
    //   wVarf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
    //   // Augment the n.Grad(u) term with a*u on the Robin portion of boundary
    //   wVarf->AddBdrFaceIntegrator(new BoundaryMassIntegrator(wFuncCoeff, _mu*_Pe));  //this won't work: I need to compute w*n!
    // }else{
    //   wVarf->AddBoundaryIntegrator(new MassIntegrator(wCoeff, _mu*_Pe) );
    // }
  }
  
  wVarf->Assemble();
  wVarf->Finalize();
  

  _Wp = wVarf->SpMat();
  _Wp.SetGraphOwner(true);
  _Wp.SetDataOwner(true);
  wVarf->LoseMat();

  delete wVarf;
  delete wCoeff;

  _WpAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Wp.dat";
    myfile.open( myfilename );
    _Wp.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial part of PCD operator Wp assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







// void StokesSTOperatorAssembler::UpdateWp( const VectorFunctionCoefficient& wFuncCoeff ){

//   if( _Pe == 0. ){
//     return;
//   }


//   BilinearForm *wVarf( new BilinearForm(_QhFESpace) );
//   ConstantCoefficient mu( _mu );
//   wVarf->AddDomainIntegrator(new DiffusionIntegrator( mu ));
//   if ( _ordP == 0 ){  // DG
//     double sigma = -1.0;
//     double kappa =  1.0;
//     wVarf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
//     // wVarf->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));  // to weakly impose Dirichlet BC - don't bother for now
//   }

//   // include convection if necessary
//   // TODO: should I impose Robin, then? Like this I'm still applying Neumann
//   wVarf->AddDomainIntegrator(new ConvectionIntegrator( wFuncCoeff, _mu*_Pe ));

//   // // This includes Robin -> can't be bothered to implement it / test it: just pick a w: w*n = 0 on the bdr in your tests
//   // if( _ordP == 0 ){
//   //   // Counteract the n.Grad(u) term on the Dirichlet portion of the boundary
//   //   double sigma = -1.0;
//   //   double kappa =  1.0;
//   //   wVarf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
//   //   // Augment the n.Grad(u) term with a*u on the Robin portion of boundary
//   //   wVarf->AddBdrFaceIntegrator(new BoundaryMassIntegrator(wFuncCoeff, _mu*_Pe));
//   // }else{
//   //   wVarf->AddBoundaryIntegrator(new MassIntegrator(wFuncCoeff, _mu*_Pe) );
//   // }

//   wVarf->Assemble();
//   wVarf->Finalize();
  
//   _Wp.Clear();
//   _Wp = wVarf->SpMat();
//   _Wp.SetGraphOwner(true);
//   _Wp.SetDataOwner(true);
//   wVarf->LoseMat();

//   delete wVarf;

//   _WpAssembled = true;


//   if ( _verbose>50 && _myRank == 0 ){
//     std::ofstream myfile;
//     std::string myfilename = "./results/out_final_Wp.dat";
//     myfile.open( myfilename );
//     _Wp.PrintMatlab(myfile);
//     myfile.close( );
//   }

//   if( _verbose>5 ){
//     if ( _myRank==0 ){
//       std::cout<<"Spatial part of PCD operator Wp assembled\n";
//     }
//     MPI_Barrier(_comm);
//   }  

// }








// <<  Deprecated >>>
// Solve Space-time system for velocity via SEQUENTIAL time-stepping
void StokesSTOperatorAssembler::TimeStepVelocity( const HypreParVector& rhs, HypreParVector*& sol ){

  AssembleFuVarf();
  AssembleMuVarf();

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
  AssembleFuVarf();

  // - subidagonal = -M
  AssembleMuVarf();


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


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time divergence operator BB assembled\n";
    }
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
  AssembleWp();

  _pSchur = new StokesSTPreconditioner( _comm, _dt, _mu, NULL, NULL, NULL, _essQhTDOF, _verbose );

  _pSchur->SetAp( &_Ap );
  _pSchur->SetMp( &_Mp );

  if( _Pe != 0. ){
    _pSchur->SetWp( &_Wp, false );    // if there is convection, then clearly Wp differs from Ap (must include pressure convection)
  }else if( _essQhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
    _pSchur->SetWp( &_Wp, true );    
  }else{
    // _pSchur->SetWp( &_Wp, false );
    _pSchur->SetWp( &_Wp, true );     // should be false, according to E/S/W!
    if( _myRank == 0 ){
      std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
               <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
               <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
    }
  }

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time pressure Schur complement inverse approximation XX^-1 assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}











// Assemble FF^-1 (top-left in preconditioner)
void StokesSTOperatorAssembler::AssembleFFinv( const int spaceTimeSolverType = 0 ){
  if ( _FFinvAssembled ){
    return;
  }


  switch (spaceTimeSolverType){
    // Use sequential time-stepping to solve for velocity space-time block
    case 0:{
      if(!( _MuAssembled && _FuAssembled ) && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble mass matrix and spatial operator for velocity first!"<<std::endl;
        return;
      }
      //                                             flag as time-dependent only if Pe is non-zero
      SpaceTimeSolver *temp  = new SpaceTimeSolver( _comm, NULL, NULL, _essVhTDOF, _Pe!=0., _verbose );

      temp->SetF( &_Fu );
      temp->SetM( &_Mu );
      _FFinv = temp;

      _FFinvAssembled = true;
      
      break;
    }

    // Use BoomerAMG with AIR set-up
    case 1:{
      if(! _FFAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble velocity space-time matrix first!"<<std::endl;
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
        std::cerr<<"ERROR: AssembleFFinv: need to assemble velocity space-time matrix first!"<<std::endl;
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

    // Use Parareal with coarse/fine solver of different accuracies
    case 3:{

      const int maxIt = 2;

      if( _numProcs <= maxIt ){
        if( _myRank == 0 ){
          std::cerr<<"ERROR: AssembleFFinv: Trying to set solver as "<<maxIt<<" iterations of Parareal, but the fine discretisation only has "<<_numProcs<<" nodes. "
                   <<"This is equivalent to time-stepping, so I'm picking that as a solver instead."<<std::endl;
        }
        
        AssembleFFinv( 0 );
        return;
      }


      PararealSolver *temp  = new PararealSolver( _comm, NULL, NULL, NULL, maxIt, _verbose );

      temp->SetF( &_Fu );
      temp->SetC( &_Fu ); // same operator is used for both! it's the solver that changes, eventually...
      temp->SetM( &_Mu );
      _FFinv = temp;

      _FFinvAssembled = true;
      
      break;
    }
    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<spaceTimeSolverType<<" not recognised."<<std::endl;
      }
    }
  }


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time velocity block inverse (approximation) FF^-1 assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







void StokesSTOperatorAssembler::SetUpBoomerAMG( HYPRE_Solver& FFinv, const int maxiter ){
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

  // - initialise relevant bilinear forms
  AssembleFuVarf();
  AssembleMuVarf();
  AssembleBVarf();

  if ( _verbose>50 ){
    if ( _myRank == 0 ){
      std::ofstream myfile;
      std::string myfilename = "./results/out_original_B.dat";
      myfile.open( myfilename );
      (_bVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_F.dat";
      myfile.open( myfilename );
      (_fuVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_M.dat";
      myfile.open( myfilename );
      (_muVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
    }
    MPI_Barrier(_comm);
  }



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

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for velocity assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // -- include initial conditions
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


    if ( _verbose>100 ){
      std::cout<<"Contribution from IC on u: "; u0form->Print(std::cout, u0form->Size());
    }

    // remember to reset function evaluation for u to the current time
    uFuncCoeff.SetTime( _dt*(_myRank+1) );


    delete u0form;

    if(_verbose>10){
      std::cout<<"Contribution from initial condition included\n"<<std::endl;
    }
  }




//   // -- adjust rhs to take dirichlet BC for previous time-step into account
//   if( _myRank > 0 ){
//     uFuncCoeff.SetTime( _dt*_myRank );
//     GridFunction uBC(_VhFESpace);
//     uBC.ProjectCoefficient(uFuncCoeff);
//     uBC.SetSubVectorComplement( _essVhTDOF, 0.0 );    // consider only BC nodes

//     Vector um1BC;
//     _muVarf->Mult( uBC, um1BC );                      // multiply Mu by it
//     um1BC.SetSubVector( _essVhTDOF, 0.0 );            // keep only contribution to non-BC nodes
// #ifndef MULT_BY_DT
//     um1BC *= (1./_dt);
// #endif
//     fform->operator-=( um1BC );                       // add to rhs (remember _muVarf is assembled with a minus sign)

//     // remember to reset function evaluation for u to the current time
//     uFuncCoeff.SetTime( _dt*(_myRank+1) );
 
//   }





  // - for pressure
  LinearForm *gform( new LinearForm );
  gform->Update( _QhFESpace );
  gform->AddDomainIntegrator( new DomainLFIntegrator( gFuncCoeff ) );  //int_\Omega g*q
  gform->Assemble();

#ifdef MULT_BY_DT
  gform->operator*=( _dt );
#endif


  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for pressure assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - adjust rhs to take dirichlet BC for current time-step into account
  // -- initialise function with BC
  GridFunction uBC(_VhFESpace);//, pBC(_QhFESpace);
  uBC.ProjectCoefficient(uFuncCoeff);
  // pBC.ProjectCoefficient(pFuncCoeff);
  // -- initialise local rhs
  Vector fRhsLoc(  fform->Size() );
  Vector gRhsLoc(  gform->Size() );
  // -- initialise local initial guess to exact solution
  Vector iguLoc( uBC.Size() );
  Vector igpLoc( gform->Size() );
  Vector empty2;
  iguLoc = uBC;
  iguLoc.SetSubVectorComplement( _essVhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  igpLoc = 0.;                                      // dirichlet BC are not actually imposed on p
  // igpLoc.SetSubVectorComplement( _essQhTDOF, 0.0);
  Array<int> empty;




  // ASSEMBLE LOCAL LINEAR SYSTEMS (PARTICULARLY, CONSTRAINED MATRICES) -----
  // - Assemble _Fu (and modify rhs to take dirichlet on u into account)
  // _fuVarf->FormLinearSystem(           _essVhTDOF,        uBC, *fform, _Fu, empty2, fRhsLoc );   % this causes err for Pe>=10, so instead do as below
  _Fu = _fuVarf->SpMat();
  // _Fu.SetGraphOwner(true);
  // _Fu.SetDataOwner(true);
  // _fuVarf->LoseMat();
  // delete _fuVarf;
  fRhsLoc = *fform;
  mfem::Array<int> cols(_Fu.Height());
  cols = 0;
  for (int i = 0; i < _essVhTDOF.Size(); ++i){
    cols[_essVhTDOF[i]] = 1;
  }
  _Fu.EliminateCols( cols, &uBC, &fRhsLoc );
  for (int i = 0; i < _essVhTDOF.Size(); ++i){
    _Fu.EliminateRow( _essVhTDOF[i], mfem::Matrix::DIAG_ONE );
    fRhsLoc(_essVhTDOF[i]) = uBC(_essVhTDOF[i]);
  }
  // - Assemble _B (and modify rhs to take dirichlet on u into account)
  _bVarf->FormRectangularLinearSystem( _essVhTDOF, empty, uBC, *gform, _B,  empty2, gRhsLoc );  // iguloc should still be initialised to uBC
  // - Assemble _Mu (and modify rhs to take dirichlet on u into account)
  uFuncCoeff.SetTime( _dt*_myRank );                // set uFunc to previous time-step
  GridFunction um1BC(_VhFESpace);
  um1BC.ProjectCoefficient(uFuncCoeff);

  Vector um1Rel( fRhsLoc.Size() );
  um1Rel = 0.0;
  _muVarf->EliminateVDofs( _essVhTDOF, um1BC, um1Rel, Matrix::DiagonalPolicy::DIAG_ZERO ); 


  if( _myRank > 0 ){
    // add to rhs (um1Rel should already take minus sign on _Mu into account)
    // NB: - no need to rescale by dt, as _Mu will be already scaled accordingly.
    //     - no need to flip sign, as _Mu carries with it already
    fRhsLoc += um1Rel;
  }

  // remember to reset function evaluation for u to the current time
  uFuncCoeff.SetTime( _dt*(_myRank+1) );

  // store velocity mass matrix - without contribution from dirichlet nodes
  _Mu = _muVarf->SpMat();


  _FuAssembled = true;
  _MuAssembled = true;
  _BAssembled  = true;

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Effect from Dirichlet BC (if prescribed) included in assembled blocks\n";
    }
    MPI_Barrier(_comm);
  }  




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
//   P^-1 = [ FF^-1  ///// ]
//          [ 0      XX^-1 ],
// where FF contains space-time matrix for velocity,
void StokesSTOperatorAssembler::AssemblePreconditioner( Operator*& FFi, Operator*& XXi, const int spaceTimeSolverType=0 ){

  //Assemble top-left block
  AssembleFFinv( spaceTimeSolverType );
  AssemblePS( );

  FFi = _FFinv;
  XXi = _pSchur;

}







// //<< deprecated >>
// void StokesSTOperatorAssembler::ApplySTOperatorVelocity( const HypreParVector*& u, HypreParVector*& res ){
//   // Initialise handy functions
//   const int     lclSize = u->Size();
//   const double* lclData = u->GetData();

//   Vector Umine( lclSize ), Uprev( lclSize ), lclres( lclSize ), temp( lclSize );
//   for ( int i = 0; i < lclSize; ++i ){
//     Umine.GetData()[i]   = lclData[i];
//     // lclres.GetData()[i] = 0.0;
//   }

//   // send my local solution to the following processor  
//   MPI_Request reqSend, reqRecv;

//   if( _myRank < _numProcs ){
//     MPI_Isend( Umine.GetData(), lclSize, MPI_DOUBLE, _myRank+1, _myRank,   _comm, &reqSend );
//   }
//   if( _myRank > 0 ){
//     MPI_Irecv( Uprev.GetData(), lclSize, MPI_DOUBLE, _myRank-1, _myRank-1, _comm, &reqRecv );
//   }

//   // assemble relevant matrices
//   AssembleFuVarf();
//   AssembleMuVarf();

//   // diagonal part
//   _Fu.Mult( Umine, lclres );

//   if( _myRank > 0 ){
//     // sub-diagonal part
//     MPI_Wait( &reqRecv, MPI_STATUS_IGNORE );
//     _Mu.Mult( Uprev, temp );    //Mu is minus mass matrix
//     lclres += temp;
//   }


//   // assemble resulting vector
//   Array<int> rowStarts(2);
//   rowStarts[0] = ( lclSize )*_myRank;
//   rowStarts[1] = ( lclSize )*(_myRank+1);

//   res = new HypreParVector( _comm, lclSize*_numProcs, lclres.StealData(), rowStarts.GetData() );

// }








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

  u->SetOwnership( 1 );
  p->SetOwnership( 1 );

}



// Each processor computes L2 error of solution at its time-step
void StokesSTOperatorAssembler::ComputeL2Error( const Vector& uh, const Vector& ph, double& err_u, double& err_p ){

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


  err_u  = u.ComputeL2Error(uFuncCoeff, irs);
  err_p  = p.ComputeL2Error(pFuncCoeff, irs);

  // for ( int i = 0; i < _numProcs; ++i ){
  //   if ( _myRank == i ){
  //     std::cout << "Instant t="       << _dt*(_myRank+1) << std::endl;
  //     std::cout << "|| uh - uEx ||_L2= " << err_u << "\n";
  //     std::cout << "|| ph - pEx ||_L2= " << err_p << "\n";
  //   }
  //   MPI_Barrier( _comm );
  // }
}


// // Each processor computes error (in the FE space norm) of solution at its time-step
// void StokesSTOperatorAssembler::ComputeVQError( const Vector& uh, const Vector& ph, double& err_u, double& err_p ){
//   const GridFunction u( _VhFESpace, uh.GetData() );
//   const GridFunction p( _QhFESpace, ph.GetData() );

//   int order_quad = 5;
//   const IntegrationRule *irs[Geometry::NumGeom];
//   for (int i=0; i < Geometry::NumGeom; ++i){
//     irs[i] = &(IntRules.Get(i, order_quad));
//   }

//   VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
//   FunctionCoefficient       pFuncCoeff(_pFunc);
//   uFuncCoeff.SetTime( _dt*(_myRank+1) );
//   pFuncCoeff.SetTime( _dt*(_myRank+1) );

//   if( _ordU == 0 ){
//     err_u  = u.ComputeL2Error(uFuncCoeff, irs);
//   }else{
//     err_u  = u.ComputeH1Error(uFuncCoeff, irs);
//   }

//   if( _ordP == 0 ){
//     err_p  = p.ComputeL2Error(pFuncCoeff, irs);
//   }else{
//     err_p  = p.ComputeH1Error(pFuncCoeff, irs);
//   }
  
//   // for ( int i = 0; i < _numProcs; ++i ){
//   //   if ( _myRank == i ){
//   //     std::cout << "Instant t="       << _dt*(_myRank+1) << std::endl;
//   //     std::cout << "|| uh - uEx ||_L2= " << err_u << "\n";
//   //     std::cout << "|| ph - pEx ||_L2= " << err_p << "\n";
//   //   }
//   //   MPI_Barrier( _comm );
//   // }
// }



void StokesSTOperatorAssembler::SaveExactSolution( const std::string& path="ParaView",
                                                   const std::string& filename="STstokes_Ex" ){
  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _VhFESpace );
    GridFunction *pFun = new GridFunction( _QhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
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

// store given approximate solution in paraview format
void StokesSTOperatorAssembler::SaveSolution( const HypreParVector& uh, const HypreParVector& ph,
                                              const std::string& path="ParaView", const std::string& filename="STstokes" ){
  
  // gather parallel vector
  Vector *uGlb = uh.GlobalVector();
  Vector *pGlb = ph.GlobalVector();


  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _VhFESpace );
    GridFunction *pFun = new GridFunction( _QhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
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

// This function is the same as above, but it doesn't rely on HypreParVector's
void StokesSTOperatorAssembler::SaveSolution( const Vector& uh, const Vector& ph,
                                              const std::string& path="ParaView",
                                              const std::string& filename="STstokes" ){
  const int blockSizeU = uh.Size();
  const int blockSizeP = ph.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( uh.GetData(), blockSizeU, MPI_DOUBLE, 0, 2*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 2*_myRank+1, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _VhFESpace );
    GridFunction *pFun = new GridFunction( _QhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
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


    // this will store the approximate solution at current time-step
    Vector uLcl(blockSizeU), pLcl(blockSizeP);

    // handle first time-step separately
    *uFun = uh;
    *pFun = ph;
    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( uLcl.GetData(), blockSizeU, MPI_DOUBLE, t-1, 2*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 2*(t-1)+1, _comm, MPI_STATUS_IGNORE );

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





// Saves a plot of the error
void StokesSTOperatorAssembler::SaveError( const Vector& uh, const Vector& ph,
                                           const std::string& path="ParaView",
                                           const std::string& filename="STstokes" ){
  const int blockSizeU = uh.Size();
  const int blockSizeP = ph.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( uh.GetData(), blockSizeU, MPI_DOUBLE, 0, 2*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 2*_myRank+1, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _VhFESpace );
    GridFunction *pFun = new GridFunction( _QhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link uFun and pFun
    paraviewDC.RegisterField( "u-uh", uFun );
    paraviewDC.RegisterField( "p-ph", pFun );

    // this will store the approximate solution at current time-step
    Vector uLcl(blockSizeU), pLcl(blockSizeP);

    // these will provide exact solution
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);

    // error at instant 0 is 0 (IC)
    *uFun = 0.;
    *pFun = 0.;
    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();

    // handle first time-step separately
    uFuncCoeff.SetTime( _dt );
    pFuncCoeff.SetTime( _dt );
    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );

    uFun->operator-=( uh );
    pFun->operator-=( ph );

    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( uLcl.GetData(), blockSizeU, MPI_DOUBLE, t-1, 2*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 2*(t-1)+1, _comm, MPI_STATUS_IGNORE );

      // - assign to linked variables
      uFuncCoeff.SetTime( _dt*t );
      pFuncCoeff.SetTime( _dt*t );
      uFun->ProjectCoefficient( uFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      uFun->operator-=( uLcl );
      pFun->operator-=( pLcl );
      
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
  delete _FFinvPrec;
  delete _FFF;
  if( _FFAssembled )
    HYPRE_IJMatrixDestroy( _FF );
  if( _BBAssembled )
    HYPRE_IJMatrixDestroy( _BB );

  delete _fuVarf;
  delete _muVarf;
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










