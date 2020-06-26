#include "mfem.hpp"
// #include "petsc.h"
#include "stokesstoperatorassembler.hpp"
#include <fstream>
#include <iostream>
//---------------------------------------------------------------------------
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;
//---------------------------------------------------------------------------
void   uFun_ex( const Vector & x, const double t, Vector & u );
double pFun_ex( const Vector & x, const double t             );
void   fFun(    const Vector & x, const double t, Vector & f );
void   nFun(    const Vector & x, const double t, Vector & f );
double gFun(    const Vector & x, const double t             );
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
   StopWatch chrono;

   // for now, assume no spatial parallelisation: each processor handles a time-step
   int numProcs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool verbose = (myid == 0);
   double Tend = 1.;
   // Types of solver for preconditioner:
   // 0 -> CG+jacobi precon (doesn't work well for Ap, probably because it's singular?)
   // 1 -> 1 jacobi iteration
   // 2 -> 1 jacobi iteration with lumped operator (not available for A)
   int MpsolveType = 0;
   int ApsolveType = 1;
   int FsolveType  = 0;

   const char *mesh_file = "../../../3rdParty/MFEM/data/inline-tri.mesh";   
   int ordV = 2;
   int ordP = 1;
   int ref_levels = 0;
   const char *petscrc_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ordV, "-oU", "--order",
                  "Finite element order (polynomial degree) for velocity field.");
   args.AddOption(&ordP, "-oP", "--order",
                  "Finite element order (polynomial degree) for pressure field.");
   args.AddOption(&ref_levels, "-r", "--rlevel",
                  "Refinement level.");
   args.AddOption(&Tend, "-T", "--Tend",
                  "Final time.");
   args.AddOption(&MpsolveType, "-M", "--Msolve",
                  "Solver for pressure mass matrix: 0 -CG+Jacobi prec, 1 -Jacobi, 2 -Jacobi with lumped operator.");
   args.AddOption(&ApsolveType, "-A", "--Asolve",
                  "Solver for pressure 'laplacian': 0 -CG+Jacobi prec, 1 -Jacobi.");
   args.AddOption(&FsolveType, "-F", "--Fsolve",
                  "Solver for velocity spatial operator: 0 -CG+Jacobi prec, 1 -Jacobi, 2 -Jacobi with lumped operator");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.Parse();
   if(!args.Good())
   {
      if(verbose)
      {
        args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if(verbose)
   {
      args.PrintOptions(cout);
   }

   double dt = Tend / numProcs;
   double mu = 1.;                // CAREFUL NOT TO CHANGE THIS! Or if you do, re-define the normal derivative, too


   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);



  // Assembles block matrices composing the system
  StokesSTOperatorAssembler stokesAssembler( MPI_COMM_WORLD, mesh_file, ref_levels, ordV, ordP,
                                             dt, mu, fFun, nFun, uFun_ex, pFun_ex );

  HypreParMatrix *FFF, *BBB, *BBt;
  Operator *FFi, *XXi;
  HypreParVector  *frhs, *uEx, *pEx;
  stokesAssembler.AssembleOperator( FFF, BBB );

  stokesAssembler.AssemblePreconditioner( FFi, XXi, MpsolveType, ApsolveType, FsolveType );
  stokesAssembler.AssembleRhs( frhs );
  BBt = BBB->Transpose( );


  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = FFF->NumRows();
  offsets[2] = BBB->NumRows();
  offsets.PartialSum();

  BlockVector sol(offsets), rhs(offsets);
  rhs.GetBlock(0) = *frhs;

  BlockOperator *stokesOp = new BlockOperator( offsets );
  stokesOp->SetBlock(0, 0, FFF);
  stokesOp->SetBlock(0, 1, BBt);
  stokesOp->SetBlock(1, 0, BBB);

  // BlockOperator *stokesPr = new BlockOperator( offsets );
  // stokesPr->SetBlock(0, 0, FFi);
  // stokesPr->SetBlock(1, 1, XXi);


  stokesAssembler.ExactSolution( uEx, pEx );
  BlockVector solEx(offsets), rhsEx(offsets);
  solEx.GetBlock(0) = *uEx;
  solEx.GetBlock(1) = *pEx;
  stokesOp->Mult( solEx, rhsEx );


  // HypreParVector myRhs( *FFF, 1 );
  // HypreParVector *solU;
  // BBB->MultTranspose( *pEx, myRhs );
  // myRhs.Neg();
  // myRhs += *frhs;
  // stokesAssembler.TimeStepVelocity( myRhs, solU );

  // stokesAssembler.ComputeL2Error( *solU, *pEx );


  // PetscLinearSolver *solver;
  // solver = new PetscLinearSolver(MPI_COMM_WORLD,"ksp_type gmres");
  // PetscPreconditioner *pstokesPr = NULL;

  FGMRESSolver solver(MPI_COMM_WORLD);
  BlockDiagonalPreconditioner pstokesPr(offsets);
  pstokesPr.SetDiagonalBlock( 0, FFi );
  pstokesPr.SetDiagonalBlock( 1, XXi );

  solver.SetPreconditioner(pstokesPr);



  solver.SetOperator(*stokesOp);
  // solver.SetTol(1e-10);
  solver.SetAbsTol(1e-10);
  // solver.SetMaxIter((stokesOp->NumRows())*numProcs);
  solver.SetMaxIter( FFF->GetGlobalNumRows() + BBB->GetGlobalNumRows() );
  solver.SetPrintLevel(1);
  solver.Mult(rhs, sol);
  // solver.Mult(rhsEx, sol);


  if (myid == 0){
    if (solver.GetConverged()){
      std::cout << "Solver converged in "         << solver.GetNumIterations();
    }else{
      std::cout << "Solver did not converge in "  << solver.GetNumIterations();
    }
    std::cout << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
  }

  int colsV[2] = { myid*(FFF->NumRows()), (myid+1)*(FFF->NumRows()) };
  int colsP[2] = { myid*(BBB->NumRows()), (myid+1)*(BBB->NumRows()) };

  HypreParVector uh( MPI_COMM_WORLD, numProcs*(FFF->NumRows()), sol.GetBlock(0).GetData(), colsV ); 
  HypreParVector ph( MPI_COMM_WORLD, numProcs*(BBB->NumRows()), sol.GetBlock(1).GetData(), colsP ); 

  stokesAssembler.SaveSolution( uh, ph );
  stokesAssembler.SaveExactSolution( );

  // {
  //   Vector uga( rhs.GetBlock(0).Size() );
  //   for ( int i = 0; i < uga.Size(); ++i ){
  //     uga.GetData()[i] = rhs.GetBlock(0).GetData()[i] - rhsEx.GetBlock(0).GetData()[i];
  //   }
  //   std::cout<<"Delta between rhs: "<<uga.Max() <<", "<<uga.Min()<<std::endl;
  // }


  // stokesAssembler.ExactSolution( uEx, pEx );
  // BlockVector solEx(offsets), rhsEx(offsets);
  // solEx.GetBlock(0) = *uEx;
  // solEx.GetBlock(1) = *pEx;

  // stokesOp->Mult( solEx, rhsEx );
  // // - these should all print 0   - pressure seems fine (duh), velocity not really
  // if( myid == 1){
  //   for ( int i = 0; i < rhsEx.Size(); ++i ){
  //     std::cout<<rhsEx.GetData()[i] - rhs.GetData()[i]<<", ";
  //   }
  // }

  // HypreParVector fullRhs( *uEx );
  // BBB->MultTranspose( *pEx, fullRhs );
  // fullRhs.Neg();
  // fullRhs += *frhs;

  // HypreParVector *uh2;
  // stokesAssembler.TimeStepVelocity( fullRhs, uh2 );
  // // // - these should all print 0
  // // for ( int i = 0; i < FFF->NumRows(); ++i ){
  // //   std::cout<<uh2->GetData()[i] - uEx->GetData()[i]<<", ";
  // // }

  // stokesAssembler.ComputeL2Error( *uh2, *pEx );
  


  // delete uh;
  delete FFF;
  delete BBB;
  delete BBt;
  delete frhs;
  delete uEx;
  delete pEx;

   

  MFEMFinalizePetsc();
  MPI_Finalize();

  return 0;
}





/*
//Constant velocity ------------------------------------------------------
// Exact solution (velocity)
void uFun_ex(const Vector & x, const double t, Vector & u){
  u(0) = 1.;
  u(1) = 1.;
}

// Exact solution (pressure)
double pFun_ex(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return (t+1.) * xx * yy;
}

// Rhs (velocity)
void fFun(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  f(0) =  (t+1.) * yy;
  f(1) =  (t+1.) * xx;
}

// Normal derivative of velocity * mu
void nFun(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun(const Vector & x, const double t ){
  return 0.0;
}
//*/


/*
//Constant pressure = 0------------------------------------------------------
// Exact solution (velocity)
void uFun_ex(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = (t+1.) * sin(M_PI*xx) * sin(M_PI*yy);
  u(1) = (t+1.) * cos(M_PI*xx) * cos(M_PI*yy);
}

// Exact solution (pressure)
double pFun_ex(const Vector & x, const double t ){
  return 0.0;
}

// Rhs (velocity)
void fFun(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  f(0) = (t+1.) * ( 2.0*M_PI*M_PI*sin(M_PI*xx)*sin(M_PI*yy) ) + sin(M_PI*xx)*sin(M_PI*yy);
  f(1) = (t+1.) * ( 2.0*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) ) + cos(M_PI*xx)*cos(M_PI*yy);
}

// Normal derivative of velocity * mu
void nFun(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n(0) = 0.0;
  if ( xx == 1. || xx == 0. ){
    n(0) = -(t+1.) * M_PI * sin( M_PI*yy );
  }
  if ( yy == 1. || yy == 0. ){
    n(0) = -(t+1.) * M_PI * sin( M_PI*xx );
  }
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun(const Vector & x, const double t ){
  return 0.0;
}
//*/



/*
// Simple stuff to check everything works  ----------------------------------
// Exact solution (velocity)
void uFun_ex(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = t*yy*yy;
  u(1) = t*xx*xx;
}


// Exact solution (pressure)
double pFun_ex(const Vector & x, const double t ){
  return 0.;
}

// Rhs (velocity)
void fFun(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  f(0) = 2.*t;
  f(1) = 2.*t;
}

// Normal derivative of velocity * mu
void nFun(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n(0) = 0.0;
  n(1) = 0.0;
  if(      xx==0. ) n(1) = -xx;
  else if( xx==1. ) n(1) =  xx;
  else if( yy==0. ) n(0) = -yy;
  else if( yy==1. ) n(0) =  yy;
  n(0) *= 2*t;
  n(1) *= 2*t;
}

// Rhs (pressure) - unused
double gFun(const Vector & x, const double t ){
  return 0.0;
}
//*/







/*
// Velocity with null flux --------------------------------------------------
// Exact solution (velocity)
void uFun_ex(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));

  u(0) =   (t+1.) * xx*xx*xx * yy*yy * (yy-1)*(yy-1) * ( 6*xx*xx - 15*xx + 10 )/10.;
  u(1) = - (t+1.) * yy*yy*yy * xx*xx * (xx-1)*(xx-1) * ( 6*yy*yy - 15*yy + 10 )/10.;
}


// Exact solution (pressure)
double pFun_ex(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return xx*yy;
}

// Rhs (velocity)
void fFun(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  // Laplacian (minus sign):
  f(0) = -(t+1.) * yy*yy * xx * ( 2*xx*xx - 3*xx + 1 ) * (yy-1)*(yy-1) * 6. + xx*xx*xx * ( 6*xx*xx - 15*xx + 10 ) * ( 6*yy*yy- 6*yy + 1 ) / 5.;
  f(1) =  (t+1.) * xx*xx * yy * ( 2*yy*yy - 3*yy + 1 ) * (xx-1)*(xx-1) * 6. - yy*yy*yy * ( 6*yy*yy - 15*yy + 10 ) * ( 6*xx*xx- 6*xx + 1 ) / 5.;
  // + time derivative
  f(0) += xx*xx*xx * yy*yy * (yy-1)*(yy-1) * ( 6*xx*xx - 15*xx + 10 )/10.;
  f(1) -= yy*yy*yy * xx*xx * (xx-1)*(xx-1) * ( 6*yy*yy - 15*yy + 10 )/10.;
  // + pressure gradient
  f(0) += (t+1.) * yy;
  f(1) += (t+1.) * xx;
}

// Normal derivative of velocity * mu
void nFun(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun(const Vector & x, const double t ){
  return 0.0;
}
//*/







//Test case of some actual relevance-----------------------------------------
// Exact solution (velocity)
void uFun_ex(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = (t+1.) * sin(M_PI*xx) * sin(M_PI*yy);
  u(1) = (t+1.) * cos(M_PI*xx) * cos(M_PI*yy);
}

// Exact solution (pressure)
double pFun_ex(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return (t+1.) * sin(M_PI*xx) * cos(M_PI*yy);
}

// Rhs (velocity)
void fFun(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  f(0) = (t+1.) * ( 2.0*M_PI*M_PI*sin(M_PI*xx)*sin(M_PI*yy) + M_PI*cos(M_PI*xx)*cos(M_PI*yy) ) + sin(M_PI*xx)*sin(M_PI*yy);
  f(1) = (t+1.) * ( 2.0*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) - M_PI*sin(M_PI*xx)*sin(M_PI*yy) ) + cos(M_PI*xx)*cos(M_PI*yy);
}

// Normal derivative of velocity * mu
void nFun(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n(0) = 0.0;
  if ( xx == 1. || xx == 0. ){
    n(0) = -(t+1.) * M_PI * sin( M_PI*yy );
  }
  if ( yy == 1. || yy == 0. ){
    n(0) = -(t+1.) * M_PI * sin( M_PI*xx );
  }
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun(const Vector & x, const double t ){
  return 0.0;
}
//*/












