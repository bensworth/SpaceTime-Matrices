//---------------------------------------------------------------------------
// Implementation of Kelvin-Helmholtz Instability equation
//  dt(w) + v•∇w -mu ∇•∇w     = f
//             w - ∇•∇\phi    = g
//             v + ∇x(\phi k) = h
// with space-time block preconditioning. Notice the curl term in the last
// equation can be rewritten as ∇x(\phi k) = J ∇(\phi), with J =[0,1; -1,0]
// 
// - Solver support from PETSc
//
// Author: Federico Danieli, Numerical Analysis Group
// University of Oxford, Dept. of Mathematics
// email address: federico.danieli@maths.ox.ac.uk  
// September 2020; Last revision: Sep-2020
//
//
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "khistoperatorassembler.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
//---------------------------------------------------------------------------
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;
//---------------------------------------------------------------------------
// - double-glazing problem
double wFun_1( const Vector & x, const double t             );
double pFun_1( const Vector & x, const double t             );
void   vFun_1( const Vector & x, const double t, Vector & u );
double fFun_1( const Vector & x, const double t             );
double gFun_1( const Vector & x, const double t             );
void   hFun_1( const Vector & x, const double t, Vector & f );
//---------------------------------------------------------------------------
// Handy function for monitoring quantities of interest - predefinition
struct UPErrorMonitorCtx{// Context of function to monitor actual error
  int lenghtW;
  int lenghtV;
  int lenghtP;
  KHISTOperatorAssembler* STassembler;
  MPI_Comm comm;
  PetscViewerAndFormat *vf;
  std::string path;
};

PetscErrorCode UPErrorMonitorDestroy( void ** mctx );
PetscErrorCode UPErrorMonitor( KSP ksp, PetscInt n, PetscReal rnorm, void *mctx );
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  StopWatch chrono;

  // for now, assume no spatial parallelisation: each processor handles a time-step
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  double Tend = 1.;

  // Initialise problem parameters
  int ordW = 2;
  int ordP = 2;
  int ordV = 1;
  int ref_levels = 4;
  const char *petscrc_file = "rc_SpaceTimeKHI";
  int verbose = 0;
  int precType = 1;
  int STSolveType = 0;
  int pbType = 1;   
  string pbName = "";
  int output = 2;

  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&ordW, "-oW", "--orderW",
                "Finite element order (polynomial degree) for w field (default: 2)");
  args.AddOption(&ordP, "-oP", "--orderPhi",
                "Finite element order (polynomial degree) for phi field (default: 2)");
  args.AddOption(&ordV, "-oV", "--orderV",
                "Finite element order (polynomial degree) for velocity field (default: 1)");
  args.AddOption(&ref_levels, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&Tend, "-T", "--Tend",
                "Final time (default: 1.0)");
  args.AddOption(&pbType, "-Pb", "--problem",
                "Problem.");
  args.AddOption(&precType, "-P", "--preconditioner",
                "Type of preconditioner: 0-block diagonal, 1-block triangular (default)");
  args.AddOption(&STSolveType, "-ST", "--spacetimesolve",
                "Type of solver for velocity space-time matrix: 0-time stepping (default), 1-boomerAMG (AIR), 2-GMRES+boomerAMG (AIR), 3-Parareal");
  args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                "PetscOptions file to use.");
  args.AddOption(&verbose, "-V", "--verbose",
                "Control how much info to print to terminal.");
  args.AddOption(&output, "-out", "--outputsol",
                "Choose how much info to store on disk. 0-nothing, 1+#it to convergence, 2+residual evolution (default), 3+paraview plot of exact (if available) and approximate solution");
  args.Parse();
  if(!args.Good()){
    if(myid == 0)
    {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }

  // -initialise remaining parameters
  const double dt = Tend / numProcs;
  const double mu = 1.;                // CAREFUL NOT TO CHANGE THIS! Or if you do, re-define the normal derivative, too

  double(*wFun)( const Vector & x, const double t             );
  double(*pFun)( const Vector & x, const double t             );
  void(  *vFun)( const Vector & x, const double t, Vector & u );
  double(*fFun)( const Vector & x, const double t             );
  double(*gFun)( const Vector & x, const double t             );
  void(  *hFun)( const Vector & x, const double t, Vector & f );

  std::string mesh_file;

  switch (pbType){
    // analytical test-case
    case 0:{
      mesh_file = "./meshes/tri-square-open.mesh";
      
      wFun = wFun_1;
      pFun = pFun_1;
      vFun = vFun_1;
      fFun = fFun_1;
      gFun = gFun_1;
      hFun = hFun_1;
      pbName = "Analytic";
      break;
    }
    default:
      std::cerr<<"ERROR: Problem type "<<pbType<<" not recognised."<<std::endl;
  }

  if(myid == 0){
    args.PrintOptions(cout);
    std::cout<<"   --np "<<numProcs<<std::endl;
    std::cout<<"   --dt "<<Tend/numProcs<<std::endl;
  }


  // - initialise petsc
  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);



  // ASSEMBLE OPERATORS -----------------------------------------------------
  // Assembles block matrices composing the system
  TODO
  KHISTOperatorAssembler *khiAssembler = new KHISTOperatorAssembler( MPI_COMM_WORLD, mesh_file, ref_levels, ordW, ordP, ordV,
                                                                     dt, mu, Pe, fFun, gFun, nFun, wFun, uFun_ex, pFun_ex, verbose );

  HypreParMatrix *FFF, *AAA, *MMv, *BBB, *Mwp, *CCC;
  HypreParVector  *frhs, *grhs, *hrhs, *wEx, *pEx, *vEx, *W0, *P0, *V0;
  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators and rhs in main system blocks ********\n";
  }

  khiAssembler->AssembleSystem( FFF,  AAA,  MMv, BBB,  Mwp,  CCC,
                                frhs, grhs, hrhs, IGw,  IGp,  IGv );


  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators for preconditioner *******************\n";
  }

  Operator *FFi, *AAi, *Mvi;
  khiAssembler->AssemblePreconditioner( FFi, AAi, Mvi, STSolveType );



  if( myid == 0 && verbose > 0 ){
    std::cout << "Set-up system, solver and preconditioner ******************\n";
  }

  Array<int> offsets(4);
  BlockVector *sol, *rhs;
  BlockOperator *KHIOp;

  // Define solver
  PetscLinearSolver *solver = new PetscLinearSolver(MPI_COMM_WORLD, "solver_");
  // TODO: iterative mode triggers use of initial guess: check if it is indeed iterative
  bool isIterative = true;
  solver->iterative_mode = isIterative;

  // Define preconditioner
  BlockUpperTriangularPreconditioner *KHIPr;


  switch(precType){
    case 0:{
      // (v,phi,w) ordering
      offsets[0] = 0;
      offsets[1] = MMv->NumRows();
      offsets[2] = AAA->NumRows();
      offsets[3] = FFF->NumRows();
      offsets.PartialSum();

      sol = new BlockVector(offsets);
      rhs = new BlockVector(offsets);

      KHIOp = new BlockOperator( offsets );
      KHIOp->SetBlock(0, 0, MMv);
      KHIOp->SetBlock(0, 1, CCC);
      KHIOp->SetBlock(1, 1, AAA);
      KHIOp->SetBlock(1, 2, Mwp);
      KHIOp->SetBlock(2, 2, FFF);
      KHIOp->SetBlock(2, 0, BBB);

      KHIPr = new BlockUpperTriangularPreconditioner(offsets);
      KHIPr->iterative_mode = false;
      KHIPr->SetDiagonalBlock( 0, Mvi );
      KHIPr->SetDiagonalBlock( 1, AAi );
      KHIPr->SetDiagonalBlock( 2, FFi );
      KHIPr->SetBlock( 0, 1, CCC );
      KHIPr->SetBlock( 1, 2, Mwp );


      break;
    }
    case 1:{
      TODO

      break;
    }
    default:{
      if ( myid == 0 ){
        std::cerr<<"ERROR: Option for preconditioner "<<precType<<" not recognised"<<std::endl;
      }
      break;
    }
  }



  solver->SetPreconditioner(*KHIPr);
  solver->SetOperator(*KHIOp);


  TODO
  // Save residual evolution to file
  if ( output>1 ){
    // - create folder which will store all files with various convergence evolution 
    string path = string("./results/convergence_results") + "_Prec" + to_string(precType) + "_STsolve" + to_string(STSolveType)
                           + "_oW" + to_string(ordW) + "_oP" + to_string(ordP) + "_oV" + to_string(ordV) + "_Pb" + to_string(pbType);
    if(pbType == 4 ){
      path += "_Pe" + to_string(Pe);
    }
    path += string("_") + petscrc_file + "/";

    if (!std::experimental::filesystem::exists( path )){
      std::experimental::filesystem::create_directories( path );
    }
    string filename = path + "NP" + to_string(numProcs) + "_r"  + to_string(ref_levels) + ".txt";
    // - create viewer to instruct KSP object how to print residual evolution to file
    PetscViewer    viewer;
    PetscViewerAndFormat *vf;
    PetscViewerCreate( PETSC_COMM_WORLD, &viewer );
    PetscViewerSetType( viewer, PETSCVIEWERASCII );
    PetscViewerFileSetMode( viewer, FILE_MODE_APPEND );
    PetscViewerFileSetName( viewer, filename.c_str() );
    // - register it to the ksp object
    KSP ksp = *solver;
    PetscViewerAndFormatCreate( viewer, PETSC_VIEWER_DEFAULT, &vf );
    PetscViewerDestroy( &viewer );
    // - create a more complex context if fancier options must be printed (error wrt analytical solution)
    // UPErrorMonitorCtx mctx;
    // mctx.lenghtU = offsets[1];
    // mctx.lenghtP = offsets[2] - offsets[1];
    // mctx.STassembler = stokesAssembler;
    // mctx.comm = MPI_COMM_WORLD;
    // mctx.vf   = vf;
    // mctx.path = path + "/ParaView/NP" + to_string(numProcs) + "_r"  + to_string(ref_levels)+"/";
    // UPErrorMonitorCtx* mctxptr = &mctx;
    // if( pbType == 2 || pbType == 20 ){
    //   if ( myid == 0 ){
    //     std::cout<<"Warning: we're printing the error wrt the analytical solution at each iteration."<<std::endl
    //              <<"         This is bound to slow down GMRES *a lot*, so leave this code only for testing purposes!"<<std::endl;
    //   }
    //   KSPMonitorSet( ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))UPErrorMonitor, mctxptr, NULL );
    // }else
    KSPMonitorSet( ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorDefault,   vf, (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy );
  }




  // SOLVE SYSTEM -----------------------------------------------------------
  if( myid == 0 && verbose > 0 ){
    std::cout << "SOLVE! ****************************************************\n";
  }
  solver->Mult(rhs, sol);


  if( myid == 0 && verbose > 0 ){
    std::cout << "Post-processing *******************************************\n";
  }

  // OUTPUT -----------------------------------------------------------------
  if (myid == 0){
    if (solver->GetConverged()){
      std::cout << "Solver converged in "         << solver->GetNumIterations();
    }else{
      std::cout << "Solver did not converge in "  << solver->GetNumIterations();
    }
    std::cout << " iterations. Residual norm is " << solver->GetFinalNorm() << ".\n";
  
    if( output>0 ){
      double hmin, hmax, kmin, kmax;
      stokesAssembler->GetMeshSize( hmin, hmax, kmin, kmax );

      ofstream myfile;
      string fname = string("./results/convergence_results") + "_Prec" + to_string(precType) + "_STsolve" + to_string(STSolveType)
                          + "_oU"  + to_string(ordU) + "_oP" + to_string(ordP) + "_Pb" + to_string(pbType);
      if( pbType == 4 ){
        fname += "_Pe" + to_string(Pe);
      }
      fname += string("_") + petscrc_file + ".txt";
      myfile.open( fname, std::ios::app );
      myfile << Tend << ",\t" << dt   << ",\t" << numProcs   << ",\t"
             << hmax << ",\t" << hmin << ",\t" << ref_levels << ",\t"
             << solver->GetNumIterations() << std::endl;
      myfile.close();
    }
  }





  if( output>2 ){
    int colsV[2] = { myid*(FFF->NumRows()), (myid+1)*(FFF->NumRows()) };
    int colsP[2] = { myid*(BBB->NumRows()), (myid+1)*(BBB->NumRows()) };

    HypreParVector uh( MPI_COMM_WORLD, numProcs*(FFF->NumRows()), sol.GetBlock(0).GetData(), colsV ); 
    HypreParVector ph( MPI_COMM_WORLD, numProcs*(BBB->NumRows()), sol.GetBlock(1).GetData(), colsP ); 

    string outFilePath = "ParaView";
    string outFileName = "STstokes_" + pbName;
    stokesAssembler->SaveSolution( uh, ph, outFilePath, outFileName );
    stokesAssembler->SaveExactSolution(    outFilePath, outFileName+"_Ex" );
  }
  

  if( myid == 0 && verbose > 0 ){
    std::cout << "Clean-up **************************************************\n";
  }


  delete FFF;
  delete BBB;
  delete BBt;
  delete frhs;
  delete grhs;
  delete U0;
  delete P0;
  delete uEx;
  delete pEx;

  delete solver;
  delete stokesOp;
  delete stokesPr;
  delete stokesAssembler;

   

  MFEMFinalizePetsc();
  // HYPRE_Finalize();  //?
  MPI_Finalize();

  return 0;
}




// Function to destroy context of function to monitor actual error
PetscErrorCode UPErrorMonitorDestroy( void ** mctx ){
  PetscErrorCode ierr;
  
  UPErrorMonitorCtx *ctx = (UPErrorMonitorCtx*)mctx;
  
  ierr = PetscViewerAndFormatDestroy( &(ctx->vf)); CHKERRQ(ierr);
  delete ctx;

  return 0;
}

// Function to monitor actual error
PetscErrorCode UPErrorMonitor( KSP ksp, PetscInt n, PetscReal rnorm, void *mctx ){
  UPErrorMonitorCtx *ctx = (UPErrorMonitorCtx*)mctx;
  Vec x;
  double errU, errP, glbErrU, glbErrP;
  PetscInt lclSize;

  // recover current solution
  KSPBuildSolution( ksp, NULL, &x );
  VecGetLocalSize( x, &lclSize );

  if( lclSize != (ctx->lenghtU + ctx->lenghtP) ){
    std::cerr<<"ERROR! ErrorMonitor: Something went very wrong"<<std::endl
             <<"       seems like the solution is stored in a weird way in the PETSC solver,"<<std::endl
             <<"       and sizes mismatch: "<< lclSize << "=/="<<ctx->lenghtU<<"+"<<ctx->lenghtP<<std::endl;
  }
  
  // get local raw pointer
  double* vals;
  VecGetArray( x, &vals );

  // store in MFEM-friendly variables
  Vector uh( vals,                ctx->lenghtU );  // hopefully that's the way it's stored in x, with u and p contiguos
  Vector ph( vals + ctx->lenghtU, ctx->lenghtP );
  

  // compute error per each time step
  ctx->STassembler->ComputeL2Error( uh, ph, errU, errP );

  // compute Linf norm in time or errors
  MPI_Reduce( &errU, &glbErrU, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  MPI_Reduce( &errP, &glbErrP, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  
  // print
  PetscViewer viewer =  ctx->vf->viewer;
  PetscViewerPushFormat( viewer, ctx->vf->format);
  // PetscViewerASCIIAddTab( viewer,((PetscObject)ksp)->tablevel );
  if (n == 0 ) {// && ((PetscObject)ksp)->prefix) {
    // PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);
    PetscViewerASCIIPrintf(viewer,"  Residual norms for outer solver.\n");
  }
  PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm_erru_errp %14.12e\t%14.12e\t%14.12e \n",n,(double)rnorm,(double)errU,(double)errP);
  // PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);
  PetscViewerPopFormat(viewer);

  // if( n%10 == 0 ){
  //   int myid;
  //   MPI_Comm_rank(ctx->comm, &myid);

  //   if ( myid == 0 ){
  //     std::cout<<"Saving plot of error in Paraview-friendly format"<<std::endl;
  //   }
  //   ctx->STassembler->SaveError( uh, ph, ctx->path + "it" + to_string(n) + "/", "error");
  // }

  // clean-up
  // vector x should *not* be destroyed by user)
  // delete vals;

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






//***************************************************************************
//TEST CASES OF SOME ACTUAL RELEVANCE
//***************************************************************************
void wFun_zero(const Vector & x, const double t, Vector & w){
  w(0) = 0.;
  w(1) = 0.;
}
double wnFun_zero(const Vector & x, const double t ){
  return 0.;
}


// Simple example -----------------------------------------------------------
// Exact solution (velocity)
void uFun_ex_an(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = (t+1.) * sin(M_PI*xx) * sin(M_PI*yy);
  u(1) = (t+1.) * cos(M_PI*xx) * cos(M_PI*yy);
}

// Exact solution (pressure)
double pFun_ex_an(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return (t+1.) * sin(M_PI*xx) * cos(M_PI*yy);
}

// Rhs (velocity)
void fFun_an(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  f(0) = (t+1.) * ( 2.0*M_PI*M_PI*sin(M_PI*xx)*sin(M_PI*yy) + M_PI*cos(M_PI*xx)*cos(M_PI*yy) ) + sin(M_PI*xx)*sin(M_PI*yy);
  f(1) = (t+1.) * ( 2.0*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) - M_PI*sin(M_PI*xx)*sin(M_PI*yy) ) + cos(M_PI*xx)*cos(M_PI*yy);
}

// Normal derivative of velocity * mu
void nFun_an(const Vector & x, const double t, Vector & n){
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
double gFun_an(const Vector & x, const double t ){
  return 0.0;
}





// Driven cavity flow (speed ramping up)-------------------------------------
// Exact solution (velocity) - only for BC
void uFun_ex_cavity(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = 0.0;
  u(1) = 0.0;

  if( yy==1.0 ){
    u(0) = t * 8*xx*(1-xx)*(2.*xx*xx-2.*xx+1.);   // regularised (1-x^4 mapped from -1,1 to 0,1)
    // u(0) = t;                      // leaky
    // if( xx > 1.0 && xx < 1.0 )
    //   u(0) = t;                    // watertight
  }
}

// Exact solution (pressure) - unused
double pFun_ex_cavity(const Vector & x, const double t ){
  return 0.0;
}

// Rhs (velocity) - no forcing
void fFun_cavity(const Vector & x, const double t, Vector & f){
  f(0) = 0.0;
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - unused
void nFun_cavity(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun_cavity(const Vector & x, const double t ){
  return 0.0;
}



// Driven cavity flow (constant speed) --------------------------------------
// Exact solution (velocity) - only for BC
void uFun_ex_cavityC(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = 0.0;
  u(1) = 0.0;

  if( yy==1.0 ){
    u(0) = 8*xx*(1-xx)*(2.*xx*xx-2.*xx+1.);   // regularised (1-x^4 mapped from -1,1 to 0,1)
    // u(0) = 1.0;                      // leaky
    // if( xx > 0. && xx < 1.0 )
    //   u(0) = 1.0;                    // watertight
  }
}

// Exact solution (pressure) - unused
double pFun_ex_cavityC(const Vector & x, const double t ){
  return 0.0;
}

// Rhs (velocity) - no forcing
void fFun_cavityC(const Vector & x, const double t, Vector & f){
  f(0) = 0.0;
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - unused
void nFun_cavityC(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun_cavityC(const Vector & x, const double t ){
  return 0.0;
}






// Poiseuille flow (speed ramping up)----------------------------------------
// Exact solution (velocity)
void uFun_ex_poiseuille(const Vector & x, const double t, Vector & u){
  double yy(x(1));
  u(0) = t * 4.*yy*(1.-yy);
  u(1) = 0.0;
}

// Exact solution (pressure)
double pFun_ex_poiseuille(const Vector & x, const double t ){
  double xx(x(0));
  return -8.*t*(xx-8.);   // pressure is zero at outflow (xx=8)- this way we can use the same function for both the long [0,8]x[0,1] and short [7,8]x[0,1] domains
}

// Rhs (velocity) - counterbalance dt term
void fFun_poiseuille(const Vector & x, const double t, Vector & f){
  double yy(x(1));
  f(0) = 4.*yy*(1.-yy);
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - used only at outflow
void nFun_poiseuille(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun_poiseuille(const Vector & x, const double t ){
  return 0.0;
}





// Poiseuille flow (constant speed) ----------------------------------------
// Exact solution (velocity)
void uFun_ex_poiseuilleC(const Vector & x, const double t, Vector & u){
  double yy(x(1));
  u(0) = 4.*yy*(1.-yy);
  u(1) = 0.0;
}

// Exact solution (pressure)
double pFun_ex_poiseuilleC(const Vector & x, const double t ){
  double xx(x(0));
  return -8.*(xx-8.);   // pressure is zero at outflow (xx=8)
}

// Rhs (velocity) - no source
void fFun_poiseuilleC(const Vector & x, const double t, Vector & f){
  f(0) = 0.0;
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - used only at outflow
void nFun_poiseuilleC(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun_poiseuilleC(const Vector & x, const double t ){
  return 0.0;
}




// Poiseuille flow (opposite velocity flow) ---------------------------------
// Exact solution (velocity)
void uFun_ex_poiseuilleM(const Vector & x, const double t, Vector & u){
  double yy(x(1));
  u(0) = - t * 4.*yy*(1.-yy);
  u(1) = 0.0;
}

// Exact solution (pressure)
double pFun_ex_poiseuilleM(const Vector & x, const double t ){
  double xx(x(0));
  return 8.*t*(xx-7.);   // pressure is zero at outflow (xx=7)
}

// Rhs (velocity) - counterbalance dt term
void fFun_poiseuilleM(const Vector & x, const double t, Vector & f){
  double yy(x(1));
  f(0) = - 4.*yy*(1.-yy);
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - used only at outflow
void nFun_poiseuilleM(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun_poiseuilleM(const Vector & x, const double t ){
  return 0.0;
}







// Flow over step (speed ramping up)----------------------------------------
// Exact solution (velocity) - only for BC
void uFun_ex_step(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = 0.0;
  u(1) = 0.0;

  if( xx==0.0 ){
    u(0) = t * 4.*yy*(1.-yy);
  }
}

// Exact solution (pressure) - unused
double pFun_ex_step(const Vector & x, const double t ){
  return 0.0;
}

// Rhs (velocity) - no forcing
void fFun_step(const Vector & x, const double t, Vector & f){
  f(0) = 0.0;
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - used only at outflow
// - setting both this and the pressure to zero imposes zero average p at outflow
void nFun_step(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// Rhs (pressure) - unused
double gFun_step(const Vector & x, const double t ){
  return 0.0;
}






// Double-glazing problem (speed ramping up)--------------------------------
// Exact solution (velocity) - only for BC
void uFun_ex_glazing(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) = 0.0;
  u(1) = 0.0;

  // just like cavity flow
  if( yy==1.0 ){
    u(0) = t * 8*xx*(1-xx)*(2.*xx*xx-2.*xx+1.);   // regularised (1-x^4 mapped from -1,1 to 0,1)
    // u(0) = 1.0;                      // leaky
    // if( xx > 0. && xx < 1.0 )
    //   u(0) = 1.0;                    // watertight
  }

  // if( xx==1.0 ){
  //   u(1) = -t * (1.-yy*yy*yy*yy);   // regularised
  //   // u(1) = -t;                      // leaky
  //   // if( yy > -1. && yy < 1.0 )
  //   //   u(1) = -t;                    // watertight
  // }
}

// Exact solution (pressure) - unused
double pFun_ex_glazing(const Vector & x, const double t ){
  return 0.0;
}

// Rhs (velocity) - no forcing
void fFun_glazing(const Vector & x, const double t, Vector & f){
  f(0) = 0.0;
  f(1) = 0.0;
}

// Normal derivative of velocity * mu - unused
void nFun_glazing(const Vector & x, const double t, Vector & n){
  n(0) = 0.0;
  n(1) = 0.0;
}

// velocity field
void wFun_glazing(const Vector & x, const double t, Vector & w){
  double xx(x(0));
  double yy(x(1));
  w(0) = - t * 2.*(2*yy-1)*(4*xx*xx-4*xx+1); // (-t*2.*yy*(1-xx*xx) mapped from -1,1 to 0,1)
  w(1) =   t * 2.*(2*xx-1)*(4*yy*yy-4*yy+1); // ( t*2.*xx*(1-yy*yy) mapped from -1,1 to 0,1)
}

// Rhs (pressure) - unused
double gFun_glazing(const Vector & x, const double t ){
  return 0.0;
}




