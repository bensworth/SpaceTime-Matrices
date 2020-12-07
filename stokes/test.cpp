//---------------------------------------------------------------------------
// Implementation of time-dependent Oseen equation
//  dt(u) -mu ∇•∇u + mu Pe w•∇u + ∇p  = f
//                              - ∇•u = g
// with space-time block preconditioning.
// 
// - Solver support from PETSc
//    Some PETSc solvers are used within this program: the tags to use to
//    prescribe the corresponding PETSc options at runtime are, resp.:
//    - solver_  : outer solver (usually GMRES or FGMRES)
//    - Vsolver_ : solver for velocity spatial operator, used when
//                 time-stepping is prescribed (usually LU/AMG)
//    - VCoarseSolver_ : coarse solver for velocity spatial operator, used
//                       when Parareal is prescribed (usually a simpler AMG)
//    - PSolverLaplacian_ : solver for pressure 'laplacian' (usually LU/AMG)
//                          Careful to signal non-trivial kernel if needed!
//    - PSolverMass_ : solver for pressure mass matrix (usually LU, or
//                     chebyshev iterations)
//  
// - For information on the components of the block preconditioner, see:
//    H. Elman, D. Silvester, and A. Wathen. Finite elements and fast
//    iterative solvers: with applications in incompressible fluid dynamics.
//
// Author: Federico Danieli, Numerical Analysis Group
// University of Oxford, Dept. of Mathematics
// email address: federico.danieli@maths.ox.ac.uk  
// April 2020; Last revision: Oct-2020
//
// Acknowledgement to: S. Rhebergen (University of Waterloo)
// Code based on MFEM's example ex5p.cpp.
//
//
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "stokesstoperatorassembler.hpp"
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
// PB parameters pre-definition:
void   wFun_zero(  const Vector & x, const double t, Vector & w );
// - simple cooked-up pb with analytical solution
void   uFun_ex_an( const Vector & x, const double t, Vector & u );
double pFun_ex_an( const Vector & x, const double t             );
void   fFun_an(    const Vector & x, const double t, Vector & f );
void   nFun_an(    const Vector & x, const double t, Vector & f );
double gFun_an(    const Vector & x, const double t             );
// - driven cavity flow
void   uFun_ex_cavity( const Vector & x, const double t, Vector & u );
double pFun_ex_cavity( const Vector & x, const double t             );
void   fFun_cavity(    const Vector & x, const double t, Vector & f );
void   nFun_cavity(    const Vector & x, const double t, Vector & f );
double gFun_cavity(    const Vector & x, const double t             );
void   uFun_ex_cavityC(const Vector & x, const double t, Vector & u );  // constant-in-time counterpart
double pFun_ex_cavityC(const Vector & x, const double t             );
void   fFun_cavityC(   const Vector & x, const double t, Vector & f );
void   nFun_cavityC(   const Vector & x, const double t, Vector & f );
double gFun_cavityC(   const Vector & x, const double t             );
// - poiseuille flow
void   uFun_ex_poiseuille( const Vector & x, const double t, Vector & u );
double pFun_ex_poiseuille( const Vector & x, const double t             );
void   fFun_poiseuille(    const Vector & x, const double t, Vector & f );
void   nFun_poiseuille(    const Vector & x, const double t, Vector & f );
double gFun_poiseuille(    const Vector & x, const double t             );
void   uFun_ex_poiseuilleC(const Vector & x, const double t, Vector & u );  // constant-in-time counterpart
double pFun_ex_poiseuilleC(const Vector & x, const double t             );
void   fFun_poiseuilleC(   const Vector & x, const double t, Vector & f );
void   nFun_poiseuilleC(   const Vector & x, const double t, Vector & f );
double gFun_poiseuilleC(   const Vector & x, const double t             );
void   uFun_ex_poiseuilleM(const Vector & x, const double t, Vector & u );  // mirrored wrt y axis counterpart
double pFun_ex_poiseuilleM(const Vector & x, const double t             );  //  this is just for debugging: errors seemed to accumulate
void   fFun_poiseuilleM(   const Vector & x, const double t, Vector & f );  //  at inflow corners, so I'm checking if it's an implementation
void   nFun_poiseuilleM(   const Vector & x, const double t, Vector & f );  //  issue, or if it's just the way it is - seem like implementation is ok
double gFun_poiseuilleM(   const Vector & x, const double t             );
// - flow over step
void   uFun_ex_step( const Vector & x, const double t, Vector & u );
double pFun_ex_step( const Vector & x, const double t             );
void   fFun_step(    const Vector & x, const double t, Vector & f );
void   nFun_step(    const Vector & x, const double t, Vector & f );
double gFun_step(    const Vector & x, const double t             );
// - double-glazing problem
void   uFun_ex_glazing( const Vector & x, const double t, Vector & u );
double pFun_ex_glazing( const Vector & x, const double t             );
void   fFun_glazing(    const Vector & x, const double t, Vector & f );
void   nFun_glazing(    const Vector & x, const double t, Vector & f );
void   wFun_glazing(    const Vector & x, const double t, Vector & f );
double gFun_glazing(    const Vector & x, const double t             );
//---------------------------------------------------------------------------
// Handy function for monitoring quantities of interest - predefinition
struct UPErrorMonitorCtx{// Context of function to monitor actual error
  int lenghtU;
  int lenghtP;
  StokesSTOperatorAssembler* STassembler;
  MPI_Comm comm;
  PetscViewerAndFormat *vf;
  std::string path;
};

PetscErrorCode UPErrorMonitorDestroy( void ** mctx );
PetscErrorCode UPErrorMonitor( KSP ksp, PetscInt n, PetscReal rnorm, void *mctx );
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{

  // for now, assume no spatial parallelisation: each processor handles a time-step
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  double Tend = 1.;

  // Initialise problem parameters
  int ordU = 2;
  int ordP = 1;
  int ref_levels = 4;
  const char *petscrc_file = "rc_SpaceTimeStokes";
  int verbose = 0;
  int precType = 1;
  int STSolveType = 0;
  int pbType = 1;   
  string pbName = "";
  int output = 2;
  double Pe = 0.;

  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&ordU, "-oU", "--orderU",
                "Finite element order (polynomial degree) for velocity field (default: 2)");
  args.AddOption(&ordP, "-oP", "--orderP",
                "Finite element order (polynomial degree) for pressure field (default: 1)");
  args.AddOption(&ref_levels, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&Tend, "-T", "--Tend",
                "Final time (default: 1.0)");
  args.AddOption(&pbType, "-Pb", "--problem",
                "Problem: 0-analytical test, 1-driven cavity flow (default), 2-poiseuille, 3-flow over step, 4-double-glazing, 5-NS driven cavity flow.");
  args.AddOption(&Pe, "-Pe", "--peclet",
                "Peclet number (only valid for Pb 4-double glazing) (default: 1.0)");
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
  const double mu = 1.;                // CAREFUL NOT TO CHANGE THIS! Or if you do, re-define the normal derivative, too.

  void(  *fFun)(   const Vector &, double, Vector& );
  double(*gFun)(   const Vector &, double );
  void(  *nFun)(   const Vector &, double, Vector& );
  void(  *wFun)(   const Vector &, double, Vector& );
  void(  *uFun_ex)(const Vector &, double, Vector& );
  double(*pFun_ex)(const Vector &, double );
  std::string mesh_file;

  switch (pbType){
    // analytical test-case
    case 0:{
      mesh_file = "./meshes/tri-square-open.mesh";
      fFun    = &fFun_an;
      gFun    = &gFun_an;
      nFun    = &nFun_an;
      wFun    = &wFun_zero;
      uFun_ex = &uFun_ex_an;
      pFun_ex = &pFun_ex_an;
      pbName = "Analytic";
      Pe = 0.0;
      break;
    }
    // driven cavity flow
    case 1:{
      mesh_file = "./meshes/tri-square-cavity.mesh";
      fFun    = &fFun_cavity;
      gFun    = &gFun_cavity;
      nFun    = &nFun_cavity;
      wFun    = &wFun_zero;
      uFun_ex = &uFun_ex_cavity;
      pFun_ex = &pFun_ex_cavity;
      pbName = "Cavity";
      Pe = 0.0;
      break;
    }
    case 2:{
      // mesh_file = "./meshes/tri-rectangle-poiseuille.mesh";
      mesh_file = "./meshes/tri-square-poiseuille.mesh";
      fFun    = &fFun_poiseuille;
      gFun    = &gFun_poiseuille;
      nFun    = &nFun_poiseuille;
      wFun    = &wFun_zero;
      uFun_ex = &uFun_ex_poiseuille;
      pFun_ex = &pFun_ex_poiseuille;
      pbName = "Poiseuille";
      Pe = 0.0;
      break;
    }
    case 3:{
      mesh_file = "./meshes/tri-step.mesh";
      fFun    = &fFun_step;
      gFun    = &gFun_step;
      nFun    = &nFun_step;
      wFun    = &wFun_zero;
      uFun_ex = &uFun_ex_step;
      pFun_ex = &pFun_ex_step;
      pbName = "Step";
      Pe = 0.0;
      break;
    }
    case 4:{
      mesh_file = "./meshes/tri-square-glazing.mesh";
      fFun    = &fFun_glazing;
      gFun    = &gFun_glazing;
      nFun    = &nFun_glazing;
      wFun    = &wFun_glazing;
      uFun_ex = &uFun_ex_glazing;
      pFun_ex = &pFun_ex_glazing;
      pbName = "Glazing";
      if (Pe <= 0.){
        Pe = 1.0;
        if ( myid == 0 ){
          std::cerr<<"Warning: Double-glazing selected but Pe=0? Changed to Pe=1"<<std::endl;
        }
      }
      if ( myid == 0 ){
        std::cerr<<"Warning: Double-glazing flow only works if w*n = 0 on bdr for now."<<std::endl
                 <<"         Also, some sort of stabilisation must be included for high Peclet."<<std::endl;
      }
      break;
    }
    // driven cavity flow - Navier Stokes
    case 5:{
      mesh_file = "./meshes/tri-square-cavity.mesh";
      fFun    = &fFun_cavity;
      gFun    = &gFun_cavity;
      nFun    = &nFun_cavity;
      wFun    = &wFun_zero;
      uFun_ex = &uFun_ex_cavity;
      pFun_ex = &pFun_ex_cavity;
      pbName = "CavityNS";
      // ugly hack: the convection term is multiplied by mu*Pe. For NS, this should be 1, regardless, so I'm tweaking Pe here.
      Pe = 1.0/mu;
      if ( myid == 0 ){
        std::cerr<<"Warning: Picard iteration for Navier Stokes not (yet) implemented. Abort."<<std::endl;
      }
      return 1;
      break;
    }
    case 20:{
      // mesh_file = "./meshes/tri-rectangle-poiseuille.mesh";
      mesh_file = "./meshes/tri-square-poiseuille-mirror.mesh";
      fFun    = &fFun_poiseuilleM;
      gFun    = &gFun_poiseuilleM;
      nFun    = &nFun_poiseuilleM;
      wFun    = &wFun_zero;
      uFun_ex = &uFun_ex_poiseuilleM;
      pFun_ex = &pFun_ex_poiseuilleM;
      pbName = "PoiseuilleM";
      Pe = 0.0;
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
  StokesSTOperatorAssembler *stokesAssembler = new StokesSTOperatorAssembler( MPI_COMM_WORLD, mesh_file, ref_levels, ordU, ordP,
                                                                              dt, mu, Pe, fFun, gFun, nFun, wFun, uFun_ex, pFun_ex, verbose );

  HypreParMatrix *FFF, *BBB, *BBt;
  Operator *FFi, *XXi;
  HypreParVector  *frhs, *grhs, *uEx, *pEx, *U0, *P0;
  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling system and rhs *********************************\n";
  }
  stokesAssembler->AssembleSystem( FFF, BBB, frhs, grhs, U0, P0 );
  BBt = BBB->Transpose( );

  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = FFF->NumRows();
  offsets[2] = BBB->NumRows();
  offsets.PartialSum();

  BlockVector sol(offsets), rhs(offsets);
  rhs.GetBlock(0) = *frhs;
  rhs.GetBlock(1) = *grhs;
  sol.GetBlock(0) = *U0;
  sol.GetBlock(1) = *P0;

  BlockOperator *stokesOp = new BlockOperator( offsets );
  stokesOp->SetBlock(0, 0, FFF);
  stokesOp->SetBlock(0, 1, BBt);
  stokesOp->SetBlock(1, 0, BBB);


  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators for preconditioner *******************\n";
  }
  stokesAssembler->AssemblePreconditioner( FFi, XXi, STSolveType );


  stokesAssembler->ExactSolution( uEx, pEx );


  if( myid == 0 && verbose > 0 ){
    std::cout << "Set-up solver *********************************************\n";
  }

  // Define solver
  PetscLinearSolver *solver = new PetscLinearSolver(MPI_COMM_WORLD, "solver_");
  // TODO: iterative mode triggers use of initial guess: check if it is indeed iterative
  bool isIterative = true;
  solver->iterative_mode = isIterative;

  // Define preconditioner
  Solver *stokesPr;

  // std::string filename = "operator";
  // stokesAssembler->PrintMatrices(filename);

  // - Flag that most extreme eigs of precon system must be computed, if only one processor is given
  if ( myid==0 && numProcs == 1 ){

    // store also the actual matrices, if you want to compute the eigs externally
    if( verbose > 3 ){
      string path = string("./results/Operators/Pb")  + to_string(pbType) + "_Prec" + to_string(precType) + "_STsolve" + to_string(STSolveType)
                         + "_oU"   + to_string(ordU)   + "_oP"             + to_string(ordP)  + "/";
      if (!std::experimental::filesystem::exists( path )){
        std::experimental::filesystem::create_directories( path );
      }
      path += "dt"+ to_string(dt) + "_r"+ to_string(ref_levels);
      stokesAssembler->PrintMatrices(path);
    }

    KSP ksp = *solver;
    KSPSetComputeEigenvalues( ksp, PETSC_TRUE );
  }



  switch(precType){
    case 0:{
      BlockDiagonalPreconditioner *temp = new BlockDiagonalPreconditioner(offsets);
      temp->iterative_mode = false;
      temp->SetDiagonalBlock( 0, FFi );
      temp->SetDiagonalBlock( 1, XXi );
      stokesPr = temp;
      break;
    }
    case 1:{
      // BlockLowerTriangularPreconditioner *temp = new BlockLowerTriangularPreconditioner(offsets);
      // temp->iterative_mode = false;
      // temp->SetDiagonalBlock( 0, FFi );
      // temp->SetDiagonalBlock( 1, XXi );
      // temp->SetBlock( 1, 0, BBB );
      BlockUpperTriangularPreconditioner *temp = new BlockUpperTriangularPreconditioner(offsets);
      temp->iterative_mode = false;
      temp->SetDiagonalBlock( 0, FFi );
      temp->SetDiagonalBlock( 1, XXi );
      temp->SetBlock( 0, 1, BBt );
      stokesPr = temp;
      break;
    }
    default:{
      if ( myid == 0 ){
        std::cerr<<"ERROR: Option for preconditioner "<<precType<<" not recognised"<<std::endl;
      }
      break;
    }
  }


  solver->SetPreconditioner(*stokesPr);



  solver->SetOperator(*stokesOp);


  // Save residual evolution to file
  if ( output>1 ){
    // - create folder which will store all files with various convergence evolution 
    string path = string("./results/convergence_results") + "_Prec" + to_string(precType) + "_STsolve" + to_string(STSolveType)
                           + "_oU" + to_string(ordU) + "_oP" + to_string(ordP) + "_Pb" + to_string(pbType);
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

  // - Main solve routine
  solver->Mult(rhs, sol);

  // - Eventually compute eigs
  if ( myid==0 && numProcs == 1 ){

    KSP ksp = *solver;
    PetscInt Neigs = 100;
    double *realPart = new double(Neigs);
    double *imagPart = new double(Neigs);
// #ifdef USE_SLEPC
//     EPS eps;
//     EPSType type; 
//     PetscInt Nconveigs;
//     PetscErrorCode ierr;
//     Vec dummyr,dummyi;

//     ierr = MatCreateVecs( (KSP)(*solver), NULL, &dummyr );CHKERRQ(ierr);
//     ierr = MatCreateVecs( (KSP)(*solver), NULL, &dummyi );CHKERRQ(ierr);
    
//     ierr = EPSCreate( PETSC_COMM_WORLD, &eps );CHKERRQ(ierr);
//     ierr = EPSSetOperators( eps, (KSP)(*solver), NULL );CHKERRQ(ierr);

//     if( pbType == 4 ){
//       ierr = EPSSetProblemType( eps, EPS_NHEP );CHKERRQ(ierr); // matrix is non-symmetric
//     }else{
//       ierr = EPSSetProblemType( eps, EPS_HEP );CHKERRQ(ierr);  // matrix is symmetric
//     }
//     ierr = EPSSolve( eps );CHKERRQ(ierr);
//     ierr = EPSGetConverged( eps, &Nconveigs );CHKERRQ(ierr);

//     Neigs = min(Neigs,Nconveigs);

//     for ( int i = 0; i < floor(Neigs/2); ++i ){
//       ierr = EPSGetEigenpair( eps,             i, &(realPart[i]),                &(imagPart[i]),                dummyr, dummyi );CHKERRQ(ierr); //largest half
//       ierr = EPSGetEigenpair( eps, Nconveigs-1-i, &(realPart[floor(Neigs/2)+i]), &(imagPart[floor(Neigs/2)+i]), dummyr, dummyi );CHKERRQ(ierr); //smallest half
//     }
    
//     ierr = EPSDestroy(&eps);CHKERRQ(ierr);
//     ierr = VecDestroy(&dummyr);CHKERRQ(ierr);
//     ierr = VecDestroy(&dummyi);CHKERRQ(ierr);
// #else

    KSPComputeEigenvalues( ksp, Neigs, realPart, imagPart, &Neigs );

    // and store them
    ofstream myfile;
    string fname = string("./results/Operators/eigs_") + "dt"+ to_string(dt) + "_r"+ to_string(ref_levels)
                        + "_Prec" + to_string(precType) + "_STsolve" + to_string(STSolveType)
                        + "_oU"   + to_string(ordU)     + "_oP"      + to_string(ordP)       + "_Pb" + to_string(pbType);
    if( pbType == 4 ){
      fname += "_Pe" + to_string(Pe);
    }
    fname += string("_") + petscrc_file + ".txt";
    myfile.open( fname );
    for ( int i = 0; i < Neigs; ++i ){
      myfile << realPart[i] << ",\t" << imagPart[i] << std::endl;
    }
    myfile.close();

    delete realPart;
    delete imagPart;

  }




  // OUTPUT -----------------------------------------------------------------
  if( myid == 0 && verbose > 0 ){
    std::cout << "Post-processing *******************************************\n";
  }

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




