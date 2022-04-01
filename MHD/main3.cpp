//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "testcases.hpp"
#include "imhd2dstoperatorassembler.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include "operatorssequence.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <limits>
//---------------------------------------------------------------------------
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;
//---------------------------------------------------------------------------
// Handy functions for monitoring quantities of interest - predefinition
struct UPErrorMonitorCtx{// Context of function to monitor actual error
  int lenghtU;
  int lenghtP;
  int lenghtZ;
  int lenghtA;
  IMHD2DSTOperatorAssembler* STassembler;
  MPI_Comm comm;
  PetscViewerAndFormat *vf;
  std::string path;
};
PetscErrorCode UPErrorMonitorDestroy( void ** mctx );
PetscErrorCode UPErrorMonitor( KSP ksp, PetscInt n, PetscReal rnorm, void *mctx );

struct UPSplitResidualMonitorCtx{// Context of function to monitor actual error
  int lenghtU;
  int lenghtP;
  int lenghtZ;
  int lenghtA;
  MPI_Comm comm;
  PetscViewerAndFormat *vf;
  std::string path;
};
PetscErrorCode UPSplitResidualMonitorDestroy( void ** mctx );
PetscErrorCode UPSplitResidualMonitor( KSP ksp, PetscInt n, PetscReal rnorm, void *mctx );
//---------------------------------------------------------------------------
// Handy functions for assembling single factors of the preconditioner
void AssembleLub( const Operator* Y,   const Operator* Fui,
                  const Operator* Z1,  const Operator* Mzi, BlockLowerTriangularPreconditioner* Lub );
void AssembleUub( const Operator* Z1,  const Operator* Z2, const Operator* Mzi, const Operator* K,
                  const Operator* aSi, BlockUpperTriangularPreconditioner* Uub );
void AssembleLup( const Operator* Fui, const Operator* B,   BlockLowerTriangularPreconditioner* Lup );
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* X1, const Operator* X2,
                  const Operator* pSi, BlockUpperTriangularPreconditioner* Uup );
//---------------------------------------------------------------------------
// Misc
void computeBlockedNorm( const BlockVector& u, Vector& norm );

//---------------------------------------------------------------------------
int main(int argc, char *argv[]){

  //*************************************************************************
  // Initialise
  //*************************************************************************

  // for now, assume no spatial parallelisation: each processor handles a time-step
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  // - discretisation
  int ordU = 3;
  int ordP = 2;
  int ordZ = 1;
  int ordA = 1;
  int ref_levels = 4;
  double Tend = 1.;
  double dt;    // will be initialised after Tend is parsed
  // - solver
  bool stab = false;
  const char *petscrc_file = "rc_SpaceTimeIMHD2D";
  int precType     = 2;
  int STSolveTypeU = 0;
  int STSolveTypeA = 3;
  const int   maxNewtonIt  = 10;
  const double  newtonRTol = 0;     //1e-5;
  const double  newtonATol = 1e-10; //0.;  
  // - misc
  int pbType  = 6;
  int output  = 2;
  int verbose = 0;


  // Parse parameters *******************************************************
  OptionsParser args(argc, argv);
  args.AddOption(&ordU, "-oU", "--orderU",
                "Finite element order (polynomial degree) for velocity field (default: 3)");
  args.AddOption(&ordP, "-oP", "--orderP",
                "Finite element order (polynomial degree) for pressure field (default: 2)");
  args.AddOption(&ordZ, "-oZ", "--orderZ",
                "Finite element order (polynomial degree) for Laplacian of vector potential (default: 1)");
  args.AddOption(&ordA, "-oA", "--orderA",
                "Finite element order (polynomial degree) for vector potential field (default: 1)");
  args.AddOption(&ref_levels, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&Tend, "-T", "--Tend",
                "Final time (default: 1.0)");
  args.AddOption(&pbType, "-Pb", "--problem",
                "Problem: 0 to 4-Analytical test cases\n"
                "              5-Kelvin-Helmholtz instability (UNTESTED)\n"
                "              6-Island coalescence (xy sym) (default)\n"
                "              7-Tearing mode (xy sym)\n"
                "              9-Tearing mode (full domain)\n"
                "             11-Driven cavity flow (NS only)\n"
        );
  args.AddOption(&precType, "-P", "--preconditioner",
                "Type of preconditioner: 0-Space-time Cyr et al: Uupi*Lupi*Uubi*Lubi\n"
                "                        1-Space-time Cyr et al simplified: Uupi*Lupi*Uubi\n"
                "                        2-Space-time Cyr et al uber simplified: Uupi*Uubi (default)\n"
        );
  args.AddOption(&stab, "-S", "--stab", "-noS", "--noStab",
                "Stabilise via SUPG (default: false) - UNTESTED!\n"
        );
  args.AddOption(&STSolveTypeU, "-STU", "--spacetimesolveU",
                "Type of solver for velocity space-time matrix: 0-time stepping (default)\n"
                // "                                               1-boomerAMG (AIR)\n"
                "                                               5-GMRES+boomerAMG (AIR)\n"
                // "                                               3-Parareal (not fully tested)\n"
                "                                               9-Sequential time-stepping for whole ST system - ignores many other options\n"
        );
  args.AddOption(&STSolveTypeA, "-STA", "--spacetimesolveA",
                "Type of solver for potential wave space-time matrix: 0 -time stepping - implicit leapfrog\n"
                " (Ca, appearing inside the magnetic Schur comp)      1 -time stepping - explicit leapfrog\n"
                "                                                     2 -time stepping on Fa*Mai*Fa+|B|/mu*Aa matrix (uses dMai on main diag of CCa, Mai on subdiag)\n"
                "                                                     3 -time stepping on   Ma  Fai (Fa*d(Ma)i*Fa+|B|/mu*Aa) matrix (uses dMai on all diags of CCa) (default)\n"
                "                                                     4 -time stepping on d(Ma) Fai (Fa*d(Ma)i*Fa+|B|/mu*Aa) matrix (uses dMai everywhere)\n"
                "                                                     5 -GMRES+boomerAMG (AIR) - Schur as in 4\n"
                "                                                     6 -GMRES+boomerAMG (AIR) - Schur as in 8\n"
                "                                                     8 -time-stepping on Fa only (simplify the whole Schur complement Sa->Fa)\n"
                "                                                     9 -sequential time-stepping for whole ST system (Schur as in 3)- ignores many other options\n"
                "                                                     10-sequential time-stepping for whole ST system (Schur as in 8)- ignores many other options\n"
        );
  args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                "PetscOptions file to use: rc_SpaceTimeIMHD2D         (direct (LU) solvers - The code handles singular Ap by adding a zero-mean constraint now,"
                "                                                                             so no need to flag it)\n"
                "                          rc_SpaceTimeIMHD2D_approx  (FGMRES + AMG solver for Ap, AIR for FFu and FFa/CCa, Cheb for Mp, Mz and Ma )");
  args.AddOption(&verbose, "-V", "--verbose",
                "Control how much info to print to terminal:(=-1   print large block matrices, and trigger eigs analysis - bit of a hack)\n"
                "                                            >0    basic info\n"
                "                                            >1   +info on large (space-time) block assembly\n"
                "                                            >5   +info on small (single time-step) blocks assembly\n"
                "                                            >10  +more details on single time-step assembly\n"
                "                                            >20  +details on each iteration\n"
                "                                            >50  +prints matrices (careful of memory usage!)\n"
                "                                            >100 +prints partial vector results from each iteration\n"
        );
  args.AddOption(&output, "-out", "--outputsol",
                "Choose how much info to store on disk: 0  nothing\n"
                "                                       1 +#it to convergence\n"
                "                                       2 +residual evolution and timing info (default)\n"
                "                                       3 +paraview plot of exact (if available) and approximate solution (careful of memory usage!)\n"
                "                                       4 +operators and intermediate vector results at each Newton iteration (VERY careful of memory usage!)\n"
        );
  args.Parse();
  if(!args.Good()){
    if(myid == 0)
    {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }

  // adjust solver selection for the global time-stepping case
  if ( STSolveTypeA == 9 || STSolveTypeA == 10 ){
    if ( myid == 0 && STSolveTypeU != 9 ){
      std::cout<<"Warning: Global time-stepping selected for A, but not for U: adjusting settings to -STU 9."<<std::endl;
    }
    STSolveTypeU = 9;
  }else if ( STSolveTypeU == 9 ){
    if ( myid == 0 ){
      std::cout<<"Warning: Global time-stepping selected for U, but not for A: adjusting settings to -STA 9."<<std::endl;
    }
    STSolveTypeA = 9;  // defaults to using Cyr's preconditioner rather than Fa
  }


  // - initialise petsc
  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

  // - initialise last parameter
  dt  = Tend / numProcs;

  // - initialise time counter
  StopWatch assemblyStopwatch;
  StopWatch solverStopwatch;
  double totAssemblyTime = 0.;
  double totSolverTime   = 0.;



  // Define problem *********************************************************
  std::string pbName;
  std::string mesh_file;
  // - tags for identifying dirichlet BC for u, p (only used in precon) and A
  Array<int> essTagsU(0);
  Array<int> essTagsV(0);
  Array<int> essTagsP(0);
  Array<int> essTagsA(0);
  // - problem parameters
  double mu  = 0.;
  double eta = 0.;
  double mu0 = 0.;
  // - problem data
  void(  *uFun)( const Vector & x, const double t, Vector & u );
  double(*pFun)( const Vector & x, const double t             );
  double(*zFun)( const Vector & x, const double t             );
  double(*aFun)( const Vector & x, const double t             );
  void(  *fFun)( const Vector & x, const double t, Vector & f );
  double(*gFun)( const Vector & x, const double t             );
  double(*hFun)( const Vector & x, const double t             );
  void(  *nFun)( const Vector & x, const double t, Vector & f );
  double(*mFun)( const Vector & x, const double t             );
  void(  *wFun)( const Vector & x, const double t, Vector & u );
  double(*qFun)( const Vector & x, const double t             );
  double(*yFun)( const Vector & x, const double t             );
  double(*cFun)( const Vector & x, const double t             );
  // - fetch relevant details for selected problem
  MHDTestCaseSelector( pbType, 
                       uFun, pFun, zFun, aFun,
                       fFun, gFun, hFun, nFun, mFun,
                       wFun, qFun, yFun, cFun,
                       mu, eta, mu0,
                       pbName, mesh_file,
                       essTagsU, essTagsV, essTagsP, essTagsA );




  if(myid == 0){
    args.PrintOptions(cout);
    std::cout<<"   --np "<<numProcs<<std::endl;
    std::cout<<"   --mu "<<mu<<std::endl;
    std::cout<<"   --eta "<<eta<<std::endl;
    std::cout<<"   --mu0 "<<mu0<<std::endl;
    std::cout<<"   --Pb "<<pbName<<std::endl;
    std::cout<<"   --mesh "<<mesh_file<<std::endl;
    std::cout<<"   --dt "<<Tend/numProcs<<std::endl;
  }





  //*************************************************************************
  // Assemble operators
  //*************************************************************************
  MPI_Barrier(MPI_COMM_WORLD);
  assemblyStopwatch.Start();
  IMHD2DSTOperatorAssembler *mhdAssembler = new IMHD2DSTOperatorAssembler( MPI_COMM_WORLD, mesh_file, ref_levels, ordU, ordP, ordZ, ordA,
                                                                           dt, mu, eta, mu0, fFun, gFun, hFun, nFun, mFun,
                                                                           wFun, qFun, yFun, cFun,
                                                                           uFun, pFun, zFun, aFun, 
                                                                           essTagsU, essTagsV, essTagsP, essTagsA, stab, verbose );


  ParBlockLowTriOperator *FFFu, *MMMz, *FFFa, *BBB, *BBBt, *CCCs, *ZZZ1, *ZZZ2, *XXX1, *XXX2, *KKK, *YYY;
  Vector   fres,   gres,  zres,  hres,  U0,  P0,  Z0,  A0;
  // Vector   *uEx, *pEx, *zEx, *aEx;

  // For the system *********************************************************
  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators for system ***************************\n";
  }

  mhdAssembler->AssembleSystem( FFFu, MMMz, FFFa, BBB,  BBBt, CCCs,
                                ZZZ1, ZZZ2, XXX1, XXX2, KKK,  YYY,
                                fres, gres, zres, hres,
                                U0,   P0,   Z0,   A0  );
  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = FFFu->NumRows();
  offsets[2] =  BBB->NumRows();
  offsets[3] = MMMz->NumRows();
  offsets[4] = FFFa->NumRows();
  offsets.PartialSum();
  // - assemble original matrix
  BlockOperator *MHDOp = new BlockOperator( offsets );
  MHDOp->SetBlock(0, 0, FFFu);
  MHDOp->SetBlock(0, 1, BBBt);
  MHDOp->SetBlock(0, 2, ZZZ1);
  MHDOp->SetBlock(0, 3, ZZZ2);
  MHDOp->SetBlock(1, 0,  BBB);
  MHDOp->SetBlock(1, 1, CCCs);
  MHDOp->SetBlock(1, 2, XXX1);
  MHDOp->SetBlock(1, 3, XXX2);
  MHDOp->SetBlock(2, 2, MMMz);
  MHDOp->SetBlock(2, 3,  KKK);
  MHDOp->SetBlock(3, 0,  YYY);
  MHDOp->SetBlock(3, 3, FFFa);
  // - assemble the original residual
  BlockVector res(offsets);
  res.GetBlock(0) = fres;
  res.GetBlock(1) = gres;
  res.GetBlock(2) = zres;
  res.GetBlock(3) = hres;
  // - assemble initial guess on solution
  BlockVector sol(offsets);
  sol.GetBlock(0) = U0;
  sol.GetBlock(1) = P0;
  sol.GetBlock(2) = Z0;
  sol.GetBlock(3) = A0;




  // For the preconditioner *************************************************
  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators for preconditioner *******************\n";
  }
  // - recover block-operators
  Operator *FFui, *MMzi, *pSi, *aSi;
  mhdAssembler->AssemblePreconditioner( FFui, MMzi, pSi, aSi, STSolveTypeU, STSolveTypeA );

  // - recover factors
  BlockUpperTriangularPreconditioner *Uub = new BlockUpperTriangularPreconditioner( offsets ),
                                     *Uup = new BlockUpperTriangularPreconditioner( offsets );
  BlockLowerTriangularPreconditioner *Lub = new BlockLowerTriangularPreconditioner( offsets ),
                                     *Lup = new BlockLowerTriangularPreconditioner( offsets );
  AssembleLub( YYY,  FFui, ZZZ1, MMzi,      Lub );
  AssembleUub( ZZZ1, ZZZ2, MMzi, KKK,  aSi, Uub );
  AssembleLup( FFui, BBB,                   Lup );
  AssembleUup( FFui, BBBt, XXX1, XXX2, pSi, Uup );

  // - combine them together
  Array<const Operator*> precOps;
  Array<bool>            precOwn;
  switch (precType){
    // full Preconditioner Uupi*Lupi*Uubi*Lubi
    case 0:{
      precOps.SetSize(4);
      precOwn.SetSize(4);
      precOps[0] = Lub;  precOwn[0] = true;
      precOps[1] = Uub;  precOwn[1] = true;
      precOps[2] = Lup;  precOwn[2] = true;
      precOps[3] = Uup;  precOwn[3] = true;
      break;
    }
    // simplified: Uupi*Lupi*Uubi
    case 1:{
      precOps.SetSize(3);
      precOwn.SetSize(3);
      precOps[0] = Uub;  precOwn[0] = true;
      precOps[1] = Lup;  precOwn[1] = true;
      precOps[2] = Uup;  precOwn[2] = true;
      break;
    }
    // uber simplified: Uupi*Uubi
    case 2:{
      precOps.SetSize(2);
      precOwn.SetSize(2);
      precOps[0] = Uub;  precOwn[0] = true;
      precOps[1] = Uup;  precOwn[1] = true;
      break;
    }
    default:
    std::cerr<<"ERROR: Preconditioner type "<<pbType<<" not recognised."<<std::endl;
  }
  // - define preconditioner
  OperatorsSequence *MHDPr = new OperatorsSequence( precOps, precOwn );



  // -- Actually disregard prescribed initial guess for z, and instead set it so that it matches the discrete laplacian of A
  if ( myid == 0 ){
    std::cout<<"Warning: considering IG on z as discrete laplacian of A (ie, disregarding analytical solution)"<<std::endl;
  }
  BlockVector tempSol(sol), tempRes(offsets);
  tempSol.GetBlock(0) = 0.;
  tempSol.GetBlock(1) = 0.;
  tempSol.GetBlock(2) = 0.; // only keep A
  mhdAssembler->ApplyOperator( tempSol, tempRes );     // this contains the rhs (particularly, zrhs - KA in block 2)
  MMzi->Mult( tempRes.GetBlock(2), sol.GetBlock(2) );  // solve for z and overwrite solution
  // sol.GetBlock(2).Neg();
  mhdAssembler->UpdateLinearisedOperators( sol );     // update operators evaluating them at the new IG
  mhdAssembler->ApplyOperator( sol, res );            // update residual
  res.GetBlock(2) = 0.;                               // should be basically zero anyway


  MPI_Barrier(MPI_COMM_WORLD);
  assemblyStopwatch.Stop();
  totAssemblyTime += assemblyStopwatch.RealTime();


  // Initialise output folders **********************************************
  string convPath = string("./results/") + "Pb" + to_string(pbType)
                         + "_Prec" + to_string(precType) + "_STsolveU" + to_string(STSolveTypeU) + "_STsolveA" + to_string(STSolveTypeA)
                         + "_oU"   + to_string(ordU) + "_oP" + to_string(ordP) + "_oZ" + to_string(ordZ) + "_oA" + to_string(ordA)
                         + "_"     + petscrc_file + "/";
  if (!std::experimental::filesystem::exists( convPath ) && myid == 0){
    std::experimental::filesystem::create_directories( convPath );
  }
  string innerConvpath = convPath + "NP" + to_string(numProcs) + "_r"  + to_string(ref_levels) + "/";
  if (output>0 && !std::experimental::filesystem::exists( innerConvpath ) && myid == 0){
    std::experimental::filesystem::create_directories( innerConvpath );
  }
  string operPath = string("./results/operators/") + "Pb" + to_string(pbType)
                         + "_Prec" + to_string(precType) + "_STsolveU" + to_string(STSolveTypeU) + "_STsolveA" + to_string(STSolveTypeA)
                         + "_oU"   + to_string(ordU) + "_oP" + to_string(ordP) + "_oZ" + to_string(ordZ) + "_oA" + to_string(ordA)
                         + "_"     + petscrc_file + "/";
  if (output>3 && !std::experimental::filesystem::exists( operPath ) && myid == 0){
    std::experimental::filesystem::create_directories( operPath );
  }
  string innerOperPath = operPath + "NP" + to_string(numProcs) + "_r"  + to_string(ref_levels) + "/";
  if (output>3 && !std::experimental::filesystem::exists( innerOperPath ) && myid == 0){
    std::experimental::filesystem::create_directories( innerOperPath );
  }








  //*************************************************************************
  // Solve system
  //*************************************************************************
  if( myid == 0 && verbose > 0 ){
    std::cout << "SOLVE! ****************************************************\n";
  }

  // In this case, just solve the system normally, via time-stepping
  if ( STSolveTypeU == 9 || STSolveTypeA == 9 ){
    if( myid == 0 && verbose > 0 ){
      std::cout << "USING CLASSICAL TIME-STEPPING *****************************\n";
    }

    // - solve via time-stepping
    mhdAssembler->TimeStep( res, sol, convPath, ref_levels, precType, output, assemblyStopwatch );

  // otherwise, things get serious
  }else{

    // - compute norm of residual at zeroth-iteration
    MPI_Barrier(MPI_COMM_WORLD);
    assemblyStopwatch.Start();
    Vector newtonBlockRes, newtonBlockErrWRTPrevIt;
    computeBlockedNorm( res, newtonBlockRes );
    MPI_Barrier(MPI_COMM_WORLD);
    assemblyStopwatch.Stop();
    double newtonRes  = newtonBlockRes(4);
    double newtonRes0 = newtonRes;
    double newtonErrWRTPrevIt = newtonATol;
    int newtonIt = 0;
    int GMRESits = 0;
    int GMRESNoConv = 0;
    double totGMRESit = 0.; //leave it as double, so that when I'll average it, it won't round-off

    if( myid == 0 ){
      std::cout << "***********************************************************\n";
      std::cout << "Newton iteration "<<newtonIt<<", initial residual "<< newtonRes
                << ", (u,p,z,A) = ("<< newtonBlockRes(0) <<","
                                    << newtonBlockRes(1) <<","
                                    << newtonBlockRes(2) <<","
                                    << newtonBlockRes(3) <<")" << std::endl
                << "Pre - Assembly took: "<< assemblyStopwatch.RealTime() << "s" <<std::endl;
      std::cout << "***********************************************************\n";
      if ( output>0 ){
        string filename = innerConvpath +"NEWTconv.txt";
        ofstream myfile;
        myfile.open( filename, std::ios::app );
        myfile <<  "#It\t"           << "Res_norm_tot\t"        
                << "Res_norm_u\t"    << "Res_norm_p\t"    << "Res_norm_z\t"      << "Res_norm_a\t"
                << "Rel_res_norm\t"  << "Update_norm\t"   << "Inner_converged\t" <<"Inner_res\t"   <<"Inner_its\t"
                << "Solve_time\t"    << "Assembly_time\t" << std::endl;
        myfile << newtonIt <<"\t"<< newtonRes <<"\t"
                << newtonBlockRes(0) <<"\t"<< newtonBlockRes(1) <<"\t"<< newtonBlockRes(2) <<"\t"<< newtonBlockRes(3) <<"\t"
                << newtonRes/newtonRes0 <<"\t"<< 0.0 <<"\t" << false <<"\t"<< 0.0 <<"\t"<<GMRESits <<"\t"
                << 0.0 <<"\t"<< assemblyStopwatch.RealTime() <<std::endl;
        myfile.close();
      }
    }
    if( output > 3 ){
      string filename = innerOperPath + "Nit" + to_string(newtonIt) + "_";
      mhdAssembler->PrintMatrices( filename );
      // - Print out rhs (that's the vector which will be used for testing)
      ofstream myfile;
      myfile.precision(std::numeric_limits< double >::max_digits10);
      myfile.open( filename+"rhs"+to_string(myid)+".dat" );
      res.Print(myfile,1);
      myfile.close();
    }

    totAssemblyTime += assemblyStopwatch.RealTime();
    assemblyStopwatch.Clear();



    //***********************************************************************
    // NEWTON ITERATIONS
    //***********************************************************************
    // - stop if:       
    bool stopNewton = ( newtonIt >= maxNewtonIt )                // max it reached
                   || ( newtonRes < newtonATol )                 // residual small enough
                   || ( newtonRes/newtonRes0 < newtonRTol );     // relative residual small enough
                   // || ( newtonErrWRTPrevIt < newtonATol );    // difference wrt prev it small enough
    while(!stopNewton){

    
      solverStopwatch.Clear();

      // Define inner solver
      MPI_Barrier(MPI_COMM_WORLD);
      solverStopwatch.Start();
      PetscLinearSolver solver(MPI_COMM_WORLD, "solver_");
      bool isIterative = true;
      solver.iterative_mode = isIterative;

      // - register operator and preconditioner with the solver
      solver.SetPreconditioner(*MHDPr);
      solver.SetOperator(*MHDOp);
      MPI_Barrier(MPI_COMM_WORLD);
      solverStopwatch.Stop();

      // - eventually register viewer to print to file residual evolution for inner iterations
      if ( output>1 ){
        string filename = innerConvpath +"GMRESconv_Nit" + to_string(newtonIt) + ".txt";
        // - create viewer to instruct KSP object how to print residual evolution to file
        PetscViewer    viewer;
        PetscViewerAndFormat *vf;
        PetscViewerCreate( PETSC_COMM_WORLD, &viewer );
        PetscViewerSetType( viewer, PETSCVIEWERASCII );
        PetscViewerFileSetMode( viewer, FILE_MODE_APPEND );
        PetscViewerFileSetName( viewer, filename.c_str() );
        // - register it to the ksp object
        KSP ksp = solver;
        PetscViewerAndFormatCreate( viewer, PETSC_VIEWER_DEFAULT, &vf );
        PetscViewerDestroy( &viewer );
  
        if( pbType<=4 ){
          if ( myid == 0 ){
            std::cout<<"Warning: we're printing the error wrt the analytical solution at each iteration."<<std::endl
                     <<"         This is bound to slow down GMRES *a lot*, so leave this code only for testing purposes!"<<std::endl;
          }
          // - create a more complex context if fancier options must be printed (error wrt analytical solution)
          UPErrorMonitorCtx mctx;
          mctx.lenghtU = offsets[1];
          mctx.lenghtP = offsets[2] - offsets[1];
          mctx.lenghtZ = offsets[3] - offsets[2];
          mctx.lenghtA = offsets[4] - offsets[3];
          mctx.STassembler = mhdAssembler;
          mctx.comm = MPI_COMM_WORLD;
          mctx.vf   = vf;
          // mctx.path = path + "/ParaView/NP" + to_string(numProcs) + "_r"  + to_string(ref_levels)+"/";
          UPErrorMonitorCtx* mctxptr = &mctx;
          KSPMonitorSet( ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))UPErrorMonitor, mctxptr, NULL );
        }else{
          UPSplitResidualMonitorCtx mctx;
          mctx.lenghtU = offsets[1];
          mctx.lenghtP = offsets[2] - offsets[1];
          mctx.lenghtZ = offsets[3] - offsets[2];
          mctx.lenghtA = offsets[4] - offsets[3];
          mctx.comm = MPI_COMM_WORLD;
          mctx.vf   = vf;
          // mctx.path = path + "/ParaView/NP" + to_string(numProcs) + "_r"  + to_string(ref_levels)+"/";
          UPSplitResidualMonitorCtx* mctxptr = &mctx;
          KSPMonitorSet( ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))UPSplitResidualMonitor, mctxptr, NULL );
          // KSPMonitorSet( ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorDefault,
          //                vf, (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy );
        }
      }




      // Define initial guess for update to solution (all zero)
      MPI_Barrier(MPI_COMM_WORLD);
      solverStopwatch.Start();
      BlockVector deltaSol(offsets);
      deltaSol.GetBlock(0) = 0.;
      deltaSol.GetBlock(1) = 0.;
      deltaSol.GetBlock(2) = 0.;
      deltaSol.GetBlock(3) = 0.;

      // Solve for current linearisation
      solver.Mult( res, deltaSol );
      sol += deltaSol;
      MPI_Barrier(MPI_COMM_WORLD);
      solverStopwatch.Stop();


      // Update relevant quantities
      // - solution (include this into assembly time)
      // - residual
      // -- apply operator
      MPI_Barrier(MPI_COMM_WORLD);
      assemblyStopwatch.Start();
      mhdAssembler->ApplyOperator( sol, res );
      // -- compute residual norm
      computeBlockedNorm( res, newtonBlockRes );
      newtonRes  = newtonBlockRes(4);
      // -- compute norm of newton update
      computeBlockedNorm( deltaSol, newtonBlockErrWRTPrevIt );
      newtonErrWRTPrevIt = newtonBlockErrWRTPrevIt(4);
      MPI_Barrier(MPI_COMM_WORLD);
      assemblyStopwatch.Stop();
      
      // Output relevant measurements
      newtonIt++;
      GMRESits  = solver.GetNumIterations();
      totGMRESit += GMRESits;
      if( myid == 0 ){
        if (solver.GetConverged()){
          std::cout << "Inner solver converged in ";
        }else{
          std::cout << "Inner solver *DID NOT* converge in ";
          GMRESNoConv++;
        }
        std::cout<< GMRESits << " iterations. Residual "<<solver.GetFinalNorm() << std::endl
                 << " - Assembly took: "<< assemblyStopwatch.RealTime() << "s"  << std::endl
                 << " - Solver   took: "<<   solverStopwatch.RealTime() << "s"  << std::endl;
        if( output>0 ){
          string filename = innerConvpath +"NEWTconv.txt";
          ofstream myfile;
          myfile.open( filename, std::ios::app );
          myfile << newtonIt <<"\t"<< newtonRes <<"\t"
                  << newtonBlockRes(0) <<"\t"<< newtonBlockRes(1) <<"\t"<< newtonBlockRes(2) <<"\t"<< newtonBlockRes(3) <<"\t"
                  << newtonRes/newtonRes0 <<"\t"<< newtonErrWRTPrevIt << "\t"
                  << solver.GetConverged() <<"\t"<< solver.GetFinalNorm() <<"\t"<<GMRESits <<"\t"
                  << solverStopwatch.RealTime() <<"\t"<< assemblyStopwatch.RealTime() <<std::endl;
          myfile.close();
        }      
        std::cout << "***********************************************************\n";
        std::cout << "Newton iteration "<< newtonIt <<", residual "<< newtonRes
                  << ", (u,p,z,A) = ("<< newtonBlockRes(0) <<","
                                      << newtonBlockRes(1) <<","
                                      << newtonBlockRes(2) <<","
                                      << newtonBlockRes(3) <<")" << std::endl;
        std::cout << "***********************************************************\n";
      }

      if( output > 3 ){
        string filename = innerOperPath + "Nit" + to_string(newtonIt) + "_";
        // - Print out solution
        ofstream myfile;
        myfile.precision(std::numeric_limits< double >::max_digits10);
        myfile.open( filename+"deltaSol"+to_string(myid)+".dat" );
        deltaSol.Print(myfile,1);
        myfile.close();
      }

      totSolverTime   += solverStopwatch.RealTime();
      totAssemblyTime += assemblyStopwatch.RealTime();
      solverStopwatch.Clear();
      assemblyStopwatch.Clear();



      // Check stopping criterion
      stopNewton = ( newtonIt >= maxNewtonIt )                // max it reached
                || ( newtonRes < newtonATol )                 // residual small enough
                || ( newtonRes/newtonRes0 < newtonRTol );     // relative residual small enough
                // || ( newtonErrWRTPrevIt < newtonATol );    // difference wrt prev it small enough
      // - and eventually update the operators
      if( !stopNewton ){
        if( myid == 0 && verbose > 0 ){
          std::cout << "Update operators ******************************************\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        assemblyStopwatch.Start();
        mhdAssembler->UpdateLinearisedOperators( sol );
        MPI_Barrier(MPI_COMM_WORLD);
        assemblyStopwatch.Stop();
    
        if( output > 3 ){
          // - Print out updated operators
          string filename = innerOperPath + "Nit" + to_string(newtonIt) + "_";
          mhdAssembler->PrintMatrices( filename );
          // - Print out updated resitual
          ofstream myfile;
          myfile.precision(std::numeric_limits< double >::max_digits10);
          myfile.open( filename+"rhs"+to_string(myid)+".dat" );
          res.Print(myfile,1);
          myfile.close();
        }
      }

    }


    // Print info for non-linear solve
    if ( myid == 0 ){
      if( newtonIt < maxNewtonIt ){
        std::cout << "Newton outer solver converged in "          << newtonIt;
      }else{
        std::cout << "Newton outer solver *DID NOT* converge in " << maxNewtonIt;
      }
      std::cout   << " iterations. Residual norm is "             << newtonRes;
      std::cout   << ", avg internal GMRES it are "               << totGMRESit/newtonIt << "."<< std::endl;
      std::cout   << "Solve took "                                << totSolverTime <<"s";
      std::cout   << ", assembly took "                           << totAssemblyTime <<"s in total.\n";
      std::cout   << "***********************************************************\n";

      // - eventually store info on newton convergence
      if( output>0 ){
        string filename = convPath + "Newton_convergence_results.txt";
        ofstream myfile;
        myfile.open( filename, std::ios::app );
        myfile << Tend       << ",\t" << dt                  << ",\t" << numProcs            << ",\t" << ref_levels << ",\t"
               << newtonIt   << ",\t" << totGMRESit/newtonIt << ",\t" << totGMRESit/numProcs << ",\t" << GMRESNoConv << ",\t"
               << newtonRes0 << ",\t" << newtonRes           << ",\t" << totSolverTime       << ",\t" << totAssemblyTime << std::endl;
        myfile.close();
      }    
    }

  }
  


  // OUTPUT -----------------------------------------------------------------
  if( myid == 0 && verbose > 0 ){
    std::cout << "Post-processing *******************************************\n";
  }


  // - save solution (beware of memory consumption!)
  if( output>2 ){
    string outFilePath = "ParaView";
    string outFileName = "STIMHD2D_" + pbName;
    // if (newtonIt == 0){
    //   mhdAssembler->SaveExactSolution( outFilePath, outFileName+"_Ex" );
    // }
    // if ( pbType > 10 ){
    //   outFileName += "_it" + to_string(newtonIt);
    // }
    int colsU[2] = { myid*( FFFu->NumRows() ), (myid+1)*( FFFu->NumRows() ) };
    int colsP[2] = { myid*(  BBB->NumRows() ), (myid+1)*(  BBB->NumRows() ) };
    int colsZ[2] = { myid*( MMMz->NumRows() ), (myid+1)*( MMMz->NumRows() ) };
    int colsA[2] = { myid*( FFFa->NumRows() ), (myid+1)*( FFFa->NumRows() ) };

    HypreParVector uh( MPI_COMM_WORLD, numProcs*( FFFu->NumRows() ), sol.GetBlock(0).GetData(), colsU ); 
    HypreParVector ph( MPI_COMM_WORLD, numProcs*(  BBB->NumRows() ), sol.GetBlock(1).GetData(), colsP ); 
    HypreParVector zh( MPI_COMM_WORLD, numProcs*( MMMz->NumRows() ), sol.GetBlock(2).GetData(), colsZ ); 
    HypreParVector ah( MPI_COMM_WORLD, numProcs*( FFFa->NumRows() ), sol.GetBlock(3).GetData(), colsA ); 

    mhdAssembler->SaveSolution( uh, ph, zh, ah, outFilePath, outFileName );

    if ( pbType<=4 ){
      mhdAssembler->SaveError( uh, ph, zh, ah, outFilePath, outFileName+"_err" );
    }
  }







  if( myid == 0 && verbose > 0 ){
    std::cout << "Clean-up **************************************************\n";
  }


  delete MHDPr;
  delete MHDOp;
  // delete solver;

  // delete FFFu,
  // delete MMMz;
  // delete FFFa;
  // delete BBBt;
  // delete BBB;
  // delete ZZZ1;
  // delete ZZZ2;
  // delete KKK;
  // delete YYY;

  // delete fres;
  // delete gres;
  // delete zres;
  // delete hres;
  // delete uEx;
  // delete pEx;
  // delete zEx;
  // delete aEx;
  // delete U0;
  // delete P0;
  // delete Z0;
  // delete A0;


  delete mhdAssembler;




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
  double errU, errP, errZ, errA, glbErrU, glbErrP, glbErrZ, glbErrA;
  PetscInt lclSize;

  // recover current solution
  KSPBuildSolution( ksp, NULL, &x );
  VecGetLocalSize( x, &lclSize );

  if( lclSize != (ctx->lenghtU + ctx->lenghtP + ctx->lenghtZ + ctx->lenghtA) ){
    std::cerr<<"ERROR! ErrorMonitor: Something went very wrong"<<std::endl
             <<"       seems like the solution is stored in a weird way in the PETSC solver,"<<std::endl
             <<"       and sizes mismatch: "<< lclSize << "=/="<<ctx->lenghtU<<"+"<<ctx->lenghtP<<std::endl;
  }
  
  // get local raw pointer
  double* vals;
  VecGetArray( x, &vals );

  // store in MFEM-friendly variables
  Vector uh( vals,                                              ctx->lenghtU );  // hopefully that's the way it's stored in x, with u and p contiguos
  Vector ph( vals + ctx->lenghtU,                               ctx->lenghtP );
  Vector zh( vals + ctx->lenghtU + ctx->lenghtP,                ctx->lenghtZ );
  Vector ah( vals + ctx->lenghtU + ctx->lenghtP + ctx->lenghtZ, ctx->lenghtA );
  

  // compute error per each time step
  ctx->STassembler->ComputeL2Error( uh, ph, zh, ah, errU, errP, errZ , errA );

  // compute Linf norm in time or errors
  MPI_Reduce( &errU, &glbErrU, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  MPI_Reduce( &errP, &glbErrP, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  MPI_Reduce( &errZ, &glbErrZ, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  MPI_Reduce( &errA, &glbErrA, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  
  // print
  PetscViewer viewer =  ctx->vf->viewer;
  PetscViewerPushFormat( viewer, ctx->vf->format);
  // PetscViewerASCIIAddTab( viewer,((PetscObject)ksp)->tablevel );
  if (n == 0 ) {// && ((PetscObject)ksp)->prefix) {
    // PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);
    PetscViewerASCIIPrintf(viewer,"  Residual norms for outer solver.\n");
  }
  PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm_erru_errp_errz_erra %14.12e\t%14.12e\t%14.12e\t%14.12e\t%14.12e \n",
                         n,(double)rnorm,(double)glbErrU,(double)glbErrP,(double)glbErrZ,(double)glbErrA);
  // PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);
  PetscViewerPopFormat(viewer);

  // if( n%10 == 0 ){
  //   int myid;
  //   MPI_Comm_rank(ctx->comm, &myid);

  //   if ( myid == 0 ){
  //     std::cout<<"Saving plot of error in Paraview-friendly format"<<std::endl;
  //   }
  //   ctx->STassembler->SaveError( uh, ph, zh, ah, ctx->path + "it" + to_string(n) + "/", "error");
  // }

  // clean-up
  // vector x should *not* be destroyed by user)
  // delete vals;

  return 0;
    
}



// Function to destroy context of function to monitor actual error
PetscErrorCode UPSplitResidualMonitorDestroy( void ** mctx ){
  PetscErrorCode ierr;
  UPSplitResidualMonitorCtx *ctx = (UPSplitResidualMonitorCtx*)mctx;
  ierr = PetscViewerAndFormatDestroy( &(ctx->vf)); CHKERRQ(ierr);
  delete ctx;
  return 0;
}

// Function to monitor actual error
PetscErrorCode UPSplitResidualMonitor( KSP ksp, PetscInt n, PetscReal rnorm, void *mctx ){
  UPSplitResidualMonitorCtx *ctx = (UPSplitResidualMonitorCtx*)mctx;
  Vec res;
  double resU, resP, resZ, resA, glbResU, glbResP, glbResZ, glbResA;
  PetscInt lclSize;

  // recover current solution
  KSPBuildResidual( ksp, NULL, NULL, &res );
  VecGetLocalSize( res, &lclSize );

  if( lclSize != (ctx->lenghtU + ctx->lenghtP + ctx->lenghtZ + ctx->lenghtA) ){
    std::cerr<<"ERROR! ErrorMonitor: Something went very wrong"<<std::endl
             <<"       seems like the solution is stored in a weird way in the PETSC solver,"<<std::endl
             <<"       and sizes mismatch: "<< lclSize << "=/="<<ctx->lenghtU<<"+"<<ctx->lenghtP<<std::endl;
  }
  
  // get local raw pointer
  double* vals;
  VecGetArray( res, &vals );

  // store in MFEM-friendly variables
  Vector resu( vals,                                              ctx->lenghtU );  // hopefully that's the way it's stored in x, with u and p contiguos
  Vector resp( vals + ctx->lenghtU,                               ctx->lenghtP );
  Vector resz( vals + ctx->lenghtU + ctx->lenghtP,                ctx->lenghtZ );
  Vector resa( vals + ctx->lenghtU + ctx->lenghtP + ctx->lenghtZ, ctx->lenghtA );

  resU = resu.Norml2();
  resP = resp.Norml2();
  resZ = resz.Norml2();
  resA = resa.Norml2();  

  // // compute Linf norm in time of residuals
  // MPI_Reduce( &resU, &glbResU, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  // MPI_Reduce( &resP, &glbResP, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  // MPI_Reduce( &resZ, &glbResZ, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );
  // MPI_Reduce( &resA, &glbResA, 1, MPI_DOUBLE, MPI_MAX, 0, ctx->comm );

  // compute L2 norm in time of residuals
  resU *= resU;
  resP *= resP;
  resZ *= resZ;
  resA *= resA;
  MPI_Reduce( &resU, &glbResU, 1, MPI_DOUBLE, MPI_SUM, 0, ctx->comm );
  MPI_Reduce( &resP, &glbResP, 1, MPI_DOUBLE, MPI_SUM, 0, ctx->comm );
  MPI_Reduce( &resZ, &glbResZ, 1, MPI_DOUBLE, MPI_SUM, 0, ctx->comm );
  MPI_Reduce( &resA, &glbResA, 1, MPI_DOUBLE, MPI_SUM, 0, ctx->comm );
  glbResU = sqrt(glbResU);
  glbResP = sqrt(glbResP);
  glbResZ = sqrt(glbResZ);
  glbResA = sqrt(glbResA);
  
  // print
  PetscViewer viewer =  ctx->vf->viewer;
  PetscViewerPushFormat( viewer, ctx->vf->format);
  // PetscViewerASCIIAddTab( viewer,((PetscObject)ksp)->tablevel );
  if (n == 0 ) {// && ((PetscObject)ksp)->prefix) {
    // PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);
    PetscViewerASCIIPrintf(viewer,"  Residual norms for outer solver.\n");
  }
  PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm_resu_resp_resz_resa %14.12e\t%14.12e\t%14.12e\t%14.12e\t%14.12e \n",
                         n,(double)rnorm,(double)glbResU,(double)glbResP,(double)glbResZ,(double)glbResA);
  // PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);
  PetscViewerPopFormat(viewer);

  // if( n%10 == 0 ){
  //   int myid;
  //   MPI_Comm_rank(ctx->comm, &myid);

  //   if ( myid == 0 ){
  //     std::cout<<"Saving plot of error in Paraview-friendly format"<<std::endl;
  //   }
  //   ctx->STassembler->SaveError( uh, ph, zh, ah, ctx->path + "it" + to_string(n) + "/", "error");
  // }

  // clean-up
  // vector x should *not* be destroyed by user)
  // delete vals;

  return 0;
    
}










//*************************************************************************
// ASSEMBLE BLOCK OPERATORS
//*************************************************************************
// Assembles lower factor of LU factorisation of velocity/magnetic field part of preconditioner
//       ⌈ I                       ⌉
// Lub = |       I                 |
//       |          I              |
//       ⌊ Y*Fui   -Y*Fui*Z1*Mzi I ⌋
void AssembleLub( const Operator* Y,  const Operator* Fui, const Operator* Z1, const Operator* Mzi,
                  BlockLowerTriangularPreconditioner* Lub ){

  Array<const Operator*> YFuiOps(2);
  YFuiOps[0] = Fui;
  YFuiOps[1] = Y;
  OperatorsSequence* YFui = new OperatorsSequence( YFuiOps );   // does not own

  ScaledOperator* mMzi = new ScaledOperator( Mzi, -1.0 );

  Array<const Operator*> mYFuiZ1Mziops(3);
  Array<bool>            mYFuiZ1Mziown(3);
  mYFuiZ1Mziops[0] = mMzi; mYFuiZ1Mziown[0] = true;
  mYFuiZ1Mziops[1] = Z1;   mYFuiZ1Mziown[1] = false;
  mYFuiZ1Mziops[2] = YFui; mYFuiZ1Mziown[2] = false;
  OperatorsSequence* mYFuiZ1Mzi = new OperatorsSequence( mYFuiZ1Mziops, mYFuiZ1Mziown );

  Lub->iterative_mode = false;
  Lub->SetBlock( 3, 0,       YFui );
  Lub->SetBlock( 3, 2, mYFuiZ1Mzi );
  Lub->owns_blocks = true;
}


// Assembles modified upper factor of LU factorisation of velocity/magnetic field part of preconditioner
//     ⌈ Fui       ⌉   ⌈ I   Z1 Z2 ⌉
// Uub*|     I     | = |   I       |
//     |       I   |   |     Mz K  |
//     ⌊         I ⌋   ⌊        aS ⌋
void AssembleUub( const Operator* Z1, const Operator* Z2, const Operator* Mzi, const Operator* K, const Operator* aSi,
                  BlockUpperTriangularPreconditioner* Uub ){
  Uub->iterative_mode = false;

  Uub->SetBlock( 0, 2, Z1  );
  Uub->SetBlock( 0, 3, Z2  );
  Uub->SetBlock( 2, 2, Mzi );
  Uub->SetBlock( 2, 3, K   );
  Uub->SetBlock( 3, 3, aSi );
  Uub->owns_blocks = false;
}


// Assembles lower factor of LU factorisation of velocity/pressure part of preconditioner
//       ⌈    I          ⌉
// Lup = | B*Fu^-1 I     |
//       |           I   |
//       ⌊             I ⌋
void AssembleLup( const Operator* Fui, const Operator* B, BlockLowerTriangularPreconditioner* Lup ){

  Array<const Operator*> BFuiOps(2);
  BFuiOps[0] = Fui;
  BFuiOps[1] = B;
  OperatorsSequence* BFui = new OperatorsSequence( BFuiOps );   // does not own
  
  Lup->iterative_mode = false;
  Lup->SetBlock( 1, 0, BFui );
  Lup->owns_blocks = true;
}


// Assembles upper factor of LU factorisation of velocity/pressure part of preconditioner
//       ⌈ Fu Bt       ⌉
// Uup = |    pS X1 X2 |
//       |       I     |
//       ⌊          I  ⌋
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* X1, const Operator* X2,
                  const Operator* pSi, BlockUpperTriangularPreconditioner* Uup ){
  Uup->iterative_mode = false;
  Uup->SetBlock( 0, 0, Fui );
  Uup->SetBlock( 0, 1, Bt  );
  Uup->SetBlock( 1, 1, pSi );
  Uup->SetBlock( 1, 2, X1  );
  Uup->SetBlock( 1, 3, X2  );
  Uup->owns_blocks = false;
}





//*************************************************************************
// MISC
//*************************************************************************
// Returns norm of each block (norm(0-3)) and total norm (norm(4)) of a given a shared block space-time vector
void computeBlockedNorm( const BlockVector& u, Vector& norm ){
  norm.SetSize(5);
  double temp = 0.;

  for ( int i = 0; i < 4; ++i ){
    temp = u.GetBlock(i).Norml2();
    temp*= temp;
    MPI_Allreduce( MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    norm(i) = sqrt(temp);
  }

  temp = u.Norml2();
  temp*= temp;
  MPI_Allreduce( MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  norm(4) = sqrt(temp);

}



