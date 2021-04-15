//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "imhd2dstoperatorassembler.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include "operatorssequence.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <limits>
#include <cmath>
//---------------------------------------------------------------------------
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;
//---------------------------------------------------------------------------
// PB parameters pre-definition:
// - simple cooked-up pb with analytical solution
void   uFun_ex_an0( const Vector & x, const double t, Vector & u );
double pFun_ex_an0( const Vector & x, const double t             );
double zFun_ex_an0( const Vector & x, const double t             );
double aFun_ex_an0( const Vector & x, const double t             );
void   fFun_an0(    const Vector & x, const double t, Vector & f );
void   nFun_an0(    const Vector & x, const double t, Vector & f );
double gFun_an0(    const Vector & x, const double t             );
double hFun_an0(    const Vector & x, const double t             );
double mFun_an0(    const Vector & x, const double t             );
void   wFun_an0(    const Vector & x, const double t, Vector & w );
double qFun_an0(    const Vector & x, const double t             );
double cFun_an0(    const Vector & x, const double t             );
double yFun_an0(    const Vector & x, const double t             );
void   uFun_ex_an1( const Vector & x, const double t, Vector & u );
double pFun_ex_an1( const Vector & x, const double t             );
double zFun_ex_an1( const Vector & x, const double t             );
double aFun_ex_an1( const Vector & x, const double t             );
void   fFun_an1(    const Vector & x, const double t, Vector & f );
void   nFun_an1(    const Vector & x, const double t, Vector & f );
double gFun_an1(    const Vector & x, const double t             );
double hFun_an1(    const Vector & x, const double t             );
double mFun_an1(    const Vector & x, const double t             );
void   wFun_an1(    const Vector & x, const double t, Vector & w );
double qFun_an1(    const Vector & x, const double t             );
double cFun_an1(    const Vector & x, const double t             );
double yFun_an1(    const Vector & x, const double t             );
void   uFun_ex_an2( const Vector & x, const double t, Vector & u );
double pFun_ex_an2( const Vector & x, const double t             );
double zFun_ex_an2( const Vector & x, const double t             );
double aFun_ex_an2( const Vector & x, const double t             );
void   fFun_an2(    const Vector & x, const double t, Vector & f );
void   nFun_an2(    const Vector & x, const double t, Vector & f );
double gFun_an2(    const Vector & x, const double t             );
double hFun_an2(    const Vector & x, const double t             );
double mFun_an2(    const Vector & x, const double t             );
void   wFun_an2(    const Vector & x, const double t, Vector & w );
double qFun_an2(    const Vector & x, const double t             );
double cFun_an2(    const Vector & x, const double t             );
double yFun_an2(    const Vector & x, const double t             );
void   uFun_ex_an3( const Vector & x, const double t, Vector & u );
double pFun_ex_an3( const Vector & x, const double t             );
double zFun_ex_an3( const Vector & x, const double t             );
double aFun_ex_an3( const Vector & x, const double t             );
void   fFun_an3(    const Vector & x, const double t, Vector & f );
void   nFun_an3(    const Vector & x, const double t, Vector & f );
double gFun_an3(    const Vector & x, const double t             );
double hFun_an3(    const Vector & x, const double t             );
double mFun_an3(    const Vector & x, const double t             );
void   wFun_an3(    const Vector & x, const double t, Vector & w );
double qFun_an3(    const Vector & x, const double t             );
double cFun_an3(    const Vector & x, const double t             );
double yFun_an3(    const Vector & x, const double t             );
void   uFun_ex_an4( const Vector & x, const double t, Vector & u );
double pFun_ex_an4( const Vector & x, const double t             );
double zFun_ex_an4( const Vector & x, const double t             );
double aFun_ex_an4( const Vector & x, const double t             );
void   fFun_an4(    const Vector & x, const double t, Vector & f );
void   nFun_an4(    const Vector & x, const double t, Vector & f );
double gFun_an4(    const Vector & x, const double t             );
double hFun_an4(    const Vector & x, const double t             );
double mFun_an4(    const Vector & x, const double t             );
void   wFun_an4(    const Vector & x, const double t, Vector & w );
double qFun_an4(    const Vector & x, const double t             );
double cFun_an4(    const Vector & x, const double t             );
double yFun_an4(    const Vector & x, const double t             );
// Kelvin-Helmholtz instability
void   uFun_ex_KHI( const Vector & x, const double t, Vector & u );
double pFun_ex_KHI( const Vector & x, const double t             );
double zFun_ex_KHI( const Vector & x, const double t             );
double aFun_ex_KHI( const Vector & x, const double t             );
void   fFun_KHI(    const Vector & x, const double t, Vector & f );
void   nFun_KHI(    const Vector & x, const double t, Vector & f );
double gFun_KHI(    const Vector & x, const double t             );
double hFun_KHI(    const Vector & x, const double t             );
double mFun_KHI(    const Vector & x, const double t             );
void   wFun_KHI(    const Vector & x, const double t, Vector & w );
double qFun_KHI(    const Vector & x, const double t             );
double cFun_KHI(    const Vector & x, const double t             );
double yFun_KHI(    const Vector & x, const double t             );
// Island coalescing
void   uFun_ex_island( const Vector & x, const double t, Vector & u );
double pFun_ex_island( const Vector & x, const double t             );
double zFun_ex_island( const Vector & x, const double t             );
double aFun_ex_island( const Vector & x, const double t             );
void   fFun_island(    const Vector & x, const double t, Vector & f );
void   nFun_island(    const Vector & x, const double t, Vector & f );
double gFun_island(    const Vector & x, const double t             );
double hFun_island(    const Vector & x, const double t             );
double mFun_island(    const Vector & x, const double t             );
void   wFun_island(    const Vector & x, const double t, Vector & w );
double qFun_island(    const Vector & x, const double t             );
double cFun_island(    const Vector & x, const double t             );
double yFun_island(    const Vector & x, const double t             );
// Modified Hartmann flow
void   uFun_ex_hartmann( const Vector & x, const double t, Vector & u );
double pFun_ex_hartmann( const Vector & x, const double t             );
double zFun_ex_hartmann( const Vector & x, const double t             );
double aFun_ex_hartmann( const Vector & x, const double t             );
void   fFun_hartmann(    const Vector & x, const double t, Vector & f );
void   nFun_hartmann(    const Vector & x, const double t, Vector & f );
double gFun_hartmann(    const Vector & x, const double t             );
double hFun_hartmann(    const Vector & x, const double t             );
double mFun_hartmann(    const Vector & x, const double t             );
void   wFun_hartmann(    const Vector & x, const double t, Vector & w );
double qFun_hartmann(    const Vector & x, const double t             );
double cFun_hartmann(    const Vector & x, const double t             );
double yFun_hartmann(    const Vector & x, const double t             );
// Cavity flow
void   uFun_ex_cavity( const Vector & x, const double t, Vector & u );
double pFun_ex_cavity( const Vector & x, const double t             );
double zFun_ex_cavity( const Vector & x, const double t             );
double aFun_ex_cavity( const Vector & x, const double t             );
void   fFun_cavity(    const Vector & x, const double t, Vector & f );
void   nFun_cavity(    const Vector & x, const double t, Vector & f );
double gFun_cavity(    const Vector & x, const double t             );
double hFun_cavity(    const Vector & x, const double t             );
double mFun_cavity(    const Vector & x, const double t             );
void   wFun_cavity(    const Vector & x, const double t, Vector & w );
double qFun_cavity(    const Vector & x, const double t             );
double cFun_cavity(    const Vector & x, const double t             );
double yFun_cavity(    const Vector & x, const double t             );
// // - poiseuille flow
// void   uFun_ex_poiseuille( const Vector & x, const double t, Vector & u );
// double pFun_ex_poiseuille( const Vector & x, const double t             );
// void   fFun_poiseuille(    const Vector & x, const double t, Vector & f );
// void   nFun_poiseuille(    const Vector & x, const double t, Vector & f );
// double gFun_poiseuille(    const Vector & x, const double t             );
// void   uFun_ex_poiseuilleC(const Vector & x, const double t, Vector & u );  // constant-in-time counterpart
// double pFun_ex_poiseuilleC(const Vector & x, const double t             );
// void   fFun_poiseuilleC(   const Vector & x, const double t, Vector & f );
// void   nFun_poiseuilleC(   const Vector & x, const double t, Vector & f );
// double gFun_poiseuilleC(   const Vector & x, const double t             );
// // - flow over step
// void   uFun_ex_step( const Vector & x, const double t, Vector & u );
// double pFun_ex_step( const Vector & x, const double t             );
// void   fFun_step(    const Vector & x, const double t, Vector & f );
// void   nFun_step(    const Vector & x, const double t, Vector & f );
// double gFun_step(    const Vector & x, const double t             );
// // - double-glazing problem
// void   uFun_ex_glazing( const Vector & x, const double t, Vector & u );
// double pFun_ex_glazing( const Vector & x, const double t             );
// void   fFun_glazing(    const Vector & x, const double t, Vector & f );
// void   nFun_glazing(    const Vector & x, const double t, Vector & f );
// void   wFun_glazing(    const Vector & x, const double t, Vector & f );
// double gFun_glazing(    const Vector & x, const double t             );
//---------------------------------------------------------------------------
// Handy function for monitoring quantities of interest - predefinition
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
// Handy functions for assembling single components of the preconditioner
void assembleLub( const Operator* Y,   const Operator* Fui,
                  const Operator* Z1,  const Operator* Mzi, BlockLowerTriangularPreconditioner* Lub );
void assembleUub( const Operator* Z1,  const Operator* Z2, const Operator* Mzi, const Operator* K,
                  const Operator* aSi, BlockUpperTriangularPreconditioner* Uub );
void assembleLup( const Operator* Fui, const Operator* B,   BlockLowerTriangularPreconditioner* Lup );
void assembleUup( const Operator* Fui, const Operator* Bt, const Operator* pSi, BlockUpperTriangularPreconditioner* Uup );
//---------------------------------------------------------------------------
int main(int argc, char *argv[]){

  // for now, assume no spatial parallelisation: each processor handles a time-step
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  double Tend = 1.;

  bool anSolution = false;

  // Initialise problem parameters
  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;
  int ref_levels = 4;
  const char *petscrc_file = "rc_SpaceTimeIMHD2D_SingAp";
  int verbose = 0;
  int precType = 1;
  int STSolveTypeU = 0;
  int STSolveTypeA = 0;
  int pbType = 1;   
  string pbName = "";
  int output = 2;
  // these tag each boundary to identify dirichlet BC for u, p and A
  Array<int> essTagsU(0); // - the domain MUST be rectangular, and the tags for its 4 sides MUST be, respectively:
  Array<int> essTagsV(0); //    north=1 east=2 south=3 west=4  
  Array<int> essTagsP(0); 
  Array<int> essTagsA(0); 
  
  // - stop criteria for newton iterations (only used in NS: changed in the switch pbtype later on)
  double newtonTol = 1e-8;
  int maxNewtonIt  = 4;


  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&ordU, "-oU", "--orderU",
                "Finite element order (polynomial degree) for velocity field (default: 2)");
  args.AddOption(&ordP, "-oP", "--orderP",
                "Finite element order (polynomial degree) for pressure field (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--orderZ",
                "Finite element order (polynomial degree) for Laplacian of vector potential (default: 1)");
  args.AddOption(&ordA, "-oA", "--orderA",
                "Finite element order (polynomial degree) for vector potential field (default: 2)");
  args.AddOption(&ref_levels, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&Tend, "-T", "--Tend",
                "Final time (default: 1.0)");
  args.AddOption(&pbType, "-Pb", "--problem",
                "Problem: 1 to 4-Analytical test cases\n"
                "              5-Kelvin-Helmholtz instability\n"
                "              6-Island coalescensce\n"
                "              7-Modified Hartmann flow\n"
                "             11-Driven cavity flow\n"
        );
  args.AddOption(&precType, "-P", "--preconditioner",
                "Type of preconditioner: 0-Space-time Cyr et al: Uupi*Lupi*Uubi*Lubi\n"
                "                        1-Space-time Cyr et al simplified: Uupi*Lupi*Uubi (default)\n"
                "                        2-Space-time Cyr et al uber simplified: Uupi*Uubi\n"
        );
  args.AddOption(&STSolveTypeU, "-STU", "--spacetimesolveU",
                "Type of solver for velocity space-time matrix: 0-time stepping (default)\n"
        "                                                       9-Sequential time-stepping for whole ST system - ignores many other options\n"
        );
  args.AddOption(&STSolveTypeA, "-STA", "--spacetimesolveA",
                "Type of solver for potential wave space-time matrix: 0-time stepping - implicit leapfrog (default)\n"
                "                                                     1-time stepping - explicit leapfrog\n"
                "                                                     2-time stepping on full Fa*Mai*Fa+|B|/mu*Aa matrix\n"
        // "                                                             1-boomerAMG (AIR)\n"
        // "                                                             2-GMRES+boomerAMG (AIR)\n"
        // "                                                             3-Parareal (not fully tested)\n"
        "                                                             9-Sequential time-stepping for whole ST system - ignores many other options\n"
        );
  args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                "PetscOptions file to use: rc_STMHD         TODO! (direct (LU) solvers - remember to flag _SingAp if Ap is singular))\n"
        "                                  rc_STMHD_approx2 (MG solvers for Ap,Fu, Cheb for Mp)\n"
        "                                  rc_STMHD_FGMRES  (use FGMRES rather than GMRES for outer solver (useful if ST=2))");
  args.AddOption(&verbose, "-V", "--verbose",
                "Control how much info to print to terminal:(=-1   print large block matrices, and trigger eigs analysis - bit of a hack)\n"
        "                                                    >0    basic info\n"
        "                                                    >1   +info on large (space-time) block assembly\n"
        "                                                    >5   +info on small (single time-step) blocks assembly\n"
        "                                                    >10  +more details on single time-step assembly\n"
        "                                                    >20  +details on each iteration\n"
        "                                                    >50  +prints matrices (careful of memory usage!)\n"
        "                                                    >100 +prints partial vector results from each iteration\n"
        );
  args.AddOption(&output, "-out", "--outputsol",
                "Choose how much info to store on disk: 0  nothing\n"
        "                                               1 +#it to convergence\n"
        "                                               2 +residual evolution (default)\n"
        "                                               3 +paraview plot of exact (if available) and approximate solution (careful of memory usage!)\n"
        "                                               4 +operators and intermediate vector results at each Newton iteration (VERY careful of memory usage!)\n"
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

  // -initialise remaining parameters
  double dt  = Tend / numProcs;
  double mu  = 1.;
  double eta = 1.;
  double mu0 = 1.;

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
  std::string mesh_file;

  switch (pbType){
    // analytical test-case
    case 0:{
      mesh_file = "./meshes/tri-square-testAn.mesh";
      uFun = uFun_ex_an0;
      pFun = pFun_ex_an0;
      zFun = zFun_ex_an0;
      aFun = aFun_ex_an0;
      fFun = fFun_an0;
      gFun = gFun_an0;
      hFun = hFun_an0;
      nFun = nFun_an0;
      mFun = mFun_an0;
      wFun = wFun_an0;
      yFun = yFun_an0;
      cFun = cFun_an0;
      pbName = "An0";
      anSolution = true;
      break;
    }
    // analytical test-case
    case 1:{
      mesh_file = "./meshes/tri-square-testAn.mesh";
      uFun = uFun_ex_an1;
      pFun = pFun_ex_an1;
      zFun = zFun_ex_an1;
      aFun = aFun_ex_an1;
      fFun = fFun_an1;
      gFun = gFun_an1;
      hFun = hFun_an1;
      nFun = nFun_an1;
      mFun = mFun_an1;
      wFun = wFun_an1;
      qFun = qFun_an1;
      yFun = yFun_an1;
      cFun = cFun_an1;
      // Set BC:
      // - Dirichlet on u everywhere but on E
      // - Dirichlet on p on E (outflow, used only in precon)
      // - Dirichlet on A on W
      essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
      essTagsV = essTagsU;
      essTagsP.SetSize(1); essTagsP[0] = 2; // E
      essTagsA.SetSize(1); essTagsA[0] = 1;
      pbName = "An1";
      anSolution = true;
      break;
    }
    // analytical test-case
    case 2:{
      mesh_file = "./meshes/tri-square-testAn.mesh";
      uFun = uFun_ex_an2;
      pFun = pFun_ex_an2;
      zFun = zFun_ex_an2;
      aFun = aFun_ex_an2;
      fFun = fFun_an2;
      gFun = gFun_an2;
      hFun = hFun_an2;
      nFun = nFun_an2;
      mFun = mFun_an2;
      wFun = wFun_an2;
      qFun = qFun_an2;
      yFun = yFun_an2;
      cFun = cFun_an2;
      // Set BC:
      // - Dirichlet on u everywhere but on E
      // - Dirichlet on p on E (outflow, used only in precon)
      // - Dirichlet on A on W
      essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
      essTagsV = essTagsU;
      essTagsP.SetSize(1); essTagsP[0] = 2; // E
      essTagsA.SetSize(1); essTagsA[0] = 1;
      pbName = "An2";
      anSolution = true;
      break;
    }
    // analytical test-case
    case 3:{
      mesh_file = "./meshes/tri-square-testAn.mesh";
      uFun = uFun_ex_an3;
      pFun = pFun_ex_an3;
      zFun = zFun_ex_an3;
      aFun = aFun_ex_an3;
      fFun = fFun_an3;
      gFun = gFun_an3;
      hFun = hFun_an3;
      nFun = nFun_an3;
      mFun = mFun_an3;
      wFun = wFun_an3;
      qFun = qFun_an3;
      yFun = yFun_an3;
      cFun = cFun_an3;
      // Set BC:
      // - Dirichlet on u everywhere but on E
      // - Dirichlet on p on E (outflow, used only in precon)
      // - Dirichlet on A on W
      essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
      essTagsV = essTagsU;
      essTagsP.SetSize(1); essTagsP[0] = 2; // E
      essTagsA.SetSize(1); essTagsA[0] = 1;
      pbName = "An3";
      anSolution = true;
      break;
    }
    // analytical test-case
    case 4:{
      mesh_file = "./meshes/tri-square-testAn.mesh";
      uFun = uFun_ex_an4;
      pFun = pFun_ex_an4;
      zFun = zFun_ex_an4;
      aFun = aFun_ex_an4;
      fFun = fFun_an4;
      gFun = gFun_an4;
      hFun = hFun_an4;
      nFun = nFun_an4;
      mFun = mFun_an4;
      wFun = wFun_an4;
      qFun = qFun_an4;
      yFun = yFun_an4;
      cFun = cFun_an4;
      // Set BC:
      // - Dirichlet on u everywhere but on E
      // - Dirichlet on p on E (outflow, used only in precon)
      // - Dirichlet on A on W
      essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
      essTagsV = essTagsU;
      essTagsP.SetSize(1); essTagsP[0] = 2; // E
      essTagsA.SetSize(1); essTagsA[0] = 1;
      pbName = "An4";
      anSolution = true;
      break;
    }
    // Kelvin-Helmholtz instability
    case 5:{
      mesh_file = "./meshes/tri-rect-KHI.mesh";
      uFun = uFun_ex_KHI;
      pFun = pFun_ex_KHI;
      zFun = zFun_ex_KHI;
      aFun = aFun_ex_KHI;
      fFun = fFun_KHI;
      gFun = gFun_KHI;
      hFun = hFun_KHI;
      nFun = nFun_KHI;
      mFun = mFun_KHI;
      wFun = wFun_KHI;
      qFun = qFun_KHI;
      yFun = yFun_KHI;
      cFun = cFun_KHI;
      // Set BC:
      // Top+bottom=1 topLeft+botRight=2 botLeft+topRight=3
      // - Dirichlet on u,v on top left and bottom right
      // - Dirichlet on v   on top and bottom
      // - Dirichlet on v   on top right and bottom left
      // - No stress on normal component (u) on top right and bottom left
      // - No stress on tangential component (u) top and bottom
      // - Dirichlet on p on top right and bottom left (outflow, used only in precon)
      // - Dirichlet on A on top and bottom
      essTagsU.SetSize(1); essTagsU[0] = 2; //essTagsU[1] = 1;
      essTagsV.SetSize(3); essTagsV[0] = 1; essTagsV[1] = 2; essTagsV[2] = 3;
      essTagsP.SetSize(1); essTagsP[0] = 3;
      essTagsA.SetSize(1); essTagsA[0] = 1;
      // Set parameters: Re = Lu = ~ 1e3
      //  - normal derivatives m and n are 0, so no need to rescale those
      // mu  = 1.5 * 1e-3;
      // eta = 1e-3;

      pbName = "KHI";
      break;
    }
    // Island coalescence
    case 6:{
      mesh_file = "./meshes/tri-rect-island.mesh";
      uFun = uFun_ex_island;
      pFun = pFun_ex_island;
      zFun = zFun_ex_island;
      aFun = aFun_ex_island;
      fFun = fFun_island;
      gFun = gFun_island;
      hFun = hFun_island;
      nFun = nFun_island;
      mFun = mFun_island;
      wFun = wFun_island;
      qFun = qFun_island;
      yFun = yFun_island;
      cFun = cFun_island;
      // Set BC:
      // Top+bottom=1 Left+Right=2
      // - Dirichlet on u on left and right
      // - Dirichlet on v on top and bottom
      // - No stress on tangential component (u) top and bottom
      // - No stress on tangential component (v) left and right
      // - Dirichlet on p on top and bottom (outflow, used only in precon)
      // - Dirichlet on A on top and bottom
      essTagsU.SetSize(1); essTagsU[0] = 2;
      essTagsV.SetSize(1); essTagsV[0] = 1;
      essTagsP.SetSize(1); essTagsP[0] = 1;
      essTagsA.SetSize(1); essTagsA[0] = 1;
      pbName = "IslandCoalescence";
      mu  = 1e-4;
      eta = 1e-4;
      break;
    }
    // Modified Hartmann flow
    case 7:{
      mesh_file = "./meshes/tri-square-hartmann.mesh";
      uFun = uFun_ex_hartmann;
      pFun = pFun_ex_hartmann;
      zFun = zFun_ex_hartmann;
      aFun = aFun_ex_hartmann;
      fFun = fFun_hartmann;
      gFun = gFun_hartmann;
      hFun = hFun_hartmann;
      nFun = nFun_hartmann;
      mFun = mFun_hartmann;
      wFun = wFun_hartmann;
      qFun = qFun_hartmann;
      yFun = yFun_hartmann;
      cFun = cFun_hartmann;
      // Set BC:
      essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, W
      essTagsV = essTagsU;
      essTagsP.SetSize(1); essTagsP[0] = 2; // E
      essTagsA.SetSize(1); essTagsA[0] = 4; //essTagsA[1] = 3; // W
      pbName = "Hartmann";
      anSolution = true;
      break;
    }
    // Driven cavity flow
    case 11:{
      mesh_file = "./meshes/tri-square-cavity.mesh";
      uFun = uFun_ex_cavity;
      pFun = pFun_ex_cavity;
      zFun = zFun_ex_cavity;
      aFun = aFun_ex_cavity;
      fFun = fFun_cavity;
      gFun = gFun_cavity;
      hFun = hFun_cavity;
      nFun = nFun_cavity;
      mFun = mFun_cavity;
      wFun = wFun_cavity;
      qFun = qFun_cavity;
      yFun = yFun_cavity;
      cFun = cFun_cavity;
      // Set BC:
      essTagsU.SetSize(4); essTagsU[0] = 1; essTagsU[1] = 2; essTagsU[2] = 3; essTagsU[3] = 4;
      essTagsV = essTagsU;
      essTagsP.SetSize(0);
      essTagsA.SetSize(0);
      pbName = "Cavity";
      mu = 1;
      break;
    }
    // // Island coalescence - periodic in x
    // case 7:{
    //   mesh_file = "./meshes/tri-rect-island-xper.mesh";
    //   uFun = uFun_ex_island;
    //   pFun = pFun_ex_island;
    //   zFun = zFun_ex_island;
    //   aFun = aFun_ex_island;
    //   fFun = fFun_island;
    //   gFun = gFun_island;
    //   hFun = hFun_island;
    //   nFun = nFun_island;
    //   mFun = mFun_island;
    //   wFun = wFun_island;
    //   yFun = yFun_island;
    //   cFun = cFun_island;
    //   // Set BC:
    //   // TODO!!!
    //   essTagsU.SetSize(2); essTagsU[0] = 1; essTagsU[1] = 3;
    //   essTagsV = essTagsU;
    //   essTagsA.SetSize(2); essTagsA[0] = 1; essTagsA[1] = 3; // N S
    //   pbName = "IslandCoalescenceXPer";
    //   mu0 = 1e-4;
    //   eta = 1e-4;
    //   break;
    // }

    default:
      std::cerr<<"ERROR: Problem type "<<pbType<<" not recognised."<<std::endl;
  }



  if(myid == 0){
    args.PrintOptions(cout);
    std::cout<<"   --np "<<numProcs<<std::endl;
    std::cout<<"   --dt "<<Tend/numProcs<<std::endl;
    std::cout<<"   --Pb "<<pbName<<std::endl;
  }


  // - initialise petsc
  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);



  // ASSEMBLE OPERATORS -----------------------------------------------------
  IMHD2DSTOperatorAssembler *mhdAssembler = new IMHD2DSTOperatorAssembler( MPI_COMM_WORLD, mesh_file, ref_levels, ordU, ordP, ordZ, ordA,
                                                                           dt, mu, eta, mu0, fFun, gFun, hFun, nFun, mFun,
                                                                           wFun, qFun, yFun, cFun,
                                                                           uFun, pFun, zFun, aFun, 
                                                                           essTagsU, essTagsV, essTagsP, essTagsA, verbose );


  Operator *FFFu, *MMMz, *FFFa, *BBB, *BBBt, *ZZZ1, *ZZZ2, *KKK, *YYY;
  Vector   fres,   gres,  zres,  hres,  U0,  P0,  Z0,  A0;
  // Vector   *uEx, *pEx, *zEx, *aEx;

  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators for system ***************************\n";
  }

  // Assemble the system
  mhdAssembler->AssembleSystem( FFFu, MMMz, FFFa,
                                BBB,  ZZZ1, ZZZ2,
                                KKK,  YYY,
                                fres, gres, zres, hres,
                                U0,   P0,   Z0,   A0  );

  BBBt = new TransposeOperator( BBB );

  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = FFFu->NumRows();
  offsets[2] =  BBB->NumRows();
  offsets[3] = MMMz->NumRows();
  offsets[4] = FFFa->NumRows();
  offsets.PartialSum();

  // - assemble matrix
  BlockOperator *MHDOp = new BlockOperator( offsets );
  MHDOp->SetBlock(0, 0, FFFu);
  MHDOp->SetBlock(2, 2, MMMz);
  MHDOp->SetBlock(3, 3, FFFa);
  MHDOp->SetBlock(0, 1, BBBt);
  MHDOp->SetBlock(0, 2, ZZZ1);
  MHDOp->SetBlock(0, 3, ZZZ2);
  MHDOp->SetBlock(1, 0,  BBB);
  MHDOp->SetBlock(2, 3,  KKK);
  MHDOp->SetBlock(3, 0,  YYY);


  // - assemble the original residual
  BlockVector res(offsets);
  res.GetBlock(0) = fres;
  res.GetBlock(1) = gres;
  res.GetBlock(2) = zres;
  res.GetBlock(3) = hres;

  // - assemble solution (IG)
  BlockVector sol(offsets);
  sol.GetBlock(0) = U0;
  sol.GetBlock(1) = P0;
  sol.GetBlock(2) = Z0;
  sol.GetBlock(3) = A0;


  if( myid == 0 && verbose > 0 ){
    std::cout << "Assembling operators for preconditioner *******************\n";
  }

  Operator *FFui, *MMzi, *pSi, *aSi;
  mhdAssembler->AssemblePreconditioner( FFui, MMzi, pSi, aSi, STSolveTypeU, STSolveTypeA );

  // Define preconditioner
  Solver *MHDPr;

  // - assemble single factors
  BlockUpperTriangularPreconditioner *Uub = new BlockUpperTriangularPreconditioner( offsets ),
                                     *Uup = new BlockUpperTriangularPreconditioner( offsets );
  BlockLowerTriangularPreconditioner *Lub = new BlockLowerTriangularPreconditioner( offsets ),
                                     *Lup = new BlockLowerTriangularPreconditioner( offsets );
  assembleLub( YYY, FFui, ZZZ1, MMzi, Lub );
  assembleUub( ZZZ1,  ZZZ2, MMzi, KKK, aSi, Uub );
  assembleLup( FFui, BBB,  Lup );
  assembleUup( FFui, BBBt, pSi, Uup );


  switch(precType){
    case 0:{
      // Uup^-1 Lup^-1 Uub^-1 Lub^-1
      Array<const Operator*> precOps(4);
      Array<bool>      precOwn(4);
      precOps[0] = Lub;  precOwn[0] = true;
      precOps[1] = Uub;  precOwn[1] = true;
      precOps[2] = Lup;  precOwn[2] = true;
      precOps[3] = Uup;  precOwn[3] = true;

      OperatorsSequence* temp = new OperatorsSequence( precOps, precOwn );

      // -and store
      MHDPr = temp;

      break;
    }

    case 1:{
      // Uup^-1 Lup^-1 Uub^-1
      Array<const Operator*> precOps(3);
      Array<bool>            precOwn(3);
      precOps[0] = Uub;  precOwn[0] = true;
      precOps[1] = Lup;  precOwn[1] = true;
      precOps[2] = Uup;  precOwn[2] = true;

      OperatorsSequence* temp = new OperatorsSequence( precOps, precOwn );

      // --and store
      MHDPr = temp;

      break;
    }

    case 2:{
      // Uup^-1 Uub^-1
      Array<const Operator*> precOps(2);
      Array<bool>            precOwn(2);
      precOps[0] = Uub;  precOwn[0] = true;
      precOps[1] = Uup;  precOwn[1] = true;

      OperatorsSequence* temp = new OperatorsSequence( precOps, precOwn );

      // --and store
      MHDPr = temp;

      break;
    }
    default:{
      if ( myid == 0 ){
        std::cerr<<"ERROR: Option for preconditioner "<<precType<<" not recognised"<<std::endl;
      }
      break;
    }
  }


  // Initialise folder where to store convergence results
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




  // // FOR TESTING PURPOSES ---------------------------------------------------
  // // - Print out matrices for matlab import
  // string path = string("./results/operators/") + "Pb" + to_string(pbType)
  //                        + "_Prec" + to_string(precType) + "_STsolveU" + to_string(STSolveTypeU) + "_STsolveA" + to_string(STSolveTypeA)
  //                        + "_oU"   + to_string(ordU) + "_oP" + to_string(ordP) + "_oZ" + to_string(ordZ) + "_oA" + to_string(ordA)
  //                        + "_"     + petscrc_file + "/" + "NP" + to_string(numProcs) + "_r"  + to_string(ref_levels)
  //                        + "/";
  // if (!std::experimental::filesystem::exists( path ) && myid == 0){
  //   std::experimental::filesystem::create_directories( path );
  // }
  // MPI_Barrier(MPI_COMM_WORLD);  // to wait for master to have created folder?
  // path += "Nit" + to_string(newtonIt) + "_";
  // mhdAssembler->PrintMatrices( path );


  // // - Print out rhs (that's the vector which will be used for testing)
  // ofstream myfile;
  // myfile.precision(std::numeric_limits< double >::max_digits10);
  // myfile.open( path+"rhs"+to_string(myid)+".dat" );
  // res.Print(myfile,1);
  // myfile.close();




  // SOLVE SYSTEM -----------------------------------------------------------
  if( myid == 0 && verbose > 0 ){
    std::cout << "SOLVE! ****************************************************\n";
  }



  // In this case, just solve the system normally, via time-stepping
  if ( STSolveTypeU == 9 || STSolveTypeA == 9 ){
    if( myid == 0 && verbose > 0 ){
      std::cout << "USING CLASSIC TIME-STEPPING *******************************\n";
    }

    // - solve via time-stepping
    mhdAssembler->TimeStep( res, sol, innerConvpath, output );




  // otherwise, things get serious
  }else{


    // - compute norm of residual at zeroth-iteration
    double newtonRes = res.Norml2();    // it's a bit annoying that HyperParVector doesn't overwrite the norml2 function...
    newtonRes*= newtonRes;
    MPI_Allreduce( MPI_IN_PLACE, &newtonRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    newtonRes  = sqrt(newtonRes);
    double newtonRes0 = newtonRes;
    double newtonErrWRTPrevIt = newtonTol;
    int newtonIt = 0;
    int GMRESits = 0;
    double totGMRESit = 0.; //leave it as double, so that when I'll average it, it won't round-off

    if( myid == 0 ){
      std::cout << "***********************************************************\n";
      std::cout << "Newton iteration "<<newtonIt<<", residual "<< newtonRes <<std::endl;
      std::cout << "***********************************************************\n";
      if ( output>0 ){
        string filename = innerConvpath +"NEWTconv.txt";
        ofstream myfile;
        myfile.open( filename, std::ios::app );
        myfile << "#It"    <<"\t"<< "Res norm"<<"\t"<< "Rel res norm"       <<"\t"<< "Norm of update\tInner converged\tInner res\tInner its"  <<std::endl;
        myfile << newtonIt <<"\t"<< newtonRes <<"\t"<< newtonRes/newtonRes0 <<"\t"<< 0.0         <<"\t"<< false   <<"\t"<<0.0<<"\t"<<GMRESits <<std::endl;
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



    // - stop if:       max it reached                 residual small enough        relative residual small enough       //     difference wrt prev it small enough
    bool stopNewton = ( newtonIt >= maxNewtonIt ) || ( newtonRes < newtonTol ) || ( newtonRes/newtonRes0 < newtonTol );  // || ( newtonErrWRTPrevIt < newtonTol );



    while(!stopNewton){
    
      // Define inner solver
      PetscLinearSolver solver(MPI_COMM_WORLD, "solver_");
      bool isIterative = true;
      solver.iterative_mode = isIterative;

      // - register operator and preconditioner with the solver
      solver.SetPreconditioner(*MHDPr);
      solver.SetOperator(*MHDOp);
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
  
        if( anSolution ){
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
          // KSPMonitorSet( ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorDefault,   vf, (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy );
        }
      }




      // Define initial guess for update to solution (all zero)
      BlockVector deltaSol(offsets);
      deltaSol.GetBlock(0) = 0.;
      deltaSol.GetBlock(1) = 0.;
      deltaSol.GetBlock(2) = 0.;
      deltaSol.GetBlock(3) = 0.;

      // Solve for current linearisation
      solver.Mult( res, deltaSol );

      if( output > 3 ){
        string filename = innerOperPath + "Nit" + to_string(newtonIt) + "_";
        // - Print out solution
        ofstream myfile;
        myfile.precision(std::numeric_limits< double >::max_digits10);
        myfile.open( filename+"deltaSol"+to_string(myid)+".dat" );
        deltaSol.Print(myfile,1);
        myfile.close();
      }


      // Update relevant quantities
      // - solution
      sol += deltaSol;
      // - residual
      // -- apply operator
      mhdAssembler->ApplyOperator( sol, res );
      // -- compute residual norm
      newtonRes = res.Norml2();
      newtonRes*= newtonRes;
      MPI_Allreduce( MPI_IN_PLACE, &newtonRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      newtonRes = sqrt(newtonRes);
      // -- compute norm of newton update
      newtonErrWRTPrevIt = deltaSol.Norml2();
      newtonErrWRTPrevIt*= newtonErrWRTPrevIt;
      MPI_Allreduce( MPI_IN_PLACE, &newtonErrWRTPrevIt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      newtonErrWRTPrevIt = sqrt(newtonErrWRTPrevIt);
      
      // Output relevant measurements
      newtonIt++;
      GMRESits  = solver.GetNumIterations();
      totGMRESit += GMRESits;
      if( myid == 0 ){
        if (solver.GetConverged()){
          std::cout << "Inner solver converged in ";
        }else{
          std::cout << "Inner solver *DID NOT* converge in ";
        }
        std::cout<< GMRESits << " iterations. Residual "<<solver.GetFinalNorm()<<std::endl;
        if( output>0 ){
          string filename = innerConvpath +"NEWTconv.txt";
          ofstream myfile;
          myfile.open( filename, std::ios::app );
          myfile << newtonIt << "\t" << newtonRes << "\t" << newtonRes/newtonRes0 << "\t" << newtonErrWRTPrevIt << "\t"
                 << solver.GetConverged() << "\t" << solver.GetFinalNorm() << "\t" << GMRESits<<std::endl;
          myfile.close();
        }      
        std::cout << "***********************************************************\n";
        std::cout << "Newton iteration "<<newtonIt<<", residual "<< newtonRes <<std::endl;
        std::cout << "***********************************************************\n";
      }

      // Check stopping criterion
      stopNewton = ( newtonIt >= maxNewtonIt ) || ( newtonRes < newtonTol ) || ( newtonRes/newtonRes0 < newtonTol ) || ( newtonErrWRTPrevIt < 1e-12 );
      // - and eventually update the operators
      if( !stopNewton ){
        if( myid == 0 && verbose > 0 ){
          std::cout << "Update operators ******************************************\n";
        }
        mhdAssembler->UpdateLinearisedOperators( sol );
    
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
      std::cout   << ", avg internal GMRES it are "               << totGMRESit/newtonIt  << ".\n";
      std::cout   << "***********************************************************\n";

      // - eventually store info on newton convergence
      if( output>0 ){
        string filename = convPath + "Newton_convergence_results_NP" + to_string(numProcs) + "_r"  + to_string(ref_levels) + ".txt";
        ofstream myfile;
        myfile.open( filename, std::ios::app );
        myfile << Tend     << ",\t" << dt         << ",\t" << numProcs   << ",\t" << ref_levels << ",\t"
               << newtonIt << ",\t" << totGMRESit/newtonIt << ",\t" << newtonRes << std::endl;
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

    if ( anSolution ){
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







// Assembles lower factor of LU factorisation of velocity/magnetic field part of preconditioner
//       ⌈    I                          ⌉
// Lub = |         I                     |
//       |                   I           |
//       ⌊ Y*Fu^-1   -Y*Fu^-1*Z1*Mz^-1 I ⌋
void assembleLub( const Operator* Y,  const Operator* Fui,
                  const Operator* Z1, const Operator* Mzi,
                  BlockLowerTriangularPreconditioner* Lub ){

  // std::cout<<"Size of operators: "<<std::endl
  //          <<"Fui  "<< Fui->NumRows() <<","<< Fui->NumCols() <<std::endl

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
//     ⌈ Fu^-1       ⌉   ⌈ I   Z1 Z2 ⌉
// Uub*|       I     | = |   I       |
//     |         I   |   |     Mz K  |
//     ⌊           I ⌋   ⌊        aS ⌋
void assembleUub( const Operator* Z1, const Operator* Z2, const Operator* Mzi, const Operator* K,
                  const Operator* aSi, BlockUpperTriangularPreconditioner* Uub ){
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
void assembleLup( const Operator* Fui, const Operator* B, BlockLowerTriangularPreconditioner* Lup ){

  Array<const Operator*> BFuiOps(2);
  BFuiOps[0] = Fui;
  BFuiOps[1] = B;
  OperatorsSequence* BFui = new OperatorsSequence( BFuiOps );   // does not own
  
  Lup->iterative_mode = false;
  Lup->SetBlock( 1, 0, BFui );
  Lup->owns_blocks = true;
}



// Assembles upper factor of LU factorisation of velocity/pressure part of preconditioner
//       ⌈ Fu Bt     ⌉
// Uup = |    pS     |
//       |       I   |
//       ⌊         I ⌋
void assembleUup( const Operator* Fui, const Operator* Bt, const Operator* pSi, BlockUpperTriangularPreconditioner* Uup ){
  Uup->iterative_mode = false;
  Uup->SetBlock( 0, 0, Fui );
  Uup->SetBlock( 0, 1, Bt  );
  Uup->SetBlock( 1, 1, pSi );
  Uup->owns_blocks = false;
}










//***************************************************************************
// ANALYTICAL TEST CASES
//***************************************************************************
// - define a perturbation to dirty initial guess
double perturbation(const Vector & x, const double t){
  double epsilon = 1.;
  double xx(x(0));
  double yy(x(1));
  return( t * epsilon * 0.25*( ( cos(2*M_PI*xx)-1 )*(cos(2*M_PI*yy)-1) ) );
}
// All-zero solution --------------------------------------------------------
// - null velocity
void uFun_ex_an0(const Vector & x, const double t, Vector & u){
  u = 0.;
}
// - null pressure
double pFun_ex_an0(const Vector & x, const double t ){
  return 0.;
}
// - null laplacian of vector potential
double zFun_ex_an0(const Vector & x, const double t ){
  return 0.;
}
// - null vector potential
double aFun_ex_an0(const Vector & x, const double t ){
  return 0.;
}
// - null rhs of velocity
void fFun_an0(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// - null normal derivative of velocity
void nFun_an0(const Vector & x, const double t, Vector & n){
  n = 0.;
}
// - null rhs of pressure
double gFun_an0(const Vector & x, const double t ){
  return 0.;
}
// - null rhs of vector potential
double hFun_an0( const Vector & x, const double t ){
  return 0.;
}
// - null normal derivative of vector potential
double mFun_an0( const Vector & x, const double t ){
  return 0.;
}
// - define null IG for every linearised variable
void wFun_an0(const Vector & x, const double t, Vector & w){
  w = 0.;
}
double qFun_an0(  const Vector & x, const double t ){
  return 0.;
}
double yFun_an0(  const Vector & x, const double t ){
  return 0.;
}
double cFun_an0(  const Vector & x, const double t ){
  return 0.;
}




// Only velocity ------------------------------------------------------------
// - Pick a div-free field
void uFun_ex_an1(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   t * xx*xx*yy;
  u(1) = - t * xx*yy*yy;
}
// - null pressure
double pFun_ex_an1(const Vector & x, const double t ){
  return 0.;
}
// - null laplacian of vector potential
double zFun_ex_an1(const Vector & x, const double t ){
  return 0.;
}
// - null vector potential
double aFun_ex_an1(const Vector & x, const double t ){
  return 0.;
}
// - rhs of velocity counteracts action of every term
void fFun_an1(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)            - mu Lap(u)
  f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - t * 2*yy;
  f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + t * 2*xx;
}
// - normal derivative of velocity
void nFun_an1(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n = 0.;
  // if ( yy == 0. ){ //S
  //   n(0) = -xx*xx;
  //   n(1) = 0.;
  // }
  // if ( xx == 0. ){ //W
  //   n(0) = 0.;
  //   n(1) = yy*yy;
  // }
  // if ( yy == 1. ){ //N
  //   n(0) =  xx*xx;
  //   n(1) = -2*xx;
  // }
  if ( xx == 1. ){ //E
    n(0) = 2*yy;
    n(1) = - yy*yy;
  }

  n *= t;
}
// - null rhs of pressure
double gFun_an1(const Vector & x, const double t ){
  return 0.;
}
// - null rhs of vector potential
double hFun_an1( const Vector & x, const double t ){
  return 0.;
}
// - null normal derivative of vector potential
double mFun_an1( const Vector & x, const double t ){
  return 0.;
}
// - define perturbed IG for every linearised variable
void wFun_an1(const Vector & x, const double t, Vector & w){
  uFun_ex_an1(x,t,w);
  double ds = perturbation(x,t);
  w(0) = w(0) + ds;
  w(1) = w(1) - ds;
}
double qFun_an1(  const Vector & x, const double t ){
  // double ds = perturbation(x,t);
  // return ( zFun_ex_an1(x,t) + ds );
  return 0.;
}
double yFun_an1(  const Vector & x, const double t ){
  // double ds = perturbation(x,t);
  // return ( zFun_ex_an1(x,t) + ds );
  return 0.;
}
double cFun_an1(  const Vector & x, const double t ){
  // double ds = perturbation(x,t);
  // return ( aFun_ex_an1(x,t) + ds );
  return 0.;
}





// Velocity and pressure -----------------------------------------------
// - Pick a div-free field
void uFun_ex_an2(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   t * xx*xx*yy;
  u(1) = - t * xx*yy*yy;
}
// - Pick a pressure which is null on boundaries
double pFun_ex_an2(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * sin(M_PI*xx)*sin(M_PI*yy) );
}
// - null laplacian of vector potential
double zFun_ex_an2(const Vector & x, const double t ){
  return 0.;
}
// - null vector potential
double aFun_ex_an2(const Vector & x, const double t ){
  return 0.;
}
// - rhs of velocity counteracts action of every term
void fFun_an2(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)            - mu Lap(u) + Grad(p)
  f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - t * 2*yy  + t * M_PI*cos(M_PI*xx)*sin(M_PI*yy);
  f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + t * 2*xx  + t * M_PI*sin(M_PI*xx)*cos(M_PI*yy);
}
// - normal derivative of velocity
void nFun_an2(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n = 0.;
  // if ( yy == 0. ){ //S
  //   n(0) = -xx*xx;
  //   n(1) = 0.;
  // }
  // if ( xx == 0. ){ //W
  //   n(0) = 0.;
  //   n(1) = yy*yy;
  // }
  // if ( yy == 1. ){ //N
  //   n(0) =  xx*xx;
  //   n(1) = -2*xx;
  // }
  if ( xx == 1. ){ //E
    n(0) = 2*yy;
    n(1) = - yy*yy;
  }

  n *= t;
}
// - null rhs of pressure
double gFun_an2(const Vector & x, const double t ){
  return 0.;
}
// - null rhs of vector potential
double hFun_an2( const Vector & x, const double t ){
  return 0.;
}
// - null normal derivative of vector potential
double mFun_an2( const Vector & x, const double t ){
  return 0.;
}
// - define perturbed IG for every linearised variable
void wFun_an2(const Vector & x, const double t, Vector & w){
  uFun_ex_an2(x,t,w);
  double ds = perturbation(x,t);
  w(0) = w(0) + ds;
  w(1) = w(1) - ds;
}
double qFun_an2(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return ( pFun_ex_an2(x,t) + ds );
}
double yFun_an2(  const Vector & x, const double t ){
  // double ds = perturbation(x,t);
  // return ( zFun_ex_an1(x,t) + ds );
  return 0.;
}
double cFun_an2(  const Vector & x, const double t ){
  // double ds = perturbation(x,t);
  // return ( aFun_ex_an1(x,t) + ds );
  return 0.;
}



// Velocity, pressure and Vector potential ----------------------------------
// - Pick a div-free field
void uFun_ex_an3(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   t * xx*xx*yy;
  u(1) = - t * xx*yy*yy;
}
// - Pick a pressure which is null on boundaries
double pFun_ex_an3(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * sin(M_PI*xx)*sin(M_PI*yy) );
}
// - null laplacian of vector potential
double zFun_ex_an3(const Vector & x, const double t ){
  return 0.;
}
// - constant (in space) vector potential
double aFun_ex_an3(const Vector & x, const double t ){
  return t;
}
// - rhs of velocity counteracts action of every term
void fFun_an3(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)            - mu Lap(u) + Grad(p)
  f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - t * 2*yy  + t * M_PI*cos(M_PI*xx)*sin(M_PI*yy);
  f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + t * 2*xx  + t * M_PI*sin(M_PI*xx)*cos(M_PI*yy);
}
// - normal derivative of velocity
void nFun_an3(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n = 0.;
  // if ( yy == 0. ){ //S
  //   n(0) = -xx*xx;
  //   n(1) = 0.;
  // }
  // if ( xx == 0. ){ //W
  //   n(0) = 0.;
  //   n(1) = yy*yy;
  // }
  // if ( yy == 1. ){ //N
  //   n(0) =  xx*xx;
  //   n(1) = -2*xx;
  // }
  if ( xx == 1. ){ //E
    n(0) = 2*yy;
    n(1) = - yy*yy;
  }

  n *= t;
}
// - null rhs of pressure
double gFun_an3(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts time-derivative
double hFun_an3( const Vector & x, const double t ){
  return 1.;
}
// - null normal derivative of vector potential
double mFun_an3( const Vector & x, const double t ){
  return 0.;
}
// - define perturbed IG for every linearised variable
void wFun_an3(const Vector & x, const double t, Vector & w){
  uFun_ex_an3(x,t,w);
  double ds = perturbation(x,t);
  w(0) = w(0) + ds;
  w(1) = w(1) - ds;
}
double qFun_an3(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return ( pFun_ex_an3(x,t) + ds );
}
double yFun_an3(  const Vector & x, const double t ){
  // double ds = perturbation(x,t);
  // return ( zFun_ex_an1(x,t) + ds );
  return 0.;
}
double cFun_an3(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return ( aFun_ex_an3(x,t) + ds );
}




// Velocity, pressure, vector potential and laplacian of vector potential ---
// - Pick a div-free field
void uFun_ex_an4(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   t * xx*xx*yy;
  u(1) = - t * xx*yy*yy;
}
// - Pick a pressure which is null on boundaries
double pFun_ex_an4(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * sin(M_PI*xx)*sin(M_PI*yy) );
}
// - laplacian of vector potential
double zFun_ex_an4(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( -t * 2*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) );
}
// - vector potential with null normal derivative
double aFun_ex_an4(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * cos(M_PI*xx)*cos(M_PI*yy) );
}
// - rhs of velocity counteracts action of every term
void fFun_an4(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)            - mu Lap(u) + Grad(p)                            + z grad(A) / mu0
  f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - t * 2*yy  + t * M_PI*cos(M_PI*xx)*sin(M_PI*yy) + zFun_ex_an4(x,t) * (-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy));
  f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + t * 2*xx  + t * M_PI*sin(M_PI*xx)*cos(M_PI*yy) + zFun_ex_an4(x,t) * (-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy));
}
// - normal derivative of velocity
void nFun_an4(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n = 0.;
  // if ( yy == 0. ){ //S
  //   n(0) = -xx*xx;
  //   n(1) = 0.;
  // }
  // if ( xx == 0. ){ //W
  //   n(0) = 0.;
  //   n(1) = yy*yy;
  // }
  // if ( yy == 1. ){ //N
  //   n(0) =  xx*xx;
  //   n(1) = -2*xx;
  // }
  if ( xx == 1. ){ //E
    n(0) = 2*yy;
    n(1) = - yy*yy;
  }

  n *= t;
}
// - null rhs of pressure
double gFun_an4(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts every term
double hFun_an4( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  Vector u(2);
  uFun_ex_an4(x,t,u);
  double ugradA = u(0)*(-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy))
                + u(1)*(-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy));
  //       dtA                       + u Grad(A) - eta/mu0 lap(A)
  return ( cos(M_PI*xx)*cos(M_PI*yy) + ugradA    - zFun_ex_an4(x,t) );
}
// - null normal derivative of vector potential
double mFun_an4( const Vector & x, const double t ){
  return 0.;
}
// - define perturbed IG for every linearised variable
void wFun_an4(const Vector & x, const double t, Vector & w){
  uFun_ex_an4(x,t,w);
  double ds = perturbation(x,t);
  w(0) = w(0) + ds;
  w(1) = w(1) - ds;
}
double qFun_an4(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return ( pFun_ex_an4(x,t) + ds );
}
double yFun_an4(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return ( zFun_ex_an4(x,t) + ds );
}
double cFun_an4(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return ( aFun_ex_an4(x,t) + ds );
}








//***************************************************************************
//TEST CASES OF SOME ACTUAL RELEVANCE
//***************************************************************************
// Kelvin-Helmholtz instability
namespace KHIData{
  const double delta = 0.07957747154595;
};

// - top velocity of 1.5, bottom velocity of -1.5
void uFun_ex_KHI(const Vector & x, const double t, Vector & u){
  u(0) = 0.;
  u(1) = 0.;

  if ( x(1) >= 0.5 ){
    u(0) =  1.5;
  }else{
    u(0) = -1.5;
  }
}
// - pressure - unused
double pFun_ex_KHI(const Vector & x, const double t ){
  return 0.;
}
// - laplacian of vector potential - unused
double zFun_ex_KHI(const Vector & x, const double t ){
  const double delta = KHIData::delta;
  double yy(x(1));
  return ( 1./ ( delta * cosh( yy/delta ) * cosh( yy/delta ) ) );
}
// - vector potential
double aFun_ex_KHI(const Vector & x, const double t ){
  const double delta = KHIData::delta;
  double yy(x(1));
  return ( delta * log( cosh( yy/delta ) ) );
}
// - rhs of velocity - unused
void fFun_KHI(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// - normal derivative of velocity
void nFun_KHI(const Vector & x, const double t, Vector & n){
  n = 0.;
}
// - rhs of pressure - unused
double gFun_KHI(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential - unused
double hFun_KHI( const Vector & x, const double t ){
  return 0.;
}
// - normal derivative of vector potential
double mFun_KHI( const Vector & x, const double t ){
  double yy(x(1));
  const double delta = KHIData::delta;

  if( yy ==  1.0 ){
    return   sinh(yy/delta) / cosh(yy/delta);
  }else if( yy ==  0.0 ){
    return - sinh(yy/delta) / cosh(yy/delta);
  }

  return 0.;
}
// - define IG for every linearised variable -> set them to initial conditions
void wFun_KHI(const Vector & x, const double t, Vector & w){
  uFun_ex_KHI(x,t,w);
}
double qFun_KHI(  const Vector & x, const double t ){
  return pFun_ex_KHI(x,t);
}
double yFun_KHI(  const Vector & x, const double t ){
  return zFun_ex_KHI(x,t);
}
double cFun_KHI(  const Vector & x, const double t ){
  return aFun_ex_KHI(x,t);
}





// Island coalescence
namespace IslandCoalescenceData{
  const double delta = 1./(2.*M_PI);
  const double P0 = 1.;
  const double epsilon = 0.4;
};

void uFun_ex_island(const Vector & x, const double t, Vector & u){
  u = 0.;
}
// - pressure
double pFun_ex_island(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  const double delta   = IslandCoalescenceData::delta;
  const double P0      = IslandCoalescenceData::P0;
  const double epsilon = IslandCoalescenceData::epsilon;

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( P0 + ( 1. - epsilon*epsilon ) / ( 2.*temp*temp ) );
}
// - laplacian of vector potential
double zFun_ex_island(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  const double delta   = IslandCoalescenceData::delta;
  const double epsilon = IslandCoalescenceData::epsilon;

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( ( 1. - epsilon*epsilon ) / ( delta * temp*temp ) );
}
// - vector potential
double aFun_ex_island(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  const double delta   = IslandCoalescenceData::delta;
  const double epsilon = IslandCoalescenceData::epsilon;

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( delta * log( temp ) );
}
// - rhs of velocity - unused
void fFun_island(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// - normal derivative of velocity
void nFun_island(const Vector & x, const double t, Vector & n){
  n = 0.;
}
// - rhs of pressure - unused
double gFun_island(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential - unused
double hFun_island( const Vector & x, const double t ){
  return 0.;
}
// - normal derivative of vector potential
double mFun_island( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  const double delta   = IslandCoalescenceData::delta;
  const double epsilon = IslandCoalescenceData::epsilon;

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);  

  if(xx == 1.0){
    return - epsilon*sin(xx/delta) / temp;
  }else if( xx == 0.0 ){
    return   epsilon*sin(xx/delta) / temp;
  }else if( yy ==  1.0 ){
    return   sinh(yy/delta) / temp;
  }else if( yy == -1.0 ){
    return - sinh(yy/delta) / temp;
  }

  return 0.;
}
// - define perturbed IG for every linearised variable
void wFun_island(const Vector & x, const double t, Vector & w){
  uFun_ex_island(x,t,w);
}
double qFun_island(  const Vector & x, const double t ){
  return pFun_ex_island(x,t);
}
double yFun_island(  const Vector & x, const double t ){
  return zFun_ex_island(x,t);
}
double cFun_island(  const Vector & x, const double t ){
  return aFun_ex_island(x,t);
}





// Hartmann flow
namespace HartmannData{
  const double G0  = 1.0;
  const double B0  = 1.0;
};
// - velocity
void uFun_ex_hartmann(const Vector & x, const double t, Vector & u){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  u = 0.;

  u(0) = G0/B0 * ( cosh(1.) - cosh(x(1)) )/sinh(1.);
}
// - pressure
double pFun_ex_hartmann(const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  return -G0*( x(0) + 1.0 );    // zero at outflow (x=-1)
}
// - laplacian of vector potential - unused
double zFun_ex_hartmann(const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  return -G0/B0 * ( 1. - cosh(x(1))/sinh(1.) );
}
// - vector potential
double aFun_ex_hartmann(const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  return -B0*x(0) - G0/B0 * ( x(1)*x(1)/2. - cosh(x(1))/sinh(1.) );
}
// - rhs of velocity - unused
void fFun_hartmann(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// - normal derivative of velocity
void nFun_hartmann(const Vector & x, const double t, Vector & n){
  const double yy(x(1));
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  n = 0.;

  if ( yy== 1. ){
    n(1) = -G0/B0 * sinh(yy)/sinh(1.);
  }
  if ( yy==-1. ){
    n(1) =  G0/B0 * sinh(yy)/sinh(1.);
  }
}
// - rhs of pressure - unused
double gFun_hartmann(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential
double hFun_hartmann( const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  return - G0/B0 * ( cosh(1.)/sinh(1.) - 1. );
}
// - normal derivative of vector potential
double mFun_hartmann( const Vector & x, const double t ){
  const double xx(x(0));
  const double yy(x(1));
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  if ( yy==1. || yy == -1. ){
    return yy*( -G0/B0 * ( yy - sinh(yy)/sinh(1.) ) );
  }
  if ( xx==1. || xx == -1. ){
    return -B0*xx;
  }
  return 0.;
}
// - define perturbed IG for every linearised variable
void wFun_hartmann(const Vector & x, const double t, Vector & w){
  uFun_ex_hartmann(x,t,w);
  double ds = perturbation(x,t);
  w(0) = w(0) + ds;
  w(1) = w(1) - ds;

}
double qFun_hartmann(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return pFun_ex_hartmann(x,t) + ds;
}
double yFun_hartmann(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return zFun_ex_hartmann(x,t) + ds;
}
double cFun_hartmann(  const Vector & x, const double t ){
  double ds = perturbation(x,t);
  return aFun_ex_hartmann(x,t) + ds;
}

















// Driven cavity flow (speed ramping up)-------------------------------------
// Exact solution (velocity) - only for BC
void uFun_ex_cavity(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u = 0.;

  if( yy==1.0 ){
    u(0) = t * 8*xx*(1-xx)*(2.*xx*xx-2.*xx+1.);   // regularised (1-x^4 mapped from -1,1 to 0,1)
    // u(0) = t;                      // leaky
    // if( xx > 1.0 && xx < 1.0 )
    //   u(0) = t;                    // watertight
  }
}
// Exact solution (pressure) - unused
double pFun_ex_cavity(const Vector & x, const double t ){
  return 0.;
}
// - laplacian of vector potential - unused
double zFun_ex_cavity(const Vector & x, const double t ){
  return 0.;
}
// - vector potential
double aFun_ex_cavity(const Vector & x, const double t ){
  return 0.;
}
// Rhs (velocity) - no forcing
void fFun_cavity(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// Normal derivative of velocity - unused
void nFun_cavity(const Vector & x, const double t, Vector & n){
  n = 0.;
}
// Rhs (pressure) - unused
double gFun_cavity(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential
double hFun_cavity( const Vector & x, const double t ){
  return 0.;
}
// - normal derivative of vector potential
double mFun_cavity( const Vector & x, const double t ){
  return 0.;
}
// - define IG for every linearised variable
void wFun_cavity(const Vector & x, const double t, Vector & w){
  uFun_ex_cavity(x,t,w);
}
double qFun_cavity(  const Vector & x, const double t ){
  return pFun_ex_cavity(x,t);
}
double yFun_cavity(  const Vector & x, const double t ){
  return zFun_ex_cavity(x,t);
}
double cFun_cavity(  const Vector & x, const double t ){
  return aFun_ex_cavity(x,t);
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

// Normal derivative of velocity - unused
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

// Normal derivative of velocity - used only at outflow
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

// Normal derivative of velocity - used only at outflow
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

// Normal derivative of velocity - used only at outflow
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

// Normal derivative of velocity - used only at outflow
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

// Normal derivative of velocity - unused
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




