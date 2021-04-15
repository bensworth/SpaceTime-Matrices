// Test file to check correctness of implementation of the MHD integrator
//  This is done by comparing my own implementation of the BlockNonlinearFormIntegrator
//  VS the results I would get by using classic MFEM integrators: the results should be
//  the same (up to machine precision)
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "blockUpperTriangularPreconditioner.hpp"
#include "operatorssequence.hpp"
#include "operatorsseries.hpp"
#include "imhd2dintegrator.hpp"
#include "imhd2dstmagneticschurcomplement.hpp"
#include "oseenstpressureschurcomplement.hpp"
#include "vectorconvectionintegrator.hpp"
#include "boundaryfacefluxintegrator.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <limits>
#include <cmath>

using namespace std;
using namespace mfem;

// Kelvin-Helmholtz instability
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
namespace IslandCoalescenceData{
  const double delta = 1./(2.*M_PI);
  const double P0 = 1.;
  const double epsilon = 0.2;
  const double mu  = 1e-2;
  const double eta = 1e-2;
  const double mu0 = 1.;
};
// Rayleigh flow
void   uFun_ex_rayleigh( const Vector & x, const double t, Vector & u );
double pFun_ex_rayleigh( const Vector & x, const double t             );
double zFun_ex_rayleigh( const Vector & x, const double t             );
double aFun_ex_rayleigh( const Vector & x, const double t             );
void   fFun_rayleigh(    const Vector & x, const double t, Vector & f );
void   nFun_rayleigh(    const Vector & x, const double t, Vector & f );
double gFun_rayleigh(    const Vector & x, const double t             );
double hFun_rayleigh(    const Vector & x, const double t             );
double mFun_rayleigh(    const Vector & x, const double t             );
void   wFun_rayleigh(    const Vector & x, const double t, Vector & w );
double qFun_rayleigh(    const Vector & x, const double t             );
double cFun_rayleigh(    const Vector & x, const double t             );
double yFun_rayleigh(    const Vector & x, const double t             );
namespace RayleighData{
  const double U   = 1.;
  const double B0  = 1.4494e-4;
  const double rho = 0.4e-4;
  const double mu0 = 1.256636e-6;
  const double eta = 1.256636e-6;
  const double mu  = 0.4e-4;
  const double d   = 1.;
  const double A0  = B0/sqrt(mu0*rho);
};
//---------------------------------------------------------------------------
// Handy functions for assembling various operators
void AssembleAp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                 const IntegrationRule *ir, SparseMatrix& _Ap );
void AssembleMp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                 const IntegrationRule *ir, SparseMatrix& _Mp );
void AssembleWp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF, const Array<int>& neuBdrP,
                 double _mu, FiniteElementSpace *_UhFESpace, const Vector& w,
                 const IntegrationRule *ir, const IntegrationRule *bir, SparseMatrix& _Wp );
void AssembleFp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF, const Array<int>& neuBdrP,
                 double _dt, double _mu, FiniteElementSpace *_UhFESpace, const Vector& w,
                 const IntegrationRule *ir, const IntegrationRule *bir, SparseMatrix& _Wp );
void AssembleAa( FiniteElementSpace *_AhFESpace, const IntegrationRule *ir, SparseMatrix& Aa );
void AssembleMa( FiniteElementSpace *_AhFESpace, const Array<int>& _essAhTDOF, const IntegrationRule *ir, SparseMatrix& Ma );
void AssembleWa( FiniteElementSpace *_AhFESpace, const Array<int>& _essAhTDOF,
                 double _eta, double _mu0, FiniteElementSpace *_UhFESpace, const Vector& w,
                 const IntegrationRule *ir, SparseMatrix& _Wa );
void AssembleCp( FiniteElementSpace *_AhFESpace, const Array<int>& _essAhTDOF,
                 double _dt, double _mu0, const Vector& a,
                 const SparseMatrix& _Fa, const SparseMatrix& _Ma, const SparseMatrix& _Aa,
                 SparseMatrix& _Cp );
// Handy functions for assembling single components of the preconditioner
void AssembleLub( const Operator* Y,   const Operator* Fui,
                  BlockLowerTriangularPreconditioner* Lub );
void AssembleUub( const Operator* Z,   const Operator* aSi, BlockUpperTriangularPreconditioner* Uub );
void AssembleLup( const Operator* Fui, const Operator* B,   BlockLowerTriangularPreconditioner* Lup );
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* pSi, BlockUpperTriangularPreconditioner* Uup );
//---------------------------------------------------------------------------
void SaveSolution( const Array<BlockVector>& sol,
                   const Array< FiniteElementSpace* >& feSpaces, double _dt, Mesh* _mesh,
                   const string& path, const string& filename );
void SaveError( const Array<BlockVector>& sol,
                const Array< FiniteElementSpace* >& feSpaces,
                void(  *uFun_ex)( const Vector & x, const double t, Vector & u ),
                double(*pFun_ex)( const Vector & x, const double t             ),
                double(*zFun_ex)( const Vector & x, const double t             ),
                double(*aFun_ex)( const Vector & x, const double t             ),
                double _dt, Mesh* _mesh,
                const std::string& path, const std::string& filename );
//---------------------------------------------------------------------------
int main(int argc, char *argv[]){

  //*************************************************************************
  // Problem definition
  //*************************************************************************
  // Define parameters
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int refLvl = 4;
  const int _dim = 2;
  const char *petscrc_file = "rc_SpaceTimeIMHD2D";


  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;

  int NT = 2;
  int pbType = 5;
  string pbName;
  int verbose = 0;

  double _mu  = 1.0;
  double _mu0 = 1.0;
  double _eta = 1.0;

  double Tend = 1.;
  double T0   = 0.;

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

  // Parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&NT, "-NT", "--timeSteps",
                "Number of time steps (default: 2)");
  args.AddOption(&Tend, "-T", "--Tend",
                "Final time (default: 1.0)");
  args.AddOption(&ordU, "-oU", "--ordU",
                "Velocity space polynomial order (default: 2)");
  args.AddOption(&ordP, "-oP", "--ordP",
                "Pressure space polynomial order (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--ordZ",
                "Laplacian of vector potential space polynomial order (default: 1)");
  args.AddOption(&ordA, "-oA", "--ordA",
                "Vector potential space polynomial order (default: 2)");
  args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                "PetscOptions file to use: rc_STMHD         TODO! (direct (LU) solvers - remember to flag _SingAp if Ap is singular))\n"
                "                          rc_STMHD_approx2 (MG solvers for Ap,Fu, Cheb for Mp)\n"
                "                          rc_STMHD_FGMRES  (use FGMRES rather than GMRES for outer solver (useful if ST=2))\n");
  args.AddOption(&pbType, "-Pb", "--problem",
                "Problem: 1 to 4-Analytical test cases\n"
                "              5-Kelvin-Helmholtz instability (default)\n"
                "              6-Island coalescensce\n"
                "              7-Modified Hartmann flow\n"
                "             11-Driven cavity flow\n"
        );
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

  args.Parse();


  const double _dt = ( Tend-T0 )/ NT;

  const int   maxNewtonIt = 5;
  const double  newtonTol = 1e-4;

  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);


  // Define problem
  // - boundary conditions
  Array<int> essTagsU(0);
  Array<int> essTagsV(0);
  Array<int> essTagsP(0);
  Array<int> essTagsA(0);


  switch (pbType){
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
      essTagsA.SetSize(1); essTagsA[0] = 4; // W
      pbName = "An4";
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
      _mu  = 1.5 * 1e-2;
      _eta = 1e-2;

      pbName = "KHI";
      break;
    }
    // Island coalescence
    case 6:{
      mesh_file = "./meshes/tri-square-island.mesh";
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
      // // Set BC: [0,1]x[-1,1]
      // // Top+bottom=1 Left+Right=2
      // // - Dirichlet on u on left and right
      // // - Dirichlet on v on top and bottom
      // // - No stress on tangential component (u) top and bottom
      // // - No stress on tangential component (v) left and right
      // // - Dirichlet on p on top and bottom (outflow, used only in precon)
      // // - Dirichlet on A on top and bottom
      // mesh_file = "./meshes/tri-rect-island.mesh";
      // essTagsU.SetSize(1); essTagsU[0] = 2;
      // essTagsV.SetSize(1); essTagsV[0] = 1;
      // essTagsP.SetSize(1); essTagsP[0] = 1;
      // essTagsA.SetSize(1); essTagsA[0] = 1;
      // Set BC: [0,1]x[0,1]
      // Top=1 Left+Right=2 Bottom=3
      // - Dirichlet on u on left and right
      // - Dirichlet on v on top and bottom
      // - No stress on tangential component (u) top and bottom
      // - No stress on tangential component (v) left, right and bottom
      // - Dirichlet on p on top (outflow?, used only in precon?)
      // - Dirichlet on A on top
      essTagsU.SetSize(1); essTagsU[0] = 2;
      essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3;
      essTagsP.SetSize(1); essTagsP[0] = 1;
      essTagsA.SetSize(1); essTagsA[0] = 1;
      pbName = "IslandCoalescence";
      _mu  = IslandCoalescenceData::mu;
      _eta = IslandCoalescenceData::eta;
      break;
    }
    // MHD rayleigh flow
    case 8:{
      mesh_file = "./meshes/tri-square-rayleigh.mesh";
      uFun = uFun_ex_rayleigh;
      pFun = pFun_ex_rayleigh;
      zFun = zFun_ex_rayleigh;
      aFun = aFun_ex_rayleigh;
      fFun = fFun_rayleigh;
      gFun = gFun_rayleigh;
      hFun = hFun_rayleigh;
      nFun = nFun_rayleigh;
      mFun = mFun_rayleigh;
      wFun = wFun_rayleigh;
      qFun = qFun_rayleigh;
      yFun = yFun_rayleigh;
      cFun = cFun_rayleigh;
      // Set BC:
      // Dirichlet everywhere but east (outflow?) for u
      // Dirichlet everywhere for A
      essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4;
      essTagsV = essTagsU;
      essTagsP.SetSize(1); essTagsP[0] = 2;
      essTagsA.SetSize(4); essTagsA[0] = 1; essTagsA[1] = 2; essTagsA[2] = 3; essTagsA[3] = 4;
      pbName = "MHDRayleigh";
      _mu  = RayleighData::mu  / RayleighData::rho;   // they consider rho separately, so I need to rescale everything...
      _mu0 = RayleighData::rho * RayleighData::mu0;
      _eta = RayleighData::rho * RayleighData::eta;

      break;
    }
    default:
      std::cerr<<"ERROR: Problem type "<<pbType<<" not recognised."<<std::endl;
  }

  args.PrintOptions(cout);
  std::cout<<"   --dt  "<<Tend/numProcs<<std::endl;
  std::cout<<"   --mu  "<<_mu<<std::endl;
  std::cout<<"   --eta "<<_eta<<std::endl;
  std::cout<<"   --mu0 "<<_mu0<<std::endl;
  std::cout<<"   --Pb  "<<pbName<<std::endl;



  // - initialise function data
  VectorFunctionCoefficient uFuncCoeff( _dim, uFun );
  FunctionCoefficient       pFuncCoeff(       pFun );
  FunctionCoefficient       zFuncCoeff(       zFun );
  FunctionCoefficient       aFuncCoeff(       aFun );
  VectorFunctionCoefficient wFuncCoeff( _dim, wFun );
  FunctionCoefficient       qFuncCoeff(       qFun );
  FunctionCoefficient       yFuncCoeff(       yFun );
  FunctionCoefficient       cFuncCoeff(       cFun );
  VectorFunctionCoefficient fFuncCoeff( _dim, fFun );
  FunctionCoefficient       gFuncCoeff(       gFun );
  FunctionCoefficient       hFuncCoeff(       hFun );
  VectorFunctionCoefficient nFuncCoeff( _dim, nFun );
  FunctionCoefficient       mFuncCoeff(       mFun );




  //*************************************************************************
  // FE spaces initialisation
  //*************************************************************************
  // Generate mesh
  Mesh *_mesh = new Mesh( mesh_file.c_str(), 1, 1 );
  // - refine as needed
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  // - initialise FE info
  FiniteElementCollection* _UhFEColl = new H1_FECollection( ordU, _dim );
  FiniteElementCollection* _PhFEColl = new H1_FECollection( ordP, _dim );
  FiniteElementCollection* _ZhFEColl = new H1_FECollection( ordZ, _dim );
  FiniteElementCollection* _AhFEColl = new H1_FECollection( ordA, _dim );
  FiniteElementSpace* _UhFESpace = new FiniteElementSpace( _mesh, _UhFEColl, _dim );
  FiniteElementSpace* _PhFESpace = new FiniteElementSpace( _mesh, _PhFEColl );
  FiniteElementSpace* _ZhFESpace = new FiniteElementSpace( _mesh, _ZhFEColl );
  FiniteElementSpace* _AhFESpace = new FiniteElementSpace( _mesh, _AhFEColl );

  // - identify dirichlet nodes
  Array<int> _essUhTDOF, _essPhTDOF, _essAhTDOF;
  int numAtt = _mesh->bdr_attributes.Max();
  Array<int> essBdrU( numAtt ); essBdrU = 0;
  Array<int> essBdrV( numAtt ); essBdrV = 0;
  Array<int> essBdrP( numAtt ); essBdrP = 0;
  Array<int> essBdrA( numAtt ); essBdrA = 0;
  Array<int> neuBdrU( numAtt ); neuBdrU = 0;
  Array<int> neuBdrP( numAtt ); neuBdrP = 0;
  Array<int> neuBdrA( numAtt ); neuBdrA = 0;
  if ( _mesh->bdr_attributes.Size() > 0 ) {
    // search among all possible tags
    for ( int i = 1; i <= _mesh->bdr_attributes.Max(); ++i ){
      // if that tag is marked in the corresponding array in essTags, then flag it
      if( essTagsU.Find( i ) + 1 )
        essBdrU[i-1] = 1;
      if( essTagsV.Find( i ) + 1 )
        essBdrV[i-1] = 1;
      if( essTagsP.Find( i ) + 1 ){
        essBdrP[i-1] = 1;
      }else{
        neuBdrP[i-1] = 1;
      }
      if( essTagsA.Find( i ) + 1 ){
        essBdrA[i-1] = 1;
      }else{
        neuBdrA[i-1] = 1;                              // if it's not dirichlet, then it's neumann
      }
      if( essBdrU[i-1] == 0 || essBdrV[i-1] == 0 )     // if it's not dirichlet, then it's neumann
        neuBdrU[i-1] = 1;
    }
    Array<int> essVhTDOF;
    _UhFESpace->GetEssentialTrueDofs( essBdrV,  essVhTDOF, 1 );
    _UhFESpace->GetEssentialTrueDofs( essBdrU, _essUhTDOF, 0 );
    _PhFESpace->GetEssentialTrueDofs( essBdrP, _essPhTDOF );
    _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );
    _essUhTDOF.Append( essVhTDOF ); // combine them together, as it's not necessary to treat nodes indices separately for the two components

  }
  std::cout << "***********************************************************\n";
  std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
  std::cout << "***********************************************************\n";
  // std::cout << "Dir u      "; _essUhTDOF.Print(mfem::out, _essUhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "Dir p      "; _essPhTDOF.Print(mfem::out, _essPhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "Dir A      "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "***********************************************************\n";

  
  //*************************************************************************
  // Time-independent operators assembly
  //*************************************************************************
  // Initialise solution
  Array< BlockVector > sol( NT + 1 );
  Array<int> offsets(5), offsetsReduced(4);
  offsets[0] = 0;
  offsets[1] = _UhFESpace->GetTrueVSize();
  offsets[2] = _PhFESpace->GetTrueVSize();
  offsets[3] = _ZhFESpace->GetTrueVSize();
  offsets[4] = _AhFESpace->GetTrueVSize();
  offsets.PartialSum();
  offsetsReduced[0] = 0;
  offsetsReduced[1] = _UhFESpace->GetTrueVSize();
  offsetsReduced[2] = _PhFESpace->GetTrueVSize();
  offsetsReduced[3] = _AhFESpace->GetTrueVSize();
  offsetsReduced.PartialSum();
  for ( int tt = 0; tt < NT+1; ++tt ){
    sol[tt].Update( offsets );
    uFuncCoeff.SetTime( T0 + _dt*tt );
    pFuncCoeff.SetTime( T0 + _dt*tt );
    zFuncCoeff.SetTime( T0 + _dt*tt );
    aFuncCoeff.SetTime( T0 + _dt*tt );
    GridFunction uGF(_UhFESpace); uGF.ProjectCoefficient( uFuncCoeff );
    GridFunction pGF(_PhFESpace); pGF.ProjectCoefficient( pFuncCoeff );
    GridFunction zGF(_ZhFESpace); zGF.ProjectCoefficient( zFuncCoeff );
    GridFunction aGF(_AhFESpace); aGF.ProjectCoefficient( aFuncCoeff );
    // this is used to evaluate Dirichlet nodes
    sol[tt].GetBlock(0) = uGF;
    sol[tt].GetBlock(1) = pGF;
    sol[tt].GetBlock(2) = zGF;
    sol[tt].GetBlock(3) = aGF;
  }

  // Define integration rule to be used throughout
  Array<int> ords(3);
  ords[0] = 2*ordU + ordU-1;       // ( (u·∇)u, v )
  ords[1] = ordZ + ordA-1 + ordU;  // (   z ∇A, v )
  ords[2] = ordU + ordA-1 + ordA;  // ( (u·∇A), B )
  const IntegrationRule *ir  = &IntRules.Get( Geometry::Type::TRIANGLE, ords.Max() ); // for domains
  const IntegrationRule *bir = &IntRules.Get( Geometry::Type::SEGMENT,  ords.Max() ); // for boundaries
  std::cout<<"Selecting integrator of order "<<ords.Max()<<std::endl;


  // Define own integrator
  Array< FiniteElementSpace* > feSpaces(4);
  feSpaces[0] = _UhFESpace;
  feSpaces[1] = _PhFESpace;
  feSpaces[2] = _ZhFESpace;
  feSpaces[3] = _AhFESpace;
  BlockNonlinearForm _IMHD2DOperator;
  _IMHD2DOperator.SetSpaces( feSpaces );
  _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ) );
  _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ), essBdrA );
  // _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ) );
  // std::cout<<"Warning: excluding bdr contribution from integrator"<<std::endl;

  std::cout<<"Warning: I'm messing up BC for pressure! Also on operators and rhs for f!"<<std::endl;
  Array< Array<int> * > tmpEssDofs(4);
  // Array<int> tempPhTDOF(0); // Set all to 0: Dirichlet BC are never imposed on pressure: they are only used to assemble the pressure operators    
  Array<int> tempZhTDOF(0); // Set all to 0: Dirichlet BC are never imposed on laplacian of vector potential    
  tmpEssDofs[0] = &_essUhTDOF;
  // tmpEssDofs[1] = &tempPhTDOF;
  tmpEssDofs[1] = &_essPhTDOF;
  tmpEssDofs[2] = &tempZhTDOF;
  tmpEssDofs[3] = &_essAhTDOF;
  Array< Vector * > dummy(4); dummy = NULL;
  _IMHD2DOperator.SetEssentialTrueDofs( tmpEssDofs, dummy );
  // - extract and keep some constant operators
  BlockOperator* dummyJ = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( sol[0] ) );
  SparseMatrix B  = *( dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(1,0) ) );
  SparseMatrix Ip = *( dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(1,1) ) );
  SparseMatrix Mz = *( dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(2,2) ) );
  SparseMatrix K  = *( dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(2,3) ) );
  TransposeOperator Bt(&B);
  // - inverse of mass matrix for z
  PetscParMatrix MzPetsc( &Mz );
  PetscLinearSolver Mzi( MzPetsc, "ZSolverMass_" );


  // Pressure Schur complement operators
  SparseMatrix Ap;
  SparseMatrix Mp;
  SparseMatrix Fp;
  AssembleAp( _PhFESpace, _essPhTDOF, ir, Ap );
  AssembleMp( _PhFESpace, _essPhTDOF, ir, Mp );
  // - inverse of operators for p
  PetscParMatrix ApPetsc( &Ap );
  PetscLinearSolver Api( ApPetsc, "PSolverLaplacian_" );
  PetscParMatrix MpPetsc( &Mp );
  PetscLinearSolver Mpi( MpPetsc, "PSolverMass_" );
  // // - using own Schur complement class
  // OseenSTPressureSchurComplement pSi( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essPhTDOF, verbose );
  // SparseMatrix Wp;
  // pSi.SetAp( &Ap );
  // pSi.SetMp( &Mp );


  // Magnetic Schur complement operators
  SparseMatrix Aa;
  SparseMatrix Ma;
  SparseMatrix Cp;
  AssembleAa( _AhFESpace, ir, Aa );
  AssembleMa( _AhFESpace, _essAhTDOF, ir, Ma );
  // - inverse of mass matrix for A
  PetscParMatrix MaPetsc( &Ma );
  PetscLinearSolver Mai( MaPetsc, "ASolverMass_" );
  // // - using own Schur complement class
  // IMHD2DSTMagneticSchurComplement aSi( MPI_COMM_SELF, _dt, NULL, NULL, NULL, _essAhTDOF, verbose );
  // SparseMatrix Wa;
  // aSi.SetM( &Ma );




  //*************************************************************************
  // TIME-STEPPING ROUTINE
  //*************************************************************************

  int GMRESNonConv     = 0;  
  int newtNonConv      = 0;  
  double tottotGMRESIt = 0;  //leave it as double to avoid round-off when averaging
  double totNewtonIt   = 0;  

  for ( int tt = 1; tt < NT+1; ++tt ){

    // Set data functions
    fFuncCoeff.SetTime( T0 + _dt*tt );
    gFuncCoeff.SetTime( T0 + _dt*tt );
    hFuncCoeff.SetTime( T0 + _dt*tt );
    nFuncCoeff.SetTime( T0 + _dt*tt );
    mFuncCoeff.SetTime( T0 + _dt*tt );

    // Compute rhs
    // for u
    // - actual rhs
    LinearForm frhs( _UhFESpace );
    ScalarVectorProductCoefficient muNFuncCoeff(_mu, nFuncCoeff);
    frhs.AddDomainIntegrator(   new VectorDomainLFIntegrator(         fFuncCoeff       )          );  //int_\Omega f*v
    // frhs.AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     muNFuncCoeff       ), neuBdrU );  //int_d\Omega \mu * du/dn *v
    // frhs.AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator(   pFuncCoeff, -1.0 ), neuBdrU );  //int_d\Omega -p*v*n
    frhs.GetDLFI()->operator[](0)->SetIntRule(ir);
    // frhs.GetBLFI()->operator[](0)->SetIntRule(bir);
    // frhs.GetBLFI()->operator[](1)->SetIntRule(bir);
    frhs.Assemble();
    frhs.operator*=( _dt );
    // - effect from solution at previous iteration
    LinearForm uPrevForm( _UhFESpace );
    GridFunction uPrev( _UhFESpace ); uPrev = sol[tt-1].GetBlock(0);
    VectorGridFunctionCoefficient uPrevFuncCoeff( &uPrev );
    uPrevForm.AddDomainIntegrator( new VectorDomainLFIntegrator( uPrevFuncCoeff ) );  //int_\Omega u(t-1)*v
    uPrevForm.GetDLFI()->operator[](0)->SetIntRule(ir);
    uPrevForm.Assemble();
    // - combine together
    frhs += uPrevForm;

    // for p
    LinearForm grhs( _PhFESpace );
    grhs.AddDomainIntegrator( new DomainLFIntegrator( gFuncCoeff ) );  //int_\Omega g*q
    grhs.GetDLFI()->operator[](0)->SetIntRule(ir);
    grhs.Assemble();
    grhs *= _dt;

    // for z
    LinearForm zrhs( _ZhFESpace );
    zrhs.AddBoundaryIntegrator( new BoundaryLFIntegrator( mFuncCoeff ), neuBdrA );  //int_d\Omega dA/dn *zeta
    zrhs.GetBLFI()->operator[](0)->SetIntRule(bir);
    zrhs.Assemble();
    // zrhs = 0.;
    // std::cout<<"Warning: setting rhs for z to 0"<<std::endl;


    // for A
    // - actual rhs
    LinearForm hrhs( _AhFESpace );
    ProductCoefficient etaMFuncCoeff( _eta/_mu0, mFuncCoeff );
    hrhs.AddDomainIntegrator(   new DomainLFIntegrator(      hFuncCoeff )          );  //int_\Omega h*B
    hrhs.AddBoundaryIntegrator( new BoundaryLFIntegrator( etaMFuncCoeff ), neuBdrA );  //int_d\Omega \eta/\mu0 * dA/dn *B
    hrhs.GetDLFI()->operator[](0)->SetIntRule(ir);
    hrhs.GetBLFI()->operator[](0)->SetIntRule(bir);
    hrhs.Assemble();
    hrhs.operator*=( _dt );
    // - effect from solution at previous iteration
    LinearForm aPrevForm( _AhFESpace );
    GridFunction aPrev( _AhFESpace ); aPrev = sol[tt-1].GetBlock(3);
    GridFunctionCoefficient aPrevFuncCoeff( &aPrev );
    aPrevForm.AddDomainIntegrator( new DomainLFIntegrator( aPrevFuncCoeff ) );  //int_\Omega A(t-1)*B
    aPrevForm.GetDLFI()->operator[](0)->SetIntRule(ir);
    aPrevForm.Assemble();
    // - combine together
    hrhs += aPrevForm;

    // clean Dirichlet nodes from rhs (they are already solved for, and they shouldn't dirty the solution)
    frhs.SetSubVector( _essUhTDOF, 0. );
    grhs.SetSubVector( _essPhTDOF, 0. );
    hrhs.SetSubVector( _essAhTDOF, 0. );


    // Initialise Newton iteration using sol recovered at previous time step
    BlockVector lclSol = sol[tt-1];
    // - however, remember to preserve Dirichlet nodes
    for ( int i = 0; i < _essUhTDOF.Size(); ++i )
      lclSol.GetBlock(0)(_essUhTDOF[i]) = sol[tt].GetBlock(0)(_essUhTDOF[i]);
    for ( int i = 0; i < _essPhTDOF.Size(); ++i )
      lclSol.GetBlock(1)(_essPhTDOF[i]) = sol[tt].GetBlock(1)(_essPhTDOF[i]);
    for ( int i = 0; i < _essAhTDOF.Size(); ++i )
      lclSol.GetBlock(3)(_essAhTDOF[i]) = sol[tt].GetBlock(3)(_essAhTDOF[i]);
    // - also set z to be *exactly* the laplacian of A
    BlockVector lclRes(offsets); // this will be reused later
    lclSol.GetBlock(2) = 0.;
    _IMHD2DOperator.Mult( lclSol, lclRes );
    Vector tempZ = lclRes.GetBlock(2);      // this is the "full" K*A, including effect from Dirichlet
    tempZ -= zrhs;
    tempZ.Neg();
    Mzi.Mult( tempZ, lclSol.GetBlock(2) );  // solve for z



    // - compute residual
    _IMHD2DOperator.Mult( lclSol, lclRes );  // N(x)
    lclRes.GetBlock(0) -= frhs;              // N(x) - b
    lclRes.GetBlock(1) -= grhs;
    // lclRes.GetBlock(2) -= zrhs;
    lclRes.GetBlock(2)  = 0.;                // should already be basically zero!
    lclRes.GetBlock(3) -= hrhs;
    lclRes.Neg();                            // b - N(x)

    // - compute norm of residual and initialise relevant quantities for Newton iteration
    int newtonIt = 0;
    double newtonRes = lclRes.Norml2();
    double newtonRes0 = newtonRes;
    double newtonErrWRTPrevIt = newtonTol;
    double totGMRESit = 0.; //leave it as double, so that when I'll average it, it won't round-off
    std::cout << "***********************************************************\n";
    std::cout << "Starting Newton for time-step "<<tt<<", initial residual "<< newtonRes
              << ", (u,p,z,A) = ("<< lclRes.GetBlock(0).Norml2() <<","
                                  << lclRes.GetBlock(1).Norml2() <<","
                                  << lclRes.GetBlock(2).Norml2() <<","
                                  << lclRes.GetBlock(3).Norml2() <<")" << std::endl;
    std::cout << "***********************************************************\n";





    //*************************************************************************
    // NEWTON ITERATIONS
    //*************************************************************************
    for ( newtonIt = 0; newtonIt < maxNewtonIt
                     // && newtonRes >= newtonTol
                     && newtonRes/newtonRes0 >= newtonTol;
                     // && newtonErrWRTPrevIt >= newtonTol;
                     ++newtonIt ){      

      // Define internal (GMRES) solver
      // - Get gradient and define relevant operators
      BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( lclSol ) );
      SparseMatrix Fu = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,0) ) );
      SparseMatrix Z1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) ) );
      SparseMatrix Z2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) ) );
      SparseMatrix Y  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,0) ) );
      SparseMatrix Fa = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,3) ) );

      // - Complete Z operator
      ScaledOperator* mMzi = new ScaledOperator( &Mzi, -1.0 );
      Array<const Operator*> mZ1MziKops(3);
      Array<bool>            mZ1MziKown(3);
      mZ1MziKops[0] = &K;   mZ1MziKown[0] = false;      
      mZ1MziKops[1] = mMzi; mZ1MziKown[1] = true;
      mZ1MziKops[2] = &Z1;  mZ1MziKown[2] = false;
      OperatorsSequence* mZ1MziK = new OperatorsSequence( mZ1MziKops, mZ1MziKown );
      Array<const Operator*> Zops(2);
      Array<bool>            Zown(2);
      Zops[0] = mZ1MziK;  Zown[0] = true;
      Zops[1] = &Z2;      Zown[1] = false;
      OperatorsSeries Z( Zops, Zown ); // own second, not first
      // std::cout<<" K:        "<<        K.Height() <<","<<        K.Width() <<std::endl;
      // std::cout<<"-Mzi:      "<<    mMzi->Height() <<","<<    mMzi->Width() <<std::endl;
      // std::cout<<" Z1:       "<<       Z1.Height() <<","<<       Z1.Width() <<std::endl;
      // std::cout<<" Z2:       "<<       Z2.Height() <<","<<       Z2.Width() <<std::endl;
      // std::cout<<"-Z1MziK:   "<< mZ1MziK->Height() <<","<< mZ1MziK->Width() <<std::endl;
      // std::cout<<" Fu:       "<<       Fu.Height() <<","<<       Fu.Width() <<std::endl;
      // std::cout<<" B:        "<<        B.Height() <<","<<        B.Width() <<std::endl;
      // std::cout<<" Z:        "<<        Z.Height() <<","<<        Z.Width() <<std::endl;
      // std::cout<<" Bt:       "<<       Bt.Height() <<","<<       Bt.Width() <<std::endl;
      // std::cout<<" Y:        "<<        Y.Height() <<","<<        Y.Width() <<std::endl;
      // std::cout<<" Fa:       "<<       Fa.Height() <<","<<       Fa.Width() <<std::endl;
      // offsetsReduced.Print(mfem::out, offsetsReduced.Size());

      // - Define actual system
      BlockOperator MHDOp(offsetsReduced);
      MHDOp.SetBlock( 0, 0, &Fu );
      MHDOp.SetBlock( 0, 1, &Bt );
      MHDOp.SetBlock( 0, 2, &Z  );
      MHDOp.SetBlock( 1, 0, &B  );
      MHDOp.SetBlock( 1, 1, &Ip );
      MHDOp.SetBlock( 2, 0, &Y  );
      MHDOp.SetBlock( 2, 2, &Fa );

      
      // Define preconditioner
      // - Velocity block inverse
      PetscParMatrix FuPetsc( &Fu );
      PetscLinearSolver Fui( FuPetsc, "VSolver_" );

      // // - Approximate pressure Schur complement inverse
      // bool isQuietState = true;
      // for ( int i = 0; i < lclSol.GetBlock(0).Size(); ++i ){
      //   if ( lclSol.GetBlock(0)[i] != 0. ){
      //     isQuietState = false;
      //     break;
      //   }
      // }
      // if ( isQuietState ){
      //   AssembleWp( _PhFESpace, _essPhTDOF, neuBdrP, _mu, _UhFESpace, lclSol.GetBlock(0), ir, bir, Wp );
      //   pSi.SetWp( &Wp, isQuietState );
      // }
      AssembleFp( _PhFESpace, _essPhTDOF, neuBdrP, _dt, _mu, _UhFESpace, lclSol.GetBlock(0), ir, bir, Fp );
      Array<const Operator*> pSiops(3);
      Array<bool>            pSiown(3);
      pSiops[0] = &Api;      pSiown[0] = false;
      pSiops[1] = &Fp;       pSiown[1] = false;
      pSiops[2] = &Mpi;      pSiown[2] = false;
      OperatorsSequence pSi( pSiops, pSiown );
      



      // // - Approximate magnetic Schur complement inverse
      // AssembleWa( _AhFESpace, _essAhTDOF, _eta, _mu0, _UhFESpace, lclSol.GetBlock(0), ir, Wa );
      // AssembleCp( _AhFESpace, _essAhTDOF, _dt,  _mu0, lclSol.GetBlock(3), Fa, Ma, Aa, Cp );
      // // -- Inverse of main diagonal of wave equation
      // PetscParMatrix CpPetsc( &Cp );
      // PetscLinearSolver CCi( CpPetsc, "AWaveSolver_" );
      // aSi.SetW( &Wa );
      // aSi.SetCCinv( &CCi );
      AssembleCp( _AhFESpace, _essAhTDOF, _dt,  _mu0, lclSol.GetBlock(3), Fa, Ma, Aa, Cp );
      PetscParMatrix CpPetsc( &Cp );
      PetscLinearSolver CCi( CpPetsc, "AWaveSolver_" );
      Array<const Operator*> aSiops(3);
      Array<bool>            aSiown(3);
      aSiops[0] = &Mai;      aSiown[0] = false;
      aSiops[1] = &Fa;       aSiown[1] = false;
      aSiops[2] = &CCi;      aSiown[2] = false;
      OperatorsSequence aSi( aSiops, aSiown );



      // - assemble single factors
      BlockUpperTriangularPreconditioner *Uub = new BlockUpperTriangularPreconditioner( offsetsReduced ),
                                         *Uup = new BlockUpperTriangularPreconditioner( offsetsReduced );
      BlockLowerTriangularPreconditioner *Lub = new BlockLowerTriangularPreconditioner( offsetsReduced ),
                                         *Lup = new BlockLowerTriangularPreconditioner( offsetsReduced );
      AssembleLub( &Y, &Fui,        Lub );
      AssembleUub( &Z,        &aSi, Uub );
      AssembleLup( &Fui, &B,        Lup );
      AssembleUup( &Fui, &Bt, &pSi, Uup );
      // - combine them together
      Array<const Operator*> precOps(4);
      Array<bool>            precOwn(4);
      precOps[0] = Lub;  precOwn[0] = true;
      precOps[1] = Uub;  precOwn[1] = true;
      precOps[2] = Lup;  precOwn[2] = true;
      precOps[3] = Uup;  precOwn[3] = true;
      OperatorsSequence MHDPr( precOps, precOwn );



      // Define solver
      PetscLinearSolver solver( MPI_COMM_SELF, "solver_" );
      bool isIterative = true;
      solver.iterative_mode = isIterative;
      // - register preconditioner and system
      solver.SetPreconditioner(MHDPr);
      solver.SetOperator(MHDOp);
      // - solve!
      BlockVector lclDeltaSolReduced( offsetsReduced );
      BlockVector lclResReduced(      offsetsReduced );
      lclDeltaSolReduced = 0.;
      lclResReduced.GetBlock(0) = lclRes.GetBlock(0);
      lclResReduced.GetBlock(1) = lclRes.GetBlock(1);
      lclResReduced.GetBlock(2) = lclRes.GetBlock(3);
      solver.Mult( lclResReduced, lclDeltaSolReduced );

      BlockVector tmp( offsetsReduced );
      MHDOp.Mult( lclDeltaSolReduced, tmp );
      tmp -= lclResReduced;
      std::cout<<"Residual norm: "<<tmp.Norml2()<<std::endl;

      BlockVector lclDeltaSol( offsets );
      lclDeltaSol.GetBlock(0) = lclDeltaSolReduced.GetBlock(0);
      lclDeltaSol.GetBlock(1) = lclDeltaSolReduced.GetBlock(1);
      lclDeltaSol.GetBlock(2) = 0.;
      lclDeltaSol.GetBlock(3) = lclDeltaSolReduced.GetBlock(2);

      // Update relevant quantities
      // - solution
      lclSol += lclDeltaSol;
      lclSol.GetBlock(2) = 0.;
      _IMHD2DOperator.Mult( lclSol, lclRes );
      Vector tempZ = lclRes.GetBlock(2);      // this is the "full" K*A, including effect from Dirichlet
      tempZ.Neg();
      tempZ += zrhs;
      Mzi.Mult( tempZ, lclSol.GetBlock(2) );


      // - residual
      _IMHD2DOperator.Mult( lclSol, lclRes );  // N(x)
      lclRes.GetBlock(0) -= frhs;              // N(x) - b
      lclRes.GetBlock(1) -= grhs;
      lclRes.GetBlock(2)  = 0.;                // this should already be basically zero
      lclRes.GetBlock(3) -= hrhs;
      lclRes.Neg();                            // b - N(x)
      newtonRes = lclRes.Norml2();
      newtonErrWRTPrevIt = lclDeltaSol.Norml2();




      // Output
      int GMRESits   = solver.GetNumIterations();
      totGMRESit += GMRESits;
      if (solver.GetConverged()){
        std::cout << "Inner solver converged in ";
      }else{
        std::cout << "Inner solver *DID NOT* converge in ";
        GMRESNonConv++;
      }
      std::cout<< GMRESits << " iterations. Residual "<< solver.GetFinalNorm() <<std::endl;
      std::cout << "***********************************************************\n";
      std::cout << "Newton iteration "<< newtonIt+1 <<" for time-step "<< tt <<", residual "<< newtonRes
                << ", (u,p,z,A) = ("<< lclRes.GetBlock(0).Norml2() <<","
                                    << lclRes.GetBlock(1).Norml2() <<","
                                    << lclRes.GetBlock(2).Norml2() <<","
                                    << lclRes.GetBlock(3).Norml2() <<")" << std::endl;
      std::cout << "***********************************************************\n";


      // Clean up - no need: it's owned by MHDpr
      // delete Uub;
      // delete Uup;
      // delete Lub;
      // delete Lup;
    }
  

    std::cout   << "Newton outer solver for time-step " << tt;
    if( newtonIt < maxNewtonIt ){
      std::cout << " converged in "                     << newtonIt;
    }else{
      std::cout << " *DID NOT* converge in "            << maxNewtonIt;
      newtNonConv++;
    }
    std::cout   << " iterations. Residual norm is "     << newtonRes;
    std::cout   << ", avg internal GMRES it are "       << totGMRESit/newtonIt  << ".\n";
    std::cout   << "***********************************************************\n";

    tottotGMRESIt += totGMRESit;
    totNewtonIt   += newtonIt;

    sol[tt] = lclSol;

 
  }

  std::cout<<"***********************************************************\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"\nALL DONE!\n\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"Total # of Newton iterations: "<< totNewtonIt   <<". Non-converged: "<< newtNonConv  <<". Avg per time-step: "<< totNewtonIt/NT   <<"\n";
  std::cout<<"Total # of GMRES  iterations: "<< tottotGMRESIt <<". Non-converged: "<< GMRESNonConv <<". Avg per time-step: "<< tottotGMRESIt/NT
                                                                                                   <<". Avg per Newton it: "<< tottotGMRESIt/totNewtonIt <<"\n";
  std::cout<<"***********************************************************\n";



  string outFilePath = "ParaView";
  string outFileName = "STIMHD2D_" + pbName;
 
  SaveSolution( sol, feSpaces, _dt, _mesh, outFilePath, outFileName );

  if ( pbType == 4 ){
    SaveError( sol, feSpaces, uFun, pFun, zFun, aFun, _dt, _mesh, outFilePath, outFileName + "_err" );
  }


  delete _UhFESpace;
  delete _PhFESpace;
  delete _AhFESpace;
  delete _ZhFESpace;
  delete _UhFEColl;
  delete _PhFEColl;
  delete _AhFEColl;
  delete _ZhFEColl;
  delete _mesh;


  MFEMFinalizePetsc();
  MPI_Finalize();


  return 0;
}























//*************************************************************************
// ASSEMBLE SINGLE OPERATORS
//*************************************************************************
// Involving p ------------------------------------------------------------
// Laplacian for p
void AssembleAp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                 const IntegrationRule *ir, SparseMatrix& _Ap ){

  BilinearForm aVarf( _PhFESpace );
  ConstantCoefficient one( 1.0 );     // diffusion

  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
  aVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  aVarf.Assemble();
  aVarf.Finalize();
  
  _Ap.MakeRef( aVarf.SpMat() );
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf.LoseMat();

  // Impose homogeneous dirichlet BC by simply removing corresponding equations
  mfem::Array<int> colsP(_Ap.Height());
  colsP = 0.;
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    colsP[_essPhTDOF[i]] = 1;
  }
  _Ap.EliminateCols( colsP );
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    _Ap.EliminateRow( _essPhTDOF[i], mfem::Matrix::DIAG_ONE );
  }
}


// Mass matrix for p
void AssembleMp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                 const IntegrationRule *ir, SparseMatrix& _Mp ){

  BilinearForm mVarf( _PhFESpace );
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new MassIntegrator( one ));
  mVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  mVarf.Assemble();
  mVarf.Finalize();

  _Mp.MakeRef( mVarf.SpMat() );
  _Mp.SetGraphOwner(true);
  _Mp.SetDataOwner(true);
  mVarf.LoseMat();

  // Impose homogeneous dirichlet BC by simply removing corresponding equations
  mfem::Array<int> colsP(_Mp.Height());
  colsP = 0.;
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    colsP[_essPhTDOF[i]] = 1;
  }
  _Mp.EliminateCols( colsP );
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    _Mp.EliminateRow( _essPhTDOF[i], mfem::Matrix::DIAG_ONE );
  }
}


// Spatial part for p: Wp(w) <-> mu*( ∇p, ∇q ) + ( w·∇p, q )
void AssembleWp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF, const Array<int>& neuBdrP,
                 double _mu, FiniteElementSpace *_UhFESpace, const Vector& w,
                 const IntegrationRule *ir, const IntegrationRule *bir, SparseMatrix& _Wp ){

  BilinearForm wVarf( _PhFESpace );
  ConstantCoefficient mu( _mu );
  GridFunction wFuncCoeff(_UhFESpace);
  wFuncCoeff = w;
  VectorGridFunctionCoefficient wCoeff( &wFuncCoeff );
  wVarf.AddDomainIntegrator(new DiffusionIntegrator( mu ));
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( wCoeff, 1.0 ));
  // wVarf.AddBdrFaceIntegrator( new BoundaryFaceFluxIntegrator( wCoeff ) );
  wVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  wVarf.GetDBFI()->operator[](1)->SetIntRule(ir);
  // wVarf.GetBFBFI()->operator[](0)->SetIntRule(bir);
  
  wVarf.Assemble();
  wVarf.Finalize();
  
  _Wp.MakeRef( wVarf.SpMat() );
  _Wp.SetGraphOwner(true);
  _Wp.SetDataOwner(true);
  wVarf.LoseMat();

  // Impose homogeneous dirichlet BC by simply removing corresponding equations
  mfem::Array<int> colsP(_Wp.Height());
  colsP = 0.;
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    colsP[_essPhTDOF[i]] = 1;
  }
  _Wp.EliminateCols( colsP );
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    _Wp.EliminateRow( _essPhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }

}


// Complete PCD operator: Fp(w) <-> (p,q) + dt*mu*( ∇p, ∇q ) + dt*( w·∇p, q )
void AssembleFp( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF, const Array<int>& neuBdrP,
                 double _dt, double _mu, FiniteElementSpace *_UhFESpace, const Vector& w,
                 const IntegrationRule *ir, const IntegrationRule *bir, SparseMatrix& _Fp ){

  BilinearForm pVarf( _PhFESpace );
  ConstantCoefficient dtmu( _dt*_mu );
  ConstantCoefficient one( 1. );
  GridFunction wFuncCoeff(_UhFESpace);
  wFuncCoeff = w;
  VectorGridFunctionCoefficient wCoeff( &wFuncCoeff );
  pVarf.AddDomainIntegrator(new MassIntegrator( one ));
  pVarf.AddDomainIntegrator(new DiffusionIntegrator( dtmu ));
  pVarf.AddDomainIntegrator(new ConvectionIntegrator( wCoeff, _dt ));
  // wVarf.AddBdrFaceIntegrator( new BoundaryFaceFluxIntegrator( wCoeff ) );
  pVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  pVarf.GetDBFI()->operator[](1)->SetIntRule(ir);
  pVarf.GetDBFI()->operator[](2)->SetIntRule(ir);
  // wVarf.GetBFBFI()->operator[](0)->SetIntRule(bir);
  
  pVarf.Assemble();
  pVarf.Finalize();
  
  _Fp.MakeRef( pVarf.SpMat() );
  _Fp.SetGraphOwner(true);
  _Fp.SetDataOwner(true);
  pVarf.LoseMat();

  // Impose homogeneous dirichlet BC by simply removing corresponding equations
  mfem::Array<int> colsP(_Fp.Height());
  colsP = 0.;
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    colsP[_essPhTDOF[i]] = 1;
  }
  _Fp.EliminateCols( colsP );
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    _Fp.EliminateRow( _essPhTDOF[i], mfem::Matrix::DIAG_ONE );
  }



}



// Involving A ------------------------------------------------------------
// Laplacian for A
void AssembleAa( FiniteElementSpace *_AhFESpace, const IntegrationRule *ir, SparseMatrix& Aa ){
  BilinearForm aVarf( _AhFESpace );
  ConstantCoefficient one( 1.0 );
  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
  aVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  aVarf.Assemble();
  aVarf.Finalize();

  Aa.MakeRef( aVarf.SpMat() );
  Aa.SetGraphOwner(true);
  Aa.SetDataOwner(true);
  aVarf.LoseMat();
}


// Mass matrix for A (dirichlet nodes set to identity)
void AssembleMa( FiniteElementSpace *_AhFESpace, const Array<int>& _essAhTDOF, const IntegrationRule *ir, SparseMatrix& Ma ){

  BilinearForm mVarf( _AhFESpace );
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new MassIntegrator( one ));
  mVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  mVarf.Assemble();
  mVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  Ma.MakeRef( mVarf.SpMat() );
  Ma.SetGraphOwner(true);
  Ma.SetDataOwner(true);
  mVarf.LoseMat();

  // - impose Dirichlet BC
  mfem::Array<int> colsA(Ma.Height());
  colsA = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  Ma.EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    Ma.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE );
  }
}


// Spatial part for A: Wa(w) <-> eta/mu0*( ∇A, ∇C ) + ( w·∇A, C )
void AssembleWa( FiniteElementSpace *_AhFESpace, const Array<int>& _essAhTDOF,
                 double _eta, double _mu0, FiniteElementSpace *_UhFESpace, const Vector& w,
                 const IntegrationRule *ir, SparseMatrix& _Wa ){

  BilinearForm wVarf( _AhFESpace );
  ConstantCoefficient etaOverMu( _eta/_mu0 );
  wVarf.AddDomainIntegrator(new DiffusionIntegrator( etaOverMu ));
  wVarf.GetDBFI()->operator[](0)->SetIntRule(ir);

  // include convection
  // - NB: multiplication by dt is handled inside the Schur complement approximation
  GridFunction wFuncCoeff(_UhFESpace);
  wFuncCoeff = w;
  VectorGridFunctionCoefficient wCoeff( &wFuncCoeff );
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( wCoeff, 1.0 ));
  wVarf.GetDBFI()->operator[](1)->SetIntRule(ir);

  wVarf.Assemble();
  wVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  _Wa.MakeRef( wVarf.SpMat() );
  _Wa.SetGraphOwner(true);
  _Wa.SetDataOwner(true);
  wVarf.LoseMat();
  
  // - impose Dirichlet BC
  mfem::Array<int> colsA(_Wa.Height());
  colsA = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  _Wa.EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _Wa.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }

}


// Main diagonal of wave equation Fa*Ma^-1*Fa + dt^2 B/mu0 Aa
void AssembleCp( FiniteElementSpace *_AhFESpace, const Array<int>& _essAhTDOF,
                 double _dt, double _mu0, const Vector& a,
                 const SparseMatrix& _Fa, const SparseMatrix& _Ma, const SparseMatrix& _Aa,
                 SparseMatrix& _Cp ){

  // Clear content of result matrix
  _Cp.Clear();

  // Compute ||B_0||_2, the L2 norm of the space-time average of the magnetic field B = ∇x(kA)
  GridFunction cFuncCoeff(_AhFESpace);
  cFuncCoeff = a;
  Vector B0(2); B0 = 0.;
  double _area = 0.;
  for (int i = 0; i < _AhFESpace->GetNE(); i++){
    const FiniteElement *fe = _AhFESpace->GetFE(i);
    ElementTransformation *Tr = _AhFESpace->GetElementTransformation(i);
    int intorder = 2*fe->GetOrder() + 3;
    const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));
    Array<int> dofs;
    _AhFESpace->GetElementDofs(i, dofs);
    for (int j = 0; j < ir->GetNPoints(); j++){
      const IntegrationPoint &ip = ir->IntPoint(j);
      Tr->SetIntPoint(&ip);
      Vector grad;
      cFuncCoeff.GetGradient(*Tr,grad);
      // NB: this will contain \int_{\Omega}[Ax,Ay] dx ...
      grad *= ip.weight * Tr->Weight();
      B0 += grad;
    }
    _area += Tr->Weight();
  }
  _area *= 0.5;
  double B0norm2 = ( B0(0)*B0(0) + B0(1)*B0(1) ) / _area; 

  // Assemble operator
  Vector MDiagInv;
  _Ma.GetDiag( MDiagInv );
  for ( int i = 0; i < MDiagInv.Size(); ++i ){  // invert it
    MDiagInv(i) = 1./MDiagInv(i);
  }
  SparseMatrix MaLinv(MDiagInv);
  SparseMatrix *MiF  = Mult(MaLinv,_Fa);
  SparseMatrix *FMiF = Mult(_Fa,*MiF);
  _Cp = *FMiF;
  delete MiF;
  delete FMiF;
  _Cp.Add( _dt*_dt*B0norm2/_mu0, _Aa );
  // - impose Dirichlet BC
  mfem::Array<int> colsA(_Cp.Height());
  colsA = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  _Cp.EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _Cp.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE );
  }
}





//*************************************************************************
// ASSEMBLE BLOCK OPERATORS
//*************************************************************************
// Assembles lower factor of LU factorisation of velocity/magnetic field part of preconditioner
//       ⌈    I         ⌉
// Lub = |         I    |
//       ⌊ Y*Fu^-1    I ⌋
void AssembleLub( const Operator* Y,  const Operator* Fui,
                  BlockLowerTriangularPreconditioner* Lub ){

  Array<const Operator*> YFuiOps(2);
  YFuiOps[0] = Fui;
  YFuiOps[1] = Y;
  OperatorsSequence* YFui = new OperatorsSequence( YFuiOps );   // does not own

  Lub->iterative_mode = false;
  Lub->SetBlock( 2, 0,       YFui );
  Lub->owns_blocks = true;
}


// Assembles modified upper factor of LU factorisation of velocity/magnetic field part of preconditioner
//     ⌈ Fu^-1     ⌉   ⌈ I   Z  ⌉
// Uub*|       I   | = |   I    |, with Z = Z2 - Z1*Mzi*K
//     ⌊         I ⌋   ⌊     aS ⌋
void AssembleUub( const Operator* Z, const Operator* aSi,
                  BlockUpperTriangularPreconditioner* Uub ){
  Uub->iterative_mode = false;

  Uub->SetBlock( 0, 2, Z   );
  Uub->SetBlock( 2, 2, aSi );
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
//       ⌈ Fu Bt     ⌉
// Uup = |    pS     |
//       |       I   |
//       ⌊         I ⌋
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* pSi, BlockUpperTriangularPreconditioner* Uup ){
  Uup->iterative_mode = false;
  Uup->SetBlock( 0, 0, Fui );
  Uup->SetBlock( 0, 1, Bt  );
  Uup->SetBlock( 1, 1, pSi );
  Uup->owns_blocks = false;
}






//***************************************************************************
//TEST CASES OF SOME ACTUAL RELEVANCE
//***************************************************************************
// Analytical test-case
// - define a perturbation to dirty initial guess
double perturbation(const Vector & x, const double t){
  double epsilon = 1.;
  double xx(x(0));
  double yy(x(1));
  return( t * epsilon * 0.25*( ( cos(2*M_PI*xx)-1 )*(cos(2*M_PI*yy)-1) ) );
}
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
  if ( yy == 0. || yy == 1.){ //N/S
    n(0) =    xx*xx;
    n(1) = -2*xx*yy;
    if ( yy == 0. ){
      n*=-1.;
    }
  }
  if ( xx == 0. || xx == 1. ){ //E/W
    n(0) = 2*xx*yy;
    n(1) = -yy*yy;
    if ( xx == 0. ){
      n*=-1.;
    }
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
void uFun_ex_island(const Vector & x, const double t, Vector & u){
  u = 0.;
}
// - pressure
double pFun_ex_island(const Vector & x, const double t ){
  using namespace IslandCoalescenceData;
  double xx(x(0));
  double yy(x(1));

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( P0 + ( 1. - epsilon*epsilon ) / ( 2.*temp*temp ) );
}
// - laplacian of vector potential
double zFun_ex_island(const Vector & x, const double t ){
  using namespace IslandCoalescenceData;
  double xx(x(0));
  double yy(x(1));

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( ( 1. - epsilon*epsilon ) / ( delta * temp*temp ) );
}
// - vector potential
double aFun_ex_island(const Vector & x, const double t ){
  using namespace IslandCoalescenceData;
  double xx(x(0));
  double yy(x(1));
  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  if ( t==0 ){
    return ( delta * log( temp ) + 0.001*cos(M_PI*.5*yy)*cos(M_PI*xx) );  // perturb IC
  }
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
// - rhs of vector potential
double hFun_island( const Vector & x, const double t ){
  using namespace IslandCoalescenceData;
  return - eta/mu0 * zFun_ex_island(x,t);
}
// - normal derivative of vector potential
double mFun_island( const Vector & x, const double t ){
  using namespace IslandCoalescenceData;
  double xx(x(0));
  double yy(x(1));

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);  

  if(xx == 1.0){
    return - epsilon*sin(xx/delta) / temp;
  }else if( xx == 0.0 ){
    return   epsilon*sin(xx/delta) / temp;
  }else if( yy ==  1.0 ){
    return   sinh(yy/delta) / temp;
  }else if( yy == 0.0 ){    // this way I also cover the case of a domain [0,1]x[0,1]
    return 0.;
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




// MHD Rayleigh flow
void uFun_ex_rayleigh(const Vector & x, const double t, Vector & u){
  using namespace RayleighData;
  double yy(x(1));

  u(0) = U/4. * (  exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) )  -  erf( (yy-A0*t)/(2*sqrt(d*t)) ) 
                 + exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) )  -  erf( (yy+A0*t)/(2*sqrt(d*t)) ) + 2. );
  u(1) = 0.;

}
// - pressure - unused?
double pFun_ex_rayleigh(const Vector & x, const double t ){
  return 0.;
}
// - laplacian of vector potential
double zFun_ex_rayleigh(const Vector & x, const double t ){
  double dh = 1e-8;

  // use centered differences
  Vector xp=x, xm=x;
  xp(1) += dh;
  xm(1) -= dh;
  return ( aFun_ex_rayleigh(xp,t) - 2*aFun_ex_rayleigh(x,t) + aFun_ex_rayleigh(xm,t) ) / (dh*dh);

  // z = - U/4.*sqrt(mu*rho/(M_PI*d*t)) * (-exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))*(yy-A0*t)/(2*d*t) * (yy-A0*t)          +  exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))
  //                                       +exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t))*(yy+A0*t)/(2*d*t) * (yy+A0*t)          -  exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
  //     + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) *(A0*t-yy)/(2*d*t)
  //                                                 - exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) *(A0*t+yy)/(2*d*t) ) / (2*sqrt(d*t))
  //     - U/4.*sqrt(mu*rho)/A0 *( -A0/d*exp(-A0*yy/d) * ( ( -(A0+A0*A0*exp(A0*yy/d)*yy/d) + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
  //                                                        + (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) ) )
  //     - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * ( ( -( A0*A0/d*( A0*exp(A0*yy/d)/d*yy + exp(A0*yy/d) ) ) 
  //                                                 +  A0*( A0*exp(A0*yy/d)/d + A0/d*exp(A0*yy/d) + A0*A0/d*exp(A0*yy/d)/d*yy ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
  //                      ( -(A0+A0*A0*exp(A0*yy/d)*yy/d)  + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) ) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/(2*sqrt(d*t)) 
  //                                                   - (d+A0*exp(A0*yy/d)*yy) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))  * (yy-A0*t)/(2*d*t)    /sqrt(d*t*M_PI) 
  //                                                   + (A0*A0*exp(A0*yy/d)/d*yy + A0*exp(A0*yy/d)) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/sqrt(d*t*M_PI) 
  //                                                   )
  //     THIS IS DONE
  //     - U/4.*sqrt(mu*rho)/A0 * (   (A0*exp(A0*yy/d)-A0)   * 2/sqrt(M_PI)*exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) / (2*sqrt(d*t))
  //                                + (A0*A0*exp(A0*yy/d)/d) * erfc((A0*t+yy)/(2*sqrt(d*t)))
  //                                + (A0*exp(A0*yy/d)-A0)   * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) 
  //                                - (d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) * 2*(A0*t+yy)/(4*d*t) /(2*sqrt(d*t)) );
}
// - vector potential
double aFun_ex_rayleigh(const Vector & x, const double t ){
  using namespace RayleighData;
  double xx(x(0));
  double yy(x(1));

  double a = -B0*xx + U/2.*sqrt(d*t*mu*rho/M_PI) * ( exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) - exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
                    + U/4.*sqrt(mu*rho)/A0 * (d+A0*A0*t) * ( erf((A0*t-yy)/(2*sqrt(d*t))) - erf((A0*t+yy)/(2*sqrt(d*t))) )
                    - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * erfc((yy-A0*t)/(2*sqrt(d*t)))
                    - U/4.*sqrt(mu*rho)/A0 * (d*exp(A0*yy/d)-A0*yy) * erfc((A0*t+yy)/(2*sqrt(d*t)));
  return a;
}
// - rhs of velocity - unused
void fFun_rayleigh(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// - normal derivative of velocity
void nFun_rayleigh(const Vector & x, const double t, Vector & n){
  // using namespace RayleighData;
  // double xx(x(0));
  // double yy(x(1));
  n = 0.;

  // if( yy==0 || yy==5 )
  //   return;
  
  // if( xx==0 || xx==5 ){
  //   n(0) = U/4. * (  -A0/d*exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp(-A0*yy/d)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) )/(2*sqrt(d*t))
  //                    -  1./sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) ) /(2*sqrt(d*t)) 
  //                    +A0/d*exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp( A0*yy/d)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) )/(2*sqrt(d*t))
  //                    -  1./sqrt(M_PI)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) ) /(2*sqrt(d*t)) );
  //   if (xx==0)
  //     n *= -1.;
  // }

}
// - rhs of pressure - unused
double gFun_rayleigh(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential - unused
double hFun_rayleigh( const Vector & x, const double t ){
  using namespace RayleighData;
  return B0*U/2.;
}
// - normal derivative of vector potential
double mFun_rayleigh( const Vector & x, const double t ){
  using namespace RayleighData;
  double xx(x(0));
  double yy(x(1));

  double m = 0.;

  if ( xx==0)
    m = B0;
  if ( xx==5)
    m = -B0;
  if ( yy==0 || yy==5 ){
    m =   U/2.*sqrt(d*t*mu*rho/M_PI) * ( exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * (-(yy-A0*t)/(2*d*t)) - exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) * (-(yy+A0*t)/(2*d*t)) )
        + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) + exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) ) / (2*sqrt(d*t))
        - U/4.*sqrt(mu*rho)/A0 * ( -A0/d * exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * erfc((yy-A0*t)/(2*sqrt(d*t)))
                                  + exp(-A0*yy/d) * (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) * erfc((yy-A0*t)/(2*sqrt(d*t)))
                                  + exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) )
        - U/4.*sqrt(mu*rho)/A0 * ( (A0*exp(A0*yy/d)-A0) * erfc((A0*t+yy)/(2*sqrt(d*t)))
                                  +(d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) );
    if( yy==0 )
      m *= -1.;
  }

  return m;
}
// - define perturbed IG for every linearised variable
void wFun_rayleigh(const Vector & x, const double t, Vector & w){
  uFun_ex_rayleigh(x,t,w);
}
double qFun_rayleigh(  const Vector & x, const double t ){
  return pFun_ex_rayleigh(x,t);
}
double yFun_rayleigh(  const Vector & x, const double t ){
  return zFun_ex_rayleigh(x,t);
}
double cFun_rayleigh(  const Vector & x, const double t ){
  return aFun_ex_rayleigh(x,t);
}















//***************************************************************************
// OUTPUT
//***************************************************************************
void SaveSolution( const Array<BlockVector>& sol,
                   const Array< FiniteElementSpace* >& feSpaces, double _dt, Mesh* _mesh,
                   const std::string& path="ParaView", const std::string& filename="STIMHD2D" ){
  // handy functions which will contain solution at single time-steps
  GridFunction *uFun = new GridFunction( feSpaces[0] );
  GridFunction *pFun = new GridFunction( feSpaces[1] );
  GridFunction *zFun = new GridFunction( feSpaces[2] );
  GridFunction *aFun = new GridFunction( feSpaces[3] );

  // set up paraview data file
  ParaViewDataCollection paraviewDC( filename, _mesh );
  paraviewDC.SetPrefixPath(path);
  paraviewDC.SetLevelsOfDetail( 2 );
  paraviewDC.SetDataFormat(VTKFormat::BINARY);
  paraviewDC.SetHighOrderOutput(true);
  // - link wFun, pFun and vFun
  paraviewDC.RegisterField( "u", uFun );
  paraviewDC.RegisterField( "p", pFun );
  paraviewDC.RegisterField( "z", zFun );
  paraviewDC.RegisterField( "A", aFun );

  // main time loop
  for ( int t = 0; t < sol.Size(); ++t ){      
    // - assign to linked variables
    *uFun = sol[t].GetBlock(0);
    *pFun = sol[t].GetBlock(1);
    *zFun = sol[t].GetBlock(2);
    *aFun = sol[t].GetBlock(3);
    
    // - store
    paraviewDC.SetCycle( t );
    paraviewDC.SetTime( _dt*t );
    paraviewDC.Save();

  }

  delete uFun;
  delete pFun;
  delete zFun;
  delete aFun;

}





// Saves a plot of the error
void SaveError( const Array<BlockVector>& sol,
                const Array< FiniteElementSpace* >& feSpaces,
                void(  *uFun_ex)( const Vector & x, const double t, Vector & u ),
                double(*pFun_ex)( const Vector & x, const double t             ),
                double(*zFun_ex)( const Vector & x, const double t             ),
                double(*aFun_ex)( const Vector & x, const double t             ),
                double _dt, Mesh* _mesh,
                const std::string& path="ParaView", const std::string& filename="STIMHD2D_err" ){

  // handy functions which will contain solution at single time-steps
  GridFunction *uFun = new GridFunction( feSpaces[0] );
  GridFunction *pFun = new GridFunction( feSpaces[1] );
  GridFunction *zFun = new GridFunction( feSpaces[2] );
  GridFunction *aFun = new GridFunction( feSpaces[3] );

  // set up paraview data file
  ParaViewDataCollection paraviewDC( filename, _mesh );
  paraviewDC.SetPrefixPath(path);
  paraviewDC.SetLevelsOfDetail( 2 );
  paraviewDC.SetDataFormat(VTKFormat::BINARY);
  paraviewDC.SetHighOrderOutput(true);
  // - link wFun, pFun and vFun
  paraviewDC.RegisterField( "u-uh", uFun );
  paraviewDC.RegisterField( "p-ph", pFun );
  paraviewDC.RegisterField( "z-zh", zFun );
  paraviewDC.RegisterField( "A-Ah", aFun );


  // these will provide exact solution
  VectorFunctionCoefficient uFuncCoeff(2,uFun_ex);
  FunctionCoefficient       pFuncCoeff(  pFun_ex);
  FunctionCoefficient       zFuncCoeff(  zFun_ex);
  FunctionCoefficient       aFuncCoeff(  aFun_ex);

  // error at instant 0 is 0 (IC)
  *uFun = 0.;
  *pFun = 0.;
  *zFun = 0.;
  *aFun = 0.;
  paraviewDC.SetCycle( 0 );
  paraviewDC.SetTime( 0.0 );
  paraviewDC.Save();

  // main time loop
  for ( int t = 1; t < sol.Size(); ++t ){      
    
    // - assign to linked variables
    uFuncCoeff.SetTime( _dt*t );
    pFuncCoeff.SetTime( _dt*t );
    zFuncCoeff.SetTime( _dt*t );
    aFuncCoeff.SetTime( _dt*t );
    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    zFun->ProjectCoefficient( zFuncCoeff );
    aFun->ProjectCoefficient( aFuncCoeff );
    uFun->operator-=( sol[t].GetBlock(0) );
    pFun->operator-=( sol[t].GetBlock(1) );
    zFun->operator-=( sol[t].GetBlock(2) );
    aFun->operator-=( sol[t].GetBlock(3) );
    
    // - store
    paraviewDC.SetCycle( t );
    paraviewDC.SetTime( _dt*t );
    paraviewDC.Save();

  }

  delete uFun;
  delete pFun;
  delete zFun;
  delete aFun;

}

