// Test file to check correctness of implementation of time-stepper and
// preconditioner when using reduced system (without z)
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "testcases.hpp"
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

  int output = 0;
  int precType = 2;

  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;

  int NT = 2;
  int pbType = 5;
  string pbName;
  bool stab = false;
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
  args.AddOption(&precType, "-P", "--preconditioner",
                "Type of preconditioner: 0-Cyr et al: Uupi*Lupi*Uubi*Lubi\n"
                "                        1-Cyr et al simplified: Uupi*Lupi*Uubi\n"
                "                        2-Cyr et al uber simplified: Uupi*Uubi (default)\n"
        );
  args.AddOption(&output, "-out", "--output",
                "Print paraview solution\n"
        );
  args.AddOption(&stab, "-S", "--stab", "-noS", "-noStab",
                "Stabilise via SUPG (default: false)\n"
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
  const double  newtonRTol = 1e-4;
  const double  newtonATol = 0.;

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
      mesh_file = Analytical4Data::_meshFile;
      pbName    = Analytical4Data::_pbName;

      uFun = Analytical4Data::uFun_ex;
      pFun = Analytical4Data::pFun_ex;
      zFun = Analytical4Data::zFun_ex;
      aFun = Analytical4Data::aFun_ex;
      fFun = Analytical4Data::fFun;
      gFun = Analytical4Data::gFun;
      hFun = Analytical4Data::hFun;
      nFun = Analytical4Data::nFun;
      mFun = Analytical4Data::mFun;
      wFun = Analytical4Data::wFun;
      qFun = Analytical4Data::qFun;
      yFun = Analytical4Data::yFun;
      cFun = Analytical4Data::cFun;

      _mu  = Analytical4Data::_mu;
      _eta = Analytical4Data::_eta;
      _mu0 = Analytical4Data::_mu0;
      
      Analytical4Data::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }
    // Kelvin-Helmholtz instability
    case 5:{
      mesh_file = KHIData::_meshFile;
      pbName    = KHIData::_pbName;

      uFun = KHIData::uFun_ex;
      pFun = KHIData::pFun_ex;
      zFun = KHIData::zFun_ex;
      aFun = KHIData::aFun_ex;
      fFun = KHIData::fFun;
      gFun = KHIData::gFun;
      hFun = KHIData::hFun;
      nFun = KHIData::nFun;
      mFun = KHIData::mFun;
      wFun = KHIData::wFun;
      qFun = KHIData::qFun;
      yFun = KHIData::yFun;
      cFun = KHIData::cFun;

      _mu  = KHIData::_mu;
      _eta = KHIData::_eta;
      _mu0 = KHIData::_mu0;
      
      KHIData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }
    // Island coalescence
    case 6:{
      mesh_file = IslandCoalescenceData::_meshFile;
      pbName    = IslandCoalescenceData::_pbName;

      uFun = IslandCoalescenceData::uFun_ex;
      pFun = IslandCoalescenceData::pFun_ex;
      zFun = IslandCoalescenceData::zFun_ex;
      aFun = IslandCoalescenceData::aFun_ex;
      fFun = IslandCoalescenceData::fFun;
      gFun = IslandCoalescenceData::gFun;
      hFun = IslandCoalescenceData::hFun;
      nFun = IslandCoalescenceData::nFun;
      mFun = IslandCoalescenceData::mFun;
      wFun = IslandCoalescenceData::wFun;
      qFun = IslandCoalescenceData::qFun;
      yFun = IslandCoalescenceData::yFun;
      cFun = IslandCoalescenceData::cFun;

      _mu  = IslandCoalescenceData::_mu;
      _eta = IslandCoalescenceData::_eta;
      _mu0 = IslandCoalescenceData::_mu0;
      
      IslandCoalescenceData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }
    // Tearing mode
    case 9:{
      mesh_file = TearingModeData::_meshFile;
      pbName    = TearingModeData::_pbName;

      uFun = TearingModeData::uFun_ex;
      pFun = TearingModeData::pFun_ex;
      zFun = TearingModeData::zFun_ex;
      aFun = TearingModeData::aFun_ex;
      fFun = TearingModeData::fFun;
      gFun = TearingModeData::gFun;
      hFun = TearingModeData::hFun;
      nFun = TearingModeData::nFun;
      mFun = TearingModeData::mFun;
      wFun = TearingModeData::wFun;
      qFun = TearingModeData::qFun;
      yFun = TearingModeData::yFun;
      cFun = TearingModeData::cFun;

      _mu  = TearingModeData::_mu;
      _eta = TearingModeData::_eta;
      _mu0 = TearingModeData::_mu0;
      
      TearingModeData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }
    // Tearing mode flipped
    case 10:{
      mesh_file = TearingModeFlippedData::_meshFile;
      pbName    = TearingModeFlippedData::_pbName;

      uFun = TearingModeFlippedData::uFun_ex;
      pFun = TearingModeFlippedData::pFun_ex;
      zFun = TearingModeFlippedData::zFun_ex;
      aFun = TearingModeFlippedData::aFun_ex;
      fFun = TearingModeFlippedData::fFun;
      gFun = TearingModeFlippedData::gFun;
      hFun = TearingModeFlippedData::hFun;
      nFun = TearingModeFlippedData::nFun;
      mFun = TearingModeFlippedData::mFun;
      wFun = TearingModeFlippedData::wFun;
      qFun = TearingModeFlippedData::qFun;
      yFun = TearingModeFlippedData::yFun;
      cFun = TearingModeFlippedData::cFun;

      _mu  = TearingModeFlippedData::_mu;
      _eta = TearingModeFlippedData::_eta;
      _mu0 = TearingModeFlippedData::_mu0;
      
      TearingModeFlippedData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }
    // Driven cavity flow
    case 11:{
      mesh_file = CavityDrivenData::_meshFile;
      pbName    = CavityDrivenData::_pbName;

      uFun = CavityDrivenData::uFun_ex;
      pFun = CavityDrivenData::pFun_ex;
      zFun = CavityDrivenData::zFun_ex;
      aFun = CavityDrivenData::aFun_ex;
      fFun = CavityDrivenData::fFun;
      gFun = CavityDrivenData::gFun;
      hFun = CavityDrivenData::hFun;
      nFun = CavityDrivenData::nFun;
      mFun = CavityDrivenData::mFun;
      wFun = CavityDrivenData::wFun;
      qFun = CavityDrivenData::qFun;
      yFun = CavityDrivenData::yFun;
      cFun = CavityDrivenData::cFun;

      _mu  = CavityDrivenData::_mu;
      _eta = CavityDrivenData::_eta;
      _mu0 = CavityDrivenData::_mu0;
      
      CavityDrivenData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }

    default:
      std::cerr<<"ERROR: Problem type "<<pbType<<" not recognised."<<std::endl;
  }

  args.PrintOptions(cout);
  std::cout<<"   --dt   "<<_dt<<std::endl;
  std::cout<<"   --mu   "<<_mu<<std::endl;
  std::cout<<"   --eta  "<<_eta<<std::endl;
  std::cout<<"   --mu0  "<<_mu0<<std::endl;
  std::cout<<"   --Pb   "<<pbName<<std::endl;
  std::cout<<"   --mesh "<<mesh_file<<std::endl;



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

  // - initialise paraview output
  string outFilePath = "ParaView";
  string outFileName = "STIMHD2D_" + pbName;
  // -- GridFunction representing solution
  GridFunction uGF( _UhFESpace );
  GridFunction pGF( _PhFESpace );
  GridFunction zGF( _ZhFESpace );
  GridFunction aGF( _AhFESpace );
  // -- set up paraview data file
  ParaViewDataCollection paraviewDC( outFileName, _mesh );
  paraviewDC.SetPrefixPath(outFilePath);
  paraviewDC.SetLevelsOfDetail( 2 );
  paraviewDC.SetDataFormat(VTKFormat::BINARY);
  paraviewDC.SetHighOrderOutput(true);
  // -- link wFun, pFun and vFun
  paraviewDC.RegisterField( "u", &uGF );
  paraviewDC.RegisterField( "p", &pGF );
  paraviewDC.RegisterField( "z", &zGF );
  paraviewDC.RegisterField( "A", &aGF );





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
    uGF.ProjectCoefficient( uFuncCoeff );
    pGF.ProjectCoefficient( pFuncCoeff );
    zGF.ProjectCoefficient( zFuncCoeff );
    aGF.ProjectCoefficient( aFuncCoeff );
    // this is used to evaluate Dirichlet nodes
    sol[tt].GetBlock(0) = uGF;
    sol[tt].GetBlock(1) = pGF;
    sol[tt].GetBlock(2) = zGF;
    sol[tt].GetBlock(3) = aGF;
  }

  // Define integration rule to be used throughout
  Array<int> ords(3);
  if ( !_stab ){
    ords[0] = 2*ordU + ordU-1;         // ( (u·∇)u, v )
    ords[1] =   ordU + ordA-1 + ordZ;  // (   z ∇A, v )
    ords[2] =   ordU + ordA-1 + ordA;  // ( (u·∇A), B )
  }else{
    ords[0] = 2*ordU + 2*(ordU-1);                 // ( (u·∇)u, (w·∇)v )
    ords[1] =   ordU +    ordU-1 + ordA-1 + ordZ;  // (   z ∇A, (w·∇)v )
    ords[2] = 2*ordU + 2*(ordA-1);                 // ( (u·∇A),  w·∇B )    
  }
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
  SparseMatrix Mz = *( dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(2,2) ) );
  SparseMatrix K  = *( dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(2,3) ) );
  // - inverse of mass matrix for z
  PetscParMatrix MzPetsc( &Mz );
  PetscLinearSolver Mzi( MzPetsc, "ZSolverMass_" );


  // Pressure Schur complement operators
  SparseMatrix Ap;
  SparseMatrix Mp;
  SparseMatrix Fp;
  AssembleAp( _PhFESpace, _essPhTDOF, ir, Ap );
  AssembleMp( _PhFESpace, _essPhTDOF, ir, Mp );
  // // - assembling each component explicitly (implemented later)
  // PetscParMatrix ApPetsc( &Ap );
  // PetscLinearSolver Api( ApPetsc, "PSolverLaplacian_" );
  // PetscParMatrix MpPetsc( &Mp );
  // PetscLinearSolver Mpi( MpPetsc, "PSolverMass_" );
  // - using own Schur complement class
  OseenSTPressureSchurComplement pSi( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essPhTDOF, verbose );
  SparseMatrix Wp;
  pSi.SetAp( &Ap );
  pSi.SetMp( &Mp );


  // Magnetic Schur complement operators
  SparseMatrix Aa;
  SparseMatrix Ma;
  SparseMatrix Cp;
  AssembleAa( _AhFESpace, ir, Aa );
  AssembleMa( _AhFESpace, _essAhTDOF, ir, Ma );
  // // - assembling each component explicitly (implemented later)
  // PetscParMatrix MaPetsc( &Ma );
  // PetscLinearSolver Mai( MaPetsc, "ASolverMass_" );
  // - using own Schur complement class
  IMHD2DSTMagneticSchurComplement aSi( MPI_COMM_SELF, _dt, NULL, NULL, NULL, _essAhTDOF, verbose );
  SparseMatrix Wa;
  aSi.SetM( &Ma );




  //*************************************************************************
  // TIME-STEPPING ROUTINE
  //*************************************************************************

  int GMRESNonConv     = 0;  
  int newtNonConv      = 0;  
  double tottotGMRESIt = 0;  //leave it as double to avoid round-off when averaging
  double totNewtonIt   = 0;  

  // print to paraview
  if ( output ){
    // -- assign to linked variables
    uGF = sol[0].GetBlock(0);
    pGF = sol[0].GetBlock(1);
    zGF = sol[0].GetBlock(2);
    aGF = sol[0].GetBlock(3);
    // -- store
    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0. );
    paraviewDC.Save();
  }



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
    double newtonErrWRTPrevIt = newtonATol;
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
                     && newtonRes >= newtonATol
                     && newtonRes/newtonRes0 >= newtonRTol;
                     // && newtonErrWRTPrevIt >= newtonTol;
                     ++newtonIt ){      

      // Define internal (GMRES) solver
      // - Get gradient and define relevant operators
      BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( lclSol ) );
      SparseMatrix Fu = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,0) ) );
      SparseMatrix Z1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) ) );
      SparseMatrix Z2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) ) );
      SparseMatrix Bt = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,1) ) );
      SparseMatrix B  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,0) ) );
      SparseMatrix Cp = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,1) ) );
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
      MHDOp.SetBlock( 1, 1, &Cp );
      MHDOp.SetBlock( 2, 0, &Y  );
      MHDOp.SetBlock( 2, 2, &Fa );

      
      // Define preconditioner
      // - Velocity block inverse
      PetscParMatrix FuPetsc( &Fu );
      PetscLinearSolver Fui( FuPetsc, "VSolver_" );

      // - Approximate pressure Schur complement inverse
      // -- Using own class
      bool isQuietState = true;
      for ( int i = 0; i < lclSol.GetBlock(0).Size(); ++i ){
        if ( lclSol.GetBlock(0)[i] != 0. ){
          isQuietState = false;
          break;
        }
      }
      if ( isQuietState ){
        AssembleWp( _PhFESpace, _essPhTDOF, neuBdrP, _mu, _UhFESpace, lclSol.GetBlock(0), ir, bir, Wp );
        pSi.SetWp( &Wp, isQuietState );
      }
      // // -- Assembling each component explicitly
      // AssembleFp( _PhFESpace, _essPhTDOF, neuBdrP, _dt, _mu, _UhFESpace, lclSol.GetBlock(0), ir, bir, Fp );
      // Array<const Operator*> pSiops(3);
      // Array<bool>            pSiown(3);
      // pSiops[0] = &Api;      pSiown[0] = false;
      // pSiops[1] = &Fp;       pSiown[1] = false;
      // pSiops[2] = &Mpi;      pSiown[2] = false;
      // OperatorsSequence pSi( pSiops, pSiown );
      



      // - Approximate magnetic Schur complement inverse
      // -- Using own class
      AssembleWa( _AhFESpace, _essAhTDOF, _eta, _mu0, _UhFESpace, lclSol.GetBlock(0), ir, Wa );
      AssembleCp( _AhFESpace, _essAhTDOF, _dt,  _mu0, lclSol.GetBlock(3), Fa, Ma, Aa, Cp );
      // --- Inverse of main diagonal of wave equation
      PetscParMatrix CpPetsc( &Cp );
      PetscLinearSolver CCi( CpPetsc, "AWaveSolver_" );
      aSi.SetW( &Wa );
      aSi.SetCCinv( &CCi );
      // // -- Assembling each component explicitly
      // AssembleCp( _AhFESpace, _essAhTDOF, _dt,  _mu0, lclSol.GetBlock(3), Fa, Ma, Aa, Cp );
      // PetscParMatrix CpPetsc( &Cp );
      // PetscLinearSolver CCi( CpPetsc, "AWaveSolver_" );
      // Array<const Operator*> aSiops(3);
      // Array<bool>            aSiown(3);
      // aSiops[0] = &Mai;      aSiown[0] = false;
      // aSiops[1] = &Fa;       aSiown[1] = false;
      // aSiops[2] = &CCi;      aSiown[2] = false;
      // OperatorsSequence aSi( aSiops, aSiown );



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
      Array<const Operator*> precOps;
      Array<bool>            precOwn;
      switch (precType){
        // Full Preconditioner Uupi*Lupi*Uubi*Lubi
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

      // BlockVector tmp( offsetsReduced );
      // MHDOp.Mult( lclDeltaSolReduced, tmp );
      // tmp -= lclResReduced;
      // std::cout<<"Residual norm: "<<tmp.Norml2()<<std::endl;

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


    // print to paraview
    if ( output ){
      // -- assign to linked variables
      uGF = sol[tt].GetBlock(0);
      pGF = sol[tt].GetBlock(1);
      zGF = sol[tt].GetBlock(2);
      aGF = sol[tt].GetBlock(3);
      // -- store
      paraviewDC.SetCycle( tt );
      paraviewDC.SetTime( _dt*tt );
      paraviewDC.Save();
    }

  }

  std::cout<<"***********************************************************\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"\nALL DONE!\n\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"Total # of Newton iterations: "<< totNewtonIt   <<". Non-converged: "<< newtNonConv  <<". Avg per time-step: "<< totNewtonIt/NT   <<"\n";
  std::cout<<"Total # of GMRES  iterations: "<< tottotGMRESIt <<". Non-converged: "<< GMRESNonConv <<". Avg per time-step: "<< tottotGMRESIt/NT
                                                                                                   <<". Avg per Newton it: "<< tottotGMRESIt/totNewtonIt <<"\n";
  std::cout<<"***********************************************************\n";


 
  // SaveSolution( sol, feSpaces, _dt, _mesh, outFilePath, outFileName );

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

