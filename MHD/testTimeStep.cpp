// Test file to check correctness of implementation of time-stepper and
// preconditioner
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "testcases.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include "operatorssequence.hpp"
#include "operatorsseries.hpp"
#include "imhd2dspaceintegrator.hpp"
#include "imhd2dtimeintegrator.hpp"
#include "imhd2dstmagneticschurcomplement.hpp"
#include "oseenstpressureschurcomplement.hpp"
// #include "oseenstpressureschurcomplementsimple.hpp"
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
void AssembleApAug( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                    const IntegrationRule *ir, SparseMatrix& _Ap );
void AssembleBMuBt( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                    FiniteElementSpace *_UhFESpace, const Array<int>& _essUhTDOF,
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
                  const Operator* Z1,  const Operator* Mzi, BlockLowerTriangularPreconditioner* Lub );
void AssembleUub( const Operator* Z1,  const Operator* Z2, const Operator* Mzi, const Operator* K,
                  const Operator* aSi, BlockUpperTriangularPreconditioner* Uub );
void AssembleLup( const Operator* Fui, const Operator* B,   BlockLowerTriangularPreconditioner* Lup );
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* X1, const Operator* X2,
                  const Operator* pSi, BlockUpperTriangularPreconditioner* Uup );

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
void PrintMatrices( const std::string& filename,
       const SparseMatrix& _Fu,  const SparseMatrix& _Bt,  const SparseMatrix& _Z1, const SparseMatrix& _Z2, const SparseMatrix& _Mu,
       const SparseMatrix& _B ,  const SparseMatrix& _Cs,  const SparseMatrix& _X1, const SparseMatrix& _X2,
       const SparseMatrix& _Mz,  const SparseMatrix& _K,
       const SparseMatrix& _Y ,  const SparseMatrix& _Fa,  const SparseMatrix& _Ma,
       const SparseMatrix& _Mp,  const SparseMatrix& _Ap,  const SparseMatrix& _Wp, const SparseMatrix& _Mps,
       const SparseMatrix& _Aa,  const SparseMatrix& _Cp,  const SparseMatrix& _Wa,
       const Array<int>& _essUhTDOF, const Array<int>& _essPhTDOF, const Array<int>& _essAhTDOF  );
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

  bool outputSol = false;
  bool outputRes = false;
  int precType = 2;
  int output = 2;

  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;

  int NT = 2;
  int pbType = 6;
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
  args.AddOption(&ordU, "-oU", "--orderU",
                "Velocity space polynomial order (default: 2)");
  args.AddOption(&ordP, "-oP", "--orderP",
                "Pressure space polynomial order (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--orderZ",
                "Laplacian of vector potential space polynomial order (default: 1)");
  args.AddOption(&ordA, "-oA", "--orderA",
                "Vector potential space polynomial order (default: 2)");
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&Tend, "-T", "--Tend",
                "Final time (default: 1.0)");
  args.AddOption(&pbType, "-Pb", "--problem",
                "Problem: 1 to 4-Analytical test cases\n"
                "              5-Kelvin-Helmholtz instability\n"
                "              6-Island coalescensce (default)\n"
                "             11-Driven cavity flow\n"
        );
  args.AddOption(&precType, "-P", "--preconditioner",
                "Type of preconditioner: 0-Cyr et al: Uupi*Lupi*Uubi*Lubi\n"
                "                        1-Cyr et al simplified: Uupi*Lupi*Uubi\n"
                "                        2-Cyr et al uber simplified: Uupi*Uubi (default)\n"
        );
  args.AddOption(&stab, "-S", "--stab", "-noS", "--noStab",
                "Stabilise via SUPG (default: false)\n"
        );
  args.AddOption(&outputSol, "-outSol", "--outputSolution", "-noOutSol", "--noOutputSolution",
                "Print paraview solution\n"
        );
  args.AddOption(&outputRes, "-outRes", "--outputResidual", "-noOutRes", "--noOutputResidual",
                "Print paraview residual\n"
        );
  args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                "PetscOptions file to use: rc_STMHD         TODO! (direct (LU) solvers - remember to flag _SingAp if Ap is singular))\n"
                "                          rc_STMHD_approx2 (MG solvers for Ap,Fu, Cheb for Mp)\n"
                "                          rc_STMHD_FGMRES  (use FGMRES rather than GMRES for outer solver (useful if ST=2))\n");
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
  args.AddOption(&output, "-out", "--output",
                "Choose how much info to store on disk: 0  nothing\n"
        "                                               1 +#it to convergence\n"
        "                                               2 +residual evolution (default)\n"
        );
  args.AddOption(&NT, "-NT", "--timeSteps",
                "Number of time steps (default: 2)");

  args.Parse();


  const double _dt = ( Tend-T0 )/ NT;

  const int   maxNewtonIt  = 10;
  const double  newtonRTol = 0.;               // 1e-5;
  const double  newtonATol = 1e-10/sqrt(NT);   // 0;

  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);


  // Define problem
  // - boundary conditions
  Array<int> essTagsU(0);
  Array<int> essTagsV(0);
  Array<int> essTagsP(0);
  Array<int> essTagsA(0);

  MHDTestCaseSelector( pbType, 
                       uFun, pFun, zFun, aFun,
                       fFun, gFun, hFun, nFun, mFun,
                       wFun, qFun, yFun, cFun,
                       _mu, _eta, _mu0,
                       pbName, mesh_file,
                       essTagsU, essTagsV, essTagsP, essTagsA );


  args.PrintOptions(cout);
  std::cout<<"   --mu "<<_mu<<std::endl;
  std::cout<<"   --eta "<<_eta<<std::endl;
  std::cout<<"   --mu0 "<<_mu0<<std::endl;
  std::cout<<"   --Pb "<<pbName<<std::endl;
  std::cout<<"   --mesh "<<mesh_file<<std::endl;
  std::cout<<"   --dt "<<_dt<<std::endl;



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

  double hMin, hMax, rMin, rMax;
  _mesh->GetCharacteristics( hMin, hMax, rMin, rMax );
  std::cout<<"   --dx   "<<hMin<<", "<<hMax<<std::endl;


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
  std::cout << "Tags dir u      "; essBdrU.Print(mfem::out, essBdrU.Size() );
  std::cout << "Tags dir v      "; essBdrV.Print(mfem::out, essBdrV.Size() );
  std::cout << "Tags dir p      "; essBdrP.Print(mfem::out, essBdrP.Size() );
  std::cout << "Tags dir A      "; essBdrA.Print(mfem::out, essBdrA.Size() );
  std::cout << "***********************************************************\n";
  // std::cout << "Dir u      "; _essUhTDOF.Print(mfem::out, _essUhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "Dir p      "; _essPhTDOF.Print(mfem::out, _essPhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "Dir A      "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "***********************************************************\n";

  // - initialise paraview output solution
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

  // - initialise paraview output residual
  string outFilePathRes = "ParaView";
  string outFileNameRes = "STIMHD2D_" + pbName + "_res";
  // -- GridFunction representing residual
  GridFunction uresGF( _UhFESpace );
  GridFunction presGF( _PhFESpace );
  GridFunction zresGF( _ZhFESpace );
  GridFunction aresGF( _AhFESpace );
  // -- set up paraview data file
  ParaViewDataCollection paraviewDCRes( outFileNameRes, _mesh );
  paraviewDCRes.SetPrefixPath(outFilePathRes);
  paraviewDCRes.SetLevelsOfDetail( 2 );
  paraviewDCRes.SetDataFormat(VTKFormat::BINARY);
  paraviewDCRes.SetHighOrderOutput(true);
  // -- link wFun, pFun and vFun
  paraviewDCRes.RegisterField( "u_res", &uresGF );
  paraviewDCRes.RegisterField( "p_res", &presGF );
  paraviewDCRes.RegisterField( "z_res", &zresGF );
  paraviewDCRes.RegisterField( "A_res", &aresGF );

  // - initialise paraview output error
  string outFilePathErr = "ParaView";
  string outFileNameErr = "STIMHD2D_" + pbName + "_err";
  // -- GridFunction representing residual
  GridFunction uerrGF( _UhFESpace );
  GridFunction perrGF( _PhFESpace );
  GridFunction zerrGF( _ZhFESpace );
  GridFunction aerrGF( _AhFESpace );
  // -- set up paraview data file
  ParaViewDataCollection paraviewDCErr( outFileNameErr, _mesh );
  paraviewDCErr.SetPrefixPath(outFilePathErr);
  paraviewDCErr.SetLevelsOfDetail( 2 );
  paraviewDCErr.SetDataFormat(VTKFormat::BINARY);
  paraviewDCErr.SetHighOrderOutput(true);
  // -- link wFun, pFun and vFun
  paraviewDCErr.RegisterField( "u-uh", &uerrGF );
  paraviewDCErr.RegisterField( "p-ph", &perrGF );
  paraviewDCErr.RegisterField( "z-zh", &zerrGF );
  paraviewDCErr.RegisterField( "A-Ah", &aerrGF );




  //*************************************************************************
  // Time-independent operators assembly
  //*************************************************************************
  // Initialise solution
  Array< BlockVector > sol( NT + 1 );
  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = _UhFESpace->GetTrueVSize();
  offsets[2] = _PhFESpace->GetTrueVSize();
  offsets[3] = _ZhFESpace->GetTrueVSize();
  offsets[4] = _AhFESpace->GetTrueVSize();
  offsets.PartialSum();
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
  if ( !stab ){
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
  BlockVector lclSol = sol[0];    // this will contain the update on the solution at each time-step
  GridFunction _wGridFunc(  _UhFESpace, lclSol.GetBlock(0).GetData() ); // these will be used in the integrator, so link these variables to
  GridFunction _cGridFunc(  _AhFESpace, lclSol.GetBlock(3).GetData() ); //  lclSol once and for all, so that it gets updated automatically
  VectorGridFunctionCoefficient _wFuncCoeff( &_wGridFunc );             //  and store them in a handy decorator used in the integrator
  GridFunctionCoefficient       _cFuncCoeff( &_cGridFunc );
  Array< FiniteElementSpace* > feSpaces(4);
  feSpaces[0] = _UhFESpace;
  feSpaces[1] = _PhFESpace;
  feSpaces[2] = _ZhFESpace;
  feSpaces[3] = _AhFESpace;
  BlockNonlinearForm _IMHD2DOperator, _IMHD2DMassOperator; // operators for returning spatial and temporal (mass) part of MHD equations
  _IMHD2DOperator.SetSpaces( feSpaces );
  _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DSpaceIntegrator( _dt, _mu, _mu0, _eta, stab,
                                                                               &fFuncCoeff, &hFuncCoeff, &_wFuncCoeff, &_cFuncCoeff ) ); // for bilinforms
  _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DSpaceIntegrator( _dt, _mu, _mu0, _eta ), essBdrA );  // for flux in third equation
  _IMHD2DMassOperator.SetSpaces( feSpaces );
  _IMHD2DMassOperator.AddDomainIntegrator(  new IncompressibleMHD2DTimeIntegrator( _dt, _mu, _mu0, _eta, stab, &_wFuncCoeff, &_cFuncCoeff ) );

  std::cout<<"Warning: I'm not considering BC on pressure, and I'm imposing Neumann integral of velocity =0!"<<std::endl;
  Array< Array<int> * > tmpEssDofs(4);
  Array<int> tempPhTDOF(0); // Set all to 0: Dirichlet BC are never imposed on pressure: they are only used to assemble the pressure operators    
  Array<int> tempZhTDOF(0); // Set all to 0: Dirichlet BC are never imposed on laplacian of vector potential    
  tmpEssDofs[0] = &_essUhTDOF;
  tmpEssDofs[1] = &tempPhTDOF;
  // tmpEssDofs[1] = &_essPhTDOF;
  tmpEssDofs[2] = &tempZhTDOF;
  tmpEssDofs[3] = &_essAhTDOF;
  Array< Vector * > dummy(4); dummy = NULL;
  _IMHD2DOperator.SetEssentialTrueDofs(     tmpEssDofs, dummy );  dummy = NULL;
  _IMHD2DMassOperator.SetEssentialTrueDofs( tmpEssDofs, dummy );

  // - extract and keep some constant operators
  BlockOperator* dummyJ = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( sol[0] ) );
  SparseMatrix Mz = *(    dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(2,2) ) );
  SparseMatrix K  = *(    dynamic_cast<SparseMatrix*>( &dummyJ->GetBlock(2,3) ) );
  // - inverse of mass matrix for z
  PetscParMatrix MzPetsc( &Mz );
  PetscLinearSolver Mzi( MzPetsc, "ZSolverMass_" );


  // Pressure Schur complement operators
  SparseMatrix Ap;
  SparseMatrix Mp;
  AssembleApAug( _PhFESpace, _essPhTDOF, ir, Ap );
  // AssembleBMuBt( _PhFESpace, _essPhTDOF, _UhFESpace, _essUhTDOF, ir, Ap );
  // AssembleAp( _PhFESpace, _essPhTDOF, ir, Ap );
  AssembleMp( _PhFESpace, _essPhTDOF, ir, Mp );
  // // - assembling each component explicitly (implemented later)
  // PetscParMatrix ApPetsc( &Ap );
  // PetscLinearSolver Api( ApPetsc, "PSolverLaplacian_" );
  // PetscParMatrix MpPetsc( &Mp );
  // PetscLinearSolver Mpi( MpPetsc, "PSolverMass_" );
  // - using own simplified Schur complement class
  // OseenSTPressureSchurComplementSimple pSi( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essPhTDOF, verbose );
  // SparseMatrix Fp;
  // pSi.SetAp( &Ap );
  // pSi.SetMp( &Mp );
  // // - using own Schur complement class
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
  int totGMRESNonConv  = 0;  
  int newtNonConv      = 0;  
  double tottotGMRESIt = 0;  //leave it as double to avoid round-off when averaging
  double totNewtonIt   = 0;  

  double STresu = 0.;  // these will store the final space-time residual norm
  double STresp = 0.;
  double STresz = 0.;
  double STresa = 0.;
  double STres  = 0.;

  // print to paraview
  if ( outputSol ){
    // -- assign to linked variables
    uGF = sol[0].GetBlock(0);
    pGF = sol[0].GetBlock(1);
    zGF = sol[0].GetBlock(2);
    aGF = sol[0].GetBlock(3);
    // -- store
    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0. );
    paraviewDC.Save();
    if ( pbType <= 4 ){ // if analytical solution is available, print error, too
      uFuncCoeff.SetTime( T0 );
      pFuncCoeff.SetTime( T0 );
      zFuncCoeff.SetTime( T0 );
      aFuncCoeff.SetTime( T0 );
      uerrGF.ProjectCoefficient( uFuncCoeff );
      perrGF.ProjectCoefficient( pFuncCoeff );
      zerrGF.ProjectCoefficient( zFuncCoeff );
      aerrGF.ProjectCoefficient( aFuncCoeff );
      uerrGF -= sol[0].GetBlock(0);
      perrGF -= sol[0].GetBlock(1);
      zerrGF -= sol[0].GetBlock(2);
      aerrGF -= sol[0].GetBlock(3);
      // -- store
      paraviewDCErr.SetCycle( 0 );
      paraviewDCErr.SetTime( 0. );
      paraviewDCErr.Save();
    }

  }

  if ( outputRes ){
    // -- assign to linked variables
    uresGF = 0.;
    presGF = 0.;
    zresGF = 0.;
    aresGF = 0.;
    // -- store
    paraviewDCRes.SetCycle( 0 );
    paraviewDCRes.SetTime( 0. );
    paraviewDCRes.Save();
  }

  string convPath = string("./results/") + "Pb" + to_string(pbType) + "_Prec" + to_string(precType) + "_STsolveU9_STsolveA9"
                         + "_oU"   + to_string(ordU) + "_oP" + to_string(ordP) + "_oZ" + to_string(ordZ) + "_oA" + to_string(ordA)
                         + "_"     + petscrc_file + "/";
  if (!std::experimental::filesystem::exists( convPath ) && myid == 0){
    std::experimental::filesystem::create_directories( convPath );
  }
  string innerConvpath = convPath + "NT" + to_string(NT) + "_r"  + to_string(refLvl) + "/";
  if (output>0 && !std::experimental::filesystem::exists( innerConvpath ) && myid == 0){
    std::experimental::filesystem::create_directories( innerConvpath );
  }

  for ( int tt = 1; tt < NT+1; ++tt ){
    int GMRESNonConv = 0;  
    string innerInnerConvpath = innerConvpath + "tt" + to_string(tt) + "/";
    if (output>0 && !std::experimental::filesystem::exists( innerInnerConvpath ) && myid == 0){
      std::experimental::filesystem::create_directories( innerInnerConvpath );
    }

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
    frhs.AddDomainIntegrator(   new VectorDomainLFIntegrator(         fFuncCoeff       )          );  //int_\Omega f*v
    // ScalarVectorProductCoefficient muNFuncCoeff(_mu, nFuncCoeff);
    // frhs.AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     muNFuncCoeff       ), neuBdrU );  //int_d\Omega \mu * du/dn *v
    // frhs.AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator(   pFuncCoeff, -1.0 ), neuBdrU );  //int_d\Omega -p*v*n
    frhs.GetDLFI()->operator[](0)->SetIntRule(ir);
    // frhs.GetBLFI()->operator[](0)->SetIntRule(bir);
    // frhs.GetBLFI()->operator[](1)->SetIntRule(bir);
    frhs.Assemble();
    frhs *= _dt;
    // // - effect from solution at previous iteration
    // LinearForm uPrevForm( _UhFESpace );
    // GridFunction uPrev( _UhFESpace ); uPrev = sol[tt-1].GetBlock(0);
    // VectorGridFunctionCoefficient uPrevFuncCoeff( &uPrev );
    // uPrevForm.AddDomainIntegrator( new VectorDomainLFIntegrator( uPrevFuncCoeff ) );  //int_\Omega u(t-1)*v
    // uPrevForm.GetDLFI()->operator[](0)->SetIntRule(ir);
    // uPrevForm.Assemble();
    // // - combine together
    // frhs += uPrevForm;

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
    hrhs *= _dt;
    // // - effect from solution at previous iteration
    // LinearForm aPrevForm( _AhFESpace );
    // GridFunction aPrev( _AhFESpace ); aPrev = sol[tt-1].GetBlock(3);
    // GridFunctionCoefficient aPrevFuncCoeff( &aPrev );
    // aPrevForm.AddDomainIntegrator( new DomainLFIntegrator( aPrevFuncCoeff ) );  //int_\Omega A(t-1)*B
    // aPrevForm.GetDLFI()->operator[](0)->SetIntRule(ir);
    // aPrevForm.Assemble();
    // // - combine together
    // hrhs += aPrevForm;

    // clean Dirichlet nodes from rhs (they are already solved for, and they shouldn't dirty the solution)
    frhs.SetSubVector( _essUhTDOF, 0. );
    // grhs.SetSubVector( _essPhTDOF, 0. );
    hrhs.SetSubVector( _essAhTDOF, 0. );


    // Initialise Newton iteration
    if ( tt==1 ){ // using provided initial guess, if first time-step
      wFuncCoeff.SetTime( T0 + _dt*tt );
      qFuncCoeff.SetTime( T0 + _dt*tt );
      yFuncCoeff.SetTime( T0 + _dt*tt );
      cFuncCoeff.SetTime( T0 + _dt*tt );
      uGF.ProjectCoefficient( wFuncCoeff );
      pGF.ProjectCoefficient( qFuncCoeff );
      zGF.ProjectCoefficient( yFuncCoeff );
      aGF.ProjectCoefficient( cFuncCoeff );
      lclSol.GetBlock(0) = uGF; 
      lclSol.GetBlock(1) = pGF;
      lclSol.GetBlock(2) = zGF;
      lclSol.GetBlock(3) = aGF;
    }else{        // using solution at previous iteration otherwise
      lclSol = sol[tt-1];
    }

    // - either case, remember to preserve Dirichlet nodes
    for ( int i = 0; i < _essUhTDOF.Size(); ++i )
      lclSol.GetBlock(0)(_essUhTDOF[i]) = sol[tt].GetBlock(0)(_essUhTDOF[i]);
    // for ( int i = 0; i < _essPhTDOF.Size(); ++i )
    //   lclSol.GetBlock(1)(_essPhTDOF[i]) = sol[tt].GetBlock(1)(_essPhTDOF[i]);
    for ( int i = 0; i < _essAhTDOF.Size(); ++i )
      lclSol.GetBlock(3)(_essAhTDOF[i]) = sol[tt].GetBlock(3)(_essAhTDOF[i]);


    // - compute residual
    BlockVector lclRes(offsets);
    if ( tt==1 ){
      // - at the beginnig, set z to be *exactly* the laplacian of A
      
      // TODO
      // So, Im observing terrible convergence if I dont set z by solving 
      //  Mz z = - K A - zrhs
      // but only at the very *first* time step! And I really have no clue why!!

      lclSol.GetBlock(2) = 0.;
      _IMHD2DOperator.Mult( lclSol, lclRes );
      Vector tempZ = lclRes.GetBlock(2);      // this is the "full" K*A, including effect from Dirichlet
      tempZ -= zrhs;
      tempZ.Neg();
      Mzi.Mult( tempZ, lclSol.GetBlock(2) );  // solve for z
      //  -- spatial contribution
      _IMHD2DOperator.Mult( lclSol, lclRes );        // N(x)
      lclRes.GetBlock(0) -= frhs;                    // N(x) - b
      lclRes.GetBlock(1) -= grhs;
      lclRes.GetBlock(2) -= zrhs;
      // lclRes.GetBlock(2)  = 0.;                      // should already be basically zero!
      lclRes.GetBlock(3) -= hrhs;
      lclRes.Neg();                                  // b - N(x)
    }else{
      //  -- spatial contribution
      _IMHD2DOperator.Mult( lclSol, lclRes );        // N(x)
      lclRes.GetBlock(0) -= frhs;                    // N(x) - b
      lclRes.GetBlock(1) -= grhs;
      lclRes.GetBlock(2) -= zrhs;
      lclRes.GetBlock(3) -= hrhs;
      lclRes.Neg();                                  // b - N(x)
    }
    //  -- temporal contribution
    BlockVector dtu(offsets), tempN(offsets);
    _IMHD2DMassOperator.Mult( lclSol,    dtu   );  // - u^{n+1}
    _IMHD2DMassOperator.Mult( sol[tt-1], tempN );  // - u^{n}
    dtu -= tempN;                                  // -(u^{n+1}-u^{n})
    //  -- combine together
    lclRes += dtu;
    // - Shouldnt be necessary (Dirichlet nodes are set to 0 when multiplying by _IMHD2D(Mass)Operator) but does no harm:
    for ( int i = 0; i < _essUhTDOF.Size(); ++i )
      lclRes.GetBlock(0)(_essUhTDOF[i]) = 0.;
    for ( int i = 0; i < _essAhTDOF.Size(); ++i )
      lclRes.GetBlock(3)(_essAhTDOF[i]) = 0.;




    // - compute norm of residual and initialise relevant quantities for Newton iteration
    int newtonIt = 0;
    double erru, errp, errz, erra, errtot;
    double newtonRes = lclRes.Norml2();
    double newtonRes0 = newtonRes;
    double newtonErrWRTPrevIt = newtonATol;
    double tempResu = 0.;  // these will store the current contribution to the full space-time residual
    double tempResp = 0.;
    double tempResz = 0.;
    double tempResa = 0.;
    double tempRes  = 0.;
    double totGMRESit = 0.; //leave it as double, so that when I'll average it, it won't round-off
    std::cout << "***********************************************************\n";
    std::cout << "Pb " << pbName <<". Starting Newton for time-step "<<tt<<", initial residual "<< newtonRes
              << ", (u,p,z,A) = ("<< lclRes.GetBlock(0).Norml2() <<","
                                  << lclRes.GetBlock(1).Norml2() <<","
                                  << lclRes.GetBlock(2).Norml2() <<","
                                  << lclRes.GetBlock(3).Norml2() <<")" << std::endl;
    if ( pbType<=4 ){ // if analytical solution is available, print info on error, too
      uFuncCoeff.SetTime( T0 + _dt*tt );
      pFuncCoeff.SetTime( T0 + _dt*tt );
      zFuncCoeff.SetTime( T0 + _dt*tt );
      aFuncCoeff.SetTime( T0 + _dt*tt );
      uerrGF.ProjectCoefficient( uFuncCoeff );
      perrGF.ProjectCoefficient( pFuncCoeff );
      zerrGF.ProjectCoefficient( zFuncCoeff );
      aerrGF.ProjectCoefficient( aFuncCoeff );
      uerrGF -= lclSol.GetBlock(0);
      perrGF -= lclSol.GetBlock(1);
      zerrGF -= lclSol.GetBlock(2);
      aerrGF -= lclSol.GetBlock(3);
      erru   = uerrGF.Norml2();
      errp   = perrGF.Norml2();
      errz   = zerrGF.Norml2();
      erra   = aerrGF.Norml2();
      errtot = sqrt(erru*erru + errp*errp + errz*errz + erra*erra);
      std::cout << "Initial error "<< errtot
                << ", (u,p,z,A) = ("<< erru <<","<< errp <<","<< errz <<","<< erra <<")" << std::endl;
    }
    std::cout << "***********************************************************\n";

    if( output>0 ){
      if ( tt==1 ){
        // initialise per time-step data collection
        string filename = innerConvpath +"NEWTconv.txt";
        ofstream myfile;
        myfile.open( filename, std::ios::app );
        myfile << "tt,\tRes_norm_tot\tRel_res_norm\tNewton_its,\tNewton_conv\tAvg_GMRES_its\tGMRES_noConv" << std::endl;    
        myfile << 0 << ",\t" << newtonRes0 << ",\t" << 1.0 << ",\t"
               << 0 << ",\t" << false      << ",\t" << 0   << ",\t" << 0  << std::endl;
        myfile.close();
      }

      // initialise data collection for specific time-step
      string filename = innerInnerConvpath +"NEWTconv.txt";
      ofstream myfile;
      myfile.open( filename, std::ios::app );
      myfile << "#It\t"        << "Res_norm_tot\t"        
              << "Res_norm_u\t"          << "Res_norm_p\t"          << "Res_norm_z\t"          << "Res_norm_a\t"
              << "Rel_res_norm\t"        << "Norm of update\t"
              <<"Inner converged\t"<<"Inner res\t"<<"Inner its"<<std::endl;
      myfile << newtonIt <<"\t"<< newtonRes <<"\t"
              << lclRes.GetBlock(0).Norml2()<<"\t"<<lclRes.GetBlock(1).Norml2()<<"\t"<<lclRes.GetBlock(2).Norml2()<<"\t"<<lclRes.GetBlock(3).Norml2()<<"\t"
              << newtonRes/newtonRes0 <<"\t"<< 0.0 <<"\t"
              << false       <<"\t"<< 0.0   <<"\t"<<0   <<std::endl;
      myfile.close();
    }    






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
      SparseMatrix Bt = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,1) ) );
      SparseMatrix Z1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) ) );
      SparseMatrix Z2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) ) );
      SparseMatrix B  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,0) ) );
      SparseMatrix Cs = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,1) ) );
      SparseMatrix X1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,2) ) );
      SparseMatrix X2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,3) ) );
      SparseMatrix Y  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,0) ) );
      SparseMatrix Fa = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,3) ) );
      //  -- Include temporal part
      BlockOperator* JM = dynamic_cast<BlockOperator*>( &_IMHD2DMassOperator.GetGradient( lclSol ) );
      SparseMatrix MuS = *( dynamic_cast<SparseMatrix*>( &JM->GetBlock(0,0) ) );
      SparseMatrix MaS = *( dynamic_cast<SparseMatrix*>( &JM->GetBlock(3,3) ) );
      //  --- for the mass matrix we need to kill the Dirichlet nodes, otherwise
      //       we risk dirtying the diagonal (should already be 1)
      for (int i = 0; i < _essUhTDOF.Size(); ++i){
        MuS( _essUhTDOF[i], _essUhTDOF[i] ) = 0.;
      }
      for (int i = 0; i < _essAhTDOF.Size(); ++i){
        MaS( _essAhTDOF[i], _essAhTDOF[i] ) = 0.;
      }
      Fu.Add( -1., MuS ); // remember mass matrices are assembled with negative sign!
      Fa.Add( -1., MaS );

      SparseMatrix Mps = *( dynamic_cast<SparseMatrix*>( &JM->GetBlock(1,0) ) );
      B.Add( -1., Mps );

      // std::string matricesFileName = "results/_uga";
      // PrintMatrices(matricesFileName,
      //               Fu,  Bt, Z1, Z2, MuS,
      //               B ,  Cs, X1, X2,
      //               Mz,  K,
      //               Y ,  Fa, MaS,
      //               Mp,  Ap, Wp, Mps,
      //               Aa,  Cp, Wa,
      //               _essUhTDOF, _essPhTDOF, _essAhTDOF);
      // int uga;
      // std::cin>>uga;




      // - Define system
      BlockOperator MHDOp(offsets);
      MHDOp.SetBlock( 0, 0, &Fu );
      MHDOp.SetBlock( 0, 1, &Bt );
      MHDOp.SetBlock( 0, 2, &Z1 );
      MHDOp.SetBlock( 0, 3, &Z2 );
      MHDOp.SetBlock( 1, 0, &B  );
      MHDOp.SetBlock( 1, 1, &Cs );
      MHDOp.SetBlock( 1, 2, &X1 );
      MHDOp.SetBlock( 1, 3, &X2 );
      MHDOp.SetBlock( 2, 2, &Mz );
      MHDOp.SetBlock( 2, 3, &K  );
      MHDOp.SetBlock( 3, 0, &Y  );
      MHDOp.SetBlock( 3, 3, &Fa );

      
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
      if ( !isQuietState ){
        AssembleWp( _PhFESpace, _essPhTDOF, neuBdrP, _mu, _UhFESpace, lclSol.GetBlock(0), ir, bir, Wp );
        pSi.SetWp( &Wp, isQuietState );
      }
      // -- Using own simplified class
      // AssembleFp( _PhFESpace, _essPhTDOF, neuBdrP, _dt, _mu, _UhFESpace, lclSol.GetBlock(0), ir, bir, Fp );
      // pSi.SetFp( &Fp );
      // -- Assembling each component explicitly
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
      BlockUpperTriangularPreconditioner *Uub = new BlockUpperTriangularPreconditioner( offsets ),
                                         *Uup = new BlockUpperTriangularPreconditioner( offsets );
      BlockLowerTriangularPreconditioner *Lub = new BlockLowerTriangularPreconditioner( offsets ),
                                         *Lup = new BlockLowerTriangularPreconditioner( offsets );
      AssembleLub( &Y,   &Fui, &Z1,  &Mzi,       Lub );
      AssembleUub( &Z1,  &Z2,  &Mzi, &K,   &aSi, Uub );
      AssembleLup( &Fui, &B,                     Lup );
      AssembleUup( &Fui, &Bt,  &X1,  &X2,  &pSi, Uup );
      // std::cout<<" Lub:      "<< Lub->Height() <<","<< Lub->Width() <<std::endl;
      // std::cout<<" Uub:      "<< Uub->Height() <<","<< Uub->Width() <<std::endl;
      // std::cout<<" Lup:      "<< Lup->Height() <<","<< Lup->Width() <<std::endl;
      // std::cout<<" Uup:      "<< Uup->Height() <<","<< Uup->Width() <<std::endl;
      
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
        std::cerr<<"ERROR: Preconditioner type "<<precType<<" not recognised."<<std::endl;
      }
      OperatorsSequence MHDPr( precOps, precOwn );
      // std::cout<<" P:        "<< MHDPr.Height() <<","<< MHDPr.Width() <<std::endl;

      // // rhs:
      // mfem::out.precision(std::numeric_limits< double >::max_digits10);
      // std::cout<<"rhsu: "; lclRes.GetBlock(0).Print(mfem::out, lclRes.GetBlock(0).Size());std::cout<<std::endl;
      // std::cout<<"rhsp: "; lclRes.GetBlock(1).Print(mfem::out, lclRes.GetBlock(1).Size());std::cout<<std::endl;
      // std::cout<<"rhsz: "; lclRes.GetBlock(2).Print(mfem::out, lclRes.GetBlock(2).Size());std::cout<<std::endl;
      // std::cout<<"rhsA: "; lclRes.GetBlock(3).Print(mfem::out, lclRes.GetBlock(3).Size());std::cout<<std::endl;

      // std::string matricesFileName = "results/_uga";
      // PrintMatrices(matricesFileName,
      //               Fu,  Bt, Z1, Z2, MuS,
      //               B ,  Cs, X1, X2,
      //               Mz,  K,
      //               Y ,  Fa, MaS,
      //               Mp,  Ap, Wp, Mps,
      //               Aa,  Cp, Wa,
      //               _essUhTDOF, _essPhTDOF, _essAhTDOF);
      // int uga;
      // std::cin>>uga;



      // Define solver
      PetscLinearSolver solver( MPI_COMM_SELF, "solver_" );
      bool isIterative = true;
      solver.iterative_mode = isIterative;
      // - register preconditioner and system
      solver.SetPreconditioner(MHDPr);
      solver.SetOperator(MHDOp);
      // - eventually register viewer to print to file residual evolution for inner iterations
      if ( output>1 ){
        string filename = innerInnerConvpath +"GMRESconv_Nit" + to_string(newtonIt) + ".txt";
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







      // - solve!
      BlockVector lclDeltaSol( offsets );
      lclDeltaSol = 0.;
      solver.Mult( lclRes, lclDeltaSol );
      // - Shouldnt be necessary (Dirichlet nodes are set to 0 when multiplying by _IMHD2D(Mass)Operator) but does no harm:
      for ( int i = 0; i < _essUhTDOF.Size(); ++i )
        lclDeltaSol.GetBlock(0)(_essUhTDOF[i]) = 0.;
      for ( int i = 0; i < _essAhTDOF.Size(); ++i )
        lclDeltaSol.GetBlock(3)(_essAhTDOF[i]) = 0.;



      // Update relevant quantities
      // - solution
      lclSol += lclDeltaSol;

      // - compute residual
      //  -- spatial contribution
      _IMHD2DOperator.Mult( lclSol, lclRes );        // N(x)
      lclRes.GetBlock(0) -= frhs;                    // N(x) - b
      lclRes.GetBlock(1) -= grhs;
      lclRes.GetBlock(2) -= zrhs;
      lclRes.GetBlock(3) -= hrhs;
      lclRes.Neg();                                  // b - N(x)
      //  -- temporal contribution
      _IMHD2DMassOperator.Mult( lclSol,    dtu   );  // - u^{n+1}
      _IMHD2DMassOperator.Mult( sol[tt-1], tempN );  // - u^{n}
      dtu -= tempN;                                  // -(u^{n+1}-u^{n})
      //  -- combine together
      lclRes += dtu;
      // - Shouldnt be necessary (Dirichlet nodes are set to 0 when multiplying by _IMHD2D(Mass)Operator) but does no harm:
      for ( int i = 0; i < _essUhTDOF.Size(); ++i )
        lclRes.GetBlock(0)(_essUhTDOF[i]) = 0.;
      for ( int i = 0; i < _essAhTDOF.Size(); ++i )
        lclRes.GetBlock(3)(_essAhTDOF[i]) = 0.;
      newtonRes = lclRes.Norml2();
      newtonErrWRTPrevIt = lclDeltaSol.Norml2();

      tempResu = lclRes.GetBlock(0).Norml2();
      tempResp = lclRes.GetBlock(1).Norml2();
      tempResz = lclRes.GetBlock(2).Norml2();
      tempResa = lclRes.GetBlock(3).Norml2();
      tempRes  = newtonRes;



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
                << ", (u,p,z,A) = ("<< tempResu <<"," << tempResp <<"," << tempResz <<"," << tempResa <<")" << std::endl;
      if ( pbType<=4 ){ // if analytical solution is available, print info on error, too
        uerrGF.ProjectCoefficient( uFuncCoeff );
        perrGF.ProjectCoefficient( pFuncCoeff );
        zerrGF.ProjectCoefficient( zFuncCoeff );
        aerrGF.ProjectCoefficient( aFuncCoeff );
        uerrGF -= lclSol.GetBlock(0);
        perrGF -= lclSol.GetBlock(1);
        zerrGF -= lclSol.GetBlock(2);
        aerrGF -= lclSol.GetBlock(3);
        erru   = uerrGF.Norml2();
        errp   = perrGF.Norml2();
        errz   = zerrGF.Norml2();
        erra   = aerrGF.Norml2();
        errtot = sqrt(erru*erru + errp*errp + errz*errz + erra*erra);
        std::cout << "Error "<< errtot
                  << ", (u,p,z,A) = ("<< erru <<","<< errp <<","<< errz <<","<< erra <<")" << std::endl;
      }
      if( output>0 ){
        string filename = innerInnerConvpath +"NEWTconv.txt";
        ofstream myfile;
        myfile.open( filename, std::ios::app );
        myfile << newtonIt <<"\t"<< newtonRes <<"\t"
               << tempResu <<"\t"<< tempResp  <<"\t" <<tempResz << "\t" << tempResa << "\t"
               << newtonRes/newtonRes0 <<"\t"<< newtonErrWRTPrevIt << "\t"
               << solver.GetConverged() <<"\t"<< solver.GetFinalNorm() <<"\t"<<GMRESits <<std::endl;
        myfile.close();
      }      
      std::cout << "***********************************************************\n";



      // Clean up - no need: it's owned by MHDpr
      // delete Uub;
      // delete Uup;
      // delete Lub;
      // delete Lup;
    }
  
    bool newtConv = (newtonIt < maxNewtonIt);

    std::cout   << "Newton outer solver for time-step " << tt;
    if( newtConv && newtonIt > 0 ){
      std::cout << " converged in "                     << newtonIt;
    }else{
      if(newtonIt > 0){
        std::cout << " *DID NOT* converge in "          << maxNewtonIt;
      }else{
        std::cout << " *HAS STALLED*: converges in  "   << newtonIt;
      }
      newtNonConv++;
    }
    std::cout   << " iterations. Residual norm is "     << newtonRes;
    if ( pbType<=4 ){
      std::cout << ", error norm is "                   << errtot;
    }
    std::cout   << ", avg internal GMRES it are "       << totGMRESit/newtonIt  << ".\n";
    std::cout   << "***********************************************************\n";

    if( output>0 ){
      string filename = innerConvpath +"NEWTconv.txt";
      ofstream myfile;
      myfile.open( filename, std::ios::app );
      myfile << tt       << ",\t" << newtonRes << ",\t" << newtonRes/newtonRes0 << ",\t"
             << newtonIt << ",\t" << newtConv  << ",\t" << totGMRESit/newtonIt  << ",\t" << GMRESNonConv  << std::endl;
      myfile.close();
    }    




    tottotGMRESIt   += totGMRESit;
    totNewtonIt     += newtonIt;
    totGMRESNonConv += GMRESNonConv;

    // - save solution
    sol[tt] = lclSol;
    // - save contribution to space-time residual norm
    STresu += tempResu * tempResu;
    STresp += tempResp * tempResp;
    STresz += tempResz * tempResz;
    STresa += tempResa * tempResa;
    STres  += tempRes  * tempRes;


    // print to paraview
    // if ( tt == NT || tt<10 || (tt%5 == 1 ) ){     // output every 5 instants (and also last)
    if ( outputSol ){
      // -- assign to linked variables
      uGF = sol[tt].GetBlock(0);
      pGF = sol[tt].GetBlock(1);
      zGF = sol[tt].GetBlock(2);
      aGF = sol[tt].GetBlock(3);
      // -- store
      paraviewDC.SetCycle( tt );
      paraviewDC.SetTime( _dt*tt );
      paraviewDC.Save();
  
      if ( pbType <= 4 ){ // if analytical solution is available, print error, too
        uerrGF.ProjectCoefficient( uFuncCoeff );
        perrGF.ProjectCoefficient( pFuncCoeff );
        zerrGF.ProjectCoefficient( zFuncCoeff );
        aerrGF.ProjectCoefficient( aFuncCoeff );
        uerrGF -= sol[tt].GetBlock(0);
        perrGF -= sol[tt].GetBlock(1);
        zerrGF -= sol[tt].GetBlock(2);
        aerrGF -= sol[tt].GetBlock(3);
        // -- store
        paraviewDCErr.SetCycle( tt );
        paraviewDCErr.SetTime( _dt*tt );
        paraviewDCErr.Save();
      }

    }
  
    if ( outputRes ){
      // -- assign to linked variables
      uresGF = lclRes.GetBlock(0);
      presGF = lclRes.GetBlock(1);
      zresGF = lclRes.GetBlock(2);
      aresGF = lclRes.GetBlock(3);
      // -- store
      paraviewDCRes.SetCycle( tt );
      paraviewDCRes.SetTime( _dt*tt );
      paraviewDCRes.Save();
    }
    // }



  }


  // compute final space-time norm
  STresu = sqrt(STresu);
  STresp = sqrt(STresp);
  STresz = sqrt(STresz);
  STresa = sqrt(STresa);
  STres  = sqrt(STres);

  std::cout<<"***********************************************************\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"\nALL DONE!\n\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"Total # of Newton iterations: "<<totNewtonIt  <<". Non-converged: "<<newtNonConv    <<". Avg per time-step: "<<totNewtonIt/NT           <<"\n";
  std::cout<<"Total # of GMRES  iterations: "<<tottotGMRESIt<<". Non-converged: "<<totGMRESNonConv<<". Avg per time-step: "<<tottotGMRESIt/NT
                                                                                                  <<". Avg per Newton it: "<<tottotGMRESIt/totNewtonIt<<"\n";
  std::cout<<"Final space-time norm of residual: "<< STres
           << ", (u,p,z,A) = ("<<STresu<<","<<STresp<<","<<STresz<<","<<STresa<<")"<< std::endl;
  std::cout<<"***********************************************************\n";

  if( output>0 ){
    string filename = convPath +"Newton_convergence_results.txt";
    ofstream myfile;
    myfile.open( filename, std::ios::app );
    myfile << Tend          << ",\t" << NT              << ",\t" << refLvl           << ",\t" << STres<< ",\t"
           << totNewtonIt   << ",\t" << newtNonConv     << ",\t" << totNewtonIt/((double)NT)  << ",\t"
           << tottotGMRESIt << ",\t" << totGMRESNonConv << ",\t" << tottotGMRESIt/(double(NT))<< ",\t" << tottotGMRESIt/totNewtonIt << std::endl;
    myfile.close();
  }    

 
  // SaveSolution( sol, feSpaces, _dt, _mesh, outFilePath, outFileName );

  // if ( pbType == 4 ){
  //   SaveError( sol, feSpaces, uFun, pFun, zFun, aFun, _dt, _mesh, outFilePath, outFileName + "_err" );
  // }


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

// Laplacian, augmented with lagrangian multiplier for zero-mean pressure
void AssembleApAug( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                    const IntegrationRule *ir, SparseMatrix& _Ap ){
  if ( _essPhTDOF.Size() != 0 ){
    AssembleAp( _PhFESpace, _essPhTDOF, ir, _Ap );
    return;
  }
  std::cerr<<"Warning: augmenting pressure 'laplacian' with zero-mean condition\n";


  ConstantCoefficient one( 1.0 );

  // Diffusion operator
  BilinearForm aVarf( _PhFESpace );
  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
  aVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  aVarf.Assemble();
  aVarf.Finalize();
  SparseMatrix Apr;
  Apr.MakeRef( aVarf.SpMat() );

  // Constraint
  LinearForm pInt( _PhFESpace );
  pInt.AddDomainIntegrator( new DomainLFIntegrator( one ) );  //int_ q
  pInt.GetDLFI()->operator[](0)->SetIntRule(ir);
  pInt.Assemble();

  

  Array<int> myii( Apr.NumRows()         + 2 );
  Array<int> myjj( Apr.NumNonZeroElems() + 2*( pInt.Size() ) + 1 );
  Vector     mydd( Apr.NumNonZeroElems() + 2*( pInt.Size() ) + 1 );

  myii[0] = 0;
  for ( int ii = 0; ii < Apr.NumRows(); ++ii ){
    int off = Apr.GetI()[ii+1] - Apr.GetI()[ii];
    myii[ii+1] = myii[ii] + off + 1;
    for ( int jj = 0; jj < off; ++jj ){
      myjj[ myii[ii] + jj ] = Apr.GetJ()[    Apr.GetI()[ii] + jj ];
      mydd[ myii[ii] + jj ] = Apr.GetData()[ Apr.GetI()[ii] + jj ];
    }
    myjj[ myii[ii] + off ] = Apr.NumCols();
    mydd[ myii[ii] + off ] = pInt.GetData()[ii];
  }
  for ( int jj = 0; jj < pInt.Size(); ++jj ){
    myjj[ myii[ Apr.NumRows() ] + jj ] = jj;
    mydd[ myii[ Apr.NumRows() ] + jj ] = pInt.GetData()[jj];
  }
  myii[ Apr.NumRows() + 1 ] = myii[ Apr.NumRows() ] + pInt.Size() + 1;
  myjj[ myii[Apr.NumRows()+1] - 1 ] = pInt.Size();
  mydd[ myii[Apr.NumRows()+1] - 1 ] = 0.;

  // Finally assemble sparse matrix
  SparseMatrix myAp( myii.GetData(), myjj.GetData(), mydd.GetData(), _PhFESpace->GetTrueVSize()+1, _PhFESpace->GetTrueVSize()+1, false, false, true );

  _Ap = myAp;

  // std::string filename = "results/_uga";
  // std::string myfilename;
  // std::ofstream myfile;
  // myfile.precision(std::numeric_limits< double >::max_digits10);
  // std::cout<<"Printing matrices to "<<filename<<std::endl;
  // myfilename = filename + "_ApAug.dat";
  // myfile.open( myfilename );
  // _Ap.PrintMatlab(myfile);
  // myfile.close( );

  // AssembleAp(_PhFESpace, _essPhTDOF, ir, myAp);
  // myfilename = filename + "_Ap.dat";
  // myfile.open( myfilename );
  // myAp.PrintMatlab(myfile);
  // myfile.close( );

  // std::cout<<"Done "<<filename<<std::endl;
  // int uga;
  // std::cin>>uga;
}


// "Laplacian" for p - alternative version, Ap = B*diag(Mu)^-1*B^t
void AssembleBMuBt( FiniteElementSpace *_PhFESpace, const Array<int>& _essPhTDOF,
                    FiniteElementSpace *_UhFESpace, const Array<int>& _essUhTDOF,
                    const IntegrationRule *ir, SparseMatrix& _Ap ){

  // Assemble Mu
  BilinearForm mVarf( _UhFESpace );
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new VectorMassIntegrator( one ));
  mVarf.GetDBFI()->operator[](0)->SetIntRule(ir);
  mVarf.Assemble();
  mVarf.Finalize();

  SparseMatrix Mu;
  Mu.MakeRef( mVarf.SpMat() );
  Mu.SetGraphOwner(true);
  Mu.SetDataOwner(true);
  mVarf.LoseMat();

  // extract its diagonal
  Vector MDiagInv;
  Mu.GetDiag( MDiagInv );
  for ( int i = 0; i < MDiagInv.Size(); ++i ){  // invert it
    MDiagInv(i) = 1./MDiagInv(i);
  }
  // - impose homogeneous dirichlet BC by fixing at 1 the corresponding equations
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    MDiagInv( _essUhTDOF[i] ) = 1.0;
  }




  // Assemble B
  MixedBilinearForm bVarf( _UhFESpace, _PhFESpace );
  ConstantCoefficient mone( -1.0 );
  bVarf.AddDomainIntegrator(new VectorDivergenceIntegrator(mone) );

  bVarf.Assemble();
  bVarf.Finalize();

  SparseMatrix B = bVarf.SpMat();
  B.SetGraphOwner(true);
  B.SetDataOwner(true);
  bVarf.LoseMat();

  // - impose homogeneous dirichlet BC by simply removing corresponding equations
  Array<int> colsP(B.Height()), colsU(B.Width());
  colsP = 0.;
  colsU = 0.;
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    colsP[_essPhTDOF[i]] = 1;
  }
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    colsU[_essUhTDOF[i]] = 1;
  }
  B.EliminateCols( colsU );
  for (int i = 0; i < _essPhTDOF.Size(); ++i){
    B.EliminateRow( _essPhTDOF[i] );
  }


  SparseMatrix MuLinv(MDiagInv);

  // Array<const Operator*> pSiops(3);
  // Array<bool>            pSiown(3);
  // pSiops[0] = &Api;      pSiown[0] = false;
  // pSiops[1] = &Fp;       pSiown[1] = false;
  // pSiops[2] = &Mpi;      pSiown[2] = false;
  // OperatorsSequence pSi( pSiops, pSiown );


  // SparseMatrix *MiF  = Mult(MaLinv,_Fa);
  // SparseMatrix *FMiF = Mult(_Fa,*MiF);
  // _Cp = *FMiF;
  std::cerr<<"AssembleBMuBt Not yet implemented!"<<std::endl;

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



void PrintMatrices( const std::string& filename,
       const SparseMatrix& _Fu,  const SparseMatrix& _Bt,  const SparseMatrix& _Z1, const SparseMatrix& _Z2, const SparseMatrix& _Mu,
       const SparseMatrix& _B ,  const SparseMatrix& _Cs,  const SparseMatrix& _X1, const SparseMatrix& _X2,
       const SparseMatrix& _Mz,  const SparseMatrix& _K,
       const SparseMatrix& _Y ,  const SparseMatrix& _Fa,  const SparseMatrix& _Ma,
       const SparseMatrix& _Mp,  const SparseMatrix& _Ap,  const SparseMatrix& _Wp, const SparseMatrix& _Mps,
       const SparseMatrix& _Aa,  const SparseMatrix& _Cp,  const SparseMatrix& _Wa,
       const Array<int>& _essUhTDOF, const Array<int>& _essPhTDOF, const Array<int>& _essAhTDOF  ){

  std::cout<<"Printing matrices to "<<filename<<std::endl;


  std::string myfilename;
  std::ofstream myfile;
  myfile.precision(std::numeric_limits< double >::max_digits10);


  myfilename = filename + "_Mz.dat";
  myfile.open( myfilename );
  _Mz.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_K.dat";
  myfile.open( myfilename );
  _K.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Mp.dat";
  myfile.open( myfilename );
  _Mp.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Mps.dat";
  myfile.open( myfilename );
  _Mps.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Ap.dat";
  myfile.open( myfilename );
  _Ap.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Cs.dat";
  myfile.open( myfilename );
  _Cs.PrintMatlab(myfile);
  myfile.close( );

  // myfilename = filename + "_MaNZ.dat";
  // myfile.open( myfilename );
  // _MaNoZero.PrintMatlab(myfile);
  // myfile.close( );

  // myfilename = filename + "_MaNZL.dat";
  // myfile.open( myfilename );
  // _MaNoZeroLumped.PrintMatlab(myfile);
  // myfile.close( );

  myfilename = filename + "_Aa.dat";
  myfile.open( myfilename );
  _Aa.PrintMatlab(myfile);
  myfile.close( );


  // Dirichlet nodes
  myfilename = filename + "_essU.dat";
  myfile.open( myfilename );
  _essUhTDOF.Print(myfile,1);
  myfile.close( );

  myfilename = filename + "_essP.dat";
  myfile.open( myfilename );
  _essPhTDOF.Print(myfile,1);
  myfile.close( );

  // myfilename = filename + "essZ.dat";
  // myfile.open( myfilename );
  // _essZhTDOF.Print(myfile,1);
  // myfile.close( );

  myfilename = filename + "_essA.dat";
  myfile.open( myfilename );
  _essAhTDOF.Print(myfile,1);
  myfile.close( );



  myfilename = filename + "_Mu.dat";
  myfile.open( myfilename );
  _Mu.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Ma.dat";
  myfile.open( myfilename );
  _Ma.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_B.dat";
  myfile.open( myfilename );
  _B.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Bt.dat";
  myfile.open( myfilename );
  _Bt.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_X1.dat";
  myfile.open( myfilename );
  _X1.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_X2.dat";
  myfile.open( myfilename );
  _X2.PrintMatlab(myfile);
  myfile.close( );


  myfilename = filename + "_Fu.dat";
  myfile.open( myfilename );
  _Fu.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Fa.dat";
  myfile.open( myfilename );
  _Fa.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Z1.dat";
  myfile.open( myfilename );
  _Z1.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Z2.dat";
  myfile.open( myfilename );
  _Z2.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Y.dat";
  myfile.open( myfilename );
  _Y.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Wp.dat";
  myfile.open( myfilename );
  _Wp.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Wa.dat";
  myfile.open( myfilename );
  _Wa.PrintMatlab(myfile);
  myfile.close( );

  // myfilename = filename + "_dtuWa_" + std::to_string(_myRank) + ".dat";
  // myfile.open( myfilename );
  // _dtuWa.PrintMatlab(myfile);
  // myfile.close( );

  myfilename = filename + "_Cp.dat";
  myfile.open( myfilename );
  _Cp.PrintMatlab(myfile);
  myfile.close( );

  // myfilename = filename + "_C0.dat";
  // myfile.open( myfilename );
  // _C0.PrintMatlab(myfile);
  // myfile.close( );

  // myfilename = filename + "_Cm.dat";
  // myfile.open( myfilename );
  // _Cm.PrintMatlab(myfile);
  // myfile.close( );


  // Vector B0(_dim);
  // ComputeAvgB( B0 );
  // double B0norm2 = ( B0(0)*B0(0) + B0(1)*B0(1) ) / (_area*_mu0); 
  // myfilename = filename + "_B0.dat";
  // myfile.open( myfilename );
  // myfile<<B0norm2;
  // myfile.close( );


  // // rhs of non-linear operator
  // myfilename = filename + "_NLrhsU.dat";
  // myfile.open( myfilename );
  // _frhs.Print(myfile,1);
  // myfile.close( );

  // myfilename = filename + "_NLrhsP.dat";
  // myfile.open( myfilename );
  // _grhs.Print(myfile,1);
  // myfile.close( );

  // myfilename = filename + "_NLrhsZ.dat";
  // myfile.open( myfilename );
  // _zrhs.Print(myfile,1);
  // myfile.close( );

  // myfilename = filename + "_NLrhsA.dat";
  // myfile.open( myfilename );
  // _hrhs.Print(myfile,1);
  // myfile.close( );

  std::cout<<"Matrices printed"<<std::endl;


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
  double resU, resP, resZ, resA;
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
  
  // print
  PetscViewer viewer =  ctx->vf->viewer;
  PetscViewerPushFormat( viewer, ctx->vf->format);
  // PetscViewerASCIIAddTab( viewer,((PetscObject)ksp)->tablevel );
  if (n == 0 ) {// && ((PetscObject)ksp)->prefix) {
    // PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);
    PetscViewerASCIIPrintf(viewer,"  Residual norms for outer solver.\n");
  }
  PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm_resu_resp_resz_resa %14.12e\t%14.12e\t%14.12e\t%14.12e\t%14.12e \n",
                         n,(double)rnorm,(double)resU,(double)resP,(double)resZ,(double)resA);
  // PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);
  PetscViewerPopFormat(viewer);

  return 0;
    
}
