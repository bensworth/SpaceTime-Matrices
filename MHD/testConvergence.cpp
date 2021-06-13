// Test file to check correctness of order of convergence
//
// For a number of refinement levels for both space and time, rX and rT,
//  performs one time step and computes error wrt to analytical solution,
//  using a very tight convergence criteria for the newton iteration
// The errors considered are
//  erru = ||u-uh||_H1, printed to results/solutionConvergenceU
//  errp = ||p-ph||_L2, printed to results/solutionConvergenceP
//  errz = ||z-zh||_L2, printed to results/solutionConvergenceZ
//  erra = ||A-Ah||_H1, printed to results/solutionConvergenceA
//  as well as their sum 
//  errTot = sqrt( erru^2 + errp^2 + errz^2 + erra^2 ), printed to results/solutionConvergence
// When using Cinf analytical solutions, and inf/sup-stable discretisations,
//  if I got my theory right, convergence should be
//  ~ dx^s (with s being the polynomial order of the basis functions used)
//  ~ dt^2 (since we're using a *single* time step of 1st order backward Euler)
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
                  BlockLowerTriangularPreconditioner* Lub );
void AssembleUub( const Operator* Z,   const Operator* aSi, BlockUpperTriangularPreconditioner* Uub );
void AssembleLup( const Operator* Fui, const Operator* B,   BlockLowerTriangularPreconditioner* Lup );
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* X, const Operator* pSi, BlockUpperTriangularPreconditioner* Uup );

//---------------------------------------------------------------------------

int main(int argc, char *argv[]){
  // Define parameters
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if ( numProcs > 1 ){
    std::cerr<<"This test is intended for single processors only";
    return 1;
  }

  // Initialise problem parameters
  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;
  int refLvlX = 4;
  int maxRefLvlX = 8;
  int refLvlT = 4;
  int maxRefLvlT = 8;
  bool stab = false;
  int verbose = 0;
  int pbType = 4;
  int precType = 2;
  const char *petscrc_file = "rc_SpaceTimeIMHD2D";
  double T0 = 0.;

  int _dim = 2;

  const int     maxNewtonIt = 20;
  const double  newtonRTol  = 1e-13;
  const double  newtonATol  = 1e-15;

  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvlX, "-rX", "--rXlevel",
                "Refinement level for space (default: 4)");
  args.AddOption(&maxRefLvlX, "-maxrX", "--maxrXlevel",
                "Max refinement level for space (default: 8)");
  args.AddOption(&refLvlT, "-rT", "--rTlevel",
                "Refinement level for time (default: 4)");
  args.AddOption(&maxRefLvlT, "-maxrT", "--maxrTlevel",
                "Max refinement level for time (default: 8)");
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
        );
  args.AddOption(&precType, "-P", "--preconditioner",
                "Type of preconditioner: 0-Cyr et al: Uupi*Lupi*Uubi*Lubi\n"
                "                        1-Cyr et al simplified: Uupi*Lupi*Uubi\n"
                "                        2-Cyr et al uber simplified: Uupi*Uubi (default)\n"
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

  // - open a bunch of files for output
  std::string myfilename  = "./results/solutionConvergence.dat";
  std::string myfilenameU = "./results/solutionConvergenceU.dat";
  std::string myfilenameP = "./results/solutionConvergenceP.dat";
  std::string myfilenameZ = "./results/solutionConvergenceZ.dat";
  std::string myfilenameA = "./results/solutionConvergenceA.dat";
  std::ofstream myfile;
  std::ofstream myfileU;
  std::ofstream myfileP;
  std::ofstream myfileZ;
  std::ofstream myfileA;
  myfile.open(  myfilename  );
  myfileU.open( myfilenameU );
  myfileP.open( myfilenameP );
  myfileZ.open( myfilenameZ );
  myfileA.open( myfilenameA );
  myfile.precision(std::numeric_limits< double >::max_digits10);
  myfileU.precision(std::numeric_limits< double >::max_digits10);
  myfileP.precision(std::numeric_limits< double >::max_digits10);
  myfileZ.precision(std::numeric_limits< double >::max_digits10);
  myfileA.precision(std::numeric_limits< double >::max_digits10);



  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);


  void(  *uFun)( const Vector & x, const double t, Vector & u );
  double(*pFun)( const Vector & x, const double t             );
  double(*zFun)( const Vector & x, const double t             );
  double(*aFun)( const Vector & x, const double t             );
  void(  *fFun)( const Vector & x, const double t, Vector & f );
  double(*gFun)( const Vector & x, const double t             );
  double(*hFun)( const Vector & x, const double t             );
  void(  *nFun)( const Vector & x, const double t, Vector & f );
  double(*mFun)( const Vector & x, const double t             );
  void  (*wFun)( const Vector & x, const double t, Vector & w );
  double(*qFun)( const Vector & x, const double t             );
  double(*cFun)( const Vector & x, const double t             );
  double(*yFun)( const Vector & x, const double t             );
  
  void( *guFun)( const Vector & x, const double t, Vector & f );
  void( *gvFun)( const Vector & x, const double t, Vector & f );
  void( *gaFun)( const Vector & x, const double t, Vector & f );



  std::string mesh_file;
  std::string pbName;

  // extra variables useful in defining the Pb
  Array<int> essTagsU(0);
  Array<int> essTagsV(0);
  Array<int> essTagsP(0);
  Array<int> essTagsA(0);

  double _mu;  
  double _eta; 
  double _mu0; 

  MHDTestCaseSelector( pbType, 
                       uFun, pFun, zFun, aFun,
                       fFun, gFun, hFun, nFun, mFun,
                       wFun, qFun, yFun, cFun,
                       _mu, _eta, _mu0,
                       pbName, mesh_file,
                       essTagsU, essTagsV, essTagsP, essTagsA,
                       &guFun, &gvFun, &gaFun );

  // - define integration rule to be used throughout
  Array<int> ords(3);
  // if ( !stab ){
  //   ords[0] = 2*ordU + ordU-1;         // ( (u·∇)u, v )
  //   ords[1] =   ordU + ordA-1 + ordZ;  // (   z ∇A, v )
  //   ords[2] =   ordU + ordA-1 + ordA;  // ( (u·∇A), B )
  // }else{
    ords[0] = 2*ordU + 2*(ordU-1);                 // ( (u·∇)u, (w·∇)v )
    ords[1] =   ordU +    ordU-1 + ordA-1 + ordZ;  // (   z ∇A, (w·∇)v )
    ords[2] = 2*ordU + 2*(ordA-1);                 // ( (u·∇A),  w·∇B )    
  // }
  const IntegrationRule *ir  = &IntRules.Get( Geometry::Type::TRIANGLE, ords.Max() ); // for domains
  const IntegrationRule *bir = &IntRules.Get( Geometry::Type::SEGMENT,  ords.Max() ); // for boundaries
  std::cout<<"Selecting integrator of order "<<ords.Max()<<std::endl;

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

  VectorFunctionCoefficient guFuncCoeff( _dim, guFun );
  VectorFunctionCoefficient gvFuncCoeff( _dim, gvFun );
  VectorFunctionCoefficient gaFuncCoeff( _dim, gaFun );



  // Generate mesh
  Mesh *_mesh = new Mesh( mesh_file.c_str(), 1, 1 );
  // - refine as needed to begin
  for (int i = 0; i < refLvlX; i++)
    _mesh->UniformRefinement();



  //*************************************************************************
  // SPACE REFINEMENT
  //*************************************************************************
  for ( ; refLvlX <= maxRefLvlX; ++refLvlX ){


    double hMin, hMax, rMin, rMax;
    _mesh->GetCharacteristics( hMin, hMax, rMin, rMax );
    std::cout<<"   --dx   "<<hMin<<", "<<hMax<<std::endl;


    // - initialise FE info
    FiniteElementCollection* _UhFEColl = new H1_FECollection( ordU, _dim );
    FiniteElementCollection* _PhFEColl = new H1_FECollection( ordP, _dim );
    FiniteElementCollection* _ZhFEColl = new H1_FECollection( ordZ, _dim );
    FiniteElementCollection* _AhFEColl = new H1_FECollection( ordA, _dim );
    FiniteElementSpace* _UhFESpace   = new FiniteElementSpace( _mesh, _UhFEColl, _dim );
    FiniteElementSpace* _UhFESpace1C = new FiniteElementSpace( _mesh, _UhFEColl, 1 ); // single-comonent counterpart
    FiniteElementSpace* _PhFESpace   = new FiniteElementSpace( _mesh, _PhFEColl );
    FiniteElementSpace* _ZhFESpace   = new FiniteElementSpace( _mesh, _ZhFEColl );
    FiniteElementSpace* _AhFESpace   = new FiniteElementSpace( _mesh, _AhFEColl );

    Array< FiniteElementSpace* > feSpaces(4);
    feSpaces[0] = _UhFESpace;
    feSpaces[1] = _PhFESpace;
    feSpaces[2] = _ZhFESpace;
    feSpaces[3] = _AhFESpace;


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
    // std::cout << "***********************************************************\n";
    // std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
    // std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
    // std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
    // std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
    // std::cout << "***********************************************************\n";
    // std::cout << "Tags dir u      "; essBdrU.Print(mfem::out, essBdrU.Size() );
    // std::cout << "Tags dir v      "; essBdrV.Print(mfem::out, essBdrV.Size() );
    // std::cout << "Tags dir p      "; essBdrP.Print(mfem::out, essBdrP.Size() );
    // std::cout << "Tags dir A      "; essBdrA.Print(mfem::out, essBdrA.Size() );
    // std::cout << "***********************************************************\n";
    // std::cout << "Dir u      "; _essUhTDOF.Print(mfem::out, _essUhTDOF.Size() ); std::cout<< "\n";
    // std::cout << "Dir p      "; _essPhTDOF.Print(mfem::out, _essPhTDOF.Size() ); std::cout<< "\n";
    // std::cout << "Dir A      "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
    // std::cout << "***********************************************************\n";



    // Handy variables
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
    Array<BlockVector> sol( 2 );
    sol[0].Update(offsets); sol[0] = 0.;
    sol[1].Update(offsets); sol[1] = 0.;

    BlockVector lclSol = sol[0];    // this will contain the update on the solution at each time-step
    GridFunction _wGridFunc(  _UhFESpace, lclSol.GetBlock(0).GetData() ); // these will be used in the integrator, so link these variables to
    GridFunction _cGridFunc(  _AhFESpace, lclSol.GetBlock(3).GetData() ); //  lclSol once and for all, so that it gets updated automatically
    VectorGridFunctionCoefficient _wFuncCoeff( &_wGridFunc );             //  and store them in a handy decorator used in the integrator
    GridFunctionCoefficient       _cFuncCoeff( &_cGridFunc );

    // - these will store the error
    GridFunction uerrGF( _UhFESpace ); 
    GridFunction perrGF( _PhFESpace ); 
    GridFunction zerrGF( _ZhFESpace ); 
    GridFunction aerrGF( _AhFESpace ); 
    GridFunction guerrGF(_UhFESpace1C);
    GridFunction gverrGF(_UhFESpace1C);
    double erru;
    double errp;
    double errz;
    double erra;
    double errtot;


    //*************************************************************************
    // TIME REFINEMENT
    //*************************************************************************
    for ( int rLvlT = refLvlT; rLvlT <= maxRefLvlT; ++rLvlT ){
      double _dt = pow(2,-rLvlT);


      // initialise solution
      for ( int tt = 0; tt < 2; ++tt ){
        sol[tt].Update( offsets );
        uFuncCoeff.SetTime( T0 + _dt*tt );
        pFuncCoeff.SetTime( T0 + _dt*tt );
        zFuncCoeff.SetTime( T0 + _dt*tt );
        aFuncCoeff.SetTime( T0 + _dt*tt );
        uerrGF.ProjectCoefficient( uFuncCoeff );
        perrGF.ProjectCoefficient( pFuncCoeff );
        zerrGF.ProjectCoefficient( zFuncCoeff );
        aerrGF.ProjectCoefficient( aFuncCoeff );
        // this is used to evaluate Dirichlet nodes
        sol[tt].GetBlock(0) = uerrGF;
        sol[tt].GetBlock(1) = perrGF;
        sol[tt].GetBlock(2) = zerrGF;
        sol[tt].GetBlock(3) = aerrGF;
      }

      // assemble integrators
      BlockNonlinearForm _IMHD2DOperator, _IMHD2DMassOperator; // operators for returning spatial and temporal (mass) part of MHD equations
      _IMHD2DOperator.SetSpaces( feSpaces );
      _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DSpaceIntegrator( _dt, _mu, _mu0, _eta, stab,
                                                                                   &fFuncCoeff, &hFuncCoeff, &_wFuncCoeff, &_cFuncCoeff ) ); // for bilinforms
      _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DSpaceIntegrator( _dt, _mu, _mu0, _eta ), essBdrA );  // for flux in third equation
      _IMHD2DMassOperator.SetSpaces( feSpaces );
      _IMHD2DMassOperator.AddDomainIntegrator(  new IncompressibleMHD2DTimeIntegrator( _dt, _mu, _mu0, _eta, stab, &_wFuncCoeff, &_cFuncCoeff ) );
      // _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ) );
      // std::cout<<"Warning: excluding bdr contribution from integrator"<<std::endl;

      std::cout<<"Warning: I'm messing up BC for pressure! Also on operators and rhs for f!"<<std::endl;
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
      // ASSEMBLE SYSTEM
      //*************************************************************************

      // Set data functions
      uFuncCoeff.SetTime( T0 + _dt );
      pFuncCoeff.SetTime( T0 + _dt );
      zFuncCoeff.SetTime( T0 + _dt );
      aFuncCoeff.SetTime( T0 + _dt );
      fFuncCoeff.SetTime( T0 + _dt );
      gFuncCoeff.SetTime( T0 + _dt );
      hFuncCoeff.SetTime( T0 + _dt );
      nFuncCoeff.SetTime( T0 + _dt );
      mFuncCoeff.SetTime( T0 + _dt );

      guFuncCoeff.SetTime( T0 + _dt );
      gvFuncCoeff.SetTime( T0 + _dt );
      gaFuncCoeff.SetTime( T0 + _dt );


      // Compute rhs
      // for u
      // - actual rhs
      LinearForm frhs( _UhFESpace );
      // ScalarVectorProductCoefficient muNFuncCoeff(_mu, nFuncCoeff);
      frhs.AddDomainIntegrator(   new VectorDomainLFIntegrator(         fFuncCoeff       )          );  //int_\Omega f*v
      // frhs.AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     muNFuncCoeff       ), neuBdrU );  //int_d\Omega \mu * du/dn *v
      // frhs.AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator(   pFuncCoeff, -1.0 ), neuBdrU );  //int_d\Omega -p*v*n
      frhs.GetDLFI()->operator[](0)->SetIntRule(ir);
      // frhs.GetBLFI()->operator[](0)->SetIntRule(bir);
      // frhs.GetBLFI()->operator[](1)->SetIntRule(bir);
      frhs.Assemble();
      frhs *= _dt;


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
      ProductCoefficient etaMu0MFuncCoeff( _eta/_mu0, mFuncCoeff );
      hrhs.AddDomainIntegrator(   new DomainLFIntegrator(         hFuncCoeff )          );  //int_\Omega h*B
      hrhs.AddBoundaryIntegrator( new BoundaryLFIntegrator( etaMu0MFuncCoeff ), neuBdrA );  //int_d\Omega \eta/\mu0 * dA/dn *B
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
      grhs.SetSubVector( _essPhTDOF, 0. );
      hrhs.SetSubVector( _essAhTDOF, 0. );



      // Initialise Newton iteration using IC
      lclSol = sol[0];
      // - however, remember to preserve Dirichlet nodes
      for ( int i = 0; i < _essUhTDOF.Size(); ++i )
        lclSol.GetBlock(0)(_essUhTDOF[i]) = sol[1].GetBlock(0)(_essUhTDOF[i]);
      // for ( int i = 0; i < _essPhTDOF.Size(); ++i )
      //   lclSol.GetBlock(1)(_essPhTDOF[i]) = sol[1].GetBlock(1)(_essPhTDOF[i]);
      for ( int i = 0; i < _essAhTDOF.Size(); ++i )
        lclSol.GetBlock(3)(_essAhTDOF[i]) = sol[1].GetBlock(3)(_essAhTDOF[i]);
      // - also set z to be *exactly* the laplacian of A
      BlockVector lclRes(offsets); // this will be reused later
      lclSol.GetBlock(2) = 0.;
      _IMHD2DOperator.Mult( lclSol, lclRes );
      Vector tempZ = lclRes.GetBlock(2);      // this is the "full" K*A, including effect from Dirichlet
      tempZ -= zrhs;
      tempZ.Neg();
      Mzi.Mult( tempZ, lclSol.GetBlock(2) );  // solve for z


      // - compute residual
      //  -- spatial contribution
      _IMHD2DOperator.Mult( lclSol, lclRes );        // N(x)
      lclRes.GetBlock(0) -= frhs;                    // N(x) - b
      lclRes.GetBlock(1) -= grhs;
      // lclRes.GetBlock(2) -= zrhs;
      lclRes.GetBlock(2)  = 0.;                      // should already be basically zero!
      lclRes.GetBlock(3) -= hrhs;
      lclRes.Neg();                                  // b - N(x)
      //  -- temporal contribution
      BlockVector dtu(offsets), tempN(offsets);
      _IMHD2DMassOperator.Mult( lclSol, dtu   );  // - u^{n+1}
      _IMHD2DMassOperator.Mult( sol[0], tempN );  // - u^{n}
      dtu -= tempN;                               // -(u^{n+1}-u^{n})
      //  -- combine together
      lclRes += dtu;
      // - Shouldnt be necessary (Dirichlet nodes are set to 0 when multiplying by _IMHD2D(Mass)Operator) but does no harm:
      for ( int i = 0; i < _essUhTDOF.Size(); ++i )
        lclRes.GetBlock(0)(_essUhTDOF[i]) = 0.;
      for ( int i = 0; i < _essAhTDOF.Size(); ++i )
        lclRes.GetBlock(3)(_essAhTDOF[i]) = 0.;




      // - compute norm of residual and initialise relevant quantities for Newton iteration
      int newtonIt = 0;
      double newtonRes = lclRes.Norml2();
      double newtonRes0 = newtonRes;
      std::cout << "***********************************************************\n";
      std::cout << "Pb " << pbName <<". Starting Newton, initial residual "<< newtonRes
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
        std::cout<<"Warning: setting Z1 and X1 to 0\n";
        Z1 = 0.;
        X1 = 0.;
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
        // - Complete X operator
        ScaledOperator* mMzi2 = new ScaledOperator( &Mzi, -1.0 );
        Array<const Operator*> mX1MziKops(3);
        Array<bool>            mX1MziKown(3);
        mX1MziKops[0] = &K;    mX1MziKown[0] = false;      
        mX1MziKops[1] = mMzi2; mX1MziKown[1] = true;
        mX1MziKops[2] = &X1;   mX1MziKown[2] = false;
        OperatorsSequence* mX1MziK = new OperatorsSequence( mX1MziKops, mX1MziKown );
        Array<const Operator*> Xops(2);
        Array<bool>            Xown(2);
        Xops[0] = mX1MziK;  Xown[0] = true;
        Xops[1] = &X2;      Xown[1] = false;
        OperatorsSeries X( Xops, Xown ); // own second, not first

        // - Define actual system
        BlockOperator MHDOp(offsetsReduced);
        MHDOp.SetBlock( 0, 0, &Fu );
        MHDOp.SetBlock( 0, 1, &Bt );
        MHDOp.SetBlock( 0, 2, &Z  );
        MHDOp.SetBlock( 1, 0, &B  );
        MHDOp.SetBlock( 1, 1, &Cs );
        MHDOp.SetBlock( 1, 2, &X  );
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
        BlockUpperTriangularPreconditioner *Uub = new BlockUpperTriangularPreconditioner( offsetsReduced ),
                                           *Uup = new BlockUpperTriangularPreconditioner( offsetsReduced );
        BlockLowerTriangularPreconditioner *Lub = new BlockLowerTriangularPreconditioner( offsetsReduced ),
                                           *Lup = new BlockLowerTriangularPreconditioner( offsetsReduced );
        AssembleLub( &Y, &Fui,            Lub );
        AssembleUub( &Z,            &aSi, Uub );
        AssembleLup( &Fui, &B,            Lup );
        AssembleUup( &Fui, &Bt, &X, &pSi, Uup );
        
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
        // - Shouldnt be necessary (Dirichlet nodes are set to 0 when multiplying by _IMHD2D(Mass)Operator) but does no harm:
        for ( int i = 0; i < _essUhTDOF.Size(); ++i )
          lclDeltaSolReduced.GetBlock(0)(_essUhTDOF[i]) = 0.;
        for ( int i = 0; i < _essAhTDOF.Size(); ++i )
          lclDeltaSolReduced.GetBlock(2)(_essAhTDOF[i]) = 0.;

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
        tempZ -= zrhs;
        tempZ.Neg();
        Mzi.Mult( tempZ, lclSol.GetBlock(2) );

        // - compute residual
        //  -- spatial contribution
        _IMHD2DOperator.Mult( lclSol, lclRes );        // N(x)
        lclRes.GetBlock(0) -= frhs;                    // N(x) - b
        lclRes.GetBlock(1) -= grhs;
        // lclRes.GetBlock(2) -= zrhs;
        lclRes.GetBlock(2)  = 0.;                      // should already be basically zero!
        lclRes.GetBlock(3) -= hrhs;
        lclRes.Neg();                                  // b - N(x)
        //  -- temporal contribution
        _IMHD2DMassOperator.Mult( lclSol, dtu   );  // - u^{n+1}
        _IMHD2DMassOperator.Mult( sol[0], tempN );  // - u^{n}
        dtu -= tempN;                                  // -(u^{n+1}-u^{n})
        //  -- combine together
        lclRes += dtu;
        // - Shouldnt be necessary (Dirichlet nodes are set to 0 when multiplying by _IMHD2D(Mass)Operator) but does no harm:
        for ( int i = 0; i < _essUhTDOF.Size(); ++i )
          lclRes.GetBlock(0)(_essUhTDOF[i]) = 0.;
        for ( int i = 0; i < _essAhTDOF.Size(); ++i )
          lclRes.GetBlock(3)(_essAhTDOF[i]) = 0.;

        newtonRes = lclRes.Norml2();
        // newtonErrWRTPrevIt = lclDeltaSol.Norml2();




        // Output
        uerrGF = lclSol.GetBlock(0);
        perrGF = lclSol.GetBlock(1);
        zerrGF = lclSol.GetBlock(2);
        aerrGF = lclSol.GetBlock(3);
        errp   = perrGF.ComputeL2Error(pFuncCoeff);
        errz   = zerrGF.ComputeL2Error(zFuncCoeff);
        erra   = aerrGF.ComputeH1Error(&aFuncCoeff, &gaFuncCoeff);
        // erra   = aerrGF.ComputeL2Error(aFuncCoeff);
        // - to get the H1 norm of u, things are a bit trickier
        erru   = uerrGF.ComputeL2Error(uFuncCoeff);
        for ( int i = 0; i < uerrGF.Size()/2; ++i ){
          guerrGF.operator()(i) = uerrGF( i );
          gverrGF.operator()(i) = uerrGF( i + uerrGF.Size()/2 );
        }
        double errgu = guerrGF.ComputeGradError( &guFuncCoeff );
        double errgv = gverrGF.ComputeGradError( &gvFuncCoeff );
        erru = sqrt( erru*erru + errgu*errgu + errgv*errgv );

        errtot = sqrt(erru*erru + errp*errp + errz*errz + erra*erra);



        int GMRESits   = solver.GetNumIterations();
        if (solver.GetConverged()){
          std::cout << "Inner solver converged in ";
        }else{
          std::cout << "Inner solver *DID NOT* converge in ";
        }
        std::cout<< GMRESits << " iterations. Residual "<< solver.GetFinalNorm() <<std::endl;
        std::cout << "***********************************************************\n";
        std::cout << "Newton iteration "<< newtonIt+1 <<", residual "<< newtonRes
                  << ", (u,p,z,A) = ("<< lclRes.GetBlock(0).Norml2() <<","
                                      << lclRes.GetBlock(1).Norml2() <<","
                                      << lclRes.GetBlock(2).Norml2() <<","
                                      << lclRes.GetBlock(3).Norml2() <<")" << std::endl;
        std::cout << "Error: "<< errtot
                  << ", (u,p,z,A) = ("<< erru <<","
                                      << errp <<","
                                      << errz <<","
                                      << erra <<")" << std::endl;
        std::cout << "***********************************************************\n";

      } // newton it

      std::cout   << "Newton outer solver";
      if( newtonIt < maxNewtonIt ){
        std::cout << " converged in "                     << newtonIt;
      }else{
        std::cout << " *DID NOT* converge in "            << maxNewtonIt;
      }
      std::cout   << " iterations. Residual norm is "     << newtonRes;
      if ( pbType<=4 ){
        std::cout << ", error norm is "                   << errtot;
      }
      std::cout   << std::endl;
      std::cout   << "***********************************************************\n";


      // check the semilogy plot the output file
      myfile  << errtot  << "\t";
      myfileU << erru    << "\t";
      myfileP << errp    << "\t";
      myfileZ << errz    << "\t";
      myfileA << erra    << "\t";


    } // time refinement

    myfile  << std::endl;
    myfileU << std::endl;
    myfileP << std::endl;
    myfileZ << std::endl;
    myfileA << std::endl;


    // clean-up
    delete _UhFEColl;
    delete _PhFEColl;
    delete _ZhFEColl;
    delete _AhFEColl;
    delete _UhFESpace;
    delete _PhFESpace;
    delete _ZhFESpace;
    delete _AhFESpace;

    
    // refine for next level (unless I'm already at the end)
    if ( refLvlX < maxRefLvlX ){
      _mesh->UniformRefinement();
    }


  } // space refinement

  myfile.close();
  myfileU.close();
  myfileP.close();
  myfileZ.close();
  myfileA.close();



  delete _mesh;


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
//       ⌈ Fu Bt       ⌉
// Uup = |    pS X1 X2 |
//       |       I     |
//       ⌊          I  ⌋
void AssembleUup( const Operator* Fui, const Operator* Bt, const Operator* X, const Operator* pSi, BlockUpperTriangularPreconditioner* Uup ){
  Uup->iterative_mode = false;
  Uup->SetBlock( 0, 0, Fui );
  Uup->SetBlock( 0, 1, Bt  );
  Uup->SetBlock( 1, 1, pSi );
  Uup->SetBlock( 1, 2, X   );
  Uup->owns_blocks = false;
}


