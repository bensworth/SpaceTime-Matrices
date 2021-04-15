// Test file to check correctness of implementation of the MHD integrator
//  This is done by comparing my own implementation of the BlockNonlinearFormIntegrator
//  VS the results I would get by using classic MFEM integrators: the results should be
//  the same (up to machine precision)
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "imhd2dintegrator.hpp"
#include "mymixedbilinearform.hpp"
#include "boundaryfacediffusionintegrator.hpp"

using namespace std;
using namespace mfem;

//---------------------------------------------------------------------------
double constant(const Vector & x, const double t){
  return 1.;
}
double lapconstant(const Vector & x, const double t){
  return 0.;
}
double ganconstant(const Vector & x, const double t){
  return 0.;
}
double linear(const Vector & x, const double t){
  return x(0)+x(1);
}
double laplinear(const Vector & x, const double t){
  return 0.;
}
double ganlinear(const Vector & x, const double t){
  double xx(x(0));
  double yy(x(1));

  if( xx == 1 || yy == 1 )
    return 1.;
  if( xx == 0 || yy == 0 )
    return -1;
  return 0.;
}
double quadratic(const Vector & x, const double t ){
  return (x(1)-1.)*(x(1)-1.) + (x(0)-1.)*(x(0)-1.) + (x(0)-1.)*(x(1)-1.);
}
double lapquadratic(const Vector & x, const double t ){
  return 4.;
}
double ganquadratic(const Vector & x, const double t){
  double xx(x(0));
  double yy(x(1));
  if( xx == 1 )
    return yy-1;
  if( yy == 1 )
    return xx-1;
  if( xx == 0 )
    return 3-yy;
  if( yy == 0 )
    return 3-xx;
  return 0.;
}
double cubic(const Vector & x, const double t ){
  return x(0)*x(0)*x(0) + x(1)*x(1)*x(1) + x(1)*x(0)*x(0);
}
double lapcubic(const Vector & x, const double t ){
  return 6*x(0) + 6*x(1) + 2*x(1);
}
double gancubic(const Vector & x, const double t){
  double xx(x(0));
  double yy(x(1));
  if( xx == 1 )
    return 3+2*yy;
  if( yy == 1 )
    return 3+xx*xx;
  if( xx == 0 )
    return 0;
  if( yy == 0 )
    return -xx*xx;
  return 0.;
}
double trig(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));

  double delta   = 1./(2*M_PI);
  double epsilon = 0.2;

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( delta * log( temp ) );
}
double laptrig(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));

  double delta   = 1./(2*M_PI);
  double epsilon = 0.2;

  double temp = cosh(yy/delta) + epsilon*cos(xx/delta);

  return ( ( 1. - epsilon*epsilon ) / ( delta * temp*temp ) );
}
double gantrig(const Vector & x, const double t){
  double xx(x(0));
  double yy(x(1));

  double delta   = 1./(2*M_PI);
  double epsilon = 0.2;

  if( xx == 1 || xx == 0)
    return 0;
  if( yy == 1 )
    return   sinh(1./delta)/( cosh(1./delta) + epsilon*cos(xx/delta) );
  if( yy == 0 )
    return - sinh(0)/( cosh(0) + epsilon*cos(xx/delta) );
  return 0.;
}
// double trigy(const Vector & x, const double t ){
//   Vector myX(2);  // swap x and y
//   myX(0) = x(1);
//   myX(1) = x(0);

//   return trig(myX,t);
// }
// double laptrigy(const Vector & x, const double t ){
//   Vector myX(2);  // swap x and y
//   myX(0) = x(1);
//   myX(1) = x(0);

//   return laptrig(myX,t);
// }
// double gantrigy(const Vector & x, const double t ){
//   Vector myX(2);  // swap x and y
//   myX(0) = x(1);
//   myX(1) = x(0);

//   return gantrig(myX,t); NOT SURE THIS WILL WORK
// }

//---------------------------------------------------------------------------
void Plot( const Vector& z, const Vector& zMFEM, const Vector& a,
           const Array< FiniteElementSpace* >& feSpaces,
           double(*zFun_ex)( const Vector & x, const double t             ),
           Mesh* _mesh,
           const std::string& path, const std::string& filename);
void Plot( const Vector& z, const Vector& a,
           const Array< FiniteElementSpace* >& feSpaces,
           double(*zFun_ex)( const Vector & x, const double t             ),
           Mesh* _mesh,
           const std::string& path, const std::string& filename);
//---------------------------------------------------------------------------

int main(int argc, char *argv[]){

  MPI_Init(&argc, &argv);


  int refLvl = 4;
  string mesh_file = "./meshes/tri-square-testAn.mesh";

  const char *petscrc_file = "rc_SpaceTimeIMHD2D";

  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;


  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Max refinement level (default: 4)");
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
  args.Parse();


  MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

  const double _dt  = 0.5;
  const double _mu  = 0.5;
  const double _eta = 0.5;
  const double _mu0 = 0.5;

  //- generate mesh
  Mesh *_mesh = new Mesh( mesh_file.c_str(), 1, 1 );
  int _dim = _mesh->Dimension();
  


  // Declare a bunch of functions to use as tests
  int numFuncs = 5;
  double( *funs[numFuncs])( const Vector & x, const double t );
  funs[0]    = constant;
  funs[1]    = linear;
  funs[2]    = quadratic;
  funs[3]    = cubic;
  funs[4]    = trig;
  // funs[5]    = trigy;
  double( *Lapfuns[numFuncs])( const Vector & x, const double t );
  Lapfuns[0] = lapconstant;
  Lapfuns[1] = laplinear;
  Lapfuns[2] = lapquadratic;
  Lapfuns[3] = lapcubic;
  Lapfuns[4] = laptrig;
  // Lapfuns[5] = laptrigy;
  double( *ganfuns[numFuncs])( const Vector & x, const double t );
  ganfuns[0] = ganconstant;
  ganfuns[1] = ganlinear;
  ganfuns[2] = ganquadratic;
  ganfuns[3] = gancubic;
  ganfuns[4] = gantrig;
  // Ganfuns[5] = gantrigy;
  std::string names[numFuncs];
  names[0] = "constant";
  names[1] = "linear";
  names[2] = "quadratic";
  names[3] = "cubic";
  names[4] = "trig";
  // names[5] = "trigy";




  for (int r = 0; r < refLvl; r++){
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
    Array<int> essTagsA(1); essTagsA[0] = 1; essTagsA[1] = 2; essTagsA[2] = 3; essTagsA[3] = 4; // 1N 2E 3S 4W
    Array<int> _essAhTDOF;
    int numAtt = _mesh->bdr_attributes.Max();
    Array<int> essBdrA( numAtt ); essBdrA = 0;
    Array<int> neuBdrA( numAtt ); neuBdrA = 0;
    if ( _mesh->bdr_attributes.Size() > 0 ) {
      // search among all possible tags
      for ( int i = 1; i <= _mesh->bdr_attributes.Max(); ++i ){
        // if that tag is marked in the corresponding array in essTags, then flag it
        if( essTagsA.Find( i ) + 1 )
          essBdrA[i-1] = 1;
        else
          neuBdrA[i-1] = 1;
      }
      _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );
      // _ZhFESpace->GetEssentialTrueDofs( essBdrA, _essZhTDOF );
    }


    // std::cout << "***********************************************************\n";
    // std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
    // std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
    // std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
    // std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
    // std::cout << "***********************************************************\n";
    
    Array<int> offsets(5);
    offsets[0] = 0;
    offsets[1] = _UhFESpace->GetTrueVSize();
    offsets[2] = _PhFESpace->GetTrueVSize();
    offsets[3] = _ZhFESpace->GetTrueVSize();
    offsets[4] = _AhFESpace->GetTrueVSize();
    offsets.PartialSum();

    ConstantCoefficient one( 1.0 );
    ConstantCoefficient mone( -1.0 );

    Array<int> ords(3);
    ords[0] = 2*ordU + ordU-1;       // ( (u·∇)u, v )
    ords[1] = ordZ + ordA-1 + ordU;  // (   z ∇A, v )
    ords[2] = ordU + ordA-1 + ordA;  // ( (u·∇A), B )
    const IntegrationRule *ir  = &IntRules.Get( Geometry::Type::TRIANGLE, ords.Max() );
    const IntegrationRule *bir = &IntRules.Get( Geometry::Type::SEGMENT,  ords.Max() );



    // Assemble matrices using own integrator ---------------------------------
    Array< FiniteElementSpace* > feSpaces(4);
    feSpaces[0] = _UhFESpace;
    feSpaces[1] = _PhFESpace;
    feSpaces[2] = _ZhFESpace;
    feSpaces[3] = _AhFESpace;
    BlockNonlinearForm _IMHD2DOperator;
    _IMHD2DOperator.SetSpaces( feSpaces );
    _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ) );
    _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ), essBdrA );
    Array< Array<int> * > tmpEssDofs(4);
    Array<int> tempTDOF(0); // Set all to 0
    tmpEssDofs[0] = &tempTDOF;
    tmpEssDofs[1] = &tempTDOF;
    tmpEssDofs[2] = &tempTDOF;
    tmpEssDofs[3] = &_essAhTDOF;
    Array< Vector * > dummy(4); dummy = NULL;
    _IMHD2DOperator.SetEssentialTrueDofs( tmpEssDofs, dummy );
    BlockVector x(offsets);
    x = 0.;
    BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( x ) );
    SparseMatrix Mz = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(2,2) ) );
    SparseMatrix K  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(2,3) ) );
    // - inverse of mass matrix for z
    PetscParMatrix MzPetsc( &Mz );
    PetscLinearSolver Mzi( MzPetsc, "ZSolverMass_" );


    // // Assemble matrices using MFEM integrators ---------------------------------
    // BilinearForm zzvarf( _ZhFESpace );
    // zzvarf.AddDomainIntegrator(new MassIntegrator( one ));                              // <z,x>
    // zzvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    // zzvarf.Assemble();
    // zzvarf.Finalize();
    // SparseMatrix MzMFEM;
    // MzMFEM.MakeRef( zzvarf.SpMat() );
    // PetscParMatrix MzMFEMPetsc( &MzMFEM );
    // PetscLinearSolver MziMFEM( MzMFEMPetsc, "ZSolverMass_" );
    // MyMixedBilinearForm zavarf( _AhFESpace, _ZhFESpace );
    // zavarf.AddDomainIntegrator(  new DiffusionIntegrator(one) );                        // <∇A,∇x>
    // zavarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    // zavarf.AddBdrFaceIntegrator( new BoundaryFaceDiffusionIntegrator(mone), essBdrA );  // -<∇A·n,x>_GD
    // zavarf.GetBFBFI()->operator[](0)->SetIntRule(bir);
    // zavarf.Assemble();
    // zavarf.Finalize();
    // SparseMatrix KMFEM;
    // KMFEM.MakeRef( zavarf.SpMat() );
    // - impose Dirichlet BC
    // Array<int> colsA(zavarf.Width());
    // colsA = 0.;
    // for (int i = 0; i < _essAhTDOF.Size(); ++i){
    //   colsA[_essAhTDOF[i]] = 1;
    // }
    // KMFEM.EliminateCols( colsA );


    
    
    for ( int i = 0; i < numFuncs; ++i ){

      // assemble A
      FunctionCoefficient AFuncCoeff( funs[i] );
      GridFunction aFun(_AhFESpace);
      aFun.ProjectCoefficient( AFuncCoeff );

      // assemble z
      FunctionCoefficient zFuncCoeff( Lapfuns[i] );

      // assemble gA*n
      FunctionCoefficient mFuncCoeff( ganfuns[i] );
      LinearForm zrhs( _ZhFESpace );
      zrhs.AddBoundaryIntegrator( new BoundaryLFIntegrator( mFuncCoeff ), neuBdrA );  //int_d\Omega dA/dn *zeta
      zrhs.GetBLFI()->operator[](0)->SetIntRule(bir);
      zrhs.Assemble();

      x = 0.;
      x.GetBlock(3) = aFun;
      x.GetBlock(3).SetSubVectorComplement( _essAhTDOF, 0. );
      BlockVector y(offsets);
      _IMHD2DOperator.Mult(x,y);

      // compare against recovered solution
      // - using my integrator
      GridFunction zFun(_ZhFESpace);
      Vector tmp(_ZhFESpace->GetTrueVSize());
      K.Mult(aFun,tmp);
      tmp.Neg();
      tmp += zrhs;
      tmp -= y.GetBlock(2);
      Mzi.Mult(tmp,zFun);
      // // - using MFEM integrator
      // GridFunction zFunMFEM(_ZhFESpace);
      // KMFEM.Mult(aFun,tmp);
      // tmp.Neg();
      // MziMFEM.Mult(tmp,zFunMFEM);

      std::cout.precision(std::numeric_limits< double >::max_digits10);
      std::cout<<zFun.ComputeL2Error( zFuncCoeff )<<"\t";
      // std::cout<<zFunMFEM.ComputeL2Error( zFuncCoeff )<<"\t";

      std::string filename = "STMHDLapA_r" + to_string(r) + names[i];

      Plot( zFun, aFun, feSpaces, Lapfuns[i], _mesh, "ParaView", filename );


    }// func

    std::cout<<std::endl;

    delete _UhFESpace;
    delete _PhFESpace;
    delete _AhFESpace;
    delete _ZhFESpace;
    delete _UhFEColl;
    delete _PhFEColl;
    delete _AhFEColl;
    delete _ZhFEColl;


 
  }// reflvl









  delete _mesh;

  MFEMFinalizePetsc();
  MPI_Finalize();


  return 0;
}







// Saves a plot of the error
void Plot( const Vector& z, const Vector& a,
           const Array< FiniteElementSpace* >& feSpaces,
           double(*zFun_ex)( const Vector & x, const double t             ),
           Mesh* _mesh,
           const std::string& path="ParaView", const std::string& filename="STIMHD2D_err" ){

  // handy functions which will contain solution at single time-steps
  GridFunction zFun( feSpaces[2] ); zFun = z;
  GridFunction zEx(  feSpaces[2] ); zEx  = z;
  GridFunction zErr( feSpaces[2] ); zErr = z;
  GridFunction aFun( feSpaces[3] ); aFun = a;
  FunctionCoefficient  zFuncCoeff(zFun_ex);
  zEx.ProjectCoefficient(  zFuncCoeff );
  zErr -= zEx;
  zErr.Neg();


  // set up paraview data file
  ParaViewDataCollection paraviewDC( filename, _mesh );
  paraviewDC.SetPrefixPath(path);
  paraviewDC.SetLevelsOfDetail( 2 );
  paraviewDC.SetDataFormat(VTKFormat::BINARY);
  paraviewDC.SetHighOrderOutput(true);
  // - link wFun, pFun and vFun
  paraviewDC.RegisterField( "zEx",    &zEx  );
  paraviewDC.RegisterField( "zh",     &zFun );
  paraviewDC.RegisterField( "z-zh",   &zErr );
  paraviewDC.RegisterField( "A",      &aFun );

  paraviewDC.SetCycle( 0 );
  paraviewDC.SetTime( 0.0 );
  paraviewDC.Save();


}




// Saves a plot of the error
void Plot( const Vector& z, const Vector& zMFEM, const Vector& a,
           const Array< FiniteElementSpace* >& feSpaces,
           double(*zFun_ex)( const Vector & x, const double t             ),
           Mesh* _mesh,
           const std::string& path="ParaView", const std::string& filename="STIMHD2D_err" ){

  // handy functions which will contain solution at single time-steps
  GridFunction zFMF( feSpaces[2] ); zFMF = zMFEM;
  GridFunction zFun( feSpaces[2] ); zFun = z;
  GridFunction zEx(  feSpaces[2] ); zEx  = z;
  GridFunction zErr( feSpaces[2] ); zErr = z;
  GridFunction zEMF( feSpaces[2] ); zEMF = zMFEM;
  GridFunction aFun( feSpaces[3] ); aFun = a;
  FunctionCoefficient  zFuncCoeff(zFun_ex);
  zEx.ProjectCoefficient(  zFuncCoeff );
  zErr -= zEx;
  zErr.Neg();
  zEMF -= zEx;
  zEMF.Neg();


  // set up paraview data file
  ParaViewDataCollection paraviewDC( filename, _mesh );
  paraviewDC.SetPrefixPath(path);
  paraviewDC.SetLevelsOfDetail( 2 );
  paraviewDC.SetDataFormat(VTKFormat::BINARY);
  paraviewDC.SetHighOrderOutput(true);
  // - link wFun, pFun and vFun
  paraviewDC.RegisterField( "zEx",    &zEx  );
  paraviewDC.RegisterField( "zh",     &zFun );
  paraviewDC.RegisterField( "zhMF",   &zFMF );
  paraviewDC.RegisterField( "z-zh",   &zErr );
  paraviewDC.RegisterField( "zM-zhM", &zEMF );
  paraviewDC.RegisterField( "A",      &aFun );

  paraviewDC.SetCycle( 0 );
  paraviewDC.SetTime( 0.0 );
  paraviewDC.Save();


}
