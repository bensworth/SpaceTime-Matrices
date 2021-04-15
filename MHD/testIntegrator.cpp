// Test file to check correctness of implementation of the MHD integrator
//  This is done by comparing my own implementation of the BlockNonlinearFormIntegrator
//  VS the results I would get by using classic MFEM integrators: the results should be
//  the same (up to machine precision)
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "boundaryfacediffusionintegrator.hpp"
#include "mymixedbilinearform.hpp"
#include "imhd2dintegrator.hpp"

using namespace std;
using namespace mfem;

//---------------------------------------------------------------------------
double constant(const Vector & x, const double t){
  return 1.;
}
void  constantV(const Vector & x, const double t, Vector & w){
  w(0)  = 1.;
  w(1)  = 1.;
}
double Dxconstant(const Vector & x, const double t){
  return 0.;
}
double Dyconstant(const Vector & x, const double t){
  return 0.;
}

double linear(const Vector & x, const double t){
  return x(1);
}
double Dxlinear(const Vector & x, const double t){
  return 0.;
}
double Dylinear(const Vector & x, const double t){
  return 1.;
}
void  linearV(const Vector & x, const double t, Vector & w){
  w(0)  = x(0);
  w(1)  = x(1);
}

double quadratic(const Vector & x, const double t ){
  return (x(1)-1.)*(x(1)-1.);
}
double Dxquadratic(const Vector & x, const double t ){
  return 0.;
}
double Dyquadratic(const Vector & x, const double t ){
  return 2*(x(1)-1.);
}
void  quadraticV(const Vector & x, const double t, Vector & w){
  w(0)  = (x(0)-1.)*(x(0)-1.);
  w(1)  = (x(1)-1.)*(x(1)-1.);
}

double quadratic2(const Vector & x, const double t ){
  return x(0)*x(1);
}
double Dxquadratic2(const Vector & x, const double t ){
  return x(1);
}
double Dyquadratic2(const Vector & x, const double t ){
  return x(0);
}
void  quadratic2V(const Vector & x, const double t, Vector & w){
  w(0)  = x(0)*x(1);
  w(1)  = x(0)*x(1);
}
// double cubic(const Vector & x, const double t ){
//   return x(0)*x(0)*x(0) + x(1)*x(1)*x(1) + x(0)*x(1)*x(1);
// }
// double Dxcubic(const Vector & x, const double t ){
//   return 3*x(0)*x(0) + x(1)*x(1);
// }
// double Dycubic(const Vector & x, const double t ){
//   return 3*x(1)*x(1) + 2*x(0)*x(1);
// }
// void  cubicV(const Vector & x, const double t, Vector & w){
//   w(0)  = x(0)*x(0)*x(0) + x(1)*x(1)*x(1) + x(0)*x(1)*x(1);
//   w(1)  = x(0)*x(0)*x(0) + x(1)*x(1)*x(1) + x(1)*x(0)*x(0);
// }
double cubic(const Vector & x, const double t ){
  return x(0)*x(0)*x(0);
}
double Dxcubic(const Vector & x, const double t ){
  return 3*x(0)*x(0);
}
double Dycubic(const Vector & x, const double t ){
  return 0.;
}
void  cubicV(const Vector & x, const double t, Vector & w){
  w(0)  = x(0)*x(0)*x(0);
  w(1)  = x(1)*x(1)*x(1);
}

//---------------------------------------------------------------------------

int main(int argc, char *argv[]){

  int refLvl = 4;
  string mesh_file = "./meshes/tri-square-testAn.mesh";

  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;


  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&ordU, "-oU", "--ordU",
                "Velocity space polynomial order (default: 2)");
  args.AddOption(&ordP, "-oP", "--ordP",
                "Pressure space polynomial order (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--ordZ",
                "Laplacian of vector potential space polynomial order (default: 1)");
  args.AddOption(&ordA, "-oA", "--ordA",
                "Vector potential space polynomial order (default: 2)");
  args.Parse();

  const double _dt = 0.5;

  //- generate mesh
  Mesh *_mesh = new Mesh( mesh_file.c_str(), 1, 1 );
  int _dim = _mesh->Dimension();
  
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
  FiniteElementSpace* _UhFES1com = new FiniteElementSpace( _mesh, _UhFEColl );

  // - identify dirichlet nodes
  Array<int> essTagsA(4); essTagsA[0] = 1; essTagsA[1] = 2; essTagsA[2] = 3; essTagsA[3] = 4; // 1N 2E 3S 4W
  Array<int> _essAhTDOF;
  // Array<int> _essZhTDOF;
  int numAtt = _mesh->bdr_attributes.Max();
  Array<int> essBdrA( numAtt ); essBdrA = 0;
  if ( _mesh->bdr_attributes.Size() > 0 ) {
    // search among all possible tags
    for ( int i = 1; i <= _mesh->bdr_attributes.Max(); ++i ){
      // if that tag is marked in the corresponding array in essTags, then flag it
      if( essTagsA.Find( i ) + 1 )
        essBdrA[i-1] = 1;
    }
    _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );
    // _ZhFESpace->GetEssentialTrueDofs( essBdrA, _essZhTDOF );
  }


  std::cout << "***********************************************************\n";
  std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
  std::cout << "***********************************************************\n";
  // std::cout << "Dir A      "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "essBdr A   ";  essBdrA.Print(  mfem::out,    essBdrA.Size() ); std::cout<< "\n";
  // std::cout << "essTags A  ";  essTagsA.Print( mfem::out,   essTagsA.Size() ); std::cout<< "\n";
  // std::cout << "Z on dir A "; _essZhTDOF.Print(mfem::out, _essZhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "Bdr atts   "; _mesh->bdr_attributes.Print(mfem::out, _mesh->bdr_attributes.Size() ); std::cout<< "\n";
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
  ConstantCoefficient dt( _dt );
  ConstantCoefficient mdt( -_dt );

  Array<int> ords(3);
  ords[0] = 2*ordU + ordU-1;       // ( (u·∇)u, v )
  ords[1] = ordZ + ordA-1 + ordU;  // (   z ∇A, v )
  ords[2] = ordU + ordA-1 + ordA;  // ( (u·∇A), B )
  const IntegrationRule *ir = &IntRules.Get( Geometry::Type::TRIANGLE, ords.Max() );
  std::cout<<"Selecting integrator of order "<<ords.Max()<<std::endl;

  // Declare a bunch of functions to use as tests
  int numFuncs = 5;
  void(  *funsV[numFuncs])( const Vector & x, const double t, Vector & u );
  funsV[0]  = constantV;
  funsV[1]  = linearV;
  funsV[2]  = quadraticV;
  funsV[3]  = quadratic2V;
  funsV[4]  = cubicV;
  double( *funs[numFuncs])( const Vector & x, const double t );
  funs[0]   = constant;
  funs[1]   = linear;
  funs[2]   = quadratic;
  funs[3]   = quadratic2;
  funs[4]   = cubic;
  double( *Dxfuns[numFuncs])( const Vector & x, const double t );
  Dxfuns[0] = Dxconstant;
  Dxfuns[1] = Dxlinear;
  Dxfuns[2] = Dxquadratic;
  Dxfuns[3] = Dxquadratic2;
  Dxfuns[4] = Dxcubic;
  double( *Dyfuns[numFuncs])( const Vector & x, const double t );
  Dyfuns[0] = Dyconstant;
  Dyfuns[1] = Dylinear;
  Dyfuns[2] = Dyquadratic;
  Dyfuns[3] = Dyquadratic2;
  Dyfuns[4] = Dycubic;
  std::string names[numFuncs];
  names[0]  = "constant";
  names[1]  = "linear";
  names[2]  = "quadratic";
  names[3]  = "quadratic2";
  names[4]  = "cubic";
  double intLapA[numFuncs];
  intLapA[0]= 0.;
  intLapA[1]= 0.;
  intLapA[2]= 2.;
  intLapA[3]= 0.;
  intLapA[4]= 3.;
  
  
  for ( int i = 0; i < numFuncs; ++i ){
    std::cout<<"*************************************************************"<<std::endl;
    std::cout<<"Testing for " << names[i] << " functions"                     <<std::endl;
    std::cout<<"*************************************************************"<<std::endl;


    // assemble u
    VectorFunctionCoefficient uFuncCoeff( _dim, funsV[i] );
    GridFunction uFun( _UhFESpace );
    uFun.ProjectCoefficient( uFuncCoeff );
    VectorGridFunctionCoefficient uCoeff( &uFun );

    // assemble p
    FunctionCoefficient pFuncCoeff( funs[i] );
    GridFunction pFun( _PhFESpace );
    pFun.ProjectCoefficient( pFuncCoeff );
    GridFunctionCoefficient pCoeff( &pFun );

    // assemble A and ∇A
    FunctionCoefficient AFuncCoeff( funs[i] );
    GridFunction aFun(_AhFESpace);
    aFun.ProjectCoefficient( AFuncCoeff );
    VectorArrayCoefficient gradACoeff( _dim );
    // - this mimics what happens inside my own integrator: there is an extra error associated with computing the derivative this way
    Array< GridFunction > gradAGridFunc( _dim );
    for ( int i = 0; i < _dim; ++i ){
      gradAGridFunc[i].SetSpace( _AhFESpace );
      aFun.GetDerivative( 1, i, gradAGridFunc[i] );   // pass 1 here: seems like for GetDerivative the components are 1-idxed and not 0-idxed...
      gradACoeff.Set( i, new GridFunctionCoefficient( &gradAGridFunc[i] ) );               //gradACoeff should take ownership here
    }
    // - this computes the derivative exaclty, instead
    // VectorArrayCoefficient gradACoeff( _dim );
    // gradACoeff.Set( 0, new FunctionCoefficient( Dxfuns[i] ) );               //gradACoeff should take ownership here
    // gradACoeff.Set( 1, new FunctionCoefficient( Dyfuns[i] ) );               //gradACoeff should take ownership here


    // assemble z and [z,0]', [0,z]'
    FunctionCoefficient zFuncCoeff( funs[i] );
    GridFunction zFun( _ZhFESpace );
    zFun.ProjectCoefficient( zFuncCoeff );
    GridFunctionCoefficient zCoeff( &zFun );
    VectorArrayCoefficient z0Coeff( _dim );
    z0Coeff.Set( 0, new GridFunctionCoefficient( zCoeff ) );
    z0Coeff.Set( 1, new ConstantCoefficient( 0. ) );
    VectorArrayCoefficient z1Coeff( _dim );
    z1Coeff.Set( 0, new ConstantCoefficient( 0. ) );
    z1Coeff.Set( 1, new GridFunctionCoefficient( zCoeff ) );


    // put them all together
    BlockVector x(offsets);
    x.GetBlock(0) = uFun;
    x.GetBlock(1) = pFun;
    x.GetBlock(2) = zFun;
    x.GetBlock(3) = aFun;
    std::cout<<"Initialised"<<std::endl;


    // Assemble matrices using MFEM integrators -------------------------------
    // Fu
    NonlinearForm uuvarf( _UhFESpace );
    uuvarf.AddDomainIntegrator(new VectorMassIntegrator( one ));                        // <u,v>
    uuvarf.AddDomainIntegrator(new VectorDiffusionIntegrator( dt ));                    // dt*<∇u,∇v>
    uuvarf.AddDomainIntegrator(new VectorConvectionNLFIntegrator( dt ));                // dt*<(u·∇)u,v>
    uuvarf.GetDNFI()->operator[](0)->SetIntRule(ir);
    uuvarf.GetDNFI()->operator[](1)->SetIntRule(ir);
    uuvarf.GetDNFI()->operator[](2)->SetIntRule(ir);
    SparseMatrix Fu = *( dynamic_cast<SparseMatrix*>( &uuvarf.GetGradient( x.GetBlock(0) ) ) );
    std::cout<<"Assembled Fu"<<std::endl;

    // B
    MixedBilinearForm puvarf( _UhFESpace, _PhFESpace );
    puvarf.AddDomainIntegrator(new VectorDivergenceIntegrator(mdt) );                   // dt*<-∇·u,q>
    puvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    puvarf.Assemble();
    puvarf.Finalize();
    SparseMatrix B;
    B.MakeRef( puvarf.SpMat() );
    std::cout<<"Assembled B"<<std::endl;


    // Z1
    // - gotta implement it one component at a time 'coz life sucks
    MixedBilinearForm uzvarf( _ZhFESpace, _UhFES1com );
    uzvarf.AddDomainIntegrator(new MixedScalarMassIntegrator( *(gradACoeff.GetCoeff(0)) ) );        // dt*<z ∇A,v>, fixed A
    uzvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    uzvarf.Assemble();
    uzvarf.Finalize();
    SparseMatrix Z1u;
    Z1u.MakeRef( uzvarf.SpMat() );
    Z1u *= _dt;
    MixedBilinearForm vzvarf( _ZhFESpace, _UhFES1com );
    vzvarf.AddDomainIntegrator(new MixedScalarMassIntegrator( *(gradACoeff.GetCoeff(1)) ) );        // dt*<z ∇A,v>, fixed A
    vzvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    vzvarf.Assemble();
    vzvarf.Finalize();
    SparseMatrix Z1v;
    Z1v.MakeRef( vzvarf.SpMat() );
    Z1v *= _dt;
    // - the resulting matrix is two stacked blocks containing Z1u and Z2v
    // -- Initialize data for Z1 collating data from Z1u and Z1v
    Array<int>    Z1ii( Z1u.NumRows()         + Z1v.NumRows()         + 1 );
    Array<int>    Z1jj( Z1u.NumNonZeroElems() + Z1v.NumNonZeroElems()     ); 
    Array<double> Z1dd( Z1u.NumNonZeroElems() + Z1v.NumNonZeroElems()     );
    for ( int ii = 0; ii < Z1u.NumRows(); ++ii )
      Z1ii[ii]                 = Z1u.GetI()[ii];
    for ( int ii = 0; ii < Z1v.NumRows() + 1; ++ii )
      Z1ii[ii + Z1u.NumRows()] = Z1v.GetI()[ii] + Z1u.NumNonZeroElems();
    for ( int ii = 0; ii < Z1u.NumNonZeroElems(); ++ii ){
      Z1jj[ii] = Z1u.GetJ()[ii];
      Z1dd[ii] = Z1u.GetData()[ii];
    }
    for ( int ii = 0; ii < Z1v.NumNonZeroElems(); ++ii ){
      Z1jj[ii+Z1u.NumNonZeroElems()] = Z1v.GetJ()[ii];
      Z1dd[ii+Z1u.NumNonZeroElems()] = Z1v.GetData()[ii];
    }
    // -- Finally assemble sparse matrix
    SparseMatrix Z1( Z1ii.GetData(), Z1jj.GetData(), Z1dd.GetData(), _UhFESpace->GetTrueVSize(), _ZhFESpace->GetTrueVSize(), false, false, true );
    std::cout<<"Assembled Z1"<<std::endl;


    // Z2
    // - gotta implement it one component at a time 'coz life sucks
    MixedBilinearForm uavarf( _AhFESpace, _UhFES1com );
    uavarf.AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator( z0Coeff ) );           // dt*<z ∇A, v>, fixed z
    uavarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    uavarf.Assemble();
    uavarf.Finalize();
    SparseMatrix Z2u;
    Z2u.MakeRef( uavarf.SpMat() );
    Z2u *= _dt;
    MixedBilinearForm vavarf( _AhFESpace, _UhFES1com );
    vavarf.AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator( z1Coeff ) );           // dt*<z ∇A, v>, fixed z
    vavarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    vavarf.Assemble();
    vavarf.Finalize();
    SparseMatrix Z2v;
    Z2v.MakeRef( vavarf.SpMat() );
    Z2v *= _dt;
    // - the resulting matrix is two stacked blocks containing Z2u and Z2v
    // -- Initialize data for Z2 collating data from Z2u and Z2v
    Array<int>    Z2ii( Z2u.NumRows()         + Z2v.NumRows()         + 1 );
    Array<int>    Z2jj( Z2u.NumNonZeroElems() + Z2v.NumNonZeroElems()     ); 
    Array<double> Z2dd( Z2u.NumNonZeroElems() + Z2v.NumNonZeroElems()     );
    for ( int ii = 0; ii < Z2u.NumRows(); ++ii )
      Z2ii[ii]                 = Z2u.GetI()[ii];
    for ( int ii = 0; ii < Z2v.NumRows() + 1; ++ii )
      Z2ii[ii + Z2u.NumRows()] = Z2v.GetI()[ii] + Z2u.NumNonZeroElems();
    for ( int ii = 0; ii < Z2u.NumNonZeroElems(); ++ii ){
      Z2jj[ii] = Z2u.GetJ()[ii];
      Z2dd[ii] = Z2u.GetData()[ii];
    }
    for ( int ii = 0; ii < Z2v.NumNonZeroElems(); ++ii ){
      Z2jj[ii+Z2u.NumNonZeroElems()] = Z2v.GetJ()[ii];
      Z2dd[ii+Z2u.NumNonZeroElems()] = Z2v.GetData()[ii];
    }
    // -- Finally assemble sparse matrix
    SparseMatrix Z2( Z2ii.GetData(), Z2jj.GetData(), Z2dd.GetData(), _UhFESpace->GetTrueVSize(), _AhFESpace->GetTrueVSize(), false, false, true );
    std::cout<<"Assembled Z2"<<std::endl;


    // Mz
    BilinearForm zzvarf( _ZhFESpace );
    zzvarf.AddDomainIntegrator(new MassIntegrator( one ));                              // <z,x>
    zzvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    zzvarf.Assemble();
    zzvarf.Finalize();
    SparseMatrix Mz;
    Mz.MakeRef( zzvarf.SpMat() );
    std::cout<<"Assembled Mz"<<std::endl;


    // K
    MyMixedBilinearForm zavarf( _AhFESpace, _ZhFESpace );
    zavarf.AddDomainIntegrator(  new DiffusionIntegrator(one) );                        // <∇A,∇x>
    zavarf.AddBdrFaceIntegrator( new BoundaryFaceDiffusionIntegrator(mone), essBdrA );  // -<∇A·n,x>_GD
    zavarf.Assemble();
    zavarf.Finalize();
    SparseMatrix K;
    K.MakeRef( zavarf.SpMat() );
    std::cout<<"Assembled K"<<std::endl;


    // Y
    // - gotta implement it one component at a time 'coz life sucks
    MixedBilinearForm auvarf( _UhFES1com, _AhFESpace );
    auvarf.AddDomainIntegrator(new MixedScalarMassIntegrator( *(gradACoeff.GetCoeff(0)) ) );           // dt*<u ∇A,B>, fixed A
    auvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    auvarf.Assemble();
    auvarf.Finalize();
    SparseMatrix Yu;
    Yu.MakeRef( auvarf.SpMat() );
    Yu *= _dt;
    MixedBilinearForm avvarf( _UhFES1com, _AhFESpace );
    avvarf.AddDomainIntegrator(new MixedScalarMassIntegrator( *(gradACoeff.GetCoeff(1)) ) );           // dt*<u ∇A,B>, fixed A
    avvarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    avvarf.Assemble();
    avvarf.Finalize();
    SparseMatrix Yv;
    Yv.MakeRef( avvarf.SpMat() );
    Yv *= _dt;
    // - the resulting matrix is two side-by-side blocks containing Yu and Yv
    // -- Initialize data for Y collating data from Yu and Yv
    Array<int>    Yii( _AhFESpace->GetTrueVSize() + 1 );
    Array<int>    Yjj( Yu.NumNonZeroElems() + Yv.NumNonZeroElems() );
    Array<double> Ydd( Yu.NumNonZeroElems() + Yv.NumNonZeroElems() );
    Yii[0] = 0;
    for ( int ii = 0; ii < _AhFESpace->GetTrueVSize(); ++ii ){
      Yii[ii+1] = Yu.GetI()[ii+1] + Yv.GetI()[ii+1];
      int dOffU = Yu.GetI()[ii+1] - Yu.GetI()[ii];
      int dOffV = Yv.GetI()[ii+1] - Yv.GetI()[ii];
      for ( int jj = 0; jj < dOffU; ++jj ){
        Yjj[ Yii[ii]         + jj ] = Yu.GetJ()[    Yu.GetI()[ii] + jj ];
        Ydd[ Yii[ii]         + jj ] = Yu.GetData()[ Yu.GetI()[ii] + jj ];
      }
      for ( int jj = 0; jj < dOffV; ++jj ){
        Yjj[ Yii[ii] + dOffU + jj ] = Yv.GetJ()[    Yv.GetI()[ii] + jj ] + _UhFES1com->GetTrueVSize();
        Ydd[ Yii[ii] + dOffU + jj ] = Yv.GetData()[ Yv.GetI()[ii] + jj ];
      }
    }
    // -- Finally assemble sparse matrix
    SparseMatrix Y( Yii.GetData(), Yjj.GetData(), Ydd.GetData(), _AhFESpace->GetTrueVSize(), _UhFESpace->GetTrueVSize(), false, false, true );
    std::cout<<"Assembled Y"<<std::endl;


    // Fa
    BilinearForm aavarf( _AhFESpace );
    aavarf.AddDomainIntegrator(new MassIntegrator( one ));                              // <A,B>
    aavarf.AddDomainIntegrator(new DiffusionIntegrator( dt ));                          // dt*<∇A,∇B>
    aavarf.AddDomainIntegrator(new ConvectionIntegrator( uCoeff, _dt ));                // dt*<(u·∇)A,B>
    aavarf.GetDBFI()->operator[](0)->SetIntRule(ir);
    aavarf.GetDBFI()->operator[](1)->SetIntRule(ir);
    aavarf.GetDBFI()->operator[](2)->SetIntRule(ir);
    aavarf.Assemble();
    aavarf.Finalize();
    SparseMatrix Fa;
    Fa.MakeRef( aavarf.SpMat() );
    std::cout<<"Assembled Fa"<<std::endl;



    // Assemble matrices using own integrator ---------------------------------
    Array< FiniteElementSpace* > feSpaces(4);
    feSpaces[0] = _UhFESpace;
    feSpaces[1] = _PhFESpace;
    feSpaces[2] = _ZhFESpace;
    feSpaces[3] = _AhFESpace;
    BlockNonlinearForm _IMHD2DOperator;
    _IMHD2DOperator.SetSpaces( feSpaces );
    _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DIntegrator( _dt, 1, 1, 1 ) );
    _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( _dt, 1, 1, 1 ), essBdrA );
    BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( x ) );
    SparseMatrix _Fu = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,0) ) );
    SparseMatrix _Z1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) ) );
    SparseMatrix _Z2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) ) );
    SparseMatrix _B  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,0) ) );
    SparseMatrix _Mz = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(2,2) ) );
    SparseMatrix _K  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(2,3) ) );
    SparseMatrix _Y  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,0) ) );
    SparseMatrix _Fa = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,3) ) );
    std::cout<<"Assembled own integrator"<<std::endl;



    // Compare Gradients ------------------------------------------------------
    // - The matrices themselves should be equal (up to machine precision)
    SparseMatrix* dFu = Add( 1., _Fu, -1., Fu );
    SparseMatrix* dZ1 = Add( 1., _Z1, -1., Z1 );
    SparseMatrix* dZ2 = Add( 1., _Z2, -1., Z2 );
    SparseMatrix* dB  = Add( 1., _B , -1., B  );
    SparseMatrix* dMz = Add( 1., _Mz, -1., Mz );
    SparseMatrix* dK  = Add( 1., _K , -1., K  );
    SparseMatrix* dY  = Add( 1., _Y , -1., Y  );
    SparseMatrix* dFa = Add( 1., _Fa, -1., Fa );
    std::cout<<"|dFu|_inf = "<< dFu->MaxNorm() <<std::endl;
    std::cout<<"|dZ1|_inf = "<< dZ1->MaxNorm() <<std::endl;
    std::cout<<"|dZ2|_inf = "<< dZ2->MaxNorm() <<std::endl;
    std::cout<<"|dB |_inf = "<<  dB->MaxNorm() <<std::endl;
    std::cout<<"|dMz|_inf = "<< dMz->MaxNorm() <<std::endl;
    std::cout<<"|dK |_inf = "<<  dK->MaxNorm() <<std::endl;
    std::cout<<"|dY |_inf = "<<  dY->MaxNorm() <<std::endl;
    std::cout<<"|dFa|_inf = "<< dFa->MaxNorm() <<std::endl;
    delete dFu; 
    delete dZ1; 
    delete dZ2; 
    delete dB;  
    delete dMz; 
    delete dK;  
    delete dY;  
    delete dFa; 

    std::string myfilename;
    std::ofstream myfile;
    myfile.precision(std::numeric_limits< double >::max_digits10);

    myfilename = "./results/operators/MFEM_Fu.dat";
    myfile.open( myfilename );
    Fu.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_Z1.dat";
    myfile.open( myfilename );
    Z1.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_Z2.dat";
    myfile.open( myfilename );
    Z2.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_B.dat";
    myfile.open( myfilename );
    B.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_Mz.dat";
    myfile.open( myfilename );
    Mz.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_K.dat";
    myfile.open( myfilename );
    K.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_Y.dat";
    myfile.open( myfilename );
    Y.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/MFEM_Fa.dat";
    myfile.open( myfilename );
    Fa.PrintMatlab(myfile);
    myfile.close( );

    myfilename = "./results/operators/my_Fu.dat";
    myfile.open( myfilename );
    _Fu.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_Z1.dat";
    myfile.open( myfilename );
    _Z1.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_Z2.dat";
    myfile.open( myfilename );
    _Z2.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_B.dat";
    myfile.open( myfilename );
    _B.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_Mz.dat";
    myfile.open( myfilename );
    _Mz.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_K.dat";
    myfile.open( myfilename );
    _K.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_Y.dat";
    myfile.open( myfilename );
    _Y.PrintMatlab(myfile);
    myfile.close( );
    myfilename = "./results/operators/my_Fa.dat";
    myfile.open( myfilename );
    _Fa.PrintMatlab(myfile);
    myfile.close( );



    // Compare application ----------------------------------------------------
    BlockVector  y(offsets);  y = 0.;
    BlockVector _y(offsets); _y = 0.;
    BlockVector dy(offsets); dy = 0.;

    // using MFEM integrators:
    // u
    uuvarf.Mult(        x.GetBlock(0), y.GetBlock(0) ); // multiply the nonlinear form
    B.AddMultTranspose( x.GetBlock(1), y.GetBlock(0) );
    Z1.AddMult(         x.GetBlock(2), y.GetBlock(0) );    //<z ∇A,v> is a trilinear form! Newton linearisation would "double" its contribution,
    // Z2.AddMult(         x.GetBlock(3), y.GetBlock(0) ); // so just include one of Z1 or Z2 (and check that they give the same result)
    // p
    B.AddMult(          x.GetBlock(0), y.GetBlock(1) );
    // z
    Mz.AddMult(         x.GetBlock(2), y.GetBlock(2) );
    K.AddMult(          x.GetBlock(3), y.GetBlock(2) );
    // A
    // Y.AddMult(          x.GetBlock(0), y.GetBlock(3) ); //same here: <u ∇A,B> is trilinear, and one contribution is already included in Fa
    Fa.AddMult(         x.GetBlock(3), y.GetBlock(3) );


    // using own integrator:
    _IMHD2DOperator.Mult( x, _y );


    // compare
    dy = y; dy -= _y;
    Vector Z1zmZ2A( x.GetBlock(0).Size() ); Z1zmZ2A = 0.;
    Z1.AddMult( x.GetBlock(2), Z1zmZ2A      );
    Z2.AddMult( x.GetBlock(3), Z1zmZ2A, -1. );
    Vector du = dy.GetBlock(0);
    Vector dp = dy.GetBlock(1);
    Vector dz = dy.GetBlock(2);
    Vector dA = dy.GetBlock(3);
    std::cout<<"|du|2 =        "<<      du.Norml2() <<",\t|du|inf        = "<<      du.Normlinf() <<std::endl;
    std::cout<<"|dp|2 =        "<<      dp.Norml2() <<",\t|dp|inf        = "<<      dp.Normlinf() <<std::endl;
    std::cout<<"|dz|2 =        "<<      dz.Norml2() <<",\t|dz|inf        = "<<      dz.Normlinf() <<std::endl;
    std::cout<<"|dA|2 =        "<<      dA.Norml2() <<",\t|dA|inf        = "<<      dA.Normlinf() <<std::endl;
    std::cout<<"|Z1*z-Z2*A|2 = "<< Z1zmZ2A.Norml2() <<",\t|Z1*z-Z2*A|inf = "<< Z1zmZ2A.Normlinf() <<std::endl;
    // I've implemented my own K = - <∇·∇A,x> = <∇A,∇x> - <∇A·n,x>_bdr. One way to check its correctness, is if the integral of the laplacian on the domain matches it
    //  (that is, multiply K by the vector representation of A, and then sum its entries)
    Vector KA(y.GetBlock(2).Size());
    K.Mult( x.GetBlock(3), KA );
    std::cout<<"d(int divgrad A)= "<< KA.Sum() + intLapA[i] <<std::endl;

 
  }









  // // // assemble bilinear form
  // // MyMixedBilinearForm mVarf( _AhFESpace, _ZhFESpace );
  // // ConstantCoefficient mone( -1.0 );
  // // mVarf.AddBdrFaceIntegrator( new BoundaryFaceDiffusionIntegrator(mone), essBdrA );
  // // mVarf.Assemble();
  // // mVarf.Finalize();

  // // assemble bilinear form
  // MyMixedBilinearForm mVarf( _AhFESpace, _ZhFESpace );
  // mVarf.AddDomainIntegrator(  new DiffusionIntegrator(one) );
  // mVarf.AddBdrFaceIntegrator( new BoundaryFaceDiffusionIntegrator(mone), essBdrA );
  // mVarf.Assemble();
  // mVarf.Finalize();
  // SparseMatrix Kd;

  // Array<int> empty(0);
  // // mVarf.FormRectangularSystemMatrix( _essAhTDOF, empty, Kd );
  // mVarf.FormRectangularSystemMatrix( empty, empty, Kd );
  // Kd.MakeRef( mVarf.SpMat() );
  // Kd.SetGraphOwner(true);
  // Kd.SetDataOwner(true);
  // // Kd.Print(mfem::out, _essAhTDOF.Size());
  // mVarf.LoseMat();


  // Array< FiniteElementSpace* > feSpaces(4);
  // feSpaces[0] = _UhFESpace;
  // feSpaces[1] = _PhFESpace;
  // feSpaces[2] = _ZhFESpace;
  // feSpaces[3] = _AhFESpace;
  // BlockNonlinearForm _IMHD2DOperator;
  // _IMHD2DOperator.SetSpaces( feSpaces );
  // Array<int> newEssBdrA( numAtt ); newEssBdrA = essBdrA;
  // _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DIntegrator( 1, 1, 1, 1 ) );
  // _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( 1, 1, 1, 1 ), newEssBdrA );
  // Array<int> offsets(5);
  // offsets[0] = 0;
  // offsets[1] = _UhFESpace->GetTrueVSize();
  // offsets[2] = _PhFESpace->GetTrueVSize();
  // offsets[3] = _ZhFESpace->GetTrueVSize();
  // offsets[4] = _AhFESpace->GetTrueVSize();
  // offsets.PartialSum();
  // BlockVector dummy(offsets), dummy2(offsets), dummy3(offsets); dummy = 0.; dummy2 = 0.; dummy3 = 0.;
  // std::cout << "Computing gradient\n";
  // Operator* J = &_IMHD2DOperator.GetGradient( dummy );
  // BlockOperator* Jb = dynamic_cast<BlockOperator*>(J);

  // std::cout << "Casting to matrix\n";
  // SparseMatrix _K  = *( dynamic_cast<SparseMatrix*>( &Jb->GetBlock(2,3) ) );

  // GridFunction myA(_AhFESpace);
  // myA.ProjectCoefficient(one);

  // Vector res(Kd.NumRows());
  // Kd.Mult(myA,res);
  // std::cout << "Kd*A for constant A:\n    "; res.Print(mfem::out, res.Size() );
  // _K.Mult(myA,res);
  // std::cout << " vs "; res.Print(mfem::out, res.Size() );
  // dummy2.GetBlock(3) = myA;
  // _IMHD2DOperator.Mult(dummy2,dummy3);
  // std::cout << " vs "; dummy3.GetBlock(2).Print(mfem::out, dummy3.GetBlock(2).Size() ); std::cout<<"\n";

  // FunctionCoefficient aFuncoeff( linear );
  // myA.ProjectCoefficient(aFuncoeff);
  // Kd.Mult(myA,res);
  // std::cout << "Kd*A for linear A:\n    "; res.Print(mfem::out, res.Size() );
  // _K.Mult(myA,res);
  // std::cout << " vs "; res.Print(mfem::out, res.Size() );
  // dummy2.GetBlock(3) = myA;
  // _IMHD2DOperator.Mult(dummy2,dummy3);
  // std::cout << " vs "; dummy3.GetBlock(2).Print(mfem::out, dummy3.GetBlock(2).Size() ); std::cout<<"\n";

  // FunctionCoefficient aFuncoeff2( quadratic );
  // myA.ProjectCoefficient(aFuncoeff2);
  // Kd.Mult(myA,res);
  // std::cout << "Kd*A for quadratic A:\n    "; res.Print(mfem::out, res.Size() );
  // _K.Mult(myA,res);
  // std::cout << " vs "; res.Print(mfem::out, res.Size() );
  // dummy2.GetBlock(3) = myA;
  // _IMHD2DOperator.Mult(dummy2,dummy3);
  // std::cout << " vs "; dummy3.GetBlock(2).Print(mfem::out, dummy3.GetBlock(2).Size() ); std::cout<<"\n";

  // FunctionCoefficient aFuncoeff3( quadratic2 );
  // myA.ProjectCoefficient(aFuncoeff3);
  // Kd.Mult(myA,res);
  // std::cout << "Kd*A for quadratic2 A:\n    "; res.Print(mfem::out, res.Size() );
  // _K.Mult(myA,res);
  // std::cout << " vs "; res.Print(mfem::out, res.Size() );
  // dummy2.GetBlock(3) = myA;
  // _IMHD2DOperator.Mult(dummy2,dummy3);
  // std::cout << " vs "; dummy3.GetBlock(2).Print(mfem::out, dummy3.GetBlock(2).Size() ); std::cout<<"\n";



  // string myfilename;
  // ofstream myfile;
  // myfile.precision(std::numeric_limits< double >::max_digits10);

  // std::cout<<"All done\n";

  // myfilename = "./results/ugaKd.dat";
  // myfile.open( myfilename );
  // Kd.PrintMatlab(myfile);
  // myfile.close( );
  // myfilename = "./results/ugaK.dat";
  // myfile.open( myfilename );
  // _K.PrintMatlab(myfile);
  // myfile.close( );

  delete _UhFESpace;
  delete _PhFESpace;
  delete _AhFESpace;
  delete _ZhFESpace;
  delete _UhFEColl;
  delete _PhFEColl;
  delete _AhFEColl;
  delete _ZhFEColl;
  delete _mesh;



  return 0;
}







