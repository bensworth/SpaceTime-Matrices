//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "imhd2dintegrator.hpp"
#include "vectorconvectionintegrator.hpp"
//---------------------------------------------------------------------------
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;

#define MULT_BY_DT

//---------------------------------------------------------------------------
// handy funcs for debug
void vFun_Quad(const Vector & x, Vector & w){
  w(0) = x(0)*x(0);
  w(1) = x(1)*x(1);
}
void vFun_Lin(const Vector & x, Vector & w){
  w(0) = x(0);
  w(1) = x(1);
}
double vFun_Const(const Vector & x){
  return 1.;
}
double aFun_Quad(const Vector & x){
  return x(0)*x(0) + x(1)*x(1);
}
double aFun_Lin(const Vector & x){
  return x(0) + x(1);
}
double aFun_Const(const Vector & x){
  return 1.;
}
//---------------------------------------------------------------------------
int main(int argc, char *argv[]){

  // problem parameters
  const double _dt =  1.0;
  const double _mu =  1.0;
  const double _mu0 = 1.0;
  const double _eta = 1.0;

  // initialise mesh
  std::string meshName = "./meshes/tri-square-test.mesh";

  Mesh* _mesh = new Mesh( meshName.c_str(), 1, 1 );
  int _dim = _mesh->Dimension();
  int refLvl = 1;

  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();


  // initialise FE info
  H1_FECollection*     _UhFEColl = new H1_FECollection( 2, _dim );
  H1_FECollection*     _PhFEColl = new H1_FECollection( 1, _dim );
  H1_FECollection*     _ZhFEColl = new H1_FECollection( 1, _dim );
  H1_FECollection*     _AhFEColl = new H1_FECollection( 2, _dim );

  FiniteElementSpace* _UhFESpace = new FiniteElementSpace( _mesh, _UhFEColl, _dim );
  FiniteElementSpace* _PhFESpace = new FiniteElementSpace( _mesh, _PhFEColl );
  FiniteElementSpace* _ZhFESpace = new FiniteElementSpace( _mesh, _ZhFEColl );
  FiniteElementSpace* _AhFESpace = new FiniteElementSpace( _mesh, _AhFEColl );

  std::cout << "***********************************************************\n";
  std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
  std::cout << "***********************************************************\n";

  // dirichlet nodes
  // Array<int> _essUhTDOF;
  // Array<int> _essPhTDOF;
  // Array<int> _essZhTDOF;
  // Array<int> _essAhTDOF;
  Array<int> essBdrU, essBdrP, essBdrZ, essBdrA;




  if ( _mesh->bdr_attributes.Size() > 0 ) {
    int numAtts = _mesh->bdr_attributes.Max();
    essBdrU.SetSize( numAtts ); essBdrP.SetSize( numAtts ); essBdrZ.SetSize( numAtts ); essBdrA.SetSize( numAtts );
    essBdrU = 0; essBdrP = 0; essBdrZ = 0; essBdrA = 0;
    for ( int i = 0; i < _mesh->bdr_attributes.Max(); ++i ){
      if( _mesh->bdr_attributes[i] == 1 )
        essBdrU[i] = 1;
      if( _mesh->bdr_attributes[i] == 2 )
        essBdrP[i] = 1;
      if( _mesh->bdr_attributes[i] == 3 )
        essBdrZ[i] = 1;
      if( _mesh->bdr_attributes[i] == 4 )
        essBdrA[i] = 1;
    }
  }

  Array< Array<int> * > _essTags(4);
  _essTags[0] = &essBdrU;
  _essTags[1] = &essBdrP;
  _essTags[2] = &essBdrZ;
  _essTags[3] = &essBdrA;

  Array<int> _essUhTDOF; _UhFESpace->GetEssentialTrueDofs( essBdrU, _essUhTDOF );
  Array<int> _essPhTDOF; _PhFESpace->GetEssentialTrueDofs( essBdrP, _essPhTDOF );
  Array<int> _essZhTDOF; _ZhFESpace->GetEssentialTrueDofs( essBdrZ, _essZhTDOF );
  Array<int> _essAhTDOF; _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );



  // initialise block operator
  Array< FiniteElementSpace * > _feSpaces(4);
  _feSpaces[0] = _UhFESpace;
  _feSpaces[1] = _PhFESpace;
  _feSpaces[2] = _ZhFESpace;
  _feSpaces[3] = _AhFESpace;
  BlockNonlinearForm _MHDOperator( _feSpaces );
  _MHDOperator.AddDomainIntegrator( new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ) );
  Array<Vector *> temp(4);
  Vector tempu(_UhFESpace->GetTrueVSize()); tempu= 0.;
  Vector tempp(_PhFESpace->GetTrueVSize()); tempp= 0.;
  Vector tempz(_ZhFESpace->GetTrueVSize()); tempz= 0.;
  Vector tempa(_AhFESpace->GetTrueVSize()); tempa= 0.;
  temp[0] = &tempu;
  temp[1] = &tempp;
  temp[2] = &tempz;
  temp[3] = &tempa;
  _MHDOperator.SetEssentialBC( _essTags, temp );


  // initialise linearised state
  GridFunction _w(_UhFESpace);
  GridFunction _r(_PhFESpace);
  GridFunction _y(_ZhFESpace);
  GridFunction _c(_AhFESpace);
  // -- only values on dirichlet nodes
   _w = 0.; _w.SetSubVector( _essUhTDOF, 1.);
   _r = 0.; _r.SetSubVector( _essPhTDOF, 2.);
   _y = 0.; _y.SetSubVector( _essZhTDOF, 3.);
   _c = 0.; _c.SetSubVector( _essAhTDOF, 4.);
  // // -- constant fields
  // _w = 1.;
  // _r = 0.;
  // _y = 1.;
  // _c = 1.;
  // -- constant-gradient fields
  // VectorFunctionCoefficient vfun(_dim,vFun_Quad);
  // FunctionCoefficient       afun(     aFun_Quad);
  // _w.ProjectCoefficient( vfun );
  // _r.ProjectCoefficient( afun );
  // _y.ProjectCoefficient( afun );
  // _c.ProjectCoefficient( afun );
  

  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = _UhFESpace->GetTrueVSize();
  offsets[2] = _PhFESpace->GetTrueVSize();
  offsets[3] = _ZhFESpace->GetTrueVSize();
  offsets[4] = _AhFESpace->GetTrueVSize();
  offsets.PartialSum();
  BlockVector x(offsets);
  BlockVector y(offsets);
  x.GetBlock(0) = _w;
  x.GetBlock(1) = _r;
  x.GetBlock(2) = _y;
  x.GetBlock(3) = _c;

  // apply nonlinear operator
  _MHDOperator.Mult(x,y);
  std::cout<<"N(x): - u = "; y.GetBlock(0).Print(); std::cout<<std::endl;
  std::cout<<"      - p = "; y.GetBlock(1).Print(); std::cout<<std::endl;
  std::cout<<"      - z = "; y.GetBlock(2).Print(); std::cout<<std::endl;
  std::cout<<"      - A = "; y.GetBlock(3).Print(); std::cout<<std::endl;

  // compute gradient of nonlinear operator
  BlockOperator* J = dynamic_cast<BlockOperator*>( &_MHDOperator.GetGradient( x ) );



  const SparseMatrix* new00 = dynamic_cast<SparseMatrix*>( &J->GetBlock(0,0) );
  const SparseMatrix* new01 = dynamic_cast<SparseMatrix*>( &J->GetBlock(0,1) );
  const SparseMatrix* new02 = dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) );
  const SparseMatrix* new03 = dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) );
  const SparseMatrix* new10 = dynamic_cast<SparseMatrix*>( &J->GetBlock(1,0) );
  const SparseMatrix* new22 = dynamic_cast<SparseMatrix*>( &J->GetBlock(2,2) );
  const SparseMatrix* new23 = dynamic_cast<SparseMatrix*>( &J->GetBlock(2,3) );
  const SparseMatrix* new30 = dynamic_cast<SparseMatrix*>( &J->GetBlock(3,0) );
  const SparseMatrix* new33 = dynamic_cast<SparseMatrix*>( &J->GetBlock(3,3) );

  std::string   myfilenamenew00 = "./results/new00.dat";
  std::string   myfilenamenew01 = "./results/new01.dat";
  std::string   myfilenamenew02 = "./results/new02.dat";
  std::string   myfilenamenew03 = "./results/new03.dat";
  std::string   myfilenamenew10 = "./results/new10.dat";
  std::string   myfilenamenew22 = "./results/new22.dat";
  std::string   myfilenamenew23 = "./results/new23.dat";
  std::string   myfilenamenew30 = "./results/new30.dat";
  std::string   myfilenamenew33 = "./results/new33.dat";

  std::ofstream myfilenew00;
  std::ofstream myfilenew01;
  std::ofstream myfilenew02;
  std::ofstream myfilenew03;
  std::ofstream myfilenew10;
  std::ofstream myfilenew22;
  std::ofstream myfilenew23;
  std::ofstream myfilenew30;
  std::ofstream myfilenew33;

  myfilenew00.open( myfilenamenew00 );
  myfilenew01.open( myfilenamenew01 );
  myfilenew02.open( myfilenamenew02 );
  myfilenew03.open( myfilenamenew03 );
  myfilenew10.open( myfilenamenew10 );
  myfilenew22.open( myfilenamenew22 );
  myfilenew23.open( myfilenamenew23 );
  myfilenew30.open( myfilenamenew30 );
  myfilenew33.open( myfilenamenew33 );
    
  new00->PrintMatlab(myfilenew00);
  new01->PrintMatlab(myfilenew01);
  new02->PrintMatlab(myfilenew02);
  new03->PrintMatlab(myfilenew03);
  new10->PrintMatlab(myfilenew10);
  new22->PrintMatlab(myfilenew22);
  new23->PrintMatlab(myfilenew23);
  new30->PrintMatlab(myfilenew30);
  new33->PrintMatlab(myfilenew33);

  myfilenew00.close();
  myfilenew01.close();
  myfilenew02.close();
  myfilenew03.close();
  myfilenew10.close();
  myfilenew22.close();
  myfilenew23.close();
  myfilenew30.close();
  myfilenew33.close();







//   //*************************************************************************
//   // Check single blocks
//   //*************************************************************************

  // Fu (0,0) ---------------------------------------------------------------
  NonlinearForm fuVarf(_UhFESpace);

#ifdef MULT_BY_DT
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient Dt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( one ));                // Mu
  fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( muDt ));       // mu*dt*K
  fuVarf.AddDomainIntegrator(new VectorConvectionNLFIntegrator( Dt ));   // dt*W2(w)
#else
  ConstantCoefficient mu( _mu );
  ConstantCoefficient dtinv( 1./_dt );
  ConstantCoefficient one1( 1.0 );
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( dtinv ));
  fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( mu ));
  fuVarf.AddDomainIntegrator(new VectorConvectionNLFIntegrator( one1 ));   // dt*W2(w)
#endif

  const SparseMatrix *pointold00 = dynamic_cast<const SparseMatrix*>( &(fuVarf.GetGradient(_w)) );
  const SparseMatrix old00 = *pointold00;

//   BilinearForm fuVarf(_UhFESpace);

  GridFunction wGridFun( _UhFESpace );
  wGridFun = _w;
  VectorGridFunctionCoefficient* wCoeff = new VectorGridFunctionCoefficient( &wGridFun );

  // W2 can also be represented as ( (u·∇)w, v ) = ( M u, v ), where M = ∇w
  //  so here we assemble ∇w
  MatrixArrayCoefficient* gradWCoeff = new MatrixArrayCoefficient( _dim );
  Array2D< GridFunction > gradWGridFunc( _dim ,_dim );
  for ( int i = 0; i < _dim; ++i ){
    for ( int j = 0; j < _dim; ++j ){
      gradWGridFunc(i,j).SetSpace( _UhFESpace );
      wGridFun.GetDerivative( i+1, j, gradWGridFunc(i,j) );   // pass i+1 here: seems like for GetDerivative the components are 1-idxed and not 0-idxed...
      gradWCoeff->Set( i, j, new GridFunctionCoefficient( &gradWGridFunc(i,j) ) );            //gradWCoeff should take ownership here
    }
  }

// #ifdef MULT_BY_DT
//   ConstantCoefficient muDt( _mu*_dt );
//   ConstantCoefficient one( 1.0 );
//   fuVarf.AddDomainIntegrator(new VectorMassIntegrator( one ));                // Mu
//   fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( muDt ));       // mu*dt*K
//   fuVarf.AddDomainIntegrator(new VectorConvectionIntegrator( *wCoeff, _dt )); // dt*W1(w)
//   fuVarf.AddDomainIntegrator(new VectorMassIntegrator( *gradWCoeff, _dt ));   // dt*W2(w)
// #else
//   ConstantCoefficient mu( _mu );
//   ConstantCoefficient dtinv( 1./_dt );
//   fuVarf.AddDomainIntegrator(new VectorMassIntegrator( dtinv ));
//   fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( mu ));
//   fuVarf.AddDomainIntegrator(new VectorConvectionIntegrator( *wCoeff, _dt ));
//   fuVarf.AddDomainIntegrator(new VectorMassIntegrator( *gradWCoeff, 1.0 ));
// #endif


//   fuVarf.Assemble();
//   fuVarf.Finalize();
//   SparseMatrix old00 = fuVarf.SpMat();


//   delete gradWCoeff;



//   // Z1 (0,2) ---------------------------------------------------------------
  // MixedBilinearForm z1Varf( _ZhFESpace, _UhFESpace );

  GridFunction cGridFun( _AhFESpace );
  cGridFun = _c;

  // assemble ∇c from c
  VectorArrayCoefficient* gradCCoeff = new VectorArrayCoefficient( _dim );
  Array< GridFunction > gradCGridFunc( _dim );
  for ( int i = 0; i < _dim; ++i ){
    gradCGridFunc[i].SetSpace( _AhFESpace );
    cGridFun.GetDerivative( 1, i, gradCGridFunc[i] );   // pass 1 here: seems like for GetDerivative the components are 1-idxed and not 0-idxed...
    gradCCoeff->Set( i, new GridFunctionCoefficient( &gradCGridFunc[i] ) );               //gradCCoeff should take ownership here
  }

  // TODO the situation here is a bit embarassing because you're explicitly building the derivative of c (that is, A).
  //  An alternative would be integration by parts: this would give rise to the boundary term int(zcv·n), and the
  //  two domain terms -int(c∇z·v) and -int(cz∇·v) (but possibly the last is zero? By forcibly imposing ∇·v=0 also on test function? Meh...)

//   z1Varf.AddDomainIntegrator(new MixedVectorProductIntegrator( *gradCCoeff ) );  // ( z ∇c, v )

//   z1Varf.Assemble();
//   z1Varf.Finalize();

//   // - once the matrix is generated, we can get rid of the operator
//   SparseMatrix old02 = z1Varf.SpMat();

//   old02 *= 1./_mu0;

// #ifdef MULT_BY_DT
//   old02 *= _dt;  
// #endif






//   // Z2 (0,3) ---------------------------------------------------------------
//   MixedBilinearForm z2Varf( _AhFESpace, _UhFESpace );

//   GridFunction yGridFun( _ZhFESpace );
//   yGridFun = _y;
//   GridFunctionCoefficient* yCoeff = new GridFunctionCoefficient( &yGridFun );

//   z2Varf.AddDomainIntegrator(new MixedVectorGradientIntegrator( *yCoeff ) );  // ( y ∇A, v )

//   z2Varf.Assemble();
//   z2Varf.Finalize();

//   // - once the matrix is generated, we can get rid of the operator
//   SparseMatrix old03 = z2Varf.SpMat();
//   old03 *= 1./_mu0;

// #ifdef MULT_BY_DT
//   old03 *= _dt;  
// #endif

//   delete yCoeff;



  // B (1,0) ----------------------------------------------------------------
  MixedBilinearForm bVarf( _UhFESpace, _PhFESpace );

#ifdef MULT_BY_DT  
  ConstantCoefficient minusDt( -_dt );
  bVarf.AddDomainIntegrator(new VectorDivergenceIntegrator(minusDt) );
#else
  ConstantCoefficient mone( -1.0 );
  bVarf.AddDomainIntegrator(new VectorDivergenceIntegrator(mone) );
#endif

  bVarf.Assemble();
  bVarf.Finalize();

  SparseMatrix old10 = bVarf.SpMat();



//   Bt (0,1) ---------------------------------------------------------------
//   SparseMatrix old01 = old10.Transpose();




  // Mz (2,2) ---------------------------------------------------------------
  BilinearForm mzVarf(_ZhFESpace);

  // TODO: multiplying by dt shouldn't be necessary either case, but it changes the relative scaling between
  //  this equation and the other in the system. Try and see what happens if you don't rescale (just be consistent
  //  with the definition of Kz)
#ifdef MULT_BY_DT
  ConstantCoefficient dt( _dt );
  mzVarf.AddDomainIntegrator(new MassIntegrator( dt ));                  // Mz
#else
  mzVarf.AddDomainIntegrator(new MassIntegrator( one ));
#endif

  mzVarf.Assemble();
  mzVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  SparseMatrix old22 = mzVarf.SpMat();




  // K (2,3) ----------------------------------------------------------------
  MixedBilinearForm kVarf( _AhFESpace, _ZhFESpace );

  // TODO: multiplying by dt shouldn't be necessary either case, but it changes the relative scaling between
  //  this equation and the other in the system. Try and see what happens if you don't rescale
#ifdef MULT_BY_DT
  kVarf.AddDomainIntegrator(new MixedGradGradIntegrator( dt ) );
#else
  kVarf.AddDomainIntegrator(new MixedGradGradIntegrator( one ) );
#endif

  kVarf.Assemble();
  kVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  SparseMatrix old23 = kVarf.SpMat();



  // Fa (3,3) ---------------------------------------------------------------
  BilinearForm faVarf(_AhFESpace);

#ifdef MULT_BY_DT
  ConstantCoefficient etaDt( _eta*_dt );
  faVarf.AddDomainIntegrator(new MassIntegrator( one ));                 // Ma
  faVarf.AddDomainIntegrator(new DiffusionIntegrator( etaDt ));     // eta *dt K
  faVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, _dt ));  // dt*V(w)

#else
  ConstantCoefficient eta( _eta );
  faVarf.AddDomainIntegrator(new MassIntegrator( dtinv ));
  faVarf.AddDomainIntegrator(new DiffusionIntegrator( eta ));
  faVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, 1.0 ));
#endif


  faVarf.Assemble();
  faVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  SparseMatrix old33 = faVarf.SpMat();



  // // Y (3,0) ----------------------------------------------------------------
  // MixedBilinearForm yVarf( _UhFESpace, _AhFESpace );

  // // assemble ∇c from c
  // yVarf.AddDomainIntegrator(new MixedDotProductIntegrator( *gradCCoeff ) );

  // // TODO Similar considerations as for Z1 apply here: integration by parts would give
  // //  the boundary term int(bcu·n), and the domain term -int(c∇b·u)

  // yVarf.Assemble();
  // yVarf.Finalize();

  // // - once the matrix is generated, we can get rid of the operator
  // SparseMatrix old30 = yVarf.SpMat();

  // delete gradCCoeff;
  // delete wCoeff;






  std::string   myfilenameold00 = "./results/old00.dat";
  // std::string   myfilenameold02 = "./results/old02.dat";
  // std::string   myfilenameold03 = "./results/old03.dat";
  std::string   myfilenameold10 = "./results/old10.dat";
  std::string   myfilenameold22 = "./results/old22.dat";
  std::string   myfilenameold23 = "./results/old23.dat";
  // std::string   myfilenameold30 = "./results/old30.dat";
  std::string   myfilenameold33 = "./results/old33.dat";

  std::ofstream myfileold00;
  // std::ofstream myfileold02;
  // std::ofstream myfileold03;
  std::ofstream myfileold10;
  std::ofstream myfileold22;
  std::ofstream myfileold23;
  // std::ofstream myfileold30;
  std::ofstream myfileold33;

  myfileold00.open( myfilenameold00 );
  // myfileold02.open( myfilenameold02 );
  // myfileold03.open( myfilenameold03 );
  myfileold10.open( myfilenameold10 );
  myfileold22.open( myfilenameold22 );
  myfileold23.open( myfilenameold23 );
  // myfileold30.open( myfilenameold30 );
  myfileold33.open( myfilenameold33 );
    
  old00.PrintMatlab(myfileold00);
  // old02.PrintMatlab(myfileold02);
  // old03.PrintMatlab(myfileold03);
  old10.PrintMatlab(myfileold10);
  old22.PrintMatlab(myfileold22);
  old23.PrintMatlab(myfileold23);
  // old30.PrintMatlab(myfileold30);
  old33.PrintMatlab(myfileold33);

  myfileold00.close();
  // myfileold02.close();
  // myfileold03.close();
  myfileold10.close();
  myfileold22.close();
  myfileold23.close();
  // myfileold30.close();
  myfileold33.close();







  delete _UhFESpace;
  delete _PhFESpace;
  delete _ZhFESpace;
  delete _AhFESpace;

  delete  _UhFEColl;
  delete  _PhFEColl;
  delete  _ZhFEColl;
  delete  _AhFEColl;

  delete _mesh;




  return 0;
}