#include "imhd2dstoperatorassembler.hpp"
#include "vectorconvectionintegrator.hpp"
#include "imhd2dintegrator.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include "spacetimesolver.hpp"
#include "spacetimewavesolver.hpp"

#include <string>
#include <cstring>
#include <iostream>
#include "HYPRE.h"
#include "petsc.h"
#include "mfem.hpp"
#include "operatorssequence.hpp"
#include <experimental/filesystem>
#include <limits>

using namespace mfem;









//******************************************************************************
// Space-time block assembly
//******************************************************************************
// These functions are useful for assembling all the necessary space-time operators

// constructor (uses analytical function for linearised fields)
IMHD2DSTOperatorAssembler::IMHD2DSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
                                                      const int refLvl, const int ordU, const int ordP, const int ordZ, const int ordA,
                                                      const double dt, const double mu, const double eta, const double mu0,
                                                      void(  *f)(const Vector &, double, Vector &),
                                                      double(*g)(const Vector &, double ),
                                                      double(*h)(const Vector &, double ),
                                                      void(  *n)(const Vector &, double, Vector &),
                                                      double(*m)(const Vector &, double ),
                                                      void(  *w)(const Vector &, double, Vector &),
                                                      double(*y)(const Vector &, double ),
                                                      double(*c)(const Vector &, double ),
                                                      void(  *u)(const Vector &, double, Vector &),
                                                      double(*p)(const Vector &, double ),
                                                      double(*z)(const Vector &, double ),
                                                      double(*a)(const Vector &, double ),
                                                      const Array<int>& essTagsU, const Array<int>& essTagsP, const Array<int>& essTagsA,
                                                      int verbose ):
  _comm(comm), _dt(dt),
  _mu(mu),     _eta(eta),                _mu0(mu0), 
  _fFunc(f),   _gFunc(g),                _hFunc(h), 
  _nFunc(n),                             _mFunc(m), 
  _uFunc(u),   _pFunc(p),     _zFunc(z), _aFunc(a),
  _ordU(ordU), _ordP(ordP), _ordZ(ordZ), _ordA(ordA),
  _essTagsU(essTagsU), _essTagsP(essTagsP), _essTagsA(essTagsA),
  _FFu(comm), _MMz(comm), _FFa(comm), _BB(comm), _ZZ1(comm), _ZZ2(comm), _KK(comm), _YY(comm),
  _verbose(verbose){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

  // this is useful otherwise valgrind complains that we didn't initialise variables...
  SetEverythingUnassembled();

	// For each processor:
	//- generate mesh
	_mesh = new Mesh( meshName.c_str(), 1, 1 );
  _dim = _mesh->Dimension();
  
  if ( _dim != 2 && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: IMHD2D only works for 2D domains\n";
  }

  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  // - initialise FE info
  _UhFEColl = new H1_FECollection( ordU, _dim );
  _PhFEColl = new H1_FECollection( ordP, _dim );
  _ZhFEColl = new H1_FECollection( ordZ, _dim );
  _AhFEColl = new H1_FECollection( ordA, _dim );
  _UhFESpace = new FiniteElementSpace( _mesh, _UhFEColl, _dim );
  _PhFESpace = new FiniteElementSpace( _mesh, _PhFEColl );
  _ZhFESpace = new FiniteElementSpace( _mesh, _ZhFEColl );
  _AhFESpace = new FiniteElementSpace( _mesh, _AhFEColl );

  if (_myRank == 0 ){
    std::cout << "***********************************************************\n";
    std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
    std::cout << "***********************************************************\n";
  }




  // - identify dirichlet nodes
  int numAtt = _mesh->bdr_attributes.Max();
  Array<int> essBdrU( numAtt ); essBdrU = 0;
  Array<int> essBdrP( numAtt ); essBdrP = 0;
  Array<int> essBdrZ( numAtt ); essBdrZ = 0;  // unused for dirichlet
  Array<int> essBdrA( numAtt ); essBdrA = 0;
  if ( _mesh->bdr_attributes.Size() > 0 ) {
    // search among all possible tags
    for ( int i = 0; i < _mesh->bdr_attributes.Max(); ++i ){
      // if that tag is marked in the corresponding array in essTags, then flag it
      if( essTagsU.Find( i ) + 1 )
        essBdrU[i] = 1;
      if( essTagsP.Find( i ) + 1 )
        essBdrP[i] = 1;
      if( essTagsA.Find( i ) + 1 )
        essBdrA[i] = 1;
    }
    _UhFESpace->GetEssentialTrueDofs( essBdrU, _essUhTDOF );
    _PhFESpace->GetEssentialTrueDofs( essBdrP, _essPhTDOF );
    _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );
  }
  // if (_myRank == 0 ){
  //   std::cout << "Dir U  "; _essUhTDOF.Print(mfem::out, _essUhTDOF.Size() ); std::cout<< "\n";
  //   std::cout << "Dir P  "; _essPhTDOF.Print(mfem::out, _essPhTDOF.Size() ); std::cout<< "\n";
  //   std::cout << "Dir A  "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
  //   std::cout << "Tags U ";  essBdrU.Print(  mfem::out,  essBdrU.Size()   ); std::cout<< "\n";
  //   std::cout << "Tags P ";  essBdrP.Print(  mfem::out,  essBdrP.Size()   ); std::cout<< "\n";
  //   std::cout << "Tags A ";  essBdrA.Print(  mfem::out,  essBdrA.Size()   ); std::cout<< "\n";
  //   std::cout << "***********************************************************\n";
  // }



  // - initialise non-lin block operator for single time-step
  Array< FiniteElementSpace* > feSpaces(4);
  feSpaces[0] = _UhFESpace;
  feSpaces[1] = _PhFESpace;
  feSpaces[2] = _ZhFESpace;
  feSpaces[3] = _AhFESpace;
  _IMHD2DOperator.SetSpaces( feSpaces );
  _IMHD2DOperator.AddDomainIntegrator( new IncompressibleMHD2DIntegrator( _dt, _mu, _mu0, _eta ) );

  Array< Array<int> * > tmpEssTags(4);
  essBdrP = 0; // Set all to 0: Dirichlet BC are never imposed on pressure: they are only used to assemble the pressure operators    
  essBdrZ = 0; // Set all to 0: Dirichlet BC are never imposed on laplacian of vector potential    
  tmpEssTags[0] = &essBdrU;
  tmpEssTags[1] = &essBdrP;
  tmpEssTags[2] = &essBdrZ;
  tmpEssTags[3] = &essBdrA;
  Array< Vector * > dummy(4); dummy = NULL;
  _IMHD2DOperator.SetEssentialBC( tmpEssTags, dummy );



  // - initialise guess on solution from provided analytical functions (if given)
  _wFuncCoeff.SetSpace(_UhFESpace); _wFuncCoeff = 0.;
  _yFuncCoeff.SetSpace(_ZhFESpace); _yFuncCoeff = 0.;
  _cFuncCoeff.SetSpace(_AhFESpace); _cFuncCoeff = 0.;
  // -- set them to the provided linearised functions (if given)
  if ( w != NULL ){
    VectorFunctionCoefficient coeff( _dim, w );
    coeff.SetTime( _dt*(_myRank+1) );
    _wFuncCoeff.ProjectCoefficient( coeff );
  }
  if ( y != NULL ){
    FunctionCoefficient coeff( y );
    coeff.SetTime( _dt*(_myRank+1) );
    _yFuncCoeff.ProjectCoefficient( coeff );
  }
  if ( c != NULL ){
    FunctionCoefficient coeff( c );
    coeff.SetTime( _dt*(_myRank+1) );
    _cFuncCoeff.ProjectCoefficient( coeff );
  }


  // -- but re-set the Dirichlet nodes to the provided analytical solution
  VectorFunctionCoefficient uFuncCoeff( _dim, _uFunc );
  FunctionCoefficient       aFuncCoeff( _aFunc );
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  aFuncCoeff.SetTime( _dt*(_myRank+1) );
  GridFunction uBC(_UhFESpace);
  GridFunction aBC(_AhFESpace);
  uBC.ProjectCoefficient(uFuncCoeff);
  aBC.ProjectCoefficient(aFuncCoeff);
  for ( int i = 0; i < _essUhTDOF.Size(); ++i ){
    _wFuncCoeff(_essUhTDOF[i]) = uBC(_essUhTDOF[i]);
  }
  for ( int i = 0; i < _essAhTDOF.Size(); ++i ){
    _cFuncCoeff(_essAhTDOF[i]) = aBC(_essAhTDOF[i]);
  }

}





// // constructor (uses vector of node values to initialise linearised fields)
// IMHD2DSTOperatorAssembler::IMHD2DSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
//                                                       const int refLvl, const int ordU, const int ordP, const int ordZ, const int ordA,
//                                                       const double dt, const double mu, const double eta, const double mu0,
//                                                       void(  *f)(const Vector &, double, Vector &),
//                                                       double(*g)(const Vector &, double ),
//                                                       double(*h)(const Vector &, double ),
//                                                       void(  *n)(const Vector &, double, Vector &),
//                                                       double(*m)(const Vector &, double ),
//                                                       const Vector& w,
//                                                       const Vector& y,
//                                                       const Vector& c,
//                                                       void(  *u)(const Vector &, double, Vector &),
//                                                       double(*p)(const Vector &, double ),
//                                                       double(*z)(const Vector &, double ),
//                                                       double(*a)(const Vector &, double ),
//                                                       int verbose ):
//   IMHD2DSTOperatorAssembler(  comm, meshName, refLvl, ordU, ordP, ordZ, ordA, dt, mu, eta, mu0,
//                               f, g, h, n, m, NULL, NULL, NULL, u, p, z, a, verbose ) {
//     _wFuncCoeff = w;
//     _yFuncCoeff = y;
//     _cFuncCoeff = c;
//   }










//-----------------------------------------------------------------------------
// Assemble operators for single time-steps
//-----------------------------------------------------------------------------
/*
// Operators for target system ------------------------------------------------

// Assemble operator on main diagonal of space-time matrix for u block:
//  Fu = Mu + mu*dt Ku + dt*W1(w)+ dt*W2(w)
// where:
//  Mu    <-> ( u, v )
//  Ku    <-> ( ∇u, ∇v )
//  W1(w) <-> ( (w·∇)u, v )
//  W2(w) <-> ( (u·∇)w, v )
void IMHD2DSTOperatorAssembler::AssembleFu( ){
  if( _FuAssembled ){
    return;
  }

  BilinearForm fuVarf(_UhFESpace);

  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for VectorConvectionIntegrator and VectorUDotGradQIntegrator
  GridFunction wGridFun( _UhFESpace );
  wGridFun = _wFuncCoeff;
  if ( _wFunc == NULL ){
    wCoeff = new VectorGridFunctionCoefficient( &wGridFun );
  }else{
    wCoeff = new VectorFunctionCoefficient( _dim, _wFunc );
    wCoeff->SetTime( _dt*(_myRank+1) );
  }

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


#ifdef MULT_BY_DT
  // ConstantCoefficient muDt( _mu*_dt );
  // ConstantCoefficient one( 1.0 );
  // ConstantCoefficient dt( _dt );
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( 1.0 ));                // Mu
  fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( _mu*_dt ));       // mu*dt*K
  fuVarf.AddDomainIntegrator(new VectorConvectionIntegrator( *wCoeff, _dt )); // dt*W1(w)
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( *gradWCoeff, _dt ));   // dt*W2(w)
#else
  // ConstantCoefficient mu( _mu );
  // ConstantCoefficient dtinv( 1./_dt );
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( 1./_dt ));
  fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( _mu ));
  fuVarf.AddDomainIntegrator(new VectorConvectionIntegrator( *wCoeff, _dt ));
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( *gradWCoeff, 1.0 ));
#endif


  fuVarf.Assemble();
  fuVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  _Fu = fuVarf.SpMat();
  _Fu.SetGraphOwner(true);
  _Fu.SetDataOwner(true);
  fuVarf.LoseMat();

  // still need BC!
  // _FwAssembled = true;

  delete wCoeff;
  delete gradWCoeff;

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial operator Fu assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







// Assemble operator on main diagonal of z block (mass matrix):
//  Mz <-> dt * ( z, x )
void IMHD2DSTOperatorAssembler::AssembleMz( ){
  if( _MzAssembled ){
    return;
  }

  BilinearForm mzVarf(_ZhFESpace);

  // TODO: multiplying by dt shouldn't be necessary either case, but it changes the relative scaling between
  //  this equation and the other in the system. Try and see what happens if you don't rescale (just be consistent
  //  with the definition of Kz)
#ifdef MULT_BY_DT
  mzVarf.AddDomainIntegrator(new MassIntegrator( _dt ));                  // Mz
#else
  mzVarf.AddDomainIntegrator(new MassIntegrator( 1.0 ));
#endif


  mzVarf.Assemble();
  mzVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  _Mz = mzVarf.SpMat();
  _Mz.SetGraphOwner(true);
  _Mz.SetDataOwner(true);
  mzVarf.LoseMat();


  // still need BC!
  // _FaAssembled = true;


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass matrix for z Mz assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}










// Assemble operator on main diagonal of space-time matrix for A block:
//  Fa = Ma + eta *dt Ka + dt* V(w)
// where
//  Ma   <-> ( a, b )
//  Ka   <-> ( ∇a, ∇b )
//  V(w) <-> ( w·∇a, b )
void IMHD2DSTOperatorAssembler::AssembleFa( ){
  if( _FaAssembled ){
    return;
  }

  BilinearForm faVarf(_AhFESpace);

  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for ConvectionIntegrator
  GridFunction wGridFun( _UhFESpace );
  wGridFun = _wFuncCoeff;
  if ( _wFunc == NULL ){
    wCoeff = new VectorGridFunctionCoefficient( &wGridFun );
  }else{
    wCoeff = new VectorFunctionCoefficient( _dim, _wFunc );
    wCoeff->SetTime( _dt*(_myRank+1) );
  }


#ifdef MULT_BY_DT
  // ConstantCoefficient one( 1.0 );
  faVarf.AddDomainIntegrator(new MassIntegrator( 1.0 ));                 // Ma
  faVarf.AddDomainIntegrator(new DiffusionIntegrator( _dt * _eta ));     // eta *dt K
  faVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, _dt ));  // dt*V(w)

#else
  // ConstantCoefficient dtinv( 1./_dt );
  faVarf.AddDomainIntegrator(new MassIntegrator( 1./_dt ));
  faVarf.AddDomainIntegrator(new DiffusionIntegrator( _eta ));
  faVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, 1.0 ));
#endif


  faVarf.Assemble();
  faVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  _Fa = faVarf.SpMat();
  _Fa.SetGraphOwner(true);
  _Fa.SetDataOwner(true);
  faVarf.LoseMat();

  delete wCoeff;

  // still need BC!
  // _FaAssembled = true;


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial operator Fa assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







// Assemble -divergence operator (coupling u and p):
//  B <-> -dt * ( ∇·u, q )
void StokesSTOperatorAssembler::AssembleB( ){

  if( _BAssembled ){
    return;
  }

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

  // - once the matrix is generated, we can get rid of the operator
  _B = bVarf.SpMat();
  _B.SetGraphOwner(true);
  _B.SetDataOwner(true);
  bVarf.LoseMat();


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Divergence operator (negative) B assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}





// Assemble first part of linearised Lorentz coupling operator (coupling u and z):
//  Z1(c) <-> dt/mu0 * ( z ∇c, v )
void StokesSTOperatorAssembler::AssembleZ1( ){

  if( _Z1Assembled ){
    return;
  }

  MixedBilinearForm z1Varf( _ZhFESpace, _UhFESpace );

  GridFunction cGridFun( _AhFESpace );
  if ( _cFunc == NULL ){
    cGridFun = _cFuncCoeff;
  }else{
    FunctionCoefficient cCoeff( _cFunc );
    cCoeff.SetTime( _dt*(_myRank+1) );
    cGridFunc.ProjectCoefficient( cCoeff );
  }

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

  z1Varf.AddDomainIntegrator(new MixedVectorProductIntegrator( *gradCCoeff ) );  // ( z ∇c, v )

  z1Varf.Assemble();
  z1Varf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Z1 = z1Varf.SpMat();
  _Z1.SetGraphOwner(true);
  _Z1.SetDataOwner(true);
  z1Varf.LoseMat();

  _Z1 *= 1./_mu0;

#ifdef MULT_BY_DT
  _Z1 *= _dt;  
#endif


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"First part of linearised Lorentz coupling operator Z1(A) assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}




// Assemble second part of linearised Lorentz coupling operator (coupling u and A):
//  Z2(y) <-> dt/mu0 * ( y ∇A, v )
void StokesSTOperatorAssembler::AssembleZ2( ){

  if( _Z2Assembled ){
    return;
  }

  MixedBilinearForm z2Varf( _AhFESpace, _UhFESpace );

  Coefficient* yCoeff = NULL;   // need to define them here otherwise they go out of scope
  GridFunction yGridFun( _ZhFESpace );
  yGridFun = _yFuncCoeff;
  if ( _yFunc == NULL ){
    yCoeff = new GridFunctionCoefficient( &yGridFun );
  }else{
    yCoeff = new FunctionCoefficient( _yFunc );
    yCoeff->SetTime( _dt*(_myRank+1) );
  }

  z2Varf.AddDomainIntegrator(new MixedVectorGradientIntegrator( *yCoeff ) );  // ( y ∇A, v )



  z2Varf.Assemble();
  z2Varf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Z2 = z2Varf.SpMat();
  _Z2.SetGraphOwner(true);
  _Z2.SetDataOwner(true);
  z2Varf.LoseMat();

  _Z2 *= 1./_mu0;

#ifdef MULT_BY_DT
  _Z2 *= _dt;  
#endif

  delete yCoeff;

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Second part of linearised Lorentz coupling operator Z2(z) assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}








// Assemble linearised convection operator (coupling A and u):
//  Y(c) <-> dt * ( u·∇c, b )
void StokesSTOperatorAssembler::AssembleY( ){

  if( _YAssembled ){
    return;
  }

  MixedBilinearForm yVarf( _UhFESpace, _AhFESpace );

  Coefficient* cCoeff = NULL;   // need to define them here otherwise they go out of scope
  GridFunction cGridFun( _AhFESpace );
  cGridFun = _cFuncCoeff;
  if ( _cFunc == NULL ){
    cCoeff = new GridFunctionCoefficient( &cGridFun );
  }else{
    cCoeff = new FunctionCoefficient( _cFunc );
    cCoeff->SetTime( _dt*(_myRank+1) );
  }

  // assemble ∇c from c
  VectorArrayCoefficient* gradCCoeff = new VectorArrayCoefficient( _dim );
  for ( int i = 0; i < _dim; ++i ){
    GridFunction* temp = new GridFunction( _AhFESpace );
    cGridFun->GetDerivative( 0, i, *temp );   // possibly pass 1 here: seems like it counts components weirdly?
    gradCCoeff->Set( i, temp );               //gradCCoeff should take ownership here
    delete temp;
  }

  yVarf.AddDomainIntegrator(new MixedDotProductIntegrator( *gradCCoeff ) );

  // TODO Similar considerations as for Z1 apply here: integration by parts would give
  //  the boundary term int(bcu·n), and the domain term -int(c∇b·u)


  yVarf.Assemble();
  yVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Y = yVarf.SpMat();
  _Y.SetGraphOwner(true);
  _Y.SetDataOwner(true);
  yVarf.LoseMat();

#ifdef MULT_BY_DT
  _Y *= _dt;  
#endif

  delete gradCCoeff;

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Linearised convection operator Y assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}








// Assemble mixed Laplacian operator (coupling A and z):
//  K <-> dt * ( ∇A, ∇x )
void StokesSTOperatorAssembler::AssembleK( ){

  if( _KAssembled ){
    return;
  }

  MixedBilinearForm kVarf( _AhFESpace, _ZhFESpace );

  // TODO: multiplying by dt shouldn't be necessary either case, but it changes the relative scaling between
  //  this equation and the other in the system. Try and see what happens if you don't rescale
#ifdef MULT_BY_DT
  ConstantCoefficient dt( _dt );
  kVarf.AddDomainIntegrator(new MixedGradGradIntegrator( dt ) );
#else
  ConstantCoefficient one( 1.0 );
  kVarf.AddDomainIntegrator(new MixedGradGradIntegrator( one ) );
#endif


  kVarf.Assemble();
  kVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _K = kVarf.SpMat();
  _K.SetGraphOwner(true);
  _K.SetDataOwner(true);
  kVarf.LoseMat();


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mixed Laplacian operator K assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}

*/




// Assemble operator on subdiagonal of space-time matrix for A block (negative mass matrix):
//  Ma <-> - (a,b)
void IMHD2DSTOperatorAssembler::AssembleMa( ){
  if( _MaAssembled ){
    return;
  }

  BilinearForm maVarf(_AhFESpace);
// #ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  maVarf.AddDomainIntegrator(new MassIntegrator( mone ));
// #else
//   ConstantCoefficient mdtinv( -1./_dt );
//   maVarf.AddDomainIntegrator(new MassIntegrator( mdtinv ));
// #endif
  maVarf.Assemble();
  maVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  // _Ma = maVarf.SpMat();
  _Ma.MakeRef( maVarf.SpMat() );
  _Ma.SetGraphOwner(true);
  _Ma.SetDataOwner(true);
  maVarf.LoseMat();

  // NOT YET! DO IT AS YOU ASSEMBLE THE SYSTEM
  // // - include dirichlet BC
  // Array<int> colsA(_Ma.Height());
  // colsA = 0;
  // for (int i = 0; i < _essAhTDOF.Size(); ++i){
  //   colsA[_essAhTDOF[i]] = 1;
  // }
  // _Ma.EliminateCols( colsA );
  // for (int i = 0; i < _essAhTDOF.Size(); ++i){
  //   _Ma.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
  // }

  // _MaAssembled = true;



  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass matrix for A Ma assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}

// Assemble operator on subdiagonal of space-time matrix for u block (negative mass matrix):
//  Mu <-> - (u,v)
void IMHD2DSTOperatorAssembler::AssembleMu( ){
  if( _MuAssembled ){
    return;
  }

  BilinearForm muVarf(_UhFESpace);
// #ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  muVarf.AddDomainIntegrator(new VectorMassIntegrator( mone ));
// #else
//   ConstantCoefficient mdtinv( -1./_dt );
//   muVarf.AddDomainIntegrator(new VectorMassIntegrator( mdtinv ));
// #endif
  muVarf.Assemble();
  muVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  // _Mu = muVarf.SpMat();
  _Mu.MakeRef( muVarf.SpMat() );
  _Mu.SetGraphOwner(true);
  _Mu.SetDataOwner(true);
  muVarf.LoseMat();


  // - include dirichlet BC
  // NOT YET! DO IT AS YOU ASSEMBLE THE SYSTEM
  // Array<int> colsU(_Mu.Height());
  // colsU = 0;
  // for (int i = 0; i < _essUhTDOF.Size(); ++i){
  //   colsU[_essUhTDOF[i]] = 1;
  // }
  // _Mu.EliminateCols( colsU );
  // for (int i = 0; i < _essUhTDOF.Size(); ++i){
  //   _Mu.EliminateRow( _essUhTDOF[i], mfem::Matrix::DIAG_ZERO );
  // }

  // _MuAssembled = true;




  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass-matrix for u (negative) Mu assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}





// Operators for preconditioners ----------------------------------------------

// Assemble pressure mass matrix
//  Mp <->  (p,q)
void IMHD2DSTOperatorAssembler::AssembleMp( ){
  if( _MpAssembled ){
    return;
  }

  BilinearForm mVarf( _PhFESpace );
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new MassIntegrator( one ));
  mVarf.Assemble();
  mVarf.Finalize();

  // TODO: let's not impose dirichlet here. We do it inside the Schur complement operator
  // - impose dirichlet BC on outflow
  // mVarf.FormSystemMatrix( _essPhTDOF, _Mp );
  // - once the matrix is generated, we can get rid of the operator
  _Mp.MakeRef( mVarf.SpMat() );
  _Mp.SetGraphOwner(true);
  _Mp.SetDataOwner(true);
  mVarf.LoseMat();

  _MpAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Mp.dat";
    myfile.open( myfilename );
    _Mp.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass matrix for p Mp assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}





// Assemble "laplacian" operator for pressure block:
//  Ap <->  ( ∇p, ∇q )
// - This should be assembled as if it had homogeneous dirichlet BC on the outflow boundary
//    and homogeneous Neumann BC on the inflow boundary (dirichlet for u)
void IMHD2DSTOperatorAssembler::AssembleAp( ){

  if( _ApAssembled ){
    return;
  }

  BilinearForm aVarf( _PhFESpace );
  ConstantCoefficient one( 1.0 );     // diffusion
  // ConstantCoefficient beta( 1e6 );    // penalty term for weakly imposing dirichlet BC

  if ( _ordP == 0 ){  // DG
    double sigma = -1.0;
    double kappa =  1.0;
    aVarf.AddDomainIntegrator(      new DiffusionIntegrator( one ));                 // classical grad-grad inside each element
    aVarf.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));   // contribution to jump across elements
    // aVarf->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));   // TODO: includes boundary contributions (otherwise you'd be imposing neumann?)
  }else{
    aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
    // Impose homogeneous dirichlet BC weakly via penalty method  -> Andy says it's not a good idea (and indeed results are baaad)
    // if( _essQhTDOF.Size()>0 ){
    //   aVarf->AddBoundaryIntegrator(new BoundaryMassIntegrator( beta ), _essQhTDOF );
    // }
  }
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


  _ApAssembled = true;

  if( _essPhTDOF.Size() == 0 ){
    if( _myRank == 0 ){
      std::cout<<"Warning: the pressure 'laplacian' has non-trivial kernel (constant functions)."<<std::endl
               <<"         Make sure to flag that in the petsc options prescribing:"<<std::endl
               <<"         -for iterative solver: -PSolverLaplacian_ksp_constant_null_space TRUE"<<std::endl
               <<"         -for direct solver: -PSolverLaplacian_pc_factor_shift_type NONZERO"<<std::endl
               <<"                         and -PSolverLaplacian_pc_factor_shift_amount 1e-10"<<std::endl
               <<"                         (this will hopefully save us from 0 pivots in the singular mat)"<<std::endl;
    }
  }



  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Ap.dat";
    myfile.open( myfilename );
    _Ap.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Pressure stiffness matrix Ap assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}




// Assemble spatial part of pressure (convection) diffusion operator
//  Wp <-> mu*( ∇p, ∇q ) + ( w·∇p, q )
// According to Elman/Silvester/Wathen:
// - If convection is included, this should be assembled as if it had Robin BC: dp/dn + (w*n)*p = 0 (w is convective velocity)
// - If convection is not included, this should be assembled as if it had Neumann BC: dp/dn = 0
// However, it seems like it's best to just leave dirichlet BC in outflow, just like how Ap is assembled :/
// NB: This bit of code is hence intended to be used *only* for the double-glazing problem, where the velocity
//     field has Dirichlet BC everywhere!
void IMHD2DSTOperatorAssembler::AssembleWp( ){

  if( _WpAssembled ){
    return;
  }

  // if ( _myRank == 0 ){
  //   std::cout<<"Warning: The boundary tags are hard coded! Make sure that the domain is a rectangle, and that"<<std::endl
  //            <<"         the tags for its 4 sides are, respectively: north=1 east=2 south=3 west=4"           <<std::endl;
  // }

  if ( _myRank == 0 ){
    std::cout<<"Warning: The assembly of the spatial part of the PCD considers Neumann BC on pressure,"      <<std::endl
             <<"         This goes against what Elman/Silvester/Wathen says (should be Robin everywhere)."   <<std::endl 
             <<"         However, this is still in line with E/S/W if:"                                      <<std::endl
             <<"          - Spatial part of Fp and Ap are forcibly imposed equal (which bypasses this func)" <<std::endl
             <<"      and - The advection field is always tangential to the boundary (enclosed flow, w*n=0)."<<std::endl
             <<"         Regardless, choosing Neumann seems to provide better results pretty much always"    <<std::endl;
  }


  BilinearForm wVarf( _PhFESpace );
  ConstantCoefficient mu( _mu );
  wVarf.AddDomainIntegrator(new DiffusionIntegrator( mu ));
  if ( _ordP == 0 ){  // DG
    double sigma = -1.0;
    double kappa =  1.0;
    wVarf.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
    // wVarf->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));  // to weakly impose Dirichlet BC - don't bother for now
  }

  // include convection
  VectorGridFunctionCoefficient wCoeff( &_wFuncCoeff );
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( wCoeff, 1.0 ));  // if used for NS, make sure both _mu*_Pe=1.0!!

  // // include Robin term
  // // - manually assemble the outward-facing normals
  // Vector nN(2); nN(0)= 0.; nN(1)= 1.; // northern side: (  0,  1 )^T
  // Vector nE(2); nE(0)= 1.; nE(1)= 0.; // eastern  side: (  1,  0 )^T
  // Vector nS(2); nS(0)= 0.; nS(1)=-1.; // southern side: (  0, -1 )^T
  // Vector nW(2); nW(0)=-1.; nW(1)= 0.; // western  side: ( -1,  0 )^T
  // VectorConstantCoefficient nNcoeff( nN );
  // VectorConstantCoefficient nEcoeff( nE );
  // VectorConstantCoefficient nScoeff( nS );
  // VectorConstantCoefficient nWcoeff( nW );
  // // - manually assemble the inner products w * n
  // InnerProductCoefficient wnN(wCoeff,nNcoeff);
  // InnerProductCoefficient wnE(wCoeff,nEcoeff);
  // InnerProductCoefficient wnS(wCoeff,nScoeff);
  // InnerProductCoefficient wnW(wCoeff,nWcoeff);
  // // - identify tags corresponding to each side: N=0 E=1 S=2 W=3 -> notice we are shifting by 1 wrt the tag definition in the mesh file
  // Array<int> Ntag(4), Etag(4), Stag(4), Wtag(4);
  // Ntag=0; Etag=0; Stag=0; Wtag=0;
  // Ntag[0]=1; Etag[1]=1; Stag[2]=1; Wtag[3]=1;
  // // - finally include Robin contribution
  // wVarf.AddBoundaryIntegrator(new MassIntegrator( wnN ), Ntag );
  // wVarf.AddBoundaryIntegrator(new MassIntegrator( wnE ), Etag );
  // wVarf.AddBoundaryIntegrator(new MassIntegrator( wnS ), Stag );
  // wVarf.AddBoundaryIntegrator(new MassIntegrator( wnW ), Wtag );

  // TODO: Robin for DG?

  
  wVarf.Assemble();
  wVarf.Finalize();
  
  // _Wp = wVarf.SpMat();
  _Wp.MakeRef( wVarf.SpMat() );
  _Wp.SetGraphOwner(true);
  _Wp.SetDataOwner(true);
  wVarf.LoseMat();

  _WpAssembled = true;


  if ( _verbose>50 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Wp_" + std::to_string(_myRank) + ".dat";
    myfile.open( myfilename );
    _Wp.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial part of PCD operator Wp assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







// Assemble vector potential mass-matrix (its inverse is used in the Schur complement)
//  this time however we don't kill contribution from dirichlet nodes, but set them to 1
//  MaNoZero <-> ( A, B )
void IMHD2DSTOperatorAssembler::AssembleMaNoZero( ){
  if( _MaNoZeroAssembled ){
    return;
  }

  BilinearForm mVarf( _AhFESpace );
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new MassIntegrator( one ));
  mVarf.Assemble();
  mVarf.Finalize();

  // mVarf.FormSystemMatrix( _essAhTDOF, _MaNoZero );
  // _MaNoZero = mVarf.SpMat();

  // - once the matrix is generated, we can get rid of the operator
  _MaNoZero.MakeRef( mVarf.SpMat() );
  _MaNoZero.SetGraphOwner(true);
  _MaNoZero.SetDataOwner(true);
  mVarf.LoseMat();

  // - impose Dirichlet BC
  mfem::Array<int> colsA(_MaNoZero.Height());
  colsA = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  _MaNoZero.EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _MaNoZero.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE );
  }



  _MaNoZeroAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_MaNoZero.dat";
    myfile.open( myfilename );
    _MaNoZero.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass matrix for A Ma (with Dirichlet nodes set to identity) assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}



// Assemble vector potential lumped mass-matrix (its inverse is used in the Schur complement)
//  this time we lump it and set dirichlet nodes to 1
//  MaNoZeroLumped <-> Lump( ( A, B ) )
void IMHD2DSTOperatorAssembler::AssembleMaNoZeroLumped( ){
  if( _MaNoZeroLumpedAssembled ){
    return;
  }

  BilinearForm maLumpedVarf(_AhFESpace);
  ConstantCoefficient one( 1.0 );
  maLumpedVarf.AddDomainIntegrator(new LumpedIntegrator( new MassIntegrator( one ) ));
  maLumpedVarf.Assemble();
  maLumpedVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _MaNoZeroLumped.MakeRef( maLumpedVarf.SpMat() );
  _MaNoZeroLumped.SetGraphOwner(true);
  _MaNoZeroLumped.SetDataOwner(true);
  maLumpedVarf.LoseMat();


  // - impose Dirichlet BC
  mfem::Array<int> colsA(_MaNoZeroLumped.Height());
  colsA = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  _MaNoZeroLumped.EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _MaNoZeroLumped.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE );
  }



  _MaNoZeroLumpedAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_MaNoZeroLump.dat";
    myfile.open( myfilename );
    _MaNoZeroLumped.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass matrix for A Ma (lumped, and with Dirichlet nodes set to identity) assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}





// Assemble spatial part of original space-time operator for A (used in Schur complement)
//  Wa(w) <-> eta*( ∇A, ∇C ) + ( w·∇A, C )
void IMHD2DSTOperatorAssembler::AssembleWa(){
  
  if( _WaAssembled ){
    return;
  }


  BilinearForm wVarf( _AhFESpace );
  ConstantCoefficient eta( _eta );
  wVarf.AddDomainIntegrator(new DiffusionIntegrator( eta ));

  // include convection
  // - NB: multiplication by dt is handled inside the Schur complement approximation
  VectorGridFunctionCoefficient wCoeff( &_wFuncCoeff );
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( wCoeff, 1.0 ));

  
  wVarf.Assemble();
  wVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  // _Wa = wVarf.SpMat();
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




  _WaAssembled = true;


  if ( _verbose>50 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Wa_" + std::to_string(_myRank) + ".dat";
    myfile.open( myfilename );
    _Wa.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial part of space-time matrix for A Wa assembled\n";
    }
    MPI_Barrier(_comm);
  }  



}




// Assemble a small component of Fa*Mai*Fa:
//  dtuWa(w) <-> ( dtw·∇A, C )
void IMHD2DSTOperatorAssembler::AssembledtuWa(){
  if( _dtuWaAssembled ){
    return;
  }

  // Assemble bilinear form
  // - define function containing dtw
  //    Use a central difference scheme for this: need to fetch w from previous and next procs
  const int NU = _wFuncCoeff.Size();
  GridFunction wCoeffPost(_UhFESpace), wCoeffPrev(_UhFESpace);
  if( _myRank < _numProcs-1 ){
    MPI_Send( _wFuncCoeff.GetData(), NU, MPI_DOUBLE, _myRank+1, 2*(_myRank),     _comm );
  }
  if( _myRank > 0 ){
    MPI_Recv( wCoeffPrev.GetData(),  NU, MPI_DOUBLE, _myRank-1, 2*(_myRank-1),   _comm, MPI_STATUS_IGNORE );
  }
  if( _myRank > 0 ){
    MPI_Send( _wFuncCoeff.GetData(), NU, MPI_DOUBLE, _myRank-1, 2*(_myRank)+1,   _comm );
  }
  if( _myRank < _numProcs-1 ){
    MPI_Recv( wCoeffPost.GetData(),  NU, MPI_DOUBLE, _myRank+1, 2*(_myRank+1)+1, _comm, MPI_STATUS_IGNORE );
  }
  // -- we need to adjust for proc 0 (missing prev) and proc _numProcs-1 (missing post)
  // --- use initial conditions for prev
  if ( _myRank == 0 ){
    VectorFunctionCoefficient uFuncCoeff( _dim, _uFunc );
    uFuncCoeff.SetTime( 0 );
    wCoeffPrev.ProjectCoefficient(uFuncCoeff);
  }
  // --- use backward difference for _numProcs-1
  if ( _myRank == _numProcs-1 ){
    // I want to reuse the same formula below, so I just need to be a bit original in how I define wCoeffPost
    wCoeffPost  = _wFuncCoeff;
    wCoeffPost *= 2.;
    wCoeffPost -= wCoeffPrev;
  }

  // -- finally use finite difference in time to get derivative
  GridFunction dtwFuncCoeff(_UhFESpace);
  for ( int i = 0; i < NU; ++i ){
    dtwFuncCoeff(i) = 0.5*( wCoeffPost(i) - wCoeffPrev(i) ); // should be divided by dt, but leave it unscaled for now
  }
  VectorGridFunctionCoefficient dtwCoeff( &dtwFuncCoeff );

  // - include convection integrator
  BilinearForm wVarf( _AhFESpace );
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( dtwCoeff, 1.0 ));

  wVarf.Assemble();
  wVarf.Finalize();
  
  // - once the matrix is generated, we can get rid of the operator
  _dtuWa.MakeRef( wVarf.SpMat() );
  _dtuWa.SetGraphOwner(true);
  _dtuWa.SetDataOwner(true);
  wVarf.LoseMat();
  
  // - impose Dirichlet BC
  mfem::Array<int> colsA(_dtuWa.Height());
  colsA = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  _dtuWa.EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _dtuWa.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }


  _dtuWaAssembled = true;


  if ( _verbose>50 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_dtuWa_" + std::to_string(_myRank) + ".dat";
    myfile.open( myfilename );
    _dtuWa.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Component of Fa*(Ma)^-1*Fa, dtuWa assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}





// Assemble stifness matrix for A (used in Schur complement)
//  Aa <-> ( ∇A, ∇C )
void IMHD2DSTOperatorAssembler::AssembleAa(){
  
  if( _AaAssembled ){
    return;
  }
  
  BilinearForm aVarf( _AhFESpace );
  ConstantCoefficient one( 1.0 );
  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
  aVarf.Assemble();
  aVarf.Finalize();
  
  // _Aa = aVarf.SpMat();
  _Aa.MakeRef( aVarf.SpMat() );
  _Aa.SetGraphOwner(true);
  _Aa.SetDataOwner(true);
  aVarf.LoseMat();

  _AaAssembled = true;

  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Aa.dat";
    myfile.open( myfilename );
    _Aa.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Laplacian for A Aa assembled\n";
    }
    MPI_Barrier(_comm);
  }  



}







// Assemble operators for space-time magnetic wave eq for A
//  The actual operators depend on the discType:
//  - Implicit Leapfrog (discType=0)
//   -- Cp <->    ( ( A, C ) + dt^2*B/(4*mu_0) ( ∇A, ∇C ) )
//   -- C0 <-> -2*( ( A, C ) - dt^2*B/(4*mu_0) ( ∇A, ∇C ) )
//   -- Cm <->    ( ( A, C ) + dt^2*B/(4*mu_0) ( ∇A, ∇C ) )
//  - Explicit Leapfrog (discType=1)
//   -- Cp <->    ( A, C )
//   -- C0 <-> -2*( A, C ) + dt^2*B/mu_0 ( ∇A, ∇C )
//   -- Cm <->    ( A, C )
//  where B = ||B_0||_2, with B_0 being a space(-time) average of the magnetic field
void IMHD2DSTOperatorAssembler::AssembleCs(){

  // I'm gonna need a mass matrix and a stiffness matrix regardless
  AssembleMaNoZero();
  AssembleAa();

  // - for dirichlet BC (used later)
  mfem::Array<int> colsA(_Aa.Height());
  colsA = 0;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }


  // Now I need to compute ||B_0||_2, the L2 norm of the space-time average of the magnetic
  //  field B = ∇x(kA)
  // - loop over all elements and integrate to recover the average magnetic field
  // -- NB: this will contain \int_{\Omega}[Ax,Ay] dx, while actually B = [Ay,-Ax], but nothing much changes
  // TODO: here I'm hacking the ComputeGradError method of GridFunction: I hope no weird reordering of the nodes
  //       of FESpace occurs in the function, otherwise I'm doomed
  Vector B0(_dim);
  double area = 0.;
  B0 = 0.;
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
      _cFuncCoeff.GetGradient(*Tr,grad);
      grad *= ip.weight * Tr->Weight();
      B0 += grad;
    }
    // We should consider linear elements: if so, the Jacobian is constant inside each element
    //  The area of the mapped element is then given by the area of the reference element (0.5)
    //   times the determinant of the Jacobian, or:
    area += 0.5 * (Tr->Weight());
  }
  // - average over time as well
  MPI_Allreduce( MPI_IN_PLACE, B0.GetData(), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  // - finally, compute its norm and rescale by domain size to get the average
  // -- B0 *should have been* rescaled by the space-time domain size, to get the actual average. When computing its ||B0||^2_L2st norm, 
  //     then, I should integrate over the space-time domain the square of the averaged value. Combining this, I'm missing a 1/( |\Omega|*|T|) factor
  double B0norm = sqrt( ( B0(0)*B0(0) + B0(1)*B0(1) ) / ( area * _dt*_numProcs) ); 
  if ( _myRank == 0 ){
    std::cout<<" Area of domain is                 "  << area                   << std::endl
             <<" Average magnetic field is         (" << B0(1)<<","<<-B0(0)<<")"<< std::endl
             <<" Average norm of magnetic field is "  << B0norm                 << std::endl;
  }




  // All the ingredients are ready, now I just need to properly combine them
  // - IMPLICIT LEAPFROG
  switch (_ASTSolveType){
    case 0:{
      _Cp  = _MaNoZero;
      _C0  = _MaNoZero;

      _Cp.Add(  _dt*_dt*B0norm/(4*_mu0), _Aa );
      _C0.Add( -_dt*_dt*B0norm/(4*_mu0), _Aa );
      _C0 *= -2.;

      _Cm  = _Cp;

      // - impose Dirichlet BC
      _Cp.EliminateCols( colsA );
      _C0.EliminateCols( colsA );
      _Cm.EliminateCols( colsA );
      for (int i = 0; i < _essAhTDOF.Size(); ++i){
        _Cp.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE  ); // this one is one the main diag and must be inverted -> set to 1
        _C0.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO ); // these ones are on the subdiag and only impact rhs  -> set to 0
        _Cm.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
      }

      break;
    }

    // - EXPLICIT LEAPFROG
    case 1:{
      _Cp  = _MaNoZero;
      _Cm  = _MaNoZero;

      _C0  = _MaNoZero;
      _C0 *= -2.;
      _C0.Add( _dt*_dt*B0norm/( 2*_mu0 ), _Aa );


      // - impose Dirichlet BC
      _Cp.EliminateCols( colsA );
      _C0.EliminateCols( colsA );
      _Cm.EliminateCols( colsA );
      for (int i = 0; i < _essAhTDOF.Size(); ++i){
        _Cp.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE  ); // this one is one the main diag and must be inverted -> set to 1
        _C0.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO ); // these ones are on the subdiag and only impact rhs  -> set to 0
        _Cm.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
      }
      break;
    }


    default:
      std::cerr<<"ERROR: Discretisation type for wave equation "<<_ASTSolveType<<" not recognised."<<std::endl;
  }

}

















//-----------------------------------------------------------------------------
// Assemble space-time operators
//-----------------------------------------------------------------------------
/*
// Handy function to assemble a monolithic space-time block-bidiagonal matrix starting from its block(s)
void IMHD2DSTOperatorAssembler::AssembleSTBlockBiDiagonal( const SparseMatrix& F, const SparseMatrix& M, HYPRE_IJMatrix& FF, (HypreParMatrix*)& FFF,
                                                           const std::string& STMatName, const bool blocksAssembled, bool& STMatAssembled ){ 

  if( STMatAssembled )
    return;

  if( !blocksAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble its blocks before assembling "<< STMatName <<std::endl;
    return;
  }



  // Create space-time block **************************************************
  // Initialize HYPRE matrix
  // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
  //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
  // - get info on matrix structure
  const int blockSizeFF = F.NumRows();
 
  Array<int> nnzPerRowD( blockSizeFF );   // num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<int> nnzPerRowO( blockSizeFF );   // ..and in off-diagonal block
  const int  *offIdxsD = F.GetI();        // has size blockSizeFF+1, contains offsets for data in J for each row
  const int  *offIdxsO = M.GetI();
  for ( int i = 0; i < blockSizeFF; ++i ){
    nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
    if ( _myRank > 0 ){
      nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
    }else{
      nnzPerRowO[i] = 0;  // first block only has elements on block-diag
    }
  }


  // - initialise matrix
  HYPRE_IJMatrixCreate( _comm, blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1,
                               blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1, &FF );
  HYPRE_IJMatrixSetObjectType( FF, HYPRE_PARCSR );
  HYPRE_IJMatrixSetDiagOffdSizes( FF, nnzPerRowD.GetData(), nnzPerRowO.GetData() );    // this gives issues :/
  HYPRE_IJMatrixInitialize( FF );


  // - fill it with matrices assembled above
  // -- diagonal block
  Array<int> rowsGlbIdxD( blockSizeFF );
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxD[i] = i + blockSizeFF*_myRank;
  }
  Array<int> colsGlbIdxD( F.NumNonZeroElems() );
  for ( int i=0; i<F.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = F.GetJ()[i] + blockSizeFF*_myRank;
  }
  HYPRE_IJMatrixSetValues( FF, blockSizeFF, nnzPerRowD.GetData(),
                           rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), F.GetData() );     // setvalues *copies* the data

  // -- off-diagonal block
  Array<int> rowsGlbIdxO( blockSizeFF );      // TODO: just use rowsGlbIdx once for both matrices?
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxO[i] = i + blockSizeFF*_myRank;
  }
  if ( _myRank > 0 ){
    Array<int> colsGlbIdxO( M.NumNonZeroElems() );
    for ( int i=0; i<M.NumNonZeroElems(); i++ ) {
      colsGlbIdxO[i] = M.GetJ()[i] + blockSizeFF*(_myRank-1);
    }
    HYPRE_IJMatrixSetValues( FF, blockSizeFF, nnzPerRowO.GetData(),
                             rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), M.GetData() );
  }


  // - assemble
  HYPRE_IJMatrixAssemble( FF );
  STMatAssembled = true;

  // - convert to a MFEM operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( FF, (void **) &FFref);
  FFF = new HypreParMatrix( FFref, false ); //"false" doesn't take ownership of data


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time operator " << STMatName <<" assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}




// Handy function to assemble a monolithic space-time block-diagonal matrix starting from its block(s)
void IMHD2DSTOperatorAssembler::AssembleSTBlockDiagonal( const SparseMatrix& D, HYPRE_IJMatrix& DD,
                                                         const std::string& STMatName, const bool blockAssembled, bool& STMatAssembled ){ 
  if( STMatAssembled )
    return;

  if( !blockAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble block before assembling "<< STMatName <<std::endl;
    return;
  }


  // recover info on matrix structure
  const int numRowsPerBlock = D.NumRows();
  const int numColsPerBlock = D.NumCols();

  Array<int> nnzPerRow( numRowsPerBlock );    // num of non-zero els per row in main (diagonal) block (for preallocation)
  const int  *offIdxs = D.GetI();             // has size numrows, contains offsets for data in J for each row
  for ( int i = 0; i < numRowsPerBlock; ++i ){
    nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
  }


  // initialise matrix
  HYPRE_IJMatrixCreate( _comm, numRowsPerBlock*_myRank, numRowsPerBlock*(_myRank+1)-1,
                               numColsPerBlock*_myRank, numColsPerBlock*(_myRank+1)-1, &DD );
  HYPRE_IJMatrixSetObjectType( DD, HYPRE_PARCSR );
  HYPRE_IJMatrixSetRowSizes( DD, nnzPerRow.GetData() );
  HYPRE_IJMatrixInitialize( DD );


  // fill it with matrices assembled above
  Array<int> rowsGlbIdxBB( numRowsPerBlockBB );
  for ( int i = 0; i < numRowsPerBlockBB; ++i ){
    rowsGlbIdxBB[i] = i + numRowsPerBlockBB*_myRank;
  }
  Array<int> colsGlbIdx( D.NumNonZeroElems() );
  for ( int i=0; i<D.NumNonZeroElems(); i++ ) {
    colsGlbIdx[i] = D.GetJ()[i] + numColsPerBlock*_myRank;
  }
  HYPRE_IJMatrixSetValues( DD, numRowsPerBlock, nnzPerRow.GetData(),
                           rowsGlbIdx.GetData(), colsGlbIdx.GetData(), D.GetData() );


  // assemble
  HYPRE_IJMatrixAssemble( DD );

  STMatAssembled = true;


  if(_verbose>1 && _myRank == 0 ){
    std::cout<<" - Space-time operator "<<STMatName<<" assembled\n";
  }  


}

*/


// Code duplication at its worst:
// - assemble a bunch of block-bidiagonal space-time matrices (representing time-stepping operators)
inline void IMHD2DSTOperatorAssembler::AssembleFFu( ){ 
  AssembleSTBlockBiDiagonal( _Fu, _Mu, _FFu, "FFu", _FuAssembled && _MuAssembled, _FFuAssembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleFFa( ){ 
  AssembleSTBlockBiDiagonal( _Fa, _Ma, _FFa, "FFa", _FaAssembled && _MaAssembled, _FFaAssembled );
}
// - assemble a bunch of block-diagonal space-time matrices (representing couplings for all space-time operators)
inline void IMHD2DSTOperatorAssembler::AssembleMMz( ){ 
  AssembleSTBlockDiagonal( _Mz, _MMz, "MMz", _MzAssembled, _MMzAssembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleBB( ){ 
  AssembleSTBlockDiagonal(  _B,  _BB,  "BB",  _BAssembled,  _BBAssembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleZZ1( ){ 
  AssembleSTBlockDiagonal( _Z1, _ZZ1, "ZZ1", _Z1Assembled, _ZZ1Assembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleZZ2( ){ 
  AssembleSTBlockDiagonal( _Z2, _ZZ2, "ZZ2", _Z2Assembled, _ZZ2Assembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleKK( ){ 
  AssembleSTBlockDiagonal(  _K,  _KK,  "KK",  _KAssembled,  _KKAssembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleYY( ){ 
  AssembleSTBlockDiagonal(  _Y,  _YY,  "YY",  _YAssembled,  _YYAssembled );
}




// Handy function to assemble a monolithic space-time block-diagonal matrix starting from its block(s)
void IMHD2DSTOperatorAssembler::AssembleSTBlockDiagonal( const SparseMatrix& D, ParBlockLowTriOperator& DD,
                                                         const std::string& STMatName, const bool blockAssembled, bool& STMatAssembled ){ 
  if( STMatAssembled )
    return;

  if( !blockAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble block before assembling "<< STMatName <<std::endl;
    return;
  }

  DD.SetBlockDiag( &D, 0, false );  // false doesn't own

  STMatAssembled = true;

  if(_verbose>1 && _myRank == 0 ){
    std::cout<<" - Space-time operator "<<STMatName<<" assembled\n";
  }  

}





// Handy function to assemble a monolithic space-time block-bidiagonal matrix starting from its block(s)
void IMHD2DSTOperatorAssembler::AssembleSTBlockBiDiagonal( const SparseMatrix& F, const SparseMatrix& M, ParBlockLowTriOperator& FF,
                                                           const std::string& STMatName, const bool blocksAssembled, bool& STMatAssembled ){ 

  if( STMatAssembled )
    return;

  if( !blocksAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble its blocks before assembling "<< STMatName <<std::endl;
    return;
  }

  FF.SetBlockDiag( &F, 0, false );  // false doesn't own
  FF.SetBlockDiag( &M, 1, false );  // false doesn't own

  STMatAssembled = true;

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time operator " << STMatName <<" assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







// Assemble FFu^-1 (used in preconditioner)
void IMHD2DSTOperatorAssembler::AssembleFFuinv( ){
  if ( _FFuinvAssembled ){
    return;
  }

  switch (_USTSolveType){
    // Use sequential time-stepping to solve for space-time block
    case 0:{
      if(!( _MuAssembled && _FuAssembled ) && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFuinv: need to assemble Mu and Fu before assembling FFuinv\n";
        return;
      }
      //                                             flag as time-dependent regardless of anything
      SpaceTimeSolver *temp  = new SpaceTimeSolver( _comm, NULL, NULL, _essUhTDOF, true, _verbose );

      temp->SetF( &_Fu );
      temp->SetM( &_Mu );
      _FFuinv = temp;

      _FFuinvAssembled = true;
      
      break;
    }

    // // Use BoomerAMG with AIR set-up
    // case 1:{
    //   if(! _FFuAssembled  && _myRank == 0 ){
    //     std::cerr<<"ERROR: AssembleFFuinv: need to assemble FFu before assembling FFuinv\n";
    //     return;
    //   }

    //   // Initialise MFEM wrapper for BoomerAMG solver
    //   HypreBoomerAMG *temp = new HypreBoomerAMG( *_FFFu );

    //   // Cast as HYPRE_Solver to get the underlying hypre object
    //   HYPRE_Solver FFuinv( *temp );

    //   // Set it up
    //   SetUpBoomerAMG( FFuinv );

    //   _FFuinv = temp;
  
    //   _FFuinvAssembled = true;

    //   break;
    // }



    // // Use GMRES with BoomerAMG precon
    // case 2:{
    //   if(! _FFuAssembled  && _myRank == 0 ){
    //     std::cerr<<"ERROR: AssembleFFuinv: need to assemble FFu before assembling FFuinv\n";
    //     return;
    //   }
    //   if( _myRank == 0 ){
    //     std::cout<<"WARNING: Since you're using GMRES to solve the space-time block inside the preconditioner"<<std::endl
    //              <<"         make sure that flexible GMRES is used as the outer solver!"<<std::endl;
    //   }


    //   // Initialise MFEM wrappers for GMRES solver and preconditioner
    //   HypreGMRES     *temp  = new HypreGMRES(     *_FFFu );
    //   HypreBoomerAMG *temp2 = new HypreBoomerAMG( *_FFFu );

    //   // Cast preconditioner as HYPRE_Solver to get the underlying hypre object
    //   HYPRE_Solver FFuinvPrecon( *temp2 );
    //   // Set it up
    //   SetUpBoomerAMG( FFuinvPrecon, 1 );   // with just one iteration this time around

    //   // Attach preconditioner to solver
    //   temp->SetPreconditioner( *temp2 );

    //   // adjust gmres options
    //   temp->SetKDim( 50 );
    //   temp->SetTol( 0.0 );   // to ensure fixed number of iterations
    //   temp->SetMaxIter( 15 );

    //   _FFuinv     = temp;
    //   _FFuinvPrec = temp2;
  
    //   _FFuinvAssembled = true;

    //   break;
    // }

    // // Use Parareal with coarse/fine solver of different accuracies
    // case 3:{

    //   const int maxIt = 2;

    //   if( _numProcs <= maxIt ){
    //     if( _myRank == 0 ){
    //       std::cerr<<"ERROR: AssembleFFinv: Trying to set solver as "<<maxIt<<" iterations of Parareal, but the fine discretisation only has "<<_numProcs<<" nodes. "
    //                <<"This is equivalent to time-stepping, so I'm picking that as a solver instead."<<std::endl;
    //     }
        
    //     AssembleFFinv( 0 );
    //     return;
    //   }


    //   PararealSolver *temp  = new PararealSolver( _comm, NULL, NULL, NULL, maxIt, _verbose );

    //   temp->SetF( &_Fw );
    //   temp->SetC( &_Fw ); // same operator is used for both! it's the solver that changes, eventually...
    //   temp->SetM( &_Mw );
    //   _FFinv = temp;

    //   _FFinvAssembled = true;
      
    //   break;
    // }
    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<_USTSolveType<<" not recognised."<<std::endl;
      }
    }
  }


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time block inverse (approximation) FF^-1 assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}






// Assemble MMz^-1 (used in preconditioner)
void IMHD2DSTOperatorAssembler::AssembleMMzinv(){
  if ( _MMzinvAssembled ){
    return;
  }

  if(!( _MzAssembled ) && _myRank == 0 ){
    std::cerr<<"Need to assemble Mz before assembling MMzinv\n";
    return;
  }


  _Mztemp = new PetscParMatrix( &_Mz );

  _MMzinv = new PetscLinearSolver( *_Mztemp, "ZSolverMass_" );


  _MMzinvAssembled = true;
}






// Assemble pressure Schur complement
void IMHD2DSTOperatorAssembler::AssemblePS(){
  if ( _pSAssembled ){
    return;
  }

  // Assemble relevant operators
  AssembleAp();
  AssembleMp();
  AssembleWp();

  if( _myRank == 0 && _essPhTDOF.Size() !=0 ){
    std::cout<<"Warning: On the PCD operator, we're imposing Dirichlet BC in outflow, just like for Ap."<<std::endl
             <<"         This goes against what Elman/Silvester/Wathen says (should be Robin everywhere for PCD,"<<std::endl
             <<"         while for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl
             <<"         Particularly, for tolerances up to 1e-6 the convergence profile with Dirichlet in outflow"<<std::endl
             <<"         is better, while for stricter tolerances the one with Robin seems to work better."<<std::endl
  }
  // pass an empty vector instead of _essPhTDOF to follow E/S/W!
  _pSinv = new OseenSTPressureSchurComplement( _comm, _dt, _mu, NULL, NULL, NULL, _essPhTDOF, _verbose );

  _pSinv->SetAp( &_Ap );
  _pSinv->SetMp( &_Mp );


  // Check if there is advection (if not, some simplifications can be made)
  bool isQuietState = true;
  // - check whether all node values of w are 0
  for ( int i = 0; i < _wFuncCoeff.Size(); ++i ){
    if ( _wFuncCoeff[i] != 0. ){
      isQuietState = false;
      break;
    }
  }

  // if there is advection, then clearly Wp differs from Ap (must include pressure convection)
  //  otherwise, they're the same and some simplifications can be made
  _pSinv->SetWp( &_Wp, isQuietState );


  _pSAssembled = true;

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time pressure Schur complement inverse approximation assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}




// Assemble CCa^-1 (used in magnetic Schur complement)
void IMHD2DSTOperatorAssembler::AssembleCCainv( ){

  if ( _CCainvAssembled ){
    return;
  }

  switch ( _ASTSolveType ){
    // Use sequential time-stepping on implicit / explicit leapfrog discretisation to solve for space-time block
    case 0:
    case 1:{
      AssembleCs();

      SpaceTimeWaveSolver *temp  = new SpaceTimeWaveSolver( _comm, NULL, NULL, NULL, _essAhTDOF, false, true, _verbose);
      temp->SetDiag( &_Cp, 0 );
      temp->SetDiag( &_C0, 1 );
      temp->SetDiag( &_Cm, 2 );
      _CCainv = temp;

      _CCainvAssembled = true;
      
      break;
    }


    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<_ASTSolveType<<" not recognised."<<std::endl;
      }
    }
  }


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time block inverse (approximation) of magnetic wave eq CCa^-1 assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}






// Assemble magnetic Schur complement
void IMHD2DSTOperatorAssembler::AssembleAS( ){
  if ( _aSAssembled ){
    return;
  }

  AssembleCCainv( );
  AssembleMaNoZero();
  AssembleMaNoZeroLumped(); // TODO: these are not necessary...
  AssembledtuWa();          // TODO: these are not necessary...
  AssembleWa();

  _aSinv = new IMHD2DSTMagneticSchurComplement( _comm, _dt, NULL, NULL, NULL, _essAhTDOF, _verbose );

  _aSinv->SetM( &_MaNoZero );
  _aSinv->SetW( &_Wa );
  _aSinv->SetCCinv( _CCainv );


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time magnetic Schur complement inverse approximation assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}







// Utility function for setting up FFuinv
void IMHD2DSTOperatorAssembler::SetUpBoomerAMG( HYPRE_Solver& FFinv, const int maxiter ){
  int printLevel = 0;

  // AIR parameters for diffusion equation:
  double distance_R = 1.5;
  std::string prerelax = "A";
  std::string postrelax = "FFC";
  int interp_type = 0;
  int relax_type = 8;
  int coarsen_type = 6;
  double strength_tolC = 0.005;
  double strength_tolR = 0.005;
  double filter_tolR = 0.0;
  double filter_tolA = 0.0;
  int cycle_type = 1;


  // // AIR parameters:
  // double distance_R = 1.5;
  // std::string prerelax = "A";
  // std::string postrelax = "FFC";
  // int interp_type = 100;
  // int relax_type = 3;
  // int coarsen_type = 6;
  // double strength_tolC = 0.005;
  // double strength_tolR = 0.005;
  // double filter_tolR = 0.0;
  // double filter_tolA = 0.0;
  // int cycle_type = 1;

  // AMG parameters
  // double distance_R = -1;
  // std::string prerelax = "AA";
  // std::string postrelax = "AA";
  // int interp_type = 6;
  // int relax_type = 3;
  // int coarsen_type = 6;
  // double strength_tolC = 0.1;
  // double strength_tolR = -1;
  // double filter_tolR = 0.0;
  // double filter_tolA = 0.0;
  // int cycle_type = 1;

  // // AIR hyperbolic parameters
  // double distance_R = 1.5;
  // std::string prerelax = "A";
  // std::string postrelax = "F";
  // int interp_type = 100;
  // int relax_type = 10;
  // int coarsen_type = 6;
  // double strength_tolC = 0.005;
  // double strength_tolR = 0.005;
  // double filter_tolR = 0.0;
  // double filter_tolA = 0.0001;
  // int cycle_type = 1;


  // Create preconditioner
  HYPRE_BoomerAMGSetTol( FFinv, 0 );    // set tolerance to 0 so to have a fixed number of iterations
  HYPRE_BoomerAMGSetMaxIter( FFinv, maxiter );
  HYPRE_BoomerAMGSetPrintLevel( FFinv, printLevel );

  unsigned int ns_down = prerelax.length();
  unsigned int ns_up   = postrelax.length();
  int ns_coarse = 1;
  std::string Fr("F");
  std::string Cr("C");
  std::string Ar("A");
  int* *grid_relax_points = new int* [4];
  grid_relax_points[0] = NULL;
  grid_relax_points[1] = new int[ns_down];
  grid_relax_points[2] = new int [ns_up];
  grid_relax_points[3] = new int[1];
  grid_relax_points[3][0] = 0;

  // set down relax scheme 
  for(unsigned int i = 0; i<ns_down; i++) {
    if (prerelax.compare(i,1,Fr) == 0) {
      grid_relax_points[1][i] = -1;
    }
    else if (prerelax.compare(i,1,Cr) == 0) {
      grid_relax_points[1][i] = 1;
    }
    else if (prerelax.compare(i,1,Ar) == 0) {
      grid_relax_points[1][i] = 0;
    }
  }

  // set up relax scheme 
  for(unsigned int i = 0; i<ns_up; i++) {
    if (postrelax.compare(i,1,Fr) == 0) {
      grid_relax_points[2][i] = -1;
    }
    else if (postrelax.compare(i,1,Cr) == 0) {
      grid_relax_points[2][i] = 1;
    }
    else if (postrelax.compare(i,1,Ar) == 0) {
      grid_relax_points[2][i] = 0;
    }
  }


  if (distance_R > 0) {
    HYPRE_BoomerAMGSetRestriction( FFinv, distance_R );
    HYPRE_BoomerAMGSetStrongThresholdR( FFinv, strength_tolR );
    HYPRE_BoomerAMGSetFilterThresholdR( FFinv, filter_tolR );
  }
  HYPRE_BoomerAMGSetInterpType( FFinv, interp_type );
  HYPRE_BoomerAMGSetCoarsenType( FFinv, coarsen_type );
  HYPRE_BoomerAMGSetAggNumLevels( FFinv, 0 );
  HYPRE_BoomerAMGSetStrongThreshold( FFinv, strength_tolC );
  HYPRE_BoomerAMGSetGridRelaxPoints( FFinv, grid_relax_points );     // TODO: THIS FUNCTION IS DEPRECATED!! nobody knows whose responsibility it is to free grid_relax_points
  if (relax_type > -1) {
    HYPRE_BoomerAMGSetRelaxType( FFinv, relax_type );
  }
  HYPRE_BoomerAMGSetCycleNumSweeps( FFinv, ns_coarse, 3 );
  HYPRE_BoomerAMGSetCycleNumSweeps( FFinv, ns_down,   1 );
  HYPRE_BoomerAMGSetCycleNumSweeps( FFinv, ns_up,     2 );
  if (filter_tolA > 0) {
    HYPRE_BoomerAMGSetADropTol( FFinv, filter_tolA );
  }
  // type = -1: drop based on row inf-norm
  else if (filter_tolA == -1) {
    HYPRE_BoomerAMGSetADropType( FFinv, -1 );
  }

  // Set cycle type for solve 
  HYPRE_BoomerAMGSetCycleType( FFinv, cycle_type );


}











// Assembles operators appearing in space-time KHI block system, and other vectors of interest
//  DN(x0) Δx0 = rhs - N(x0), with
//           ⌈ FFFu BBB ZZZ1 ZZZ2 ⌉
//  DN(x0) = | BBBt               |.
//           |          MMMz KKK  |
//           ⌊ YYY           FFFa ⌋
// This function is designed to be invoked at the beginning of the Newton iteration, and it assembles:
//  - A suitable initial guess for the system: all-zero, but with dirichlet BC included (inside the IG* vecs)
//  - The rhs of the Newton system at the zeroth iteration: rhs - N(x0) (inside the *rhs vecs)
//  - The various blocks composing the operator gradient at the initial state
// NB: notice all vectors have zeroed essential nodes, apart from IG (=x0)
void IMHD2DSTOperatorAssembler::AssembleSystem( Operator*& FFFu, Operator*& MMMz, Operator*& FFFa,
                                                Operator*& BBB,  Operator*& ZZZ1, Operator*& ZZZ2,
                                                Operator*& KKK,  Operator*& YYY,
                                                Vector*& frhs,   Vector*& grhs,   Vector*& zrhs,   Vector*& hrhs,
                                                Vector*& IGu,    Vector*& IGp,    Vector*& IGz,    Vector*& IGa  ){



  // ASSEMBLE LOCAL RHS -----------------------------------------------------
  // Initialise handy functions for rhs
  VectorFunctionCoefficient uFuncCoeff( _dim, _uFunc );
  FunctionCoefficient       pFuncCoeff( _pFunc );
  FunctionCoefficient       zFuncCoeff( _zFunc );
  FunctionCoefficient       aFuncCoeff( _aFunc );
  VectorFunctionCoefficient fFuncCoeff( _dim, _fFunc );
  FunctionCoefficient       gFuncCoeff( _gFunc );
  FunctionCoefficient       hFuncCoeff( _hFunc );
  VectorFunctionCoefficient nFuncCoeff( _dim, _nFunc );
  FunctionCoefficient       mFuncCoeff( _mFunc );

  // - specify evaluation time
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  zFuncCoeff.SetTime( _dt*(_myRank+1) );
  aFuncCoeff.SetTime( _dt*(_myRank+1) );
  fFuncCoeff.SetTime( _dt*(_myRank+1) );
  gFuncCoeff.SetTime( _dt*(_myRank+1) );
  hFuncCoeff.SetTime( _dt*(_myRank+1) );
  nFuncCoeff.SetTime( _dt*(_myRank+1) );
  mFuncCoeff.SetTime( _dt*(_myRank+1) );


  // Assemble local part of rhs
  // - for u
  // -- identify neumann nodes for U
  Array<int> neuBdrU(_mesh->bdr_attributes.Max());
  neuBdrU = 0;
  if ( _mesh->bdr_attributes.Size() > 0 ) {
    // search among all possible tags
    for ( int i = 0; i < neuBdrU.Size(); ++i ){
      // if that tag is NOT marked in the corresponding array in essTags, then flag it
      if( !( _essTagsU.Find( i ) + 1 ) )
        neuBdrU[i] = 1;
    }
  }
  // if (_myRank == 0 ){
  //   std::cout << "Tags neuU ";  neuBdrU.Print(  mfem::out,  neuBdrU.Size()   ); std::cout<< "\n";
  // }
  LinearForm *fform( new LinearForm );
  fform->Update( _UhFESpace );
  fform->AddDomainIntegrator(   new VectorDomainLFIntegrator(       fFuncCoeff       )          );  //int_\Omega f*v
  fform->AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     nFuncCoeff       ), neuBdrU );  //int_d\Omega \mu * du/dn *v
  fform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ), neuBdrU );  //int_d\Omega -p*v*n

  fform->Assemble();

// #ifdef MULT_BY_DT
  fform->operator*=( _dt );
// #endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for u assembled\n";
    }
    MPI_Barrier(_comm);
  }  

  // -- include initial conditions
  if( _myRank == 0 ){
    uFuncCoeff.SetTime( 0.0 );
    LinearForm *u0form( new LinearForm );
    u0form->Update( _UhFESpace );
    u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) );  //int_\Omega u0*v
    u0form->Assemble();

// #ifndef MULT_BY_DT
//     u0form->operator*=(1./_dt);
// #endif
    fform->operator+=( *u0form );


    if ( _verbose>100 ){
      std::cout<<"Contribution from IC on u: "; u0form->Print(std::cout, u0form->Size());
    }

    // remember to reset function evaluation for w to the current time
    uFuncCoeff.SetTime( _dt*(_myRank+1) );


    delete u0form;

    if(_verbose>10){
      std::cout<<"Contribution from initial condition on u included\n"<<std::endl;
    }
  }



  // - for p
  LinearForm *gform( new LinearForm );
  gform->Update( _PhFESpace );
  gform->AddDomainIntegrator( new DomainLFIntegrator( gFuncCoeff ) );  //int_\Omega g*q
  gform->Assemble();

// #ifdef MULT_BY_DT
  gform->operator*=( _dt );
// #endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for p assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // - for z
  // -- identify neumann nodes for A
  Array<int> neuBdrA(_mesh->bdr_attributes.Max());
  neuBdrA = 0;
  if ( _mesh->bdr_attributes.Size() > 0 ) {
    // search among all possible tags
    for ( int i = 0; i < neuBdrA.Size(); ++i ){
      // if that tag is NOT marked in the corresponding array in essTags, then flag it
      if( !( _essTagsA.Find( i ) + 1 ) )
        neuBdrA[i] = 1;
    }
  }
  // if (_myRank == 0 ){
  //   std::cout << "Tags neuA ";  neuBdrA.Print(  mfem::out,  neuBdrA.Size()   ); std::cout<< "\n";
  // }
  LinearForm *zform( new LinearForm );
  // -- after integration by parts, I end up with a \int_\partial\Omega dA/dn * zeta
  zform->Update( _ZhFESpace );
  zform->AddBoundaryIntegrator( new BoundaryLFIntegrator( mFuncCoeff ), neuBdrA );  //int_d\Omega \eta * dA/dn *zeta, so I need to rescale by eta
  zform->Assemble();
  zform->operator*=( 1./_eta );

// #ifdef MULT_BY_DT
  zform->operator*=( _dt );
// #endif



  // - for A
  LinearForm *hform( new LinearForm );
  hform->Update( _AhFESpace );
  hform->AddDomainIntegrator(   new DomainLFIntegrator(   hFuncCoeff )          );  //int_\Omega h*B
  hform->AddBoundaryIntegrator( new BoundaryLFIntegrator( mFuncCoeff ), neuBdrA );  //int_d\Omega \eta * dA/dn *B
  hform->Assemble();

// #ifdef MULT_BY_DT
  hform->operator*=( _dt );
// #endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for A assembled\n";
    }
    MPI_Barrier(_comm);
  }  

  // -- include initial conditions
  if( _myRank == 0 ){
    aFuncCoeff.SetTime( 0.0 );
    LinearForm *a0form( new LinearForm );
    a0form->Update( _AhFESpace );
    a0form->AddDomainIntegrator( new DomainLFIntegrator( aFuncCoeff ) );  //int_\Omega A0*B
    a0form->Assemble();

// #ifndef MULT_BY_DT
//     a0form->operator*=(1./_dt);
// #endif
    hform->operator+=( *a0form );


    if ( _verbose>100 ){
      std::cout<<"Contribution from IC on A: "; a0form->Print(std::cout, a0form->Size());
    }

    // remember to reset function evaluation for A to the current time
    aFuncCoeff.SetTime( _dt*(_myRank+1) );


    delete a0form;

    if(_verbose>10){
      std::cout<<"Contribution from initial condition on A included\n"<<std::endl;
    }
  }



  // Initialise local rhs with contribution from linear forms
  Vector fRhsLoc = *fform;
  Vector gRhsLoc = *gform;
  Vector zRhsLoc = *zform;
  Vector hRhsLoc = *hform;

  // - the linear forms have now served their purpose
  delete fform;
  delete gform;
  delete zform;
  delete hform;

  // - include effect from BC (kill contributions there)
  fRhsLoc.SetSubVector( _essUhTDOF, 0.0);
  // gRhsLoc.SetSubVector( _essPhTDOF, 0.0);  // no dirichlet BC imposed on pressure
  // zRhsLoc.SetSubVector( _essZhTDOF, 0.0);
  hRhsLoc.SetSubVector( _essAhTDOF, 0.0);






  // ASSEMBLE LOCAL IG ------------------------------------------------------
  // // - initialise function with BC
  // GridFunction uBC(_UhFESpace);
  // GridFunction zBC(_ZhFESpace);
  // GridFunction aBC(_AhFESpace);
  // uBC.ProjectCoefficient(uFuncCoeff);
  // zBC.ProjectCoefficient(zFuncCoeff);
  // aBC.ProjectCoefficient(aFuncCoeff);
  // // - initialise local initial guess to exact solution on Dirichlet nodes
  // Vector iguLoc = uBC;
  // Vector igpLoc(_PhFESpace->GetTrueVSize()); igpLoc = 0.;     // dirichlet BC are not actually imposed on p
  // Vector igzLoc = zBC;
  // Vector igaLoc = aBC;
  // iguLoc.SetSubVectorComplement( _essUhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  // igzLoc.SetSubVectorComplement( _essZhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  // igaLoc.SetSubVectorComplement( _essAhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?

  // use linearised state as IG:
  Vector iguLoc = _wFuncCoeff;
  Vector igpLoc(_PhFESpace->GetTrueVSize()); igpLoc = 0.;     // dirichlet BC are not actually imposed on p
  Vector igzLoc = _yFuncCoeff;
  Vector igaLoc = _cFuncCoeff;








  // ASSEMBLE LOCAL MATRICES ------------------------------------------------
  // - Compute local operator gradient at IG
  //  -- since we already flagged the relevant esential BC to the local operator,
  //      the matrices are already constrained
  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = _UhFESpace->GetTrueVSize();
  offsets[2] = _PhFESpace->GetTrueVSize();
  offsets[3] = _ZhFESpace->GetTrueVSize();
  offsets[4] = _AhFESpace->GetTrueVSize();
  offsets.PartialSum();
  BlockVector x(offsets);
  x.GetBlock(0) = iguLoc;
  x.GetBlock(1) = igpLoc;
  x.GetBlock(2) = igzLoc;
  x.GetBlock(3) = igaLoc;

  BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( x ) );
  _Fu = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,0) ) );
  _Z1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) ) );
  _Z2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) ) );
  _B  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(1,0) ) );
  _Mz = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(2,2) ) );
  _K  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(2,3) ) );
  _Y  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,0) ) );
  _Fa = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,3) ) );
  _FuAssembled = true;
  _Z1Assembled = true;
  _Z2Assembled = true;
  _BAssembled  = true; 
  _MzAssembled = true;
  _KAssembled  = true; 
  _YAssembled  = true; 
  _FaAssembled = true;



  // - Only missing are the local mass matrices for time-stepping
  // -- handy variables indicating dirichlet nodes
  Array<int> colsU(_Fu.Height());
  Array<int> colsA(_Fa.Height());
  colsU = 0;
  colsA = 0;
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    colsU[_essUhTDOF[i]] = 1;
  }
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }


  // -- Assemble _Mu (and modify rhs to take dirichlet on u into account within space-time structure)
  AssembleMu();
  uFuncCoeff.SetTime( _dt*_myRank );                // set uFunc to previous time-step
  GridFunction um1BC(_UhFESpace);
  um1BC.ProjectCoefficient(uFuncCoeff);

  Vector um1Rel( fRhsLoc.Size() );
  um1Rel = 0.0;
  _Mu.EliminateCols( colsU, &um1BC, &um1Rel );
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    _Mu.EliminateRow( _essUhTDOF[i], mfem::Matrix::DIAG_ZERO );
    // um1Rel(_essUhTDOF[i]) = 0.0;  // rhs will be killed later at dirichlet BC
  }

  if( _myRank > 0 ){
    // add to rhs (um1Rel should already take minus sign on _Mu into account)
    // NB: - no need to rescale by dt, as _Mu will be already scaled accordingly.
    //     - no need to flip sign, as _Mu carries with it already
    fRhsLoc += um1Rel;
  }

  // -- remember to reset function evaluation for u to the current time
  uFuncCoeff.SetTime( _dt*(_myRank+1) );

  // -- assembly of the velocity matrices is now complete
  _MuAssembled = true;



  // -- Assemble _Ma (and modify rhs to take dirichlet on A into account within space-time structure)
  AssembleMa();
  aFuncCoeff.SetTime( _dt*_myRank );                // set aFunc to previous time-step
  GridFunction am1BC(_AhFESpace);
  am1BC.ProjectCoefficient(aFuncCoeff);

  Vector am1Rel( hRhsLoc.Size() );
  am1Rel = 0.0;
  _Ma.EliminateCols( colsA, &am1BC, &am1Rel );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _Ma.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
    // am1Rel(_essAhTDOF[i]) = 0.0;  // rhs will be killed later at dirichlet BC
  }

  if( _myRank > 0 ){
    // add to rhs (am1Rel should already take minus sign on _Ma into account)
    // NB: - no need to rescale by dt, as _Ma will be already scaled accordingly.
    //     - no need to flip sign, as _Ma carries with it already
    hRhsLoc += am1Rel;
  }

  // -- remember to reset function evaluation for A to the current time
  aFuncCoeff.SetTime( _dt*(_myRank+1) );

  // -- assembly of the vector potential matrices is now complete
  _MaAssembled = true;






  // ASSEMBLE LOCAL OPERATOR EVALUATIONS ------------------------------------
  // Compute local action of non-linear operator on initial guess
  // - alright, so, at this stage the rhs vectors contain:
  //  -- evaluations from the boundary integrals, initial conditions, and forcing terms
  //  -- impacts of dirichlet conditions on the space-time structure (the subdiagonal containing the mass matrix)
  //   but they're still lacking the impact of the dir BC on the remaining part of the operator!
  //   Hopefully the following will include this:
  BlockVector y(offsets);

  ApplyOperator(x,y);
  fRhsLoc -= y.GetBlock(0);
  gRhsLoc -= y.GetBlock(1);
  zRhsLoc -= y.GetBlock(2);
  hRhsLoc -= y.GetBlock(3);
  // ...the advantage of including the impact of the space-time structure directly on the rhs (rather than having to evaluate
  //  it inside ApplyOperator ) is that this way I can store the matrices Mu and Ma with zeroed columns, and be sure that
  //  eventual dirtying of the BC won't have an impact on the operator application. Is it useful? Meh, dunno, but this should
  //  work anyway...

  // - finally, kill the rhs on the Dirichlet nodes (the system we'll solve involves Δx: don't wanna risk dirtying the BC)
  fRhsLoc.SetSubVector( _essUhTDOF, 0. );
  // zRhsLoc.SetSubVector( _essZhTDOF, 0. );
  hRhsLoc.SetSubVector( _essAhTDOF, 0. );






  // ASSEMBLE GLOBAL (PARALLEL) RHS -----------------------------------------
  // - for velocity
  int colPartU[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPartU );
  // frhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for velocity block f assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for pressure
  int colPartP[2] = {_myRank*gRhsLoc.Size(), (_myRank+1)*gRhsLoc.Size()};
  grhs = new HypreParVector( _comm, gRhsLoc.Size()*_numProcs, gRhsLoc.StealData(), colPartP );
  // grhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for pressure block g assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for Laplacian of vector potential
  int colPartZ[2] = {_myRank*zRhsLoc.Size(), (_myRank+1)*zRhsLoc.Size()};
  zrhs = new HypreParVector( _comm, zRhsLoc.Size()*_numProcs, zRhsLoc.StealData(), colPartZ );
  // zrhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for Laplacian of vector potential block '0' assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for vector potential
  int colPartA[2] = {_myRank*hRhsLoc.Size(), (_myRank+1)*hRhsLoc.Size()};
  hrhs = new HypreParVector( _comm, hRhsLoc.Size()*_numProcs, hRhsLoc.StealData(), colPartA );
  // hrhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for vector potential block h assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  // ASSEMBLE INITIAL GUESS -------------------------------------------------
  // Assemble global vectors
  IGu = new HypreParVector( _comm, iguLoc.Size()*_numProcs, iguLoc.StealData(), colPartU );
  IGp = new HypreParVector( _comm, igpLoc.Size()*_numProcs, igpLoc.StealData(), colPartP );
  IGz = new HypreParVector( _comm, igzLoc.Size()*_numProcs, igzLoc.StealData(), colPartZ );
  IGa = new HypreParVector( _comm, igaLoc.Size()*_numProcs, igaLoc.StealData(), colPartA );
  // IGu->SetOwnership( 1 );
  // IGp->SetOwnership( 1 );
  // IGz->SetOwnership( 1 );
  // IGa->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time initial guesses assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  // ASSEMBLE SPACE-TIME OPERATORS ------------------------------------------
  AssembleFFu();
  FFFu = &_FFu;

  AssembleMMz();
  MMMz = &_MMz;

  AssembleFFa();
  FFFa = &_FFa;

  AssembleBB();
  BBB = &_BB;
  
  AssembleZZ1();
  ZZZ1 = &_ZZ1;
  
  AssembleZZ2();
  ZZZ2 = &_ZZ2;
  
  AssembleKK();
  KKK = &_KK;

  AssembleYY();
  YYY = &_YY;



  /*
  //Assemble space-time velocity block
  AssembleFFu();
  // - pass handle to mfem matrix
  FFFu = new HypreParMatrix();
  FFFu->MakeRef( *_FFFu );

  //Assemble space-time Laplacian of potential block
  AssembleMMz();
  // - pass handle to mfem matrix
  MMMz = new HypreParMatrix();
  MMMz->MakeRef( *_MMMz );

  //Assemble space-time potential block
  AssembleFFa();
  // - pass handle to mfem matrix
  FFFa = new HypreParMatrix();
  FFFa->MakeRef( *_FFFa );


  //Assemble divergence block
  AssembleBB();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  BBref;
  HYPRE_IJMatrixGetObject( _BB, (void **) &BBref);
  BBB = new HypreParMatrix( BBref, false ); //"false" doesn't take ownership of data


  //Assemble Lorentz force blocks
  AssembleZZ1();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  ZZ1ref;
  HYPRE_IJMatrixGetObject( _ZZ1, (void **) &ZZ1ref);
  ZZZ1 = new HypreParMatrix( ZZ1ref, false ); //"false" doesn't take ownership of data
  AssembleZZ2();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  ZZ2ref;
  HYPRE_IJMatrixGetObject( _ZZ2, (void **) &ZZ2ref);
  ZZZ2 = new HypreParMatrix( ZZ2ref, false ); //"false" doesn't take ownership of data


  //Assemble mixed Laplacian block
  AssembleKK();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  KKref;
  HYPRE_IJMatrixGetObject( _KK, (void **) &KKref);
  KKK = new HypreParMatrix( KKref, false ); //"false" doesn't take ownership of data


  //Assemble potential advection block
  AssembleYY();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  YYref;
  HYPRE_IJMatrixGetObject( _YY, (void **) &YYref);
  YYY = new HypreParMatrix( YYref, false ); //"false" doesn't take ownership of data
  */

  

  if ( _verbose>50 ){
    // std::string myfilename = std::string("./results/IGu.dat");
    // IGu->Print(  myfilename.c_str() );
    // myfilename = std::string("./results/RHSu.dat");
    // frhs->Print( myfilename.c_str() );
    // myfilename = std::string("./results/IGp.dat");
    // IGp->Print(  myfilename.c_str() );
    // myfilename = std::string("./results/RHSp.dat");
    // grhs->Print( myfilename.c_str() );
    // myfilename = std::string("./results/IGz.dat");
    // IGz->Print(  myfilename.c_str() );
    // myfilename = std::string("./results/RHSz.dat");
    // zrhs->Print( myfilename.c_str() );
    // myfilename = std::string("./results/IGa.dat");
    // IGa->Print(  myfilename.c_str() );
    // myfilename = std::string("./results/RHSa.dat");
    // hrhs->Print( myfilename.c_str() );

    if ( _myRank == 0 ){
      std::ofstream myfile;
      std::string myfilename = "./results/out_final_B.dat";
      myfile.open( myfilename );
      _B.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Z1.dat";
      myfile.open( myfilename );
      _Z1.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Z2.dat";
      myfile.open( myfilename );
      _Z2.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Y.dat";
      myfile.open( myfilename );
      _Y.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_K.dat";
      myfile.open( myfilename );
      _K.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Fu.dat";
      myfile.open( myfilename );
      _Fu.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Mu.dat";
      myfile.open( myfilename );
      _Mu.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Mz.dat";
      myfile.open( myfilename );
      _Mz.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Fa.dat";
      myfile.open( myfilename );
      _Fa.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_final_Ma.dat";
      myfile.open( myfilename );
      _Ma.PrintMatlab(myfile);
      myfile.close( );

      myfilename = "./results/out_essU.dat";
      myfile.open( myfilename );
      _essUhTDOF.Print(myfile,1);
      myfile.close( );

      myfilename = "./results/out_essP.dat";
      myfile.open( myfilename );
      _essPhTDOF.Print(myfile,1);
      myfile.close( );

      // myfilename = "./results/out_essZ.dat";
      // myfile.open( myfilename );
      // _essZhTDOF.Print(myfile,1);
      // myfile.close( );

      myfilename = "./results/out_essA.dat";
      myfile.open( myfilename );
      _essAhTDOF.Print(myfile,1);
      myfile.close( );

      std::cout<<"U essential nodes: ";_essUhTDOF.Print(std::cout, _essUhTDOF.Size());
      std::cout<<"P essential nodes: ";_essPhTDOF.Print(std::cout, _essPhTDOF.Size());
      // std::cout<<"Z essential nodes: ";_essZhTDOF.Print(std::cout, _essZhTDOF.Size());
      std::cout<<"A essential nodes: ";_essAhTDOF.Print(std::cout, _essAhTDOF.Size());

    }

  }

}






// Compute application of non-linear operator to state vector (eventually scaled by dt)
// - When using this function, make sure to populate the state vector with its dirichlet BC,
//    as they have an impact on the non-linear part of the operator.
// - Notice however that the Dirichlet nodes *do not* have an impact on the time-derivative part
//    of this operator (which is linear). Their contribution should've been included once and for all 
//    in the rhs of the AssembleSystem method. The reasoning behind this choice is that this way I can store
//    the mass matrices as square operators surrounded by zeroes (corresponding to the dir nodes), and I don't
//    risk dirtying the dirichlet nodes when I reuse them in some preconditioner. Maybe a bit of a mess, but oh, well...
//  dt * N(xk)
void IMHD2DSTOperatorAssembler::ApplyOperator( const BlockVector& x, BlockVector& y ){
  if ( _myRank == 0 ){
    std::cout<<"Evaluating action of non-linear operator: did you make sure to include Dirichlet BC on the state x?"<<std::endl;
  }

  if ( !( _MuAssembled && _MaAssembled ) ){ 
    std::cerr<<"ERROR: ApplyOperator: need to assemble Mu and Ma in order to apply the operator\n";
    return;
  }

  // Assemble local part of operator evaluation
  _IMHD2DOperator.Mult(x,y);


  // Include contribution from time derivative ------------------------------
  // - for u
  // -- Send lcl solution to next proc
  const int sizeU = x.GetBlock(0).Size();
  Vector prevU( sizeU ), tempU( sizeU );
  prevU = 0.;
  if ( _myRank < _numProcs-1 ){
    MPI_Send( x.GetBlock(0).GetData(), sizeU, MPI_DOUBLE, _myRank+1, 2*_myRank,     _comm );
  }
  if ( _myRank > 0 ){
    MPI_Recv(         prevU.GetData(), sizeU, MPI_DOUBLE, _myRank-1, 2*(_myRank-1), _comm, MPI_STATUS_IGNORE );
  }

  // -- Mu should already be stored with negative sign and scaling by dt
  _Mu.Mult( prevU, tempU );
  y.GetBlock(0) += tempU;



  // - for A
  // -- Send lcl solution to next proc
  const int sizeA = x.GetBlock(3).Size();
  Vector prevA( sizeA ), tempA( sizeA );
  prevA = 0.;
  if ( _myRank < _numProcs-1 ){
    MPI_Send( x.GetBlock(3).GetData(), sizeA, MPI_DOUBLE, _myRank+1, 2*_myRank+1,     _comm );
  }
  if ( _myRank > 0 ){
    MPI_Recv(         prevA.GetData(), sizeA, MPI_DOUBLE, _myRank-1, 2*(_myRank-1)+1, _comm, MPI_STATUS_IGNORE );
  }

  // -- Ma should already be stored with negative sign and scaling by dt
  _Ma.Mult( prevA, tempA );
  y.GetBlock(3) += tempA;




  // This might be superfluous, but kill all Dirichlet contributions
  //  (this is in line with what the Mult() method of NonLinearForm does)
  for (int i = 0; i < _essUhTDOF.Size(); ++i)
    y.GetBlock(0)(_essUhTDOF[i]) = 0.;
  // for (int i = 0; i < _essZhTDOF.Size(); ++i)
  //   y.GetBlock(2)(_essZhTDOF[i]) = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i)
    y.GetBlock(3)(_essAhTDOF[i]) = 0.;


}







// Update linearisation
// We need to re-assemble all the operators which come from linearisations of non-linear ones
// NB: make sure the Dirichlet conditions are already included in x! Just to be on the safe side...
void IMHD2DSTOperatorAssembler::UpdateLinearisedOperators( const BlockVector& x ){
  if ( _myRank == 0 ){
    std::cout<<"Updating Gradient: did you make sure to include Dirichlet BC on the state x?"<<std::endl;
  }
  

  // Update internal variables
  for ( int i = 0; i < x.GetBlock(0).Size(); ++i ){
    _wFuncCoeff.GetData()[i] = x.GetBlock(0).GetData()[i];
  }
  for ( int i = 0; i < x.GetBlock(2).Size(); ++i ){
    _yFuncCoeff.GetData()[i] = x.GetBlock(2).GetData()[i];
  }
  for ( int i = 0; i < x.GetBlock(3).Size(); ++i ){
    _cFuncCoeff.GetData()[i] = x.GetBlock(3).GetData()[i];
  }


  // Re-evaluate operator gradient
  _Fu.Clear();
  _Z1.Clear();
  _Z2.Clear();
  _Y.Clear();
  _Fa.Clear();

  BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( x ) );
  _Fu = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,0) ) );
  _Z1 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,2) ) );
  _Z2 = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(0,3) ) );
  _Y  = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,0) ) );
  _Fa = *( dynamic_cast<SparseMatrix*>( &J->GetBlock(3,3) ) );


  // Re-assemble relevant operators
  _FFu.SetBlockDiag( &_Fu, 0, false );
  _FFa.SetBlockDiag( &_Fa, 0, false );
  _ZZ1.SetBlockDiag( &_Z1, 0, false );
  _ZZ2.SetBlockDiag( &_Z2, 0, false );
  _YY.SetBlockDiag(  &_Y,  0, false );


  /*
  // -- reassemble space-time velocity block
  if( _FFuAssembled )
    HYPRE_IJMatrixDestroy( _FFu );
  _FFuAssembled = false;
  delete _FFFu; // do I need this?
  AssembleFFu();

  // -- reassemble space-time vector potential block
  if( _FFaAssembled )
    HYPRE_IJMatrixDestroy( _FFa );
  _FFaAssembled = false;
  delete _FFFa; // do I need this?
  AssembleFFa();

  // -- reassemble space-time lorentz block
  if( _ZZ1Assembled )
    HYPRE_IJMatrixDestroy( _ZZ1 );
  _ZZ1Assembled = false;
  AssembleZZ1();
  if( _ZZ2Assembled )
    HYPRE_IJMatrixDestroy( _ZZ2 );
  _ZZ2Assembled = false;
  AssembleZZ2();

  // -- reassemble space-time magnetic convection block
  if( _YYAssembled )
    HYPRE_IJMatrixDestroy( _YY );
  _YYAssembled = false;
  AssembleYY();
  */


  // - whole space-time operators for preconditioners
  // -- update pSchur
  _WpAssembled = false;
  _Wp.Clear();
  AssembleWp();
  // --- check if there is advection (if not, some simplifications can be made)
  bool isQuietState = true;
  for ( int i = 0; i < _wFuncCoeff.Size(); ++i ){
    if ( _wFuncCoeff[i] != 0 ){
      isQuietState = false;
      break;
    }
  }

  _pSinv->SetWp( &_Wp, isQuietState );

  // if( !isQuietState ){
  //   _pSinv->SetWp( &_Wp, false );    // if there is advection, then clearly Wp differs from Ap (must include pressure convection)
  // }else if( _essPhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
  //   _pSinv->SetWp( &_Wp, true );    
  // }else{
  //   // _pSchur->SetWp( &_Wp, false );
  //   _pSinv->SetWp( &_Wp, true );     // should be false, according to E/S/W!
  //   if( _myRank == 0 ){
  //     std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
  //              <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
  //              <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
  //   }
  // }




  // update FFuinv
  switch (_USTSolveType){
    // Use sequential time-stepping to solve for space-time block
    case 0:{
      ( dynamic_cast<SpaceTimeSolver*>( _FFuinv ) )->SetF( &_Fu );

      break;
    }
    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<_USTSolveType<<" not recognised."<<std::endl;
      }
    }
  }



  // update aSchur
  _WaAssembled = false;
  _Wa.Clear();
  AssembleWa();
  _aSinv->SetW( &_Wa );

  switch ( _ASTSolveType ){
    // Use sequential time-stepping on implicit / explicit leapfrog discretisation to solve for space-time block
    case 0:
    case 1:{
      _CpAssembled = false;
      _C0Assembled = false;
      _CmAssembled = false;
      _Cp.Clear();
      _C0.Clear();
      _Cm.Clear();      
      AssembleCs();
      ( dynamic_cast<SpaceTimeWaveSolver*>( _CCainv ) )->SetDiag( &_Cp, 0 );
      ( dynamic_cast<SpaceTimeWaveSolver*>( _CCainv ) )->SetDiag( &_C0, 1 );
      ( dynamic_cast<SpaceTimeWaveSolver*>( _CCainv ) )->SetDiag( &_Cm, 2 );
      
      break;
    }

    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<_ASTSolveType<<" not recognised."<<std::endl;
      }
    }
  }



}















// Assemble inverse operators appearing in the various factors of the preconditioner. In particular, for the blocks
// 
//  ⌈ FFu^-1 \\\\\ \\ \\ ⌉     ⌈ I \\ \\\\\ \\\\\ ⌉
//  |        Sp^-1 \\ \\ |     |    I \\\\\ \\\\\ |
//  |               I \\ | and |      Mz^-1 \\\\\ |
//  ⌊                  I ⌋     ⌊            Sa^-1 ⌋
// where:
//  FFu contains space-time matrix for  velocity
//  Mz  contains space-time mass matrix for Laplacian of vector potential
//  Sp  contains the (approximate) space-time pressure Schur complement
//  Sa  contains the (approximate) space-time magnetic Schur complement
void IMHD2DSTOperatorAssembler::AssemblePreconditioner( Operator*& Fuinv, Operator*& Mzinv, Operator*& pSinv, Operator*& aSinv,
                                                        const int spaceTimeSolverTypeU, const int spaceTimeSolverTypeA ){
  //Assemble inverses
  _USTSolveType = spaceTimeSolverTypeU;
  _ASTSolveType = spaceTimeSolverTypeA;
  AssembleFFuinv( );
  AssembleMMzinv( );

  AssemblePS();
  AssembleAS();

  Fuinv = _FFuinv;
  Mzinv = _MMzinv;
  pSinv = _pSinv;
  aSinv = _aSinv;

}


















// Actual Time-stepper
void IMHD2DSTOperatorAssembler::TimeStep( const BlockVector& x, BlockVector& y,
                                          const std::string &fname1, const std::string &path2, int refLvl ){

  // Define operator
  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = _Fu.NumRows();
  offsets[2] =  _B.NumRows();
  offsets[3] = _Mz.NumRows();
  offsets[4] = _Fa.NumRows();
  offsets.PartialSum();
  PetscParMatrix myB( &_B );

  // - assemble matrix
  BlockOperator myMHDOp( offsets );
  myMHDOp.SetBlock(0, 0, &_Fu);
  myMHDOp.SetBlock(2, 2, &_Mz);
  myMHDOp.SetBlock(3, 3, &_Fa);
  myMHDOp.SetBlock(0, 1, myB.Transpose());
  myMHDOp.SetBlock(0, 2, &_Z1);
  myMHDOp.SetBlock(0, 3, &_Z2);
  myMHDOp.SetBlock(1, 0, &_B);
  myMHDOp.SetBlock(2, 3, &_K);
  myMHDOp.SetBlock(3, 0, &_Y);


  // Define preconditioner components
  // - Inverse of pressure Schur complement - reuse code from ST case, but with single processor
  OseenSTPressureSchurComplement myPSinv( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essPhTDOF, _verbose );
  AssembleAp();
  AssembleMp();
  AssembleWp();
  // -- check if there is advection (if not, some simplifications can be made)
  bool isQuietState = true;
  // --- check whether all node values of w are 0
  for ( int i = 0; i < _wFuncCoeff.Size(); ++i ){
    if ( _wFuncCoeff[i] != 0. ){
      isQuietState = false;
      break;
    }
  }
  // -- if there is advection, then clearly Wp differs from Ap (must include pressure convection)
  //     otherwise, they're the same and some simplifications can be made
  myPSinv.SetWp( &_Wp, isQuietState );
  myPSinv.SetAp( &_Ap );
  myPSinv.SetMp( &_Mp );


  // - Inverse of magnetic Schur complement - reuse code from ST case, but with single processor
  IMHD2DSTMagneticSchurComplement myASinv( MPI_COMM_SELF, _dt, NULL, NULL, NULL, _essAhTDOF, _verbose );
  AssembleMaNoZero();
  AssembleWa();
  myASinv.SetM( &_MaNoZero );
  myASinv.SetW( &_Wa );
  // -- compute operators for wave equation:
  // --- laplacian
  AssembleAa();
  // --- inverse of mass matrix (consider only its diagonal)
  AssembleMaNoZero();
  Vector MDiagInv;
  _MaNoZero.GetDiag( MDiagInv );
  for ( int i = 0; i < MDiagInv.Size(); ++i ){  // invert it
    MDiagInv(i) = 1./MDiagInv(i);
  }
  SparseMatrix MaLinv(MDiagInv);
  // --- approximation of wave equation
  // ---- compute the L2 norm of the space-time average of the magnetic field B = ∇x(kA)
  Vector B0(_dim);
  double area = 0.;
  B0 = 0.;
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
      _cFuncCoeff.GetGradient(*Tr,grad);
      grad *= ip.weight * Tr->Weight();
      B0 += grad;
    }
    area += 0.5 * (Tr->Weight());
  }
  MPI_Allreduce( MPI_IN_PLACE, B0.GetData(), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  double B0norm = sqrt( ( B0(0)*B0(0) + B0(1)*B0(1) ) / ( area * _dt*_numProcs) ); 
  // ---- assemble matrix for approximate wave equation
  SparseMatrix *MiF  = Mult(MaLinv,_Fa);
  SparseMatrix *FMiF = Mult(_Fa,*MiF);
  SparseMatrix *CC   = Add( 1., *FMiF, _dt*_dt*B0norm/_mu0, _Aa );
  delete MiF;
  delete FMiF;
  // ---- impose dirichlet BC
  mfem::Array<int> colsA(_Aa.Height());
  colsA = 0;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }
  CC->EliminateCols( colsA );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    CC->EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE  );
  }
  // ---- set solver
  PetscParMatrix myCC( CC );
  PetscLinearSolver myCCinv( myCC, "AWaveSolver_" );
  myASinv.SetCCinv( &myCCinv );




  // - Inverse of velocity operator
  const PetscParMatrix myFu( &_Fu );
  PetscLinearSolver myFuinv( myFu, "VSolver_" );



  // - Inverse of mass matrix of laplacian of vector field
  const PetscParMatrix myMz( &_Mz );
  PetscLinearSolver myMzinv( myMz, "ZSolverMass_" );





  // Assemble preconditioner
  // - inverse of Lub
  BlockLowerTriangularPreconditioner *Lub = new BlockLowerTriangularPreconditioner( offsets );
  Array<const Operator*> YFuiOps(2);
  YFuiOps[0] = &myFuinv;
  YFuiOps[1] = &_Y;
  OperatorsSequence* YFui = new OperatorsSequence( YFuiOps );   // does not own
  ScaledOperator* mMzi    = new ScaledOperator( &myMzinv, -1.0 );
  Array<const Operator*> mYFuiZ1Mziops(3);
  Array<bool>            mYFuiZ1Mziown(3);
  mYFuiZ1Mziops[0] = mMzi; mYFuiZ1Mziown[0] = true;
  mYFuiZ1Mziops[1] = &_Z1; mYFuiZ1Mziown[1] = false;
  mYFuiZ1Mziops[2] = YFui; mYFuiZ1Mziown[2] = false;
  OperatorsSequence* mYFuiZ1Mzi = new OperatorsSequence( mYFuiZ1Mziops, mYFuiZ1Mziown );
  Lub->iterative_mode = false;
  Lub->SetBlock( 3, 0,       YFui );
  Lub->SetBlock( 3, 2, mYFuiZ1Mzi );
  Lub->owns_blocks = true;

  // - inverse of Uub
  BlockUpperTriangularPreconditioner *Uub = new BlockUpperTriangularPreconditioner( offsets );
  Uub->iterative_mode = false;
  Uub->SetBlock( 0, 2, &_Z1     );
  Uub->SetBlock( 0, 3, &_Z2     );
  Uub->SetBlock( 2, 2, &myMzinv );
  Uub->SetBlock( 2, 3, &_K      );
  Uub->SetBlock( 3, 3, &myASinv );
  Uub->owns_blocks = false;
  
  // - inverse of Lup
  BlockLowerTriangularPreconditioner *Lup = new BlockLowerTriangularPreconditioner( offsets );
  Array<const Operator*> BFuiOps(2);
  BFuiOps[0] = &myFuinv;
  BFuiOps[1] = &_B;
  OperatorsSequence* BFui = new OperatorsSequence( BFuiOps );   // does not own
  Lup->iterative_mode = false;
  Lup->SetBlock( 1, 0, BFui );
  Lup->owns_blocks = true;
  
  // - inverse of Uup
  BlockUpperTriangularPreconditioner *Uup = new BlockUpperTriangularPreconditioner( offsets );
  Uup->iterative_mode = false;
  Uup->SetBlock( 0, 0, &myFuinv );
  Uup->SetBlock( 0, 1, myB.Transpose() );
  Uup->SetBlock( 1, 1, &myPSinv );
  Uup->owns_blocks = false;
  
  // - combine them together
  Array<const Operator*> precOps(4);
  Array<bool>      precOwn(4);
  precOps[0] = Lub;  precOwn[0] = true;
  precOps[1] = Uub;  precOwn[1] = true;
  precOps[2] = Lup;  precOwn[2] = true;
  precOps[3] = Uup;  precOwn[3] = true;
  OperatorsSequence myMHDPr( precOps, precOwn );



  // Define solver
  PetscLinearSolver solver( MPI_COMM_SELF, "solver_" );
  bool isIterative = true;
  solver.iterative_mode = isIterative;
  solver.SetPreconditioner(myMHDPr);
  solver.SetOperator(myMHDOp);

  double tol = 1e-10 / sqrt( _numProcs );
  if( _myRank == 0 ){
    std::cout<<"Warning: Considering a fixed overall tolerance of 1e-10, scaled by the number of time steps."<<std::endl
             <<"          This option gets overwritten if a tolerance is prescribed in the petsc option file,"<<std::endl    
             <<"          so make sure to delete it from there!"<<std::endl;    
  }
  solver.SetTol(tol);
  solver.SetRelTol(tol);
  solver.SetAbsTol(tol);



  // Main "time-stepping" routine *******************************************
  std::cerr<<"CHECK TIME-STEPPING! We are solving for gradient, so initial guess should be 0? And what about the temporal contribution to rhs?"<<std::endl;

  const int totDofs = x.Size();
  // - for each time-step, this will contain rhs
  BlockVector b( offsets );
  b = 0.0;
  
  // - receive solution from previous processor (unless initial time-step)
  if ( _myRank > 0 ){
    // - use it as initial guess for the solver (so store in y), unless initial step!
    MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
    // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
    _Mu.Mult( y.GetBlock(0), b.GetBlock(0) );
    _Ma.Mult( y.GetBlock(3), b.GetBlock(3) );
    // - M is stored with negative sign, so flip it
    b.GetBlock(0).Neg();
    b.GetBlock(3).Neg();
  }

  // - define rhs for this step (includes contribution from sol at previous time-step
  b += x;

  // - solve for current time-step
  //  --y acts as an initial guess! So use prev sol, unless first time step
  solver.Mult( b, y );

  int GMRESit    = solver.GetNumIterations();
  double resnorm = solver.GetFinalNorm();
  std::cout<<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

  if (!std::experimental::filesystem::exists( path2 )){
    std::experimental::filesystem::create_directories( path2 );
  }
  std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
  std::ofstream myfile;
  myfile.open( fname2, std::ios::app );
  myfile <<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
  myfile.close();

  //TODO: I should probably add a check that all the print to file are done in order, but cmon...



  // - send solution to following processor (unless last time-step)
  if ( _myRank < _numProcs-1 ){
    MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
  }

  // sum residuals at each time step
  resnorm*= resnorm;
  MPI_Allreduce( MPI_IN_PLACE, &resnorm, 1, MPI_DOUBLE, MPI_SUM, _comm );
  double finalResNorm = sqrt(resnorm);

  // average out iterations at each time step
  MPI_Allreduce( MPI_IN_PLACE, &GMRESit, 1, MPI_INT,    MPI_SUM, _comm );
  double avgGMRESit = double(GMRESit) / _numProcs;




  // OUTPUT -----------------------------------------------------------------
  // - save #it to convergence to file
  if (_myRank == 0){
    std::cout<<"Solver converged in "    <<avgGMRESit<<" average GMRES it per time-step.";
    std::cout<<" Final residual norm is "<<finalResNorm   <<".\n";
  
    double hmin, hmax, kmin, kmax;
    this->GetMeshSize( hmin, hmax, kmin, kmax );

    std::ofstream myfile;
    myfile.open( fname1, std::ios::app );
    myfile << _dt*_numProcs << ",\t" << _dt  << ",\t" << _numProcs   << ",\t"
           << hmax << ",\t" << hmin << ",\t" << refLvl << ",\t"
           << avgGMRESit << ",\t"  << finalResNorm  << std::endl;
    myfile.close();
  }

  // wait for write to be completed..possibly non-necessary
  MPI_Barrier(_comm);


  // clean up
  delete CC;
}























//-----------------------------------------------------------------------------
// Utils functions
//-----------------------------------------------------------------------------

// Returns vector containing the space-time exact solution (if available)
void IMHD2DSTOperatorAssembler::ExactSolution( HypreParVector*& u, HypreParVector*& p, HypreParVector*& z, HypreParVector*& a ) const{
  // Initialise handy functions
  VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
  FunctionCoefficient       pFuncCoeff(     _pFunc);
  FunctionCoefficient       zFuncCoeff(     _zFunc);
  FunctionCoefficient       aFuncCoeff(     _aFunc);
  // - specify evaluation time
  // -- notice first processor actually refers to instant dt
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  zFuncCoeff.SetTime( _dt*(_myRank+1) );
  aFuncCoeff.SetTime( _dt*(_myRank+1) );

  GridFunction uFun( _UhFESpace );
  GridFunction pFun( _PhFESpace );
  GridFunction zFun( _ZhFESpace );
  GridFunction aFun( _AhFESpace );

  uFun.ProjectCoefficient( uFuncCoeff );
  pFun.ProjectCoefficient( pFuncCoeff );
  zFun.ProjectCoefficient( zFuncCoeff );
  aFun.ProjectCoefficient( aFuncCoeff );
  

  Array<int> rowStartsU(2), rowStartsP(2), rowStartsZ(2), rowStartsA(2);
  rowStartsU[0] = ( uFun.Size() )*_myRank;
  rowStartsU[1] = ( uFun.Size() )*(_myRank+1);
  rowStartsP[0] = ( pFun.Size() )*_myRank;
  rowStartsP[1] = ( pFun.Size() )*(_myRank+1);
  rowStartsZ[0] = ( zFun.Size() )*_myRank;
  rowStartsZ[1] = ( zFun.Size() )*(_myRank+1);
  rowStartsA[0] = ( aFun.Size() )*_myRank;
  rowStartsA[1] = ( aFun.Size() )*(_myRank+1);

  u = new HypreParVector( _comm, (uFun.Size())*_numProcs, uFun.StealData(), rowStartsU.GetData() );
  p = new HypreParVector( _comm, (pFun.Size())*_numProcs, pFun.StealData(), rowStartsP.GetData() );
  z = new HypreParVector( _comm, (zFun.Size())*_numProcs, zFun.StealData(), rowStartsZ.GetData() );
  a = new HypreParVector( _comm, (aFun.Size())*_numProcs, aFun.StealData(), rowStartsA.GetData() );

  u->SetOwnership( 1 );
  p->SetOwnership( 1 );
  z->SetOwnership( 1 );
  a->SetOwnership( 1 );

}



// Each processor computes L2 error of solution at its time-step
void IMHD2DSTOperatorAssembler::ComputeL2Error( const Vector& uh, const Vector& ph, const Vector& zh, const Vector& ah,
                                                  double& err_u,    double& err_p,    double& err_z,    double& err_a ) const{

  const GridFunction u( _UhFESpace, uh.GetData() );
  const GridFunction p( _PhFESpace, ph.GetData() );
  const GridFunction z( _ZhFESpace, zh.GetData() );
  const GridFunction a( _AhFESpace, ah.GetData() );

  int order_quad = 5;
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i=0; i < Geometry::NumGeom; ++i){
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
  FunctionCoefficient       pFuncCoeff(     _pFunc);
  FunctionCoefficient       zFuncCoeff(     _zFunc);
  FunctionCoefficient       aFuncCoeff(     _aFunc);
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  zFuncCoeff.SetTime( _dt*(_myRank+1) );
  aFuncCoeff.SetTime( _dt*(_myRank+1) );


  err_u  = u.ComputeL2Error(uFuncCoeff, irs);
  err_p  = p.ComputeL2Error(pFuncCoeff, irs);
  err_z  = z.ComputeL2Error(zFuncCoeff, irs);
  err_a  = a.ComputeL2Error(aFuncCoeff, irs);

  // for ( int i = 0; i < _numProcs; ++i ){
  //   if ( _myRank == i ){
  //     std::cout << "Instant t="       << _dt*(_myRank+1) << std::endl;
  //     std::cout << "|| uh - uEx ||_L2= " << err_u << "\n";
  //     std::cout << "|| ph - pEx ||_L2= " << err_p << "\n";
  //     std::cout << "|| zh - zEx ||_L2= " << err_z << "\n";
  //     std::cout << "|| ah - aEx ||_L2= " << err_a << "\n";
  //   }
  //   MPI_Barrier( _comm );
  // }
}




void IMHD2DSTOperatorAssembler::SaveExactSolution( const std::string& path="ParaView",
                                                   const std::string& filename="STIMHD2D_Ex" ) const{
  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _UhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *zFun = new GridFunction( _ZhFESpace );
    GridFunction *aFun = new GridFunction( _AhFESpace );

    // set wpv paraview data file
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
    for ( int t = 0; t < _numProcs+1; ++t ){
      VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
      FunctionCoefficient       pFuncCoeff(     _pFunc);
      FunctionCoefficient       zFuncCoeff(     _zFunc);
      FunctionCoefficient       aFuncCoeff(     _aFunc);
      uFuncCoeff.SetTime( t*_dt );
      pFuncCoeff.SetTime( t*_dt );
      zFuncCoeff.SetTime( t*_dt );
      aFuncCoeff.SetTime( t*_dt );

      uFun->ProjectCoefficient( uFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      zFun->ProjectCoefficient( zFuncCoeff );
      aFun->ProjectCoefficient( aFuncCoeff );

      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete uFun;
    delete pFun;
    delete zFun;
    delete aFun;

  }
}

// store given approximate solution in paraview format
void IMHD2DSTOperatorAssembler::SaveSolution( const HypreParVector& uh, const HypreParVector& ph, const HypreParVector& zh, const HypreParVector& ah,
                                              const std::string& path="ParaView", const std::string& filename="STIMHD2D" ) const{
  
  // gather parallel vector
  Vector *uGlb = uh.GlobalVector();
  Vector *pGlb = ph.GlobalVector();
  Vector *zGlb = zh.GlobalVector();
  Vector *aGlb = ah.GlobalVector();


  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _UhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *zFun = new GridFunction( _ZhFESpace );
    GridFunction *aFun = new GridFunction( _AhFESpace );

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


    // store initial conditions
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    FunctionCoefficient       pFuncCoeff(     _pFunc);
    FunctionCoefficient       zFuncCoeff(     _zFunc);
    FunctionCoefficient       aFuncCoeff(     _aFunc);
    uFuncCoeff.SetTime( 0. );
    pFuncCoeff.SetTime( 0. );
    zFuncCoeff.SetTime( 0. );
    aFuncCoeff.SetTime( 0. );

    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    zFun->ProjectCoefficient( zFuncCoeff );
    aFun->ProjectCoefficient( aFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // handy variables for time loop
    const int blockSizeU = uh.Size();
    const int blockSizeP = ph.Size();
    const int blockSizeZ = zh.Size();
    const int blockSizeA = ah.Size();
    Vector uLcl,
           pLcl,
           zLcl,
           aLcl;
    Array<int> idxU(blockSizeU),
               idxP(blockSizeP),
               idxZ(blockSizeZ),
               idxA(blockSizeA);

    // main time loop
    for ( int t = 1; t < _numProcs+1; ++t ){
      // - identify correct sub-vector idx in global vectors
      for ( int i = 0; i < blockSizeU; ++i ){
        idxU[i] = blockSizeU*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeP; ++i ){
        idxP[i] = blockSizeP*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeZ; ++i ){
        idxZ[i] = blockSizeZ*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeA; ++i ){
        idxA[i] = blockSizeA*(t-1) + i;
      }

      // - extract subvector
      uGlb->GetSubVector( idxU, uLcl );
      pGlb->GetSubVector( idxP, pLcl );
      zGlb->GetSubVector( idxZ, zLcl );
      aGlb->GetSubVector( idxA, aLcl );
      
      // - assign to linked variables
      *uFun = uLcl;
      *pFun = pLcl;
      *zFun = zLcl;
      *aFun = aLcl;
      
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
}

// This function is the same as above, but it doesn't rely on HypreParVector's
void IMHD2DSTOperatorAssembler::SaveSolution( const Vector& uh, const Vector& ph, const Vector& zh, const Vector& ah,
                                              const std::string& path="ParaView",
                                              const std::string& filename="STIMHD2D" ) const{
  const int blockSizeU = uh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeZ = zh.Size();
  const int blockSizeA = ah.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( uh.GetData(), blockSizeU, MPI_DOUBLE, 0, 4*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 4*_myRank+1, _comm );
    MPI_Send( zh.GetData(), blockSizeZ, MPI_DOUBLE, 0, 4*_myRank+2, _comm );
    MPI_Send( ah.GetData(), blockSizeA, MPI_DOUBLE, 0, 4*_myRank+3, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _UhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *zFun = new GridFunction( _ZhFESpace );
    GridFunction *aFun = new GridFunction( _AhFESpace );

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


    // store initial conditions
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    FunctionCoefficient       pFuncCoeff(     _pFunc);
    FunctionCoefficient       zFuncCoeff(     _zFunc);
    FunctionCoefficient       aFuncCoeff(     _aFunc);
    uFuncCoeff.SetTime( 0.0 );
    pFuncCoeff.SetTime( 0.0 );
    zFuncCoeff.SetTime( 0.0 );
    aFuncCoeff.SetTime( 0.0 );
    
    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    zFun->ProjectCoefficient( zFuncCoeff );
    aFun->ProjectCoefficient( aFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // this will store the approximate solution at current time-step
    Vector uLcl(blockSizeU),
           pLcl(blockSizeP),
           zLcl(blockSizeP),
           aLcl(blockSizeA);

    // handle first time-step separately
    *uFun = uh;
    *pFun = ph;
    *zFun = zh;
    *aFun = ah;
    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( uLcl.GetData(), blockSizeU, MPI_DOUBLE, t-1, 4*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 4*(t-1)+1, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( zLcl.GetData(), blockSizeZ, MPI_DOUBLE, t-1, 4*(t-1)+2, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( aLcl.GetData(), blockSizeA, MPI_DOUBLE, t-1, 4*(t-1)+3, _comm, MPI_STATUS_IGNORE );

      // - assign to linked variables
      *uFun = uLcl;
      *pFun = pLcl;
      *zFun = zLcl;
      *aFun = aLcl;
      
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
}





// Saves a plot of the error
void IMHD2DSTOperatorAssembler::SaveError( const Vector& uh, const Vector& ph, const Vector& zh, const Vector& ah,
                                           const std::string& path="ParaView",
                                           const std::string& filename="STIMHD2D_err" ) const{
  const int blockSizeU = uh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeZ = zh.Size();
  const int blockSizeA = ah.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( uh.GetData(), blockSizeU, MPI_DOUBLE, 0, 4*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 4*_myRank+1, _comm );
    MPI_Send( zh.GetData(), blockSizeZ, MPI_DOUBLE, 0, 4*_myRank+2, _comm );
    MPI_Send( ah.GetData(), blockSizeA, MPI_DOUBLE, 0, 4*_myRank+3, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *uFun = new GridFunction( _UhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *zFun = new GridFunction( _ZhFESpace );
    GridFunction *aFun = new GridFunction( _AhFESpace );

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

    // this will store the approximate solution at current time-step
    Vector uLcl(blockSizeU),
           pLcl(blockSizeP),
           zLcl(blockSizeP),
           aLcl(blockSizeA);

    // these will provide exact solution
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    FunctionCoefficient       pFuncCoeff(     _pFunc);
    FunctionCoefficient       zFuncCoeff(     _zFunc);
    FunctionCoefficient       aFuncCoeff(     _aFunc);

    // error at instant 0 is 0 (IC)
    *uFun = 0.;
    *pFun = 0.;
    *zFun = 0.;
    *aFun = 0.;
    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();

    // handle first time-step separately
    uFuncCoeff.SetTime( _dt );
    pFuncCoeff.SetTime( _dt );
    zFuncCoeff.SetTime( _dt );
    aFuncCoeff.SetTime( _dt );
    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    zFun->ProjectCoefficient( zFuncCoeff );
    aFun->ProjectCoefficient( aFuncCoeff );

    uFun->operator-=( uh );
    pFun->operator-=( ph );
    zFun->operator-=( zh );
    aFun->operator-=( ah );

    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( uLcl.GetData(), blockSizeU, MPI_DOUBLE, t-1, 4*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 4*(t-1)+1, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( zLcl.GetData(), blockSizeZ, MPI_DOUBLE, t-1, 4*(t-1)+2, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( aLcl.GetData(), blockSizeA, MPI_DOUBLE, t-1, 4*(t-1)+3, _comm, MPI_STATUS_IGNORE );
      
      // - assign to linked variables
      uFuncCoeff.SetTime( _dt*t );
      pFuncCoeff.SetTime( _dt*t );
      zFuncCoeff.SetTime( _dt*t );
      aFuncCoeff.SetTime( _dt*t );
      uFun->ProjectCoefficient( uFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      zFun->ProjectCoefficient( zFuncCoeff );
      aFun->ProjectCoefficient( aFuncCoeff );
      uFun->operator-=( uLcl );
      pFun->operator-=( pLcl );
      zFun->operator-=( zLcl );
      aFun->operator-=( aLcl );
      
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
}










void IMHD2DSTOperatorAssembler::GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const{
  if(_mesh == NULL)
    std::cerr<<"Mesh not yet set"<<std::endl;
  else
    _mesh->GetCharacteristics( h_min, h_max, k_min, k_max );
}




void IMHD2DSTOperatorAssembler::PrintMatrices( const std::string& filename ) const{

  if( ! ( _FuAssembled       && _MuAssembled
       && _MzAssembled
       && _FaAssembled       && _MaAssembled
       && _BAssembled        && _Z1Assembled && _Z2Assembled
       && _KAssembled        && _YAssembled 
       && _MpAssembled       && _ApAssembled && _WpAssembled
       && _MaNoZeroAssembled && _AaAssembled && _WaAssembled && _dtuWaAssembled && _CCainvAssembled && _MaNoZeroLumpedAssembled ) ){
    if( _myRank == 0){
        std::cerr<<"Make sure all matrices have been initialised, otherwise they can't be printed"<<std::endl;
    }
    return;
  }

  std::string myfilename;
  std::ofstream myfile;
  myfile.precision(std::numeric_limits< double >::max_digits10);

  if ( _myRank == 0 ){
    myfilename = filename + "_Mu.dat";
    myfile.open( myfilename );
    _Mu.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Mz.dat";
    myfile.open( myfilename );
    _Mz.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Ma.dat";
    myfile.open( myfilename );
    _Ma.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_B.dat";
    myfile.open( myfilename );
    _B.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_K.dat";
    myfile.open( myfilename );
    _K.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Mp.dat";
    myfile.open( myfilename );
    _Mp.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Ap.dat";
    myfile.open( myfilename );
    _Ap.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_MaNZ.dat";
    myfile.open( myfilename );
    _MaNoZero.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_MaNZL.dat";
    myfile.open( myfilename );
    _MaNoZeroLumped.PrintMatlab(myfile);
    myfile.close( );

    myfilename = filename + "_Aa.dat";
    myfile.open( myfilename );
    _Aa.PrintMatlab(myfile);
    myfile.close( );


    // Dirichlet nodes
    myfilename = filename + "essU.dat";
    myfile.open( myfilename );
    _essUhTDOF.Print(myfile,1);
    myfile.close( );

    myfilename = filename + "essP.dat";
    myfile.open( myfilename );
    _essPhTDOF.Print(myfile,1);
    myfile.close( );

    // myfilename = filename + "essZ.dat";
    // myfile.open( myfilename );
    // _essZhTDOF.Print(myfile,1);
    // myfile.close( );

    myfilename = filename + "essA.dat";
    myfile.open( myfilename );
    _essAhTDOF.Print(myfile,1);
    myfile.close( );

  }

  myfilename = filename + "_Fu_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Fu.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Fa_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Fa.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Z1_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Z1.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Z2_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Z2.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Y_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Y.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Wp_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Wp.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Wa_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Wa.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_dtuWa_" + std::to_string(_myRank) + ".dat";
  myfile.open( myfilename );
  _dtuWa.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Cp_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Cp.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_C0_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _C0.PrintMatlab(myfile);
  myfile.close( );

  myfilename = filename + "_Cm_" + std::to_string(_myRank) +".dat";
  myfile.open( myfilename );
  _Cm.PrintMatlab(myfile);
  myfile.close( );

}





inline void IMHD2DSTOperatorAssembler::SetEverythingUnassembled(){
  _MuAssembled = false;
  _FuAssembled = false;
  _MzAssembled = false;
  _MaAssembled = false;
  _FaAssembled = false;
  _BAssembled = false;
  _Z1Assembled = false;
  _Z2Assembled = false;
  _KAssembled = false;
  _YAssembled = false;
  _MpAssembled = false;
  _ApAssembled = false;
  _WpAssembled = false;
  _AaAssembled = false;
  _CpAssembled = false;
  _C0Assembled = false;
  _CmAssembled = false;
  _MaNoZeroAssembled = false;
  _MaNoZeroLumpedAssembled = false;
  _WaAssembled = false;
  _dtuWaAssembled = false;
  _FFuAssembled = false;
  _MMzAssembled = false;
  _FFaAssembled = false;
  _BBAssembled = false;
  _ZZ1Assembled = false;
  _ZZ2Assembled = false;
  _YYAssembled = false;
  _KKAssembled = false;
  _pSAssembled = false;
  _aSAssembled = false;
  _FFuinvAssembled = false;
  _MMzinvAssembled = false;
  _CCainvAssembled = false;

  _FFuinv = NULL;
  _FFuinvPrec = NULL;
  _MMzinv = NULL;
  _CCainv = NULL;
  _CCainvPrec = NULL;
}









/*

// Actual Time-stepper
void IMHD2DSTOperatorAssembler::TimeStep( const BlockVector& x, BlockVector& y,
                                          const std::string &fname1, const std::string &path2, int refLvl ){
  TODO
  // Define operator
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = _Fu.NumRows();
  offsets[2] =  _B.NumRows();
  offsets.PartialSum();
  PetscParMatrix myB( &_B );

  BlockOperator stokesOp( offsets );
  stokesOp.SetBlock(0, 0, &_Fu );
  stokesOp.SetBlock(0, 1, myB.Transpose() );
  stokesOp.SetBlock(1, 0, &_B  );


  // Define preconditioner
  // - inverse of pressure Schur complement (bottom-right block in precon) - reuse code from ST case, but with single processor
  StokesSTPreconditioner myPSchur( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essQhTDOF, _verbose );
  AssembleAp();
  AssembleMp();
  AssembleWp();
  myPSchur.SetAp( &_Ap );
  myPSchur.SetMp( &_Mp );
  if( _Pe != 0. ){
    myPSchur.SetWp( &_Wp, false );   // if there is convection, then clearly Wp differs from Ap (must include pressure convection)
  }else if( _essQhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
    myPSchur.SetWp( &_Wp, true );    
  }else{
    // _pSchur->SetWp( &_Wp, false );
    myPSchur.SetWp( &_Wp, true );     // should be false, according to E/S/W!
    if( _myRank == 0 ){
      std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
               <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
               <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
    }
  }

  // - inverse of velocity operator (top-left block in precon)
  const PetscParMatrix myFu( &_Fu );
  PetscLinearSolver Fsolve( myFu, "VSolver_" );

  // - assemble precon
  BlockUpperTriangularPreconditioner stokesPr(offsets);
  stokesPr.iterative_mode = false;
  stokesPr.SetDiagonalBlock( 0, &Fsolve );
  stokesPr.SetDiagonalBlock( 1, &myPSchur );
  stokesPr.SetBlock( 0, 1, myB.Transpose() );


  // Define solver
  PetscLinearSolver solver( MPI_COMM_SELF, "solver_" );
  bool isIterative = true;
  solver.iterative_mode = isIterative;
  solver.SetPreconditioner(stokesPr);
  solver.SetOperator(stokesOp);

  double tol = 1e-10 / sqrt( _numProcs );
  if( _myRank == 0 ){
    std::cout<<"Warning: Considering a fixed overall tolerance of 1e-10, scaled by the number of time steps."<<std::endl
             <<"          This option gets overwritten if a tolerance is prescribed in the petsc option file,"<<std::endl    
             <<"          so make sure to delete it from there!"<<std::endl;    
  }
  solver.SetTol(tol);
  solver.SetRelTol(tol);
  solver.SetAbsTol(tol);


  // Main "time-stepping" routine
  const int totDofs = x.Size();
  // - for each time-step, this will contain rhs
  BlockVector b( offsets );
  b = 0.0;
  
  // - receive solution from previous processor (unless initial time-step)
  if ( _myRank > 0 ){
    // - use it as initial guess for the solver (so store in y), unless initial step!
    MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
    // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
    _Mu.Mult( y.GetBlock(0), b.GetBlock(0) );
    // - M is stored with negative sign for velocity, so flip it
    b.GetBlock(0).Neg();
  }

  // - define rhs for this step (includes contribution from sol at previous time-step
  b += x;

  // - solve for current time-step
  //  --y acts as an initial guess! So use prev sol, unless first time step
  solver.Mult( b, y );

  int GMRESit    = solver.GetNumIterations();
  double resnorm = solver.GetFinalNorm();
  std::cout<<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

  if (!std::experimental::filesystem::exists( path2 )){
    std::experimental::filesystem::create_directories( path2 );
  }
  std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
  std::ofstream myfile;
  myfile.open( fname2, std::ios::app );
  myfile <<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
  myfile.close();

  //TODO: I should probably add a check that all the print to file are done in order, but cmon...



  // - send solution to following processor (unless last time-step)
  if ( _myRank < _numProcs-1 ){
    MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
  }

  // sum residuals at each time step
  resnorm*= resnorm;
  MPI_Allreduce( MPI_IN_PLACE, &resnorm, 1, MPI_DOUBLE, MPI_SUM, _comm );
  double finalResNorm = sqrt(resnorm);

  // average out iterations at each time step
  MPI_Allreduce( MPI_IN_PLACE, &GMRESit, 1, MPI_INT,    MPI_SUM, _comm );
  double avgGMRESit = double(GMRESit) / _numProcs;




  // OUTPUT -----------------------------------------------------------------
  // - save #it to convergence to file
  if (_myRank == 0){
    std::cout<<"Solver converged in "    <<avgGMRESit<<" average GMRES it per time-step.";
    std::cout<<" Final residual norm is "<<finalResNorm   <<".\n";
  
    double hmin, hmax, kmin, kmax;
    this->GetMeshSize( hmin, hmax, kmin, kmax );

    std::ofstream myfile;
    myfile.open( fname1, std::ios::app );
    myfile << _dt*_numProcs << ",\t" << _dt  << ",\t" << _numProcs   << ",\t"
           << hmax << ",\t" << hmin << ",\t" << refLvl << ",\t"
           << avgGMRESit << ",\t"  << finalResNorm  << std::endl;
    myfile.close();
  }

  // wait for write to be completed..possibly non-necessary
  MPI_Barrier(_comm);

}
















// Optimised Time-stepper (only triggers one solve per time-step if necessary, otherwise master takes care of solving everything)
void IMHD2DSTOperatorAssembler::TimeStep( const BlockVector& x, BlockVector& y,
                                          const std::string &fname1, const std::string &path2, int refLvl, int pbType ){

  TODO
  // Define operator
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = _Fu.NumRows();
  offsets[2] =  _B.NumRows();
  offsets.PartialSum();
  PetscParMatrix myB( &_B );

  BlockOperator stokesOp( offsets );
  stokesOp.SetBlock(0, 0, &_Fu );
  stokesOp.SetBlock(0, 1, myB.Transpose() );
  stokesOp.SetBlock(1, 0, &_B  );

  BlockUpperTriangularPreconditioner* stokesPr = NULL;
  StokesSTPreconditioner* myPSchur = NULL;

  PetscParMatrix* myFu = NULL;
  PetscLinearSolver* Fsolve = NULL;
  PetscLinearSolver* solver = NULL;

  const int totDofs = x.Size();
  double finalResNorm = 0.0;
  double avgGMRESit   = 0.0;


  // Define operators
  if ( pbType == 4 || _myRank == 0 ){
    // Define preconditioner
    // - inverse of pressure Schur complement (bottom-right block in precon) - reuse code from ST case, but with single processor
    myPSchur = new StokesSTPreconditioner( MPI_COMM_SELF, _dt, _mu, NULL, NULL, NULL, _essQhTDOF, _verbose );
    AssembleAp();
    AssembleMp();
    AssembleWp();
    myPSchur->SetAp( &_Ap );
    myPSchur->SetMp( &_Mp );
    if( _Pe != 0. ){
      myPSchur->SetWp( &_Wp, false );   // if there is convection, then clearly Wp differs from Ap (must include pressure convection)
    }else if( _essQhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
      myPSchur->SetWp( &_Wp, true );    
    }else{
      // _pSchur->SetWp( &_Wp, false );
      myPSchur->SetWp( &_Wp, true );     // should be false, according to E/S/W!
      if( _myRank == 0 ){
        std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
                 <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
                 <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
      }
    }

    // - inverse of velocity operator (top-left block in precon)
    myFu   = new PetscParMatrix( &_Fu );
    Fsolve = new PetscLinearSolver( *myFu, "VSolver_" );

    // - assemble precon
    stokesPr = new BlockUpperTriangularPreconditioner(offsets);
    stokesPr->iterative_mode = false;
    stokesPr->SetDiagonalBlock( 0, Fsolve );
    stokesPr->SetDiagonalBlock( 1, myPSchur );
    stokesPr->SetBlock( 0, 1, myB.Transpose() );


    // Define solver
    solver = new PetscLinearSolver( MPI_COMM_SELF, "solver_" );
    bool isIterative = true;
    solver->iterative_mode = isIterative;
    solver->SetPreconditioner(*stokesPr);
    solver->SetOperator(stokesOp);

    double tol = 1e-10 / sqrt( _numProcs );
    if( _myRank == 0 ){
      std::cout<<"Warning: Considering a fixed overall tolerance of 1e-10, scaled by the number of time steps."<<std::endl
               <<"          This option gets overwritten if a tolerance is prescribed in the petsc option file,"<<std::endl    
               <<"          so make sure to delete it from there!"<<std::endl;    
    }
    solver->SetTol(tol);
    solver->SetRelTol(tol);
    solver->SetAbsTol(tol);
  }



  // Main "time-stepping" routine
  // - if each operator is different, each time step needs to solve its own
  if ( pbType == 4 ){
    // - for each time-step, this will contain rhs
    BlockVector b( offsets );
    b = 0.0;
    
    // - receive solution from previous processor (unless initial time-step)
    if ( _myRank > 0 ){
      // - use it as initial guess for the solver (so store in y), unless initial step!
      MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, _myRank-1, _myRank, _comm, MPI_STATUS_IGNORE );
      // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
      _Mu.Mult( y.GetBlock(0), b.GetBlock(0) );
      // - M is stored with negative sign for velocity, so flip it
      b.GetBlock(0).Neg();
    }

    // - define rhs for this step (includes contribution from sol at previous time-step
    b += x;

    // - solve for current time-step
    //  --y acts as an initial guess! So use prev sol, unless first time step
    solver->Mult( b, y );

    int GMRESit    = solver->GetNumIterations();
    double resnorm = solver->GetFinalNorm();
    std::cout<<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

    if (!std::experimental::filesystem::exists( path2 )){
      std::experimental::filesystem::create_directories( path2 );
    }
    std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
    std::ofstream myfile;
    myfile.open( fname2, std::ios::app );
    myfile <<"Solved for time-step "<<_myRank+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
    myfile.close();

    // - send solution to following processor (unless last time-step)
    if ( _myRank < _numProcs-1 ){
      MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, _myRank+1, _myRank+1, _comm );
    }

    // sum residuals at each time step
    resnorm*= resnorm;
    MPI_Allreduce( MPI_IN_PLACE, &resnorm, 1, MPI_DOUBLE, MPI_SUM, _comm );
    finalResNorm = sqrt(resnorm);

    // average out iterations at each time step
    MPI_Allreduce( MPI_IN_PLACE, &GMRESit, 1, MPI_INT,    MPI_SUM, _comm );
    avgGMRESit = double(GMRESit) / _numProcs;


  // - otherwise, master takes care of it all
  }else{
    if ( _myRank == 0 ){

      // first time-step is a bit special
      solver->Mult( x, y );
      int GMRESit    = solver->GetNumIterations();
      double resnorm = solver->GetFinalNorm();

      avgGMRESit   += GMRESit;
      finalResNorm += resnorm*resnorm;

      std::cout<<"Solved for time-step "<<1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

      if (!std::experimental::filesystem::exists( path2 )){
        std::experimental::filesystem::create_directories( path2 );
      }
      std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
      std::ofstream myfile;
      myfile.open( fname2, std::ios::app );
      myfile <<"Solved for time-step "<<1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
      myfile.close();

      // this will keep track of sol at current time-step
      BlockVector y0 = y;

      // after first time-step, you need to
      for ( int t = 1; t < _numProcs; ++t ){
        // - this will contain rhs
        BlockVector b( offsets ), temp( offsets );
        b = 0.0; temp = 0.0;

        MPI_Recv( b.GetData(), totDofs, MPI_DOUBLE, t, 2*t,   _comm, MPI_STATUS_IGNORE );
        // nah, use prev sol as IG
        // MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, t, 3*t+1, _comm, MPI_STATUS_IGNORE );

        // - M should be the same for every proc, so it doesn't really matter which one performs the multiplication
        _Mu.Mult( y0.GetBlock(0), temp.GetBlock(0) );
        // - M is stored with negative sign for velocity, so flip it
        b -= temp;
    
        // - solve for current time-step
        //  --y acts as an initial guess! So use prev sol
        solver->Mult( b, y0 );

        int GMRESit    = solver->GetNumIterations();
        double resnorm = solver->GetFinalNorm();

        avgGMRESit   += GMRESit;
        finalResNorm += resnorm*resnorm;

        std::cout<<"Solved for time-step "<<t+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;

        std::string fname2 = path2 + "NP" + std::to_string(_numProcs) + "_r"  + std::to_string(refLvl) + ".txt";
        std::ofstream myfile;
        myfile.open( fname2, std::ios::app );
        myfile <<"Solved for time-step "<<t+1<<" in "<<GMRESit<<" iterations. Residual "<<resnorm<<std::endl;
        myfile.close();

        // - send solution to right processor
        MPI_Send( y0.GetData(), totDofs, MPI_DOUBLE, t, 2*t+1, _comm );
      }

      avgGMRESit = double(avgGMRESit)/_numProcs;
      finalResNorm = sqrt(finalResNorm);

    }else{
      // - send rhs to master
      MPI_Send( x.GetData(), totDofs, MPI_DOUBLE, 0, 2*_myRank,   _comm );
      // - send IG to master
      // MPI_Send( y.GetData(), totDofs, MPI_DOUBLE, 0, 3*_myRank+1, _comm );
      // - receive solution from master
      MPI_Recv( y.GetData(), totDofs, MPI_DOUBLE, 0, 2*_myRank+1, _comm, MPI_STATUS_IGNORE );
    }
  }





  // OUTPUT -----------------------------------------------------------------
  // - save #it to convergence to file
  if (_myRank == 0){
    std::cout<<"Solver converged in "    <<avgGMRESit<<" average GMRES it per time-step.";
    std::cout<<" Final residual norm is "<<finalResNorm   <<".\n";
  
    double hmin, hmax, kmin, kmax;
    this->GetMeshSize( hmin, hmax, kmin, kmax );

    std::ofstream myfile;
    myfile.open( fname1, std::ios::app );
    myfile << _dt*_numProcs << ",\t" << _dt  << ",\t" << _numProcs   << ",\t"
           << hmax << ",\t" << hmin << ",\t" << refLvl << ",\t"
           << avgGMRESit << ",\t"  << finalResNorm  << std::endl;
    myfile.close();
  }

  // wait for write to be completed..possibly non-necessary
  MPI_Barrier(_comm);


  // clean up
  delete stokesPr;
  delete myPSchur;
  delete myFu;
  delete Fsolve;
  delete solver;

}

*/
















IMHD2DSTOperatorAssembler::~IMHD2DSTOperatorAssembler(){
  delete _pSinv;
  delete _aSinv;
  delete _FFuinv;
  delete _FFuinvPrec;
  delete _CCainv;
  delete _CCainvPrec;
  delete _Mztemp;
  delete _MMzinv;
  // delete _FFFu;
  // delete _MMMz;
  // delete _FFFa;
  // if( _FFuAssembled )
  //   HYPRE_IJMatrixDestroy( _FFu );
  // if( _FFaAssembled )
  //   HYPRE_IJMatrixDestroy( _FFa );
  // if( _MMzAssembled )
  //   HYPRE_IJMatrixDestroy( _MMz );
  // if( _BBAssembled )
  //   HYPRE_IJMatrixDestroy( _BB );
  // if( _ZZ1Assembled )
  //   HYPRE_IJMatrixDestroy( _ZZ1 );
  // if( _ZZ2Assembled )
  //   HYPRE_IJMatrixDestroy( _ZZ2 );
  // if( _KKAssembled )
  //   HYPRE_IJMatrixDestroy( _KK );
  // if( _YYAssembled )
  //   HYPRE_IJMatrixDestroy( _YY );

  delete _UhFESpace;
  delete _PhFESpace;
  delete _ZhFESpace;
  delete _AhFESpace;
  delete _UhFEColl;
  delete _PhFEColl;
  delete _ZhFEColl;
  delete _AhFEColl;

  delete _mesh;
}


























