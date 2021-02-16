#include "imhd2dstoperatorassembler.hpp"
#include "vectorconvectionintegrator.hpp"
#include "blockUpperTriangularPreconditioner.hpp"
#include "spacetimesolver.hpp"

#include <mpi.h>
#include <string>
#include <cstring>
#include <iostream>
#include "HYPRE.h"
#include "petsc.h"
#include "mfem.hpp"
#include <experimental/filesystem>

using namespace mfem;

// Seems like multiplying every operator by dt gives slightly better results.
#define MULT_BY_DT









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
                                                      void(  *m)(const Vector &, double ),
                                                      void(  *w)(const Vector &, double, Vector &),
                                                      double(*y)(const Vector &, double ),
                                                      double(*c)(const Vector &, double ),
                                                      void(  *u)(const Vector &, double, Vector &),
                                                      double(*p)(const Vector &, double ),
                                                      double(*z)(const Vector &, double ),
                                                      double(*a)(const Vector &, double ),
                                                      int verbose ):
  _comm(comm), _dt(dt),
  _mu(mu),     _eta(eta),                _mu0(mu0), 
  _fFunc(f),   _gFunc(g),                _hFunc(h), 
  _nFunc(n),                             _mFunc(m), 
  _wFunc(w),                  _yFunc(y), _cFunc(c),
  _uFunc(u),   _pFunc(p),     _zFunc(z), _aFunc(a),
  _ordU(ordU), _ordP(ordP), _ordZ(ordZ), _ordA(ordA),
  _verbose(verbose){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

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


  if ( _mesh->bdr_attributes.Size() > 0 ) {
    Array<int> essBdrU( _mesh->bdr_attributes.Max() ), essBdrP( _mesh->bdr_attributes.Max() ),
               essBdrZ( _mesh->bdr_attributes.Max() ), essBdrA( _mesh->bdr_attributes.Max() );
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

    _UhFESpace->GetEssentialTrueDofs( essBdrU, _essUhTDOF );
    _PhFESpace->GetEssentialTrueDofs( essBdrP, _essPhTDOF );
    _ZhFESpace->GetEssentialTrueDofs( essBdrZ, _essZhTDOF );
    _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );
  }


  if (_myRank == 0 ){
    std::cout << "***********************************************************\n";
    std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
    std::cout << "***********************************************************\n";
  }

}



// constructor (uses vector of node values to initialise linearised fields)
IMHD2DSTOperatorAssembler::IMHD2DSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
                                                      const int refLvl, const int ordU, const int ordP, const int ordZ, const int ordA,
                                                      const double dt, const double mu, const double eta, const double mu0,
                                                      void(  *f)(const Vector &, double, Vector &),
                                                      double(*g)(const Vector &, double ),
                                                      double(*h)(const Vector &, double ),
                                                      void(  *n)(const Vector &, double, Vector &),
                                                      void(  *m)(const Vector &, double ),
                                                      const Vector& w,
                                                      const Vector& y,
                                                      const Vector& c,
                                                      void(  *u)(const Vector &, double, Vector &),
                                                      double(*p)(const Vector &, double ),
                                                      double(*z)(const Vector &, double ),
                                                      double(*a)(const Vector &, double ),
                                                      int verbose ):
  _wFuncCoeff(w), _yFuncCoeff(y), _cFuncCoeff(c),
  IMHD2DSTOperatorAssembler(  comm, meshName, refLvl, ordU, ordP, ordZ, ordA, dt, mu, eta, mu0,
                              f, g, h, n, m, NULL, NULL, NULL, u, p, z, a, verbose ) {}










//-----------------------------------------------------------------------------
// Assemble operators for single time-steps
//-----------------------------------------------------------------------------

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




// Assemble operator on subdiagonal of space-time matrix for u block (negative mass matrix):
//  Mu <-> - (u,v)
void IMHD2DSTOperatorAssembler::AssembleMu( ){
  if( _MuAssembled ){
    return;
  }

  BilinearForm muVarf(_UhFESpace);
#ifdef MULT_BY_DT
  // ConstantCoefficient mone( -1.0 );
  muVarf.AddDomainIntegrator(new VectorMassIntegrator( -1.0 ));
#else
  // ConstantCoefficient mdtinv( -1./_dt );
  muVarf.AddDomainIntegrator(new VectorMassIntegrator( -1./_dt ));
#endif
  muVarf.Assemble();
  muVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Mu = muVarf.SpMat();
  _Mu.SetGraphOwner(true);
  _Mu.SetDataOwner(true);
  muVarf.LoseMat();

  // still need BC!
  // _MuAssembled = true;




  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass-matrix for u (negative) Mu assembled\n";
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




// Assemble operator on subdiagonal of space-time matrix for A block (negative mass matrix):
//  Ma <-> - (a,b)
void IMHD2DSTOperatorAssembler::AssembleMa( ){
  if( _MaAssembled ){
    return;
  }

  BilinearForm maVarf(_AhFESpace);
#ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  maVarf.AddDomainIntegrator(new MassIntegrator( mone ));
#else
  ConstantCoefficient mdtinv( -1./_dt );
  maVarf.AddDomainIntegrator(new MassIntegrator( mdtinv ));
#endif
  maVarf.Assemble();
  maVarf.Finalize();

  // - once the matrix is generated, we can get rid of the operator
  _Ma = maVarf.SpMat();
  _Ma.SetGraphOwner(true);
  _Ma.SetDataOwner(true);
  maVarf.LoseMat();

  // still need BC!
  // _MvAssembled = true;



  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass matrix for A Ma assembled\n";
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





// Operators for preconditioners ----------------------------------------------

// Assemble pressure mass matrix
//  Mp <->  (p,q)
void IMHD2DSTOperatorAssembler::AssembleMp( ){
  if( _MpAssembled ){
    return;
  }

  BilinearForm mVarf( new BilinearForm(_PhFESpace) );
  ConstantCoefficient one( 1.0 );
  mVarf.AddDomainIntegrator(new MassIntegrator( one ));
  mVarf.Assemble();
  mVarf.Finalize();

  // - impose dirichlet BC on outflow
  mVarf.FormSystemMatrix( _essPhTDOF, _Mp );
  // _Mp = mVarf->SpMat();

  // - once the matrix is generated, we can get rid of the operator
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
void StokesSTOperatorAssembler::AssembleAp( ){

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
  // Impose homogeneous dirichlet BC by simply removing corresponding equations
  aVarf.Assemble();
  aVarf.Finalize();
  
  aVarf.FormSystemMatrix( _essPhTDOF, _Ap );
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf.LoseMat();


  delete aVarf;

  _ApAssembled = true;

  if( _essQhTDOF.Size() == 0 ){
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
void StokesSTOperatorAssembler::AssembleWp( ){

  if( _WpAssembled ){
    return;
  }

  if ( _myRank == 0 ){
    std::cout<<"Warning: The assembly of the spatial part of the PCD considers only Neumann BC on pressure."<<std::endl
             <<"          This conflicts with the definition of the other pressure operators (which include"<<std::endl
             <<"          Dirichlet BC on outflow). For this to make sense, make sure that either:"         <<std::endl
             <<"          - Spatial part of Fp and Ap are forcibly imposed equal (which bypasses this func)"<<std::endl
             <<"          - There is no outflow (in which case they would have Neumann everywhere anyway)"  <<std::endl
             <<"         Moreover, if solving Oseen, we further need to impose that the prescribed"         <<std::endl
             <<"          advection field is tangential to the boundary (enclosed flow, w*n=0)."            <<std::endl;
  }


  BilinearForm wVarf( _PhFESpace );
  ConstantCoefficient mu( _mu );
  wVarfAddDomainIntegrator(new DiffusionIntegrator( mu ));
  if ( _ordP == 0 ){  // DG
    double sigma = -1.0;
    double kappa =  1.0;
    wVarfAddInteriorFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
    // wVarf->AddBdrFaceIntegrator(     new DGDiffusionIntegrator(one, sigma, kappa));  // to weakly impose Dirichlet BC - don't bother for now
  }

  // include convection
  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for ConvectionIntegrator
  GridFunction wGridFun( _UhFESpace );
  wGridFun = _wFuncCoeff;
  if ( _wFunc == NULL ){
    wCoeff = new VectorGridFunctionCoefficient( &wGridFun );
  }else{
    wCoeff = new VectorFunctionCoefficient( _dim, _wFunc );
    wCoeff->SetTime( _dt*(_myRank+1) );
  }
  // TODO: should I impose Robin, then? Like this I'm still applying Neumann
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, 1.0 ));  // if used for NS, make sure both _mu*_Pe=1.0!!

  // // This includes Robin -> can't be bothered to implement it / test it: just pick a w: w*n = 0 on the bdr in your tests
  // if( _ordP == 0 ){
  //   // Counteract the n.Grad(u) term on the Dirichlet portion of the boundary
  //   double sigma = -1.0;
  //   double kappa =  1.0;
  //   wVarf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(mu, sigma, kappa));
  //   // Augment the n.Grad(u) term with a*u on the Robin portion of boundary
  //   wVarf->AddBdrFaceIntegrator(new BoundaryMassIntegrator(wFuncCoeff, _mu*_Pe));  //this won't work: I need to compute w*n!
  // }else{
  //   wVarf->AddBoundaryIntegrator(new MassIntegrator(wCoeff, _mu*_Pe) );
  // }
  
  wVarf.Assemble();
  wVarf.Finalize();
  

  _Wp = wVarf.SpMat();
  _Wp.SetGraphOwner(true);
  _Wp.SetDataOwner(true);
  wVarf.LoseMat();

  delete wCoeff;

  _WpAssembled = true;


  if ( _verbose>50 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Wp_" + to_string(_myRank) + ".dat";
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

  // - impose dirichlet BC
  mVarf.FormSystemMatrix( _essAhTDOF, _MaNoZero );
  // _Mp = mVarf->SpMat();

  // - once the matrix is generated, we can get rid of the operator
  _MaNoZero.SetGraphOwner(true);
  _MaNoZero.SetDataOwner(true);
  mVarf.LoseMat();

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
      std::cout<<"Mass matrix for A Ma (without zeros for dir BC) assembled\n";
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
  VectorCoefficient* wCoeff = NULL;   // need to define them here otherwise they go out of scope for ConvectionIntegrator
  GridFunction wGridFun( _UhFESpace );
  wGridFun = _wFuncCoeff;
  if ( _wFunc == NULL ){
    wCoeff = new VectorGridFunctionCoefficient( &wGridFun );
  }else{
    wCoeff = new VectorFunctionCoefficient( _dim, _wFunc );
    wCoeff->SetTime( _dt*(_myRank+1) );
  }

  ConstantCoefficient one( 1.0 );
  wVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, one ));

  // wVarf.AddDomainIntegrator(new ConvectionIntegrator( *wCoeff, 1.0 ));

  
  wVarf.Assemble();
  wVarf.Finalize();
  
  // Include dirichlet BC
  wVarf.FormSystemMatrix( _essAhTDOF, _Wa );
  // - once the matrix is generated, we can get rid of the operator
  _Wa.SetGraphOwner(true);
  _Wa.SetDataOwner(true);
  wVarf.LoseMat();

  delete wCoeff;


  _WaAssembled = true;


  if ( _verbose>50 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Wa_" + to_string(_myRank) + ".dat";
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
void IMHD2DSTOperatorAssembler::AssembleCs( const int discType ){

  // I'm gonna need a mass matrix and a stiffness matrix regardless
  // - mass
  AssembleMaNoZero();
  // - stiffness
  BilinearForm aVarf( new BilinearForm(_AhFESpace) );
  ConstantCoefficient one( 1.0 );
  aVarf.AddDomainIntegrator(  new DiffusionIntegrator( one ));
  aVarf.Assemble();
  aVarf.Finalize();
  SparseMatrix Aa = aVarf.SpMat();
  Aa.SetGraphOwner(true);
  Aa.SetDataOwner(true);
  aVarf.LoseMat();
  // - for dirichlet BC (used later)
  mfem::Array<int> colsA(Aa.Height()) = 0.;
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }


  // Now I need to compute ||B_0||_2, the L2 norm of the space-time average of the magnetic
  //  field B = ∇x(kA)
  // - first, define a GridFunction representing A (so that I can use its GetGradient method
  //    to get its derivatives)
  GridFunction cGridFun( _AhFESpace );
  if ( _cFunc == NULL ){
    cGridFun = _cFuncCoeff;
  }else{
    FunctionCoefficient cCoeff( _cFunc );
    cCoeff->SetTime( _dt*(_myRank+1) );
    GridFunction cGridFun( _AhFESpace );
    cGridFun.ProjectCoefficient( cCoeff );  
  }
  // - then, loop over all elements and integrate to recover the average magnetic field
  // -- NB: this will contain \int_{\Omega}[Ax,Ay] dx, while actually B = [Ay,-Ax], but nothing much changes
  // TODO: here I'm hacking the ComputeGradError method of GridFunction: I hope no weird reordering of the nodes
  //       of FESpace occurs in the function, otherwise I'm doomed
  Vector B0(_dim) = 0.;
  for (int i = 0; i < _AhFESpace->GetNE(); i++){
    const FiniteElement *fe = _AhFESpace->GetFE(i);
    ElementTransformation *Tr = _AhFESpace->GetElementTransformation(i);
    int intorder = 2*fe->GetOrder() + 3;
    const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));
    Array<int> dofs;
    fes->GetElementDofs(i, dofs);
    for (int j = 0; j < ir->GetNPoints(); j++){
      const IntegrationPoint &ip = ir->IntPoint(j);
      Tr->SetIntPoint(&ip);
      Vector grad;
      cGridFun.GetGradient(*Tr,grad);
      B0 += ip.weight * Tr->Weight() * grad;
    }
  }
  // - average over time as well
  MPI_Allreduce( MPI_IN_PLACE, B0.GetData(), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  // - finally, compute its norm and rescale by domain size to get the average
  // -- B0 *should have been* rescaled by the space-time domain size, to get the actual average. When computing its ||B0||^2_L2st norm, 
  //     then, I should integrate over the space-time domain the square of the averaged value. Combining this, I'm missing a 1/( |\Omega|*|T|) factor
  double area = 1.0;    // TODO find a way to compute the area of the spatial domain?
  double B0norm = sqrt( ( B0*B0 ) / ( area * _dt*_numProcs) ); 
  if ( _myRank == 0 ){
    std::cout<<"Warning: When computing the average magnetic field, I'm assuming that the domain has area 1.0!"<<std::endl;
  }




  // All the ingredients are ready, now I just need to properly combine them
  // - IMPLICIT LEAPFROG
  switch (discType){
    case 0:{
      _Cp  = Aa;
      _Cp *= _dt*_dt*B0norm/( 4*_mu0 );
      
      _C0  = _Cp;
      _C0 *= -1.;
      _C0 += _MaNoZero;
      _C0 *= -2.;

      _Cp += _MaNoZero;
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

      _C0  = Aa;
      _C0 *= -_dt*_dt*B0norm/( 2*_mu0 );
      _C0 += _MaNoZero;
      _C0 *= -2.;

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
      std::cerr<<"ERROR: Discretisation type for wave equation "<<discType<<" not recognised."<<std::endl;
  }

}

















//-----------------------------------------------------------------------------
// Assemble space-time operators
//-----------------------------------------------------------------------------
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




// Code duplication at its worst:
// - assemble a bunch of block-bidiagonal space-time matrices (representing time-stepping operators)
inline void IMHD2DSTOperatorAssembler::AssembleFFu( ){ 
  AssembleSTBlockBiDiagonal( _Fu, _Mu, _FFu, _FFFu, "FFu", _FuAssembled && _MuAssembled, _FFuAssembled );
}
inline void IMHD2DSTOperatorAssembler::AssembleFFa( ){ 
  AssembleSTBlockBiDiagonal( _Fa, _Ma, _FFa, _FFFa, "FFa", _FaAssembled && _MaAssembled, _FFaAssembled );
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










// Assemble FFu^-1 (used in preconditioner)
void IMHD2DSTOperatorAssembler::AssembleFFuinv( const int spaceTimeSolverType = 0 ){
  if ( _FFuinvAssembled ){
    return;
  }

  switch (spaceTimeSolverType){
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

    // Use BoomerAMG with AIR set-up
    case 1:{
      if(! _FFuAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFuinv: need to assemble FFu before assembling FFuinv\n";
        return;
      }

      // Initialise MFEM wrapper for BoomerAMG solver
      HypreBoomerAMG *temp = new HypreBoomerAMG( *_FFFu );

      // Cast as HYPRE_Solver to get the underlying hypre object
      HYPRE_Solver FFuinv( *temp );

      // Set it up
      SetUpBoomerAMG( FFuinv );

      _FFuinv = temp;
  
      _FFuinvAssembled = true;

      break;
    }



    // Use GMRES with BoomerAMG precon
    case 2:{
      if(! _FFuAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFuinv: need to assemble FFu before assembling FFuinv\n";
        return;
      }
      if( _myRank == 0 ){
        std::cout<<"WARNING: Since you're using GMRES to solve the space-time block inside the preconditioner"<<std::endl
                 <<"         make sure that flexible GMRES is used as the outer solver!"<<std::endl;
      }


      // Initialise MFEM wrappers for GMRES solver and preconditioner
      HypreGMRES     *temp  = new HypreGMRES(     *_FFFu );
      HypreBoomerAMG *temp2 = new HypreBoomerAMG( *_FFFu );

      // Cast preconditioner as HYPRE_Solver to get the underlying hypre object
      HYPRE_Solver FFuinvPrecon( *temp2 );
      // Set it up
      SetUpBoomerAMG( FFuinvPrecon, 1 );   // with just one iteration this time around

      // Attach preconditioner to solver
      temp->SetPreconditioner( *temp2 );

      // adjust gmres options
      temp->SetKDim( 50 );
      temp->SetTol( 0.0 );   // to ensure fixed number of iterations
      temp->SetMaxIter( 15 );

      _FFuinv     = temp;
      _FFuinvPrec = temp2;
  
      _FFuinvAssembled = true;

      break;
    }

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
        std::cerr<<"Space-time solver type "<<spaceTimeSolverType<<" not recognised."<<std::endl;
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

  AssembleMz();

  _Mztemp = new PetscParMatrix( &_Mz );

  _MMzinv = new PetscLinearSolver( *_Mztemp, "zSolverMass_" );


  _MMzinvAssembled = true;
}






// Assemble pressure Schur complement
void IMHD2DSTOperatorAssembler::AssemblePS(){
  if ( _pSAssembled ){
    return;
  }

  if( _myRank == 0 ){
    std::cout<<"Warning: The original preconditioner for Oseen wasn't treating Dirichlet BC well."<<std::endl
             <<"         I think I corrected this now, but do double check!"<<std::endl;
  }

  // Assemble relevant operators
  AssembleAp();
  AssembleMp();
  AssembleWp();

  _pSinv = new OseenSTPressureSchurComplement( _comm, _dt, _mu, NULL, NULL, NULL, _essPhTDOF, _verbose );

  _pSinv->SetAp( &_Ap );
  _pSinv->SetMp( &_Mp );


  // Check if there is advection (if not, some simplifications can be made)
  bool isQuietState = true;
  if ( _wFunc != NULL ){  // if advection is prescribed as a function, I assume there is a wind
    isQuietState == false;
  }else{                  // if it's prescribed as a vector of node values, check whether they're all zero
    for ( int i = 0; i < _wFuncCoeff.Size(); ++i ){
      if ( _wFuncCoeff[i] != 0 ){
        isQuietState = false;
        break;
      }
    }
  }

  if( !isQuietState ){
    _pSinv->SetWp( &_Wp, false );    // if there is advection, then clearly Wp differs from Ap (must include pressure convection)
  }else if( _essQhTDOF.Size() == 0 ){ // otherwise, if there is no outflow
    _pSinv->SetWp( &_Wp, true );    
  }else{
    // _pSchur->SetWp( &_Wp, false );
    _pSinv->SetWp( &_Wp, true );     // should be false, according to E/S/W!
    if( _myRank == 0 ){
      std::cout<<"Warning: spatial part of Fp and Ap flagged to be the same, even though there is outflow."<<std::endl
               <<"         This goes against what Elman/Silvester/Wathen says (BC for Fp should be Robin"<<std::endl
               <<"         BC for Ap should be neumann/dirichlet on out). However, results seem much better."<<std::endl;
    }
  }


  _pSAssembled = true;

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time pressure Schur complement inverse approximation assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}




// Assemble CCa^-1 (used in magnetic Schur complement)
void IMHD2DSTOperatorAssembler::AssembleCCainv( const int spaceTimeSolverType = 0 ){

  if ( _CCainvAssembled ){
    return;
  }

  switch (spaceTimeSolverType){
    // Use sequential time-stepping on implicit / explicit leapfrog discretisation to solve for space-time block
    case 0:
    case 1:{
      AssembleCs( spaceTimeSolverType );

      SpaceTimeWaveSolver *temp  = new SpaceTimeWaveSolver( _comm, NULL, NULL, NULL, false, true, _verbose);
      temp->SetDiag( &_Cp, 0 );
      temp->SetDiag( &_C0, 1 );
      temp->SetDiag( &_Cm, 2 );
      _CCainv = temp;

      _CCainvAssembled = true;
      
      break;
    }


    default:{
      if ( _myRank == 0 ){
        std::cerr<<"Space-time solver type "<<spaceTimeSolverType<<" not recognised."<<std::endl;
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
void IMHD2DSTOperatorAssembler::AssembleAS( const int spaceTimeSolverType ){
  if ( _aSAssembled ){
    return;
  }

  AssembleCCainv( spaceTimeSolverType );
  AssembleMaNoZero();
  AssembleWa();

  _aSinv = new IMHD2DSTMagneticSchurComplement( _comm, _dt, _eta, NULL, NULL, NULL, _essAhTDOF, _verbose );

  _aSinv->SetMa( &_MaNoZero );
  _aSinv->SetWa( &_Wa );
  _aSinv->SetCCainv( &_CCainv );


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











// Assembles space-time KHI block system
//              ⌈ FFFu BBB ZZZ1 ZZZ2 ⌉⌈u⌉ ⌈f⌉
//   Ax = b <-> | BBBt               ||p|=|g|,
//              |          MMMz KKK  ||z|=|0|,
//              ⌊ YYY           FFFa ⌋⌊a⌋ ⌊h⌋
// Function also provides suitable initial guess for system (initialised with dirichlet BC)
void IMHD2DSTOperatorAssembler::AssembleSystem( HypreParMatrix*& FFFu, HypreParMatrix*& MMMz, HypreParMatrix*& FFFa,
                                                HypreParMatrix*& BBB,  HypreParMatrix*& ZZZ1, HypreParMatrix*& ZZZ2,
                                                HypreParMatrix*& KKK,  HypreParMatrix*& YYY,
                                                HypreParVector*& frhs, HypreParVector*& grhs, HypreParVector*& zrhs, HypreParVector*& hrhs,
                                                HypreParVector*& IGu,  HypreParVector*& IGp,  HypreParVector*& IGz,  HypreParVector*& IGa ){

  // - initialise relevant matrices
  AssembleFu();
  AssembleMu();
  AssembleMz();
  AssembleFa();
  AssembleMa();
  AssembleB();
  AssembleZ1();
  AssembleZ2();
  AssembleK();
  AssembleY();

  if ( _verbose>50 ){
    if ( _myRank == 0 ){
      std::ofstream myfile;
      std::string myfilename;

      myfilename = "./results/out_original_Fu.dat";
      myfile.open( myfilename );
      _Fu.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mu.dat";
      myfile.open( myfilename );
      _Mu.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mz.dat";
      myfile.open( myfilename );
      _Mz.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Fa.dat";
      myfile.open( myfilename );
      _Fa.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Ma.dat";
      myfile.open( myfilename );
      _Ma.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_B.dat";
      myfile.open( myfilename );
      _B.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Z1.dat";
      myfile.open( myfilename );
      _Z1.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Z2.dat";
      myfile.open( myfilename );
      _Z2.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_K.dat";
      myfile.open( myfilename );
      _K.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Y.dat";
      myfile.open( myfilename );
      _Y.PrintMatlab(myfile);
      myfile.close( );  
    }
    MPI_Barrier(_comm);
  }






  // ASSEMBLE RHS -----------------------------------------------------------
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
  LinearForm *fform( new LinearForm );
  fform->Update( _UhFESpace );
  fform->AddDomainIntegrator(   new VectorDomainLFIntegrator( fFuncCoeff       ) );  //int_\Omega f*v
  fform->AddBoundaryIntegrator( new BoundaryLFIntegrator(     nFuncCoeff       ) );  //int_d\Omega \mu * du/dn *v
  fform->AddBoundaryIntegrator( new BoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n

  fform->Assemble();

#ifdef MULT_BY_DT
  fform->operator*=( _dt );
#endif

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

#ifndef MULT_BY_DT
    u0form->operator*=(1./_dt);
#endif
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

#ifdef MULT_BY_DT
  gform->operator*=( _dt );
#endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for p assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // - for z
  LinearForm *zform( new LinearForm );
  // -- after integration by parts, I end up with a \int_\partial\Omega dA/dn * zeta
  zform->Update( _ZhFESpace );
  zform->AddBoundaryIntegrator( new BoundaryLFIntegrator( mFuncCoeff ) );  //int_d\Omega \eta * dA/dn *zeta, so I need to rescale by eta
  zform->Assemble();
  zform->operator*=( 1./_eta );

#ifdef MULT_BY_DT
  zform->operator*=( _dt );
#endif



  // - for A
  LinearForm *hform( new LinearForm );
  hform->Update( _AhFESpace );
  hform->AddDomainIntegrator(   new DomainLFIntegrator(   hFuncCoeff ) );  //int_\Omega h*B
  hform->AddBoundaryIntegrator( new BoundaryLFIntegrator( mFuncCoeff ) );  //int_d\Omega \eta * dA/dn *B
  hform->Assemble();

#ifdef MULT_BY_DT
  hform->operator*=( _dt );
#endif

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

#ifndef MULT_BY_DT
    a0form->operator*=(1./_dt);
#endif
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









  // - adjust rhs to take dirichlet BC for current time-step into account
  // -- initialise function with BC
  GridFunction uBC(_UhFESpace); zBC(_ZhFESpace); aBC(_AhFESpace);
  uBC.ProjectCoefficient(uFuncCoeff);
  zBC.ProjectCoefficient(zFuncCoeff);
  aBC.ProjectCoefficient(aFuncCoeff);
  // pBC.ProjectCoefficient(pFuncCoeff);
  // -- initialise local rhs
  Vector fRhsLoc( fform->Size() );
  Vector gRhsLoc( gform->Size() );
  Vector zRhsLoc( zform->Size() );
  Vector hRhsLoc( hform->Size() );
  // -- initialise local initial guess to exact solution
  Vector iguLoc( fform->Size() );
  Vector igpLoc( gform->Size() );
  Vector igzLoc( zform->Size() );
  Vector igaLoc( hform->Size() );
  Vector empty2;
  iguLoc = uBC;
  iguLoc.SetSubVectorComplement( _essUhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  igaLoc = aBC;
  igaLoc.SetSubVectorComplement( _essAhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  igzLoc = zBC;
  igzLoc.SetSubVectorComplement( _essZhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  igpLoc = 0.;                                      // dirichlet BC are not actually imposed on p
  // igpLoc.SetSubVectorComplement( _essQhTDOF, 0.0);
  Array<int> empty;



  // ASSEMBLE LOCAL LINEAR SYSTEMS (PARTICULARLY, CONSTRAINED MATRICES) -----
  // -- handy variables indicating dirichlet nodes
  mfem::Array<int> colsU(_Fu.Height()), colsZ(_Mz.Height()), colsA(_Fa.Height());
  colsU = 0; colsZ = 0; colsA = 0;
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    colsU[_essUhTDOF[i]] = 1;
  }
  for (int i = 0; i < _essZhTDOF.Size(); ++i){
    colsZ[_essZhTDOF[i]] = 1;
  }
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    colsA[_essAhTDOF[i]] = 1;
  }

  // - For velocity ---------------------------------------------------------
  // -- Assemble _Fu (and modify rhs to take dirichlet on u into account)
  fRhsLoc = *fform;
  _Fu.EliminateCols( colsU, &uBC, &fRhsLoc );
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    _Fu.EliminateRow( _essUhTDOF[i], mfem::Matrix::DIAG_ONE );
    fRhsLoc(_essUhTDOF[i]) = uBC(_essUhTDOF[i]);
  }
  

  // -- Assemble _Mu (and modify rhs to take dirichlet on u into account)
  uFuncCoeff.SetTime( _dt*_myRank );                // set uFunc to previous time-step
  GridFunction um1BC(_UhFESpace);
  um1BC.ProjectCoefficient(uFuncCoeff);

  Vector um1Rel( fRhsLoc.Size() );
  um1Rel = 0.0;
  _Mu.EliminateCols( colsU, &um1BC, &um1Rel );
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    _Mu.EliminateRow( _essUhTDOF[i], mfem::Matrix::DIAG_ZERO );
    um1Rel(_essUhTDOF[i]) = 0.0;                    // dir BC are already set in fRhsLoc: kill them here
  }

  if( _myRank > 0 ){
    // add to rhs (um1Rel should already take minus sign on _Mu into account)
    // NB: - no need to rescale by dt, as _Mu will be already scaled accordingly.
    //     - no need to flip sign, as _Mu carries with it already
    fRhsLoc += um1Rel;
  }


  // -- Assemble _Z1 (and modify rhs to take dirichlet on z into account)
  _Z1.EliminateCols( colsZ, &zBC, &fRhsLoc );
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    _Z1.EliminateRow( _essUhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }



  // -- Assemble _Z2 (and modify rhs to take dirichlet on A into account)
  _Z2.EliminateCols( colsA, &aBC, &fRhsLoc );
  for (int i = 0; i < _essUhTDOF.Size(); ++i){
    _Z2.EliminateRow( _essUhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }


  // -- remember to reset function evaluation for u to the current time
  uFuncCoeff.SetTime( _dt*(_myRank+1) );


  // -- assembly of the velocity matrices is now complete
  _FuAssembled = true;
  _MuAssembled = true;
  _Z1Assembled = true;
  _Z2Assembled = true;




  // - For pressure ---------------------------------------------------------
  // -- Assemble _B (and modify rhs to take dirichlet on u into account)
  gRhsLoc = *gform;
  _B.EliminateCols( colsU, &uBC, &gRhsLoc );

  // -- assembly of the pressure matrices is now complete
  _BAssembled  = true;




  // - For Laplacian of vector potential ------------------------------------
  // -- Assemble _Mz (and modify rhs to take dirichlet on z into account)
  zRhsLoc = *zform;
  _Mz.EliminateCols( colsZ, &zBC, &zRhsLoc );
  for (int i = 0; i < _essZhTDOF.Size(); ++i){
    _Mz.EliminateRow( _essZhTDOF[i], mfem::Matrix::DIAG_ONE );
    zRhsLoc(_essZhTDOF[i]) = zBC(_essZhTDOF[i]);
  }

  // -- Assemble _K (and modify rhs to take dirichlet on A into account)
  _K.EliminateCols( colsA, &aBC, &zRhsLoc );
  for (int i = 0; i < _essZhTDOF.Size(); ++i){
    _K.EliminateRow(  _essZhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }


  // -- assembly of the Laplacian of vector potential matrices is now complete
  _MzAssembled = true;
  _KAssembled  = true;





  // - For vector potential ------------------------------------------------
  // -- Assemble _Fa (and modify rhs to take dirichlet on A into account)
  hRhsLoc = *hform;
  _Fa.EliminateCols( colsA, &aBC, &hRhsLoc );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _Fa.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ONE );
    hRhsLoc(_essAhTDOF[i]) = aBC(_essAhTDOF[i]);
  }
  

  // -- Assemble _Y (and modify rhs to take dirichlet on u into account)
  _Y.EliminateCols( colsU, &uBC, &hRhsLoc );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _Y.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
  }


  // -- Assemble _Ma (and modify rhs to take dirichlet on A into account)
  aFuncCoeff.SetTime( _dt*_myRank );                // set aFunc to previous time-step
  GridFunction am1BC(_AhFESpace);
  am1BC.ProjectCoefficient(aFuncCoeff);

  Vector am1Rel( hRhsLoc.Size() );
  am1Rel = 0.0;
  _Ma.EliminateCols( colsA, &am1BC, &am1Rel );
  for (int i = 0; i < _essAhTDOF.Size(); ++i){
    _Ma.EliminateRow( _essAhTDOF[i], mfem::Matrix::DIAG_ZERO );
    am1Rel(_essAhTDOF[i]) = 0.0;                    // dir BC are already set in hRhsLoc: kill them here
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
  _FaAssembled = true;
  _MaAssembled = true;
  _YAssembled  = true;




  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Effect from Dirichlet BC (if prescribed) included in assembled blocks\n";
    }
    MPI_Barrier(_comm);
  }  






  // ASSEMBLE GLOBAL (PARALLEL) RHS -----------------------------------------
  // - for velocity
  int colPartU[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPartU );
  frhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for velocity block f assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for pressure
  int colPartP[2] = {_myRank*gRhsLoc.Size(), (_myRank+1)*gRhsLoc.Size()};
  grhs = new HypreParVector( _comm, gRhsLoc.Size()*_numProcs, gRhsLoc.StealData(), colPartP );
  grhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for pressure block g assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for Laplacian of vector potential
  int colPartZ[2] = {_myRank*zRhsLoc.Size(), (_myRank+1)*zRhsLoc.Size()};
  zrhs = new HypreParVector( _comm, zRhsLoc.Size()*_numProcs, zRhsLoc.StealData(), colPartZ );
  zrhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for Laplacian of vector potential block '0' assembled\n";
    }
    MPI_Barrier(_comm);
  }  


  // - for vector potential
  int colPartA[2] = {_myRank*hRhsLoc.Size(), (_myRank+1)*hRhsLoc.Size()};
  hrhs = new HypreParVector( _comm, hRhsLoc.Size()*_numProcs, hRhsLoc.StealData(), colPartA );
  hrhs->SetOwnership( 1 );

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
  IGu->SetOwnership( 1 );
  IGp->SetOwnership( 1 );
  IGz->SetOwnership( 1 );
  IGa->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time initial guesses assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  // ASSEMBLE SPACE-TIME OPERATOR -------------------------------------------
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


  

  if ( _verbose>50 ){
    std::string myfilename = std::string("./results/IGu.dat");
    IGu->Print(  myfilename.c_str() );
    myfilename = std::string("./results/RHSu.dat");
    frhs->Print( myfilename.c_str() );
    myfilename = std::string("./results/IGp.dat");
    IGp->Print(  myfilename.c_str() );
    myfilename = std::string("./results/RHSp.dat");
    grhs->Print( myfilename.c_str() );
    myfilename = std::string("./results/IGz.dat");
    IGz->Print(  myfilename.c_str() );
    myfilename = std::string("./results/RHSz.dat");
    zrhs->Print( myfilename.c_str() );
    myfilename = std::string("./results/IGa.dat");
    IGa->Print(  myfilename.c_str() );
    myfilename = std::string("./results/RHSa.dat");
    hrhs->Print( myfilename.c_str() );

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

      myfilename = "./results/out_essZ.dat";
      myfile.open( myfilename );
      _essZhTDOF.Print(myfile,1);
      myfile.close( );

      myfilename = "./results/out_essA.dat";
      myfile.open( myfilename );
      _essAhTDOF.Print(myfile,1);
      myfile.close( );

      std::cout<<"U essential nodes: ";_essUhTDOF.Print(std::cout, _essUhTDOF.Size());
      std::cout<<"P essential nodes: ";_essPhTDOF.Print(std::cout, _essPhTDOF.Size());
      std::cout<<"Z essential nodes: ";_essZhTDOF.Print(std::cout, _essZhTDOF.Size());
      std::cout<<"A essential nodes: ";_essAhTDOF.Print(std::cout, _essAhTDOF.Size());

    }

  }

}









// Compute application of non-linear operator to state vector
void IMHD2DSTOperatorAssembler::ApplyOperator( HypreParVector*& resU, HypreParVector*& resP, HypreParVector*& resZ, HypreParVector*& resA ){

  if ( !( _FuAssembled && _BAssembled && _Z2Assembled && _MzAssembled && _KAssembled && _YAssembled ) ){
    std::cerr<<"Need to assemble operators before computing the action"<<std::endl;
  }

  // Assemble local part of operator evaluation

  // - for u ----------------------------------------------------------------

  // -- Send prev solution for temporal part
  Vector lclu( _Fu.Height() ), wFuncCoeffPrev( _Fu.Height() ), tempu( _Fu.Height() );
  if ( _myRank < _numProcs-1 ){
    MPI_Send(    _wFuncCoeff.GetData(), _Fu.Height(), MPI_DOUBLE, _myRank+1, 2*_myRank,     _comm );
  }
  if ( _myRank > 0 ){
    MPI_Recv( wFuncCoeffPrev.GetData(), _Fu.Height(), MPI_DOUBLE, _myRank-1, 2*(_myRank-1), _comm, MPI_STATUS_IGNORE );
  }

  // -- include time derivative
  _Mu.Mult( wFuncCoeffPrev, lclu );
  lclu.Neg(); // -M*u(t-1)


  // -- include main operator
  // --- NB: _Fu includes also the u-linearisation of ( (u·∇)u, v ): if the integration rule is accurate enough,
  //          multiplying this term by w should include the effect of the non-linear term TWICE...?
  _Fu.AddMult( _wFuncCoeff, lclu );


  // GridFunction wGridFun( _UhFESpace );
  // wGridFun = _wFuncCoeff;
  // VectorGridFunctionCoefficient wCoeff( &wGridFun );
  // GridFunction wGridFunPrev( _UhFESpace );
  // wFuncCoeffPrev.Neg();
  // wGridFunPrev = wFuncCoeffPrev;
  // VectorGridFunctionCoefficient wCoeffPrev( &wGridFunPrev );

  // LinearForm *fform( new LinearForm );
  // fform->Update( _UhFESpace );
  // fform->AddDomainIntegrator( new VectorDomainLFIntegrator( wCoeff     ) );  //  int_\Omega u(t)*v
  // fform->AddDomainIntegrator( new VectorDomainLFIntegrator( wCoeffPrev ) );  // -int_\Omega u(t-1)0*v
  // fform->Assemble();

  // lclu = fform;

// #ifndef MULT_BY_DT
//   lclu *= (1./_dt);
// #endif

  
//   // -- include Laplacian
//   BilinearForm fuVarf(_UhFESpace);
// #ifdef MULT_BY_DT
//   fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( _mu*_dt ));       // mu*dt*K
// #else
//   fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( _mu ));
// #endif
//   fuVarf.Assemble();
//   fuVarf.Finalize();
//   fuVarf.AddMult( _wFuncCoeff, lclu );


  // -- include pressure part
  _B.AddMultTranspose( _wFuncCoeff, lclu );

  
  // -- remove velocity non-linear part (since it was included twice multiplying _Fu?)
  NonLinearForm nuVarf(_UhFESpace);
#ifdef MULT_BY_DT
  nuVarf.AddDomainIntegrator(new VectorConvectionNLFIntegrator( _dt ));
#else
  nuVarf.AddDomainIntegrator(new VectorConvectionNLFIntegrator( 1.0 ));
#endif
  nuVarf.Mult( _wFuncCoeff, tempu );
  lclu -= tempu;


  // -- include magnetic non-linear part
  // TODO: this should be equivalent to _Z1*y, provided that the integration rule used is accurate enough...?
  _Z2.AddMult( _cFuncCoeff, lclu );


  // -- include Dirichlet BC
  VectorFunctionCoefficient uFuncCoeff( _dim, _uFunc );
  uFuncCoeff.SetTime( _dt*(_myRank+1) );
  GridFunction uBC(_UhFESpace);
  uBC.ProjectCoefficient(uFuncCoeff);
  lclu.SetSubVector( _essUhTDOF, uBC );



  // - for p ----------------------------------------------------------------
  // -- This is easy since everything is linear
  Vector lclp( _B.Height() );
  _B.Mult( _wFuncCoeff, lclp );



  // - for z ----------------------------------------------------------------
  // -- This is easy since everything is linear
  Vector lclz( _Mz.Height() );
  _Mz.Mult( _yFuncCoeff, lclz );
  _K.AddMult(  _cFuncCoeff, lclz );

  // -- Include Dirichlet BC
  FunctionCoefficient zFuncCoeff( _zFunc );
  zFuncCoeff.SetTime( _dt*(_myRank+1) );
  GridFunction zBC(_ZhFESpace);
  zBC.ProjectCoefficient(zFuncCoeff);
  lclz.SetSubVector( _essZhTDOF, zBC );



  // - for A ----------------------------------------------------------------
  // -- Send prev solution for temporal part
  Vector lcla( _Fa.Height() ), cFuncCoeffPrev( _Fa.Height() );
  if ( _myRank < _numProcs-1 ){
    MPI_Send(    _cFuncCoeff.GetData(), _Fa.Height(), MPI_DOUBLE, _myRank+1, 2*_myRank+1,     _comm );
  }
  if ( _myRank > 0 ){
    MPI_Recv( cFuncCoeffPrev.GetData(), _Fa.Height(), MPI_DOUBLE, _myRank-1, 2*(_myRank-1)+1, _comm, MPI_STATUS_IGNORE );
  }


  // -- include time derivative
  _Ma.Mult( cFuncCoeffPrev, lcla );
  lclA.Neg(); // -M*A(t-1)


  // -- include main operator
  // --- NB: _Fa includes also the A-linearisation of ( u·∇A, B ): if the integration rule is accurate enough,
  //          multiplying this term by c should include also the effect of the non-linear term...?
  _Fa.AddMult( _cFuncCoeff, lcla );


//   GridFunction cGridFun( _AhFESpace );
//   cGridFun = _cFuncCoeff;
//   GridFunctionCoefficient cCoeff( &cGridFun );
//   GridFunction cGridFunPrev( _AhFESpace );
//   cFuncCoeffPrev.Neg();
//   cGridFunPrev = cFuncCoeffPrev;
//   GridFunctionCoefficient cCoeffPrev( &cGridFunPrev );

//   LinearForm *hform( new LinearForm );
//   hform->Update( _AhFESpace );
//   hform->AddDomainIntegrator( new DomainLFIntegrator( cCoeff     ) );  //  int_\Omega A(t)*B
//   hform->AddDomainIntegrator( new DomainLFIntegrator( cCoeffPrev ) );  // -int_\Omega A(t-1)*B
//   hform->Assemble();

//   lcla = hform;

// #ifndef MULT_BY_DT
//   lcla *= (1./_dt);
// #endif

  
//   // -- include Laplacian
//   BilinearForm faVarf(_AhFESpace);
// #ifdef MULT_BY_DT
//   faVarf.AddDomainIntegrator(new DiffusionIntegrator( _eta*_dt ));       // eta*dt*K
// #else
//   faVarf.AddDomainIntegrator(new DiffusionIntegrator( _eta ));
// #endif
//   faVarf.Assemble();
//   faVarf.Finalize();
//   faVarf.AddMult( _cFuncCoeff, lcla );


//   // -- include velocity non-linear part
//   // TODO: this should be equivalent to assemble ( w·∇A, B ) and multiply by c,
//   //       provided that the integration rule used is accurate enough...?
//   _Y.AddMult( _wFuncCoeff, lcla );


  // -- Include Dirichlet BC
  FunctionCoefficient aFuncCoeff( _aFunc );
  aFuncCoeff.SetTime( _dt*(_myRank+1) );
  GridFunction aBC(_AhFESpace);
  aBC.ProjectCoefficient(aFuncCoeff);
  lcla.SetSubVector( _essAhTDOF, aBC );





  // Assemble global vectors
  int colPartU[2] = {_myRank*lcl.Size(), (_myRank+1)*lclu.Size()};
  resU = new HypreParVector( _comm, lclu.Size()*_numProcs, lclu.StealData(), colPartU );
  resU->SetOwnership( 1 );

  int colPartP[2] = {_myRank*lcl.Size(), (_myRank+1)*lclp.Size()};
  resP = new HypreParVector( _comm, lclp.Size()*_numProcs, lclp.StealData(), colPartP );
  resP->SetOwnership( 1 );

  int colPartZ[2] = {_myRank*lcl.Size(), (_myRank+1)*lclz.Size()};
  resZ = new HypreParVector( _comm, lclz.Size()*_numProcs, lclz.StealData(), colPartZ );
  resZ->SetOwnership( 1 );

  int colPartA[2] = {_myRank*lcl.Size(), (_myRank+1)*lcla.Size()};
  resA = new HypreParVector( _comm, lcla.Size()*_numProcs, lcla.StealData(), colPartA );
  resA->SetOwnership( 1 );





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
  AssembleFFuinv( spaceTimeSolverTypeU );
  AssembleMMzinv( );

  AssemblePS();
  AssembleAS( spaceTimeSolverTypeA );

  Fuinv = _FFuinv;
  Mzinv = _MMzinv;
  pSinv = _pSinv;
  aSinv = _aSinv;

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
  const int blockSizeU = wh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeZ = zh.Size();
  const int blockSizeA = vh.Size();


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

  if( ! ( _FuAssembled && _MuAssembled
       && _MzAssembled
       && _FaAssembled && _MaAssembled
       && _BAssembled  && _Z1Assembled && _Z2Assembled
       && _KAssembled  && _YAssembled 
       && _MpAssembled && _ApAssembled && _WpAssembled ) ){
    if( _myRank == 0){
        std::cerr<<"Make sure all matrices have been initialised, otherwise they can't be printed"<<std::endl;
    }
    return;
  }

  std::string myfilename;
  std::ofstream myfile;


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
  delete _pSchur;
  delete _aSchur;
  delete _FFuinv;
  delete _FFuinvPrec;
  delete _CCainv;
  delete _CCainvPrec;
  delete _Mztemp;
  delete _MMzinv;
  delete _FFFu;
  delete _MMMz;
  delete _FFFa;
  if( _FFuAssembled )
    HYPRE_IJMatrixDestroy( _FFu );
  if( _FFaAssembled )
    HYPRE_IJMatrixDestroy( _FFa );
  if( _MMzAssembled )
    HYPRE_IJMatrixDestroy( _MMz );
  if( _BBAssembled )
    HYPRE_IJMatrixDestroy( _BB );
  if( _ZZ1Assembled )
    HYPRE_IJMatrixDestroy( _ZZ1 );
  if( _ZZ2Assembled )
    HYPRE_IJMatrixDestroy( _ZZ2 );
  if( _KKAssembled )
    HYPRE_IJMatrixDestroy( _KK );
  if( _YYAssembled )
    HYPRE_IJMatrixDestroy( _YY );

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


























