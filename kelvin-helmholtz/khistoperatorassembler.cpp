#include "KHISTOperatorAssembler.hpp"
#include "vectorconvectionintegrator.hpp"
#include "spacetimesolver.hpp"
#include "pararealsolver.hpp"

#include <mpi.h>
#include <string>
#include <cstring>
#include <iostream>
#include "HYPRE.h"
#include "petsc.h"
#include "mfem.hpp"

using namespace mfem;

// Seems like multiplying every operator by dt gives slightly better results.
#define MULT_BY_DT




//******************************************************************************
// Space-time block assembly
//******************************************************************************
// These functions are useful for assembling all the necessary space-time operators

KHISTOperatorAssembler::KHISTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
                                                const int refLvl, const int ordU, const int ordP,
                                                const double dt, const double mu, const double Pe,
                                                void(  *f)(const Vector &, double, Vector &),
                                                double(*g)(const Vector &, double ),
                                                void(  *n)(const Vector &, double, Vector &),
                                                void(  *w)(const Vector &, double, Vector &),
		                         							      void(  *u)(const Vector &, double, Vector &),
		                         							      double(*p)(const Vector &, double ),
                                                int verbose ):
	_comm(comm), _dt(dt), _mu(mu), _Pe(Pe), _fFunc(f), _gFunc(g), _nFunc(n), _wFunc(w), _uFunc(u), _pFunc(p), _ordU(ordU), _ordP(ordP),
  _MuAssembled(false), _FuAssembled(false), _MpAssembled(false), _ApAssembled(false), _WpAssembled(false), _BAssembled(false),
  _FFinvPrec(NULL), _FFAssembled(false), _BBAssembled(false), _pSAssembled(false), _FFinvAssembled(false),
  _verbose(verbose){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

	// For each processor:
	//- generate mesh
	_mesh = new Mesh( meshName.c_str(), 1, 1 );
  
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  _dim = _mesh->Dimension();

  if ( _dim != 2 && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: KHI only works for 2D domains\n";
  }

  // - initialise FE info
  _WhFEColl = new H1_FECollection( ordW, _dim );
  _PhFEColl = new H1_FECollection( ordP, _dim );
  _VhFEColl = new H1_FECollection( ordV, _dim );
  _WhFESpace = new FiniteElementSpace( _mesh, _WhFEColl );
  _PhFESpace = new FiniteElementSpace( _mesh, _PhFEColl );
  _VhFESpace = new FiniteElementSpace( _mesh, _VhFEColl, _dim );


  if ( _mesh->bdr_attributes.Size() > 0 ) {
    Array<int> essBdrW( _mesh->bdr_attributes.Max() ), essBdrP( _mesh->bdr_attributes.Max() ), essBdrV( _mesh->bdr_attributes.Max() );
    essBdrW = 0; essBdrP = 0; essBdrV = 0;
    for ( int i = 0; i < _mesh->bdr_attributes.Max(); ++i ){
      if( _mesh->bdr_attributes[i] == 1 )
        essBdrW[i] = 1;
      if( _mesh->bdr_attributes[i] == 2 )
        essBdrP[i] = 1;
      if( _mesh->bdr_attributes[i] == 3 )
        essBdrV[i] = 1;
    }

    _WhFESpace->GetEssentialTrueDofs( essBdrW, _essWhTDOF );
    _PhFESpace->GetEssentialTrueDofs( essBdrP, _essPhTDOF );
    _VhFESpace->GetEssentialTrueDofs( essBdrV, _essVhTDOF );
  }


  if (_myRank == 0 ){
    std::cout << "***********************************************************\n";
    std::cout << "dim(Wh) = " << _WhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
    std::cout << "dim(Vh) = " << _VhFESpace->GetTrueVSize() << "\n";
    std::cout << "***********************************************************\n";
  }

}





//-----------------------------------------------------------------------------
// Assemble operators for single time-steps
//-----------------------------------------------------------------------------

// Assemble operator on main diagonal of space-time matrix for w block:
//  Fw = M + mu*dt K + dt*W(v)
void KHISTOperatorAssembler::AssembleFwVarf( ){
  if( _FwAssembled ){
    return;
  }

  _fwVarf =  new BilinearForm(_WhFESpace);

  VectorFunctionCoefficient vFuncCoeff( _dim, _vFunc );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );

#ifdef MULT_BY_DT
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
	_fwVarf->AddDomainIntegrator(new MassIntegrator( one ));
	_fwVarf->AddDomainIntegrator(new DiffusionIntegrator( muDt ));
  _fwVarf->AddDomainIntegrator(new ConvectionIntegrator( vFuncCoeff, _dt ));
#else
  ConstantCoefficient mu( _mu );
  ConstantCoefficient dtinv( 1./_dt );
  _fwVarf->AddDomainIntegrator(new MassIntegrator( dtinv ));
  _fwVarf->AddDomainIntegrator(new DiffusionIntegrator( mu ));
  _fwVarf->AddDomainIntegrator(new ConvectionIntegrator( vFuncCoeff, 1.0 ));
#endif

  _fwVarf->Assemble();
  _fwVarf->Finalize();
  


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Spatial operator Fw assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}



// Assemble operator on subdiagonal of space-time matrix for w block:
//  Mw = -M
void KHISTOperatorAssembler::AssembleMwVarf( ){
  if( _MwAssembled ){
    return;
  }

	_mwVarf = new BilinearForm(_WhFESpace);
#ifdef MULT_BY_DT
  ConstantCoefficient mone( -1.0 );
  _mwVarf->AddDomainIntegrator(new MassIntegrator( mone ));
#else
  ConstantCoefficient mdtinv( -1./_dt );
  _mwVarf->AddDomainIntegrator(new MassIntegrator( mdtinv ));
#endif
  _mwVarf->Assemble();
  _mwVarf->Finalize();

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass-matrix for w (negative) Mw assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}


// Assemble mass matrix for v block:
//  Mv = dt*u*v
void KHISTOperatorAssembler::AssembleMvVarf( ){
  if( _MvAssembled ){
    return;
  }

  _mvVarf = new BilinearForm(_VhFESpace);
#ifdef MULT_BY_DT
  ConstantCoefficient one( 1.0 );
  _mvVarf->AddDomainIntegrator(new VectorMassIntegrator( one ));
#else
  ConstantCoefficient Dt( _dt );
  _mvVarf->AddDomainIntegrator(new VectorMassIntegrator( Dt ));
#endif
  _mvVarf->Assemble();
  _mvVarf->Finalize();

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mass-matrix for v Mv assembled\n";
    }
    MPI_Barrier(_comm);
  }  


}


// Assemble laplacian for potential block:
//  Ap = dt * div(grad(\phi))
void KHISTOperatorAssembler::AssembleAp( ){

  if( _ApAssembled ){
    return;
  }

  BilinearForm *aVarf( new BilinearForm(_PhFESpace) );
#ifdef MULT_BY_DT  
  ConstantCoefficient Dt( _dt );
  aVarf->AddDomainIntegrator(  new DiffusionIntegrator( Dt ));
#else
  ConstantCoefficient one( 1.0 );
  aVarf->AddDomainIntegrator(  new DiffusionIntegrator( one ));
#endif

  // Impose homogeneous dirichlet BC by simply removing corresponding equations?
  TODO
  aVarf->Assemble();
  aVarf->Finalize();
  
  aVarf->FormSystemMatrix( _essPhTDOF, _Ap );
  _Ap.SetGraphOwner(true);
  _Ap.SetDataOwner(true);
  aVarf->LoseMat();


  delete aVarf;

  _ApAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Ap.dat";
    myfile.open( myfilename );
    _Ap.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Laplacian operator Ap for potential assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}



// Assemble divergence operator coupling v to w:
//  B(w) = dt * div(w * u), v
void KHISTOperatorAssembler::AssembleBVarf( ){

  if( _BAssembled ){
    return;
  }

	_bVarf = new MixedBilinearForm( _VhFESpace, _WhFESpace );
  FunctionCoefficient wFuncCoeff( _wFunc );
  wFuncCoeff.SetTime( _dt*(_myRank+1) );

  // TODO: how to include multiplication by dt?

#ifdef MULT_BY_DT  
  ConstantCoefficient Dt( _dt );
  _bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(Dt*wFuncCoeff) );
#else
  ConstantCoefficient one( 1.0 );
  _bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(one*wFuncCoeff) );
#endif

  _bVarf->Assemble();
  _bVarf->Finalize();


  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Divergence operator B(w) assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}





// Assemble mass matrix coupling w to potential:
//  Mwp = dt * w * \phi
void KHISTOperatorAssembler::AssembleMwp( ){

  if( _MwpAssembled ){
    return;
  }

	BilinearForm *mVarf( new MixedBilinearForm( _WhFESpace, _PhFESpace ) );

#ifdef MULT_BY_DT  
  ConstantCoefficient Dt( _dt );
  mVarf->AddDomainIntegrator(new MixedScalarMassIntegrator( Dt ));
#else
  ConstantCoefficient one( 1.0 );
  mVarf->AddDomainIntegrator(new MixedScalarMassIntegrator( one ));
#endif

  mVarf->Assemble();
  mVarf->Finalize();


  // - once the matrix is generated, we can get rid of the operator
  _Mwp = mVarf->SpMat();
  _Mwp.SetGraphOwner(true);
  _Mwp.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

  _MwpAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_Mwp.dat";
    myfile.open( myfilename );
    _Mwp.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Mixed w-phi mass matrix Mwp assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}




// Assemble scalar curl operator coupling potential to v:
//  C = dt * curl(k*phi) v = dt * (v, J grad(phi) ), with J = [0,1;-1,0]
void KHISTOperatorAssembler::AssembleC( ){

  if( _CAssembled ){
    return;
  }

  BilinearForm *cVarf( new MixedBilinearForm( _PhFESpace, _VhFESpace ) );

  // ok, this is tricky. MixedVectorWeakDivergenceIntegrator assembles
  // a(v,phi) = - (Q v, grad(phi))
  // so in order to assemble (v, J grad(phi)), I need Q = - J^T = J :)
  DenseMatrix mJT(2);
#ifdef MULT_BY_DT
  mJT.GetData()[0] = 0.;
  mJT.GetData()[1] =  _dt;
  mJT.GetData()[2] = -_dt;
  mJT.GetData()[3] = 0.;
#else
  mJT.GetData()[0] = 0.;
  mJT.GetData()[1] =  1.;
  mJT.GetData()[2] = -1.;
  mJT.GetData()[3] = 0.;
#endif
  MatrixConstantCoefficient Q( mJT );
  cVarf->AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator ( Q ));

  cVarf->Assemble();
  cVarf->Finalize();


  // - once the matrix is generated, we can get rid of the operator
  _C = cVarf->SpMat();
  _C.SetGraphOwner(true);
  _C.SetDataOwner(true);
  cVarf->LoseMat();
  delete cVarf;

  _CAssembled = true;


  if ( _verbose>50 && _myRank == 0 ){
    std::ofstream myfile;
    std::string myfilename = "./results/out_final_C.dat";
    myfile.open( myfilename );
    _C.PrintMatlab(myfile);
    myfile.close( );
  }

  if( _verbose>5 ){
    if ( _myRank==0 ){
      std::cout<<"Scalar curl operator C assembled\n";
    }
    MPI_Barrier(_comm);
  }  
}










//-----------------------------------------------------------------------------
// Assemble space-time operators
//-----------------------------------------------------------------------------
// Assemble space-time matrix FF (w block)
TODO
void KHISTOperatorAssembler::AssembleFF( ){ 
  
  if( _FFAssembled ){
    return;
  }

  if( !( _FwAssembled && _MwAssembled ) && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble Mw and Fw before assembling FF\n";
    return;
  }


  // Create FF block ********************************************************
  // Initialize HYPRE matrix
  // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
  //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
  // - get info on matrix structure
  const int blockSizeFF = _Fw.NumRows();
 
  Array<int> nnzPerRowD( blockSizeFF );   // num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<int> nnzPerRowO( blockSizeFF );   // ..and in off-diagonal block
  const int  *offIdxsD = _Fw.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  const int  *offIdxsO = _Mw.GetI();
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
                               blockSizeFF*_myRank, blockSizeFF*(_myRank+1)-1, &_FF );
  HYPRE_IJMatrixSetObjectType( _FF, HYPRE_PARCSR );
  HYPRE_IJMatrixSetDiagOffdSizes( _FF, nnzPerRowD.GetData(), nnzPerRowO.GetData() );    // this gives issues :/
  HYPRE_IJMatrixInitialize( _FF );


  // - fill it with matrices assembled above
  // -- diagonal block
  Array<int> rowsGlbIdxD( blockSizeFF );
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxD[i] = i + blockSizeFF*_myRank;
  }
  Array<int> colsGlbIdxD( _Fw.NumNonZeroElems() );
  for ( int i=0; i<_Fw.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = _Fw.GetJ()[i] + blockSizeFF*_myRank;
  }
  HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowD.GetData(),
                           rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), _Fw.GetData() );     // setvalues *copies* the data

  // -- off-diagonal block
  Array<int> rowsGlbIdxO( blockSizeFF );      // TODO: just use rowsGlbIdx once for both matrices?
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxO[i] = i + blockSizeFF*_myRank;
  }
  if ( _myRank > 0 ){
    Array<int> colsGlbIdxO( _Mw.NumNonZeroElems() );
    for ( int i=0; i<_Mw.NumNonZeroElems(); i++ ) {
      colsGlbIdxO[i] = _Mw.GetJ()[i] + blockSizeFF*(_myRank-1);
    }
    HYPRE_IJMatrixSetValues( _FF, blockSizeFF, nnzPerRowO.GetData(),
                             rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), _Mw.GetData() );
  }


  // - assemble
  HYPRE_IJMatrixAssemble( _FF );
  _FFAssembled = true;

  // - convert to a MFEM operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( _FF, (void **) &FFref);
  _FFF = new HypreParMatrix( FFref, false ); //"false" doesn't take ownership of data


  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time operator FF assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}




// code duplication at its worst: assemble a bunch of block-diagonal space-time matrices
inline void KHISTOperatorAssembler::AssembleAA( ){ 
  AssembleSTBlockDiagonal( _Ap, _AA, "AA", _ApAssembled, _AAAssembled );
}
inline void KHISTOperatorAssembler::AssembleMMv( ){ 
  AssembleSTBlockDiagonal( _Mv, _MMv, "MMv", _MvAssembled, _MMvAssembled );
}
inline void KHISTOperatorAssembler::AssembleBB( ){ 
  AssembleSTBlockDiagonal( _B, _BB, "BB", _BAssembled, _BBAssembled );
}
inline void KHISTOperatorAssembler::AssembleCC( ){ 
  AssembleSTBlockDiagonal( _C, _CC, "CC", _CAssembled, _CCAssembled );
}
inline void KHISTOperatorAssembler::AssembleMMwp( ){ 
  AssembleSTBlockDiagonal( _Mwp, _MMwp, "MMwp", _MwpAssembled, _MMwpAssembled );
}




// Handy function to assemble a monolithic space-time block-diagonal matrix starting from its block(s)
void KHISTOperatorAssembler::AssembleSTBlockDiagonal( const SparseMatrix& D, HYPRE_IJMatrix& DD,
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












// Assemble inverse of AA (used in preconditioner)
void KHISTOperatorAssembler::AssembleAinv(){
  if ( _AinvAssembled ){
    return;
  }

  if( !_ApAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble A before assembling Ainv\n";
    return;
  }

  _Ainv = new PetscLinearSolver( _Ap, "ApSolver_" );
  
  _AinvAssembled = true;

}


// Assemble inverse of Mv (used in preconditioner)
void KHISTOperatorAssembler::AssembleMvinv(){
  if ( _MvinvAssembled ){
    return;
  }

  if( !_MvAssembled && _myRank == 0 ){
    std::cerr<<"FATAL ERROR: Need to assemble Mv before assembling Mvinv\n";
    return;
  }

  _Mvinv = new PetscLinearSolver( _Mv, "MvSolver_" );
  
  _MvinvAssembled = true;

}




// Assemble FF^-1 (used in preconditioner)
void KHISTOperatorAssembler::AssembleFFinv( const int spaceTimeSolverType = 0 ){
  if ( _FFinvAssembled ){
    return;
  }

  switch (spaceTimeSolverType){
    // Use sequential time-stepping to solve for space-time block
    case 0:{
      if(!( _MwAssembled && _FwAssembled ) && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble Mw and Fw before assembling Finv\n";
        return;
      }
      //                                             flag as time-dependent regardless of anything
      SpaceTimeSolver *temp  = new SpaceTimeSolver( _comm, NULL, NULL, _essVhTDOF, true, _verbose );

      temp->SetF( &_Fw );
      temp->SetM( &_Mw );
      _FFinv = temp;

      _FFinvAssembled = true;
      
      break;
    }

    // Use BoomerAMG with AIR set-up
    case 1:{
      if(! _FFAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble FF before assembling Finv\n";
        return;
      }

      // Initialise MFEM wrapper for BoomerAMG solver
      HypreBoomerAMG *temp = new HypreBoomerAMG( *_FFF );

      // Cast as HYPRE_Solver to get the underlying hypre object
      HYPRE_Solver FFinv( *temp );

      // Set it up
      SetUpBoomerAMG( FFinv );

      _FFinv = temp;
  
      _FFinvAssembled = true;

      break;
    }



    // Use GMRES with BoomerAMG precon
    case 2:{
      if(! _FFAssembled  && _myRank == 0 ){
        std::cerr<<"ERROR: AssembleFFinv: need to assemble FF before assembling Finv\n";
        return;
      }
      if( _myRank == 0 ){
        std::cout<<"WARNING: Since you're using GMRES to solve the space-time block inside the preconditioner"<<std::endl
                 <<"         make sure that flexible GMRES is used as the outer solver!"<<std::endl;
      }


      // Initialise MFEM wrappers for GMRES solver and preconditioner
      HypreGMRES     *temp  = new HypreGMRES(     *_FFF );
      HypreBoomerAMG *temp2 = new HypreBoomerAMG( *_FFF );

      // Cast preconditioner as HYPRE_Solver to get the underlying hypre object
      HYPRE_Solver FFinvPrecon( *temp2 );
      // Set it up
      SetUpBoomerAMG( FFinvPrecon, 1 );   // with just one iteration this time around

      // Attach preconditioner to solver
      temp->SetPreconditioner( *temp2 );

      // adjust gmres options
      temp->SetKDim( 50 );
      temp->SetTol( 0.0 );   // to ensure fixed number of iterations
      temp->SetMaxIter( 15 );

      _FFinv     = temp;
      _FFinvPrec = temp2;
  
      _FFinvAssembled = true;

      break;
    }

    // Use Parareal with coarse/fine solver of different accuracies
    case 3:{

      const int maxIt = 2;

      if( _numProcs <= maxIt ){
        if( _myRank == 0 ){
          std::cerr<<"ERROR: AssembleFFinv: Trying to set solver as "<<maxIt<<" iterations of Parareal, but the fine discretisation only has "<<_numProcs<<" nodes. "
                   <<"This is equivalent to time-stepping, so I'm picking that as a solver instead."<<std::endl;
        }
        
        AssembleFFinv( 0 );
        return;
      }


      PararealSolver *temp  = new PararealSolver( _comm, NULL, NULL, NULL, maxIt, _verbose );

      temp->SetF( &_Fw );
      temp->SetC( &_Fw ); // same operator is used for both! it's the solver that changes, eventually...
      temp->SetM( &_Mw );
      _FFinv = temp;

      _FFinvAssembled = true;
      
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
      std::cout<<"Space-time block inverse (approximation) FF^-1 assembled\n";
    }
    MPI_Barrier(_comm);
  }  

}







void KHISTOperatorAssembler::SetUpBoomerAMG( HYPRE_Solver& FFinv, const int maxiter ){
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
//              ⌈ FFF BBB     ⌉⌈w⌉ ⌈f⌉
//   Ax = b <-> | Mwp AAA     ||p|=|g|,
//              ⌊     CCC MMv ⌋⌊v⌋ ⌊h⌋
// where:
//  - FF is space-time matrix for w
//  - AA is block-diagonal with potential laplacian
//  - Mv is block-diagonal with v mass matrix
//  - BB is block-diagonal with div(w * )
//  - Mw is block-diagonal with mixed w-p mass matrix
//  - CC is block-diagonal with curl(k * )
//  - f  is the rhs for w
//  - g  is the rhs for p
//  - h  is the rhs for v
// Function also provides suitable initial guess for system (initialised with dirichlet BC)
void KHISTOperatorAssembler::AssembleSystem( HypreParMatrix*& FFF,  HypreParMatrix*& AAA,  HypreParMatrix*& MMv,
                                             HypreParMatrix*& BBB,  HypreParMatrix*& Mwp,  HypreParMatrix*& CCC,
                                             HypreParVector*& frhs, HypreParVector*& grhs, HypreParVector*& hrhs,
                                             HypreParVector*& IGw,  HypreParVector*& IGp,  HypreParVector*& IGv ){

  // - initialise relevant bilinear forms
  AssembleFwVarf();
  AssembleApVarf();
  AssembleMvVarf();
  AssembleMwVarf();
  AssembleBVarf();
  AssembleCVarf();

  if ( _verbose>50 ){
    if ( _myRank == 0 ){
      std::ofstream myfile;
      std::string myfilename;

      myfilename = "./results/out_original_Fw.dat";
      myfile.open( myfilename );
      (_fwVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mw.dat";
      myfile.open( myfilename );
      (_mwVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mv.dat";
      myfile.open( myfilename );
      (_mvVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Ap.dat";
      myfile.open( myfilename );
      (_apVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      std::string myfilename = "./results/out_original_B.dat";
      myfile.open( myfilename );
      (_bVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_Mwp.dat";
      myfile.open( myfilename );
      (_mwpVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_original_C.dat";
      myfile.open( myfilename );
      (_cVarf->SpMat()).PrintMatlab(myfile);
      myfile.close( );  
    }
    MPI_Barrier(_comm);
  }



  // ASSEMBLE RHS -----------------------------------------------------------
  // Initialise handy functions for rhs
  FunctionCoefficient       wFuncCoeff( _wFunc );
  FunctionCoefficient       pFuncCoeff( _pFunc );
  FunctionCoefficient       fFuncCoeff( _fFunc );
  FunctionCoefficient       gFuncCoeff( _gFunc );
  VectorFunctionCoefficient vFuncCoeff( _dim, _vFunc );
  VectorFunctionCoefficient hFuncCoeff( _dim, _hFunc );

  // - specify evaluation time
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  fFuncCoeff.SetTime( _dt*(_myRank+1) );
  gFuncCoeff.SetTime( _dt*(_myRank+1) );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );
  hFuncCoeff.SetTime( _dt*(_myRank+1) );

  // Assemble local part of rhs
  // - for w
  LinearForm *fform( new LinearForm );
  fform->Update( _WhFESpace );
  fform->AddDomainIntegrator(   new DomainLFIntegrator(       fFuncCoeff       ) );  //int_\Omega f*v
  TODO
  // fform->AddBoundaryIntegrator( new BoundaryLFIntegrator(     nFuncCoeff       ) );  //int_d\Omega \mu * du/dn *v
  // fform->AddBoundaryIntegrator( new BoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n

  fform->Assemble();

#ifdef MULT_BY_DT
  fform->operator*=( _dt );
#endif

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for W assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // -- include initial conditions
  if( _myRank == 0 ){
    wFuncCoeff.SetTime( 0.0 );
    LinearForm *w0form( new LinearForm );
    w0form->Update( _WhFESpace );
    w0form->AddDomainIntegrator( new DomainLFIntegrator( wFuncCoeff ) );  //int_\Omega w0*v
    w0form->Assemble();

#ifndef MULT_BY_DT
    w0form->operator*=(1./_dt);
#endif
    fform->operator+=( *w0form );


    if ( _verbose>100 ){
      std::cout<<"Contribution from IC on w: "; w0form->Print(std::cout, w0form->Size());
    }

    // remember to reset function evaluation for w to the current time
    wFuncCoeff.SetTime( _dt*(_myRank+1) );


    delete w0form;

    if(_verbose>10){
      std::cout<<"Contribution from initial condition included\n"<<std::endl;
    }
  }





  // - for p
  LinearForm *gform( new LinearForm );
  gform->Update( _PhFESpace );
  gform->AddDomainIntegrator( new DomainLFIntegrator( gFuncCoeff ) );  //int_\Omega g*q
  TODO
  gform->Assemble();

#ifdef MULT_BY_DT
  gform->operator*=( _dt );
#endif


  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for phi assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // - for v
  LinearForm *hform( new LinearForm );
  hform->Update( _VhFESpace );
  hform->AddDomainIntegrator( new VectorDomainLFIntegrator( hFuncCoeff ) );  //int_\Omega h*v
  TODO
  hform->Assemble();

#ifdef MULT_BY_DT
  hform->operator*=( _dt );
#endif


  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Local (single time-step) contribution to rhs for v assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  // - adjust rhs to take dirichlet BC for current time-step into account
  // -- initialise function with BC
  GridFunction wBC(_WhFESpace);//, pBC(_QhFESpace);
  wBC.ProjectCoefficient(wFuncCoeff);
  // pBC.ProjectCoefficient(pFuncCoeff);
  // -- initialise local rhs
  Vector fRhsLoc(  fform->Size() );
  Vector gRhsLoc(  gform->Size() );
  Vector hRhsLoc(  hform->Size() );
  // -- initialise local initial guess to exact solution
  Vector igwLoc( fform->Size() );
  Vector igpLoc( gform->Size() );
  Vector igvLoc( hform->Size() );
  Vector empty2;
  igwLoc = uBC;
  igwLoc.SetSubVectorComplement( _essWhTDOF, 0.0); // set to zero on interior (non-essential) nodes - TODO: initialise everything to IC?
  TODO
  igpLoc = 0.;                                      // dirichlet BC are not actually imposed on p
  igvLoc = 0.;                                      // dirichlet BC are not actually imposed on v
  // igpLoc.SetSubVectorComplement( _essQhTDOF, 0.0);
  Array<int> empty;




  // ASSEMBLE LOCAL LINEAR SYSTEMS (PARTICULARLY, CONSTRAINED MATRICES) -----
  TODO
  // - Assemble _Fw (and modify rhs to take dirichlet on w into account)
  _fwVarf->FormLinearSystem(           _essWhTDOF,        wBC, *fform, _Fw, empty2, fRhsLoc );
  // - Assemble _B (and modify rhs to take dirichlet on w into account)
  _bVarf->FormRectangularLinearSystem( _essWhTDOF, empty, wBC, *fform, _B,  empty2, fRhsLoc );  // iguloc should still be initialised to uBC
  // - Assemble _Mw (and modify rhs to take dirichlet on w into account)
  //  -- this is quite tricky, actually, since we need to include the effect from dirichlet through the mass matrix appearing in the sub-diagonal into the rhs
  wFuncCoeff.SetTime( _dt*_myRank );                // set wFunc to previous time-step
  GridFunction wm1BC(_WhFESpace);
  wm1BC.ProjectCoefficient(wFuncCoeff);
  Vector wm1Rel( fRhsLoc.Size() );
  wm1Rel = 0.0;
  //  -- remove dirichlet from mass matrix in sub-diag (and save effect in wm1Rel)
  _mwVarf->EliminateVDofs( _essWhTDOF, wm1BC, wm1Rel, Matrix::DiagonalPolicy::DIAG_ZERO ); 
  //  -- include effect from dirichlet in rhs
  if( _myRank > 0 ){
    // NB: - no need to rescale by dt, as _Mw will be already scaled accordingly.
    //     - no need to flip sign, as _Mw carries with it already
    fRhsLoc += wm1Rel;
  }
  // -- remember to reset function evaluation for w to the current time
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  // -- store w mass matrix - now with the dirichlet nodes killed
  _Mw = _mwVarf->SpMat();

  TODO
  _FwAssembled = true;
  _MwAssembled = true;
  _BAssembled  = true;

  if( _verbose>10 ){
    if ( _myRank==0 ){
      std::cout<<"Effect from Dirichlet BC (if prescribed) included in assembled blocks\n";
    }
    MPI_Barrier(_comm);
  }  




  // ASSEMBLE GLOBAL (PARALLEL) RHS -----------------------------------------
  // - for w
  int colPartW[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPartW );
  frhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for w block f assembled\n";
    }
    MPI_Barrier(_comm);
  }  

  // - for p
  int colPartP[2] = {_myRank*gRhsLoc.Size(), (_myRank+1)*gRhsLoc.Size()};
  grhs = new HypreParVector( _comm, gRhsLoc.Size()*_numProcs, gRhsLoc.StealData(), colPartP );
  grhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for p block g assembled\n";
    }
    MPI_Barrier(_comm);
  }  

  // - for v
  int colPartV[2] = {_myRank*hRhsLoc.Size(), (_myRank+1)*hRhsLoc.Size()};
  hrhs = new HypreParVector( _comm, hRhsLoc.Size()*_numProcs, hRhsLoc.StealData(), colPartV );
  hrhs->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time rhs for v block h assembled\n";
    }
    MPI_Barrier(_comm);
  }  



  // ASSEMBLE INITIAL GUESS -------------------------------------------------
  // Assemble global vectors
  IGw = new HypreParVector( _comm, igwLoc.Size()*_numProcs, igwLoc.StealData(), colPartW );
  IGp = new HypreParVector( _comm, igpLoc.Size()*_numProcs, igpLoc.StealData(), colPartp );
  IGv = new HypreParVector( _comm, igvLoc.Size()*_numProcs, igvLoc.StealData(), colPartV );
  IGw->SetOwnership( 1 );
  IGp->SetOwnership( 1 );
  IGv->SetOwnership( 1 );

  if( _verbose>1 ){
    if ( _myRank==0 ){
      std::cout<<"Space-time initial guess assembled\n";
    }
    MPI_Barrier(_comm);
  }  




  // ASSEMBLE SPACE-TIME OPERATORS ------------------------------------------
  //Assemble w-w block
  AssembleFF();
  // - pass handle to mfem matrix
  FFF = new HypreParMatrix();
  FFF->MakeRef( *_FFF );

  //Assemble p-p block
  AssembleAA();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  AAref;
  HYPRE_IJMatrixGetObject( _AA, (void **) &AAref);
  AAA = new HypreParMatrix( AAref, false ); //"false" doesn't take ownership of data

  //Assemble v-v block
  AssembleMMv();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  MMvref;
  HYPRE_IJMatrixGetObject( _MMv, (void **) &MMvref);
  MMv = new HypreParMatrix( MMvref, false ); //"false" doesn't take ownership of data

  //Assemble v-w block
  AssembleBB();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  BBref;
  HYPRE_IJMatrixGetObject( _BB, (void **) &BBref);
  BBB = new HypreParMatrix( BBref, false ); //"false" doesn't take ownership of data

  //Assemble w-p block
  AssembleMwp();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  Mwpref;
  HYPRE_IJMatrixGetObject( _MMwp, (void **) &Mwpref);
  Mwp = new HypreParMatrix( Mwpref, false ); //"false" doesn't take ownership of data

  //Assemble p-v block
  AssembleCC();
  // - convert to mfem operator
  HYPRE_ParCSRMatrix  CCref;
  HYPRE_IJMatrixGetObject( _CC, (void **) &CCref);
  CCC = new HypreParMatrix( CCref, false ); //"false" doesn't take ownership of data
  



  if ( _verbose>50 ){
    std::string myfilename
    myfilename = std::string("./results/IGw.dat");
    IGw->Print( myfilename.c_str() );
    myfilename = std::string("./results/IGp.dat");
    IGp->Print( myfilename.c_str() );
    myfilename = std::string("./results/IGv.dat");
    IGv->Print( myfilename.c_str() );
    myfilename = std::string("./results/RHSw.dat");
    frhs->Print( myfilename.c_str() );
    myfilename = std::string("./results/RHSp.dat");
    grhs->Print( myfilename.c_str() );
    myfilename = std::string("./results/RHSv.dat");
    hrhs->Print( myfilename.c_str() );

    if ( _myRank == 0 ){

      std::ofstream myfile;

      myfilename = "./results/out_final_Fw.dat";
      myfile.open( myfilename );
      _Fw.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_final_Mw.dat";
      myfile.open( myfilename );
      _Mw.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_final_Mv.dat";
      myfile.open( myfilename );
      _Mv.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_final_Ap.dat";
      myfile.open( myfilename );
      _Ap.PrintMatlab(myfile);
      myfile.close( );  
      std::string myfilename = "./results/out_final_B.dat";
      myfile.open( myfilename );
      _B.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_final_Mwp.dat";
      myfile.open( myfilename );
      _Mwp.PrintMatlab(myfile);
      myfile.close( );  
      myfilename = "./results/out_final_C.dat";
      myfile.open( myfilename );
      _C.PrintMatlab(myfile);
      myfile.close( );  


      // myfilename = "./results/out_essV.dat";
      // myfile.open( myfilename );
      // _essVhTDOF.Print(myfile,1);
      // myfile.close( );

      // myfilename = "./results/out_essQ.dat";
      // myfile.open( myfilename );
      // _essQhTDOF.Print(myfile,1);
      // myfile.close( );

      // std::cout<<"U essential nodes: ";_essVhTDOF.Print(std::cout, _essVhTDOF.Size());
      // std::cout<<"P essential nodes: ";_essQhTDOF.Print(std::cout, _essQhTDOF.Size());

    }

  }

}

























/*
void StokesSTOperatorAssembler::AssembleRhs( HypreParVector*& frhs ){
  // Initialise handy functions
  VectorFunctionCoefficient fFuncCoeff(_dim,_fFunc);
  VectorFunctionCoefficient nFuncCoeff(_dim,_nFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  FunctionCoefficient       gFuncCoeff(_gFunc);
  // - specify evaluation time
  fFuncCoeff.SetTime( _dt*(_myRank+1) );
  nFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  gFuncCoeff.SetTime( _dt*(_myRank+1) );

  // Assemble local part of rhs
  LinearForm *fform( new LinearForm ), *nform( new LinearForm ), *pform( new LinearForm );
  fform->Update( _VhFESpace ); nform->Update( _VhFESpace ); pform->Update( _VhFESpace );
  fform->AddDomainIntegrator(   new VectorDomainLFIntegrator(       fFuncCoeff       ) );  //int_\Omega f*v
  nform->AddBoundaryIntegrator( new VectorBoundaryLFIntegrator(     nFuncCoeff       ) );  //int_d\Omega \mu * du/dn *v
  pform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff, -1.0 ) );  //int_d\Omega -p*v*n
  fform->Assemble(); nform->Assemble(); pform->Assemble();
  if(_verbose ){
    // for ( int rank = 0; rank < _numProcs; ++rank ){
      // if ( rank==_myRank ){        
    std::cout<<"Rank "<<_myRank<<" - RHS assembled: "<<std::endl;
        // std::cout<<"\t\tf,\t\tn,\t\t-p:"<<std::endl;
        // for ( int i = 0; i < fform->Size(); ++i ){
        //   std::cout<<"Row: "<<i<<",\t\t"<<fform->GetData()[i]<<",\t\t"<<nform->GetData()[i]<<",\t\t"<<pform->GetData()[i]<<std::endl;
        // }
      // }
    MPI_Barrier(_comm);
    // } 
  }

  Vector fRhsLoc( fform->Size() );
  fRhsLoc.SetData( fform->StealData() );
  fRhsLoc += *nform;
  fRhsLoc += *pform;

#ifdef MULT_BY_DT
  fRhsLoc *= _dt;
#endif

  delete fform;     // once data is stolen, we can delete the linear form
  delete nform;     // once data is stolen, we can delete the linear form
  delete pform;     // once data is stolen, we can delete the linear form


  // - include initial conditions
  if( _myRank == 0 ){
    VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
    uFuncCoeff.SetTime( 0.0 );
    LinearForm *u0form( new LinearForm );
    u0form->Update( _VhFESpace );
    u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) ); //int_\Omega u0*v
    u0form->Assemble();

#ifdef MULT_BY_DT
    fRhsLoc += *u0form;
#else
    Vector temp = *u0form;
    temp *= (1./_dt);
    fRhsLoc += temp;
#endif

    if(_verbose){
      std::cout<<"Initial condition included "<<std::endl;
      // std::cout<<"(Row, value) = ";
      // for ( int i = 0; i < u0form->Size(); ++i ){
      //   std::cout<<"("<<i<<","<<u0form->GetData()[i]<<"), ";
      // }
      // std::cout<<std::endl;
    }

    delete u0form;
  }


  // Assemble global (parallel) rhs
  // Array<HYPRE_Int> rowStarts(2);
 //  rowStarts[0] = ( fRhsLoc.Size() )*_myRank;
 //  rowStarts[1] = ( fRhsLoc.Size() )*(_myRank+1);
  // HypreParVector *frhs = new HypreParVector( _comm, (fRhsLoc.Size())*_numProcs, fRhsLoc.GetData(), rowStarts.GetData() );
  // HypreParVector *frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
  // frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
  int colPart[2] = {_myRank*fRhsLoc.Size(), (_myRank+1)*fRhsLoc.Size()};
  frhs = new HypreParVector( _comm, fRhsLoc.Size()*_numProcs, fRhsLoc.StealData(), colPart );
  frhs->SetOwnership( 1 );

}



// Assemble space-time Stokes operator
//   A = [ FF  BB^T ]
//       [ BB  0    ],
// where FF contains space-time matrix for velocity,
// while BB is block-diagonal with -div operator in it  
void StokesSTOperatorAssembler::AssembleOperator( HypreParMatrix*& FFF, HypreParMatrix*& BBB ){

  //Assemble top-left block
  AssembleFF();

  // - convert to mfem operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( _FF, (void **) &FFref);
  FFF = new HypreParMatrix( FFref, false ); //"false" doesn't take ownership of data



  //Assemble bottom-left block
  AssembleBB();

  // - convert to mfem operator
  HYPRE_ParCSRMatrix  BBref;
  HYPRE_IJMatrixGetObject( _BB, (void **) &BBref);
  BBB = new HypreParMatrix( BBref, false ); //"false" doesn't takes ownership of data

}
*/













// Assemble operators appearing in diagonal of preconditioner
//          ⌈ FF^-1 \\\\  \\\\\  ⌉
//   P^-1 = |       AA^-1 \\\\\  |
//          ⌊             MMv^-1 ⌋
// (not necessarily in this order)
// where FF contains space-time matrix for w
void KHISTOperatorAssembler::AssemblePreconditioner( Operator*& Finv, Operator*& Ainv, Operator*& Mvinv, const int spaceTimeSolverType=0 ){

  //Assemble inverses
  AssembleFFinv( spaceTimeSolverType );
  AssembleAAinv( );
  AssembleMvinv( );

  Finv  = _FFinv;
  Ainv  = _Ainv;
  Mvinv = _Mvinv;

}









void KHISTOperatorAssembler::ExactSolution( HypreParVector*& w, HypreParVector*& p, HypreParVector*& v ) const{
  // Initialise handy functions
  FunctionCoefficient       wFuncCoeff(_wFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
  // - specify evaluation time
  // -- notice first processor actually refers to instant dt
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );

  GridFunction wFun( _WhFESpace );
  GridFunction pFun( _PhFESpace );
  GridFunction vFun( _VhFESpace );

  wFun.ProjectCoefficient( wFuncCoeff );
  pFun.ProjectCoefficient( pFuncCoeff );
  vFun.ProjectCoefficient( vFuncCoeff );
  

  Array<int> rowStartsW(2), rowStartsP(2), rowStartsV(2);
  rowStartsW[0] = ( wFun.Size() )*_myRank;
  rowStartsW[1] = ( wFun.Size() )*(_myRank+1);
  rowStartsP[0] = ( pFun.Size() )*_myRank;
  rowStartsP[1] = ( pFun.Size() )*(_myRank+1);
  rowStartsV[0] = ( vFun.Size() )*_myRank;
  rowStartsV[1] = ( vFun.Size() )*(_myRank+1);

  w = new HypreParVector( _comm, (wFun.Size())*_numProcs, wFun.StealData(), rowStartsW.GetData() );
  p = new HypreParVector( _comm, (pFun.Size())*_numProcs, pFun.StealData(), rowStartsP.GetData() );
  v = new HypreParVector( _comm, (vFun.Size())*_numProcs, vFun.StealData(), rowStartsV.GetData() );

  w->SetOwnership( 1 );
  p->SetOwnership( 1 );
  v->SetOwnership( 1 );

}



// Each processor computes L2 error of solution at its time-step
void KHISTOperatorAssembler::ComputeL2Error( const Vector& wh, const Vector& ph, const Vector& vh,
                                             double& err_w, double& err_p, double& err_v ) const{

  const GridFunction w( _WhFESpace, wh.GetData() );
  const GridFunction p( _PhFESpace, ph.GetData() );
  const GridFunction v( _VhFESpace, vh.GetData() );

  int order_quad = 5;
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i=0; i < Geometry::NumGeom; ++i){
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  FunctionCoefficient       wFuncCoeff(_wFunc);
  FunctionCoefficient       pFuncCoeff(_pFunc);
  VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
  wFuncCoeff.SetTime( _dt*(_myRank+1) );
  pFuncCoeff.SetTime( _dt*(_myRank+1) );
  vFuncCoeff.SetTime( _dt*(_myRank+1) );


  err_w  = w.ComputeL2Error(wFuncCoeff, irs);
  err_p  = p.ComputeL2Error(pFuncCoeff, irs);
  err_v  = v.ComputeL2Error(vFuncCoeff, irs);

  // for ( int i = 0; i < _numProcs; ++i ){
  //   if ( _myRank == i ){
  //     std::cout << "Instant t="       << _dt*(_myRank+1) << std::endl;
  //     std::cout << "|| uh - uEx ||_L2= " << err_u << "\n";
  //     std::cout << "|| ph - pEx ||_L2= " << err_p << "\n";
  //   }
  //   MPI_Barrier( _comm );
  // }
}




void KHISTOperatorAssembler::SaveExactSolution( const std::string& path="ParaView",
                                                const std::string& filename="STKHI_Ex" ) const{
  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set wpv paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w",   wFun );
    paraviewDC.RegisterField( "phi", pFun );
    paraviewDC.RegisterField( "v",   vFun );

    // main time loop
    for ( int t = 0; t < _numProcs+1; ++t ){
      FunctionCoefficient       wFuncCoeff(_wFunc);
      FunctionCoefficient       pFuncCoeff(_pFunc);
      VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
      wFuncCoeff.SetTime( t*_dt );
      pFuncCoeff.SetTime( t*_dt );
      vFuncCoeff.SetTime( t*_dt );

      wFun->ProjectCoefficient( wFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      vFun->ProjectCoefficient( vFuncCoeff );

      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}

// store given approximate solution in paraview format
void KHISTOperatorAssembler::SaveSolution( const HypreParVector& wh, const HypreParVector& ph, const HypreParVector& vh,
                                           const std::string& path="ParaView", const std::string& filename="STKHI" ) const{
  
  // gather parallel vector
  Vector *wGlb = wh.GlobalVector();
  Vector *pGlb = ph.GlobalVector();
  Vector *vGlb = vh.GlobalVector();


  if( _myRank == 0 ){

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w",   wFun );
    paraviewDC.RegisterField( "phi", pFun );
    paraviewDC.RegisterField( "v",   vFun );


    // store initial conditions
    FunctionCoefficient       wFuncCoeff(_wFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
    wFuncCoeff.SetTime( 0. );
    pFuncCoeff.SetTime( 0. );
    vFuncCoeff.SetTime( 0. );

    wFun->ProjectCoefficient( wFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    vFun->ProjectCoefficient( vFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // handy variables for time loop
    const int blockSizeW = wh.Size();
    const int blockSizeP = ph.Size();
    const int blockSizeV = vh.Size();
    Vector wLcl, pLcl, vLcl;
    Array<int> idxW(blockSizeW), idxP(blockSizeP), idxV(blockSizeV);

    // main time loop
    for ( int t = 1; t < _numProcs+1; ++t ){
      // - identify correct sub-vector idx in global vectors
      for ( int i = 0; i < blockSizeW; ++i ){
        idxW[i] = blockSizeW*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeP; ++i ){
        idxP[i] = blockSizeP*(t-1) + i;
      }
      for ( int i = 0; i < blockSizeV; ++i ){
        idxV[i] = blockSizeV*(t-1) + i;
      }

      // - extract subvector
      wGlb->GetSubVector( idxW, wLcl );
      pGlb->GetSubVector( idxP, pLcl );
      vGlb->GetSubVector( idxV, vLcl );
      
      // - assign to linked variables
      *wFun = wLcl;
      *pFun = pLcl;
      *vFun = vLcl;
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}

// This function is the same as above, but it doesn't rely on HypreParVector's
void KHISTOperatorAssembler::SaveSolution( const Vector& wh, const Vector& ph, const Vector& vh,
                                           const std::string& path="ParaView",
                                           const std::string& filename="STKHI" ) const{
  const int blockSizeW = wh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeV = vh.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( wh.GetData(), blockSizeW, MPI_DOUBLE, 0, 3*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 3*_myRank+1, _comm );
    MPI_Send( vh.GetData(), blockSizeV, MPI_DOUBLE, 0, 3*_myRank+2, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w",   wFun );
    paraviewDC.RegisterField( "phi", pFun );
    paraviewDC.RegisterField( "v",   vFun );


    // store initial conditions
    FunctionCoefficient       wFuncCoeff(_wFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    VectorFunctionCoefficient vFuncCoeff(_dim,_vFunc);
    wFuncCoeff.SetTime( 0.0 );
    pFuncCoeff.SetTime( 0.0 );
    vFuncCoeff.SetTime( 0.0 );
    
    wFun->ProjectCoefficient( wFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    vFun->ProjectCoefficient( vFuncCoeff );

    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();


    // this will store the approximate solution at current time-step
    Vector wLcl(blockSizeW), pLcl(blockSizeP), vLcl(blockSizeV);

    // handle first time-step separately
    *wFun = wh;
    *pFun = ph;
    *vFun = vh;
    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( wLcl.GetData(), blockSizeW, MPI_DOUBLE, t-1, 3*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 3*(t-1)+1, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( vLcl.GetData(), blockSizeV, MPI_DOUBLE, t-1, 3*(t-1)+2, _comm, MPI_STATUS_IGNORE );

      // - assign to linked variables
      *wFun = wLcl;
      *pFun = pLcl;
      *vFun = vLcl;
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}





// Saves a plot of the error
void KHISTOperatorAssembler::SaveError( const Vector& wh, const Vector& ph, const Vector& vh,
                                        const std::string& path="ParaView",
                                        const std::string& filename="STKHI_err" ) const{
  const int blockSizeW = wh.Size();
  const int blockSizeP = ph.Size();
  const int blockSizeV = vh.Size();


  // only the master will print stuff. The slaves just need to send their part of data
  if( _myRank != 0 ){
    MPI_Send( wh.GetData(), blockSizeW, MPI_DOUBLE, 0, 3*_myRank,   _comm );
    MPI_Send( ph.GetData(), blockSizeP, MPI_DOUBLE, 0, 3*_myRank+1, _comm );
    MPI_Send( vh.GetData(), blockSizeV, MPI_DOUBLE, 0, 3*_myRank+2, _comm );
  
  }else{

    // handy functions which will contain solution at single time-steps
    GridFunction *wFun = new GridFunction( _WhFESpace );
    GridFunction *pFun = new GridFunction( _PhFESpace );
    GridFunction *vFun = new GridFunction( _VhFESpace );

    // set up paraview data file
    ParaViewDataCollection paraviewDC( filename, _mesh );
    paraviewDC.SetPrefixPath(path);
    paraviewDC.SetLevelsOfDetail( 2 );
    paraviewDC.SetDataFormat(VTKFormat::BINARY);
    paraviewDC.SetHighOrderOutput(true);
    // - link wFun, pFun and vFun
    paraviewDC.RegisterField( "w-wh",   wFun );
    paraviewDC.RegisterField( "p-ph", pFun );
    paraviewDC.RegisterField( "v-vh",   vFun );

    // this will store the approximate solution at current time-step
    Vector wLcl(blockSizeW), pLcl(blockSizeP), vLcl(blockSizeV);

    // these will provide exact solution
    VectorFunctionCoefficient wFuncCoeff(_dim,_wFunc);
    FunctionCoefficient       pFuncCoeff(_pFunc);
    FunctionCoefficient       vFuncCoeff(_vFunc);

    // error at instant 0 is 0 (IC)
    *wFun = 0.;
    *pFun = 0.;
    *vFun = 0.;
    paraviewDC.SetCycle( 0 );
    paraviewDC.SetTime( 0.0 );
    paraviewDC.Save();

    // handle first time-step separately
    wFuncCoeff.SetTime( _dt );
    pFuncCoeff.SetTime( _dt );
    vFuncCoeff.SetTime( _dt );
    uFun->ProjectCoefficient( uFuncCoeff );
    pFun->ProjectCoefficient( pFuncCoeff );
    vFun->ProjectCoefficient( vFuncCoeff );

    wFun->operator-=( wh );
    pFun->operator-=( ph );
    vFun->operator-=( vh );

    paraviewDC.SetCycle( 1 );
    paraviewDC.SetTime( _dt );
    paraviewDC.Save();


    // main time loop
    for ( int t = 2; t < _numProcs+1; ++t ){

      MPI_Recv( wLcl.GetData(), blockSizeW, MPI_DOUBLE, t-1, 3*(t-1),   _comm, MPI_STATUS_IGNORE );
      MPI_Recv( pLcl.GetData(), blockSizeP, MPI_DOUBLE, t-1, 3*(t-1)+1, _comm, MPI_STATUS_IGNORE );
      MPI_Recv( vLcl.GetData(), blockSizeV, MPI_DOUBLE, t-1, 3*(t-1)+2, _comm, MPI_STATUS_IGNORE );
      
      // - assign to linked variables
      wFuncCoeff.SetTime( _dt*t );
      pFuncCoeff.SetTime( _dt*t );
      vFuncCoeff.SetTime( _dt*t );
      wFun->ProjectCoefficient( wFuncCoeff );
      pFun->ProjectCoefficient( pFuncCoeff );
      vFun->ProjectCoefficient( vFuncCoeff );
      wFun->operator-=( wLcl );
      pFun->operator-=( pLcl );
      vFun->operator-=( vLcl );
      
      // - store
      paraviewDC.SetCycle( t );
      paraviewDC.SetTime( _dt*t );
      paraviewDC.Save();

    }

    delete wFun;
    delete pFun;
    delete vFun;

  }
}










void KHISTOperatorAssembler::GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const{
  if(_mesh == NULL)
    std::cerr<<"Mesh not yet set"<<std::endl;
  else
    _mesh->GetCharacteristics( h_min, h_max, k_min, k_max );
}




void KHISTOperatorAssembler::PrintMatrices( const std::string& filename ) const{

  if( _myRank == 0){
    if( ! ( _FuAssembled && _MuAssembled && _MpAssembled && _ApAssembled && _BAssembled ) ){
      std::cerr<<"Make sure all matrices have been initialised, otherwise they can't be printed"<<std::endl;
      return;
    }


    std::ofstream myfile;
    std::string myfilename;

    myfilename = "./results/out_Fw.dat";
    myfile.open( myfilename );
    _Fw.PrintMatlab(myfile);
    myfile.close( );  
    myfilename = "./results/out_Mw.dat";
    myfile.open( myfilename );
    _Mw.PrintMatlab(myfile);
    myfile.close( );  
    myfilename = "./results/out_Mv.dat";
    myfile.open( myfilename );
    _Mv.PrintMatlab(myfile);
    myfile.close( );  
    myfilename = "./results/out_Ap.dat";
    myfile.open( myfilename );
    _Ap.PrintMatlab(myfile);
    myfile.close( );  
    std::string myfilename = "./results/out_B.dat";
    myfile.open( myfilename );
    _B.PrintMatlab(myfile);
    myfile.close( );  
    myfilename = "./results/out_Mwp.dat";
    myfile.open( myfilename );
    _Mwp.PrintMatlab(myfile);
    myfile.close( );  
    myfilename = "./results/out_C.dat";
    myfile.open( myfilename );
    _C.PrintMatlab(myfile);
    myfile.close( );  
  }
}




KHISTOperatorAssembler::~KHISTOperatorAssembler(){
  delete _Ainv;
  delete _MvFinv;
  delete _FFinv;
  delete _FFinvPrec;
  delete _FFF;
  if( _FFAssembled )
    HYPRE_IJMatrixDestroy( _FF );
  if( _AAAssembled )
    HYPRE_IJMatrixDestroy( _AA );
  if( _MMvAssembled )
    HYPRE_IJMatrixDestroy( _MMv );
  if( _BBAssembled )
    HYPRE_IJMatrixDestroy( _BB );
  if( _CCAssembled )
    HYPRE_IJMatrixDestroy( _CC );
  if( _MMwpAssembled )
    HYPRE_IJMatrixDestroy( _MMwp );

  delete _fwVarf;
  delete _mwVarf;
  delete _apVarf;
  delete _mvVarf;
  delete _bVarf;
  delete _cVarf;
  delete _mwpVarf;

  delete _WhFESpace;
  delete _PhFESpace;
  delete _VhFESpace;
  delete _WhFEColl;
  delete _PhFEColl;
  delete _VhFEColl;

  delete _mesh;
}













