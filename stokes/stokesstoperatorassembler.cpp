#include "stokesstoperatorassembler.hpp"

#include <mpi.h>
#include <string>
#include "HYPRE.h"
// #include "petsc.h"
#include "mfem.hpp"

using namespace mfem;





StokesSTOperatorAssembler::StokesSTOperatorAssembler( const MPI_Comm& comm, const char* meshName, const int ordV, const int ordP, const double dt, const double mu,
		                         							            void(  *f)(const Vector &, double, Vector &),
		                         							            void(  *u)(const Vector &, double, Vector &),
		                         							            double(*p)(const Vector &, double )          ):
	_comm(comm), _dt(dt), _mu(mu), _fFunc(f), _uFunc(u), _pFunc(p){

	MPI_Comm_size( comm, &_numProcs );
	MPI_Comm_rank( comm, &_myRank );

	// For each processor:
	//- generate mesh (don't refine)
	_mesh = new Mesh( meshName, 1, 0 );
  _dim = _mesh->Dimension();

  // - initialise FE info
  _VhFEColl  = new H1_FECollection( ordV, _dim );
  _QhFEColl  = new H1_FECollection( ordP, _dim );
  _VhFESpace = new FiniteElementSpace( _mesh, _VhFEColl, _dim );
  _QhFESpace = new FiniteElementSpace( _mesh, _QhFEColl );

   if (_myRank == 0 ){
      std::cout << "***********************************************************\n";
      std::cout << "dim(Vh) = " << _VhFESpace->GetTrueVSize() << "\n";
      std::cout << "dim(Qh) = " << _QhFESpace->GetTrueVSize() << "\n";
      std::cout << "***********************************************************\n";
   }

}



// Assemble operator on main diagonal of space-time matrix for velocity block:
//  Fu = M + mu*dt K
void StokesSTOperatorAssembler::AssembleFu( ){
  Array<int> essVhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
  }

  BilinearForm *fVarf( new BilinearForm(_VhFESpace) );
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
	fVarf->AddDomainIntegrator(new VectorMassIntegrator( one ));
	fVarf->AddDomainIntegrator(new VectorDiffusionIntegrator( muDt ));
  fVarf->Assemble();
  fVarf->Finalize();
  
  fVarf->FormSystemMatrix( essVhTDOF, _Fu );
  // - once the matrix is generated, we can get rid of the operator
  _Fu.SetGraphOwner(true);
  _Fu.SetDataOwner(true);
  fVarf->LoseMat();
  delete fVarf;

}



// Assemble operator on subdiagonal of space-time matrix for velocity block:
//  Mu = -M
void StokesSTOperatorAssembler::AssembleMu( ){
  Array<int> essVhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
  }

	BilinearForm *mVarf( new BilinearForm(_VhFESpace) );
  ConstantCoefficient mone( -1.0 );
	mVarf->AddDomainIntegrator(new VectorMassIntegrator( mone ));
  mVarf->Assemble();
  mVarf->Finalize();
  mVarf->FormSystemMatrix( essVhTDOF, _Mu );
  // - once the matrix is generated, we can get rid of the operator
  _Mu.SetGraphOwner(true);
  _Mu.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

}



// Assemble -divergence operator:
//  B = -div
void StokesSTOperatorAssembler::AssembleB( ){
  Array<int> essVhTDOF;
  Array<int> essQhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
    _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  }

	MixedBilinearForm *bVarf(new MixedBilinearForm( _VhFESpace, _QhFESpace ));
  ConstantCoefficient mone( -1.0 );
  bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(mone) );
  bVarf->Assemble();
  bVarf->Finalize();

  Array<int> emptyVhTDOF;
  Array<int> emptyQhTDOF;

  // bVarf->FormRectangularSystemMatrix( essQhTDOF, essVhTDOF, B );	//TODO: imposing essential nodes causes memory corruption??
  bVarf->FormRectangularSystemMatrix( emptyQhTDOF, emptyVhTDOF, _B );
	// - once the matrix is generated, we can get rid of the operator
  _B.SetGraphOwner(true);
  _B.SetDataOwner(true);
  bVarf->LoseMat();
  delete bVarf;
}





// Assemble operator on main diagonal of space-time matrix for pressure block:
//  Fp = M + mu*dt K
void StokesSTOperatorAssembler::AssembleFp( ){
  Array<int> essQhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    // _VhFESpace->GetEssentialTrueDofs( ess_bdr, essVhTDOF );
    _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  }

  BilinearForm *fVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient muDt( _mu*_dt );
  ConstantCoefficient one( 1.0 );
	fVarf->AddDomainIntegrator(new MassIntegrator( one ));
	fVarf->AddDomainIntegrator(new DiffusionIntegrator( muDt ));
  fVarf->Assemble();
  fVarf->Finalize();
  
  fVarf->FormSystemMatrix( essQhTDOF, _Fp );
  // - once the matrix is generated, we can get rid of the operator
  _Fp.SetGraphOwner(true);
  _Fp.SetDataOwner(true);
  fVarf->LoseMat();
  delete fVarf;

}



// Assemble operator on subdiagonal of space-time matrix for pressure block:
//  Mp = -M
void StokesSTOperatorAssembler::AssembleMp( ){
  Array<int> essQhTDOF;
  if ( _mesh->bdr_attributes.Size() ) {
    Array<int> ess_bdr( _mesh->bdr_attributes.Max() );
    ess_bdr = 1;
    _QhFESpace->GetEssentialTrueDofs( ess_bdr, essQhTDOF );
  }

	BilinearForm *mVarf( new BilinearForm(_QhFESpace) );
  ConstantCoefficient mone( -1.0 );
	mVarf->AddDomainIntegrator(new MassIntegrator( mone ));
  mVarf->Assemble();
  mVarf->Finalize();
  mVarf->FormSystemMatrix( essQhTDOF, _Mp );
  // - once the matrix is generated, we can get rid of the operator
  _Mp.SetGraphOwner(true);
  _Mp.SetDataOwner(true);
  mVarf->LoseMat();
  delete mVarf;

}








// Assemble space-time Stokes operator
//   A = [ FF  BB^T ]
//       [ BB  0    ],
// where FF contains space-time matrix for velocity,
// while BB is block-diagonal with -div operator in it	
// void StokesSTOperatorAssembler::AssembleSystem( BlockOperator*& stokesOp, BlockVector*& rhs ){
void StokesSTOperatorAssembler::AssembleSystem( HypreParMatrix*& FFF, HypreParMatrix*& BBB, 
                                                HypreParVector*& frhs ){

	// Define general structure of time-dep Stokes operator
	// Array<int> block_offsets(3); // number of variables + 1
 //  block_offsets[0] = 0;
 //  block_offsets[1] = _VhFESpace->GetTrueVSize(); // * _numProcs; TODO: yeah, I know the actual size is different, but seems like it wants size on single proc.
 //  block_offsets[2] = _QhFESpace->GetTrueVSize(); // * _numProcs;
 //  block_offsets.PartialSum();

	// stokesOp = new BlockOperator( block_offsets );
 //  rhs      = new BlockVector(   block_offsets );


  //*************************************************************************
	// Fill FF (top-left)
	//*************************************************************************
	// For each processor, define main operators
	// - main diagonal = M + mu*dt K
	AssembleFu();              // TODO: check it's not assembled already

	// - subidagonal = -M
	AssembleMu();              // TODO: check it's not assembled already


  // Create FF block ********************************************************
  // Initialize HYPRE matrix
  // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
  //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
  // - get info on matrix structure
	const int blockSizeFF = _Fu.NumRows();
 
  Array<int> nnzPerRowD( blockSizeFF );  	// num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<int> nnzPerRowO( blockSizeFF );  	// ..and in off-diagonal block
  const int  *offIdxsD = _Fu.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  const int  *offIdxsO = _Mu.GetI();
  for ( int i = 0; i < blockSizeFF; ++i ){
  	nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
    if ( _myRank > 0 ){
    	nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
    }else{
      nnzPerRowO[i] = 0;  // first block only has elements on block-diag
    }
  }


  // - initialise matrix
  HYPRE_IJMatrix FF;
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
  Array<int> colsGlbIdxD( _Fu.NumNonZeroElems() );
  for ( int i=0; i<_Fu.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = _Fu.GetJ()[i] + blockSizeFF*_myRank;
  }
  HYPRE_IJMatrixSetValues( FF, blockSizeFF, nnzPerRowD.GetData(),
  	                       rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), _Fu.GetData() );     // setvalues *copies* the data

  // -- off-diagonal block
  Array<int> rowsGlbIdxO( blockSizeFF );      // TODO: just use rowsGlbIdx once for both matrices?
  for ( int i = 0; i < blockSizeFF; ++i ){
    rowsGlbIdxO[i] = i + blockSizeFF*_myRank;
  }
  if ( _myRank > 0 ){
    Array<int> colsGlbIdxO( _Mu.NumNonZeroElems() );
    for ( int i=0; i<_Mu.NumNonZeroElems(); i++ ) {
      colsGlbIdxO[i] = _Mu.GetJ()[i] + blockSizeFF*(_myRank-1);
    }
    HYPRE_IJMatrixSetValues( FF, blockSizeFF, nnzPerRowO.GetData(),
    	                       rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), _Mu.GetData() );
  }


  // - assemble
  HYPRE_IJMatrixAssemble( FF );
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( FF, (void **) &FFref);

	// - convert to mfem operator
  // HypreParMatrix *FFF = new HypreParMatrix( FFref, true ); //"true" takes ownership of data
  FFF = new HypreParMatrix( FFref, true ); //"true" takes ownership of data


	// store in the stokes space-time operator
	// stokesOp->SetBlock( 0, 0, FFF );




  //*************************************************************************
	// Fill BB and BB^T (bottom-left / top-right)
	//*************************************************************************
	// For each processor, define -div operator
	AssembleB();


  // Assemble BB and BB^T blocks
  // - recover info on matrix structure
  const int numRowsPerBlockBB = _QhFESpace->GetTrueVSize();
  const int numColsPerBlockBB = _VhFESpace->GetTrueVSize();

  Array<int> nnzPerRow( numRowsPerBlockBB );  	// num of non-zero els per row in main (diagonal) block (for preallocation)
  const int  *offIdxs = _B.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  for ( int i = 0; i < numRowsPerBlockBB; ++i ){
  	nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
  }


  // - initialise matrix
  HYPRE_IJMatrix BB;
  HYPRE_IJMatrixCreate( _comm, numRowsPerBlockBB*_myRank, numRowsPerBlockBB*(_myRank+1)-1,
                               numColsPerBlockBB*_myRank, numColsPerBlockBB*(_myRank+1)-1, &BB );
  HYPRE_IJMatrixSetObjectType( BB, HYPRE_PARCSR );
  HYPRE_IJMatrixSetRowSizes( BB, nnzPerRow.GetData() );
  HYPRE_IJMatrixInitialize( BB );


  // - fill it with matrices assembled above
  Array<int> rowsGlbIdxBB( numRowsPerBlockBB );
  for ( int i = 0; i < numRowsPerBlockBB; ++i ){
  	rowsGlbIdxBB[i] = i + numRowsPerBlockBB*_myRank;
  }
  Array<int> colsGlbIdx( _B.NumNonZeroElems() );
  for ( int i=0; i<_B.NumNonZeroElems(); i++ ) {
    colsGlbIdx[i] = _B.GetJ()[i] + numColsPerBlockBB*_myRank;
  }
  HYPRE_IJMatrixSetValues( BB, numRowsPerBlockBB, nnzPerRow.GetData(),
  	                       rowsGlbIdxBB.GetData(), colsGlbIdx.GetData(), _B.GetData() );



  // - assemble
  HYPRE_IJMatrixAssemble( BB );
  HYPRE_ParCSRMatrix  BBref;
  HYPRE_IJMatrixGetObject( BB, (void **) &BBref);

	// - convert to mfem operator
	// HypreParMatrix *BBB = new HypreParMatrix( BBref, true ); //"true" takes ownership of data
  BBB = new HypreParMatrix( BBref, true ); //"true" takes ownership of data
  // HypreParMatrix *BBt = BBB->Transpose( );                 //TODO: does it reference the same data as in BBB?




	// // store in the stokes space-time operator
 //  stokesOp->SetBlock( 0, 1, BBt );
 //  stokesOp->SetBlock( 1, 0, BBB );



 //  // Clean up
 //  // HYPRE_IJMatrixDestroy( BB );
 //  // - set stokeOp as the owner of its own blocks
 //  stokesOp->owns_blocks = true;
 //  BBB->SetOwnerFlags( false, false, false );
 //  BBt->SetOwnerFlags( false, false, false );
 //  FFF->SetOwnerFlags( false, false, false );

 // //  // - clean up
 // //  HYPRE_IJMatrixDestroy( BB );
 //  // delete FFF;
	// // delete BBB;
	// // delete BBt;







  //*************************************************************************
	// Assemble rhs
	//*************************************************************************
  // Initialise handy functions
  FunctionCoefficient       pFuncCoeff(_pFunc);
  VectorFunctionCoefficient uFuncCoeff(_dim,_uFunc);
	VectorFunctionCoefficient fFuncCoeff(_dim,_fFunc);
	// - specify evaluation time
	pFuncCoeff.SetTime( _dt*(_myRank+1) );
	uFuncCoeff.SetTime( _dt*(_myRank+1) );
	fFuncCoeff.SetTime( _dt*(_myRank+1) );

	// Assemble local part of rhs
  LinearForm *fform( new LinearForm );
  fform->Update( _VhFESpace );
  fform->AddDomainIntegrator( new VectorDomainLFIntegrator( fFuncCoeff ) );					 //int_\Omega f*v
  fform->AddBoundaryIntegrator( new VectorBoundaryFluxLFIntegrator( pFuncCoeff ) );  //int_d\Omega -p*v*n + \mu*grad u*v *n (remember to put a minus in the function definition)
  fform->Assemble();
  Vector fRhsLoc( fform->Size() );        // should be blockSizeFF
  fRhsLoc.SetData( fform->StealData() );
  fRhsLoc *= _dt;
  delete fform;			// once data is stolen, we can delete the linear form


	// - include initial conditions
	if( _myRank == 0 ){
		uFuncCoeff.SetTime( 0.0 );
	  LinearForm *u0form( new LinearForm );
	  u0form->Update( _VhFESpace );
	  u0form->AddDomainIntegrator( new VectorDomainLFIntegrator( uFuncCoeff ) ); //int_\Omega u0*v
	  u0form->Assemble();

	  fRhsLoc += *u0form;
 	  delete u0form;
	}


	// Assemble global (parallel) rhs
	// Array<HYPRE_Int> rowStarts(2);
 //  rowStarts[0] = ( fRhsLoc.Size() )*_myRank;
 //  rowStarts[1] = ( fRhsLoc.Size() )*(_myRank+1);
  // HypreParVector *frhs = new HypreParVector( _comm, (fRhsLoc.Size())*_numProcs, fRhsLoc.GetData(), rowStarts.GetData() );
  // HypreParVector *frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
  // frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.GetData(), FFF->ColPart() );
  frhs = new HypreParVector( _comm, FFF->GetGlobalNumCols(), fRhsLoc.StealData(), FFF->ColPart() );
  frhs->SetOwnership( 1 );



  // - store in rhs
  // rhs->GetBlock( 0 ).SetData( frhs->StealData() );




// hypre_ParCSRMatrixOwnsData(     FFref) = false;
// hypre_ParCSRMatrixOwnsRowStarts(FFref) = false;
// hypre_ParCSRMatrixOwnsColStarts(FFref) = false;
// hypre_ParCSRMatrixOwnsData(     BBref) = false;
// hypre_ParCSRMatrixOwnsRowStarts(BBref) = false;
// hypre_ParCSRMatrixOwnsColStarts(BBref) = false;
FFF->SetOwnerFlags(true, true, true);
// BBB->SetOwnerFlags(true, true, true);
// HYPRE_IJMatrixDestroy( FF );
// HYPRE_IJMatrixDestroy( BB );

// HYPRE_ParCSRMatrixDestroy ( FFref );


// TODO: this is me trying to figure out what the hell is going on...
// {  SparseMatrix diag;
//   FFF->GetDiag( diag );
//   if ( _myRank == 0 ){
//     for ( int i = 0; i < diag.NumRows(); ++i ){
//       std::cout<<"Row: "<<i<<"-";
//       for ( int j = diag.GetI()[i]; j < diag.GetI()[i+1]; ++j ){
//         std::cout<<" Col "<<diag.GetJ()[j]<<": "<<diag.GetData()[j];
//       }
//       std::cout<<std::endl;
//     }
//   }
// }
//     {int uga;
//     std::cin>>uga;
//     MPI_Barrier( MPI_COMM_WORLD );}


// {  HypreParVector buga( *FFF, 1 );
//   FFF->Mult( *frhs, buga );
//   if ( _myRank==0 ){
//     for ( int i = 0; i < buga.Partitioning()[1] - buga.Partitioning()[0]; ++i ){
//       std::cout<<"Rank "<<_myRank<<": "<<buga.GetData()[i]<<std::endl;
//     }
//   }
// }

//     {int uga;
//     std::cin>>uga;
//     MPI_Barrier( MPI_COMM_WORLD );}








  /* 
  // Initialise Petsc matrix
  // TODO: there MUST be a better way to do this. All info is already neatly stored in M and F, and it seems like
  //       in order to assemble the block-matrix I need to reassemble them. Such a waste.
  // - get info on matrix structure
  const PetscInt blockSizeFF = F.NumRows();
  const PetscInt glbSizeFF   = F.NumRows() * _numProcs;
  

  Array<PetscInt> nnzPerRowD( blockSizeFF );  	// num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<PetscInt> nnzPerRowO( blockSizeFF );  	// ..and in off-diagonal block
  int  *offIdxsD = F.GetI(); // has size blockSizeFF+1, contains offsets for data in J for each row
  int  *offIdxsO = M.GetI();
  for ( int i = 0; i < blockSizeFF; ++i ){
  	nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
  	nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
  }



  // - initialise matrix
	PetscErrorCode petscErr;
  Mat FF;
  petscErr = MatCreateAIJ( _comm, blockSizeFF, blockSizeFF, glbSizeFF, glbSizeFF, 
  	                       nnzPerRowD.Max(), nnzPerRowD.GetData(),
  	                       nnzPerRowO.Max(), nnzPerRowO.GetData(), &FF ); CHKERRV(petscErr);
  
  // - fill it with data collected above - one row at a time
  // -- diagonal block
  for ( PetscInt i = 0; i < blockSizeFF; ++i ){
  	const PetscInt rowGlbIdx =  i + blockSizeFF * _myRank;
  	Array<PetscInt> colGlbIdx( nnzPerRowD[i] );
  	for ( int j = 0; j < nnzPerRowD[i]; ++j ){
  		colGlbIdx[j] = (F.GetJ())[ offIdxsD[i] +j ] + blockSizeFF * _myRank; 
  	}
	  petscErr = MatSetValues( FF,         1, &rowGlbIdx,
	  	                       nnzPerRowD[i], colGlbIdx.GetData(),
	  	                       &((F.GetData())[offIdxsD[i]]), INSERT_VALUES ); CHKERRV(petscErr);
  }
  // -- off-diagonal block
  for ( PetscInt i = 0; i < blockSizeFF; ++i ){
  	const PetscInt rowGlbIdx =  i + blockSizeFF * _myRank;
  	Array<PetscInt> colGlbIdx( nnzPerRowO[i] );
  	for ( int j = 0; j < nnzPerRowO[i]; ++j ){
  		colGlbIdx[j] = (M.GetJ())[ offIdxsO[i] +j ] + blockSizeFF * (_myRank-1); //will be skipped for _myRank==0
  	}
	  petscErr = MatSetValues( FF,         1, &rowGlbIdx,
	  	                       nnzPerRowO[i], colGlbIdx.GetData(),
	  	                       &((M.GetData())[offIdxsO[i]]), INSERT_VALUES ); CHKERRV(petscErr);
  }

  // - assemble
	petscErr = MatAssemblyBegin( FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);
	petscErr = MatAssemblyEnd(   FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);


	// convert to mfem operator
	PetscParMatrix *FFF = new PetscParMatrix( FF, true ); //"true" increases ref counts: now even if FF dies, there should be no memory loss


	// // finally store in the stokes space-time operator
	stokesOp->SetBlock( 0, 0, FFF );

  petscErr = MatDestroy( &FF ); CHKERRV(petscErr);*/












  /* Yeah, it'd be nice to use block matrices, but it seems like building them is a pain: delete this code
  PetscErrorCode petscErr;
  Mat FF;
  const int maxNZBlocksPerRow = 2;
  const int glbSizeFF   = _VhFESpace->GetVSize()*_numProcs;
  const int blockSizeFF = _VhFESpace->GetVSize();
  petscErr = MatCreateBAIJ( _comm, blockSizeFF, blockSizeFF, blockSizeFF, glbSizeFF, glbSizeFF,
                            1, NULL, 1, NULL, &FF ); CHKERRV(petscErr);

  // petscErr = MatCreateBlockMat( _comm, glbSizeFF, glbSizeFF, blockSizeFF,
  // 	                            maxNZBlocksPerRow, maxNZPerBlockRow.GetData(), &FF ); CHKERRV(petscErr); // maxNZPerBlockRow is actually nnz blocks, rather than elems?

  petscErr = MatSetUp( FF ); CHKERRV(petscErr);
  
  // for each proc, build a map for local 2 global rows and col indeces (block to gl matrix),
  //  for both the block on the Diagonal and that on the SubDiagonal
	ISLocalToGlobalMapping l2gColMapD, l2gRowMapD, l2gColMapSD, l2gRowMapSD;
  PetscInt *l2gColDIdx, *l2gRowDIdx, *l2gColSDIdx, *l2gRowSDIdx;
  petscErr = PetscMalloc( sizeof(PetscInt), &l2gRowDIdx);  CHKERRV(petscErr);	// shouldn't need to petscfree if using PETSC_OWN_POINTER in ISLocalToGlobalMappingCreate()
  petscErr = PetscMalloc( sizeof(PetscInt), &l2gColDIdx);  CHKERRV(petscErr); //  otherwise, just use PETSC_COPY_VALUES and whatever
  petscErr = PetscMalloc( sizeof(PetscInt), &l2gRowSDIdx); CHKERRV(petscErr);
  petscErr = PetscMalloc( sizeof(PetscInt), &l2gColSDIdx); CHKERRV(petscErr);

  *l2gRowDIdx  = _myRank;
  *l2gColDIdx  = _myRank;
	*l2gRowSDIdx = _myRank;
  *l2gColSDIdx = _myRank-1; // should be invalid for myRank = 1 	

  petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gRowDIdx,  PETSC_COPY_VALUES, &l2gRowMapD  ); CHKERRV(petscErr);
  petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gColDIdx,  PETSC_COPY_VALUES, &l2gColMapD  ); CHKERRV(petscErr);
  petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gRowSDIdx, PETSC_COPY_VALUES, &l2gRowMapSD ); CHKERRV(petscErr);
  petscErr = ISLocalToGlobalMappingCreate( _comm, blockSizeFF, 1, l2gColSDIdx, PETSC_COPY_VALUES, &l2gColMapSD ); CHKERRV(petscErr);
 

	petscErr = PetscFree( l2gRowDIdx  ); CHKERRV(petscErr);
	petscErr = PetscFree( l2gColDIdx  ); CHKERRV(petscErr);
	petscErr = PetscFree( l2gRowSDIdx ); CHKERRV(petscErr);
	petscErr = PetscFree( l2gColSDIdx ); CHKERRV(petscErr);


  // fill each block: main diagonal
	petscErr = MatSetLocalToGlobalMapping( FF, l2gRowMapD, l2gColMapD); CHKERRV(petscErr);
  int  *rowIdxD = F.GetI();
  int  *colIdxD = F.GetJ();
  double *dataD = F.GetData();
  // F.LoseData();	// we can get rid of the matrix now
  // delete F;
	// petscErr = MatSetValuesBlockedLocal( FF, blockSizeFF, rowIdxD, blockSizeFF, colIdxD, dataD, INSERT_VALUES ); CHKERRV(petscErr);

  petscErr = ISLocalToGlobalMappingDestroy( &l2gRowMapD  ); CHKERRV(petscErr);
  petscErr = ISLocalToGlobalMappingDestroy( &l2gColMapD  ); CHKERRV(petscErr);
  petscErr = ISLocalToGlobalMappingDestroy( &l2gRowMapSD ); CHKERRV(petscErr);
  petscErr = ISLocalToGlobalMappingDestroy( &l2gColMapSD ); CHKERRV(petscErr);



	petscErr = MatAssemblyBegin( FF, MAT_FLUSH_ASSEMBLY ); CHKERRV(petscErr);
	petscErr = MatAssemblyEnd(   FF, MAT_FLUSH_ASSEMBLY ); CHKERRV(petscErr);

  // fill each block: sub-diagonal
	petscErr = MatSetLocalToGlobalMapping( FF, l2gRowMapSD, l2gColMapSD); CHKERRV(petscErr);
  int  *rowIdxSD = M.GetI();
  int  *colIdxSD = M.GetJ();
  double *dataSD = M.GetData();
  M.LoseData();	// we can get rid of the matrix now
  // delete M;
	petscErr = MatSetValuesBlockedLocal( FF, blockSizeFF, rowIdxSD, blockSizeFF, colIdxSD, dataSD, INSERT_VALUES ); CHKERRV(petscErr);

	// assemble the whole thing
	petscErr = MatAssemblyBegin( FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);
	petscErr = MatAssemblyEnd(   FF, MAT_FINAL_ASSEMBLY ); CHKERRV(petscErr);

  petscErr = MatDestroy( &FF ); CHKERRV(petscErr); */


	// // convert to mfem operator
	// PetscParMatrix *FFF = new PetscParMatrix( FF, true ); //"true" increases ref counts: now even if FF dies, there should be no memory loss

	// // finally store in the stokes space-time operator
	// stokesOp->SetBlock( 0, 0, FFF );



	// petscErr = MatView( FF, 	PETSC_VIEWER_STDOUT_(_comm) ); CHKERRV(petscErr);








  /* Again, Petsc is a pain
 //  // Assemble BB and BB^T blocks
 //  // - recover info on matrix structure
 //  const int glbNumRows = _QhFESpace->GetVSize() * _numProcs;
 //  const int glbNumCols = _VhFESpace->GetVSize() * _numProcs;
	// Array<PetscInt> rowStarts(_numProcs+1);
	// Array<PetscInt> colStarts(_numProcs+1);
 //  rowStarts[0] = 0;  colStarts[0] = 0;
 //  for ( int i = 1; i < _numProcs+1; ++i ){
 //  	rowStarts[i] = _QhFESpace->GetVSize();
 //  	colStarts[i] = _VhFESpace->GetVSize();
 //  }
 //  rowStarts.PartialSum();  colStarts.PartialSum();
 //  // - assemble actual matrix
  // TODO: seems like this function doesn't build a block-diagonal matrix, with blocks specified in B, but
  //  rather assumes B is already block diagonal, and somehow it breaks it into a parallel matrix??
	// PetscParMatrix *BB = new PetscParMatrix( _comm, glbNumRows, glbNumCols,
	// 	                  										 rowStarts.GetData(), colStarts.GetData(),
	//                     										 &B, mfem::Operator::PETSC_MATAIJ ); //PETSC_MATNEST is unsupported?
 //  PetscParMatrix *BBt = BB->Transpose( true );

	// if (_myRank == 0 ){
	//   std::cout << "***********************************************************\n";
	//   std::cout << "B  is a  " << B.NumRows()   << "x" << B.NumCols()   << " matrix\n";
	//   std::cout << "BB is a  " << BB->NumRows()  << "x" << BB->NumCols()  << " matrix\n";
	//   std::cout << "BBt is a " << BBt->NumRows() << "x" << BBt->NumCols() << " matrix\n";
	//   std::cout << "A is a "   << stokesOp->NumRows() << "x" << stokesOp->NumCols() << " matrix\n";
	//   // std::cout << "F is a " << F.NumRows() << "x" << F.NumCols() << " matrix\n";
	//   // std::cout << "M is a " << M.NumRows() << "x" << M.NumCols() << " matrix\n";
	//   std::cout << "***********************************************************\n";
	// }

	// // finally store in the stokes space-time operator
 //  stokesOp->SetBlock( 0, 1, BBt);
 //  stokesOp->SetBlock( 1, 0, BB );

  // TODO: think about how to deal with data ownership
	// BB->ReleaseMat(true);

	// stokesOp.owns_block = true;
	// // B.LoseData();	// we can get rid of the matrix now
 //  // delete B;
 //  PetscParMatrix *BBt = BB->Transpose( true );

	// // finally store in the stokes space-time operator
 //  stokesOp->SetBlock( 0, 1, BBt);
 //  stokesOp->SetBlock( 1, 0, BB );

 */



//  //  clean up
//  //  delete FF;
//  //  delete FFF; // should I delete the underlying FFF operator?
// 	// delete BB;
// 	// delete BBt;
}






StokesSTOperatorAssembler::~StokesSTOperatorAssembler(){
	delete _VhFESpace;
	delete _QhFESpace;
	delete _VhFEColl;
	delete _QhFEColl;
	delete _mesh;
}







