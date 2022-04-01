#include "mfem.hpp"
#include "HYPRE.h"
#include "vectorconvectionintegrator.hpp"
#include <string>
#include <iostream>

using namespace mfem;

// void AssembleParallelMatrix( const SparseMatrix& Mu, const SparseMatrix& Fu, HypreParMatrix* &FFuHypre );
// void UpdateParallelMatrix(   const SparseMatrix& FuNew, HypreParMatrix* &FFuHypre );

void fun(const Vector & x, const double t, Vector & w){
  double xx(x(0));
  double yy(x(1));
  w(0) = - t * 2.*(2*yy-1)*(4*xx*xx-4*xx+1); // (-t*2.*yy*(1-xx*xx) mapped from -1,1 to 0,1)
  w(1) =   t * 2.*(2*xx-1)*(4*yy*yy-4*yy+1); // ( t*2.*xx*(1-yy*yy) mapped from -1,1 to 0,1)
}


int main(int argc, char *argv[]){

  //*************************************************************************
  // Initialise
  //*************************************************************************

  // Mpi::Init();
  // int myRank = Mpi::WorldRank();
  // Hypre::Init();
  int numProcs, myRank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


  if ( myRank==0 ) 
	  std::cout<<"Initialise FE space\n";
	
	std::string meshName = "./meshes/tri-square-testAn.mesh";
  Mesh* mesh = new Mesh( meshName.c_str(), 1, 1 );
  for (int i = 0; i < 0; i++)
    mesh->UniformRefinement();

  int dim = mesh->Dimension();
	FiniteElementCollection* UhFEColl  = new H1_FECollection( 1, dim );
  FiniteElementSpace*      UhFESpace = new FiniteElementSpace( mesh, UhFEColl, dim );

 	if ( myRank==0 ) 
	  std::cout<<"Initialise Galerkin matrices\n";

  BilinearForm muVarf(UhFESpace);
  ConstantCoefficient myCoeff( myRank+1 );
  muVarf.AddDomainIntegrator(new VectorMassIntegrator( myCoeff ));
  muVarf.Assemble();
  muVarf.Finalize();
 	SparseMatrix Mu = muVarf.SpMat();
  Mu.Finalize();
 	if ( myRank==0 ) 
 		std::cout<<"Mu has "<< Mu.NumNonZeroElems()<<" non-zero elems\n";

  BilinearForm fuVarf(UhFESpace);
  fuVarf.AddDomainIntegrator(new VectorMassIntegrator( myCoeff ));
  fuVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( myCoeff ));
  fuVarf.Assemble();
  fuVarf.Finalize();
 	SparseMatrix Fu = fuVarf.SpMat();
  Fu.Finalize();
 	if ( myRank==0 ) 
 		std::cout<<"Fu has "<< Fu.NumNonZeroElems()<<" non-zero elems\n";


  VectorFunctionCoefficient vecCoeff( dim, fun );
  vecCoeff.SetTime( myRank+1 );
  BilinearForm fuNewVarf(UhFESpace);
  ConstantCoefficient myNewCoeff( 2*(myRank+1) );
  fuNewVarf.AddDomainIntegrator(new VectorMassIntegrator( myNewCoeff ));
  fuNewVarf.AddDomainIntegrator(new VectorDiffusionIntegrator( myNewCoeff ));
  fuNewVarf.AddDomainIntegrator(new VectorConvectionIntegrator( vecCoeff, 1. )); // dt*W1(w)
  fuNewVarf.Assemble();
  fuNewVarf.Finalize();
 	SparseMatrix FuNew = fuNewVarf.SpMat();
  FuNew.Finalize();
 	if ( myRank==0 ) 
 		std::cout<<"FuNew has "<< FuNew.NumNonZeroElems()<<" non-zero elems\n";



/*
  StopWatch stopwatch1, stopwatch2, stopwatch3;



 	if ( myRank==0 ) 
	  std::cout<<"Assemble Parallel matrix\n";
  HypreParMatrix* FFuHypre;
  MPI_Barrier(MPI_COMM_WORLD);
  stopwatch1.Start();
  AssembleParallelMatrix( Mu, Fu, FFuHypre );
  MPI_Barrier(MPI_COMM_WORLD);
  stopwatch1.Stop();
  std::cout<<"\tRank "<<myRank<<", Total time: "<<stopwatch1.RealTime()<<"\n";
  // std::string matName = "debug_matrixFFuOrig.dat";
  // FFuHypre->Print( matName.c_str() );

  MPI_Barrier(MPI_COMM_WORLD);


 	if ( myRank==0 ) 
	  std::cout<<"Update Parallel matrix\n";
  MPI_Barrier(MPI_COMM_WORLD);
  stopwatch2.Start();
  UpdateParallelMatrix( FuNew, FFuHypre );
  MPI_Barrier(MPI_COMM_WORLD);
  stopwatch2.Stop();
  std::cout<<"\tRank "<<myRank<<", Total time: "<<stopwatch2.RealTime()<<"\n";
  // matName = "debug_matrixFFuUpdated.dat";
  // FFuHypre->Print( matName.c_str() );
  delete FFuHypre;


  MPI_Barrier(MPI_COMM_WORLD);


 	if ( myRank==0 ) 
	  std::cout<<"Rebuild Parallel matrix\n";
  MPI_Barrier(MPI_COMM_WORLD);
  stopwatch3.Start();
  AssembleParallelMatrix( Mu, Fu, FFuHypre );
  MPI_Barrier(MPI_COMM_WORLD);
  stopwatch3.Stop();
  std::cout<<"\tRank "<<myRank<<", Total time: "<<stopwatch3.RealTime()<<"\n";
  // matName = "debug_matrixFFuRebuilt.dat";
  // FFuHypre->Print( matName.c_str() );
  delete FFuHypre;




  // Array<HYPRE_BigInt> rowStart(2), colStart(2);
  // rowStart[0] = myRank*lclSize; rowStart[1] = (myRank+1)*lclSize-1;
  // colStart[0] = 0;              colStart[1] =   numProcs*lclSize-1;

  // Array<HYPRE_BigInt> colMap(lclSize);
  // for ( int i = 0; i < colMap.Size(); ++i ){
  // 	if ( myRank < numProcs-1 ){
  // 	  colMap[i] = myRank*lclSize + i;
  // 	}else{
  // 		colMap[i] = (myRank-1)*lclSize + i;
  // 	}
  // }



  // HypreParMatrix* FFaHypre = new HypreParMatrix( MPI_COMM_WORLD, numProcs * Mu.NumRows(), numProcs * Mu.NumCols(),
  //                                 rowStart.GetData(), colStart.GetData(), 
  //                                 &Mu, &Mu, colMap.GetData() ); //"false" does not take ownership of data
  // HypreParMatrix* FFaHypre = new HypreParMatrix( MPI_COMM_WORLD, numProcs * Mu.NumRows(), numProcs * Mu.NumCols(),
  //                                 rowStart.GetData(), colStart.GetData(), 
  //                                 &Mu ); //"false" does not take ownership of data
  // HypreParMatrix* FFaHypre = new HypreParMatrix( MPI_COMM_WORLD, glbSize, rowStart.GetData(), &Mu );
  // hypre_ParCSRMatrix* A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, glbSize, glbSize,
  //                              										rowStart, rowStart,
  //                              										0, Mu.NumNonZeroElems(),0);


 	// // if ( myRank==0 ) 
	 //  std::cout<<"Print Parallel matrix\n";

  // std::string matName2 = "debug_matrixFFa.dat";
  // FFaHypre->Print( matName2.c_str(), myRank*lclSize, 0 );

  // delete FFaHypre;

*/
  delete UhFESpace;
  delete UhFEColl;
  delete mesh;

  return 0;

 }









/*
void UpdateParallelMatrix(   const SparseMatrix& FuNew, HypreParMatrix* &FFuHypre ){

  int myRank = Mpi::WorldRank();
	HYPRE_BigInt lclSize = FuNew.NumRows();

  Array<int> nnzPerRowD( lclSize );   // num of non-zero els per row in main (diagonal) block (for preallocation)
  const int  *offIdxsD = FuNew.GetI(); // has size lclSize+1, contains offsets for data in J for each row
  for ( int i = 0; i < lclSize; ++i ){
    nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
  }


  HYPRE_ParCSRMatrix* FFuTemp = (HYPRE_ParCSRMatrix*) FFuHypre;

  // - initialise matrix
  HYPRE_IJMatrixInitialize( *FFuTemp );

  // - fill it with matrices assembled above
  // -- diagonal block
  Array<int> rowsGlbIdxD( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    rowsGlbIdxD[i] = i + lclSize*myRank;
  }
  Array<int> colsGlbIdxD( FuNew.NumNonZeroElems() );
  for ( int i=0; i<FuNew.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = FuNew.GetJ()[i] + lclSize*myRank;
  }
  HYPRE_IJMatrixSetValues( *FFuTemp, lclSize, nnzPerRowD.GetData(),
                           rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), FuNew.GetData() );     // setvalues *copies* the data


  // - assemble
  HYPRE_IJMatrixAssemble( *FFuTemp );

  // - convert to a MFEM operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( *FFuTemp, (void **) &FFref);
  FFuHypre = new HypreParMatrix( FFref, true ); //"true" takes ownership of data


}














void AssembleParallelMatrix( const SparseMatrix& Mu, const SparseMatrix& Fu, HypreParMatrix* &FFuHypre ){

  int myRank = Mpi::WorldRank();
	HYPRE_BigInt lclSize = Mu.NumRows();

  Array<int> nnzPerRowD( lclSize );   // num of non-zero els per row in main (diagonal) block (for preallocation)
  Array<int> nnzPerRowO( lclSize );   // ..and in off-diagonal block
  const int  *offIdxsD = Fu.GetI(); // has size lclSize+1, contains offsets for data in J for each row
  const int  *offIdxsO = Mu.GetI();
  for ( int i = 0; i < lclSize; ++i ){
    nnzPerRowD[i] = offIdxsD[i+1] - offIdxsD[i];
    if ( myRank > 0 ){
      nnzPerRowO[i] = offIdxsO[i+1] - offIdxsO[i];
    }else{
      nnzPerRowO[i] = 0;  // first block only has elements on block-diag
    }
  }

  HYPRE_IJMatrix FFuTemp;

  // - initialise matrix
  HYPRE_IJMatrixCreate( MPI_COMM_WORLD, lclSize*myRank, lclSize*(myRank+1)-1,
                                        lclSize*myRank, lclSize*(myRank+1)-1, &FFuTemp );
  HYPRE_IJMatrixSetObjectType( FFuTemp, HYPRE_PARCSR );
  HYPRE_IJMatrixSetDiagOffdSizes( FFuTemp, nnzPerRowD.GetData(), nnzPerRowO.GetData() );    // this gives issues :/
  HYPRE_IJMatrixInitialize( FFuTemp );


  // - fill it with matrices assembled above
  // -- diagonal block
  Array<int> rowsGlbIdxD( lclSize );
  for ( int i = 0; i < lclSize; ++i ){
    rowsGlbIdxD[i] = i + lclSize*myRank;
  }
  Array<int> colsGlbIdxD( Fu.NumNonZeroElems() );
  for ( int i=0; i<Fu.NumNonZeroElems(); i++ ) {
    colsGlbIdxD[i] = Fu.GetJ()[i] + lclSize*myRank;
  }
  HYPRE_IJMatrixSetValues( FFuTemp, lclSize, nnzPerRowD.GetData(),
                           rowsGlbIdxD.GetData(), colsGlbIdxD.GetData(), Fu.GetData() );     // setvalues *copies* the data

  // -- off-diagonal block
  Array<int> rowsGlbIdxO( lclSize );      // TODO: just use rowsGlbIdx once for both matrices?
  for ( int i = 0; i < lclSize; ++i ){
    rowsGlbIdxO[i] = i + lclSize*myRank;
  }
  if ( myRank > 0 ){
    Array<int> colsGlbIdxO( Mu.NumNonZeroElems() );
    for ( int i=0; i<Mu.NumNonZeroElems(); i++ ) {
      colsGlbIdxO[i] = Mu.GetJ()[i] + lclSize*(myRank-1);
    }
    HYPRE_IJMatrixSetValues( FFuTemp, lclSize, nnzPerRowO.GetData(),
                             rowsGlbIdxO.GetData(), colsGlbIdxO.GetData(), Mu.GetData() );
  }


  // - assemble
  HYPRE_IJMatrixAssemble( FFuTemp );

  // - convert to a MFEM operator
  HYPRE_ParCSRMatrix  FFref;
  HYPRE_IJMatrixGetObject( FFuTemp, (void **) &FFref);
  FFuHypre = new HypreParMatrix( FFref, true ); //"true" takes ownership of data

  return;

}
*/