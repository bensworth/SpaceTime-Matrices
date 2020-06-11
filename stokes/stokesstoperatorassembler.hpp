#ifndef STOKESSTOPERATORASSEMBLER_HPP
#define STOKESSTOPERATORASSEMBLER_HPP

#include <mpi.h>
#include "mfem.hpp"



using namespace mfem;


/** Placeholder class for handling space-time stokes
- at least until I sort out how to make Ben's class work*/
class StokesSTOperatorAssembler{

private:
	
	// info for parallelisation
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;


	// problem parameters
	const double _dt; 	//time step (constant)
	const double _mu;		//viscosity
  int _dim;						//domain dimension (R^d)
	void(  *_fFunc)( const Vector &, double, Vector & );	// function returning rhs (time-dep)
	void(  *_uFunc)( const Vector &, double, Vector & );	// function returning velocity solution (time-dep, used to implement IC and BC)
	double(*_pFunc)( const Vector &, double )          ; // function returning pressure solution (time-dep, used to implement IC and BC)


	// info on FE
  Mesh* _mesh;
  FiniteElementCollection *_VhFEColl;
  FiniteElementCollection *_QhFEColl;
  FiniteElementSpace      *_VhFESpace;
  FiniteElementSpace      *_QhFESpace;

  // relevant matrices
  SparseMatrix _Mu;
  SparseMatrix _Fu;
  SparseMatrix _Mp;
  SparseMatrix _Fp;
  SparseMatrix _B;




public:
	StokesSTOperatorAssembler( const MPI_Comm& comm, const char *meshName, const int ordV, const int ordP, const double dt, const double mu,
		                         void(  *f)(const Vector &, double, Vector &),
		                         void(  *u)(const Vector &, double, Vector &),
		                         double(*p)(const Vector &, double )          );
	~StokesSTOperatorAssembler();

	// void AssembleSystem( BlockOperator*& stokesOp, BlockVector*& rhs );
	void AssembleSystem( HypreParMatrix*& FFF, HypreParMatrix*& BBB, 
                       HypreParVector*& frhs );



private:
	void AssembleFu();
	void AssembleMu();
	void AssembleFp();
	void AssembleMp();
	void AssembleB();







}; //StokesSTOperatorAssembler



#endif //STOKESSTOPERATORASSEMBLER