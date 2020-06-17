#ifndef STOKESSTOPERATORASSEMBLER_HPP
#define STOKESSTOPERATORASSEMBLER_HPP

#include <mpi.h>
#include "mfem.hpp"



using namespace mfem;




// "Inverse" of space-time matrix
class SpaceTimeSolver: public Operator{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;

  double _tol;
  
  const SparseMatrix *_F;
  const SparseMatrix *_M;
  
  mutable HypreParVector* _X;
  mutable HypreParVector* _Y;



public:

	SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F, const SparseMatrix* M, const double tol=1e-12 );

  void Mult( const Vector& x, Vector& y ) const;

  inline void SetF( const SparseMatrix* F ){ _F = F; height = F->Height(); width = F->Width(); }
  inline void SetM( const SparseMatrix* M ){ _M = M; height = M->Height(); width = M->Width();  }

	~SpaceTimeSolver();

}; //SpaceTimeSolver







// Approximation to pressure Schur complement
class StokesSTPreconditioner: public Operator{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;

  double _dt;
  double _mu;
  double _tol;
  const SparseMatrix *_Ap;
  const SparseMatrix *_Mp;



public:

	StokesSTPreconditioner( const MPI_Comm& comm, const double dt, const double mu,
		                      const SparseMatrix* Ap, const SparseMatrix* Mp, const double tol=1e-12 );

  void Mult( const Vector& x, Vector& y ) const;

  inline void SetAp( const SparseMatrix* Ap ){ _Ap = Ap; height = Ap->Height(); width = Ap->Width(); }
  inline void SetMp( const SparseMatrix* Mp ){ _Mp = Mp; height = Mp->Height(); width = Mp->Width();  }


}; //StokesSTPreconditioner














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
	double(*_pFunc)( const Vector &, double )          ;  // function returning pressure solution (time-dep, used to implement IC and BC)

	const double _tol;	//tolerance for solvers

	// info on FE
  Mesh *_mesh;
  FiniteElementCollection *_VhFEColl;
  FiniteElementCollection *_QhFEColl;
  FiniteElementSpace      *_VhFESpace;
  FiniteElementSpace      *_QhFESpace;

  // relevant matrices
  // - blocks for single time-steps
  SparseMatrix _Mu;
  SparseMatrix _Fu;
  SparseMatrix _Mp;
  SparseMatrix _Ap;
  SparseMatrix _B;
  bool _MuAssembled;
  bool _FuAssembled;
  bool _MpAssembled;
  bool _ApAssembled;
  bool _BAssembled;

  // - space-time blocks
  HYPRE_IJMatrix _FF;								// Space-time velocity block
  HYPRE_IJMatrix _BB;								// space-time -div block
  StokesSTPreconditioner _pSchur;   // Approximation to space-time pressure Schur complement
  SpaceTimeSolver        _FFinv;    // Space-time velocity block solver
  bool _FFAssembled;
  bool _BBAssembled;
	bool _pSAssembled;
	bool _FFinvAssembled;


  // // - full-fledged operators
  // BlockOperator _STstokes;					 // space-time stokes operator
  // BlockOperator _STstokesPrec;      // space-time block preconditioner




public:
	StokesSTOperatorAssembler( const MPI_Comm& comm, const char *meshName, const int refLvl,
		                         const int ordV, const int ordP, const double dt, const double mu,
		                         void(  *f)(const Vector &, double, Vector &),
		                         void(  *u)(const Vector &, double, Vector &),
		                         double(*p)(const Vector &, double ),
		                         const double tol=1e-12 );
	~StokesSTOperatorAssembler();


	void AssembleOperator( HypreParMatrix*& FFF, HypreParMatrix*& BBB );

	void AssemblePreconditioner( Operator*& Finv, Operator*& XXX );

	void AssembleRhs( HypreParVector*& frhs );


	void ExactSolution( HypreParVector*& u, HypreParVector*& p );

	void TimeStepVelocity( const HypreParVector& rhs, HypreParVector*& sol );
	void TimeStepPressure( const HypreParVector& rhs, HypreParVector*& sol );

	void ComputeL2Error( const HypreParVector& uh, const HypreParVector& ph );


private:
	// assemble blocks for single time-step 
	void AssembleFu();
	void AssembleMu();
	void AssembleAp();
	void AssembleMp();
	void AssembleB();

	// assemble blocks for whole Space-time operators 
	void AssembleFF();
	void AssembleBB();
	void AssemblePS();
	void AssembleFFinv();


	void TimeStep( const SparseMatrix &F, const SparseMatrix &M, const HypreParVector& rhs, HypreParVector*& sol );







}; //StokesSTOperatorAssembler












#endif //STOKESSTOPERATORASSEMBLER