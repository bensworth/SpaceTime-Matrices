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
  
  // relevant operators
  const SparseMatrix *_F;
  const SparseMatrix *_M;

	// solvers for relevant operator (with corresponding preconditioner, if iterative)
  Solver   *_Fsolve;
	Operator *_Fprec;
	int _solveType;

  
  mutable HypreParVector* _X;
  mutable HypreParVector* _Y;

	const bool _verbose;


public:

	SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F=NULL, const SparseMatrix* M=NULL,
		               int solveType=0, double tol=1e-12, bool verbose=false);

  void Mult( const Vector& x, Vector& y ) const;

  void SetF( const SparseMatrix* F, int solvetype );
  void SetM( const SparseMatrix* M );

	~SpaceTimeSolver();

private:
	void SetFSolve();


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

  // relevant operators
  const SparseMatrix *_Ap;
  const SparseMatrix *_Mp;

  // solvers for relevant operators (with corresponding preconditioners, if iterative)
  Solver   *_Asolve;
  Solver   *_Msolve;
  Operator *_Aprec;
  Operator *_Mprec;
	int _ASolveType;
	int _MSolveType;

	const bool _verbose;


public:

	StokesSTPreconditioner( const MPI_Comm& comm, double dt, double mu,
		                      const SparseMatrix* Ap = NULL, const SparseMatrix* Mp = NULL,
		                      int ASolveType = 0, int MSolveType = 0,
		                      double tol=1e-12, bool verbose = false );
	~StokesSTPreconditioner();



  void Mult( const Vector& x, Vector& y ) const;

  void SetAp( const SparseMatrix* Ap, int solvetype );
  void SetMp( const SparseMatrix* Mp, int solvetype );

private:
	void SetMpSolve();
	void SetApSolve();



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
	void(  *_fFunc)( const Vector &, double, Vector & );	// function returning forcing term (time-dep)
	void(  *_nFunc)( const Vector &, double, Vector & );  // function returning mu * du/dn (time-dep, used to implement BC)
	void(  *_uFunc)( const Vector &, double, Vector & );	// function returning velocity solution (time-dep, used to implement IC, and compute error)
	double(*_pFunc)( const Vector &, double )          ;  // function returning pressure solution (time-dep, used to implement IC and BC, and compute error)

	const double _tol;	//tolerance for solvers

	// info on FE
  Mesh *_mesh;
  FiniteElementCollection *_VhFEColl;
  FiniteElementCollection *_QhFEColl;
  FiniteElementSpace      *_VhFESpace;
  FiniteElementSpace      *_QhFESpace;
  const int _ordU;
  const int _ordP;


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


	const bool _verbose;



public:
	StokesSTOperatorAssembler( const MPI_Comm& comm, const char *meshName, const int refLvl,
		                         const int ordU, const int ordP, const double dt, const double mu,
		                         void(  *f)(const Vector &, double, Vector &),
		                         void(  *n)(const Vector &, double, Vector &),
		                         void(  *u)(const Vector &, double, Vector &),
		                         double(*p)(const Vector &, double ),
		                         double tol=1e-12 );
	~StokesSTOperatorAssembler();


	void AssembleOperator( HypreParMatrix*& FFF, HypreParMatrix*& BBB );

	void AssemblePreconditioner( Operator*& Finv, Operator*& XXX,
		                           int MsolveType=0, int AsolveType=0, int FsolveType=0 );

	void AssembleRhs( HypreParVector*& frhs );

	void ApplySTOperatorVelocity( const HypreParVector*& u, HypreParVector*& res );


	void ExactSolution( HypreParVector*& u, HypreParVector*& p );

	void TimeStepVelocity( const HypreParVector& rhs, HypreParVector*& sol );
	void TimeStepPressure( const HypreParVector& rhs, HypreParVector*& sol );

	void ComputeL2Error( const HypreParVector& uh, const HypreParVector& ph );
	void SaveSolution(   const HypreParVector& uh, const HypreParVector& ph );
	void SaveExactSolution( );

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
	void AssemblePS( int Msolvetype, int Asolvetype );
	void AssembleFFinv( int FsolveType );


	void TimeStep( const SparseMatrix &F, const SparseMatrix &M, const HypreParVector& rhs, HypreParVector*& sol );







}; //StokesSTOperatorAssembler












#endif //STOKESSTOPERATORASSEMBLER