#ifndef KHISTOPERATORASSEMBLER_HPP
#define KHISTOPERATORASSEMBLER_HPP

#include <mpi.h>
#include "mfem.hpp"
#include <string>



using namespace mfem;



// Generates operators necessary to discretise Kelvin-Helmholtz instability equations
class KHISTOperatorAssembler{

private:
	
	// info for parallelisation
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;


	// problem parameters
	const double _dt; 	//time step (constant over time-steps for now)
	const double _mu;		//viscosity
  int _dim;			      //domain dimension (R^d) - only dim=2 supported
  double(*_wFunc)( const Vector & x, const double t             ); // function returning w solution (time-dep, used to implement IC, and compute error)
  double(*_pFunc)( const Vector & x, const double t             ); // function returning phi solution (time-dep, used to implement IC, and compute error)
  void(  *_vFunc)( const Vector & x, const double t, Vector & u ); // function returning v solution (time-dep, used to implement IC, and compute error)
  double(*_fFunc)( const Vector & x, const double t             );	// function returning forcing term for w (time-dep)
  double(*_gFunc)( const Vector & x, const double t             );	// function returning forcing term for potential (time-dep)
  void(  *_hFunc)( const Vector & x, const double t, Vector & f );	// function returning forcing term for velocity (time-dep)

	// info on FE
  Mesh *_mesh;
  FiniteElementCollection *_WhFEColl;
  FiniteElementCollection *_PhFEColl;
  FiniteElementCollection *_VhFEColl;
  FiniteElementSpace      *_WhFESpace;
  FiniteElementSpace      *_PhFESpace;
  FiniteElementSpace      *_VhFESpace;
  const int _ordW;
  const int _ordP;
  const int _ordV;

	Array<int> _essWhTDOF;
	Array<int> _essPhTDOF;
	Array<int> _essVhTDOF;

  // relevant operators and corresponding matrices
  // - blocks for single time-steps
  BilinearForm      *_fwVarf;
  BilinearForm      *_mwVarf;
  BilinearForm      *_apVarf;
  BilinearForm      *_mvVarf;
  MixedBilinearForm *_bVarf;
  MixedBilinearForm *_cVarf;
  MixedBilinearForm *_mwpVarf;
  SparseMatrix _Fw;
  SparseMatrix _Mw;
  SparseMatrix _Ap;
  SparseMatrix _Mv;
  SparseMatrix _B;
  SparseMatrix _C;
  SparseMatrix _Mwp;
  bool _FwAssembled;
  bool _MwAssembled;
  bool _ApAssembled;
  bool _MvAssembled;
  bool _CAssembled;
  bool _BAssembled;
  bool _MwpAssembled;


  // - space-time blocks
  HYPRE_IJMatrix _FF;     // Space-time velocity block for w
  HypreParMatrix *_FFF;   // Space-time velocity block for w (reference)
  HYPRE_IJMatrix _AA;     // Space-time velocity block for p
  HYPRE_IJMatrix _MMv;    // Space-time velocity block for v
  HYPRE_IJMatrix _BB;     // space-time v*grad(w) block
  HYPRE_IJMatrix _CC;     // space-time curl(k*\phi) block
  HYPRE_IJMatrix _MMwp;   // space-time pairing btw w and p block
  Solver *_FFinv;         // Space-time block solver for w
  Solver *_FFinvPrec;     //  - with its preconditioner, in case
  Solver *_Ainv;          // (block) solver for laplacian of p
  Solver *_Mvinv;         // (block) solver for mass matrix of v
  bool _FFAssembled;
  bool _AAAssembled;
  bool _MMvAssembled;
  bool _BBAssembled;
  bool _CCAssembled;
  bool _MMwpAssembled;
	bool _FFinvAssembled;
	bool _AinvAssembled;
	bool _MvinvAssembled;



	const int _verbose;



public:
	KHISTOperatorAssembler( const MPI_Comm& comm, const std::string &meshName,
													const int refLvl, const int ordU, const int ordP,
		                      const double dt, const double mu, const double Pe,
		                      void(  *f)(const Vector &, double, Vector &),
		                      double(*g)(const Vector &, double ),
		                      void(  *n)(const Vector &, double, Vector &),
		                      void(  *w)(const Vector &, double, Vector &),
		                      void(  *u)(const Vector &, double, Vector &),
		                      double(*p)(const Vector &, double ),
		                      int verbose );
	~KHISTOperatorAssembler();
	void AssembleSystem( HypreParMatrix*& FFF,  HypreParMatrix*& AAA,  HypreParMatrix*& MMv,
                       HypreParMatrix*& BBB,  HypreParMatrix*& Mwp,  HypreParMatrix*& CCC,
                       HypreParVector*& frhs, HypreParVector*& grhs, HypreParVector*& hrhs,
                       HypreParVector*& IGw,  HypreParVector*& IGp,  HypreParVector*& IGv );

	void AssemblePreconditioner( Operator*& Finv, Operator*& Ainv, Operator*& Mvinv, const int spaceTimeSolverType );

	void ExactSolution( HypreParVector*& w, HypreParVector*& p, HypreParVector*& v );

	void GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const;
	void SaveSolution(      const HypreParVector& wh, const HypreParVector& ph, const HypreParVector& vh, const std::string& path, const std::string& filename ) const;
	void SaveExactSolution(                                                                               const std::string& path, const std::string& filename ) const;
	void SaveError(         const Vector& wh,         const Vector& ph,         const Vector& vh,         const std::string& path, const std::string& filename ) const;
	void SaveSolution(      const Vector& wh,         const Vector& ph,         const Vector& vh,         const std::string& path, const std::string& filename ) const;
	void ComputeL2Error(    const Vector& wh,         const Vector& ph,         const Vector& vh, double& err_w, double& err_p, double& err_v ) const;
	void PrintMatrices( const std::string& filename ) const;


private:
	TODO
	// assemble blocks for single time-step 
	void AssembleFuVarf();
	void AssembleMuVarf();
	void AssembleBVarf();
	void AssembleAp();
	void AssembleMp();
	void AssembleMwp();

	// assemble blocks for whole Space-time operators 
	void AssembleSTBlockDiagonal( const SparseMatrix& D, HYPRE_IJMatrix& DD, const std::string& STMatName,
		                            const bool blockAssembled, bool& STMatAssembled );
	void AssembleFF();
	void AssembleAA();
	void AssembleMMv();
	void AssembleBB();
	void AssembleCC();
	void AssembleMMwp();
	void AssembleFFinv( const int spaceTimeSolverType );
	void AssembleAinv();
	void AssembleMvinv();


	void SetUpBoomerAMG( HYPRE_Solver& FFinv, const int maxiter=15 );







}; //KHISTOperatorAssembler












#endif //KHISTOPERATORASSEMBLER