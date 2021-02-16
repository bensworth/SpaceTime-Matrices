#ifndef IMHD2DSTOPERATORASSEMBLER_HPP
#define IMHD2DSTOPERATORASSEMBLER_HPP

#include <mpi.h>
#include "mfem.hpp"
#include "imhd2dstmagneticschurcomplement.hpp"
#include "oseenstpressureschurcomplement.hpp"
#include <string>



using namespace mfem;



class IMHD2DSTOperatorAssembler{

private:
	
	// info for parallelisation
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;


	// problem parameters
	const double _dt; 	//time step (constant over time-steps for now)
	const double _mu;		//viscosity
	const double _eta;	//magnetic resistivity
	const double _mu0;	//magnetic permeability
  int _dim;						//domain dimension (R^d)
	void(  *_fFunc)( const Vector &, double, Vector & );	// function returning forcing term for velocity (time-dep)
	double(*_gFunc)( const Vector &, double )          ;  // function returning forcing term for pressure (time-dep)
	double(*_hFunc)( const Vector &, double )          ;  // function returning forcing term for vector potential (time-dep)
	void(  *_nFunc)( const Vector &, double, Vector & );  // function returning mu  * du/dn (time-dep, used to implement BC)
	void(  *_mFunc)( const Vector &, double );            // function returning eta * dA/dn (time-dep, used to implement BC)
	void(  *_wFunc)( const Vector &, double, Vector & );  // function returning (linearised) velocity field
	double(*_yFunc)( const Vector &, double );            // function returning (linearised) Laplacian of vector potential
	double(*_cFunc)( const Vector &, double );            // function returning (linearised) vector potential
	const Vector _wFuncCoeff;                             // coefficients of (linearised) velocity field
	const Vector _yFuncCoeff;                             // coefficients of (linearised) Laplacian of vector potential
	const Vector _cFuncCoeff;                             // coefficients of (linearised) vector potential
	void(  *_uFunc)( const Vector &, double, Vector & );	// function returning velocity solution (time-dep, used to implement IC, and compute error)
	double(*_pFunc)( const Vector &, double )          ;  // function returning pressure solution (time-dep, used to implement IC and BC, and compute error)
	double(*_zFunc)( const Vector &, double )          ;  // function returning Laplacian of vector potential solution (time-dep, used to BC, and compute error)
	double(*_aFunc)( const Vector &, double )          ;  // function returning vector potential solution (time-dep, used to implement IC and BC, and compute error)


	// info on FE
  Mesh *_mesh;
  FiniteElementCollection *_UhFEColl;
  FiniteElementCollection *_PhFEColl;
  FiniteElementCollection *_ZhFEColl;
  FiniteElementCollection *_AhFEColl;
  FiniteElementSpace      *_UhFEspace;
  FiniteElementSpace      *_PhFEspace;
  FiniteElementSpace      *_ZhFEspace;
  FiniteElementSpace      *_AhFEspace;
  const int _ordU;
  const int _ordP;
  const int _ordZ;
  const int _ordA;

	Array<int> _essUhTDOF;
	Array<int> _essPhTDOF;
	Array<int> _essZhTDOF;
	Array<int> _essAhTDOF;

  // relevant matrices
  // - blocks for single time-steps
  SparseMatrix _Mu;
  SparseMatrix _Fu;
  SparseMatrix _Mz;
  SparseMatrix _Ma;
  SparseMatrix _Fa;
  SparseMatrix _B;
  SparseMatrix _Z1;
  SparseMatrix _Z2;
  SparseMatrix _K;
  SparseMatrix _Y;
	// -- for pressure Schur comp
  SparseMatrix _Mp;
  SparseMatrix _Ap;
  SparseMatrix _Wp;
	// -- for magnetic Schur comp
  SparseMatrix _Cp;
  SparseMatrix _C0;
  SparseMatrix _Cm;
  SparseMatrix _MaNoZero;
  SparseMatrix _Wa;
  // -- for auxiliary variable
  PetscParMatrix* _Mztemp;


  bool _MuAssembled;
  bool _FuAssembled;
  bool _MzAssembled;
  bool _MaAssembled;
  bool _FaAssembled;
  bool _BAssembled;
  bool _Z1Assembled;
  bool _Z2Assembled;
  bool _KAssembled;
  bool _YAssembled;
  bool _MpAssembled;
  bool _ApAssembled;
  bool _WpAssembled;
	bool _CpAssembled;
	bool _C0Assembled;
	bool _CmAssembled;
	bool _MaNoZeroAssembled;
	bool _WaAssembled;


  // - space-time blocks
  HYPRE_IJMatrix _FFu;                     // Space-time velocity block
  HypreParMatrix *_FFFu;                   // Space-time velocity block (ref)
  HYPRE_IJMatrix _MMz;                     // Space-time Laplacian of vector potential block
  HypreParMatrix *_MMMz;                   // Space-time Laplacian of vector potential block (ref)
  HYPRE_IJMatrix _FFa;                     // Space-time vector potential block
  HypreParMatrix *_FFFa;                   // Space-time vector potential block (ref)
  HYPRE_IJMatrix _BB;                      // space-time -div block
  HYPRE_IJMatrix _ZZ1;                     // space-time Lorentz block 1
  HYPRE_IJMatrix _ZZ2;                     // space-time Lorentz block 2
  HYPRE_IJMatrix _KK;                      // space-time mixed Laplacian block
  HYPRE_IJMatrix _YY;                      // space-time magnetic convection block
  OseenSTPressureSchurComplement  *_pSinv; // Approximation to space-time pressure Schur complement inverse
  IMHD2DSTMagneticSchurComplement *_aSinv; // Approximation to space-time magnetic Schur complement inverse
  Solver *_FFuinv;                         // Space-time velocity block solver
  Solver *_FFuinvPrec;                     //  - with its preconditioner, in case
  Solver *_MMzinv;                         // Space-time Laplacian of vector potential solver
  Solver *_CCainv;                         // Space-time magnetic wave block solver
  Solver *_CCainvPrec;                     //  - with its preconditioner, in case
  bool _FFuAssembled;
  bool _MMzAssembled;
  bool _FFaAssembled;
  bool _BBAssembled;
  bool _ZZ1Assembled;
	bool _ZZ2Assembled;
  bool _ZZAssembled;
  bool _YYAssembled;
	bool _pSAssembled;
	bool _aSAssembled;
	bool _FFuinvAssembled;
	bool _MMzinvAssembled;
	bool _CCainvAssembled;



	const int _verbose;



public:

	// constructor (uses analytical function for linearised fields)
	IMHD2DSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
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
	                           int verbose=0 );

	// constructor (uses vector of node values to initialise linearised fields)
	IMHD2DSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
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
                             int verbose=0 );

	~IMHD2DSTOperatorAssembler();



	void AssembleSystem( HypreParMatrix*& FFFu, HypreParMatrix*& MMMz, HypreParMatrix*& FFFa,
                       HypreParMatrix*& BBB,  HypreParMatrix*& ZZZ1, HypreParMatrix*& ZZZ2,
                       HypreParMatrix*& KKK,  HypreParMatrix*& YYY,
                       HypreParVector*& frhs, HypreParVector*& grhs, HypreParVector*& zrhs, HypreParVector*& hrhs,
                       HypreParVector*& IGu,  HypreParVector*& IGp,  HypreParVector*& IGz,  HypreParVector*& IGa );
	
	void ApplyOperator(  HypreParVector*& resU, HypreParVector*& resP, HypreParVector*& resZ, HypreParVector*& resA );

	void AssemblePreconditioner( Operator*& Finv, Operator*& Mzinv, Operator*& pSinv, Operator*& aSinv, const int spaceTimeSolverTypeU, const int spaceTimeSolverTypeA );


	void ApplySTOperatorVelocity( const HypreParVector*& u, HypreParVector*& res );


	void ExactSolution( HypreParVector*& u, HypreParVector*& p, HypreParVector*& z, HypreParVector*& a );

	void GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const;
	void ComputeL2Error(       const Vector& uh,        const Vector& ph,        const Vector& zh,        const Vector& ah,
	                             double& err_u,           double& err_p,           double& err_z,           double& err_a );
	void SaveSolution( const HypreParVector& uh, const HypreParVector& ph, const HypreParVector& zh, const HypreParVector& ah, const std::string& path,const std::string& fname );
	void SaveSolution(         const Vector& uh,         const Vector& ph,         const Vector& zh,         const Vector& ah, const std::string& path,const std::string& fname );
	void SaveError(            const Vector& uh,         const Vector& ph,         const Vector& zh,         const Vector& ah, const std::string& path,const std::string& fname );
	void SaveExactSolution( const std::string& path, const std::string& filename );
	void PrintMatrices( const std::string& filename ) const;
	// TODO:
	// void TimeStep(      const BlockVector& rhs, BlockVector& sol, const std::string &fname1, const std::string &path2, int refLvl );
	// void TimeStep(      const BlockVector& rhs, BlockVector& sol, const std::string &fname1, const std::string &path2, int refLvl, int pbType );

private:
	// assemble blocks for single time-step 
	void AssembleFu();
	void AssembleMu();
	void AssembleMz();
	void AssembleFa();
	void AssembleMa();
	void AssembleB();
	void AssembleZ1();
	void AssembleZ2();
	void AssembleK();
	void AssembleY();
	void AssembleAp();
	void AssembleMp();
	void AssembleWp();
	void AssembleCs( const int discType );
	void AssembleMaNoZero();
	void AssembleWa();



	// assemble blocks for whole Space-time operators 
	void AssembleSTBlockDiagonal( const SparseMatrix& D, HYPRE_IJMatrix& DD, const std::string& STMatName,
		                            const bool blockAssembled, bool& STMatAssembled );
	void AssembleSTBlockBiDiagonal( const SparseMatrix& F, const SparseMatrix& M, HYPRE_IJMatrix& FF, (HypreParMatrix*)& FFF,
                                  const std::string& STMatName, const bool blocksAssembled, bool& STMatAssembled ); 
	void AssembleMMz();
	void AssembleFFa();
	void AssembleBB();
	void AssembleZZ1();
	void AssembleZZ2();
	void AssembleKK();
	void AssembleYY();
	void AssemblePS();
	void AssembleAS();
	void AssembleFFuinv( const int spaceTimeSolverType );
	void AssembleMMzinv();
	void AssembleCCainv( const int spaceTimeSolverType );


	void SetUpBoomerAMG( HYPRE_Solver& FFuinv, const int maxiter=15 );

	






}; //IMHD2DSTOperatorAssembler












#endif //IMHD2DSTOPERATORASSEMBLER