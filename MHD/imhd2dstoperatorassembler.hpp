#ifndef IMHD2DSTOPERATORASSEMBLER_HPP
#define IMHD2DSTOPERATORASSEMBLER_HPP

#include "mfem.hpp"
#include "imhd2dstmagneticschurcomplement.hpp"
#include "oseenstpressureschurcomplement.hpp"
#include "parblocklowtrioperator.hpp"
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
	const double _eta;	//electric resistivity
	const double _mu0;	//magnetic permeability
  int _dim;						//domain dimension (R^d)
	void(  *_fFunc)( const Vector &, double, Vector & );	// function returning forcing term for velocity (time-dep)
	double(*_gFunc)( const Vector &, double )          ;  // function returning forcing term for pressure (time-dep)
	double(*_hFunc)( const Vector &, double )          ;  // function returning forcing term for vector potential (time-dep)
	void(  *_nFunc)( const Vector &, double, Vector & );  // function returning mu  * du/dn (time-dep, used to implement BC)
	double(*_mFunc)( const Vector &, double );            // function returning eta * dA/dn (time-dep, used to implement BC)
	GridFunction                  _wGridFunc;   	        // coefficients of (linearised) velocity field
	GridFunction                  _qGridFunc;   	        // coefficients of pressure field (initial guess)
	GridFunction                  _yGridFunc;   	        // coefficients of (linearised) Laplacian of vector potential
	GridFunction                  _cGridFunc;   	        // coefficients of (linearised) vector potential
	VectorGridFunctionCoefficient _wFuncCoeff;	          // (linearised) velocity field (cast as a function coefficient)
	GridFunctionCoefficient       _cFuncCoeff;	          // coefficients of (linearised) vector potential (cast as a function coefficient)
	VectorFunctionCoefficient     _fFuncCoeff;	          // coefficients of rhs for velocity
	FunctionCoefficient           _hFuncCoeff;            // coefficients of rhs for vector potential
	void(  *_uFunc)( const Vector &, double, Vector & );	// function returning velocity solution (time-dep, used to implement IC, and compute error)
	double(*_pFunc)( const Vector &, double )          ;  // function returning pressure solution (time-dep, used to implement IC and BC, and compute error)
	double(*_zFunc)( const Vector &, double )          ;  // function returning Laplacian of vector potential solution (time-dep, used to BC, and compute error)
	double(*_aFunc)( const Vector &, double )          ;  // function returning vector potential solution (time-dep, used to implement IC and BC, and compute error)


	// rhs of non-linear operator (assembled once and for all, and then stored internally)
	Vector _frhs;
	Vector _grhs;
	Vector _zrhs;
	Vector _hrhs;
	

	// info on FE
  Mesh *_mesh;
  FiniteElementCollection *_UhFEColl;
  FiniteElementCollection *_PhFEColl;
  FiniteElementCollection *_ZhFEColl;
  FiniteElementCollection *_AhFEColl;
	// - FE spaces
  FiniteElementSpace      *_UhFESpace;
  FiniteElementSpace      *_PhFESpace;
  FiniteElementSpace      *_ZhFESpace;
  FiniteElementSpace      *_AhFESpace;
	// - order of polynomial FE spaces
  const int _ordU;
  const int _ordP;
  const int _ordZ;
  const int _ordA;
  // - info on Dirichlet BC
	// -- for each bdr tag, identifies whether it's Dirichlet (1) or not (0)
  Array<int> _isEssBdrU; // x component of velocity
  Array<int> _isEssBdrV; // y component of velocity
  Array<int> _isEssBdrA; // vector potential
	Array<int> _essUhTDOF;
	Array<int> _essPhTDOF;
	Array<int> _essAhTDOF;

	// size of domain
	double _area;

  // relevant matrices
  // - blocks for single time-steps
  BlockNonlinearForm _IMHD2DOperator;
  SparseMatrix _Mu;
  SparseMatrix _Fu;
  SparseMatrix _Mz;
  SparseMatrix _Ma;
  SparseMatrix _Fa;
  SparseMatrix _B;
  SparseMatrix _Bt;
  SparseMatrix _Z1;
  SparseMatrix _Z2;
  SparseMatrix _K;
  SparseMatrix _Y;
  // - for stabilisation
  BlockNonlinearForm _IMHD2DMassStabOperator;
  SparseMatrix _Cs;
  SparseMatrix _Mus;
  SparseMatrix _Mps;
  SparseMatrix _Mas;
	// -- for pressure Schur comp
  SparseMatrix _Mp;
  SparseMatrix _Ap;
  SparseMatrix _Wp;
	// -- for magnetic Schur comp
  SparseMatrix _Cp;
  SparseMatrix _C0;
  SparseMatrix _Cm;
  SparseMatrix _MaNoZero;
  SparseMatrix _MaNoZeroLumped;
  SparseMatrix _Wa;
  SparseMatrix _Aa;
  SparseMatrix _dtuWa;
  // -- for auxiliary variable
  PetscParMatrix* _Mztemp;


  bool _MuAssembled;
  bool _FuAssembled;
  bool _MzAssembled;
  bool _MaAssembled;
  bool _FaAssembled;
  bool _BAssembled;
  bool _BtAssembled;
	bool _CsAssembled;
  bool _Z1Assembled;
  bool _Z2Assembled;
  bool _KAssembled;
  bool _YAssembled;
  bool _MpAssembled;
  bool _ApAssembled;
  bool _WpAssembled;
  bool _AaAssembled;
	bool _CpAssembled;
	bool _C0Assembled;
	bool _CmAssembled;
	bool _MaNoZeroAssembled;
	bool _MaNoZeroLumpedAssembled;
	bool _WaAssembled;
	bool _dtuWaAssembled;

	int _USTSolveType;		// flag for space-time solver for velocity
	int _ASTSolveType;		// flag for space-time solver for vector potential

  // - space-time blocks
  ParBlockLowTriOperator _FFu;             // Space-time velocity block
  ParBlockLowTriOperator _MMz;             // Space-time Laplacian of vector potential block
  ParBlockLowTriOperator _FFa;             // Space-time vector potential block
  ParBlockLowTriOperator _BB;              // space-time -div block
  ParBlockLowTriOperator _BBt;             // space-time -div block (transpose)
  ParBlockLowTriOperator _CCs;             // space-time pressure stabilisation
  ParBlockLowTriOperator _ZZ1;             // space-time Lorentz block 1
  ParBlockLowTriOperator _ZZ2;             // space-time Lorentz block 2
  ParBlockLowTriOperator _KK;              // space-time mixed Laplacian block
  ParBlockLowTriOperator _YY;              // space-time magnetic convection block
  // HYPRE_IJMatrix _FFu;                     // Space-time velocity block
  // HypreParMatrix *_FFFu;                   // Space-time velocity block (ref)
  // HYPRE_IJMatrix _MMz;                     // Space-time Laplacian of vector potential block
  // HypreParMatrix *_MMMz;                   // Space-time Laplacian of vector potential block (ref)
  // HYPRE_IJMatrix _FFa;                     // Space-time vector potential block
  // HypreParMatrix *_FFFa;                   // Space-time vector potential block (ref)
  // HYPRE_IJMatrix _BB;                      // space-time -div block
  // HYPRE_IJMatrix _ZZ1;                     // space-time Lorentz block 1
  // HYPRE_IJMatrix _ZZ2;                     // space-time Lorentz block 2
  // HYPRE_IJMatrix _KK;                      // space-time mixed Laplacian block
  // HYPRE_IJMatrix _YY;                      // space-time magnetic convection block
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
  bool _BBtAssembled;
  bool _CCsAssembled;
  bool _ZZ1Assembled;
	bool _ZZ2Assembled;
  bool _YYAssembled;
  bool _KKAssembled;
	bool _pSAssembled;
	bool _aSAssembled;
	bool _FFuinvAssembled;
	bool _MMzinvAssembled;
	bool _CCainvAssembled;


	// whether to use stabilisation
	const bool _stab;

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
	                           double(*m)(const Vector &, double ),
	                           void(  *w)(const Vector &, double, Vector &),
	                           double(*q)(const Vector &, double ),
	                           double(*y)(const Vector &, double ),
	                           double(*c)(const Vector &, double ),
	                           void(  *u)(const Vector &, double, Vector &),
	                           double(*p)(const Vector &, double ),
	                           double(*z)(const Vector &, double ),
	                           double(*a)(const Vector &, double ),
 														 const Array<int>& essTagsU, const Array<int>& essTagsV, 
 														 const Array<int>& essTagsP, const Array<int>& essTagsA,
                             bool stab=false, int verbose=0 );

	// // constructor (uses vector of node values to initialise linearised fields)
	// IMHD2DSTOperatorAssembler( const MPI_Comm& comm, const std::string& meshName,
 //                             const int refLvl, const int ordU, const int ordP, const int ordZ, const int ordA,
 //                             const double dt, const double mu, const double eta, const double mu0,
 //                             void(  *f)(const Vector &, double, Vector &),
 //                             double(*g)(const Vector &, double ),
 //                             double(*h)(const Vector &, double ),
 //                             void(  *n)(const Vector &, double, Vector &),
 //                             double(*m)(const Vector &, double ),
 //                             const Vector& w,
 //                             const Vector& y,
 //                             const Vector& c,
 //                             void(  *u)(const Vector &, double, Vector &),
 //                             double(*p)(const Vector &, double ),
 //                             double(*z)(const Vector &, double ),
 //                             double(*a)(const Vector &, double ),
 //                             int verbose=0 );

	~IMHD2DSTOperatorAssembler();



	void AssembleSystem( Operator*& FFFu, Operator*& MMMz, Operator*& FFFa,
	                     Operator*& BBB,  Operator*& BBBt, Operator*& CCCs,  
	                     Operator*& ZZZ1, Operator*& ZZZ2,
	                     Operator*& KKK,  Operator*& YYY,
	                     Vector&  fres,   Vector&  gres,   Vector& zres,    Vector& hres,
	                     Vector&  IGu,    Vector&  IGp,    Vector& IGz,     Vector& IGa  );


	void ApplyOperator( const BlockVector& x, BlockVector& y );

	// void UpdateLinearisedOperators( const Vector& u, const Vector& p,  const Vector& z, const Vector& a );
	void UpdateLinearisedOperators( const BlockVector& x );


	void AssemblePreconditioner( Operator*& Finv, Operator*& Mzinv, Operator*& pSinv, Operator*& aSinv, const int spaceTimeSolverTypeU, const int spaceTimeSolverTypeA );


	void ApplySTOperatorVelocity( const HypreParVector*& u, HypreParVector*& res );


	void ExactSolution( HypreParVector*& u, HypreParVector*& p, HypreParVector*& z, HypreParVector*& a ) const;

	void GetMeshSize( double& h_min, double& h_max, double& k_min, double& k_max ) const;
	void ComputeL2Error(       const Vector& uh,        const Vector& ph,        const Vector& zh,        const Vector& ah,
	                             double& err_u,           double& err_p,           double& err_z,           double& err_a ) const;
	void SaveSolution( const HypreParVector& uh,const HypreParVector& ph,const HypreParVector& zh,const HypreParVector& ah,const std::string& pth,const std::string& fnnm ) const;
	void SaveSolution(         const Vector& uh,        const Vector& ph,        const Vector& zh,        const Vector& ah,const std::string& pth,const std::string& fnnm ) const;
	void SaveError(            const Vector& uh,        const Vector& ph,        const Vector& zh,        const Vector& ah,const std::string& pth,const std::string& fnnm ) const;
	void SaveExactSolution( const std::string& path, const std::string& filename ) const;
	void PrintMatrices( const std::string& filename ) const;
	// TODO:
	void TimeStep( const BlockVector& x, BlockVector& y, const std::string &innerConvpath, int output );


private:
	// get an integration rule for the various integrators
	const IntegrationRule& GetRule();
	const IntegrationRule& GetBdrRule();

	// assemble blocks for single time-step 
	// void AssembleFu();
	void AssembleMu();
	// void AssembleMz();
	// void AssembleFa();
	void AssembleMa();
	// void AssembleB();
	// void AssembleZ1();
	// void AssembleZ2();
	// void AssembleK();
	// void AssembleY();
	void AssembleAp();
	void AssembleMp();
	void AssembleWp();
	void AssembleCCaBlocks();
	void AssembleMaNoZero();
	void AssembleMaNoZeroLumped();
	void AssembleAa();
	void AssembleWa();
	void AssembledtuWa();


	// assemble blocks for whole Space-time operators 
	void AssembleSTBlockDiagonal(   const SparseMatrix& D, ParBlockLowTriOperator& DD,
																  const std::string& STMatName, const bool blockAssembled, bool& STMatAssembled );
	void AssembleSTBlockBiDiagonal( const SparseMatrix& F, const SparseMatrix& M, ParBlockLowTriOperator& FF,
                                  const std::string& STMatName, const bool blocksAssembled, bool& STMatAssembled ); 
	void AssembleFFu();
	void AssembleMMz();
	void AssembleFFa();
	void AssembleBB();
	void AssembleBBt();
	void AssembleCCs();
	void AssembleZZ1();
	void AssembleZZ2();
	void AssembleKK();
	void AssembleYY();
	void AssemblePS();
	void AssembleAS();
	void AssembleFFuinv();
	void AssembleMMzinv();
	void AssembleCCainv();


	void SetUpBoomerAMG( HYPRE_Solver& FFuinv, const int maxiter=15 );

	
	void SetEverythingUnassembled();
	void ComputeDomainArea();
	void ComputeAvgB( Vector& B ) const;




}; //IMHD2DSTOperatorAssembler












#endif //IMHD2DSTOPERATORASSEMBLER