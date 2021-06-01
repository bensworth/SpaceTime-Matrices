#ifndef OSEENSTPRESSSCHURCOMPSIMPLE_HPP
#define OSEENSTPRESSSCHURCOMPSIMPLE_HPP

#include "mfem.hpp"




namespace mfem{

// Approximation to pressure Schur complement
class OseenSTPressureSchurComplementSimple: public Solver{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;

  // relevant operators
  SparseMatrix          _myAp;  //  spatial part of pressure convection-diffusion operator
  const PetscParMatrix *_Ap;	  //	pressure "laplacian"
  const PetscParMatrix *_Mp;	  //	pressure mass matrix
  SparseMatrix          _Fp;	  //  spatial part of pressure convection-diffusion operator

  // solvers for relevant operators
  PetscLinearSolver *_Asolve;
  PetscLinearSolver *_Msolve;

  // dofs for pressure (useful not to dirty dirichlet BC in the solution procedure) - TODO: check this
	const Array<int> _essQhTDOF;

	const int _verbose;


public:

	OseenSTPressureSchurComplementSimple( const MPI_Comm& comm, double dt, double mu,
		                              const SparseMatrix* Ap = NULL, const SparseMatrix* Mp = NULL, const SparseMatrix* Fp = NULL,
                                  const Array<int>& essQhTDOF = Array<int>(), int verbose = 0 );

	~OseenSTPressureSchurComplementSimple();



  void Mult( const Vector& x, Vector& y ) const;

  void SetAp( const SparseMatrix* Ap );
  void SetMp( const SparseMatrix* Mp );
  void SetFp( const SparseMatrix* Fp );

	// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"OseenSTPressureSchurComplementSimple::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void SetMpSolve();
	void SetApSolve();



}; //OseenSTPressureSchurComplementSimple

} // namespace mfem





#endif //OSEENSTPRESSSCHURCOMPSIMPLE_HPP

