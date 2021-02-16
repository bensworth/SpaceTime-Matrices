#ifndef OSEENSTPRESSSCHURCOMP_HPP
#define OSEENSTPRESSSCHURCOMP_HPP

#include <mpi.h>
#include "mfem.hpp"




namespace mfem{

// Approximation to pressure Schur complement
class OseenSTPressureSchurComplement: public Solver{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;

  double _dt;
  double _mu;

  // relevant operators
  const PetscParMatrix *_Ap;	//	pressure "laplacian"
  const PetscParMatrix *_Mp;	//	pressure mass matrix
  SparseMatrix          _Wp;	//  spatial part of pressure convection-diffusion operator
	bool _WpEqualsAp;						//  allows for simplifications if Wp and Ap coincide

  // solvers for relevant operators
  PetscLinearSolver *_Asolve;
  PetscLinearSolver *_Msolve;

  // dofs for pressure (useful not to dirty dirichlet BC in the solution procedure) - TODO: check this
	const Array<int> _essQhTDOF;

	const int _verbose;


public:

	OseenSTPressureSchurComplement( const MPI_Comm& comm, double dt, double mu,
		                              const SparseMatrix* Ap = NULL, const SparseMatrix* Mp = NULL, const SparseMatrix* Wp = NULL,
                                  const Array<int>& essQhTDOF = Array<int>(), int verbose = 0 );

	~OseenSTPressureSchurComplement();



  void Mult( const Vector& x, Vector& y ) const;

  void SetAp( const SparseMatrix* Ap );
  void SetMp( const SparseMatrix* Mp );
  void SetWp( const SparseMatrix* Wp, bool WpEqualsAp=false );

	// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"OseenSTPressureSchurComplement::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void SetMpSolve();
	void SetApSolve();



}; //OseenSTPressureSchurComplement

} // namespace mfem





#endif //OSEENSTPRESSSCHURCOMP_HPP

