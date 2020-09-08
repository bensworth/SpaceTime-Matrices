#ifndef PARAREALSOLVER_HPP
#define PARAREALSOLVER_HPP

#include <mpi.h>
#include "mfem.hpp"
#include <string>



using namespace mfem;




// "Inverse" of space-time matrix
class PararealSolver: public Solver{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;


  // relevant operators
  const PetscParMatrix *_F;
  const PetscParMatrix *_C;
  const SparseMatrix   *_M;

	// solvers for relevant operator (with corresponding preconditioner, if iterative)
  PetscLinearSolver *_Fsolve;
  PetscLinearSolver *_Csolve;

  // number of iterations for Parareal
	const int _maxIT;

  mutable HypreParVector* _X;
  mutable HypreParVector* _Y;

	const int _verbose;


public:

	PararealSolver( const MPI_Comm& comm, const SparseMatrix* F=NULL, const SparseMatrix* C=NULL, const SparseMatrix* M=NULL,
		              int maxIT=1, int verbose=0);

  void Mult( const Vector& x, Vector& y ) const;

  void SetF( const SparseMatrix* F );
  void SetC( const SparseMatrix* C );
  void SetM( const SparseMatrix* M );

	~PararealSolver();

// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"PararealSolver::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void SetFSolve();
	void SetCSolve();

   


}; //PararealSolver















#endif //PARAREALSOLVER_HPP