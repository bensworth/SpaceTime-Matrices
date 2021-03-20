#ifndef SPACETIMESOLVER_HPP
#define SPACETIMESOLVER_HPP

#include "mfem.hpp"



using namespace mfem;




// "Inverse" of space-time matrix
// This class represents the inverse of a space-time matrix, built using
// an implicit Euler discretisation. That is, it approximates the
// inverse of FF:
//      ⌈ F*dt         ⌉
// FF = |  -M  F*dt    |,
//      |       -M  \\ |
//      ⌊           \\ ⌋
//  where F = M/dt + the spatial operator
class SpaceTimeSolver: public Solver{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;

	const bool _timeDep;		// flag identifying whether the spatial operator depends on time or not
  
  // relevant operators
  const PetscParMatrix *_F;
  const SparseMatrix   *_M;

	// solvers for relevant operator (with corresponding preconditioner, if iterative)
  PetscLinearSolver *_Fsolve;

  // dirichlet dofs (unused?)
 	const Array<int> _essVhTDOF;

  mutable HypreParVector* _X;
  mutable HypreParVector* _Y;

	const int _verbose;


public:

	SpaceTimeSolver( const MPI_Comm& comm, const SparseMatrix* F=NULL, const SparseMatrix* M=NULL,
		               const Array<int>& essVhTDOF=Array<int>(), bool timeDependent = true, int verbose=0);

  void Mult( const Vector& x, Vector& y ) const;

  void SetF( const SparseMatrix* F );
  void SetM( const SparseMatrix* M );

	~SpaceTimeSolver();

// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"SpaceTimeSolver::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void SetFSolve();

   


}; //SpaceTimeSolver







#endif //SPACETIMESOLVER