#ifndef SPACETIMEWAVESOLVER_HPP
#define SPACETIMEWAVESOLVER_HPP

#include "mfem.hpp"



using namespace mfem;




// "Inverse" of space-time discretisation of linear wave equation
// At heart, this is a solver for a block lower tridiagonal matrix, although
//  some modifications are in place to consider the impact of initial conditions.
//  In particular, we consider a uniform temporal discretisation, acting on a
//  3-node stencil (this includes, eg, Implicit/Explicit Leapfrog, and the
//  backward difference formula. IC on derivative is approximated using a CD
//  formula, and automatically included in the system.
// Such a system can be represented with the recurring formula
//    Cp u^{n+1} + C0 u^{n} + Cm u^{n-1} = b^n
//  where the first equation is modified to include IC on derivative u', as such
//   (Cp+Cm) u^{1} + Cm u^{n-1} = b^n ( + dt*Cm u' - C0 u^{0} )
//  (and the second should also include IC on solution, as such
//    Cp u^{2} + C0 u^{1} = b^n ( - Cm u^{0} ),
//   but here we don't modify rhs! Only the system!).
// The monolithic system then looks like this:
//       ⌈ Cp+Cm            ⌉
//       |   C0  Cp         |
//  CC = |   Cm  C0 Cp      |,
//       |       Cm C0 Cp   |
//       ⌊          \\ \\ \\⌋
//  where Cp, C0, Cm are the spatial operators evaluated at the +1, 0, -1 nodes
//  in the stencil, respectively
class SpaceTimeWaveSolver: public Solver{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;


  const bool _timeDep;    // flag identifying whether the spatial operator depends on time or not
  const bool _symmetric;  // flag identifying whether the discretisation is symmetric
  // if both above are true, then Cm == Cp, and we can automatically apply some simplifications
  // if not, then it's the user responsibility to pass (Cp+Cm as an operator to the first processor)
  
  // relevant operators
  const PetscParMatrix *_Cp;
  const SparseMatrix   *_C0;
  const SparseMatrix   *_Cm;

	// solvers for relevant operator (with corresponding preconditioner, if iterative)
  PetscLinearSolver *_Cpsolve;

  // dirichlet dofs (unused?)
 	const Array<int> _essTDOF;

  mutable HypreParVector* _X;
  mutable HypreParVector* _Y;

	const int _verbose;


public:

	SpaceTimeWaveSolver( const MPI_Comm& comm, const SparseMatrix* Cp=NULL, const SparseMatrix* C0=NULL, const SparseMatrix* Cm=NULL,
		                   const Array<int>& essTDOF=Array<int>(), bool timeDep=true, bool symmetric=false, int verbose=0);

  void Mult( const Vector& x, Vector& y ) const;

  void SetDiag( const SparseMatrix* Op, int diag );

	~SpaceTimeWaveSolver();

// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"SpaceTimeWaveSolver::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void SetCpSolve();

   


}; //SpaceTimeWaveSolver







#endif //SPACETIMEWAVESOLVER