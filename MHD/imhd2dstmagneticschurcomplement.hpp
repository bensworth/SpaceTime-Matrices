#ifndef IMHD2DMAGNETICSCHURCOMP_HPP
#define IMHD2DMAGNETICSCHURCOMP_HPP

#include <mpi.h>
#include "mfem.hpp"




namespace mfem{

// Approximation to pressure Schur complement
class IMHD2DSTMagneticSchurComplement: public Solver{

private:
	const MPI_Comm _comm;
	int _numProcs;
	int _myRank;

  double _dt;

  // relevant operators
  const PetscParMatrix *_M;	     //	mass matrix
  SparseMatrix          _W;      // spatial part magnetic convection-diffusion operator
  Solver               *_CCinv;  // solver for magnetic wave equation

  // solvers for relevant operators
  PetscLinearSolver *_Msolve;

  // dofs for vector potential (useful not to dirty dirichlet BC in the solution procedure)
	const Array<int> _essTDOF;

	const int _verbose;


public:
  IMHD2DSTMagneticSchurComplement( const MPI_Comm& comm, double dt,
                                   const SparseMatrix* M=NULL, const SparseMatrix* W=NULL, const SparseMatrix* CCinv=NULL,
                                   const Array<int>& essTDOF=Array<int>(), int verbose=0 );

	~IMHD2DSTMagneticSchurComplement();



  void Mult( const Vector& x, Vector& y ) const;

  void SetCCinv( const Solver*       CCinv );
  void SetM(     const SparseMatrix* M );
  void SetW(     const SparseMatrix* W );

	// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"IMHD2DSTMagneticSchurComplement::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void SetMSolve();



}; //IMHD2DSTMagneticSchurComplement

} // namespace mfem




#endif // IMHD2DMAGNETICSCHURCOMP_HPP