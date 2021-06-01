#ifndef CHOLESKYSOLVER_HPP
#define CHOLESKYSOLVER_HPP

#include "mfem.hpp"




namespace mfem{

// Invert using cholesky
class CholeskySolver: public Solver{

private:

  // relevant operators
  DenseMatrix _A;
  DenseMatrix _L;

public:

	CholeskySolver( const SparseMatrix* A );




  void Mult( const Vector& x, Vector& y ) const;


	// to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
  	std::cerr<<"CholeskySolver::SetOperator( op ): You shouldn't invoke this function"<<std::endl;
  };


private:
	void Factorise();



}; //CholeskySolver

} // namespace mfem





#endif //CHOLESKYSOLVE_HPP

