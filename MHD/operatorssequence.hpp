#ifndef MFEM_OPERATORSSEQUENCE
#define MFEM_OPERATORSSEQUENCE

#include "mfem.hpp"


namespace mfem
{

//! @class OperatorsSequence
/**
 * \brief A class to handle operators defined as products of a sequence of operators
 *
 * Usage:
 * - Use the constructors to define the sequence of operators
 * - Use the method Mult() and MultTranspose() to apply the sequence to a vector.
 *
 */

class OperatorsSequence : public Solver
{
public:
  //! Constructor for OperatorsSequence.
  /**
   *  @param ops  Sequence of operators to apply. Notice they are applied left
   *               to right, ie, @a ops = [A1,A2,A3] represents the operator
   *               A3*A2*A1
   *  @param ownsOps  Flag which operators this class takes ownership of
   *
   *
   *  @note By default, OperatorsSequence will \em not own/copy the data contained in @a ops.
   */
  OperatorsSequence( const Array< const Operator*> & ops, const Array<bool>& ownsOps = Array<bool>() );



  //! Return the number of operators
  int NumOps() const { return _ops.Size(); }

  // //! Return a reference to i-th operator
  // Operator & GetOp(int iOp)
  // { MFEM_VERIFY(_ops[iOp], ""); return *_ops[iOp]; }


  /// Apply sequence of operators
  virtual void Mult (const Vector & x, Vector & y) const;

  //! Apply transpose of sequence of operators
  /**
   *  @note This is \em not the operator-wise transpose, but the \em actual
   *         transpose: (A3*A2*A1)^T -> A1^T*A2^T*A3^T
   */
  virtual void MultTranspose (const Vector & x, Vector & y) const;

  ~OperatorsSequence();

  //! Controls the ownership of the operators: if nonzero,
  //! OperatorsSequence will delete all operators that are set
  //! (non-NULL); the default value is zero.
  const Array<bool> _ownsOps;

  // to ensure implementation of Solver interface
  inline void SetOperator(const Operator &op){
    std::cerr<<"You shouldn't invoke this function"<<std::endl;
  };


private:
   //! Array that stores each operator in the sequence
   const Array<const Operator *> _ops;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose
   //! methods.
   mutable Vector _tmp;
   mutable Vector _tmp2;
};

}
#endif /* MFEM_OPERATORSSEQUENCE */