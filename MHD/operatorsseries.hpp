#ifndef MFEM_OPERATORSSERIES
#define MFEM_OPERATORSSERIES

#include "mfem.hpp"


namespace mfem
{

//! @class OperatorsSeries
/**
 * \brief A class to handle operators defined as series (sum of) of operators
 *
 * Usage:
 * - Use the constructors to define the series of operators
 * - Use the method Mult() and MultTranspose() to apply the series to a vector.
 *
 */

class OperatorsSeries : public Operator
{
public:
  //! Constructor for OperatorsSeries.
  /**
   *  @param ops  Series of operators to sum
   *  @param ownsOps  Flag which operators this class takes ownership of
   *
   *
   *  @note By default, OperatorsSeries will \em not own/copy the data contained in @a ops.
   */
  OperatorsSeries( const Array< const Operator*> & ops, const Array<bool>& ownsOps = Array<bool>() );



  //! Return the number of operators
  int NumOps() const { return _ops.Size(); }

  // //! Return a reference to i-th operator
  // Operator & GetOp(int iOp)
  // { MFEM_VERIFY(_ops[iOp], ""); return *_ops[iOp]; }


  /// Apply sequence of operators
  virtual void Mult (const Vector & x, Vector & y) const;

  //! Apply transpose of series of operators
  virtual void MultTranspose (const Vector & x, Vector & y) const;

  ~OperatorsSeries();

  //! Controls the ownership of the operators: if nonzero,
  //! OperatorsSeries will delete all operators that are set
  //! (non-NULL); the default value is zero.
  const Array<bool> _ownsOps;



private:
   //! Array that stores each operator in the sequence
   const Array<const Operator *> _ops;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose
   //! methods.
   mutable Vector _tmp;
   mutable Vector _tmp2;
};

}
#endif /* MFEM_OPERATORSSERIES */