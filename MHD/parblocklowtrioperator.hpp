#ifndef MFEM_PARBLOCKLOWTRIOPERATOR
#define MFEM_PARBLOCKLOWTRIOPERATOR

#include "mfem.hpp"


namespace mfem
{

//! @class ParBlockLowTriOperator
/**
 * \brief A class to handle block-lower-triangular operators defined in parallel, where each block is handled by a different processor
 *
 * Usage:
 * - Use the method SetBlockDiag() to define the operator in each diagonal (concurrent)
 * - Use the method Mult() and MultTranspose() to apply this operator to a vector.
 *
 */

class ParBlockLowTriOperator : public Operator
{
public:

   //! Parallel "view" of the operator.
   enum ParBlockView{ 
    ROW_VIEW, ///< Each processor holds blocks in the same *row*
    COL_VIEW  ///< Each processor holds blocks in the same *column*
   };

   //! Constructor for ParBlockLowTriOperator.
   /**
    *  @param comm  The communicator over which the operator is shared. It's used to 
    *                define the block size of the matrix (=number of processors)
    *  @param view  Defines the "view" each processor has of the whole operators
    *                (ie, whether it holds its row -default- or column blocks)
    *
    */
   ParBlockLowTriOperator( const MPI_Comm& comm, const ParBlockView& view=ROW_VIEW );


   //! Set block diagonal.
   /**
    *  @param mat  The operator to include in the block diagonal
    *  @param i    The index of the block diagonal to fill
    *  @param own  Wether the operator is owned or not by this class
    *
    *  @note By default, ParBlockLowTriOperator will \em not own/copy the data contained in @a mat.
    *  @note If a diagonal was already assigned, it gets deleted (if owned) and substituted.
    *
    */
   void SetBlockDiag( const SparseMatrix * mat, int i=0, bool own=false);

   //! Get block diagonal.
   /**
    *  @param i    The index of the block diagonal to fill
    *  @return     Pointer to selected operator
    *
    *  @note Returns NULL if diagonal is not assigned.
    *
    */
   const Operator* GetBlockDiag( int i );



   // //! Return a reference to i-th operator
   // Operator & GetOp(int iOp)
   // { MFEM_VERIFY( iOp<_numProcs && _ops[iOp], ""); return *_ops[iOp]; }


   /// Apply lower triangular operator
   virtual void Mult (const Vector & x, Vector & y) const;

   //! Apply transpose of lower triangular operator
   virtual void MultTranspose (const Vector & x, Vector & y) const; //{std::cerr<<"NOT IMPLEMENTED!!";};

   ~ParBlockLowTriOperator();

   //! Controls the ownership of the operators: if nonzero,
   //! ParBlockLowTriOperator will delete all operators that are set
   //! (non-NULL); the default value is zero.
   Array<bool> _ownsOps;

private:

  const MPI_Comm _comm;
  const ParBlockView _view;

  int _numProcs;
  int _myRank;


   //! Array that stores each operator in the diagonals
   Array<const Operator *> _ops;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose
   //! methods.
   mutable Vector _tmp;
   mutable Vector _tmp2;
};

}
#endif /* MFEM_PARBLOCKLOWTRIOPERATOR */