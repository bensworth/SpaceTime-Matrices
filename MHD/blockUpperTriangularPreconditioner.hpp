#ifndef MFEM_BLOCKUPPERTRIANGULARPRECONDITIONER
#define MFEM_BLOCKUPPERTRIANGULARPRECONDITIONER

#include "mfem.hpp"


namespace mfem
{

//! @class BlockUpperTriangularPreconditioner
/**
 * \brief A class to handle Block upper triangular preconditioners in a
 * matrix-free implementation.
 *
 * Usage:
 * - Use the constructors to define the block structure
 * - Use SetBlock() to fill the BlockOperator
 * - Diagonal blocks of the preconditioner should approximate the inverses of
 *   the diagonal block of the matrix
 * - Off-diagonal blocks of the preconditioner should match/approximate those of
 *   the original matrix
 * - Use the method Mult() and MultTranspose() to apply the operator to a vector.
 *
 * If a diagonal block is not set, it is assumed to be an identity block, if an
 * off-diagonal block is not set, it is assumed to be a zero block.
 *
 */

class BlockUpperTriangularPreconditioner : public Solver
{
public:
   //! Constructor for BlockUpperTriangularPreconditioner with the same
   //! block-structure for rows and columns.
   /**
    *  @param offsets  Offsets that mark the start of each row/column block
    *                  (size nBlocks+1).
    *
    *  @note BlockUpperTriangularPreconditioner will not own/copy the data
    *  contained in @a offsets.
    */
   BlockUpperTriangularPreconditioner(const Array<int> & offsets);

   //! Add block op in the block-entry (iblock, iblock).
   /**
    * @param iblock  The block will be inserted in location (iblock, iblock).
    * @param op      The Operator to be inserted.
    */
   void SetDiagonalBlock(int iblock, const Operator *op);
   //! Add a block op in the block-entry (iblock, jblock).
   /**
    * @param iRow, iCol  The block will be inserted in location (iRow, iCol).
    * @param op          The Operator to be inserted.
    */
   void SetBlock(int iRow, int iCol, const Operator *op);
   //! This method is present since required by the abstract base class Solver
   virtual void SetOperator(const Operator &op) { }

   //! Return the number of blocks
   int NumBlocks() const { return nBlocks; }

   // //! Return a reference to block i,j.
   // Operator & GetBlock(int iblock, int jblock)
   // { MFEM_VERIFY(op(iblock,jblock), ""); return *op(iblock,jblock); }

   //! Return the offsets for block starts
   Array<int> & Offsets() { return offsets; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const;

   ~BlockUpperTriangularPreconditioner();

   //! Controls the ownership of the blocks: if nonzero,
   //! BlockUpperTriangularPreconditioner will delete all blocks that are set
   //! (non-NULL); the default value is zero.
   int owns_blocks;

private:
   //! Number of block rows/columns
   int nBlocks;
   //! Offsets for the starting position of each block
   Array<int> offsets;
   //! 2D array that stores each block of the operator.
   Array2D<const Operator *> op;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose
   //! methods.
   mutable BlockVector xblock;
   mutable BlockVector yblock;
   mutable Vector tmp;
   mutable Vector tmp2;
};

}
#endif /* MFEM_BLOCKUPPERTRIANGULAROPERATOR */