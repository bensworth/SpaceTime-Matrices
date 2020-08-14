#include "blockUpperTriangularPreconditioner.hpp"

namespace mfem
{

BlockUpperTriangularPreconditioner::BlockUpperTriangularPreconditioner(
   const Array<int> & offsets_)
   : Solver(offsets_.Last()),
     owns_blocks(0),
     nBlocks(offsets_.Size() - 1),
     offsets(0),
     op(nBlocks, nBlocks)
{
   op = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockUpperTriangularPreconditioner::SetDiagonalBlock(int iblock,
                                                          Operator *op)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == op->Height() &&
               offsets[iblock+1] - offsets[iblock] == op->Width(),
               "incompatible Operator dimensions");

   SetBlock(iblock, iblock, op);
}

void BlockUpperTriangularPreconditioner::SetBlock(int iRow, int iCol,
                                                  Operator *opt)
{
   MFEM_VERIFY(iRow <= iCol,"cannot set block in lower triangle");
   MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == opt->NumRows() &&
               offsets[iCol+1] - offsets[iCol] == opt->NumCols(),
               "incompatible Operator dimensions");

   op(iRow, iCol) = opt;
}

// Operator application
void BlockUpperTriangularPreconditioner::MultTranspose (const Vector & x,
                                                        Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect output Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect input Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);

   y = 0.0;
   for (int iRow=0; iRow < nBlocks; ++iRow)
   {
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=0; jCol < iRow; ++jCol)
      {
         if (op(jCol,iRow))
         {
            op(jCol,iRow)->MultTranspose(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (op(iRow,iRow))
      {
         op(iRow,iRow)->MultTranspose(tmp2, yblock.GetBlock(iRow));
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;
      }
   }
}

// Operator application
void BlockUpperTriangularPreconditioner::Mult (const Vector & x,
                                               Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);

   // int myid;
   // MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // std::cout<<"Inside precon: Rank: "<<myid<< ", rhs for u: "; xblock.GetBlock(0).Print(std::cout, xblock.GetBlock(0).Size());
   // std::cout<<"Inside precon: Rank: "<<myid<< ", IG  for u: "; yblock.GetBlock(0).Print(std::cout, yblock.GetBlock(0).Size());
   // std::cout<<"Inside precon: Rank: "<<myid<< ", rhs for p: "; xblock.GetBlock(1).Print(std::cout, xblock.GetBlock(1).Size());
   // std::cout<<"Inside precon: Rank: "<<myid<< ", IG  for p: "; yblock.GetBlock(1).Print(std::cout, yblock.GetBlock(1).Size());

   y = 0.0;
   for (int iRow=nBlocks-1; iRow >=0; --iRow)
   {
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=iRow+1; jCol < nBlocks; ++jCol)
      {
         if (op(iRow,jCol))
         {
            op(iRow,jCol)->Mult(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (op(iRow,iRow))
      {
         op(iRow,iRow)->Mult(tmp2, yblock.GetBlock(iRow));
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;
      }
   }
}

BlockUpperTriangularPreconditioner::~BlockUpperTriangularPreconditioner()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nBlocks; ++jCol)
         {
            delete op(jCol,iRow);
         }
      }
   }
}

}