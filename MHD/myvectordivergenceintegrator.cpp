#include "myvectordivergenceintegrator.hpp"

namespace mfem{


void MyVectorDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dim  = trial_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   double c;

   dshape.SetSize (trial_dof, dim);
   gshape.SetSize (trial_dof, dim);
   Jadj.SetSize (dim);
   divshape.SetSize (dim*trial_dof);
   shape.SetSize (test_dof);

   elmat.SetSize (test_dof, dim*trial_dof);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                            Trans);

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trial_fe.CalcDShape (ip, dshape);
      test_fe.CalcShape (ip, shape);

      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      Mult (dshape, Jadj, gshape);

      gshape.GradToDiv (divshape);

      c = ip.weight;
      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      // elmat += c * shape * divshape ^ t
      shape *= c;
      AddMultVWt (shape, divshape, elmat);
   }
  elmat.Print(mfem::out, elmat.NumCols());  

}




const IntegrationRule &MyVectorDivergenceIntegrator::GetRule(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), order);
}


}