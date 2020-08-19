#include "vectorconvectionintegrator.hpp"


namespace mfem{


void VectorConvectionIntegrator::AssembleElementMatrix(
	const FiniteElement &el,
	ElementTransformation &Trans,
	DenseMatrix &elmat ){
  
  int dof = el.GetDof();
  int dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape, adjJ, Q_ir, pelmat;
  Vector shape, vec2, BdFidxT;
#endif
  elmat.SetSize(dof*dim);
  pelmat.SetSize(dof);

  dshape.SetSize(dof,dim);
  adjJ.SetSize(dim);
  shape.SetSize(dof);
  vec2.SetSize(dim);
  BdFidxT.SetSize(dof);

  Vector vec1;

  const IntegrationRule *ir = IntRule;
  if (ir == NULL){
    int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
    ir = &IntRules.Get(el.GetGeomType(), order);
  }

  Q->Eval(Q_ir, Trans, *ir);

  elmat = 0.0;
  for (int i = 0; i < ir->GetNPoints(); i++){
  	const IntegrationPoint &ip = ir->IntPoint(i);
  	el.CalcDShape(ip, dshape);
  	el.CalcShape(ip, shape);

    Trans.SetIntPoint(&ip);
    CalcAdjugate(Trans.Jacobian(), adjJ);
    Q_ir.GetColumnReference(i, vec1);
    vec1 *= alpha * ip.weight;

    adjJ.Mult(vec1, vec2);
    dshape.Mult(vec2, BdFidxT);

    MultVWt(shape, BdFidxT, pelmat);

    for (int d = 0; d < dim; d++){
	   	for (int k = 0; k < dof; k++){
	   	  for (int l = 0; l < dof; l++){
	   	    elmat (dof*d+k, dof*d+l) += pelmat (k, l);
	   	  }
	   	}
    }
  }
}








// void VectorConvectionIntegrator::AssembleElementVector(
//   const FiniteElement &el,
//   ElementTransformation &Trans,
//   const Vector &elfun,
//   Vector &elvect)
// {
//   const int dof = el.GetDof();
//   const int dim = el.GetDim();

// #ifdef MFEM_THREAD_SAFE
//   DenseMatrix dshape, adjJ, Q_ir, pelmat;
//   Vector shape, vec2, BdFidxT;
// #endif

//   shape.SetSize(dof);
//   dshape.SetSize(dof, dim);
//   elvect.SetSize(dof * dim);
//   gradEF.SetSize(dim);

//   EF.UseExternalData(elfun.GetData(), dof, dim);
//   ELV.UseExternalData(elvect.GetData(), dof, dim);

//   Vector vec1(dim), vec2(dim);
//   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Trans);
//   ELV = 0.0;

//   for (int i = 0; i < ir->GetNPoints(); i++){
//     const IntegrationPoint &ip = ir->IntPoint(i);
//     Trans.SetIntPoint(&ip);
//     el.CalcShape(ip, shape);
//     el.CalcPhysDShape(Trans, dshape);
//     double w = ip.weight * Trans.Weight();
      
//     if (Q){
//     	w *= Q->Eval(Trans, ip);
//     }
      
//     MultAtB(EF, dshape, gradEF);
//     EF.MultTranspose(shape, vec1);
//     gradEF.Mult(vec1, vec2);
//     vec2 *= w;
//     AddMultVWt(shape, vec2, ELV);
//   }
// }











} //mfem