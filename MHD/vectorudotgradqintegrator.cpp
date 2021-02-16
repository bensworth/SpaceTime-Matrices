#include "vectorudotgradqintegrator.hpp"


namespace mfem{



void VectorMassIntegrator::AssembleElementMatrix( const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat ){
  // I need to compute ((u•∇) Q, v) = ( ∇Q' u , v), where Q is a vector-valued grid function.
  //  Luckily, VectorMassIntegrator already computes ( M u , v), so I can just copy-paste that code,
  //  making sure that I compute the gradient, and that I transpose it correctly 

  int nd = el.GetDof();
  int spaceDim = Trans.GetSpaceDim();

  double norm;

  // If vdim is not set, set it to the space dimension
  vdim = (vdim == -1) ? spaceDim : vdim;

  elmat.SetSize(nd*vdim);
  shape.SetSize(nd);
  partelmat.SetSize(nd);
  mcoeff.SetSize(vdim);

  const IntegrationRule *ir = IntRule;
  if (ir == NULL){
  int order = 2 * el.GetOrder() + Trans.OrderW() + Q_order;

  if (el.Space() == FunctionSpace::rQk){
      ir = &RefinedIntRules.Get(el.GetGeomType(), order);
    }else{
      ir = &IntRules.Get(el.GetGeomType(), order);
    }
  }

  elmat = 0.0;
  for (int s = 0; s < ir->GetNPoints(); s++){
    const IntegrationPoint &ip = ir->IntPoint(s);
    el.CalcShape(ip, shape);

    Trans.SetIntPoint (&ip);
    norm = ip.weight * Trans.Weight();

    MultVVt(shape, partelmat);

    // get gradient of Q
    Q->GetGridFunction()->EvalGradients( mcoeff, Trans, ip );
    for (int i = 0; i < vdim; i++){
      for (int j = 0; j < vdim; j++){
        // flip i and j wrt VectorMassMatrix to transpose
        elmat.AddMatrix(norm*mcoeff(j,i), partelmat, nd*i, nd*j);
      }
    }
  }
}






















void VectorUDotGradQIntegrator::AssembleElementMatrix(
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

  Q->GetGridFunction()->EvalGradients(Q_ir, Trans, *ir);

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