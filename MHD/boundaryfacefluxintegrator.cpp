#include "boundaryfacefluxintegrator.hpp"

namespace mfem{



void BoundaryFaceFluxIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                        const FiniteElement &el2,
                        FaceElementTransformations &Trans,
                        DenseMatrix &elmat){

  if (Trans.Elem2No >= 0){
    MFEM_ABORT("AssembleFaceMatrix only works for boundary faces");
  }

  int dim, dof;
  double w;

  dof = el1.GetDof();
  dim = el1.GetDim();

  qev.SetSize(dim);
  nor.SetSize(dim);
  adjJ.SetSize(dim);

  shape.SetSize(dof);

  elmat.SetSize(dof);
  elmat = 0.0;


  const IntegrationRule *ir = IntRule;
  if (ir == NULL){
    // a simple choice for the integration order; is this OK?
    int order = 2*el1.GetOrder();
    // ir = &IntRules.Get(Trans.GetGeometryType(), order);
    ir = &IntRules.Get(Trans.FaceGeom, order);
  }

  // assemble: < {Q.n} p,q >
  for (int p = 0; p < ir->GetNPoints(); p++){
    const IntegrationPoint &ip = ir->IntPoint(p);

    IntegrationPoint eip1;
    Trans.Loc1.Transform(ip, eip1);
    Trans.Face->SetIntPoint(&ip);


    // // Set the integration point in the face and the neighboring elements
    // Trans.SetAllIntPoints(&ip);

    // // Access the neighboring elements' integration points
    // const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

    if (dim == 1){
      nor(0) = 2*eip1.x - 1.0;
    }else{
      // CalcOrtho(Trans.Jacobian(), nor);
      CalcOrtho(Trans.Face->Jacobian(), nor);
    }

    el1.CalcShape(eip1, shape);

    Trans.Elem1->SetIntPoint(&eip1);
    Q->Eval(qev, *Trans.Elem1, eip1);

    w = ip.weight;
    w *= ( qev * nor );

    AddMult_a_VVt( w, shape, elmat );



  }

}



}