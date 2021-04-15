#include "myboundarylfintegrator.hpp"

namespace mfem{

void MyBoundaryLFIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
  int dof = el.GetDof();

  shape.SetSize(dof);      // vector of size dof
  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = IntRule;
  if (ir == NULL)
  {
    int intorder = oa * el.GetOrder() + ob;  // <----------
    ir = &IntRules.Get(el.GetGeomType(), intorder);
  }

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetIntPoint (&ip);
    double val = Tr.Weight() * Q.Eval(Tr, ip);

    std::cout<<"Q eval: "<< Q.Eval(Tr, ip) <<std::endl;

    el.CalcShape(ip, shape);

    add(elvect, ip.weight * val, shape, elvect);
  }

  std::cout<<"Partial element assembly: "; elvect.Print();


}

void MyBoundaryLFIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
  int dof = el.GetDof();

  shape.SetSize(dof);      // vector of size dof
  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = IntRule;
  if (ir == NULL)
  {
    int intorder = oa * el.GetOrder() + ob;   // <------ user control
    ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
  }

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);

    // Set the integration point in the face and the neighboring element
    Tr.SetAllIntPoints(&ip);

    // Access the neighboring element's integration point
    const IntegrationPoint &eip = Tr.GetElement1IntPoint();

    double val = Tr.Face->Weight() * ip.weight * Q.Eval(*Tr.Face, ip);

    std::cout<<"Q eval: "<< Q.Eval(Tr, ip) << "weight" << Tr.Face->Weight() <<std::endl;
  
    el.CalcShape(eip, shape);

    add(elvect, val, shape, elvect);
  }

  std::cout<<"Partial element assembly: "; elvect.Print();


}
}