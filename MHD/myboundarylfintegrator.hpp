#include "mfem.hpp"

namespace mfem{

class MyBoundaryLFIntegrator : public LinearFormIntegrator
{
  Vector shape;
  Coefficient &Q;
  int oa, ob;
public:
  /** @brief Constructs a boundary integrator with a given Coefficient @a QG.
     Integration order will be @a a * basis_order + @a b. */
  MyBoundaryLFIntegrator(Coefficient &QG, int a = 1, int b = 1)
    : Q(QG), oa(a), ob(b) { }

  /** Given a particular boundary Finite Element and a transformation (Tr)
     computes the element boundary vector, elvect. */
  virtual void AssembleRHSElementVect(const FiniteElement &el,
                          ElementTransformation &Tr,
                          Vector &elvect);
  virtual void AssembleRHSElementVect(const FiniteElement &el,
                          FaceElementTransformations &Tr,
                          Vector &elvect);
};


}