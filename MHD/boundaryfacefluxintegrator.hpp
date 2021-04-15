#ifndef BOUNDARYFACEFLUXINTEGRATOR_HPP
#define BOUNDARYFACEFLUXINTEGRATOR_HPP

#include "mfem.hpp"

namespace mfem{

/** Integrator for the form:

  < {Q.n} p, q >_dO

  where Q is a vector coefficient and p, q are the trial
  and test spaces, respectively.*/
class BoundaryFaceFluxIntegrator : public BilinearFormIntegrator
{
protected:
  VectorCoefficient *Q;

  // these are not thread-safe!
  Vector shape, nor, qev;
  DenseMatrix adjJ;

public:
  BoundaryFaceFluxIntegrator(VectorCoefficient &q)
    : Q(&q) { }
  using BilinearFormIntegrator::AssembleElementMatrix2;
  virtual void AssembleFaceMatrix(const FiniteElement &el1,
                        const FiniteElement &el2,
                        FaceElementTransformations &Trans,
                        DenseMatrix &elmat);
  // virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
  // 											const FiniteElement &test_fe1, const FiniteElement &test_fe2,
  // 											FaceElementTransformations &Trans,
  //                       DenseMatrix &elmat);
	// void AssembleElementMatrix2(const FiniteElement &trial_fe,const FiniteElement &test_fe,
	// 														ElementTransformation &Trans, DenseMatrix &elmat);


};

}


#endif //BOUNDARYFACEFLUXINTEGRATOR_HPP
