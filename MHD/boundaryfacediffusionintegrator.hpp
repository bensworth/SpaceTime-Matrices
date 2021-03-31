#ifndef BOUNDARYFACEDIFFUSIONINTEGRATOR_HPP
#define BOUNDARYFACEDIFFUSIONINTEGRATOR_HPP

#include "mfem.hpp"

namespace mfem{

/** Integrator for the form:

  - < {(Q grad(u)).n}, v >

  where Q is a scalar or matrix diffusion coefficient and u, v are the trial
  and test spaces, respectively.*/
class BoundaryFaceDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
  Coefficient *Q;
  MatrixCoefficient *MQ;

  // these are not thread-safe!
  Vector shape_z, shape_Dan, nor, nh, ni;
  DenseMatrix shape_Da, mq, adjJ;

public:
  BoundaryFaceDiffusionIntegrator()
    : Q(NULL), MQ(NULL) { }
  BoundaryFaceDiffusionIntegrator(Coefficient &q)
    : Q(&q), MQ(NULL) { }
  BoundaryFaceDiffusionIntegrator(MatrixCoefficient &q)
    : Q(NULL), MQ(&q) { }
  using BilinearFormIntegrator::AssembleElementMatrix2;
  // virtual void AssembleFaceMatrix(const FiniteElement &el1,
  //                       const FiniteElement &el2,
  //                       FaceElementTransformations &Trans,
  //                       DenseMatrix &elmat);
  virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
  											const FiniteElement &test_fe1, const FiniteElement &test_fe2,
  											FaceElementTransformations &Trans,
                        DenseMatrix &elmat);
	// void AssembleElementMatrix2(const FiniteElement &trial_fe,const FiniteElement &test_fe,
	// 														ElementTransformation &Trans, DenseMatrix &elmat);


};

}


#endif //BOUNDARYFACEDIFFUSIONINTEGRATOR_HPP
