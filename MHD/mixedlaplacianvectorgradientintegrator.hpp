#ifndef MFEM_MIXLAPVECGRADINTEG
#define MFEM_MIXLAPVECGRADINTEG

#include "mfem.hpp"


namespace mfem{


/** Class for integrating the bilinear form a(u,v) := (C grad u, v) in either 2D
 		or 3D and where C is a scalar coefficient computed as the laplacian of Q, u
 		is in H1 and v is in H(Curl) or H(Div). */
class MixedLaplacianVectorGradientIntegrator : public MixedVectorIntegrator{
public:
  MixedLaplacianVectorGradientIntegrator(GridFunctionCoefficient &q): MixedVectorIntegrator(q) { *_myQ = q; }

protected:
  inline virtual bool VerifyFiniteElementTypes( const FiniteElement & trial_fe, const FiniteElement & test_fe) const{
    return (trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
            test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
  }

  inline virtual const char * FiniteElementTypeFailureMessage() const{
    return "MixedVectorGradientIntegrator:  "
           "Trial spaces must be H1 and the test space must be a "
           "vector field in 2D or 3D";
  }

  inline virtual void CalcTrialShape(const FiniteElement & trial_fe, ElementTransformation &Trans, DenseMatrix & shape){
  	trial_fe.CalcPhysDShape(Trans, shape);
  }

  using BilinearFormIntegrator::AssemblePA;
  virtual void AssemblePA(const FiniteElementSpace &trial_fes,
                          const FiniteElementSpace &test_fes);

  virtual void AddMultPA(const Vector&, Vector&) const;

private:
  DenseMatrix Jinv;

  // bit of a hack
  GridFunctionCoefficient *_myQ; // not owned, so no need to delete

  // PA extension
  Vector pa_data;
  const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
  const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
  const GeometricFactors *geom;   ///< Not owned
  int dim, ne, dofs1D, quad1D;
};

}

#endif //MFEM_MIXLAPVECGRADINTEG