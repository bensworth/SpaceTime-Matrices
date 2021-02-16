#ifndef INCOMPRESSIBLEMHD2DINTEGRATOR_HPP
#define INCOMPRESSIBLEMHD2DINTEGRATOR_HPP

#include "mfem.hpp"
namespace mfem{

/** 2D incompressible MHD equations in vector potential formulation,
   augmented with the extra variable z = (∇·∇A),
   ⎧ dt(u) + (u·∇)u - mu ∇·∇u + ∇p + 1/mu0 z ∇A = f
   ⎨                                        ∇·u = g
   |                                 z - (∇·∇A) = 0
   ⎩                    dt(A) + u·∇A - eta ∇·∇A = h

	NB: This integrator assumes that the FE spaces are polynomial
	     (particularly for the velocity variable)
*/
class IncompressibleMHD2DIntegrator : public BlockNonlinearFormIntegrator{
private:
	const double _dt;
	const double _mu;
	const double _mu0;
	const double _eta;

	Vector shape_u, shape_p, shape_z, shape_a;
  DenseMatrix shape_Du, shape_Dz, shape_Da,
  						gshape_u, gshape_z, gshape_a;

  DenseMatrix uCoeff;

public:
  IncompressibleMHD2DIntegrator( double dt, double mu, double mu0, double eta):
                                    _dt(dt),   _mu(mu),  _mu0(mu0),  _eta(eta){}

  // virtual double GetElementEnergy(const Array<const FiniteElement *>&el,
  //                                 ElementTransformation &Tr,
  //                                 const Array<const Vector *> &elfun);

  /// Perform the local action of the NonlinearFormIntegrator
  virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                     ElementTransformation &Tr,
                                     const Array<const Vector *> &elfun,
                                     const Array<Vector *> &elvec);

  /// Assemble the local gradient matrix
  virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *> &elfun,
                                   const Array2D<DenseMatrix *> &elmats);

  /// Get adequate GLL rule for element integration
  static const IntegrationRule& GetRule(const Array<const FiniteElement *> &fe, ElementTransformation &T);


};


} //namespace mfem

#endif //INCOMPRESSIBLEMHD2DINTEGRATOR_HPP


