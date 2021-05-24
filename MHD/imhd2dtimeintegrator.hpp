#ifndef INCOMPRESSIBLEMHD2DTIMEINTEGRATOR_HPP
#define INCOMPRESSIBLEMHD2DTIMEINTEGRATOR_HPP

#include "mfem.hpp"
namespace mfem{

/** Handy integrator to assemble the "temporal part" of the IMHD operator.
    This basically computes mass matrices (eventually rescaled by dt).
    In particular it assembles the bilinear forms:
    - < u,v > and  - < A,B >
    plus eventually (if stabilisation is included)
    \sum_el < - \tau_u(w,C)  u , (w·∇)v >_el
    \sum_el < - \tau_u(w,C)  u , ∇q     >_el
    \sum_el < - \tau_A(w,C)  A , w·∇B   >_el
    (notice the negative sign) for a given velocity field w and vector
    potential C

	NB: This integrator assumes that the FE spaces are polynomial
	     (particularly for the velocity variable)
*/
class IncompressibleMHD2DTimeIntegrator : public BlockNonlinearFormIntegrator{
private:
	const double _dt;
  const double _mu;
  const double _mu0;
  const double _eta;

  // for stabilisation
  const bool _stab;
  VectorGridFunctionCoefficient *_w;
  GridFunctionCoefficient       *_c;

	Vector shape_u, shape_a;
  DenseMatrix shape_Du, shape_Dp, shape_Da,
  						gshape_u, gshape_p, gshape_a;

  DenseMatrix uCoeff;


public:
  IncompressibleMHD2DTimeIntegrator( double dt, double mu, double mu0, double eta, bool stab=false, 
                                     VectorGridFunctionCoefficient* w=NULL, GridFunctionCoefficient* c=NULL ):
                                        _dt(dt),   _mu(mu),  _mu0(mu0),  _eta(eta), _stab(stab), _w(w), _c(c) {};

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

#endif //INCOMPRESSIBLEMHD2DTIMEINTEGRATOR_HPP


