#ifndef INCOMPRESSIBLEMHD2DMASSSTABINTEGRATOR_HPP
#define INCOMPRESSIBLEMHD2DMASSSTABINTEGRATOR_HPP

#include "mfem.hpp"
namespace mfem{

/** Handy integrator to assemble "mass matrices" stemming from
    SUPG stabilisation. In particular it assembles the bilinear forms:
    \sum_el < - \tau_u(w,C)  u , (w·∇)v >_el
    \sum_el < - \tau_u(w,C)  u , ∇q     >_el
    \sum_el < - \tau_A(w,C)  A , w·∇B   >_el
    (notice the negative sign) for a given velocity field w and vector
    potential C

	NB: This integrator assumes that the FE spaces are polynomial
	     (particularly for the velocity variable)
*/
class IncompressibleMHD2DMassStabIntegrator : public BlockNonlinearFormIntegrator{
private:
	const double _dt;
  const double _mu;
  const double _mu0;
  const double _eta;

  // for stabilisation parameters
  const double C1 = 1.;
  const double C2 = 10.;

  VectorGridFunctionCoefficient *_w;
  GridFunctionCoefficient       *_c;

	Vector shape_u, shape_a;
  DenseMatrix shape_Du, shape_Dp, shape_Da,
  						gshape_u, gshape_p, gshape_a;

  DenseMatrix uCoeff;


public:
  IncompressibleMHD2DMassStabIntegrator( double dt, double mu, double mu0, double eta,
                                         VectorGridFunctionCoefficient* w, GridFunctionCoefficient* c ):
                                            _dt(dt),   _mu(mu),  _mu0(mu0),  _eta(eta), _w(w), _c(c) {};

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

  void GetTaus(const Vector& u, const Vector& gA, const DenseMatrix& Jinv, double& tauU, double& tauA) const; // stabilisation coefficients
  
  /// Assembles the contravariant metric tensor of the transformation from local to physical coordinates
  void GetGC( const DenseMatrix& Jinv, DenseMatrix& GC ) const;


};


} //namespace mfem

#endif //INCOMPRESSIBLEMHD2DMASSSTABINTEGRATOR_HPP


