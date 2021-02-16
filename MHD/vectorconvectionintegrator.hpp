#ifndef MFEM_VECCONVINTEG
#define MFEM_VECCONVINTEG

#include "mfem.hpp"


namespace mfem{

/** Integrator for
     a (Q \dot div u, v) = a sum_i (Q \dot div u_i, v_i) e_i e_i^T
    for FE spaces defined by 'dim' copies of a scalar FE space. Where e_i
    is the unit vector in the i-th direction.  The resulting local element
    matrix is a block-diagonal matrix consisting of 'dim' copies of a scalar
    convection matrix in each diagonal block. */
class VectorConvectionIntegrator : public BilinearFormIntegrator{

protected:
  VectorCoefficient *Q;
  double alpha;
  // PA extension
  Vector pa_data;
  const DofToQuad *maps;         ///< Not owned
  const GeometricFactors *geom;  ///< Not owned
  int dim, ne, nq, dofs1D, quad1D;

private:
#ifndef MFEM_THREAD_SAFE
  DenseMatrix dshape, adjJ, Q_ir, pelmat;
  Vector shape, vec2, BdFidxT;
#endif

public:
  VectorConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
    : Q(&q) { alpha = a; }
  virtual void AssembleElementMatrix(const FiniteElement &,
                                     ElementTransformation &,
                                     DenseMatrix &);
  // virtual void AssembleElementVector(const FiniteElement &,
  // 																	 ElementTransformation &,
  // 																	 const Vector &,
  // 																	 Vector &);

  // using BilinearFormIntegrator::AssemblePA;

  // virtual void AssemblePA(const FiniteElementSpace&);
//   virtual void AssembleDiagonalPA(Vector &diag);

  // virtual void AddMultPA(const Vector&, Vector&) const;

  // static const IntegrationRule &GetRule(const FiniteElement &el,
  //                                       ElementTransformation &Trans);

  // static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
  //                                       const FiniteElement &test_fe,
  //                                       ElementTransformation &Trans);

};// VectorConvectionIntegrator



// // This is taken from VectorDiffusionIntegrator
// class VectorConvectionIntegrator : public BilinearFormIntegrator{
// protected:
//   VectorCoefficient *Q;

//   // PA extension
//   const DofToQuad *maps;         ///< Not owned
//   const GeometricFactors *geom;  ///< Not owned
//   int dim, ne, dofs1D, quad1D;
//   Vector pa_data;

// private:
//   DenseMatrix Jinv;
//   DenseMatrix dshape;
//   DenseMatrix gshape;
//   DenseMatrix pelmat;

// public:
//   VectorConvectionIntegrator() { Q = NULL; }
//   VectorConvectionIntegrator(VectorCoefficient &w) { W = &w; }

//   virtual void AssembleElementMatrix(const FiniteElement &el,
//                                      ElementTransformation &Trans,
//                                      DenseMatrix &elmat);
//   virtual void AssembleElementVector(const FiniteElement &el,
//                                      ElementTransformation &Tr,
//                                      const Vector &elfun, Vector &elvect);
//   using BilinearFormIntegrator::AssemblePA;
//   virtual void AssemblePA(const FiniteElementSpace &fes);
//   virtual void AssembleDiagonalPA(Vector &diag);
//   virtual void AddMultPA(const Vector &x, Vector &y) const;
















} //mfem


#endif //MFEM_VECCONVINTEG