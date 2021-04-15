#ifndef MYVECTORDIVERGENCEINTEGRATOR_HPP
#define MYVECTORDIVERGENCEINTEGRATOR_HPP

#include "mfem.hpp"

namespace mfem{

/** Integrator for (Q div u, p) where u=(v1,...,vn) and all vi are in the same
    scalar FE space; p is also in a (different) scalar FE space.  */
class MyVectorDivergenceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
   Vector shape;
   Vector divshape;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;
   // PA extension
   Vector pa_data;
   const DofToQuad *trial_maps, *test_maps; ///< Not owned
   const GeometricFactors *geom;            ///< Not owned
   int dim, ne, nq;
   int trial_dofs1D, test_dofs1D, quad1D;

public:
   MyVectorDivergenceIntegrator() :
      Q(NULL), trial_maps(NULL), test_maps(NULL), geom(NULL)
   {  }
   MyVectorDivergenceIntegrator(Coefficient *_q) :
      Q(_q), trial_maps(NULL), test_maps(NULL), geom(NULL)
   { }
   MyVectorDivergenceIntegrator(Coefficient &q) :
      Q(&q), trial_maps(NULL), test_maps(NULL), geom(NULL)
   { }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   // using BilinearFormIntegrator::AssemblePA;
   // virtual void AssemblePA(const FiniteElementSpace &trial_fes,
   //                         const FiniteElementSpace &test_fes);

   // virtual void AddMultPA(const Vector &x, Vector &y) const;
   // virtual void AddMultTransposePA(const Vector &x, Vector &y) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};

}

#endif