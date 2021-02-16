#include "mixedlaplacianvectorgradientintegrator.hpp"


namespace mfem{



void MixedLaplacianVectorGradientIntegrator::AssemblePA(const FiniteElementSpace &trial_fes, const FiniteElementSpace &test_fes){
  // Assumes tensor-product elements, with a vector test space and H^1 trial space.
  Mesh *mesh = trial_fes.GetMesh();
  const FiniteElement *trial_fel = trial_fes.GetFE(0);
  const FiniteElement *test_fel = test_fes.GetFE(0);

  const NodalTensorFiniteElement *trial_el =
    dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
  MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

  const VectorTensorFiniteElement *test_el =
    dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
  MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

  const IntegrationRule *ir
    = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
  const int dims = trial_el->GetDim();
  MFEM_VERIFY(dims == 2 || dims == 3, "");

  const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
  const int nq = ir->GetNPoints();
  dim = mesh->Dimension();
  MFEM_VERIFY(dim == 2 || dim == 3, "");

  MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

  ne = trial_fes.GetNE();
  geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
  mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
  mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
  dofs1D = mapsC->ndof;
  quad1D = mapsC->nqpt;

  MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

  pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

  Vector coeff(ne * nq);
  coeff = 1.0;

  if (Q){
    for (int e=0; e<ne; ++e){
		  Vector laps(nq);
		  // Will this work out of the box? isn't it a mess because of all the mappings to ref element?
  		_myQ->GetGridFunction()->GetLaplacians( e, *ir, laps, dim );	// TODO: is dim the right value to pass here?
    	for (int p=0; p<nq; ++p){
        coeff[p + (e * nq)] = laps[p];
      }
    }
  }


  // // This was the original code taken from MixedVectorGradientIntegrator
  // if (Q){
  //   for (int e=0; e<ne; ++e){
  //   	ElementTransformation *tr = mesh->GetElementTransformation(e);
  //   	for (int p=0; p<nq; ++p){
  //       coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
  //     }
  //   }
  // }

  // Use the same setup functions as VectorFEMassIntegrator.
  if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3){
    PADiffusionSetup3D(quad1D, 1, ne, ir->GetWeights(), geom->J, coeff, pa_data);
  }else if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2){
    PADiffusionSetup2D<2>(quad1D, 1, ne, ir->GetWeights(), geom->J, coeff, pa_data);
  }else{
    MFEM_ABORT("Unknown kernel.");
  }
}

void MixedLaplacianVectorGradientIntegrator::AddMultPA(const Vector &x, Vector &y) const{
  if (dim == 3)
    PAHcurlH1Apply3D(dofs1D, quad1D, ne, mapsC->B, mapsC->G, mapsO->Bt, mapsC->Bt, pa_data, x, y);
  else if (dim == 2)
    PAHcurlH1Apply2D(dofs1D, quad1D, ne, mapsC->B, mapsC->G, mapsO->Bt, mapsC->Bt, pa_data, x, y);
  else{
    MFEM_ABORT("Unsupported dimension!");
  }
}

} // namespace mfem
