#include "mymixedbilinearform.hpp"
#include <iostream>

namespace mfem{


void MyMixedBilinearForm::Assemble(int skip_zeros){

  // call method of base class
	MixedBilinearForm::Assemble(skip_zeros);


  Array<int> tr_vdofs, te_vdofs;
  DenseMatrix elemmat;

  Mesh *mesh = test_fes -> GetMesh();

  if (bfbfi.Size()){
    FaceElementTransformations *tr;
    const FiniteElement *trial_fe, *test_fe1, *test_fe2;

    // Which boundary attributes need to be processed?
    Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                               mesh->bdr_attributes.Max() : 0);
    bdr_attr_marker = 0;
    for (int k = 0; k < bfbfi.Size(); k++){
      if (bfbfi_marker[k] == NULL){
        bdr_attr_marker = 1;
        break;
      }
      Array<int> &bdr_marker = *bfbfi_marker[k];
      MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
              "invalid boundary marker for boundary face integrator #"
              << k << ", counting from zero");
      for (int i = 0; i < bdr_attr_marker.Size(); i++){
        bdr_attr_marker[i] |= bdr_marker[i];
      }
    }

    for (int i = 0; i < trial_fes -> GetNBE(); i++){
      const int bdr_attr = mesh->GetBdrAttribute(i);
      if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

      tr = mesh -> GetBdrFaceTransformations (i);
      if (tr != NULL){
        test_fes  -> GetElementVDofs (tr -> Elem1No, te_vdofs);
        trial_fes -> GetElementVDofs (tr -> Elem1No, tr_vdofs);
        test_fe1 = test_fes  -> GetFE (tr -> Elem1No);
        trial_fe = trial_fes -> GetFE (tr -> Elem1No);
        // The fe2 object is really a dummy and not used on the boundaries,
        // but we can't dereference a NULL pointer, and we don't want to
        // actually make a fake element.
        test_fe2 = test_fe1;
        for (int k = 0; k < bfbfi.Size(); k++){
          if (bfbfi_marker[k] &&
             (*bfbfi_marker[k])[bdr_attr-1] == 0) { continue; }

          bfbfi[k] -> AssembleFaceMatrix (*trial_fe, *test_fe1, *test_fe2, *tr, elemmat);
          mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
        }
      }
    }
  }

}




void MyMixedBilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi){
  bfbfi.Append(bfi);
  bfbfi_marker.Append(NULL); // NULL marker means apply everywhere
}

void MyMixedBilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi,
			                                         Array<int> &bdr_marker){
  bfbfi.Append(bfi);
  bfbfi_marker.Append(&bdr_marker);
}












} //mfem


  // if (bfbfi.Size())
  // {
  //   FaceElementTransformations *ftr;
  //   Array<int> te_vdofs2;
  //   const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

  //   // Which boundary attributes need to be processed?
  //   Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
  //                     mesh->bdr_attributes.Max() : 0);
  //   bdr_attr_marker = 0;
  //   for (int k = 0; k < btfbfi.Size(); k++)
  //   {
  //     if (btfbfi_marker[k] == NULL)
  //     {
  //       bdr_attr_marker = 1;
  //       break;
  //     }
  //     Array<int> &bdr_marker = *btfbfi_marker[k];
  //     MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
  //             "invalid boundary marker for boundary trace face integrator #"
  //             << k << ", counting from zero");
  //     for (int i = 0; i < bdr_attr_marker.Size(); i++)
  //     {
  //       bdr_attr_marker[i] |= bdr_marker[i];
  //     }
  //   }

  //   for (int i = 0; i < trial_fes -> GetNBE(); i++)
  //   {
  //     const int bdr_attr = mesh->GetBdrAttribute(i);
  //     if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

  //     ftr = mesh->GetBdrFaceTransformations(i);
  //     if (ftr)
  //     {
  //       trial_fes->GetFaceVDofs(i, tr_vdofs);
  //       test_fes->GetElementVDofs(ftr->Elem1No, te_vdofs);
  //       trial_face_fe = trial_fes->GetFaceElement(i);
  //       test_fe1 = test_fes->GetFE(ftr->Elem1No);
  //       // The test_fe2 object is really a dummy and not used on the
  //       // boundaries, but we can't dereference a NULL pointer, and we don't
  //       // want to actually make a fake element.
  //       test_fe2 = test_fe1;
  //       for (int k = 0; k < btfbfi.Size(); k++)
  //       {
  //         if (btfbfi_marker[k] &&
  //            (*btfbfi_marker[k])[bdr_attr-1] == 0) { continue; }

  //         btfbfi[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1, *test_fe2,
  //                             *ftr, elemmat);
  //         mat->AddSubMatrix(te_vdofs, tr_vdofs, elemmat, skip_zeros);
  //       }
  //     }
  //   }
  // }
