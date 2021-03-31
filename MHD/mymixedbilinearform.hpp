#ifndef MYMIXEDBILINEARFORM_HPP
#define MYMIXEDBILINEARFORM_HPP

#include "mfem.hpp"
namespace mfem{


class MyMixedBilinearForm : public MixedBilinearForm{

protected:
  /// Boundary face (skeleton) integrators.
  Array<BilinearFormIntegrator*> bfbfi;
  Array<Array<int>*>             bfbfi_marker;///< Entries are not owned.



public:
  /// Copy construction is not supported; body is undefined.
  MyMixedBilinearForm(FiniteElementSpace *tr_fes, FiniteElementSpace *te_fes)
  : MixedBilinearForm(tr_fes,te_fes){};

  /// Copy construction is not supported; body is undefined.
  MyMixedBilinearForm(const MyMixedBilinearForm &);

  /// Copy assignment is not supported; body is undefined.
  MyMixedBilinearForm &operator=(const MyMixedBilinearForm &);

  /// Adds a boundary face integrator. Assumes ownership of @a bfi.
  void AddBdrFaceIntegrator (BilinearFormIntegrator * bfi);

  /// Adds a boundary face integrator. Assumes ownership of @a bfi.
  void AddBdrFaceIntegrator (BilinearFormIntegrator * bfi,
                             Array<int> &bdr_marker);

  /// Access all integrators added with AddBdrFaceIntegrator().
  Array<BilinearFormIntegrator*> *GetBFBFI() { return &bfbfi; }
  /** @brief Access all boundary markers added with AddBdrFaceIntegrator().
     If no marker was specified when the integrator was added, the
     corresponding pointer (to Array<int>) will be NULL. */
  Array<Array<int>*> *GetBFBFI_Marker() { return &bfbfi_marker; }


  /// Assembles the form i.e. sums over all domain/bdr integrators.
  void Assemble(int skip_zeros = 1);


  ~MyMixedBilinearForm(){
    if (!extern_bfs){
      int k;
      for (k=0; k < bfbfi.Size(); k++) { delete bfbfi[k]; }
    }
  }


};



}


#endif
