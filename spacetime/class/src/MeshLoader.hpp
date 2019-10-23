#ifndef __MESHLOADER_HH__
#define __MESHLOADER_HH__

#include <memory>
#include <map>
#include <vector>
#include <string>
#include "mfem.hpp"

/// \brief Loads a mesh using.
///  This loads a mesh from a file, changes the order, refines it, then distorts
///  it, if that's what the user wanted.
class MeshLoader
{
  public:
   MeshLoader(const std::string& filename,
              const int meshOrder,
              const int feOrder,
              const int refinementLevels,
              const std::string& transformName,
              const double alpha);
   
   MeshLoader(const std::string& filename,
              const int meshOrder,
              const int feOrder,
              const int refinementLevels);

   
   mfem::Mesh& getMesh() const { return *(mMesh.get()); }

   const std::vector<int>& getBoundaryElements(const int b)
   {
      // Note - this creates an empty vector if b doesn't exist.
      return mBCE[b];
   }

  private:
   // Using auto_ptr here so that I don't have to delete them manually
   // later.  Since MFEM itself just takes raw pointers, I don't need
   // to use a shared pointer.  I'm using pointers instead of real
   // obects so we don't even allocate the memory if we don't use
   // this option.
   std::auto_ptr<mfem::Mesh> mMesh;
   std::auto_ptr<mfem::FiniteElementCollection> mFEC;
   std::auto_ptr<mfem::FiniteElementSpace> mFES;

   std::map<int, std::vector<int> > mBCE;
};

#endif  // __MESHLOADER_HH__