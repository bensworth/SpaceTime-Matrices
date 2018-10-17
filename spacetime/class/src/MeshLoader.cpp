#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <memory>

#include "mfem.hpp"
#include "Distortions.hpp"
#include "MeshLoader.hpp"

MeshLoader::MeshLoader(const std::string& filename,
                       const int meshOrder,
                       const int feOrder,
                       const int refinementLevels,
                       const std::string& transformName,
                       const double alpha)
{
   // Read the mesh from the file.
   std::ifstream imesh(filename.c_str());

   MFEM_VERIFY( imesh, "Could not load the mesh file: " << filename);

   mMesh.reset(new mfem::Mesh(imesh, 1, 1));
   imesh.close();

   int dim = mMesh->Dimension();

   // If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-feOrder mesh.
   if (mMesh->NURBSext or meshOrder != -1) {
      int mesh_order = std::max(feOrder, 1);
      if (meshOrder != -1) {
         mesh_order = meshOrder;
      }
      mFEC.reset(new mfem::H1_FECollection(mesh_order, dim));
      mFES.reset(new mfem::FiniteElementSpace(mMesh.get(), mFEC.get(), dim));
      mMesh->SetNodalFESpace(mFES.get());
   }

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < refinementLevels; ++l)
   {
      mMesh->UniformRefinement();
   }

   if (transformName == "none")
   {
      // Just skip this.
   }
   else if (transformName == "z")
   {
      if (alpha >= 0.0)
      {
         // Global variable in transforms
         zmeshAlpha = alpha;
      }
      // Transform the mesh
      mMesh->Transform(zmeshTransform);
   }
   else if (transformName == "CrookedPipe"
            || filename.find("CrookedPipe") != std::string::npos)
   {
      mMesh->Transform(crookedPipeTransform);
   }
   else if (transformName == "twist")
   {
      if (alpha >= 0.0)
      {
         // Global variable in transforms
         twistAlpha = alpha;
      }
      // Transform the mesh
      mMesh->Transform(twistTransform);
   }
   else if (transformName == "rotate")
   {
      // Transform the mesh
      mMesh->Transform(rotateTransform);
   }
   else if (transformName == "vortex")
   {
      if (alpha >= 0.0)
      {
         // Global variable in transforms
         TaylorGreenAlpha = alpha;
      }
      // Transform the mesh
      mMesh->Transform(TaylorGreenTransform);
   }
   else if (transformName == "zvortex")
   {
      if (alpha >= 0.0)
      {
         // Global variable in transforms
         TaylorGreenAlpha = alpha;
      }
      // Transform the mesh
      mMesh->Transform(zmeshTransform);
      mMesh->Transform(TaylorGreenTransform);
   }
   else if (transformName == "sine")
   {
      if (alpha >= 0.0)
      {
         // Global variable in transforms
         sinAlpha = alpha;
      }
      // Transform the mesh
      mMesh->Transform(sineTransform);
   }
   else if (transformName == "radial")
   {
      // Transform the mesh
      mMesh->Transform(radialTransform);
   }
   else if (transformName == "zsine")
   {
      if (alpha >= 0.0)
      {
         sinAlpha = alpha;
      }
      // Transform the mesh
      mMesh->Transform(zmeshTransform);
      mMesh->Transform(sineTransform);
   }
   else if (transformName != "default")
   {
      MFEM_ABORT("Unsupported mesh distortion: "
                 << transformName
                 << ".  Allowed are 'none', 'z', 'sine', 'twist', 'vortex', "
                    "'zsine', 'radial', and 'zvortex'.  Case matters.");
   }

   // TODO: Check for inverted zones here.

   // ------------------------
   // fill in boundary condition arrays
   for (int e = 0; e < mMesh->GetNBE(); ++e)
   {
      mBCE[mMesh->GetBdrAttribute(e)].push_back(e);
   }
}




MeshLoader::MeshLoader(const std::string& filename,
                       const int meshOrder,
                       const int feOrder,
                       const int refinementLevels)
{
   // Read the mesh from the file.
   std::ifstream imesh(filename.c_str());

   MFEM_VERIFY( imesh, "Could not load the mesh file: " << filename);

   mMesh.reset(new mfem::Mesh(imesh, 1, 1));
   imesh.close();

   int dim = mMesh->Dimension();

   // If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-feOrder mesh.
   if (mMesh->NURBSext or meshOrder != -1) {
      int mesh_order = std::max(feOrder, 1);
      if (meshOrder != -1) {
         mesh_order = meshOrder;
      }
      mFEC.reset(new mfem::H1_FECollection(mesh_order, dim));
      mFES.reset(new mfem::FiniteElementSpace(mMesh.get(), mFEC.get(), dim));
      mMesh->SetNodalFESpace(mFES.get());
   }

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < refinementLevels; ++l)
   {
      mMesh->UniformRefinement();
   }


   // TODO: Check for inverted zones here.

   // ------------------------
   // fill in boundary condition arrays
   for (int e = 0; e < mMesh->GetNBE(); ++e)
   {
      mBCE[mMesh->GetBdrAttribute(e)].push_back(e);
   }
}
