#ifndef __DISTORTIONS_HH__
#define __DISTORTIONS_HH__

/// \file Distortions.hh
/// \brief Some free functions to distort the mesh.
///
/// All these transformations assume the mesh is a unit box or cube
/// from (0,0,0) to (1,1,1).  This could be easily relaxed, if necessary.

namespace mfem
{
class Vector;
}

/// Paramter to control how much distortion in the sineTransform
extern double sinAlpha;
/// Distorts the grid with a sin funciton.
void sineTransform(const mfem::Vector& in, mfem::Vector& out);

/// Controls the amount of twist.  This is the amount of radians to twist the
/// center.
extern double twistAlpha;
/// Twist the mesh.  The Z coordinate is left alone
void twistTransform(const mfem::Vector& in, mfem::Vector& out);

/// Scale factor for Z mesh distortion.
extern double zmeshAlpha;
/// The Kershaw Z mesh distortion.
void zmeshTransform(const mfem::Vector& in, mfem::Vector& out);

/// Scale factor for Taylor-Green distortion
extern double TaylorGreenAlpha;
/// Distort the mesh according to the solution for the Taylor-Green Vortex hydro
/// problem.
void TaylorGreenTransform(const mfem::Vector& in, mfem::Vector& out);

/// For the crooked pipe problem, the mesh that's stored is square.  We want to
/// make it round.
void crookedPipeTransform(const mfem::Vector& in, mfem::Vector& out);

/// \brief Define a transformation from a unit box to a cylinder with small hole
/// in the middle.
void radialTransform(const mfem::Vector& in, mfem::Vector& out);

/// The size of the hole in the middle of the mesh.
extern const double holeRadius = 1.0e-6;

/// \brief Rotate the entire mesh by 45 degrees.
void rotateTransform(const mfem::Vector& in, mfem::Vector& out);

#endif  // __DISTORTIONS_HH__