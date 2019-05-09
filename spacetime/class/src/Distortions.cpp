#include "mfem.hpp"
#include <cmath>

/// \file Distortions.cc
/// \brief Define some transformations to distort the mesh.
///
/// \todo Add a randomizer.  Hard to do without access to the edge lengths.
///

////////////////////////////////////////////////////////////////////////////////

double sinAlpha = 0.06;

// From: Berndt, Lipnikov, Shshkov, Wheeler, Yotov, "Superconvergence of the
// velocity in Mimetic Finite Difference Methods on Quadrilaterals," SIAM J.
// Numer.  Anal., Vol 43, Num 4, pp 1728-1749, 2005.  Scale maximum distortion
// with square of grid size?  I think, basically, don't invert/boomerang
// zones...
void sineTransform(const mfem::Vector& in, mfem::Vector& out)
{
   const double width = 1.0;

   double x = in(0);
   double y = in(1);
   double z = width / 4.0;
   if (in.Size() == 3)
   {
      z = in(2);
   }

   using namespace std;

   double xnew = x
                 + sinAlpha * sin(2 * M_PI * y / width) * sin(2 * M_PI * x / width)
                       * sin(2 * M_PI * z / width);
   double ynew = y
                 + sinAlpha * sin(2 * M_PI * y / width) * sin(2 * M_PI * x / width)
                       * sin(2 * M_PI * z / width);
   double znew = z
                 + sinAlpha * sin(2 * M_PI * y / width) * sin(2 * M_PI * x / width)
                       * sin(2 * M_PI * z / width);
   out(0) = xnew;
   out(1) = ynew;
   if (in.Size() == 3)
   {
      out(2) = znew;
   }
}

////////////////////////////////////////////////////////////////////////////////

// Parameter to determine how much to twist the mesh.
double twistAlpha = 0.7;

/// This twists the mesh be first transforming the square into a circle, then
/// rotates each point around the center by an increasing amount as you move
/// toward the center.  (The outter boundary is fixed.)  Then transforming
/// back into a square.
///
/// This is similar to the distortion in
///
/// Dobrev, Ellis, Kolev, and Rieben, "Curvilinear finite elements for
/// Lagrangian
/// hydrodynamics," International Journal for Numerical Methods in Fluids, 2010,
/// DOI: 10.1002/fld.2366
void twistTransform(const mfem::Vector& in, mfem::Vector& out)
{
   using namespace std;

   const double width = 1.0;
   double half = 0.5 * width;
   double r_corner = half * sqrt(2.0);

   // Get the coordinates
   double x = in(0) - half;
   double y = in(1) - half;

   // Convert to polar
   double r = hypot(x, y);
   double theta = atan2(x, y);

   // Scale the box out to be a circle
   double big = max(abs(cos(theta)), abs(sin(theta)));
   double r_boundary = half / big;
   double r_scale = r_corner / r_boundary;
   double rtmp = r * r_scale;

   // Rotate the coordinates
   double normDistToCorner = (1.0 - rtmp / r_corner);
   double theta_new = theta + twistAlpha * normDistToCorner;  //*normDistToCorner;

   // Now square up the circle
   // This takes the new theta, and resquares it.  It doesn't look as curvy as
   // just using the old radius
   // big = max( abs( cos(theta_new) ), abs(sin(theta_new) ) );
   // r_boundary = half / big;
   // r_scale = r_corner / r_boundary;
   // double r_new = rtmp / r_scale;

   double r_new = r;

   // Convert back to Cartesian coordinates
   out(0) = r_new * cos(theta_new) + half;
   out(1) = r_new * sin(theta_new) + half;
   if (in.Size() == 3)
   {
      /// \todo Do something interesting with the Z coordinate.
      out(2) = in(2);
   }
}

////////////////////////////////////////////////////////////////////////////////

double zmeshAlpha = 0.1;

/// This gives highly skewed meshes.  First published in:

///  David S Kershaw, Differencing of the diffusion equation in Lagrangian
///  hydrodynamic codes, Journal of Computational Physics, Volume 39, Issue 2,
///  February 1981, Pages 375-395, ISSN 0021-9991, 10.1016/0021-9991(81)90158-3.
///  http://www.sciencedirect.com/science/article/pii/0021999181901583
///
///  This assumes the mesh has some multiple of 5 zones in each direction
/// through
///  the definition of the y1, y2, y3, and y4 points.  You could move those to
/// be
///  on the undistorted grid lines if you had a multiple of 8 zones.
///
void zmeshTransform(const mfem::Vector& in, mfem::Vector& out)
{
   const double width = 1.0;

   double x = in(0);
   double y = in(1);
   double z = 0.0;
   if (in.Size() == 3)
   {
      z = in(2);
   }

   const double ymin = 0.0;
   const double ymax = width;
   const double y1 = ymin + 0.2 * (ymax - ymin);
   const double y2 = ymin + 0.4 * (ymax - ymin);
   const double y3 = ymin + 0.6 * (ymax - ymin);
   const double y4 = ymin + 0.8 * (ymax - ymin);

   // Bigger than y4
   double f = 1.0 - zmeshAlpha;
   if (y <= y1)
   {
      f = zmeshAlpha;
   }
   else if (y <= y2)
   {
      f = zmeshAlpha + (1.0 - 2.0 * zmeshAlpha) / (y2 - y1) * (y - y1);
   }
   else if (y <= y3)
   {
      f = 1.0 - zmeshAlpha + (2.0 * zmeshAlpha - 1.0) / (y3 - y2) * (y - y2);
   }
   else if (y <= y4)
   {
      f = zmeshAlpha + (1.0 - 2.0 * zmeshAlpha) / (y4 - y3) * (y - y3);
   }

   const double xmin = 0.0;
   const double xmax = width;
   const double zmin = 0.0;
   const double zmax = width;
   double fracx = (x - xmin) / (xmax - xmin);
   double fracz = (z - zmin) / (zmax - zmin);
   out(1) = y;
   if (fracx <= 0.5)
   {
      out(0) = xmin + 2.0 * fracx * f * (xmax - xmin);
   }
   else
   {
      out(0) = xmin + (f + 2.0 * (1.0 - f) * (fracx - 0.5)) * (xmax - xmin);
   }

   if (out.Size() == 3)
   {
      if (fracz <= 0.5)
      {
         out(2) = zmin + 2.0 * fracz * f * (zmax - zmin);
      }
      else
      {
         out(2) = zmin + (f + 2.0 * (1.0 - f) * (fracz - 0.5)) * (zmax - zmin);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////

double TaylorGreenAlpha = 0.5;

/// Twist the mesh following the solution to the Taylor-Green vortex problem.
/// The usual solution on the domain \f$0<x,y<2\pi\f$ to this is
/// \f{gather}{
///      v_x = \sin x \cos y e^{-2\nu t} \\
///      v_y = -\cos x \sin y
/// e^{-2\nu t}
/// \f}
/// where \f$v_x\f$ and \f$v_y\f$ are the \f$x\f$ and \f$y\f$ velocities,
/// \f$\nu\f$ is the kinematic viscosity, and \f$t\f$ is time.  We will
/// assume \f$\nu\f$ is zero for this distortion.
///
/// We numerically integrate mesh points through this velocity field
/// by scaling our unit box up by a factor of \f$ \pi \f$.  There is likely
/// Some analytic solution for this distortion.
///
void TaylorGreenTransform(const mfem::Vector& in, mfem::Vector& out)
{
   const unsigned int numStepsPerBD = 1000;
   const unsigned int numSteps
       = static_cast<unsigned int>(numStepsPerBD * TaylorGreenAlpha + 0.5);
   const double dt = (M_PI * TaylorGreenAlpha) / numSteps;

   // scale mesh up to the solution space
   double x = in(0) * M_PI;
   double y = in(1) * M_PI;

   for (unsigned int s = 0; s < numSteps; ++s)
   {
      const double v_x = -std::sin(x) * std::cos(y);
      const double v_y = +std::cos(x) * std::sin(y);
      x += v_x * dt;
      y += v_y * dt;
   }

   // scale back down.
   out(0) = x / (M_PI);
   out(1) = y / (M_PI);
   if (in.Size() == 3)
   {
      /// \todo Do something interesting with the Z coordinate.
      out(2) = in(2);
   }
}

////////////////////////////////////////////////////////////////////////////////

/// This distortion is a little different.  The mesh is supposed to be
/// circular, but we've squared it up in the mesh file.  This is so we
/// can refine it and increase the mesh order, then we transform it back
/// so that the new points live on the curved surfaces we want.
void crookedPipeTransform(const mfem::Vector& in, mfem::Vector& out)
{
   const double x = in(0);
   const double y = in(1);
   double xnew = x;
   double ynew = y;
   double r = 0.0;

   if (x > 1.0e-8 && y > 1.0e-8)
   {
      if (x >= y)
      {
         r = x;
         ynew = std::sqrt(x * x / ((x * x) / (y * y) + 1.0));
         xnew = ynew * x / y;
      }
      else
      {
         r = y;
         xnew = std::sqrt(y * y / ((y * y) / (x * x) + 1.0));
         ynew = xnew * y / x;
      }
   }

   // We want to scale the box part of the mesh less than the rest,
   // since those zones get really distorted looking.
   if (r > 0.0 && r < 0.5)
   {
      const double factor = r / 0.5;
      const double dx = factor * (xnew - x);
      const double dy = factor * (ynew - y);
      xnew = x + dx;
      ynew = y + dy;
   }

   out(0) = xnew;
   out(1) = ynew;
   // This should always be true, but just in case...
   if (in.Size() == 3)
   {
      out(2) = in(2);
   }
}

////////////////////////////////////////////////////////////////////////////////

const double holeRadius = 1.0e-6;
void radialTransform(const mfem::Vector& in, mfem::Vector& out)
{
   // We take the x coordinate and use it as the radius,
   // This is scaled so we have a smooth transform.
   double r = (1.0 - holeRadius) * in(0) + holeRadius;
   // We use the y coordinate, and scale it to theta.
   // double theta = (0.99*in(1)+0.01) * M_PI_2;
   double theta = in(1) * M_PI_2;
   // now we compute a new x and y
   double xprime = std::cos(theta);
   double yprime = std::sin(theta);
   if (xprime < std::numeric_limits<double>::epsilon())
   {
      xprime = 0.0;
   }
   if (yprime < std::numeric_limits<double>::epsilon())
   {
      yprime = 0.0;
   }
   out(0) = r * xprime;
   out(1) = r * yprime;
   if (in.Size() == 3)
   {
      out(2) = in(2);
   }
}

////////////////////////////////////////////////////////////////////////////////

void rotateTransform(const mfem::Vector& in, mfem::Vector& out)
{
   using namespace std;

   const double width = 1.0;
   double half = 0.5 * width;

   // Get the coordinates
   double x = in(0) - half;
   double y = in(1) - half;

   // Convert to polar
   double r = hypot(x, y);
   double theta = atan2(x, y);

   // Rotate the coordinates
   double r_new = r;
   double theta_new = theta + M_PI * 0.25;

   // Convert back to Cartesian coordinates
   out(0) = r_new * cos(theta_new) + half;
   out(1) = r_new * sin(theta_new) + half;
   if (in.Size() == 3)
   {
      /// \todo Do something interesting with the Z coordinate.
      out(2) = in(2);
   }
}
