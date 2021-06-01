#ifndef TEST_CASES_HPP
#define TEST_CASES_HPP

#include "mfem.hpp"
#include <string>




namespace mfem{

// Analytical test case 4
namespace Analytical4Data{
  void   uFun_ex( const Vector & x, const double t, Vector & u );
  double pFun_ex( const Vector & x, const double t             );
  double zFun_ex( const Vector & x, const double t             );
  double aFun_ex( const Vector & x, const double t             );
  void   fFun(    const Vector & x, const double t, Vector & f );
  void   nFun(    const Vector & x, const double t, Vector & f );
  double gFun(    const Vector & x, const double t             );
  double hFun(    const Vector & x, const double t             );
  double mFun(    const Vector & x, const double t             );
  void   wFun(    const Vector & x, const double t, Vector & w );
  double qFun(    const Vector & x, const double t             );
  double cFun(    const Vector & x, const double t             );
  double yFun(    const Vector & x, const double t             );
  // - define a perturbation to dirty initial guess
  double perturbation(const Vector & x, const double t);
  const double _mu   = 1.;
  const double _eta  = 1.;
  const double _mu0  = 1.;
  const std::string _pbName   = "Analytical4";
  const std::string _meshFile ="./meshes/tri-square-testAn.mesh";
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
};


// // Modified Hartmann flow instability
// namespace HartmannData{
//   void   uFun_ex( const Vector & x, const double t, Vector & u );
//   double pFun_ex( const Vector & x, const double t             );
//   double zFun_ex( const Vector & x, const double t             );
//   double aFun_ex( const Vector & x, const double t             );
//   void   fFun(    const Vector & x, const double t, Vector & f );
//   void   nFun(    const Vector & x, const double t, Vector & f );
//   double gFun(    const Vector & x, const double t             );
//   double hFun(    const Vector & x, const double t             );
//   double mFun(    const Vector & x, const double t             );
//   void   wFun(    const Vector & x, const double t, Vector & w );
//   double qFun(    const Vector & x, const double t             );
//   double cFun(    const Vector & x, const double t             );
//   double yFun(    const Vector & x, const double t             );
//   const double _G0   = 1.;
//   const double _B0   = 1.;
//   const double _mu   = 1.;
//   const double _eta  = 1.;
//   const double _mu0  = 1.;
//   const std::string _pbName   = "Hartmann";
//   const std::string _meshFile = "./meshes/tri-square-hartmann.mesh";
//   void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
// };





// Kelvin-Helmholtz instability
namespace KHIData{
  void   uFun_ex( const Vector & x, const double t, Vector & u );
  double pFun_ex( const Vector & x, const double t             );
  double zFun_ex( const Vector & x, const double t             );
  double aFun_ex( const Vector & x, const double t             );
  void   fFun(    const Vector & x, const double t, Vector & f );
  void   nFun(    const Vector & x, const double t, Vector & f );
  double gFun(    const Vector & x, const double t             );
  double hFun(    const Vector & x, const double t             );
  double mFun(    const Vector & x, const double t             );
  void   wFun(    const Vector & x, const double t, Vector & w );
  double qFun(    const Vector & x, const double t             );
  double cFun(    const Vector & x, const double t             );
  double yFun(    const Vector & x, const double t             );
  const double _delta = 0.07957747154595;
  const double _mu   = 1.;
  const double _eta  = 1.;
  const double _mu0  = 1.;
  const std::string _pbName   = "KHI";
  const std::string _meshFile ="./meshes/tri-rect-KHI.mesh";
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
};



// Island coalescing
namespace IslandCoalescenceData{
  void   uFun_ex( const Vector & x, const double t, Vector & u );
  double pFun_ex( const Vector & x, const double t             );
  double zFun_ex( const Vector & x, const double t             );
  double aFun_ex( const Vector & x, const double t             );
  void   fFun(    const Vector & x, const double t, Vector & f );
  void   nFun(    const Vector & x, const double t, Vector & f );
  double gFun(    const Vector & x, const double t             );
  double hFun(    const Vector & x, const double t             );
  double mFun(    const Vector & x, const double t             );
  void   wFun(    const Vector & x, const double t, Vector & w );
  double qFun(    const Vector & x, const double t             );
  double cFun(    const Vector & x, const double t             );
  double yFun(    const Vector & x, const double t             );
  const double _delta = 1./(2.*M_PI);
  const double _beta  = 1e-3;
  const double _P0    = 1.;
  const double _eps   = 0.2;
  const double _mu    = 1e-2;
  const double _eta   = 1e-2;
  const double _mu0   = 1.;
  const std::string _pbName   = "IslandCoalescence";
  const std::string _meshFile ="./meshes/tri-square-island.mesh";
  // const std::string _meshFile ="./meshes/test.msh";
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
};




// // Rayleigh flow
// namespace RayleighData{
//   void   uFun_ex_rayleigh( const Vector & x, const double t, Vector & u );
//   double pFun_ex_rayleigh( const Vector & x, const double t             );
//   double zFun_ex_rayleigh( const Vector & x, const double t             );
//   double aFun_ex_rayleigh( const Vector & x, const double t             );
//   void   fFun_rayleigh(    const Vector & x, const double t, Vector & f );
//   void   nFun_rayleigh(    const Vector & x, const double t, Vector & f );
//   double gFun_rayleigh(    const Vector & x, const double t             );
//   double hFun_rayleigh(    const Vector & x, const double t             );
//   double mFun_rayleigh(    const Vector & x, const double t             );
//   void   wFun_rayleigh(    const Vector & x, const double t, Vector & w );
//   double qFun_rayleigh(    const Vector & x, const double t             );
//   double cFun_rayleigh(    const Vector & x, const double t             );
//   double yFun_rayleigh(    const Vector & x, const double t             );
//   const double U   = 1.;
//   const double B0  = 1.4494e-4;
//   const double rho = 0.4e-4;
//   const double mu0 = 1.256636e-6;
//   const double eta = 1.256636e-6;
//   const double mu  = 0.4e-4;
//   const double d   = 1.;
//   const double A0  = B0/sqrt(mu0*rho);
// };




// Tearing mode
namespace TearingModeData{
  void   uFun_ex( const Vector & x, const double t, Vector & u );
  double pFun_ex( const Vector & x, const double t             );
  double zFun_ex( const Vector & x, const double t             );
  double aFun_ex( const Vector & x, const double t             );
  void   fFun(    const Vector & x, const double t, Vector & f );
  void   nFun(    const Vector & x, const double t, Vector & f );
  double gFun(    const Vector & x, const double t             );
  double hFun(    const Vector & x, const double t             );
  double mFun(    const Vector & x, const double t             );
  void   wFun(    const Vector & x, const double t, Vector & w );
  double qFun(    const Vector & x, const double t             );
  double cFun(    const Vector & x, const double t             );
  double yFun(    const Vector & x, const double t             );
  const double _lambda = 5.;
  const double _Lx     = 3.;
  const double _beta   = 1e-3;
  const double _mu     = 1e-2;
  const double _eta    = 1e-2;
  const double _mu0    = 1.;
  const std::string _pbName   = "TearingMode";
  // const std::string _meshFile ="./meshes/quad-rect-tearing-xper.mesh";
  // const std::string _meshFile ="./meshes/tri-rect-tearing-sym.mesh";
  const std::string _meshFile ="./meshes/tri-rect-tearing.mesh";
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
};


// Tearing mode (flipped, to debug oscillations in y component)
namespace TearingModeFlippedData{
  void   uFun_ex( const Vector & x, const double t, Vector & u );
  double pFun_ex( const Vector & x, const double t             );
  double zFun_ex( const Vector & x, const double t             );
  double aFun_ex( const Vector & x, const double t             );
  void   fFun(    const Vector & x, const double t, Vector & f );
  void   nFun(    const Vector & x, const double t, Vector & f );
  double gFun(    const Vector & x, const double t             );
  double hFun(    const Vector & x, const double t             );
  double mFun(    const Vector & x, const double t             );
  void   wFun(    const Vector & x, const double t, Vector & w );
  double qFun(    const Vector & x, const double t             );
  double cFun(    const Vector & x, const double t             );
  double yFun(    const Vector & x, const double t             );
  const double _mu     = TearingModeData::_mu;
  const double _eta    = TearingModeData::_eta;
  const double _mu0    = TearingModeData::_mu0;
  const std::string _pbName   = "TearingModeFlipped";
  // const std::string _meshFile ="./meshes/quad-rect-tearing-xper.mesh";
  // const std::string _meshFile ="./meshes/tri-rect-tearing-sym.mesh";
  const std::string _meshFile ="./meshes/tri-rect-tearing-flip.mesh";
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
};



// // Tilt mode
// namespace TiltModeData{
//   void   uFun_ex( const Vector & x, const double t, Vector & u );
//   double pFun_ex( const Vector & x, const double t             );
//   double zFun_ex( const Vector & x, const double t             );
//   double aFun_ex( const Vector & x, const double t             );
//   void   fFun(    const Vector & x, const double t, Vector & f );
//   void   nFun(    const Vector & x, const double t, Vector & f );
//   double gFun(    const Vector & x, const double t             );
//   double hFun(    const Vector & x, const double t             );
//   double mFun(    const Vector & x, const double t             );
//   void   wFun(    const Vector & x, const double t, Vector & w );
//   double qFun(    const Vector & x, const double t             );
//   double cFun(    const Vector & x, const double t             );
//   double yFun(    const Vector & x, const double t             );
//   void xy2rth(const Vector & x, Vector& r); // map from cartesian to cylindrical coords
//   const double _k    = 3.83170597020751; // zero of bessel function of first kind, k : J1(k)=0
//   const double _J0k  = -0.4027593957026; // J0(k)
//   const double _beta = 1e-3;
//   const double _mu   = 1e-2;
//   const double _eta  = 1e-2;
//   const double _mu0  = 1.;
//   const std::string _pbName   = "TiltMode";
//   // const std::string _meshFile ="./meshes/tri-rect-tilt-xper.mesh";
//   const std::string _meshFile ="./meshes/tri-rect-tilt.mesh";
//   void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
// }






// Driven cavity flow
namespace CavityDrivenData{
  void   uFun_ex( const Vector & x, const double t, Vector & u );
  double pFun_ex( const Vector & x, const double t             );
  double zFun_ex( const Vector & x, const double t             );
  double aFun_ex( const Vector & x, const double t             );
  void   fFun(    const Vector & x, const double t, Vector & f );
  void   nFun(    const Vector & x, const double t, Vector & f );
  double gFun(    const Vector & x, const double t             );
  double hFun(    const Vector & x, const double t             );
  double mFun(    const Vector & x, const double t             );
  void   wFun(    const Vector & x, const double t, Vector & w );
  double qFun(    const Vector & x, const double t             );
  double cFun(    const Vector & x, const double t             );
  double yFun(    const Vector & x, const double t             );
  const double _mu  = 1e-2;
  const double _eta = 1.;
  const double _mu0 = 1.;
  const std::string _pbName   = "DrivenCavity";
  const std::string _meshFile ="./meshes/tri-square-cavity.mesh";
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA );
};
















}
#endif
















// namespace mfem{

// // THIS IS USELESS! MFEM NEEDS A POINTER TO FUNCTION, NOT A POINTER TO MEMBER FUNCTION!
// // - Use template?

// //***************************************************************************
// // BASE CLASS FOR TEST CASE INFO
// //***************************************************************************
// class TestCase{

// protected:
//   typedef  void   (TestCase::*VFuncPtrT)( const Vector & x, const double t, Vector & u ); // type of functor for vector functions
//   typedef  double (TestCase::*SFuncPtrT)( const Vector & x, const double t );             // type of functor for scalar functions

//   const double _mu;
//   const double _eta;
//   const double _mu0;

//   Array<int> _essTagsU;
//   Array<int> _essTagsV;
//   Array<int> _essTagsP;
//   Array<int> _essTagsA;

//   const std::string _pbName;
//   const std::string _meshFile;

// public:

//   // Constructor with parameter values
//   TestCase(double mu, double eta, double mu0, std::string name, std::string meshFile ):
//   _mu(mu),_eta(eta),_mu0(mu0),_pbName(name),_meshFile(meshFile){};
//   // Default constructor
//   TestCase()
//   : TestCase(1,1,1,"UnnamedTest","./meshes/tri-square-testAn.mesh"){}; // only from c++11


//   // Functions to use for rhs data
//   virtual void   uFun_ex( const Vector & x, const double t, Vector & u) const = 0;
//   virtual double pFun_ex( const Vector & x, const double t )            const = 0;
//   virtual double zFun_ex( const Vector & x, const double t )            const = 0;
//   virtual double aFun_ex( const Vector & x, const double t )            const = 0;
//   virtual void   fFun(    const Vector & x, const double t, Vector & f) const = 0;
//   virtual void   nFun(    const Vector & x, const double t, Vector & n) const = 0;
//   virtual double gFun(    const Vector & x, const double t )            const = 0;
//   virtual double hFun(    const Vector & x, const double t )            const = 0;
//   virtual double mFun(    const Vector & x, const double t )            const = 0;
//   // Functions to use for initialisation
//   // - set everything to IC by default
//   void   wFun( const Vector & x, const double t, Vector & w) const {        uFun_ex(x,0,w); };
//   double qFun( const Vector & x, const double t )            const { return pFun_ex(x,0);   };
//   double yFun( const Vector & x, const double t )            const { return zFun_ex(x,0);   };
//   double cFun( const Vector & x, const double t )            const { return aFun_ex(x,0);   };

//   // Functions to use to recover Pb parameters
//   double getMu(  ) const { return _mu;  };
//   double getEta( ) const { return _eta; };
//   double getMu0( ) const { return _mu0; };

//   // Functions to use to recover type of BC
//   Array<int> getEssTagsU() const { return _essTagsU; };
//   Array<int> getEssTagsV() const { return _essTagsV; };
//   Array<int> getEssTagsP() const { return _essTagsP; };
//   Array<int> getEssTagsA() const { return _essTagsA; };

//   // Returns name of meshfile
//   std::string getMeshFile() const { return _meshFile; };
  
//   // Returns name of problem
//   std::string getPbName() const {   return _pbName; };

// };







// //***************************************************************************
// // Analytical test case 4
// //***************************************************************************
// class Analytical4: public TestCase{
// private:
//   const double _delta = 0.07957747154595;

//   // - define a perturbation to dirty initial guess
//   double perturbation(const Vector & x, const double t){
//     double epsilon = 1.;
//     double xx(x(0));
//     double yy(x(1));
//     return( t * epsilon * 0.25*( ( cos(2*M_PI*xx)-1 )*(cos(2*M_PI*yy)-1) ) );
//   }



// public:

//   Analytical4()
//   : TestCase(1,1,1,"An4","./meshes/tri-square-testAn.mesh"){
//     // Set BC:
//     // - Dirichlet on u everywhere but on E
//     // - Dirichlet on p on E (outflow, used only in precon)
//     // - Dirichlet on A on W
//     _essTagsU.SetSize(3); _essTagsU[0] = 1; _essTagsU[1] = 3; _essTagsU[2] = 4; // N, S, w
//     _essTagsV = _essTagsU;
//     _essTagsP.SetSize(1); _essTagsP[0] = 2; // E
//     _essTagsA.SetSize(1); _essTagsA[0] = 4; // W
//   };

  
// // Analytical test-case
// // - Pick a div-free field
// void uFun_ex(const Vector & x, const double t, Vector & u) const {
//   double xx(x(0));
//   double yy(x(1));
//   u(0) =   t * xx*xx*yy;
//   u(1) = - t * xx*yy*yy;
// }
// // - Pick a pressure which is null on boundaries
// double pFun_ex(const Vector & x, const double t ) const {
//   double xx(x(0));
//   double yy(x(1));
//   return ( t * sin(M_PI*xx)*sin(M_PI*yy) );
// }
// // - laplacian of vector potential
// double zFun_ex(const Vector & x, const double t ) const {
//   double xx(x(0));
//   double yy(x(1));
//   return ( -t * 2*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) );
// }
// // - vector potential with null normal derivative
// double aFun_ex(const Vector & x, const double t ) const {
//   double xx(x(0));
//   double yy(x(1));
//   return ( t * cos(M_PI*xx)*cos(M_PI*yy) );
// }
// // - rhs of velocity counteracts action of every term
// void fFun(const Vector & x, const double t, Vector & f) const {
//   double xx(x(0));
//   double yy(x(1));
//   //      dt u     + u Grad(u)            - mu Lap(u)     + Grad(p)                            + z grad(A) / mu0
//   f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - _mu* t * 2*yy + t * M_PI*cos(M_PI*xx)*sin(M_PI*yy) + zFun_ex(x,t) * (-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy)) / _mu0;
//   f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + _mu* t * 2*xx + t * M_PI*sin(M_PI*xx)*cos(M_PI*yy) + zFun_ex(x,t) * (-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy)) / _mu0;
// }
// // - normal derivative of velocity
// void nFun(const Vector & x, const double t, Vector & n) const {
//   double xx(x(0));
//   double yy(x(1));
//   n = 0.;
//   if ( yy == 0. || yy == 1.){ //N/S
//     n(0) =    xx*xx;
//     n(1) = -2*xx*yy;
//     if ( yy == 0. ){
//       n*=-1.;
//     }
//   }
//   if ( xx == 0. || xx == 1. ){ //E/W
//     n(0) = 2*xx*yy;
//     n(1) = -yy*yy;
//     if ( xx == 0. ){
//       n*=-1.;
//     }
//   }

//   n *= t;
// }
// // - null rhs of pressure
// double gFun(const Vector & x, const double t ) const {
//   return 0.;
// }
// // - rhs of vector potential counteracts every term
// double hFun( const Vector & x, const double t ) const {
//   double xx(x(0));
//   double yy(x(1));
//   Vector u(2);
//   uFun_ex(x,t,u);
//   double ugradA = u(0)*(-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy))
//                 + u(1)*(-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy));
//   //       dtA                       + u Grad(A) - eta/mu0 lap(A)
//   return ( cos(M_PI*xx)*cos(M_PI*yy) + ugradA    - _eta/_mu0 * zFun_ex(x,t) );
// }
// // - null normal derivative of vector potential
// double mFun( const Vector & x, const double t ) const {
//   return 0.;
// }
// // - define perturbed IG for every linearised variable
// void wFun(const Vector & x, const double t, Vector & w) const {
//   uFun_ex(x,t,w);
//   double ds = perturbation(x,t);
//   w(0) = w(0) + ds;
//   w(1) = w(1) - ds;
// }
// double qFun(  const Vector & x, const double t ) const {
//   double ds = perturbation(x,t);
//   return ( pFun_ex(x,t) + ds );
// }
// double yFun(  const Vector & x, const double t ) const {
//   double ds = perturbation(x,t);
//   return ( zFun_ex(x,t) + ds );
// }
// double cFun(  const Vector & x, const double t ) const {
//   double ds = perturbation(x,t);
//   return ( aFun_ex(x,t) + ds );
// }




// };










// //***************************************************************************
// // Kelvin-Helmholtz instability
// //***************************************************************************
// class KelvinHelmholtzInstability: public TestCase{
// private:
//   const double _delta = 0.07957747154595;

// public:

//   KelvinHelmholtzInstability()
//   : TestCase(1,1,1,"KHI","./meshes/tri-rect-KHI.mesh"){
//     // Top+bottom=1 topLeft+botRight=2 botLeft+topRight=3
//     // - Dirichlet on u,v on top left and bottom right
//     // - Dirichlet on v   on top and bottom
//     // - Dirichlet on v   on top right and bottom left
//     // - No stress on normal component (u) on top right and bottom left
//     // - No stress on tangential component (u) top and bottom
//     // - Dirichlet on p on top right and bottom left (outflow, used only in precon)
//     // - Dirichlet on A on top and bottom
//     _essTagsU.SetSize(1); _essTagsU[0] = 2;
//     _essTagsV.SetSize(3); _essTagsV[0] = 1; _essTagsV[1] = 2; _essTagsV[2] = 3;
//     _essTagsP.SetSize(1); _essTagsP[0] = 3;
//     _essTagsA.SetSize(1); _essTagsA[0] = 1;    
//   };

  
//   // - smoothly vary from top velocity ~ 1.5 to bottom velocity of ~ -1.5, ramping up in time
//   void uFun_ex(const Vector & x, const double t, Vector & u) const {
//     double yy(x(1));
//     u(0) = std::min( t, 1. ) * atan(6*M_PI*(yy-.5));
//     u(1) = 0.;
//   }
//   // - pressure - unused
//   double pFun_ex(const Vector & x, const double t ) const {
//     return 0.;
//     // return _P0 - ( 1./ ( 2* cosh( yy/_delta ) * cosh( yy/_delta ) ) ); // if you want p to be in equilibrium with A with u=0
//   }
//   // - laplacian of vector potential
//   double zFun_ex(const Vector & x, const double t ) const {
//     double yy(x(1));
//     return ( 1./ ( _delta * cosh( yy/_delta ) * cosh( yy/_delta ) ) );
//   }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ) const {
//     double yy(x(1));
//     return ( _delta * log( cosh( yy/_delta ) ) );
//   }
//   // - rhs of velocity - unused
//   void fFun(const Vector & x, const double t, Vector & f) const {
//     f = 0.;
//   }
//   // - normal derivative of velocity
//   void nFun(const Vector & x, const double t, Vector & n) const {
//     n = 0.;
//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ) const {
//     return 0.;
//   }
//   // - rhs of vector potential - unused
//   double hFun( const Vector & x, const double t ) const {
//     return 0.;
//     // return -eta * ( 1./ ( _delta * cosh( yy/_delta ) * cosh( yy/_delta ) ) ); // if you want A to be at equilibrium with u=0

//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ) const {
//     double yy(x(1));

//     if( yy ==  1.0 ){
//       return   sinh(yy/_delta) / cosh(yy/_delta);
//     }else if( yy ==  0.0 ){
//       return - sinh(yy/_delta) / cosh(yy/_delta);
//     }

//     return 0.;
//   }


// };








// //***************************************************************************
// // Island coalescence
// //***************************************************************************
// class IslandCoalescence: public TestCase{

// private:
//   const double _delta = 1./(2.*M_PI);
//   const double _P0    = 1.;
//   const double _eps   = 0.4;

// public:
//   IslandCoalescence()
//   : TestCase(1e-2,1e-2,1,"IslandCoalescence","./meshes/tri-square-island.mesh"){
//     // Top=1 Left+Right=2 Bottom=3
//     // - Dirichlet on u on left and right
//     // - Dirichlet on v on top and bottom
//     // - No stress on tangential component (u) top and bottom
//     // - No stress on tangential component (v) left, right and bottom
//     // - Dirichlet on p on top (outflow?, used only in precon?)
//     // - Dirichlet on A on top
//     _essTagsU.SetSize(1); _essTagsU[0] = 2;
//     _essTagsV.SetSize(2); _essTagsV[0] = 1; _essTagsV[1] = 3;
//     _essTagsP.SetSize(1); _essTagsP[0] = 1;
//     _essTagsA.SetSize(1); _essTagsA[0] = 1;
//   };


//   void uFun_ex(const Vector & x, const double t, Vector & u) const {
//     u = 0.;
//   }
//   // - pressure
//   double pFun_ex(const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));

//     double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);

//     return ( _P0 + ( 1. - _eps*_eps ) / ( 2.*temp*temp ) )/_mu0;
//   }
//   // - laplacian of vector potential
//   double zFun_ex(const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));

//     double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);

//     if ( t==0 ){
//       return ( ( 1. - _eps*_eps ) / ( _delta * temp*temp ) - 0.001* 5./4.*M_PI*M_PI*cos(M_PI*.5*yy)*cos(M_PI*xx) );  // perturb IC
//     }

//     return ( ( 1. - _eps*_eps ) / ( _delta * temp*temp ) );
//   }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));
//     double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);

//     if ( t==0 ){
//       return ( _delta * log( temp ) + 0.001*cos(M_PI*.5*yy)*cos(M_PI*xx) );  // perturb IC
//     }
//     return ( _delta * log( temp ) );
//   }
//   // - rhs of velocity - unused
//   void fFun(const Vector & x, const double t, Vector & f) const {
//     f = 0.;
//   }
//   // - normal derivative of velocity
//   void nFun(const Vector & x, const double t, Vector & n) const {
//     n = 0.;
//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ) const {
//     return 0.;
//   }
//   // - rhs of vector potential
//   double hFun( const Vector & x, const double t ) const {
//     return - _eta/_mu0 * zFun_ex(x,t) ;
//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));

//     double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);  

//     if(xx == 1.0){
//       return - _eps*sin(xx/_delta) / temp;
//     }else if( xx == 0.0 ){
//       return   _eps*sin(xx/_delta) / temp;
//     }else if( yy ==  1.0 ){
//       return   sinh(yy/_delta) / temp;
//     }else if( yy == 0.0 ){    // this way I also cover the case of a domain [0,1]x[0,1]
//       return 0.;
//     }else if( yy == -1.0 ){
//       return - sinh(yy/_delta) / temp;
//     }

//     return 0.;
//   }



// };






// //***************************************************************************
// // Tearing mode
// //***************************************************************************
// class TearingMode: public TestCase{

// private:
//   const double _lambda = 5.;
//   const double _Lx     = 3.;
//   const double _beta   = 1e-3;

// public:
//   TearingMode()
//   : TestCase(1,1,1,"TearingMode","./meshes/tri-rect-tearing.mesh"){
//     // Top=1 Left+Right=2 Bottom=3
//     // - Dirichlet on u on left and right
//     // - Dirichlet on v on top and bottom
//     // - No stress on tangential component (u) top and bottom
//     // - No stress on tangential component (v) left, right and bottom
//     // - Dirichlet on p on top (outflow?, used only in precon?)
//     // - Dirichlet on A on top
//     _essTagsU.SetSize(1); _essTagsU[0] = 2;
//     _essTagsV.SetSize(2); _essTagsV[0] = 1; _essTagsV[1] = 3;
//     _essTagsP.SetSize(1); _essTagsP[0] = 1;
//     _essTagsA.SetSize(1); _essTagsA[0] = 1;
//   };



//   void uFun_ex(const Vector & x, const double t, Vector & u) const {
//     u = 0.;
//   }
//   // - pressure
//   double pFun_ex(const Vector & x, const double t ) const {
//     double yy(x(1));
//     return 1./(2*_lambda*cosh(_lambda*yy)*cosh(_lambda*yy)) / _mu0;  //if you want unperturbed A to be equilibrium
//   }
//   // - laplacian of vector potential
//   double zFun_ex(const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));
    
//     if ( t==0 ){
//       return _lambda/(cosh(_lambda*yy)*cosh(_lambda*yy)) - _beta*( sin(M_PI*(yy+.5))*cos(2.0*M_PI/_Lx*(xx+3./2.)*M_PI*M_PI*(4./_Lx+1.) )); // perturb IC by _beta
//     }
//     return _lambda/(cosh(_lambda*yy)*cosh(_lambda*yy));
//   }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));

//     if ( t==0 ){
//       return log(cosh(_lambda*yy))/_lambda + _beta*sin(M_PI*(yy+.5))*cos(2.0*M_PI/_Lx*(xx+3./2.)); // perturb IC by _beta
//     }
//     return log(cosh(_lambda*yy))/_lambda;
//   }
//   // - rhs of velocity - unused
//   void fFun(const Vector & x, const double t, Vector & f) const {
//     f = 0.;
//   }
//   // - normal derivative of velocity
//   void nFun(const Vector & x, const double t, Vector & n) const {
//     n = 0.;
//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ) const {
//     return 0.;
//   }
//   // - rhs of vector potential - unused
//   double hFun( const Vector & x, const double t ) const {
//     return -_eta/_mu0*zFun_ex(x,t);  //if you want unperturbed A to be equilibrium
//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ) const {
//     double xx(x(0));
//     double yy(x(1));
//     if ( xx == 0. || xx == 1.5 || xx == -1.5 ){
//       return 0.;
//     }
//     if ( yy == 0. || yy == 0.5 || yy == -0.5 ){
//       return abs(tanh(_lambda*yy)); // if yy negative, must be multiplied by -1: this takes care of that
//     }
//     return 0.;
//   }




// };













// //***************************************************************************
// // MHD Rayleigh flow - UNTESTED
// //***************************************************************************
// class Rayleigh: public TestCase{

// private:
//   static const double _U   = 1.;
//   static const double _B0  = 1.4494e-4;
//   static const double _rho = 0.4e-4;
//   static const double _mu0 = 1.256636e-6;
//   static const double _eta = 1.256636e-6;
//   static const double _mu  = 0.4e-4;
//   static const double _d   = 1.;
//   static const double _A0  = _B0/sqrt(_mu0*_rho);

// public:

//   std::string GetPbName(){return "MHDRayleigh";};  
//   std::string GetMeshFile(){return "./meshes/tri-square-rayleigh.mesh";};

//   double getMu() {return ( _mu  / _rho ) };
//   double getMu0(){return ( _mu0 * _rho ) };
//   double getEta(){return ( _eta * _rho ) };


//   void uFun_ex(const Vector & x, const double t, Vector & u){
//     double xx(x(0));
//     double yy(x(1));

//     u(0) = U/4. * (  exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) )  -  erf( (yy-A0*t)/(2*sqrt(d*t)) ) 
//                    + exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) )  -  erf( (yy+A0*t)/(2*sqrt(d*t)) ) + 2. );
//     u(1) = 0.;

//   }
//   // - pressure - unused?
//   double pFun_ex(const Vector & x, const double t ){
//     return 0.;
//   }
//   // - laplacian of vector potential
//   double zFun_ex(const Vector & x, const double t ){
//     double dh = 1e-8;

//     // use centered differences
//     Vector xp=x, xm=x;
//     xp(1) += dh;
//     xm(1) -= dh;
//     return ( aFun_ex(xp,t) - 2*aFun_ex(x,t) + aFun_ex(xm,t) ) / (dh*dh);

//     // z = - U/4.*sqrt(mu*rho/(M_PI*d*t)) * (-exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))*(yy-A0*t)/(2*d*t) * (yy-A0*t)          +  exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))
//     //                                       +exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t))*(yy+A0*t)/(2*d*t) * (yy+A0*t)          -  exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
//     //     + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) *(A0*t-yy)/(2*d*t)
//     //                                                 - exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) *(A0*t+yy)/(2*d*t) ) / (2*sqrt(d*t))
//     //     - U/4.*sqrt(mu*rho)/A0 *( -A0/d*exp(-A0*yy/d) * ( ( -(A0+A0*A0*exp(A0*yy/d)*yy/d) + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
//     //                                                        + (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) ) )
//     //     - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * ( ( -( A0*A0/d*( A0*exp(A0*yy/d)/d*yy + exp(A0*yy/d) ) ) 
//     //                                                 +  A0*( A0*exp(A0*yy/d)/d + A0/d*exp(A0*yy/d) + A0*A0/d*exp(A0*yy/d)/d*yy ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
//     //                      ( -(A0+A0*A0*exp(A0*yy/d)*yy/d)  + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) ) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/(2*sqrt(d*t)) 
//     //                                                   - (d+A0*exp(A0*yy/d)*yy) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))  * (yy-A0*t)/(2*d*t)    /sqrt(d*t*M_PI) 
//     //                                                   + (A0*A0*exp(A0*yy/d)/d*yy + A0*exp(A0*yy/d)) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/sqrt(d*t*M_PI) 
//     //                                                   )
//     //     THIS IS DONE
//     //     - U/4.*sqrt(mu*rho)/A0 * (   (A0*exp(A0*yy/d)-A0)   * 2/sqrt(M_PI)*exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) / (2*sqrt(d*t))
//     //                                + (A0*A0*exp(A0*yy/d)/d) * erfc((A0*t+yy)/(2*sqrt(d*t)))
//     //                                + (A0*exp(A0*yy/d)-A0)   * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) 
//     //                                - (d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) * 2*(A0*t+yy)/(4*d*t) /(2*sqrt(d*t)) );
//   }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ){
//     double xx(x(0));
//     double yy(x(1));

//     double a = -B0*xx + U/2.*sqrt(d*t*mu*rho/M_PI) * ( exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) - exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
//                       + U/4.*sqrt(mu*rho)/A0 * (d+A0*A0*t) * ( erf((A0*t-yy)/(2*sqrt(d*t))) - erf((A0*t+yy)/(2*sqrt(d*t))) )
//                       - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * erfc((yy-A0*t)/(2*sqrt(d*t)))
//                       - U/4.*sqrt(mu*rho)/A0 * (d*exp(A0*yy/d)-A0*yy) * erfc((A0*t+yy)/(2*sqrt(d*t)));
//     return a;
//   }
//   // - rhs of velocity - unused
//   void fFun(const Vector & x, const double t, Vector & f){
//     f = 0.;
//   }
//   // - normal derivative of velocity
//   void nFun(const Vector & x, const double t, Vector & n){
//     double xx(x(0));
//     double yy(x(1));
//     n = 0.;

//     if( yy==0 || yy==5 )
//       return;
    
//     if( xx==0 || xx==5 ){
//       n(0) = U/4. * (  -A0/d*exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp(-A0*yy/d)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) )/(2*sqrt(d*t))
//                        -  1./sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) ) /(2*sqrt(d*t)) 
//                        +A0/d*exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp( A0*yy/d)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) )/(2*sqrt(d*t))
//                        -  1./sqrt(M_PI)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) ) /(2*sqrt(d*t)) );
//       if (xx==0)
//         n *= -1.;
//     }

//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ){
//     return 0.;
//   }
//   // - rhs of vector potential - unused
//   double hFun( const Vector & x, const double t ){
//     return B0*U/2.;
//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ){
//     double xx(x(0));
//     double yy(x(1));

//     double m = 0.;

//     if ( xx==0)
//       m = B0;
//     if ( xx==5)
//       m = -B0;
//     if ( yy==0 || yy==5 ){
//       m =   U/2.*sqrt(d*t*mu*rho/M_PI) * ( exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * (-(yy-A0*t)/(2*d*t)) - exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) * (-(yy+A0*t)/(2*d*t)) )
//           + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) + exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) ) / (2*sqrt(d*t))
//           - U/4.*sqrt(mu*rho)/A0 * ( -A0/d * exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * erfc((yy-A0*t)/(2*sqrt(d*t)))
//                                     + exp(-A0*yy/d) * (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) * erfc((yy-A0*t)/(2*sqrt(d*t)))
//                                     + exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) )
//           - U/4.*sqrt(mu*rho)/A0 * ( (A0*exp(A0*yy/d)-A0) * erfc((A0*t+yy)/(2*sqrt(d*t)))
//                                      (d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) );
//       if( yy==0 )
//         m *= -1.;
//     }

//     return m;
//   }

// };


