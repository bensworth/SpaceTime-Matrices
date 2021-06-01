#include "mfem.hpp"
#include "testcases.hpp"
#include <cmath>


namespace mfem{


//***************************************************************************
// Analytical test case 4
//***************************************************************************
namespace Analytical4Data{
  // - define a perturbation to dirty initial guess
  double perturbation(const Vector & x, const double t){
    double epsilon = 1.;
    double xx(x(0));
    double yy(x(1));
    return( t * epsilon * 0.25*( ( cos(2*M_PI*xx)-1 )*(cos(2*M_PI*yy)-1) ) );
  }
  // - Pick a div-free field
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));
    u(0) =   t * xx*xx*yy;
    u(1) = - t * xx*yy*yy;
  }
  // - Pick a pressure which is null on boundaries
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( t * sin(M_PI*xx)*sin(M_PI*yy) );
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( -t * 2*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) );
  }
  // - vector potential with null normal derivative
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( t * cos(M_PI*xx)*cos(M_PI*yy) );
  }
  // - rhs of velocity counteracts action of every term
  void fFun(const Vector & x, const double t, Vector & f){
    double xx(x(0));
    double yy(x(1));
    //      dt u     + u Grad(u)            - mu Lap(u)     + Grad(p)                            + z grad(A) / mu0
    f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - _mu* t * 2*yy + t * M_PI*cos(M_PI*xx)*sin(M_PI*yy) + zFun_ex(x,t) * (-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy)) / _mu0;
    f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + _mu* t * 2*xx + t * M_PI*sin(M_PI*xx)*cos(M_PI*yy) + zFun_ex(x,t) * (-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy)) / _mu0;
  }
  // - normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    double xx(x(0));
    double yy(x(1));
    n = 0.;
    if ( yy == 0. || yy == 1.){ //N/S
      n(0) =    xx*xx;
      n(1) = -2*xx*yy;
      if ( yy == 0. ){
        n*=-1.;
      }
    }
    if ( xx == 0. || xx == 1. ){ //E/W
      n(0) = 2*xx*yy;
      n(1) = -yy*yy;
      if ( xx == 0. ){
        n*=-1.;
      }
    }

    n *= t;
  }
  // - null rhs of pressure
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential counteracts every term
  double hFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    Vector u(2);
    uFun_ex(x,t,u);
    double ugradA = u(0)*(-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy))
                  + u(1)*(-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy));
    //       dtA                       + u Grad(A) - eta/mu0 lap(A)
    return ( cos(M_PI*xx)*cos(M_PI*yy) + ugradA    - _eta/_mu0 * zFun_ex(x,t) );
  }
  // - null normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    return 0.;
  }
  // - define perturbed IG for every linearised variable
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,t,w);
    double ds = perturbation(x,t);
    w(0) = w(0) + ds;
    w(1) = w(1) - ds;
  }
  double qFun(  const Vector & x, const double t ){
    double ds = perturbation(x,t);
    return ( pFun_ex(x,t) + ds );
  }
  double yFun(  const Vector & x, const double t ){
    double ds = perturbation(x,t);
    return ( zFun_ex(x,t) + ds );
  }
  double cFun(  const Vector & x, const double t ){
    double ds = perturbation(x,t);
    return ( aFun_ex(x,t) + ds );
  }


  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Set BC:
    // - Dirichlet on u everywhere but on E
    // - Dirichlet on p on E (outflow, used only in precon)
    // - Dirichlet on A on W
    essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
    essTagsV = essTagsU;
    essTagsP.SetSize(1); essTagsP[0] = 2; // E
    essTagsA.SetSize(1); essTagsA[0] = 4; // W
  }

}




// //***************************************************************************
// // Modified Hartmann flow TODO
// //***************************************************************************
// // Hartmann flow
// namespace HartmannData{
//   // - velocity
//   void uFun_ex(const Vector & x, const double t, Vector & u){
//     u = 0.;

//     u(0) = _G0/_B0 * ( cosh(1.) - cosh(x(1)) )/sinh(1.);
//   }
//   // - pressure
//   double pFun_ex(const Vector & x, const double t ){
//     return -_G0*( x(0) + 1.0 );    // zero at outflow (x=-1)
//   }
//   // - laplacian of vector potential - unused
//   double zFun_ex(const Vector & x, const double t ){

//     return -_G0/_B0 * ( 1. - cosh(x(1))/sinh(1.) );
//   }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ){
//     return -_B0*x(0) - _G0/_B0 * ( x(1)*x(1)/2. - cosh(x(1))/sinh(1.) );
//   }
//   // - rhs of velocity - unused
//   void fFun(const Vector & x, const double t, Vector & f){
//     f = 0.;
//   }
//   // - normal derivative of velocity
//   void nFun(const Vector & x, const double t, Vector & n){
//     const double yy(x(1));
//     n = 0.;

//     if ( yy== 1. ){
//       n(1) = -_G0/_B0 * sinh(yy)/sinh(1.);
//     }
//     if ( yy==-1. ){
//       n(1) =  _G0/_B0 * sinh(yy)/sinh(1.);
//     }
//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ){
//     return 0.;
//   }
//   // - rhs of vector potential
//   double hFun( const Vector & x, const double t ){
//     const double _G0  = HartmannData::_G0;
//     const double _B0  = HartmannData::_B0;

//     return - _G0/_B0 * ( cosh(1.)/sinh(1.) - 1. );
//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ){
//     const double xx(x(0));
//     const double yy(x(1));

//     if ( yy==1. || yy == -1. ){
//       return yy*( -_G0/_B0 * ( yy - sinh(yy)/sinh(1.) ) );
//     }
//     if ( xx==1. || xx == -1. ){
//       return -_B0*xx;
//     }
//     return 0.;
//   }
//   // - define perturbed IG for every linearised variable
//   void wFun(const Vector & x, const double t, Vector & w){
//     uFun_ex(x,t,w);
//     double ds = perturbation(x,t);
//     w(0) = w(0) + ds;
//     w(1) = w(1) - ds;

//   }
//   double qFun(  const Vector & x, const double t ){
//     double ds = perturbation(x,t);
//     return pFun_ex(x,t) + ds;
//   }
//   double yFun(  const Vector & x, const double t ){
//     double ds = perturbation(x,t);
//     return zFun_ex(x,t) + ds;
//   }
//   double cFun(  const Vector & x, const double t ){
//     double ds = perturbation(x,t);
//     return aFun_ex(x,t) + ds;
//   }

//   void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
//     // Set BC:
//     // - Dirichlet on u everywhere but on E
//     // - Dirichlet on p on E (outflow, used only in precon)
//     // - Dirichlet on A on W
//     essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
//     essTagsV = essTagsU;
//     essTagsP.SetSize(1); essTagsP[0] = 2; // E
//     essTagsA.SetSize(2); essTagsA[0] = 1; essTagsA[1] = 3; // N, S
//   }

// }






















//***************************************************************************
// Kelvin-Helmholtz instability
//***************************************************************************
namespace KHIData{
  // - smoothly vary from top velocity ~ 1.5 to bottom velocity of ~ -1.5, ramping up in time
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double yy(x(1));
    u(0) = std::min( t, 1. ) * atan(6*M_PI*(yy-.5));
    u(1) = 0.;
  }
  // - pressure - unused
  double pFun_ex(const Vector & x, const double t ){
    return 0.;
    // return _P0 - ( 1./ ( 2* cosh( yy/_delta ) * cosh( yy/_delta ) ) ); // if you want p to be in equilibrium with A with u=0
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double yy(x(1));
    return ( 1./ ( _delta * cosh( yy/_delta ) * cosh( yy/_delta ) ) );
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double yy(x(1));
    return ( _delta * log( cosh( yy/_delta ) ) );
  }
  // - rhs of velocity - unused
  void fFun(const Vector & x, const double t, Vector & f){
    f = 0.;
  }
  // - normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    n = 0.;
  }
  // - rhs of pressure - unused
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential - unused
  double hFun( const Vector & x, const double t ){
    return 0.;
    // return -eta * ( 1./ ( _delta * cosh( yy/_delta ) * cosh( yy/_delta ) ) ); // if you want A to be at equilibrium with u=0

  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    double yy(x(1));

    if( yy ==  1.0 ){
      return   sinh(yy/_delta) / cosh(yy/_delta);
    }else if( yy ==  0.0 ){
      return - sinh(yy/_delta) / cosh(yy/_delta);
    }

    return 0.;
  }
  // - define IG for every linearised variable -> set them to initial conditions?
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,t,w);
  }
  double qFun(  const Vector & x, const double t ){
    return pFun_ex(x,t);
  }
  double yFun(  const Vector & x, const double t ){
    return zFun_ex(x,t);
  }
  double cFun(  const Vector & x, const double t ){
    return aFun_ex(x,t);
  }

  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Set BC:
    // Top+bottom=1 topLeft+botRight=2 botLeft+topRight=3
    // - Dirichlet on u,v on top left and bottom right
    // - Dirichlet on v   on top and bottom
    // - Dirichlet on v   on top right and bottom left
    // - No stress on normal component (u) on top right and bottom left
    // - No stress on tangential component (u) top and bottom
    // - Dirichlet on p on top right and bottom left (outflow, used only in precon)
    // - Dirichlet on A on top and bottom
    essTagsU.SetSize(1); essTagsU[0] = 2; //essTagsU[1] = 1;
    essTagsV.SetSize(3); essTagsV[0] = 1; essTagsV[1] = 2; essTagsV[2] = 3;
    essTagsP.SetSize(1); essTagsP[0] = 3;
    essTagsA.SetSize(1); essTagsA[0] = 1;
  }
}











//***************************************************************************
// Island Coalescence
//***************************************************************************
namespace IslandCoalescenceData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    u = 0.;
  }
  // - pressure
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);

    return ( _P0 + ( 1. - _eps*_eps ) / ( 2.*temp*temp ) )/_mu0;
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);

    if ( t==0 ){
      return ( ( 1. - _eps*_eps ) / ( _delta * temp*temp ) - _beta* 5./4.*M_PI*M_PI*cos(M_PI*.5*yy)*cos(M_PI*xx) );  // perturb IC
    }

    return ( ( 1. - _eps*_eps ) / ( _delta * temp*temp ) );
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);

    if ( t==0 ){
      return ( _delta * log( temp ) + _beta*cos(M_PI*.5*yy)*cos(M_PI*xx) );  // perturb IC
    }
    return ( _delta * log( temp ) );
  }
  // - rhs of velocity - unused
  void fFun(const Vector & x, const double t, Vector & f){
    f = 0.;
  }
  // - normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    n = 0.;
  }
  // - rhs of pressure - unused
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential
  double hFun( const Vector & x, const double t ){
    return - _eta/_mu0 * zFun_ex(x,t) ;
  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _eps*cos(xx/_delta);  

    if(xx == 1.0){
      return - _eps*sin(xx/_delta) / temp;
    }else if( xx == 0.0 ){
      return   _eps*sin(xx/_delta) / temp;
    }else if( yy ==  1.0 ){
      return   sinh(yy/_delta) / temp;
    }else if( yy == 0.0 ){    // this way I also cover the case of a domain [0,1]x[0,1]
      return 0.;
    }else if( yy == -1.0 ){
      return - sinh(yy/_delta) / temp;
    }

    return 0.;
  }
  // - define IG for every linearised variable -> set them to initial conditions?
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,t,w);
  }
  double qFun(  const Vector & x, const double t ){
    return pFun_ex(x,t);
  }
  double yFun(  const Vector & x, const double t ){
    return zFun_ex(x,t);
  }
  double cFun(  const Vector & x, const double t ){
    return aFun_ex(x,t);
  }

  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // // Set BC: [0,1]x[-1,1]
    // // Top+bottom=1 Left+Right=2
    // // - Dirichlet on u on left and right
    // // - Dirichlet on v on top and bottom
    // // - No stress on tangential component (u) top and bottom
    // // - No stress on tangential component (v) left and right
    // // - Dirichlet on p on top and bottom (outflow, used only in precon)
    // // - Dirichlet on A on top and bottom
    // mesh_file = "./meshes/tri-rect-island.mesh";
    // essTagsU.SetSize(1); essTagsU[0] = 2;
    // essTagsV.SetSize(1); essTagsV[0] = 1;
    // essTagsP.SetSize(1); essTagsP[0] = 1;
    // essTagsA.SetSize(1); essTagsA[0] = 1;
    // Set BC: [0,1]x[0,1]
    // Top=1 Left+Right=2 Bottom=3
    // - Dirichlet on u on left and right
    // - Dirichlet on v on top and bottom
    // - No stress on tangential component (u) top and bottom
    // - No stress on tangential component (v) left, right and bottom
    // - Dirichlet on p on top (outflow?, used only in precon?)
    // - Dirichlet on A on top
    essTagsU.SetSize(1); essTagsU[0] = 2;
    essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3;
    // essTagsP.SetSize(1); essTagsP[0] = 1;
    essTagsP.SetSize(0);
    essTagsA.SetSize(1); essTagsA[0] = 1;
    std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
  }

}





//***************************************************************************
// MHD Rayleigh flow
//***************************************************************************
// namespace RayleighData{
//   void uFun_ex(const Vector & x, const double t, Vector & u){
//     using namespace RayleighData;
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

//     // z=- U/4.*sqrt(mu*rho/(M_PI*d*t)) * (-exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))*(yy-A0*t)/(2*d*t) * (yy-A0*t)          +  exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))
//     //                                     +exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t))*(yy+A0*t)/(2*d*t) * (yy+A0*t)          -  exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
//     //   + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) *(A0*t-yy)/(2*d*t)
//     //                                               - exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) *(A0*t+yy)/(2*d*t) ) / (2*sqrt(d*t))
//     //   - U/4.*sqrt(mu*rho)/A0 *( -A0/d*exp(-A0*yy/d) * ( ( -(A0+A0*A0*exp(A0*yy/d)*yy/d) + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
//     //                                                      + (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) ) )
//     //   - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * ( ( -( A0*A0/d*( A0*exp(A0*yy/d)/d*yy + exp(A0*yy/d) ) ) 
//     //                                               +  A0*( A0*exp(A0*yy/d)/d + A0/d*exp(A0*yy/d) + A0*A0/d*exp(A0*yy/d)/d*yy ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
//     //                    ( -(A0+A0*A0*exp(A0*yy/d)*yy/d)  + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) ) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/(2*sqrt(d*t)) 
//     //                                                 - (d+A0*exp(A0*yy/d)*yy) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))  * (yy-A0*t)/(2*d*t)    /sqrt(d*t*M_PI) 
//     //                                                 + (A0*A0*exp(A0*yy/d)/d*yy + A0*exp(A0*yy/d)) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/sqrt(d*t*M_PI) 
//     //                                                 )
//     //   THIS IS DONE
//     //   - U/4.*sqrt(mu*rho)/A0 * (   (A0*exp(A0*yy/d)-A0)   * 2/sqrt(M_PI)*exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) / (2*sqrt(d*t))
//     //                              + (A0*A0*exp(A0*yy/d)/d) * erfc((A0*t+yy)/(2*sqrt(d*t)))
//     //                              + (A0*exp(A0*yy/d)-A0)   * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) 
//     //                              - (d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) * 2*(A0*t+yy)/(4*d*t) /(2*sqrt(d*t)) );
//   }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ){
//     using namespace RayleighData;
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
//     // using namespace RayleighData;
//     // double xx(x(0));
//     // double yy(x(1));
//     n = 0.;

//     // if( yy==0 || yy==5 )
//     //   return;
    
//     // if( xx==0 || xx==5 ){
//     //   n(0) = U/4. * (  -A0/d*exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp(-A0*yy/d)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) )/(2*sqrt(d*t))
//     //                    -  1./sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) ) /(2*sqrt(d*t)) 
//     //                    +A0/d*exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp( A0*yy/d)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) )/(2*sqrt(d*t))
//     //                    -  1./sqrt(M_PI)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) ) /(2*sqrt(d*t)) );
//     //   if (xx==0)
//     //     n *= -1.;
//     // }

//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ){
//     return 0.;
//   }
//   // - rhs of vector potential - unused
//   double hFun( const Vector & x, const double t ){
//     using namespace RayleighData;
//     return B0*U/2.;
//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ){
//     using namespace RayleighData;
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
//                                     +(d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) );
//       if( yy==0 )
//         m *= -1.;
//     }

//     return m;
//   }
//   // - define perturbed IG for every linearised variable
//   void wFun(const Vector & x, const double t, Vector & w){
//     uFun_ex(x,t,w);
//   }
//   double qFun(  const Vector & x, const double t ){
//     return pFun_ex(x,t);
//   }
//   double yFun(  const Vector & x, const double t ){
//     return zFun_ex(x,t);
//   }
//   double cFun(  const Vector & x, const double t ){
//     return aFun_ex(x,t);
//   }
// }










//***************************************************************************
// Tearing mode
//***************************************************************************
namespace TearingModeData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    u = 0.;
  }
  // - pressure
  double pFun_ex(const Vector & x, const double t ){
    double yy(x(1));
    return 1./(2*cosh(_lambda*(yy-0.5))*cosh(_lambda*(yy-0.5))) / _mu0;  //if you want unperturbed A to be equilibrium
    // re-mapped on symmetric domain:
    // return 1./(2*cosh(_lambda*yy)*cosh(_lambda*yy)) / _mu0;  //if you want unperturbed A to be equilibrium
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double val = _lambda/(cosh(_lambda*(yy-0.5))*cosh(_lambda*(yy-0.5)));
    if ( t==0 )
      val -= _beta*( sin(M_PI*yy)*cos(2.0*M_PI/_Lx*xx)*M_PI*M_PI*(4./(_Lx*_Lx)+1.)); // perturb IC by _beta

    return val;
    
    // // re-mapped on symmetric domain:
    // if ( t==0 ){
    //   return _lambda/(cosh(_lambda*yy)*cosh(_lambda*yy)) - _beta*( sin(M_PI*(yy+.5))*cos(2.0*M_PI/_Lx*(xx+3./2.))*M_PI*M_PI*(4./(_Lx*_Lx)+1.) ); // perturb IC by _beta
    // }
    // return _lambda/(cosh(_lambda*yy)*cosh(_lambda*yy));
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double val = log(cosh(_lambda*(yy-0.5)))/_lambda;
    if ( t==0 )
      val += _beta*sin(M_PI*yy)*cos(2.0*M_PI/_Lx*xx); // perturb IC by _beta

    return val;

    // re-mapped on symmetric domain:
    // if ( t==0 ){
    //   return log(cosh(_lambda*yy))/_lambda + _beta*sin(M_PI*(yy+.5))*cos(2.0*M_PI/_Lx*(xx+3./2.)); // perturb IC by _beta
    // }
    // return log(cosh(_lambda*yy))/_lambda;
  }
  // - rhs of velocity - unused
  void fFun(const Vector & x, const double t, Vector & f){
    f = 0.;
  }
  // - normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    n = 0.;
  }
  // - rhs of pressure - unused
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential - unused
  double hFun( const Vector & x, const double t ){
    return -_eta/_mu0*zFun_ex(x,t);  //if you want unperturbed A to be equilibrium
  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    if ( xx == 0. || xx == 3. || xx == 1.5 ){ // this takes care also of the symmetric case
      return 0.;
    }
    if ( yy == 0. || yy == 1. || yy == 0.5 ){
      double val = tanh(_lambda*(yy-0.5));
      if ( yy == 0. )
        val*=-1;
      return val;
    }

    // re-mapped on symmetric domain:
    // if ( xx == 0. || xx == 1.5 || xx == -1.5 ){
    //   return 0.;
    // }
    // if ( yy == 0. || yy == 0.5 || yy == -0.5 ){
    //   return abs(tanh(_lambda*yy)); // if yy negative, must be multiplied by -1: this takes care of that
    // }
    return 0.;
  }
  // - define IG for every linearised variable -> set them to equilibrium
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,t,w);
  }
  double qFun(  const Vector & x, const double t ){
    return pFun_ex(x,t);
  }
  double yFun(  const Vector & x, const double t ){
    return zFun_ex(x,t);
  }
  double cFun(  const Vector & x, const double t ){
    return aFun_ex(x,t);
  }

  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // periodic in x
    // // Top+Bottom=1
    // // - Dirichlet on v on top and bottom
    // // - No stress on tangential component (u) top and bottom
    // // - Dirichlet on p on top and bottom
    // // - Dirichlet on A on top and bottom
    // essTagsU.SetSize(0);
    // essTagsV.SetSize(1); essTagsV[0] = 1;
    // // essTagsP.SetSize(1); essTagsP[0] = 1;
    // essTagsA.SetSize(1); essTagsA[0] = 1;
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
    // whole domain, no periodicity
    // Top+Bottom=1 Left+Right=2
    // - Dirichlet on u on left and right
    // - Dirichlet on v on top and bottom
    // - No stress on tangential component (u) top and bottom
    // - No stress on tangential component (v) left and right
    // - Dirichlet on p on top and bottom
    // - Dirichlet on A on top and bottom
    essTagsU.SetSize(1); essTagsU[0] = 2;
    essTagsV.SetSize(1); essTagsV[0] = 1;
    // essTagsP.SetSize(1); essTagsP[0] = 1;
    essTagsA.SetSize(1); essTagsA[0] = 1;
    std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
    // exploit symmetric structure
    // // Top=1 Left+Right=2 Bottom=3
    // // - Dirichlet on u on left and right
    // // - Dirichlet on v on top and bottom
    // // - No stress on tangential component (u) top and bottom
    // // - No stress on tangential component (v) left, right and bottom
    // // - Dirichlet on p on top (outflow?, used only in precon?)
    // // - Dirichlet on A on top
    // essTagsU.SetSize(1); essTagsU[0] = 2;
    // essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3;
    // // essTagsP.SetSize(1); essTagsP[0] = 1;
    // essTagsP.SetSize(0);
    // essTagsA.SetSize(1); essTagsA[0] = 1;
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
  }


}



// Tearing mode (flipped, to debug oscillations in y component)
namespace TearingModeFlippedData{
  void   uFun_ex(const Vector& x,const double t,Vector& u){Vector xx(2);xx(0)=x(1);xx(1)=x(0);       TearingModeData::uFun_ex(xx,t,u);double temp=u(0);u(0)=u(1);u(1)=temp;}
  double pFun_ex(const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::pFun_ex(xx,t  );}
  double zFun_ex(const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::zFun_ex(xx,t  );}
  double aFun_ex(const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::aFun_ex(xx,t  );}
  void   fFun(   const Vector& x,const double t,Vector& f){Vector xx(2);xx(0)=x(1);xx(1)=x(0);       TearingModeData::fFun(   xx,t,f);double temp=f(0);f(0)=f(1);f(1)=temp;}
  void   nFun(   const Vector& x,const double t,Vector& f){Vector xx(2);xx(0)=x(1);xx(1)=x(0);       TearingModeData::nFun(   xx,t,f);double temp=f(0);f(0)=f(1);f(1)=temp;}
  double gFun(   const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::gFun(   xx,t  );}
  double hFun(   const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::hFun(   xx,t  );}
  double mFun(   const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::mFun(   xx,t  );}
  void   wFun(   const Vector& x,const double t,Vector& w){Vector xx(2);xx(0)=x(1);xx(1)=x(0);       TearingModeData::wFun(   xx,t,w);double temp=w(0);w(0)=w(1);w(1)=temp;}
  double qFun(   const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::qFun(   xx,t  );}
  double cFun(   const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::cFun(   xx,t  );}
  double yFun(   const Vector& x,const double t          ){Vector xx(2);xx(0)=x(1);xx(1)=x(0);return TearingModeData::yFun(   xx,t  );}
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    return TearingModeData::setEssTags(essTagsV,essTagsU,essTagsP,essTagsA);  // flip U and V, too
  }
};









//***************************************************************************
// Tilt mode
//***************************************************************************
//  INITIAL A IS NOT PERIODIC!!!!!
// namespace TiltModeData{
//   // map from cartesian to cylindrical coords
//   void xy2rth(const Vector & x, Vector& r){
//     double xx(x(0));
//     double yy(x(1));
//     double rr = sqrt(xx*xx+yy*yy);
//     double th = atan(yy/xx);
//     if ( rr==0. ){
//       th=0.;
//     }
//     if ( xx<0 ){
//       th += M_PI;
//     }
//     r(0) = rr;
//     r(1) = th;
//   }
//   // Bessel function of first kind and its derivatives
//   double J0(double x){
//     double fct = 1.;
//     double sum = 0.;
//     for(int i = 0; i < 8; fct*=++i) {
//       sum += std::pow(-1, i)*std::pow((x/2.),2.*i) / std::pow(fct,2);
//     }
//     return sum;
//   }
//   double dJ0(double x){
//     double fct = 1.;
//     double sum = 0.;
//     for(int i = 1; i < 8; fct*=++i) {
//       sum += std::pow(-1, i)*std::pow((x/2.),2.*i-1)*(2.*i) / std::pow(fct,2);
//     }
//     return sum;
//   }
//   double ddJ0(double x){
//     double fct = 1.;
//     double sum = 0.;
//     for(int i = 1; i < 8; fct*=++i) {
//       sum += std::pow(-1, i)*std::pow((x/2.),2.*i-2)*(2.*i)*(2.*i-1) / std::pow(fct,2);
//     }
//     return sum;
//   }

//   // - velocity
//   void uFun_ex(const Vector & x, const double t, Vector & u){
//     u = 0.;
//   }
//   // - pressure
//   double pFun_ex(const Vector & x, const double t ){
//     return 0.;
//   }
//   // - laplacian of vector potential
//   double zFun_ex(const Vector & x, const double t ){
//     double eps = 1e-7;
//     Vector xphx(x); xphx(0)+=eps;
//     Vector xmhx(x); xmhx(0)-=eps;
//     Vector xphy(x); xphy(1)+=eps;
//     Vector xmhy(x); xmhy(1)-=eps;
//     // CD approx of laplacian
//     return (-4*aFun_ex(x,t) + aFun_ex(xphx,t) + aFun_ex(xmhx,t) + aFun_ex(xphy,t) + aFun_ex(xmhy,t) )/ (eps*eps);
//   }  
//   //   // map to cyl
//   //   Vector myX(x); myX *= 2*M_PI;
//   //   Vector r(2);
//   //   xy2rth(myX,r);
//   //   double rr(r(0));
//   //   double th(r(1));

//   //   double val;

//   //   if ( rr >= 1. ){
//   //     val = (1./rr-1./(rr*rr*rr))*cos(th) - (rr-1./rr)*cos(th)/(rr*rr);
//   //   }else{
//   //     if(rr == 0.){
//   //       dJ0r = - 1;  //dJ0(0)/r
//   //       J0rr = ?????; //  J0(0)/r^2
//   //       val = 2./( _k*_J0k ) * ( ( _k*dJ0r + _k*_k*ddJ0(_k*rr) )*cos(th) - J0rr*cos(th) );
//   //     }else{
//   //       val = 2./( _k*_J0k ) * ( ( _k*dJ0(_k*rr) + _k*_k*rr*ddJ0(_k*rr) )*cos(th)/rr - J0(_k*rr)*cos(th)/(rr*rr) );
//   //     }
//   //   }
//   //   // perturb IC
//   //   if ( t==0 ){
//   //     val += _beta * 4*exp(-rr*rr)*(rr*rr-1.);
//   //   }
//     // return val / (4*M_PI*M_PI);
//   // }
//   // - vector potential
//   double aFun_ex(const Vector & x, const double t ){
//     // map to cyl
//     Vector myX(x); myX *= 2*M_PI;
//     myX(0) *= 2*M_PI;
//     Vector r(2);
//     xy2rth(myX,r);
//     double rr(r(0));
//     double th(r(1));

//     double val;

//     if ( rr >= 1. ){
//       val = (rr-1./rr) * cos(th);
//     }else{
//       val = 2./( _k*_J0k ) * J0(_k*rr) * cos(th);
//     }
//     // perturb IC
//     if ( t==0 ){
//       val += _beta * exp(-rr*rr);
//     }
//     return val;
//   }
//   // - rhs of velocity - unused
//   void fFun(const Vector & x, const double t, Vector & f){
//     f = 0.;
//   }
//   // - normal derivative of velocity
//   void nFun(const Vector & x, const double t, Vector & n){
//     n = 0.;
//   }
//   // - rhs of pressure - unused
//   double gFun(const Vector & x, const double t ){
//     return 0.;
//   }
//   // - rhs of vector potential
//   double hFun( const Vector & x, const double t ){
//     return - _eta/_mu0 * zFun_ex(x,t) ;
//   }
//   // - normal derivative of vector potential
//   double mFun( const Vector & x, const double t ){
//     // map to cyl
//     Vector myX(x); myX *= 2*M_PI;
//     Vector r(2);
//     xy2rth(myX,r);
//     double rr(r(0));
//     double th(r(1));
//     // get gradient
//     Vector dA(2), dAx(2);
//     // - get gradient in cyl
//     if ( rr >= 1. ){
//       dA(0) =   (1.+1./(rr*rr)) * cos(th);
//       dA(1) = - (1.-1./(rr*rr)) * sin(th);
//     }else{
//       dA(0) =   2./( _k*_J0k ) * dJ0(_k*rr) * cos(th) * _k;
//       dA(1) = - 2./( _k*_J0k ) *  J0(_k*rr) * sin(th)/rr;
//     }
//     // - get jacobian of transformation
//     DenseMatrix J(2);
//     J(0,0)=cos(th); J(0,1)=-rr*sin(th);
//     J(1,0)=sin(th); J(1,1)= rr*cos(th);
//     // - mult to get grad in cart
//     J.Inverse()->Mult( dA, dAx );
//     dAx *= 1./(2*M_PI);

//     if ( abs(x(0)) == 1. ){
//       return x(0) * dAx(0);
//     }else{ //if ( abs(x(1)) == 5./(2*M_PI) ){
//       std::cout<<"ERROR! you should only impose neumann on left or right!"<<std::endl;
//       return 0.;
//     }
//     return 0.;
//   }
//   // - define IG for every linearised variable -> set them to initial conditions?
//   void wFun(const Vector & x, const double t, Vector & w){
//     uFun_ex(x,t,w);
//   }
//   double qFun(  const Vector & x, const double t ){
//     return pFun_ex(x,t);
//   }
//   double yFun(  const Vector & x, const double t ){
//     return zFun_ex(x,t);
//   }
//   double cFun(  const Vector & x, const double t ){
//     return aFun_ex(x,t);
//   }

//   void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
//     // For periodic mesh
//     // // Set BC: [-1,1]x[-5/2pi,5/2pi]
//     // // Top+bottom=1
//     // // - Periodic in x
//     // // - Dirichlet on v on top and bottom
//     // // - No stress on tangential component (u) top and bottom
//     // // - Dirichlet on p on top and bottom (outflow, used only in precon?)
//     // // - Dirichlet on A on top and bottom
//     // essTagsU.SetSize(0);
//     // essTagsV.SetSize(1); essTagsV[0] = 1;
//     // // essTagsP.SetSize(1); essTagsP[0] = 1;
//     // essTagsP.SetSize(0);
//     // essTagsA.SetSize(1); essTagsA[0] = 1;
//     // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;


//     // Set BC: [-1,1]x[-5/2pi,5/2pi]
//     // Top+bottom=1 Left+Right=2
//     // - Dirichlet on u on left and right
//     // - Dirichlet on v on top and bottom
//     // - No stress on tangential component (u) top and bottom
//     // - No stress on tangential component (v) left and right
//     // - Dirichlet on p on top and bottom (outflow, used only in precon?)
//     // - Dirichlet on A on top and bottom
//     essTagsU.SetSize(1); essTagsU[0] = 2;
//     essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3;
//     // essTagsP.SetSize(1); essTagsP[0] = 1;
//     essTagsP.SetSize(0);
//     essTagsA.SetSize(1); essTagsA[0] = 1;
//     std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
//   }

// }















//***************************************************************************
// Cavity driven flow
//***************************************************************************
namespace CavityDrivenData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));
    u = 0.;

    if( yy==1.0 ){
      u(0) = t * 8*xx*(1-xx)*(2.*xx*xx-2.*xx+1.);   // regularised (1-x^4 mapped from -1,1 to 0,1)
      // u(0) = t;                      // leaky
      // if( xx > 1.0 && xx < 1.0 )
      //   u(0) = t;                    // watertight
    }
  }
  // - pressure
  double pFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of velocity - unused
  void fFun(const Vector & x, const double t, Vector & f){
    f = 0.;
  }
  // - normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    n = 0.;
  }
  // - rhs of pressure - unused
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential - unused
  double hFun( const Vector & x, const double t ){
    return 0.;
  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    return 0.;
  }
  // - define IG for every linearised variable -> set them to initial conditions?
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,0,w);
  }
  double qFun(  const Vector & x, const double t ){
    return pFun_ex(x,0);
  }
  double yFun(  const Vector & x, const double t ){
    return zFun_ex(x,0);
  }
  double cFun(  const Vector & x, const double t ){
    return aFun_ex(x,0);
  }

  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Top 1 Right 2 Bottom=3 Left=4
    // - Dirichlet on u, v, and A everywhere
    essTagsU.SetSize(4); essTagsU[0] = 1; essTagsU[1] = 2; essTagsU[2] = 3; essTagsU[3] = 4;
    essTagsV = essTagsU;
    essTagsP.SetSize(0);
    essTagsA.SetSize(4); essTagsA[0] = 1; essTagsA[1] = 2; essTagsA[2] = 3; essTagsA[3] = 4;
  }


}

































}






















// namespace mfem{
// namespace testcases{

// 	// Static variables definition
// 	// - Generic test:
// 	template< class T > constexpr double TestBase<T>::_mu  = 1.;
// 	template< class T > constexpr double TestBase<T>::_eta = 1.;
// 	template< class T > constexpr double TestBase<T>::_mu0 = 1.;

//   template< class T > constexpr std::array<int> TestBase<T>::_essTagsU = {0};
//   template< class T > constexpr std::array<int> TestBase<T>::_essTagsV = {0};
//   template< class T > constexpr std::array<int> TestBase<T>::_essTagsP = {0};
//   template< class T > constexpr std::array<int> TestBase<T>::_essTagsA = {0};

//   template< class T > constexpr std::string TestBase<T>::_pbName   = "UnnamedTest";
//   template< class T > constexpr std::string TestBase<T>::_meshFile = "./meshes/tri-square-testAn.mesh";





// 	// - Kelvin-Helmholtz Instability
// 	template< > constexpr double TestBase<KelvinHelmholtzInstability>::_mu  = 1e-1;
// 	template< > constexpr double TestBase<KelvinHelmholtzInstability>::_eta = 1e-1;
// 	template< > constexpr double TestBase<KelvinHelmholtzInstability>::_mu0 = 1.;

//   template< > constexpr std::array<int> TestBase<KelvinHelmholtzInstability>::_essTagsU = Array<int>;
//   template< > constexpr std::array<int> TestBase<KelvinHelmholtzInstability>::_essTagsV = Array<int>;
//   template< > constexpr std::array<int> TestBase<KelvinHelmholtzInstability>::_essTagsP = Array<int>;
//   template< > constexpr std::array<int> TestBase<KelvinHelmholtzInstability>::_essTagsA = Array<int>;

//   template< > constexpr std::string TestBase<KelvinHelmholtzInstability>::_pbName   = "KelvinHelmholtzInstability";
//   template< > constexpr std::string TestBase<KelvinHelmholtzInstability>::_meshFile = "./meshes/tri-rect-KHI.mesh";



// }
// }


