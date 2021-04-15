#ifndef TEST_CASES_HPP
#define TEST_CASES_HPP

#include "mfem.hpp"
#include <string>

//***************************************************************************
//TEST CASES OF SOME ACTUAL RELEVANCE
//***************************************************************************


NEED TO THINK ABOUT HOW TO HANDLE THIS!!!


namespace mfem{
namespace testCases{

// Generic interface class for test cases info
class TestCase{
private:
  typedef  void   (TestCase::*VFuncPtrT)( const Vector & x, const double t, Vector & u ); // type of functor for vector functions
  typedef  double (TestCase::*SFuncPtrT)( const Vector & x, const double t );             // type of functor for scalar functions


public:

  // Functions to use for rhs data
  virtual void   uFun_ex( const Vector & x, const double t, Vector & u) = 0;
  virtual double pFun_ex( const Vector & x, const double t )            = 0;
  virtual double zFun_ex( const Vector & x, const double t )            = 0;
  virtual double aFun_ex( const Vector & x, const double t )            = 0;
  virtual void   fFun(    const Vector & x, const double t, Vector & f) = 0;
  virtual void   nFun(    const Vector & x, const double t, Vector & n) = 0;
  virtual double gFun(    const Vector & x, const double t )            = 0;
  virtual double hFun(    const Vector & x, const double t )            = 0;
  virtual double mFun(    const Vector & x, const double t )            = 0;
  // Functions to use for initialisation
  // - set everything to IC by default
  void   wFun( const Vector & x, const double t, Vector & w){        uFun_ex(x,0,w); };
  double qFun( const Vector & x, const double t )           { return pFun_ex(x,0);   };
  double yFun( const Vector & x, const double t )           { return zFun_ex(x,0);   };
  double cFun( const Vector & x, const double t )           { return aFun_ex(x,0);   };

  // Functions to use to recover Pb parameters
  // - set everything to 1 by default
  double getMu(  ){return 1.};
  double getMu0( ){return 1.};
  double getEta( ){return 1.};

  // Functions to use to recover type of BC
  void getUEssBdr( Array<int>& essBdr ){ essBdr.SetSize(4); essBdr(0)=1; essBdr(1)=2; essBdr(2)=3; essBdr(3)=4; };  // set four sides as dirichlet
  void getVEssBdr( Array<int>& essBdr ){ getUEssBdr( essBdr ); };                                                   // copy U
  void getPEssBdr( Array<int>& essBdr ){ essBdr.SetSize(0);    };                                                   // no dirichlet on P
  void getAEssBdr( Array<int>& essBdr ){ getUEssBdr( essBdr ); };                                                   // copy U

  // Returns name of meshfile
  std::string GetMeshFile(){ return "./meshes/tri-square-testAn.mesh"; };
  
  // Returns name of problem
  virtual std::string GetPbName() = 0;



}





//***************************************************************************
// Kelvin-Helmholtz instability
//***************************************************************************
class KHI: public TestCase{
private:
  static const double _delta = 0.07957747154595;

public:
  std::string GetPbName(){return "KHI";};  
  
  // - top velocity of 1.5, bottom velocity of -1.5
  void uFun_ex(const Vector & x, const double t, Vector & u){
    u(0) = 0.;
    u(1) = 0.;

    if ( x(1) >= 0.5 ){
      u(0) =  1.5;
    }else{
      u(0) = -1.5;
    }
  }
  // - pressure - unused
  double pFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - laplacian of vector potential - unused
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


};








//***************************************************************************
// Island coalescence
//***************************************************************************
class IslandCoalescence: public TestCase{

private:
  static const double _delta = 1./(2.*M_PI);
  static const double _P0 = 1.;
  static const double _epsilon = 0.4;

public:
  std::string GetPbName(){return "IslandCoalescence";};  


  void uFun_ex(const Vector & x, const double t, Vector & u){
    u = 0.;
  }
  // - pressure
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _epsilon*cos(xx/_delta);

    return ( _P0 + ( 1. - _epsilon*_epsilon ) / ( 2.*temp*temp ) );
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _epsilon*cos(xx/_delta);

    return ( ( 1. - _epsilon*_epsilon ) / ( _delta * temp*temp ) );
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _epsilon*cos(xx/_delta);

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
  // - rhs of vector potential - unused
  double hFun( const Vector & x, const double t ){
    return 0.;
  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double temp = cosh(yy/_delta) + _epsilon*cos(xx/_delta);  

    if(xx == 1.0){
      return - _epsilon*sin(xx/_delta) / temp;
    }else if( xx == 0.0 ){
      return   _epsilon*sin(xx/_delta) / temp;
    }else if( yy ==  1.0 ){
      return   sinh(yy/_delta) / temp;
    }else if( yy == -1.0 ){
      return - sinh(yy/_delta) / temp;
    }

    return 0.;
  }


};







//***************************************************************************
// MHD Rayleigh flow - UNTESTED
//***************************************************************************
class Rayleigh: public TestCase{

private:
  static const double _U   = 1.;
  static const double _B0  = 1.4494e-4;
  static const double _rho = 0.4e-4;
  static const double _mu0 = 1.256636e-6;
  static const double _eta = 1.256636e-6;
  static const double _mu  = 0.4e-4;
  static const double _d   = 1.;
  static const double _A0  = _B0/sqrt(_mu0*_rho);

public:

  std::string GetPbName(){return "MHDRayleigh";};  
  std::string GetMeshFile(){return "./meshes/tri-square-rayleigh.mesh";};

  double getMu() {return ( _mu  / _rho ) };
  double getMu0(){return ( _mu0 * _rho ) };
  double getEta(){return ( _eta * _rho ) };


  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));

    u(0) = U/4. * (  exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) )  -  erf( (yy-A0*t)/(2*sqrt(d*t)) ) 
                   + exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) )  -  erf( (yy+A0*t)/(2*sqrt(d*t)) ) + 2. );
    u(1) = 0.;

  }
  // - pressure - unused?
  double pFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double dh = 1e-8;

    // use centered differences
    Vector xp=x, xm=x;
    xp(1) += dh;
    xm(1) -= dh;
    return ( aFun_ex(xp,t) - 2*aFun_ex(x,t) + aFun_ex(xm,t) ) / (dh*dh);

    // z = - U/4.*sqrt(mu*rho/(M_PI*d*t)) * (-exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))*(yy-A0*t)/(2*d*t) * (yy-A0*t)          +  exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))
    //                                       +exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t))*(yy+A0*t)/(2*d*t) * (yy+A0*t)          -  exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
    //     + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) *(A0*t-yy)/(2*d*t)
    //                                                 - exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) *(A0*t+yy)/(2*d*t) ) / (2*sqrt(d*t))
    //     - U/4.*sqrt(mu*rho)/A0 *( -A0/d*exp(-A0*yy/d) * ( ( -(A0+A0*A0*exp(A0*yy/d)*yy/d) + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
    //                                                        + (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) ) )
    //     - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * ( ( -( A0*A0/d*( A0*exp(A0*yy/d)/d*yy + exp(A0*yy/d) ) ) 
    //                                                 +  A0*( A0*exp(A0*yy/d)/d + A0/d*exp(A0*yy/d) + A0*A0/d*exp(A0*yy/d)/d*yy ) )* erfc((yy-A0*t)/(2*sqrt(d*t)))
    //                      ( -(A0+A0*A0*exp(A0*yy/d)*yy/d)  + (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) ) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/(2*sqrt(d*t)) 
    //                                                   - (d+A0*exp(A0*yy/d)*yy) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))  * (yy-A0*t)/(2*d*t)    /sqrt(d*t*M_PI) 
    //                                                   + (A0*A0*exp(A0*yy/d)/d*yy + A0*exp(A0*yy/d)) * exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t))/sqrt(d*t*M_PI) 
    //                                                   )
    //     THIS IS DONE
    //     - U/4.*sqrt(mu*rho)/A0 * (   (A0*exp(A0*yy/d)-A0)   * 2/sqrt(M_PI)*exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) / (2*sqrt(d*t))
    //                                + (A0*A0*exp(A0*yy/d)/d) * erfc((A0*t+yy)/(2*sqrt(d*t)))
    //                                + (A0*exp(A0*yy/d)-A0)   * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) 
    //                                - (d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) * 2*(A0*t+yy)/(4*d*t) /(2*sqrt(d*t)) );
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double a = -B0*xx + U/2.*sqrt(d*t*mu*rho/M_PI) * ( exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) - exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) )
                      + U/4.*sqrt(mu*rho)/A0 * (d+A0*A0*t) * ( erf((A0*t-yy)/(2*sqrt(d*t))) - erf((A0*t+yy)/(2*sqrt(d*t))) )
                      - U/4.*sqrt(mu*rho)/A0 * exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * erfc((yy-A0*t)/(2*sqrt(d*t)))
                      - U/4.*sqrt(mu*rho)/A0 * (d*exp(A0*yy/d)-A0*yy) * erfc((A0*t+yy)/(2*sqrt(d*t)));
    return a;
  }
  // - rhs of velocity - unused
  void fFun(const Vector & x, const double t, Vector & f){
    f = 0.;
  }
  // - normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    double xx(x(0));
    double yy(x(1));
    n = 0.;

    if( yy==0 || yy==5 )
      return;
    
    if( xx==0 || xx==5 ){
      n(0) = U/4. * (  -A0/d*exp(-A0*yy/d)*erfc( (yy-A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp(-A0*yy/d)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) )/(2*sqrt(d*t))
                       -  1./sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t) ) /(2*sqrt(d*t)) 
                       +A0/d*exp( A0*yy/d)*erfc( (yy+A0*t)/(2*sqrt(d*t)) ) + 1./sqrt(M_PI)*exp( A0*yy/d)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) )/(2*sqrt(d*t))
                       -  1./sqrt(M_PI)*exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t) ) /(2*sqrt(d*t)) );
      if (xx==0)
        n *= -1.;
    }

  }
  // - rhs of pressure - unused
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential - unused
  double hFun( const Vector & x, const double t ){
    return B0*U/2.;
  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));

    double m = 0.;

    if ( xx==0)
      m = B0;
    if ( xx==5)
      m = -B0;
    if ( yy==0 || yy==5 ){
      m =   U/2.*sqrt(d*t*mu*rho/M_PI) * ( exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * (-(yy-A0*t)/(2*d*t)) - exp(-(yy+A0*t)*(yy+A0*t)/(4*d*t)) * (-(yy+A0*t)/(2*d*t)) )
          + U/2.*sqrt(mu*rho/M_PI)/A0 * (d+A0*A0*t) * ( exp(-(A0*t-yy)*(A0*t-yy)/(4*d*t)) + exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) ) / (2*sqrt(d*t))
          - U/4.*sqrt(mu*rho)/A0 * ( -A0/d * exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * erfc((yy-A0*t)/(2*sqrt(d*t)))
                                    + exp(-A0*yy/d) * (d+A0*( exp(A0*yy/d) + A0/d*exp(A0*yy/d)*yy ) ) * erfc((yy-A0*t)/(2*sqrt(d*t)))
                                    + exp(-A0*yy/d) * (d+A0*exp(A0*yy/d)*yy) * 2/sqrt(M_PI)*exp(-(yy-A0*t)*(yy-A0*t)/(4*d*t)) * 1/(2*sqrt(d*t)) )
          - U/4.*sqrt(mu*rho)/A0 * ( (A0*exp(A0*yy/d)-A0) * erfc((A0*t+yy)/(2*sqrt(d*t)))
                                     (d*exp(A0*yy/d)-A0*yy) * exp(-(A0*t+yy)*(A0*t+yy)/(4*d*t)) /(2*sqrt(d*t)) );
      if( yy==0 )
        m *= -1.;
    }

    return m;
  }

};










};
};





#endif