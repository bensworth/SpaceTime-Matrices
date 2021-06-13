#include "mfem.hpp"
#include "testcases.hpp"
#include <cmath>


namespace mfem{

void MHDTestCaseSelector( int pbType, 
                          VecFuncPtr_t &uFun,       FuncPtr_t &pFun,   FuncPtr_t &zFun,    FuncPtr_t &aFun,
                          VecFuncPtr_t &fFun,       FuncPtr_t &gFun,   FuncPtr_t &hFun, VecFuncPtr_t &nFun,   FuncPtr_t &mFun,
                          VecFuncPtr_t &wFun,       FuncPtr_t &qFun,   FuncPtr_t &yFun,    FuncPtr_t &cFun,
                          double& _mu, double& _eta, double& _mu0,
                          std::string &pbName, std::string &meshFile,
                          Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA,
                          VecFuncPtr_t *guFun,  VecFuncPtr_t *gvFun,                    VecFuncPtr_t *gaFun ){

  switch (pbType){
    // Analytical test-cases ------------------------------------------------
    // Constant in space
    case 0:{
      meshFile = ConstantTData::_meshFile;
      pbName   = ConstantTData::_pbName;

      uFun = ConstantTData::uFun_ex;
      pFun = ConstantTData::pFun_ex;
      zFun = ConstantTData::zFun_ex;
      aFun = ConstantTData::aFun_ex;
      fFun = ConstantTData::fFun;
      gFun = ConstantTData::gFun;
      hFun = ConstantTData::hFun;
      nFun = ConstantTData::nFun;
      mFun = ConstantTData::mFun;
      wFun = ConstantTData::wFun;
      qFun = ConstantTData::qFun;
      yFun = ConstantTData::yFun;
      cFun = ConstantTData::cFun;

      _mu  = ConstantTData::_mu;
      _eta = ConstantTData::_eta;
      _mu0 = ConstantTData::_mu0;
      
      ConstantTData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }

    // Quadratic in space, constant in time
    case 1:{
      meshFile = QuadraticData::_meshFile;
      pbName   = QuadraticData::_pbName;

      uFun = QuadraticData::uFun_ex;
      pFun = QuadraticData::pFun_ex;
      zFun = QuadraticData::zFun_ex;
      aFun = QuadraticData::aFun_ex;
      fFun = QuadraticData::fFun;
      gFun = QuadraticData::gFun;
      hFun = QuadraticData::hFun;
      nFun = QuadraticData::nFun;
      mFun = QuadraticData::mFun;
      wFun = QuadraticData::wFun;
      qFun = QuadraticData::qFun;
      yFun = QuadraticData::yFun;
      cFun = QuadraticData::cFun;

      _mu  = QuadraticData::_mu;
      _eta = QuadraticData::_eta;
      _mu0 = QuadraticData::_mu0;
      
      QuadraticData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }

    // Cubic in space, constant in time
    case 2:{
      meshFile = CubicData::_meshFile;
      pbName   = CubicData::_pbName;

      uFun = CubicData::uFun_ex;
      pFun = CubicData::pFun_ex;
      zFun = CubicData::zFun_ex;
      aFun = CubicData::aFun_ex;
      fFun = CubicData::fFun;
      gFun = CubicData::gFun;
      hFun = CubicData::hFun;
      nFun = CubicData::nFun;
      mFun = CubicData::mFun;
      wFun = CubicData::wFun;
      qFun = CubicData::qFun;
      yFun = CubicData::yFun;
      cFun = CubicData::cFun;

      _mu  = CubicData::_mu;
      _eta = CubicData::_eta;
      _mu0 = CubicData::_mu0;
      
      CubicData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }



    // Cubic in space, Linear in time
    case 3:{
      meshFile = CubicTData::_meshFile;
      pbName   = CubicTData::_pbName;

      uFun = CubicTData::uFun_ex;
      pFun = CubicTData::pFun_ex;
      zFun = CubicTData::zFun_ex;
      aFun = CubicTData::aFun_ex;
      fFun = CubicTData::fFun;
      gFun = CubicTData::gFun;
      hFun = CubicTData::hFun;
      nFun = CubicTData::nFun;
      mFun = CubicTData::mFun;
      wFun = CubicTData::wFun;
      qFun = CubicTData::qFun;
      yFun = CubicTData::yFun;
      cFun = CubicTData::cFun;

      _mu  = CubicTData::_mu;
      _eta = CubicTData::_eta;
      _mu0 = CubicTData::_mu0;
      
      CubicTData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }


    // Non-polynomial
    case 4:{
      meshFile = Analytical4Data::_meshFile;
      pbName   = Analytical4Data::_pbName;

      uFun = Analytical4Data::uFun_ex;
      pFun = Analytical4Data::pFun_ex;
      zFun = Analytical4Data::zFun_ex;
      aFun = Analytical4Data::aFun_ex;
      fFun = Analytical4Data::fFun;
      gFun = Analytical4Data::gFun;
      hFun = Analytical4Data::hFun;
      nFun = Analytical4Data::nFun;
      mFun = Analytical4Data::mFun;
      wFun = Analytical4Data::wFun;
      qFun = Analytical4Data::qFun;
      yFun = Analytical4Data::yFun;
      cFun = Analytical4Data::cFun;

      _mu  = Analytical4Data::_mu;
      _eta = Analytical4Data::_eta;
      _mu0 = Analytical4Data::_mu0;
      
      Analytical4Data::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }



    // Relevant MHD test-cases ----------------------------------------------
    // Kelvin-Helmholtz instability
    case 5:{
      meshFile = KHIData::_meshFile;
      pbName   = KHIData::_pbName;

      uFun = KHIData::uFun_ex;
      pFun = KHIData::pFun_ex;
      zFun = KHIData::zFun_ex;
      aFun = KHIData::aFun_ex;
      fFun = KHIData::fFun;
      gFun = KHIData::gFun;
      hFun = KHIData::hFun;
      nFun = KHIData::nFun;
      mFun = KHIData::mFun;
      wFun = KHIData::wFun;
      qFun = KHIData::qFun;
      yFun = KHIData::yFun;
      cFun = KHIData::cFun;

      _mu  = KHIData::_mu;
      _eta = KHIData::_eta;
      _mu0 = KHIData::_mu0;
      
      KHIData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);

      break;
    }
    // Island coalescence
    case 6:{
      meshFile = IslandCoalescenceData::_meshFile;
      pbName   = IslandCoalescenceData::_pbName;

      uFun = IslandCoalescenceData::uFun_ex;
      pFun = IslandCoalescenceData::pFun_ex;
      zFun = IslandCoalescenceData::zFun_ex;
      aFun = IslandCoalescenceData::aFun_ex;
      fFun = IslandCoalescenceData::fFun;
      gFun = IslandCoalescenceData::gFun;
      hFun = IslandCoalescenceData::hFun;
      nFun = IslandCoalescenceData::nFun;
      mFun = IslandCoalescenceData::mFun;
      wFun = IslandCoalescenceData::wFun;
      qFun = IslandCoalescenceData::qFun;
      yFun = IslandCoalescenceData::yFun;
      cFun = IslandCoalescenceData::cFun;

      _mu  = IslandCoalescenceData::_mu;
      _eta = IslandCoalescenceData::_eta;
      _mu0 = IslandCoalescenceData::_mu0;
      
      IslandCoalescenceData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }
    // Tearing mode
    case 9:{
      meshFile = TearingModeData::_meshFile;
      pbName   = TearingModeData::_pbName;

      uFun = TearingModeData::uFun_ex;
      pFun = TearingModeData::pFun_ex;
      zFun = TearingModeData::zFun_ex;
      aFun = TearingModeData::aFun_ex;
      fFun = TearingModeData::fFun;
      gFun = TearingModeData::gFun;
      hFun = TearingModeData::hFun;
      nFun = TearingModeData::nFun;
      mFun = TearingModeData::mFun;
      wFun = TearingModeData::wFun;
      qFun = TearingModeData::qFun;
      yFun = TearingModeData::yFun;
      cFun = TearingModeData::cFun;

      _mu  = TearingModeData::_mu;
      _eta = TearingModeData::_eta;
      _mu0 = TearingModeData::_mu0;
      
      TearingModeData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }
    // Tearing mode (periodic in x)
    case 8:{
      meshFile = TearingModePerData::_meshFile;
      pbName   = TearingModePerData::_pbName;

      uFun = TearingModePerData::uFun_ex;
      pFun = TearingModePerData::pFun_ex;
      zFun = TearingModePerData::zFun_ex;
      aFun = TearingModePerData::aFun_ex;
      fFun = TearingModePerData::fFun;
      gFun = TearingModePerData::gFun;
      hFun = TearingModePerData::hFun;
      nFun = TearingModePerData::nFun;
      mFun = TearingModePerData::mFun;
      wFun = TearingModePerData::wFun;
      qFun = TearingModePerData::qFun;
      yFun = TearingModePerData::yFun;
      cFun = TearingModePerData::cFun;

      _mu  = TearingModePerData::_mu;
      _eta = TearingModePerData::_eta;
      _mu0 = TearingModePerData::_mu0;
      
      TearingModePerData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }
    // Tearing mode (on symmetric domain)
    case 7:{
      meshFile = TearingModeSymData::_meshFile;
      pbName   = TearingModeSymData::_pbName;

      uFun = TearingModeSymData::uFun_ex;
      pFun = TearingModeSymData::pFun_ex;
      zFun = TearingModeSymData::zFun_ex;
      aFun = TearingModeSymData::aFun_ex;
      fFun = TearingModeSymData::fFun;
      gFun = TearingModeSymData::gFun;
      hFun = TearingModeSymData::hFun;
      nFun = TearingModeSymData::nFun;
      mFun = TearingModeSymData::mFun;
      wFun = TearingModeSymData::wFun;
      qFun = TearingModeSymData::qFun;
      yFun = TearingModeSymData::yFun;
      cFun = TearingModeSymData::cFun;

      _mu  = TearingModeSymData::_mu;
      _eta = TearingModeSymData::_eta;
      _mu0 = TearingModeSymData::_mu0;
      
      TearingModeSymData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }

    // Tearing mode (flipped domain)
    case 10:{
      meshFile = TearingModeFlippedData::_meshFile;
      pbName   = TearingModeFlippedData::_pbName;

      uFun = TearingModeFlippedData::uFun_ex;
      pFun = TearingModeFlippedData::pFun_ex;
      zFun = TearingModeFlippedData::zFun_ex;
      aFun = TearingModeFlippedData::aFun_ex;
      fFun = TearingModeFlippedData::fFun;
      gFun = TearingModeFlippedData::gFun;
      hFun = TearingModeFlippedData::hFun;
      nFun = TearingModeFlippedData::nFun;
      mFun = TearingModeFlippedData::mFun;
      wFun = TearingModeFlippedData::wFun;
      qFun = TearingModeFlippedData::qFun;
      yFun = TearingModeFlippedData::yFun;
      cFun = TearingModeFlippedData::cFun;

      _mu  = TearingModeFlippedData::_mu;
      _eta = TearingModeFlippedData::_eta;
      _mu0 = TearingModeFlippedData::_mu0;
      
      TearingModeFlippedData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }


    // Relevant test-cases for NS -------------------------------------------
    // Driven cavity flow
    case 11:{
      meshFile = CavityDrivenData::_meshFile;
      pbName   = CavityDrivenData::_pbName;

      uFun = CavityDrivenData::uFun_ex;
      pFun = CavityDrivenData::pFun_ex;
      zFun = CavityDrivenData::zFun_ex;
      aFun = CavityDrivenData::aFun_ex;
      fFun = CavityDrivenData::fFun;
      gFun = CavityDrivenData::gFun;
      hFun = CavityDrivenData::hFun;
      nFun = CavityDrivenData::nFun;
      mFun = CavityDrivenData::mFun;
      wFun = CavityDrivenData::wFun;
      qFun = CavityDrivenData::qFun;
      yFun = CavityDrivenData::yFun;
      cFun = CavityDrivenData::cFun;

      _mu  = CavityDrivenData::_mu;
      _eta = CavityDrivenData::_eta;
      _mu0 = CavityDrivenData::_mu0;
      
      CavityDrivenData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }

    // Analytical test case for NS only
    case 30:{
      meshFile = AnalyticalNS0Data::_meshFile;
      pbName   = AnalyticalNS0Data::_pbName;

      uFun = AnalyticalNS0Data::uFun_ex;
      pFun = AnalyticalNS0Data::pFun_ex;
      zFun = AnalyticalNS0Data::zFun_ex;
      aFun = AnalyticalNS0Data::aFun_ex;
      fFun = AnalyticalNS0Data::fFun;
      gFun = AnalyticalNS0Data::gFun;
      hFun = AnalyticalNS0Data::hFun;
      nFun = AnalyticalNS0Data::nFun;
      mFun = AnalyticalNS0Data::mFun;
      wFun = AnalyticalNS0Data::wFun;
      qFun = AnalyticalNS0Data::qFun;
      yFun = AnalyticalNS0Data::yFun;
      cFun = AnalyticalNS0Data::cFun;

      _mu  = AnalyticalNS0Data::_mu;
      _eta = AnalyticalNS0Data::_eta;
      _mu0 = AnalyticalNS0Data::_mu0;

      if ( guFun ){ *guFun = AnalyticalNS0Data::guFun_ex; };
      if ( gvFun ){ *gvFun = AnalyticalNS0Data::gvFun_ex; };
      if ( gaFun ){ *gaFun = AnalyticalNS0Data::gaFun_ex; };

      
      AnalyticalNS0Data::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }

    // Analytical test case for separated Mag - NS only
    case 40:{
      meshFile = AnalyticalMagSepData::_meshFile;
      pbName   = AnalyticalMagSepData::_pbName;

      uFun = AnalyticalMagSepData::uFun_ex;
      pFun = AnalyticalMagSepData::pFun_ex;
      zFun = AnalyticalMagSepData::zFun_ex;
      aFun = AnalyticalMagSepData::aFun_ex;
      fFun = AnalyticalMagSepData::fFun;
      gFun = AnalyticalMagSepData::gFun;
      hFun = AnalyticalMagSepData::hFun;
      nFun = AnalyticalMagSepData::nFun;
      mFun = AnalyticalMagSepData::mFun;
      wFun = AnalyticalMagSepData::wFun;
      qFun = AnalyticalMagSepData::qFun;
      yFun = AnalyticalMagSepData::yFun;
      cFun = AnalyticalMagSepData::cFun;

      _mu  = AnalyticalMagSepData::_mu;
      _eta = AnalyticalMagSepData::_eta;
      _mu0 = AnalyticalMagSepData::_mu0;

      if ( guFun ){ *guFun = AnalyticalMagSepData::guFun_ex; };
      if ( gvFun ){ *gvFun = AnalyticalMagSepData::gvFun_ex; };
      if ( gaFun ){ *gaFun = AnalyticalMagSepData::gaFun_ex; };

      
      AnalyticalMagSepData::setEssTags(essTagsU, essTagsV, essTagsP, essTagsA);
      break;
    }

    default:
      std::cerr<<"ERROR: Problem type "<<pbType<<" not recognised."<<std::endl;
  }

}










//***************************************************************************
// Constant (in space only) analytical test case
//***************************************************************************
namespace ConstantTData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    u(0) =   t*t;
    u(1) = - t*t;
  }
  double pFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  double zFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  double aFun_ex(const Vector & x, const double t ){
    return t;
  }
  // - rhs of velocity counteracts action of every term
  void fFun(const Vector & x, const double t, Vector & f){
    f(0) =  2*t;
    f(1) = -2*t;
  }
  void nFun(const Vector & x, const double t, Vector & n){
    n = 0.;
  }
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential counteracts every term
  double hFun( const Vector & x, const double t ){
    return 1.;
  }
  double mFun( const Vector & x, const double t ){
    return 0.;
  }
  // - IC as IG
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,0.,w);
  }
  double qFun(  const Vector & x, const double t ){
    return ( pFun_ex(x,0.) );
  }
  double yFun(  const Vector & x, const double t ){
    return ( zFun_ex(x,0.) );
  }
  double cFun(  const Vector & x, const double t ){
    return ( aFun_ex(x,0.) );
  }
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Set BC:
    // - Dirichlet on u on N, S, W
    // - Dirichlet on v on N, S
    // - Dirichlet on p on E (outflow, used only in precon)
    // - Dirichlet on A on N
    essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
    essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
    essTagsP.SetSize(1); essTagsP[0] = 2; // E
    essTagsA.SetSize(1); essTagsA[0] = 1;
  }



}






//***************************************************************************
// Quadratic analytical test case
//***************************************************************************
namespace QuadraticData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));
    u(0) =   xx*yy;
    u(1) = - xx*yy;
  }
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    return ( xx );
  }
  double zFun_ex(const Vector & x, const double t ){
    return 2.;
  }
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    return ( xx * xx  );
  }
  // - rhs of velocity counteracts action of every term
  void fFun(const Vector & x, const double t, Vector & f){
    double xx(x(0));
    double yy(x(1));
    //    dt u + u Grad(u)       + Grad(p) + z grad(A) / mu0
    f(0) =       xx*yy*( yy-xx ) + 1.0     + zFun_ex(x,t) * 2*xx / _mu0;
    f(1) =       xx*yy*( xx-yy ) + 0.0     + zFun_ex(x,t) * 0.   / _mu0;
  }
  void nFun(const Vector & x, const double t, Vector & n){
    double xx(x(0));
    double yy(x(1));
    n = 0.;
    if ( yy == 1. ){ //N
      n(0) =  xx;
      n(1) = -xx;
    }
    if ( xx == 1. ){ //E
      n(0) =  yy;
      n(1) = -yy;
    }
    if ( yy == 0. ){ //S
      n(0) = -xx;
      n(1) =  xx;
    }
    if ( xx == 0. ){ //W
      n(0) = -yy;
      n(1) =  yy;
    }
  }
  double gFun(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return xx-yy;
  }
  // - rhs of vector potential counteracts every term
  double hFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    //       dtA      + u Grad(A) - eta/mu0 lap(A)
    return (          2*xx*xx*yy  - _eta/_mu0 * zFun_ex(x,t) );
  }
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double n = 0.;
    if ( yy == 1. ){ //N
      n = 0.;
    }
    if ( xx == 1. ){ //E
      n = 2*xx;
    }
    if ( yy == 0. ){ //S
      n = 0.;
    }
    if ( xx == 0. ){ //W
      n =-2*xx;
    }
    return n;
  }
  // - IC as IG
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,0.,w);
  }
  double qFun(  const Vector & x, const double t ){
    return ( pFun_ex(x,0.) );
  }
  double yFun(  const Vector & x, const double t ){
    return ( zFun_ex(x,0.) );
  }
  double cFun(  const Vector & x, const double t ){
    return ( aFun_ex(x,0.) );
  }
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Set BC:
    // - Dirichlet on u on N, S, W
    // - Dirichlet on v on N, S
    // - Dirichlet on p on E (outflow, used only in precon)
    // - Dirichlet on A on N
    essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
    essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
    essTagsP.SetSize(1); essTagsP[0] = 2; // E
    essTagsA.SetSize(1); essTagsA[0] = 1;
  }
}








//***************************************************************************
// Cubic analytical test case
//***************************************************************************
namespace CubicData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));
    u(0) =   xx*xx*yy;
    u(1) = - xx*yy*yy;
  }
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( xx * yy );
  }
  double zFun_ex(const Vector & x, const double t ){
    double yy(x(1));
    return ( 2 * yy );
  }
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( xx * xx * yy  );
  }
  // - rhs of velocity counteracts action of every term
  void fFun(const Vector & x, const double t, Vector & f){
    double xx(x(0));
    double yy(x(1));
    //      dt u     + u Grad(u)      - mu Lap(u)  + Grad(p) + z grad(A) / mu0
    f(0) =             xx*xx*yy*yy*xx - _mu * 2*yy + yy      + zFun_ex(x,t) * 2.*xx*yy / _mu0;
    f(1) =             xx*xx*yy*yy*yy + _mu * 2*xx + xx      + zFun_ex(x,t) *    xx*xx / _mu0;
  }
  void nFun(const Vector & x, const double t, Vector & n){
    double xx(x(0));
    double yy(x(1));
    n = 0.;
    if ( yy == 1. ){ //N
      n(0) =    xx*xx;
      n(1) = -2*xx*yy;
    }
    if ( xx == 1. ){ //E
      n(0) = 2*xx*yy;
      n(1) = - yy*yy;
    }
    if ( yy == 0. ){ //S
      n(0) = - xx*xx;
      n(1) = 2*xx*yy;
    }
    if ( xx == 0. ){ //W
      n(0) = -2*xx*yy;
      n(1) =    yy*yy;
    }
  }
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential counteracts every term
  double hFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    //       dtA      + u Grad(A)          - eta/mu0 lap(A)
    return (            xx*xx*yy*( xx*yy ) - _eta/_mu0 * zFun_ex(x,t) );
  }
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double n = 0.;
    if ( yy == 1. ){ //N
      n = xx*xx;
    }
    if ( xx == 1. ){ //E
      n = 2*xx*yy;
    }
    if ( yy == 0. ){ //S
      n =-xx*xx;
    }
    if ( xx == 0. ){ //W
      n =-2*xx*yy;
    }
    return n;
  }
  // - IC as IG
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,0.,w);
  }
  double qFun(  const Vector & x, const double t ){
    return ( pFun_ex(x,0.) );
  }
  double yFun(  const Vector & x, const double t ){
    return ( zFun_ex(x,0.) );
  }
  double cFun(  const Vector & x, const double t ){
    return ( aFun_ex(x,0.) );
  }
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Set BC:
    // - Dirichlet on u on N, S, W
    // - Dirichlet on v on N, S
    // - Dirichlet on p on E (outflow, used only in precon)
    // - Dirichlet on A on N
    essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
    essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
    essTagsP.SetSize(1); essTagsP[0] = 2; // E
    essTagsA.SetSize(1); essTagsA[0] = 1;
  }
}



//***************************************************************************
// Cubic in space, linear in time analytical test case
//***************************************************************************
namespace CubicTData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));
    u(0) =   t * xx*xx*yy;
    u(1) = - t * xx*yy*yy;
  }
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( t * xx * yy );
  }
  double zFun_ex(const Vector & x, const double t ){
    double yy(x(1));
    return ( t * 2 * yy );
  }
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return ( t * xx * xx * yy  );
  }
  // - rhs of velocity counteracts action of every term
  void fFun(const Vector & x, const double t, Vector & f){
    double xx(x(0));
    double yy(x(1));
    //      dt u     + u Grad(u)           - mu Lap(u)      + Grad(p) + z grad(A) / mu0
    f(0) =  xx*xx*yy + t*t* xx*xx*yy*yy*xx - t * _mu * 2*yy + t* yy   + t* zFun_ex(x,t) * 2.*xx*yy / _mu0;
    f(1) = -xx*yy*yy + t*t* xx*xx*yy*yy*yy + t * _mu * 2*xx + t* xx   + t* zFun_ex(x,t) *    xx*xx / _mu0;
  }
  void nFun(const Vector & x, const double t, Vector & n){
    double xx(x(0));
    double yy(x(1));
    n = 0.;
    if ( yy == 1. ){ //N
      n(0) =    xx*xx;
      n(1) = -2*xx*yy;
    }
    if ( xx == 1. ){ //E
      n(0) = 2*xx*yy;
      n(1) = - yy*yy;
    }
    if ( yy == 0. ){ //S
      n(0) = - xx*xx;
      n(1) = 2*xx*yy;
    }
    if ( xx == 0. ){ //W
      n(0) = -2*xx*yy;
      n(1) =    yy*yy;
    }
    n *= t;
  }
  double gFun(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of vector potential counteracts every term
  double hFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    //       dtA      + u Grad(A)               - _eta/_mu0 lap(A)
    return ( xx*xx*yy + t*t* xx*xx*yy*( xx*yy ) - _eta/_mu0 * zFun_ex(x,t) );
  }
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double n = 0.;
    if ( yy == 1. ){ //N
      n = xx*xx;
    }
    if ( xx == 1. ){ //E
      n = 2*xx*yy;
    }
    if ( yy == 0. ){ //S
      n =-xx*xx;
    }
    if ( xx == 0. ){ //W
      n =-2*xx*yy;
    }
    return n*t;
  }
  // - IC as IG
  void wFun(const Vector & x, const double t, Vector & w){
    uFun_ex(x,0.,w);
  }
  double qFun(  const Vector & x, const double t ){
    return ( pFun_ex(x,0.) );
  }
  double yFun(  const Vector & x, const double t ){
    return ( zFun_ex(x,0.) );
  }
  double cFun(  const Vector & x, const double t ){
    return ( aFun_ex(x,0.) );
  }
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Set BC:
    // - Dirichlet on u on N, S, W
    // - Dirichlet on v on N, S
    // - Dirichlet on p on E (outflow, used only in precon)
    // - Dirichlet on A on N
    essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
    essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
    essTagsP.SetSize(1); essTagsP[0] = 2; // E
    essTagsA.SetSize(1); essTagsA[0] = 1;
  }
}










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
    // - Dirichlet on u on N and W
    // - Dirichlet on v on N and S
    // - Dirichlet on p on E (outflow, used only in precon)
    // - Dirichlet on A on W
    essTagsU.SetSize(2); essTagsU[0] = 1; essTagsU[1] = 4; // N, W
    essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
    essTagsP.SetSize(1); essTagsP[0] = 2; // E
    // essTagsP.SetSize(0);
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;    
    essTagsA.SetSize(1); essTagsA[0] = 1; // N
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
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
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
    return (1./(2*cosh(_lambda*(yy-0.5))*cosh(_lambda*(yy-0.5))) - _P0 )/ _mu0;  //if you want unperturbed A to be equilibrium
    // return 0.;
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double val = _lambda/(cosh(_lambda*(yy-0.5))*cosh(_lambda*(yy-0.5)));
    if ( t==0 )
      val -= _beta*( sin(M_PI*yy)*cos(2.0*M_PI/_Lx*xx)*M_PI*M_PI*(4./(_Lx*_Lx)+1.)); // perturb IC by _beta

    return val;
    
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    double val = log(cosh(_lambda*(yy-0.5)))/_lambda;
    if ( t==0 )
      val += _beta*sin(M_PI*yy)*cos(2.0*M_PI/_Lx*xx); // perturb IC by _beta

    return val;

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
    essTagsP.SetSize(0);
    essTagsA.SetSize(1); essTagsA[0] = 1;
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
  }

}



// Tearing mode (on reduced domain, exploiting symmetry)
namespace TearingModeSymData{
  void uFun_ex(const Vector & x, const double t, Vector & u){
    u = 0.;
  }
  // - pressure
  double pFun_ex(const Vector & x, const double t ){
    double yy(x(1));
    // re-mapped on symmetric domain:
    return (1./(2*cosh(_lambda*yy)*cosh(_lambda*yy)) - _P0)/ _mu0;  //if you want unperturbed A to be equilibrium
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    // re-mapped on symmetric domain:
    double val = _lambda/(cosh(_lambda*yy)*cosh(_lambda*yy));
    if ( t==0 ){
      val -= _beta*( sin(M_PI*(yy+.5))*cos(2.0*M_PI/_Lx*(xx+3./2.))*M_PI*M_PI*(4./(_Lx*_Lx)+1.) ); // perturb IC by _beta
    }
    return val;
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    // re-mapped on symmetric domain:
    double val = log(cosh(_lambda*yy))/_lambda;
    if ( t==0 ){
      val += _beta*sin(M_PI*(yy+.5))*cos(2.0*M_PI/_Lx*(xx+3./2.)); // perturb IC by _beta
    }
    return val;
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

    // re-mapped on symmetric domain:
    if ( xx == 0. || xx == 1.5 || xx == -1.5 ){
      return 0.;
    }
    if ( yy == 0. || yy == 0.5 || yy == -0.5 ){
      return abs(tanh(_lambda*yy)); // if yy negative, must be multiplied by -1: this takes care of that
    }
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
    // exploit symmetric structure
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
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;
  }


}



// Tearing mode (periodic in x)
namespace TearingModePerData{
  // reuse data from TearingMode
  void   uFun_ex(const Vector& x,const double t,Vector& u){       TearingModeData::uFun_ex(x,t,u);}
  double pFun_ex(const Vector& x,const double t          ){return TearingModeData::pFun_ex(x,t  );}
  double zFun_ex(const Vector& x,const double t          ){return TearingModeData::zFun_ex(x,t  );}
  double aFun_ex(const Vector& x,const double t          ){return TearingModeData::aFun_ex(x,t  );}
  void   fFun(   const Vector& x,const double t,Vector& f){       TearingModeData::fFun(   x,t,f);}
  void   nFun(   const Vector& x,const double t,Vector& f){       TearingModeData::nFun(   x,t,f);}
  double gFun(   const Vector& x,const double t          ){return TearingModeData::gFun(   x,t  );}
  double hFun(   const Vector& x,const double t          ){return TearingModeData::hFun(   x,t  );}
  double mFun(   const Vector& x,const double t          ){return TearingModeData::mFun(   x,t  );}
  void   wFun(   const Vector& x,const double t,Vector& w){       TearingModeData::wFun(   x,t,w);}
  double qFun(   const Vector& x,const double t          ){return TearingModeData::qFun(   x,t  );}
  double cFun(   const Vector& x,const double t          ){return TearingModeData::cFun(   x,t  );}
  double yFun(   const Vector& x,const double t          ){return TearingModeData::yFun(   x,t  );}
  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // periodic in x
    // Top+Bottom=1
    // - Dirichlet on v on top and bottom
    // - No stress on tangential component (u) top and bottom
    // - Dirichlet on p on top and bottom
    // - Dirichlet on A on top and bottom
    essTagsU.SetSize(0);
    essTagsV.SetSize(1); essTagsV[0] = 1;
    // essTagsP.SetSize(1); essTagsP[0] = 1;
    essTagsP.SetSize(0);
    essTagsA.SetSize(1); essTagsA[0] = 1;
    // std::cout<<"Warning: removed dirichlet BC from pressure"<<std::endl;

  }
};





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




//***************************************************************************
// Analytical solutions for NS
//***************************************************************************
namespace AnalyticalNS0Data{
  // - velocity
  void uFun_ex(const Vector & x, const double t, Vector & u){
    double xx(x(0));
    double yy(x(1));
    u(0) = (t*t+1.) * sin(M_PI*xx) * sin(M_PI*yy);
    u(1) = (t*t+1.) * cos(M_PI*xx) * cos(M_PI*yy);
  }
  // - pressure
  double pFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    return (t*t+1.) * sin(M_PI*xx) * cos(M_PI*yy);
  }
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    return 0.;
  }
  // - rhs of velocity
  void fFun(const Vector & x, const double t, Vector & f){
    double xx(x(0));
    double yy(x(1));
    double sx = sin(M_PI*xx), cx = cos(M_PI*xx);
    double sy = sin(M_PI*yy), cy = cos(M_PI*yy);
    //     dt u                   + u Grad(u)          -  mu Lap(u)             + Grad(p)
    f(0) = 2*t*sx*sy + (t*t+1.)*(  (t*t+1.)*M_PI*cx*sx + _mu*2.*M_PI*M_PI*sx*sy + M_PI*cx*cy );
    f(1) = 2*t*cx*cy + (t*t+1.)*( -(t*t+1.)*M_PI*cy*sy + _mu*2.*M_PI*M_PI*cx*cy - M_PI*sx*sy );
  }

  // Normal derivative of velocity
  void nFun(const Vector & x, const double t, Vector & n){
    double xx(x(0));
    double yy(x(1));
    n(0) = 0.0;
    if ( xx == 1. || xx == 0. ){
      n(0) = -(t*t+1.) * M_PI * sin( M_PI*yy );
    }
    if ( yy == 1. || yy == 0. ){
      n(0) = -(t*t+1.) * M_PI * sin( M_PI*xx );
    }
    n(1) = 0.0;
  }

  // Rhs (pressure) - unused
  double gFun(const Vector & x, const double t ){
    return 0.0;
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

  // gradient of u
  // - first component
  void guFun_ex(   const Vector & x, const double t, Vector & u ){
    double xx(x(0));
    double yy(x(1));
    double sx = sin(M_PI*xx), cx = cos(M_PI*xx);
    double sy = sin(M_PI*yy), cy = cos(M_PI*yy);
    u(0) =  (t*t+1.) * M_PI * cx*sy;
    u(1) =  (t*t+1.) * M_PI * cy*sx;
  }
  // - second component
  void gvFun_ex(   const Vector & x, const double t, Vector & u ){
    double xx(x(0));
    double yy(x(1));
    double sx = sin(M_PI*xx), cx = cos(M_PI*xx);
    double sy = sin(M_PI*yy), cy = cos(M_PI*yy);
    u(0) = -(t*t+1.) * M_PI * sx*cy;
    u(1) = -(t*t+1.) * M_PI * cx*sy;
  }
  // gradient of A
  void gaFun_ex(   const Vector & x, const double t, Vector & u ){
    u = 0.;
  }


  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Top 1 Right 2 Bottom 3 Left 4
    // - Dirichlet on u, v, and A everywhere
    essTagsU.SetSize(4); essTagsU[0] = 1; essTagsU[1] = 2; essTagsU[2] = 3; essTagsU[3] = 4;
    essTagsV = essTagsU;
    essTagsP.SetSize(0);
    essTagsA.SetSize(4); essTagsA[0] = 1; essTagsA[1] = 2; essTagsA[2] = 3; essTagsA[3] = 4;
  }

}





//***************************************************************************
// Analytical solution for "separated" NS - magnetic
//***************************************************************************
namespace AnalyticalMagSepData{
  // reuse analytic ns for the fluid part
  void   uFun_ex( const Vector & x, const double t, Vector & u ){        mfem::AnalyticalNS0Data::uFun_ex(x,t,u);  };
  double pFun_ex( const Vector & x, const double t             ){ return mfem::AnalyticalNS0Data::pFun_ex(x,t);    };
  void   nFun(    const Vector & x, const double t, Vector & u ){        mfem::AnalyticalNS0Data::nFun(x,t,u);     };
  double gFun(    const Vector & x, const double t             ){ return mfem::AnalyticalNS0Data::gFun(x,t);       };
  void   wFun(    const Vector & x, const double t, Vector & u ){        mfem::AnalyticalNS0Data::wFun(x,t,u);     };
  double qFun(    const Vector & x, const double t             ){ return mfem::AnalyticalNS0Data::qFun(x,t);       };
  void   guFun_ex(const Vector & x, const double t, Vector & u ){        mfem::AnalyticalNS0Data::guFun_ex(x,t,u); };
  void   gvFun_ex(const Vector & x, const double t, Vector & u ){        mfem::AnalyticalNS0Data::gvFun_ex(x,t,u); };
  // - laplacian of vector potential
  double zFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));    
    return  ( -(t*t*t+1) * M_PI*M_PI * 2 * (cos(M_PI*xx)*cos(M_PI*yy)) );
  }
  // - vector potential
  double aFun_ex(const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));    
    return  ( (t*t*t+1) * cos(M_PI*xx)*cos(M_PI*yy) ); // very unimaginative
  }
  // - rhs of velocity
  void fFun(const Vector & x, const double t, Vector & f){
    Vector f0(2), gA(2);
    AnalyticalNS0Data::fFun( x, t, f0 );  //rhs of fluid part (dtu+uGrad(u)-muLap(u)+Grad(p))
    gaFun_ex( x, t, gA );
    //     f0    + z*Grad(A)/mu0
    f(0) = f0(0) + zFun_ex(x,t) * gA(0)/_mu0;
    f(1) = f0(1) + zFun_ex(x,t) * gA(1)/_mu0 ;
  }
  // - rhs of vector potential
  double hFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    Vector u(2), gA(2);
    AnalyticalNS0Data::uFun_ex(x,t,u);
    gaFun_ex( x, t, gA );    
    //     dtA                             + u*Grad(A) - eta/mu0 Lap(A)
    return 3*t*t*cos(M_PI*xx)*cos(M_PI*yy) + u*gA      - (_eta/_mu0)*zFun_ex(x,t);
  }
  // - normal derivative of vector potential
  double mFun( const Vector & x, const double t ){
    double xx(x(0));
    double yy(x(1));
    Vector gA(2), n(2);
    n=0.;
    gaFun_ex( x, t, gA );    

    if ( xx==0. || xx==1. ){
      if ( xx==1 ){
        n(0)=  1; 
      }else{
        n(0)= -1; 
      }
    }else if ( yy==0. || yy==1. ){
      if ( yy==1 ){
        n(1)=  1; 
      }else{
        n(1)= -1; 
      }
    }
    return gA*n;
  }
  double yFun(  const Vector & x, const double t ){
    return zFun_ex(x,0);
  }
  double cFun(  const Vector & x, const double t ){
    return aFun_ex(x,0);
  }

  // gradient of A
  void gaFun_ex(   const Vector & x, const double t, Vector & gA ){
    double xx(x(0));
    double yy(x(1));
    gA(0) = - (t*t*t+1)*M_PI * sin(M_PI*xx)*cos(M_PI*yy);
    gA(1) = - (t*t*t+1)*M_PI * cos(M_PI*xx)*sin(M_PI*yy);
  }


  void setEssTags( Array<int>& essTagsU, Array<int>& essTagsV, Array<int>& essTagsP, Array<int>& essTagsA ){
    // Top 1 Right 2 Bottom 3 Left 4
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


