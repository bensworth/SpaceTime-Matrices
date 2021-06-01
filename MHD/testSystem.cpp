// Test file to check correctness of implementation of the system assembly
//  As the mesh is refined, substituting the analytical solution should send the residual to 0
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "imhd2dstoperatorassembler.hpp"
#include "parblocklowtrioperator.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

void uFun_ex_constant(const Vector & x, const double t, Vector & u){
  u(0) =   t*t;
  u(1) = - t*t;
}
double pFun_ex_constant(const Vector & x, const double t ){
  return 0.;
}
double zFun_ex_constant(const Vector & x, const double t ){
  return 0.;
}
double aFun_ex_constant(const Vector & x, const double t ){
  return t;
}
// - rhs of velocity counteracts action of every term
void fFun_constant(const Vector & x, const double t, Vector & f){
  f(0) =  2*t;
  f(1) = -2*t;
}
void nFun_constant(const Vector & x, const double t, Vector & n){
  n = 0.;
}
double gFun_constant(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts every term
double hFun_constant( const Vector & x, const double t ){
  return 1.;
}
double mFun_constant( const Vector & x, const double t ){
  return 0.;
}






namespace LinearData{
  const double eta = 1.;
  const double mu0 = 1.;
  const double mu  = 1.;
}

void uFun_ex_linear(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   yy;
  u(1) = - xx;
}
double pFun_ex_linear(const Vector & x, const double t ){
  return 1.;
}
double zFun_ex_linear(const Vector & x, const double t ){
  return 0.;
}
double aFun_ex_linear(const Vector & x, const double t ){
  double xx(x(0));
  return ( xx  );
}
// - rhs of velocity counteracts action of every term
void fFun_linear(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //    dt u + u Grad(u) + Grad(p) + z grad(A) / mu0
  f(0) =     - xx        + 0.0     + zFun_ex_linear(x,t) * 1. /LinearData::mu0;
  f(1) =     - yy        + 0.0     + zFun_ex_linear(x,t) * 0. /LinearData::mu0;
}
void nFun_linear(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n = 0.;
  if ( yy == 1. ){ //N
    n(0) = 1.;
    n(1) = 0.;
  }
  if ( xx == 1. ){ //E
    n(0) =  0.;
    n(1) = -1.;
  }
  if ( yy == 0. ){ //S
    n(0) = -1.;
    n(1) =  0.;
  }
  if ( xx == 0. ){ //W
    n(0) =  0.;
    n(1) =  1.;
  }
}
double gFun_linear(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts every term
double hFun_linear( const Vector & x, const double t ){
  double yy(x(1));
  //       dtA + u Grad(A) - eta lap(A)
  return (     yy          -  LinearData::eta * zFun_ex_linear(x,t) );
}
double mFun_linear( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  double n = 0.;
  if ( yy == 1. ){ //N
    n = 0.;
  }
  if ( xx == 1. ){ //E
    n = 1.;
  }
  if ( yy == 0. ){ //S
    n = 0.;
  }
  if ( xx == 0. ){ //W
    n =-1.;
  }
  return n;
}










namespace QuadraticData{
  const double eta = 1.;
  const double mu0 = 1.;
  const double mu  = 1.;
}

void uFun_ex_quadratic(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   xx*yy;
  u(1) = - xx*yy;
}
double pFun_ex_quadratic(const Vector & x, const double t ){
  double xx(x(0));
  return ( xx );
}
double zFun_ex_quadratic(const Vector & x, const double t ){
  return 2.;
}
double aFun_ex_quadratic(const Vector & x, const double t ){
  double xx(x(0));
  return ( xx * xx  );
}
// - rhs of velocity counteracts action of every term
void fFun_quadratic(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //    dt u + u Grad(u)       + Grad(p) + z grad(A) / mu0
  f(0) =       xx*yy*( yy-xx ) + 1.0     + zFun_ex_quadratic(x,t) * 2*xx / QuadraticData::mu0;
  f(1) =       xx*yy*( xx-yy ) + 0.0     + zFun_ex_quadratic(x,t) * 0.   / QuadraticData::mu0;
}
void nFun_quadratic(const Vector & x, const double t, Vector & n){
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
double gFun_quadratic(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return xx-yy;
}
// - rhs of vector potential counteracts every term
double hFun_quadratic( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  //       dtA      + u Grad(A) - eta lap(A)
  return (          2*xx*xx*yy  - QuadraticData::eta * zFun_ex_quadratic(x,t) );
}
double mFun_quadratic( const Vector & x, const double t ){
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




namespace CubicData{
  const double eta = 1.;
  const double mu0 = 1.;
  const double mu  = 1.;
}

void uFun_ex_cubic(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   xx*xx*yy;
  u(1) = - xx*yy*yy;
}
double pFun_ex_cubic(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( xx * yy );
}
double zFun_ex_cubic(const Vector & x, const double t ){
  double yy(x(1));
  return ( 2 * yy );
}
double aFun_ex_cubic(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( xx * xx * yy  );
}
// - rhs of velocity counteracts action of every term
void fFun_cubic(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)      - mu Lap(u)            + Grad(p) + z grad(A) / mu0
  f(0) =             xx*xx*yy*yy*xx - CubicData::mu * 2*yy + yy      + zFun_ex_cubic(x,t) * 2.*xx*yy / CubicData::mu0;
  f(1) =             xx*xx*yy*yy*yy + CubicData::mu * 2*xx + xx      + zFun_ex_cubic(x,t) *    xx*xx / CubicData::mu0;
}
void nFun_cubic(const Vector & x, const double t, Vector & n){
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
double gFun_cubic(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts every term
double hFun_cubic( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  //       dtA      + u Grad(A)          - eta lap(A)
  return (            xx*xx*yy*( xx*yy ) - CubicData::eta * zFun_ex_cubic(x,t) );
}
double mFun_cubic( const Vector & x, const double t ){
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





namespace CubicTData{
  const double eta = 1.;
  const double mu0 = 1.;
  const double mu  = 1.;
}

void uFun_ex_cubicT(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   t * xx*xx*yy;
  u(1) = - t * xx*yy*yy;
}
double pFun_ex_cubicT(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * xx * yy );
}
double zFun_ex_cubicT(const Vector & x, const double t ){
  double yy(x(1));
  return ( t * 2 * yy );
}
double aFun_ex_cubicT(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * xx * xx * yy  );
}
// - rhs of velocity counteracts action of every term
void fFun_cubicT(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)           - mu Lap(u)                 + Grad(p) + z grad(A) / mu0
  f(0) =  xx*xx*yy + t*t* xx*xx*yy*yy*xx - t * CubicTData::mu * 2*yy + t* yy   + t* zFun_ex_cubicT(x,t) * 2.*xx*yy / CubicTData::mu0;
  f(1) = -xx*yy*yy + t*t* xx*xx*yy*yy*yy + t * CubicTData::mu * 2*xx + t* xx   + t* zFun_ex_cubicT(x,t) *    xx*xx / CubicTData::mu0;
}
void nFun_cubicT(const Vector & x, const double t, Vector & n){
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
double gFun_cubicT(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts every term
double hFun_cubicT( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  //       dtA      + u Grad(A)               - eta lap(A)
  return ( xx*xx*yy + t*t* xx*xx*yy*( xx*yy ) - CubicTData::eta * zFun_ex_cubicT(x,t) );
}
double mFun_cubicT( const Vector & x, const double t ){
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






// Define analytical solution
// Velocity, pressure, vector potential and laplacian of vector potential ---
namespace An4Data{
  const double eta = 1.;
  const double mu0 = 1.;
  const double mu  = 1.;
}
// - Pick a div-free field
void uFun_ex_an4(const Vector & x, const double t, Vector & u){
  double xx(x(0));
  double yy(x(1));
  u(0) =   t * xx*xx*yy;
  u(1) = - t * xx*yy*yy;
}
// - Pick a pressure which is null on boundaries
double pFun_ex_an4(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * sin(M_PI*xx)*sin(M_PI*yy) );
}
// - laplacian of vector potential
double zFun_ex_an4(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( -t * 2*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) );
}
// - vector potential with null normal derivative
double aFun_ex_an4(const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  return ( t * cos(M_PI*xx)*cos(M_PI*yy) );
}
// - rhs of velocity counteracts action of every term
void fFun_an4(const Vector & x, const double t, Vector & f){
  double xx(x(0));
  double yy(x(1));
  //      dt u     + u Grad(u)            - mu Lap(u) + Grad(p)                            + z grad(A) / mu0
  f(0) =  xx*xx*yy + t*t * xx*xx*xx*yy*yy - t * 2*yy  + t * M_PI*cos(M_PI*xx)*sin(M_PI*yy) + zFun_ex_an4(x,t) * (-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy));
  f(1) = -xx*yy*yy + t*t * xx*xx*yy*yy*yy + t * 2*xx  + t * M_PI*sin(M_PI*xx)*cos(M_PI*yy) + zFun_ex_an4(x,t) * (-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy));
}
// - normal derivative of velocity
void nFun_an4(const Vector & x, const double t, Vector & n){
  double xx(x(0));
  double yy(x(1));
  n = 0.;
  if ( xx == 1. ){ //E
    n(0) = 2*yy;
    n(1) = - yy*yy;
  }
  n *= t;
}
// - null rhs of pressure
double gFun_an4(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential counteracts every term
double hFun_an4( const Vector & x, const double t ){
  double xx(x(0));
  double yy(x(1));
  Vector u(2);
  uFun_ex_an4(x,t,u);
  double ugradA = u(0)*(-t*M_PI*sin(M_PI*xx)*cos(M_PI*yy))
                + u(1)*(-t*M_PI*cos(M_PI*xx)*sin(M_PI*yy));
  //       dtA                       + u Grad(A) - eta lap(A)
  return ( cos(M_PI*xx)*cos(M_PI*yy) + ugradA    - zFun_ex_an4(x,t) );
}
// - null normal derivative of vector potential
double mFun_an4( const Vector & x, const double t ){
  return 0.;
}





// Hartmann flow
namespace HartmannData{
  const double G0  = 1.0;
  const double B0  = 1.0;
};
// - velocity
void uFun_ex_hartmann(const Vector & x, const double t, Vector & u){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  u = 0.;

  u(0) = G0/B0 * ( cosh(1.) - cosh(x(1)) )/sinh(1.);
}
// - pressure
double pFun_ex_hartmann(const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  return -G0*( x(0) + 1.0 );    // zero at outflow (x=-1)
}
// - laplacian of vector potential - unused
double zFun_ex_hartmann(const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  return -G0/B0 * ( 1. - cosh(x(1))/sinh(1.) );
}
// - vector potential
double aFun_ex_hartmann(const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  return -B0*x(0) - G0/B0 * ( x(1)*x(1)/2. - cosh(x(1))/sinh(1.) );
}
// - rhs of velocity - unused
void fFun_hartmann(const Vector & x, const double t, Vector & f){
  f = 0.;
}
// - normal derivative of velocity
void nFun_hartmann(const Vector & x, const double t, Vector & n){
  const double yy(x(1));
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  n = 0.;

  if ( yy== 1. ){
    n(1) = -G0/B0 * sinh(yy)/sinh(1.);
  }
  if ( yy==-1. ){
    n(1) =  G0/B0 * sinh(yy)/sinh(1.);
  }
}
// - rhs of pressure - unused
double gFun_hartmann(const Vector & x, const double t ){
  return 0.;
}
// - rhs of vector potential
double hFun_hartmann( const Vector & x, const double t ){
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  return - G0/B0 * ( cosh(1.)/sinh(1.) - 1. );
}
// - normal derivative of vector potential
double mFun_hartmann( const Vector & x, const double t ){
  const double xx(x(0));
  const double yy(x(1));
  const double G0  = HartmannData::G0;
  const double B0  = HartmannData::B0;

  if ( yy==1. || yy == -1. ){
    return yy*( -G0/B0 * ( yy - sinh(yy)/sinh(1.) ) );
  }
  if ( xx==1. || xx == -1. ){
    return -B0*xx;
  }
  return 0.;
}






//---------------------------------------------------------------------------

int main(int argc, char *argv[]){

  // for now, assume no spatial parallelisation: each processor handles a time-step
  int numProcs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  double Tend = 1;

  // Initialise problem parameters
  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;
  int refLvl = 4;
  int maxRefLvl = 8;
  bool stab = false;
  int verbose = 0;
  // these tag each boundary to identify dirichlet BC for u, p and A
  Array<int> essTagsU(0);
  Array<int> essTagsV(0);
  Array<int> essTagsP(0);
  Array<int> essTagsA(0);


  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&maxRefLvl, "-maxr", "--maxrlevel",
                "Max refinement level (default: 8)");
  args.AddOption(&ordU, "-oU", "--ordU",
                "Velocity space polynomial order (default: 2)");
  args.AddOption(&ordP, "-oP", "--ordP",
                "Pressure space polynomial order (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--ordZ",
                "Laplacian of vector potential space polynomial order (default: 1)");
  args.AddOption(&ordA, "-oA", "--ordA",
                "Vector potential space polynomial order (default: 2)");
  args.AddOption(&stab, "-S", "--stab", "-noS", "--noStab",
                "Stabilise via SUPG (default: false)");
  args.AddOption(&verbose, "-V", "--verbose",
                "Verbosity");
  args.Parse();

  void(  *uFun)( const Vector & x, const double t, Vector & u );
  double(*pFun)( const Vector & x, const double t             );
  double(*zFun)( const Vector & x, const double t             );
  double(*aFun)( const Vector & x, const double t             );
  void(  *fFun)( const Vector & x, const double t, Vector & f );
  double(*gFun)( const Vector & x, const double t             );
  double(*hFun)( const Vector & x, const double t             );
  void(  *nFun)( const Vector & x, const double t, Vector & f );
  double(*mFun)( const Vector & x, const double t             );

  // std::string mesh_file = "./meshes/tri-square-hartmann.mesh";
  // uFun = uFun_ex_hartmann;
  // pFun = pFun_ex_hartmann;
  // zFun = zFun_ex_hartmann;
  // aFun = aFun_ex_hartmann;
  // fFun = fFun_hartmann;
  // gFun = gFun_hartmann;
  // hFun = hFun_hartmann;
  // nFun = nFun_hartmann;
  // mFun = mFun_hartmann;
  // essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, W
  // essTagsV = essTagsU;
  // essTagsP.SetSize(1); essTagsP[0] = 2; // E
  // essTagsA.SetSize(1); essTagsA[0] = 4; //essTagsA[1] = 3; // W
  // double dt  = Tend / numProcs;
  // double mu  = 1.;
  // double eta = 1.;
  // double mu0 = 1.;



  std::string mesh_file = "./meshes/tri-square-testAn-ref.mesh";
  uFun = uFun_ex_an4;
  pFun = pFun_ex_an4;
  zFun = zFun_ex_an4;
  aFun = aFun_ex_an4;
  fFun = fFun_an4;
  gFun = gFun_an4;
  hFun = hFun_an4;
  nFun = nFun_an4;
  mFun = mFun_an4;
  essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
  essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
  essTagsP.SetSize(1); essTagsP[0] = 2; // E
  essTagsA.SetSize(1); essTagsA[0] = 1;
  double dt  = Tend / numProcs;
  double mu  = An4Data::mu;
  double eta = An4Data::eta;
  double mu0 = An4Data::mu0;




  if(myid == 0){
    args.PrintOptions(cout);
    std::cout<<"   --np "<<numProcs<<std::endl;
    std::cout<<"   --dt "<<Tend/numProcs<<std::endl;
  }


  // - initialise petsc
  MFEMInitializePetsc(NULL,NULL,NULL,NULL);


  if(myid == 0){
    std::cout << "Space-time residual norm:\nTot\t\tu\t\tp\t\tz\t\tA"<<std::endl;
  }


  for ( ; refLvl < maxRefLvl; ++refLvl ){
    // ASSEMBLE OPERATORS -----------------------------------------------------
    IMHD2DSTOperatorAssembler *mhdAssembler = new IMHD2DSTOperatorAssembler( MPI_COMM_WORLD, mesh_file, refLvl, ordU, ordP, ordZ, ordA,
                                                                             dt, mu, eta, mu0, fFun, gFun, hFun, nFun, mFun,
                                                                             uFun, pFun, zFun, aFun,
                                                                             uFun, pFun, zFun, aFun, 
                                                                             essTagsU, essTagsV, essTagsP, essTagsA, stab, verbose );


    ParBlockLowTriOperator *FFFu, *MMMz, *FFFa, *BBB, *BBBt, *CCCs, *ZZZ1, *ZZZ2, *XXX1, *XXX2, *KKK, *YYY;
    Vector   fres,   gres,  zres,  hres,  U0,  P0,  Z0,  A0;


    // Assemble the system
    mhdAssembler->AssembleSystem( FFFu, MMMz, FFFa,
                                  BBB,  BBBt, CCCs,  
                                  ZZZ1, ZZZ2, XXX1, XXX2, 
                                  KKK,  YYY,
                                  fres, gres, zres, hres,
                                  U0,   P0,   Z0,   A0  );


    Array<int> offsets(5);
    offsets[0] = 0;
    offsets[1] = FFFu->NumRows();
    offsets[2] =  BBB->NumRows();
    offsets[3] = MMMz->NumRows();
    offsets[4] = FFFa->NumRows();
    offsets.PartialSum();

    // recover residual and its norm
    BlockVector res(offsets);
    res.GetBlock(0) = fres;
    res.GetBlock(1) = gres;
    res.GetBlock(2) = zres;
    res.GetBlock(3) = hres;

    // check how the residual reduces: it should be bang 0 if the prescribed functions are 
    //  polys of order < max integration order of the integrators used, otherwise it should
    //  decrease at a rate proportional to the integration order
    double resnorm = res.Norml2();
    resnorm*= resnorm;
    MPI_Allreduce( MPI_IN_PLACE, &resnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    resnorm  = sqrt(resnorm);

    double fresnorm = fres.Norml2();
    fresnorm*= fresnorm;
    MPI_Allreduce( MPI_IN_PLACE, &fresnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    fresnorm  = sqrt(fresnorm);
    double gresnorm = gres.Norml2();
    gresnorm*= gresnorm;
    MPI_Allreduce( MPI_IN_PLACE, &gresnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    gresnorm  = sqrt(gresnorm);
    double zresnorm = zres.Norml2();
    zresnorm*= zresnorm;
    MPI_Allreduce( MPI_IN_PLACE, &zresnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    zresnorm  = sqrt(zresnorm);
    double hresnorm = hres.Norml2();
    hresnorm*= hresnorm;
    MPI_Allreduce( MPI_IN_PLACE, &hresnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    hresnorm  = sqrt(hresnorm);

    if(myid == 0){
      std::cout << resnorm <<"\t"<< fresnorm <<"\t"<< gresnorm <<"\t"<< zresnorm <<"\t"<< hresnorm <<std::endl;
    }

    if (  refLvl == maxRefLvl - 1 ){

      // To check implementation of gradient, compare gradient application with evaluation of nonlinear operator

      // - Identify non-dirichlet components
      Array<int> essU, essP, essA;
      mhdAssembler->GetDirichletIdx( essU, essP, essA );
      Array<bool> isEssU( res.GetBlock(0).Size() ); isEssU = false;
      Array<bool> isEssP( res.GetBlock(1).Size() ); isEssP = false;
      Array<bool> isEssA( res.GetBlock(3).Size() ); isEssA = false;
      for(int i=0; i<essU.Size(); ++i ){ isEssU[essU[i]] = true; };
      for(int i=0; i<essP.Size(); ++i ){ isEssP[essP[i]] = true; };
      for(int i=0; i<essA.Size(); ++i ){ isEssA[essA[i]] = true; };


      // - open a bunch of files
      std::string myfilename  = "./results/gradientConvergence.dat";
      std::string myfilenameU = "./results/gradientConvergenceU.dat";
      std::string myfilenameP = "./results/gradientConvergenceP.dat";
      std::string myfilenameZ = "./results/gradientConvergenceZ.dat";
      std::string myfilenameA = "./results/gradientConvergenceA.dat";
      std::ofstream myfile;
      std::ofstream myfileU;
      std::ofstream myfileP;
      std::ofstream myfileZ;
      std::ofstream myfileA;
      if ( myid == 0 ){
        myfile.open(  myfilename  );
        myfileU.open( myfilenameU );
        myfileP.open( myfilenameP );
        myfileZ.open( myfilenameZ );
        myfileA.open( myfilenameA );
        myfile.precision(std::numeric_limits< double >::max_digits10);
        myfileU.precision(std::numeric_limits< double >::max_digits10);
        myfileP.precision(std::numeric_limits< double >::max_digits10);
        myfileZ.precision(std::numeric_limits< double >::max_digits10);
        myfileA.precision(std::numeric_limits< double >::max_digits10);
      }
      // std::string myfilenameMat;
      // std::ofstream myfileMat;
      // myfileMat.precision(std::numeric_limits< double >::max_digits10);


      // - Assemble gradient
      BlockOperator MHDOp( offsets );
      MHDOp.SetBlock(0, 0, FFFu);
      MHDOp.SetBlock(0, 1, BBBt);
      MHDOp.SetBlock(0, 2, ZZZ1);
      MHDOp.SetBlock(0, 3, ZZZ2);
      MHDOp.SetBlock(1, 0,  BBB);
      MHDOp.SetBlock(1, 1, CCCs);
      MHDOp.SetBlock(1, 2, XXX1);
      MHDOp.SetBlock(1, 3, XXX2);
      MHDOp.SetBlock(2, 2, MMMz);
      MHDOp.SetBlock(2, 3,  KKK);
      MHDOp.SetBlock(3, 0,  YYY);
      MHDOp.SetBlock(3, 3, FFFa);

      // myfilenameMat = "./results/ugaFu_pre.dat";
      // myfileMat.open( myfilenameMat );
      // FFFu->GetBlockDiag(0)->PrintMatlab(myfileMat);
      // myfileMat.close( );

      // Consider X0 as the point where we evaluate the gradient ****************

      // // - Assemble x0 and recover Nx = b - N(x0) = res
      // BlockVector x(offsets);
      // x.GetBlock(0) = U0;
      // x.GetBlock(1) = P0;
      // x.GetBlock(2) = Z0;
      // x.GetBlock(3) = A0;
      // BlockVector Nx(res);

      // Or randomly generate a new point ***************************************
      // (this is an additional check on UpdateLinearisedOperators doing its job)
      srand( myid );

      BlockVector x(offsets), Nx(offsets);
      for ( int ii = 0; ii < x.Size(); ++ii ){
        x(ii) = double( ( ( rand() % 100 ) - 50. ) ) / 5.0; // values in [-10,10]
      }
      // - remember however to reset its Dirichlet nodes
      for(int i=0; i<essU.Size(); ++i ){ x.GetBlock(0)(essU[i]) = U0(essU[i]); };
      for(int i=0; i<essP.Size(); ++i ){ x.GetBlock(1)(essP[i]) = P0(essP[i]); };
      for(int i=0; i<essA.Size(); ++i ){ x.GetBlock(3)(essA[i]) = A0(essA[i]); };

      mhdAssembler->UpdateLinearisedOperators( x ); // this should (hopefully) update MHDOp, too
      mhdAssembler->ApplyOperator( x, Nx );         // compute b-N(x)


      // myfilenameMat = "./results/ugaFu_post.dat";
      // myfileMat.open( myfilenameMat );
      // FFFu->GetBlockDiag(0)->PrintMatlab(myfileMat);
      // myfileMat.close( );



      // - Here the fun begins
      // For each block
      for ( int bb = 0; bb < 4; ++bb ){
        // For each processor
        for ( int pp = 0; pp < numProcs; ++pp ){
          // For each node
          for ( int ii = 0; ii < x.GetBlock(bb).Size(); ++ii ){
            BlockVector dx(offsets); dx = 0.;
            // if it's dirichlet, skip
            if ( ( bb==0 && isEssU[ii] ) || ( bb==3 && isEssA[ii] ) ){
              continue;
            }
            // if it's a non-dirichlet (and it belongs to this processor) perturb
            if ( myid == pp ){
              dx.GetBlock(bb).operator()(ii) = 1.;
            }

            // - compute linearised perturbation (J*dx) (ie, extract column in Jacobian)
            BlockVector Jdx(offsets);
            MHDOp.Mult(dx, Jdx);
            // std::cout<<"("<<pp<<","<<bb<<","<<ii<<"): ";Jdx.Print(mfem::out,Jdx.Size());std::cout<<std::endl;

            // - compute N(x+eps*dx) for various perturbation sizes eps
            BlockVector xpedx(x), error(offsets);
            for ( int jj = 0; jj > -10; --jj ){
              // -- perturb x
              double eps = pow(10,jj);
              
              if ( myid == pp ){
                xpedx.GetBlock(bb).operator()(ii) += eps;
                // std::cout<<"("<<pp<<","<<bb<<","<<ii<<"): ";xpedx.Print(mfem::out,xpedx.Size());std::cout<<std::endl;
              }
              // -- compute error = (N(x+eps*dx) - Nx) - Jdx*eps
              mhdAssembler->ApplyOperator( xpedx, error );  // this actually gives b-N(x+eps*dx)
              // std::cout<<"Nxpedx          "<<myid<<"("<<pp<<","<<bb<<","<<ii<<"): ";error.Print(mfem::out,error.Size());std::cout<<std::endl;
              error -= Nx;                                  // this gives b-N(x+eps*dx) - (b-N(x)) = N(x) - N(x+eps*dx)
              // std::cout<<"Nxpedx-Nx:      "<<myid<<"("<<pp<<","<<bb<<","<<ii<<"): ";error.Print(mfem::out,error.Size());std::cout<<std::endl;
              error.Add( eps, Jdx );                        // this gives  eps*J*dx - ( N(x+eps*dx) - N(x) )
              // std::cout<<"Nxpedx-Nx-eJdx: "<<myid<<"("<<pp<<","<<bb<<","<<ii<<"): ";error.Print(mfem::out,error.Size());std::cout<<std::endl;

              // -- compute and store its norm
              double errnorm  = error.Norml2();             errnorm  *= errnorm;
              double errnormU = error.GetBlock(0).Norml2(); errnormU *= errnormU;
              double errnormP = error.GetBlock(1).Norml2(); errnormP *= errnormP;
              double errnormZ = error.GetBlock(2).Norml2(); errnormZ *= errnormZ;
              double errnormA = error.GetBlock(3).Norml2(); errnormA *= errnormA;
              MPI_Allreduce( MPI_IN_PLACE, &errnorm,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
              MPI_Allreduce( MPI_IN_PLACE, &errnormU, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
              MPI_Allreduce( MPI_IN_PLACE, &errnormP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
              MPI_Allreduce( MPI_IN_PLACE, &errnormZ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
              MPI_Allreduce( MPI_IN_PLACE, &errnormA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
              errnorm  = sqrt(errnorm);
              errnormU = sqrt(errnormU);
              errnormP = sqrt(errnormP);
              errnormZ = sqrt(errnormZ);
              errnormA = sqrt(errnormA);
              if ( myid == 0 ){
                // check the semilogy plot the output file: it should decrease with order 2
                myfile  << errnorm  << "\t";
                myfileU << errnormU << "\t";
                myfileP << errnormP << "\t";
                myfileZ << errnormZ << "\t";
                myfileA << errnormA << "\t";
              }


              // -- reset perturbation
              if ( myid == pp ){
                xpedx.GetBlock(bb).operator()(ii) = x.GetBlock(bb).operator()(ii);
              }

            }
            // - new component
            if ( myid == 0 ){
              myfile  << std::endl;
              myfileU << std::endl;
              myfileP << std::endl;
              myfileZ << std::endl;
              myfileA << std::endl;
            }

          }

        }
      }
      if ( myid == 0 ){
        myfile.close();
        myfileU.close();
        myfileP.close();
        myfileZ.close();
        myfileA.close();
      }
    }
  }

  MFEMFinalizePetsc();
  MPI_Finalize();


  return 0;
}







