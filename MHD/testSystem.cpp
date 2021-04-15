// Test file to check correctness of implementation of the system assembly
//  As the mesh is refined, substituting the analytical solution should send the residual to 0
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "imhd2dstoperatorassembler.hpp"
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
  const double eta = 1e-3;
  const double mu0 = 1e-3;
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
  const double eta = 1e-3;
  const double mu0 = 1e-3;
  const double mu  = 1;
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
  const double eta = 1e-1;
  const double mu0 = 1e-2;
  const double mu  = 1e-3;
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
  const double eta = 1e-1;
  const double mu0 = 1e-2;
  const double mu  = 1e-3;
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
  args.AddOption(&ordU, "-oU", "--ordU",
                "Velocity space polynomial order (default: 2)");
  args.AddOption(&ordP, "-oP", "--ordP",
                "Pressure space polynomial order (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--ordZ",
                "Laplacian of vector potential space polynomial order (default: 1)");
  args.AddOption(&ordA, "-oA", "--ordA",
                "Vector potential space polynomial order (default: 2)");
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


  std::string mesh_file = "./meshes/tri-square-testAn.mesh";
  uFun = uFun_ex_cubicT;
  pFun = pFun_ex_cubicT;
  zFun = zFun_ex_cubicT;
  aFun = aFun_ex_cubicT;
  fFun = fFun_cubicT;
  gFun = gFun_cubicT;
  hFun = hFun_cubicT;
  nFun = nFun_cubicT;
  mFun = mFun_cubicT;
  essTagsU.SetSize(3); essTagsU[0] = 1; essTagsU[1] = 3; essTagsU[2] = 4; // N, S, w
  essTagsV.SetSize(2); essTagsV[0] = 1; essTagsV[1] = 3; // N, S
  essTagsP.SetSize(1); essTagsP[0] = 2; // E
  essTagsA.SetSize(1); essTagsA[0] = 1;
  double dt  = Tend / numProcs;
  double mu  = CubicTData::mu;
  double eta = CubicTData::eta;
  double mu0 = CubicTData::mu0;




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


  for ( ; refLvl < 8; ++refLvl ){
    // ASSEMBLE OPERATORS -----------------------------------------------------
    IMHD2DSTOperatorAssembler *mhdAssembler = new IMHD2DSTOperatorAssembler( MPI_COMM_WORLD, mesh_file, refLvl, ordU, ordP, ordZ, ordA,
                                                                             dt, mu, eta, mu0, fFun, gFun, hFun, nFun, mFun,
                                                                             uFun, pFun, zFun, aFun,
                                                                             uFun, pFun, zFun, aFun, 
                                                                             essTagsU, essTagsV, essTagsP, essTagsA, verbose );


    Operator *FFFu, *MMMz, *FFFa, *BBB, *ZZZ1, *ZZZ2, *KKK, *YYY;
    Vector   fres,   gres,  zres,  hres,  U0,  P0,  Z0,  A0;

    // Assemble the system
    mhdAssembler->AssembleSystem( FFFu, MMMz, FFFa,
                                  BBB,  ZZZ1, ZZZ2,
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

    BlockVector res(offsets);
    res.GetBlock(0) = fres;
    res.GetBlock(1) = gres;
    res.GetBlock(2) = zres;
    res.GetBlock(3) = hres;


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

  }

  MFEMFinalizePetsc();
  MPI_Finalize();


  return 0;
}







