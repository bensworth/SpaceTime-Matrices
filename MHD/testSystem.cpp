// Test file to check correctness of implementation of the system assembly
//  As the mesh is refined, substituting the analytical solution should send the residual to 0
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "petsc.h"
#include "testcases.hpp"
#include "imhd2dstoperatorassembler.hpp"
#include "parblocklowtrioperator.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


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
  int pbType = 4;

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
  args.AddOption(&pbType, "-Pb", "--problem",
              "Problem: 0 to 4-Analytical test cases (4 default)\n"
              "              5-Kelvin-Helmholtz instability\n"
              "              6-Island coalescensce\n"
              "              9-Tearing mode\n"
              "             11-Driven cavity flow\n");
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
  void  (*wFun)( const Vector & x, const double t, Vector & w );
  double(*qFun)( const Vector & x, const double t             );
  double(*cFun)( const Vector & x, const double t             );
  double(*yFun)( const Vector & x, const double t             );

  double dt  = Tend / numProcs;

  std::string mesh_file;
  std::string pbName;

  // extra variables useful in defining the Pb
  Array<int> essTagsU(0);
  Array<int> essTagsV(0);
  Array<int> essTagsP(0);
  Array<int> essTagsA(0);

  double mu;  
  double eta; 
  double mu0; 

  MHDTestCaseSelector( pbType, 
                       uFun, pFun, zFun, aFun,
                       fFun, gFun, hFun, nFun, mFun,
                       wFun, qFun, yFun, cFun,
                       mu, eta, mu0,
                       pbName, mesh_file,
                       essTagsU, essTagsV, essTagsP, essTagsA );




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
                                                                             uFun, pFun, zFun, aFun, // give exact solution as IG
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







