//---------------------------------------------------------------------------
// * Taylor-Hood finite element implementation for time-dependent Stokes
//   equations in MFEM
// * Solver support from PETSc
// * For information on the block preconditioner, see the book:
//   H. Elman, D. Silvester, and A. Wathen. Finite elements and fast
//   iterative solvers: with applications in incompressible fluid dynamics.
//
// Implemented by S. Rhebergen (University of Waterloo). The code is based
// on MFEM's example ex5p.cpp.
//
// Compile and run:
// make && mpirun -np 4 ./stokesp --petscopts rc_stokesp_fieldsplit -r 4
//
//---------------------------------------------------------------------------
#include "mfem.hpp"
// #include "petsc.h"
#include <fstream>
#include <iostream>
//---------------------------------------------------------------------------
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;
//---------------------------------------------------------------------------
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
   StopWatch chrono;

   // for now, assume no spatial parallelisation: each processor handles a time-step
   int num_procs, myid;
   double dt = 0.001;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool verbose = (myid == 0);

   const char *mesh_file = "../3rdParty/MFEM/data/inline-tri.mesh";   
   int order = 2;
   int ref_levels = 0;
   bool visualization = 1;
   bool use_petsc = true;
   const char *petscrc_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the linear system.");
   args.AddOption(&ref_levels, "-r", "--rlevel",
                  "Refinement level.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.Parse();
   if (!args.Good())
   {
      if (verbose)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (verbose)
   {
      args.PrintOptions(cout);
   }

   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   for (int ii = 0; ii < ref_levels; ii++)
     mesh->UniformRefinement();

   if (verbose)
     std::cout << "Number of elements: " << mesh->GetNE() << std::endl;

   // For now, just have each proc build his own mesh   
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   // delete mesh;

   FiniteElementCollection *Vh_coll(new H1_FECollection(order,   dim));
   FiniteElementCollection *Qh_coll(new H1_FECollection(order-1, dim));

   FiniteElementSpace *Vh_space = new FiniteElementSpace(mesh, Vh_coll, dim);
   FiniteElementSpace *Qh_space = new FiniteElementSpace(mesh, Qh_coll);

   HYPRE_Int dimVh = Vh_space->GlobalTrueVSize();
   HYPRE_Int dimQh = Qh_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(Vh) = " << dimVh << "\n";
      std::cout << "dim(Qh) = " << dimQh << "\n";
      std::cout << "dim(Vh+Qh) = " << dimVh + dimQh << "\n";
      std::cout << "***********************************************************\n";
   }

   // each block is an instant in time
   Array<int> block_offsets( 2*num_procs + 1 );
   block_offsets[0] = 0
   for ( int i = 1; i < num_procs + 1; ++i ){
      block_offsets[i]             = i * ( Vh_space->GetVSize() ); 
      block_offsets[num_procs + i] = i * ( Qh_space->GetVSize() );   
   }
   block_offsets.PartialSum();

   // In theory no need for this, since mesh is not parallel
   // block_trueOffsets[0] = 0
   // for ( int i = 1; i < num_procs + 1; ++i ){
   //    block_trueOffsets[i]             = i * ( Vh_space->TrueVSize() ); 
   //    block_trueOffsets[num_procs + i] = i * ( Qh_space->TrueVSize() );   
   // }
   // block_offsets.PartialSum();


   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient gcoeff(gFun);
   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dof;
   Vh_space->GetEssentialVDofs(ess_bdr, ess_dof);

   BlockVector x(block_offsets), rhs(block_offsets);
   // BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);


   // each proc is associated with the corresponding myid-th block of the rhs
   LinearForm *fform(new LinearForm);
   fform->Update(Vh_space, rhs.GetBlock(myid), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   fform->Assemble();

   LinearForm *gform(new LinearForm);
   gform->Update(Qh_space, rhs.GetBlock(num_procs+myid), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();

   // Assemble the finite element matrices 
   // for the Stokes operator
   //
   //   D = [ A  B^T ]
   //       [ B   0  ]
   //
   GridFunction ubc;
   ubc.MakeRef(Vh_space, x.GetBlock(myid), 0);
   ubc.ProjectCoefficient(ucoeff);

   BilinearForm *avVarf(new BilinearForm(Vh_space));
   avVarf->AddDomainIntegrator(new VectorDiffusionIntegrator);
   avVarf->Assemble();
   avVarf->EliminateEssentialBC(ess_bdr, ubc, rhs.GetBlock(myid));
   avVarf->Finalize();
   // HypreParMatrix *A;
   // A = aVarf->ParallelAssemble();
   SparseMatrix *AV( avVarf->SpMat() );

   BilinearForm *mvVarf(new BilinearForm(Vh_space));
   ConstantCoefficient invdt( 1.0/dt );
   mvVarf->AddDomainIntegrator(new VectorMassIntegrator(invdt) );
   mvVarf->Assemble();
   mvVarf->EliminateEssentialBC(ess_bdr, ubc, rhs.GetBlock(myid));
   mvVarf->Finalize();
   SparseMatrix *MV( mvVarf->SpMat() );
   SparseMatrix *mMV = *MV *(-1.0);

   SparseMatrix *FV = *MV + *AV;


   ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(Vh_space, Qh_space));
   ConstantCoefficient mone(-1.0);
   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(mone));
   bVarf->Assemble();
   bVarf->EliminateTrialDofs(ess_bdr, ubc, rhs.GetBlock(1));
   bVarf->Finalize();
   // HypreParMatrix *B;
   // B = bVarf->ParallelAssemble();
   // HypreParMatrix *BT = B->Transpose();
   SparseMatrix *B( bVarf->SpMat() );
   SparseMatrix *BT = B->Transpose();


   // fform->ParallelAssemble(trueRhs.GetBlock(0));
   // gform->ParallelAssemble(trueRhs.GetBlock(1));

   BlockOperator *stokesOp = new BlockOperator(block_offsets);
   stokesOp->SetBlock( myid,myid, AV );
   if( myid>0 )
      stokesOp->SetBlock(myid-1,myid, mMV );
   stokesOp->SetBlock( myid, myid+num_procs, BT );
   stokesOp->SetBlock( myid+num_procs, myid, B );

   // Construct the operators for preconditioner
   //
   //   P = [  A         0 ]
   //       [  0         Q ]
   //
   BilinearForm *mpVarf(new BilinearForm(Qh_space));
   mpVarf->AddDomainIntegrator(new MassIntegrator);
   mpVarf->Assemble();
   mpVarf->Finalize();
   SparseMatrix *MP( mpVarf->SpMat() );
   // HypreParMatrix *Q;
   // Q = qVarf->ParallelAssemble();
   // (*Q) *= -1.0; // comment when using MINRES
   
   chrono.Clear();
   chrono.Start();

   BlockOperator *stokesPr = new BlockOperator(block_offsets);
   stokesPr->SetBlock(0, 0, A);
   stokesPr->SetBlock(0, 1, BT);
   stokesPr->SetBlock(1, 0, B);
   stokesPr->SetBlock(1, 1, Q);

   PetscLinearSolver *solver;
   PetscPreconditioner *pstokesPr = NULL;
   solver = new PetscLinearSolver(MPI_COMM_WORLD,"solver_");
   
   pstokesPr = new PetscFieldSplitSolver(MPI_COMM_WORLD,*stokesPr,"solver_");
   solver->SetOperator(*stokesOp);
   solver->SetPreconditioner(*pstokesPr);
   solver->SetTol(1e-12);
   solver->SetAbsTol(0.0);
   solver->SetMaxIter(300);
   solver->Mult(trueRhs, trueX);
   
   chrono.Stop();
   if (verbose)
     {
       if (solver->GetConverged())
	 std::cout << "Solver converged in " << solver->GetNumIterations()
		   << " iterations with a residual norm of " << solver->GetFinalNorm() << ".\n";
       else
	 std::cout << "Solver did not converge in " << solver->GetNumIterations()
		   << " iterations. Residual norm is " << solver->GetFinalNorm() << ".\n";
       std::cout << "Solver solver took " << chrono.RealTime() << "s. \n";
     }
   delete solver;
   delete stokesPr;
   delete pstokesPr;

   ParGridFunction *u(new ParGridFunction);
   ParGridFunction *p(new ParGridFunction);
   u->MakeRef(Vh_space, x.GetBlock(0), 0);
   p->MakeRef(Qh_space, x.GetBlock(1), 0);
   u->Distribute(&(trueX.GetBlock(0)));
   p->Distribute(&(trueX.GetBlock(1)));

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u->ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeGlobalLpNorm(2, ucoeff, *pmesh, irs);
   double err_p  = p->ComputeL2Error(pcoeff, irs);
   double norm_p = ComputeGlobalLpNorm(2, pcoeff, *pmesh, irs);

   if (verbose)
   {
      std::cout << "|| u_h - u_ex ||= " << err_u << "\n";
      std::cout << "|| p_h - p_ex || = " << err_p << "\n";
   }

   VisItDataCollection visit_dc("Stokes-Parallel", pmesh);
   visit_dc.RegisterField("velocity", u);
   visit_dc.RegisterField("pressure", p);
   visit_dc.Save();

   delete fform;
   delete gform;
   delete u;
   delete p;
   delete stokesOp;
   delete BT;
   delete B;
   delete A;
   delete Q;
   delete aVarf;
   delete bVarf;
   delete qVarf;
   delete Vh_space;
   delete Qh_space;
   delete Vh_coll;
   delete Qh_coll;
   delete pmesh;

   MFEMFinalizePetsc();
   MPI_Finalize();

   return 0;
}
//---------------------------------------------------------------------------
void uFun_ex(const Vector & x, Vector & u)
{
   double xx(x(0));
   double yy(x(1));
   u(0) = sin(M_PI*xx) * sin(M_PI*yy);
   u(1) = cos(M_PI*xx) * cos(M_PI*yy);
}
//---------------------------------------------------------------------------
double pFun_ex(const Vector & x)
{
   double xx(x(0));
   double yy(x(1));
   return sin(M_PI*xx) * cos(M_PI*yy);
}
//---------------------------------------------------------------------------
void fFun(const Vector & x, Vector & f)
{
   double xx(x(0));
   double yy(x(1));
   f(0) = 2.0*M_PI*M_PI*sin(M_PI*xx)*sin(M_PI*yy) + M_PI*cos(M_PI*xx)*cos(M_PI*yy);
   f(1) = 2.0*M_PI*M_PI*cos(M_PI*xx)*cos(M_PI*yy) - M_PI*sin(M_PI*xx)*sin(M_PI*yy);
}
//---------------------------------------------------------------------------
double gFun(const Vector & x)
{
   return 0;
}
//---------------------------------------------------------------------------
