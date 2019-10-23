#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "mfem.hpp"
#include "HYPRE.h"

using namespace mfem;


void BDF1(int rank, int ntPerProc, double dt, bool isTimeDependent,
          int *&rowptr, int *&colinds, double *&data, double *&B, double *&X,
          int &onProcSize, int ref_levels, int order);
void BDF2(int rank, int ntPerProc, double dt, bool isTimeDependent,
          int *&rowptr, int *&colinds, double *&data, double *&B, double *&X,
          int &onProcSize, int ref_levels, int order);
void BDF3(int rank, int ntPerProc, double dt, bool isTimeDependent,
          int *&rowptr, int *&colinds, double *&data, double *&B, double *&X,
          int &onProcSize, int ref_levels, int order);
void AM2(int rank, int ntPerProc, double dt, bool isTimeDependent,
         int *&rowptr, int *&colinds, double *&data, double *&B,
         double *&X, int &onProcSize, int ref_levels, int order);
void AB1(int rank, int ntPerProc, double dt, bool isTimeDependent,
         int *&rowptr, int *&colinds, double *&data, double *&B,
         double *&X, int &onProcSize, int ref_levels, int order);
void AB2(int rank, int ntPerProc, double dt, bool isTimeDependent,
         int *&rowptr, int *&colinds, double *&data, double *&B,
         double *&X, int &onProcSize, int ref_levels, int order);

//  TODO:
// --------
//  + Add implementation so that can update RHS w/o rebuilding entire matrix
//  + Are these error still accurate??    
//      - Error in running hypre on AB1 and AB2 in serial


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    std::cout << "Initialized MPI\n";

    // Parameters
    bool isTimeDependent = true;
    int numTimeSteps = 4;
    int ref_levels = 1;
    int order = 1;
    int bdf = -1;
    int am = -1;
    int ab = -1;
    bool save_mat = false;
    int distance = 2;
    std::string prerelax("");
    std::string postrelax("FFC");
    double strength_tolC = 0.1;
    double strength_tolR = 0.1;
    int interp_type = 100;
    int relax_type = 1;
    double filterA_tol = 0.0;
    int coarsening = 22;
    int print_level = 3;
    for(int i=1; i<argc; i++) {
        if(strcmp(argv[i],"-ref") == 0) {
            i += 1;
            ref_levels = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-order") == 0) {
            i += 1;
            order = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-nt") == 0) {
            i += 1;
            numTimeSteps = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-BDF") == 0) {
            i += 1;
            bdf = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-AM") == 0) {
            i += 1;
            am = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-AB") == 0) {
            i += 1;
            ab = atoi(argv[i]);
        }
        else if (strcmp(argv[i],"-save") == 0) {
            save_mat = true;
        }
    }
    double dt = 1.0/numTimeSteps;

    // Check for non-conflicting time integration routines; set BDF1 as default. 
    if ((am > 0) && (bdf > 0)) {
        std::cout << "WARNING: Cannot specify BDF and Adams-Moulton time integrators.\n";
        return 1;
    }
    else if ((am > 0) && (ab > 0)) {
        std::cout << "WARNING: Cannot specify Adams-Bashforth and Adams-Moulton time integrators.\n";
        return 1;
    }
    else if ((ab > 0) && (bdf > 0)) {
        std::cout << "WARNING: Cannot specify Adams-Bashforth and BDF time integrators.\n";
        return 1;
    }
    else if ((am < 0) && (bdf < 0) && (ab < 0)) {
        bdf = 1;
    }

    // Check that number of processes divides the total number of time steps 
    if (numTimeSteps % numProcess  != 0) {
        if (rank == 0) {
            std::cout << "Error: number of processes does not divide number of time steps.\n";
        }
        MPI_Finalize();
        return 1;
    }
    int ntPerProc = numTimeSteps / numProcess;

    // Array to store relaxation scheme and pass to Hypre
    int ns_down = prerelax.length();
    int ns_up = postrelax.length();
    int ns_coarse = 1;
    std::string Fr("F");
    std::string Cr("C");
    std::string Ar("A");
    int **grid_relax_points = new int *[4];
    grid_relax_points[0] = NULL;
    grid_relax_points[1] = new int[ns_down];
    grid_relax_points[2] = new int [ns_up];
    grid_relax_points[3] = new int[1];
    grid_relax_points[3][0] = 0;

    // set down relax scheme 
    for(unsigned int i = 0; i<ns_down; i++) {
        if (prerelax.compare(i,1,Fr) == 0) {
            grid_relax_points[1][i] = -1;
        }
        else if (prerelax.compare(i,1,Cr) == 0) {
            grid_relax_points[1][i] = 1;
        }
        else if (prerelax.compare(i,1,Ar) == 0) {
            grid_relax_points[1][i] = 0;
        }
    }

    // set up relax scheme 
    for(unsigned int i = 0; i<ns_up; i++) {
        if (postrelax.compare(i,1,Fr) == 0) {
            grid_relax_points[2][i] = -1;
        }
        else if (postrelax.compare(i,1,Cr) == 0) {
            grid_relax_points[2][i] = 1;
        }
        else if (postrelax.compare(i,1,Ar) == 0) {
            grid_relax_points[2][i] = 0;
        }
    }

    // Get space-time matrix for BDF1
    int *rowptr;
    int *colinds;
    double *data;
    double *B;
    double *X;
    int onProcSize;
    if (bdf == 1) {
        BDF1(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else if (bdf == 2) {
        BDF2(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else if (bdf == 3) {
        BDF3(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else if (am == 1) {
        BDF1(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else if (am == 2) {
        AM2(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else if (ab == 1) {
        AB1(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else if (ab == 2) {
        AB2(rank, ntPerProc, dt, isTimeDependent, rowptr, colinds, data,
             B, X, onProcSize, ref_levels, order);
    }
    else {
        std::cout << "WARNING: invalid integration parameters.\n";
        MPI_Finalize();
        return 1;
    }

    // Initialize matrix
    int ilower = rank*onProcSize;
    int iupper = (rank+1)*onProcSize - 1;
    HYPRE_IJMatrix      spaceTimeMat0;
    HYPRE_ParCSRMatrix  spaceTimeMat;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &spaceTimeMat0);
    HYPRE_IJMatrixSetObjectType(spaceTimeMat0, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(spaceTimeMat0);

    // Set matrix coefficients
    int *rows = new int[onProcSize];
    int *cols_per_row = new int[onProcSize];
    for (int i=0; i<onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row[i] = rowptr[i+1] - rowptr[i];
    }
    HYPRE_IJMatrixSetValues(spaceTimeMat0, onProcSize, cols_per_row, rows, colinds, data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(spaceTimeMat0);
    HYPRE_IJMatrixGetObject(spaceTimeMat0, (void **) &spaceTimeMat);
    
    // Save matrix to file to debug
    if (save_mat) {
        const char* filename1 = "./test.mm";
        HYPRE_IJMatrixPrint (spaceTimeMat0, filename1);
    }

    /* Create sample rhs and solution vectors */
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorSetValues(b, onProcSize, rows, B);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    HYPRE_IJVector x;
    HYPRE_ParVector par_x;  
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);
    HYPRE_IJVectorSetValues(x, onProcSize, rows, X);
    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **) &par_x);

    // /* Create a preconditioner and solve the problem */
    int maxiter = 100;
    double tol = 1e-6;
    HYPRE_Solver precond;
    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetTol(precond, tol);    
    HYPRE_BoomerAMGSetMaxIter(precond, maxiter);
    HYPRE_BoomerAMGSetPrintLevel(precond, print_level);
    HYPRE_BoomerAMGSetRestriction(precond, distance);
    HYPRE_BoomerAMGSetInterpType(precond, interp_type);
    // HYPRE_BoomerAMGSetCoarsenType(precond, coarsening);
    // HYPRE_BoomerAMGSetAggNumLevels(precond, 0);
    // HYPRE_BoomerAMGSetStrongThreshold(precond, strength_tolC);
    // HYPRE_BoomerAMGSetStrongThresholdR(precond, strength_tolR);
    // HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
    // if (relax_type > -1) {
    //     HYPRE_BoomerAMGSetRelaxType(precond, relax_type);
    // }
    // HYPRE_BoomerAMGSetCycleNumSweeps(precond, ns_coarse, 3);
    // HYPRE_BoomerAMGSetCycleNumSweeps(precond, ns_down,   1);
    // HYPRE_BoomerAMGSetCycleNumSweeps(precond, ns_up,     2);
    // HYPRE_BoomerAMGSetADropTol(precond, filterA_tol);
    // HYPRE_BoomerAMGSetADropType(precond, -1);          /* type = -1: drop based on row inf-norm */

    /* Setup and solve system. */
    HYPRE_BoomerAMGSetup(precond, spaceTimeMat, par_b, par_x);
    HYPRE_BoomerAMGSolve(precond, spaceTimeMat, par_b, par_x);

    // HYPRE_Solver gmres;
    // HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &gmres);
    // HYPRE_GMRESSetMaxIter(gmres, maxiter);
    // HYPRE_GMRESSetTol(gmres, tol);
    // HYPRE_GMRESSetPrintLevel(gmres, 1);
    // HYPRE_GMRESSetPrecond(gmres, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
    //                       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
    // HYPRE_GMRESSetup(gmres, (HYPRE_Matrix)spaceTimeMat, (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
    // HYPRE_GMRESSolve(gmres, (HYPRE_Matrix)spaceTimeMat, (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);


    // Finalize MPI, clean up
    HYPRE_BoomerAMGDestroy(precond);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);
    HYPRE_IJMatrixDestroy(spaceTimeMat0);
    delete[] rows;
    delete[] cols_per_row;
    MPI_Finalize();
    return 0;
}


/* Time-independent spatial discretization of Laplacian */
void getSpatialDiscretization(int *&A_rowptr, int *&A_colinds, double *&A_data,
                              double *&B, double *&X, int &spatialDOFs, double t,
                              int ref_levels, int order)
{
    // Read mesh from mesh file
    const char *mesh_file = "../meshes/beam-quad.mesh";
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine mesh
    // double max_elements = 50000;
    // int ref_levels = (int)floor(log(max_elements./mesh->GetNE())/log(2.)/dim);
    for (int l=0; l<ref_levels; l++) {
        mesh->UniformRefinement();
    }

    // Define finite element space on mesh
    FiniteElementCollection *fec;
    fec = new H1_FECollection(order, dim);
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear form b(.) 
    LinearForm *b = new LinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Define solution vector x as finite element grid function corresponding to fespace.
    // Initialize x with initial guess of zero, which satisfies the boundary conditions.
    GridFunction x(fespace);
    x = 0.0;

    // Set up bilinear form a(.,.) on finite element space corresponding to Laplacian
    // operator -Delta, by adding the Diffusion domain integrator.
    BilinearForm *a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // Assemble bilinear form and corresponding linear system
    Vector B0;
    Vector X0;
    SparseMatrix A;
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X0, B0, 0);

    // Change ownership of matrix data to sparse matrix from bilinear form
    spatialDOFs = A.NumRows();
    a->LoseMat();
    A_rowptr = A.GetI();
    A_colinds = A.GetJ();
    A_data = A.GetData();
    A.LoseData();

    // TODO : think I want to steal data from B0, X0, but they do not own
    B = b->StealData();
    X = x.StealData();

    delete a;
    delete b; 
    if (fec) {
      delete fespace;
      delete fec;
    }
    delete mesh;

    // TODO: remove
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


/* First-order Adams-Bashforth (also known as Backward Euler, BDF1, and
   first-order Adams-Moulton) */
void BDF1(int rank, int ntPerProc, double dt, bool isTimeDependent,
          int *&rowptr, int *&colinds, double *&data, double *&B,
          double *&X, int &onProcSize, int ref_levels, int order)
{
    int tInd0 = rank*ntPerProc;
    int tInd1 = tInd0 + ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             B0, X0, spatialDOFs, dt*tInd0,
                             ref_levels, order);
    int nnzPerTime = T_rowptr[spatialDOFs];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = ntPerProc * spatialDOFs;
    int procNnz    = ntPerProc * (spatialDOFs + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= spatialDOFs;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Steps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                                     B0, X0, spatialDOFs, dt*ti,
                                     ref_levels, order);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + dt*T_data[j];
                    }
                    else {
                        data[dataInd] = dt*T_data[j];
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + T_rowptr[i+1] - T_rowptr[i];
                thisRow += 1;
            }
        }
        else {

            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd + i;
                data[dataInd] = -1.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + dt*T_data[j];
                    }
                    else {
                        data[dataInd] = dt*T_data[j];
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] B0;
    delete[] X0;
}


void BDF2(int rank, int ntPerProc, double dt, bool isTimeDependent,
          int *&rowptr, int *&colinds, double *&data, double *&B,
          double *&X, int &onProcSize, int ref_levels, int order)
{
    int tInd0 = rank*ntPerProc;
    int tInd1 = tInd0 + ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             B0, X0, spatialDOFs, dt*tInd0,
                             ref_levels, order);
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = ntPerProc * spatialDOFs;
    int procNnz    = ntPerProc * (2*spatialDOFs + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= 2*spatialDOFs;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= spatialDOFs;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Steps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                                     B0, X0, spatialDOFs, dt*ti,
                                     ref_levels, order);
        }

        int colPlusDiag   = ti*spatialDOFs;
        int colPlusOffd_1 = (ti - 1)*spatialDOFs;
        int colPlusOffd_2 = (ti - 2)*spatialDOFs;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 2.0*dt*T_data[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*dt*T_data[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*dt*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + T_rowptr[i+1] - T_rowptr[i];
                thisRow += 1;
            }
        }
        // At time t=1, only have 1 off-diagonal block
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time t1
            for (int i=0; i<spatialDOFs; i++) {
                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -4.0/3.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 3dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 2.0*dt*T_data[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*dt*T_data[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*dt*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                // Add off-diagonal block, -u_{i-2}
                colinds[dataInd] = colPlusOffd_2 + i;
                data[dataInd] = 1.0/3.0;
                dataInd += 1;

                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -4.0/3.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 2.0*dt*T_data[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*dt*T_data[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*dt*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 2 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }
    
        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] B0;
    delete[] X0;
}


void BDF3(int rank, int ntPerProc, double dt, bool isTimeDependent,
          int *&rowptr, int *&colinds, double *&data, double *&B,
          double *&X, int &onProcSize, int ref_levels, int order)
{
    int tInd0 = rank*ntPerProc;
    int tInd1 = tInd0 + ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             B0, X0, spatialDOFs, dt*tInd0,
                             ref_levels, order);
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = ntPerProc * spatialDOFs;
    int procNnz    = ntPerProc * (3*spatialDOFs + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= 3*spatialDOFs;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= 2*spatialDOFs;
    if ((tInd0 <= 2) && (tInd1 >= 2)) procNnz -= spatialDOFs;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Steps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                                     B0, X0, spatialDOFs, dt*ti,
                                     ref_levels, order);
        }

        int colPlusDiag   = ti*spatialDOFs;
        int colPlusOffd_1 = (ti - 1)*spatialDOFs;
        int colPlusOffd_2 = (ti - 2)*spatialDOFs;
        int colPlusOffd_3 = (ti - 3)*spatialDOFs;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + T_rowptr[i+1] - T_rowptr[i];
                thisRow += 1;
            }
        }
        // At time t=1, only have 1 off-diagonal block
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time t1
            for (int i=0; i<spatialDOFs; i++) {
                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -18.0/11.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 3dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }
        else if (ti == 2) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                // Add off-diagonal block, -u_{i-2}
                colinds[dataInd] = colPlusOffd_2 + i;
                data[dataInd] = 9.0/11.0;
                dataInd += 1;

                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -18.0/11.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 2 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                // Add off-diagonal block, -u_{i-3}
                colinds[dataInd] = colPlusOffd_3 + i;
                data[dataInd] = -2.0/11.0;
                dataInd += 1;

                // Add off-diagonal block, -u_{i-2}
                colinds[dataInd] = colPlusOffd_2 + i;
                data[dataInd] = 9.0/11.0;
                dataInd += 1;

                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -18.0/11.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 3 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] B0;
    delete[] X0;
}


/* Second-order Adams-Moulton implicit scheme (trapezoid method) */
void AM2(int rank, int ntPerProc, double dt, bool isTimeDependent,
         int *&rowptr, int *&colinds, double *&data, double *&B,
         double *&X, int &onProcSize, int ref_levels, int order)
{
    int tInd0 = rank*ntPerProc;
    int tInd1 = tInd0 + ntPerProc -1;

    // Get spatial discretization for previous time step, or first step if tInd0=0
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *Bi;
    double *Xi;
    int *T_rowptr_1 = NULL;
    int *T_colinds_1 = NULL;
    double *T_data_1 = NULL;
    double *Bi_1 = NULL;
    double *Xi_1 = NULL;
    int spatialDOFs;
    if (tInd0 > 0) {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             Bi, Xi, spatialDOFs, dt*(tInd0-1),
                             ref_levels, order);
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             Bi, Xi, spatialDOFs, dt*tInd0,
                             ref_levels, order);
    }
    if (!isTimeDependent) {
        T_rowptr_1 = T_rowptr;
        T_colinds_1 = T_colinds;
        T_data_1 = T_data;
        Bi_1 = Bi;
        Xi_1 = Xi;   
    }
    int nnzPerTime = T_rowptr[spatialDOFs];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = ntPerProc * spatialDOFs;
    int procNnz     = 2 * ntPerProc * nnzPerTime;     // nnzs on this processor
    if (tInd0 == 0) procNnz -= nnzPerTime;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;

    rowptr[0] = 0;
    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Steps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        int colPlusOffd = (ti - 1)*spatialDOFs;
        int colPlusDiag = ti*spatialDOFs;

        // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
        // to build one spatial matrix each iteration. Matrix at ti for previous iteration
        //  is used as ti-1 on this iteration.
        if ((ti != 0) && isTimeDependent) {
            delete[] T_rowptr_1;
            delete[] T_colinds_1;
            delete[] T_data_1;
            delete[] Bi_1;
            delete[] Xi_1;
            T_rowptr_1 = T_rowptr;
            T_colinds_1 = T_colinds;
            T_data_1 = T_data;
            Bi_1 = Bi;
            Xi_1 = Xi;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                                     Bi, Xi, spatialDOFs, dt*ti,
                                     ref_levels, order);
        }

        // At time t=0, only have spatial discretization block.
        if (ti == 0) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + dt*T_data[j] / 2.0;
                    }
                    else {
                        data[dataInd] = dt*T_data[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*Bi[i] / 2.0;
                X[thisRow] = Xi[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + T_rowptr[i+1] - T_rowptr[i];
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add row for spatial discretization of off-diagonal block 
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (-I + dt*L/2), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds_1[j]) {
                        data[dataInd] = -1 + dt*T_data_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = dt*T_data_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd + T_colinds_1[j];
                    dataInd += 1;
                }

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L/2), otherwise data is dt*L/2
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + dt*T_data[j] / 2.0;
                    }
                    else {
                        data[dataInd] = dt*T_data[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*(Bi[i] + Bi_1[i]) / 2.0;
                X[thisRow] = Xi[i];

                // Total nonzero for this row on processor is the total nnz in this row
                // the spatial discretization at time ti and ti-1.
                rowptr[thisRow+1] = rowptr[thisRow] + (T_rowptr[i+1] - T_rowptr[i]) +
                                    (T_rowptr_1[i+1] - T_rowptr_1[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] Bi;
    delete[] Xi;
    if (isTimeDependent) {
        delete[] T_rowptr_1;
        delete[] T_colinds_1;
        delete[] T_data_1;
        delete[] Bi_1;
        delete[] Xi_1;   
    }
}


/* First-order Adams-Bashforth (Forward Euler) */
void AB1(int rank, int ntPerProc, double dt, bool isTimeDependent,
         int *&rowptr, int *&colinds, double *&data, double *&B,
         double *&X, int &onProcSize, int ref_levels, int order)
{
    int tInd0 = rank*ntPerProc;
    int tInd1 = tInd0 + ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    if (tInd0 > 0) {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             B0, X0, spatialDOFs, dt*(tInd0-1),
                             ref_levels, order);
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                             B0, X0, spatialDOFs, dt*tInd0,
                             ref_levels, order);
    }
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = ntPerProc * spatialDOFs;
    int procNnz = ntPerProc * (spatialDOFs + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= nnzPerTime;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Steps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0, ti > 1)
        if ((ti > tInd0) && isTimeDependent && (ti > 1)) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data,
                                     B0, X0, spatialDOFs, dt*ti,
                                     ref_levels, order);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;

        // At time t=0, only have identity block on diagonal
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Assume user implements boundary conditions to rhs
                B[thisRow] = 0.0;
                X[thisRow] = X0[i];

                // One nonzero for this row
                rowptr[thisRow+1] = rowptr[thisRow] + 1;
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add spatial discretization at time ti-1 to off-diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Subtract identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = -1 + dt*T_data[j];
                    }
                    else {
                        data[dataInd] = dt*T_data[j];
                    }
                    colinds[dataInd] = colPlusOffd + T_colinds[j];
                    dataInd += 1;
                }

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] B0;
    delete[] X0;
}


/* Second-order Adams-Bashforth explicit scheme */
void AB2(int rank, int ntPerProc, double dt, bool isTimeDependent,
         int *&rowptr, int *&colinds, double *&data, double *&B,
         double *&X, int &onProcSize, int ref_levels, int order)
{
    int tInd0 = rank*ntPerProc;
    int tInd1 = tInd0 + ntPerProc -1;

    // Pointers to CSR arrays for A_{ti} and A_{ti-1}
    int *T_rowptr_1;
    int *T_colinds_1;
    double *T_data_1;
    double *Bi_1;
    double *Xi_1;
    int *T_rowptr_2 = NULL;
    int *T_colinds_2 = NULL;
    double *T_data_2 = NULL;
    double *Bi_2 = NULL;
    double *Xi_2 = NULL;
    int spatialDOFs;
    if (tInd0 <= 1) {
        getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1,
                                 Bi_1, Xi_1, spatialDOFs, 0,
                                 ref_levels, order);
    }
    else {
        getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1,
                                 Bi_1, Xi_1, spatialDOFs, dt*(tInd0-2),
                                 ref_levels, order);
    }
    if (!isTimeDependent) {
        T_rowptr_2 = T_rowptr_1;
        T_colinds_2 = T_colinds_1;
        T_data_2 = T_data_1;
        Bi_2 = Bi_1;
        Xi_2 = Xi_1;   
    }

    int nnzPerTime = T_rowptr_1[spatialDOFs];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = ntPerProc * spatialDOFs;
    int procNnz = ntPerProc * (2*nnzPerTime + spatialDOFs);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= 2*nnzPerTime;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= nnzPerTime;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Steps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        int colPlusOffd_1 = (ti - 1)*spatialDOFs;
        int colPlusOffd_2 = (ti - 2)*spatialDOFs;
        int colPlusDiag = ti*spatialDOFs;

         // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
        // to build one spatial matrix each iteration. Matrix at ti-1 for previous iteration
        //  is used as ti-2 on this iteration.
        if ((ti > 1) && isTimeDependent) {
            delete[] T_rowptr_2;
            delete[] T_colinds_2;
            delete[] T_data_2;
            delete[] Bi_2;
            delete[] Xi_2;
            T_rowptr_2 = T_rowptr_1;
            T_colinds_2 = T_colinds_1;
            T_data_2 = T_data_1;
            Bi_2 = Bi_1;
            Xi_2 = Xi_1;
            getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1,
                                     Bi_1, Xi_1, spatialDOFs, dt*(ti-1),
                                     ref_levels, order);
        }

        // At time t=0, only have identity block on diagonal
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Assume user implements boundary conditions to rhs
                B[thisRow] = 0.0;
                X[thisRow] = Xi_1[i];

                // One nonzero for this row
                rowptr[thisRow+1] = rowptr[thisRow] + 1;
                thisRow += 1;
            }
        }
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to first off-diagonal block
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds_1[j]) {
                        data[dataInd] = -1 + 3.0*dt*T_data_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = 3.0*dt*T_data_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd_1 + T_colinds_1[j];
                    dataInd += 1;
                }

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 3.0*dt*Bi_1[i] / 2.0;
                X[thisRow] = Xi_1[i];

                // Total nonzero for this row on processor is one for diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + T_rowptr_1[i+1] - T_rowptr_1[i];
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add row for spatial discretization of second off-diagonal block 
                for (int j=T_rowptr_2[i]; j<T_rowptr_2[i+1]; j++) {

                    // Add spatial block -dt*Lu_{ti-2}
                    data[dataInd] = -dt*T_data_2[j] / 2.0;
                    colinds[dataInd] = colPlusOffd_2 + T_colinds_2[j];
                    dataInd += 1;
                }

                // Add this row of spatial discretization to first off-diagonal block
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L/2), otherwise data is dt*L/2
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds_1[j]) {
                        data[dataInd] = -1 + 3.0*dt*T_data_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = 3.0*dt*T_data_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd_1 + T_colinds_1[j];
                    dataInd += 1;
                }

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*(3.0*Bi_1[i] - Bi_2[i]) / 2.0;
                X[thisRow] = Xi_1[i];

                // Total nonzero for this row on processor is one for diagonal plue the
                // total nnz in this row the spatial discretization at time ti and ti-1.
                rowptr[thisRow+1] = rowptr[thisRow] + (T_rowptr_1[i+1] - T_rowptr_1[i]) +
                                    (T_rowptr_2[i+1] - T_rowptr_2[i]) + 1;
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }
    delete[] T_rowptr_1;
    delete[] T_colinds_1;
    delete[] T_data_1;
    delete[] Bi_1;
    delete[] Xi_1;
    if (isTimeDependent) {
        delete[] T_rowptr_2;
        delete[] T_colinds_2;
        delete[] T_data_2;
        delete[] Bi_2;
        delete[] Xi_2;   
    }
}