#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "mfem.hpp"
#include "HYPRE.h"

using namespace mfem;

void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr, int *&A_colinds,
                              double *&A_data, double *&B, double *&X, int &localMinRow,
                              int &localMaxRow, int &spatialDOFs, double t,
                              int ref_levels, int order);
void BDF1(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
          int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
          int &localMaxRow, int &spatialDOFs, int ref_levels, int order);
void BDF2(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
          int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
          int &localMaxRow, int &spatialDOFs, int ref_levels, int order);
void BDF3(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
          int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
          int &localMaxRow, int &spatialDOFs, int ref_levels, int order);
void AM2(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
         int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
         int &localMaxRow, int &spatialDOFs, int ref_levels, int order);
void AB1(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr, int *&colinds,
          double *&data, double *&B, double *&X, int &localMinRow, int &localMaxRow,
          int &spatialDOFs, int ref_levels, int order);
void AB2(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr, int *&colinds,
          double *&data, double *&B, double *&X, int &localMinRow, int &localMaxRow,
          int &spatialDOFs, int ref_levels, int order);


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess, myTimeInd, spatialRank, Np_x;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);
    MPI_Comm spatialComm;

    std::cout << "Initialized MPI, " << rank << "/" << numProcess << "\n";

    // Parameters
    bool use_spatial_parallel;
    bool isTimeDependent = true;
    int ntPerProc;
    int numTimeSteps = 2;
    int ref_levels = 1;
    int order = 1;
    int bdf = -1;
    int am = -1;
    int ab = -1;
    bool save_mat = false;
    int distance = 1;
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

    // Check that number of time steps divides the number MPI processes or vice versa.
    if (numTimeSteps <= numProcess) {
        use_spatial_parallel = true;
        if (numProcess % numTimeSteps != 0) {
            if (rank == 0) {
                std::cout << "Error: number of time steps does not divide number of processes.\n";
            }
            MPI_Finalize();
            return 1;
        }
        else {
            Np_x = numProcess / numTimeSteps;
        }
        // Set up communication group for spatial discretizations.
        myTimeInd = rank / Np_x;
        int spatialSize;
        MPI_Comm_split(MPI_COMM_WORLD, myTimeInd, rank, &spatialComm);
        MPI_Comm_rank(spatialComm, &spatialRank);
        MPI_Comm_size(spatialComm, &spatialSize);
        std::cout << "Global comm size: " << numProcess << ", local comm size: " << spatialSize << "\n";
    }
    else {
        use_spatial_parallel = false;
        if (numTimeSteps % numProcess  != 0) {
            if (rank == 0) {
                std::cout << "Error: number of processes does not divide number of time steps.\n";
            }
            MPI_Finalize();
            return 1;
        }
        // Time steps computed per processor. 
        ntPerProc = numTimeSteps / numProcess;
    }

    // Get CSR space-time matrix structure
    int *rowptr;
    int *colinds;
    double *data;
    double *B;
    double *X;
    int localMinRow;
    int localMaxRow;
    int spatialDOFs;
    if (use_spatial_parallel) {
        if (bdf == 1) {
            BDF1(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
        else if (bdf == 2) {
            BDF2(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
        else if (bdf == 3) {
            BDF3(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
        else if (am == 1) {
            BDF1(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
        else if (am == 2) {
            AM2(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
        else if (ab == 1) {
            AB1(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
#if 0
        else if (ab == 2) {
            AB2(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
#endif
        else {
            std::cout << "WARNING: invalid integration parameters.\n";
            MPI_Finalize();
            return 1;
        }
    }
    // More than one time-step per processor
    else {
        if (bdf == 1) {
            BDF1(spatialComm, myTimeInd, dt, rowptr, colinds, data, B, X,
                 localMinRow, localMaxRow, spatialDOFs, ref_levels, order);
        }
#if 0
        else if (bdf == 2) {
            BDF2(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
        else if (bdf == 3) {
            BDF3(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
        else if (am == 1) {
            BDF1(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
        else if (am == 2) {
            AM2(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
        else if (ab == 1) {
            AB1(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
        else if (ab == 2) {
            AB2(rank, ntPerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
        }
        else {
            std::cout << "WARNING: invalid integration parameters.\n";
            MPI_Finalize();
            return 1;
        }
#endif
    }

    // Initialize matrix
    int onProcSize = localMaxRow - localMinRow + 1;
    int ilower = myTimeInd*spatialDOFs + localMinRow;
    int iupper = myTimeInd*spatialDOFs + localMaxRow;
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
    HYPRE_BoomerAMGSetCoarsenType(precond, coarsening);
    HYPRE_BoomerAMGSetAggNumLevels(precond, 0);
    HYPRE_BoomerAMGSetStrongThreshold(precond, strength_tolC);
    HYPRE_BoomerAMGSetStrongThresholdR(precond, strength_tolR);
    HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
    if (relax_type > -1) {
        HYPRE_BoomerAMGSetRelaxType(precond, relax_type);
    }
    HYPRE_BoomerAMGSetCycleNumSweeps(precond, ns_coarse, 3);
    HYPRE_BoomerAMGSetCycleNumSweeps(precond, ns_down,   1);
    HYPRE_BoomerAMGSetCycleNumSweeps(precond, ns_up,     2);
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
/*      - Given spatial communication group, must return the same row distribution
/*        over processors each time called.                                             */
void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr, int *&A_colinds,
                              double *&A_data, double *&B, double *&X, int &localMinRow,
                              int &localMaxRow, int &spatialDOFs, double t,
                              int ref_levels, int order)
{
    // Read mesh from mesh file
    const char *mesh_file = "../meshes/beam-quad.mesh";
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine mesh
    // double max_elements = 50000;
    // int ref_levels = (int)floor(log(max_elements./mesh->GetNE())/log(2.)/dim);
    int ser_ref_levels = std::min(3,ref_levels);
    for (int l=0; l<ser_ref_levels; l++) {
        mesh->UniformRefinement();
    }

    // Define parallel mesh by a partitioning of the serial mesh.
    ParMesh *pmesh = new ParMesh(spatialComm, *mesh);
    delete mesh;
    int par_ref_levels = ref_levels - 3;
    for (int l = 0; l < par_ref_levels; l++) {
        pmesh->UniformRefinement();
    }

    // Define finite element space on mesh
    FiniteElementCollection *fec;
    fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (pmesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear form b(.) 
    ParLinearForm *b = new ParLinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Define solution vector x as finite element grid function corresponding to fespace.
    // Initialize x with initial guess of zero, which satisfies the boundary conditions.
    ParGridFunction x(fespace);
    x = 0.0;

    // Set up bilinear form a(.,.) on finite element space corresponding to Laplacian
    // operator -Delta, by adding the Diffusion domain integrator.
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // Assemble bilinear form and corresponding linear system
    int spatialRank;
    MPI_Comm_rank(spatialComm, &spatialRank);
    HypreParMatrix A;
    Vector B0;
    Vector X0;
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X0, B0);

    spatialDOFs = A.GetGlobalNumRows();
    int *rowStarts = A.GetRowStarts();
    localMinRow = rowStarts[0];
    localMaxRow = rowStarts[1]-1;

    // Steal vector data to pointers
    B = B0.StealData();
    X = X0.StealData();

    // Compress diagonal and off-diagonal blocks of hypre matrix to local CSR
    SparseMatrix A_loc;
    A.GetProcRows(A_loc);
    A_rowptr = A_loc.GetI();
    A_colinds = A_loc.GetJ();
    A_data = A_loc.GetData();
    A_loc.LoseData();

    delete a;
    delete b;
 
    if (fec) {
      delete fespace;
      delete fec;
    }
    delete pmesh;

    // TODO: remove
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


/* First-order Adams-Bashforth (also known as Backward Euler and first-order Adams-Moulton) */
void BDF1(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr, int *&colinds,
          double *&data, double *&B, double *&X, int &localMinRow, int &localMaxRow,
          int &spatialDOFs, int ref_levels, int order)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    getSpatialDiscretization(spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             dt*tInd, ref_levels, order);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get number nnz on this processor.
    int procNnz;
    if (tInd == 0) procNnz = nnzPerTime;
    else procNnz = nnzPerTime + procRows;

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    std::cout << "Time index: " << tInd << "\n\tSpatial DOFs: " << spatialDOFs
              << "\n\tRows/processor: " << procRows << "\n\tAllocated nnz: "
              << procNnz << "\n";

    // Column indices for this processor given the time index and the first row
    // stored on this processor of the spatial discretization (zero if whole
    // spatial problem stored on one processor).
    int colPlusOffd = (tInd - 1)*spatialDOFs + localMinRow;
    int colPlusDiag = tInd*spatialDOFs;

    // At time t=0, only have spatial discretization block
    if (tInd == 0) {
        
        // Loop over each on-processor row in spatial discretization at time tInd
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + dt*T_data[j];
                }
                else {
                    data[dataInd] = dt*T_data[j];
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= dt;

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    else {
        // Loop over each row in spatial discretization at time tInd
        for (int i=0; i<procRows; i++) {
            // Add off-diagonal block, -u_{i-1}
            colinds[dataInd] = colPlusOffd + i;
            data[dataInd] = -1.0;
            dataInd += 1;

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + dt*T_data[j];
                }
                else {
                    data[dataInd] = dt*T_data[j];
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= dt;

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
}


void BDF2(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
          int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
          int &localMaxRow, int &spatialDOFs, int ref_levels, int order)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    getSpatialDiscretization(spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             dt*tInd, ref_levels, order);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    
    
    // Get number nnz on this processor.
    int procNnz;
    if (tInd == 0) procNnz = nnzPerTime;
    else if (tInd == 1) procNnz = nnzPerTime + procRows;
    else procNnz = nnzPerTime + 2*procRows;

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    std::cout << "Time index: " << tInd << "\n\tSpatial DOFs: " << spatialDOFs
              << "\n\tRows/processor: " << procRows << "\n\tAllocated nnz: "
              << procNnz << "\n";

    // Column indices for this processor given the time index and the first row
    // stored on this processor of the spatial discretization (zero if whole
    // spatial problem stored on one processor).
    int colPlusOffd_2 = (tInd - 2)*spatialDOFs + localMinRow;
    int colPlusOffd_1 = (tInd - 1)*spatialDOFs + localMinRow;
    int colPlusDiag   = tInd*spatialDOFs;

    // At time t=0, only have spatial discretization block
    if (tInd == 0) {
        
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 2.0*dt*T_data[j] / 3.0;
                }
                else {
                    data[dataInd] = 2.0*dt*T_data[j] / 3.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0*dt/3.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    // At time t=1, only have 1 off-diagonal block
    else if (tInd == 1) {
        // Loop over each row in spatial discretization at time t1
        for (int i=0; i<procRows; i++) {
            // Add off-diagonal block, -u_{i-1}
            colinds[dataInd] = colPlusOffd_1 + i;
            data[dataInd] = -4.0/3.0;
            dataInd += 1;

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 3dt*L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 2.0*dt*T_data[j] / 3.0;
                }
                else {
                    data[dataInd] = 2.0*dt*T_data[j] / 3.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0*dt/3.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
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
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 2.0*dt*T_data[j] / 3.0;
                }
                else {
                    data[dataInd] = 2.0*dt*T_data[j] / 3.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0*dt/3.0);

            // Total nonzero for this row on processor is two for off-diagonal blocks
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 2 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
}


void BDF3(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
          int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
          int &localMaxRow, int &spatialDOFs, int ref_levels, int order)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    getSpatialDiscretization(spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             dt*tInd, ref_levels, order);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    
    
    // Get number nnz on this processor.
    int procNnz;
    if (tInd == 0) procNnz = nnzPerTime;
    else if (tInd == 1) procNnz = nnzPerTime + procRows;
    else if (tInd == 2) procNnz = nnzPerTime + 2*procRows;
    else procNnz = nnzPerTime + 3*procRows;

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    std::cout << "Time index: " << tInd << "\n\tSpatial DOFs: " << spatialDOFs
              << "\n\tRows/processor: " << procRows << "\n\tAllocated nnz: "
              << procNnz << "\n";

    // Column indices for this processor given the time index and the first row
    // stored on this processor of the spatial discretization (zero if whole
    // spatial problem stored on one processor).
    int colPlusOffd_3 = (tInd - 3)*spatialDOFs + localMinRow;
    int colPlusOffd_2 = (tInd - 2)*spatialDOFs + localMinRow;
    int colPlusOffd_1 = (tInd - 1)*spatialDOFs + localMinRow;
    int colPlusDiag   = tInd*spatialDOFs;

    // At time t=0, only have spatial discretization block
    if (tInd == 0) {
        
        // Loop over each row in spatial discretization at time tInd
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0*dt/11.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    // At time t=1, only have 1 off-diagonal block
    else if (tInd == 1) {
        // Loop over each row in spatial discretization at time t1
        for (int i=0; i<procRows; i++) {
            // Add off-diagonal block, -u_{i-1}
            colinds[dataInd] = colPlusOffd_1 + i;
            data[dataInd] = -18.0/11.0;
            dataInd += 1;

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 3dt*L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0*dt/11.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }
    else if (tInd == 2) {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
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
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0*dt/11.0);

            // Total nonzero for this row on processor is two for off-diagonal blocks
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 2 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
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
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*dt*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*dt*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0*dt/11.0);

            // Total nonzero for this row on processor is two for off-diagonal blocks
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 3 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
}



/* Second-order Adams-Moulton implicit scheme (trapezoid method) */
//  TODO: What to do with B and X here as passed to getSpatialDisc()?
//        Must fix in loops as well when defining B, X
void AM2(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
         int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
         int &localMaxRow, int &spatialDOFs, int ref_levels, int order)
{
    // Get spatial discretization for previous time step, or first step if tInd0=0
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *Bi = NULL;
    double *Xi = NULL;
    int *T_rowptr_1 = NULL;
    int *T_colinds_1 = NULL;
    double *T_data_1 = NULL;
    double *Bi_1 = NULL;
    double *Xi_1 = NULL;
    int procNnz;
    getSpatialDiscretization(spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             dt*(tInd), ref_levels, order);
    int procRows = localMaxRow - localMinRow + 1;
    procNnz = T_rowptr[procRows];

    // Get discretization at time ti-1 for Adams-Moulton if tInd!=0. 
    if (tInd > 0) {
        int localMinRow_1;
        int localMaxRow_1;
        getSpatialDiscretization(spatialComm, T_rowptr_1, T_colinds_1, T_data_1,
                                 B, X, localMinRow_1, localMaxRow_1,
                                 spatialDOFs, dt*(tInd-1), ref_levels, order);
     
        // Check that discretization at time ti and ti-1 allocate the same rows
        // to this processor.
        if ((localMinRow != localMinRow_1) || (localMaxRow != localMaxRow_1)) {
            std::cout << "WARNING: different rows allocated to processor at time "
                         "t_i and t_{i-1}. Ending program.\n";
            MPI_Finalize();
        }
        procNnz += T_rowptr_1[procRows];
    }

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    std::cout << "Time index: " << tInd << "\n\tSpatial DOFs: " << spatialDOFs
              << "\n\tRows/processor: " << procRows << "\n\tAllocated nnz: "
              << procNnz << "\n";

    // Local CSR matrices have (spatially) global column indices. Do not need
    // to account for the min row indexing as in BDF.
    int colPlusOffd = (tInd - 1)*spatialDOFs;
    int colPlusDiag = tInd*spatialDOFs;

    // At time t=0, only have spatial discretization at t0.
    if (tInd == 0) {
        // Loop over each row in spatial discretization at time t0
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + dt*T_data[j] / 2.0;
                }
                else {
                    data[dataInd] = dt*T_data[j] / 2.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            //  TODO: FIX
            // B[i] = dt*Bi[i] / 2.0;
            // X[i] = Xi[i];

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

            // Add row for spatial discretization of off-diagonal block 
            for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {

                // Add identity to diagonal, (-I + dt*L/2), otherwise data is dt*L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds_1[j]) {
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
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + dt*T_data[j] / 2.0;
                }
                else {
                    data[dataInd] = dt*T_data[j] / 2.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            //  TODO: FIX
            // B[i] = dt*(Bi[i] + Bi_1[i]) / 2.0;
            // X[i] = Xi[i];

            // Total nonzero for this row on processor is the total nnz in this row
            // the spatial discretization at time ti and ti-1.
            rowptr[i+1] = rowptr[i] + (T_rowptr[i+1] - T_rowptr[i]) +
                                (T_rowptr_1[i+1] - T_rowptr_1[i]);
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] Bi;
    delete[] Xi;
    delete[] T_rowptr_1;
    delete[] T_colinds_1;
    delete[] T_data_1;
    delete[] Bi_1;
    delete[] Xi_1;
}


/* First-order Adams-Bashforth (Forward Euler) */
void AB1(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr, int *&colinds,
          double *&data, double *&B, double *&X, int &localMinRow, int &localMaxRow,
          int &spatialDOFs, int ref_levels, int order)
{
    int spatialRank;
    MPI_Comm_rank(spatialComm, &spatialRank);

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    if (tInd == 0) {    
        getSpatialDiscretization(spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 dt*tInd, ref_levels, order);
    }
    else {
        getSpatialDiscretization(spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 dt*(tInd-1), ref_levels, order);
    }
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int procNnz;
    if (tInd == 0) procNnz = procRows;
    else procNnz = nnzPerTime + procRows;

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    std::cout << "Time index: " << tInd << "\n\tSpatial DOFs: " << spatialDOFs
              << "\n\tRows/processor: " << procRows << "\n\tAllocated nnz: "
              << procNnz << "\n";

    // Local CSR matrices in off-diagonal blocks here have (spatially) global
    // column indices. Only need to account for min row indexing for the
    // diagonal block.
    int colPlusOffd = (tInd - 1)*spatialDOFs;
    int colPlusDiag = tInd*spatialDOFs + localMinRow;

    // At time t=0, only have identity block on diagonal
    if (tInd == 0) {
        
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

            // Add identity as diagonal block, u_ti
            colinds[dataInd] = colPlusDiag + i;
            data[dataInd] = 1.0;
            dataInd += 1;

            // Assume user implements boundary conditions to rhs
            B[i] = 0.0;
            // X[i] = X0[i];

            // One nonzero for this row
            rowptr[i+1] = rowptr[i] + 1;
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

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
            B[i] *= dt;
            // X[i] = X0[i];

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }

    // Check if sufficient data was allocated
    if (dataInd != procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
}


/* Second-order Adams-Bashforth explicit scheme */
void AB2(const MPI_Comm &spatialComm, int tInd, double dt, int *&rowptr,
         int *&colinds, double *&data, double *&B, double *&X, int &localMinRow,
         int &localMaxRow, int &spatialDOFs, int ref_levels, int order)
{
    // Get spatial discretization for previous time step, or first step if tInd0=0
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *Bi = NULL;
    double *Xi = NULL;
    int *T_rowptr_1 = NULL;
    int *T_colinds_1 = NULL;
    double *T_data_1 = NULL;
    double *Bi_1 = NULL;
    double *Xi_1 = NULL;
    int procNnz;
    if (tInd0 <= 1) {    
        getSpatialDiscretization(Tb, Bb, Xb, 0.0, ref_levels, order);
    }
    else {
        getSpatialDiscretization(Tb, Bb, Xb, dt*(tInd0-2), ref_levels, order);
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int spatialDOFs = Tb.NumRows();
    int nnzPerTime  = Tb.NumNonZeroElems();
    int procNnz     = ntPerProc * (2*nnzPerTime + spatialDOFs);     // nnzs on this processor
    int procRows     = ntPerProc * spatialDOFs;
    if (tInd0 == 0) procNnz -= 2*nnzPerTime;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= spatialDOFs;

    int *rowptr  = new int[procRows + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRows);
    X.SetSize(procRows);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << ntPerProc
                  << "\nDOFs per time step: " << spatialDOFs << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    int colPlusOffd_1 = (ti - 1)*spatialDOFs;
    int colPlusOffd_2 = (ti - 2)*spatialDOFs;
    int colPlusDiag = ti*spatialDOFs;


    // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
    // to build one spatial matrix each iteration. Matrix at ti for previous iteration
    //  is used as ti-1 on this iteration.
    if (ti <= 1) {
        T_rowptr_1  = Tb.GetI();
        T_colinds_1 = Tb.GetJ();
        T_data_1    = Tb.GetData();
        Bi_1          = &Bb[0];
        Xi_1          = &Xb[0];
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
