#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "mfem.hpp"
#include "HYPRE.h"

using namespace mfem;

void getSpatialDiscretization(SparseMatrix &A, Vector &B, Vector &X, double t,
                              int ref_levels, int order);
void BDF1(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);
void BDF2(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);
void BDF3(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);
void AM2(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);
void AB1(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);
void AB2(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);



//  TODO:
// --------
//      - Error in running hypre on AB1 and AB2 in serial
//      - Error in most cases when running hypre with mutiple processors
//      - Here we assume spatial problem always fits on one processor. Implement
//        comm groups so that spatial disc. can be put on multiple processors. 


/* Sample routine to split processors over space and time. */
// braid_Int
// braid_SplitCommworld(const MPI_Comm  *comm_world,
//                      braid_Int       px,
//                      MPI_Comm        *comm_x,
//                      MPI_Comm        *comm_t)
// {
//    braid_Int myid, xcolor, tcolor;

//    /* Create communicators for the time and space dimensions */
//    /* The communicators are based on colors and keys (= myid) */
//    MPI_Comm_rank( *comm_world, &myid );
//    xcolor = myid / px;
//    tcolor = myid % px;

//    MPI_Comm_split( *comm_world, xcolor, myid, comm_x );
//    MPI_Comm_split( *comm_world, tcolor, myid, comm_t );

//    return _braid_error_flag;
// }




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
    int timePerProc = numTimeSteps / numProcess;

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
    SparseMatrix A;
    Vector B, X;
    if (bdf == 1) {
        BDF1(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else if (bdf == 2) {
        BDF2(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else if (bdf == 3) {
        BDF3(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else if (am == 1) {
        BDF1(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else if (am == 2) {
        AM2(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else if (ab == 1) {
        AB1(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else if (ab == 2) {
        AB2(rank, timePerProc, numTimeSteps, dt, isTimeDependent, A, B, X, ref_levels, order);
    }
    else {
        std::cout << "WARNING: invalid integration parameters.\n";
        MPI_Finalize();
        return 1;
    }
    A.Finalize(1);

    // Initialize matrix
    int onprocSize = A.NumRows();
    int ilower = rank*onprocSize;
    int iupper = (rank+1)*onprocSize - 1;
    HYPRE_IJMatrix      spaceTimeMat0;
    HYPRE_ParCSRMatrix  spaceTimeMat;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &spaceTimeMat0);
    HYPRE_IJMatrixSetObjectType(spaceTimeMat0, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(spaceTimeMat0);

    // Set matrix coefficients
    int *rowptr  = A.GetI();
    int *colinds = A.GetJ();
    double *data = A.GetData();   
    int *rows = new int[onprocSize];
    int *cols_per_row = new int[onprocSize];
    for (int i=0; i<onprocSize; i++) {
        rows[i] = ilower + i;
        cols_per_row[i] = rowptr[i+1] - rowptr[i];
    }
    HYPRE_IJMatrixSetValues(spaceTimeMat0, onprocSize, cols_per_row, rows, colinds, data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(spaceTimeMat0);
    HYPRE_IJMatrixGetObject(spaceTimeMat0, (void **) &spaceTimeMat);
    
    // Save matrix to file to debug
    if (save_mat) {
        const char* filename1 = "./test.mm";
        HYPRE_IJMatrixPrint (spaceTimeMat0, filename1);
    }

    /* Create sample rhs and solution vectors */
    double *temp = B.GetData();
    std::cout << B.Size() << ": ";
    // for (int i=0; i<onprocSize; i++) {
    //     std::cout << temp[i] << ", ";
    // }
    std::cout << "\ndone\n";
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorSetValues(b, onprocSize, rows, temp);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    double *temp2 = X.GetData();
    std::cout << X.Size() << ": ";
    // for (int i=0; i<onprocSize; i++) {
    //     std::cout << temp[i] << ", ";
    // }
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;  
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);
    HYPRE_IJVectorSetValues(x, onprocSize, rows, temp2);
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

    // Save to file by MPI rank
    if (save_mat) {
        std::stringstream filename;
        filename << "test_mat_" << rank << ".mm";
        std::ofstream outfile(filename.str()); 
        A.PrintMM(outfile);
    }

    std::cout << "cleaning up\n";


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
void getSpatialDiscretization(SparseMatrix &A, Vector &B, Vector &X, double t,
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
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B, 0);

    // Change ownership of matrix data to sparse matrix from bilinear form
    A.SetGraphOwner(true);
    A.SetDataOwner(true);
    a->LoseMat();
    delete a;
    X.SetData(x.StealData());
    B.SetData(b->StealData());
    delete b;
 
    if (fec) {
      delete fespace;
      delete fec;
    }
    delete mesh;

    // TODO: remove
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


/* First-order Adams-Bashforth (also known as Backward Euler and first-order Adams-Moulton) */
void BDF1(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order)
{
    int tInd0 = rank*timePerProc;
    int tInd1 = tInd0 + timePerProc -1;

    // Get spatial discretization for first time step on this processor
    SparseMatrix T;
    Vector B0, X0;
    getSpatialDiscretization(T, B0, X0, dt*tInd0, ref_levels, order);
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int rowsPerTime = T.NumRows();
    int nnzPerTime  = T.NumNonZeroElems();
    int procNnz     = timePerProc * (nnzPerTime + rowsPerTime);     // nnzs on this processor
    int procRow     = timePerProc * rowsPerTime;
    if (tInd0 == 0) procNnz -= rowsPerTime;

    int *rowptr  = new int[procRow + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRow);
    X.SetSize(procRow);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << timePerProc
                  << "\nDOFs per time step: " << rowsPerTime << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            getSpatialDiscretization(T, B0, X0, dt*ti, ref_levels, order); 
        }

        int *tempRowptr  = T.GetI();
        int *tempColinds = T.GetJ();
        double *tempData = T.GetData();
        int colPlusOffd  = (ti - 1)*rowsPerTime;
        int colPlusDiag  = ti*rowsPerTime;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + dt*tempData[j];
                    }
                    else {
                        data[dataInd] = dt*tempData[j];
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + tempRowptr[i+1] - tempRowptr[i];
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {
                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd + i;
                data[dataInd] = -1.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + dt*tempData[j];
                    }
                    else {
                        data[dataInd] = dt*tempData[j];
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }

    // Build sparse matrix structure for all rows on this processor
    A = SparseMatrix(rowptr, colinds, data, procRow, numTimeSteps*rowsPerTime);
    if (rank == 1) std::cout << "\nBuilt mat.\n";
}


void BDF2(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order)
{
    int tInd0 = rank*timePerProc;
    int tInd1 = tInd0 + timePerProc -1;

    // Get spatial discretization for first time step on this processor
    SparseMatrix T;
    Vector B0, X0;
    getSpatialDiscretization(T, B0, X0, dt*tInd0, ref_levels, order);
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int rowsPerTime = T.NumRows();
    int nnzPerTime  = T.NumNonZeroElems();
    int procNnz     = timePerProc * (nnzPerTime + 2*rowsPerTime);     // nnzs on this processor
    int procRow     = timePerProc * rowsPerTime;
    if (tInd0 == 0) procNnz -= 2*rowsPerTime;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= rowsPerTime;

    int *rowptr  = new int[procRow + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRow);
    X.SetSize(procRow);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << timePerProc
                  << "\nDOFs per time step: " << rowsPerTime << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            getSpatialDiscretization(T, B0, X0, dt*ti, ref_levels, order); 
        }

        int *tempRowptr   = T.GetI();
        int *tempColinds  = T.GetJ();
        double *tempData  = T.GetData();
        int colPlusDiag   = ti*rowsPerTime;
        int colPlusOffd_1 = (ti - 1)*rowsPerTime;
        int colPlusOffd_2 = (ti - 2)*rowsPerTime;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 2.0*dt*tempData[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*dt*tempData[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*dt*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + tempRowptr[i+1] - tempRowptr[i];
                thisRow += 1;
            }
        }
        // At time t=1, only have 1 off-diagonal block
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time t1
            for (int i=0; i<rowsPerTime; i++) {
                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -4.0/3.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 3dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 2.0*dt*tempData[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*dt*tempData[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*dt*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {
                // Add off-diagonal block, -u_{i-2}
                colinds[dataInd] = colPlusOffd_2 + i;
                data[dataInd] = 1.0/3.0;
                dataInd += 1;

                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -4.0/3.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 2.0*dt*tempData[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*dt*tempData[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*dt*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 2 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }
    
        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }

    // Build sparse matrix structure for all rows on this processor
    A = SparseMatrix(rowptr, colinds, data, procRow, numTimeSteps*rowsPerTime);
    A.Finalize(1);
    if (rank == 1) std::cout << "\nBuilt mat.\n";
}


void BDF3(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order)
{
    int tInd0 = rank*timePerProc;
    int tInd1 = tInd0 + timePerProc -1;

    // Get spatial discretization for first time step on this processor
    SparseMatrix T;
    Vector B0, X0;
    getSpatialDiscretization(T, B0, X0, dt*tInd0, ref_levels, order);
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int rowsPerTime = T.NumRows();
    int nnzPerTime  = T.NumNonZeroElems();
    int procNnz     = timePerProc * (nnzPerTime + 3*rowsPerTime);     // nnzs on this processor
    int procRow     = timePerProc * rowsPerTime;
    if (tInd0 == 0) procNnz -= 3*rowsPerTime;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= 2*rowsPerTime;
    if ((tInd0 <= 2) && (tInd1 >= 2)) procNnz -= rowsPerTime;

    int *rowptr  = new int[procRow + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRow);
    X.SetSize(procRow);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << timePerProc
                  << "\nDOFs per time step: " << rowsPerTime << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            getSpatialDiscretization(T, B0, X0, dt*ti, ref_levels, order); 
        }

        int *tempRowptr   = T.GetI();
        int *tempColinds  = T.GetJ();
        double *tempData  = T.GetData();
        int colPlusDiag   = ti*rowsPerTime;
        int colPlusOffd_1 = (ti - 1)*rowsPerTime;
        int colPlusOffd_2 = (ti - 2)*rowsPerTime;
        int colPlusOffd_3 = (ti - 3)*rowsPerTime;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*tempData[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*tempData[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + tempRowptr[i+1] - tempRowptr[i];
                thisRow += 1;
            }
        }
        // At time t=1, only have 1 off-diagonal block
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time t1
            for (int i=0; i<rowsPerTime; i++) {
                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -18.0/11.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 3dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*tempData[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*tempData[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }
        else if (ti == 2) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {
                // Add off-diagonal block, -u_{i-2}
                colinds[dataInd] = colPlusOffd_2 + i;
                data[dataInd] = 9.0/11.0;
                dataInd += 1;

                // Add off-diagonal block, -u_{i-1}
                colinds[dataInd] = colPlusOffd_1 + i;
                data[dataInd] = -18.0/11.0;
                dataInd += 1;

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*tempData[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*tempData[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 2 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {
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
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + 2dt*L/3), otherwise data is 2dt*L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + 6.0*dt*tempData[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*dt*tempData[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*dt*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 3 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }

    // Build sparse matrix structure for all rows on this processor
    A = SparseMatrix(rowptr, colinds, data, procRow, numTimeSteps*rowsPerTime);
    A.Finalize(1);
    if (rank == 1) std::cout << "\nBuilt mat.\n";
}



/* Second-order Adams-Moulton implicit scheme (trapezoid method) */
void AM2(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order)
{
    int tInd0 = rank*timePerProc;
    int tInd1 = tInd0 + timePerProc -1;

    // Get spatial discretization for previous time step, or first step if tInd0=0
    SparseMatrix Ta, Tb;
    Vector Ba, Bb, Xa, Xb;
    if (tInd0 > 0) {
        getSpatialDiscretization(Tb, Bb, Xb, dt*(tInd0-1), ref_levels, order);
    }
    else {
        getSpatialDiscretization(Tb, Bb, Xb, dt*tInd0, ref_levels, order);
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int rowsPerTime = Tb.NumRows();
    int nnzPerTime  = Tb.NumNonZeroElems();
    int procNnz     = 2 * timePerProc * nnzPerTime;     // nnzs on this processor
    int procRow     = timePerProc * rowsPerTime;
    if (tInd0 == 0) procNnz -= nnzPerTime;

    int *rowptr  = new int[procRow + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRow);
    X.SetSize(procRow);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << timePerProc
                  << "\nDOFs per time step: " << rowsPerTime << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Set index to swap pointers to zero if spatial problem is not time dependent. Avoids
    // rebuilding matrix each iteration. 
    int swap_ind;
    if (isTimeDependent) {
        swap_ind = 1;
    }
    else {
        swap_ind = 0;
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        int colPlusOffd = (ti - 1)*rowsPerTime;
        int colPlusDiag = ti*rowsPerTime;

        // Pointers to CSR arrays for A_{ti} and A_{ti-1}
        int *tempRowptr;
        int *tempColinds;
        double *tempData;
        double *Bi;
        int *tempRowptr_1;
        int *tempColinds_1;
        double *tempData_1;
        double *Bi_1;
        double *Xi;

        // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
        // to build one spatial matrix each iteration. Matrix at ti for previous iteration
        //  is used as ti-1 on this iteration.
        if (ti == 0) {
            tempRowptr    = Tb.GetI();
            tempColinds   = Tb.GetJ();
            tempData      = Tb.GetData();
            Bi            = &Bb[0];
            Xi            = &Xb[0];
        }
        else if (swap_ind > 0) {
            getSpatialDiscretization(Ta, Ba, Xa, dt*ti, ref_levels, order); 
            tempRowptr    = Ta.GetI();
            tempColinds   = Ta.GetJ();
            tempData      = Ta.GetData();
            Bi            = &Ba[0];
            Xi            = &Xa[0];
            tempRowptr_1  = Tb.GetI();
            tempColinds_1 = Tb.GetJ();
            tempData_1    = Tb.GetData();
            Bi_1          = &Bb[0];
        }
        else if (swap_ind < 0) {
            getSpatialDiscretization(Tb, Bb, Xb, dt*ti, ref_levels, order); 
            tempRowptr    = Tb.GetI();
            tempColinds   = Tb.GetJ();
            tempData      = Tb.GetData();
            Bi            = &Bb[0];
            Xi            = &Xb[0];
            tempRowptr_1  = Ta.GetI();
            tempColinds_1 = Ta.GetJ();
            tempData_1    = Ta.GetData();
            Bi_1          = &Ba[0];
        }
        else {
            tempRowptr    = Tb.GetI();
            tempColinds   = Tb.GetJ();
            tempData      = Tb.GetData();
            Bi            = &Bb[0];
            tempRowptr_1  = Tb.GetI();
            tempColinds_1 = Tb.GetJ();
            tempData_1    = Tb.GetData();            
            Bi_1          = &Bb[0];
            Xi            = &Xb[0];
        }

        // At time t=0, only have spatial discretization block.
        if (ti == 0) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + dt*tempData[j] / 2.0;
                    }
                    else {
                        data[dataInd] = dt*tempData[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*Bi[i] / 2.0;
                X[thisRow] = Xi[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + tempRowptr[i+1] - tempRowptr[i];
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

                // Add row for spatial discretization of off-diagonal block 
                for (int j=tempRowptr_1[i]; j<tempRowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (-I + dt*L/2), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds_1[j]) {
                        data[dataInd] = -1 + dt*tempData_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = dt*tempData_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd + tempColinds_1[j];
                    dataInd += 1;
                }

                // Add this row of spatial discretization to diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L/2), otherwise data is dt*L/2
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = 1 + dt*tempData[j] / 2.0;
                    }
                    else {
                        data[dataInd] = dt*tempData[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusDiag + tempColinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = dt*(Bi[i] + Bi_1[i]) / 2.0;
                X[thisRow] = Xi[i];

                // Total nonzero for this row on processor is the total nnz in this row
                // the spatial discretization at time ti and ti-1.
                rowptr[thisRow+1] = rowptr[thisRow] + (tempRowptr[i+1] - tempRowptr[i]) +
                                    (tempRowptr_1[i+1] - tempRowptr_1[i]);
                thisRow += 1;
            }
        }

        // Change sign of pointer index (except for on time t=0 iteration)
        if (ti > 0) swap_ind *= -1;

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
        if (thisRow > procRow) {
            std::cout << "WARNING: Matrix has more rows than allocated.\n";
        }
    }

    // Build sparse matrix structure for all rows on this processor
    A = SparseMatrix(rowptr, colinds, data, procRow, numTimeSteps*rowsPerTime);
    if (rank == 1) std::cout << "\nBuilt mat.\n";
}


/* First-order Adams-Bashforth (Forward Euler) */
void AB1(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order)
{
    int tInd0 = rank*timePerProc;
    int tInd1 = tInd0 + timePerProc -1;

    // Get spatial discretization for first time step on this processor
    SparseMatrix T;
    Vector B0, X0;
    if (tInd0 == 0) {    
        getSpatialDiscretization(T, B0, X0, dt*tInd0, ref_levels, order);
    }
    else {
        getSpatialDiscretization(T, B0, X0, dt*(tInd0-1), ref_levels, order);
    }
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int rowsPerTime = T.NumRows();
    int nnzPerTime  = T.NumNonZeroElems();
    int procNnz     = timePerProc * (nnzPerTime + rowsPerTime);     // nnzs on this processor
    int procRow     = timePerProc * rowsPerTime;
    if (tInd0 == 0) procNnz -= rowsPerTime;

    int *rowptr  = new int[procRow + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRow);
    X.SetSize(procRow);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << timePerProc
                  << "\nDOFs per time step: " << rowsPerTime << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && isTimeDependent) {
            getSpatialDiscretization(T, B0, X0, dt*(ti-1), ref_levels, order); 
        }

        int *tempRowptr  = T.GetI();
        int *tempColinds = T.GetJ();
        double *tempData = T.GetData();
        int colPlusOffd  = (ti - 1)*rowsPerTime;
        int colPlusDiag  = ti*rowsPerTime;

        // At time t=0, only have identity block on diagonal
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

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
            for (int i=0; i<rowsPerTime; i++) {

                // Add spatial discretization at time ti-1 to off-diagonal block
                for (int j=tempRowptr[i]; j<tempRowptr[i+1]; j++) {

                    // Subtract identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds[j]) {
                        data[dataInd] = -1 + dt*tempData[j];
                    }
                    else {
                        data[dataInd] = dt*tempData[j];
                    }
                    colinds[dataInd] = colPlusOffd + tempColinds[j];
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
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + (tempRowptr[i+1] - tempRowptr[i]);
                thisRow += 1;
            }
        }

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
    }

    // Build sparse matrix structure for all rows on this processor
    A = SparseMatrix(rowptr, colinds, data, procRow, numTimeSteps*rowsPerTime);
    if (rank == 1) std::cout << "\nBuilt mat.\n";
}


/* Second-order Adams-Bashforth explicit scheme */
/* -- verified time ordering of blocks correct */
void AB2(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order)
{
    int tInd0 = rank*timePerProc;
    int tInd1 = tInd0 + timePerProc -1;

    // Get spatial discretization for previous time step, or first step if tInd0=0
    SparseMatrix Ta, Tb;
    Vector Ba, Bb, Xa, Xb;
    if (tInd0 <= 1) {    
        getSpatialDiscretization(Tb, Bb, Xb, 0.0, ref_levels, order);
    }
    else {
        getSpatialDiscretization(Tb, Bb, Xb, dt*(tInd0-2), ref_levels, order);
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int rowsPerTime = Tb.NumRows();
    int nnzPerTime  = Tb.NumNonZeroElems();
    int procNnz     = timePerProc * (2*nnzPerTime + rowsPerTime);     // nnzs on this processor
    int procRow     = timePerProc * rowsPerTime;
    if (tInd0 == 0) procNnz -= 2*nnzPerTime;
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= rowsPerTime;

    int *rowptr  = new int[procRow + 1];
    int *colinds = new int[procNnz];
    double *data = new double[procNnz];
    B.SetSize(procRow);
    X.SetSize(procRow);
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // if (rank == 0) {
    if (rank > -1) {
        std::cout << "Total time steps: " << numTimeSteps << "\nSteps per proc: " << timePerProc
                  << "\nDOFs per time step: " << rowsPerTime << "\nNnz per time: " << nnzPerTime
                  << "\nAllocated nnz: " << procNnz << "\n";
    }

    // Set index to swap pointers to zero if spatial problem is not time dependent. Avoids
    // rebuilding matrix each iteration. 
    int swap_ind;
    if (isTimeDependent) {
        swap_ind = 1;
    }
    else {
        swap_ind = 0;
    }

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        int colPlusOffd_1 = (ti - 1)*rowsPerTime;
        int colPlusOffd_2 = (ti - 2)*rowsPerTime;
        int colPlusDiag = ti*rowsPerTime;

        // Pointers to CSR arrays for A_{ti} and A_{ti-1}
        int *tempRowptr_1;
        int *tempColinds_1;
        double *tempData_1;
        double *Bi_1;
        double *Xi_1;
        int *tempRowptr_2;
        int *tempColinds_2;
        double *tempData_2;
        double *Bi_2;

        // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
        // to build one spatial matrix each iteration. Matrix at ti for previous iteration
        //  is used as ti-1 on this iteration.
        if (ti <= 1) {
            tempRowptr_1  = Tb.GetI();
            tempColinds_1 = Tb.GetJ();
            tempData_1    = Tb.GetData();
            Bi_1          = &Bb[0];
            Xi_1          = &Xb[0];
        }
        else if (swap_ind > 0) {
            getSpatialDiscretization(Ta, Ba, Xa, dt*(ti-1), ref_levels, order); 
            tempRowptr_1  = Ta.GetI();
            tempColinds_1 = Ta.GetJ();
            tempData_1    = Ta.GetData();
            Bi_1          = &Ba[0];
            Xi_1            = &Xa[0];
            tempRowptr_2  = Tb.GetI();
            tempColinds_2 = Tb.GetJ();
            tempData_2    = Tb.GetData();
            Bi_2          = &Bb[0];
        }
        else if (swap_ind < 0) {
            getSpatialDiscretization(Tb, Bb, Xb, dt*(ti-1), ref_levels, order); 
            tempRowptr_1  = Tb.GetI();
            tempColinds_1 = Tb.GetJ();
            tempData_1    = Tb.GetData();
            Bi_1          = &Bb[0];
            Xi_1            = &Xb[0];
            tempRowptr_2  = Ta.GetI();
            tempColinds_2 = Ta.GetJ();
            tempData_2    = Ta.GetData();
            Bi_2          = &Ba[0];
        }
        else {
            tempRowptr_1  = Tb.GetI();
            tempColinds_1 = Tb.GetJ();
            tempData_1    = Tb.GetData();
            Bi_1          = &Bb[0];
            tempRowptr_2  = Tb.GetI();
            tempColinds_2 = Tb.GetJ();
            tempData_2    = Tb.GetData();            
            Bi_2          = &Bb[0];
            Xi_1            = &Xb[0];
        }

        // At time t=0, only have identity block on diagonal
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

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
            for (int i=0; i<rowsPerTime; i++) {

                // Add this row of spatial discretization to first off-diagonal block
                for (int j=tempRowptr_1[i]; j<tempRowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L), otherwise data is dt*L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds_1[j]) {
                        data[dataInd] = -1 + 3.0*dt*tempData_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = 3.0*dt*tempData_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd_1 + tempColinds_1[j];
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
                // and the total nnz in this row of T (tempRowptr[i+1] - tempRowptr[i]).
                rowptr[thisRow+1] = rowptr[thisRow] + 1 + tempRowptr_1[i+1] - tempRowptr_1[i];
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<rowsPerTime; i++) {

                // Add row for spatial discretization of second off-diagonal block 
                for (int j=tempRowptr_2[i]; j<tempRowptr_2[i+1]; j++) {

                    // Add spatial block -dt*Lu_{ti-2}
                    data[dataInd] = -dt*tempData_2[j] / 2.0;
                    colinds[dataInd] = colPlusOffd_2 + tempColinds_2[j];
                    dataInd += 1;
                }

                // Add this row of spatial discretization to first off-diagonal block
                for (int j=tempRowptr_1[i]; j<tempRowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (I + dt*L/2), otherwise data is dt*L/2
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == tempColinds_1[j]) {
                        data[dataInd] = -1 + 3.0*dt*tempData_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = 3.0*dt*tempData_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd_1 + tempColinds_1[j];
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
                rowptr[thisRow+1] = rowptr[thisRow] + (tempRowptr_1[i+1] - tempRowptr_1[i]) +
                                    (tempRowptr_2[i+1] - tempRowptr_2[i]) + 1;
                thisRow += 1;
            }
        }

        // Change sign of pointer index (except for on time t=0 and t=1 iterations)
        if (ti > 1) swap_ind *= -1;

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
        }
        if (thisRow > procRow) {
            std::cout << "WARNING: Matrix has more rows than allocated.\n";
        }
    }

    // Build sparse matrix structure for all rows on this processor
    A = SparseMatrix(rowptr, colinds, data, procRow, numTimeSteps*rowsPerTime);
    if (rank == 1) std::cout << "\nBuilt mat.\n";
}
