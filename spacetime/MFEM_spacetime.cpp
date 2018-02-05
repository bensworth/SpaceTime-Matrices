// #include "mpi.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "mfem.hpp"

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
void AM3(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);
void AB1(const int &rank, const int &timePerProc, const int &numTimeSteps,
          const double &dt, const bool &isTimeDependent, SparseMatrix &A,
          Vector &B, Vector &X, const int &ref_levels, const int &order);



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
    bool isTimeDependent = false;
    int numTimeSteps = 4;
    int ref_levels = 1;
    int order = 1;
    int bdf = -1;
    int am = -1;
    int ab = -1;
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
        else if(strcmp(argv[i],"-bdf") == 0) {
            i += 1;
            bdf = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-am") == 0) {
            i += 1;
            am = atoi(argv[i]);
        }
        else if(strcmp(argv[i],"-ab") == 0) {
            i += 1;
            ab = atoi(argv[i]);
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
    else if ((am > 0) && (bdf > 0)) {
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
    A.Finalize(1);

    // Save to file by MPI rank
    std::stringstream filename;
    filename << "test_mat_" << rank << ".mm";
    std::ofstream outfile(filename.str()); 
    A.PrintMM(outfile);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}


/* Time-independent spatial discretization of Laplacian */
void getSpatialDiscretization(SparseMatrix &A, Vector &B, Vector &X, double t,
                              int ref_levels, int order)
{
    // Read mesh from mesh file
    const char *mesh_file = "./meshes/beam-quad.mesh";
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
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B, 1);
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
    int procNnz     = timePerProc * (nnzPerTime + rowsPerTime);     // nnzs on this processor for BDF1
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

        // At time t=0, only have spatial discretization block; need to move initial
        // conditions to RHS
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
    int procNnz     = timePerProc * (nnzPerTime + 2*rowsPerTime);     // nnzs on this processor for BDF2 
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

        // At time t=0, only have spatial discretization block; need to move initial
        // conditions to RHS
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
        // At time t=1, only have 1 off-diagonal block and need to move initial
        // conditions to RHS for 2nd-order accuracy
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
    int procNnz     = timePerProc * (nnzPerTime + 3*rowsPerTime);     // nnzs on this processor for BDF2 
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

        // At time t=0, only have spatial discretization block; need to move initial
        // conditions to RHS
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
        // At time t=1, only have 1 off-diagonal block and need to move initial
        // conditions to RHS for 2nd-order accuracy
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
    int procNnz     = 2 * timePerProc * nnzPerTime;     // nnzs on this processor for BDF1
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

        // Change sign of pointer index
        swap_ind *= -1;

        // Check if sufficient data was allocated
        if (dataInd > procNnz) {
            std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
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
    int procNnz     = timePerProc * (nnzPerTime + rowsPerTime);     // nnzs on this processor for BDF1
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

        // At time t=0, only have spatial discretization block; need to move initial
        // conditions to RHS
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
