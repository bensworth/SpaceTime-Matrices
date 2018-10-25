#include <iostream>
#include <fstream>
#include <map>
#include "SpaceTimeMatrix.hpp"

// TODO:
//      - Add scaling by mass inverse in MySpaceTime.cpp. **Include dt -> (M / dt)^{-1}.
//          Then don't scale by dt in SpaceTimeMatrix.cpp
//      - Add function or capability to eliminate zeros from spatial
//        discretization.
//      - Add isTimeDependent option
//      - Finish AB2 for spatial parallel
//      - Add hypre timing for setup and solve (see ij.c)
//      - Something wrong with GMRES
//            srun -N 4 -n 64 -p pdebug driver -nt 512 -t 12 -o 2 -l 4 -Af 0 -Ai 100 -AIR 2 -gmres 1
//        doesn't converge, but does if you turn off gmres...



SpaceTimeMatrix::SpaceTimeMatrix(MPI_Comm globComm, int timeDisc,
                                 int numTimeSteps, double t0, double t1)
    : m_globComm{globComm}, m_timeDisc{timeDisc}, m_numTimeSteps{numTimeSteps},
      m_t0{t0}, m_t1{t1}, m_solver(NULL), m_gmres(NULL), m_bij(NULL), m_xij(NULL), 
      m_rebuildSolver(false)
{
    // Set member variables
    m_dt = (m_t1-m_t0) / m_numTimeSteps;

    // Get number of processes
    MPI_Comm_rank(m_globComm, &m_globRank);
    MPI_Comm_size(m_globComm, &m_numProc);

    // Check that number of time steps divides the number MPI processes or vice versa.
    if (m_numTimeSteps <= m_numProc) {
        m_useSpatialParallel = true;
        if (m_numProc % m_numTimeSteps != 0) {
            if (m_globRank == 0) {
                std::cout << "Error: number of time steps does not divide number of processes.\n";
            }
            MPI_Finalize();
            return;
        }
        else {
            m_Np_x = m_numProc / m_numTimeSteps;
        }

        // Set up communication group for spatial discretizations.
        m_timeInd = m_globRank / m_Np_x;
        MPI_Comm_split(m_globComm, m_timeInd, m_globRank, &m_spatialComm);
        MPI_Comm_rank(m_spatialComm, &m_spatialRank);
        MPI_Comm_size(m_spatialComm, &m_spCommSize);
    }
    else {
        m_useSpatialParallel = false;
        if (m_numTimeSteps % m_numProc  != 0) {
            if (m_globRank == 0) {
                std::cout << "Error: number of processes does not divide number of time steps.\n";
            }
            MPI_Finalize();
            return;
        }
        // Time steps computed per processor. 
        m_ntPerProc = m_numTimeSteps / m_numProc;
    }
}


SpaceTimeMatrix::~SpaceTimeMatrix()
{
    if (m_solver) HYPRE_BoomerAMGDestroy(m_solver);
    if (m_gmres) HYPRE_ParCSRGMRESDestroy(m_gmres);
    if (m_Aij) HYPRE_IJMatrixDestroy(m_Aij);   // This destroys parCSR matrix too
    if (m_bij) HYPRE_IJVectorDestroy(m_bij);   // This destroys parVector too
    if (m_xij) HYPRE_IJVectorDestroy(m_xij);
}


void SpaceTimeMatrix::BuildMatrix()
{
    
    int* T_rowptr = NULL;
    int* T_colinds = NULL;
    double* T_data = NULL;
    double* B0 = NULL;
    double* X0 = NULL;
    int * cols_per_row_T = NULL;
    int spatialDOFs;
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    
    // // Calling getMassMatrix here doesn't yield weird results, except I can't free the mass pointers...
    // getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0,
    //                                     X0, spatialDOFs, 0.0, cols_per_row_T);
    // 
    // getMassMatrix(M_rowptr, M_colinds, M_data);
    // std::cout << "----------- spatialDOFs = " << spatialDOFs << '\n';
    // std::cout << "----------- nnz(L) = " << T_rowptr[spatialDOFs] << '\n';
    // std::cout << "----------- nnz(M) = " << M_rowptr[spatialDOFs] << '\n';
    // // I can free these...
    // delete[] T_rowptr;
    // delete[] T_colinds;
    // delete[] T_data;
    // // But when I try to free these pointers I get null pointer....
    // // delete[] M_rowptr;
    // // delete[] M_colinds;
    // // delete[] M_data;
    
    if (m_globRank == 0) std::cout << "Building matrix.\n";
    if (m_useSpatialParallel) GetMatrix_ntLE1();
    else GetMatrix_ntGT1();
    
    // Now when I use getMassMatrix down here, I get weird behaviour..., 
    // e.g. nnz(M) is trying to access something it shouldn't be...
    // but some of the time it's right...
    getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0,
                                        X0, spatialDOFs, 0.0, cols_per_row_T);   
    getMassMatrix(M_rowptr, M_colinds, M_data);
    std::cout << "zzzzzzzzzzz spatialDOFs = " << spatialDOFs << '\n';
    std::cout << "zzzzzzzzzzz nnz(L) = " << T_rowptr[spatialDOFs] << '\n';
    std::cout << "zzzzzzzzzzz nnz(M) = " << M_rowptr[spatialDOFs] << '\n';
    // Can't free these arrays
    // delete[] M_rowptr;
    // delete[] M_colinds;
    // delete[] M_data;
    
    if (m_globRank == 0) std::cout << "Space-time matrix assembled.\n";
    
    // Update global RHS vector
    // TODO : is this the right way to just have a single processor execute this update RHS code?
    if (m_globComm == 0) {
        updateMultiRHS_ntGT1(MPI_COMM_SELF);
    }
    
}


/* Get space-time matrix for at most 1 time step per processor */
void SpaceTimeMatrix::GetMatrix_ntLE1()
{
    // Get local CSR structure
    int* rowptr;
    int* colinds;
    double* data;
    double* B;
    double* X;
    int localMinRow;
    int localMaxRow;
    int spatialDOFs;
    if (m_timeDisc == 11) {
        BDF1(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
    else if (m_timeDisc == 12) {
        BDF2(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
    else if (m_timeDisc == 13) {
        BDF3(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
    else if (m_timeDisc == 21) {
        BDF1(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
    else if (m_timeDisc == 22) {
        AM2(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
    else if (m_timeDisc == 31) {
        AB1(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
    // else if (m_timeDisc == 32) {
    //     // AB2(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    //     // TODO: implement
    // }
    else {
        std::cout << "WARNING: invalid choice of time integration.\n";
        MPI_Finalize();
        return;
    }

    // Initialize matrix
    int onProcSize = localMaxRow - localMinRow + 1;
    int ilower = m_timeInd*spatialDOFs + localMinRow;
    int iupper = m_timeInd*spatialDOFs + localMaxRow;
    HYPRE_IJMatrixCreate(m_globComm, ilower, iupper, ilower, iupper, &m_Aij);
    HYPRE_IJMatrixSetObjectType(m_Aij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(m_Aij);

    // Set matrix coefficients
    int* rows = new int[onProcSize];
    int* cols_per_row = new int[onProcSize];
    for (int i=0; i<onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row[i] = rowptr[i+1] - rowptr[i];
    }
    HYPRE_IJMatrixSetValues(m_Aij, onProcSize, cols_per_row, rows, colinds, data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(m_Aij);
    HYPRE_IJMatrixGetObject(m_Aij, (void **) &m_A);

    /* Create rhs and solution vectors */
    HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &m_bij);
    HYPRE_IJVectorSetObjectType(m_bij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(m_bij);
    HYPRE_IJVectorSetValues(m_bij, onProcSize, rows, B);
    HYPRE_IJVectorAssemble(m_bij);
    HYPRE_IJVectorGetObject(m_bij, (void **) &m_b);

    HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &m_xij);
    HYPRE_IJVectorSetObjectType(m_xij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(m_xij);
    HYPRE_IJVectorSetValues(m_xij, onProcSize, rows, X);
    HYPRE_IJVectorAssemble(m_xij);
    HYPRE_IJVectorGetObject(m_xij, (void **) &m_x);

    // Remove pointers that should have been copied by Hypre
    delete[] rowptr;
    delete[] colinds;
    delete[] data;
    delete[] B;
    delete[] X;
    delete[] rows;
    delete[] cols_per_row;
}


/* Get space-time matrix for more than 1 time step per processor */
void SpaceTimeMatrix::GetMatrix_ntGT1()
{
    // Get local CSR structure
    int* rowptr;
    int* colinds;
    double* data;
    double* B;
    double* X;
    int onProcSize;
    if (m_timeDisc == 11) {
        BDF1(rowptr, colinds, data, B, X, onProcSize);
    }
    else if (m_timeDisc == 12) {
        BDF2(rowptr, colinds, data, B, X, onProcSize);
    }
    else if (m_timeDisc == 13) {
        BDF3(rowptr, colinds, data, B, X, onProcSize);
    }
    else if (m_timeDisc == 21) {
        BDF1(rowptr, colinds, data, B, X, onProcSize);
    }
    else if (m_timeDisc == 22) {
        AM2(rowptr, colinds, data, B, X, onProcSize);
    }
    else if (m_timeDisc == 31) {
        AB1(rowptr, colinds, data, B, X, onProcSize);
    }
    else if (m_timeDisc == 32) {
        AB2(rowptr, colinds, data, B, X, onProcSize);
    }
    else {
        std::cout << "WARNING: invalid choice of time integration.\n";
        MPI_Finalize();
        return;
    }

    // Initialize matrix
    int ilower = m_globRank*onProcSize;
    int iupper = (m_globRank+1)*onProcSize - 1;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &m_Aij);
    HYPRE_IJMatrixSetObjectType(m_Aij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(m_Aij);

    // Set matrix coefficients
    int* rows = new int[onProcSize];
    int* cols_per_row = new int[onProcSize];
    for (int i=0; i<onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row[i] = rowptr[i+1] - rowptr[i];
    }
    HYPRE_IJMatrixSetValues(m_Aij, onProcSize, cols_per_row, rows, colinds, data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(m_Aij);
    HYPRE_IJMatrixGetObject(m_Aij, (void **) &m_A);

    /* Create sample rhs and solution vectors */
    HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &m_bij);
    HYPRE_IJVectorSetObjectType(m_bij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(m_bij);
    HYPRE_IJVectorSetValues(m_bij, onProcSize, rows, B);
    HYPRE_IJVectorAssemble(m_bij);
    HYPRE_IJVectorGetObject(m_bij, (void **) &m_b);

    HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &m_xij);
    HYPRE_IJVectorSetObjectType(m_xij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(m_xij);
    HYPRE_IJVectorSetValues(m_xij, onProcSize, rows, X);
    HYPRE_IJVectorAssemble(m_xij);
    HYPRE_IJVectorGetObject(m_xij, (void **) &m_x);

    // Remove pointers that should have been copied by Hypre
    delete[] rowptr;
    delete[] colinds;
    delete[] data;
    delete[] B;
    delete[] X;
    delete[] rows;
    delete[] cols_per_row;
}


/* Set classical AMG parameters for BoomerAMG solve. */
void SpaceTimeMatrix::SetAMG()
{
   m_solverOptions.prerelax = "AA";
   m_solverOptions.postrelax = "AA";
   m_solverOptions.relax_type = 5;
   m_solverOptions.interp_type = 6;
   m_solverOptions.strength_tolC = 0.1;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = -1;
   m_solverOptions.strength_tolR = -1;
   m_solverOptions.filterA_tol = 0.0;
   m_rebuildSolver = true;
}


/* Set standard AIR parameters for BoomerAMG solve. */
void SpaceTimeMatrix::SetAIR()
{
   m_solverOptions.prerelax = "FFFC";
   m_solverOptions.postrelax = "A";
   m_solverOptions.relax_type = 3;
   m_solverOptions.interp_type = 100;
   m_solverOptions.strength_tolC = 0.005;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = 2;
   m_solverOptions.strength_tolR = 0.005;
   m_solverOptions.filterA_tol = 0.0001;
   m_rebuildSolver = true;
}


/* Set AIR parameters assuming triangular matrix in BoomerAMG solve. */
void SpaceTimeMatrix::SetAIRHyperbolic()
{
   m_solverOptions.prerelax = "F";
   m_solverOptions.postrelax = "A";
   m_solverOptions.relax_type = 10;
   m_solverOptions.interp_type = 100;
   m_solverOptions.strength_tolC = 0.005;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = 2;
   m_solverOptions.strength_tolR = 0.005;
   m_solverOptions.filterA_tol = 0.0001;
   m_rebuildSolver = true;
}


/* Provide BoomerAMG parameters struct for solve. */
void SpaceTimeMatrix::SetAMGParameters(AMG_parameters &params)
{
    // TODO: does this copy the structure by value?
    m_solverOptions = params;
}


/* Initialize AMG solver based on parameters in m_solverOptions struct. */
void SpaceTimeMatrix::SetupBoomerAMG(int printLevel, int maxiter, double tol)
{
    // If solver exists and rebuild bool is false, return
    if (m_solver && !m_rebuildSolver){
        return;
    }
    // Build/rebuild solver
    else {
        if (m_solver) {
            std::cout << "Rebuilding solver.\n";
            HYPRE_BoomerAMGDestroy(m_solver);
        }

        // Array to store relaxation scheme and pass to Hypre
        //      TODO: does hypre clean up grid_relax_points
        int ns_down = m_solverOptions.prerelax.length();
        int ns_up = m_solverOptions.postrelax.length();
        int ns_coarse = 1;
        std::string Fr("F");
        std::string Cr("C");
        std::string Ar("A");
        int* *grid_relax_points = new int* [4];
        grid_relax_points[0] = NULL;
        grid_relax_points[1] = new int[ns_down];
        grid_relax_points[2] = new int [ns_up];
        grid_relax_points[3] = new int[1];
        grid_relax_points[3][0] = 0;

        // set down relax scheme 
        for(unsigned int i = 0; i<ns_down; i++) {
            if (m_solverOptions.prerelax.compare(i,1,Fr) == 0) {
                grid_relax_points[1][i] = -1;
            }
            else if (m_solverOptions.prerelax.compare(i,1,Cr) == 0) {
                grid_relax_points[1][i] = 1;
            }
            else if (m_solverOptions.prerelax.compare(i,1,Ar) == 0) {
                grid_relax_points[1][i] = 0;
            }
        }

        // set up relax scheme 
        for(unsigned int i = 0; i<ns_up; i++) {
            if (m_solverOptions.postrelax.compare(i,1,Fr) == 0) {
                grid_relax_points[2][i] = -1;
            }
            else if (m_solverOptions.postrelax.compare(i,1,Cr) == 0) {
                grid_relax_points[2][i] = 1;
            }
            else if (m_solverOptions.postrelax.compare(i,1,Ar) == 0) {
                grid_relax_points[2][i] = 0;
            }
        }

        // Create preconditioner
        HYPRE_BoomerAMGCreate(&m_solver);
        HYPRE_BoomerAMGSetTol(m_solver, tol);    
        HYPRE_BoomerAMGSetMaxIter(m_solver, maxiter);
        HYPRE_BoomerAMGSetPrintLevel(m_solver, printLevel);

        if (m_solverOptions.distance_R > 0) {
            HYPRE_BoomerAMGSetRestriction(m_solver, m_solverOptions.distance_R);
        }
        HYPRE_BoomerAMGSetInterpType(m_solver, m_solverOptions.interp_type);
        HYPRE_BoomerAMGSetCoarsenType(m_solver, m_solverOptions.coarsen_type);
        HYPRE_BoomerAMGSetAggNumLevels(m_solver, 0);
        HYPRE_BoomerAMGSetStrongThreshold(m_solver, m_solverOptions.strength_tolC);
        HYPRE_BoomerAMGSetStrongThresholdR(m_solver, m_solverOptions.strength_tolR);
        HYPRE_BoomerAMGSetGridRelaxPoints(m_solver, grid_relax_points);
        if (m_solverOptions.relax_type > -1) {
            HYPRE_BoomerAMGSetRelaxType(m_solver, m_solverOptions.relax_type);
        }
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_coarse, 3);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_down,   1);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_up,     2);
        if (m_solverOptions.filterA_tol > 0) {
            HYPRE_BoomerAMGSetADropTol(m_solver, m_solverOptions.filterA_tol);
        }
        // type = -1: drop based on row inf-norm
        else if (m_solverOptions.filterA_tol == -1) {
            HYPRE_BoomerAMGSetADropType(m_solver, -1);
        }

        // Do not rebuild solver unless parameters are changed.
        m_rebuildSolver = false;
    }
}


void SpaceTimeMatrix::SolveAMG(double tol, int maxiter, int printLevel)
{
    SetupBoomerAMG(printLevel, maxiter, tol);
    HYPRE_BoomerAMGSetup(m_solver, m_A, m_b, m_x);
    HYPRE_BoomerAMGSolve(m_solver, m_A, m_b, m_x);
}


void SpaceTimeMatrix::SolveGMRES(double tol, int maxiter, int printLevel, int precondition) 
{
    HYPRE_ParCSRGMRESCreate(m_globComm, &m_gmres);
    HYPRE_GMRESSetMaxIter(m_gmres, maxiter);
    HYPRE_GMRESSetTol(m_gmres, tol);
    HYPRE_GMRESSetPrintLevel(m_gmres, printLevel);
    
    // AMG preconditioning (setup boomerAMG with 1 max iter and print level 1)
    if (precondition == 1) {
        SetupBoomerAMG(1, 1, tol);
        HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, m_solver);
    }
    // Diagonally scaled preconditioning?
    else if (precondition == 2) {
        int temp = -1;
    }
    // TODO: Implement block-diagonal AMG preconditioning, with BoomerAMG on each time block?
    else if (precondition == 3) {
        int temp = -1;
    }

    HYPRE_GMRESSetup(m_gmres, (HYPRE_Matrix)m_A, (HYPRE_Vector)m_b, (HYPRE_Vector)m_x);
    HYPRE_GMRESSolve(m_gmres, (HYPRE_Matrix)m_A, (HYPRE_Vector)m_b, (HYPRE_Vector)m_x);
}


/* ------------------------------------------------------------------------- */
/* ----------------------- Multistep initialization ------------------------ */
/* ------------------------------------------------------------------------- */

/* Update RHS when using more than 1 time step per processor */
/* Even though this is only done on one processor, we need a communicator to 
use HYPRE on. So just create communicator with a single proc. */

/* TODO : 
    - Need to get initial condition u0, where is this?
    - how to get spatialDOFs without calling the spatial discretization?
    - is spatialDOFs always time independent?
*/

// Helper to get spatial discretization to save retyping these things all the time
void SpaceTimeMatrix::getSpatialDiscretization_helper(int *&T_rowptr, int *&T_colinds, double *&T_data, 
                                                        double *&B0, double *&X0, int &spatialDOFs, double t, 
                                                        int *&cols_per_row_T) {
    
    // If these components are previously allocated, clear them first
    // Note this requires that on the first pass the be explicitly set to NULL.
    if (T_rowptr) {
        delete[] T_rowptr;
        delete[] T_colinds;
        delete[] T_data;
        delete[] B0;
        delete[] X0;
        delete[] cols_per_row_T;
    }
    
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, t);
    
    int onProcSize = spatialDOFs;
    cols_per_row_T = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        cols_per_row_T[i] = T_rowptr[i+1] - T_rowptr[i];
    }
}

void SpaceTimeMatrix::updateMultiRHS_ntGT1(MPI_Comm comm) {
    
    // int* T_rowptr = NULL;
    // int* T_colinds = NULL;
    // double* T_data = NULL;
    // double* B0 = NULL;
    // double* X0 = NULL;
    // int * cols_per_row_T = NULL;
    // int spatialDOFs = 18;
    // getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0,
    //                                     X0, spatialDOFs, 0.0, cols_per_row_T);
    // 
    // std::cout << "................ENTERED update RHS..............\n";
    // int rank, size;
    // MPI_Comm_rank(comm, &rank);
    // MPI_Comm_size(comm, &size);
    // std::cout << "I am process " << rank << " out of " << size << " processes\n";
    // 
    // 
    // RK_butcher butch;
    // 
    // // TODO : how do we obtain this number without calling spatial disc???
    // 
    // int onProcSize = spatialDOFs; // TODO : is this and the things below it correct?
    // int ilower = 0;
    // int iupper = onProcSize-1;
    // 
    // // Get mass matrix 
    // int* M_rowptr;
    // int* M_colinds;
    // double* M_data;
    // 
    // getMassMatrix(M_rowptr, M_colinds, M_data);
    // std::cout << "spatialDOFs = " << spatialDOFs << '\n';
    // std::cout << "nnz(M) = " << M_rowptr[spatialDOFs] << '\n';
    // 
    // 
    // double * M_data_scaled = new double[M_rowptr[spatialDOFs]];
    // for (int i = 0; i < M_rowptr[spatialDOFs]; i++) {
    //     M_data_scaled[i] = M_data[i] / m_dt; // Scale data by 1/dt
    //     std::cout << "M_data[%d] = "<< i << "%.2e" << M_data_scaled[i] << '\n';
    // }
    // 
    // // Process M so that it can be added to HYPRE matrices later 
    // int * rows = new int[onProcSize]; // Reuse this many times
    // int * cols_per_row_M = new int[onProcSize];
    // for (int i = 0; i < onProcSize; i++) {
    //     rows[i] = ilower + i;
    //     cols_per_row_M[i] = M_rowptr[i+1] - M_rowptr[i];
    //     std::cout << "rows[%d] = "<< i << rows[i] << "cols_per_row = " << cols_per_row_M[i] << '\n';
    // }
    // 
    // std::cout << "passed 0" << '\n';
    // 
    // // // 
    // // // 
    // // // // Initialize solution vectors to be computed
    // HYPRE_IJVector temp; // Need 1 dummy vector as a place holder
    // HYPRE_ParVector par_temp;
    // HYPRE_IJVector u0;
    // HYPRE_ParVector par_u0;
    // HYPRE_IJVector u1;
    // HYPRE_ParVector par_u1;
    // HYPRE_IJVector u2;
    // HYPRE_ParVector par_u2;
    // HYPRE_IJVectorCreate(comm, ilower, iupper, &temp);
    // HYPRE_IJVectorSetObjectType(temp, HYPRE_PARCSR);
    // HYPRE_IJVectorInitialize(temp);
    // HYPRE_IJVectorAssemble(temp);
    // HYPRE_IJVectorGetObject(temp, (void **) &par_temp);
    // std::cout << "passed 1" << '\n';
    // HYPRE_IJVectorCreate(comm, ilower, iupper, &u0); 
    // HYPRE_IJVectorSetObjectType(u0, HYPRE_PARCSR);
    // HYPRE_IJVectorInitialize(u0);
    // HYPRE_IJVectorAssemble(u0);
    // HYPRE_IJVectorGetObject(u0, (void **) &par_u0);
    // std::cout << "passed 2" << '\n';
    // HYPRE_IJVectorCreate(comm, ilower, iupper, &u1);
    // HYPRE_IJVectorSetObjectType(u1, HYPRE_PARCSR);
    // HYPRE_IJVectorInitialize(u1);
    // HYPRE_IJVectorAssemble(u1);
    // HYPRE_IJVectorGetObject(u1, (void **) &par_u1);
    // std::cout << "passed 3" << '\n';
    // HYPRE_IJVectorCreate(comm, ilower, iupper, &u2); 
    // HYPRE_IJVectorSetObjectType(u2, HYPRE_PARCSR);
    // HYPRE_IJVectorInitialize(u2);
    // HYPRE_IJVectorAssemble(u2);
    // HYPRE_IJVectorGetObject(u2, (void **) &par_u2);
    // std::cout << "passed 4" << '\n';
    // // // 
    // // // Create HYPRE version of mass matrix to perform MATVECs
    // HYPRE_IJMatrix M_scaled;
    // HYPRE_ParCSRMatrix parcsr_M_scaled;
    // HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &M_scaled); // TODO : is this spatialDOFs or spatialDOFs-1??
    // HYPRE_IJMatrixSetObjectType(M_scaled, HYPRE_PARCSR);
    // HYPRE_IJMatrixInitialize(M_scaled);
    // 
    // HYPRE_IJMatrixSetValues(M_scaled, onProcSize, cols_per_row_M, rows, M_colinds, M_data_scaled); // M_scaled <- M/dt
    // 
    // std::cout << "passed 4.5" << '\n';
    // 
    // HYPRE_IJMatrixAssemble(M_scaled);
    // HYPRE_IJMatrixGetObject(M_scaled, (void **) &parcsr_M_scaled);
    // std::cout << "passed 5" << '\n';
    
    // // // // Might need to build spatial discretization, so make appropriate declarations 
    // HYPRE_IJMatrix T;
    // HYPRE_ParCSRMatrix parcsr_T;
    // int built_spatial_disc = 0;
    // // int* T_rowptr;
    // // int* T_colinds;
    // // double* T_data;
    // // double* B0;
    // // double* X0;
    // // int * cols_per_row_T;
    // 
    // // All of these discretizations require L0, so build it once only
    // if (m_timeDisc == 22 || m_timeDisc == 23 || m_timeDisc == 31 || m_timeDisc == 32 || m_timeDisc == 33) {
    //     built_spatial_disc = 1; // Set flag
    //     getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0,
    //                                         X0, spatialDOFs, 0.0, cols_per_row_T);
    // 
    //     // Put spatial disc into HYPRE matrix
    //     HYPRE_IJMatrixCreate(comm, 0, spatialDOFs-1, 0, spatialDOFs-1, &T); // TODO : is this spatialDOFs or spatialDOFs-1??
    //     HYPRE_IJMatrixSetObjectType(T, HYPRE_PARCSR);
    //     HYPRE_IJMatrixInitialize(T);
    //     HYPRE_IJMatrixSetValues(T, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
    //     HYPRE_IJMatrixAssemble(T);
    //     HYPRE_IJMatrixGetObject(T, (void **) &parcsr_T);
    // }
    // 
    // 
    // 
    // // BDF1 or AM1 (they're the same). No need to use RK.
    // if (m_timeDisc == 11 || m_timeDisc == 21) {
    //     //HYPRE_ParCSRMatrixMatvec(1.0, parcsr_M_scaled, par_u0, 0.0, par_temp); // temp <- 1/(dt)*M*u0
    //     // TODO: update first block of global vector: b1 <- b1 + temp
    // 
    // // BDF2
    // } else if (m_timeDisc == 12) {
    //     getButcher(butch, 22);
    //     DIRK(comm, butch, &par_u1, &par_u0, 0.0, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u1 from u0
    // 
    //     // Vector for updating first block of global RHS vector
    //     HYPRE_ParVectorCopy(par_u0, par_temp); // temp <- u0
    //     HYPRE_ParVectorScale(-1.0, par_temp); // temp <- -u0
    //     HYPRE_ParVectorAxpy(4.0, par_u2, par_temp); // temp <- 4*u1 - u0
    //     HYPRE_ParCSRMatrixMatvec(1.0/3.0, parcsr_M_scaled, par_temp, 0.0, par_u0); // u0 <- 1/(3dt)*M*temp
    //     // TODO: update first block of global vector: b2 <- b2 + u0
    // 
    //     // Vector for updating second block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(-1.0/3.0, parcsr_M_scaled, par_u1, 0.0, par_temp); // temp <- -1/(3dt)*M*u1
    //     // TODO: update second block of global vector: b3 <- b3 + temp
    // }
    // 
    // // BDF3
    // else if (m_timeDisc == 13) {
    //     getButcher(butch, 23);
    //     DIRK(comm, butch, &par_u1, &par_u0, 0.0, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u1 from u0
    //     DIRK(comm, butch, &par_u2, &par_u1, 0.0 + m_dt, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u2 from u1
    // 
    //     // Vector for updating first block of global RHS vector
    //     HYPRE_ParVectorCopy(par_u0, par_temp); // temp <- u0
    //     HYPRE_ParVectorScale(2.0, par_temp); // temp <- 2*u0
    //     HYPRE_ParVectorAxpy(-9.0, par_u1, par_temp); // temp <- -9*u1 + 2*u0
    //     HYPRE_ParVectorAxpy(18.0, par_u2, par_temp); // temp <- 18*u2 - 9*u1 + 2*u0
    //     HYPRE_ParCSRMatrixMatvec(1.0/11.0, parcsr_M_scaled, par_temp, 0.0, par_u0); // u0 <- 1/(11dt)*M*temp
    //     // TODO: update first block of global vector: b3 <- b3 + u0
    // 
    //     // Vector for updating second block of global RHS vector
    //     HYPRE_ParVectorCopy(par_u1, par_temp); // temp <- u1
    //     HYPRE_ParVectorScale(2.0, par_temp); // temp <- 2*u1
    //     HYPRE_ParVectorAxpy(-9.0, par_u2, par_temp); // temp <- -9*u2 + 2*u1
    //     HYPRE_ParCSRMatrixMatvec(1.0/11.0, parcsr_M_scaled, par_temp, 0.0, par_u1); // u1 <- 1/(11dt)*M*temp
    //     // TODO: update second block of global vector: b4 <- b4 + u1
    // 
    //     // Vector for updating third block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(2.0/11.0, parcsr_M_scaled, par_u2, 0.0, par_temp); // temp <- 2/(11dt)*M*u2
    //     // TODO: update third block of global vector: b5 <- b5 + temp
    // 
    // 
    // // AM2. No need to use RK.
    // } else if (m_timeDisc == 22) {
    //     // Vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_M_scaled, par_u0, 0.0, par_temp); // temp <- 1/(dt)*M*u0
    //     HYPRE_ParCSRMatrixMatvec(-0.5, parcsr_T, par_u0, 0.0, par_temp); // temp <- -1/2*L0*u0 + 1/(dt)*M*u0
    //     // TODO: update first block of global vector: b1 <- b1 + temp
    // 
    // // AM3
    // } else if (m_timeDisc == 23) {
    //     getButcher(butch, 13);  
    //     ERK(comm, butch, &par_u1, &par_u0, 0.0, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u1 from u0      
    // 
    //     // Create vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(1.0/12.0, parcsr_T, par_u0, 0.0, par_temp); // temp <- 1/12*L0*u0
    // 
    //     // Rebuild L at t = dt if it's time dependent: L1
    //     if (m_isTimeDependent) {
    //         getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0,
    //                                             X0, spatialDOFs, m_dt, cols_per_row_T);
    //         HYPRE_IJMatrixSetValues(T, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
    //     } 
    // 
    //     // Update vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(-2.0/3.0, parcsr_T, par_u1, 0.0, par_temp); // temp <- -2/3*L1*u1 + 1/12*L0*u0
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_M_scaled, par_u1, 0.0, par_temp); // temp <- 1/dt*M*u1 -2/3*L1*u1 + 1/12*L0*u0
    //     // TODO: update first block of global vector: b2 <- b2 + temp
    // 
    // 
    // // AB1. No need to use RK
    // } else if (m_timeDisc == 31) {
    //     // Vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_M_scaled, par_u0, 0.0, par_temp); // temp <- 1/(dt)*M*u0
    //     HYPRE_ParCSRMatrixMatvec(-1.0, parcsr_T, par_u0, 0.0, par_temp); // temp <- -L0*u0 + 1/(dt)*M*u0
    //     // TODO: update first block of global vector: b1 <- b1 + temp    
    // 
    // // AB2
    // } else if (m_timeDisc == 32) {
    //     getButcher(butch, 12);
    //     ERK(comm, butch, &par_u1, &par_u0, 0.0, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u1 from u0
    // 
    //     // Create vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(0.5, parcsr_T, par_u0, 0.0, par_temp); // temp <- 1/2*L0*u0
    // 
    //     // Rebuild L at t = dt if it's time dependent: L1
    //     if (m_isTimeDependent) {
    //         getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, m_dt, cols_per_row_T);
    //         HYPRE_IJMatrixSetValues(T, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
    //     }
    // 
    //     // Create vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(-3.0/2.0, parcsr_T, par_u1, 1.0, par_temp); // temp <- -3/2*L1*u1 + 1/2*L0*u0
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_M_scaled, par_u1, 1.0, par_u0); // u0 <- 1/(dt)*M*u1 -3/2*L1*u1 + 1/2*L0*u0
    //     // TODO: b2 <- b2 + u0
    // 
    //     // Create vector for updating second block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(0.5, parcsr_T, par_u1, 0.0, par_temp); // temp <- 1/2*L1*u1
    //     // TODO: b3 <- b3 + temp
    // 
    // // AB3
    // } else if (m_timeDisc == 33) {
    //     getButcher(butch, 13);
    //     ERK(comm, butch, &par_u1, &par_u0, 0.0, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u1 from u0
    //     ERK(comm, butch, &par_u2, &par_u1, 0.0 + m_dt, m_dt, 0, spatialDOFs-1, M_rowptr, M_colinds, M_data); // Compute u2 from u1
    // 
    //     // Create vector for updating first block of global RHS vector
    //     HYPRE_ParCSRMatrixMatvec(0.5, parcsr_T, par_u0, 0.0, par_temp); // temp <- -5/12*L0*u0
    //     HYPRE_ParVectorCopy(par_temp, par_u0); // u0 <- -5/12*L0*u0
    // 
    //     // Rebuild L at t = dt if it's time dependent: L1
    //     if (m_isTimeDependent) {
    //         getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, m_dt, cols_per_row_T);
    //         HYPRE_IJMatrixSetValues(T, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
    //     }
    // 
    //     // Append to vector for updating first block of global vector
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_T, par_u1, 0.0, par_temp); // temp <- L1*u1
    //     HYPRE_ParVectorAxpy(4.0/3.0, par_temp, par_u0); // u0 <- 4/3*L1*u1 - 5/12*L0*u0
    // 
    //     // Create vector for updating second block of global RHS vector
    //     HYPRE_ParVectorCopy(par_temp, par_u1); // u1 <- L1*u1
    //     HYPRE_ParVectorScale(-5.0/12.0 , par_u1); // u1 <- -5/12*L1*u1
    // 
    //     // Rebuild L at t = 2*dt if it's time dependent: L2
    //     if (m_isTimeDependent) {
    //         getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, 2*m_dt, cols_per_row_T);
    //         HYPRE_IJMatrixSetValues(T, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
    //     }
    // 
    //     // Do matvec L2*u2
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_T, par_u2, 0.0, par_temp); // temp <- L2*u2
    // 
    //     // Append to vector for updating first block of global vector
    //     HYPRE_ParVectorAxpy(-23.0/12.0, par_temp, par_u0); // u0 <- -23/12*L2*u2 + 4/3*L1*u1 - 5/12*L0*u0
    //     HYPRE_ParCSRMatrixMatvec(1.0, parcsr_M_scaled, par_u2, 1.0, par_u0); // u0 <- 1/(dt)*M*u2 -23/12*L2*u2 + 4/3*L1*u1 - 5/12*L0*u0
    //     // TODO: b3 <- b3 + u0
    // 
    //     // Append to vector for updating second block of global RHS vector
    //     HYPRE_ParVectorAxpy(4.0/3.0, par_temp, par_u1); // u1 <- 4.0/3.0*L2*u2 -5/12*L1*u1
    //     // TODO: b4 <- b4 + u1
    // 
    //     // Create vector for updating third block of global RHS vector
    //     HYPRE_ParVectorScale(-5.0/12.0 , par_temp); // temp <- -5/12*L2*u2
    //     // TODO: b5 <- b5 + temp
    // }
    // 
    // // Clean up
    // if (built_spatial_disc) {
    //     HYPRE_IJMatrixDestroy(T);
    //     delete[] T_rowptr;
    //     delete[] T_colinds;
    //     delete[] T_data;
    //     delete[] B0;
    //     delete[] X0;
    //     delete[] cols_per_row_T;
    // }
    // 
    // delete[] M_rowptr;
    // delete[] M_colinds;
    // delete[] M_data;
    // delete[] M_data_scaled;
    // delete[] rows;
    // delete[] cols_per_row_M;
    // 
    // HYPRE_IJVectorDestroy(temp);
    // HYPRE_IJVectorDestroy(u0);
    // HYPRE_IJVectorDestroy(u1);
    // HYPRE_IJVectorDestroy(u2);
    // HYPRE_IJMatrixDestroy(M_scaled);
    
    std::cout << ".......Exiting update RHS.........\n\n\n";
}



/* Update RHS when using at most 1 time step per processor */
void SpaceTimeMatrix::updateMultiRHS_ntLT1() {
    
}

/* ------------------------------------------------------------------------- */
/* ------------------------- Runge--Kutta schemes -------------------------- */
/* ------------------------------------------------------------------------- */
/* 
-TODO:
    -Linear solvers & Preconditioners for inverting matrices
    -I'm not really sure what the procedure is after altering a HYPRE object's values. Should it be reinitialized?
    -Get some Butcher tables. Probably steal these from MFEM ODE solvers...
*/


void SpaceTimeMatrix::getButcher(RK_butcher & butch, int option) {
    // options prefixed with 1 are ERK
    // options prefixed with 2 are DIRK

    /* --- ERK tables --- */
    // Forward Euler: 1st-order
    if (option == 11) {
        butch.isSDIRK = 0;
        butch.isImplicit = 0;
        butch.num_stages = 1;
        butch.b[0] = 1;
        butch.c[0] = 0.0;

    // Heun's method: 2nd-order
    } else if (option == 12) {
        butch.isSDIRK = 0;
        butch.isImplicit = 0;
        butch.num_stages = 2;
        butch.a[1][0] = 1.0;
        butch.b[0] = 0.5;
        butch.b[1] = 0.5;
        butch.c[1] = 1.0;
        
    // TODO : 3rd-order
    } else if (option == 13) {
        butch.isSDIRK = 0;
        butch.isImplicit = 0;
        
        

    /* --- DIRK tables --- */
    // Backward Euler: 1st-order
    } else if (option == 21) {
        butch.isImplicit = 1;
        butch.isSDIRK = 1;
        butch.isImplicit = 1;
        butch.num_stages = 1;
        butch.a[0][0] = 1.0;
        butch.b[0] = 1.0;
        butch.c[1] = 0.0;

    // TODO : 2nd-order
    } else if (option == 22) {
        butch.isImplicit = 1;
        butch.isSDIRK = 1;
    
    
    // TODO : 3rd-order
    } else if (option == 23) { 
        butch.isImplicit = 1;   
        butch.isSDIRK = 1;
    }
}


/* DIRK2/3 with mass matrix: 
    step solution of M*du/dt = -L(t)*u + g(t) from solution u0 at t=t0 up to t=t0 + dt.

-The RK updates are
    u = u0 + dt*(b1*k1 + b2*k2 + b3*k3), with
    [M + a11*dt*L(t_n+c1*dt)]*k1 = -L(t0)*u0 + g(t0)
    [M + a22*dt*L(t_n+c1*dt)]*k2 = -L(t + c2*dt)*(u0 + dt*a21*k1) + g(t0 + c2*dt)
    [M + a33*dt*L(t_n+c1*dt)]*k3 = -L(t + c3*dt)*(u0 + dt*a31*k1 + dt*a32*k2) + g(t0 + c3*dt)

-NOTES:
*/
void SpaceTimeMatrix::DIRK(MPI_Comm comm, RK_butcher butch, HYPRE_ParVector * par_u, 
                            HYPRE_ParVector * par_u0, double t0, double dt, 
                            int ilower, int iupper, int * M_rowptr, int * M_colinds, 
                            double * M_data) {

    
    /* ---------------------------------------------------------------- */
    /* --- Initial processing of data for creation of HYPRE objects --- */
    /* ---------------------------------------------------------------- */
    int rebuild_L = m_isTimeDependent; // Do we need to rebuild L for each stage?
    int rebuild_A = (rebuild_L || !butch.isSDIRK); // Do we need to rebuild A to compute each stage?
    
    // Build spatial disc components L and g at t = t0 + c[0]*dt.
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, t0 + butch.c[0]*dt);
    int T_nnz = T_rowptr[spatialDOFs]; // TODO : is this the Length of T_colinds and T_data?

    // The number of rows owned by this processor?
    int onProcSize = iupper - ilower - 1; // TODO: Is this correct??

    // Process M so that it can be added to HYPRE matrices later 
    int * rows = new int[onProcSize]; // Reuse this many times
    int * cols_per_row_T = new int[onProcSize];
    int * cols_per_row_M = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row_T[i] = T_rowptr[i+1] - T_rowptr[i];
        cols_per_row_M[i] = M_rowptr[i+1] - M_rowptr[i];
    } 
    
    // Get L data to populate A.
    double * A_data = new double[T_nnz];
    double fac = butch.a[0][0]*dt;
    for (int i = 0; i < T_nnz; i++) {
        A_data[i] = fac*T_data[i];
    }   
    
    /* ------------------------------------------ */
    /* --- Initial set up of HYPRE components --- */
    /* ------------------------------------------ */
    // Initialize soluton vector: u <- u0
    HYPRE_ParVectorCopy(*par_u0, *par_u); 

    // Stage vectors to be found
    HYPRE_IJVector k1;
    HYPRE_ParVector par_k1;
    HYPRE_IJVector k2;
    HYPRE_ParVector par_k2;
    HYPRE_IJVector k3;
    HYPRE_ParVector par_k3;
    
    // Initialize a dummy RHS vector b.
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVectorCreate(comm, ilower, iupper, &b); 
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    // Initialize a dummy temp vector; it's used in constructing later stage vectors.
    HYPRE_IJVector temp;
    HYPRE_ParVector par_temp;
    if (butch.num_stages > 1) {
        HYPRE_IJVectorCreate(comm, ilower, iupper, &temp); 
        HYPRE_IJVectorSetObjectType(temp, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(temp);
        HYPRE_IJVectorAssemble(temp);
        HYPRE_IJVectorGetObject(temp, (void **) &par_temp);
    }

    // Spatial discretization vector g
    HYPRE_IJVector g;
    HYPRE_ParVector par_g;
    HYPRE_IJVectorCreate(comm, ilower, iupper, &g);
    HYPRE_IJVectorSetObjectType(g, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(g);
    HYPRE_IJVectorSetValues(g, onProcSize, rows, B0);
    HYPRE_IJVectorAssemble(g);
    HYPRE_IJVectorGetObject(g, (void **) &par_g);

    // Matrix in the stage linear systems & spatial disc matrix.
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrix L; 
    HYPRE_ParCSRMatrix parcsr_L;

    // Initialize matrices
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &L);
    HYPRE_IJMatrixSetObjectType(L, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(L);

    // Set matrix values
    HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_T, rows, T_colinds, A_data);
    HYPRE_IJMatrixAddToValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data); // A <- L + M.
    HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_T, rows, T_colinds, T_data);

    // Finalize construction of matrices
    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJMatrixGetObject(A, (void **) &parcsr_A);
    HYPRE_IJMatrixAssemble(L);
    HYPRE_IJMatrixGetObject(L, (void **) &parcsr_L);

    
    /* --------------------------------------------------------------------------------------------- */
    /* --- SOLVE for k1: A*k1 == [M + a11*dt*L(t0+c1*dt)]*k1 = b == -L(t0)*u0 + g(t0) --- */
    /* --------------------------------------------------------------------------------------------- */ 
    // Populate RHS vector b <- -L*u0 + g
    hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, *par_u0, 1.0, par_g, par_b);

    // Set up and finalize k1, but don't set its values yet.
    HYPRE_IJVectorCreate(comm, ilower, iupper, &k1); 
    HYPRE_IJVectorSetObjectType(k1, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(k1);
    HYPRE_IJVectorAssemble(k1);
    HYPRE_IJVectorGetObject(k1, (void **) &par_k1);

    
    // Solve linear system for k1
    // TODO: What should the initial guess for k1 be?? Probably b?
    HYPRE_ParVectorCopy(par_b, par_k1); // Populate k1 with b as initial guess.

    // TODO: apply linear solve
    // HYPRE_Solve(comm, &parcsr_A, par_b, par_k1);

    // Update ODE solution: u <- u + dt*b1*k1
    HYPRE_ParVectorAxpy(dt*butch.b[0], par_k1, *par_u);


    /* ------------------------------------------------------------------------- */
    /* --- SOLVE for k2: 
        A*k2 == [M + a22*dt*L(t0+c1*dt)]*k2 = b 
            == -L(t0 + c2*dt)*(u0 + dt*a21*k1) + g(t0 + c2*dt) --- */
    /* ------------------------------------------------------------------------- */
    if (butch.num_stages > 1) {
        // Rebuild the spatial disc at time t=t0 + c2*delta_t ONLY if it's time dependent. 
        // Allow for changes in sparsity pattern.
        if (rebuild_L) {            
            getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, t0 + butch.c[1]*dt, cols_per_row_T);
            // Update L matrix and G vector values
            // TODO : will this erase all previous elements nz in L if they don't fall inside the possibly new sparsity pattern? Test this.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, B0);            
        }

        // Rebuild A if neccessary
        if (rebuild_A) {
            delete[] A_data;
            A_data = new double[T_nnz];
            fac = butch.a[1][1]*dt;
            for (int i = 0; i < T_nnz; i++) {
                A_data[i] = fac*T_data[i];
            }
            // TODO : as above with L, will this call remove existing entries of A even if they aren't in its possibly new sparsity pattern
            HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_T, rows, T_colinds, A_data);
            HYPRE_IJMatrixAddToValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data); // A <- A + M.
        }

        // Populate RHS vector b <- -L*(u0 + dt*a21*k1) + g
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[1][0], par_k1, par_temp); // temp <- temp + a21*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b = -L*temp + g

        // Set up and finalize k2, but don't set its values yet.
        HYPRE_IJVectorCreate(comm, ilower, iupper, &k2); 
        HYPRE_IJVectorSetObjectType(k2, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(k2);
        HYPRE_IJVectorAssemble(k2);
        HYPRE_IJVectorGetObject(k2, (void **) &par_k2);

        // TODO: What should the initial guess for k2 be?? Probably b?
        HYPRE_ParVectorCopy(par_b, par_k2); // Populate k2 with b as initial guess.

        // TODO: Apply linear solver
        // HYPRE_Solve(comm, &parcsr_A, par_b, par_k2);

        // Update ODE solution: u += dt*b2*k2
        HYPRE_ParVectorAxpy(dt*butch.b[1], par_k2, *par_u); 
    }


    /* --------------------------------------------------------------------------------------- */
    /* --- SOLVE for k3: 
        A*k1 == [M + a33*dt*L(t_n+c1*dt)]*k3 = 
            -L(t + c3*dt)*(u0 + dt*a31*k1 + dt*a32*k2) + g(t0 + c3*dt) --- */
    /* --------------------------------------------------------------------------------------- */
    if (butch.num_stages > 2) {
        // Rebuild the spatial disc at time t0 + c3*dt ONLY if it's time dependent. 
        // Allow for changes in sparsity pattern.
        if (rebuild_L) {            
            getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, t0 + butch.c[2]*dt, cols_per_row_T);
            // Update L matrix and G vector values
            // TODO : will this erase all previous elements nz in L if they don't fall inside the possibly new sparsity pattern? Test this.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, B0);            
        }

        // Rebuild A if neccessary
        if (rebuild_A) {
            delete[] A_data;
            A_data = new double[T_nnz];
            fac = butch.a[2][2]*dt;
            for (int i = 0; i < onProcSize; i++) {
                A_data[i] = fac*T_data[i];
            }
            // TODO : as above with L, will this call remove existing entries of A even if they aren't in its possibly new sparsity pattern
            HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_T, rows, T_colinds, A_data);
            HYPRE_IJMatrixAddToValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data); // A <- A + M.
        }

        // Populate RHS vector b <- -L*(u0 + dt*a31*k1 + dt*a32*k2) + g(t0 + dt*c3)
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[2][0], par_k1, par_temp); // temp <- temp + a31*k1
        HYPRE_ParVectorAxpy(butch.a[2][1], par_k2, par_temp); // temp <- temp + a32*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b <- -L*temp + g

        // Set up and finalize k3, but don't set its values yet.
        HYPRE_IJVectorCreate(comm, ilower, iupper, &k3); 
        HYPRE_IJVectorSetObjectType(k3, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(k3);
        HYPRE_IJVectorAssemble(k3);
        HYPRE_IJVectorGetObject(k3, (void **) &par_k3);

        
        // Solve linear system for k3
        // TODO: What should the initial guess for k3 be?? Probably b?
        HYPRE_ParVectorCopy(par_b, par_k3); // Populate k3 with b as initial guess.

        // TODO: Apply linear solve
        // HYPRE_Solve(comm, &parcsr_A, par_b, par_k3);
    
        // Update ODE solution: u <- u + dt*b3*k3;
        HYPRE_ParVectorAxpy(dt*butch.b[2], par_k3, *par_u); 
    }

      
    /*----------------- */
    /* --- Clean up --- */
    /* ---------------- */
    delete[] rows;
    delete[] cols_per_row_M;
    delete[] cols_per_row_T;
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] B0;
    delete[] X0;
    
    HYPRE_IJVectorDestroy(g);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(k1);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJMatrixDestroy(L);
    
    if (butch.num_stages > 1) {
        HYPRE_IJVectorDestroy(k2);
        HYPRE_IJVectorDestroy(temp);
        if (butch.num_stages > 2) {
            HYPRE_IJVectorDestroy(k3);
        } 
    }
}
/* End of DIRK with mass matrix */

/* ERK with mass matrix */
void SpaceTimeMatrix::ERK(MPI_Comm comm, RK_butcher butch, HYPRE_ParVector * par_u, 
                            HYPRE_ParVector * par_u0, double t0, double dt, 
                            int ilower, int iupper, int * M_rowptr, int * M_colinds, 
                            double * M_data) {

    /* ---------------------------------------------------------------- */
    /* --- Initial processing of data for creation of HYPRE objects --- */
    /* ---------------------------------------------------------------- */
    int rebuild_L = m_isTimeDependent; // Do we need to rebuild L for each stage?
        
    // Build spatial disc components L and g at t = t0 + c[0]*dt.
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, t0 + butch.c[0]*dt);

    // The number of rows owned by this processor?
    int onProcSize = iupper - ilower - 1; // TODO: Is this correct??

    // Process M so that it can be added to HYPRE matrices later 
    int * rows = new int[onProcSize]; // Reuse this many times
    int * cols_per_row_T = new int[onProcSize];
    int * cols_per_row_M = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row_T[i] = T_rowptr[i+1] - T_rowptr[i];
        cols_per_row_M[i] = M_rowptr[i+1] - M_rowptr[i];
    }    
    
    /* ------------------------------------------ */
    /* --- Initial set up of HYPRE components --- */
    /* ------------------------------------------ */
    // Initialize soluton vector: u <- u0
    HYPRE_ParVectorCopy(*par_u0, *par_u); 

    // Stage vectors to be found
    HYPRE_IJVector k1;
    HYPRE_ParVector par_k1;
    HYPRE_IJVector k2;
    HYPRE_ParVector par_k2;
    HYPRE_IJVector k3;
    HYPRE_ParVector par_k3;
    
    // Initialize a dummy RHS vector b.
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVectorCreate(comm, ilower, iupper, &b); 
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    // Initialize a dummy temp vector; it's used in constructing later stage vectors.
    HYPRE_IJVector temp;
    HYPRE_ParVector par_temp;
    if (butch.num_stages > 1) {
        HYPRE_IJVectorCreate(comm, ilower, iupper, &temp); 
        HYPRE_IJVectorSetObjectType(temp, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(temp);
        HYPRE_IJVectorAssemble(temp);
        HYPRE_IJVectorGetObject(temp, (void **) &par_temp);
    }

    // Spatial discretization vector g
    HYPRE_IJVector g;
    HYPRE_ParVector par_g;
    HYPRE_IJVectorCreate(comm, ilower, iupper, &g);
    HYPRE_IJVectorSetObjectType(g, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(g);
    HYPRE_IJVectorSetValues(g, onProcSize, rows, B0);
    HYPRE_IJVectorAssemble(g);
    HYPRE_IJVectorGetObject(g, (void **) &par_g);

    // Matrix in the stage linear systems & spatial disc matrix.
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrix L; 
    HYPRE_ParCSRMatrix parcsr_L;

    // Initialize matrices
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &L);
    HYPRE_IJMatrixSetObjectType(L, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(L);

    // Set matrix values
    HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data); // A <- M.
    HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_T, rows, T_colinds, T_data);

    // Finalize construction of matrices
    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJMatrixGetObject(A, (void **) &parcsr_A);
    HYPRE_IJMatrixAssemble(L);
    HYPRE_IJMatrixGetObject(L, (void **) &parcsr_L);

    
    /* --------------------------------------------------------------------------------------------- */
    /* --- SOLVE for k1: 
        M*k1 = b == -L(t0)*u0 + g(t0) --- */
    /* --------------------------------------------------------------------------------------------- */ 
    // Populate RHS vector b <- -L*u0 + g
    hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, *par_u0, 1.0, par_g, par_b);

    // Set up and finalize k1, but don't set its values yet.
    HYPRE_IJVectorCreate(comm, ilower, iupper, &k1); 
    HYPRE_IJVectorSetObjectType(k1, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(k1);
    HYPRE_IJVectorAssemble(k1);
    HYPRE_IJVectorGetObject(k1, (void **) &par_k1);

    
    // Solve linear system for k1
    // TODO: What should the initial guess for k1 be?? Probably b?
    HYPRE_ParVectorCopy(par_b, par_k1); // Populate k1 with b as initial guess.

    // TODO: apply linear solve
    // HYPRE_Solve(comm, &parcsr_A, par_b, par_k1);

    // Update ODE solution: u <- u + dt*b1*k1
    HYPRE_ParVectorAxpy(dt*butch.b[0], par_k1, *par_u);


    /* ------------------------------------------------------------------------- */
    /* --- SOLVE for k2: 
        M*k2 = b == -L(t0 + c2*dt)*(u0 + dt*a21*k1) + g(t0 + c2*dt) --- */
    /* ------------------------------------------------------------------------- */
    if (butch.num_stages > 1) {
        // Rebuild the spatial disc at time t=t0 + c2*delta_t ONLY if it's time dependent. 
        // Allow for changes in sparsity pattern.
        if (rebuild_L) {            
            getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, t0 + butch.c[1]*dt, cols_per_row_T);
            // Update L matrix and G vector values
            // TODO : will this erase all previous elements nz in L if they don't fall inside the possibly new sparsity pattern? Test this.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, B0);            
        }
        
        // Populate RHS vector b <- -L*(u0 + dt*a21*k1) + g
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[1][0], par_k1, par_temp); // temp <- temp + a21*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b = -L*temp + g

        // Set up and finalize k2, but don't set its values yet.
        HYPRE_IJVectorCreate(comm, ilower, iupper, &k2); 
        HYPRE_IJVectorSetObjectType(k2, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(k2);
        HYPRE_IJVectorAssemble(k2);
        HYPRE_IJVectorGetObject(k2, (void **) &par_k2);

        // TODO: What should the initial guess for k2 be?? Probably b?
        HYPRE_ParVectorCopy(par_b, par_k2); // Populate k2 with b as initial guess.

        // TODO: Apply linear solver
        // HYPRE_Solve(comm, &parcsr_A, par_b, par_k2);

        // Update ODE solution: u += dt*b2*k2
        HYPRE_ParVectorAxpy(dt*butch.b[1], par_k2, *par_u); 
    }


    /* --------------------------------------------------------------------------------------- */
    /* --- SOLVE for k3: 
        M*k3 = -L(t + c3*dt)*(u0 + dt*a31*k1 + dt*a32*k2) + g(t0 + c3*dt) --- */
    /* --------------------------------------------------------------------------------------- */
    if (butch.num_stages > 2) {
        // Rebuild the spatial disc at time t0 + c3*dt ONLY if it's time dependent. 
        // Allow for changes in sparsity pattern.
        if (rebuild_L) {            
            getSpatialDiscretization_helper(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, t0 + butch.c[2]*dt, cols_per_row_T);
            // Update L matrix and G vector values
            // TODO : will this erase all previous elements nz in L if they don't fall inside the possibly new sparsity pattern? Test this.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_T, rows, T_colinds, T_data);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, B0);            
        }

        // Populate RHS vector b <- -L*(u0 + dt*a31*k1 + dt*a32*k2) + g(t0 + dt*c3)
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[2][0], par_k1, par_temp); // temp <- temp + a31*k1
        HYPRE_ParVectorAxpy(butch.a[2][1], par_k2, par_temp); // temp <- temp + a32*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b <- -L*temp + g

        // Set up and finalize k3, but don't set its values yet.
        HYPRE_IJVectorCreate(comm, ilower, iupper, &k3); 
        HYPRE_IJVectorSetObjectType(k3, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(k3);
        HYPRE_IJVectorAssemble(k3);
        HYPRE_IJVectorGetObject(k3, (void **) &par_k3);

        
        // Solve linear system for k3
        // TODO: What should the initial guess for k3 be?? Probably b?
        HYPRE_ParVectorCopy(par_b, par_k3); // Populate k3 with b as initial guess.

        // TODO: Apply linear solve
        // HYPRE_Solve(comm, &parcsr_A, par_b, par_k3);
    
        // Update ODE solution: u <- u + dt*b3*k3;
        HYPRE_ParVectorAxpy(dt*butch.b[2], par_k3, *par_u); 
    }

      
    /*----------------- */
    /* --- Clean up --- */
    /* ---------------- */
    delete[] rows;
    delete[] cols_per_row_M;
    delete[] cols_per_row_T;
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] B0;
    delete[] X0;
    
    HYPRE_IJVectorDestroy(g);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(k1);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJMatrixDestroy(L);
    
    if (butch.num_stages > 1) {
        HYPRE_IJVectorDestroy(k2);
        HYPRE_IJVectorDestroy(temp);
        if (butch.num_stages > 2) {
            HYPRE_IJVectorDestroy(k3);
        } 
    }
}
/* End of ERK with mass matrix */
/* ---------------------- End of Runge--Kutta schemes ---------------------- */

/* ------------------------------------------------------------------------- */
/* --------------------------- Multistep methods --------------------------- */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* ----------------- More than one time step per processor ----------------- */
/* ------------------------------------------------------------------------- */

/* First-order BDF implicit scheme (Backward Euler / 1st-order Adams-Moulton). */
void SpaceTimeMatrix::BDF1(int* &rowptr, int* &colinds, double* &data,
                           double* &B, double* &X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, m_dt*tInd0);
    int nnzPerTime = T_rowptr[spatialDOFs];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }
    
    std::cout << "spatialDOFs = " << spatialDOFs << '\n';
    std::cout << "nnz(M) = " << M_rowptr[spatialDOFs] << '\n';
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (M_rowptr[spatialDOFs] + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= M_rowptr[spatialDOFs];

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (m_isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && m_isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                                     X0, spatialDOFs, m_dt*ti);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;
        std::map<int, double>::iterator it;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += T_data[j];
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else {

            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                
                std::map<int, double> entries;

                // Add row of off-diagonal block, -(M/dt)*u_{i-1}
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd + M_colinds[j];
                    data[dataInd] = -M_data[j];
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += T_data[j];
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
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
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] X0;
}


/* Second-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF2(int* &rowptr, int* &colinds, double* &data,
                           double* &B, double* &X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, m_dt*tInd0);
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (2*M_rowptr[spatialDOFs] + nnzPerTime);   // nnzs on this processor
    if (tInd0 == 0) procNnz -= 2*M_rowptr[spatialDOFs];
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= M_rowptr[spatialDOFs];

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (m_isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && m_isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                                     X0, spatialDOFs, m_dt*ti);
        }

        int colPlusDiag   = ti*spatialDOFs;
        int colPlusOffd_1 = (ti - 1)*spatialDOFs;
        int colPlusOffd_2 = (ti - 2)*spatialDOFs;
        std::map<int, double>::iterator it;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 2.0*T_data[j] / 3.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        // At time t=1, only have 1 off-diagonal block
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time t1
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Add row of off-diagonal block, -4M/3dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                    data[dataInd] = -4.0*M_data[j] / 3.0;
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 2.0*T_data[j] / 3.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Add row of 2nd off-diagonal block, M/3dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_2 + M_colinds[j];
                    data[dataInd] = 1.0*M_data[j] / 3.0;
                    dataInd += 1;
                }

                // Add row of off-diagonal block, -4M/3dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                    data[dataInd] = -4.0*M_data[j] / 3.0;
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 2.0*T_data[j] / 3.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*B0[i] / 3.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
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
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] X0;
}


/* Third-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF3(int* &rowptr, int* &colinds, double* &data,
                           double* &B, double* &X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, m_dt*tInd0);
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (3*M_rowptr[spatialDOFs] + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= 3*M_rowptr[spatialDOFs];
    if ((tInd0 <= 1) && (tInd1 >= 1)) procNnz -= 2*M_rowptr[spatialDOFs];
    if ((tInd0 <= 2) && (tInd1 >= 2)) procNnz -= M_rowptr[spatialDOFs];

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (m_isTimeDependent) and matrix has not been built yet (ti > tInd0)
        if ((ti > tInd0) && m_isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                                     X0, spatialDOFs, m_dt*ti);
        }

        int colPlusDiag   = ti*spatialDOFs;
        int colPlusOffd_1 = (ti - 1)*spatialDOFs;
        int colPlusOffd_2 = (ti - 2)*spatialDOFs;
        int colPlusOffd_3 = (ti - 3)*spatialDOFs;
        std::map<int, double>::iterator it;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        // At time t=1, only have 1 off-diagonal block
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time t1
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Add row of off-diagonal block, -18M/11dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                    data[dataInd] = -18.0*M_data[j] / 11.0;
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else if (ti == 2) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
               std::map<int, double> entries;

                // Add row of 2nd off-diagonal block, 9M/11dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_2 + M_colinds[j];
                    data[dataInd] = 9.0*M_data[j] / 11.0;
                    dataInd += 1;
                }

                // Add row of off-diagonal block, -18M/11dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                    data[dataInd] = -18.0*M_data[j] / 11.0;
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Add row of 3rd off-diagonal block, -2M/11dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_3 + M_colinds[j];
                    data[dataInd] = -2.0*M_data[j] / 11.0;
                    dataInd += 1;
                }

                // Add row of 2nd off-diagonal block, 9M/11dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_2 + M_colinds[j];
                    data[dataInd] = 9.0*M_data[j] / 11.0;
                    dataInd += 1;
                }

                // Add row of off-diagonal block, -18M/11dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                    data[dataInd] = -18.0*M_data[j] / 11.0;
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is two for off-diagonal blocks
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
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
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] X0;
}


/* Second-order Adams-Moulton implicit scheme (trapezoid method). */
void SpaceTimeMatrix::AM2(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for previous time step, or first step if tInd0=0
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* Bi;
    double* Xi;
    int* T_rowptr_1 = NULL;
    int* T_colinds_1 = NULL;
    double* T_data_1 = NULL;
    double* Bi_1 = NULL;
    double* Xi_1 = NULL;
    int spatialDOFs;
    if (tInd0 > 0) {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, Bi, Xi,
                                 spatialDOFs, m_dt*(tInd0-1));
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, Bi, Xi,
                                 spatialDOFs, m_dt*tInd0);
    }
    if (!m_isTimeDependent) {
        T_rowptr_1 = T_rowptr;
        T_colinds_1 = T_colinds;
        T_data_1 = T_data;
        Bi_1 = Bi;
        Xi_1 = Xi;   
    }
    int nnzPerTime = T_rowptr[spatialDOFs];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = m_ntPerProc * spatialDOFs;
    int procNnz     = 2 * m_ntPerProc * nnzPerTime;     // nnzs on this processor
    if (tInd0 == 0) procNnz -= nnzPerTime;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        int colPlusOffd = (ti - 1)*spatialDOFs;
        int colPlusDiag = ti*spatialDOFs;
        std::map<int, double>::iterator it;

        // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
        // to build one spatial matrix each iteration. Matrix at ti for previous iteration
        //  is used as ti-1 on this iteration.
        if ((ti != 0) && m_isTimeDependent) {
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
            getSpatialDiscretization(T_rowptr, T_colinds, T_data, Bi,
                                     Xi, spatialDOFs, m_dt*ti);
        }

        // At time t=0, only have diagonal spatial discretization block.
        if (ti == 0) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += T_data[j] / 2.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = Bi[i] / 2.0;
                X[thisRow] = Xi[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] -= M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {
                    entries[T_colinds_1[j]] += T_data_1[j] / 2.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                // for off-diagonal block
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusOffd + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                std::map<int, double> entries1;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries1[M_colinds[j]] += M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries1[T_colinds[j]] += T_data[j] / 2.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                // for diagonal block
                for (it=entries1.begin(); it!=entries1.end(); it++) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = (Bi[i] + Bi_1[i]) / 2.0;
                X[thisRow] = Xi[i];

                // Total nonzero for this row on processor is the total nnz in this row
                // the spatial discretization at time ti and ti-1.
                rowptr[thisRow+1] = dataInd;
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
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] Bi;
    delete[] Xi;
    if (m_isTimeDependent) {
        delete[] T_rowptr_1;
        delete[] T_colinds_1;
        delete[] T_data_1;
        delete[] Bi_1;
        delete[] Xi_1;   
    }
}


/* First-order Adams-Bashforth explicit scheme (Forward Euler). */
void SpaceTimeMatrix::AB1(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    if (tInd0 > 0) {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0,
                                 spatialDOFs, m_dt*(tInd0-1));
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0,
                                 spatialDOFs, m_dt*tInd0);
    }
    int nnzPerTime = T_rowptr[spatialDOFs];    
 
    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (M_rowptr[spatialDOFs] + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= nnzPerTime;

    rowptr  = new int[onProcSize + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    B = new double[onProcSize];
    X = new double[onProcSize];
    int dataInd = 0;
    int thisRow = 0;
    rowptr[0] = 0;

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        // Build spatial matrix for next time step if coefficients are time dependent
        // (m_isTimeDependent) and matrix has not been built yet (ti > tInd0, ti > 1)
        if ((ti > tInd0) && m_isTimeDependent && (ti > 1)) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                                     X0, spatialDOFs, m_dt*ti);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;
        std::map<int, double>::iterator it;

        // At time t=0, only have identity block on diagonal
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                
                // Add mass matrix as diagonal block, M/dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusDiag + M_colinds[j];
                    data[dataInd] = M_data[j];
                    dataInd += 1;
                }

                // Assume user implements boundary conditions to rhs
                B[thisRow] = 0.0;
                X[thisRow] = X0[i];

                // One nonzero for this row
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] -= M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += T_data[j];
                }

                // Add spatial discretization and mass matrix to global matrix
                // for off-diagonal block
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusOffd + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add mass matrix as diagonal block, M/dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusDiag + M_colinds[j];
                    data[dataInd] = M_data[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = B0[i];
                X[thisRow] = X0[i];

                // Total nonzero for this row on processor is one for off-diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
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
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] X0;
}


/* Second-order Adams-Bashforth explicit scheme. */
void SpaceTimeMatrix::AB2(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc - 1;

    // Pointers to CSR arrays for A_{ti} and A_{ti-1}
    int* T_rowptr_1;
    int* T_colinds_1;
    double* T_data_1;
    double* Bi_1;
    double* Xi_1;
    int* T_rowptr_2 = NULL;
    int* T_colinds_2 = NULL;
    double* T_data_2 = NULL;
    double* Bi_2 = NULL;
    double* Xi_2 = NULL;
    int spatialDOFs;
    if (tInd0 <= 1) {
        getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1,
                                 Bi_1, Xi_1, spatialDOFs, 0);
    }
    else {
        getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1, Bi_1,
                                 Xi_1, spatialDOFs, m_dt*(tInd0-2));
    }
    if (!m_isTimeDependent) {
        T_rowptr_2 = T_rowptr_1;
        T_colinds_2 = T_colinds_1;
        T_data_2 = T_data_1;
        Bi_2 = Bi_1;
        Xi_2 = Xi_1;   
    }

    int nnzPerTime = T_rowptr_1[spatialDOFs];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (2*nnzPerTime + M_rowptr[spatialDOFs]);     // nnzs on this processor
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

    // Loop over each time index and build sparse space-time matrix rows on this processor
    for (int ti=tInd0; ti<=tInd1; ti++) {

        int colPlusOffd_1 = (ti - 1)*spatialDOFs;
        int colPlusOffd_2 = (ti - 2)*spatialDOFs;
        int colPlusDiag = ti*spatialDOFs;
        std::map<int, double>::iterator it;

         // Swap pointer between A_{ti} and A_{ti-1} each iteration. Makes so that only have
        // to build one spatial matrix each iteration. Matrix at ti-1 for previous iteration
        //  is used as ti-2 on this iteration.
        if ((ti > 1) && m_isTimeDependent) {
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
            getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1, Bi_1,
                                     Xi_1, spatialDOFs, m_dt*(ti-1));
        }

        // At time t=0, only have identity block on diagonal
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add mass matrix as diagonal block, M/dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusDiag + M_colinds[j];
                    data[dataInd] = M_data[j];
                    dataInd += 1;
                }

                // Assume user implements boundary conditions to rhs
                B[thisRow] = 0.0;
                X[thisRow] = Xi_1[i];

                // One nonzero for this row
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else if (ti == 1) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] -= M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {
                    entries[T_colinds_1[j]] += 3.0*T_data_1[j] / 2.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                // for off-diagonal block
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusOffd_1 + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add mass matrix as diagonal block, M/dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusDiag + M_colinds[j];
                    data[dataInd] = M_data[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 3.0*Bi_1[i] / 2.0;
                X[thisRow] = Xi_1[i];

                // Total nonzero for this row on processor is one for diagonal block
                // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;
            }
        }
        else {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add row for spatial discretization of second off-diagonal block 
                for (int j=T_rowptr_2[i]; j<T_rowptr_2[i+1]; j++) {
                    // Add spatial block -Lu_{ti-2}
                    data[dataInd] = -T_data_2[j] / 2.0;
                    colinds[dataInd] = colPlusOffd_2 + T_colinds_2[j];
                    dataInd += 1;
                }

                std::map<int, double> entries;

                // Get row of mass matrix
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    entries[M_colinds[j]] -= M_data[j];
                }

                // Get row of spatial discretization
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {
                    entries[T_colinds_1[j]] += 3.0*T_data_1[j] / 2.0;
                }

                // Add spatial discretization and mass matrix to global matrix
                // for off-diagonal block
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = colPlusOffd_1 + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }

                // Add mass matrix as diagonal block, M/dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    colinds[dataInd] = colPlusDiag + M_colinds[j];
                    data[dataInd] = M_data[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = (3.0*Bi_1[i] - Bi_2[i]) / 2.0;
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
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] Bi_1;
    delete[] Xi_1;
    if (m_isTimeDependent) {
        delete[] T_rowptr_2;
        delete[] T_colinds_2;
        delete[] T_data_2;
        delete[] Bi_2;
        delete[] Xi_2;   
    }
}


/* ------------------------------------------------------------------------- */
/* ------------------ At most one time step per processor ------------------ */
/* ------------------------------------------------------------------------- */

/* First-order BDF implicit scheme (Backward Euler / 1st-order Adams-Moulton). */
void SpaceTimeMatrix::BDF1(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*m_timeInd);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
    }

    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzPerTime;
    else procNnz = nnzPerTime + (M_rowptr[procRows] - M_rowptr[0]);

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    // Column indices for this processor given the time index and the first row
    // stored on this processor of the spatial discretization (zero if whole
    // spatial problem stored on one processor).
    int colPlusOffd = (m_timeInd - 1)*spatialDOFs + localMinRow;
    int colPlusDiag = m_timeInd*spatialDOFs;
    std::map<int, double>::iterator it;

    // At time t=0, only have spatial discretization block
    if (m_timeInd == 0) {
        
        // Loop over each on-processor row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += T_data[j];
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= 1;

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    else {
        // Loop over each row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Add row of off-diagonal block, -(M/dt)*u_{i-1}
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd + M_colinds[j];
                data[dataInd] = -M_data[j];
                dataInd += 1;
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += T_data[j];
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= 1;

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
}


/* Second-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF2(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*m_timeInd);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
    }
    
    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzPerTime;
    else if (m_timeInd == 1) procNnz = nnzPerTime + (M_rowptr[procRows] - M_rowptr[0]);
    else procNnz = nnzPerTime + 2*(M_rowptr[procRows] - M_rowptr[0]);

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    // Column indices for this processor given the time index and the first row
    // stored on this processor of the spatial discretization (zero if whole
    // spatial problem stored on one processor).
    int colPlusOffd_2 = (m_timeInd - 2)*spatialDOFs + localMinRow;
    int colPlusOffd_1 = (m_timeInd - 1)*spatialDOFs + localMinRow;
    int colPlusDiag   = m_timeInd*spatialDOFs;
    std::map<int, double>::iterator it;

    // At time t=0, only have spatial discretization block
    if (m_timeInd == 0) {
        
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 2.0*T_data[j] / 3.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0/3.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    // At time t=1, only have 1 off-diagonal block
    else if (m_timeInd == 1) {
        // Loop over each row in spatial discretization at time t1
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Add row of off-diagonal block, -4M/3dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                data[dataInd] = -4.0*M_data[j] / 3.0;
                dataInd += 1;
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 2.0*T_data[j] / 3.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0/3.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Add row of 2nd off-diagonal block, M/3dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_2 + M_colinds[j];
                data[dataInd] = 1.0*M_data[j] / 3.0;
                dataInd += 1;
            }

            // Add row of off-diagonal block, -4M/3dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                data[dataInd] = -4.0*M_data[j] / 3.0;
                dataInd += 1;
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 2.0*T_data[j] / 3.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0/3.0);

            // Total nonzero for this row on processor is two for off-diagonal blocks
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
}


/* Third-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF3(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*m_timeInd);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
    }
    
    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzPerTime;
    else if (m_timeInd == 1) procNnz = nnzPerTime + (M_rowptr[procRows] - M_rowptr[0]);
    else if (m_timeInd == 2) procNnz = nnzPerTime + 2*(M_rowptr[procRows] - M_rowptr[0]);
    else procNnz = nnzPerTime + 3*(M_rowptr[procRows] - M_rowptr[0]);

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    // Column indices for this processor given the time index and the first row
    // stored on this processor of the spatial discretization (zero if whole
    // spatial problem stored on one processor).
    int colPlusOffd_3 = (m_timeInd - 3)*spatialDOFs + localMinRow;
    int colPlusOffd_2 = (m_timeInd - 2)*spatialDOFs + localMinRow;
    int colPlusOffd_1 = (m_timeInd - 1)*spatialDOFs + localMinRow;
    int colPlusDiag   = m_timeInd*spatialDOFs;
    std::map<int, double>::iterator it;

    // At time t=0, only have spatial discretization block
    if (m_timeInd == 0) {
        
        // Loop over each row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    // At time t=1, only have 1 off-diagonal block
    else if (m_timeInd == 1) {
        // Loop over each row in spatial discretization at time t1
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Add row of off-diagonal block, -18M/11dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                data[dataInd] = -18.0*M_data[j] / 11.0;
                dataInd += 1;
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    else if (m_timeInd == 2) {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
           std::map<int, double> entries;

            // Add row of 2nd off-diagonal block, 9M/11dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_2 + M_colinds[j];
                data[dataInd] = 9.0*M_data[j] / 11.0;
                dataInd += 1;
            }

            // Add row of off-diagonal block, -18M/11dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                data[dataInd] = -18.0*M_data[j] / 11.0;
                dataInd += 1;
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

            // Total nonzero for this row on processor is two for off-diagonal blocks
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Add row of 3rd off-diagonal block, -2M/11dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_3 + M_colinds[j];
                data[dataInd] = -2.0*M_data[j] / 11.0;
                dataInd += 1;
            }

            // Add row of 2nd off-diagonal block, 9M/11dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_2 + M_colinds[j];
                data[dataInd] = 9.0*M_data[j] / 11.0;
                dataInd += 1;
            }

            // Add row of off-diagonal block, -18M/11dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusOffd_1 + M_colinds[j];
                data[dataInd] = -18.0*M_data[j] / 11.0;
                dataInd += 1;
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += 6.0*T_data[j] / 11.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

            // Total nonzero for this row on processor is two for off-diagonal blocks
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
}


/* Second-order Adams-Moulton implicit scheme (trapezoid method). */
void SpaceTimeMatrix::AM2(int* &rowptr, int* &colinds, double* &data,
                          double* &B, double* &X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for previous time step, or first step if m_timeInd0=0
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* Bi = NULL;
    double* Xi = NULL;
    int* T_rowptr_1 = NULL;
    int* T_colinds_1 = NULL;
    double* T_data_1 = NULL;
    double* Bi_1 = NULL;
    double* Xi_1 = NULL;
    int procNnz;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*(m_timeInd));
    int procRows = localMaxRow - localMinRow + 1;
    procNnz = T_rowptr[procRows];

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
    }

    // Get discretization at time ti-1 for Adams-Moulton if m_timeInd!=0. 
    if (m_timeInd > 0) {
        int localMinRow_1;
        int localMaxRow_1;
        getSpatialDiscretization(m_spatialComm, T_rowptr_1, T_colinds_1, T_data_1,
                                 B, X, localMinRow_1, localMaxRow_1,
                                 spatialDOFs, m_dt*(m_timeInd-1));
     
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

    // Local CSR matrices have (spatially) global column indices. Do not need
    // to account for the min row indexing as in BDF.
    int colPlusOffd = (m_timeInd - 1)*spatialDOFs;
    int colPlusDiag = m_timeInd*spatialDOFs;
    std::map<int, double>::iterator it;

    // At time t=0, only have spatial discretization at t0.
    if (m_timeInd == 0) {
        // Loop over each row in spatial discretization at time t0
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += T_data[j] / 2.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            //  TODO: FIX
            // B[i] = Bi[i] / 2.0;
            // X[i] = Xi[i];

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries[M_colinds[j]] -= M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {
                entries[T_colinds_1[j]] += T_data_1[j] / 2.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            // for off-diagonal block
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusOffd + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            std::map<int, double> entries1;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries1[M_colinds[j]] += M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries1[T_colinds[j]] += T_data[j] / 2.0;
            }

            // Add spatial discretization and mass matrix to global matrix
            // for diagonal block
            for (it=entries1.begin(); it!=entries1.end(); it++) {
                colinds[dataInd] = colPlusDiag + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            //  TODO: FIX
            // B[i] = (Bi[i] + Bi_1[i]) / 2.0;
            // X[i] = Xi[i];

            // Total nonzero for this row on processor is the total nnz in this row
            // the spatial discretization at time ti and ti-1.
            rowptr[i+1] = dataInd;
        }
    }

    // Check if sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] Bi;
    delete[] Xi;
    delete[] T_rowptr_1;
    delete[] T_colinds_1;
    delete[] T_data_1;
    delete[] Bi_1;
    delete[] Xi_1;
}


/* First-order Adams-Bashforth explicit scheme (Forward Euler). */
void SpaceTimeMatrix::AB1(int* &rowptr, int* &colinds,  double* &data,
                          double* &B, double* &X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    if (m_timeInd == 0) {    
        getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 m_dt*m_timeInd);
    }
    else {
        getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 m_dt*(m_timeInd-1));
    }
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
    }

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int procNnz;
    if (m_timeInd == 0) procNnz = (M_rowptr[procRows] - M_rowptr[0]);
    else procNnz = nnzPerTime + (M_rowptr[procRows] - M_rowptr[0]);

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    // Local CSR matrices in off-diagonal blocks here have (spatially) global
    // column indices. Only need to account for min row indexing for the
    // diagonal block.
    int colPlusOffd = (m_timeInd - 1)*spatialDOFs;
    int colPlusDiag = m_timeInd*spatialDOFs + localMinRow;
    std::map<int, double>::iterator it;

    // At time t=0, only have identity block on diagonal
    if (m_timeInd == 0) {
        
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

            // Add mass matrix as diagonal block, M/dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusDiag + M_colinds[j];
                data[dataInd] = M_data[j];
                dataInd += 1;
            }

            // Assume user implements boundary conditions to rhs
            B[i] = 0.0;
            // X[i] = X0[i];

            // One nonzero for this row
            rowptr[i+1] = dataInd;
        }
    }
    else {
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

            std::map<int, double> entries;

            // Get row of mass matrix
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                entries[M_colinds[j]] -= M_data[j];
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += T_data[j];
            }

            // Add spatial discretization and mass matrix to global matrix
            // for off-diagonal block
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = colPlusOffd + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }

            // Add mass matrix as diagonal block, M/dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                colinds[dataInd] = colPlusDiag + M_colinds[j];
                data[dataInd] = M_data[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            // X[i] = X0[i];

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = dataInd;
        }
    }

    // Check if sufficient data was allocated
    if (dataInd != procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
}

