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
    if (m_globRank == 0) std::cout << "Building matrix.\n";
    if (m_useSpatialParallel) GetMatrix_ntLE1();
    else GetMatrix_ntGT1();
    if (m_globRank == 0) std::cout << "Space-time matrix assembled.\n";
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

