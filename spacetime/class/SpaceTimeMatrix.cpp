#include <iostream>
#include <fstream>
#include "SpaceTimeMatrix.hpp"

// TODO:
//      - Add scaling by mass inverse in MySpaceTime.cpp. **Include dt -> (M / dt)^{-1}.
//          Then don't scale by dt in SpaceTimeMatrix.cpp
//      - Add function or capability to eliminate zeros from spatial
//        discretization.
//      - Add isTimeDependent option
//      - Finish AB2 for spatial parallel
//      - Add hypre timing for setup and solve (see ij.c)
//      - May be something wrong with BDF3 implementation --> singular matrix?
//          + lots of sing submatrices, bad/stalling overall conv.
//          + Seems related to number of processors (actually relaxation: GS better on less proc)
//              CON: srun -N 16 -n 256 -p pdebug driver -nt 128 -t 12 -o 1 -l 4 -Af 0 -Ai 8 -Ar 3 -Ac 3
//              DNC: srun -N 1 -n 16 -p pdebug driver -nt 128 -t 12 -o 1 -l 4 -Af 0 -Ai 8 -Ar 3 -Ac 3
//      - Seems like parallel coarsening can be a problem.
//          CON (no gmres, n16): srun -N 1 -n 16 -p pdebug driver -nt 128 -t 12 -o 1 -l 4 -Af 0 -Ai 6 -Ar 0 -Ac 3 -AIR 2 -AsC 0.1 -AsR 0.01 -gmres 0
//          DNC (gmres, n16): srun -N 1 -n 16 -p pdebug driver -nt 128 -t 12 -o 1 -l 4 -Af 0 -Ai 6 -Ar 0 -Ac 3 -AIR 2 -AsC 0.1 -AsR 0.01 -gmres 1
//          DNC (no gmres, n256): srun -N 16 -n 256 -p pdebug driver -nt 128 -t 12 -o 1 -l 4 -Af 0 -Ai 6 -Ar 0 -Ac 3 -AIR 2 -AsC 0.1 -AsR 0.01 -gmres 0
//      - Figure out why can't destroy IJ matrix
//      - May be something wrong with BDF3 implementation --> singular matrix?
//          + lots of sing submatrices, bad/stalling overall conv.
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
    if (m_useSpatialParallel) GetMatrix_ntLE1();
    else GetMatrix_ntGT1();
    if (m_globRank == 0) std::cout << "Space-time matrix assembled.\n";
}


/* Get space-time matrix for at most 1 time step per processor */
void SpaceTimeMatrix::GetMatrix_ntLE1()
{
    // Get local CSR structure
    int *rowptr;
    int *colinds;
    double *data;
    double *B;
    double *X;
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
    int *rows = new int[onProcSize];
    int *cols_per_row = new int[onProcSize];
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
    int *rowptr;
    int *colinds;
    double *data;
    double *B;
    double *X;
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
    int *rows = new int[onProcSize];
    int *cols_per_row = new int[onProcSize];
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
        int **grid_relax_points = new int *[4];
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
        delete[] grid_relax_points;
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
/* ------------------------- Runge--Kutta schemes -------------------------- */
/* ------------------------------------------------------------------------- */
/* 
-TODO:
    -Is it a good idea to have L and M passed to the schemes?? Maybe M yes. But maybe not L... It makes things kind of messy.
    -Linear solver & Preconditioner for inverting mass matrix...
        Is it appropriate to use the ones set up for the ST linear system? Maybe not...
        These can be set up and passed to the function?
    -Make a function that builds the linear system for arbitrary stage? Then it can just return the appropriate vectors...
    -I'm not really sure what the procedure is after alterning a HYPRE object's values. Should it be reinitialized?
    -Does the sparsity pattern of L change with time? I was thinking it doesn't but I guess it could if a wavespeed, for example changes sign... because then the upwinding direction will also alternate... Maybe I should go through and remove this assumption...
    -Get some Butcher tables. Probably steal these from MFEM ODE solvers...

-NOTES:
    -Pass mass, M, and spatial disc, L, CSR info since they'll likely be beign built outside of this scheme. RK code won't alter those arrays.
    -Reuires u and u0 to be HYPRE vectors.
    -Requires M arrays to be initialized to NULL if no mass matrix present...
    -Written assuming the sparsity pattern of spatial disc and mass doesn't depend on time.
*/


void getButcher(RK_butcher butch, int option) {
    // options prefixed with 1 are ERK
    // optiosn prefixed with 2 are DIRK

    // Forward Euler
    if (option == 11) {
        butch.isImplicit = 0;
        butch.num_stages = 1;
        butch.b[0] = 1;
        // todo

    // Heun's method (aka improved Euler?)
    } else if (option == 12) {
        butch.isImplicit = 0;
        butch.num_stages = 2;
        butch.a[1][0] = 1.0;
        butch.b[0] = 0.5;
        butch.b[1] = 0.5;
        butch.c[1] = 1.0;
        
    } else if (option == 13) {
        butch.isImplicit = 0;
        //todo

    // Backward Euler
    } else if (option == 21) {
        butch.isImplicit = 1;
        butch.isSDIRK = 1;
        butch.isImplicit = 1;
        butch.num_stages = 1;
        butch.a[0][0] = 1.0;
        butch.b[0] = 1.0;
        butch.c[1] = 0.0;

    } else if (option == 22) {
        butch.isImplicit = 1;
        butch.isSDIRK = 1;
        // todo
    } else if (option == 23) { 
        butch.isImplicit = 1;   
        butch.isSDIRK = 1;
        // todo
    }
}


/* DIRK2/3 with mass matrix: 
    step solution of M*du/dt = -L(t)*u + g(t) from t0 to t0 + deltat.

-The RK updates are
    u = u0 + delta_t*(b1*k1 + b2*k2 + b3*k3), with
    [M + a11*delta_t*L(t_n+c1*delta_t)]*k1 = -L(t0)*u0 + g(t0)
    [M + a22*delta_t*L(t_n+c1*delta_t)]*k2 = -L(t + c2*delta_t)*(u0 + delta_t*a21*k1) + g(t0 + c2*delta_t)
    [M + a33*delta_t*L(t_n+c1*delta_t)]*k3 = -L(t + c3*delta_t)*(u0 + delta_t*a31*k1 + delta_t*a32*k2) + g(t0 + c3*delta_t)

-NOTES:
    -Assumes that the sparsity pattern of A == M + constant*L(t) doesn't depend on time! This probably isn't right...

-TODO:
    -What's more sparse, L or M? We need to add these things together, so it's probably best to set initial values in HYPRE matrix with the denser of the two, then add the sparser matrix to the result (since they won't have same sparsity pattern, this will be more efficient). I'll assume L is the denser of the two here...

*/
void SpaceTimeMatrix::DIRK(MPI_Comm comm, RK_butcher butch, HYPRE_ParVector * par_u, HYPRE_ParVector * par_u0, double t0, double delta_t, int ilower, int iupper, int * M_rowptr, int * M_colinds, double * M_data, int * L_rowptr, int * L_colinds, double * L_data, double * g_data) {

    /* ---------------------- */
    /* --- Initial set up --- */
    /* ---------------------- */

    // Initialize ODE soluton: u = u0
    HYPRE_ParVectorCopy(*par_u0, *par_u); 

    // The stage vectors to be found
    HYPRE_IJVector k1;
    HYPRE_ParVector par_k1;
    HYPRE_IJVector k2;
    HYPRE_ParVector par_k2;
    HYPRE_IJVector k3;
    HYPRE_ParVector par_k3;

    // The rows owned by this processor?
    int onProcSize = iupper - ilower - 1; // TODO: Is this correct??

    // Do we have a mass matrix M? If not. Populate CSR structures of M with identity.
    int mass = 0;
    if (M_data == NULL) {
        int * M_rowptr = new int[onProcSize+1];
        int * M_colinds = new int[onProcSize];
        double * M_data = new double[onProcSize];

        // TODO: are these right for the identity??
        for (int i = 0; i < onProcSize; i++) {
            M_rowptr[i] = i;
            M_colinds[i] = i + ilower; 
            M_data[i] = 1.0;
        }
        M_rowptr[onProcSize] = onProcSize;
    } else {
        mass = 1;
    }

    // Will we need to rebuild A for each stage?
    int rebuild_A = (m_isTimeDependent || !butch.isSDIRK);

    // Components for if we need to rebuild spatial disc, as don't want to overwrite ones passed to function.
    int * L_rowptr_new;
    int * L_colinds_new;
    double * L_data_new;
    double * g_data_new;
    double * dummy; // I don't think we need this here...
    int spatialDOFs;

    int built_L_new = 0; 
    // Rebuild L at t0 + c1*delta_t if c1 != 0
    // TODO: is it safe to compare doubles like this??
    if (butch.c[0] != 0.0 && m_isTimeDependent) {
        getSpatialDiscretization(comm, L_rowptr_new, L_colinds_new, L_data_new, g_data_new, dummy, ilower, iupper, spatialDOFs, t0 + butch.c[1], delta_t);
        built_L_new = 1;
    }

    // Just use pointer to get whatever is the appropriate L and g data.
    double * temp_L_data, * temp_g_data;
    if (!built_L_new) {
        temp_L_data = L_data;
        temp_g_data = g_data;
    } else {
        temp_L_data = L_data_new;
        temp_g_data = g_data;
    }

    // Initialize RHS vector b.
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVectorCreate(comm, ilower, iupper, &b); 
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    // Initialize temp vector; it's used as a temporary vector in building later stage vectors.
    HYPRE_IJVector temp;
    HYPRE_ParVector par_temp;
    if (butch.num_stages > 1) {
        HYPRE_IJVectorCreate(comm, ilower, iupper, &temp); 
        HYPRE_IJVectorSetObjectType(temp, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(temp);
        HYPRE_IJVectorAssemble(temp);
        HYPRE_IJVectorGetObject(temp, (void **) &par_temp);
    }

    // The coefficient matrix in the linear systems.
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;

    // Place spatial discretization and mass components into HYPRE objects.
    HYPRE_IJMatrix L; 
    HYPRE_ParCSRMatrix parcsr_L;
    HYPRE_IJVector g;
    HYPRE_ParVector par_g;

    // Initialize matrices
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &L);
    HYPRE_IJMatrixSetObjectType(L, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(L);

    // Set matrix coefficients
    int * rows = new int[onProcSize];
    int * cols_per_row_M = new int[onProcSize];
    int * cols_per_row_L = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row_M[i] = M_rowptr[i+1] - M_rowptr[i];
        cols_per_row_L[i] = L_rowptr[i+1] - L_rowptr[i];
    }
    HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_L, rows, L_colinds, temp_L_data);


    // Create data to initially populate A with.
    double * A_data = new double[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        A_data[i] = butch.a[0][0]*delta_t*temp_L_data[i];
    }
    HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_L, rows, L_colinds, A_data);


    // Finalize construction
    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJMatrixGetObject(A, (void **) &parcsr_A);
    HYPRE_IJMatrixAssemble(L);
    HYPRE_IJMatrixGetObject(L, (void **) &parcsr_L);

    // Add M values into existing values of A. Now A is ready to be used to find k1.
    HYPRE_IJMatrixAddToValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data);

    // Create g vector
    HYPRE_IJVectorCreate(comm, ilower, iupper, &g);
    HYPRE_IJVectorSetObjectType(g, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(g);
    HYPRE_IJVectorSetValues(g, onProcSize, rows, temp_g_data);
    HYPRE_IJVectorAssemble(g);
    HYPRE_IJVectorGetObject(g, (void **) &par_g);


    /* ------------------------------------------------------------------------------------- */
    /* --- SOLVE for k1: [M + a11*delta_t*L(t_n+c1*delta_t)]*k1 = b == -L(t0)*u0 + g(t0) --- */
    /* ------------------------------------------------------------------------------------- */ 
    // Populate RHS vector b = -L*u0 + g
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

    // Update ODE solution: u += delta_t*b1 * k1
    HYPRE_ParVectorAxpy(delta_t*butch.b[0], par_k1, *par_u);


    /* --------------------------------------------------------------------------------------------------------------------------------- */
    /* --- SOLVE for k2: [M + a22*delta_t*L(t_n+c1*delta_t)]*k2 = b == -L(t + c2*delta_t)*(u0 + delta_t*a21*k1) + g(t0 + c2*delta_t) --- */
    /* --------------------------------------------------------------------------------------------------------------------------------- */
    if (butch.num_stages > 1) {

        // Rebuild the spatial disc at time t0 + c2*delta_t ONLY if it's time dependent.
        if (m_isTimeDependent) {
            
            // Free these things if they've been set previously.
            if (built_L_new) {
                delete[] L_rowptr_new;
                delete[] L_colinds_new;
                delete[] L_data_new;
                delete[] g_data_new;
                delete[] dummy;
            }

            getSpatialDiscretization(comm, L_rowptr_new, L_colinds_new, L_data_new, g_data_new, dummy, ilower, iupper, spatialDOFs, t0 + butch.c[1], delta_t);

            temp_L_data = L_data_new; // Reassign pointer to L data.

            // Update HYPRE matrix's and vector's values.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_L, rows, L_colinds_new, L_data_new);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, g_data_new);
        }

        // Do we need to rebuild A?
        if (rebuild_A) {
            // Create data to initially populate A with.
            for (int i = 0; i < onProcSize; i++) {
                A_data[i] = butch.a[1][1]*delta_t*temp_L_data[i];
            }

            // Since the sparsity of A doesn't change with time, this call will overwrite all exisiting entries in it. 
            HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_L, rows, L_colinds, A_data);
            HYPRE_IJMatrixAddToValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data); // Add M to exisitng A.
        }

        // Populate RHS vector b = -L*(u0 + delta_t*a21*k1) + g
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[1][0], par_k1, par_temp); // temp += a21*k1
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

        // Update ODE solution: u += delta_t*b2 * k2
        HYPRE_ParVectorAxpy(delta_t*butch.b[1], par_k2, *par_u); 
    }


    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    /* --- SOLVE for k3: [M + a33*delta_t*L(t_n+c1*delta_t)]*k3 = -L(t + c3*delta_t)*(u0 + delta_t*a31*k1 + delta_t*a32*k2) + g(t0 + c3*delta_t) --- */
    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    if (butch.num_stages > 2) {

        // Rebuild the spatial disc at time t0 + c3*delta_t ONLY if it's time dependent.
        if (m_isTimeDependent) {
            delete[] L_rowptr_new;
            delete[] L_colinds_new;
            delete[] L_data_new;
            delete[] g_data_new;
            delete[] dummy;
            getSpatialDiscretization(comm, L_rowptr_new, L_colinds_new, L_data_new, g_data_new, dummy, ilower, iupper, spatialDOFs, t0 + butch.c[2], delta_t);

            temp_L_data = L_data_new;

            // Update HYPRE matrix's and vector's values.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_L, rows, L_colinds_new, L_data_new);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, g_data_new);
        }

        // Do we need to rebuild A?
        if (rebuild_A) {
            // Create data to initially populate A with.
            for (int i = 0; i < onProcSize; i++) {
                A_data[i] = butch.a[2][2]*delta_t*temp_L_data[i];
            }

            // Since the sparsity of A doesn't change with time, this call will overwrite all exisiting entries in it. 
            HYPRE_IJMatrixSetValues(A, onProcSize, cols_per_row_L, rows, L_colinds, A_data);
            HYPRE_IJMatrixAddToValues(A, onProcSize, cols_per_row_M, rows, M_colinds, M_data); // Add M to exisitng A.
        }

        // Populate RHS vector b = -L*(u0 + delta_t*a31*k1 + delta_t*a32*k2) + g
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[2][0], par_k1, par_temp); // temp += a31*k1
        HYPRE_ParVectorAxpy(butch.a[2][1], par_k2, par_temp); // temp += a32*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b = -L*temp + g

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
    
        // Update ODE solution: u += delta_t*b3 * k3;
        HYPRE_ParVectorAxpy(delta_t*butch.b[2], par_k3, *par_u); 
    }

      
    /*----------------- */
    /* --- Clean up --- */
    /* ---------------- */
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJMatrixDestroy(L);
    HYPRE_IJVectorDestroy(g);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(k1);

    delete[] rows;
    delete[] cols_per_row_M;
    delete[] cols_per_row_L;

    // Did we build these things?
    if (!mass) {
        delete[] M_rowptr;
        delete[] M_colinds;
        delete[] M_data;
    }
    if (built_L_new) {
        delete[] L_rowptr_new;
        delete[] L_colinds_new;
        delete[] L_data_new;
        delete[] g_data_new;
        delete[] dummy;
    }

    if (butch.num_stages > 1) {
        HYPRE_IJVectorDestroy(k2);
        HYPRE_IJVectorDestroy(temp);
    }
    if (butch.num_stages > 2) {
        HYPRE_IJVectorDestroy(k3);
    }  
}


/* ERK2/3 with mass matrix: 
    step solution of M*du/dt = -L(t)*u + g(t) from t0 to t0 + deltat.

-The RK updates are
    u = u0 + delta_t*(b1*k1 + b2*k2 + b3*k3), with
    M*k1 = -L(t0)*u0 + g(t0)
    M*k2 = -L(t + c2*delta_t)*(u0 + delta_t*a21*k1) + g(t0 + c2*delta_t)
    M*k3 = -L(t + c3*delta_t)*(u0 + delta_t*a31*k1 + delta_t*a32*k2) + g(t0 + c3*delta_t)
*/
void SpaceTimeMatrix::ERK(MPI_Comm comm, RK_butcher butch, HYPRE_ParVector * par_u, HYPRE_ParVector * par_u0, double t0, double delta_t, int ilower, int iupper, int * M_rowptr, int * M_colinds, double * M_data, int * L_rowptr, int * L_colinds, double * L_data, double * g_data) {

    /* ---------------------- */
    /* --- Initial set up --- */
    /* ---------------------- */

    // Initialize ODE soluton: u = u0
    HYPRE_ParVectorCopy(*par_u0, *par_u); 

    // The stage vectors to be found
    HYPRE_IJVector k1;
    HYPRE_ParVector par_k1;
    HYPRE_IJVector k2;
    HYPRE_ParVector par_k2;
    HYPRE_IJVector k3;
    HYPRE_ParVector par_k3;

    // Do we have a mass matrix M?
    int mass = 0;
    if (M_data != NULL) {
        mass = 1;
    }

    // Initialize RHS vector b.
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVectorCreate(comm, ilower, iupper, &b); 
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    // Initialize temp vector; it's used as a temporary vector in building later stage vectors.
    HYPRE_IJVector temp;
    HYPRE_ParVector par_temp;
    if (butch.num_stages > 1) {
        HYPRE_IJVectorCreate(comm, ilower, iupper, &temp); 
        HYPRE_IJVectorSetObjectType(temp, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(temp);
        HYPRE_IJVectorAssemble(temp);
        HYPRE_IJVectorGetObject(temp, (void **) &par_temp);
    }


    // Place spatial discretization and mass components into HYPRE objects.
    HYPRE_IJMatrix M; 
    HYPRE_ParCSRMatrix parcsr_M;
    HYPRE_IJMatrix L; 
    HYPRE_ParCSRMatrix parcsr_L;
    HYPRE_IJVector g;
    HYPRE_ParVector par_g;

    // Initialize matrices
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &M);
    HYPRE_IJMatrixSetObjectType(M, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(M);
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &L);
    HYPRE_IJMatrixSetObjectType(L, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(L);

    // Set matrix coefficients
    int onProcSize = iupper - ilower - 1; // TODO: Is this correct??
    int * rows = new int[onProcSize];
    int * cols_per_row_M = new int[onProcSize];
    int * cols_per_row_L = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row_M[i] = M_rowptr[i+1] - M_rowptr[i];
        cols_per_row_L[i] = L_rowptr[i+1] - L_rowptr[i];
    }
    HYPRE_IJMatrixSetValues(M, onProcSize, cols_per_row_M, rows, M_colinds, M_data);
    HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_L, rows, L_colinds, L_data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(M);
    HYPRE_IJMatrixGetObject(M, (void **) &parcsr_M);
    HYPRE_IJMatrixAssemble(L);
    HYPRE_IJMatrixGetObject(L, (void **) &parcsr_L);

    // Create g vector
    HYPRE_IJVectorCreate(comm, ilower, iupper, &g);
    HYPRE_IJVectorSetObjectType(g, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(g);
    HYPRE_IJVectorSetValues(g, onProcSize, rows, g_data);
    HYPRE_IJVectorAssemble(g);
    HYPRE_IJVectorGetObject(g, (void **) &par_g);

    // Do we need to rebuild L here?
    int rebuild_L = (m_isTimeDependent && butch.num_stages>1);

    // Components for if we need to rebuild spatial disc, as don't want to overwrite ones pass to function.
    int * L_rowptr_new;
    int * L_colinds_new;
    double * L_data_new;
    double * g_data_new; 
    double * dummy; // I don't think we need this here...
    int spatialDOFs;


    /* --------------------------------------------------- */
    /* --- SOLVE for k1: M*k1 = b == -L(t0)*u0 + g(t0) --- */
    /* --------------------------------------------------- */ 
    // Populate RHS vector b = -L*u0 + g
    hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, *par_u0, 1.0, par_g, par_b);

    // Set up and finalize k1, but don't set its values yet.
    HYPRE_IJVectorCreate(comm, ilower, iupper, &k1); 
    HYPRE_IJVectorSetObjectType(k1, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(k1);
    HYPRE_IJVectorAssemble(k1);
    HYPRE_IJVectorGetObject(k1, (void **) &par_k1);

    
    // Solve linear system for k1 if M != I.
    if (mass) {

        // TODO: What should the initial guess for k1 be?? Probably b?
        HYPRE_ParVectorCopy(par_b, par_k1); // Populate k1 with b as initial guess.

        // TODO: apply linear solve
        // HYPRE_Solve(comm, &parcsr_M, par_b, par_k1);
    
    // Update is explicit: copy values from b into k1.
    } else {
        HYPRE_ParVectorCopy(par_b, par_k1);
    }


    // Update ODE solution: u += delta_t*b1 * k1
    HYPRE_ParVectorAxpy(delta_t*butch.b[0], par_k1, *par_u);


    /* ------------------------------------------------------------------------------------------------ */
    /* --- SOLVE for k2: M*k2 = b == -L(t0 + c2*delta_t)*(u0 + delta_t*a21*k1) + g(t0 + c2*delta_t) --- */
    /* ------------------------------------------------------------------------------------------------ */
    if (butch.num_stages > 1) {
        // Rebuild the spatial disc at time t0 + c2*delta_t ONLY if it's time dependent.
        if (m_isTimeDependent) {
            getSpatialDiscretization(comm, L_rowptr_new, L_colinds_new, L_data_new, g_data_new, dummy, ilower, iupper, spatialDOFs, t0 + butch.c[1], delta_t);

            // Update HYPRE matrix's and vector's values.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_L, rows, L_colinds_new, L_data_new);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, g_data_new);
        }

        // Populate RHS vector b = -L*(u0 + a21*k1) + g
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[1][0], par_k1, par_temp); // temp += a21*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b = -L*temp + g

        // Set up and finalize k2, but don't set its values yet.
        HYPRE_IJVectorCreate(comm, ilower, iupper, &k2); 
        HYPRE_IJVectorSetObjectType(k2, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(k2);
        HYPRE_IJVectorAssemble(k2);
        HYPRE_IJVectorGetObject(k2, (void **) &par_k2);

        
        // Solve linear system for k2 if M != I.
        if (mass) {

            // TODO: What should the initial guess for k2 be?? Probably b?
            HYPRE_ParVectorCopy(par_b, par_k2); // Populate k2 with b as initial guess.

            // TODO: Apply linear solver
            // HYPRE_Solve(comm, &parcsr_M, par_b, par_k2);
        
        // Update is explicit: copy values from b into k2.
        } else {
            HYPRE_ParVectorCopy(par_b, par_k2);
        }

        // Update ODE solution: u += delta_t*b2 * k2
        HYPRE_ParVectorAxpy(delta_t*butch.b[1], par_k2, *par_u); 
    }


    /* ----------------------------------------------------------------------------------------------------------------- */
    /* --- SOLVE for k3: M*k3 = b == -L(t0 + c3*delta_t)*(u0 + delta_t*a31*k1 + delta_t*a32*k2) + g(t0 + c3*delta_t) --- */
    /* ----------------------------------------------------------------------------------------------------------------- */
    if (butch.num_stages > 2) {

        // Rebuild the spatial disc at time t0 + c3*delta_t ONLY if it's time dependent.
        if (m_isTimeDependent) {
            delete[] L_rowptr_new;
            delete[] L_colinds_new;
            delete[] L_data_new;
            delete[] g_data_new;
            delete[] dummy;
            getSpatialDiscretization(comm, L_rowptr_new, L_colinds_new, L_data_new, g_data_new, dummy, ilower, iupper, spatialDOFs, t0 + butch.c[2], delta_t);

            // Update HYPRE matrix's and vector's values.
            HYPRE_IJMatrixSetValues(L, onProcSize, cols_per_row_L, rows, L_colinds_new, L_data_new);
            HYPRE_IJVectorSetValues(g, onProcSize, rows, g_data_new);
        }

        // Populate RHS vector b = -L*(u0 + a31*k1 + a32*k2) + g
        HYPRE_ParVectorCopy(*par_u0, par_temp); // temp <- u0
        HYPRE_ParVectorAxpy(butch.a[2][0], par_k1, par_temp); // temp += a31*k1
        HYPRE_ParVectorAxpy(butch.a[2][1], par_k2, par_temp); // temp += a32*k1
        hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, parcsr_L, par_temp, 1.0, par_g, par_b); // b = -L*temp + g

        // Set up and finalize k3, but don't set its values yet.
        HYPRE_IJVectorCreate(comm, ilower, iupper, &k3); 
        HYPRE_IJVectorSetObjectType(k3, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(k3);
        HYPRE_IJVectorAssemble(k3);
        HYPRE_IJVectorGetObject(k3, (void **) &par_k3);

        
        // Solve linear system for k3 if M != I.
        if (mass) {
            // TODO: What should the initial guess for k3 be?? Probably b?
            HYPRE_ParVectorCopy(par_b, par_k3); // Populate k3 with b as initial guess.

            // TODO: Apply linear solve
            // HYPRE_Solve(comm, &parcsr_M, par_b, par_k3);
        
        // Update is explicit: copy values from b into k3.
        } else {
            HYPRE_ParVectorCopy(par_b, par_k3);
        }

        // Update ODE solution: u += delta_t*b3 * k3;
        HYPRE_ParVectorAxpy(delta_t*butch.b[2], par_k3, *par_u); 
    }

      
    /*----------------- */
    /* --- Clean up --- */
    /* ---------------- */
    HYPRE_IJMatrixDestroy(M);
    HYPRE_IJMatrixDestroy(L);
    HYPRE_IJVectorDestroy(g);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(k1);

    delete[] rows;
    delete[] cols_per_row_M;
    delete[] cols_per_row_L;

    if (rebuild_L) {
        delete[] L_rowptr_new;
        delete[] L_colinds_new;
        delete[] L_data_new;
        delete[] g_data_new;
        delete[] dummy;
    }

    if (butch.num_stages > 1) {
        HYPRE_IJVectorDestroy(k2);
        HYPRE_IJVectorDestroy(temp);
    }
    if (butch.num_stages > 2) {
        HYPRE_IJVectorDestroy(k3);
    }  
}





/* ------------------------------------------------------------------------- */
/* ----------------- More than one time step per processor ----------------- */
/* ------------------------------------------------------------------------- */
/* First-order BDF implicit scheme (Backward Euler / 1st-order Adams-Moulton). */
void SpaceTimeMatrix::BDF1(int *&rowptr, int *&colinds, double *&data,
                           double *&B, double *&X, int &onProcSize)
{

    // B is the RHS vector. X is the solution vector, populated here with initial guess.

    int tInd0 = m_globRank*m_ntPerProc; // Index of first time on processor
    int tInd1 = tInd0 + m_ntPerProc -1; // Index of last time on processor

    // Get spatial discretization for first time step on this processor: t = m_dt * tInd0. 
    // Spatial disc is in CSR format and called T.
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, m_dt*tInd0, m_dt);
    int nnzPerTime = T_rowptr[spatialDOFs];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = m_ntPerProc * spatialDOFs;
    int procNnz    = m_ntPerProc * (spatialDOFs + nnzPerTime);     // nnzs on this processor
    if (tInd0 == 0) procNnz -= spatialDOFs;

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
                                     X0, spatialDOFs, m_dt*ti, m_dt);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;

        // At time t=0, only have spatial discretization block
        if (ti == 0) {
            
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + L), otherwise data is L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + T_data[j];
                    }
                    else {
                        data[dataInd] = T_data[j];
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = B0[i];
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

                    // Add identity to diagonal, (I + L), otherwise data is L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + T_data[j];
                    }
                    else {
                        data[dataInd] = T_data[j];
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = B0[i];
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


/* Second-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF2(int *&rowptr, int *&colinds, double *&data,
                           double *&B, double *&X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, m_dt*tInd0, m_dt);
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = m_ntPerProc * spatialDOFs;
    int procNnz    = m_ntPerProc * (2*spatialDOFs + nnzPerTime);     // nnzs on this processor
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
                                     X0, spatialDOFs, m_dt*ti, m_dt);
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 2.0*T_data[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*T_data[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*B0[i] / 3.0;
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 3L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 2.0*T_data[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*T_data[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*B0[i] / 3.0;
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 2.0*T_data[j] / 3.0;
                    }
                    else {
                        data[dataInd] = 2.0*T_data[j] / 3.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 2.0*B0[i] / 3.0;
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


/* Third-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF3(int *&rowptr, int *&colinds, double *&data,
                           double *&B, double *&X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0,
                             X0, spatialDOFs, m_dt*tInd0, m_dt);
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize = m_ntPerProc * spatialDOFs;
    int procNnz    = m_ntPerProc * (3*spatialDOFs + nnzPerTime);     // nnzs on this processor
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
                                     X0, spatialDOFs, m_dt*ti, m_dt);
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 3L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
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

                    // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                    }
                    else {
                        data[dataInd] = 6.0*T_data[j] / 11.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 6.0*B0[i] / 11.0;
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


/* Second-order Adams-Moulton implicit scheme (trapezoid method). */
void SpaceTimeMatrix::AM2(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

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
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, Bi, Xi,
                                 spatialDOFs, m_dt*(tInd0-1), m_dt);
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, Bi, Xi,
                                 spatialDOFs, m_dt*tInd0, m_dt);
    }
    if (!m_isTimeDependent) {
        T_rowptr_1 = T_rowptr;
        T_colinds_1 = T_colinds;
        T_data_1 = T_data;
        Bi_1 = Bi;
        Xi_1 = Xi;   
    }
    int nnzPerTime = T_rowptr[spatialDOFs];    

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
                                     Xi, spatialDOFs, m_dt*ti, m_dt);
        }

        // At time t=0, only have spatial discretization block.
        if (ti == 0) {
            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + L), otherwise data is L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + T_data[j] / 2.0;
                    }
                    else {
                        data[dataInd] = T_data[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = Bi[i] / 2.0;
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

                    // Add identity to diagonal, (-I + L/2), otherwise data is L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds_1[j]) {
                        data[dataInd] = -1 + T_data_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = T_data_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd + T_colinds_1[j];
                    dataInd += 1;
                }

                // Add this row of spatial discretization to diagonal block
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                    // Add identity to diagonal, (I + L/2), otherwise data is L/2
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = 1 + T_data[j] / 2.0;
                    }
                    else {
                        data[dataInd] = T_data[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusDiag + T_colinds[j];
                    dataInd += 1;
                }

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = (Bi[i] + Bi_1[i]) / 2.0;
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
    if (m_isTimeDependent) {
        delete[] T_rowptr_1;
        delete[] T_colinds_1;
        delete[] T_data_1;
        delete[] Bi_1;
        delete[] Xi_1;   
    }
}


/* First-order Adams-Bashforth explicit scheme (Forward Euler). */
void SpaceTimeMatrix::AB1(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    double *B0;
    double *X0;
    int spatialDOFs;
    if (tInd0 > 0) {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0,
                                 spatialDOFs, m_dt*(tInd0-1), m_dt);
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0,
                                 spatialDOFs, m_dt*tInd0, m_dt);
    }
    int nnzPerTime = T_rowptr[spatialDOFs];    
    
    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (spatialDOFs + nnzPerTime);     // nnzs on this processor
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
                                     X0, spatialDOFs, m_dt*ti, m_dt);
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

                    // Subtract identity to diagonal, (I + L), otherwise data is L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds[j]) {
                        data[dataInd] = -1 + T_data[j];
                    }
                    else {
                        data[dataInd] = T_data[j];
                    }
                    colinds[dataInd] = colPlusOffd + T_colinds[j];
                    dataInd += 1;
                }

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = B0[i];
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


/* Second-order Adams-Bashforth explicit scheme. */
void SpaceTimeMatrix::AB2(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &onProcSize)
{
    int tInd0 = m_globRank*m_ntPerProc;
    int tInd1 = tInd0 + m_ntPerProc -1;

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
                                 Bi_1, Xi_1, spatialDOFs, 0, m_dt);
    }
    else {
        getSpatialDiscretization(T_rowptr_1, T_colinds_1, T_data_1, Bi_1,
                                 Xi_1, spatialDOFs, m_dt*(tInd0-2), m_dt);
    }
    if (!m_isTimeDependent) {
        T_rowptr_2 = T_rowptr_1;
        T_colinds_2 = T_colinds_1;
        T_data_2 = T_data_1;
        Bi_2 = Bi_1;
        Xi_2 = Xi_1;   
    }

    int nnzPerTime = T_rowptr_1[spatialDOFs];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (2*nnzPerTime + spatialDOFs);     // nnzs on this processor
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
                                     Xi_1, spatialDOFs, m_dt*(ti-1), m_dt);
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

                    // Add identity to diagonal, (I + L), otherwise data is L
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds_1[j]) {
                        data[dataInd] = -1 + 3.0*T_data_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = 3.0*T_data_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd_1 + T_colinds_1[j];
                    dataInd += 1;
                }

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

                // Add right hand side and initial guess for this row to global problem
                B[thisRow] = 3.0*Bi_1[i] / 2.0;
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

                    // Add spatial block -Lu_{ti-2}
                    data[dataInd] = -T_data_2[j] / 2.0;
                    colinds[dataInd] = colPlusOffd_2 + T_colinds_2[j];
                    dataInd += 1;
                }

                // Add this row of spatial discretization to first off-diagonal block
                for (int j=T_rowptr_1[i]; j<T_rowptr_1[i+1]; j++) {

                    // Add identity to diagonal, (I + L/2), otherwise data is L/2
                    //     - NOTE: assume here that spatial disc. has nonzero diagonal
                    if (i == T_colinds_1[j]) {
                        data[dataInd] = -1 + 3.0*T_data_1[j] / 2.0;
                    }
                    else {
                        data[dataInd] = 3.0*T_data_1[j] / 2.0;
                    }
                    colinds[dataInd] = colPlusOffd_1 + T_colinds_1[j];
                    dataInd += 1;
                }

                // Add identity as diagonal block, u_ti
                colinds[dataInd] = colPlusDiag + i;
                data[dataInd] = 1.0;
                dataInd += 1;

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
void SpaceTimeMatrix::BDF1(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*m_timeInd, m_dt);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzPerTime;
    else procNnz = nnzPerTime + procRows;

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

    // At time t=0, only have spatial discretization block
    if (m_timeInd == 0) {
        
        // Loop over each on-processor row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + L), otherwise data is L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + T_data[j];
                }
                else {
                    data[dataInd] = T_data[j];
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= m_dt;

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    else {
        // Loop over each row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {
            // Add off-diagonal block, -u_{i-1}
            colinds[dataInd] = colPlusOffd + i;
            data[dataInd] = -1.0;
            dataInd += 1;

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + L), otherwise data is L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + T_data[j];
                }
                else {
                    data[dataInd] = T_data[j];
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= m_dt;

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


/* Second-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF2(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*m_timeInd, m_dt);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    
    
    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzPerTime;
    else if (m_timeInd == 1) procNnz = nnzPerTime + procRows;
    else procNnz = nnzPerTime + 2*procRows;

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

    // At time t=0, only have spatial discretization block
    if (m_timeInd == 0) {
        
        // Loop over each row in spatial discretization at time ti
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 2.0*T_data[j] / 3.0;
                }
                else {
                    data[dataInd] = 2.0*T_data[j] / 3.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0/3.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    // At time t=1, only have 1 off-diagonal block
    else if (m_timeInd == 1) {
        // Loop over each row in spatial discretization at time t1
        for (int i=0; i<procRows; i++) {
            // Add off-diagonal block, -u_{i-1}
            colinds[dataInd] = colPlusOffd_1 + i;
            data[dataInd] = -4.0/3.0;
            dataInd += 1;

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2L/3), otherwise data is 3L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 2.0*T_data[j] / 3.0;
                }
                else {
                    data[dataInd] = 2.0*T_data[j] / 3.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0/3.0);

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

                // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 2.0*T_data[j] / 3.0;
                }
                else {
                    data[dataInd] = 2.0*T_data[j] / 3.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (2.0/3.0);

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


/* Third-order BDF implicit scheme. */
void SpaceTimeMatrix::BDF3(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*m_timeInd, m_dt);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    
    
    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzPerTime;
    else if (m_timeInd == 1) procNnz = nnzPerTime + procRows;
    else if (m_timeInd == 2) procNnz = nnzPerTime + 2*procRows;
    else procNnz = nnzPerTime + 3*procRows;

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

    // At time t=0, only have spatial discretization block
    if (m_timeInd == 0) {
        
        // Loop over each row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + T_rowptr[i+1] - T_rowptr[i];
        }
    }
    // At time t=1, only have 1 off-diagonal block
    else if (m_timeInd == 1) {
        // Loop over each row in spatial discretization at time t1
        for (int i=0; i<procRows; i++) {
            // Add off-diagonal block, -u_{i-1}
            colinds[dataInd] = colPlusOffd_1 + i;
            data[dataInd] = -18.0/11.0;
            dataInd += 1;

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + 2L/3), otherwise data is 3L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

            // Total nonzero for this row on processor is one for off-diagonal block
            // and the total nnz in this row of T (T_rowptr[i+1] - T_rowptr[i]).
            rowptr[i+1] = rowptr[i] + 1 + (T_rowptr[i+1] - T_rowptr[i]);
        }
    }
    else if (m_timeInd == 2) {
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

                // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

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

                // Add identity to diagonal, (I + 2L/3), otherwise data is 2L/3
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + 6.0*T_data[j] / 11.0;
                }
                else {
                    data[dataInd] = 6.0*T_data[j] / 11.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            B[i] *= (6.0/11.0);

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


/* Second-order Adams-Moulton implicit scheme (trapezoid method). */
void SpaceTimeMatrix::AM2(int *&rowptr, int *&colinds, double *&data,
                          double *&B, double *&X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for previous time step, or first step if m_timeInd0=0
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
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                             B, X, localMinRow, localMaxRow, spatialDOFs,
                             m_dt*(m_timeInd), m_dt);
    int procRows = localMaxRow - localMinRow + 1;
    procNnz = T_rowptr[procRows];

    // Get discretization at time ti-1 for Adams-Moulton if m_timeInd!=0. 
    if (m_timeInd > 0) {
        int localMinRow_1;
        int localMaxRow_1;
        getSpatialDiscretization(m_spatialComm, T_rowptr_1, T_colinds_1, T_data_1,
                                 B, X, localMinRow_1, localMaxRow_1,
                                 spatialDOFs, m_dt*(m_timeInd-1), m_dt);
     
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

    // At time t=0, only have spatial discretization at t0.
    if (m_timeInd == 0) {
        // Loop over each row in spatial discretization at time t0
        for (int i=0; i<procRows; i++) {

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + L), otherwise data is L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + T_data[j] / 2.0;
                }
                else {
                    data[dataInd] = T_data[j] / 2.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            //  TODO: FIX
            // B[i] = Bi[i] / 2.0;
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

                // Add identity to diagonal, (-I + L/2), otherwise data is L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds_1[j]) {
                    data[dataInd] = -1 + T_data_1[j] / 2.0;
                }
                else {
                    data[dataInd] = T_data_1[j] / 2.0;
                }
                colinds[dataInd] = colPlusOffd + T_colinds_1[j];
                dataInd += 1;
            }

            // Add this row of spatial discretization to diagonal block
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {

                // Add identity to diagonal, (I + L/2), otherwise data is L/2
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if ((i+localMinRow) == T_colinds[j]) {
                    data[dataInd] = 1 + T_data[j] / 2.0;
                }
                else {
                    data[dataInd] = T_data[j] / 2.0;
                }
                colinds[dataInd] = colPlusDiag + T_colinds[j];
                dataInd += 1;
            }

            // Add right hand side and initial guess for this row to global problem
            //  TODO: FIX
            // B[i] = (Bi[i] + Bi_1[i]) / 2.0;
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


/* First-order Adams-Bashforth explicit scheme (Forward Euler). */
void SpaceTimeMatrix::AB1(int *&rowptr, int *&colinds,  double *&data,
                          double *&B, double *&X, int &localMinRow,
                          int &localMaxRow, int &spatialDOFs)
{
    // Get spatial discretization for first time step on this processor
    int *T_rowptr;
    int *T_colinds;
    double *T_data;
    if (m_timeInd == 0) {    
        getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 m_dt*m_timeInd, m_dt);
    }
    else {
        getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 m_dt*(m_timeInd-1), m_dt);
    }
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int procNnz;
    if (m_timeInd == 0) procNnz = procRows;
    else procNnz = nnzPerTime + procRows;

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

    // At time t=0, only have identity block on diagonal
    if (m_timeInd == 0) {
        
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

                // Subtract identity to diagonal, (I + L), otherwise data is L
                //     - NOTE: assume here that spatial disc. has nonzero diagonal
                if (i == T_colinds[j]) {
                    data[dataInd] = -1 + T_data[j];
                }
                else {
                    data[dataInd] = T_data[j];
                }
                colinds[dataInd] = colPlusOffd + T_colinds[j];
                dataInd += 1;
            }

            // Add identity as diagonal block, u_ti
            colinds[dataInd] = colPlusDiag + i;
            data[dataInd] = 1.0;
            dataInd += 1;

            // Add right hand side and initial guess for this row to global problem
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

