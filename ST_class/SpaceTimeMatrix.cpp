#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include "SpaceTimeMatrix.hpp"
#include <iomanip> // Need this for std::setprecision
#include <cmath> // OAK: I have to include this... There is some ambiguity with "abs". 
// Something to do with how new OSX is setup now. But on stack exchage some people say this 
// is not a good solution. But I think it'll do for now...

// TODO:
//      - Add A-stable schemes, compare whether this matters vs. L-stable (it does for MGRiT)
//      - Add isTimeDependent option
//      - Add hypre timing for setup and solve (see ij.c)
//      - Plotting code for spatial solution
//      - namespace for RK tableaux/AMG parameters? Keep structs separate from class? 


// TODO remove these functions when implemented in all spatial discretizations..
void SpaceTimeMatrix::getSpatialDiscretizationL(const MPI_Comm &spatialComm, int* &A_rowptr, 
                                            int* &A_colinds, double* &A_data,
                                            double* &U0, bool getU0, 
                                            int &localMinRow, int &localMaxRow, 
                                            int &spatialDOFs,
                                            double t, int &bsize)  
{
    std::cout << "WARNING: The `getSpatialDiscretizationL' has not been implemented in the derived spatial discretization class" << '\n';
    MPI_Finalize();
    exit(1);
}                                                                            
void SpaceTimeMatrix::getSpatialDiscretizationG(const MPI_Comm &spatialComm, double* &G, 
                                            int &localMinRow, int &localMaxRow,
                                            int &spatialDOFs, double t)
{
    std::cout << "WARNING: The `getSpatialDiscretizationG' has not been implemented in the derived spatial discretization class" << '\n';
    MPI_Finalize();
    exit(1);
}   
void SpaceTimeMatrix::getSpatialDiscretizationL(int* &A_rowptr, 
                                            int* &A_colinds, double* &A_data,
                                            double* &U0, bool getU0, 
                                            int &spatialDOFs,
                                            double t, int &bsize)  
{
    std::cout << "WARNING: The `getSpatialDiscretizationL' has not been implemented in the derived spatial discretization class" << '\n';
    MPI_Finalize();
    exit(1);
}                                                        
void SpaceTimeMatrix::getSpatialDiscretizationG(double* &G, int &spatialDOFs, double t)
{
    std::cout << "WARNING: The `getSpatialDiscretizationG' has not been implemented in the derived spatial discretization class" << '\n';
    MPI_Finalize();
    exit(1);
}                                                                          




SpaceTimeMatrix::SpaceTimeMatrix(MPI_Comm globComm, int timeDisc,
                                 int nt, double dt, bool pit)
    : m_globComm{globComm}, m_timeDisc{timeDisc}, m_nt{nt},
      m_dt{dt}, m_pit{pit}, m_solver(NULL), m_gmres(NULL), m_bij(NULL), m_xij(NULL), m_Aij(NULL),
      m_M_exists(true),
      m_M_rowptr(NULL), m_M_colinds(NULL), m_M_data(NULL), m_rebuildSolver(false),
      m_bsize(1), m_hmin(-1), m_hmax(-1), m_is_ERK(false), m_is_DIRK(false), m_is_SDIRK(false),
      m_spatialComm(NULL), m_L_isTimedependent(true), m_g_isTimedependent(true)
{
    
    m_isTimeDependent = true; // TODO : this will need to be removed later... when L and G are treated separately
    
    // Get number of processes
    MPI_Comm_rank(m_globComm, &m_globRank);
    MPI_Comm_size(m_globComm, &m_numProc);


    // Get RK Butcher tabelaux
    GetButcherTableaux();
    // Set member variables
    
    /* ---------------------------------------------- */
    /* ------ Solving global space-time system ------ */
    /* ---------------------------------------------- */
    if (m_pit) {
        // Check that number of time steps times number of stages divides the number MPI processes or vice versa.
        /* ------ Temporal + spatial parallelism ------ */
        if (m_numProc > (m_nt * m_s_butcher)) {
            if (m_globRank == 0) {
                std::cout << "Space-time system: Spatial + temporal parallelism!\n";    
            }
            
            m_useSpatialParallel = true;
            if (m_numProc % (m_nt * m_s_butcher) != 0) {
                if (m_globRank == 0) {
                    std::cout << "Error: number of processes " << m_numProc << " does not divide number of time points (" << m_nt << ") * number of RK stages (" << m_s_butcher << ") == " << m_nt * m_s_butcher << "\n";
                }
                MPI_Finalize();
                exit(1);
                //return;
            }
            else {
                m_spatialCommSize = m_numProc / (m_nt * m_s_butcher);
                m_Np_x = m_spatialCommSize; // TODO : remove. 
            }
             
            // Set up communication group for spatial discretizations.
            m_timeInd = m_globRank / m_Np_x; // TODO. Delete this...
            m_DOFInd = m_globRank / m_Np_x;
            MPI_Comm_split(m_globComm, m_DOFInd, m_globRank, &m_spatialComm);
            MPI_Comm_rank(m_spatialComm, &m_spatialRank);
            MPI_Comm_size(m_spatialComm, &m_spCommSize);


        }
        /* ------ Temporal parallelism only ------ */
        else {
            if (m_globRank == 0) {
                if (m_numProc > 1) std::cout << "Space-time system: Temporal parallelism only!\n";    
                else std::cout << "Space-time system: No parallelism!\n";    
            }
            
            m_useSpatialParallel = false;
            if ( (m_nt * m_s_butcher) % m_numProc  != 0) {
                if (m_globRank == 0) {
                    std::cout << "Error: number of time points (" << m_nt << ") * number of RK stages (" << m_s_butcher << ") == " << m_nt * m_s_butcher << " does not divide number of processes " << m_numProc << "\n";
                }
                MPI_Finalize();
                exit(1);
                //return;
            }
            m_nDOFPerProc = (m_nt * m_s_butcher) / m_numProc; // Number of temporal DOFs per proc, be they solution and/or stage DOFs
            m_ntPerProc = m_nt / m_numProc; //  TOOD: delete... This variable is for the old implementation...     
        }
    
    
    /* -------------------------------------- */
    /* ------ Sequential time stepping ------ */
    /* -------------------------------------- */
    } else {
        if (m_numProc > 1)  {
            if (m_globRank == 0) {
                std::cout << "Time-stepping: Spatial parallelism!\n";    
            }
            m_useSpatialParallel = true;
            
            // Update spatial communicator variable to be those of global communicator.
            m_spatialComm     = m_globComm;
            m_spatialRank     = m_globRank;
            m_spatialCommSize = m_numProc;
            m_Np_x = m_spatialCommSize; // TODO delete...
        } else {
            std::cout << "Time-stepping: No parallelism!\n";    
            m_useSpatialParallel = false;
        }  
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


/* Call appropiate sequential time-stepping routine */
void SpaceTimeMatrix::RKSolve(double solve_tol, int max_iter, int printLevel,
                                bool binv_scale, int precondition, int AMGiters)
{
    if (m_is_ERK) {
        ERKSolve(solve_tol, max_iter, printLevel, binv_scale, precondition, AMGiters);
    } else if (m_is_DIRK) {
        DIRKSolve(solve_tol, max_iter, printLevel, binv_scale, precondition, AMGiters);
    } else {
        std::cout << "WARNING: Only ERK and DIRK solvers available" << '\n';
        MPI_Finalize();
        exit(1);
    }
}


/* Sequential time-stepping routine for general DIRK schemes */
void SpaceTimeMatrix::DIRKSolve(double solve_tol, int max_iter, int printLevel,
                                 bool binv_scale, int precondition, int AMGiters) 
{   
    /* ---------------------------------------------------------------------- */
    /* ------------------------ Setup/initialization ------------------------ */
    /* ---------------------------------------------------------------------- */
    double t = 0.0;         // Initial time

    HYPRE_ParVector    u;   // Solution vector
    HYPRE_IJVector     uij;
    HYPRE_ParVector    g;   // Spatial discretization vector
    HYPRE_IJVector     gij = NULL;
    HYPRE_ParCSRMatrix L;   // Spatial discretization matrix  
    HYPRE_IJMatrix     Lij = NULL;
    HYPRE_ParCSRMatrix DIRK_matrix; // Matrix to be inverted in linear solve
    HYPRE_IJMatrix     DIRK_matrixij;

    // Place-holder vectors
    std::vector<HYPRE_ParVector> vectors;
    std::vector<HYPRE_IJVector>  vectorsij;
    int numVectors = m_s_butcher + 2; // Have s stage vectors + 2 temporary vectors

    // Get initial condition and initialize place-holder vectors
    RKInitializeHypreVectors(u, uij, numVectors, vectors, vectorsij);

    // Shallow copy vectors into variables with meaningful names
    HYPRE_ParVector              b1   = vectors[0];  // Temporary vector
    HYPRE_IJVector               b1ij = vectorsij[0];
    HYPRE_ParVector              b2   = vectors[1];  // Temporary vector
    HYPRE_IJVector               b2ij = vectorsij[1];
    std::vector<HYPRE_ParVector> k; // Stage vectors
    std::vector<HYPRE_IJVector>  kij;    
    k.reserve(m_s_butcher);
    kij.reserve(m_s_butcher);
    for (int i = 0; i < m_s_butcher; i++) {
        kij[i] = vectorsij[i+2];
        k[i]   = vectors[i+2];
    }

    // Mass matrix components
    int    * M_rowptr;
    int    * M_colinds;
    double * M_data;
    double * M_scaled_data;
    int    * M_rows;
    int    * M_cols_per_row;
    int      ilower;
    int      iupper;
    int      onProcSize;

    // For monitoring convergence of linear solver
    int    num_iters;
    double rel_res_norm;

    // Is it necessary to build spatial disc. matrix/DIRK matrix more than once?
    bool rebuildMatrix = ((m_L_isTimedependent) || (!m_is_SDIRK));


    /* ------------------------------------------------------------ */
    /* ------------------------ Time march ------------------------ */
    /* ------------------------------------------------------------ */
    // Take nt-1 steps
    int step = 0;
    for (step = 0; step < m_nt-1; step++) {
        /* -------------- Build RHS vector, b2, in linear system (M+a_ii*dt*L)*k[i]=b2 -------------- */
        for (int i = 0; i < m_s_butcher; i++) {
            // Compute spatial discretization at t + c[i]*dt
            // Solution-independent term
            if (m_g_isTimedependent || (i == 0 && step == 0)) {
                //HYPRE_IJVectorDestroy(gij);
                RKGetHypreSpatialDiscretizationG(g, gij, t + m_dt * m_c_butcher[i]);
            }
            // Solution-dependent term
            if (rebuildMatrix || (i == 0 && step == 0)) {
                //HYPRE_IJMatrixDestroy(Lij);
                RKGetHypreSpatialDiscretizationL(L, Lij, t + m_dt * m_c_butcher[i]);
            } 
            
            // Assemble RHS of linear system in b2
            HYPRE_ParVectorCopy(u, b1); // b1 <- u
            for (int j = 0; j < i; j++) {
                double temp = m_dt * m_A_butcher[i][j];
                if (temp != 0.0) HYPRE_ParVectorAxpy(temp, k[j], b1); // b1 <- b1 + dt*aij*k[j]
            }
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, L, b1, 1.0, g, b2); // b2 <- -L*b1 + g 
            
            
            // Set inital guess at ith stage
            HYPRE_ParVectorCopy(b2, k[i]); // Stage is RHS of linear system
            
            //HYPRE_ParVectorCopy(u, k[i]); // Stage is u at start of current interval
            
            // See Carpenter p.55 (115). But assumes L is evaluated at t which is not the case if it's time dependent
            // double temp2 = m_dt * m_c_butcher[i];
            // hypre_ParCSRMatrixMatvecOutOfPlace(-temp2, L, u, temp2, g, k[i]); // k[i] <- dt*ci*[ -L*u + g ]
            // HYPRE_ParVectorAxpy(1.0, u, k[i]); // k[i] <- k[i] + u
            
            
            /* -------------- Solve linear system, (M+a_ii*dt*L)*k[i]=b2, for ith stage vector, k[i] -------------- */
            // Get components of mass matrix only after spatial discretization has been assembled for the first time
            // Mass matrix is NOT time dependent: Only needs to be assembled once
            if ((step == 0) && (i == 0)) {
                getMassMatrix(M_rowptr, M_colinds, M_data);
                // Get rows this process owns from M; assumes rows of M and L are partitioned the same in memory
                int jdummy1, jdummy2;
                HYPRE_IJMatrixGetLocalRange(Lij, &ilower, &iupper, &jdummy1, &jdummy2);
                onProcSize     = iupper - ilower + 1;
                M_rows         = new int[onProcSize];
                M_cols_per_row = new int[onProcSize];
                for (int rowIdx = 0; rowIdx < onProcSize; rowIdx++) {
                    M_rows[rowIdx]         = ilower + rowIdx;
                    M_cols_per_row[rowIdx] = M_rowptr[rowIdx+1] - M_rowptr[rowIdx];
                }
                M_scaled_data = new double[M_rowptr[onProcSize]]; // Temporary vector to use below...
            }
            
            // Throw error if there is a zero diagonal entry in A!
            if (m_A_butcher[i][i] == 0.0) {
                std::cout << "WARNING: DIRK solver not implemented to handle Butcher matrix A with 0s on diagonal!" << '\n';
                MPI_Finalize();
                exit(1);
            }
            
            // Scale RHS vector by 1/dt*a_ii for the moment 
            double temp = 1.0/(m_dt * m_A_butcher[i][i]); 
            HYPRE_ParVectorScale(temp, b2); // b2 <- b2/(dt*a_ii)
            
            // Rescale mass matrix data by 1/dt*a_ii; only need to do this once if using SDIRK 
            if (!m_is_SDIRK || ((step == 0) && (i == 0))) {
                for (int dataInd = 0; dataInd < M_rowptr[onProcSize]; dataInd++) {
                    M_scaled_data[dataInd] = temp * M_data[dataInd]; // M <- M / (dt*a_ii)
                }
            }
            
            // Get DIRK matrix, DIRK_matrix <- M + a_ii*dt*L
            // Build DIRK matrix once only if L time independent, and using SDIRK
            if (!rebuildMatrix) {
                // Build DIRK matrix on first iteration only
                if ((step == 0) && (i == 0)) { 
                    RKGetHypreSpatialDiscretizationL(DIRK_matrix, DIRK_matrixij, t);
                    HYPRE_IJMatrixAddToValues(DIRK_matrixij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_scaled_data);            
                }
            // Reuse/update L since it's rebuilt at next iteration, DIRK_matrix <- L <- M + a_ii*dt*L    
            } else {
                HYPRE_IJMatrixAddToValues(Lij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_scaled_data);            
                DIRK_matrix   = L;
                DIRK_matrixij = Lij;
                // DIRK matrix has changed, will need to rebuild linear solver 
                // (this variable is reset to false when the solver (re)built)
                m_rebuildSolver = true; // TODO : make this optional/have a frequency with which it's rebuilt!
            }
            
            
            
            
            // Point member variables to local variables so linear solver can access them
            m_A = DIRK_matrix;
            m_x = k[i]; // Initial guess at solution is value from previous time step
            m_b = b2;
            
            if ((printLevel > 0) && (m_globRank == 0)) {
                std::cout << "Time step " << step+1 << "/" << m_nt-1 << ": Solving for stage " << i+1 << "/" << m_s_butcher << '\n';
                std::cout << "-----------------------------------------\n\n";
            }
            
            // Solve linear system and get convergence statistics
            if (precondition) {
                SolveGMRES(solve_tol, max_iter, printLevel, true, precondition, AMGiters);
                HYPRE_GMRESGetNumIterations(m_gmres, &num_iters);
                HYPRE_GMRESGetFinalRelativeResidualNorm(m_gmres, &rel_res_norm);
            } else {
                SolveAMG(solve_tol, max_iter, printLevel);
                HYPRE_BoomerAMGGetNumIterations(m_solver, &num_iters);
                HYPRE_BoomerAMGGetFinalRelativeResidualNorm(m_solver, &rel_res_norm);
            }
                
            // Ensure desired tolerance was reached in allowable number of iterations, otherwise quit
            if (rel_res_norm > solve_tol) {
                std::cout << "=================================\n =========== WARNING ===========\n=================================\n";
                std::cout << "Time step " << step+1 << "/" << m_nt-1 << ": Solving for stage " << i+1 << "/" << m_s_butcher << '\n';
                std::cout << "Tol after " << num_iters << " iters (max iterations) = " << rel_res_norm << " > desired tol = " << solve_tol << "\n\n";
                MPI_Finalize();
                exit(1);
            }
        }

        // Sum solution
        for (int i = 0; i < m_s_butcher; i++)  {
            double temp = m_dt * m_b_butcher[i];
            if (temp != 0.0) HYPRE_ParVectorAxpy(temp, k[i], u); // u <- u + dt*k[i]*k[i]; 
        }
        t += m_dt; // Increment time
    }


    /* ---------------------------------------------------------- */
    /* ------------------------ Clean up ------------------------ */
    /* ---------------------------------------------------------- */
    for (int i = 0; i < vectors.size(); i++) {
        HYPRE_IJVectorDestroy(vectorsij[i]);
    }
    if (step > 0) {
        HYPRE_IJVectorDestroy(gij);
        HYPRE_IJMatrixDestroy(Lij);
        if (!rebuildMatrix) HYPRE_IJMatrixDestroy(DIRK_matrixij); // Clean up extra matrix if we used it
        m_A = NULL;
        m_b = NULL;
    }

    // Assign final solution to member variable for saving etc.
    m_xij = uij;
}


/* Sequential time-stepping routine for general ERK schemes without mass matrix */
void SpaceTimeMatrix::ERKSolve(double solve_tol, int max_iter, int printLevel,
                                 bool binv_scale, int precondition, int AMGiters) 
{    
    /* ---------------------------------------------------------------------- */
    /* ------------------------ Setup/initialization ------------------------ */
    /* ---------------------------------------------------------------------- */
    double t = 0.0;         // Initial time

    HYPRE_ParVector    u;   // Solution vector
    HYPRE_IJVector     uij;
    HYPRE_ParVector    g;   // Spatial discretization vector
    HYPRE_IJVector     gij = NULL;
    HYPRE_ParCSRMatrix L;   // Spatial discretization matrix  
    HYPRE_IJMatrix     Lij = NULL;

    // Place-holder vectors
    std::vector<HYPRE_ParVector> vectors;
    std::vector<HYPRE_IJVector>  vectorsij;
    int numVectors = m_s_butcher + 1; // Have s stage vectors + 1 temporary vector

    // Get initial condition and initialize place-holder vectors
    RKInitializeHypreVectors(u, uij, numVectors, vectors, vectorsij);

    // Shallow copy place-holder vectors into vectors with meaningful names
    HYPRE_ParVector              b   = vectors[0];  // Temporary vector
    HYPRE_IJVector               bij = vectorsij[0];
    std::vector<HYPRE_ParVector> k; // Stage vectors
    std::vector<HYPRE_IJVector>  kij;    
    k.reserve(m_s_butcher);
    kij.reserve(m_s_butcher);
    for (int i = 0; i < m_s_butcher; i++) {
        kij[i] = vectorsij[i+1];
        k[i]   = vectors[i+1];
    }


    /* ------------------------------------------------------------ */
    /* ------------------------ Time march ------------------------ */
    /* ------------------------------------------------------------ */
    // Take nt-1 steps
    int step = 0;
    for (step = 0; step < m_nt-1; step++) {
        
        // Build ith stage vector, k[i]
        for (int i = 0; i < m_s_butcher; i++) {
            // Compute spatial discretization at t + c[i]*dt
            if (m_g_isTimedependent || (i == 0 && step == 0)) {
                RKGetHypreSpatialDiscretizationG(g, gij, t + m_dt * m_c_butcher[i]);
            }
            // Solution-dependent term
            if (m_L_isTimedependent || (i == 0 && step == 0)) {
                RKGetHypreSpatialDiscretizationL(L, Lij, t + m_dt * m_c_butcher[i]);
            } 

            HYPRE_ParVectorCopy(u, b); // b <- u
            for (int j = 0; j < i; j++) {
                double temp = m_dt * m_A_butcher[i][j];
                if (temp !=  0.0) HYPRE_ParVectorAxpy(temp, k[j], b); // b <- b + dt*aij*k[j]
            }

            // Set final value of stage by computing MATVEC
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, L, b, 1.0, g, k[i]); // k[i] <- -L*b + g 
        }

        if ((printLevel > 0) && (m_globRank == 0)) {
            std::cout << "Time step " << step+1 << "/" << m_nt-1 << '\n';
        }

        // Sum solution
        for (int i = 0; i < m_s_butcher; i++)  {
            double temp = m_dt * m_b_butcher[i];
            if (temp != 0.0) HYPRE_ParVectorAxpy(temp, k[i], u); // u <- u + dt*k[i]*k[i]; 
        }
        t += m_dt; // Increment time
    }


    /* ---------------------------------------------------------- */
    /* ------------------------ Clean up ------------------------ */
    /* ---------------------------------------------------------- */
    for (int i = 0; i < vectors.size(); i++) {
        HYPRE_IJVectorDestroy(vectorsij[i]);
    }
    if (step > 0) {
        HYPRE_IJVectorDestroy(gij);
        HYPRE_IJMatrixDestroy(Lij);
    }

    // Assign final solution to member variable for saving etc.
    m_xij = uij;
}


/* Get initial condition, u0, as a HYPRE vector, and initialize all the HYPRE vectors 
    in "vectors" to the same values as the intial conditon */
void SpaceTimeMatrix::RKInitializeHypreVectors(HYPRE_ParVector               &u0, 
                                                HYPRE_IJVector               &u0ij, 
                                                int                           numVectors,
                                                std::vector<HYPRE_ParVector> &vectors, 
                                                std::vector<HYPRE_IJVector>  &vectorsij) 
{
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    int      onProcSize;
    double * U;
    int    * rows;

    // No parallelism: Spatial discretization on single processor
    if (!m_useSpatialParallel) {
        getInitialCondition(U, spatialDOFs);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator
    } else {
        getInitialCondition(m_spatialComm, U, ilower, iupper, spatialDOFs);    
    }
    
    onProcSize = iupper - ilower + 1; // Number of rows on current process
    rows       = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        rows[i] = ilower + i;
    }

    HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &u0ij);
    HYPRE_IJVectorSetObjectType(u0ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(u0ij);
    HYPRE_IJVectorSetValues(u0ij, onProcSize, rows, U);
    HYPRE_IJVectorAssemble(u0ij);
    HYPRE_IJVectorGetObject(u0ij, (void **) &u0);

    // Reserve space for the specified number of vectors
    vectors.reserve(numVectors);
    vectorsij.reserve(numVectors);
    // Create and initialize all vectors in vectorsij, setting their values to those of u
    for (int i = 0; i < numVectors; i++) {
        HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &vectorsij[i]);
        HYPRE_IJVectorSetObjectType(vectorsij[i], HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(vectorsij[i]);
        HYPRE_IJVectorSetValues(vectorsij[i], onProcSize, rows, U);
        HYPRE_IJVectorAssemble(vectorsij[i]);
        HYPRE_IJVectorGetObject(vectorsij[i], (void **) &vectors[i]);
    }

    delete[] rows;
    delete[] U;
}


/* Get solution-independent component of spatial discretization, the vector g, as a HYPRE vector */
void SpaceTimeMatrix::RKGetHypreSpatialDiscretizationG(HYPRE_ParVector &g,
                                                        HYPRE_IJVector &gij,
                                                        double          t)  
{
    // Free vector if currently allocated memory
    if (gij) HYPRE_IJVectorDestroy(gij);
    
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    int      onProcSize;
    int    * G_rows;
    double * G;

    // No parallelism: Spatial discretization on single processor
    if (!m_useSpatialParallel) {
        getSpatialDiscretizationG(G, spatialDOFs, t);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute spatial discretization across spatial communicator
    } else {
        getSpatialDiscretizationG(m_spatialComm, G, ilower, iupper, spatialDOFs, t);
    }

    // Get global row indices
    onProcSize = iupper - ilower + 1; // Number of rows of spatial disc I own
    G_rows = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        G_rows[i] = ilower + i;
    }
    
    // Create HYPRE vector
    HYPRE_IJVectorCreate(m_globComm, ilower, iupper, &gij);
    HYPRE_IJVectorSetObjectType(gij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(gij);
    HYPRE_IJVectorSetValues(gij, onProcSize, G_rows, G);
    HYPRE_IJVectorAssemble(gij);
    HYPRE_IJVectorGetObject(gij, (void **) &g);

    // Remove pointers that should have been copied by Hypre
    delete[] G_rows;
    delete[] G;
}


/* Get solution-dependent component of spatial discretization, the matrix L, as a HYPRE matrix */
void SpaceTimeMatrix::RKGetHypreSpatialDiscretizationL(HYPRE_ParCSRMatrix &L,
                                                        HYPRE_IJMatrix    &Lij,
                                                        double             t)  
{
    // Free matrix if currently allocated memory
    if (Lij) HYPRE_IJMatrixDestroy(Lij);
    
    int      m_bsize;
    int      ilower;
    int      iupper;
    int      spatialDOFs;
    int      onProcSize;
    int    * L_rowptr;
    int    * L_colinds;
    int    * L_rows;
    int    * L_cols_per_row;
    double * L_data;
    double * U0; // Dummy variable
    bool     getU0 = false; // No need to get initial guess at the solution

    // No parallelism: Spatial discretization on single processor
    if (!m_useSpatialParallel) {
        getSpatialDiscretizationL(L_rowptr, L_colinds, L_data, U0, getU0, spatialDOFs, t, m_bsize);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator    
    } else {
        getSpatialDiscretizationL(m_spatialComm, L_rowptr, L_colinds, L_data, 
                                    U0, getU0, ilower, iupper, spatialDOFs, 
                                    t, m_bsize);
    }

    // Initialize matrix
    onProcSize = iupper - ilower + 1; // Number of rows on process
    HYPRE_IJMatrixCreate(m_globComm, ilower, iupper, ilower, iupper, &Lij);
    HYPRE_IJMatrixSetObjectType(Lij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(Lij);

    // Set matrix coefficients
    L_rows         = new int[onProcSize];
    L_cols_per_row = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        L_rows[i] = ilower + i;
        L_cols_per_row[i] = L_rowptr[i+1] - L_rowptr[i];
    }
    HYPRE_IJMatrixSetValues(Lij, onProcSize, L_cols_per_row, L_rows, L_colinds, L_data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(Lij);
    HYPRE_IJMatrixGetObject(Lij, (void **) &L);

    // Remove pointers that should have been copied by Hypre
    delete[] L_rowptr;
    delete[] L_colinds;
    delete[] L_data;
    delete[] L_rows;
    delete[] L_cols_per_row;
}





void SpaceTimeMatrix::BuildMatrix()
{
    // Check not using time stepping, since this doesn't make sense!
    if (!m_pit) {
        std::cout << "WARNING: BuildMatrix() only available when solving space-time system!" << '\n';
        MPI_Finalize();
        exit(1);        
        //return;
    }
    if (m_globRank == 0) std::cout << "Building matrix, " << m_useSpatialParallel << "\n";
    if (m_useSpatialParallel) GetMatrix_ntLE1();
    else GetMatrix_ntGT1();
    if (m_globRank == 0) std::cout << "Space-time matrix assembled.\n";
}


/* Print data to file that allows one to extract the 
    relevant data for plotting, etc. from the saved solution. Pass
    in dictionary with information also to be saved to file that isn't a member 
    variable (e.g. space disc info)
*/
void SpaceTimeMatrix::SaveSolInfo(std::string filename, std::map<std::string, std::string> additionalInfo) 
{
    std::ofstream solinfo;
    solinfo.open(filename);
    solinfo << std::setprecision(17); // Must accurately print out dt to file!
    solinfo << "pit " << int(m_pit) << "\n";
    solinfo << "P " << m_numProc << "\n";
    solinfo << "nt " << m_nt << "\n";
    solinfo << "dt " << m_dt << "\n";
    solinfo << "s " << m_s_butcher << "\n";
    solinfo << "timeDisc " << m_timeDisc << "\n";
    solinfo << "spatialParallel " << int(m_useSpatialParallel) << "\n";
    if (m_useSpatialParallel) solinfo << "np_xTotal " << m_spatialCommSize << "\n";
    
    // Print out contents from additionalInfo to file too
    std::map<std::string, std::string>::iterator it;
    for (it=additionalInfo.begin(); it!=additionalInfo.end(); it++) {
        solinfo << it->first << " " << it->second << "\n";
    }

    solinfo.close();
}


// timeDisc with "1" as 1st digit are ERK, timeDisc with "2" as 1st digit are DIRK
// 2nd digit == number of stages
// 3rd digit == order of method
void SpaceTimeMatrix::GetButcherTableaux() {

    m_s_butcher = m_timeDisc / 10 % 10; //  Extract number of stages; assumes number has 3 digits    
    // Resize Butcher arrays
    m_A_butcher.resize(m_s_butcher, std::vector<double>(m_s_butcher, 0.0));
    m_b_butcher.resize(m_s_butcher);
    m_c_butcher.resize(m_s_butcher);

    /* --- ERK tables --- */
    // Forward Euler: 1st-order
    if (m_timeDisc == 111) {
        m_is_ERK             = true;
        m_s_butcher       = 1;
        m_A_butcher[0][0] = 0.0;
        m_b_butcher[0]    = 1.0; 
        m_c_butcher[0]    = 0.0; 
    
    // 2nd-order Heun's method    
    } else if (m_timeDisc == 122) {
        m_is_ERK             = true;
        m_s_butcher       = 2;
        m_A_butcher[0][0] = 0.0;
        m_A_butcher[1][0] = 1.0;
        m_A_butcher[0][1] = 0.0;
        m_A_butcher[1][1] = 0.0;
        m_b_butcher[0]    = 0.5;
        m_b_butcher[1]    = 0.5;
        m_c_butcher[0]    = 0.0;
        m_c_butcher[1]    = 1.0;
        
    // 3rd-order optimal SSPERK
    } else if (m_timeDisc == 133) {
        m_is_ERK             = true;
        m_s_butcher       = 3;
        m_A_butcher[0][0] = 0.0; // 1st col
        m_A_butcher[1][0] = 1.0;
        m_A_butcher[2][0] = 1.0/4.0; 
        m_A_butcher[0][1] = 0.0; // 2nd col
        m_A_butcher[1][1] = 0.0;
        m_A_butcher[2][1] = 1.0/4.0;
        m_A_butcher[0][2] = 0.0;  // 3rd col
        m_A_butcher[1][2] = 0.0;
        m_A_butcher[2][2] = 0.0;
        m_b_butcher[0]    = 1.0/6.0;
        m_b_butcher[1]    = 1.0/6.0;
        m_b_butcher[2]    = 2.0/3.0;
        m_c_butcher[0]    = 0.0;
        m_c_butcher[1]    = 1.0;
        m_c_butcher[2]    = 1.0/2.0;

    // Classical 4th-order ERK
    } else if (m_timeDisc == 144){
        m_is_ERK             = true;
        m_s_butcher       = 4;
        m_A_butcher[0][0] = 0.0; // 1st col
        m_A_butcher[1][0] = 1.0/2.0;
        m_A_butcher[2][0] = 0.0;
        m_A_butcher[3][0] = 0.0;
        m_A_butcher[0][1] = 0.0; // 2nd col
        m_A_butcher[1][1] = 0.0;
        m_A_butcher[2][1] = 1.0/2.0;
        m_A_butcher[3][1] = 0.0;
        m_A_butcher[0][2] = 0.0; // 3rd col
        m_A_butcher[1][2] = 0.0;
        m_A_butcher[2][2] = 0.0;
        m_A_butcher[3][2] = 1.0;
        m_A_butcher[0][3] = 0.0; // 4th col
        m_A_butcher[1][3] = 0.0;
        m_A_butcher[2][3] = 0.0;
        m_A_butcher[3][3] = 0.0;
        m_b_butcher[0]    = 1.0/6.0;
        m_b_butcher[1]    = 1.0/3.0;
        m_b_butcher[2]    = 1.0/3.0;
        m_b_butcher[3]    = 1.0/6.0;
        m_c_butcher[0]    = 0.0;
        m_c_butcher[1]    = 1.0/2.0;
        m_c_butcher[2]    = 1.0/2.0;
        m_c_butcher[3]    = 1.0;
    
    
    /* --- SDIRK tables --- */
    // Backward Euler, 1st-order
    } else if (m_timeDisc == 211) {
        m_is_DIRK            = true;
        m_is_SDIRK           = true;
        m_s_butcher       = 1;
        m_A_butcher[0][0] = 1.0;
        m_b_butcher[0]    = 1.0; 
        m_c_butcher[0]    = 1.0; 
    
    // 2nd-order L-stable SDIRK (there are a few different possibilities here. This is from the Dobrev et al.)
    } else if (m_timeDisc == 222) {
        double sqrt2      = 1.414213562373095;
        m_is_DIRK            = true;
        m_is_SDIRK           = true;
        m_s_butcher       = 2;
        m_A_butcher[0][0] = 1.0 - sqrt2/2.0;
        m_A_butcher[1][0] = sqrt2 - 1.0;
        m_A_butcher[0][1] = 0.0;
        m_A_butcher[1][1] = 1 - sqrt2/2.0;
        m_b_butcher[0]    = 0.5;
        m_b_butcher[1]    = 0.5;
        m_c_butcher[0]    = 1 - sqrt2/2.0;
        m_c_butcher[1]    = sqrt2/2.0;
        
    // 3rd-order (3-stage) L-stable SDIRK (see Butcher's book, p.261--262)
    } else if (m_timeDisc == 233) {
        double zeta       =  0.43586652150845899942;
        double alpha      =  0.5*(1.0 + zeta);
        double beta       =  0.5*(1.0 - zeta); 
        double gamma      = -3.0/2.0*zeta*zeta + 4.0*zeta - 0.25;
        double epsilon    =  3.0/2.0*zeta*zeta - 5.0*zeta + 1.25;
        m_is_DIRK            =  true;
        m_is_SDIRK           =  true;
        m_s_butcher       =  3;
        m_A_butcher[0][0] =  zeta; // 1st col
        m_A_butcher[1][0] =  beta;
        m_A_butcher[2][0] =  gamma; 
        m_A_butcher[0][1] =  0.0; // 2nd col
        m_A_butcher[1][1] =  zeta;
        m_A_butcher[2][1] =  epsilon;
        m_A_butcher[0][2] =  0.0;  // 3rd col
        m_A_butcher[1][2] =  0.0;
        m_A_butcher[2][2] =  zeta;
        m_b_butcher[0]    =  gamma;
        m_b_butcher[1]    =  epsilon;
        m_b_butcher[2]    =  zeta;
        m_c_butcher[0]    =  zeta;
        m_c_butcher[1]    =  alpha;
        m_c_butcher[2]    =  1.0;
        
    // 4th-order (5-stage) L-stable SDIRK (see Wanner's & Hairer's, Solving ODEs II, 1996, eq. 6.16)
    } else if (m_timeDisc == 254) {
        m_is_DIRK            =  true;
        m_is_SDIRK           =  true;
        m_s_butcher       =  5;
        // 1st col of A
        m_A_butcher[0][0] =  1.0/4.0; 
        m_A_butcher[1][0] =  1.0/2.0;
        m_A_butcher[2][0] =  17.0/50.0; 
        m_A_butcher[3][0] =  371.0/1360.0; 
        m_A_butcher[4][0] =  25.0/24.0; 
        // 2nd col of A
        m_A_butcher[0][1] =  0.0; 
        m_A_butcher[1][1] =  1.0/4.0;
        m_A_butcher[2][1] = -1.0/25.0;
        m_A_butcher[3][1] = -137.0/2720.0;
        m_A_butcher[4][1] = -49.0/48.0;
        // 3rd col of A
        m_A_butcher[0][2] =  0.0;  
        m_A_butcher[1][2] =  0.0;
        m_A_butcher[2][2] =  1.0/4.0;
        m_A_butcher[3][2] =  15.0/544.0;
        m_A_butcher[4][2] =  125.0/16.0;
        // 4th col of A
        m_A_butcher[0][3] =  0.0;  
        m_A_butcher[1][3] =  0.0;
        m_A_butcher[2][3] =  0.0;
        m_A_butcher[3][3] =  1.0/4.0;
        m_A_butcher[4][3] = -85.0/12.0;
        // 5th col of A
        m_A_butcher[0][4] =  0.0;  
        m_A_butcher[1][4] =  0.0;
        m_A_butcher[2][4] =  0.0;
        m_A_butcher[3][4] =  0.0;
        m_A_butcher[4][4] =  1.0/4.0;
        
        // b 
        m_b_butcher[0]    =  25.0/24.0;
        m_b_butcher[1]    = -49.0/48.0;
        m_b_butcher[2]    =  125.0/16.0;
        m_b_butcher[3]    = -85.0/12.0;
        m_b_butcher[4]    =  1.0/4.0;
        // c
        m_c_butcher[0]    =  1.0/4.0;
        m_c_butcher[1]    =  3.0/4.0;
        m_c_butcher[2]    =  11.0/20.0;
        m_c_butcher[3]    =  1.0/2.0;
        m_c_butcher[4]    =  1.0;
    
    } else {
        std::cout << "WARNING: invalid choice of time integration.\n";
        MPI_Finalize();
        exit(1);
        //return;
    }
}


/* Get space-time matrix for at less than 1 temporal DOF per processor: Uses spatial parallelism */
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
    RK(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    // TODO: remove code below, but keep for the moment. 
    // if (m_timeDisc == 11) {
    //     BDF1(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    // }
    // else if (m_timeDisc == 31) {
    //     AB1(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    // }
    // else {
    //     std::cout << "WARNING: invalid choice of time integration.\n";
    //     MPI_Finalize();
    //     return;
    // }

    // Initialize matrix
    int onProcSize = localMaxRow - localMinRow + 1;
    int ilower = m_DOFInd*spatialDOFs + localMinRow;
    int iupper = m_DOFInd*spatialDOFs + localMaxRow;
    HYPRE_IJMatrixCreate(m_globComm, ilower, iupper, ilower, iupper, &m_Aij);
    HYPRE_IJMatrixSetObjectType(m_Aij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(m_Aij);

    // Set matrix coefficients
    int* rows         = new int[onProcSize];
    int* cols_per_row = new int[onProcSize];
    for (int i=0; i<onProcSize; i++) {
        rows[i] = ilower + i;
        cols_per_row[i] = rowptr[i+1] - rowptr[i];
    }
    HYPRE_IJMatrixSetValues(m_Aij, onProcSize, cols_per_row, rows, colinds, data);

    // Finalize construction
    HYPRE_IJMatrixAssemble(m_Aij);
    HYPRE_IJMatrixGetObject(m_Aij, (void **) &m_A);

    // Create rhs and solution vectors
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


/* Get space-time matrix for at least 1 temporal DOF per processor: Doesn't use spatial parallelism */
void SpaceTimeMatrix::GetMatrix_ntGT1()
{
    // Get local CSR structure
    int* rowptr;
    int* colinds;
    double* data;
    double* B;
    double* X;
    int onProcSize;
    RK(rowptr, colinds, data, B, X, onProcSize);
    // TODO : Delete the stuff below. But just keep for the moment.. 
   // if (m_timeDisc == 11) {
   //      BDF1(rowptr, colinds, data, B, X, onProcSize);
   //  }
   //  else if (m_timeDisc == 31) {
   //      AB1(rowptr, colinds, data, B, X, onProcSize);
   //  }
   //  else {
   //      std::cout << "WARNING: invalid choice of time integration.\n";
   //      MPI_Finalize();
   //      return;
   //  }

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
    
    // Create sample rhs and solution vectors
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
   m_solverOptions.relax_type = 3;
   m_solverOptions.interp_type = 6;
   m_solverOptions.strength_tolC = 0.1;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = -1;
   m_solverOptions.strength_tolR = -1;
   m_solverOptions.filter_tolA = 0.0;
   m_solverOptions.filter_tolR = 0.0;
   m_solverOptions.cycle_type = 1;
   m_rebuildSolver = true;
}


/* Set standard AIR parameters for BoomerAMG solve. */
void SpaceTimeMatrix::SetAIR()
{
   m_solverOptions.prerelax = "A";
   m_solverOptions.postrelax = "FFC";
   m_solverOptions.relax_type = 3;
   m_solverOptions.interp_type = 100;
   m_solverOptions.strength_tolC = 0.005;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = 1.5;
   m_solverOptions.strength_tolR = 0.005;
   m_solverOptions.filter_tolA = 0.0;
   m_solverOptions.filter_tolR = 0.0;
   m_solverOptions.cycle_type = 1;
   m_rebuildSolver = true;
}


/* Set AIR parameters assuming triangular matrix in BoomerAMG solve. */
void SpaceTimeMatrix::SetAIRHyperbolic()
{
   m_solverOptions.prerelax = "A";
   m_solverOptions.postrelax = "F";
   m_solverOptions.relax_type = 10;
   m_solverOptions.interp_type = 100;
   m_solverOptions.strength_tolC = 0.005;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = 1.5;
   m_solverOptions.strength_tolR = 0.005;
   m_solverOptions.filter_tolA = 0.0001;
   m_solverOptions.filter_tolR = 0.0;
   m_solverOptions.cycle_type = 1;
   m_rebuildSolver = true;
}


/* Provide BoomerAMG parameters struct for solve. */
void SpaceTimeMatrix::SetAMGParameters(AMG_parameters &params)
{
    // TODO: does this copy the structure by value?
    m_solverOptions = params;
}


void SpaceTimeMatrix::PrintMeshData()
{
    if (m_globRank == 0) {
        std::cout << "Space-time mesh:\n\thmin = " << m_hmin <<
        "\n\thmax = " << m_hmax << "\n\tdt   = " << m_dt << "\n\n";
    }
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
            if (m_globRank == 0) std::cout << "Rebuilding solver.\n";
            HYPRE_BoomerAMGDestroy(m_solver);
        } else {
            if (m_globRank == 0) std::cout << "Building solver.\n";
        }
        
        // Do not rebuild solver unless parameters are changed.
        m_rebuildSolver = false;

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
            HYPRE_BoomerAMGSetStrongThresholdR(m_solver, m_solverOptions.strength_tolR);
            HYPRE_BoomerAMGSetFilterThresholdR(m_solver, m_solverOptions.filter_tolR);
        }
        HYPRE_BoomerAMGSetInterpType(m_solver, m_solverOptions.interp_type);
        HYPRE_BoomerAMGSetCoarsenType(m_solver, m_solverOptions.coarsen_type);
        HYPRE_BoomerAMGSetAggNumLevels(m_solver, 0);
        HYPRE_BoomerAMGSetStrongThreshold(m_solver, m_solverOptions.strength_tolC);
        HYPRE_BoomerAMGSetGridRelaxPoints(m_solver, grid_relax_points);
        if (m_solverOptions.relax_type > -1) {
            HYPRE_BoomerAMGSetRelaxType(m_solver, m_solverOptions.relax_type);
        }
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_coarse, 3);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_down,   1);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_up,     2);
        if (m_solverOptions.filter_tolA > 0) {
            HYPRE_BoomerAMGSetADropTol(m_solver, m_solverOptions.filter_tolA);
        }
        // type = -1: drop based on row inf-norm
        else if (m_solverOptions.filter_tolA == -1) {
            HYPRE_BoomerAMGSetADropType(m_solver, -1);
        }

        // Set cycle type for solve 
        HYPRE_BoomerAMGSetCycleType(m_solver, m_solverOptions.cycle_type);
    }
}


void SpaceTimeMatrix::SolveAMG(double tol, int maxiter, int printLevel,
                               bool binv_scale)
{
    SetupBoomerAMG(printLevel, maxiter, tol);

    if (binv_scale) {
        HYPRE_ParCSRMatrix A_s;
        hypre_ParcsrBdiagInvScal(m_A, m_bsize, &A_s);
        hypre_ParCSRMatrixDropSmallEntries(A_s, 1e-15, 1);
        HYPRE_ParVector b_s;
        hypre_ParvecBdiagInvScal(m_b, m_bsize, &b_s, m_A);

        HYPRE_BoomerAMGSetup(m_solver, A_s, b_s, m_x);
        HYPRE_BoomerAMGSolve(m_solver, A_s, b_s, m_x);
    }
    else {
        HYPRE_BoomerAMGSetup(m_solver, m_A, m_b, m_x);
        HYPRE_BoomerAMGSolve(m_solver, m_A, m_b, m_x);
    }
}


void SpaceTimeMatrix::SolveGMRES(double tol, int maxiter, int printLevel,
                                 bool binv_scale, int precondition, int AMGiters) 
{
    HYPRE_ParCSRGMRESCreate(m_globComm, &m_gmres);
    
    // AMG preconditioning (setup boomerAMG with 1 max iter and print level 1)
    if (precondition == 1) {
        SetupBoomerAMG(1, AMGiters, 0.0);
        HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, m_solver);
    }
    // Diagonally scaled preconditioning?
    else if (precondition == 2) {
        HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_ParCSROnProcTriSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSROnProcTriSetup, m_solver);  
    }

    HYPRE_GMRESSetKDim(m_gmres, 50);
    HYPRE_GMRESSetMaxIter(m_gmres, maxiter);
    HYPRE_GMRESSetTol(m_gmres, tol);
    HYPRE_GMRESSetPrintLevel(m_gmres, printLevel);
    HYPRE_GMRESSetLogging(m_gmres, 1);

   if (binv_scale) {
        HYPRE_ParCSRMatrix A_s;
        hypre_ParcsrBdiagInvScal(m_A, m_bsize, &A_s);
        hypre_ParCSRMatrixDropSmallEntries(A_s, 1e-15, 1);
        HYPRE_ParVector b_s;
        hypre_ParvecBdiagInvScal(m_b, m_bsize, &b_s, m_A);
        HYPRE_ParCSRGMRESSetup(m_gmres, A_s, b_s, m_x);
        HYPRE_ParCSRGMRESSolve(m_gmres, A_s, b_s, m_x);
    }
    else {
        HYPRE_ParCSRGMRESSetup(m_gmres, m_A, m_b, m_x);
        HYPRE_ParCSRGMRESSolve(m_gmres, m_A, m_b, m_x);
    }
}


// TODO : this function may be unnecessary if mass matrix is stored as member variables...
void SpaceTimeMatrix::getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data)
{
    // TODO : set to sparse identity matrix here
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {

    }

    // TODO : set input pointers to address of member variables

}

/* ------------------------------------------------------------------------- */
/* ----------------- At least 1 temporal DOF per processor ------------------ */
/* ------------------------------------------------------------------------- */
// (No spatial parallelism)
// NOTES:
//  Does not make any assumption about overlap in the sparsity pattern of the spatial 
//  discretization and mass matrix when estimating matrix nnz, 
//  assumes nnz of spatial discretization does not depend on time

/* Arbitrary component(s) of s-stage RK block(s) w/ last stage eliminated. */
void SpaceTimeMatrix::RK(int* &rowptr, int* &colinds, double* &data,
                           double* &B, double* &X, int &onProcSize)
{
    int globalInd0 = m_globRank * m_nDOFPerProc;        // Global index of first variable I own
    int globalInd1 = globalInd0 + m_nDOFPerProc - 1;    // Global index of last variable I own

    int * localInd = new int[m_nDOFPerProc];            // Local index of each DOF inside block DOF
    int * blockInd = new int[m_nDOFPerProc];            // Block DOF, v, that each DOF belongs to
    for (int globalInd = globalInd0; globalInd <= globalInd1; globalInd++) {
        localInd[globalInd - globalInd0] = globalInd % m_s_butcher;
        blockInd[globalInd - globalInd0] = std::floor( (double) globalInd / m_s_butcher ); // TODO: Just use integer division here rather than floor?
    }
    
    // Get spatial discretization at time required by first 1st DOF.
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double* B0;
    double* X0;
    int spatialDOFs;
    double t = 0.0; // Time to evaluate spatial discretization, as required by first DOF. 
    if (localInd[0] == 0) { // Solution-type DOF that's not the initial condition
        if (blockInd[0] != 0) t = m_dt * (blockInd[0]-1) + m_dt * m_c_butcher[m_s_butcher-1]; 
    } else { // Stage-type DOF
        t = m_dt * blockInd[0] + m_dt * m_c_butcher[localInd[0]-1]; 
    }
    getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, t, m_bsize);
    int nnzL = T_rowptr[spatialDOFs];   
    
    
    // for (int i = 0; i < nnzL; i++) {
    //     std::cout << "T_data[" << i << "] = " << T_data[i] << '\n';
    // }
    // 
    // exit(1);
    
    
    // Get mass matrix
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    int nnzM = M_rowptr[spatialDOFs];
    // TODO: Why is this not making sense  for DG??
    
    // std::cout << "nnzL on proc " << m_globRank << " = " << nnzL << '\n';
    // std::cout << "spatialDOFs on proc " << m_globRank << " = " << spatialDOFs << '\n';
    // std::cout << "nnzM on proc " << m_globRank << " = " << nnzM << '\n';
    // 
    
    /* ------ Get total NNZ on this processor. ------ */
    //  -Assumes NNZ of spatial discretization does not change with time
    //  -Doesn't assume sparsity of M and L overlap: This nnz count is an upperbound.
    int procNnz = 0;
    for (int i = 0; i <= globalInd1 - globalInd0; i++) {
        int k = localInd[i];
        // Solution-type variable 
        if (k == 0) {
            // Initial condition just has identity coupling
            if (blockInd[i] == 0) {
                procNnz += spatialDOFs;
            // All solution DOFs at t > 0
            } else {
                // Coupling to itself
                procNnz += nnzM;
                if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] != 0.0) procNnz += nnzL;
    
                // Coupling to solution DOF at previous time 
                procNnz += nnzM;
                if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] != 0.0) procNnz += nnzL; 
    
                // Coupling to stage variables at previous time 
                for (int j = 0; j < m_s_butcher-1; j++) {
                    if (m_b_butcher[j] != 0.0) procNnz += nnzM;
                    if (m_b_butcher[j]*m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1]*m_A_butcher[m_s_butcher-1][j] != 0.0) procNnz += nnzL;
                }
            }
    
        // Stage-type DOF
        } else {
            k -= 1; // Work with this to index Butcher arrays directly.
            // Coupling to itself
            procNnz += nnzM;
            if (m_A_butcher[k][k] != 0.0) procNnz += nnzL;
    
            // Coupling to solution at previous time
            procNnz += nnzL;
    
            // Coupling to previous stage variables at current time
            for (int j = 0; j < k; j++) {
                if (m_A_butcher[k][j] != 0.0) procNnz += nnzL;
            }
        }
    }
    //std::cout << "nnz on proc " << m_globRank << " = " << procNnz << '\n';
    
    onProcSize = m_nDOFPerProc * spatialDOFs; // Number of rows I own
    rowptr     = new int[onProcSize + 1];
    colinds    = new int[procNnz];
    data       = new double[procNnz];
    B          = new double[onProcSize];
    X          = new double[onProcSize];
    rowptr[0]  = 0; 
    
    int dataInd         = 0;
    int rowOffset       = 0;    // Row offset for current DOF w.r.t rows on proc
    int globalColOffset = 0;    // Global index of first column for furthest DOF that we couple back to
    int localColOffset  = 0;    // Temporary variable to help indexing
    double temp         = 0.0;  // Temporary constant 
    
    
    /* ------------------------------------------------------------------------------------------------- */
    /* ------ Loop over all DOFs on this processor building their block rows in space-time matrix ------ */
    /* ------------------------------------------------------------------------------------------------- */
    for (int globalInd = globalInd0; globalInd <= globalInd1; globalInd++) {
    
        // Rebuild spatial discretization if it's time dependent
        if ((globalInd > globalInd0) && m_isTimeDependent) {
            delete[] T_rowptr;
            delete[] T_colinds;
            delete[] T_data;
            delete[] B0;
            delete[] X0;
            // Time to evaluate spatial discretization at
            // Solution-type DOF
            if (localInd[globalInd-globalInd0] == 0) { 
                t = m_dt * (blockInd[globalInd-globalInd0] - 1) + m_dt * m_c_butcher[m_s_butcher-1];
            //  Stage-type DOF    
            } else { 
                t = m_dt * blockInd[globalInd-globalInd0]       + m_dt * m_c_butcher[localInd[globalInd-globalInd0]-1];
            }
            getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0, spatialDOFs, t, m_bsize);
        }
    
    
        std::map<int, double>::iterator it;
        /* -------------------------------------------------------- */
        /* ------ Assemble block row for a solution-type DOF ------ */
        /* -------------------------------------------------------- */
        if (localInd[globalInd-globalInd0] == 0) {
            // Initial condition: Set matrix to identity and fix RHS and initial guess to ICs.
            if (blockInd[globalInd-globalInd0] == 0) {
    
                // Loop over each row
                for (int row=0; row<spatialDOFs; row++) {
                    colinds[dataInd] = row; // Note globalColOffset == 0 for this DOF
                    data[dataInd] = 1.0;
                    dataInd += 1;
                    rowptr[row+1] = dataInd;
                    // Set these to 0 then add IC to them below.
                    B[row] = 0.0;
                    X[row] = 0.0;
                }
                addInitialCondition(B);
                addInitialCondition(X); 
    
            // Solution-type DOF at time t > 0
            } else {
                // This DOF couples s DOFs back to the solution at the previous time
                globalColOffset = spatialDOFs * (globalInd - m_s_butcher);
    
                // Loop over each row in spatial discretization, working from left-most column/variables to right-most
                for (int row=0; row<spatialDOFs; row++) {
                    /* ------ Coupling to solution at previous time ------ */
                    std::map<int, double> entries; 
                    // Get mass-matrix data
                    for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                        if (std::abs(M_data[j]) > 1e-16) {
                            entries[M_colinds[j]] = M_data[j] / m_dt;
                        }
                    }
    
                    // Add spatial discretization data to mass-matrix data
                    temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1];
                    if (temp != 0.0) {
                        for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                            if (std::abs(T_data[j]) > 1e-16) {
                                entries[T_colinds[j]] += temp * T_data[j];
                            }
                        }
                    }
    
                    // Add this data to global matrix
                    for (it=entries.begin(); it!=entries.end(); it++) {
                        colinds[dataInd] = globalColOffset + it->first;
                        data[dataInd] = -it->second;
                        dataInd += 1;
                    }
    
    
                    /* ------ Coupling to stages at previous time ------ */
                    // Loop over all stage DOFs
                    for (int i = 0; i < m_s_butcher - 1; i++) {
                        std::map<int, double> entries2; 
                        // Get mass-matrix data
                        temp = m_b_butcher[i];
                        if (temp != 0.0) {
                            for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                                if (std::abs(M_data[j]) > 1e-16) {
                                    entries2[M_colinds[j]] = temp * M_data[j];
                                }
                            }
                        }
    
                        // Add spatial discretization data to mass-matrix data
                        temp = m_b_butcher[i] * m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] * m_A_butcher[m_s_butcher-1][i];
                        if (temp != 0.0) {
                            temp *= m_dt;
                            for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                                if (std::abs(T_data[j]) > 1e-16) {
                                    entries2[T_colinds[j]] += temp * T_data[j];
                                }
                            }
                        }
    
                        // Add this data to global matrix
                        localColOffset = globalColOffset + (i+1) * spatialDOFs;
                        for (it=entries2.begin(); it!=entries2.end(); it++) {
                            colinds[dataInd] = localColOffset + it->first;
                            data[dataInd] = -it->second;
                            dataInd += 1;
                        }
                    }
    
    
                    /* ------ Coupling to myself/solution at current time ------ */
                    std::map<int, double> entries3; 
                    // Get mass-matrix data
                    for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                        if (std::abs(M_data[j]) > 1e-16) {
                            entries3[M_colinds[j]] = M_data[j] / m_dt;
                        }
                    }
    
                    // Add spatial discretization data to mass-matrix data
                    temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1];
                    if (temp != 0.0) {
                        for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                            if (std::abs(T_data[j]) > 1e-16) {
                                entries3[T_colinds[j]] += temp * T_data[j];
                            }
                        }
                    }
    
                    // Add this data to global matrix
                    localColOffset = globalColOffset + m_s_butcher * spatialDOFs;
                    for (it=entries3.begin(); it!=entries3.end(); it++) {
                        colinds[dataInd] = localColOffset + it->first;
                        data[dataInd] = it->second;
                        dataInd += 1;
                    }
    
    
                    // RHS and initial guess
                    B[rowOffset + row] = m_b_butcher[m_s_butcher-1] * B0[row];
                    X[rowOffset + row] = X0[row];
    
                    // Move to next row for current variable
                    rowptr[rowOffset + row+1] = dataInd;
                }
            }
    
        /* ----------------------------------------------------- */
        /* ------ Assemble block row for a stage-type DOF ------ */
        /* ----------------------------------------------------- */
        } else {
            int kInd = localInd[globalInd-globalInd0]; // 1's-based index of current stage DOF
            globalColOffset = spatialDOFs * (globalInd - kInd); // We couple back to the solution at current time
    
            // Loop over each row in spatial discretization, working from left-most column/variables to right-most
            for (int row=0; row<spatialDOFs; row++) {
                /* ------ Coupling to solution at current time ------ */
                for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                    colinds[dataInd] = globalColOffset + T_colinds[j];
                    data[dataInd] = T_data[j];
                    dataInd += 1;
                }
    
                /* ------ Coupling to stages that come before me ------ */
                for (int i=0; i<kInd-1; i++) {
                    temp = m_A_butcher[kInd-1][i];
                    if (temp != 0.0) {
                        temp *= m_dt;
                        localColOffset = globalColOffset + (i+1) * spatialDOFs;
                        for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                            if (std::abs(T_data[j]) > 1e-16) {
                                colinds[dataInd] = localColOffset + T_colinds[j];
                                data[dataInd] = temp * T_data[j];
                                dataInd += 1;
                            }
                        }
                    }
                }
    
                /* ------ Coupling to myself ------ */
                std::map<int, double> entries; 
                // Get mass-matrix data
                for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        entries[M_colinds[j]] = M_data[j];
                    }
                }
    
                // Add spatial discretization data to mass-matrix data
                temp = m_A_butcher[kInd-1][kInd-1];
                if (temp != 0.0) {
                    temp *= m_dt;
                    for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                        if (std::abs(T_data[j]) > 1e-16) {
                            entries[T_colinds[j]] += temp * T_data[j];
                        }
                    }
                }
    
                // Add this data to global matrix
                localColOffset = globalColOffset + kInd * spatialDOFs;
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = localColOffset + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }
    
                // RHS and initial guess
                B[rowOffset + row] = B0[row];
                X[rowOffset + row] = X0[row];
    
                // Move to next row for current variable
                rowptr[rowOffset + row+1] = dataInd;
            }
        }
        // Move to next variable
        rowOffset += spatialDOFs;
    }
    // Finished assembling component of global space-time matrix
    
    // Check that sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    
    //std::cout << "Actual nnz on proc " << m_globRank << " = " << rowptr[onProcSize] << "\n";
    
    // Clean up.
    delete[] localInd;
    delete[] blockInd;
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] X0;
    
    // MPI_Finalize();
    // exit(1);
}





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
                             X0, spatialDOFs, m_dt*tInd0, m_bsize);
    int nnzPerTime = T_rowptr[spatialDOFs];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[spatialDOFs]; i++) {
        M_data[i] /= m_dt;
    }

    std::cout << "nnzL on proc " << m_globRank << " = " << T_rowptr[spatialDOFs] << '\n';
    std::cout << "spatialDOFs on proc " << m_globRank << " = " << spatialDOFs << '\n';
    std::cout << "nnzM on proc " << m_globRank << " = " << M_rowptr[spatialDOFs] << '\n';

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    onProcSize  = m_ntPerProc * spatialDOFs;
    int procNnz = m_ntPerProc * (2*M_rowptr[spatialDOFs] + nnzPerTime);     // nnzs on this processor
    // Account for only diagonal block at time t=0
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
                                     X0, spatialDOFs, m_dt*ti, m_bsize);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;
        std::map<int, double>::iterator it;

        // At time t=0, have fixed initial condition. Set matrix to identity
        // and fix RHS and initial guess to ICs.
        if (ti == 0) {
            
            // Add initial condition to rhs
            addInitialCondition(B);

            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                colinds[dataInd] = i;
                data[dataInd] = 1.0;
                dataInd += 1;
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;

                // Set solution equal to initial condition
                X[thisRow] = B[i];
            }
        }
        else {

            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {
                
                std::map<int, double> entries;

                // Add row of off-diagonal block, -(M/dt)*u_{i-1}
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16){
                        colinds[dataInd] = colPlusOffd + M_colinds[j];
                        data[dataInd] = -M_data[j];
                        dataInd += 1;
                        entries[M_colinds[j]] += M_data[j];
                    }
                }

                // Get row of spatial discretization
                for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                    entries[T_colinds[j]] += T_data[j];
                }

                // Add spatial discretization and mass matrix to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    if (std::abs(it->second) > 1e-16) {
                        colinds[dataInd] = colPlusDiag + it->first;
                        data[dataInd] = it->second;
                        dataInd += 1;
                    }
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
                                 spatialDOFs, m_dt*(tInd0-1), m_bsize);
    }
    else {
        getSpatialDiscretization(T_rowptr, T_colinds, T_data, B0, X0,
                                 spatialDOFs, m_dt*tInd0, m_bsize);
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
    int procNnz = m_ntPerProc * (2*M_rowptr[spatialDOFs] + nnzPerTime);     // nnzs on this processor
    // Account for only diagonal block (mass matrix) at time t=0
    if (tInd0 == 0) procNnz -= (M_rowptr[spatialDOFs] + nnzPerTime);

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
                                     X0, spatialDOFs, m_dt*ti, m_bsize);
        }

        int colPlusOffd  = (ti - 1)*spatialDOFs;
        int colPlusDiag  = ti*spatialDOFs;
        std::map<int, double>::iterator it;

        // At time t=0, have fixed initial condition. Set matrix to identity
        // and fix RHS and initial guess to ICs.
        if (ti == 0) {
            
            // Add initial condition to rhs
            addInitialCondition(B);

            // Loop over each row in spatial discretization at time ti
            for (int i=0; i<spatialDOFs; i++) {

                colinds[dataInd] = i;
                data[dataInd] = 1.0;
                dataInd += 1;
                rowptr[thisRow+1] = dataInd;
                thisRow += 1;

                // Set solution equal to initial condition
                X[thisRow] = B[i];
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
                    if (std::abs(it->second) > 1e-16) {
                        colinds[dataInd] = colPlusOffd + it->first;
                        data[dataInd] = it->second;
                        dataInd += 1;
                    }
                }

                // Add mass matrix as diagonal block, M/dt
                for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        colinds[dataInd] = colPlusDiag + M_colinds[j];
                        data[dataInd] = M_data[j];
                        dataInd += 1;
                    }
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


/* ------------------------------------------------------------------------- */
/* ------------------ At most one time step per processor ------------------ */
/* ------------------------------------------------------------------------- */
// (Spatial parallelism)

void SpaceTimeMatrix::RK(int *&rowptr, int *&colinds, double *&data, 
                        double *&B, double *&X, int &localMinRow, 
                        int &localMaxRow, int &spatialDOFs)
{
    
    //std::cout << "Global rank = " << m_globRank << "; Local rank = " << m_spatialRank << "\n";
    
    int globalInd = m_DOFInd;                   // Global index of variable I own
    int localInd  = globalInd % m_s_butcher;    // Local index of variable I own
    // TODO: I guess I can just use integer division here? 
    int blockInd  = std::floor( (double) globalInd / m_s_butcher ); // Block index of variable I own
    
    // Get spatial discretization 
    int* T_rowptr;
    int* T_colinds;
    double* T_data;
    double t = 0.0; // Time to evaluate spatial discretization
    if (localInd == 0) { // Solution-type DOF that's not the initial condition
        if (blockInd != 0) t = m_dt * (blockInd-1) + m_dt * m_c_butcher[m_s_butcher-1]; 
    } else { // Stage-type DOF
        t = m_dt * blockInd + m_dt * m_c_butcher[localInd-1]; 
    }
    getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data, 
                            B, X, localMinRow, localMaxRow, spatialDOFs, 
                            t, m_bsize);
    int onProcSize = localMaxRow - localMinRow + 1; // Number of rows of spatial disc I own
    int nnzL       = T_rowptr[onProcSize] - T_rowptr[0];    // Nnz in the component of spatial disc I own
    
    //std::cout << "nnzL on proc " << m_globRank << " = " << nnzL << '\n';
    //std::cout << "spatialDOFs on proc " << m_globRank << " = " << onProcSize << '\n';
    
    //MPI_Finalize();
    //exit(0);
    
    // Get mass matrix
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    int nnzM = M_rowptr[onProcSize] - M_rowptr[0];
    
    // std::cout << "nnzL on proc " << m_globRank << " = " << nnzL << '\n';
    // std::cout << "spatialDOFs on proc " << m_globRank << " = " << spatialDOFs << '\n';
    // std::cout << "nnzM on proc " << m_globRank << " = " << nnzM << '\n';
    // 
    
    /* ------ Get total NNZ on this processor. ------ */
    //  -Doesn't assume sparsity of M and L overlap: This nnz count is an upperbound.
    int procNnz = 0;
    int k = localInd;
    // Solution-type variable 
    if (k == 0) {
        // Initial condition just has identity coupling
        if (blockInd == 0) {
            procNnz += spatialDOFs;
        // All solution DOFs at t > 0
        } else {
            // Coupling to itself
            procNnz += nnzM;
            if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] != 0.0) procNnz += nnzL;
    
            // Coupling to solution DOF at previous time 
            procNnz += nnzM;
            if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] != 0.0) procNnz += nnzL; 
    
            // Coupling to stage variables at previous time 
            for (int j = 0; j < m_s_butcher-1; j++) {
                if (m_b_butcher[j] != 0.0) procNnz += nnzM;
                if (m_b_butcher[j]*m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1]*m_A_butcher[m_s_butcher-1][j] != 0.0) procNnz += nnzL;
            }
        }
    
    // Stage-type DOF
    } else {
        k -= 1; // Work with this to index Butcher arrays directly.
        // Coupling to itself
        procNnz += nnzM;
        if (m_A_butcher[k][k] != 0.0) procNnz += nnzL;
    
        // Coupling to solution at previous time
        procNnz += nnzL;
    
        // Coupling to previous stage variables at current time
        for (int j = 0; j < k; j++) {
            if (m_A_butcher[k][j] != 0.0) procNnz += nnzL;
        }
    }
    
    //std::cout << "nnz on proc " << m_globRank << " = " << procNnz << '\n';
    
    rowptr    = new int[onProcSize + 1];
    colinds   = new int[procNnz];
    data      = new double[procNnz];
    rowptr[0] = 0; 
    
    int dataInd         = 0;
    int globalColOffset = 0;    // Global index of first column for furthest DOF that we couple back to
    int localColOffset  = 0;    // Temporary variable to help indexing
    double temp         = 0.0;  // Temporary constant 
    
    /* ---------------------------------------------------------------------------------------------- */
    /* ------ Build my component of the block row of the space-time matrix belonging to my DOF ------ */
    /* ---------------------------------------------------------------------------------------------- */
    
    std::map<int, double>::iterator it;
    /* -------------------------------------------------------- */
    /* ------ Assemble block row for a solution-type DOF ------ */
    /* -------------------------------------------------------- */
    if (localInd == 0) {
        // Initial condition: Set matrix to identity and fix RHS and initial guess to ICs.
        if (blockInd == 0) {
    
            // Loop over each row
            for (int row=0; row<onProcSize; row++) {
                colinds[dataInd] = localMinRow + row; // Note globalColOffset == 0 for this DOF
                data[dataInd] = 1.0;
                dataInd += 1;
                rowptr[row+1] = dataInd;
    
                // Set these to 0 then replace it with IC below
                B[row] = 0.0;
                X[row] = 0.0; 
            }
            addInitialCondition(m_spatialComm, B);
            addInitialCondition(m_spatialComm, X); 
    
            //std::cout << "Global rank = " << m_globRank << "; Local rank = " << m_spatialRank << "\n";
    
        // Solution-type DOF at time t > 0
        } else {
            // This DOF couples s DOFs back to the solution at the previous time
            globalColOffset = spatialDOFs * (globalInd - m_s_butcher);
    
            // Loop over each row in spatial discretization, working from left-most column/variables to right-most
            for (int row=0; row<onProcSize; row++) {
                /* ------ Coupling to solution at previous time ------ */
                std::map<int, double> entries; 
                // Get mass-matrix data
                for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        entries[M_colinds[j]] = M_data[j] / m_dt;
                    }
                }
    
                // Add spatial discretization data to mass-matrix data
                temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1];
                if (temp != 0.0) {
                    for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                        if (std::abs(T_data[j]) > 1e-16) {
                            entries[T_colinds[j]] += temp * T_data[j];
                        }
                    }
                }
    
                // Add this data to global matrix
                for (it=entries.begin(); it!=entries.end(); it++) {
                    colinds[dataInd] = globalColOffset + it->first;
                    data[dataInd] = -it->second;
                    dataInd += 1;
                }
    
    
                /* ------ Coupling to stages at previous time ------ */
                // Loop over all stage DOFs
                for (int i = 0; i < m_s_butcher - 1; i++) {
                    std::map<int, double> entries2; 
                    // Get mass-matrix data
                    temp = m_b_butcher[i];
                    if (temp != 0.0) {
                        for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                            if (std::abs(M_data[j]) > 1e-16) {
                                entries2[M_colinds[j]] = temp * M_data[j];
                            }
                        }
                    }
    
                    // Add spatial discretization data to mass-matrix data
                    temp = m_b_butcher[i] * m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] * m_A_butcher[m_s_butcher-1][i];
                    if (temp != 0.0) {
                        temp *= m_dt;
                        for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                            if (std::abs(T_data[j]) > 1e-16) {
                                entries2[T_colinds[j]] += temp * T_data[j];
                            }
                        }
                    }
    
                    // Add this data to global matrix
                    localColOffset = globalColOffset + (i+1) * spatialDOFs;
                    for (it=entries2.begin(); it!=entries2.end(); it++) {
                        colinds[dataInd] = localColOffset + it->first;
                        data[dataInd] = -it->second;
                        dataInd += 1;
                    }
                }
    
    
                /* ------ Coupling to myself/solution at current time ------ */
                std::map<int, double> entries3; 
                // Get mass-matrix data
                for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        entries3[M_colinds[j]] = M_data[j] / m_dt;
                    }
                }
    
                // Add spatial discretization data to mass-matrix data
                temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1];
                if (temp != 0.0) {
                    for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                        if (std::abs(T_data[j]) > 1e-16) {
                            entries3[T_colinds[j]] += temp * T_data[j];
                        }
                    }
                }
    
                // Add this data to global matrix
                localColOffset = globalColOffset + m_s_butcher * spatialDOFs;
                for (it=entries3.begin(); it!=entries3.end(); it++) {
                    colinds[dataInd] = localColOffset + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }
    
    
                // Scale solution-independent spatial component by coefficient
                B[row] *= m_b_butcher[m_s_butcher-1];
    
                // Move to next row for current variable
                rowptr[row+1] = dataInd;
            }
        }
    /* ----------------------------------------------------- */
    /* ------ Assemble block row for a stage-type DOF ------ */
    /* ----------------------------------------------------- */
    } else {
        int kInd = localInd; // 1's-based index of current stage DOF
        globalColOffset = spatialDOFs * (globalInd - kInd); // We couple back to the solution at current time
    
        // Loop over each row in spatial discretization, working from left-most column/variables to right-most
        for (int row=0; row<onProcSize; row++) {
            /* ------ Coupling to solution at current time ------ */
            for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                colinds[dataInd] = globalColOffset + T_colinds[j];
                data[dataInd] = T_data[j];
                dataInd += 1;
            }
    
            /* ------ Coupling to stages that come before me ------ */
            for (int i=0; i<kInd-1; i++) {
                temp = m_A_butcher[kInd-1][i];
                if (temp != 0.0) {
                    temp *= m_dt;
                    localColOffset = globalColOffset + (i+1) * spatialDOFs;
                    for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                        if (std::abs(T_data[j]) > 1e-16) {
                            colinds[dataInd] = localColOffset + T_colinds[j];
                            data[dataInd] = temp * T_data[j];
                            dataInd += 1;
                        }
                    }
                }
            }
    
            /* ------ Coupling to myself ------ */
            std::map<int, double> entries; 
            // Get mass-matrix data
            for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                if (std::abs(M_data[j]) > 1e-16) {
                    entries[M_colinds[j]] = M_data[j];
                }
            }
    
            // Add spatial discretization data to mass-matrix data
            temp = m_A_butcher[kInd-1][kInd-1];
            if (temp != 0.0) {
                temp *= m_dt;
                for (int j=T_rowptr[row]; j<T_rowptr[row+1]; j++) {
                    if (std::abs(T_data[j]) > 1e-16) {
                        entries[T_colinds[j]] += temp * T_data[j];
                    }
                }
            }
    
            // Add this data to global matrix
            localColOffset = globalColOffset + kInd * spatialDOFs;
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = localColOffset + it->first;
                data[dataInd] = it->second;
                dataInd += 1;
            }
    
            // Move to next row for current variable
            rowptr[row+1] = dataInd;
        }
    }
    // Finished assembling component of global space-time matrix
    
    // Check that sufficient data was allocated
    if (dataInd > procNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    
    //std::cout << "Actual nnz on proc " << m_globRank << " = " << rowptr[onProcSize] << "\n";
    
    // Clean up.
    delete[] T_rowptr;
    delete[] T_colinds;
    delete[] T_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
}

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
                             m_dt*m_timeInd, m_bsize);
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows] - T_rowptr[0];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
    }
    int nnzMPerTime = M_rowptr[procRows] - M_rowptr[0];    

    // Get number nnz on this processor.
    int procNnz;
    if (m_timeInd == 0) procNnz = procRows;
    else procNnz = nnzPerTime + 2*nnzMPerTime;

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    // Mass matrix and spatial matrix have column indices global w.r.t. the spatial
    // discretization. Get global index w.r.t. space-time discretization.
    int colPlusOffd = (m_timeInd - 1)*spatialDOFs;
    int colPlusDiag = m_timeInd*spatialDOFs;
    std::map<int, double>::iterator it;

    // At time t=0, have fixed initial condition. Set matrix to identity
    // and fix RHS and initial guess to ICs.
    if (m_timeInd == 0) {
        
        // Add initial condition to rhs
        // addInitialCondition(m_spatialComm, B);

        // Loop over each on-processor row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {

            colinds[dataInd] = localMinRow + i;
            data[dataInd] = 1.0;
            dataInd += 1;
            rowptr[i+1] = dataInd;

            // Set solution equal to initial condition
            // X[i] = B[i];
            X[i] = 0.0;
        }
    }
    else {
        // Loop over each row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {
            std::map<int, double> entries;

            // Add row of off-diagonal block, -(M/dt)*u_{i-1}
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                if (std::abs(M_data[j]) > 1e-16) {
                    colinds[dataInd] = colPlusOffd + M_colinds[j];
                    data[dataInd] = -M_data[j];
                    dataInd += 1;
                    entries[M_colinds[j]] += M_data[j];
                }
            }

            // Get row of spatial discretization
            for (int j=T_rowptr[i]; j<T_rowptr[i+1]; j++) {
                entries[T_colinds[j]] += T_data[j];
            }

            // Add spatial discretization and mass matrix to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                if (std::abs(it->second) > 1e-16) {
                    colinds[dataInd] = colPlusDiag + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }
            }

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
                                 m_dt*m_timeInd, m_bsize);
    }
    else {
        getSpatialDiscretization(m_spatialComm, T_rowptr, T_colinds, T_data,
                                 B, X, localMinRow, localMaxRow, spatialDOFs,
                                 m_dt*(m_timeInd-1), m_bsize);
    }
    int procRows = localMaxRow - localMinRow + 1;
    int nnzPerTime = T_rowptr[procRows] - T_rowptr[0];    

    // Get mass matrix and scale by 1/dt
    int* M_rowptr;
    int* M_colinds;
    double* M_data;
    getMassMatrix(M_rowptr, M_colinds, M_data);
    for (int i=0; i<M_rowptr[procRows]; i++) {
        M_data[i] /= m_dt;
        // if (m_globRank == 3) std::cout << M_data[i] << ", ";
    }
    int nnzMPerTime = M_rowptr[procRows] - M_rowptr[0];    

    // Get size/nnz of spatial discretization and total for rows on this processor.
    // Allocate CSR structure.
    int procNnz;
    if (m_timeInd == 0) procNnz = nnzMPerTime;
    else procNnz = nnzPerTime + 2*nnzMPerTime;

    // Allocate CSR structure.
    rowptr  = new int[procRows + 1];
    colinds = new int[procNnz];
    data = new double[procNnz];
    int dataInd = 0;
    rowptr[0] = 0;

    // Mass matrix and spatial matrix have column indices global w.r.t. the spatial
    // discretization. Get global index w.r.t. space-time discretization.
    int colPlusOffd = (m_timeInd - 1)*spatialDOFs;
    int colPlusDiag = m_timeInd*spatialDOFs;
    std::map<int, double>::iterator it;

    // At time t=0, have fixed initial condition. Set matrix to identity
    // and fix RHS and initial guess to ICs.
    if (m_timeInd == 0) {
        
        // Add initial condition to rhs
        addInitialCondition(m_spatialComm, B);

        // Loop over each on-processor row in spatial discretization at time m_timeInd
        for (int i=0; i<procRows; i++) {

            colinds[dataInd] = localMinRow + i;
            data[dataInd] = 1.0;
            dataInd += 1;
            rowptr[i+1] = dataInd;

            // Set solution equal to initial condition
            X[i] = B[i];
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
                if (std::abs(it->second) > 1e-16) {
                    colinds[dataInd] = colPlusOffd + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }
            }

            // Add mass matrix as diagonal block, M/dt
            for (int j=M_rowptr[i]; j<M_rowptr[i+1]; j++) {                
                if (std::abs(M_data[j]) > 1e-16) {
                    colinds[dataInd] = colPlusDiag + M_colinds[j];
                    data[dataInd] = M_data[j];
                    dataInd += 1;
                }
            }

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

