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


/* Update member variables for describing which component of the identity 
mass-matrix the current process owns 

This is a helper function for when the spatial discretization doesn't use a mass 
matrix but we still need one in the form of an identity matrix

TODO : This is a little funny... I don't think the member variable should change over the life 
of the code since I think a process will always own the same local range of a spatial
discretization... But not convinced due to the way some of Ben's code is written, so just account for 
the possiblity that they might change
*/
void SpaceTimeMatrix::setIdentityMassLocalRange(int localMinRow, int localMaxRow) 
{    
    // Reset local range if : Not previously set, has changed, or the rebuild flag has been set
    if (m_M_localMinRow == -1 || m_M_localMaxRow == -1 || localMinRow != m_M_localMinRow || localMaxRow != m_M_localMaxRow || m_rebuildMass) {
        m_M_localMinRow = localMinRow;
        m_M_localMaxRow = localMaxRow;
        m_rebuildMass   = true; // Ensures identity mass matrix will be build or rebuilt
    }
}


/* Assmble some component of an identity mass matrix based on the member variables incidcating the local range owned by current process */
void SpaceTimeMatrix::getMassMatrix(int * &M_rowptr, int * &M_colinds, double * &M_data)
{
    // Ensure this function is not being called when the user has indicated that the spatial discretization does use a mass matrix
    if (m_M_exists) {
        std::cout << "WARNING: Spatial discretization subclass must implement mass matrix!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    
    // Ensure that local range variables have been set before we try to assmble
    if (m_M_localMinRow == -1 || m_M_localMaxRow == -1) {
        std::cout << "WARNING: Local range of identity mass matrix must be set before 'getMassMatrix' is called" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    // Only build or rebuild mass matrix if this flag has been set
    if (m_rebuildMass) {
        m_rebuildMass = false; // Don't rebuild unless this flag is changed
        
        // Free existing member variables since we're updating them
        if (m_M_rowptr)  delete[] m_M_rowptr;
        if (m_M_colinds) delete[] m_M_colinds;
        if (m_M_data)    delete[] m_M_data;
        
        int onProcSize = m_M_localMaxRow - m_M_localMinRow + 1; // Number of rows to assemble

        // Build some component of the sparse identity
        m_M_rowptr    = new int[onProcSize+1];
        m_M_colinds   = new int[onProcSize];
        m_M_data      = new double[onProcSize];
        m_M_rowptr[0] = 0;
        int rowcount  = 0;
        for (int row = m_M_localMinRow; row <= m_M_localMaxRow; row++) {
            m_M_colinds[rowcount]  = row;
            m_M_data[rowcount]     = 1.0;
            m_M_rowptr[rowcount+1] = rowcount+1;
            rowcount += 1;
        }
        
        // As a test for inverting mass matrix, make it be the 1D Laplacian....
        // int nnz = 3*onProcSize;
        // m_M_rowptr    = new int[onProcSize+1];
        // m_M_colinds   = new int[nnz];
        // m_M_data      = new double[nnz];
        // m_M_rowptr[0] = 0;
        // int dataInd = 0;
        // int rowcount = 0;
        // for (int row = m_M_localMinRow; row <= m_M_localMaxRow; row++) {
        //     if (row == m_M_localMinRow) {
        //         m_M_data[dataInd] = 2.0;
        //         m_M_colinds[dataInd] = row;
        //         dataInd += 1;
        //         m_M_data[dataInd] = -1.0;
        //         m_M_colinds[dataInd] = row+1;
        //         dataInd += 1;
        //     } else if (row == m_M_localMaxRow) {
        //         m_M_data[dataInd] = -1.0;
        //         m_M_colinds[dataInd] = row-1;
        //         dataInd += 1;
        //         m_M_data[dataInd] = 2.0;
        //         m_M_colinds[dataInd] = row;
        //         dataInd += 1;
        //     } else {
        //         m_M_data[dataInd] = -1.0;
        //         m_M_colinds[dataInd] = row-1;
        //         dataInd += 1;
        //         m_M_data[dataInd] = 2.0;
        //         m_M_colinds[dataInd] = row;
        //         dataInd += 1;
        //         m_M_data[dataInd] = -1.0;
        //         m_M_colinds[dataInd] = row+1;
        //         dataInd += 1;
        //     }
        //     m_M_rowptr[rowcount+1] = dataInd;
        //     rowcount += 1;
        // }
    }
    
    // Direct input references to existing member variables
    M_rowptr  = m_M_rowptr;
    M_colinds = m_M_colinds; 
    M_data    = m_M_data; 
}


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




SpaceTimeMatrix::SpaceTimeMatrix(MPI_Comm globComm, bool pit, bool M_exists, 
                                    int timeDisc, int nt, double dt)
    : m_globComm{globComm}, m_pit{pit}, m_M_exists{M_exists}, m_timeDisc{timeDisc}, m_nt{nt}, m_dt{dt},
      m_solverComm(NULL), m_solver(NULL), m_gmres(NULL), m_pcg(NULL), m_bij(NULL), m_xij(NULL), m_Aij(NULL),
      m_u_multi({}), m_u_multi_ij({}),
      m_Mij(NULL), m_invMij(NULL), m_iterative(true), 
      m_RK(false), m_ERK(false), m_DIRK(false), m_SDIRK(false),
      m_multi(false), m_AB(false), m_AM(false), m_BDF(false),
      m_a_multi({}), m_b_multi({}), 
      m_M_rowptr(NULL), m_M_colinds(NULL), m_M_data(NULL), m_rebuildSolver(false),
      m_spatialComm(NULL), m_L_isTimedependent(true), m_g_isTimedependent(true),
      m_bsize(1), m_hmin(-1), m_hmax(-1),
      m_M_localMinRow(-1), m_M_localMaxRow(-1),  m_rebuildMass(true)
{
    m_isTimeDependent = true; // TODO : this will need to be removed later... when L and G are treated separately
    
    // Get number of processes
    MPI_Comm_rank(m_globComm, &m_globRank);
    MPI_Comm_size(m_globComm, &m_numProc);


    // Swap AM0 to BDF1 to simplify implementation of AM schemes.
    if (m_timeDisc == 20) {
        m_timeDisc = 31;
        std::cout << "WARNING: 1st-order Adams--Moulton is equivalent to BDF1 and so a BDF1 implementation will be used!\n";
    }


    // Runge-Kutta time integration
    if (m_timeDisc >= 111 && m_timeDisc < 300) {
        m_RK = true;
        GetButcherTableaux(); // Get RK Butcher tabelaux
        
    // Adams--Bashforth time integration
    } else if (m_timeDisc >= 10 && m_timeDisc < 20) {
        m_multi = true;
        m_AB = true;
        SetABTableaux(); // Get AB coefficients
        
        std::cout << "WARNING: AB TIME INTEGRATION NOT IMPLEMENTED" << '\n';
        MPI_Finalize();
        exit(1);
        
    // Adams--Moulton time integration
    } else if (m_timeDisc >= 20 && m_timeDisc < 30) {
        m_multi = true;
        m_AM = true;
        SetAMTableaux(); // Get AM coefficients
        
        std::cout << "WARNING: AM TIME INTEGRATION NOT IMPLEMENTED" << '\n';
        MPI_Finalize();
        exit(1);
        
    // BDF time integration    
    } else if (m_timeDisc >= 30 && m_timeDisc < 40) {
        m_multi = true;
        m_BDF = true;
        SetBDFTableaux(); // Get BDF coefficients
        
    } else {
        if (m_globRank == 0) std::cout << "Format of temporal-discretization '"<< m_timeDisc <<"' is invalid!\n";   
        MPI_Finalize();
        exit(1);
    }
    
    // if (!m_a_multi.empty()) {
    //     std::cout << "a = " << '\n';
    //     for (std::vector<int>::size_type i = 0; i < m_a_multi.size(); i++) std::cout << m_a_multi[i] << '\n';
    // }
    // if (!m_b_multi.empty()) {
    //     std::cout << "b = " << '\n';
    //     for (std::vector<int>::size_type i = 0; i < m_b_multi.size(); i++) std::cout << m_b_multi[i] << '\n';
    // }
    
    // MULTISTEP: Ensure number of time steps is at least number of starting values 
    // We do nt-1 steps from t=0 (integrate to u[nt-1]), and require s starting values
    if (m_multi && m_nt <= m_s_multi) {
        std::cout << "WARNING: Cannot integrate'" << m_nt-1 << "' steps from t=0 with a multistep scheme requiring '" << m_s_multi << "' starting values\n";
        MPI_Finalize();
        exit(1);
    }
    
    /* ---------------------------------------------- */
    /* ------ Solving global space-time system ------ */
    /* ---------------------------------------------- */
    if (m_pit) {
        m_solverComm = m_globComm; // All solves are done on global communicator
        
        /* ------------------------------------ */
        /* --- Runge-Kutta time integration --- */
        /* ------------------------------------ */
        if (m_RK) {        
            // Check that number of time steps times number of stages divides the number MPI processes or vice versa.
            /* ------ Temporal + spatial parallelism ------ */
            if (m_numProc > (m_nt * m_s_butcher)) {
                if (m_globRank == 0) std::cout << "Space-time system: Spatial + temporal parallelism!\n";
                
                m_useSpatialParallel = true;
                if (m_numProc % (m_nt * m_s_butcher) != 0) {
                    if (m_globRank == 0) std::cout << "Error: number of processes " << m_numProc << " does not divide number of time points (" << m_nt << ") * number of RK stages (" << m_s_butcher << ") == " << m_nt * m_s_butcher << "\n";
                    MPI_Finalize();
                    exit(1);
                }
                else {
                    m_spatialCommSize = m_numProc / (m_nt * m_s_butcher);
                    m_Np_x = m_spatialCommSize; // TODO : remove. 
                }
                 
                // Set up communication group for spatial discretizations.
                m_timeInd = m_globRank / m_Np_x; // TODO. Delete this...
                m_DOFInd = m_globRank / m_Np_x; // Temporal DOF current process owns
                MPI_Comm_split(m_globComm, m_DOFInd, m_globRank, &m_spatialComm);
                MPI_Comm_rank(m_spatialComm, &m_spatialRank);
                MPI_Comm_size(m_spatialComm, &m_spatialCommSize);
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
                }
                m_nDOFPerProc = (m_nt * m_s_butcher) / m_numProc; // Number of temporal DOFs per proc, be they solution and/or stage DOFs
                m_ntPerProc = m_nt / m_numProc; //  TOOD: delete... This variable is for the old implementation...     
            }
        
        
        /* ---------------------------- */
        /* --- BDF time integration --- */
        /* ---------------------------- */
        } else if (m_BDF) {
            /* ------ Temporal + spatial parallelism ------ */
            if (m_numProc > m_nt-m_s_multi) {
                if (m_globRank == 0) std::cout << "Space-time system: Spatial + temporal parallelism!\n";      
                
                m_useSpatialParallel = true;
                
                // Ensure DOFs can be evenly split among processors 
                if (m_numProc % (m_nt-m_s_multi) != 0) {
                    if (m_globRank == 0) std::cout << "Error: total procs (" << m_numProc << ") does not evenly divide total number of DOFs (number of time points (" << m_nt << ") - number of starting values (" << m_s_multi << ") == " << m_nt-m_s_multi << ") \n";
                    MPI_Finalize();
                    exit(1);
                }
                else {
                    m_spatialCommSize = m_numProc / (m_nt - m_s_multi);
                }
                 
                // Set up communication group for spatial discretizations.
                m_DOFInd = m_globRank / m_spatialCommSize; // Temporal DOF current process owns
                MPI_Comm_split(m_globComm, m_DOFInd, m_globRank, &m_spatialComm);
                MPI_Comm_rank(m_spatialComm, &m_spatialRank);
                MPI_Comm_size(m_spatialComm, &m_spatialCommSize);
            }
            
            
            /* ------ Temporal parallelism only ------ */
            else {
                if (m_globRank == 0) {
                    if (m_numProc > 1) std::cout << "Space-time system: Temporal parallelism only!\n";    
                    else std::cout << "Space-time system: No parallelism!\n";    
                }
                
                m_useSpatialParallel = false;
                if ( (m_nt-m_s_multi) % m_numProc != 0) {
                    if (m_globRank == 0) {
                        std::cout << "Error: total DOFs (number of time points (" << m_nt << ") - number of starting values (" << m_s_multi << ") == " << m_nt-m_s_multi << ") does not divide number of processes " << m_numProc << "\n";
                    }
                    MPI_Finalize();
                    exit(1);
                }
                m_nDOFPerProc = (m_nt-m_s_multi) / m_numProc; // Number of temporal DOFs per proc
                m_ntPerProc = m_nt / m_numProc; //  TOOD: delete... This variable is for the old implementation...     
                
                /* Setup spatial communicator to be accessed by sequential time-stepping routines 
                used to generate starting values can access it. Since there is no spatial parallelism, the 
                spatial communicator has only a single process on it. Need to do this because HYPRE matrices
                are distributed on spatial communicator in those routines */
                MPI_Comm_split(m_globComm, m_globRank, 0, &m_spatialComm);
                MPI_Comm_rank(m_spatialComm, &m_spatialRank);
                MPI_Comm_size(m_spatialComm, &m_spatialCommSize);
            }
            
        } else {
            std::cout << "WARNING: Only RK and BDF space-time system implemented" << '\n';
            MPI_Finalize();
            exit(1);
        }
    
    
    /* -------------------------------------- */
    /* ------ Sequential time stepping ------ */
    /* -------------------------------------- */
    } else {
        // Set spatial communicator to be the same as global communicator.
        // Note: This is for the purposes of distributing HYPRE matrices during time-stepping (even when there is no spatiall parallelism, i.e. a single process)
        m_spatialComm     = m_globComm;
        m_spatialRank     = m_globRank;
        m_spatialCommSize = m_numProc;
        m_Np_x = m_spatialCommSize; // TODO delete...
        
        m_solverComm = m_spatialComm; // All solves are done on spatial communicator
        
        if (m_numProc > 1)  {
            if (m_globRank == 0) {
                std::cout << "Time-stepping: Spatial parallelism!\n";    
            }
            m_useSpatialParallel = true;
            
        } else {
            std::cout << "Time-stepping: No parallelism!\n";    
            m_useSpatialParallel = false;
        }  
    }
}





SpaceTimeMatrix::~SpaceTimeMatrix()
{
    DestroyHypreMemberVariables();
        
    // if (m_solver) HYPRE_BoomerAMGDestroy(m_solver);
    // if (m_gmres)  HYPRE_ParCSRGMRESDestroy(m_gmres);
    // if (m_pcg)    HYPRE_ParCSRPCGDestroy(m_pcg); 
    // if (m_Aij)    HYPRE_IJMatrixDestroy(m_Aij);   // This destroys parCSR matrix too
    // if (m_bij)    HYPRE_IJVectorDestroy(m_bij);   // This destroys parVector too
    // if (m_xij)    HYPRE_IJVectorDestroy(m_xij);
    // if (m_Mij)    HYPRE_IJMatrixDestroy(m_Mij);   
    // if (m_invMij) HYPRE_IJMatrixDestroy(m_invMij);   
    
    // TODO : destroy mass matrix member variables here...
}

/* General solve function, calls appropriate time integration routine */
void SpaceTimeMatrix::Solve() {
    
    // Solve space-time system
    if (m_pit) {
        
        // TODO : Potentially wrap timer around this code block. This encompasses the entirety of the code 
        // used to initialize multistep schemes.
        // Construct values to be inserted into RHS of space-time system to initialize a multistep scheme
        if (m_multi) {
            // Only have processes that need starting values actually obtain them
            bool iNeedStartValues = false;
            if (m_useSpatialParallel) {
                if (m_DOFInd < m_s_multi) iNeedStartValues = true; // First s DOFs need starting values
            } else {
                int firstDOFIOwn = m_globRank * m_nDOFPerProc; // First temporal DOF on process
                if (firstDOFIOwn < m_s_multi) iNeedStartValues = true; // First s DOFs need starting values
            }
            
            if (iNeedStartValues) {                
                m_solverComm = m_spatialComm;       // Solves during sequential time-stepping must be done on spatial communicator
                SetMultistepStartValues();          // Get required starting values
                SetMultistepSpaceTimeRHSValues();   // Construct values in space-time RHS vector from starting values
                
                // Esnure we free any member variables that we've used that may be used in the forthcoming space-time solve
                DestroyHypreMemberVariables();
                m_solverComm = m_globComm;          // Remaining space-time solve is done on global communicator    
            }
            
            
            /* Global syncronization point: When obtaining starting values, member variables 
            have been used (e.g., HYPRE vectors, matrices, and solvers) that will now be used for a 
            different purpose and potentially by a different communicator alltogether than they were previously.
            To avoid any potential issues, enforce that all processes are syncronized at this point.
            */
            MPI_Barrier(m_globComm);
        }
        
        // Build the space-time matrix
        BuildSpaceTimeMatrix(); 
        
        // Call appropiate solver
        if (m_solver_parameters.use_gmres) {
            SolveGMRES(); 
        } else {
            SolveAMG();
        }
        
        
    
    // sequential time-stepping
    } else {
        TimeSteppingSolve();
    }
}


/* Destroy all HYPRE member variables; this is necessary if switching 
    from time-stepping to space-time within an instance of the code.
 */
void SpaceTimeMatrix::DestroyHypreMemberVariables() 
{
    if (m_solver) {
        HYPRE_BoomerAMGDestroy(m_solver);
        m_solver = NULL;
    }
    if (m_gmres) {
        HYPRE_ParCSRGMRESDestroy(m_gmres);
        m_gmres = NULL;
    }
    if (m_pcg) {
        HYPRE_ParCSRPCGDestroy(m_pcg); 
        m_pcg = NULL;
    }   
    if (m_Aij) {
        HYPRE_IJMatrixDestroy(m_Aij);   // This destroys parCSR matrix too
        m_Aij = NULL;
    }
    if (m_bij) {
        HYPRE_IJVectorDestroy(m_bij);   // This destroys parVector too
        m_bij = NULL;
    }
    if (m_xij) {
        HYPRE_IJVectorDestroy(m_xij);
        m_xij = NULL;
    }
    if (m_Mij) {
        HYPRE_IJMatrixDestroy(m_Mij);   
        m_Mij = NULL;
    }
    if (m_invMij) { 
        HYPRE_IJMatrixDestroy(m_invMij);  
        m_invMij = NULL;
    }
    for (int i = 0; i < m_u_multi_ij.size(); i++) {
        if (m_u_multi_ij[i]) {
            HYPRE_IJVectorDestroy(m_u_multi_ij[i]);
            m_u_multi_ij[i] = NULL;
        }
    }
}


/* Given the s-starting values, m_u_multi, create the terms necessary to 
put them straight into the RHS of the space-time linear system 

NOTE:
    -This also frees m_u_multi at its completion since this is no longer needed.
*/
void SpaceTimeMatrix::SetMultistepSpaceTimeRHSValues()
{
    if (m_AB) {
        std::cout << "WARNING: CODE TO CONSTRUCT RHS FOR AB NOT WRITTEN" << '\n';
        MPI_Finalize();
        exit(1);
    } else if (m_AM) {
        std::cout << "WARNING: CODE TO CONSTRUCT RHS FOR AM NOT WRITTEN" << '\n';
        MPI_Finalize();
        exit(1);
    } else if (m_BDF) {
        
        // Compute vector that's multiplied by the mass matrix
        for (int n = 0; n <= m_s_multi - 1; n++) {            
            // This linear combination overwrites the start value it uses first, v[n]
            for (int j = m_s_multi; j >= n+1; j--) {
                if (j == m_s_multi) {
                    HYPRE_ParVectorScale(-m_a_multi[m_s_multi-j], m_u_multi[n]); // v[n] <- -a[0] * v[n]
                } else {
                    HYPRE_ParVectorAxpy(-m_a_multi[m_s_multi-j], m_u_multi[n+m_s_multi-j], m_u_multi[n]); // v[n] <- v[n] + a[s-j] * v[j-1]
                }
            }    
        }
        
        // Scale by mass matrix if one exists
        if (m_M_exists)  {
            if (!m_Mij) {
                std::cout << "WARNING: I SHOULD HAVE ALREADY ASSEMBLED THE MASS-MATRIX!" << '\n';
                MPI_Finalize();
                exit(1);
            }
            
            // Just use m_b as a temporary vector to do the MATVEC; this should be initialized from when starting values were obtained
            //  Actually that's not true... We might have only setup initial condition, or  could have  been free'd by the time-stepping routine itself...
            // TODO: Use b if it exists, if it does not, then create and initialize a new HYPRE vector, 
            // whose  structure can mimic that of any vector in m_u_multi
            // Actually, b probably does not exist here at all since it's free'd at the completion of the time-stepping...  
            if (!m_bij) {
                std::cout << "WARNING: I require that m_x be initialized here!" << '\n';
                MPI_Finalize();
                exit(1);
            }
            
            for (int i = 0; i < m_u_multi.size(); i++) {
                HYPRE_ParVectorCopy(m_u_multi[i], m_b); // b <- m_u_multi[i]
                hypre_ParCSRMatrixMatvec(1.0, m_M, m_b, 0.0, m_u_multi[i]); // m_u_multi[i] <- 1.0*M*b + 0.0*m_u_multi[i]
            }
        }
    }
    
    
    /* Insert RHS information into non-HYPRE vectors so it can be added directly
     into the space-time RHS during its construction */
    m_w_multi.resize(m_s_multi);
    for (int i = 0; i < m_s_multi; i++) {
        
        int ilower, iupper, onProcSize;
        int * indices;
        HYPRE_IJVectorGetLocalRange(m_u_multi_ij[i], &ilower, &iupper);
        onProcSize = iupper - ilower + 1; // Number  of rows on process
        
        indices = new int[onProcSize];
        for (int i = 0; i < onProcSize; i++) {
            indices[i] = ilower + i;
        }
        
        m_w_multi[i] = new double[onProcSize];
        HYPRE_IJVectorGetValues(m_u_multi_ij[i], onProcSize, indices, m_w_multi[i]);
        
        delete[] indices;
    
        // Free values in HYPRE data structures as no longer needed
        HYPRE_IJVectorDestroy(m_u_multi_ij[i]);
        m_u_multi_ij[i] = NULL;
    }
    m_u_multi    = {};
    m_u_multi_ij = {};
}


/* Call appropiate sequential time-stepping routine */
void SpaceTimeMatrix::TimeSteppingSolve()
{
    /* Runge-Kutta routines: Integrate nt-1 steps from time=0 */
    if (m_RK) {
        m_t0 = 0.0; // Set global starting time to 0
        
        // Setup initial condition for time-integration 
        HYPRE_ParVector u0;   
        HYPRE_IJVector  u0ij;
        GetHypreInitialCondition(u0, u0ij);
        
        // Copy initial condition into member vector so that RK routines can access it
        m_x   = u0;
        m_xij = u0ij;
        
        if (m_ERK) {
            ERKTimeSteppingSolve();
        } else if (m_DIRK) {
            DIRKTimeSteppingSolve();
        }
        
    /* Multistep routines: Need to initialize first s steps using Runge-Kutta 
    integration, then integrate up to t_{nt-1} */
    } else if (m_multi) {     
        // Populate member vectors with the s starting values we need
        SetMultistepStartValues();
        
        /* Call appropiate multstep routine */
        if (m_AB) {
            ABTimeSteppingSolve();
        } else if (m_AM) {
            AMTimeSteppingSolve();
        } else if (m_BDF) {
            BDFTimeSteppingSolve();
        }
    }
}


/* Populate member vectors, m_u_multi, m_u_multi_ij, with the s starting values
required to integrate with a multistep scheme 

NOTE:
    -Resets the starting time m_t0 to 0, so multstep schemes must automatically start from the right time.
*/
void SpaceTimeMatrix::SetMultistepStartValues() 
{
    if (m_globRank == 0) std::cout << "Initializing starting values for multistep time integration\n";
    
    // Temporarily store variables while they're reset for use in RK routines
    int  timeDisc_temp = m_timeDisc; 
    int  nt_temp       = m_nt;
    bool implicit_temp = m_implicit;
    
    // Get initial condition for time-integration 
    HYPRE_ParVector u0;   
    HYPRE_IJVector  u0ij;
    GetHypreInitialCondition(u0, u0ij);
    
    // Starting-values for multistep methods
    // We need to store s starting values of u, but will explicitly insert initial conditon below 
    m_u_multi.resize(m_s_multi - 1);
    m_u_multi_ij.resize(m_s_multi - 1);
    InitializeHypreVectors(u0, u0ij, m_u_multi, m_u_multi_ij);
    
    // Insert initial condition at front of starting values vector
    m_u_multi.insert(m_u_multi.begin(), u0); // TODO : Is this really inefficient???
    m_u_multi_ij.insert(m_u_multi_ij.begin(), u0ij);
    
    // Set global starting time to 0
    m_t0 = 0.0; 
    
    /* If we need more starting values obtain them via sequential RK integration */
    if (SetMultiRKPairing()) {
        m_nt = 2; // We take only a single step at a time (RK routines are hard-codes to take m_nt-1 steps)
        
        // Get the remaining s-1 starting values, u_n
        for (int n = 1; n < m_s_multi; n++) {
            
            // Copy initial starting values into a location such that RK routines will overwrite them with solution at the end of 1 time step
            HYPRE_ParVectorCopy(m_u_multi[n-1], m_u_multi[n]); // u[n] <- u[n-1]
            
            // Copy initial value into member vector so that RK routines can access it
            m_x   = m_u_multi[n];
            m_xij = m_u_multi_ij[n];
            
            // Call appropriate RK solver
            if (m_ERK) {
                ERKTimeSteppingSolve();
            } else if (m_DIRK) {
                DIRKTimeSteppingSolve();
            }
            
            m_t0 += m_dt; // Update "starting time" of integration
        }
        
        // Point member vector x back to NULL; NOTE: it was never allocated any memory, 
        // but just pointed to an element of m_u_multi so that the RK routines could access it. 
        m_x   = NULL;
        m_xij = NULL;
        
        // Clear Runge-Kutta data structures so there's no possible confusion in future
        m_s_butcher = -1;
        m_c_butcher = {};
        m_b_butcher = {};
        m_A_butcher = {};
    }
    
    // Reset variables to their original values
    m_t0       = 0.0;
    m_timeDisc = timeDisc_temp;
    m_nt       = nt_temp;
    m_implicit = implicit_temp;
}


/* Sequential time-stepping routine for arbitrary Adams--Bashforth schemes */
void SpaceTimeMatrix::ABTimeSteppingSolve() 
{
}

/* Sequential time-stepping routine for arbitrary Adams--Moulton schemes */
void SpaceTimeMatrix::AMTimeSteppingSolve() 
{
}

/* Sequential time-stepping routine for arbitrary BDF schemes 

    After this function has executed, the solution at the final time is stored in 
        m_x, which is just a pointer to the head of the list m_u_multi

NOTE:
    -m_u_multi must contain the s starting values required to initiate s-step BDF; 
    that is, m_u_multi==[u(t_0),...,u(t_{s-1})], s>=1.
*/
void SpaceTimeMatrix::BDFTimeSteppingSolve() 
{
    /* ---------------------------------------------------------------------- */
    /* ------------------------ Setup/initialization ------------------------ */
    /* ---------------------------------------------------------------------- */
    double t = (m_s_multi-1) * m_dt; // Initial time to integrate from. The current solution is known at t_{s-1}
    
    // Check that solution vector has been initialized!
    if (m_u_multi_ij.empty()) {
        std::cout << "WARNING: Global solution vector must be allocated before beginning BDF time stepping" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    int headptr = m_s_multi - 1;    // Pointer to solution at most recent time, u(t_n)
    int tailptr = 0;                // Pointer to solution at last required time, u(t_{n+1-s})
    
    // TODO : Should rename m_x to m_u...
    // Shallow copy starting values into vectors with more meaningful names
    std::vector<HYPRE_ParVector> u   = m_u_multi;   // Initial solution vector to integrate from
    std::vector<HYPRE_IJVector>  uij = m_u_multi_ij;
    
    HYPRE_ParVector    g             = NULL; // Spatial discretization vector
    HYPRE_IJVector     gij           = NULL;
    HYPRE_ParCSRMatrix L             = NULL; // Spatial discretization matrix  
    HYPRE_IJMatrix     Lij           = NULL;
    HYPRE_ParCSRMatrix BDF_matrix    = NULL; // Matrix to be inverted in linear solve
    HYPRE_IJMatrix     BDF_matrixij  = NULL;

    // Place-holder vectors
    std::vector<HYPRE_ParVector> vectors;
    std::vector<HYPRE_IJVector>  vectorsij;
    int numVectors = 1; // We only need a single temporary vector (to store RHS of the linear system)

    // Initialize place-holder vectors
    vectors.resize(numVectors);
    vectorsij.resize(numVectors);
    InitializeHypreVectors(u[0], uij[0], vectors, vectorsij);

    // Shallow copy vectors into variables with meaningful names
    HYPRE_ParVector b   = vectors[0];  // Temporary vector
    HYPRE_IJVector  bij = vectorsij[0];

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
    int    num_iters   = 0;
    int    solve_count = 0; // Number of linear solves
    double rel_res_norm;
    double avg_iters   = 0;
    double avg_convergence_rate;

    // Is it necessary to build spatial discretization matrix/BDF matrix more than once?
    bool rebuildMatrix = m_L_isTimedependent;

    /* ------------------------------------------------------------ */
    /* ------------------------ Time march ------------------------ */
    /* ------------------------------------------------------------ */
    int step = 0; // Counter for how many time steps we take
        
    // Advance the solution from its known values at the current time, tInd*m_dt, to new time, (tInd+1)*dt.
    // For a multistep scheme, the solution is initially known at (s-1)*dt (and the s-1 times before it)
    // Solve up to time t == (m_nt-1)*dt
    for (int tInd = m_s_multi-1; tInd < m_nt - 1; tInd++) {
        /* -------------- Build RHS vector, b, in linear system (M+b_s*dt*L)*u[n+1]=b[n+1] -------------- */
        if (m_solver_parameters.printLevel > 0 && m_spatialRank == 0) {
            std::cout << "\nSolving for time level " << tInd+1 << " of " << m_nt-1 << "\n"; 
            std::cout << "-----------------------------------------\n\n";
        }
        
        // Compute spatial discretization at t + dt
        // Solution-independent term
        if (m_g_isTimedependent || step == 0) {
            GetHypreSpatialDiscretizationG(g, gij, t + m_dt);
        }
        // Solution-dependent term
        if (rebuildMatrix || step == 0) {
            GetHypreSpatialDiscretizationL(L, Lij, t + m_dt);
        } 
        
        // Assemble intermediate variable w, w == \sum_{j=1}^s a[s-j] * u[n+1-j], this is stored in u[tail]
        for (int j = m_s_multi-1; j >= 0; j--) { // Start sum from u[n+1-s] since  this is overwritten
            if (j == m_s_multi-1) {
                HYPRE_ParVectorScale(m_a_multi[0], u[tailptr]); // u[n+1-s] <- a[0] * u[n+1-s]
            } else {
                HYPRE_ParVectorAxpy(m_a_multi[m_s_multi-1 - j], u[(tailptr + m_s_multi-1 - j) % m_s_multi], u[tailptr]); // u[n+1-s] <- u[n+1-s] + a[s-j]*u[n+1-j]
            }
        }
        
        // Multiply w by mass matrix and add g. Note for BDF schemes, m_b_multi stores only b_s!
        if (m_M_exists)  {
            // Assemble as HYPRE matrix to compute MATVECS  
            if (step == 0) {
                int ilower, iupper, jdummy1, jdummy2;
                HYPRE_IJMatrixGetLocalRange(Lij, &ilower, &iupper, &jdummy1, &jdummy2);
                SetHypreMassMatrix(ilower, iupper);
            }
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, m_M, u[tailptr], m_dt*m_b_multi[0], g, b); // b <- -M*w + dt*b_s*g
        // Add g to w
        } else {
            HYPRE_ParVectorCopy(u[tailptr], b); // b <- u[tail]
            HYPRE_ParVectorScale(-1.0, b); // b <- -b
            HYPRE_ParVectorAxpy(m_dt*m_b_multi[0], g, b); // b <- b + dt*b_s*g
        }
        
        // Set inital guess at solution to be the RHS of the system
        HYPRE_ParVectorCopy(b, u[tailptr]); 
        
        
        /* -------------- Solve linear system, (M+b_s*dt*L)*u[n+1]=b -------------- */
        // Get components of mass matrix only after spatial discretization has been assembled for the first time
        // Mass matrix is NOT time dependent: Only needs to be assembled once
        if (step == 0) {
            // Get rows this process owns of M assuming rows of M and L are partitioned the same in memory
            int jdummy1, jdummy2;
            HYPRE_IJMatrixGetLocalRange(Lij, &ilower, &iupper, &jdummy1, &jdummy2);
            
            // Setup range of identity matrix to assemble if spatial discretization doesn't use a mass matrix
            if (!m_M_exists) setIdentityMassLocalRange(ilower, iupper);
            
            getMassMatrix(M_rowptr, M_colinds, M_data);
            onProcSize     = iupper - ilower + 1;
            M_rows         = new int[onProcSize];
            M_cols_per_row = new int[onProcSize];
            for (int rowIdx = 0; rowIdx < onProcSize; rowIdx++) {
                M_rows[rowIdx]         = ilower + rowIdx;
                M_cols_per_row[rowIdx] = M_rowptr[rowIdx+1] - M_rowptr[rowIdx];
            }
            M_scaled_data = new double[M_rowptr[onProcSize]]; // Temporary vector to use below...
        }
        
        // Scale RHS vector by 1/dt*a_ii for the moment 
        double temp = 1.0/(m_dt * m_b_multi[0]); 
        HYPRE_ParVectorScale(temp, b); // b <- b/(dt*b_s)
        
        // Rescale mass matrix data by 1/dt*b_s; only need to do this once
        if (step == 0) {
            for (int dataInd = 0; dataInd < M_rowptr[onProcSize]; dataInd++) {
                M_scaled_data[dataInd] = temp * M_data[dataInd]; // M <- M / (dt*b_s)
            }
        }
        
        // Get BDF matrix, BDF_matrix <- M + b_s*dt*L
        // Build BDF matrix once only if L time independent
        if (!rebuildMatrix) {
            // Build BDF matrix on first iteration only
            if (step == 0) { 
                GetHypreSpatialDiscretizationL(BDF_matrix, BDF_matrixij, t + m_dt);
                HYPRE_IJMatrixAddToValues(BDF_matrixij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_scaled_data);            
                m_rebuildSolver = true; // Ensure solver rebuilt on first iteration (may not be depending on how initial values were obtained)
            }
        // Reuse/update L since it's rebuilt at next iteration, BDF_matrix <- L <- M + b_s*dt*L    
        } else {
            HYPRE_IJMatrixAddToValues(Lij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_scaled_data);            
            BDF_matrix   = L;
            BDF_matrixij = Lij;
            
            // BDF matrix has changed, check if AMG solver is due to be rebuild
            // (m_rebuildSolver is reset to false when the solver (re)built)
            if (m_solver_parameters.rebuildRate == 0 || (m_solver_parameters.rebuildRate > 0 && (step % m_solver_parameters.rebuildRate) == 0)) m_rebuildSolver = true; 
        }
        
        // Point member variables to local variables so appropiate linear solver can access them
        m_A = BDF_matrix;
        m_x = u[tailptr]; 
        m_b = b;
        // Solve linear system and get convergence statistics
        if (m_solver_parameters.use_gmres) {
            SolveGMRES();
        } else {
            SolveAMG();
        }
        // Reset member variables to NULL
        m_A = NULL;
        m_x = NULL;
        m_b = NULL;
        solve_count += 1;
        avg_iters   += (double) m_num_iters;
        // avg_convergence_rate += ...; // TODO : Not sure how to do
        
        // Ensure desired tolerance was reached in allowable number of iterations, otherwise quit
        if (m_res_norm > m_solver_parameters.tol) {
            if (m_spatialRank == 0) std::cout << "=================================\n =========== WARNING ===========\n=================================\n";
            if (m_spatialRank == 0) std::cout << "Time step " << step+1 << "/" << m_nt-1 << "\n";
            if (m_spatialRank == 0) std::cout << "Tol after " << m_num_iters << " iters (max iterations) = " << m_res_norm << " > desired tol = " << m_solver_parameters.tol << "\n\n";
            MPI_Finalize();
            exit(1);
        }

        t    += m_dt; // Increment time
        step += 1;    // Increment step counter
        
        // Update tailptr and headptr for next iteration
        headptr = tailptr;
        tailptr = (tailptr + 1) % m_s_multi;
    }


    // Print statistics about average iteration counts and convergence factor across whole time interval
    if (step > 0 && m_solver_parameters.printLevel > 0 && m_spatialRank == 0) {
        std::cout << "=============================================\n";
        std::cout << "Summary of linear solves during time stepping\n";
        std::cout << "---------------------------------------------\n";
        std::cout << "Number of systems solved = " << solve_count << '\n';
        std::cout << "Average number of iterations = " << avg_iters/solve_count << '\n';
        std::cout << "***TODO***: Average convergence factor = ..." << '\n';
        //hypre_BoomerAMGGetRelResidualNorm(m_solver, &rel_res_norm);
    }

    /* ---------------------------------------------------------- */
    /* ------------------------ Clean up ------------------------ */
    /* ---------------------------------------------------------- */
    for (int i = 0; i < vectors.size(); i++) {
        HYPRE_IJVectorDestroy(vectorsij[i]);
    }
    
    if (Lij) HYPRE_IJMatrixDestroy(Lij);
    if (BDF_matrixij && BDF_matrixij != Lij) HYPRE_IJMatrixDestroy(BDF_matrixij); // BDF_matrix is distinct from L

    // Free all starting values except the one holding the solution at the final time
    for (int i = 0; i < m_u_multi_ij.size(); i++) {
        if (m_u_multi_ij[i] && i != headptr) {
            HYPRE_IJVectorDestroy(m_u_multi_ij[i]);
            m_u_multi_ij[i] = NULL;
        }
    }

    // Point member variable x to final solution
    m_x   = u[headptr];
    m_xij = uij[headptr];
    // Set corresponding u_multi element to NULL since x now "owns" this block of memory
    m_u_multi[headptr]    = NULL;
    m_u_multi_ij[headptr] = NULL;
}



/* Get Runge-Kutta tableaux that's compatible 
with underlying multistep scheme 

NOTE:
    -If we do not need to do RK integration (as when doing 1st-order integration) then return false
*/
bool SpaceTimeMatrix::SetMultiRKPairing() 
{
    // Adams--Bashforth: Use order s_multi ERK time-stepping
    if (m_AB) {
        // Higher than 1st-order integration requires starting values
        if (m_timeDisc != 11) {
            if      (m_timeDisc == 12) m_timeDisc = 122; // 2nd-order
            else if (m_timeDisc == 13) m_timeDisc = 133; // 3rd-order
            else if (m_timeDisc == 14) m_timeDisc = 144; // 4th-order
        // 1st-order only needs initial condition
        } else {
            return false;
        }
        
    // Adams--Moulton: Use order s_multi+1 DIRK time-stepping
    } else if (m_AM) {
        if      (m_timeDisc == 21) m_timeDisc = 222; // 2nd-order
        else if (m_timeDisc == 22) m_timeDisc = 233; // 3rd-order
        else if (m_timeDisc == 23) m_timeDisc = 254; // 4th-order

    // BDF: Use order s_multi DIRK time-stepping
    } else if (m_BDF) {
        // Higher than 1st-order integration requires starting values
        if (m_timeDisc != 31) {
            if      (m_timeDisc == 32) m_timeDisc = 222; // 2nd-order
            else if (m_timeDisc == 33) m_timeDisc = 233; // 3rd-order
            else if (m_timeDisc == 34) m_timeDisc = 254; // 4th-order
        } else {
            return false;
        }
    }
    
    GetButcherTableaux();   
    return true;
}


/* Sequential time-stepping routine for arbitrary DIRK schemes

    After this function has executed, the solution at the final time is stored in 
    the member vector m_x  
*/
void SpaceTimeMatrix::DIRKTimeSteppingSolve() 
{   
    /* ---------------------------------------------------------------------- */
    /* ------------------------ Setup/initialization ------------------------ */
    /* ---------------------------------------------------------------------- */
    double t = m_t0;        // Initial time to integrate from
    
    // Check that solution vector has been initialized!
    if (!m_xij) {
        std::cout << "WARNING: Global solution vector must be allocated before beginning RK time stepping" << '\n';
        MPI_Finalize();
        exit(1);
    }
    // Shallow copy member variables into variables with more appropiate names
    HYPRE_ParVector    u   = m_x;   // Initial solution vector to integrate from
    HYPRE_IJVector     uij = m_xij;
    
    HYPRE_ParVector    g             = NULL; // Spatial discretization vector
    HYPRE_IJVector     gij           = NULL;
    HYPRE_ParCSRMatrix L             = NULL; // Spatial discretization matrix  
    HYPRE_IJMatrix     Lij           = NULL;
    HYPRE_ParCSRMatrix DIRK_matrix   = NULL; // Matrix to be inverted in linear solve
    HYPRE_IJMatrix     DIRK_matrixij = NULL;

    // Place-holder vectors
    std::vector<HYPRE_ParVector> vectors;
    std::vector<HYPRE_IJVector>  vectorsij;
    int numVectors = m_s_butcher + 2; // Have s stage vectors + 2 temporary vectors

    // Get initial condition and initialize place-holder vectors
    vectors.resize(numVectors);
    vectorsij.resize(numVectors);
    InitializeHypreVectors(u, uij, vectors, vectorsij);

    // Shallow copy vectors into variables with meaningful names
    HYPRE_ParVector              b1   = vectors[0];  // Temporary vector
    HYPRE_IJVector               b1ij = vectorsij[0];
    HYPRE_ParVector              b2   = vectors[1];  // Temporary vector
    HYPRE_IJVector               b2ij = vectorsij[1];
    std::vector<HYPRE_ParVector> k; // Stage vectors
    std::vector<HYPRE_IJVector>  kij;    
    k.resize(m_s_butcher);
    kij.resize(m_s_butcher);
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
    double avg_iters = 0;
    double avg_convergence_rate;


    // Is it necessary to build spatial discretization matrix/DIRK matrix more than once?
    bool rebuildMatrix = ((m_L_isTimedependent) || (!m_SDIRK));

    /* ------------------------------------------------------------ */
    /* ------------------------ Time march ------------------------ */
    /* ------------------------------------------------------------ */
    // Take nt-1 steps
    int solve_count = 0; // Number of linear solves
    int step = 0;
    for (step = 0; step < m_nt-1; step++) {
        /* -------------- Build RHS vector, b2, in linear system (M+a_ii*dt*L)*k[i]=b2 -------------- */
        for (int i = 0; i < m_s_butcher; i++) {
            if ((m_solver_parameters.printLevel > 0) && (m_spatialRank == 0)) {
                std::cout << "Time step " << step+1 << "/" << m_nt-1 << ": Solving for stage " << i+1 << "/" << m_s_butcher << '\n';
                std::cout << "-----------------------------------------\n\n";
            }
            
            // Compute spatial discretization at t + c[i]*dt
            // Solution-independent term
            if (m_g_isTimedependent || (i == 0 && step == 0)) {
                //HYPRE_IJVectorDestroy(gij);
                GetHypreSpatialDiscretizationG(g, gij, t + m_dt * m_c_butcher[i]);
            }
            // Solution-dependent term
            if (rebuildMatrix || (i == 0 && step == 0)) {
                //HYPRE_IJMatrixDestroy(Lij);
                GetHypreSpatialDiscretizationL(L, Lij, t + m_dt * m_c_butcher[i]);
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
                // Get rows this process owns of M assuming rows of M and L are partitioned the same in memory
                int jdummy1, jdummy2;
                HYPRE_IJMatrixGetLocalRange(Lij, &ilower, &iupper, &jdummy1, &jdummy2);
                
                // Setup range of identity matrix to assemble if spatial discretization doesn't use a mass matrix
                if (!m_M_exists) setIdentityMassLocalRange(ilower, iupper);
                
                getMassMatrix(M_rowptr, M_colinds, M_data);
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
            if (!m_SDIRK || ((step == 0) && (i == 0))) {
                for (int dataInd = 0; dataInd < M_rowptr[onProcSize]; dataInd++) {
                    M_scaled_data[dataInd] = temp * M_data[dataInd]; // M <- M / (dt*a_ii)
                }
            }
            
            // Get DIRK matrix, DIRK_matrix <- M + a_ii*dt*L
            // Build DIRK matrix once only if L time independent, and using SDIRK
            if (!rebuildMatrix) {
                // Build DIRK matrix on first iteration only
                if ((step == 0) && (i == 0)) { 
                    GetHypreSpatialDiscretizationL(DIRK_matrix, DIRK_matrixij, t);
                    HYPRE_IJMatrixAddToValues(DIRK_matrixij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_scaled_data);            
                }
            // Reuse/update L since it's rebuilt at next iteration, DIRK_matrix <- L <- M + a_ii*dt*L    
            } else {
                HYPRE_IJMatrixAddToValues(Lij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_scaled_data);            
                DIRK_matrix   = L;
                DIRK_matrixij = Lij;
                
                // DIRK matrix has changed, check if AMG solver is due to be rebuild
                // (m_rebuildSolver is reset to false when the solver (re)built)
                if (m_solver_parameters.rebuildRate == 0 || (m_solver_parameters.rebuildRate > 0 && i == 0 && (step % m_solver_parameters.rebuildRate) == 0)) m_rebuildSolver = true; 
            }
            
            // Point member variables to local variables so appropiate linear solver can access them
            m_A = DIRK_matrix;
            m_x = k[i]; // Initial guess at solution is value from previous time step
            m_b = b2;
            // Solve linear system and get convergence statistics
            if (m_solver_parameters.use_gmres) {
                SolveGMRES();
            } else {
                SolveAMG();
            }
            // Reset member variables to NULL
            m_A = NULL;
            m_x = NULL;
            m_b = NULL;  
            solve_count += 1;
            avg_iters   += (double) m_num_iters;
            // avg_convergence_rate += ...; // TODO : Not sure how to do
            
                
            // Ensure desired tolerance was reached in allowable number of iterations, otherwise quit
            if (m_res_norm > m_solver_parameters.tol) {
                if (m_spatialRank == 0) std::cout << "=================================\n =========== WARNING ===========\n=================================\n";
                if (m_spatialRank == 0) std::cout << "Time step " << step+1 << "/" << m_nt-1 << ": Solving for stage " << i+1 << "/" << m_s_butcher << '\n';
                if (m_spatialRank == 0) std::cout << "Tol after " << m_num_iters << " iters (max iterations) = " << m_res_norm << " > desired tol = " << m_solver_parameters.tol << "\n\n";
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


    // Print statistics about average iteration counts and convergence factor across whole time interval
    if ((m_solver_parameters.printLevel > 0) && (m_spatialRank == 0)) {
        std::cout << "=============================================\n";
        std::cout << "Summary of linear solves during time stepping\n";
        std::cout << "---------------------------------------------\n";
        std::cout << "Number of systems solved = " << solve_count << '\n';
        std::cout << "Average number of iterations = " << avg_iters/solve_count << '\n';
        std::cout << "***TODO***: Average convergence factor = ..." << '\n';
        //hypre_BoomerAMGGetRelResidualNorm(m_solver, &rel_res_norm);
    }

    /* ---------------------------------------------------------- */
    /* ------------------------ Clean up ------------------------ */
    /* ---------------------------------------------------------- */
    for (int i = 0; i < vectors.size(); i++) {
        HYPRE_IJVectorDestroy(vectorsij[i]);
    }
    
    if (gij) HYPRE_IJVectorDestroy(gij);
    if (Lij) HYPRE_IJMatrixDestroy(Lij);
    if (DIRK_matrixij && DIRK_matrixij != Lij)  HYPRE_IJMatrixDestroy(DIRK_matrixij); // DIRK matrix was distinct from L
        
        
            
    // Point member variable x to final solution
    m_x   = u;
    m_xij = uij;
}


/* Sequential time-stepping routine for arbitrary ERK schemes 
    
    After this function has executed, the solution at the final time is stored in 
    the member vector m_x  
*/
void SpaceTimeMatrix::ERKTimeSteppingSolve() 
{        
    /* ---------------------------------------------------------------------- */
    /* ------------------------ Setup/initialization ------------------------ */
    /* ---------------------------------------------------------------------- */
    double t = m_t0;        // Initial time to integrate from
    
    // Check that solution vector has been initialized!
    if (!m_xij) {
        std::cout << "WARNING: Global solution vector must be allocated before beginning RK time stepping" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    // Shallow copy member variables into variables with more appropiate names
    HYPRE_ParVector    u   = m_x;  // Initial solution vector to integrate from
    HYPRE_IJVector     uij = m_xij;
    
    HYPRE_ParVector    g   = NULL; // Spatial discretization vector
    HYPRE_IJVector     gij = NULL;
    HYPRE_ParCSRMatrix L   = NULL; // Spatial discretization matrix  
    HYPRE_IJMatrix     Lij = NULL;

    // Place-holder vectors
    std::vector<HYPRE_ParVector> vectors;
    std::vector<HYPRE_IJVector>  vectorsij;
    int numVectors = m_s_butcher + 1; // Have s stage vectors + 1 temporary vector

    // Get initial condition and initialize place-holder vectors
    vectors.resize(numVectors);
    vectorsij.resize(numVectors);
    InitializeHypreVectors(u, uij, vectors, vectorsij);

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

    // For monitoring convergence of linear solver
    int    num_iters;
    double rel_res_norm;
    double avg_iters;
    double avg_convergence_rate;

    /* ------------------------------------------------------------ */
    /* ------------------------ Time march ------------------------ */
    /* ------------------------------------------------------------ */
    // Take nt-1 steps
    int solve_count = 0;
    int step = 0;
    for (step = 0; step < m_nt-1; step++) {
        
        // Build ith stage vector, k[i]
        for (int i = 0; i < m_s_butcher; i++) {
            if ((m_solver_parameters.printLevel > 0) && (m_spatialRank == 0)) {
                std::cout << "Time step " << step+1 << "/" << m_nt-1 << ": Solving for stage " << i+1 << "/" << m_s_butcher << '\n';
                std::cout << "-----------------------------------------\n\n";
            }
            
            // Compute spatial discretization at t + c[i]*dt
            if (m_g_isTimedependent || (i == 0 && step == 0)) {
                GetHypreSpatialDiscretizationG(g, gij, t + m_dt * m_c_butcher[i]);
            }

            // Solution-dependent term
            if (m_L_isTimedependent || (i == 0 && step == 0)) {
                GetHypreSpatialDiscretizationL(L, Lij, t + m_dt * m_c_butcher[i]);
            } 

            HYPRE_ParVectorCopy(u, b); // b <- u
            for (int j = 0; j < i; j++) {
                double temp = m_dt * m_A_butcher[i][j];
                if (temp !=  0.0) HYPRE_ParVectorAxpy(temp, k[j], b); // b <- b + dt*aij*k[j]
            }

            // Set final value of stage if no mass matrix, otherwise this makes a good initial guess at solution
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, L, b, 1.0, g, k[i]); // k[i] <- -L*b + g 
            
            /* -------------------------------------------------------- */
            /* --- Invert mass matrix: Find k_i such that M*k_i=b_i --- */
            /* -------------------------------------------------------- */
            if (m_M_exists) {
            //if (!m_M_exists) { // TODO : Hack for testing when I don't have a mass matrix but want to invert whatever is provided by getMassMatrix()
                // Assemble mass matrix on first iteration
                if (step == 0 && i == 0) {
                    // Get rows this process owns of M assuming rows of M and L are partitioned the same in memory
                    int ilower, iupper, jdummy1, jdummy2;
                    HYPRE_IJMatrixGetLocalRange(Lij, &ilower, &iupper, &jdummy1, &jdummy2);
                    //setIdentityMassLocalRange(ilower, iupper); // TODO : Hack for testing when I don't have a mass matrix but want to invert whatever is provided by getMassMatrix()
                    // Store inverse of M if it's lumped since we only need mass matrix for purposes of inverting it
                    if (!m_solver_parameters.lump_mass) {
                        SetHypreMassMatrix(ilower, iupper);
                    } else {
                        SetHypreInvMassMatrix(ilower, iupper);
                    }
                }
                
                /* --- Solve --- */
                HYPRE_ParVectorCopy(k[i], b); // b <- k[i]; // RHS of linear system is the value currently stored in k[i]
                //HYPRE_ParVectorScale(10.0, k[i]); // TODO : Hack for testing; set initial guess to something dumb
                
                // Point member variables to local variables so mass solver can access them
                m_x = k[i]; // RHS of linear system is used as the initial guess at the solution
                m_b = b;
                // Solve the system!
                SolveMassSystem(); 
                // Reset memeber variables to NULL
                m_x = NULL;
                m_b = NULL;
                
                // Ensure desired tolerance was reached if using iterative solver, otherwise quit
                if (m_iterative) {
                    solve_count += 1;
                    avg_iters   += (double) m_num_iters;
                    // Ensure desired tolerance was reached in allowable number of iterations, otherwise quit
                    if (m_res_norm > m_solver_parameters.tol) {
                        if (m_spatialRank == 0) std::cout << "=================================\n =========== WARNING ===========\n=================================\n";
                        if (m_spatialRank == 0) std::cout << "Time step " << step+1 << "/" << m_nt-1 << ": Solving for stage " << i+1 << "/" << m_s_butcher << '\n';
                        if (m_spatialRank == 0) std::cout << "Tol after " << m_num_iters << " iters (max iterations) = " << m_res_norm << " > desired tol = " << m_solver_parameters.tol << "\n\n";
                        MPI_Finalize();
                        exit(1);
                    }
                }
            }
        }

        // Sum solution
        for (int i = 0; i < m_s_butcher; i++) {
            double temp = m_dt * m_b_butcher[i];
            if (temp != 0.0) HYPRE_ParVectorAxpy(temp, k[i], u); // u <- u + dt*k[i]*k[i]; 
        }
        t += m_dt; // Increment time
    }


    // Print statistics about average iteration counts and convergence factor across whole time interval
    if (m_M_exists && m_iterative) {
        if ((m_solver_parameters.printLevel > 0) && (m_spatialRank == 0)) {
            std::cout << "=============================================\n";
            std::cout << "Summary of linear solves during time stepping\n";
            std::cout << "---------------------------------------------\n";
            std::cout << "Number of systems solved = " << solve_count << '\n';
            std::cout << "Average number of iterations = " << avg_iters/solve_count << '\n';
            std::cout << "***TODO***: Average convergence factor = ..." << '\n';
            //hypre_BoomerAMGGetRelResidualNorm(m_solver, &rel_res_norm);
        }
    }


    /* ---------------------------------------------------------- */
    /* ------------------------ Clean up ------------------------ */
    /* ---------------------------------------------------------- */
    for (int i = 0; i < vectors.size(); i++) {
        HYPRE_IJVectorDestroy(vectorsij[i]);
    }
    
    if (gij) HYPRE_IJVectorDestroy(gij);
    if (Lij) HYPRE_IJMatrixDestroy(Lij);
    
    
    // Point member variable x to final solution
    m_x   = u;
    m_xij = uij;
}


/* Assemble the inverse of a diagonally lumped mass matrix as a HYPRE matrix

NOTE: 
    -We know the rows owned by the current process by previously assembling 
        the spatial discretization whose rows are distributed the same way
    -If the mass matrix has been lumped to a diagonal this function will 
        store its inverse 
*/
void SpaceTimeMatrix::SetHypreInvMassMatrix(int  ilower, 
                                            int  iupper)
{
    // Just check not already set!
    if (m_invM) {
        std::cout << "WARNING: The inverse of mass matrix has already been set! Cannot be reset" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    if (!m_solver_parameters.lump_mass) {
        std::cout << "WARNING: I can only compute the inverse of a diagaonally lumped mass matrix!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    int      onProcSize;
    int *    M_rowptr;
    int *    M_colinds;
    int *    M_rows;
    int *    M_cols_per_row;
    double * M_data;
    
    // Get mass matrix components and assemble as the HYPRE matrix M
    getMassMatrix(M_rowptr, M_colinds, M_data);
    
    onProcSize = iupper - ilower + 1; // Number of rows on process
    
    int nnzM = M_rowptr[onProcSize] - M_rowptr[0]; // nnz(M) on process
    for (int i = 0; i < nnzM; i++) {
        // TODO : Is this a silly check? Is it impossible that M could become singular after lumping?
        if (std::abs(M_data[i]) < 1e-14) {
            std::cout << "WARNING: Mass matrix appears to be singular after lumping!" << '\n';
            MPI_Finalize();
            exit(1);
        }
        M_data[i] = 1/M_data[i];
    }
    
    // Initialize matrix
    HYPRE_IJMatrixCreate(m_spatialComm, ilower, iupper, ilower, iupper, &m_invMij);
    HYPRE_IJMatrixSetObjectType(m_invMij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(m_invMij);
    
    // Set matrix coefficients
    M_rows         = new int[onProcSize];
    M_cols_per_row = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        M_rows[i] = ilower + i;
        M_cols_per_row[i] = M_rowptr[i+1] - M_rowptr[i];
    }
    HYPRE_IJMatrixSetValues(m_Mij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_data);
    
    // Finalize construction
    HYPRE_IJMatrixAssemble(m_invMij);
    HYPRE_IJMatrixGetObject(m_invMij, (void **) &m_invM);
    
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_rows;
    delete[] M_cols_per_row;
    delete[] M_data;
}


/* Assemble mass matrix as a HYPRE matrix

NOTE: 
    -We know the rows owned by the current process by previously assembling 
        the spatial discretization whose rows are distributed the same way
*/
void SpaceTimeMatrix::SetHypreMassMatrix(int  ilower, 
                                         int  iupper)
{
    // Just check not already set!
    if (m_M) {
        std::cout << "WARNING: The mass matrix has already been set! Cannot be reset" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    
    int      onProcSize;
    int *    M_rowptr;
    int *    M_colinds;
    int *    M_rows;
    int *    M_cols_per_row;
    double * M_data;
    
    // Get mass matrix components and assemble as the HYPRE matrix M
    getMassMatrix(M_rowptr, M_colinds, M_data);
    
    onProcSize = iupper - ilower + 1; // Number of rows on process
    
    // Initialize matrix
    HYPRE_IJMatrixCreate(m_spatialComm, ilower, iupper, ilower, iupper, &m_Mij);
    HYPRE_IJMatrixSetObjectType(m_Mij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(m_Mij);
    
    // Set matrix coefficients
    M_rows         = new int[onProcSize];
    M_cols_per_row = new int[onProcSize];
    for (int i = 0; i < onProcSize; i++) {
        M_rows[i] = ilower + i;
        M_cols_per_row[i] = M_rowptr[i+1] - M_rowptr[i];
    }
    HYPRE_IJMatrixSetValues(m_Mij, onProcSize, M_cols_per_row, M_rows, M_colinds, M_data);
    
    // Finalize construction
    HYPRE_IJMatrixAssemble(m_Mij);
    HYPRE_IJMatrixGetObject(m_Mij, (void **) &m_M);
    
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_rows;
    delete[] M_cols_per_row;
    delete[] M_data;
}





/* Get initial condition, u0, as a HYPRE vector */
void SpaceTimeMatrix::GetHypreInitialCondition(HYPRE_ParVector &u0, 
                                                HYPRE_IJVector &u0ij) 
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

    HYPRE_IJVectorCreate(m_spatialComm, ilower, iupper, &u0ij);
    HYPRE_IJVectorSetObjectType(u0ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(u0ij);
    HYPRE_IJVectorSetValues(u0ij, onProcSize, rows, U);
    HYPRE_IJVectorAssemble(u0ij);
    HYPRE_IJVectorGetObject(u0ij, (void **) &u0);

    delete[] rows;
    delete[] U;
}

/* Initialize all the HYPRE vectors in "vectors" to the same values as in u 

NOTE: Size of vectors must have been set, using the resize function, for example.
*/
void SpaceTimeMatrix::InitializeHypreVectors(HYPRE_ParVector                 &u, 
                                                HYPRE_IJVector               &uij, 
                                                std::vector<HYPRE_ParVector> &vectors, 
                                                std::vector<HYPRE_IJVector>  &vectorsij) 
{
    if (vectors.size() > 0) {    
        int      ilower;
        int      iupper;
        int      onProcSize;
        double * U;
        int    * rows;
        
        // Get range of rows owned by current process
        HYPRE_IJVectorGetLocalRange(uij, &ilower, &iupper);
        
        // Get rows owned by current process
        onProcSize = iupper - ilower + 1; // Number of rows on current process
        rows       = new int[onProcSize];
        for (int i = 0; i < onProcSize; i++) {
            rows[i] = ilower + i;
        }
        
        // Get entries owned by current process
        U = new double[onProcSize];
        HYPRE_IJVectorGetValues(uij, onProcSize, rows, U);
        
        // Create and initialize all vectors in vectorsij, setting their values to those of u
        for (int i = 0; i < vectors.size(); i++) {
            HYPRE_IJVectorCreate(m_spatialComm, ilower, iupper, &vectorsij[i]);
            HYPRE_IJVectorSetObjectType(vectorsij[i], HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(vectorsij[i]);
            HYPRE_IJVectorSetValues(vectorsij[i], onProcSize, rows, U);
            HYPRE_IJVectorAssemble(vectorsij[i]);
            HYPRE_IJVectorGetObject(vectorsij[i], (void **) &vectors[i]);
        }
        
        delete[] rows;
        delete[] U;
    }
}


/* Get solution-independent component of spatial discretization, the vector g, as a HYPRE vector */
void SpaceTimeMatrix::GetHypreSpatialDiscretizationG(HYPRE_ParVector   &g,
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
    HYPRE_IJVectorCreate(m_spatialComm, ilower, iupper, &gij);
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
void SpaceTimeMatrix::GetHypreSpatialDiscretizationL(HYPRE_ParCSRMatrix &L,
                                                        HYPRE_IJMatrix  &Lij,
                                                        double           t)  
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
    HYPRE_IJMatrixCreate(m_spatialComm, ilower, iupper, ilower, iupper, &Lij);
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


void SpaceTimeMatrix::BuildSpaceTimeMatrix()
{
    if (m_globRank == 0) std::cout << "Building space-time matrix\n";
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
    solinfo << "spatialParallel " << int(m_useSpatialParallel) << "\n";
    if (m_useSpatialParallel) solinfo << "p_xTotal " << m_spatialCommSize << "\n";
    
    // Time-discretization-specific information
    solinfo << "timeDisc " << m_timeDisc << "\n";
    solinfo << "implicit " << m_implicit << "\n";
    if (m_RK) {
        solinfo << "s_rk " << m_s_butcher << "\n";
        solinfo << "time " << "RK\n";
    } else if (m_AB) {
        solinfo << "time " << "AB\n";
        solinfo << "s_multi " << m_s_multi << "\n";
    } else if (m_AM) {
        solinfo << "time " << "AM\n";
        solinfo << "s_multi " << m_s_multi << "\n";
    } else if (m_BDF) {
        solinfo << "time " << "BDF\n";
        solinfo << "s_multi " << m_s_multi << "\n";
    }
    
    

    
    // Print out contents from additionalInfo to file too
    std::map<std::string, std::string>::iterator it;
    for (it=additionalInfo.begin(); it!=additionalInfo.end(); it++) {
        solinfo << it->first << " " << it->second << "\n";
    }

    solinfo.close();
}


/* 
timeDisc is a 3 digit integer

timeDisc with "1" as 1st digit are ERK, timeDisc with "2" as 1st digit are DIRK
2nd digit == number of stages
3rd digit == order of method
*/
void SpaceTimeMatrix::GetButcherTableaux() {

    m_s_butcher = m_timeDisc / 10 % 10; //  Extract number of stages; assumes number has 3 digits    
    // Resize Butcher arrays
    m_A_butcher.resize(m_s_butcher, std::vector<double>(m_s_butcher, 0.0));
    m_b_butcher.resize(m_s_butcher);
    m_c_butcher.resize(m_s_butcher);

    /* --- ERK tables --- */
    // Forward Euler: 1st-order
    if (m_timeDisc == 111) {
        m_implicit        = false;
        m_ERK             = true;
        m_s_butcher       = 1;
        m_A_butcher[0][0] = 0.0;
        m_b_butcher[0]    = 1.0; 
        m_c_butcher[0]    = 0.0; 
    
    // 2nd-order Heun's method    
    } else if (m_timeDisc == 122) {
        m_implicit        = false;
        m_ERK             = true;
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
        m_implicit        = true;
        m_ERK             = true;
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
        m_implicit        = false;
        m_ERK             = true;
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
        m_implicit        = true;
        m_DIRK            = true;
        m_SDIRK           = true;
        m_s_butcher       = 1;
        m_A_butcher[0][0] = 1.0;
        m_b_butcher[0]    = 1.0; 
        m_c_butcher[0]    = 1.0; 
    
    // 2nd-order L-stable SDIRK (there are a few different possibilities here. This is from the Dobrev et al.)
    } else if (m_timeDisc == 222) {
        double sqrt2      = 1.414213562373095;
        m_implicit        = true;
        m_DIRK            = true;
        m_SDIRK           = true;
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
        m_implicit        =  true;
        m_DIRK            =  true;
        m_SDIRK           =  true;
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
        m_implicit        =  true;
        m_DIRK            =  true;
        m_SDIRK           =  true;
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


void SpaceTimeMatrix::SetABTableaux()
{
    m_implicit = false;
    m_s_multi  = m_timeDisc % 10; // Extract 2nd digit of two-digit integer
    m_b_multi.resize(m_s_multi); // No need to store the s+1th 0 coefficient
    
    if (m_timeDisc == 11) {         // 1st-order
        m_b_multi[0] = +1.0;
    } else if (m_timeDisc == 12) {  // 2nd-order
        m_b_multi[0] = -1.0/2.0;
        m_b_multi[1] = +3.0/2.0;
    } else if (m_timeDisc == 13) {  // 3rd-order
        m_b_multi[0] = +5.0/12.0;
        m_b_multi[1] = -16.0/12.0;
        m_b_multi[2] = +23.0/12.0;
    } else if (m_timeDisc == 14) {  // 4th-order
        m_b_multi[0] = -9.0/24.0;
        m_b_multi[1] = +37.0/24.0;
        m_b_multi[2] = -59.0/24.0;
        m_b_multi[3] = +55.0/24.0;
    } else {
        std::cout << "WARNING: Adams--Bashforth scheme '" << m_timeDisc << "'not recognised!\n";
        MPI_Finalize();
        exit(1);
    }
}

void SpaceTimeMatrix::SetAMTableaux()
{
    m_implicit = true;
    m_s_multi  = m_timeDisc % 10; // Extract 2nd digit of two-digit integer
    m_b_multi.resize(m_s_multi + 1); 
    
    if (m_timeDisc == 21) {  // 2nd-order
        m_b_multi[0] = +1.0/2.0;
        m_b_multi[1] = +1.0/2.0;
    } else if (m_timeDisc == 22) {  // 3rd-order
        m_b_multi[0] = -1.0/12.0;
        m_b_multi[1] = +8.0/12.0;
        m_b_multi[2] = +5.0/12.0;
    } else if (m_timeDisc == 23) {  // 4th-order
        m_b_multi[0] = +1.0/24.0;
        m_b_multi[1] = -5.0/24.0;
        m_b_multi[2] = +19.0/24.0;
        m_b_multi[3] = +9.0/24.0;
    } else {
        std::cout << "WARNING: Adams--Moulton scheme '" << m_timeDisc << "'not recognised!\n";
        MPI_Finalize();
        exit(1);
    }
}

void SpaceTimeMatrix::SetBDFTableaux()
{
    m_implicit = true;
    m_s_multi  = m_timeDisc % 10; // Extract 2nd digit of two-digit integer
    m_a_multi.resize(m_s_multi); // No need to store the s+1th coefficient that's 1
    m_b_multi.resize(1); // There is a single non-zero b coefficient
    
    if (m_timeDisc == 31) {         // 1st-order
        m_a_multi[0] = -1.0;
        
        m_b_multi[0] = +1.0;
    } else if (m_timeDisc == 32) {  // 2nd-order
        m_a_multi[0] = +1.0/3.0;
        m_a_multi[1] = -4.0/3.0;
        
        m_b_multi[0] = +2.0/3.0;
    } else if (m_timeDisc == 33) {  // 3rd-order
        m_a_multi[0] = -2.0/11.0;
        m_a_multi[1] = +9.0/11.0;
        m_a_multi[2] = -18.0/11.0;
        
        m_b_multi[0] = +6.0/11.0;
    } else if (m_timeDisc == 34) {  // 4th-order
        m_a_multi[0] = +3.0/25.0;
        m_a_multi[1] = -16.0/25.0;
        m_a_multi[2] = +36.0/25.0;
        m_a_multi[3] = -48.0/25.0;
        
        m_b_multi[0] = +12.0/25.0;
    } else {
        std::cout << "WARNING: BDF scheme '" << m_timeDisc << "'not recognised!\n";
        MPI_Finalize();
        exit(1);
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
    if (m_RK) {
        RKSpaceTimeBlock(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    } else if (m_BDF) {
        BDFSpaceTimeBlock(rowptr, colinds, data, B, X, localMinRow, localMaxRow, spatialDOFs);
    }
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
    
    if (m_RK) {
        RKSpaceTimeBlock(rowptr, colinds, data, B, X, onProcSize);
    } else if (m_BDF) {
        BDFSpaceTimeBlock(rowptr, colinds, data, B, X, onProcSize);
    }    
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



/* Provide options for solver */
void SpaceTimeMatrix::SetSolverParameters(Solver_parameters &solver_params) {
    m_solver_parameters = solver_params;
}


/* Set default options for solver */
void SpaceTimeMatrix::SetSolverParametersDefaults() {
    m_solver_parameters.tol          = 1e-8;
    m_solver_parameters.maxiter      = 250;
    m_solver_parameters.printLevel   = 3;
    
    m_solver_parameters.use_gmres    = 1;
    m_solver_parameters.gmres_preconditioner = 1;
    m_solver_parameters.AMGiters     = 10;
    m_solver_parameters.precon_printLevel = 1;
    
    m_solver_parameters.rebuildRate  = 0;
    
    m_solver_parameters.binv_scale   = false;
    m_solver_parameters.lump_mass    = true;
}


/* Set classical AMG parameters for BoomerAMG solve. */
void SpaceTimeMatrix::SetAMG()
{
    m_AMG_parameters.filter_tolR = 0.0;
    m_AMG_parameters.prerelax = "AA";
    m_AMG_parameters.postrelax = "AA";
    m_AMG_parameters.relax_type = 3;
    m_AMG_parameters.interp_type = 6;
    m_AMG_parameters.strength_tolC = 0.1;
    m_AMG_parameters.coarsen_type = 6;
    m_AMG_parameters.distance_R = -1;
    m_AMG_parameters.strength_tolR = -1;
    m_AMG_parameters.filter_tolA = 0.0;
    m_AMG_parameters.cycle_type = 1;
    m_rebuildSolver = true;
}


/* Set standard AIR parameters for BoomerAMG solve. */
void SpaceTimeMatrix::SetAIR()
{
    m_AMG_parameters.distance_R = 1.5;
    m_AMG_parameters.prerelax = "A";
    m_AMG_parameters.postrelax = "FFC";
    m_AMG_parameters.relax_type = 3;
    m_AMG_parameters.interp_type = 100;
    m_AMG_parameters.strength_tolC = 0.005;
    m_AMG_parameters.coarsen_type = 6;
    m_AMG_parameters.strength_tolR = 0.005;
    m_AMG_parameters.filter_tolR = 0.0;
    m_AMG_parameters.filter_tolA = 0.0;
    m_AMG_parameters.cycle_type = 1;
    m_rebuildSolver = true;
}


/* Set AIR parameters assuming triangular matrix in BoomerAMG solve. */
void SpaceTimeMatrix::SetAIRHyperbolic()
{
    m_AMG_parameters.distance_R = 1.5;
    m_AMG_parameters.prerelax = "A";
    m_AMG_parameters.postrelax = "F";
    m_AMG_parameters.relax_type = 10;
    m_AMG_parameters.interp_type = 100;
    m_AMG_parameters.strength_tolC = 0.005;
    m_AMG_parameters.coarsen_type = 6;
    m_AMG_parameters.strength_tolR = 0.005;
    m_AMG_parameters.filter_tolR = 0.0;
    m_AMG_parameters.filter_tolA = 0.0001;
    m_AMG_parameters.cycle_type = 1;
    m_rebuildSolver = true;
}


/* Provide BoomerAMG parameters struct for solve. */
void SpaceTimeMatrix::SetAMGParameters(AMG_parameters &AMG_params)
{
    // TODO: does this copy the structure by value?
    m_AMG_parameters = AMG_params;
}





void SpaceTimeMatrix::PrintMeshData()
{
    if (m_globRank == 0) {
        std::cout << "Space-time mesh:\n\thmin = " << m_hmin <<
        "\n\thmax = " << m_hmax << "\n\tdt   = " << m_dt << "\n\n";
    }
}

/* Initialize AMG solver based on parameters in m_AMG_parameters struct. 

NOTE: Some parameters are passed here rather than set through m_solver_parameters
since they differ depending on whether BoomerAMG is used as the solver of preconditioned
*/
void SpaceTimeMatrix::SetBoomerAMGOptions(int printLevel, int maxiter, double tol)
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
        
        // Array to store relaxation scheme and pass to Hypre
        //      TODO: does hypre clean up grid_relax_points
        int ns_down = m_AMG_parameters.prerelax.length();
        int ns_up = m_AMG_parameters.postrelax.length();
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
            if (m_AMG_parameters.prerelax.compare(i,1,Fr) == 0) {
                grid_relax_points[1][i] = -1;
            }
            else if (m_AMG_parameters.prerelax.compare(i,1,Cr) == 0) {
                grid_relax_points[1][i] = 1;
            }
            else if (m_AMG_parameters.prerelax.compare(i,1,Ar) == 0) {
                grid_relax_points[1][i] = 0;
            }
        }

        // set up relax scheme 
        for(unsigned int i = 0; i<ns_up; i++) {
            if (m_AMG_parameters.postrelax.compare(i,1,Fr) == 0) {
                grid_relax_points[2][i] = -1;
            }
            else if (m_AMG_parameters.postrelax.compare(i,1,Cr) == 0) {
                grid_relax_points[2][i] = 1;
            }
            else if (m_AMG_parameters.postrelax.compare(i,1,Ar) == 0) {
                grid_relax_points[2][i] = 0;
            }
        }

        // Create preconditioner
        HYPRE_BoomerAMGCreate(&m_solver);
        HYPRE_BoomerAMGSetTol(m_solver, tol);    
        HYPRE_BoomerAMGSetMaxIter(m_solver, maxiter);
        HYPRE_BoomerAMGSetPrintLevel(m_solver, printLevel);

        if (m_AMG_parameters.distance_R > 0) {
            HYPRE_BoomerAMGSetRestriction(m_solver, m_AMG_parameters.distance_R);
            HYPRE_BoomerAMGSetStrongThresholdR(m_solver, m_AMG_parameters.strength_tolR);
            HYPRE_BoomerAMGSetFilterThresholdR(m_solver, m_AMG_parameters.filter_tolR);
        }
        HYPRE_BoomerAMGSetInterpType(m_solver, m_AMG_parameters.interp_type);
        HYPRE_BoomerAMGSetCoarsenType(m_solver, m_AMG_parameters.coarsen_type);
        HYPRE_BoomerAMGSetAggNumLevels(m_solver, 0);
        HYPRE_BoomerAMGSetStrongThreshold(m_solver, m_AMG_parameters.strength_tolC);
        HYPRE_BoomerAMGSetGridRelaxPoints(m_solver, grid_relax_points);
        if (m_AMG_parameters.relax_type > -1) {
            HYPRE_BoomerAMGSetRelaxType(m_solver, m_AMG_parameters.relax_type);
        }
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_coarse, 3);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_down,   1);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_up,     2);
        if (m_AMG_parameters.filter_tolA > 0) {
            HYPRE_BoomerAMGSetADropTol(m_solver, m_AMG_parameters.filter_tolA);
        }
        // type = -1: drop based on row inf-norm
        else if (m_AMG_parameters.filter_tolA == -1) {
            HYPRE_BoomerAMGSetADropType(m_solver, -1);
        }

        // Set cycle type for solve 
        HYPRE_BoomerAMGSetCycleType(m_solver, m_AMG_parameters.cycle_type);
    }
}



/* Solve current linear system with AMG

NOTE: If applicable, an existing AMG solver, i.e., one based on a previously 
constructed matrix A, is used to solve the current linear system if the underlying 
AMG solver is not explicitly told to be rebuilt via the m_rebuildSolver flag */
void SpaceTimeMatrix::SolveAMG()
{
    if (!m_solver) m_rebuildSolver = true; // Ensure that if solver not build previously then it is built now
    
    // TODO : What does this code mean? Does it scale the whole matrix by inverse of block diagonal mass matrix? 
    // If so, when do we want to do this? Oh, maybe if solving the space-time problem with explicit time stepping to make the 
    // Space-time matrix lower triangular...
    if (m_solver_parameters.binv_scale) {
        HYPRE_ParCSRMatrix A_s;
        hypre_ParcsrBdiagInvScal(m_A, m_bsize, &A_s);
        hypre_ParCSRMatrixDropSmallEntries(A_s, 1e-15, 1);
        HYPRE_ParVector b_s;
        hypre_ParvecBdiagInvScal(m_b, m_bsize, &b_s, m_A);
    
    
        // TODO : wrap setup timer around this block
        // If necessary, construct AMG solver based on current value of  A
        if (m_rebuildSolver) {
            // Set or reset options for AMG solver
            SetBoomerAMGOptions(m_solver_parameters.printLevel, m_solver_parameters.maxiter, m_solver_parameters.tol);
            // Build AMG hierarchy based on current value of A_s
            HYPRE_BoomerAMGSetup(m_solver, A_s, b_s, m_x); // NOTE: Values of b and x are ignored by this function!
            if (m_globRank == 0) std::cout << "Solver assembled.\n";
            m_rebuildSolver = false; // Don't rebuild solver again unless explicitly told to
        }
        
        // TODO : wrap solve timer around this block
        // Solve linear system based on current values of A,b,x 
        m_hypre_ierr = HYPRE_BoomerAMGSolve(m_solver, A_s, b_s, m_x);
        
        // TODO : What happens to A_s and b_s here? Don't they need to be free'd? Or are they just copies?
    }
    else 
    {
        // TODO : wrap setup timer around block
        // If necessary, construct AMG solver based on current value of  A
        if (m_rebuildSolver) {
            // Set or reset options for AMG solver
            SetBoomerAMGOptions(m_solver_parameters.printLevel, m_solver_parameters.maxiter, m_solver_parameters.tol);
            // Build AMG hierarchy based on current value of A
            HYPRE_BoomerAMGSetup(m_solver, m_A, m_b, m_x); // NOTE: Values of b and x are ignored by this function!
            if (m_globRank == 0) std::cout << "Solver assembled.\n";
            m_rebuildSolver = false; // Don't rebuild solver again unless explicitly told to
        }
        
        // TODO : wrap solve timer around this block
        // Solve linear system based on current values of A,b,x 
        m_hypre_ierr = HYPRE_BoomerAMGSolve(m_solver, m_A, m_b, m_x);
    }
    
    // Get convergence statistics
    HYPRE_BoomerAMGGetNumIterations(m_solver, &m_num_iters);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(m_solver, &m_res_norm);
}



/* Initialize GMRES solver based on parameters in m_solver_parameters struct. */
void SpaceTimeMatrix::SetGMRESOptions() {
    // If GMRES solver exists and underlying preconditioner not being rebuilt then return
    if (m_gmres && !m_rebuildSolver){
        return;
    
    // Initialize or reinitalize GMRES solver if it already existed
    } else {
        if (m_gmres) HYPRE_ParCSRGMRESDestroy(m_gmres);
    
        // Create solver object
        HYPRE_ParCSRGMRESCreate(m_solverComm, &m_gmres);
    
        // AMG preconditioning 
        if (m_solver_parameters.gmres_preconditioner == 1) {
            // Setup boomerAMG with zero halting tolerance so we can do a fixed number of AMG iterations
            SetBoomerAMGOptions(m_solver_parameters.precon_printLevel, m_solver_parameters.AMGiters, 0.0);
            HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, m_solver);
        }
        // Diagonally scaled preconditioning?
        else if (m_solver_parameters.gmres_preconditioner == 2) {
            // TODO : Ben, does this make sense? m_solver has never been set before it is used below, i.e., it's currently NULL???
            HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_ParCSROnProcTriSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_ParCSROnProcTriSetup, m_solver);  
        }
    
        HYPRE_GMRESSetKDim(m_gmres, 50);
        HYPRE_GMRESSetMaxIter(m_gmres, m_solver_parameters.maxiter);
        HYPRE_GMRESSetTol(m_gmres, m_solver_parameters.tol);
        HYPRE_GMRESSetPrintLevel(m_gmres, m_solver_parameters.printLevel);
        HYPRE_GMRESSetLogging(m_gmres, 1);
    }
}


/* Solve current linear system with preconditioned GMRES

NOTE: If applicable, an existing GMRES solver, i.e., one based on a previously 
constructed AMG preconditioner (in turn based on a previously constructed matrix A), 
is used to solve the current linear system if the underlying AMG preconditioner 
is not explicitly told to be rebuilt via the m_rebuildSolver flag */
void SpaceTimeMatrix::SolveGMRES() 
{
    if (!m_gmres) m_rebuildSolver = true; // Ensure that if solver not build previously then it is built now

    if (m_solver_parameters.binv_scale) {
        HYPRE_ParCSRMatrix A_s;
        hypre_ParcsrBdiagInvScal(m_A, m_bsize, &A_s);
        hypre_ParCSRMatrixDropSmallEntries(A_s, 1e-15, 1);
        HYPRE_ParVector b_s;
        hypre_ParvecBdiagInvScal(m_b, m_bsize, &b_s, m_A);
        
        // TODO : wrap setup timer around block
        // If necessary, build GMRES solver based on current value of A_s
        if (m_rebuildSolver) {
            // Set or reset options for GMRES solver
            SetGMRESOptions();
            // Build GMRES solver based on current value of A
            HYPRE_ParCSRGMRESSetup(m_gmres, A_s, b_s, m_x); // NOTE: Values of b and x are ignored by this function!
            if (m_globRank == 0) std::cout << "Solver assembled.\n";
            m_rebuildSolver = false; // Don't rebuild solver again unless explicitly told to
        }
        
        // TODO : wrap solve timer around this block
        // Solve linear system based on current values of A,b,x
        m_hypre_ierr = HYPRE_ParCSRGMRESSolve(m_gmres, A_s, b_s, m_x);
        
        // TODO : What happens to A_s and b_s here? I think they likely need to be free'd? 
    }
    else 
    {
        
        // TODO : wrap setup timer around block
        // If necessary, build GMRES solver based on current value of A
        if (m_rebuildSolver) {
            // Set or reset options for GMRES solver
            SetGMRESOptions();
            // Build GMRES solver based on current value of A
            HYPRE_ParCSRGMRESSetup(m_gmres, m_A, m_b, m_x); // NOTE: Values of b and x are ignored by this function!
            if (m_globRank == 0) std::cout << "Solver assembled.\n";
            m_rebuildSolver = false; // Don't rebuild solver again unless explicitly told to
        }
        
        // TODO : wrap solve timer around this block
        // Solve linear system based on current values of A,b,x
        m_hypre_ierr = HYPRE_ParCSRGMRESSolve(m_gmres, m_A, m_b, m_x);
    }
    
    // Get convergence statistics
    HYPRE_GMRESGetNumIterations(m_gmres, &m_num_iters);
    HYPRE_GMRESGetFinalRelativeResidualNorm(m_gmres, &m_res_norm);
}



/* Initialize (unpreconditioned) PCG based on parameters in m_solver_parameters struct. 

NOTE:
    -No attention paid to m_rebuildSolver here since the mass matrix will not change 
        over the life-time of the solve
*/
void SpaceTimeMatrix::SetPCGOptions() {
    // Create solver object
    
    if (m_pcg) return;
    
    HYPRE_ParCSRPCGCreate(m_solverComm, &m_pcg);

    HYPRE_PCGSetTwoNorm(m_pcg, 1); // Base convergence on two-norm (I guess otherwise this is A-norm? Hard to find in docs...)
    HYPRE_PCGSetMaxIter(m_pcg, m_solver_parameters.maxiter);
    HYPRE_PCGSetTol(m_pcg, m_solver_parameters.tol);
    HYPRE_PCGSetPrintLevel(m_pcg, m_solver_parameters.printLevel);
    HYPRE_PCGSetLogging(m_pcg, 1);
}


/* Solve the linear system M*m_x = m_b where M is the mass matrix 

Options:
    1. M not lumped, and don't scale by block inverse: Solve iteratively with CG
    2. M lumped to be diagonal: Multiply by its inverse
    3. Option set to scale by block inverse of M: Multiply by its inverse
    
NOTE:
    -No attention paid to m_rebuildSolver here since the mass matrix will not change 
        over the life-time of the solve
*/
void SpaceTimeMatrix::SolveMassSystem() 
{
    // Use unpreconditioned CG if not exactly inverting linear system
    if (!m_solver_parameters.lump_mass && !m_solver_parameters.binv_scale) {
        m_iterative = true;
        
        // Setup solver if hasn't been done previously
        if (!m_pcg) {
            if (m_spatialRank == 0) std::cout << "Building solver" << '\n';
            SetPCGOptions();
            HYPRE_ParCSRPCGSetup(m_pcg, m_M, m_b, m_x); // NOTE: Values of b and x are ignored by this function!
            if (m_spatialRank == 0) std::cout << "Solver assembled" << '\n';
        }
        
        // Solve linear system
        m_hypre_ierr = HYPRE_ParCSRPCGSolve(m_pcg, m_M, m_b, m_x);
        
        // Get convergence statistics
        HYPRE_PCGGetNumIterations(m_pcg, &m_num_iters);
        HYPRE_PCGGetFinalRelativeResidualNorm(m_pcg, &m_res_norm);
        
    // Mass matrix is lumped to be diagonal: Directly multiply by its inverse!
    // Note: The inverse of the mass matrix must already be stored here!
    } else if (m_solver_parameters.lump_mass) {
        if (!m_invMij) {
            std::cout << "WARNING: Diagonally lumped mass matrix is inverted directly, but m_invM doesn't exist! You must create this first" << '\n';
            MPI_Finalize();
            exit(1);
        }
        
        m_iterative = false;
        hypre_ParCSRMatrixMatvec(1.0, m_invM, m_b, 0.0, m_x); // x <- 1.0*inv(M)*b + 0.0*x
        
    // Mass matrix is block diagonal and we directly invert it
    } else {
        m_iterative = false;
        
        // Ensure that block size has been set to something meaningful
        if (m_bsize <= 1)  {
            if (m_spatialRank == 0) std::cout << "Block diagonal mass matrix must have block size > 1 to invert by scaling by block inverse!" << '\n';
            MPI_Finalize();
            exit(1);
        }
        
        // Scale RHS by mass inverse
        hypre_ParvecBdiagInvScal(m_b, m_bsize, &m_x, m_M); // x <- inv(A) * b
    }
}


/* ----------------------------------------------------------- */
/* ----------------- No spatial parallelism ------------------ */
/* ----------------------------------------------------------- */
/* NOTES:
    -Does not make any assumption about overlap in the sparsity pattern of the spatial 
        discretization and mass matrix when estimating matrix nnz, 
        assumes nnz of spatial discretization does not depend on time
 */



/* Gets block row (or multiple if there are > 1 DOFs per process) of s-step BDF 
    space-time equations. No spatial parallelism.

NOTEs: 
    -m_t0 is assumed to be 0, and so the spatial discretization is evaluated at
        time t0 + (n+s)*dt == (n+s)*dt.
*/
void SpaceTimeMatrix::BDFSpaceTimeBlock(int    * &rowptr, 
                                        int    * &colinds, 
                                        double * &data, 
                                        double * &B, 
                                        double * &V, 
                                        int      &onProcSize)
{
    int globalInd0 = m_globRank * m_nDOFPerProc;    // Index of first DOF on process
    int globalInd1 = globalInd0 + m_nDOFPerProc - 1;// Index of last DOF on process
    
    /* --- Get spatial discretization at time required by 1st DOF on process --- */
    int      spatialDOFs;    
    int      L_nnz;
    int    * L_rowptr;
    int    * L_colinds;
    double * L_data;
    double * B0;
    double * V0;
    getSpatialDiscretization(L_rowptr, L_colinds, L_data, B0, V0, spatialDOFs, m_t0 + (globalInd0+m_s_multi)*m_dt, m_bsize);
    L_nnz = L_rowptr[spatialDOFs];   
    
    /* --- Get mass matrix ---*/
    int      M_nnz;
    int    * M_rowptr;
    int    * M_colinds;
    double * M_data;
    // Set range of identity mass matix to assemble if needed (all fits on process)
    if (!m_M_exists) setIdentityMassLocalRange(0, spatialDOFs-1); 
    getMassMatrix(M_rowptr, M_colinds, M_data);
    M_nnz = M_rowptr[spatialDOFs];

    /* --- Get total NNZ estimate on process --- */
    //  -Assume NNZ of L does not change with time
    //  -Don't assume sparsity of M and L overlap: Estimate will be an upperbound.
    int onProcNnz = 0;
    for (int globalInd = globalInd0; globalInd <= globalInd1; globalInd++) {
        // Coupling to myself
        onProcNnz += M_nnz;
        onProcNnz += L_nnz;
        
        // Coupling to DOFs at s previous times
        if (globalInd > m_s_multi - 1) {
            onProcNnz += m_s_multi * M_nnz;
        } else { // Or n times if n < s
            onProcNnz += globalInd * M_nnz;
        }
    }
    
    onProcSize = m_nDOFPerProc * spatialDOFs; // Number of rows on process
    rowptr     = new int[onProcSize + 1];
    rowptr[0]  = 0; 
    colinds    = new int[onProcNnz];
    data       = new double[onProcNnz];
    B          = new double[onProcSize];
    V          = new double[onProcSize];
    
    int dataInd      = 0;
    int colOffset    = 0;  // Global index of first column for DOF that we're coupling to
    int rowptrOffset = 0;  // Offset for accessing rowptr array for each DOF on proc
    double temp      = 0.0;
    
    
    /* ----------------------------------------------------- */
    /* ------ Build block row for all DOFs on process ------ */
    /* ----------------------------------------------------- */
    for (int globalInd = globalInd0; globalInd <= globalInd1; globalInd++) {
        
        // Rebuild spatial discretization if it's time dependent
        if (globalInd > globalInd0 && m_isTimeDependent) {
            delete[] L_rowptr;
            delete[] L_colinds;
            delete[] L_data;
            delete[] V0;
            delete[] B0;
            getSpatialDiscretization(L_rowptr, L_colinds, L_data, B0, V0, spatialDOFs, m_t0 + (globalInd+m_s_multi)*m_dt, m_bsize);
        }
    
        std::map<int, double>::iterator it;
    
        // Loop over each row in spatial discretization, working from earliest DOFs to the current one
        for (int row = 0; row < spatialDOFs; row++) {
            B[rowptrOffset + row] = m_dt*m_b_multi[0]*B0[row]; // PDE source term. NOTE: Only b_s is stored for BDF schemes 
            V[rowptrOffset + row] = V0[row];                   // Initial guess at solution
        
            // Number of previous DOFs current DOF couples to
            int s_effective = m_s_multi;
            
            // First s DOFs only couple to the n times before them rather than all s
            if (globalInd <= m_s_multi - 1) {
                s_effective = globalInd;
                // Add precomputed w[n] vector holding all necessary starting-value information
                B[rowptrOffset + row] += m_w_multi[globalInd][row];
            }
            
            // Global index of furthest DOF current DOF couples back to
            colOffset = (globalInd - s_effective) * spatialDOFs;
            
            /* ------ Off-block-diagonal component that reaches back s_effective DOFs with mass matrix couplings ------ */
            for (int j = s_effective; j >= 1; j--)  {
                // Get mass-matrix data
                for (int p = M_rowptr[row]; p < M_rowptr[row+1]; p++) {                
                    data[dataInd]    = m_a_multi[m_s_multi-j]*M_data[p];
                    colinds[dataInd] = M_colinds[p] + colOffset;
                    dataInd         += 1;
                }
                colOffset += spatialDOFs;
            }
        
            /* ------ Block-diagonal component ------ */
            std::map<int, double> entries; 
            // Get mass-matrix data
            for (int j=M_rowptr[row]; j<M_rowptr[row+1]; j++) {                
                if (std::abs(M_data[j]) > 1e-16) {
                    entries[M_colinds[j]] = M_data[j];
                }
            }

            // Add spatial discretization data to mass-matrix data
            temp = m_dt * m_b_multi[0]; // Note only b_s is stored for BDF schemes
            for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                if (std::abs(L_data[j]) > 1e-16) {
                    entries[L_colinds[j]] += temp * L_data[j];
                }
            }

            // Add this data to global matrix
            for (it=entries.begin(); it!=entries.end(); it++) {
                colinds[dataInd] = it->first + colOffset;
                data[dataInd]    = it->second;
                dataInd         += 1;
            }

            // Move to next row of spatial discretization for current DOF
            rowptr[rowptrOffset + row+1] = dataInd;
        }
        
        // Finished assembling component for current DOF, move to next DOF on process
        rowptrOffset += spatialDOFs;
    }
    
    // Check that sufficient data was allocated
    if (dataInd > onProcNnz) {
        std::cout << "WARNING: Space-time BDF matrix has more nonzeros than allocated on process " << m_globRank << " of " << m_numProc << ".\n";
    }
    
    // Clean up.
    delete[] L_rowptr;
    delete[] L_colinds;
    delete[] L_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] V0;
    
    // Information for initializing space-time RHS vector no longer needed.
    for (int i = 0; i < m_w_multi.size(); i++) {
        delete[] m_w_multi[i];
        m_w_multi[i] = NULL;
    }
}                                    


/* Get block row (or multiple if there are > 1 DOFs per process) of s-stage RK 
    space-time equations. 
    
    DOFS are blocked in groups of s: There are s-1 stages followed by the 
        solution at the new time.
    
    No spatial parallelism.
*/
void SpaceTimeMatrix::RKSpaceTimeBlock(int    * &rowptr, 
                                       int    * &colinds, 
                                       double * &data,
                                       double * &B, 
                                       double * &V,     
                                       int      &onProcSize)
{
    int globalInd0 = m_globRank * m_nDOFPerProc;        // Global index of first variable on process
    int globalInd1 = globalInd0 + m_nDOFPerProc - 1;    // Global index of last variable on process

    int * localInd = new int[m_nDOFPerProc];            // Local index of each DOF on process
    int * blockInd = new int[m_nDOFPerProc];            // Block index of each DOF on process
    for (int globalInd = globalInd0; globalInd <= globalInd1; globalInd++) {
        localInd[globalInd - globalInd0] = globalInd % m_s_butcher;
        blockInd[globalInd - globalInd0] = globalInd / m_s_butcher; 
    }
    
    /* --- Get spatial discretization at time required by 1st DOF on process --- */
    int      L_nnz;
    int    * L_rowptr;
    int    * L_colinds;
    double * L_data;
    double * B0;
    double * V0;
    int      spatialDOFs;
    double t = 0.0; // Time to evaluate spatial discretization, as required by first DOF on process. 
    if (localInd[0] == 0) { // Solution-type DOF that's not the initial condition
        if (blockInd[0] != 0) t = m_dt * (blockInd[0]-1) + m_dt * m_c_butcher[m_s_butcher-1]; 
    } else { // Stage-type DOF
        t = m_dt * blockInd[0] + m_dt * m_c_butcher[localInd[0]-1]; 
    }
    getSpatialDiscretization(L_rowptr, L_colinds, L_data, B0, V0, spatialDOFs, t, m_bsize);
    L_nnz = L_rowptr[spatialDOFs];   
    
    /* --- Get mass matrix ---*/
    int      M_nnz;
    int    * M_rowptr;
    int    * M_colinds;
    double * M_data;
    
    // Setup range of identity matrix to assemble if spatial discretization doesn't use a mass matrix
    if (!m_M_exists) setIdentityMassLocalRange(0, spatialDOFs-1); // Entire spatial discretization fits on process
    getMassMatrix(M_rowptr, M_colinds, M_data);
    M_nnz = M_rowptr[spatialDOFs];

    /* ------ Get total NNZ on this processor. ------ */
    //  -Assumes NNZ of spatial discretization does not change with time
    //  -Doesn't assume sparsity of M and L overlap: This nnz count is an upperbound.
    int procNnz = 0;
    for (int i = 0; i <= globalInd1 - globalInd0; i++) {
        // Solution-type variable
        if (localInd[i] == 0) {
            // Initial condition just has identity coupling
            if (blockInd[i] == 0) {
                procNnz += spatialDOFs;
            // All solution DOFs at t > 0
            } else {
                // Coupling to itself
                procNnz += M_nnz;
                if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] != 0.0) procNnz += L_nnz;
    
                // Coupling to solution DOF at previous time 
                procNnz += M_nnz;
                if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] != 0.0) procNnz += L_nnz; 
    
                // Coupling to stage variables at previous time 
                for (int j = 0; j < m_s_butcher-1; j++) {
                    if (m_b_butcher[j] != 0.0) procNnz += M_nnz;
                    if (m_b_butcher[j]*m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1]*m_A_butcher[m_s_butcher-1][j] != 0.0) procNnz += L_nnz;
                }
            }
    
        // Stage-type variable
        } else {
            int stageInd = localInd[i] - 1; // Work with this to index Butcher arrays directly.
            // Coupling to itself
            procNnz += M_nnz;
            if (m_A_butcher[stageInd][stageInd] != 0.0) procNnz += L_nnz;
    
            // Coupling to solution at previous time
            procNnz += L_nnz;
    
            // Coupling to previous stage variables at current time
            for (int j = 0; j < stageInd; j++) {
                if (m_A_butcher[stageInd][j] != 0.0) procNnz += L_nnz;
            }
        }
    }
    
    onProcSize = m_nDOFPerProc * spatialDOFs; // Number of rows I own
    rowptr     = new int[onProcSize + 1];
    colinds    = new int[procNnz];
    data       = new double[procNnz];
    B          = new double[onProcSize];
    V          = new double[onProcSize];
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
            delete[] L_rowptr;
            delete[] L_colinds;
            delete[] L_data;
            delete[] B0;
            delete[] V0;
            // Time to evaluate spatial discretization at
            // Solution-type DOF
            if (localInd[globalInd-globalInd0] == 0) { 
                t = m_dt * (blockInd[globalInd-globalInd0] - 1) + m_dt * m_c_butcher[m_s_butcher-1];
            //  Stage-type DOF    
            } else { 
                t = m_dt * blockInd[globalInd-globalInd0]       + m_dt * m_c_butcher[localInd[globalInd-globalInd0]-1];
            }
            getSpatialDiscretization(L_rowptr, L_colinds, L_data, B0, V0, spatialDOFs, t, m_bsize);
        }
    
    
        std::map<int, double>::iterator it;
        /* -------------------------------------------------------- */
        /* ------ Assemble block row for a solution-type DOF ------ */
        /* -------------------------------------------------------- */
        if (localInd[globalInd-globalInd0] == 0) {
            // Initial condition: Set matrix to identity and fix RHS and initial guess to ICs.
            if (blockInd[globalInd-globalInd0] == 0) {
    
                // Loop over each row
                for (int row = 0; row < spatialDOFs; row++) {
                    colinds[dataInd] = row; // Note globalColOffset == 0 for this DOF
                    data[dataInd] = 1.0;
                    dataInd += 1;
                    rowptr[row+1] = dataInd;
                    // Set these to 0 then add IC to them below.
                    B[row] = 0.0;
                    V[row] = 0.0;
                }
                addInitialCondition(B);
                addInitialCondition(V); 
    
            // Solution-type DOF at time t > 0
            } else {
                // This DOF couples s DOFs back to the solution at the previous time
                globalColOffset = spatialDOFs * (globalInd - m_s_butcher);
    
                // Loop over each row in spatial discretization, working from left-most column/variables to right-most
                for (int row=0; row<spatialDOFs; row++) {
                    /* ------ Coupling to solution at previous time ------ */
                    std::map<int, double> entries; 
                    // Get mass-matrix data
                    for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                        if (std::abs(M_data[j]) > 1e-16) {
                            entries[M_colinds[j]] = M_data[j];
                        }
                    }
    
                    // Add spatial discretization data to mass-matrix data
                    temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1];
                    if (temp != 0.0) {
                        temp *= m_dt;
                        for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                            if (std::abs(L_data[j]) > 1e-16) {
                                entries[L_colinds[j]] += temp * L_data[j];
                            }
                        }
                    }
    
                    // Add this data to global matrix
                    for (it = entries.begin(); it != entries.end(); it++) {
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
                            temp *= m_dt;
                            for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                                if (std::abs(M_data[j]) > 1e-16) {
                                    entries2[M_colinds[j]] = temp * M_data[j];
                                }
                            }
                        }
    
                        // Add spatial discretization data to mass-matrix data
                        temp = m_b_butcher[i] * m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] * m_A_butcher[m_s_butcher-1][i];
                        if (temp != 0.0) {
                            temp *= (m_dt * m_dt);
                            for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                                if (std::abs(L_data[j]) > 1e-16) {
                                    entries2[L_colinds[j]] += temp * L_data[j];
                                }
                            }
                        }
    
                        // Add this data to global matrix
                        localColOffset = globalColOffset + (i+1) * spatialDOFs;
                        for (it = entries2.begin(); it != entries2.end(); it++) {
                            colinds[dataInd] = localColOffset + it->first;
                            data[dataInd] = -it->second;
                            dataInd += 1;
                        }
                    }
    
    
                    /* ------ Coupling to myself/solution at current time ------ */
                    std::map<int, double> entries3; 
                    // Get mass-matrix data
                    for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                        if (std::abs(M_data[j]) > 1e-16) {
                            entries3[M_colinds[j]] = M_data[j];
                        }
                    }
    
                    // Add spatial discretization data to mass-matrix data
                    temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1];
                    if (temp != 0.0) {
                        temp *= m_dt;
                        for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                            if (std::abs(L_data[j]) > 1e-16) {
                                entries3[L_colinds[j]] += temp * L_data[j];
                            }
                        }
                    }
    
                    // Add this data to global matrix
                    localColOffset = globalColOffset + m_s_butcher * spatialDOFs;
                    for (it = entries3.begin(); it != entries3.end(); it++) {
                        colinds[dataInd] = localColOffset + it->first;
                        data[dataInd] = it->second;
                        dataInd += 1;
                    }
    
    
                    // RHS and initial guess
                    B[rowOffset + row] = m_dt * m_b_butcher[m_s_butcher-1] * B0[row];
                    V[rowOffset + row] = V0[row];
    
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
            for (int row = 0; row < spatialDOFs; row++) {
                /* ------ Coupling to solution at current time ------ */
                for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                    colinds[dataInd] = globalColOffset + L_colinds[j];
                    data[dataInd] = L_data[j];
                    dataInd += 1;
                }
    
                /* ------ Coupling to stages that come before me ------ */
                for (int i = 0; i < kInd-1; i++) {
                    temp = m_A_butcher[kInd-1][i];
                    if (temp != 0.0) {
                        temp *= m_dt;
                        localColOffset = globalColOffset + (i+1) * spatialDOFs;
                        for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                            if (std::abs(L_data[j]) > 1e-16) {
                                colinds[dataInd] = localColOffset + L_colinds[j];
                                data[dataInd] = temp * L_data[j];
                                dataInd += 1;
                            }
                        }
                    }
                }
    
                /* ------ Coupling to myself ------ */
                std::map<int, double> entries; 
                // Get mass-matrix data
                for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        entries[M_colinds[j]] = M_data[j];
                    }
                }
    
                // Add spatial discretization data to mass-matrix data
                temp = m_A_butcher[kInd-1][kInd-1];
                if (temp != 0.0) {
                    temp *= m_dt;
                    for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                        if (std::abs(L_data[j]) > 1e-16) {
                            entries[L_colinds[j]] += temp * L_data[j];
                        }
                    }
                }
    
                // Add this data to global matrix
                localColOffset = globalColOffset + kInd * spatialDOFs;
                for (it = entries.begin(); it != entries.end(); it++) {
                    colinds[dataInd] = localColOffset + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }
    
                // RHS and initial guess
                B[rowOffset + row] = B0[row];
                V[rowOffset + row] = V0[row];
    
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
        std::cout << "WARNING: Space-time RK matrix has more nonzeros than allocated on process " << m_globRank << " of " << m_numProc << ".\n";
    }
    
    // Clean up.
    delete[] localInd;
    delete[] blockInd;
    delete[] L_rowptr;
    delete[] L_colinds;
    delete[] L_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] V0;
}




/* --------------------------------------------------------- */
/* ------------------ Spatial parallelism ------------------ */
/* --------------------------------------------------------- */

/* Get component of block row of s-step BDF space-time equations. Uses spatial parallelism.

NOTEs: 
    -m_t0 is assumed to be 0, and so the spatial discretization is evaluated at
        time t0 + (n+s)*dt == (n+s)*dt.
*/
void SpaceTimeMatrix::BDFSpaceTimeBlock(int    * &rowptr, 
                                        int    * &colinds, 
                                        double * &data, 
                                        double * &B, 
                                        double * &V, 
                                        int      &localMinRow, 
                                        int      &localMaxRow, 
                                        int      &spatialDOFs) 
{
    int globalInd = m_DOFInd; // Index of temporal DOF on process
    
    /* --- Get spatial discretization at time required temporal DOF on process --- */
    int      L_nnz;
    int    * L_rowptr;
    int    * L_colinds;
    double * L_data;
    double * B0;
    double * V0;
    getSpatialDiscretization(m_spatialComm, L_rowptr, L_colinds, L_data, B0, V0, localMinRow, 
                                localMaxRow, spatialDOFs, m_t0 + (globalInd+m_s_multi)*m_dt, m_bsize);
    int onProcSize = localMaxRow - localMinRow + 1; // Number of rows on process
    L_nnz          = L_rowptr[onProcSize] - L_rowptr[0];   
    
    /* --- Get mass matrix ---*/
    int      M_nnz;
    int    * M_rowptr;
    int    * M_colinds;
    double * M_data;
    // Set range of identity mass matix to assemble if needed 
    if (!m_M_exists) setIdentityMassLocalRange(localMinRow, localMaxRow); 
    getMassMatrix(M_rowptr, M_colinds, M_data);
    M_nnz = M_rowptr[onProcSize] - M_rowptr[0];


    /* --- Get total NNZ estimate on process --- */
    //  -Doesn't assume sparsity of M and L overlap: Estimate will be an upperbound.
    int onProcNnz = 0;
    // Coupling to myself
    onProcNnz += M_nnz;
    onProcNnz += L_nnz;
    
    // Coupling to DOFs at s previous times
    if (globalInd > m_s_multi - 1) {
        onProcNnz += m_s_multi * M_nnz;
    } else { // Or n times if n < s
        onProcNnz += globalInd * M_nnz;
    }
    
    rowptr    = new int[onProcSize + 1];
    rowptr[0] = 0; 
    colinds   = new int[onProcNnz];
    data      = new double[onProcNnz];
    B         = new double[onProcSize];
    V         = new double[onProcSize];
    
    int dataInd      = 0;
    int colOffset    = 0;  // Global index of first column for DOF that we're coupling to
    double temp      = 0.0;
    
    
    /* ------------------------------------------------ */
    /* ------ Build block row for DOF on process ------ */
    /* ------------------------------------------------ */
    std::map<int, double>::iterator it;

    // Loop over all rows of spatial discretization on process
    for (int row = 0; row < onProcSize; row++) {
        B[row] = m_dt*m_b_multi[0]*B0[row]; // PDE source term. NOTE: Only b_s is stored for BDF schemes 
        V[row] = V0[row];                   // Initial guess at solution
    
        // Number of previous DOFs current DOF couples to
        int s_effective = m_s_multi;
        
        // First s DOFs only couple to the n times before them rather than all s
        if (globalInd <= m_s_multi - 1) {
            s_effective = globalInd;
            // Add precomputed w[n] vector holding all necessary starting-value information
            B[row] += m_w_multi[globalInd][row];
        }
        
        // Global index of furthest DOF current DOF couples back to
        colOffset = (globalInd - s_effective) * spatialDOFs;
        
        /* ------ Off-block-diagonal component that reaches back s_effective DOFs with mass matrix couplings ------ */
        for (int j = s_effective; j >= 1; j--)  {
            // Get mass-matrix data
            for (int p = M_rowptr[row]; p < M_rowptr[row+1]; p++) {                
                data[dataInd]    = m_a_multi[m_s_multi-j]*M_data[p];
                colinds[dataInd] = M_colinds[p] + colOffset;
                dataInd         += 1;
            }
            colOffset += spatialDOFs;
        }
    
        /* ------ Block-diagonal component ------ */
        std::map<int, double> entries; 
        // Get mass-matrix data
        for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
            if (std::abs(M_data[j]) > 1e-16) {
                entries[M_colinds[j]] = M_data[j];
            }
        }

        // Add spatial discretization data to mass-matrix data
        temp = m_dt * m_b_multi[0]; // Note only b_s is stored for BDF schemes
        for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
            if (std::abs(L_data[j]) > 1e-16) {
                entries[L_colinds[j]] += temp * L_data[j];
            }
        }

        // Add this data to global matrix
        for (it = entries.begin(); it != entries.end(); it++) {
            colinds[dataInd] = it->first + colOffset;
            data[dataInd]    = it->second;
            dataInd         += 1;
        }

        // Move to next row of spatial discretization for current DOF
        rowptr[row+1] = dataInd;
    }
    
    // Check sufficient data was allocated
    if (dataInd > onProcNnz) {
        std::cout << "WARNING: Space-time BDF matrix has more nonzeros than allocated on process " << m_globRank << " of " << m_numProc << ".\n";
    }
    
    // Clean up.
    delete[] L_rowptr;
    delete[] L_colinds;
    delete[] L_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
    delete[] B0;
    delete[] V0;
    
    // Information for initializing space-time RHS vector no longer needed.
    for (int i = 0; i < m_w_multi.size(); i++) {
        delete[] m_w_multi[i];
        m_w_multi[i] = NULL;
    }
}  



/* Get block row of s-stage RK space-time equations. 

    DOFS are blocked in groups of s: There are s-1 stages followed by the 
        solution at the new time.
    
    Uses spatial parallelism.
*/
void SpaceTimeMatrix::RKSpaceTimeBlock(int    * &rowptr, 
                                       int    * &colinds, 
                                       double * &data, 
                                       double * &B, 
                                       double * &V, 
                                       int      &localMinRow, 
                                       int      &localMaxRow, 
                                       int      &spatialDOFs)
{
    
    int globalInd = m_DOFInd;                   // Global index of variable on process
    int localInd  = globalInd % m_s_butcher;    // Local index of variable on process
    int blockInd  = globalInd / m_s_butcher;    // Block index of variable on process
    
    /* --- Get spatial discretization at time required by 1st DOF on process --- */
    int      L_nnz;
    int    * L_rowptr;
    int    * L_colinds;
    double * L_data;
    double t = 0.0; // Time to evaluate spatial discretization
    if (localInd == 0) { // Solution-type DOF that's not the initial condition
        if (blockInd != 0) t = m_dt * (blockInd-1) + m_dt * m_c_butcher[m_s_butcher-1]; 
    } else { // Stage-type DOF
        t = m_dt * blockInd + m_dt * m_c_butcher[localInd-1]; 
    }
    getSpatialDiscretization(m_spatialComm, L_rowptr, L_colinds, L_data, B, V, localMinRow, 
                                localMaxRow, spatialDOFs, t, m_bsize);
    int onProcSize = localMaxRow - localMinRow + 1; // Number of rows on process
    L_nnz          = L_rowptr[onProcSize] - L_rowptr[0];  
    
    /* --- Get mass matrix ---*/
    int      M_nnz;
    int    * M_rowptr;
    int    * M_colinds;
    double * M_data;
    
    // Set range of identity mass matix to assemble if needed 
    if (!m_M_exists) setIdentityMassLocalRange(localMinRow, localMaxRow); 
    getMassMatrix(M_rowptr, M_colinds, M_data);
    M_nnz = M_rowptr[onProcSize] - M_rowptr[0];


    /* ------ Get total NNZ on this processor. ------ */
    //  -Doesn't assume sparsity of M and L overlap: This nnz count is an upperbound.
    int onProcNnz = 0;
    // Solution-type variable 
    if (localInd == 0) {
        // Initial condition just has identity coupling
        if (blockInd == 0) {
            onProcNnz += spatialDOFs;
        // All solution DOFs at t > 0
        } else {
            // Coupling to itself
            onProcNnz += M_nnz;
            if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] != 0.0) onProcNnz += L_nnz;
    
            // Coupling to solution DOF at previous time 
            onProcNnz += M_nnz;
            if (m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] != 0.0) onProcNnz += L_nnz; 
    
            // Coupling to stage variables at previous time 
            for (int j = 0; j < m_s_butcher-1; j++) {
                if (m_b_butcher[j] != 0.0) onProcNnz += M_nnz;
                if (m_b_butcher[j]*m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1]*m_A_butcher[m_s_butcher-1][j] != 0.0) onProcNnz += L_nnz;
            }
        }
    
    // Stage-type variable 
    } else {
        int stageInd = localInd - 1; // Work with this to index Butcher arrays directly.
        // Coupling to itself
        onProcNnz += M_nnz;
        if (m_A_butcher[stageInd][stageInd] != 0.0) onProcNnz += L_nnz;
    
        // Coupling to solution at previous time
        onProcNnz += L_nnz;
    
        // Coupling to previous stage variables at current time
        for (int j = 0; j < stageInd; j++) {
            if (m_A_butcher[stageInd][j] != 0.0) onProcNnz += L_nnz;
        }
    }
    
    
    rowptr    = new int[onProcSize + 1];
    colinds   = new int[onProcNnz];
    data      = new double[onProcNnz];
    rowptr[0] = 0; 
    
    int dataInd         = 0;
    int globalColOffset = 0;    // Global index of first column for furthest DOF that we couple back to
    int localColOffset  = 0;    // Temporary variable to help indexing
    double temp         = 0.0;  // Temporary constant 
    
    /* ---------------------------------------------------------- */
    /* ------ Build block row of space-time matrix for DOF ------ */
    /* ---------------------------------------------------------- */
    
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
                V[row] = 0.0; 
            }
            addInitialCondition(m_spatialComm, B);
            addInitialCondition(m_spatialComm, V); 
    
    
        // Solution-type variable at time t > 0
        } else {
            // This DOF couples s DOFs back to the solution at the previous time
            globalColOffset = spatialDOFs * (globalInd - m_s_butcher);
    
            // Loop over each row in spatial discretization, working from left-most column/variables to right-most
            for (int row = 0; row < onProcSize; row++) {
                /* ------ Coupling to solution at previous time ------ */
                std::map<int, double> entries; 
                // Get mass-matrix data
                for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        entries[M_colinds[j]] = M_data[j];
                    }
                }
    
                // Add spatial discretization data to mass-matrix data
                temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1];
                if (temp != 0.0) {
                    temp *= m_dt;
                    for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                        if (std::abs(L_data[j]) > 1e-16) {
                            entries[L_colinds[j]] += temp * L_data[j];
                        }
                    }
                }
    
                // Add this data to global matrix
                for (it = entries.begin(); it != entries.end(); it++) {
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
                        temp *= m_dt;
                        for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                            if (std::abs(M_data[j]) > 1e-16) {
                                entries2[M_colinds[j]] = temp * M_data[j];
                            }
                        }
                    }
    
                    // Add spatial discretization data to mass-matrix data
                    temp = m_b_butcher[i] * m_A_butcher[m_s_butcher-1][m_s_butcher-1] - m_b_butcher[m_s_butcher-1] * m_A_butcher[m_s_butcher-1][i];
                    if (temp != 0.0) {
                        temp *= (m_dt * m_dt);
                        for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                            if (std::abs(L_data[j]) > 1e-16) {
                                entries2[L_colinds[j]] += temp * L_data[j];
                            }
                        }
                    }
    
                    // Add this data to global matrix
                    localColOffset = globalColOffset + (i+1) * spatialDOFs;
                    for (it = entries2.begin(); it != entries2.end(); it++) {
                        colinds[dataInd] = localColOffset + it->first;
                        data[dataInd] = -it->second;
                        dataInd += 1;
                    }
                }
    
    
                /* ------ Coupling to myself/solution at current time ------ */
                std::map<int, double> entries3; 
                // Get mass-matrix data
                for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                    if (std::abs(M_data[j]) > 1e-16) {
                        entries3[M_colinds[j]] = M_data[j];
                    }
                }
    
                // Add spatial discretization data to mass-matrix data
                temp = m_A_butcher[m_s_butcher-1][m_s_butcher-1];
                if (temp != 0.0) {
                    temp *= m_dt;
                    for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                        if (std::abs(L_data[j]) > 1e-16) {
                            entries3[L_colinds[j]] += temp * L_data[j];
                        }
                    }
                }
    
                // Add this data to global matrix
                localColOffset = globalColOffset + m_s_butcher * spatialDOFs;
                for (it = entries3.begin(); it != entries3.end(); it++) {
                    colinds[dataInd] = localColOffset + it->first;
                    data[dataInd] = it->second;
                    dataInd += 1;
                }
    
    
                // Scale solution-independent spatial component by coefficient
                B[row] *= (m_dt * m_b_butcher[m_s_butcher-1]);
    
                // Move to next row for current variable
                rowptr[row+1] = dataInd;
            }
        }
        
    /* ----------------------------------------------------- */
    /* ------ Assemble block row for a stage-type DOF ------ */
    /* ----------------------------------------------------- */
    } else {
        int stageInd = localInd; // 1's-based index of current stage DOF
        globalColOffset = spatialDOFs * (globalInd - stageInd); // We couple back to the solution at current time
    
        // Loop over each row in spatial discretization, working from left-most column/variables to right-most
        for (int row = 0; row < onProcSize; row++) {
            /* ------ Coupling to solution at current time ------ */
            for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                colinds[dataInd] = globalColOffset + L_colinds[j];
                data[dataInd] = L_data[j];
                dataInd += 1;
            }
    
            /* ------ Coupling to stages that come before me ------ */
            for (int i = 0; i < stageInd-1; i++) {
                temp = m_A_butcher[stageInd-1][i];
                if (temp != 0.0) {
                    temp *= m_dt;
                    localColOffset = globalColOffset + (i+1) * spatialDOFs;
                    for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                        if (std::abs(L_data[j]) > 1e-16) {
                            colinds[dataInd] = localColOffset + L_colinds[j];
                            data[dataInd] = temp * L_data[j];
                            dataInd += 1;
                        }
                    }
                }
            }
    
            /* ------ Coupling to myself ------ */
            std::map<int, double> entries; 
            // Get mass-matrix data
            for (int j = M_rowptr[row]; j < M_rowptr[row+1]; j++) {                
                if (std::abs(M_data[j]) > 1e-16) {
                    entries[M_colinds[j]] = M_data[j];
                }
            }
    
            // Add spatial discretization data to mass-matrix data
            temp = m_A_butcher[stageInd-1][stageInd-1];
            if (temp != 0.0) {
                temp *= m_dt;
                for (int j = L_rowptr[row]; j < L_rowptr[row+1]; j++) {
                    if (std::abs(L_data[j]) > 1e-16) {
                        entries[L_colinds[j]] += temp * L_data[j];
                    }
                }
            }
    
            // Add this data to global matrix
            localColOffset = globalColOffset + stageInd * spatialDOFs;
            for (it = entries.begin(); it != entries.end(); it++) {
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
    if (dataInd > onProcNnz) {
        std::cout << "WARNING: Space-time RK matrix has more nonzeros than allocated on process " << m_globRank << " of " << m_numProc << ".\n";
    }
    
    // Clean up.
    delete[] L_rowptr;
    delete[] L_colinds;
    delete[] L_data;
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
}













/*
BELOW: ALL OLD CODE TO BE REMOVED....
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
*/





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

