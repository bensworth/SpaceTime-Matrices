#include <mpi.h>
#include "HYPRE.h"
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_krylov.h"
#define SPACETIMEMATRIX

// TODO :
// - Make spatial communicator a protected variable so that spatial discretization constructor can see it/access it...

/* Struct containing basis AMG/AIR parameters to pass to hypre. */
struct AMG_parameters {
   double distance_R;
   std::string prerelax;
   std::string postrelax;
   int interp_type;
   int relax_type;
   int coarsen_type;
   double strength_tolC;
   double strength_tolR;
   double filter_tolR;
   double filter_tolA;
   int cycle_type;

    // AMG_parameters() : 
    //     prerelax(""), postrelax(""), relax_type(-1),
    //     interp_type(-1), strength_tolC(0), coarsen_type(-1),
    //     distance_R(-1), strength_tolR(0), filterA_tol(0)
    // {

    // }
};

/* Struct containing solver options */
struct Solver_parameters {
    double tol;                 /* Relative residual holting tolerance */
    int    maxiter;             /* Maximum number of solver iterations */
    int    printLevel;          /* Amount of information about solver */
    
    bool   use_gmres;           /* Use preconditioned GMRES as solver */
    int    gmres_preconditioner;/* Preconditioner for GMRES */
    int    AMGiters;            /* Number of AMG iterations to precondition a GMRES iteration by (if using AMG) */
    int    precon_printLevel;   /* Print level for preconditioner (if using one) */
    
    int    rebuildRate;         /* Time-stepping: Rate solver is rebuilt */
    
    bool   binv_scale;          /* Scale by block inverse if mass matrix is block diagonal */
    bool   lump_mass;           /* Lump mass matrix so that it's diagonal */
};


class SpaceTimeMatrix
{
private:
    
    bool    m_pit;                  /* Parallel (true) or sequential (false) in time */
    bool    m_M_exists;             /* Does the spatial discretization use a mass matrix? */
    double  m_dt;                   /* Time step (constant) */
    double  m_t0;                   /* Initial time to integrate from */
    int     m_nt;                   /* Number of time point/solution DOFs. We do Nt-1 time steps */

    int     m_DOFInd;               /* Index of DOF that spatial comm group belongs to */
    int     m_nDOFPerProc;          /* Number of temporal DOFs per proc (in general, solution variables, but for Runge-Kutta, includes stage variables too) */
    
    int     m_globRank;             /* Rank in global communicator */
    int     m_numProc;              /* Total number of procs; TODO : Change to "m_globCommSize" */
    
    
    bool    m_rebuildSolver;        /* Flag specifying when AMG solver is rebuilt */
    
    
    /* --- The time-integration scheme --- */
    int     m_timeDisc;             /* ID of time-integration scheme */
    bool    m_implicit;             /* Flag for explicit/implicit time integration */
    
    /* Runge-Kutta Butcher tableaux variables */
    bool    m_RK;                /* Runge-Kutta time integration */
    bool    m_ERK;               /* Explicit Runge-Kutta */
    bool    m_DIRK;              /* Diagonally implicit Runge-Kutta */
    bool    m_SDIRK;             /* Singly diagonally Runge-Kutta */
    int                              m_s_butcher; /* Number of stages in RK scheme */
    std::vector<std::vector<double>> m_A_butcher; /* Coefficients in RK Butcher tableaux */
    std::vector<double>              m_b_butcher; /* Coefficients in RK Butcher tableaux */
    std::vector<double>              m_c_butcher; /* Coefficients in RK Butcher tableaux */

    bool                m_multi;   /* Multistep time integration */
    bool                m_AB;      /* Adams--Bashforth time integration */
    bool                m_AM;      /* Adams--Moulton time integration */
    bool                m_BDF;     /* BDF time integration */
    int                 m_s_multi; /* Number of steps in multistep scheme */
    std::vector<double> m_a_multi; /* Coefficients in multistep scheme */
    std::vector<double> m_b_multi; /* Coefficients in multistep scheme */
    


    /* --- Relating to HYPRE solution of linear systems --- */
    MPI_Comm            m_globComm;            /* Global communicator */
    MPI_Comm            m_solverComm;          /* Communicator used for solvers; this may be globComm, spatialComm, or may swap between the two */
    HYPRE_Solver        m_solver;
    HYPRE_Solver        m_gmres;
    HYPRE_Solver        m_pcg;
    HYPRE_ParCSRMatrix  m_M;
    HYPRE_IJMatrix      m_Mij;
    HYPRE_ParCSRMatrix  m_invM;                 /* If M is lumped to be diagonal, we may store its inverse */
    HYPRE_IJMatrix      m_invMij; 
    HYPRE_ParCSRMatrix  m_A;
    HYPRE_IJMatrix      m_Aij;
    HYPRE_ParVector     m_b;
    HYPRE_IJVector      m_bij;
    HYPRE_ParVector     m_x;
    HYPRE_IJVector      m_xij;
    std::vector<HYPRE_ParVector> m_u_multi;     /* Starting values for multistep time-stepping */
    std::vector<HYPRE_IJVector>  m_u_multi_ij;
    std::vector<double *> m_w_multi;            /* Solution-dependent RHS values needed for building multistep RHS vector */
    AMG_parameters      m_AMG_parameters;
    Solver_parameters   m_solver_parameters;
    bool                m_iterative;            /* Indicating whether an iterative or direct solver is used */
    int                 m_num_iters;            /* Number of iterations solver took */
    double              m_convergence_factor;   /* Average convergence factor during solve */
    double              m_res_norm;             /* Residual norm after solve */
    int                 m_hypre_ierr;           /* Generic error flag for HYPRE functions */
    
    
    /* --- Identity-mass-matrix related --- */
    int  m_M_localMinRow;
    int  m_M_localMaxRow;
    bool m_rebuildMass;
    
    int     m_bsize;                /* DG specific variable... */

    // TODO : variables to remove    
    int     m_Np_x;     /* TODO : Remove. Replace with protected variable "m_spatialCommSize" */
    int     m_ntPerProc; // TODO: this variable doesn't really make sense...
    int     m_numTimeSteps; // TODO: I don't think we should use this variable it's confusing: we assume Nt points, but we do Nt-1 time steps. But it's the Nt that's important because there are Nt DOFs and not Nt-1...
    int     m_spCommSize; // TODO : remove... now protected
    
    double  m_t1; // TOOD :  do we use this?? delete??
    int     m_timeInd; // TODO: remove this. It no longer applies. Variable below makes more sense.

    void GetButcherTableaux();
    void SetABTableaux();
    void SetAMTableaux();
    void SetBDFTableaux();
    bool SetMultiRKPairing();
    void SetMultistepStartValues();
    void SetMultistepSpaceTimeRHSValues();
    void GetMatrix_ntLE1();
    void GetMatrix_ntGT1();
    void SetBoomerAMGOptions(int printLevel=3, int maxiter=250, double tol=1e-8);
    void SetGMRESOptions();
    void SetPCGOptions();


    // Not using spatial parallelism
    void RKSpaceTimeBlock(int    * &rowptr, 
                          int    * &colinds, 
                          double * &data, 
                          double * &B, 
                          double * &V, 
                          int      &onProcSize);
              
    void BDFSpaceTimeBlock(int    * &rowptr, 
                           int    * &colinds, 
                           double * &data, 
                           double * &B, 
                           double * &V, 
                           int      &onProcSize);

    // Using spatial parallelism
    void RKSpaceTimeBlock(int    * &rowptr, 
                          int    * &colinds, 
                          double * &data, 
                          double * &B,
                          double * &V, 
                          int      &localMinRow, 
                          int      &localMaxRow, 
                          int      &spatialDOFs);
              
    void BDFSpaceTimeBlock(int    * &rowptr, 
                           int    * &colinds, 
                           double * &data, 
                           double * &B,
                           double * &V, 
                           int      &localMinRow, 
                           int      &localMaxRow, 
                           int      &spatialDOFs);
                           
    //  TODO : remove these functions...  
    // -----------------------------------------        
    // void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
    //           double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    // void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
    //           double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    // void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
    //           double *&X, int &onProcSize);
    // void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
    //           double *&X, int &onProcSize);
    // -----------------------------------------
                              
    

    // Spatial discretization on more than one processor. Must same row distribution
    // over processors each time called, e.g., first processor in communicator gets
    // first 10 spatial DOFs, second processor next 10, and so on. 
    
    
    // TODO : Make all spatial discretization functions PURE VIRTUAL once they've been implemented!
    virtual void getSpatialDiscretizationG(const MPI_Comm &spatialComm, 
                                           double * &G, 
                                           int      &localMinRow, 
                                           int      &localMaxRow,
                                           int      &spatialDOFs, 
                                           double    t);                                   
    virtual void getSpatialDiscretizationL(const MPI_Comm &spatialComm, 
                                           int    * &A_rowptr, 
                                           int    * &A_colinds, 
                                           double * &A_data,
                                           double * &U0, 
                                           bool      getU0, 
                                           int      &localMinRow, 
                                           int      &localMaxRow, 
                                           int      &spatialDOFs,
                                           double    t, 
                                           int      &bsize);                                            
                                          
                                          
    // Spatial discretization on one processor                                  
    virtual void getSpatialDiscretizationG(double * &G, 
                                           int      &spatialDOFs, 
                                           double    t);
    virtual void getSpatialDiscretizationL(int    * &A_rowptr, 
                                           int    * &A_colinds, 
                                           double * &A_data,
                                           double * &U0, 
                                           bool      getU0, 
                                           int      &spatialDOFs,
                                           double    t, 
                                           int      &bsize);                                            
        
                                                                            
    // Get mass matrix for time integration; only for finite element discretizations.
    virtual void getMassMatrix(int    * &M_rowptr, 
                               int    * &M_colinds, 
                               double * &M_data);
    // Helper function to setup identity mass matrix when not using finite-element discretization
    void setIdentityMassLocalRange(int localMinRow, int localMaxRow);
    
    // TODO : these need to be implemented in CG and DG also...
    virtual void getInitialCondition(const MPI_Comm &spatialComm, double * &B, 
                                        int &localMinRow, int &localMaxRow, 
                                        int &spatialDOFs) = 0;
    virtual void getInitialCondition(double * &B, int &spatialDOFs) = 0;

    // Optional function to obtain exact solution of PDE at time t. Boolean return value reflects whether such a solution is implemented
    virtual bool GetExactPDESolution(double * &U, int &spatialDOFs, double t) { return false; };
    
    virtual bool GetExactPDESolution(const MPI_Comm &spatialComm, 
                                        double * &U, int &localMinRow, 
                                        int &localMaxRow, 
                                        int &spatialDOFs, double t) { return false; };
    
    
    /* ------ Sequential time integration routines ------ */
    void ERKTimeSteppingSolve();    /* General purpose ERK solver */
    void DIRKTimeSteppingSolve();   /* General purpose DIRK solver */
    void ABTimeSteppingSolve();     /* General purpose Adams--Bashforth solver */
    void AMTimeSteppingSolve();     /* General purpose Adams--Moulton solver */
    void BDFTimeSteppingSolve();    /* General purpose BDF solver */
    
    void DestroyHypreMemberVariables();
    
    void GetHypreInitialCondition(HYPRE_ParVector &u0, 
                                  HYPRE_IJVector  &u0ij); 
    
    void InitializeHypreVectors(HYPRE_ParVector              &u0, 
                                HYPRE_IJVector               &u0ij, 
                                std::vector<HYPRE_ParVector> &z, 
                                std::vector<HYPRE_IJVector>  &zij); 
                                    
    void GetHypreSpatialDiscretizationG(HYPRE_ParVector &g,
                                        HYPRE_IJVector  &gij,
                                        double           t);
                                                    
    void GetHypreSpatialDiscretizationL(HYPRE_ParCSRMatrix &L,
                                        HYPRE_IJMatrix     &Lij,
                                        double              t);
            
            
    void GetHypreMatrixFromCRSData(HYPRE_ParCSRMatrix  &A,
                                   HYPRE_IJMatrix      &Aij,
                                   MPI_Comm             A_comm,
                                   int                * A_rowptr, 
                                   int                * A_colinds, 
                                   double             * A_data, 
                                   int                  ilower, 
                                   int                  iupper);         
                    
    void GetHypreVectorFromData(HYPRE_ParVector  &x, 
                                HYPRE_IJVector   &xij,
                                MPI_Comm          x_comm,
                                double          * x_data, 
                                int               ilower, 
                                int               iupper);                                      
                    
                                            
    void SetHypreMassMatrix(int  ilower, 
                            int  iupper);
                            
    void SetHypreInvMassMatrix(int  ilower, 
                               int  iupper);                                             
    
    void SolveAMG();
    void SolveGMRES();
    void SolveMassSystem();
    
    void BuildSpaceTimeMatrix();
    
    void SpaceTimeSolve();      /* Solve full space-time system */
    void TimeSteppingSolve();   /* Sequential time integration */
    
protected:    
    bool     m_useSpatialParallel;   /*  */
    
    MPI_Comm m_spatialComm;     /* Spatial communicator; the spatial discretization code has access to this */
    int      m_spatialCommSize; /* Num processes in spatial communicator */
    int      m_spatialRank;     /* Process rank in spatial communicator */
    
    // TOOD : Make sure these variables are set in spatial discretization code...
    bool     m_L_isTimedependent; /* Is spatial discretization time dependent? */
    bool     m_G_isTimedependent; /* Is PDE source term time dependent? */

    int *    m_M_rowptr;
    int *    m_M_colinds;
    double * m_M_data;
    
    double   m_hmin;
    double   m_hmax;

public:
    SpaceTimeMatrix(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps, double dt);
    virtual ~SpaceTimeMatrix();


    void Solve(); /* General solver! */

    void SetAMG();
    void SetAIR();
    void SetAIRHyperbolic();
    void SetAMGParameters(AMG_parameters &AMG_params);
    
    void SetSolverParametersDefaults();
    void SetSolverParameters(Solver_parameters &solver_params); 
        
    
    bool GetDiscretizationError(double &e2norm);
    
    void PrintMeshData();

    // TODO : Ben, Why the different string formats for A, b, x?
    void SaveMatrix(const char* filename) { if (m_Aij) HYPRE_IJMatrixPrint(m_Aij, filename); else std::cout << "WARNING: m_Aij == NULL, cannot be printed!\n"; }
    void SaveRHS(std::string filename)    { if (m_bij) HYPRE_IJVectorPrint(m_bij, filename.c_str()); else std::cout << "WARNING: m_bij == NULL, cannot be printed!\n";  }
    void SaveX(std::string filename)      { if (m_xij) HYPRE_IJVectorPrint(m_xij, filename.c_str()); else std::cout << "WARNING: m_xij == NULL, cannot be printed!\n";  }
    void SaveSolInfo(std::string filename, std::map<std::string, std::string> additionalInfo);
};
