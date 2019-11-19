#include <mpi.h>
#include "HYPRE.h"
#include <map>
#include <vector>
#include <string>
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
    int    AMGiters;            /* Number of AMG iterations to precondition a GMRES iteration by, if preconditioning by GMRES */
    int    gmres_AMG_printLevel;/* Print level for BoomerAMG if used as a preconditioner */
    
    bool   binv_scale;          /* Scale by block inverse if mass matrix is block diagonal */
    
    int    rebuildRate;         /* Time-stepping: Rate solver is rebuilt */
};


class SpaceTimeMatrix
{
private:
    
    
    bool    m_pit;                  /* Parallel (true) or sequential (false) in time */
    double  m_dt;                   /* Time step (constant) */
    double  m_t0;                   /* Initial time to integrate from */
    int     m_nt;                   /* Number of time point/solution DOFs. We do Nt-1 time steps */

    int     m_DOFInd;               /* Index of DOF that spatial comm group belongs to */
    int     m_nDOFPerProc;          /* Number of temporal DOFs per proc, be they solution variables and/or stage variables */
    
    int     m_globRank;             /* Rank in global communicator */
    int     m_numProc;              /* Total number of procs; TODO : Change to "m_globCommSize" */
    
    
    bool    m_rebuildSolver;        /* Flag specifying when AMG solver is rebuilt */
    
    
    /* --- The time-integration scheme --- */
    int     m_timeDisc;             /* ID of time-integration scheme */
    
    /* Runge-Kutta Butcher tableaux variables */
    bool    m_RK;                   /* Runge-Kutta time integration */
    bool    m_is_ERK;               /* Explicit Runge-Kutta */
    bool    m_is_DIRK;              /* Diagonally implicit Runge-Kutta */
    bool    m_is_SDIRK;             /* Singly diagonally Runge-Kutta */
    int                              m_s_butcher; /* Number of stages in RK scheme */
    std::vector<std::vector<double>> m_A_butcher; /* Coefficients in RK Butcher tableaux */
    std::vector<double>              m_b_butcher; /* Coefficients in RK Butcher tableaux */
    std::vector<double>              m_c_butcher; /* Coefficients in RK Butcher tableaux */

    bool    m_BDF;                  /* BDF time integration */


    MPI_Comm            m_globComm;
    HYPRE_Solver        m_solver;
    HYPRE_Solver        m_gmres;
    HYPRE_ParCSRMatrix  m_A;
    HYPRE_IJMatrix      m_Aij;
    HYPRE_ParVector     m_b;
    HYPRE_IJVector      m_bij;
    HYPRE_ParVector     m_x;
    HYPRE_IJVector      m_xij;
    AMG_parameters      m_AMG_parameters;
    Solver_parameters   m_solver_parameters;

    
    int     m_bsize;                /* DG specific variable... */

    // TODO : variables to remove    
    int     m_Np_x;     /* TODO : Remove. Replace with protected variable "m_spatialCommSize" */
    int     m_ntPerProc; // TODO: this variable doesn't really make sense...
    int     m_numTimeSteps; // TODO: I don't think we should use this variable it's confusing: we assume Nt points, but we do Nt-1 time steps. But it's the Nt that's important because there are Nt DOFs and not Nt-1...
    bool    m_isTimeDependent; //  TODO : Will need to be removed ...
    int     m_spCommSize; // TODO : remove... now protected
    
    double  m_t1; // TOOD :  do we use this?? delete??
    int     m_timeInd; // TODO: remove this. It no longer applies. Variable below makes more sense.

    void GetButcherTableaux();
    void GetMatrix_ntLE1();
    void GetMatrix_ntGT1();
    void SetBoomerAMGOptions(int printLevel=3, int maxiter=250, double tol=1e-8);
    void SetGMRESOptions();

    // Routine to build space-time matrices when the spatial discretization
    // takes more than one processor.
    void RKSpaceTimeBlock(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);

    // Routine to build space-time matrices when spatial at least one DOF is allocated per processor.
    void RKSpaceTimeBlock(int* &rowptr, int* &colinds, double* &data, double* &B, 
              double* &X, int &onProcSize);
    void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);


    // TODO : these functions are to be phased out...
    // -----------------------------------------
    virtual void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                          int* &A_colinds, double* &A_data, double* &B,
                                          double* &X, int &localMinRow, int &localMaxRow,
                                          int &spatialDOFs, double t, int &bsize) = 0;

    virtual void getSpatialDiscretization(int* &A_rowptr, int* &A_colinds, double* &A_data,
                                          double* &B, double* &X, int &spatialDOFs,
                                          double t, int &bsize) = 0;                                    
    // -----------------------------------------

    // Spatial discretization on more than one processor. Must same row distribution
    // over processors each time called, e.g., first processor in communicator gets
    // first 10 spatial DOFs, second processor next 10, and so on. 
    
    virtual void getSpatialDiscretizationG(const MPI_Comm &spatialComm, double* &G, 
                                            int &localMinRow, int &localMaxRow,
                                            int &spatialDOFs, double t);                                   
    virtual void getSpatialDiscretizationL(const MPI_Comm &spatialComm, int* &A_rowptr, 
                                            int* &A_colinds, double* &A_data,
                                            double* &U0, bool getU0, 
                                            int &localMinRow, int &localMaxRow, 
                                            int &spatialDOFs,
                                            double t, int &bsize);                                            
                                          
                                          
    // Spatial discretization on one processor                                  
    virtual void getSpatialDiscretizationG(double* &G, int &spatialDOFs, double t);
    virtual void getSpatialDiscretizationL(int* &A_rowptr, int* &A_colinds, double* &A_data,
                                          double* &U0, bool getU0, int &spatialDOFs,
                                          double t, int &bsize);                                            
        
                                                                            
    // Get mass matrix for time integration; only for finite element discretizations.
    virtual void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);

    
    // TODO : these need to be implemented in CG and DG also...
    virtual void getInitialCondition(const MPI_Comm &spatialComm, double * &B, 
                                        int &localMinRow, int &localMaxRow, 
                                        int &spatialDOFs) = 0;
    virtual void getInitialCondition(double * &B, int &spatialDOFs) = 0;
    // TODO : I  don't think these make sense... Shouldn't we have as  above?
    // TODO: make optional? 
    virtual void addInitialCondition(const MPI_Comm &spatialComm, double *B) = 0;
    virtual void addInitialCondition(double *B) = 0;

    
    /* ------ Sequential time integration routines ------ */
    void ERKTimeSteppingSolve();  /* General purpose ERK solver */
    void ERKSolveWithMass();      /* General purpose ERK solver that can invert mass matrices */
    void DIRKTimeSteppingSolve(); /* General purpose DIRK solver */
    
    void GetHypreInitialCondition(HYPRE_ParVector  &u0, 
                                    HYPRE_IJVector &u0ij); 
    
    void InitializeHypreVectors(HYPRE_ParVector                  &u0, 
                                    HYPRE_IJVector               &u0ij, 
                                    int                           numVectors,
                                    std::vector<HYPRE_ParVector> &z, 
                                    std::vector<HYPRE_IJVector>  &zij); 
                                    
    void GetHypreSpatialDiscretizationG(HYPRE_ParVector    &g,
                                            HYPRE_IJVector &gij,
                                            double          t);
                                                    
    void GetHypreSpatialDiscretizationL(HYPRE_ParCSRMatrix &L,
                                            HYPRE_IJMatrix &Lij,
                                            double          t);
    
    void SolveAMG();
    void SolveGMRES();
    
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
    bool     m_g_isTimedependent; /* Is PDE source term time dependent? */

    
    bool     m_M_exists; /* Is mass-matrix the identity? */
    int *    m_M_rowptr;
    int *    m_M_colinds;
    double * m_M_data;
    
    double   m_hmin;
    double   m_hmax;

public:
    SpaceTimeMatrix(MPI_Comm globComm, int timeDisc, int numTimeSteps, double dt, bool pit);
    // SpaceTimeMatrix(MPI_Comm globComm, int timeDisc, double dt, double t0, double t1);
    virtual ~SpaceTimeMatrix();


    void Solve(); /* General solver! */

    void SetAMG();
    void SetAIR();
    void SetAIRHyperbolic();
    void SetAMGParameters(AMG_parameters &AMG_params);
    
    void SetSolverParametersDefaults();
    void SetSolverParameters(Solver_parameters &solver_params); 
        
    
    void PrintMeshData();

    void SaveMatrix(const char* filename) { HYPRE_IJMatrixPrint (m_Aij, filename); }
    void SaveRHS(std::string filename) { HYPRE_IJVectorPrint(m_bij, filename.c_str()); }
    void SaveX(std::string filename) { HYPRE_IJVectorPrint(m_xij, filename.c_str()); }
    void SaveSolInfo(std::string filename, std::map<std::string, std::string> additionalInfo);
};
