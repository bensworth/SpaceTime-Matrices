#include <mpi.h>
#include "HYPRE.h"
#include <string>
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"
#define SPACETIMEMATRIX


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


class SpaceTimeMatrix
{
private:
    
    int     m_globRank;
    int     m_timeInd;
    int     m_spatialRank;
    int     m_timeDisc;
    int     m_numTimeSteps; // TODO: I don't think we should use this variable it's confusing: we assume Nt points, but we do Nt-1 time steps. But it's the Nt that's important because there are Nt DOFs and not Nt-1...
    int     m_nt;                   /* Number of time point/solution DOFs. We do Nt-1 time steps */
    int     m_numProc;
    int     m_ntPerProc; // TODO: this variable doesn't really make sense...
    int     m_nDOFPerProc;          /* Number of temporal DOFs per proc, be they solution variables and/or stage variables */
    int     m_Np_x;
    int     m_bsize;
    int     m_spCommSize;
    bool    m_useSpatialParallel;
    bool    m_isTimeDependent;
    bool    m_rhsTimeDependent;
    bool    m_rebuildSolver;
    double  m_t0;
    double  m_t1;
    double  m_dt;
    int     m_s_butcher;            /* Number of stages in RK scheme */
    double  m_A_butcher[10][10];    /* Coefficients in RK Butcher tableaux */
    double  m_b_butcher[10];        /* Coefficients in RK Butcher tableaux */
    double  m_c_butcher[10];        /* Coefficients in RK Butcher tableaux */

    MPI_Comm            m_globComm;
    MPI_Comm            m_spatialComm;
    HYPRE_Solver        m_solver;
    HYPRE_Solver        m_gmres;
    HYPRE_ParCSRMatrix  m_A;
    HYPRE_IJMatrix      m_Aij;
    HYPRE_ParVector     m_b;
    HYPRE_IJVector      m_bij;
    HYPRE_ParVector     m_x;
    HYPRE_IJVector      m_xij;
    AMG_parameters      m_solverOptions;

    void GetButcherTableaux();
    void GetMatrix_ntLE1();
    void GetMatrix_ntGT1();
    void SetupBoomerAMG(int printLevel=3, int maxiter=250, double tol=1e-8);

    // Routines to build space-time matrices when the spatial discretization
    // takes up one or more processors.
    void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);

    // Routines to build space-time matrices when more than one time
    // step are allocated per processor.
    void RK(int* &rowptr, int* &colinds, double* &data, double* &B, 
              double* &X, int &onProcSize);
    void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);

    // Spatial discretization on more than one processor. Must same row distribution
    // over processors each time called, e.g., first processor in communicator gets
    // first 10 spatial DOFs, second processor next 10, and so on. 
    virtual void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                          int* &A_colinds, double* &A_data, double* &B,
                                          double* &X, int &localMinRow, int &localMaxRow,
                                          int &spatialDOFs, double t, int &bsize) = 0;
    // Spatial discretization on one processor
    virtual void getSpatialDiscretization(int* &A_rowptr, int* &A_colinds, double* &A_data,
                                          double* &B, double* &X, int &spatialDOFs,
                                          double t, int &bsize) = 0;

    // Get mass matrix for time integration; only for finite element discretizations.
    virtual void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);

    // TODO: make optional? 
    virtual void addInitialCondition(const MPI_Comm &spatialComm, double *B) = 0;
    virtual void addInitialCondition(double *B) = 0;


protected:

    int*    m_M_rowptr;
    int*    m_M_colinds;
    double* m_M_data;
    double  m_hmin;
    double  m_hmax;

public:

    SpaceTimeMatrix(MPI_Comm globComm, int timeDisc, int numTimeSteps, double dt);
    // SpaceTimeMatrix(MPI_Comm globComm, int timeDisc, double dt, double t0, double t1);
    virtual ~SpaceTimeMatrix();

    void BuildMatrix();
    void SaveMatrix(const char* filename) { HYPRE_IJMatrixPrint (m_Aij, filename); }
    void SaveRHS(std::string filename) { HYPRE_IJVectorPrint(m_bij, filename.c_str()); }
    void SaveX(std::string filename) { HYPRE_IJVectorPrint(m_xij, filename.c_str()); }

    void SetAMG();
    void SetAIR();
    void SetAIRHyperbolic();
    void SetAMGParameters(AMG_parameters &params);
    void SolveAMG(double tol=1e-8, int maxiter=250, int printLevel=3,
                  bool binv_scale=true);
    void SolveGMRES(double tol=1e-8, int maxiter=250, int printLevel=3,
                    bool binv_scale=true, int precondition=1, int AMGiters=10);
    void PrintMeshData();
};
