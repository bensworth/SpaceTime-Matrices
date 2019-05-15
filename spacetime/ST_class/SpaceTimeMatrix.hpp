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
    /* Struct containing Butcher table for an RK methods */
    // TODO: What's the best way to have these arrays... is it OK to just do it like this and just fill-in the parts we need?
    struct RK_butcher {
        int s;     // Number of stages 
        double a[10][10]; // Matrix of coefficients.
        double b[10]; // Quadrature weights
        double c[10]; // Quadrature nodes
    };

    int     m_globRank;
    int     m_timeInd;
    int     m_spatialRank;
    int     m_timeDisc;
    int     m_numTimeSteps;
    int     m_numProc;
    int     m_ntPerProc;
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
    RK_butcher m_tableaux;

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
