#include <mpi.h>
#include "HYPRE.h"
#include <string>
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

#include "_hypre_parcsr_mv.h"

/* Struct containing Butcher table for RK methods */
struct RK_butcher
{
    // Assume have no more than 3 stages.
    int num_stages; 
    double a[3][3]; // Matrix of coefficients.
    double b[3];
    double c[3]; 
    int isImplicit;
    int isSDIRK; 
};

/* Struct containing basis AMG/AIR parameters to pass to hypre. */
struct AMG_parameters {
    std::string prerelax;
    std::string postrelax;
    int relax_type;
    int interp_type;
    double strength_tolC;
    int coarsen_type;
    int distance_R;
    double strength_tolR;
    double filterA_tol;

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
    int     m_numTimeSteps;
    int     m_numProc;
    int     m_ntPerProc;
    int     m_Np_x;
    int     m_spCommSize;
    bool    m_useSpatialParallel;
    bool    m_isTimeDependent;
    bool    m_rhsTimeDependent;
    bool    m_rebuildSolver;
    double  m_t0;
    double  m_t1;
    double  m_dt;

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

    void GetMatrix_ntLE1();
    void GetMatrix_ntGT1();
    void SetupBoomerAMG(int printLevel=3, int maxiter=250, double tol=1e-8);

    // Routines to build space-time matrices when the spatial discretization
    // takes up one or more processors.
    void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void BDF2(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void BDF3(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void AM2(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    // void AB2(int *&rowptr, int *&colinds, double *&data, double *&B,
    //           double *&X, int &localMinRow, int &localMaxRow, int &spatialDOFs);

    // Routines to build space-time matrices when more than one time
    // step are allocated per processor.
    void BDF1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void BDF2(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void BDF3(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void AM2(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void AB1(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);
    void AB2(int *&rowptr, int *&colinds, double *&data, double *&B,
              double *&X, int &onProcSize);

    // Runge--Kutta schemes
    void getButcher(RK_butcher butch, int option);
    void ERK(MPI_Comm comm, RK_butcher butch, HYPRE_ParVector * par_u, HYPRE_ParVector * par_u0, double t0, double delta_t, int ilower, int iupper, int * M_rowptr, int * M_colinds, double * M_data, int * L_rowptr, int * L_colinds, double * L_data, double * g_data);
    void DIRK(MPI_Comm comm, RK_butcher butch, HYPRE_ParVector * par_u, HYPRE_ParVector * par_u0, double t0, double delta_t, int ilower, int iupper, int * M_rowptr, int * M_colinds, double * M_data, int * L_rowptr, int * L_colinds, double * L_data, double * g_data);

    
    // Spatial discretization on more than one processor. Must same row distribution
    // over processors each time called, e.g., first processor in communicator gets
    // first 10 spatial DOFs, second processor next 10, and so on. 
    virtual void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                          int *&A_colinds, double *&A_data, double *&B,
                                          double *&X, int &localMinRow, int &localMaxRow,
                                          int &spatialDOFs, double t, double dt) = 0;
    // Spatial discretization on one processor
    virtual void getSpatialDiscretization(int *&A_rowptr, int *&A_colinds, double *&A_data,
                                          double *&B, double *&X, int &spatialDOFs,
                                          double t, double dt) = 0;

    // TODO: Add support in functions for this; make optional? 
    // virtual void getRHS(const MPI_Comm &spatialComm, double *&B, double t) = 0;


public:

    SpaceTimeMatrix(MPI_Comm globComm, int timeDisc, int numTimeSteps, double t0, double t1);
    // SpaceTimeMatrix(MPI_Comm globComm, int timeDisc, double dt, double t0, double t1);
    virtual ~SpaceTimeMatrix();

    void BuildMatrix();
    void SaveMatrix(const char* filename) { HYPRE_IJMatrixPrint (m_Aij, filename); } // this said filename1?
    void SetAMG();
    void SetAIR();
    void SetAIRHyperbolic();
    void SetAMGParameters(AMG_parameters &params);
    void SolveAMG(double tol=1e-8, int maxiter=250, int printLevel=3);
    void SolveGMRES(double tol=1e-8, int maxiter=250, int printLevel=3, int precondition=1);

};
