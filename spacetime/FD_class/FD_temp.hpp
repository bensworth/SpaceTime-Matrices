#include <mpi.h>
#include "HYPRE.h"
#include <string>
#include <vector>
#include "HYPRE_parcsr_ls.h"
// #include "_hypre_parcsr_mv.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_krylov.h"
#define FD_TEMP

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
};


class FD_temp
{
private:

    MPI_Comm m_comm;
    int    m_rank;
    int    m_numProc;
    int    m_Pt;
    int    m_Px;
    int    m_Py;
    int    m_nt_local;
    int    m_nx_local;
    int    m_ny_local;
    int    m_numvar;
    int    m_dim;
    int    m_pt_ind;
    int    m_px_ind;
    int    m_py_ind;
    int    m_n;
    long   m_globt;
    long   m_globx;
    bool   m_rebuildSolver;
    double m_t0;
    double m_t1;
    double m_x0;
    double m_x1;
    double m_y0;
    double m_y1;
    double m_hx;
    double m_hy;
    double m_dt;
    std::vector<int> m_ilower;
    std::vector<int> m_iupper;
    AMG_parameters   m_solverOptions;

    HYPRE_SStructGrid     m_grid;
    HYPRE_SStructGraph    m_graph;
    HYPRE_SStructStencil  m_stencil_u;
    HYPRE_SStructStencil  m_stencil_v;
    HYPRE_SStructVector   m_bS;
    HYPRE_SStructVector   m_xS;
    HYPRE_SStructMatrix   m_AS;
    HYPRE_ParVector       m_x;
    HYPRE_ParVector       m_b;
    HYPRE_ParCSRMatrix    m_A;
    HYPRE_Solver          m_solver;
    HYPRE_Solver          m_gmres;

    void SetupBoomerAMG(int printLevel=3, int maxiter=250, double tol=1e-8);

public:

    FD_temp(MPI_Comm comm, int nt, int nx, int Pt, int Px);
    // SpaceTimeFD(MPI_Comm comm, int nt, int nx, int ny, int Pt, int Px, int Py);  // 3D
    

    // TODO : implement SaveMatrix
    // void SaveMatrix(const char* filename) { HYPRE_SStructMatrixPrint (m_AS, filename); }
    void SaveMatrix(std::string filename) { HYPRE_SStructMatrixPrint (filename.c_str(), m_AS, 1); }
    void SetAMG();
    void SetAIR();
    void SetAIRHyperbolic();
    void PrintMeshData();
    void SetAMGParameters(AMG_parameters &params);
    void SolveAMG(double tol=1e-8, int maxiter=250, int printLevel=3);
    void SolveGMRES(double tol=1e-8, int maxiter=250, int printLevel=3, int precondition=1);
    void Wave1D(double (*IC_u)(double), double (*IC_v)(double), double c);
};
