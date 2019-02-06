#include "DGadvection.hpp"
#include <cstdio>
#include <vector>
#include <iostream>
using namespace mfem;


class CoefficientWithState : public Coefficient
{
protected:
    double (*Function)(const Vector &, const Vector &);

public:
    /// Define a time-independent coefficient from a C-function
    CoefficientWithState(double (*f)(const Vector &, const Vector &))
    {
      Function = f;
    }

    /// Evaluate coefficient
    virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) {
        double x[3];
        Vector transip(x, 3);
        T.Transform(ip, transip);
        return ((*Function)(state_, transip));
    }
    
    void SetState(Vector state) { 
        state_.SetSize(state.Size());
        state_ = state;
    }

private:
   Vector state_;
};


// freq used in definition of psi_function2(omega,x)
#define PI 3.14159265358979323846
double freq = 1.52;
double sigma_t_function(const Vector &x);
double sigma_s_function(const Vector &x);
double psi_function2(const Vector &omega, const Vector &x);
double Q_function2(const Vector &omega, const Vector &x);
double inflow_function2(const Vector &omega, const Vector &x);

struct AIR_parameters {
   double distance;
   std::string prerelax;
   std::string postrelax;
   double strength_tolC;
   double strength_tolR;
   double filter_tolR;
   int interp_type;
   int relax_type;
   double filterA_tol;
   int coarsening;
};


DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, 0, 1),
    m_M_rowptr(NULL), m_M_colinds(NULL), m_M_data(NULL)
{
    m_order = 1;
    m_refLevels = 1;
    m_lumped = false;
}


DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double t0, double t1): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, t0, t1),
    m_M_rowptr(NULL), m_M_colinds(NULL), m_M_data(NULL)
{
    m_order = 1;
    m_refLevels = 1;
    m_lumped = false;
}


DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         int refLevels, int order): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, 0, 1),
    m_M_rowptr(NULL), m_M_colinds(NULL), m_M_data(NULL),
    m_refLevels{refLevels}, m_order{order}
{
    m_lumped = false;
}


DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double t0, double t1, int refLevels, int order): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, t0, t1),
    m_M_rowptr(NULL), m_M_colinds(NULL), m_M_data(NULL),
    m_refLevels{refLevels}, m_order{order}
{
    m_lumped = false;
}


void DGadvection::getSpatialDiscretization(const MPI_Comm &spatialComm, int* &A_rowptr,
                                           int* &A_colinds, double* &A_data, double* &B,
                                           double* &X, int &localMinRow, int &localMaxRow,
                                           int &spatialDOFs, double t)
{
    int meshOrder  = m_order;
    int angle_ind  = g_nAngles / 2;     // Select angle from all angles in "g_quad"
    int basis_type = 1;
    double alpha_mesh = 0.1;
    int blocksize;
    if (g_dim == 2) blocksize = (m_order+1)*(m_order+1);
    else blocksize = (m_order+1)*(m_order+1)*(m_order+1);

    /* Set up a curved mesh and a finite element space */
    std::string mesh_file;
    if (g_dim == 2) {
        mesh_file = "/g/g19/bs/quartz/AIR_tests/data/UnsQuad.0.mesh";
    }
    else {
        mesh_file = "/g/g19/bs/quartz/AIR_tests/data/inline-tet.mesh";
    }
    Mesh mesh(mesh_file, 1, 1);
    dim = mesh.Dimension();
    for (int lev = 0; lev<3; lev++) {
        mesh.UniformRefinement();
    }

    DG_FECollection fec(m_order, g_dim, basis_type);
    FiniteElementSpace fes(&mesh, &fec);
    if (g_dim == 2) {
        bool is_2D = true;
        g_quad.set2DFlag(is_2D);   
    }

    /* Define a parallel mesh by partitioning the serial mesh. */
    ParMesh pmesh(spatialComm, mesh);
    for (int lev = 0; lev<m_refLevels-3; lev++) {
      pmesh.UniformRefinement();
    }
    ParFiniteElementSpace pfes(&pmesh, &fec);

    /* Define angle of flow, coefficients and integrators */
    Vector omega;
    std::vector<double> omega0 = g_quad.getOmega(angle_ind);
    omega.SetSize(omega0.size());
    for (int j=0; j<omega0.size(); ++j) {
        omega(j) = omega0[j];
        // if (myid == 0) std::cout << omega0[j];
    }
    CoefficientWithState inflow_coeff(inflow_function2);
    CoefficientWithState Q_coeff(Q_function2);  
    inflow_coeff.SetState(omega);
    Q_coeff.SetState(omega);
    FunctionCoefficient sigma_t_coeff(sigma_t_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(omega); 

    /* Set up the bilinear form for this angle */
    ParBilinearForm *bl_form = new ParBilinearForm(&pfes);
    bl_form -> AddDomainIntegrator(new MassIntegrator(sigma_t_coeff));
    bl_form -> AddDomainIntegrator(new TransposeIntegrator(
                new ConvectionIntegrator(*direction, -1.0), 0));
    bl_form -> AddInteriorFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));  // Interior face integrators
    bl_form -> AddBdrFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));       // Boundary face integrators
    bl_form -> Assemble();
    bl_form -> Finalize();

    /* Form the right-hand side */
    ParLinearForm *l_form = new ParLinearForm(&pfes);
    l_form -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    l_form -> AddDomainIntegrator(new DomainLFIntegrator(Q_coeff));
    l_form -> Assemble();

    // Assemble bilinear form and corresponding linear system
    int spatialRank;
    MPI_Comm_rank(spatialComm, &spatialRank);
    HypreParMatrix *A = bl_form -> ParallelAssemble();
    HypreParVector *B0 = l_form -> ParallelAssemble();
    Vector X0(pfes.GetVSize());
    X0 = 0.0;

    spatialDOFs = A->GetGlobalNumRows();
    int *rowStarts = A->GetRowStarts();
    localMinRow = rowStarts[0];
    localMaxRow = rowStarts[1]-1;

    // Steal vector data to pointers
    B = B0->StealData();
    X = X0.StealData();

    // Compress diagonal and off-diagonal blocks of hypre matrix to local CSR
    SparseMatrix A_loc;
    A->GetProcRows(A_loc);
    A_rowptr = A_loc.GetI();
    A_colinds = A_loc.GetJ();
    A_data = A_loc.GetData();
    A_loc.LoseData();

    // Mass integrator (lumped) for time integration
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        ParBilinearForm *m = new ParBilinearForm(&pfes);
        if (m_lumped) m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
        else m->AddDomainIntegrator(new MassIntegrator);
        m->Assemble();
        m->Finalize();
        HypreParMatrix *M = m->ParallelAssemble();
        SparseMatrix M_loc;
        M->GetProcRows(M_loc);
        m_M_rowptr = M_loc.GetI();
        m_M_colinds = M_loc.GetJ();
        m_M_data = M_loc.GetData();
        M_loc.LoseData();
        delete M;
        delete m;
    }

    delete A;
    delete B0;
    delete bl_form;
    delete l_form;
    // TODO: debug
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


/* Time-independent spatial discretization of Laplacian */
void DGadvection::getSpatialDiscretization(int* &A_rowptr, int* &A_colinds,
                                           double* &A_data, double* &B, double* &X,
                                           int &spatialDOFs, double t)
{
    int meshOrder  = m_order;
    int angle_ind  = g_nAngles / 2;     // Select angle from all angles in "g_quad"
    int basis_type = 1;
    double alpha_mesh = 0.1;
    int blocksize;
    if (g_dim == 2) blocksize = (m_order+1)*(m_order+1);
    else blocksize = (m_order+1)*(m_order+1)*(m_order+1);    

    /* Set up a curved mesh and a finite element space */
    std::string mesh_file;
    if (g_dim == 2) {
        mesh_file = "/g/g19/bs/quartz/AIR_tests/data/UnsQuad.0.mesh";
    }
    else {
        mesh_file = "/g/g19/bs/quartz/AIR_tests/data/inline-tet.mesh";
    }
    Mesh mesh(mesh_file, 1, 1);
    dim = mesh.Dimension();
    for (int lev = 0; lev<3; lev++) {
        mesh.UniformRefinement();
    }

    DG_FECollection fec(m_order, g_dim, basis_type);
    FiniteElementSpace fes(&mesh, &fec);
    if (g_dim == 2) {
        bool is_2D = true;
        g_quad.set2DFlag(is_2D);   
    }

    /* Define angle of flow, coefficients and integrators */
    Vector omega;
    std::vector<double> omega0 = g_quad.getOmega(angle_ind);
    omega.SetSize(omega0.size());
    for (int j=0; j<omega0.size(); ++j) {
        omega(j) = omega0[j];
        // if (myid == 0) std::cout << omega0[j];
    }
    CoefficientWithState inflow_coeff(inflow_function2);
    CoefficientWithState Q_coeff(Q_function2);  
    inflow_coeff.SetState(omega);
    Q_coeff.SetState(omega);
    FunctionCoefficient sigma_t_coeff(sigma_t_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(omega); 

    /* Set up the bilinear form for this angle */
    BilinearForm *bl_form = new BilinearForm(&fes);
    bl_form -> AddDomainIntegrator(new MassIntegrator(sigma_t_coeff));
    bl_form -> AddDomainIntegrator(new TransposeIntegrator(
                new ConvectionIntegrator(*direction, -1.0), 0));
    bl_form -> AddInteriorFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));  // Interior face integrators
    bl_form -> AddBdrFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));       // Boundary face integrators
    bl_form -> Assemble();
    bl_form -> Finalize();

    /* Form the right-hand side */
    LinearForm *l_form = new LinearForm(&fes);
    l_form -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    l_form -> AddDomainIntegrator(new DomainLFIntegrator(Q_coeff));
    l_form -> Assemble();

    // Change ownership of matrix data to sparse matrix from bilinear form
    SparseMatrix* A = bl_form -> LoseMat();
    spatialDOFs = A->NumRows();
    A_rowptr = A->GetI();
    A_colinds = A->GetJ();
    A_data = A->GetData();
    A->LoseData();
    delete A; 

    // TODO : think I want to steal data from B0, X0, but they do not own
    B = l_form->StealData();
    Vector X0(fes.GetVSize());
    X0 = 0.0;
    X = X0.StealData();

    // Mass integrator for time integration
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        BilinearForm *m = new BilinearForm(&fes);
        if (m_lumped) m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
        else m->AddDomainIntegrator(new MassIntegrator);
        m->Assemble();
        m->Finalize();
        SparseMatrix *M = m->LoseMat();
        m_M_rowptr = M->GetI();
        m_M_colinds = M->GetJ();
        m_M_data = M->GetData();
        M->LoseData();
        delete M;
        delete m;
    }

    delete bl_form;
    delete l_form; 

    // TODO: debug
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


void DGadvection::getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data)
{
    // Check that mass matrix has been constructed
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        std::cout << "WARNING: Mass matrix not integrated.\n";
        return;
    }

    // Direct pointers to mass matrix data arrays
    M_rowptr = m_M_rowptr;
    M_colinds = m_M_colinds; 
    M_data = m_M_data;
}


double psi_function(const Vector &x) {
    if (g_dim == 2) {
        double x1 = x(0);
        double x2 = x(1);
        double psi = .5 * (x1*x1 + x2*x2 + 1.) + std::cos(g_freq*(x1+x2));
        psi = psi * (g_omega_g(0)*g_omega_g(0) + g_omega_g(1));
        return psi;
    }
    else {
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        double psi = .5 * (x1*x1 + x2*x2 + x3*x3 + 1.) + std::cos(g_freq*(x1+x2+x3));
        psi = psi * (g_omega_g(0)*g_omega_g(0) + g_omega_g(1)*g_omega_g(2) + g_omega_g(1));
        return psi;
    }
}


double psi_function2(const Vector &omega, const Vector &x) {
    if (g_dim == 2) {
        double x1 = x(0);
        double x2 = x(1);
        double psi = .5 * (x1*x1 + x2*x2 + 1.) + std::cos(g_freq*(x1+x2));
        psi = psi * (omega(0)*omega(0) + omega(1));
        return psi;
    }
    else {
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        double psi = .5 * (x1*x1 + x2*x2 + x3*x3 + 1.) + std::cos(g_freq*(x1+x2+x3));
        psi = psi * (omega(0)*omega(0) + omega(1)*omega(2) + omega(1));
        return psi;
    }
}


double sigma_s_function(const Vector &x) {
    if (g_dim == 2) {
        return 0.;
    }
    else {
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        return (x1*x2 + x1*x1 + .2);       
    }
}


double sigma_t_function(const Vector &x) {
    if (g_dim == 2) {
        double x1 = x(0);
        double x2 = x(1);
        double val_abs = x1*x2 + x1*x1 + 1.;
        double sig_s = sigma_s_function(x);
        return val_abs + sig_s;
    }
    else {
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        double val_abs = x1*x2 + x1*x1 + 1.;
        double sig_s = sigma_s_function(x);
        return val_abs + sig_s;
    }
}


double phi_function2(const Vector &x)
{
    double phi = 0;
    Vector omega(3);
    for (int angle=0; angle < g_nAngles; ++angle)
    {
        std::vector<double> omega0 = g_quad.getOmega(angle);
        double w = g_quad.getWt(angle);
        omega(0) = omega0[0];
        omega(1) = omega0[1];
        omega(2) = omega0[2];
        phi = phi + w * psi_function2(omega,x);
    }
    return phi; 
}


double Q_function2(const Vector &omega, const Vector &x) {
    if (g_dim == 2) {
        double x1 = x(0);
        double x2 = x(1);
        double sig = sigma_t_function(x);
        double val_sin = g_freq * std::sin(g_freq*(x1+x2));
        double psi_dx_dot_v = omega(0)*(x1-val_sin) + omega(1)*(x2-val_sin);
        psi_dx_dot_v = psi_dx_dot_v * (omega(0)*omega(0) + omega(1));
        double psi = psi_function2(omega, x);
        return psi_dx_dot_v + sig * psi;
    }
    else {
        double pi = 3.1415926535897;
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        double sig = sigma_t_function(x);
        double val_sin = g_freq * std::sin(g_freq*(x1+x2+x3));
        double psi_dx_dot_v = omega(0)*(x1-val_sin) + omega(1)*(x2-val_sin) + omega(2)*(x3-val_sin);
        psi_dx_dot_v = psi_dx_dot_v * (omega(0)*omega(0) + omega(1)*omega(2) + omega(1));
        double psi = psi_function2(omega, x);
        double phi = phi_function2(x);
        double sig_s = sigma_s_function(x);
        return psi_dx_dot_v + sig * psi - sig_s/(4*pi) * phi;
    }
}


double inflow_function2(const Vector &omega, const Vector &x) {
    double psi = psi_function2(omega, x);
    return psi;
}


