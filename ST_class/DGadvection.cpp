#include "DGadvection.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>


Vector mesh_min;
Vector mesh_max;

// Reaction term
double sigma_function(const Vector &x) {
    return 0.0;
    // int dim = x.Size();
    // double scale = 1e-4;
    // if (dim == 2) {
    //     double x1 = x(0);
    //     double x2 = x(1);
    //     double val_abs = x1*x2 + x1*x1 + 1.;
    //     return scale*val_abs;
    // }
    // else {
    //     double x1 = x(0);
    //     double x2 = x(1);
    //     double x3 = x(2);
    //     double val_abs = x1*x2 + x1*x1 + 1.;
    //     return scale*val_abs;
    // }
}

// Initial condition
// - Currently top-hat like area close to x-border and slightly off of y border
double initCond(const Vector &x) {

    int dim = x.Size();
    if (dim == 2) {
        double x0 = x(0);
        double x1 = x(1);
        if (x0 > 0.1 && x0 < 0.5 &&
            x1 > 0.1 && x1 < 0.5) {
            return 200.0;
        }
        else {
            return 0.0;
        }
    }
    else {
        double x0 = x(0);
        double x1 = x(1);
        double x2 = x(2);
        return 0;
    }   
}

// TODO : allow for this to support time-dependent rhs??
// --> double Q_function(const Vector &x, double t) {
// Need to modify CoefficientWithState in .hpp file. 
double Q_function(const Vector &x) {
    int dim = x.Size();
    // if (dim == 2) {
    //     return 0;
    // }
    // else {
    //     return 0;
    // }
    if (dim == 2) {
        double x1 = x(0);
        double x2 = x(1);
        double val_abs = x1*x2 + x1*x1 + 1.;
        return val_abs;
    }
    else {
        double x1 = x(0);
        double x2 = x(1);
        double x3 = x(2);
        double val_abs = x1*x2 + x1*x1 + 1.;
        return val_abs;
    }   
}


// Inflow boundaries
double inflow_function(const Vector &x) {
    int dim = x.Size();
    if (dim == 2) {
        return 0.0;
    }
    else {
        return 0.0;
    }
}

#if 1
DGadvection::DGadvection(MPI_Comm globComm, bool pit, bool M_exists, 
                         int timeDisc, int numTimeSteps, double dt): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt),
    m_order(1), m_refLevels(1), m_lumped(false), m_dim(2),
    m_basis_type(1), m_is_refined(false), m_plform(NULL), m_pbform(NULL),
    m_lform(NULL), m_bform(NULL), m_tcurrent(-1), m_pfes(NULL), m_fes(NULL),
    m_bsize(-1), m_mesh(NULL), m_pmesh(NULL), m_fec(NULL)
{
    m_G_isTimedependent = false;
    m_L_isTimedependent = false;
    Setup();
}
#else
DGadvection::DGadvection(MPI_Comm globComm, bool pit, bool M_exists, 
                         int timeDisc, int numTimeSteps, double dt): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt)
{
    m_order = 1;
    m_refLevels = 1;
    m_lumped = false;
    m_dim = 2;

    m_basis_type = 1;
    m_is_refined = false;
    m_plform = NULL;
    m_pbform = NULL;

    m_lform = NULL;
    m_bform = NULL;
    m_tcurrent = -1;
    m_pfes = NULL;
    m_fes = NULL;

    m_bsize = -1;

m_G_isTimedependent(false), m_L_isTimedependent(false)
    Setup();
}
#endif


DGadvection::DGadvection(MPI_Comm globComm, bool pit, bool M_exists, 
                            int timeDisc, int numTimeSteps,
                            double dt, int refLevels, int order): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt),
    m_refLevels{refLevels}, m_order{order}, m_lumped(false), m_dim(2),
    m_basis_type(1), m_is_refined(false), m_plform(NULL), m_pbform(NULL),
    m_lform(NULL), m_bform(NULL), m_tcurrent(-1), m_pfes(NULL), m_fes(NULL),
    m_bsize(-1), m_mesh(NULL), m_pmesh(NULL), m_fec(NULL)
{
    m_G_isTimedependent = false;
    m_L_isTimedependent = false;
    Setup();
}


DGadvection::DGadvection(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order, bool lumped): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt),
    m_refLevels{refLevels}, m_order{order}, m_lumped{lumped}, m_dim(2),
    m_basis_type(1), m_is_refined(false), m_plform(NULL), m_pbform(NULL),
    m_lform(NULL), m_bform(NULL), m_tcurrent(-1), m_pfes(NULL), m_fes(NULL),
    m_bsize(-1), m_mesh(NULL), m_pmesh(NULL), m_fec(NULL)
{
    m_G_isTimedependent = false;
    m_L_isTimedependent = false;
    Setup();
}


DGadvection::~DGadvection()
{
    ClearForms();
    ClearData();
}


void DGadvection::Setup()
{
    if (m_dim == 2) {
        // m_mesh_file = "/g/g19/bs/quartz/AIR_tests/data/UnsQuad.0.mesh";
        std::string mesh_file = "./meshes/beam-quad.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    else {
        std::string mesh_file = "./meshes/inline-tet.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    m_mesh->GetBoundingBox(mesh_min,mesh_max);
    
 
    m_fec = new DG_FECollection(m_order, m_dim, m_basis_type);

    if (m_useSpatialParallel) {

        // Serial uniform refinement
        for (int lev = 0; lev<std::min(3,m_refLevels); lev++) {
            m_mesh->UniformRefinement();
        }
        // Parallel refinement
        m_pmesh = new ParMesh(m_spatialComm, *m_mesh);
        for (int lev = 0; lev<m_refLevels-3; lev++) {
            m_pmesh->UniformRefinement();
        }
        m_is_refined = true;
        m_pfes = new ParFiniteElementSpace(m_pmesh, m_fec);
    }
    else {
        // Serial uniform refinement
        for (int lev = 0; lev<m_refLevels; lev++) {
            m_mesh->UniformRefinement();
        }
        m_is_refined = true;
        m_fes = new FiniteElementSpace(m_mesh, m_fec);
    }

    // Size of DG element blocks
    if (m_dim == 2) m_bsize = (m_order+1)*(m_order+1);
    else  m_bsize = (m_order+1)*(m_order+1)*(m_order+1);

    // Construct initial matrices at time t=0
    // TODO : t0 --> t=0 is implicitly used in SpaceTimeMatrix, but we should
    // either make that more flexible (initial time t0 != 0) or make it more clear. 
    if (m_useSpatialParallel) ConstructFormsPar(0);
    else ConstructForms(0);
}


void DGadvection::ClearData()
{
    delete m_mesh; m_mesh = NULL;
    delete m_pmesh; m_pmesh = NULL;
    delete m_fec; m_fec = NULL;
}


void DGadvection::ClearForms()
{
    delete m_plform; m_plform = NULL;
    delete m_pbform; m_pbform = NULL;
    delete m_lform; m_lform = NULL;
    delete m_bform; m_bform = NULL;
}

// Construct linear and bilinear forms using spatial parallelism
void DGadvection::ConstructFormsPar(double t)
{
    // TODO: support moving flow -> use FunctionCoefficient instead of VectorConstantCoefficient
    /* Define angle of flow, coefficients and integrators */
    m_omega.SetSize(m_dim);
    if (m_dim == 2) {
        double theta = PI/4.0;
        m_omega(0) = cos(theta);
        m_omega(1) = sin(theta);
    }
    else {
        double theta1 = PI/4.0;
        double theta2 = PI/4.0;
        m_omega(0) = sin(theta1)*cos(theta2);
        m_omega(1) = sin(theta1)*sin(theta2);       
        m_omega(1) = cos(theta1);       
    }

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    if (m_pmesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(m_pmesh->bdr_attributes.Max());
        ess_bdr = 1;
        m_pfes -> GetEssentialTrueDofs(ess_bdr, m_ess_tdof_list);
    }

    // Define functions and coefficients 
    CoefficientWithState inflow_coeff(inflow_function);
    CoefficientWithState Q_coeff(Q_function);
    FunctionCoefficient sigma_coeff(sigma_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 

    // Define solution vector x as finite element grid function corresponding to m_pfes.
    // Initialize x with initial guess of zero, which satisfies the boundary conditions.
    ParGridFunction x(m_pfes);
    x = 0.0;

    /* Set up the bilinear form for this angle */
    m_pbform = new ParBilinearForm(m_pfes);
    m_pbform -> AddDomainIntegrator(new MassIntegrator(sigma_coeff));
    m_pbform -> AddDomainIntegrator(new TransposeIntegrator(
                new ConvectionIntegrator(*direction, -1.0)));
    m_pbform -> AddInteriorFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));  // Interior face integrators
    m_pbform -> AddBdrFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));       // Boundary face integrators
    m_pbform -> Assemble();
    m_pbform -> Finalize();

    /* Form the right-hand side */
    m_plform = new ParLinearForm(m_pfes);
    m_plform -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    m_plform -> AddDomainIntegrator(new DomainLFIntegrator(Q_coeff));
    m_plform -> Assemble();

    // Mass integrator (lumped) for time integration
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        ParBilinearForm *m = new ParBilinearForm(m_pfes);
        if (m_lumped) m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
        else m->AddDomainIntegrator(new MassIntegrator);
        m->Assemble();
        m->Finalize();
        HypreParMatrix *M = m->ParallelAssemble();
        SparseMatrix M_loc;
        M -> GetProcRows(M_loc);
        m_M_rowptr = M_loc.GetI();
        m_M_colinds = M_loc.GetJ();
        m_M_data = M_loc.GetData();
        M_loc.LoseData();
        delete M;
        delete m;
    }

    m_tcurrent = t;
}


// Construct linear and bilinear forms without using spatial parallelism
void DGadvection::ConstructForms(double t)
{
    // TODO: support moving flow -> use FunctionCoefficient instead of VectorConstantCoefficient
    /* Define angle of flow, coefficients and integrators */
    m_omega.SetSize(m_dim);
    if (m_dim == 2) {
        double theta = PI/4.0;
        m_omega(0) = cos(theta);
        m_omega(1) = sin(theta);
    }
    else {
        double theta1 = PI/4.0;
        double theta2 = PI/4.0;
        m_omega(0) = sin(theta1)*cos(theta2);
        m_omega(1) = sin(theta1)*sin(theta2);       
        m_omega(1) = cos(theta1);       
    }

    // Define functions and coefficients 
    CoefficientWithState inflow_coeff(inflow_function);
    CoefficientWithState Q_coeff(Q_function);
    FunctionCoefficient sigma_coeff(sigma_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 

    /* Set up the bilinear form for this angle */
    m_bform = new BilinearForm(m_fes);
    m_bform -> AddDomainIntegrator(new MassIntegrator(sigma_coeff));
    m_bform -> AddDomainIntegrator(new TransposeIntegrator(
                new ConvectionIntegrator(*direction, -1.0), 0));
    m_bform -> AddInteriorFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));  // Interior face integrators
    m_bform -> AddBdrFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));       // Boundary face integrators
    m_bform -> Assemble();

    /* Form the right-hand side */
    m_lform = new LinearForm(m_fes);
    m_lform -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    m_lform -> AddDomainIntegrator(new DomainLFIntegrator(Q_coeff));
    m_lform -> Assemble();

    // Mass integrator for time integration
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        BilinearForm *m = new BilinearForm(m_fes);
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

    m_tcurrent = t;
}



void DGadvection::getSpatialDiscretizationL(int    * &A_rowptr, 
                                           int    * &A_colinds, 
                                           double * &A_data,
                                           double * &U0, 
                                           bool      getU0, 
                                           int      &localMinRow, 
                                           int      &localMaxRow, 
                                           int      &spatialDOFs,
                                           double    t, 
                                           int      &bsize)
{
    if (m_L_isTimedependent && (std::abs(t - m_tcurrent) > 1e-14)) {
        ClearForms();
        ConstructFormsPar(t);
    }

    // Assemble bilinear form and corresponding linear system
    HypreParMatrix* A = m_pbform -> ParallelAssemble();

    spatialDOFs = A -> GetGlobalNumRows();
    int *rowStarts = A -> GetRowStarts();
    localMinRow = rowStarts[0];
    localMaxRow = rowStarts[1]-1;

    // Steal vector data to pointers
    if (getU0) {
        U0 = new double[spatialDOFs]();
    }

    // Compress diagonal and off-diagonal blocks of hypre matrix to local CSR
    SparseMatrix A_loc;
    A -> GetProcRows(A_loc);
    A_rowptr = A_loc.GetI();
    A_colinds = A_loc.GetJ();
    A_data = A_loc.GetData();
    A_loc.LoseData();

    double temp0, temp1;
    m_pmesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);
    bsize = m_bsize;
    delete A;
}

void DGadvection::getSpatialDiscretizationG(double * &G, 
                                            int      &localMinRow, 
                                            int      &localMaxRow,
                                            int      &spatialDOFs, 
                                            double    t)
{
    if (m_G_isTimedependent && (std::abs(t - m_tcurrent) > 1e-14)) {
        ClearForms();
        ConstructFormsPar(t);
    }
    G = m_plform->StealData();
}


/* Time-independent DG spatial discretization of advection */
void DGadvection::getSpatialDiscretizationL(int    * &A_rowptr, 
                                           int    * &A_colinds, 
                                           double * &A_data,
                                           double * &U0, 
                                           bool      getU0, 
                                           int      &spatialDOFs,
                                           double    t, 
                                           int      &bsize)
{
    if (m_L_isTimedependent && (std::abs(t - m_tcurrent) > 1e-14)) {
        ClearForms();
        ConstructForms(t);        
    }

    // Change ownership of matrix data to sparse matrix from bilinear form
    m_bform -> Finalize();
    SparseMatrix* A = m_bform -> LoseMat();
    spatialDOFs = A->NumRows();
    A_rowptr = A->GetI();
    A_colinds = A->GetJ();
    A_data = A->GetData();
    A->LoseData();
    delete A; 

    // TODO : think I want to steal data from B0, X0, but they do not own
    if (getU0) {
        U0 = new double[spatialDOFs]();
    }

    double temp0, temp1;
    m_mesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);
    bsize = m_bsize;
}


void DGadvection::getSpatialDiscretizationG(double* &G, 
                                            int     &spatialDOFs, 
                                            double   t)
{
    if (m_G_isTimedependent && (std::abs(t - m_tcurrent) > 1e-14)) {
        ClearForms();
        ConstructForms(t);        
    }
    G = m_lform->StealData();
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

#if 0
void DGadvection::addInitialCondition(double *B)
{
    // Define functions and coefficients 
    FunctionCoefficient IC(initCond);  

    /* Form the right-hand side */
    FunctionCoefficient inflow_coeff(inflow_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 
    LinearForm *m_lform = new LinearForm(m_fes);
    m_lform -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    m_lform -> AddDomainIntegrator(new DomainLFIntegrator(IC));
    m_lform -> Assemble();

    for (int i=0; i<m_lform->Size(); i++) {
        B[i] = m_lform->Elem(i);
    }
    delete m_lform;
}


void DGadvection::addInitialCondition(double *B)
{
    /* Form the right-hand side */
    FunctionCoefficient inflow_coeff(inflow_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega);
    ParLinearForm *m_lform = new ParLinearForm(m_pfes);
    m_lform -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    m_lform -> AddDomainIntegrator(new DomainLFIntegrator(IC));
    m_lform -> Assemble();

    for (int i=0; i<m_lform->Size(); i++) {
        B[i] = m_lform->Elem(i);
    }
    delete m_lform;
}

#endif

// TODO :  Need to implement these functions... 
void DGadvection::getInitialCondition(double * &B, int &localMinRow,
                                      int &localMaxRow, int &spatialDOFs) 
{
        // Define functions and coefficients 
    FunctionCoefficient IC(initCond);  

    /* Form the right-hand side */
    FunctionCoefficient inflow_coeff(inflow_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega);
    ParLinearForm *plform = new ParLinearForm(m_pfes);
    plform -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    plform -> AddDomainIntegrator(new DomainLFIntegrator(IC));
    plform -> Assemble();

    B = new double[plform->Size()];
    for (int i=0; i<plform->Size(); i++) {
        B[i] = plform->Elem(i);
    }
    delete plform;   
}

void DGadvection::getInitialCondition(double * &B, int &spatialDOFs) 
{    
    // Define functions and coefficients 
    FunctionCoefficient IC(initCond);  

    /* Form the right-hand side */
    FunctionCoefficient inflow_coeff(inflow_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 
    LinearForm *lform = new LinearForm(m_fes);
    lform -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    lform -> AddDomainIntegrator(new DomainLFIntegrator(IC));
    lform -> Assemble();

    B = new double[lform->Size()];
    for (int i=0; i<lform->Size(); i++) {
        B[i] = lform->Elem(i);
    }
    delete lform;
}
