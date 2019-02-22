#include "DGadvection.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace mfem;


Vector mesh_min;
Vector mesh_max;

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

// Return top-hat like area close to x-border and slightly off of y border
double initCond(const Vector &x) {

    int dim = x.Size();
    if (dim == 2) {
        double x0 = x(0);
        double x1 = x(1);
        double s0 = (mesh_max(0) - mesh_min(0));
        double s1 = (mesh_max(1) - mesh_min(1));
        if (x0 > mesh_min(0) && x0 < mesh_max(0)/10.0 &&
            x1 > (mesh_min(1) + s1/10.0) && x1 < (mesh_min(1) + s1/5.0) ) {
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
// double Q_function(const Vector &x, double t) {
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


// Zero inflow boundaries
double inflow_function(const Vector &x) {
    int dim = x.Size();
    if (dim == 2) {
        return 0.0;
    }
    else {
        return 0.0;
    }
}

DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt)
{
    m_order = 1;
    m_refLevels = 1;
    m_lumped = false;
    m_dim = 2;
    m_basis_type = 1;
    m_is_refined = false;

    /* Define angle of flow, coefficients and integrators */
    // TODO: change this to support moving flow, i.e., FunctionCoefficient instead
    //       of VectorConstantCoefficient
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

    if (m_dim == 2) {
        // m_mesh_file = "/g/g19/bs/quartz/AIR_tests/data/UnsQuad.0.mesh";
        std::string mesh_file = "/g/g19/bs/quartz/AIR_tests/data/beam-quad.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    else {
        std::string mesh_file = "/g/g19/bs/quartz/AIR_tests/data/inline-tet.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    m_mesh->GetBoundingBox(mesh_min,mesh_max);
}


DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt),
    m_refLevels{refLevels}, m_order{order}
{
    m_lumped = false;
    m_dim = 2;
    m_basis_type = 1;
    m_is_refined = false;

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

    if (m_dim == 2) {
        // m_mesh_file = "/g/g19/bs/quartz/AIR_tests/data/UnsQuad.0.mesh";
        std::string mesh_file = "/g/g19/bs/quartz/AIR_tests/data/beam-quad.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    else {
        std::string mesh_file = "/g/g19/bs/quartz/AIR_tests/data/inline-tet.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    m_mesh->GetBoundingBox(mesh_min,mesh_max);
}


DGadvection::DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order, bool lumped): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt),
    m_refLevels{refLevels}, m_order{order}, m_lumped{lumped}
{
    m_dim = 2;
    m_basis_type = 1;
    m_is_refined = false;

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

    if (m_dim == 2) {
        // m_mesh_file = "/g/g19/bs/quartz/AIR_tests/data/UnsQuad.0.mesh";
        std::string mesh_file = "/g/g19/bs/quartz/AIR_tests/data/beam-quad.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    else {
        std::string mesh_file = "/g/g19/bs/quartz/AIR_tests/data/inline-tet.mesh";
        m_mesh = new Mesh(mesh_file.c_str(), 1, 1);
    }
    m_mesh->GetBoundingBox(mesh_min,mesh_max);
}


DGadvection::~DGadvection()
{
    delete m_mesh;
}

void DGadvection::getSpatialDiscretization(const MPI_Comm &spatialComm, int* &A_rowptr,
                                           int* &A_colinds, double* &A_data, double* &B,
                                           double* &X, int &localMinRow, int &localMaxRow,
                                           int &spatialDOFs, double t, int &bsize)
{
    int meshOrder  = m_order;
    if (m_dim == 2) {
        bsize = (m_order+1)*(m_order+1);
    }
    else {
        bsize = (m_order+1)*(m_order+1)*(m_order+1);
    }

    /* Set up mesh and a finite element space */
    if (!m_is_refined) {
        for (int lev = 0; lev<std::min(3,m_refLevels); lev++) {
            m_mesh->UniformRefinement();
        }
        m_is_refined = true;
    }

    DG_FECollection fec(m_order, m_dim, m_basis_type);

    /* Define a parallel mesh by partitioning the serial mesh. */
    ParMesh pmesh(spatialComm, *m_mesh);
    for (int lev = 0; lev<m_refLevels-3; lev++) {
      pmesh.UniformRefinement();
    }
    ParFiniteElementSpace pfes(&pmesh, &fec);

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (pmesh.bdr_attributes.Size()) {
        Array<int> ess_bdr(pmesh.bdr_attributes.Max());
        ess_bdr = 1;
        pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Define functions and coefficients 
    FunctionCoefficient inflow_coeff(inflow_function);
    FunctionCoefficient Q_coeff(Q_function);  
    FunctionCoefficient sigma_coeff(sigma_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 

    // Define solution vector x as finite element grid function corresponding to pfes.
    // Initialize x with initial guess of zero, which satisfies the boundary conditions.
    ParGridFunction x(&pfes);
    x = 0.0;

    /* Set up the bilinear form for this angle */
    ParBilinearForm *bl_form = new ParBilinearForm(&pfes);
    bl_form -> AddDomainIntegrator(new MassIntegrator(sigma_coeff));
    bl_form -> AddDomainIntegrator(new TransposeIntegrator(
                new ConvectionIntegrator(*direction, -1.0), 0));
    bl_form -> AddInteriorFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));  // Interior face integrators
    bl_form -> AddBdrFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));       // Boundary face integrators

    /* Form the right-hand side */
    ParLinearForm *l_form = new ParLinearForm(&pfes);
    l_form -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    l_form -> AddDomainIntegrator(new DomainLFIntegrator(Q_coeff));
    l_form -> Assemble();

    // Assemble bilinear form and corresponding linear system
    int spatialRank;
    MPI_Comm_rank(spatialComm, &spatialRank);
    HypreParMatrix A;
    Vector B0;
    Vector X0;
    bl_form -> Assemble();
    bl_form -> Finalize();
    bl_form -> FormLinearSystem(ess_tdof_list, x, *l_form, A, X0, B0);

    spatialDOFs = A.GetGlobalNumRows();
    int *rowStarts = A.GetRowStarts();
    localMinRow = rowStarts[0];
    localMaxRow = rowStarts[1]-1;

    // Steal vector data to pointers
    B = B0.StealData();
    X = X0.StealData();

    // Compress diagonal and off-diagonal blocks of hypre matrix to local CSR
    SparseMatrix A_loc;
    A.GetProcRows(A_loc);
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
        HypreParMatrix M;
        m->FormSystemMatrix(ess_tdof_list, M);
        SparseMatrix M_loc;
        M.GetProcRows(M_loc);
        m_M_rowptr = M_loc.GetI();
        m_M_colinds = M_loc.GetJ();
        m_M_data = M_loc.GetData();
        M_loc.LoseData();
        delete m;
    }

    double temp0, temp1;
    pmesh.GetCharacteristics(m_hmin, m_hmax, temp0, temp1);

    delete bl_form;
    delete l_form;
    // TODO: debug
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


/* Time-independent spatial discretization of Laplacian */
void DGadvection::getSpatialDiscretization(int* &A_rowptr, int* &A_colinds,
                                           double* &A_data, double* &B, double* &X,
                                           int &spatialDOFs, double t, int &bsize)
{
    if (m_dim == 2) {
        bsize = (m_order+1)*(m_order+1);
    }
    else {
        bsize = (m_order+1)*(m_order+1)*(m_order+1);    
    }

    /* Set up mesh and a finite element space */
    if (!m_is_refined) {
        for (int lev = 0; lev<m_refLevels; lev++) {
            m_mesh->UniformRefinement();
        }
        m_is_refined = true;
    }

    DG_FECollection fec(m_order, m_dim, m_basis_type);
    FiniteElementSpace fes(m_mesh, &fec);

    // Define functions and coefficients 
    FunctionCoefficient inflow_coeff(inflow_function);
    FunctionCoefficient Q_coeff(Q_function);  
    FunctionCoefficient sigma_coeff(sigma_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 

    /* Set up the bilinear form for this angle */
    BilinearForm *bl_form = new BilinearForm(&fes);
    bl_form -> AddDomainIntegrator(new MassIntegrator(sigma_coeff));
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

    double temp0, temp1;
    m_mesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);

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


void DGadvection::addInitialCondition(double *B)
{
    /* Set up mesh and a finite element space */
    if (!m_is_refined) {
        for (int lev = 0; lev<m_refLevels; lev++) {
            m_mesh->UniformRefinement();
        }
        m_is_refined = true;
    }
    DG_FECollection fec(m_order, m_dim, m_basis_type);
    FiniteElementSpace fes(m_mesh, &fec);

    // Define functions and coefficients 
    FunctionCoefficient IC(initCond);  

    /* Form the right-hand side */
    FunctionCoefficient inflow_coeff(inflow_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega); 
    LinearForm *l_form = new LinearForm(&fes);
    l_form -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    l_form -> AddDomainIntegrator(new DomainLFIntegrator(IC));
    l_form -> Assemble();

    for (int i=0; i<l_form->Size(); i++) {
        B[i] = l_form->Elem(i);
    }
    delete l_form;
}


void DGadvection::addInitialCondition(const MPI_Comm &spatialComm, double *B)
{
    /* Set up mesh and a finite element space */
    if (!m_is_refined) {
        for (int lev = 0; lev<m_refLevels; lev++) {
            m_mesh->UniformRefinement();
        }
        m_is_refined = true;
    }
    DG_FECollection fec(m_order, m_dim, m_basis_type);

    /* Define a parallel mesh by partitioning the serial mesh. */
    ParMesh pmesh(spatialComm, *m_mesh);
    for (int lev = 0; lev<m_refLevels-3; lev++) {
      pmesh.UniformRefinement();
    }
    ParFiniteElementSpace pfes(&pmesh, &fec);

    // Define functions and coefficients 
    FunctionCoefficient IC(initCond);  

    /* Form the right-hand side */
    FunctionCoefficient inflow_coeff(inflow_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(m_omega);
    ParLinearForm *l_form = new ParLinearForm(&pfes);
    l_form -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    l_form -> AddDomainIntegrator(new DomainLFIntegrator(IC));
    l_form -> Assemble();

    for (int i=0; i<l_form->Size(); i++) {
        B[i] = l_form->Elem(i);
    }
    delete l_form;
}


