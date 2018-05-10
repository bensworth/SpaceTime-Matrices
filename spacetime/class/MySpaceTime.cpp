#include "MySpaceTime.hpp"
#include "mfem.hpp"
using namespace mfem;



MySpaceTime::MySpaceTime(MPI_Comm globComm, int timeDisc, int numTimeSteps): 
	SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, 0, 1)
{
    m_order = 1;
    m_refLevels = 1;
}


MySpaceTime::MySpaceTime(MPI_Comm globComm, int timeDisc, int numTimeSteps,
						 double t0, double t1): 
	SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, t0, t1)
{
    m_order = 1;
    m_refLevels = 1;
}


MySpaceTime::MySpaceTime(MPI_Comm globComm, int timeDisc, int numTimeSteps,
						 int refLevels, int order): 
	SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, 0, 1),
	m_refLevels{refLevels}, m_order{order}
{

}


MySpaceTime::MySpaceTime(MPI_Comm globComm, int timeDisc, int numTimeSteps,
						 double t0, double t1, int refLevels, int order): 
	SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, t0, t1),
	m_refLevels{refLevels}, m_order{order}
{

}


void MySpaceTime::getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
										   int *&A_colinds, double *&A_data, double *&B,
										   double *&X, int &localMinRow, int &localMaxRow,
										   int &spatialDOFs, double t)
{
    // Read mesh from mesh file
    const char *mesh_file = "../../meshes/beam-quad.mesh";
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine mesh
    // double max_elements = 50000;
    // int m_refLevels = (int)floor(log(max_elements./mesh->GetNE())/log(2.)/dim);
    int ser_m_refLevels = std::min(3,m_refLevels);
    for (int l=0; l<ser_m_refLevels; l++) {
        mesh->UniformRefinement();
    }

    // Define parallel mesh by a partitioning of the serial mesh.
    ParMesh *pmesh = new ParMesh(spatialComm, *mesh);
    delete mesh;
    int par_m_refLevels = m_refLevels - 3;
    for (int l = 0; l < par_m_refLevels; l++) {
        pmesh->UniformRefinement();
    }

    // Define finite element space on mesh
    FiniteElementCollection *fec;
    fec = new H1_FECollection(m_order, dim);
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (pmesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear form b(.) 
    ParLinearForm *b = new ParLinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Define solution vector x as finite element grid function corresponding to fespace.
    // Initialize x with initial guess of zero, which satisfies the boundary conditions.
    ParGridFunction x(fespace);
    x = 0.0;

    // Set up bilinear form a(.,.) on finite element space corresponding to Laplacian
    // operator -Delta, by adding the Diffusion domain integrator.
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // Assemble bilinear form and corresponding linear system
    int spatialRank;
    MPI_Comm_rank(spatialComm, &spatialRank);
    HypreParMatrix A;
    Vector B0;
    Vector X0;
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X0, B0);

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

    delete a;
    delete b;
 
    if (fec) {
      delete fespace;
      delete fec;
    }
    delete pmesh;

    // TODO: debug
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}


/* Time-independent spatial discretization of Laplacian */
void MySpaceTime::getSpatialDiscretization(int *&A_rowptr, int *&A_colinds,
										   double *&A_data, double *&B, double *&X,
										   int &spatialDOFs, double t)
{
    // Read mesh from mesh file
    const char *mesh_file = "../../meshes/beam-quad.mesh";
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Refine mesh
    // double max_elements = 50000;
    // int m_refLevels = (int)floor(log(max_elements./mesh->GetNE())/log(2.)/dim);
    for (int l=0; l<m_refLevels; l++) {
        mesh->UniformRefinement();
    }

    // Define finite element space on mesh
    FiniteElementCollection *fec;
    fec = new H1_FECollection(m_order, dim);
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear form b(.) 
    LinearForm *b = new LinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Define solution vector x as finite element grid function corresponding to fespace.
    // Initialize x with initial guess of zero, which satisfies the boundary conditions.
    GridFunction x(fespace);
    x = 0.0;

    // Set up bilinear form a(.,.) on finite element space corresponding to Laplacian
    // operator -Delta, by adding the Diffusion domain integrator.
    BilinearForm *a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // Mass integrator (lumped) for time integration
    // BilinearForm *m = new BilinearForm(fespace);
    // m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
    // m->Assemble();
    // m->Finalize();
    // HypreParMatrix *M = m -> ParallelAssemble();

    // TODO : SCALE BY MASS INVERSE

    // Assemble bilinear form and corresponding linear system
    Vector B0;
    Vector X0;
    SparseMatrix A, M;
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X0, B0, 0);
    // m->FormSystemMatrix(ess_tdof_list, M);
    // SparseMatrix M = m->SpMat();

    // Change ownership of matrix data to sparse matrix from bilinear form
    spatialDOFs = A.NumRows();
    a->LoseMat();
    A_rowptr = A.GetI();
    A_colinds = A.GetJ();
    A_data = A.GetData();
    A.LoseData();

    // TODO : think I want to steal data from B0, X0, but they do not own
    B = b->StealData();
    X = x.StealData();

    delete a;
//    delete m;
    delete b; 
    if (fec) {
      delete fespace;
      delete fec;
    }
    delete mesh;

    // TODO: debug
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}
