#include "CGdiffusion.hpp"
#include "mfem.hpp"
#include <iostream>
using namespace mfem;


// TODO : make sure this is discretizing -\Delta u = f


// TODO :  Need to implement these functions... 
void CGdiffusion::getInitialCondition(const MPI_Comm &spatialComm, double * &B, int &localMinRow, int &localMaxRow, int &spatialDOFs) {    
    
    if(!m_par_fespace)
        std::cerr<<"Initial conditions were requested before setting up mesh info (par ver)"<<std::endl;

    ParGridFunction *u = new ParGridFunction( m_par_fespace );
    FunctionCoefficient u0( []( const Vector& x ){return 0.;} );        //TODO: use a more meaningful IC
    u->ProjectCoefficient(u0);
    HypreParVector *U = u->GetTrueDofs();


    spatialDOFs = U->GlobalSize();

    int* rows = U->Partitioning();
    localMinRow = rows[0];
    localMaxRow = rows[1] - 1;

    B = new double[ rows[1] - rows[0] ];                  //TODO: separating memory allocation and destruction seems like terrible practice
    for ( int i = 0; i < rows[1]-rows[0]; ++i ){          // but that's how it's handled in SpaceTimeMatrix.cpp ;_;
        B[i] = U->GetData()[i];    
    }

    delete u;

}


void CGdiffusion::getInitialCondition(double * &B, int &spatialDOFs) {    
    if(!m_fespace)
        std::cerr<<"Initial conditions were requested before setting up mesh info (ser ver)"<<std::endl;

    spatialDOFs = m_fespace->GetNDofs();

    GridFunction *u = new GridFunction( m_fespace );
    FunctionCoefficient u0( []( const Vector& x ){return 0.;} );        //TODO: use a more meaningful IC

    u->ProjectCoefficient(u0);

    B = new double[spatialDOFs];                                        //TODO: separating memory allocation and destruction seems like terrible practice
    for ( int i = 0; i < spatialDOFs; ++i ){                            // but that's how it's handled in SpaceTimeMatrix.cpp ;_;
        B[i] = u->GetData()[i];    
    }

    delete u;

}







CGdiffusion::CGdiffusion(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                         double dt): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt),
    m_fec(NULL), m_mesh(NULL), m_fespace(NULL), m_par_mesh(NULL), m_par_fespace(NULL){

    m_order = 1;
    m_refLevels = 1;
    m_lumped = false;
    m_L_isTimedependent = false;
    m_G_isTimedependent = false;

    if( !m_useSpatialParallel ){
        initialiseFEinfo();
        initialiseMassMatrix();
    }
    else{
        initialiseParFEinfo(m_spatialComm);
        initialiseParMassMatrix();
    }
}


CGdiffusion::CGdiffusion(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt),
    m_refLevels{refLevels}, m_order{order},
    m_fec(NULL), m_mesh(NULL), m_fespace(NULL), m_par_mesh(NULL), m_par_fespace(NULL){

    m_lumped = false;
    m_L_isTimedependent = false;
    m_G_isTimedependent = false;

    if( !m_useSpatialParallel ){
        initialiseFEinfo();
        initialiseMassMatrix();
    }
    else{
        initialiseParFEinfo(m_spatialComm);
        initialiseParMassMatrix();
    }
}

CGdiffusion::CGdiffusion(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order, bool lumped): 
    SpaceTimeMatrix(globComm, pit, M_exists, timeDisc, numTimeSteps, dt),
    m_refLevels{refLevels}, m_order{order}, m_lumped(lumped),
    m_fec(NULL), m_mesh(NULL), m_fespace(NULL), m_par_mesh(NULL), m_par_fespace(NULL){

    m_L_isTimedependent = false;
    m_G_isTimedependent = false;

    if( !m_useSpatialParallel ){
        initialiseFEinfo();
        initialiseMassMatrix();
    }
    else{
        initialiseParFEinfo(m_spatialComm);
        initialiseParMassMatrix();
    }
}



CGdiffusion::~CGdiffusion() {
    if(m_mesh) delete m_mesh;
    if(m_fespace) delete m_fespace;

    if(m_par_mesh) delete m_par_mesh;
    if(m_par_fespace) delete m_par_fespace;

    if(m_fec) delete m_fec;
};






// Assemble mass matrix - space serial version
void CGdiffusion::initialiseMassMatrix( ){
    
    // initialise variational form and assemble mass matrix
    Array<int> essTDOF;
    if ( m_mesh->bdr_attributes.Size() ) {
        Array<int> ess_bdr( m_mesh->bdr_attributes.Max() );
        ess_bdr = 1;
        m_fespace->GetEssentialTrueDofs( ess_bdr, essTDOF );
    }

    BilinearForm *mVarf( new BilinearForm(m_fespace) );
    if (m_lumped){ 
        mVarf->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
    }else{
        mVarf->AddDomainIntegrator(new MassIntegrator);
    }
    mVarf->Assemble();
    mVarf->Finalize();

    SparseMatrix M;
    mVarf->FormSystemMatrix( essTDOF, M );

    // store
    m_M_rowptr  = M.GetI();
    m_M_colinds = M.GetJ();
    m_M_data    = M.GetData();
    mVarf->LoseMat();
    M.LoseData();



    // // store mass matrix in hypre format   
    // HYPRE_IJMatrixCreate( MPI_COMM_SELF, 0, M.NumRows()-1, 0, M.numCols()-1, &m_Mij );

    // Array<int> nnzPerRow( M.NumRows() );     // num of non-zero els per row
    // int  *offIdxs = M.GetI();                // contains offsets for data in J for each row
    // for ( int i = 0; i < M.NumRows(); ++i ){
    //     nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
    // }
    // Array<int> rowsIdx( M.numRows() );
    // for ( int i = 0; i < M.numRows(); ++i ){
    //     rowsIdx[i] = i;
    // }
    // HYPRE_IJMatrixSetValues( m_Mij, M.NumRows(), nnzPerRow.GetData(),
    //                          rowsIdx.GetData(), M.GetJ(), M.GetData() );     // setvalues *copies* the data
    
    // HYPRE_IJMatrixAssemble( m_Mij );
    // HYPRE_IJMatrixGetObject( m_Mij, (void **) &m_M);

    // m_M_localMinRow = 0;
    // m_M_localMaxRow = M.NumROws()-1;


    // // This should be handled by base class
    // // if(m_lumped){
    // //     HYPRE_IJMatrixCreate( MPI_COMM_SELF, 0, M.NumRows()-1, 0, M.numCols()-1, &m_invMij );
    // //
    // //     // bit of a hack: manually invert values
    // //     for ( int i = 0; i < M.NumNonZeroElems(); ++i ){
    // //         ( M.GetData() )[i] = 1./( M.GetData() )[i];
    // //     }
    // //     HYPRE_IJMatrixSetValues( m_invMij, M.NumRows(), nnzPerRow.GetData(),
    // //                              rowsIdx.GetData(), M.GetJ(), M.GetData() );        
    // //     HYPRE_IJMatrixAssemble( m_invMij );
    // //     HYPRE_IJMatrixGetObject( m_invMij, (void **) &m_invM );

    // // }


    // - once the matrix is generated, we can get rid of the operator
    delete mVarf;
}



// Assemble mass matrix - space parallel version
void CGdiffusion::initialiseParMassMatrix( ){

    // initialise variational form and assemble mass matrix
    Array<int> essTDOF;
    if ( m_par_mesh->bdr_attributes.Size() ) {
        Array<int> ess_bdr( m_par_mesh->bdr_attributes.Max() );
        ess_bdr = 1;
        m_par_fespace->GetEssentialTrueDofs( ess_bdr, essTDOF );
    }

    ParBilinearForm *mVarf = new ParBilinearForm(m_par_fespace);
    if (m_lumped){ 
        mVarf->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
    }else{
        mVarf->AddDomainIntegrator(new MassIntegrator);
    }
    mVarf->Assemble();
    mVarf->Finalize();
    HypreParMatrix M;


    mVarf->FormSystemMatrix( essTDOF, M );

    int *rowStarts = M.GetRowStarts();

    //TODO: Now I need a way to extract from HypreParMatrix the info pertaining single processors

    // ATTEMPT 1: use built-in function GetProcRows()
    // TODO: this doesn't seem to work well with the GetJ() method of SparseMatrix
    // SparseMatrix M_loc;
    // M.GetProcRows(M_loc);
    // m_M_rowptr  = M_loc.GetI();
    // m_M_colinds = M_loc.GetJ();
    // m_M_data    = M_loc.GetData();

    // M_loc.LoseData();




    // ATTEMPT 2: Extract underlying hypre matrix, and use hypre-specific functions
    // TODO: This returns garbage: conversion between HYPRE_IJMatrix and HYPRE_ParCSRMatrix is not as simple
    // Array<int> nnzPerRow( rowStarts[1] - rowStarts[0] ); 
    // HYPRE_IJMatrixGetValues( HYPRE_IJMatrix( HYPRE_ParCSRMatrix( M ) ), rowStarts[1] - rowStarts[0], nnzPerRow.GetData(), m_M_rowptr, m_M_colinds, m_M_data );



    // ATTEMPT 3: Forcefully disassemble and re-assemble matrix
    // TODO: How does the mapping for the columns of offd work?
    SparseMatrix diag, offd;
    M.GetDiag(diag);
    int* cmap;
    M.GetOffd(offd,cmap);

    m_M_rowptr  = new int[ rowStarts[1] - rowStarts[0] + 1 ];
    m_M_colinds = new int[ diag.NumNonZeroElems() + offd.NumNonZeroElems() ];
    m_M_data    = new double[ diag.NumNonZeroElems() + offd.NumNonZeroElems() ];

    m_M_rowptr[0] = diag.GetI()[0];       // = 0...
    // TODO: re-check this code once you've figured out what to do with cmap
    for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
        int jInDiag = diag.GetI()[i+1] - diag.GetI()[i];
        int jInOffd = offd.GetI()[i+1] - offd.GetI()[i];
        m_M_rowptr[i+1] = m_M_rowptr[i] + jInDiag + jInOffd;           //Need to collect info on both:
        for ( int j = 0; j < jInDiag; ++j ){                           // diagonal ...
            m_M_data[    m_M_rowptr[i] + j ] = diag.GetData()[ diag.GetI()[i] + j];
            m_M_colinds[ m_M_rowptr[i] + j ] = diag.GetJ()[    diag.GetI()[i] + j];
        }
        for ( int j = 0; j < jInOffd; ++j ){                           // ...and off-diagonal parts
            m_M_data[    m_M_rowptr[i] + jInDiag + j ] =       offd.GetData()[ offd.GetI()[i] + j];
            m_M_colinds[ m_M_rowptr[i] + jInDiag + j ] = cmap[ offd.GetJ()[    offd.GetI()[i] + j] ];       // TODO: triple check this
        }
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank==0)
        std::cout<<"WARNING: Make sure you're including off-diagonal info of mass matrix, correctly!"<<std::endl;


    // This is for debugging
    // {   int uga;
    //     int rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     if (rank==0){
    //         // std::cout<<"Diagonal Matrix";
    //         // for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
    //         //     std::cout<<std::endl<<"Row "<<i<<" - Cols: ";
    //         //     for ( int j = m_M_rowptr[i]; j < m_M_rowptr[i+1]; ++j ){
    //         //         std::cout<<m_M_colinds[j]<<": "<<m_M_data[j]<<" - ";
    //         //     }
    //         // }
    //         // std::cout<<"Off-Diagonal Matrix";
    //         // for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
    //         //     std::cout<<std::endl<<"Row "<<i<<" - Cols: ";
    //         //     for ( int j = offd.GetI()[i]; j < offd.GetI()[i+1]; ++j ){
    //         //         std::cout<<offd.GetJ()[j]<<": "<<offd.GetData()[j]<<" - ";
    //         //     }
    //         // }
    //         std::cout<<"Full Matrix?"<<std::endl;
    //         for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
    //             std::cout<<std::endl<<"Row "<<i<<" - Cols: ";
    //             for ( int j = m_M_rowptr[i]; j < m_M_rowptr[i+1]; ++j ){
    //                 std::cout<<m_M_colinds[j]<<": ";
    //                 std::cout<<m_M_data[j]<<" - ";
    //             }
    //         }            
    //     }
    //     std::cin>>uga;
    //     MPI_Barrier( MPI_COMM_WORLD );}



    // This is handled elsewhere...?
    // // store mass matrix in hypre format   
    // // TODO: this re-assembles the whole thing. Can't I just steal it from M?
    // HYPRE_IJMatrixCreate( M.GetComm(), (M.GetRowStarts())[0], ((M.GetRowStarts())[1])-1,
    //                                    (M.GetColStarts())[0], ((M.GetColStarts())[1])-1, &m_Mij );

    // Array<int> nnzPerRow( (M.GetRowStarts())[1] - (M.GetRowStarts())[0] );     // num of non-zero els per row
    // int  *offIdxs = M_loc.GetI();                // contains offsets for data in J for each row
    // for ( int i = 0; i < nnzPerRow.Size(); ++i ){
    //     nnzPerRow[i] = offIdxs[i+1] - offIdxs[i];
    // }
    // Array<int> rowsIdx( (M.GetRowStarts())[1] - (M.GetRowStarts())[0] );
    // for ( int i = (M.GetRowStarts())[0]; i < (M.GetRowStarts())[1]; ++i ){
    //     rowsIdx[i] = i;
    // }
    // HYPRE_IJMatrixSetValues( m_Mij, (M.GetRowStarts())[1] - (M.GetRowStarts())[0], nnzPerRow.GetData(),
    //                          rowsIdx.GetData(), M_loc.GetJ(), M_loc.GetData() );     // setvalues *copies* the data
    
    // HYPRE_IJMatrixAssemble( m_Mij );
    // HYPRE_IJMatrixGetObject( m_Mij, (void **) &m_M);

    // m_M_localMinRow = (M.GetRowStarts())[0];
    // m_M_localMaxRow = (M.GetRowStarts())[1]-1;




    // // This should be handled by base class
    // // if(m_lumped){
    // //     HYPRE_IJMatrixCreate( M.GetComm(), (M.GetRowStarts())[0], ((M.GetRowStarts())[1])-1,
    // //                                        (M.GetColStarts())[0], ((M.GetColStarts())[1])-1, &m_invMij );
    // //     // bit of a hack: manually invert values
    // //     for ( int i = 0; i < M_loc.NumNonZeroElems(); ++i ){
    // //         ( M_loc.GetData() )[i] = 1./( M_loc.GetData() )[i];
    // //     }

    // //     HYPRE_IJMatrixSetValues( m_invMij, (M.GetRowStarts())[1] - (M.GetRowStarts())[0], nnzPerRow.GetData(),
    // //                              rowsIdx.GetData(), M_loc.GetJ(), M_loc.GetData() );     // setvalues *copies* the data
        
    // //     HYPRE_IJMatrixAssemble( m_invMij );
    // //     HYPRE_IJMatrixGetObject( m_invMij, (void **) &m_invM );

    // // }



    // - once the matrix is generated, we can get rid of the operator
    delete mVarf;
}










// Handy function to initialise info on mesh and fespace - space parallel version
void CGdiffusion::initialiseParFEinfo( const MPI_Comm &spatialComm ){
    // Does the distinction between space par and not make sense? isn't the non-space-par version
    // treated as with spatialComm = a single proc?

    // Read mesh from mesh file
    const char *mesh_file = "./meshes/beam-quad.mesh";
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
    m_par_mesh = new ParMesh(spatialComm, *mesh);
    delete mesh;
    int par_m_refLevels = m_refLevels - 3;
    for (int l = 0; l < par_m_refLevels; l++) {
        m_par_mesh->UniformRefinement();
    }

    double temp0, temp1;
    m_par_mesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);

   

    // Define finite element space on mesh
    m_fec = new H1_FECollection(m_order,dim);       //TODO: I don't think I need m_fec
    m_par_fespace = new ParFiniteElementSpace(m_par_mesh, m_fec);

}



// Handy function to initialise info on mesh and fespace - space serial version
void CGdiffusion::initialiseFEinfo(){
    // Does the distinction between space par and not make sense? isn't the non-space-par version
    // treated as with spatialComm = a single proc?

    // Read mesh from mesh file
    const char *mesh_file = "./meshes/beam-quad.mesh";
    m_mesh = new Mesh(mesh_file, 1, 1);
    int dim = m_mesh->Dimension();

    // Refine mesh
    for (int l=0; l<m_refLevels; l++) {
        m_mesh->UniformRefinement();
    }

    double temp0, temp1;
    m_mesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);

    // Define finite element space on mesh
    m_fec = new H1_FECollection(m_order,dim);       //TODO: I don't think I need m_fec
    m_fespace = new FiniteElementSpace(m_mesh, m_fec);

}












// This should assemble the rhs? parallel version
void CGdiffusion::getSpatialDiscretizationG(const MPI_Comm &spatialComm, double* &G, 
                                            int &localMinRow, int &localMaxRow, int &spatialDOFs, double t){
    // initialise mesh and FEspace info if not done before
    if(!m_par_mesh){
        initialiseFEinfo();
    }


    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (m_par_mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(m_par_mesh->bdr_attributes.Max());
        ess_bdr = 1;
        m_par_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear form b(.) - for now the rhs is just == 1
    ParLinearForm *b = new ParLinearForm(m_par_fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();
    HypreParVector *B = b->ParallelAssemble();

    spatialDOFs = B->GlobalSize();
    int *rowStarts = B->Partitioning();
    localMinRow = rowStarts[0];
    localMaxRow = rowStarts[1]-1;

    // Copy vector data to pointers
    G = new double[ rowStarts[1] - rowStarts[0] ];             //TODO: separating memory allocation and destruction seems like terrible practice
    for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){   // but that's how it's handled in SpaceTimeMatrix.cpp ;_;
        G[i] = B->GetData()[i];   
    }


    delete b;
}                             





// This should assemble the rhs? non-parallel version
void CGdiffusion::getSpatialDiscretizationG(double* &G, int &spatialDOFs, double t){

    // initialise mesh and FEspace info if not done before
    if(!m_mesh){
        initialiseFEinfo();
    }

    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (m_mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(m_mesh->bdr_attributes.Max());
        ess_bdr = 1;
        m_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up linear form b(.) - for now the rhs is just == 1
    LinearForm *b = new LinearForm(m_fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // Change ownership of matrix data to sparse matrix from bilinear form
    spatialDOFs = b->Size();

    // Copy vector data to pointers
    G = new double[ spatialDOFs ];                     //TODO: separating memory allocation and destruction seems like terrible practice
    for ( int i = 0; i < spatialDOFs; ++i ){           // but that's how it's handled in SpaceTimeMatrix.cpp ;_;
        G[i] = b->GetData()[i];    
    }

    delete b;

}








// This should assemble the spatial system? parallel version
void CGdiffusion::getSpatialDiscretizationL(const MPI_Comm &spatialComm, int* &L_rowptr, 
                                            int* &L_colinds, double* &L_data,
                                            double* &U0, bool getU0, 
                                            int &localMinRow, int &localMaxRow, int &spatialDOFs,
                                            double t, int &bsize){
    //TODO: What's bsize for?

    if(!m_par_mesh){
        initialiseParFEinfo( spatialComm );
    }


    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (m_par_mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(m_par_mesh->bdr_attributes.Max());
        ess_bdr = 1;
        m_par_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up bilinear form a(.,.) on finite element space corresponding to Laplacian
    // operator -Delta, by adding the Diffusion domain integrator.
    ConstantCoefficient one(1.0);
    ParBilinearForm *a = new ParBilinearForm(m_par_fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // Assemble bilinear form and corresponding linear system
    a->Assemble();
    a->Finalize();
    HypreParMatrix A;
    a->FormSystemMatrix(ess_tdof_list, A);

    spatialDOFs = A.GetGlobalNumRows();
    int *rowStarts = A.GetRowStarts();
    localMinRow = rowStarts[0];
    localMaxRow = rowStarts[1]-1;


    //TODO: Now I need a way to extract from HypreParMatrix the info pertaining single processors

    // ATTEMPT 1: use built-in function GetProcRows()
    // TODO: this doesn't seem to work well with the GetJ() method of SparseMatrix
    // SparseMatrix A_loc;
    // A.GetProcRows(A_loc);
    // L_rowptr  = A_loc.GetI();
    // L_colinds = A_loc.GetJ();
    // L_data    = A_loc.GetData();

    // A_loc.LoseData();




    // ATTEMPT 2: Extract underlying hypre matrix, and use hypre-specific functions
    // TODO: This returns garbage: conversion between HYPRE_IJMatrix and HYPRE_ParCSRMatrix is not as simple
    // Array<int> nnzPerRow( rowStarts[1] - rowStarts[0] ); 
    // HYPRE_IJMatrixGetValues( HYPRE_IJMatrix( HYPRE_ParCSRMatrix( A ) ), rowStarts[1] - rowStarts[0], nnzPerRow.GetData(), L_rowptr, L_colinds, L_data );



    // ATTEMPT 3: Forcefully disassemble and re-assemble matrix
    // TODO: How does the mapping for the columns of offd work?
    SparseMatrix diag, offd;
    A.GetDiag(diag);
    int* cmap;
    A.GetOffd(offd,cmap);

    L_rowptr  = new int[ rowStarts[1] - rowStarts[0] + 1 ];
    L_colinds = new int[ diag.NumNonZeroElems() + offd.NumNonZeroElems() ];
    L_data    = new double[ diag.NumNonZeroElems() + offd.NumNonZeroElems() ];

    L_rowptr[0] = diag.GetI()[0];       // = 0...
    // TODO: re-check this code once you've figured out what to do with cmap
    for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
        int jInDiag = diag.GetI()[i+1] - diag.GetI()[i];
        int jInOffd = offd.GetI()[i+1] - offd.GetI()[i];
        L_rowptr[i+1] = L_rowptr[i] + jInDiag + jInOffd;               //Need to collect info on both:
        for ( int j = 0; j < jInDiag; ++j ){                           // diagonal ...
            L_data[    L_rowptr[i] + j ] = diag.GetData()[ diag.GetI()[i] + j];
            L_colinds[ L_rowptr[i] + j ] = diag.GetJ()[    diag.GetI()[i] + j];
        }
        for ( int j = 0; j < jInOffd; ++j ){                           // ...and off-diagonal parts
            L_data[    L_rowptr[i] + jInDiag + j ] =       offd.GetData()[ offd.GetI()[i] + j];
            L_colinds[ L_rowptr[i] + jInDiag + j ] = cmap[ offd.GetJ()[    offd.GetI()[i] + j] ];       // TODO: triple check this
        }
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank==0)
        std::cout<<"WARNING: Make sure you're including off-diagonal info of spatial operator, correctly!"<<std::endl;


    // This is for debugging...
    // {   int uga;
    //     int rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     if (rank==0){
    //         // std::cout<<"Diagonal Matrix";
    //         // for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
    //         //     std::cout<<std::endl<<"Row "<<i<<" - Cols: ";
    //         //     for ( int j = L_rowptr[i]; j < L_rowptr[i+1]; ++j ){
    //         //         std::cout<<L_colinds[j]<<": "<<L_data[j]<<" - ";
    //         //     }
    //         // }
    //         // std::cout<<"Off-Diagonal Matrix";
    //         // for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
    //         //     std::cout<<std::endl<<"Row "<<i<<" - Cols: ";
    //         //     for ( int j = offd.GetI()[i]; j < offd.GetI()[i+1]; ++j ){
    //         //         std::cout<<offd.GetJ()[j]<<": "<<offd.GetData()[j]<<" - ";
    //         //     }
    //         // }
    //         std::cout<<"Full Matrix?"<<std::endl;
    //         for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){
    //             std::cout<<std::endl<<"Row "<<i<<" - Cols: ";
    //             for ( int j = L_rowptr[i]; j < L_rowptr[i+1]; ++j ){
    //                 std::cout<<L_colinds[j]<<": ";
    //                 std::cout<<L_data[j]<<" - ";
    //             }
    //         }            
    //     }
    //     std::cin>>uga;
    //     MPI_Barrier( MPI_COMM_WORLD );}




    // // This is handled elsewhere
    // // Mass integrator (lumped) for time integration
    // if( !m_M )
    // if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
    //     ParBilinearForm *m = new ParBilinearForm(m_par_fespace);
    //     if (m_lumped) m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
    //     else m->AddDomainIntegrator(new MassIntegrator);
    //     m->Assemble();
    //     m->Finalize();
    //     HypreParMatrix M;
    //     m->FormSystemMatrix(ess_tdof_list, M);
    //     SparseMatrix M_loc;
    //     M.GetProcRows(M_loc);
    //     m_M_rowptr = M_loc.GetI();
    //     m_M_colinds = M_loc.GetJ();
    //     m_M_data = M_loc.GetData();
    //     M_loc.LoseData();
    //     delete m;
    // }


    //I reckon u0 gives an initial guess for the solution of the system:
    // taking an all-0 vec seems reasonable
    if (getU0){
        U0 = new double[ rowStarts[1] - rowStarts[0] ];             // TODO: separating memory allocation and destruction seems like terrible practice
        for ( int i = 0; i < rowStarts[1] - rowStarts[0]; ++i ){    //  but that's how it's handled in SpaceTimeMatrix.cpp ;_;
            U0[i] = 0.0;                                            // TODO: come up with something more meaningful?
        }
    }

    // clean up stuff
    delete a;


}





// This should assemble the spatial system? non-parallel version
void CGdiffusion::getSpatialDiscretizationL(int* &L_rowptr, int* &L_colinds, double* &L_data,
                                            double* &U0, bool getU0, int &spatialDOFs,
                                            double t, int &bsize){
    //TODO: What's bsize for?

    // initialise mesh and FEspace info if not done before
    if(!m_mesh){
        initialiseFEinfo();
    }


    // Determine list of true (i.e. conforming) essential boundary dofs.
    // In this example, the boundary conditions are defined by marking all
    // the boundary attributes from the mesh as essential (Dirichlet) and
    // converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (m_mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(m_mesh->bdr_attributes.Max());
        ess_bdr = 1;
        m_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Set up bilinear form a(.,.) on finite element space corresponding to Laplacian
    // operator -Delta, by adding the Diffusion domain integrator.
    ConstantCoefficient one(1.0);
    BilinearForm *a = new BilinearForm(m_fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // Assemble bilinear form and corresponding linear system
    SparseMatrix A;
    a->Assemble();
    a->FormSystemMatrix(ess_tdof_list, A);

    // Change ownership of matrix data to sparse matrix from bilinear form
    spatialDOFs = A.NumRows();
    a->LoseMat();
    L_rowptr = A.GetI();
    L_colinds = A.GetJ();
    L_data = A.GetData();
    A.LoseData();

    // // Mass integrator (lumped) for time integration
    // if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
    //     BilinearForm *m = new BilinearForm(m_fespace);
    //     if (m_lumped) m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
    //     else m->AddDomainIntegrator(new MassIntegrator);
    //     m->Assemble();
    //     m->Finalize();
    //     SparseMatrix M;
    //     m->FormSystemMatrix(ess_tdof_list, M);
    //     m->LoseMat();
    //     m_M_rowptr = M.GetI();
    //     m_M_colinds = M.GetJ();
    //     m_M_data = M.GetData();
    //     M.LoseData();
    //     delete m;
    // }



    //I reckon u0 gives an initial guess for the solution of the system:
    // taking an all-0 vec seems reasonable
    if (getU0){
        U0 = new double[spatialDOFs];                  // TODO: separating memory allocation and destruction seems like terrible practice
        for ( int i = 0; i < spatialDOFs; ++i ){       //  but that's how it's handled in SpaceTimeMatrix.cpp ;_;
            U0[i] = 0.;                                // TODO: come up with something more meaningful?
        }
    }


    delete a;
}






void CGdiffusion::getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data)
{
    // Check that mass matrix has been constructed
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        if( !m_useSpatialParallel ){
            initialiseMassMatrix();
        }else{
            initialiseParMassMatrix();       
        }
    }

    // Direct pointers to mass matrix data arrays
    M_rowptr  = m_M_rowptr;
    M_colinds = m_M_colinds; 
    M_data    = m_M_data;
}























/* Old code that shouldnt be necessary anymore

void CGdiffusion::getSpatialDiscretization(int* &A_rowptr, int* &A_colinds,
                                           double* &A_data, double* &B, double* &X,
                                           int &spatialDOFs, double t, int &bsize)
{
    // Read mesh from mesh file
    const char *mesh_file = "./meshes/beam-quad.mesh";
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

    // Assemble bilinear form and corresponding linear system
    Vector B0;
    Vector X0;
    SparseMatrix A;
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X0, B0, 0);

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

    // Mass integrator (lumped) for time integration
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        BilinearForm *m = new BilinearForm(fespace);
        if (m_lumped) m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
        else m->AddDomainIntegrator(new MassIntegrator);
        m->Assemble();
        m->Finalize();
        SparseMatrix M;
        m->FormSystemMatrix(ess_tdof_list, M);
        m->LoseMat();
        m_M_rowptr = M.GetI();
        m_M_colinds = M.GetJ();
        m_M_data = M.GetData();
        M.LoseData();
        delete m;
    }

    double temp0, temp1;
    mesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);

    delete a;
    delete b; 
    if (fec) {
      delete fespace;
      delete fec;
    }
    delete mesh;

    // TODO: debug
    // A *= (1.0+t);       // Scale by t to distinguish system at different times for verification
}



void CGdiffusion::getSpatialDiscretization(const MPI_Comm &spatialComm, int* &A_rowptr,
                                           int* &A_colinds, double* &A_data, double* &B,
                                           double* &X, int &localMinRow, int &localMaxRow,
                                           int &spatialDOFs, double t, int &bsize)
{
    // Read mesh from mesh file
    const char *mesh_file = "./meshes/beam-quad.mesh";
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
    a->Finalize();
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

    // Mass integrator (lumped) for time integration
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        ParBilinearForm *m = new ParBilinearForm(fespace);
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
    pmesh->GetCharacteristics(m_hmin, m_hmax, temp0, temp1);

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
//*/