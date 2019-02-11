// #include <math.h>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
// #include "vis.c"
#include "FD_temp.hpp"

// Not sure if I have to initialize all of these to NULL. Do not do ParMatrix
// and ParVector because I think these are objects within StructMatrix/Vector. 
// 2d constructor
FD_temp::FD_temp(MPI_Comm comm, int nt, int nx, int Pt, int Px) :
    m_comm{comm}, m_nt_local{nt}, m_nx_local{nx}, m_Pt{Pt}, m_Px{Px},
    m_dim(2), m_rebuildSolver{false},
    m_solver(NULL), m_gmres(NULL), m_bS(NULL), m_xS(NULL), m_AS(NULL)
{
    // Get number of processes
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProc);

    if ((m_Px*m_Pt) != m_numProc) {
        if (m_rank == 0) {
            std::cout << "Error: Invalid number of processors or processor topology \n";
        }
        throw std::domain_error("Px*Pt != P");
    }

    // Compute local indices in 2d processor array, m_px_ind and
    // m_pt_ind, from m_Px, m_Pt, and m_rank
    m_n = m_nx_local * m_nt_local;
    m_px_ind = m_rank % m_Px;
    m_pt_ind = (m_rank - m_px_ind) / m_Px;

    // TODO : should be over (m_globx + 1)?? Example had this
    m_t0 = 0.0;
    m_t1 = 1.0;
    m_x0 = 0.0;
    m_x1 = 1.0;
    m_globt = m_nt_local * m_Pt;
    m_dt = (m_t1 - m_t0) / m_globt;
    m_globx = m_nx_local * m_Px;
    m_hx = (m_x1 - m_x0) / m_globx;
    m_hy = -1;

    // Define each processor's piece of the grid in space-time. Ordered by
    // time, then space
    m_ilower.resize(2);
    m_iupper.resize(2);
    m_ilower[0] = m_pt_ind * m_nt_local;
    m_iupper[0] = m_ilower[0] + m_nt_local-1;
    m_ilower[1] = m_px_ind * m_nx_local;
    m_iupper[1] = m_ilower[1] + m_nx_local-1;
		
		std::cout << "Here...";
}

// First-order upwind discretization of the 1d-space, 1d-time homogeneous
// wave equation. Initial conditions (t = 0) are passed in function pointers
// IC_u(double x) and IC_v(double x), which give the ICs for u and v at point
// (0,x). 
void FD_temp::Wave1D(double (*IC_u)(double),
                         double (*IC_v)(double), 
                         double c)
{
    if (m_dim != 2) {
        if (m_rank == 0) {
            std::cout << "Error: class dimension is " << m_dim <<
                    ". This discretization requires dim=2.\n";
       }
       return;
    }

    if (m_rank == 0) {
        std::cout << "  1d Space-time wave equation:\n" <<
           "    (nx, nt) = (" << m_globx << ", " << m_globt << ")\n" << 
           "    (Px, Pt) = (" << m_Px << ", " << m_Pt << ")\n";
    }

    // Create an empty 2D grid object with 1 part
    HYPRE_SStructGridCreate(m_comm, m_dim, 1, &m_grid);

    // Add this processor's box to the grid (on part 0)
    HYPRE_SStructGridSetExtents(m_grid, 0, &m_ilower[0], &m_iupper[0]);

    // Define two variables on grid (on part 0)
    HYPRE_SStructVariable vartypes[2] = {HYPRE_SSTRUCT_VARIABLE_CELL,
                            HYPRE_SSTRUCT_VARIABLE_CELL };
    HYPRE_SStructGridSetVariables(m_grid, 0, m_dim, vartypes);

    // Finalize grid assembly.
    HYPRE_SStructGridAssemble(m_grid);

    /* ------------------------------------------------------------------
    *                   Define discretization stencils
    * ---------------------------------------------------------------- */
    // Stencil object for variable u (labeled as variable 0). First entry
    // is diagonal, next three are u at previous time, final three are v
    // at previous time
    HYPRE_SStructStencil  m_stencil_u;
    HYPRE_SStructStencil  m_stencil_v;

    std::vector<int> uu_indices{0, 1, 2, 3};
    std::vector<int> uv_indices{4, 5, 6};

    int n_uu_stenc = uu_indices.size();
    int n_uv_stenc = uv_indices.size();
    int stencil_size_u = n_uu_stenc + n_uv_stenc;

    std::vector<std::vector<int>> offsets_u = {{0,0}, {-1,-1}, {-1,0}, {-1,1},
                                                {-1,-1}, {-1,0}, {-1,1}};
    HYPRE_SStructStencilCreate(m_dim, stencil_size_u, &m_stencil_u);

    // Data for stencil
    double lambda = c * m_dt / m_hx;
    std::vector<double> u_data = {1.0, lambda*lambda/2.0, 1-lambda*lambda,
                                  lambda*lambda/2.0, lambda*m_dt/4.0, 
                                  m_dt*(2.0-lambda)/2.0, lambda*m_dt/4.0 };
    std::vector<double> v_data = {1.0, lambda/2.0, 1.0-lambda, lambda/2.0,
                                  lambda*lambda/m_dt, -2.0*lambda*lambda/m_dt,
                                  lambda*lambda/m_dt};

    // Set stencil for u-u connections (variable 0)
    for (auto entry : uu_indices) {
        HYPRE_SStructStencilSetEntry(m_stencil_u, entry, &offsets_u[entry][0], 0);
    }

    // Set stencil for u-v connections (variable 1)
    for (auto entry : uv_indices) {
        HYPRE_SStructStencilSetEntry(m_stencil_u, entry, &offsets_u[entry][0], 1);
    }

    // Set u-stencil entries (to be added to matrix later). Note that
    // HYPRE_SStructMatrixSetBoxValues can only set values corresponding
    // to stencil entries for one variable at a time
    int n_uu = n_uu_stenc * m_n;
    double* uu_values = new double[n_uu];
    for (int i=0; i<n_uu; i+=n_uu_stenc) {
        for (int j=0; j<n_uu_stenc; j++) {
            uu_values[i+j] = u_data[j];
        }
    }

    // Fill in stencil for u-v entries here (to be added to matrix later)
    int n_uv = n_uv_stenc * m_n;
    double* uv_values = new double[n_uv];
    for (int i=0; i<n_uv; i+=n_uv_stenc) {
        for (int j=n_uu_stenc; j<stencil_size_u; j++) {
            uv_values[i+j] = u_data[j];
        }
    }

    // Stencil object for variable v (labeled as variable 1).
    std::vector<int> vv_indices{0, 1, 2, 3};
    std::vector<int> vu_indices{4, 5, 6};

    int n_vv_stenc = vv_indices.size();
    int n_vu_stenc = vu_indices.size();
    int stencil_size_v = n_vv_stenc + n_vu_stenc;

    std::vector<std::vector<int> > offsets_v = {{0,0}, {-1,-1}, {-1,0}, {-1,1},
                                                {-1,-1}, {-1,0}, {-1,1}};
    HYPRE_SStructStencilCreate(m_dim, stencil_size_v, &m_stencil_v);

    // Set stencil for v-v connections (variable 1)
    for (auto entry : vv_indices) {
        std::cout << entry << ", ";
        HYPRE_SStructStencilSetEntry(m_stencil_v, entry, &offsets_v[entry][0], 1);
    }

    // Set stencil for v-u connections (variable 0)
    for (auto entry : vu_indices) {
        std::cout << entry << ", ";
        HYPRE_SStructStencilSetEntry(m_stencil_v, entry, &offsets_v[entry][0], 0);
    }

    // Set u-stencil entries (to be added to matrix later). Note that
    // HYPRE_SStructMatrixSetBoxValues can only set values corresponding
    // to stencil entries for one variable at a time
    int n_vv = n_vv_stenc * m_n;
    double* vv_values = new double[n_vv];
    for (int i=0; i<n_vv; i+=n_vv_stenc) {
        for (int j=0; j<n_vv_stenc; j++) {
            vv_values[i+j] = v_data[j];
        }
    }

    // Fill in stencil for u-v entries here (to be added to matrix later)
    int n_vu = n_vu_stenc * m_n;
    double* vu_values = new double[n_vu];
    for (int i=0; i<n_vu; i+=n_vu_stenc) {
        for (int j=n_vv_stenc; j<n_vu_stenc; j++) {
            vu_values[i+j] = v_data[j];
        }
    }

    /* ------------------------------------------------------------------
    *                      Fill in sparse matrix
    * ---------------------------------------------------------------- */


    // ----------------------------------------------------------------
    // TODO : memory problems here. These lines compile if I declare
    // graph here. No idea why this is different than as a clas variable??

    // Set up graph for problem (determines non-zero structure of matrix)
    HYPRE_SStructGraph    graph;

    HYPRE_SStructGraphCreate(m_comm, m_grid, &graph);
    HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);

    // ----------------------------------------------------------------

    HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);


    // Assign the u-stencil to variable u (variable 0), and the v-stencil
    // variable v (variable 1), both on part 0 of the m_grid
    HYPRE_SStructGraphSetStencil(graph, 0, 0, m_stencil_u);
    HYPRE_SStructGraphSetStencil(graph, 0, 1, m_stencil_v);

    // Assemble the graph
    HYPRE_SStructGraphAssemble(graph);

//#if 0
    // Create an empty matrix object
    HYPRE_SStructMatrixCreate(m_comm, graph, &m_AS);
    HYPRE_SStructMatrixSetObjectType(m_AS, HYPRE_PARCSR);
    HYPRE_SStructMatrixInitialize(m_AS);

    // // Set values in matrix for part 0 and variables 0 (u) and 1 (v)
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
                                    0, n_uu_stenc, &uu_indices[0], uu_values);
    delete[] uu_values;
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
                                    0, n_uv_stenc, &uv_indices[0], uv_values);
    delete[] uv_values;
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
                                    1, n_vv_stenc, &vv_indices[0], vv_values);
    delete[] vv_values;
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
                                    1, n_vu_stenc, &vu_indices[0], vu_values);
    delete[] vu_values;

    /* ------------------------------------------------------------------
    *                      Add boundary conditions
    * ---------------------------------------------------------------- */
    std::vector<int> periodic(m_dim,0);
    periodic[1] = m_globx;
    if ( (m_px_ind == 0) || (m_px_ind == (m_Px-1)) ) {
        HYPRE_SStructGridSetPeriodic(m_grid, 0, &periodic[0]);    
    }
    // TODO : do I set periodic on *all* processors, or only boundary processors??
		
    /* ------------------------------------------------------------------
    *                      Construct linear system
    * ---------------------------------------------------------------- */
    // Finalize matrix assembly
    HYPRE_SStructMatrixAssemble(m_AS);
		
    // Create an empty vector object
    HYPRE_SStructVectorCreate(m_comm, m_grid, &m_bS);
    HYPRE_SStructVectorCreate(m_comm, m_grid, &m_xS);
		
    // Set vectors to be par csr type
    HYPRE_SStructVectorSetObjectType(m_bS, HYPRE_PARCSR);
    HYPRE_SStructVectorSetObjectType(m_xS, HYPRE_PARCSR);
		
    // Indicate that vector coefficients are ready to be set
    HYPRE_SStructVectorInitialize(m_bS);
    HYPRE_SStructVectorInitialize(m_xS);
		
    // Set right hand side and inital guess. RHS is nonzero only at time t=0
    // because we are solving homogeneous wave equation. Because scheme is
    // explicit, set solution equal to rhs there because first t rows are
    // diagonal. Otherwise, rhs = 0 and we use 0 initial guess.
    //      TODO : should we be using zero initial guess for x0?
    std::vector<double> rhs(m_n, 0);
    if (m_pt_ind == 0) {
        for (int i=0; i<m_nx_local; i++) {
            double temp_x = (m_px_ind*m_nx_local + i) * m_hx;
            rhs[i] = IC_u(temp_x);
        }
    }
    HYPRE_SStructVectorSetBoxValues(m_bS, 0, &m_ilower[0], &m_iupper[0], 0, &rhs[0]);
    HYPRE_SStructVectorSetBoxValues(m_xS, 0, &m_ilower[0], &m_iupper[0], 0, &rhs[0]);
		
    if (m_pt_ind == 0) {
        for (int i=0; i<m_nx_local; i++) {
            double temp_x = (m_px_ind*m_nx_local + i) * m_hx;
            rhs[i] = IC_v(temp_x);
        }
    }
    HYPRE_SStructVectorSetBoxValues(m_bS, 0, &m_ilower[0], &m_iupper[0], 1, &rhs[0]);
    HYPRE_SStructVectorSetBoxValues(m_xS, 0, &m_ilower[0], &m_iupper[0], 1, &rhs[0]);
		
    // Finalize vector assembly
    HYPRE_SStructVectorAssemble(m_bS);
    HYPRE_SStructVectorAssemble(m_xS);
		
    // Get objects for sparse matrix and vectors.
    HYPRE_SStructMatrixGetObject(m_AS, (void **) &m_A);
    HYPRE_SStructVectorGetObject(m_bS, (void **) &m_b);
    HYPRE_SStructVectorGetObject(m_xS, (void **) &m_x);
//#endif
}
