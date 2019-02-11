#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <mpi.h>
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_krylov.h"

double IC_u(double x)
{
    double temp;
    if ((x + 0.25) >= 0) {
        temp += 1;
    }
    if ((x - 0.25) >= 0) {
        temp += 1;
    }
    return temp;
}

double IC_v(double x)
{
    return 0.0;
}

int main(int argc, char *argv[])
{
    // Initialize parallel
    int m_rank, m_numProc;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_numProc);

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

    double c = 1.0;

    // Compute local indices in 2d processor array, m_px_ind and
    // m_pt_ind, from m_Px, m_Pt, and m_rank
    int m_Pt = 1;
    int m_Px = 1;
    int m_nt_local = 6;
    int m_nx_local = 6;
    int m_dim = 2;   
    int m_n = m_nx_local * m_nt_local;
    int m_px_ind = m_rank % m_Px;
    int m_pt_ind = (m_rank - m_px_ind) / m_Px;

    // TODO : should be over (m_globx + 1)?? Example had this
    int m_t0 = 0.0;
    int m_t1 = 1.0;
    int m_x0 = 0.0;
    int m_x1 = 1.0;
    int m_globt = m_nt_local * m_Pt;
    double m_dt = (m_t1 - m_t0) / m_globt;
    long m_globx = m_nx_local * m_Px;
    double m_hx = (m_x1 - m_x0) / m_globx;

    // Define each processor's piece of the grid in space-time. Ordered by
    // time, then space
    // std::vector<int> m_ilower(2);
    // std::vector<int> m_iupper(2);
    int m_ilower[2];
    int m_iupper[2];
    m_ilower[0] = m_pt_ind * m_nt_local;
    m_iupper[0] = m_ilower[0] + m_nt_local-1;
    m_ilower[1] = m_px_ind * m_nx_local;
    m_iupper[1] = m_ilower[1] + m_nx_local-1;

    /* ------------------------------------------------------------------
    *                            Build grid
    * ---------------------------------------------------------------- */

    // Create an empty 2D grid object with 1 part
    HYPRE_SStructGridCreate(MPI_COMM_WORLD, m_dim, 1, &m_grid);

    // Add this processor's box to the grid (on part 0)
    // HYPRE_SStructGridSetExtents(m_grid, 0, &m_ilower[0], &m_iupper[0]);
    HYPRE_SStructGridSetExtents(m_grid, 0, m_ilower, m_iupper);

    // Define two variables on grid (on part 0)
    HYPRE_SStructVariable vartypes[2] = {HYPRE_SSTRUCT_VARIABLE_CELL,
                            HYPRE_SSTRUCT_VARIABLE_CELL };
    HYPRE_SStructGridSetVariables(m_grid, 0, 2, vartypes);

    // Finalize grid assembly.
    HYPRE_SStructGridAssemble(m_grid);

    /* ------------------------------------------------------------------
    *                   Define discretization stencils
    * ---------------------------------------------------------------- */
    // Stencil object for variable u (labeled as variable 0). First entry
    // is diagonal, next three are u at previous time, final three are v
    // at previous time
    // std::vector<int> uu_indices{0, 1, 2, 3};
    // std::vector<int> uv_indices{4, 5, 6};
    // int n_uu_stenc = uu_indices.size();
    // int n_uv_stenc = uv_indices.size();

    int uu_indices[4] = {0, 1, 2, 3};
    int uv_indices[3] = {4, 5, 6};
    int n_uu_stenc = 4;
    int n_uv_stenc = 3;

    int stencil_size_u = n_uu_stenc + n_uv_stenc;

    // std::vector<std::vector<int>> offsets_u = {{0,0}, {-1,-1}, {-1,0}, {-1,1},
    //                                             {-1,-1}, {-1,0}, {-1,1}};

    int offsets_u[7][2] = {{0,0}, {-1,-1}, {-1,0}, {-1,1}, {-1,-1}, {-1,0}, {-1,1}};

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
        // HYPRE_SStructStencilSetEntry(m_stencil_u, entry, &offsets_u[entry][0], 0);
        HYPRE_SStructStencilSetEntry(m_stencil_u, entry, offsets_u[entry], 0);
    }

    // Set stencil for u-v connections (variable 1)
    for (auto entry : uv_indices) {
        // HYPRE_SStructStencilSetEntry(m_stencil_u, entry, &offsets_u[entry][0], 1);
        HYPRE_SStructStencilSetEntry(m_stencil_u, entry, offsets_u[entry], 1);
    }

    // Set u-stencil entries (to be added to matrix later). Note that
    // HYPRE_SStructMatrixSetBoxValues can only set values corresponding
    // to stencil entries for one variable at a time
    int n_uu = n_uu_stenc * m_n;
    // double* uu_values = new double[n_uu];
    std::vector<double> uu_values(n_uu);
    for (int i=0; i<n_uu; i+=n_uu_stenc) {
        for (int j=0; j<n_uu_stenc; j++) {
            uu_values[i+j] = u_data[j];
        }
    }

    // Fill in stencil for u-v entries here (to be added to matrix later)
    int n_uv = n_uv_stenc * m_n;
    // double* uv_values = new double[n_uv];
    std::vector<double> uv_values(n_uv);
    for (int i=0; i<n_uv; i+=n_uv_stenc) {
        // for (int j=n_uu_stenc; j<stencil_size_u; j++) {
        for (int j=0; j<n_uv_stenc; j++) {
            uv_values[i+j] = u_data[j+n_uu_stenc];
        }
    }

    // Stencil object for variable v (labeled as variable 1).
    // std::vector<int> vv_indices{0, 1, 2, 3};
    // std::vector<int> vu_indices{4, 5, 6};
    // int n_vv_stenc = vv_indices.size();
    // int n_vu_stenc = vu_indices.size();

    int vv_indices[4] = {0, 1, 2, 3};
    int vu_indices[3] = {4, 5, 6};
    int n_vv_stenc = 4;
    int n_vu_stenc = 3;

    int stencil_size_v = n_vv_stenc + n_vu_stenc;

    // std::vector<std::vector<int> > offsets_v = {{0,0}, {-1,-1}, {-1,0}, {-1,1},
    //                                             {-1,-1}, {-1,0}, {-1,1}};
 
    int offsets_v[7][2] = {{0,0}, {-1,-1}, {-1,0}, {-1,1}, {-1,-1}, {-1,0}, {-1,1}};

    HYPRE_SStructStencilCreate(m_dim, stencil_size_v, &m_stencil_v);

    // Set stencil for v-v connections (variable 1)
    for (auto entry : vv_indices) {
        std::cout << entry << ", ";
        // HYPRE_SStructStencilSetEntry(m_stencil_v, entry, &offsets_v[entry][0], 1);
        HYPRE_SStructStencilSetEntry(m_stencil_v, entry, offsets_v[entry], 1);
    }

    // Set stencil for v-u connections (variable 0)
    for (auto entry : vu_indices) {
        std::cout << entry << ", ";
        // HYPRE_SStructStencilSetEntry(m_stencil_v, entry, &offsets_v[entry][0], 0);
        HYPRE_SStructStencilSetEntry(m_stencil_v, entry, offsets_v[entry], 0);
    }

    // Set u-stencil entries (to be added to matrix later). Note that
    // HYPRE_SStructMatrixSetBoxValues can only set values corresponding
    // to stencil entries for one variable at a time
    int n_vv = n_vv_stenc * m_n;
    // double* vv_values = new double[n_vv];
    std::vector<double> vv_values(n_vv);
    for (int i=0; i<n_vv; i+=n_vv_stenc) {
        for (int j=0; j<n_vv_stenc; j++) {
            vv_values[i+j] = v_data[j];
        }
    }

    // Fill in stencil for u-v entries here (to be added to matrix later)
    int n_vu = n_vu_stenc * m_n;
    // double* vu_values = new double[n_vu];
    std::vector<double> vu_values(n_vu);
    for (int i=0; i<n_vu; i+=n_vu_stenc) {
        // for (int j=n_vv_stenc; j<n_vu_stenc; j++) {
        for (int j=0; j<n_vu_stenc; j++) {
            vu_values[i+j] = v_data[n_vv_stenc+j];
        }
    }

    /* ------------------------------------------------------------------
    *                      Fill in sparse matrix
    * ---------------------------------------------------------------- */

    HYPRE_SStructGraphCreate(MPI_COMM_WORLD, m_grid, &m_graph);
    HYPRE_SStructGraphSetObjectType(m_graph, HYPRE_PARCSR);
    HYPRE_SStructGraphSetObjectType(m_graph, HYPRE_PARCSR);

    // Assign the u-stencil to variable u (variable 0), and the v-stencil
    // variable v (variable 1), both on part 0 of the m_grid
    HYPRE_SStructGraphSetStencil(m_graph, 0, 0, m_stencil_u);
    HYPRE_SStructGraphSetStencil(m_graph, 0, 1, m_stencil_v);

    // Assemble the m_graph
    HYPRE_SStructGraphAssemble(m_graph);

    // Create an empty matrix object
    HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, m_graph, &m_AS);
    HYPRE_SStructMatrixSetObjectType(m_AS, HYPRE_PARCSR);
    HYPRE_SStructMatrixInitialize(m_AS);
#if 1

    // // Set values in matrix for part 0 and variables 0 (u) and 1 (v)
    // HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
    //                                 0, n_uu_stenc, &uu_indices[0], uu_values);
    // delete[] uu_values;
    // HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
    //                                 0, n_uv_stenc, &uv_indices[0], uv_values);
    // delete[] uv_values;
    // HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
    //                                 1, n_vv_stenc, &vv_indices[0], vv_values);
    // delete[] vv_values;
    // HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0],
    //                                 1, n_vu_stenc, &vu_indices[0], vu_values);
    // delete[] vu_values;


    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, m_ilower, m_iupper,
                                    0, n_uu_stenc, uu_indices, &uu_values[0]);
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, m_ilower, m_iupper,
                                    0, n_uv_stenc, uv_indices, &uv_values[0]);
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, m_ilower, m_iupper,
                                    1, n_vv_stenc, vv_indices, &vv_values[0]);
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, m_ilower, m_iupper,
                                    1, n_vu_stenc, vu_indices, &vu_values[0]);

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
    HYPRE_SStructVectorCreate(MPI_COMM_WORLD, m_grid, &m_bS);
    HYPRE_SStructVectorCreate(MPI_COMM_WORLD, m_grid, &m_xS);
        
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

#endif 

    // Clean things up
    HYPRE_SStructGridDestroy(m_grid);
    HYPRE_SStructStencilDestroy(m_stencil_v);
    HYPRE_SStructStencilDestroy(m_stencil_u);
    // HYPRE_SStructGraphDestroy(m_graph);
    // HYPRE_SStructMatrixDestroy(m_AS);     // This destroys parCSR matrix too
    // HYPRE_SStructVectorDestroy(m_bS);       // This destroys parVector too
    // HYPRE_SStructVectorDestroy(m_xS);

    MPI_Finalize();
    return 0;
}