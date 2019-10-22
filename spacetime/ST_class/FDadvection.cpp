#include "FDadvection.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

// TODO: 
// Parallel in 2D...

// 
double FDadvection::InitCond(double x) 
{        
    if (m_problemID == 1) {
        return pow(cos(PI * x), 4.0);
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        return cos(PI * x);
    } else {
        return 0.0;
    }
}

double FDadvection::InitCond(double x, double y) 
{        
    if (m_problemID == 1) {
        return pow(cos(PI * x), 4.0) * pow(cos(PI * y), 4.0);
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        return cos(PI * x) * cos(PI * y);
    } else {
        return 0.0;
    }
}


// Wave speed for 1D problem
double FDadvection::WaveSpeed(double x, double t) {
    if (m_problemID == 1) {
        return 1.0;
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        return  cos(PI*(x-t)) * exp(-(1+cos(t)));
    }  else  {
        return 0.0;
    }
}


// Wave speed for 2D problem; need to choose component as 1 or 2.
double FDadvection::WaveSpeed(double x, double y, double t, int component) {
    if (m_problemID == 1) {
        if (component == 1) {
            return 1.0;
        } else {
            return 1.0;
        }
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        if (component == 1) {
            return  cos(PI*(x-t)) * sin(PI*(y-t)) * exp(-(1+cos(t)));
        } else {
            return  sin(PI*(x-t)) * cos(PI*(y-t)) * exp(-(1+cos(t)));
        }
    } else {
        return 0.0;
    }
}



// Map grid index to grid point in specified dimension
double FDadvection::MeshIndToPoint(int meshInd, int dim)
{
    return m_boundary0[dim] + m_dx[dim] * meshInd;
    //return -1.0 + xInd * m_dx; // Assuming x \in [-1,1]
}


// RHS of PDE 
double FDadvection::PDE_Source(double x, double t)
{
    if (m_problemID == 1) {
        return 0.0;
    } else if (m_problemID == 2) {
        return ( -exp(1 + cos(t)) * ( cos(PI*(t-x))*sin(t) + PI*sin(PI*(t-x)) ) + PI*sin(2*PI*(t-x)) )/pow(exp(1), 2.0);
    } else if (m_problemID == 3) {
        return ( -exp(1 + cos(t)) * ( cos(PI*(t-x))*sin(t) + PI*sin(PI*(t-x)) ) + 0.5*PI*sin(2*PI*(t-x)) )/pow(exp(1), 2.0);
    } else {
        return 0.0;
    }
}

// RHS of PDE 
double FDadvection::PDE_Source(double x, double y, double t)
{
    if (m_problemID == 1) {
        return 0.0;
    // } else if (m_problemID == 2) {
    //     return // TODO
    // } else if (m_problemID == 3) {
    //     //return // TODO
    } else {
        return 0.0;
    }
}



FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt)
{
    
}


FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, int dim, int refLevels, int order, int problemID): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt),
    m_dim{dim}, m_refLevels{refLevels}, m_problemID{problemID}
{
    
    // Can generalize this if you like to pass in distinct order and nDOFs in y-direction. This by default just makes them the same as in the x-direction
    
    double nx = pow(2, refLevels+2);
    double dx = 2.0 / nx; 
    double xboundary0 = -1.0; // Assume x \in [-1,1].
    
    if (dim >= 1) {
        m_nx.push_back(nx);
        m_dx.push_back(dx);
        m_boundary0.push_back(xboundary0);
        m_order.push_back(order);
    }
    
    // Just make domain in y-direction the same as in x-direction
    if (dim >= 2) {
        m_nx.push_back(nx);
        m_dx.push_back(dx);
        m_boundary0.push_back(xboundary0);
        m_order.push_back(order);
    }
    
    
    if ((m_problemID == 1) || (m_problemID == 2)) {
        m_conservativeForm = true; 
    } else if (m_problemID == 3) {
        m_conservativeForm = false; 
    } else {
        m_conservativeForm = true;
    }
        
    // Need to initialize these to NULL so we can distinguish them from once they have been built
    m_M_rowptr  = NULL;
    m_M_colinds = NULL;
    m_M_data    = NULL;  
}


FDadvection::~FDadvection()
{
    
}


// Get local CSR structure of FD spatial discretization matrix, L
// Can be run serially by passing spatialComm == NULL 
void FDadvection::getSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&B,
                                              double *&X, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) 
{
    // Unpack variables frequently used
    int nx          = m_nx[0];
    double dx       = m_dx[0];
    int xFD_Order   = m_order[0];
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    
    int spatialRank;
    int spatialCommSize;
    int onProcSize;
    
    // Spatial communicator is NULL: Code is serial, so entire discretization is put on single process
    if (!spatialComm) {
        spatialRank = 0;
        spatialCommSize = 1;
        onProcSize = nx;
        
    // Spatial communicator exists so ensure proccessor distribution makes sense
    } else {
        MPI_Comm_rank(spatialComm, &spatialRank);    
        MPI_Comm_size(spatialComm, &spatialCommSize);        
        if ( (nx % spatialCommSize) != 0 ) {
            if (spatialRank == 0) {
                std::cout << "Error: Number of spatial DOFs (" << nx << ") does not divide number of spatial processes (" << spatialCommSize << ")\n";
            }
            MPI_Finalize();
            exit(1);
        }
        if ( spatialCommSize > nx ) {
            if (spatialRank == 0) {
                std::cout << "Error: Number of spatial DOFs (" << nx << ") exceeds number of spatial processes (" << spatialCommSize << ")\n";
            }
            MPI_Finalize();
            exit(1);
        }
    }
    
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    onProcSize  = nx / spatialCommSize;         // Number of rows on proc
    localMinRow = spatialRank * onProcSize;     // First row I own
    localMaxRow = localMinRow + onProcSize - 1; // Last row I own    
    spatialDOFs = nx;
    
    int L_nnz   = xStencilNnz * onProcSize;  // Nnz on proc
    L_rowptr     = new int[onProcSize + 1];
    L_colinds    = new int[L_nnz];
    L_data       = new double[L_nnz];
    int rowcount = 0;
    int dataInd  = 0;
    L_rowptr[0]  = 0;
    X            = new double[onProcSize]; // Initial guesss at solution
    B            = new double[onProcSize]; // Solution-independent source terms
    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * plusInds;
    double * plusWeights;
    get1DUpwindStencil(plusInds, plusWeights, 0);
    
    // Scale entries by mesh size
    for (int i = 0; i < xStencilNnz; i++) {
        plusWeights[i] /= dx;
    }
    
    // Generate stencils for wind blowing right to left by reversing stencils
    int * minusInds       = new int[xStencilNnz];
    double * minusWeights = new double[xStencilNnz];
    for (int i = 0; i < xStencilNnz; i++) {
        minusInds[i] = -plusInds[xFD_Order-i];
        minusWeights[i]    = -plusWeights[xFD_Order-i];
    } 
    
    // Placeholder for weights to discretize derivative at each individual point in space-time
    double * localWeights = new double[xStencilNnz];
    int windDirection;
    double x;
    
    std::function<double(int)> localWaveSpeed;    
         
         
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {
    
        x = MeshIndToPoint(row, xDim); // Mesh point we're discretizing at       
        // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
        localWaveSpeed = [this, x, dx, t](int offset) { return WaveSpeed(x + dx * offset, t); };
                
        // Get weights for discretizing spatial component at current point 
        getLocalUpwindDiscretization(windDirection, localWeights,
                                        localWaveSpeed, 
                                        plusWeights, plusInds, 
                                        minusWeights, minusInds, 
                                        xStencilNnz);
        
        // Wind blows left to right
        if (windDirection > 0) {            
            // DOFs in interior whose stencils cannot hit boundary    
            if ((row > xFD_Order) && (row < nx - xFD_Order - 1)) {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = plusInds[count] + row;
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }
            // Boundary DOFs whose stencils are possibly flow over boundary
            } else {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = (plusInds[count] + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }
            }
        
        // Wind blows right to left
        } else {
            // DOFs in interior whose stencils cannot hit boundary    
            if ((row > xFD_Order) && (row < nx - xFD_Order - 1)) {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = minusInds[count] + row;
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }
            // Boundary DOFs whose stencils are possibly flow over boundary
            } else {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = (minusInds[count] + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }
            }
        }
    
        // Set source term and guess at the solution
        B[rowcount] = PDE_Source(MeshIndToPoint(row, xDim), t);
        X[rowcount] = 1.0; // TODO : Set this to a random value?    
    
        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    }    
    
    // Clean up
    delete[] plusInds;
    delete[] plusWeights;
    delete[] minusInds;
    delete[] minusWeights;
    delete[] localWeights;
    
    
    /* -------------------------------------------------------------------------------------- */
    /* ------ MASS MATRIX: Assemble identity matrix if it has not been done previously ------ */
    /* -------------------------------------------------------------------------------------- */
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        m_M_rowptr    = new int[onProcSize+1];
        m_M_colinds   = new int[onProcSize];
        m_M_data      = new double[onProcSize];
        m_M_rowptr[0] = 0;
        rowcount      = 0;
        for (int row = localMinRow; row <= localMaxRow; row++) {
            m_M_colinds[rowcount]  = row;
            m_M_data[rowcount]     = 1.0;
            m_M_rowptr[rowcount+1] = rowcount+1;
            rowcount += 1;
        } 
    }
}



/* FD spatial discretization of advection */
// Simply call the parallel code with NULL spatial communicator
// This saves doubling up on code that's almost identical
void FDadvection::getSpatialDiscretization(int * &L_rowptr, int * &L_colinds,
                                           double * &L_data, double * &B, double * &X,
                                           int &spatialDOFs, double t, int &bsize)
{
    int dummy1;
    int dummy2;
    getSpatialDiscretization(NULL, L_rowptr, L_colinds, L_data, B, X, dummy1, dummy2, spatialDOFs, t, bsize);
}



// Compute upwind direction and weights to provide upwind discretization of linear flux function
void FDadvection::getLocalUpwindDiscretization(int &windDirection, double * &localWeights,
                                    const std::function<double(int)> localWaveSpeed,
                                    double * const &plusWeights, int * const &plusInds, 
                                    double * const &minusWeights, int * const &minusInds,
                                    int nWeights)
{    
    // Wave speed at point in question. The sign of this determines the upwind direction
    double waveSpeed0 = localWaveSpeed(0); 
    
    // Wind blows from left to right
    if (waveSpeed0 >= 0.0) {
        windDirection = 1;
    
        // PDE is in conservation form: Need to discretize (wavespeed*u)_x
        if (m_conservativeForm) {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = localWaveSpeed(plusInds[ind]) * plusWeights[ind];
            }
    
        // PDE is in non-conservation form: Need to discretize wavespeed*u_x    
        } else {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = waveSpeed0 * plusWeights[ind];
            }
        }
    
    // Wind blows from right to left    
    } else {
        windDirection = -1;
        // PDE is in conservation form: Need to discretize (wavespeed*u)_x
        if (m_conservativeForm) {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = localWaveSpeed(minusInds[ind]) * minusWeights[ind];
            }
    
        // PDE is in non-conservation form: Need to discretize wavespeed*u_x      
        } else {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = waveSpeed0 * minusWeights[ind];
            }
        }
    }    
}




// The mass matrix (the identity) is assembled when the spatial discretization is assembled
// Note that it has to be done this way to account for spatial parallelism since this functions
// definition does not make reference to the spatial comm group
void FDadvection::getMassMatrix(int * &M_rowptr, int * &M_colinds, double * &M_data)
{
    // Check that mass matrix has been constructed
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        std::cout << "WARNING: Mass matrix not integrated. Spatial discretization must be assembled once first\n";
        return;
    }
    
    // Direct pointers to mass matrix data arrays
    M_rowptr = m_M_rowptr;
    M_colinds = m_M_colinds; 
    M_data = m_M_data;   
}


// Add initial condition to vector B.
void FDadvection::addInitialCondition(const MPI_Comm &spatialComm, double *B) {

    // Need to work out where my rows are.
    int spatialRank;
    int spatialCommSize;
    MPI_Comm_rank(spatialComm, &spatialRank);    
    MPI_Comm_size(spatialComm, &spatialCommSize);    
        
    int onProcSize = m_nx[0] / spatialCommSize; 
    int localMinRow = spatialRank * onProcSize; 
    int localMaxRow = localMinRow + onProcSize - 1; 
    
    int rowcount = 0;
    for (int row = localMinRow; row <= localMaxRow; row++) {
        B[rowcount] += InitCond(MeshIndToPoint(row, 0));
        rowcount += 1;
    }
}


// Add initial condition to vector B.
void FDadvection::addInitialCondition(double *B)
{
    for (int i = 0; i < m_nx[0]; i++) {
        B[i] += InitCond(MeshIndToPoint(i, 0));
    }
}


// Stencils for upwind discretizations of d/dx. Wind is assumed to blow left to right. 
void FDadvection::get1DUpwindStencil(int * &inds, double * &weights, int dim)
{    
    inds    = new int[m_order[dim]+1];
    weights = new double[m_order[dim]+1];
    
    if (m_order[dim] ==  1) 
    {
        inds[0] = -1;
        inds[1] =  0;
        weights[0] = -1.0;
        weights[1] =  1.0;
    }
    else if (m_order[dim] == 2) 
    {
        inds[0] = -2;
        inds[1] = -1;
        inds[2] =  0;
        weights[0] =  1.0/2.0;
        weights[1] = -4.0/2.0;
        weights[2] =  3.0/2.0;
    }
    else if (m_order[dim] == 3) 
    {
        inds[0] = -2;
        inds[1] = -1;
        inds[2] =  0;
        inds[3] =  1;
        weights[0] =  1.0/6.0;
        weights[1] = -6.0/6.0;
        weights[2] =  3.0/6.0;
        weights[3] =  2.0/6.0;
    }
    else if (m_order[dim] == 4) 
    {
        inds[0] = -3;
        inds[1] = -2;
        inds[2] = -1;
        inds[3] =  0;
        inds[4] =  1;
        weights[0] = -1.0/12.0;
        weights[1] =  6.0/12.0;
        weights[2] = -18.0/12.0;
        weights[3] =  10.0/12.0;
        weights[4] =  3.0/12.0;
    }
    else if (m_order[dim] == 5) 
    {    
        inds[0] = -3;
        inds[1] = -2;
        inds[2] = -1;
        inds[3] =  0;
        inds[4] =  1;
        inds[5] =  2;
        weights[0] = -2.0/60.0;
        weights[1] =  15.0/60.0;
        weights[2] = -60.0/60.0;
        weights[3] =  20.0/60.0;
        weights[4] =  30.0/60.0;
        weights[5] = -3.0/60.0;
    } 
    else 
    {
        std::cout << "WARNING: invalid choice of spatial discretization. Upwind discretizations of orders 1--5 only implemented.\n";
        MPI_Finalize();
        return;
    }
}