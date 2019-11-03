#include "FDadvection.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

// TODO: 
// The mass-matrix should not be assembled in the getSpatialDiscretization 
// code in the sense that there is no need for it to be done there. We can just 
// assemble in in the getMassMatrix if it's not been assembled previously...

 
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
        return pow(cos(PI * x), 4.0) * pow(sin(PI * y), 2.0);
        //return ;
        // if ((x >= 0) && (y >= 0)) return 1.0;
        // if ((x < 0) && (y >= 0)) return 2.0;
        // if ((x < 0) && (y < 0)) return 3.0;
        // if ((x >= 0) && (y < 0)) return 4.0;
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
        return  cos( PI*(x-t) ) * exp( -pow(sin(2*PI*t), 2.0) );
    }  else  {
        return 0.0;
    }
}


// Wave speed for 2D problem; need to choose component as 1 or 2.
double FDadvection::WaveSpeed(double x, double y, double t, int component) {
    if (m_problemID == 1) {
        return 1.0;
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        if (component == 0) {
            return cos( PI*(x-t) ) * cos(PI*y) * exp( -pow(sin(2*PI*t), 2.0) );
        } else {
            return sin(PI*x) * cos( PI*(y-t) ) * exp( -pow(sin(2*PI*t), 2.0) );
        }
    } else {
        return 0.0;
    }
}



// Map grid index to grid point in specified dimension
double FDadvection::MeshIndToPoint(int meshInd, int dim)
{
    return m_boundary0[dim] + m_dx[dim] * meshInd;
}


// RHS of PDE 
double FDadvection::PDE_Source(double x, double t)
{
    if (m_problemID == 1) {
        return 0.0;
    } else if (m_problemID == 2) {
        return PI * exp( -2*pow(sin(PI*t), 2.0)*(cos(2*PI*t) + 2) ) * ( sin(2*PI*(t-x)) 
                    - exp( pow(sin(2*PI*t), 2.0) )*(  sin(PI*(t-x)) + 2*sin(2*PI*t)*cos(PI*(t-x)) ) );
    } else if (m_problemID == 3) {
        return 0.5* PI * exp( -2*pow(sin(PI*t), 2.0)*(cos(2*PI*t) + 2) ) * ( sin(2*PI*(t-x)) 
                    - 2*exp( pow(sin(2*PI*t), 2.0) )*(  sin(PI*(t-x)) + 2*sin(2*PI*t)*cos(PI*(t-x)) ) );
    } else {
        return 0.0;
    }
}

// RHS of PDE 
double FDadvection::PDE_Source(double x, double y, double t)
{
    if (m_problemID == 1) {
        return 0.0;
    } else if (m_problemID == 2) {
        return PI*exp(-3*pow(sin(2*PI*t), 2.0)) * 
            (
            cos(PI*(t-y))*( -exp(pow(sin(2*PI*t), 2.0)) * sin(PI*(t-x)) + cos(PI*y)*sin(2*PI*(t-x)) ) +
            cos(PI*(t-x))*( -exp(pow(sin(2*PI*t), 2.0)) * (4*cos(PI*(t-y))*sin(4*PI*t) + sin(PI*(t-y))) + sin(PI*x)*sin(2*PI*(t-y)) )
            );
    } else if (m_problemID == 3) {
        return 0.5*PI*exp(-3*pow(sin(2*PI*t), 2.0)) * 
            (
            cos(PI*(t-y))*( -2*exp(pow(sin(2*PI*t), 2.0)) * sin(PI*(t-x)) + cos(PI*y)*sin(2*PI*(t-x)) ) +
            cos(PI*(t-x))*( -2*exp(pow(sin(2*PI*t), 2.0)) * (4*cos(PI*(t-y))*sin(4*PI*t) + sin(PI*(t-y))) + sin(PI*x)*sin(2*PI*(t-y)) )
            );
    } else {
        return 0.0;
    }
}



FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, bool pit): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt, pit)
{
    
}


FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, bool pit, int dim, int refLevels, int order, int problemID): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt, pit),
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
        m_spatialDOFs = m_nx[0];
    }
    
    // Just make domain in y-direction the same as in x-direction
    if (dim == 2) {
        m_nx.push_back(nx);
        m_dx.push_back(dx);
        m_boundary0.push_back(xboundary0);
        m_order.push_back(order);
        m_spatialDOFs = m_nx[0] * m_nx[1];
    }
    
    /* Set variables based on form of PDE */
    if (m_problemID == 1) {
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_g_isTimedependent = false;
    } else if (m_problemID == 2) {
        m_conservativeForm  = true; 
        m_L_isTimedependent = true;
        m_g_isTimedependent = true;
    } else if (m_problemID == 3) {
        m_conservativeForm  = false; 
        m_L_isTimedependent = true;
        m_g_isTimedependent = true;
    } else {
        m_conservativeForm  = true;
        m_L_isTimedependent = true;
        m_g_isTimedependent = true;
    }
    
    
    /* -------------------------------------------------------- */
    /* -------------- Set up spatial parallelism -------------- */
    /* -------------------------------------------------------- */
    /* Ensure spatial parallelism setup is permissible and 
    decide which variables etc current process owns, etc */
    if (m_dim == 1) {
        
        /* --- Parallel --- */
        if (m_spatialComm) {
            m_pGridInd.push_back(m_spatialRank);
            m_npx.push_back(m_spatialCommSize);
            m_nxOnProcInt.push_back(m_nx[0]/m_npx[0]);
            
            // Procs in interior have an even number of DOFS
            if (m_spatialRank < m_npx[0]-1)  {
                m_nxOnProc.push_back(m_nxOnProcInt[0]); 
            // Last proc in domain takes the remainder of DOFs
            } else {
                m_nxOnProc.push_back(m_nx[0] - (m_npx[0]-1)*m_nxOnProcInt[0]);
            }
            
        /* --- Sequential; hard-code in 1 proccess grid even though communicator==NULL --- */    
        } else {
            m_npx             = {1};        // 1 process on grid
            m_nxOnProc.push_back(m_nx[0]);  // Assumes n procs divides DOFs 
            m_pGridInd        = {0};        // Only proc in communicator
            m_spatialRank     = 0;          // Only proc in communicator
            m_spatialCommSize = 1;          // Only 1 proc in communicator
            m_nxOnProcInt.push_back(m_nx[0]/m_npx[0]);
        }
        m_localMinRow = m_pGridInd[0] * m_nxOnProcInt[0]; // Index of first DOF on proc
        m_onProcSize  = m_nxOnProc[0];
        
    } else if (m_dim == 2) {
        /* --- Parallel --- */
        if (m_spatialComm) {
            if ( (m_spatialDOFs % m_spatialCommSize) != 0 ) {
                if (m_spatialRank == 0) {
                    std::cout << "Error: Number of spatial DOFs (" << m_spatialDOFs << ") does not divide number of spatial processes (" << m_spatialCommSize << ")\n";
                }
                MPI_Finalize();
                exit(1);
            }
            if ( m_spatialCommSize > m_spatialDOFs ) {
                if (m_spatialRank == 0) {
                    std::cout << "Error: Number of spatial DOFs (" << m_spatialDOFs << ") exceeds number of spatial processes (" << m_spatialCommSize << ")\n";
                }
                MPI_Finalize();
                exit(1);
            } 
            
            m_npx.push_back(sqrt(m_spatialCommSize)); // Assumes square proc grid
            m_npx.push_back(sqrt(m_spatialCommSize));
            m_nxOnProc.push_back(m_nx[0]/m_npx[0]); // Assumes n procs divides DOFs in each direction
            m_nxOnProc.push_back(m_nx[1]/m_npx[1]);
            m_pGridInd.push_back(m_spatialRank % m_npx[0]); // x grid-index
            m_pGridInd.push_back(m_spatialRank / m_npx[0]); // y grid-index
        
        /* --- Sequential; hard-code in 1 proccess grid even though communicator==NULL --- */    
        } else {        
            m_npx             = {1, 1};     // Have a 1 x 1 proc grid
            m_nxOnProc.push_back(m_nx[0]);  // All DOFs fit on single proc
            m_nxOnProc.push_back(m_nx[1]);  
            m_pGridInd        = {0, 0};            
            m_spatialRank     = 0;              
            m_spatialCommSize = 1; 
        }
        m_onProcSize = m_nxOnProc[0] * m_nxOnProc[1];
    }
    std::cout << "I made it through constructor..." << '\n';
}


FDadvection::~FDadvection()
{
    
}


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

// Get local CSR structure of FD spatial discretization matrix, L
// Can be run serially by passing spatialComm == NULL 
void FDadvection::getSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&B,
                                              double *&X, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) 
{
    if (m_dim == 1) {
        get1DSpatialDiscretization(spatialComm, L_rowptr,
                                      L_colinds, L_data, B,
                                      X, localMinRow, localMaxRow,
                                      spatialDOFs, t, bsize);
    } else if (m_dim == 2) {
        get2DSpatialDiscretization(spatialComm, L_rowptr,
                                      L_colinds, L_data, B,
                                      X, localMinRow, localMaxRow,
                                      spatialDOFs, t, bsize);
    }
}


// Get local CSR structure of FD spatial discretization matrix, L
// Can be run serially by passing spatialComm == NULL 
void FDadvection::get2DSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&B,
                                              double *&X, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize)
{
    // Unpack variables frequently used
    // x-related variables
    int nx          = m_nx[0];
    double dx       = m_dx[0];
    int xFD_Order   = m_order[0];
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    // y-related variables
    int ny          = m_nx[1];
    double dy       = m_dx[1];
    int yFD_Order   = m_order[1];
    int yStencilNnz = yFD_Order + 1; // Width of the FD stencil
    int yDim        = 1;
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    spatialDOFs  = m_spatialDOFs;                      
    localMinRow  = m_spatialRank * m_onProcSize;   // First row on proc
    localMaxRow  = localMinRow + m_onProcSize - 1; // Last row on proc 
    int L_nnz    = (xStencilNnz + yStencilNnz - 1) * m_onProcSize; // Nnz on proc. Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    L_rowptr     = new int[m_onProcSize + 1];
    L_colinds    = new int[L_nnz];
    L_data       = new double[L_nnz];
    L_rowptr[0]  = 0;
    X            = new double[m_onProcSize]; // Initial guesss at solution
    B            = new double[m_onProcSize]; // Solution-independent source term
    int rowcount = 0;
    int dataInd  = 0;
    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * xPlusInds;
    int * yPlusInds;
    double * xPlusWeights;
    double * yPlusWeights;
    get1DUpwindStencil(xPlusInds, xPlusWeights, xDim);
    get1DUpwindStencil(yPlusInds, yPlusWeights, yDim);
    
    // Generate stencils for wind blowing right to left by reversing stencils
    int * xMinusInds       = new int[xStencilNnz];
    int * yMinusInds       = new int[yStencilNnz];
    double * xMinusWeights = new double[xStencilNnz];
    double * yMinusWeights = new double[yStencilNnz];
    for (int i = 0; i < xStencilNnz; i++) {
        xMinusInds[i]    = -xPlusInds[xFD_Order-i];
        xMinusWeights[i] = -xPlusWeights[xFD_Order-i];
    } 
    for (int i = 0; i < yStencilNnz; i++) {
        yMinusInds[i]    = -yPlusInds[yFD_Order-i];
        yMinusWeights[i] = -yPlusWeights[yFD_Order-i];
    } 
    
    // Placeholder for weights to discretize derivatives at each point 
    double * xLocalWeights = new double[xStencilNnz];
    double * yLocalWeights = new double[yStencilNnz];
    int    * xLocalInds; // This will just point to an existing array, doesn't need memory allocated!
    int    * yLocalInds; // This will just point to an existing array, doesn't need memory allocated!
    
    int    globalInd; 
    int    xIndOnProc; 
    int    yIndOnProc; 
    int    xIndGlobal;
    int    yIndGlobal;
    double x;
    double y;
    
    std::function<double(int)>   xLocalWaveSpeed;
    std::function<double(int)>   yLocalWaveSpeed; 
    std::function<int(int, int)> MeshIndsToGlobalInd; 
    
    
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    globalInd = m_spatialRank * m_onProcSize - 1; // Substract 1 here since we add it in loop straight away
    for (int row = localMinRow; row <= localMaxRow; row++) {
        globalInd  += 1;                                            // Global index of current DOF
        xIndOnProc = rowcount % m_nxOnProc[0];                      // x-index on proc
        yIndOnProc = rowcount / m_nxOnProc[0];                      // y-index on proc
        xIndGlobal = m_pGridInd[0] * m_nxOnProc[0] + xIndOnProc;    // Global x-index
        yIndGlobal = m_pGridInd[1] * m_nxOnProc[1] + yIndOnProc;    // Global y-index
        y          = MeshIndToPoint(yIndGlobal, yDim);              // y-value of current point
        x          = MeshIndToPoint(xIndGlobal, xDim);              // x-value of current point
    
        // Get global index from any pair of x,y-indices (i.e., not those above). 
        MeshIndsToGlobalInd = [=](int xIndGlobal, int yIndGlobal) { return (xIndGlobal/m_nxOnProc[0])*(m_onProcSize - m_nxOnProc[0])  
                                                                            + (yIndGlobal/m_nxOnProc[1])*m_onProcSize*(m_npx[0] - 1) 
                                                                            + xIndGlobal + yIndGlobal*m_nxOnProc[0]; };

        // Compute x- and y-components of wavespeed given some dx or dy perturbation away from the current point
        xLocalWaveSpeed = [this, x, dx, y, t, xDim](int xOffset) { return WaveSpeed(x + dx * xOffset, y, t, xDim); };
        yLocalWaveSpeed = [this, x, y, dy, t, yDim](int yOffset) { return WaveSpeed(x, y + dy * yOffset, t, yDim); };
    
        // Get stencil for discretizing x-derivative at current point 
        getLocalUpwindDiscretization(xLocalWeights, xLocalInds,
                                        xLocalWaveSpeed, 
                                        xPlusWeights, xPlusInds, 
                                        xMinusWeights, xMinusInds, 
                                        xStencilNnz);
        // Get stencil for discretizing y-derivative at current point 
        getLocalUpwindDiscretization(yLocalWeights, yLocalInds,
                                        yLocalWaveSpeed, 
                                        yPlusWeights, yPlusInds, 
                                        yMinusWeights, yMinusInds, 
                                        yStencilNnz);
    
        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, then continuing to loop over y
        // Actually, periodicity stuffs this up I think...
        for (int yNzInd = 0; yNzInd < yStencilNnz; yNzInd++) {

            // The two stencils will intersect somewhere at this y-point
            if (yLocalInds[yNzInd] == 0) {
                for (int xNzInd = 0; xNzInd < xStencilNnz; xNzInd++) {
                    // Account for periodicity here. This always puts resulting x-index in range 0,nx-1
                    L_colinds[dataInd] = MeshIndsToGlobalInd((xIndGlobal + xLocalInds[xNzInd] + nx) % nx, yIndGlobal);
                    L_data[dataInd]    = xLocalWeights[xNzInd];

                    // The two stencils intersect at this point x-y-point, i.e. they share a 
                    // column in L, so add y-derivative information to x-derivative information that exists there
                    if (xLocalInds[xNzInd] == 0) L_data[dataInd] += yLocalWeights[yNzInd]; 
                    dataInd += 1;
                }
    
            // There is no possible intersection between between x- and y-stencils
            } else {
                // Account for periodicity here. This always puts resulting y-index in range 0,ny-1
                L_colinds[dataInd] = MeshIndsToGlobalInd(xIndGlobal, (yIndGlobal + yLocalInds[yNzInd] + ny) % ny);
                L_data[dataInd]    = yLocalWeights[yNzInd];
                dataInd += 1;
            }
        }    
    
        // Set source term and guess at the solution
        B[rowcount] = PDE_Source(x, y, t);
        X[rowcount] = 1.0; // TODO : Set this to a random value?    
    
        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    }    
    
    // Check that sufficient data was allocated
    if (dataInd > L_nnz) {
        std::cout << "WARNING: FD spatial discretization matrix has more nonzeros than allocated.\n";
    }
    
    // Clean up
    delete[] xPlusInds;
    delete[] xPlusWeights;
    delete[] yPlusInds;
    delete[] yPlusWeights;
    delete[] xMinusInds;
    delete[] xMinusWeights;
    delete[] xLocalWeights;
    delete[] yLocalWeights;
} 
                             

// Get local CSR structure of FD spatial discretization matrix, L
// Can be run serially by passing spatialComm == NULL 
void FDadvection::get1DSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
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
    
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    localMinRow = m_localMinRow;                    // First row on proc
    localMaxRow = m_localMinRow + m_onProcSize - 1; // Last row on proc
    spatialDOFs = m_spatialDOFs;
    
    int L_nnz    = xStencilNnz * m_onProcSize;  // Nnz on proc
    L_rowptr     = new int[m_onProcSize + 1];
    L_colinds    = new int[L_nnz];
    L_data       = new double[L_nnz];
    int rowcount = 0;
    int dataInd  = 0;
    L_rowptr[0]  = 0;
    X            = new double[m_onProcSize]; // Initial guesss at solution
    B            = new double[m_onProcSize]; // Solution-independent source terms
    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * plusInds;
    double * plusWeights;
    get1DUpwindStencil(plusInds, plusWeights, xDim);
    
    
    // Generate stencils for wind blowing right to left by reversing stencils
    int * minusInds       = new int[xStencilNnz];
    double * minusWeights = new double[xStencilNnz];
    for (int i = 0; i < xStencilNnz; i++) {
        minusInds[i]    = -plusInds[xFD_Order-i];
        minusWeights[i] = -plusWeights[xFD_Order-i];
    } 
    
    // Placeholder for weights and indices to discretize derivative at each point
    double * localWeights = new double[xStencilNnz];
    int * localInds; // This will just point to an existing array, doesn't need memory allocated!
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
        getLocalUpwindDiscretization(localWeights, localInds,
                                        localWaveSpeed, 
                                        plusWeights, plusInds, 
                                        minusWeights, minusInds, 
                                        xStencilNnz);

        for (int count = 0; count < xStencilNnz; count++) {
            L_colinds[dataInd] = (localInds[count] + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
            L_data[dataInd]    = localWeights[count];
            dataInd += 1;
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
}



// Compute upwind weights to provide upwind discretization of linear flux function
// Note that localInds is just directed to point at the right set of indices
void FDadvection::getLocalUpwindDiscretization(double * &localWeights, int * &localInds,
                                    std::function<double(int)> localWaveSpeed,
                                    double * const &plusWeights, int * const &plusInds, 
                                    double * const &minusWeights, int * const &minusInds,
                                    int nWeights)
{    
    // Wave speed at point in question; the sign of this determines the upwind direction
    double waveSpeed0 = localWaveSpeed(0); 
    
    // Wind blows from minus to plus
    if (waveSpeed0 >= 0.0) {
        localInds = plusInds;
    
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
    
    // Wind blows from plus to minus
    } else {        
        localInds = minusInds;
        
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




// The mass matrix just the identity
void FDadvection::getMassMatrix(int * &M_rowptr, int * &M_colinds, double * &M_data)
{
    // Check if mass matrix has been constructed, if not then build it
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        int localMinRow = m_localMinRow;                    // First row on proc
        int localMaxRow = m_localMinRow + m_onProcSize - 1; // Last row on proc
        m_M_rowptr      = new int[m_onProcSize+1];
        m_M_colinds     = new int[m_onProcSize];
        m_M_data        = new double[m_onProcSize];
        m_M_rowptr[0]   = 0;
        int rowcount    = 0;
        for (int row = localMinRow; row <= localMaxRow; row++) {
            m_M_colinds[rowcount]  = row;
            m_M_data[rowcount]     = 1.0;
            m_M_rowptr[rowcount+1] = rowcount+1;
            rowcount += 1;
        }
    } 
    
    // Direct pointers to existing member mass matrix data arrays
    M_rowptr = m_M_rowptr;
    M_colinds = m_M_colinds; 
    M_data = m_M_data; 
}


// Put initial condition in vector B.
void FDadvection::getInitialCondition(const MPI_Comm &spatialComm, double * &B, 
                                        int &localMinRow, int &localMaxRow, 
                                        int &spatialDOFs) 
{
    spatialDOFs  = m_spatialDOFs;
    localMinRow  = m_localMinRow;                    // First row on proc
    localMaxRow  = m_localMinRow + m_onProcSize - 1; // Last row on proc
    int rowcount = 0;
    B = new double[m_onProcSize]; 
    
    if (m_dim == 1) {        
        for (int row = localMinRow; row <= localMaxRow; row++) {
            B[rowcount] = InitCond(MeshIndToPoint(row, 0));
            rowcount += 1;
        }
    } else if (m_dim == 2) {
        int xInd, yInd;      
        for (int row = localMinRow; row <= localMaxRow; row++) {
            xInd = m_pGridInd[0] * m_nxOnProc[0] + rowcount % m_nxOnProc[0]; // x-index of current point
            yInd = m_pGridInd[1] * m_nxOnProc[1] + rowcount / m_nxOnProc[0]; // y-index of current point
            B[rowcount] = InitCond(MeshIndToPoint(xInd, 0), MeshIndToPoint(yInd, 1));
            rowcount += 1;
        }
    }     
}


void FDadvection::getInitialCondition(double * &B, int &spatialDOFs)
{
    spatialDOFs = m_spatialDOFs;
    B = new double[m_spatialDOFs];
    
    if (m_dim == 1) {
        for (int xInd = 0; xInd < m_nx[0]; xInd++) {
            B[xInd] = InitCond(MeshIndToPoint(xInd, 0));
        }
    } else if (m_dim == 2) {
        int rowInd = 0;
        for (int yInd = 0; yInd < m_nx[1]; yInd++) {
            for (int xInd = 0; xInd < m_nx[0]; xInd++) {
                B[rowInd] = InitCond(MeshIndToPoint(xInd, 0), MeshIndToPoint(yInd, 1));
                rowInd += 1;
            }
        }
    }
}


// Add initial condition to vector B.
// TODO : Add code to do 2D
void FDadvection::addInitialCondition(const MPI_Comm &spatialComm, double * B) {

    if (m_dim > 1) {
        std::cout << "WARNING: ONLY 1D ADD IC IMPLEMENTED" << '\n';
        MPI_Finalize();
        return;
    }

    // Need to work out where my rows are.
    int localMinRow = m_localMinRow; 
    int localMaxRow = m_localMinRow + m_onProcSize - 1; 
    
    int rowcount = 0;
    for (int row = localMinRow; row <= localMaxRow; row++) {
        B[rowcount] += InitCond(MeshIndToPoint(row, 0));
        rowcount += 1;
    }
}


// Add initial condition to vector B.
// TODO: Incorporate this into the code above by passing NULL communicator??
void FDadvection::addInitialCondition(double *B)
{
    if (m_dim == 1) {
        for (int xInd = 0; xInd < m_nx[0]; xInd++) {
            B[xInd] += InitCond(MeshIndToPoint(xInd, 0));
        }
    } else if (m_dim == 2) {
        int rowInd = 0;
        for (int yInd = 0; yInd < m_nx[1]; yInd++) {
            for (int xInd = 0; xInd < m_nx[0]; xInd++) {
                B[rowInd] += InitCond(MeshIndToPoint(xInd, 0), MeshIndToPoint(yInd, 0));
                rowInd += 1;
            }
        }
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
    
    for (int i = 0; i < m_order[dim]+1; i++) {
        weights[i] /= m_dx[dim];
    }
}