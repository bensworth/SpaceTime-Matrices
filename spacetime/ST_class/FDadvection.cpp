#include "FDadvection.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

// TODO: 
// Parallel in 2D...

// 
double FDadvection::InitCond(const double x) 
{        
    if (m_problemID == 1) {
        return pow(cos(PI * x), 4.0);
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        return cos(PI*x);
    } else {
        return 0.0;
    }
}


// Wave speed of PDE as a function of x and t
double FDadvection::WaveSpeed(const double x, const double t) {
    if (m_problemID == 1) {
        return 1.0;
    } else if ((m_problemID == 2) || (m_problemID == 3)) {
        return  cos(PI*(x-t)) * exp(-(1+cos(t)));
    }  else  {
        return 0.0;
    }
}


// Return the value of a point in space given its index
double FDadvection::MeshIndToVal(const int xInd)
{
    return -1.0 + xInd * m_dx; // Assuming x \in [-1,1]
}


// RHS of PDE as a function of x and t
double FDadvection::PDE_Source(const double x, const double t)
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



FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt)
{
    
}


FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order, int problemID): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt), 
    m_refLevels{refLevels}, m_order{order}, m_problemID{problemID}
{
    m_nx = pow(2, refLevels+2);
    m_dx = 2.0 / m_nx; // Assume x \in [-1,1].
    
    if ((m_problemID == 1) || (m_problemID == 2)) {
        m_conservativeForm = 1; 
    } else if (m_problemID == 3) {
        m_conservativeForm = 0; 
    } else {
        m_conservativeForm = 1;
    }
        
    // Need to initialize these to NULL so we can distinguish them from once they have been built
    m_M_rowptr  = NULL;
    m_M_colinds = NULL;
    m_M_data    = NULL;  
    
    //std::cout << "Do I use spatial parallel:   " << m_useSpatialParallel << "\n";
}


FDadvection::~FDadvection()
{
    
}


// Get local CSR structure of FD spatial  discretization.
// Can be run serially by passing NULL spatialComm
void FDadvection::getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                  int *&A_colinds, double *&A_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize) 
{
    int spatialRank;
    int spatialCommSize;
    int onProcSize;
    
    // Spatial communicator is NULL: Code is serial, so entire discretization is put on single process
    if (!spatialComm) {
        spatialRank = 0;
        spatialCommSize = 1;
        onProcSize = m_nx;
        
    // Spatial communicator exists so ensure proccessor distribution makes sense
    } else {
        MPI_Comm_rank(spatialComm, &spatialRank);    
        MPI_Comm_size(spatialComm, &spatialCommSize);        
        if ( (m_nx % spatialCommSize) != 0 ) {
            if (spatialRank == 0) {
                std::cout << "Error: Number of spatial DOFs (" << m_nx << ") does not divide number of spatial processes (" << spatialCommSize << ")\n";
            }
            MPI_Finalize();
            exit(1);
        }
        if ( spatialCommSize > m_nx ) {
            if (spatialRank == 0) {
                std::cout << "Error: Number of spatial DOFs (" << m_nx << ") exceeds number of spatial processes (" << spatialCommSize << ")\n";
            }
            MPI_Finalize();
            exit(1);
        }
    }
    
    
    onProcSize  = m_nx / spatialCommSize;       // Number of rows on proc
    localMinRow = spatialRank * onProcSize;     // First row I own
    localMaxRow = localMinRow + onProcSize - 1; // Last row I own    
    spatialDOFs = m_nx;
    int A_nnz   = (m_order + 1) * onProcSize;  // Nnz on proc
    
    A_rowptr  = new int[onProcSize + 1];
    A_colinds = new int[A_nnz];
    A_data    = new double[A_nnz];
    int rowcount   = 0;
    int dataInd    = 0;
    A_rowptr[0]    = 0;
    X = new double[onProcSize];
    B = new double[onProcSize];
    
    // Get stencils for upwind discretizations, wind blowing left to right
    int * L_PlusColinds;
    double * L_PlusData;
    getUpwindStencil(L_PlusColinds, L_PlusData);
    
    // Scale entries by mesh size
    for (int i = 0; i < m_order + 1; i++) {
        L_PlusData[i] /= m_dx;
    }
    
    // Generate stencils for wind blowing right to left by flipping stencils
    int * L_MinusColinds = new int[m_order + 1];;
    double * L_MinusData = new double[m_order + 1];
    for (int i = 0; i < m_order + 1; i++) {
        L_MinusColinds[i] = -L_PlusColinds[m_order-i];
        L_MinusData[i]    = -L_PlusData[m_order-i];
    } 
    
    // Place holder for weights to discretize derivative at each individual point
    double * L_LocalData = new double[m_order + 1];
    int windDirection;
    
    
    // Loop over all rows of the spatial discretization on this processor
    for (int row = localMinRow; row <= localMaxRow; row++) {
    
        // Get the weight for discretizing spatial component at this point in space time
        getLocalUpwindDiscretization(row, t, windDirection, L_LocalData, L_PlusData, L_PlusColinds, L_MinusData, L_MinusColinds);
    
        // Wind blows left to right
        if (windDirection > 0) {            
            // DOFs in interior whose stencils cannot hit boundary    
            if ((row > m_order) && (row < m_nx - m_order - 1)) {
                for (int count = 0; count < m_order+1; count++) {
                    A_colinds[dataInd] = L_PlusColinds[count] + row;
                    A_data[dataInd]    = L_LocalData[count];
                    dataInd += 1;
                }
            // Boundary DOFs whose stencils are possibly flow over boundary
            } else {
                for (int count = 0; count < m_order+1; count++) {
                    A_colinds[dataInd] = (L_PlusColinds[count] + row + m_nx) % m_nx; // Account for periodicity here. This always puts in range 0,nx-1
                    A_data[dataInd]    = L_LocalData[count];
                    dataInd += 1;
                }
            }
        
        // Wind blows right to left
        } else {
            // DOFs in interior whose stencils cannot hit boundary    
            if ((row > m_order) && (row < m_nx - m_order - 1)) {
                for (int count = 0; count < m_order+1; count++) {
                    A_colinds[dataInd] = L_MinusColinds[count] + row;
                    A_data[dataInd]    = L_LocalData[count];
                    dataInd += 1;
                }
            // Boundary DOFs whose stencils are possibly flow over boundary
            } else {
                for (int count = 0; count < m_order+1; count++) {
                    A_colinds[dataInd] = (L_MinusColinds[count] + row + m_nx) % m_nx; // Account for periodicity here. This always puts in range 0,nx-1
                    A_data[dataInd]    = L_LocalData[count];
                    dataInd += 1;
                }
            }
        }
    
        // Set source term and guess at the solution
        B[rowcount] = PDE_Source(MeshIndToVal(row), t);
        X[rowcount] = 1.0; // TODO : Set this to a random value?    
    
        A_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    }
    
    // Assemble the mass matrix (identity matrix) if it has not been done previously
    if ((!m_M_rowptr) || (!m_M_colinds) || (!m_M_data)) {
        m_M_rowptr  = new int[onProcSize+1];
        m_M_colinds = new int[onProcSize];
        m_M_data    = new double[onProcSize];
        m_M_rowptr[0] = 0;
        rowcount = 0;
        for (int row = localMinRow; row <= localMaxRow; row++) {
            m_M_colinds[rowcount]  = row;
            m_M_data[rowcount]     = 1.0;
            m_M_rowptr[rowcount+1] = rowcount+1;
            rowcount += 1;
        } 
    }
    
    // Don't need these any more
    delete[] L_PlusColinds;
    delete[] L_PlusData;
    delete[] L_MinusColinds;
    delete[] L_MinusData;
    delete[] L_LocalData;
}



/* FD spatial discretization of advection */
// Simply call the parallel code with NULL spatial communicator
// This saves doubling up on code that's almost identical
void FDadvection::getSpatialDiscretization(int * &A_rowptr, int * &A_colinds,
                                           double * &A_data, double * &B, double * &X,
                                           int &spatialDOFs, double t, int &bsize)
{
    int dummy1;
    int dummy2;
    getSpatialDiscretization(NULL, A_rowptr, A_colinds, A_data, B, X, dummy1, dummy2, spatialDOFs, t, bsize);
}


// Return weights for the upwind spatial discretization of the PDE at the point (x,t)
void FDadvection::getLocalUpwindDiscretization(int xInd, double t, int &windDirection, double * &L_Data,
                                                double * &L_PlusData, int * &L_PlusColinds, 
                                                double * &L_MinusData, int * &L_MinusColinds)
{
    
    double x = MeshIndToVal(xInd); // Mesh point we're discretizing at
    double waveSpeed = WaveSpeed(x, t); // Wave speed at point in question. This alone determines the upwind direction
    
    // Wind blows from left to right
    if (waveSpeed >= 0.0) {
        windDirection = 1;
        
        // PDE is in conservation form: Need to discretize (wavespeed(x,t)*u)_x
        if (m_conservativeForm) {
            for (int colInd = 0; colInd < m_order + 1; colInd++) {
                L_Data[colInd] = WaveSpeed(x + m_dx * L_PlusColinds[colInd], t) * L_PlusData[colInd];
            }
            
        // PDE is in non-conservation form: Need to discretize wavespeed(x,t)*u_x    
        } else {
            for (int colInd = 0; colInd < m_order + 1; colInd++) {
                L_Data[colInd] = waveSpeed * L_PlusData[colInd];
            }
        }
        
    // Wind blows from right to left    
    } else {
        windDirection = -1;
        // PDE is in conservation form: Need to discretize (wavespeed(x,t)*u)_x
        if (m_conservativeForm) {
            for (int colInd = 0; colInd < m_order + 1; colInd++) {
                L_Data[colInd] = WaveSpeed(x + m_dx * L_MinusColinds[colInd], t) * L_MinusData[colInd];
            }
            
        // PDE is in non-conservation form: Need to discretize wavespeed(x,t)*u_x      
        } else {
            for (int colInd = 0; colInd < m_order + 1; colInd++) {
                L_Data[colInd] = waveSpeed * L_MinusData[colInd];
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
        
    int onProcSize = m_nx / spatialCommSize; 
    int localMinRow = spatialRank * onProcSize; 
    int localMaxRow = localMinRow + onProcSize - 1; 
    
    int rowcount = 0;
    for (int row = localMinRow; row <= localMaxRow; row++) {
        B[rowcount] += InitCond(MeshIndToVal(row));
        rowcount += 1;
    }
}


// Add initial condition to vector B.
void FDadvection::addInitialCondition(double *B)
{
    for (int i = 0; i < m_nx; i++) {
        B[i] += InitCond(MeshIndToVal(i));
    }
}


// Stencils for upwind spatial discretizations of dx * du/dx. Wind is assumed to blow left to right. 
void FDadvection::getUpwindStencil(int * &colinds, double * &data)
{
    colinds = new int[m_order+1];
    data    = new double[m_order+1];
    
    if (m_order ==  1) 
    {
        colinds[0] = -1;
        colinds[1] =  0;
        data[0] = -1.0;
        data[1] =  1.0;
    }
    else if (m_order == 2) 
    {
        colinds[0] = -2;
        colinds[1] = -1;
        colinds[2] =  0;
        data[0] =  1.0/2.0;
        data[1] = -4.0/2.0;
        data[2] =  3.0/2.0;
    }
    else if (m_order == 3) 
    {
        colinds[0] = -2;
        colinds[1] = -1;
        colinds[2] =  0;
        colinds[3] =  1;
        data[0] =  1.0/6.0;
        data[1] = -6.0/6.0;
        data[2] =  3.0/6.0;
        data[3] =  2.0/6.0;
    }
    else if (m_order == 4) 
    {
        colinds[0] = -3;
        colinds[1] = -2;
        colinds[2] = -1;
        colinds[3] =  0;
        colinds[4] =  1;
        data[0] = -1.0/12.0;
        data[1] =  6.0/12.0;
        data[2] = -18.0/12.0;
        data[3] =  10.0/12.0;
        data[4] =  3.0/12.0;
    }
    else if (m_order == 5) 
    {    
        colinds[0] = -3;
        colinds[1] = -2;
        colinds[2] = -1;
        colinds[3] =  0;
        colinds[4] =  1;
        colinds[5] =  2;
        data[0] = -2.0/60.0;
        data[1] =  15.0/60.0;
        data[2] = -60.0/60.0;
        data[3] =  20.0/60.0;
        data[4] =  30.0/60.0;
        data[5] = -3.0/60.0;
    } 
    else 
    {
        std::cout << "WARNING: invalid choice of spatial discretization. Upwind discretizations of orders 1--5 only implemented.\n";
        MPI_Finalize();
        return;
    }
}