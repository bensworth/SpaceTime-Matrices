#include "FDadvection.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

// TODO: 
// Upwinding for variable coefficient
// 2D in space

// 
double FDadvection::InitCond(const double x) 
{        
        //double c = sin(PI * x);
        //return pow(c, 4.0);
        
        //return cos(PI * x);
        
        return 1.0;
}


// Wave speed of PDE as a function of x and t
double FDadvection::WaveSpeed(const double x, const double t) {
    //return 1.0;
    
    //return 0.5*( 1 + pow(cos(PI*x), 2.0) );
    
    return -sin(PI*x) / (PI);
}


// Return the value of a point in space given its index
double FDadvection::MeshIndToVal(const int xInd)
{
    return -1.0 + xInd * m_dx;
}

// RHS of PDE
double FDadvection::PDE_Source(const double x, const double t)
{
    //return -2*PI*cos(PI*x)*cos(2*PI*t) -PI*sin(PI*x)*cos(2*PI*t);
    //return -2*PI*cos(PI*x)*cos(2*PI*t) + sin(2*PI*t)*-PI*sin(PI*x)*cos(2*PI*t);
    return 0.0;
}



FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt)
{
    
}


FDadvection::FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
                         double dt, int refLevels, int order): 
    SpaceTimeMatrix(globComm, timeDisc, numTimeSteps, dt), 
    m_refLevels{refLevels}, m_order{order}
{
    m_nx = pow(2, refLevels+2);
    m_dx = 2.0 / m_nx; // Assume x \in [-1,1].
    
    m_conservativeForm = 1; // Assume in conservation form
}


FDadvection::~FDadvection()
{
    
}


void FDadvection::getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                  int *&A_colinds, double *&A_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize) {
                                      
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


/* Time-independent DG spatial discretization of advection */
void FDadvection::getSpatialDiscretization(int * &A_rowptr, int * &A_colinds,
                                           double * &A_data, double * &B, double * &X,
                                           int &spatialDOFs, double t, int &bsize)
{
    
    std::cout << "spatial disc order = " << m_order << '\n';
    
    spatialDOFs = m_nx;
    A_rowptr  = new int[m_nx+1];
    int A_nnz = (m_order + 1) * m_nx;
    A_colinds = new int[A_nnz];
    A_data    = new double[A_nnz];
    
    int dataInd = 0;
    A_rowptr[0] = 0;
    
    X = new double[m_nx];
    B = new double[m_nx];
    
    
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
    
    
    // Place holder for weights to discretize derivative 
    double * L_LocalData = new double[m_order + 1];
    int windDirection;
    
    // DOFs on LHS of domain whose stencil is possibly influenced by boundary
    for (int row = 0; row < m_order + 1; row++) {
        
        // Get the weight for discretizing spatial component at this point in space time
        getLocalUpwindDiscretization(row, t, windDirection, L_LocalData, L_PlusData, L_PlusColinds, L_MinusData, L_MinusColinds);
        
        // Wind blows left to right
        if (windDirection == 1) {
            for (int count = 0; count < m_order+1; count++) {
                A_colinds[dataInd] = (L_PlusColinds[count] + row + m_nx) % m_nx;
                A_data[dataInd] = L_LocalData[count];
                dataInd += 1;
            }
            
        // Wind blows right to left
        } else {
            for (int count = 0; count < m_order+1; count++) {
                A_colinds[dataInd] = (L_MinusColinds[count] + row + m_nx) % m_nx;
                A_data[dataInd] = L_LocalData[count];
                dataInd += 1;
            }
        }
        
        A_rowptr[row+1] = dataInd;
        B[row] = PDE_Source(MeshIndToVal(row), t);
        X[row] = 1.0; // TOOD : Set this to a random value?
    }
    
    // DOFs in interior whose setncils cannot hit boundary
    for (int row = m_order + 1; row < m_nx - m_order - 1; row++) {
        
        // Get the weight for discretizing spatial component at this point in space time
        getLocalUpwindDiscretization(row, t, windDirection, L_LocalData, L_PlusData, L_PlusColinds, L_MinusData, L_MinusColinds);
        
        // Wind blows left to right
        if (windDirection == 1) {
            for (int count = 0; count < m_order+1; count++) {
                A_colinds[dataInd] = L_PlusColinds[count] + row;
                A_data[dataInd] = L_LocalData[count];
                dataInd += 1;
            }
            
        // Wind blows right to left
        } else {
            for (int count = 0; count < m_order+1; count++) {
                A_colinds[dataInd] = L_MinusColinds[count] + row;
                A_data[dataInd] = L_LocalData[count];
                dataInd += 1;
            }
        }
        
        A_rowptr[row+1] = dataInd;
        B[row] = PDE_Source(MeshIndToVal(row), t);
        X[row] = 1.0; // TOOD : Set this to a random value?
    }
    
    // DOFs on RHS of domain whose stencil is possibly influenced by boundary
    // DOFs in interior whose setncils cannot hit boundary
    for (int row = m_nx - m_order - 1; row < m_nx; row++) {
        
        // Get the weight for discretizing spatial component at this point in space time
        getLocalUpwindDiscretization(row, t, windDirection, L_LocalData, L_PlusData, L_PlusColinds, L_MinusData, L_MinusColinds);
        
        // Wind blows left to right
        if (windDirection == 1) {
            for (int count = 0; count < m_order+1; count++) {
                A_colinds[dataInd] = (L_PlusColinds[count] + row) % m_nx;
                A_data[dataInd] = L_LocalData[count];
                dataInd += 1;
            }
            
        // Wind blows right to left
        } else {
            for (int count = 0; count < m_order+1; count++) {
                A_colinds[dataInd] = (L_MinusColinds[count] + row) % m_nx;
                A_data[dataInd] = L_LocalData[count];
                dataInd += 1;
            }
        }
        
        A_rowptr[row+1] = dataInd;
        B[row] = PDE_Source(MeshIndToVal(row), t);
        X[row] = 1.0; // TOOD : Set this to a random value?
    }
    
    // Don't need these any more
    delete[] L_PlusColinds;
    delete[] L_PlusData;
    delete[] L_MinusColinds;
    delete[] L_MinusData;
    delete[] L_LocalData;
}

// Return the weights for the upwind spatial discretization of the PDE at the point (x,t)
void FDadvection::getLocalUpwindDiscretization(int xInd, double t, int &windDirection, double * &L_Data,
                                                double * &L_PlusData, int * &L_PlusColinds, 
                                                double * &L_MinusData, int * &L_MinusColinds)
{
    
    double x = MeshIndToVal(xInd); // Mesh point we're discretizing at
    double waveSpeed = WaveSpeed(x, t); // Wave speed at point in question. This alone determines the upwind direction
    
    // Wind blows from left to right
    if (waveSpeed > 0.0) {
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


// Just return the identity matrix: We don't have a mass matrix for FD discretizations
void FDadvection::getMassMatrix(int * &M_rowptr, int * &M_colinds, double * &M_data)
{
    M_rowptr  = new int[m_nx+1];
    M_colinds = new int[m_nx];
    M_data    = new double[m_nx];
    M_rowptr[0] = 0;
    for (int i = 0; i < m_nx; i++) {
        M_colinds[i]  = i;
        M_data[i]     = 1.0;
        M_rowptr[i+1] = i+1;
    }    
}


void FDadvection::addInitialCondition(const MPI_Comm &spatialComm, double *B) {
    
}

// Add initial condition to vector B.
void FDadvection::addInitialCondition(double *B)
{
    for (int i = 0; i < m_nx; i++) {
        B[i] += InitCond(MeshIndToVal(i));
    }
}



