#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif

#include <vector>

#define PI 3.14159265358979323846

/* Provide finite-difference discretizations to the advection PDEs. There are two forms of advection PDE:
      u_t + \grad_x (wavespeed(x,t)*u)   = source(x,t)          (conservative form; the wave speed is a scalar function of m spatial variables)
      u_t + wavespeed(x,t) \cdot \grad u = source(x,t)          (non-conservative form; the wave speed is an m-dimensional function of m spatial variables)


Several test problems implemented, as follows:
    u_t + u_x = 0,                      problemID == 1
    u_t + (a(x,t)*u)_x = s2(x,t),       problemID == 2
    u_t + a(x,t)*u_x   = s3(x,t),       problemID == 3
where the wave speeds and sources can be found in the main file...    
    
    
NOTES:
    -Only periodic boundary conditions are implemented. This means the wavespeed has to be periodic in space
*/

class FDadvection : public SpaceTimeMatrix
{
private:
    int m_problemID;        // ID for test problems
    int m_refLevels;        // Have nx == 2^(refLevels + 2) spatial DOFs
    
    bool m_conservativeForm;            // TRUE == PDE in conservative form; FALSE == PDE in non-conservative form
    int m_dim;                          // Number of spatial dimensions
    std::vector<int> m_order;           // Order of discretization
    std::vector<int> m_nx;              // Number of DOFs
    std::vector<double> m_dx;           // Mesh spacing
    std::vector<double> m_boundary0;    // Lower boundary of domain

    // TODO... This class shouldn't really provide any mass matrix support at all...
    int * m_M_rowptr;
    int * m_M_colinds; 
    double * m_M_data;

    void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                  int *&L_colinds, double *&L_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize);
	
	void getSpatialDiscretization(int *&L_rowptr, int *&L_colinds, double *&L_data,
                                  double *&B, double *&X, int &spatialDOFs, double t,
                                  int &bsize);
                                  
	void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);    
    
    void addInitialCondition(const MPI_Comm &spatialComm, double *B);
    void addInitialCondition(double *B);



    void getLocalUpwindDiscretization(int &windDirection, double * &localWeights,
                                        const std::function<double(int)> localWaveSpeed,
                                        double * const &plusWeights, int * const &plusInds, 
                                        double * const &minusWeights, int * const &minusInds,
                                        int nWeights);

                                        
                                        
    double MeshIndToPoint(int meshInd, int dim);
    void get1DUpwindStencil(int * &inds, double * &weight, int dim);
    double InitCond(double x); // 1D initial condition
    double InitCond(double x, double y); // 2D initial condition
    double WaveSpeed(double x, double t); // 1D wave speed
    double WaveSpeed(double x, double y, double t, int component); // 2D wave speed
    double PDE_Source(double x, double t); // 1D source
    double PDE_Source(double x, double y, double t); // 2D source

public:

    // Constructors
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt);
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, int dim, int refLevels, int order, int problemID);
    ~FDadvection();

};