#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif

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
    int m_conservativeForm; // Boolean: 1 == PDE in conservative form; 0 == PDE in non-conservative form
	int m_order;            // Order of disc, between 1 and 5
    int m_refLevels;        // Have nx == 2^(refLevels + 2) spatial DOFs
    int m_nx;               // Number of DOFs in space
    double m_dx;            // Mesh spacing

    // TODO... This class shouldn't really provide any mass matrix support at all...
    int * m_M_rowptr;
    int * m_M_colinds; 
    double * m_M_data;

    void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                  int *&A_colinds, double *&A_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize);
	
	void getSpatialDiscretization(int *&A_rowptr, int *&A_colinds, double *&A_data,
                                  double *&B, double *&X, int &spatialDOFs, double t,
                                  int &bsize);
                                  
	void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);    
    
    void addInitialCondition(const MPI_Comm &spatialComm, double *B);
    void addInitialCondition(double *B);


    void getLocalUpwindDiscretization(int xInd, double t, int &windDirection, double * &L_Data,
                                        double * &L_PLusData, int * &L_PlusColinds, 
                                        double * &L_MinusData, int * &L_MinusColinds);
                                        
    void getUpwindStencil(int * &colinds, double * &data);
    double InitCond(const double x);
    double WaveSpeed(const double x, const double t);
    double MeshIndToVal(const int xInd);
    double PDE_Source(const double x, const double t);

public:

    // Constructors
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt);
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, int refLevels, int order, int problemID);
    ~FDadvection();

};