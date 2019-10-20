#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif

#define PI 3.14159265358979323846

// Provide finite-difference discretizations to the advection PDEs
//      u_t + \grad_x (wavespeed(x,t)*u)   = source(x,t)          (conservative form)
//      u_t + wavespeed(x,t) \cdot \grad u = source(x,t)          (non-conservative form)

class FDadvection : public SpaceTimeMatrix
{
private:

    int m_conservativeForm;  // Boolean: 1 == PDE in conservative form; 0 == PDE in non-conservative form
	int m_order;           // Order of disc, between 1 and 5
    int m_refLevels;       // Have nx == 2^(refLevels + 2) spatial DOFs
    int m_nx;
    double m_dx;
    double* m_grid;


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
    
    //void mesh(double *&x);

public:

	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt);
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, int refLevels, int order);
    ~FDadvection();

};