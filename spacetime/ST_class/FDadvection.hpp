#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif

#define PI 3.14159265358979323846

class FDadvection : public SpaceTimeMatrix
{
private:

	int m_order; // Order of disc, an int between 1 and 5
    int m_refLevels; // Have nx == 2^(refLevels + 2) spatial DOFs
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

    void getSpatialDiscretizationStencil(int * &colinds, double * &data);
    double initCond(const double x);
    
    //void mesh(double *&x);

public:

	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt);
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, int refLevels, int order);
    ~FDadvection();

};