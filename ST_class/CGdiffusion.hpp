#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif

// TODO
//	- Add option to set h^l ~ dt^k, where l is the spatial order and k
//	  time order, that way accuracy same in space and time



class CGdiffusion : public SpaceTimeMatrix
{
private:

	bool m_lumped;
	int m_refLevels;
	int m_order;
	int m_dim;
	int* m_M_rowptr;
	int* m_M_colinds;
	double* m_M_data;

	void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                  int *&A_colinds, double *&A_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize);
	void getSpatialDiscretization(int *&A_rowptr, int *&A_colinds, double *&A_data,
                                  double *&B, double *&X, int &spatialDOFs, double t,
                                  int &bsize);
	void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);
    void addInitialCondition(const MPI_Comm &spatialComm, double *B) { };
    void addInitialCondition(double *B) { };


    // TODO :  Need to implement these functions... 
    void getInitialCondition(const MPI_Comm &spatialComm, double * &B, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void getInitialCondition(double * &B, int &spatialDOFs);
    
public:

	CGdiffusion(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, bool pit);
	CGdiffusion(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, bool pit, int refLevels, int order);
	CGdiffusion(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, bool pit, int refLevels, int order, bool lumped);
    ~CGdiffusion() { };

};