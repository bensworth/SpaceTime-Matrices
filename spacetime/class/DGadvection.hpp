#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif
#include "mfem.hpp"
using namespace mfem;


// TODO:
//	- Add option to set h^l ~ dt^k, where l is the spatial order and k
//	  time order, that way accuracy same in space and time
#define PI 3.14159265358979323846


class DGadvection : public SpaceTimeMatrix
{
private:

	bool m_lumped;
	int m_refLevels;
	int m_order;
	int m_dim;
	Vector m_omega;

	void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                  int *&A_colinds, double *&A_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize);
	void getSpatialDiscretization(int *&A_rowptr, int *&A_colinds, double *&A_data,
                                  double *&B, double *&X, int &spatialDOFs, double t,
                                  int &bsize);
	void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);
    // virtual void getRHS(const MPI_Comm &spatialComm, double *&B, double t);

	// double sigma_function(const Vector &x);

	// double Q_function(const Vector &x);
	
	// double inflow_function(const Vector &x);

public:

	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps);
	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double t0, double t1);
	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				int refLevels, int order);
	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double t0, double t1, int refLevels, int order);
    ~DGadvection() { };

};