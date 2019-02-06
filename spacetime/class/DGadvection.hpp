#ifndef SPACETIMEMATRIX
	#include "SpaceTimeMatrix.hpp"
#endif
#include "/g/g19/bs/quartz/AIR_tests/src/Quadrature.hpp"
#include "mfem.hpp"
using namespace mfem;


// TODO:
//	- Add option to set h^l ~ dt^k, where l is the spatial order and k
//	  time order, that way accuracy same in space and time


class DGadvection : public SpaceTimeMatrix
{
private:

	double m_freq;
	bool m_lumped;
	int m_nAngles;
	int m_refLevels;
	int m_order;
	int m_dim;
	Vector m_omega_g;
	std::string m_transformName;

	void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&A_rowptr,
                                  int *&A_colinds, double *&A_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t);
	void getSpatialDiscretization(int *&A_rowptr, int *&A_colinds, double *&A_data,
                                  double *&B, double *&X, int &spatialDOFs, double t);
	void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);
    // virtual void getRHS(const MPI_Comm &spatialComm, double *&B, double t);

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