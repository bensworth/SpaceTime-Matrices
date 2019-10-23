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
    bool m_is_refined;
    bool m_is_prefined;
	int m_refLevels;
	int m_order;
    int m_basis_type;
	int m_dim;
	Vector m_omega;
    Mesh* m_mesh;

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

public:

	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt);
	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, int refLevels, int order);
	DGadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, int refLevels, int order, bool lumped);
    ~DGadvection();

};