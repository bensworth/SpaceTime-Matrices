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
    
-In higher dimensions, the finite-differences are done in a dimension by dimension fashion    
    
-The wavespeed is assumed to be an m-dimensional vector function whose m components are each functions of the m spatial variables and time
    
NOTES:
    -Only periodic boundary conditions are implemented. This means the wavespeed has to be periodic in space!
    
    -In 2D: If a non-square number of processors is to be used the user 
        must(!) specify the number of processors in each of the x- and y-directions
*/

class FDadvection : public SpaceTimeMatrix
{
private:
    
    bool m_conservativeForm;            /* TRUE == PDE in conservative form; FALSE == PDE in non-conservative form */
    int m_dim;                          /* Number of spatial dimensions */
    int m_problemID;                    /* ID for test problems */
    int m_refLevels;                    /* Have nx == 2^(refLevels + 2) spatial DOFs */
    int m_onProcSize;                   /* Number of DOFs on proc */
    int m_spatialDOFs;                  /* Total number of DOFs in spatial disc */
    int m_localMinRow;                  /* Global index of first DOF on proc */
    std::vector<int>    m_order;        /* Order of discretization in each direction */
    std::vector<int>    m_nx;           /* Number of DOFs in each direction */
    std::vector<double> m_dx;           /* Mesh spacing in each direction */
    std::vector<double> m_boundary0;    /* Lower boundary of domain in each direction */
    std::vector<int>    m_px;           /* Number of procs in each direction */
    std::vector<int>    m_pGridInd;     /* Grid indices of proc */
    std::vector<int>    m_nxOnProc;     /* Number of DOFs in each direction on proc */
    std::vector<int>    m_nxOnProcInt;  /* Number of DOFs in each direction on procs in interior of proc domain */
    std::vector<int>    m_nxOnProcBnd;  /* Number of DOFs in each direction on procs on boundary of proc domain */




    int m_NLocalMinRow;
    int m_SLocalMinRow;
    int m_ELocalMinRow;
    int m_WLocalMinRow;
    std::vector<int> m_NNxOnProc;
    std::vector<int> m_SNxOnProc;
    std::vector<int> m_ENxOnProc;
    std::vector<int> m_WNxOnProc;


    void getSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                  int *&L_colinds, double *&L_data, double *&B,
                                  double *&X, int &localMinRow, int &localMaxRow,
                                  int &spatialDOFs, double t, int &bsize);
                                  
    void getSpatialDiscretization(int *&L_rowptr, int *&L_colinds, double *&L_data,
                                  double *&B, double *&X, int &spatialDOFs, double t,
                                  int &bsize);                              
                                  
    void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);                                  
                                  
    void get2DSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                    int *&L_colinds, double *&L_data, double *&B,
                                    double *&X, int &localMinRow, int &localMaxRow,
                                    int &spatialDOFs, double t, int &bsize);
                                  
    void get1DSpatialDiscretization(const MPI_Comm &spatialComm, int *&L_rowptr,
                                    int *&L_colinds, double *&L_data, double *&B,
                                    double *&X, int &localMinRow, int &localMaxRow,
                                    int &spatialDOFs, double t, int &bsize);                                  
    
    void getInitialCondition(const MPI_Comm &spatialComm, double * &B, int &localMinRow, 
                                int &localMaxRow, int &spatialDOFs);
    void getInitialCondition(double * &B, int &spatialDOFs);
    
    void addInitialCondition(const MPI_Comm &spatialComm, double *B);
    void addInitialCondition(double *B);



    void getLocalUpwindDiscretization(double * &localWeights, int * &localInds,
                                        std::function<double(int)> localWaveSpeed,
                                        double * const &plusWeights, int * const &plusInds, 
                                        double * const &minusWeights, int * const &minusInds,
                                        int nWeights);

    double MeshIndToPoint(int meshInd, int dim);
    void get1DUpwindStencil(int * &inds, double * &weight, int dim);
    
    double InitCond(double x);                          /* 1D initial condition */
    double InitCond(double x, double y);                /* 2D initial condition */
    double WaveSpeed(double x, double t);               /* 1D wave speed */
    double WaveSpeed(double x, double y, double t,      /* 2D wave speed */
                        int component);  
    double PDE_Source(double x, double t);              /* 1D source */
    double PDE_Source(double x, double y, double t);    /* 2D source */

public:

    /* Constructors */
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, bool pit);
	FDadvection(MPI_Comm globComm, int timeDisc, int numTimeSteps,
				double dt, bool pit, int dim, int refLevels, int order, 
                int problemID, std::vector<int> px = {});
    ~FDadvection();

};