#ifndef SPACETIMEMATRIX
    #include "SpaceTimeMatrix.hpp"
#endif
#include "mfem.hpp"
using namespace mfem;


// TODO:
//  - Add option to set h^l ~ dt^k, where l is the spatial order and k
//    time order, that way accuracy same in space and time
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
    ParMesh* m_pmesh;
    ParBilinearForm* m_pbform;
    ParLinearForm* m_plform;
    BilinearForm* m_bform;
    LinearForm* m_lform;
    DG_FECollection *m_fec;

    // TODO : Make all spatial discretization functions PURE VIRTUAL once they've been implemented!
    virtual void getSpatialDiscretizationG(const MPI_Comm &spatialComm, 
                                           double * &G, 
                                           int      &localMinRow, 
                                           int      &localMaxRow,
                                           int      &spatialDOFs, 
                                           double    t);                                   
    virtual void getSpatialDiscretizationL(const MPI_Comm &spatialComm, 
                                           int    * &A_rowptr, 
                                           int    * &A_colinds, 
                                           double * &A_data,
                                           double * &U0, 
                                           bool      getU0, 
                                           int      &localMinRow, 
                                           int      &localMaxRow, 
                                           int      &spatialDOFs,
                                           double    t, 
                                           int      &bsize);                                            
                                          
                                          
    // Spatial discretization on one processor                                  
    virtual void getSpatialDiscretizationG(double * &G, 
                                           int      &spatialDOFs, 
                                           double    t);
    virtual void getSpatialDiscretizationL(int    * &A_rowptr, 
                                           int    * &A_colinds, 
                                           double * &A_data,
                                           double * &U0, 
                                           bool      getU0, 
                                           int      &spatialDOFs,
                                           double    t, 
                                           int      &bsize);                                            
    void getMassMatrix(int* &M_rowptr, int* &M_colinds, double* &M_data);  
    void addInitialCondition(const MPI_Comm &spatialComm, double *B);
    void addInitialCondition(double *B);

    // TODO :  Need to implement these functions... 
    void getInitialCondition(const MPI_Comm &spatialComm, double * &B, int &localMinRow, int &localMaxRow, int &spatialDOFs);
    void getInitialCondition(double * &B, int &spatialDOFs);

public:

    DGadvection(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                double dt);
    DGadvection(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                double dt, int refLevels, int order);
    DGadvection(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps,
                double dt, int refLevels, int order, bool lumped);
    ~DGadvection();

};