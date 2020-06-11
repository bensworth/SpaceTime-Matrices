#ifndef CGDIFFUSION_HPP
#define CGDIFFUSION_HPP

#include "SpaceTimeMatrix.hpp" // for base class
#include <mpi.h>               // for MPI_Comm
#include "mfem.hpp"            // for FE objects



// TODO
//	- Add option to set h^l ~ dt^k, where l is the spatial order and k
//	  time order, that way accuracy same in space and time


// Space time matrix arising from discretizing u_t -\Delta u = 1
class CGdiffusion : public SpaceTimeMatrix
{
private:


  int m_refLevels;
  int m_order;

  bool m_lumped;
	

	// These are useful to ensure consistency, and to prevent reloading meshes
  mfem::FiniteElementCollection *m_fec;
  // - space-serial version
  mfem::Mesh                    *m_mesh;
	mfem::FiniteElementSpace      *m_fespace;
	// - space-parallel version
  mfem::ParMesh                 *m_par_mesh;
	mfem::ParFiniteElementSpace   *m_par_fespace;



	// Implement virtual functions in base class:
  // Call when using spatial parallelism                          
  void getSpatialDiscretizationG(const MPI_Comm &spatialComm, double* &G, 
                                  int &localMinRow, int &localMaxRow, int &spatialDOFs, double t);                               
  void getSpatialDiscretizationL(const MPI_Comm &spatialComm, int* &L_rowptr, 
                                  int* &L_colinds, double* &L_data,
                                  double* &U0, bool getU0, 
                                  int &localMinRow, int &localMaxRow, int &spatialDOFs,
                                  double t, int &bsize);                                            
  
  // Call when NOT using spatial parallelism                                        
  void getSpatialDiscretizationG(double* &G, int &spatialDOFs, double t); 
  void getSpatialDiscretizationL(int* &L_rowptr, int* &L_colinds, double* &L_data,
                                  double* &U0, bool getU0, int &spatialDOFs,
                                  double t, int &bsize);                                            

  // these 2 should not be necessary (remnants of an older version?)
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

  // Initialise mass matrix
  void initialiseMassMatrix( );
  void initialiseParMassMatrix( );

  // Initialise mesh and fespace info
  void initialiseParFEinfo( const MPI_Comm &spatialComm );
	void initialiseFEinfo();



  void getInitialCondition(const MPI_Comm &spatialComm, double * &B, int &localMinRow, int &localMaxRow, int &spatialDOFs);
  void getInitialCondition(double * &B, int &spatialDOFs);
    
public:

	//TODO: maybe change constructor to take in meshname too, rather than hard-coding it
	CGdiffusion(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps, double dt);
	CGdiffusion(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps, double dt, int refLevels, int order);
	CGdiffusion(MPI_Comm globComm, bool pit, bool M_exists, int timeDisc, int numTimeSteps, double dt, int refLevels, int order, bool lumped);
  ~CGdiffusion();

};






#endif //CGDIFFUSION_HPP