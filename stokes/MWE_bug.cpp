#include "mfem.hpp"
#include <string>
//---------------------------------------------------------------------------
using namespace std;
using namespace mfem;
//---------------------------------------------------------------------------
// velocity field
void wFun(const Vector & x, const double t, Vector & w){
  double xx(x(0));
  double yy(x(1));
  w(0) = - 10*t * 2.*(2*yy-1)*(4*xx*xx-4*xx+1); // (-t*2.*yy*(1-xx*xx) mapped from -1,1 to 0,1)
  w(1) =   10*t * 2.*(2*xx-1)*(4*yy*yy-4*yy+1); // ( t*2.*xx*(1-yy*yy) mapped from -1,1 to 0,1)
}
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	// Pb parameters
  const int ordU = 2;
  const int refLvl = 2;
  const double dt = 0.5;
  const double Pe = 1.;
 	const std::string meshName = "./tri-square-mod.mesh";
  

  // Import and refine mesh
	Mesh *mesh = new Mesh( meshName.c_str(), 1, 1 );
  for (int i = 0; i < refLvl; i++)
    mesh->UniformRefinement();
  const int dim = mesh->Dimension();

  // Initialise FE info
  FiniteElementCollection *VhFEColl  = new H1_FECollection( ordU, dim );  
  FiniteElementSpace      *VhFESpace = new FiniteElementSpace( mesh, VhFEColl );

  // Extract nodes corresponding to Dirichlet BC (whole boundary)
	Array<int> essVhTDOF;
  if ( mesh->bdr_attributes.Size() > 0 ) {
    Array<int> essBdrV( mesh->bdr_attributes.Max() );
    essBdrV = 0;
    for ( int i = 0; i < mesh->bdr_attributes.Max(); ++i ){
      if( mesh->bdr_attributes[i] == 1 )
        essBdrV[i] = 1;
    }
    VhFESpace->GetEssentialTrueDofs( essBdrV, essVhTDOF );
  }


  // Assemble bilinear form: dt*Pe*(w,grad(u))
  BilinearForm* wVarf =  new BilinearForm( VhFESpace);
  VectorFunctionCoefficient wFuncCoeff( dim, wFun );
  wFuncCoeff.SetTime( dt*7 );						// only this instant gives issues
  double PeDt = Pe*dt;  
  wVarf->AddDomainIntegrator(new ConvectionIntegrator( wFuncCoeff, PeDt ));
  wVarf->Assemble(0); //this still triggers error
  wVarf->Finalize();

  // Print full matrix
  // std::ofstream myfile;
  // std::string myfilename = "./bugMatrix.dat";
  // myfile.open( myfilename );
  // (wVarf->SpMat()).PrintMatlab(myfile);
  // myfile.close( );  

  
  // Assemble reduced system
  GridFunction dummyBC(VhFESpace);
  dummyBC = 0.;

  SparseMatrix W;

  W = wVarf->SpMat();
  Vector dummy1(wVarf->NumRows()), dummy2(wVarf->NumRows()), dummy3(wVarf->NumRows());
  dummy1 =0.; dummy2 =0.; dummy3 =0.;

  mfem::Array<int> cols(wVarf->Height());
  cols = 0;
  for (int i = 0; i < essVhTDOF.Size(); ++i){
    cols[essVhTDOF[i]] = 1;
  }
  W.EliminateCols(cols, &dummyBC, &dummy1);
  for (int i = 0; i < essVhTDOF.Size(); ++i){
    W.EliminateRow(essVhTDOF[i], mfem::Matrix::DIAG_ONE);
    // rhs(essVhTDOF[i]) = boundary_values(essVhTDOF[i]);
  }

  // wVarf->FormLinearSystem( essVhTDOF, dummyBC, dummy1, W, dummy2, dummy3 );


  delete wVarf;
  delete VhFESpace;
  delete VhFEColl;
  delete mesh;

  return 0;
}

