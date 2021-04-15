#include "mfem.hpp"

using namespace std;
using namespace mfem;



// Just a dummy integrator that turns on contributions only to the (0,0) block
class dummyIntegrator : public BlockNonlinearFormIntegrator{

public:

  virtual void AssembleFaceGrad ( const Array< const FiniteElement * > &  el1,
                                  const Array< const FiniteElement * > &  el2,
                                  FaceElementTransformations &  Trans,
                                  const Array< const Vector * > &   elfun,
                                  const Array2D< DenseMatrix * > &  elmats ){
    elmats(0,0)->SetSize(el1[0]->GetDof(), el1[0]->GetDof());   *elmats(0,0) = 1.;
    elmats(0,1)->SetSize(el1[0]->GetDof(), el1[1]->GetDof());   *elmats(0,1) = 0.;
    elmats(1,0)->SetSize(el1[1]->GetDof(), el1[0]->GetDof());   *elmats(1,0) = 0.;
    elmats(1,1)->SetSize(el1[1]->GetDof(), el1[1]->GetDof());   *elmats(1,1) = 0.;

  }

};




int main(int argc, char *argv[]){

	// read mesh
  string mesh_file = "./tri-square-bug.mesh";

  Mesh *mesh = new Mesh( mesh_file.c_str(), 1, 1 );

  int ordA = 1;
  int dim = mesh->Dimension();

  // create FE spaces
  FiniteElementCollection* AhFEColl = new H1_FECollection( ordA, dim );
  FiniteElementSpace* AhFESpace = new FiniteElementSpace(  mesh, AhFEColl );

  // identify dirichlet nodes on various faces
  Array<int> essTagsN(1); essTagsN[0] = 1; // North
  Array<int> essTagsE(1); essTagsE[0] = 2; // East
  Array<int> essTagsS(1); essTagsS[0] = 3; // South
  Array<int> essTagsW(1); essTagsW[0] = 4; // West

  int numAtt = mesh->bdr_attributes.Max();
  Array<int> essBdrN( numAtt ); essBdrN = 0;
  Array<int> essBdrE( numAtt ); essBdrE = 0;
  Array<int> essBdrS( numAtt ); essBdrS = 0;
  Array<int> essBdrW( numAtt ); essBdrW = 0;
  if ( mesh->bdr_attributes.Size() > 0 ) {
    for ( int i = 1; i <= mesh->bdr_attributes.Max(); ++i ){
      if( essTagsN.Find( i ) + 1 )
        essBdrN[i-1] = 1;
      if( essTagsE.Find( i ) + 1 )
        essBdrE[i-1] = 1;
      if( essTagsS.Find( i ) + 1 )
        essBdrS[i-1] = 1;
      if( essTagsW.Find( i ) + 1 )
        essBdrW[i-1] = 1;
    }
  }

  Array<int> essTDOFN, essTDOFE, essTDOFS, essTDOFW;
  AhFESpace->GetEssentialTrueDofs( essBdrN, essTDOFN );
  AhFESpace->GetEssentialTrueDofs( essBdrE, essTDOFE );
  AhFESpace->GetEssentialTrueDofs( essBdrS, essTDOFS );
  AhFESpace->GetEssentialTrueDofs( essBdrW, essTDOFW );

  std::cout << "***********************************************************\n";
  std::cout << "Nodes on N face: "; essTDOFN.Print(mfem::out, essTDOFN.Size() );
  std::cout << "Nodes on S face: "; essTDOFS.Print(mfem::out, essTDOFS.Size() );
  std::cout << "Nodes on E face: "; essTDOFE.Print(mfem::out, essTDOFE.Size() );
  std::cout << "Nodes on W face: "; essTDOFW.Print(mfem::out, essTDOFW.Size() );
  std::cout << "***********************************************************\n";


  // initialise dummy block non-linear form
  Array< FiniteElementSpace* > feSpaces(2);
  feSpaces[0] = AhFESpace;
  feSpaces[1] = AhFESpace;
  BlockNonlinearForm NLform;
  NLform.SetSpaces( feSpaces );
  NLform.AddBdrFaceIntegrator( new dummyIntegrator(), essBdrN ); // ----------> This adds contributions to the wrong element (3,0,1 which doesn't contain north face 2,3)
  // NLform.AddBdrFaceIntegrator( new dummyIntegrator(), essBdrE ); // ----------> This adds contributions to the wrong element (2,3,0, which doesn't contain east face 1,3)
  // NLform.AddBdrFaceIntegrator( new dummyIntegrator(), essBdrS ); // ----------> This triggers segfault
  // NLform.AddBdrFaceIntegrator( new dummyIntegrator(), essBdrW ); // ----------> This triggers segfault

  // compute its gradient
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = AhFESpace->GetTrueVSize();
  offsets[2] = AhFESpace->GetTrueVSize();
  offsets.PartialSum();
  BlockVector dummy(offsets); dummy = 0.;
  Operator* J = &NLform.GetGradient( dummy );
  BlockOperator* Jb = dynamic_cast<BlockOperator*>(J);
  // print it
  SparseMatrix K  = *( dynamic_cast<SparseMatrix*>( &Jb->GetBlock(0,0) ) );
  K.Print();


  delete AhFESpace;
  delete AhFEColl;
  delete mesh;

  return 0;
}
