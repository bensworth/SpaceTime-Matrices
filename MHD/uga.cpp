// Test file to check correctness of implementation

//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "boundaryfacediffusionintegrator.hpp"
#include "mymixedbilinearform.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]){

  int refLvl = 4;
  string mesh_file = "./meshes/tri-square-testAn.mesh";

  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.Parse();


  // For each processor:
  //- generate mesh
  Mesh *_mesh = new Mesh( mesh_file.c_str(), 1, 1 );
  int _dim = _mesh->Dimension();
  
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();

  // - initialise FE info
  FiniteElementCollection* _AhFEColl = new H1_FECollection( 2, _dim );
  FiniteElementCollection* _ZhFEColl = new H1_FECollection( 1, _dim );
  FiniteElementSpace*     _AhFESpace = new FiniteElementSpace( _mesh, _AhFEColl );
  FiniteElementSpace*     _ZhFESpace = new FiniteElementSpace( _mesh, _ZhFEColl );

  // - identify dirichlet nodes
  Array<int> essTagsA(1); essTagsA[0] = 4; // W
  Array<int> _essAhTDOF;
  Array<int> _essZhTDOF;
  int numAtt = _mesh->bdr_attributes.Max();
  Array<int> essBdrA( numAtt ); essBdrA = 0;
  if ( _mesh->bdr_attributes.Size() > 0 ) {
    // search among all possible tags
    for ( int i = 1; i <= _mesh->bdr_attributes.Max(); ++i ){
      // if that tag is marked in the corresponding array in essTags, then flag it
      if( essTagsA.Find( i ) + 1 )
        essBdrA[i-1] = 1;
    }
    _AhFESpace->GetEssentialTrueDofs( essBdrA, _essAhTDOF );
    _ZhFESpace->GetEssentialTrueDofs( essBdrA, _essZhTDOF );
  }


  std::cout << "***********************************************************\n";
  std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
  std::cout << "***********************************************************\n";
  std::cout << "Dir A      "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
  std::cout << "essBdr A   ";  essBdrA.Print(  mfem::out,    essBdrA.Size() ); std::cout<< "\n";
  std::cout << "essTags A  ";  essTagsA.Print( mfem::out,   essTagsA.Size() ); std::cout<< "\n";
  std::cout << "Z on dir A "; _essZhTDOF.Print(mfem::out, _essZhTDOF.Size() ); std::cout<< "\n";
  std::cout << "Bdr atts   "; _mesh->bdr_attributes.Print(mfem::out, _mesh->bdr_attributes.Size() ); std::cout<< "\n";
  std::cout << "***********************************************************\n";
  
  for (int i = 0; i < _ZhFESpace->GetNBE(); i++){
     std::cout <<"("<<i<<","<< _mesh->GetBdrAttribute(i)<<") ";
  }std::cout<<std::endl;
  for (int i = 0; i < _AhFESpace->GetNBE(); i++){
     std::cout <<"("<<i<<","<< _mesh->GetBdrAttribute(i)<<") ";
  }std::cout<<std::endl;

  FaceElementTransformations *ftr = _mesh->GetBdrFaceTransformations(3);
  Array<int> tr_vdofs0, tr_vdofs1, tr_vdofs2, tr_vdofs3, tr_vdofs4;
  Array<int> te_vdofs0, te_vdofs1, te_vdofs2, te_vdofs3, te_vdofs4, te_vdofs;
  _ZhFESpace->GetFaceVDofs(0, tr_vdofs0);
  _ZhFESpace->GetFaceVDofs(1, tr_vdofs1);
  _ZhFESpace->GetFaceVDofs(2, tr_vdofs2);
  _ZhFESpace->GetFaceVDofs(3, tr_vdofs3);
  _ZhFESpace->GetFaceVDofs(4, tr_vdofs4);
  _AhFESpace->GetFaceVDofs(0, te_vdofs0);
  _AhFESpace->GetFaceVDofs(1, te_vdofs1);
  _AhFESpace->GetFaceVDofs(2, te_vdofs2);
  _AhFESpace->GetFaceVDofs(3, te_vdofs3);
  _AhFESpace->GetFaceVDofs(4, te_vdofs4);
  _AhFESpace->GetElementVDofs(ftr->Elem1No, te_vdofs);
  std::cout << "Z nodes on 0 "; tr_vdofs0.Print(mfem::out, tr_vdofs0.Size() ); std::cout<< "\n";
  std::cout << "Z nodes on 1 "; tr_vdofs1.Print(mfem::out, tr_vdofs1.Size() ); std::cout<< "\n";
  std::cout << "Z nodes on 2 "; tr_vdofs2.Print(mfem::out, tr_vdofs2.Size() ); std::cout<< "\n";
  std::cout << "Z nodes on 3 "; tr_vdofs3.Print(mfem::out, tr_vdofs3.Size() ); std::cout<< "\n";
  std::cout << "Z nodes on 4 "; tr_vdofs4.Print(mfem::out, tr_vdofs4.Size() ); std::cout<< "\n";
  std::cout << "A nodes on 0 "; te_vdofs0.Print(mfem::out, te_vdofs0.Size() ); std::cout<< "\n";
  std::cout << "A nodes on 1 "; te_vdofs1.Print(mfem::out, te_vdofs1.Size() ); std::cout<< "\n";
  std::cout << "A nodes on 2 "; te_vdofs2.Print(mfem::out, te_vdofs2.Size() ); std::cout<< "\n";
  std::cout << "A nodes on 3 "; te_vdofs3.Print(mfem::out, te_vdofs3.Size() ); std::cout<< "\n";
  std::cout << "A nodes on 4 "; te_vdofs4.Print(mfem::out, te_vdofs4.Size() ); std::cout<< "\n";

  std::cout << "A nodes on dir A "; te_vdofs.Print(mfem::out, te_vdofs.Size() ); std::cout<< "\n";


  // assemble bilinear form
  MyMixedBilinearForm mVarf( _ZhFESpace, _AhFESpace );
  ConstantCoefficient one( 1.0 );
  // mVarf.AddBoundaryIntegrator( new DiffusionIntegrator(one), essTagsA );
  mVarf.AddBdrFaceIntegrator( new BoundaryFaceDiffusionIntegrator(one), essBdrA );
  // mVarf.AddBoundaryIntegrator( new BoundaryFaceDiffusionIntegrator(one), essTagsA );
  mVarf.Assemble();
  mVarf.Finalize();

  SparseMatrix Kd;

  Array<int> empty(0);
  mVarf.FormRectangularSystemMatrix( empty, _essAhTDOF, Kd );
  Kd.MakeRef( mVarf.SpMat() );
  Kd.SetGraphOwner(true);
  Kd.SetDataOwner(true);
  Kd.Print(mfem::out, _essZhTDOF.Size());
  mVarf.LoseMat();



  string myfilename;
  ofstream myfile;
  myfile.precision(std::numeric_limits< double >::max_digits10);

  myfilename = "./results/ugaKd.dat";
  myfile.open( myfilename );
  Kd.PrintMatlab(myfile);
  myfile.close( );

  delete _AhFESpace;
  delete _ZhFESpace;
  delete _AhFEColl;
  delete _ZhFEColl;
  delete _mesh;



  return 0;
}



