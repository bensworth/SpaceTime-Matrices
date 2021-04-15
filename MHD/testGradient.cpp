// Test file to check correctness of implementation of the MHD integrator
//  This is done by comparing my own implementation of the BlockNonlinearFormIntegrator
//  VS the results I would get by using classic MFEM integrators: the results should be
//  the same (up to machine precision)
//---------------------------------------------------------------------------
#include "mfem.hpp"
#include "imhd2dintegrator.hpp"
#include <string>
#include <iostream>

using namespace std;
using namespace mfem;


//---------------------------------------------------------------------------

int main(int argc, char *argv[]){

  int refLvl = 4;
  string mesh_file = "./meshes/tri-square-testAn.mesh";

  int ordU = 2;
  int ordP = 1;
  int ordZ = 1;
  int ordA = 2;

  int seed = 0;

  // -parse parameters
  OptionsParser args(argc, argv);
  args.AddOption(&refLvl, "-r", "--rlevel",
                "Refinement level (default: 4)");
  args.AddOption(&ordU, "-oU", "--ordU",
                "Velocity space polynomial order (default: 2)");
  args.AddOption(&ordP, "-oP", "--ordP",
                "Pressure space polynomial order (default: 1)");
  args.AddOption(&ordZ, "-oZ", "--ordZ",
                "Laplacian of vector potential space polynomial order (default: 1)");
  args.AddOption(&ordA, "-oA", "--ordA",
                "Vector potential space polynomial order (default: 2)");
  args.AddOption(&seed, "-s", "--seed",
                "Seed for random number generator (default 0)");
  args.Parse();

  const double _dt = 0.5;

  //- generate mesh
  Mesh *_mesh = new Mesh( mesh_file.c_str(), 1, 1 );
  int _dim = _mesh->Dimension();
  
  for (int i = 0; i < refLvl; i++)
    _mesh->UniformRefinement();
  

  // - initialise FE info
  FiniteElementCollection* _UhFEColl = new H1_FECollection( ordU, _dim );
  FiniteElementCollection* _PhFEColl = new H1_FECollection( ordP, _dim );
  FiniteElementCollection* _ZhFEColl = new H1_FECollection( ordZ, _dim );
  FiniteElementCollection* _AhFEColl = new H1_FECollection( ordA, _dim );
  FiniteElementSpace* _UhFESpace = new FiniteElementSpace( _mesh, _UhFEColl, _dim );
  FiniteElementSpace* _PhFESpace = new FiniteElementSpace( _mesh, _PhFEColl );
  FiniteElementSpace* _ZhFESpace = new FiniteElementSpace( _mesh, _ZhFEColl );
  FiniteElementSpace* _AhFESpace = new FiniteElementSpace( _mesh, _AhFEColl );

  // - identify dirichlet nodes
  Array<int> essTagsA(4); essTagsA[0] = 1; essTagsA[1] = 2; essTagsA[2] = 3; essTagsA[3] = 4; // 1N 2E 3S 4W
  Array<int> _essAhTDOF;
  // Array<int> _essZhTDOF;
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
    // _ZhFESpace->GetEssentialTrueDofs( essBdrA, _essZhTDOF );
  }


  std::cout << "***********************************************************\n";
  std::cout << "dim(Uh) = " << _UhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ph) = " << _PhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Zh) = " << _ZhFESpace->GetTrueVSize() << "\n";
  std::cout << "dim(Ah) = " << _AhFESpace->GetTrueVSize() << "\n";
  std::cout << "***********************************************************\n";
  // std::cout << "Dir A      "; _essAhTDOF.Print(mfem::out, _essAhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "essBdr A   ";  essBdrA.Print(  mfem::out,    essBdrA.Size() ); std::cout<< "\n";
  // std::cout << "essTags A  ";  essTagsA.Print( mfem::out,   essTagsA.Size() ); std::cout<< "\n";
  // std::cout << "Z on dir A "; _essZhTDOF.Print(mfem::out, _essZhTDOF.Size() ); std::cout<< "\n";
  // std::cout << "Bdr atts   "; _mesh->bdr_attributes.Print(mfem::out, _mesh->bdr_attributes.Size() ); std::cout<< "\n";
  // std::cout << "***********************************************************\n";


  // Inisialise own integrator ----------------------------------------------
  Array< FiniteElementSpace* > feSpaces(4);
  feSpaces[0] = _UhFESpace;
  feSpaces[1] = _PhFESpace;
  feSpaces[2] = _ZhFESpace;
  feSpaces[3] = _AhFESpace;
  BlockNonlinearForm _IMHD2DOperator;
  _IMHD2DOperator.SetSpaces( feSpaces );
  _IMHD2DOperator.AddDomainIntegrator(  new IncompressibleMHD2DIntegrator( _dt, 1, 1, 1 ) );
  _IMHD2DOperator.AddBdrFaceIntegrator( new IncompressibleMHD2DIntegrator( _dt, 1, 1, 1 ), essBdrA );

  // Initialise state
  Array<int> offsets(5);
  offsets[0] = 0;
  offsets[1] = _UhFESpace->GetTrueVSize();
  offsets[2] = _PhFESpace->GetTrueVSize();
  offsets[3] = _ZhFESpace->GetTrueVSize();
  offsets[4] = _AhFESpace->GetTrueVSize();
  offsets.PartialSum();
  BlockVector x(offsets);
  // - set to random
  std::srand(seed);
  for ( int i = 0; i < x.Size(); ++i ){
    x(i)  = ( (std::rand()%100) - 50 )/5; // lies between -10 and 10
  }


  std::string myfilename = "./results/gradientConvergence.dat";
  std::ofstream myfile;
  myfile.open( myfilename );
  myfile.precision(std::numeric_limits< double >::max_digits10);


  // Evaluate operator and gradient there
  BlockOperator* J = dynamic_cast<BlockOperator*>( &_IMHD2DOperator.GetGradient( x ) );
  BlockVector fx(offsets);
  _IMHD2DOperator.Mult( x, fx );

  // go through each component
  for ( int ii = 0; ii < x.Size(); ++ii ){
    BlockVector dx(offsets);
    dx = 0.;
    dx(ii) = 1.;

    // iteratively decrease size of perturbation
    for ( int jj = 4; jj > -6; --jj ){
      double eps = pow(10,jj);

      BlockVector xpedx = x;
      xpedx(ii) += eps;

      BlockVector dy(offsets);
      _IMHD2DOperator.Mult( xpedx, dy );
      dy -= fx;
    
      BlockVector Jdx(offsets);
      J->Mult( dx, Jdx );
      Jdx *= eps;

      Jdx -= dy;

      myfile << Jdx.Norml2() << "\t";
    }
    myfile << std::endl;

  }

  myfile.close();

 

  delete _UhFESpace;
  delete _PhFESpace;
  delete _AhFESpace;
  delete _ZhFESpace;
  delete _UhFEColl;
  delete _PhFEColl;
  delete _AhFEColl;
  delete _ZhFEColl;
  delete _mesh;



  return 0;
}







