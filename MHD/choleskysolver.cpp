#include "choleskysolver.hpp"



namespace mfem{



CholeskySolver::CholeskySolver( const SparseMatrix* A ){
  A->ToDenseMatrix( _A );
  height = A->Height();
  width  = A->Width();

  this->Factorise();

}



void CholeskySolver::Factorise(){
  const int N = _A.Height();
  _L.SetSize(N);
  _L = 0.;

  std::cout<<"A: "; _A.Print(std::cout, width);


  Array<int> p(N);
  Vector d(N);
  for ( int i = 0; i < N; ++i ){
    p[i] = i;
    d(i) = _A(i,i);
  }

  // for each row
  for ( int m = 0; m < N; ++m ){
    // find maximum diagonal element
    int    iMax = m;
    double dMax = d(p[m]);
    for ( int i = m+1; i < N; ++i ){
      if( d(p[i])> dMax){
        iMax = i;
        dMax = d(p[i]);
      }
    }
    // store in permutation "matrix" and swap
    int temp = p[m];
    p[m]     = p[iMax];
    p[iMax]  = temp;

    _L(m,p[m]) = sqrt(d(p[m]));

    double error = 0.;
    for ( int i = m+1; i < N; ++i ){
      double sum = 0.;
      for ( int j = 0; j < m; ++j ){
        sum += _L(j,p[m]) * _L(j,p[i]);
      }
      _L(m,p[i])  = (_A(p[m],p[i]) - sum) / _L(m,p[m]);
      d(p[i]) -= _L(m,p[m])*_L(m,p[i]);
      error += d(p[i]);
    }


  }

}




void CholeskySolver::Mult( const Vector &x, Vector &y ) const{
  MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
              << ", expected size = " << Width());
  MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
              << ", expected size = " << Height());

  
  std::cerr<<"TODO";
}



} // namespace mfem