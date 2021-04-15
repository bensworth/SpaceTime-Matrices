#include "operatorsseries.hpp"

namespace mfem{

OperatorsSeries::OperatorsSeries(const Array<const Operator*>& ops, const Array<bool>& ownsOps )
  : Operator( (*(ops.begin()))->Height(), (*(ops.begin()))->Width() ),
    _ownsOps(ownsOps),
    _ops(ops){

  for ( int i = 1; i < ops.Size(); ++i ){
    MFEM_VERIFY( ( ops[i-1]->Height() == ops[i]->Height() )
              && ( ops[i-1]->Width()  == ops[i]->Width()  ), "Operators of incompatible size" );
  }

}



// Operator application
void OperatorsSeries::Mult (const Vector & x, Vector & y) const{
  MFEM_VERIFY(x.Size() == width, "incorrect input Vector size");
  MFEM_VERIFY(y.Size() == height, "incorrect output Vector size");

  _tmp  = y;  // store IG
  _tmp2.SetSize(height);
  _tmp2 = 0.; // initialise to 0

  for (int iOp=0; iOp<_ops.Size(); ++iOp){

    _ops[iOp]->Mult( x, y );
    _tmp2 += y;
    y = _tmp; // reset IG
  }

  y = _tmp2;

}



// Operator transpose application
void OperatorsSeries::MultTranspose (const Vector & x, Vector & y) const{
   MFEM_VERIFY(x.Size() == height, "incorrect input Vector size");
   MFEM_VERIFY(y.Size() == width, "incorrect output Vector size");

  _tmp  = y;  // store IG
  _tmp2.SetSize(width);
  _tmp2 = 0.; // initialise to 0

  for (int iOp=0; iOp<_ops.Size(); ++iOp){
    _ops[iOp]->MultTranspose( x, y );
    _tmp2 += y;
    y = _tmp; // reset IG
  }

  y = _tmp2;

}




OperatorsSeries::~OperatorsSeries(){
  for (int i=0; i < _ops.Size(); ++i){
    if ( _ownsOps.Size()>i && _ownsOps[i] ){
      delete _ops[i];
    }
  }
}

} // mfem namespace