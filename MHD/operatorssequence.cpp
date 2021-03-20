#include "operatorssequence.hpp"

namespace mfem{

OperatorsSequence::OperatorsSequence(const Array<const Operator*>& ops, const Array<bool>& ownsOps )
  : Solver( (*(ops.end()-1))->Height(), (*(ops.begin()))->Width(), false ),   // set it as non-iterative by default
    _ownsOps(ownsOps),
    _ops(ops){

  for ( int i = 1; i < ops.Size(); ++i ){
    MFEM_VERIFY( ops[i-1]->Height() == ops[i]->Width(), "Operators of incompatible size" );
  }

}



// Operator application
void OperatorsSequence::Mult (const Vector & x, Vector & y) const{
  MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
  MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

  _tmp = x;

  for (int iOp=0; iOp<_ops.Size(); ++iOp){
    // if the first operator on the list is square, and y is of valid size, use y as initial guess
    // if ( iOp==0 && _ops[0]->Height() == _ops[0]->Width() && _ops[0]->Height() == y.Size() ){
    //   _tmp2 = y;
    // // use 0 as initial guess otherwise
    // }else{
    _tmp2.SetSize( _ops[iOp]->Height() );
    _tmp2 = 0.;
    // }
    _ops[iOp]->Mult( _tmp, _tmp2 );
    _tmp = _tmp2;
  }

  y = _tmp2;

}



// Operator transpose application
void OperatorsSequence::MultTranspose (const Vector & x, Vector & y) const{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

  _tmp = x;

  for (int iOp=_ops.Size()-1; iOp>=0; --iOp){
    // if the last operator on the list is square, and y is of valid size, use y as initial guess
    if ( iOp==_ops.Size()-1 && _ops[iOp]->Height() == _ops[iOp]->Width() && _ops[iOp]->Width() == y.Size() ){
      _tmp2 = y;
    // use 0 as initial guess otherwise
    }else{
      _tmp2.SetSize( _ops[iOp]->Width() );
      _tmp2 = 0.;
    }
    _ops[iOp]->MultTranspose( _tmp, _tmp2 );
    _tmp = _tmp2;
  }

  y = _tmp2;

}




OperatorsSequence::~OperatorsSequence(){
  for (int i=0; i < _ops.Size(); ++i){
    if ( _ownsOps.Size()>i && _ownsOps[i] ){
      delete _ops[i];
    }
  }
}

} // mfem namespace