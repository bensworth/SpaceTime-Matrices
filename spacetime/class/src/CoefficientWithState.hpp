

#ifndef CoefficientWithState_hpp
#define CoefficientWithState_hpp

#include "mfem.hpp"

using namespace mfem;


class CoefficientWithState : public Coefficient
{

protected:
   double (*Function)(const Vector &, const Vector &);

public:
   /// Define a time-independent coefficient from a C-function
   CoefficientWithState(double (*f)(const Vector &, const Vector &))
   {
      Function = f;
   }

   /// Evaluate coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
    
   void SetState(Vector state) 
   { 
       state_.SetSize(state.Size());
       state_ = state;
   }

private:

   Vector state_;

};



#endif /* CoefficientWithState_hpp */

