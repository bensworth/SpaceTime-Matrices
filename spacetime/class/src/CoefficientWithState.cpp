
#include "CoefficientWithState.hpp"

using namespace mfem;


double CoefficientWithState::Eval(ElementTransformation & T,
                                  const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);
  // Vector transip;
  // transip.SetSize(state_.Size());

  // T.SetIntPoint(&ip);
   T.Transform(ip, transip);
   return ((*Function)(state_, transip));
}


