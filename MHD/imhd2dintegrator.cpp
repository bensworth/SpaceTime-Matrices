#include "imhd2dintegrator.hpp"

namespace mfem{

void IncompressibleMHD2DIntegrator::AssembleElementGrad(
  const Array<const FiniteElement*> &el,
  ElementTransformation &Tr,
  const Array<const Vector *> &elfun,
  const Array2D<DenseMatrix *> &elmats){

  int dof_u = el[0]->GetDof();
  int dof_p = el[1]->GetDof();
  int dof_z = el[2]->GetDof();
  int dof_a = el[3]->GetDof();

  int dim = el[0]->GetDim();

  elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);
  elmats(0,1)->SetSize(dof_u*dim, dof_p);
  elmats(0,2)->SetSize(dof_u*dim, dof_z);
  elmats(0,3)->SetSize(dof_u*dim, dof_a);
  elmats(1,0)->SetSize(dof_p,     dof_u*dim);
  elmats(1,1)->SetSize(dof_p,     dof_p);
  elmats(1,2)->SetSize(dof_p,     dof_z);
  elmats(1,3)->SetSize(dof_p,     dof_a);
  elmats(2,0)->SetSize(dof_z,     dof_u*dim);
  elmats(2,1)->SetSize(dof_z,     dof_p);
  elmats(2,2)->SetSize(dof_z,     dof_z);
  elmats(2,3)->SetSize(dof_z,     dof_a);
  elmats(3,0)->SetSize(dof_a,     dof_u*dim);
  elmats(3,1)->SetSize(dof_a,     dof_p);
  elmats(3,2)->SetSize(dof_a,     dof_z);
  elmats(3,3)->SetSize(dof_a,     dof_a);

  *elmats(0,0) = 0.0;
  *elmats(0,1) = 0.0;
  *elmats(0,2) = 0.0;
  *elmats(0,3) = 0.0;
  *elmats(1,0) = 0.0;
  *elmats(1,1) = 0.0;
  *elmats(1,2) = 0.0;
  *elmats(1,3) = 0.0;
  *elmats(2,0) = 0.0;
  *elmats(2,1) = 0.0;
  *elmats(2,2) = 0.0;
  *elmats(2,3) = 0.0;
  *elmats(3,0) = 0.0;
  *elmats(3,1) = 0.0;
  *elmats(3,2) = 0.0;
  *elmats(3,3) = 0.0;

  shape_u.SetSize(dof_u);
  shape_p.SetSize(dof_p);
  shape_z.SetSize(dof_z);
  shape_a.SetSize(dof_a);
  shape_Du.SetSize(dof_u, dim);
  shape_Dz.SetSize(dof_z, dim);
  shape_Da.SetSize(dof_a, dim);

  gshape_u.SetSize(dof_u, dim);
  gshape_z.SetSize(dof_z, dim);
  gshape_a.SetSize(dof_a, dim);

  uCoeff.UseExternalData((elfun[0])->GetData(), dof_u, dim);  // coefficients of basis functions for u

  const IntegrationRule& ir = GetRule(el, Tr);

  for (int i = 0; i < ir.GetNPoints(); ++i){

    // compute quantities specific to this integration point
    // - Jacobian-related
    const IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    const DenseMatrix adJ( Tr.AdjugateJacobian() );
    const double detJ = Tr.Weight();
    // - basis functions evaluations (on reference element)
    el[0]->CalcShape(  ip, shape_u  );
    el[1]->CalcShape(  ip, shape_p  );
    el[2]->CalcShape(  ip, shape_z  );
    el[3]->CalcShape(  ip, shape_a  );
    el[0]->CalcDShape( ip, shape_Du );
    el[2]->CalcDShape( ip, shape_Dz );
    el[3]->CalcDShape( ip, shape_Da );
    // - basis functions gradient evaluations (on physical element)
    Mult(shape_Du, adJ, gshape_u);  // NB: this way they are all multiplied by detJ already!
    Mult(shape_Dz, adJ, gshape_z);
    Mult(shape_Da, adJ, gshape_a);
    // - function evaluations
    Vector Ueval(dim), gAeval(dim);
    DenseMatrix gUeval(dim);
    double Zeval = (elfun[2])->operator*(shape_z); // z
    uCoeff.MultTranspose(shape_u, Ueval);          // u
    MultAtB(uCoeff, gshape_u, gUeval);             // ∇u
    gshape_a.MultTranspose( *(elfun[3]), gAeval ); // ∇A

    double scale =0.;


    //***********************************************************************
    // u,u block
    //***********************************************************************
    // Mass (eventually rescaled by 1/dt) -----------------------------------
    DenseMatrix tempuu(dof_u);
    MultVVt(shape_u, tempuu);
    scale = ip.weight * detJ;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
    tempuu *= scale;

    // this contribution is block-diagonal: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
    }



    // Stiffness (eventually rescaled by dt) --------------------------------
    scale = ip.weight / detJ * _mu;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    MultAAt( gshape_u, tempuu );
    tempuu *= scale;

    // this contribution is block-diagonal: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
    }



    // Non-linear convection (eventually rescaled by dt) --------------------
    Vector wgu(dof_u);  
 
    scale = ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif

    // - term ((w·∇)u,v)
    gshape_u.Mult(Ueval, wgu);
    MultVWt(shape_u, wgu, tempuu);
    tempuu *= scale;
    // -- this contribution is block-diagonal: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
    }

    // - term ((u·∇)w,v)
    MultVVt(shape_u, tempuu);
    tempuu *= scale;
    // -- this is not, so we must loop over its components
    for (int k = 0; k < dim; k++){
      for (int j = 0; j < dim; j++){
        elmats(0,0)->AddMatrix(gUeval(k, j), tempuu, dof_u*k, dof_u*j );
      }
    }





    //***********************************************************************
    // p,u block
    //***********************************************************************
    // Negative Divergence (eventually rescaled by dt) ----------------------
    Vector dshape_u(dim*dof_u);
   
    gshape_u.GradToDiv(dshape_u); // get the divergence of the basis functions
    scale = - ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    AddMult_a_VWt( scale, shape_p, dshape_u, *(elmats(1,0)) );





    //***********************************************************************
    // u,p block
    //***********************************************************************
    // Negative Gradient (eventually rescaled by dt) ------------------------
    // - since this is just the transpose of elmats(1,0), we simply copy at the end
    





    //***********************************************************************
    // z,z block
    //***********************************************************************
    // Mass (eventually rescaled by dt) -------------------------------------
    scale = ip.weight * detJ;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    AddMult_a_VVt( scale, shape_z, *(elmats(2,2)) );





    //***********************************************************************
    // A,A block
    //***********************************************************************
    // Mass (eventually rescaled by 1/dt) -----------------------------------
    scale = ip.weight * detJ;
// #ifndef MULT_BY_DT
//     scale *= 1./_dt;
// #endif
    AddMult_a_VVt( scale, shape_a, *(elmats(3,3)) );


    // Stiffness (eventually rescaled by dt) --------------------------------
    scale = ip.weight / detJ * _eta;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    AddMult_a_AAt( scale, gshape_a, *(elmats(3,3)));
  

    // Semi-linear convection (eventually rescaled by dt) -------------------
    Vector gAu(dof_a);
    gshape_a.Mult( Ueval, gAu );
    scale = ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    AddMult_a_VWt(scale, shape_a, gAu, *(elmats(3,3)) );







    //***********************************************************************
    // A,u block
    //***********************************************************************
    // Semi-linear convection (eventually rescaled by dt) -------------------
    DenseMatrix tempAu(dof_a,dof_u);
    MultVWt(shape_a, shape_u, tempAu);
    scale = ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    // -- this contribution is defined block-wise: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(3,0)->AddMatrix( gAeval(k)*scale, tempAu, 0, dof_u*k );
    }



    //***********************************************************************
    // u,z block
    //***********************************************************************
    // Linearised Lorentz (eventually rescaled by dt) -----------------------
    DenseMatrix tempuz(dof_u,dof_z);
    MultVWt(shape_u, shape_z, tempuz);
    scale = ip.weight / _mu0;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    // -- this contribution is defined block-wise: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,2)->AddMatrix( gAeval(k)*scale, tempuz, dof_u*k, 0 );
    }



    //***********************************************************************
    // u,A block
    //***********************************************************************
    // Linearised Lorentz (eventually rescaled by dt) -----------------------
    DenseMatrix tempuA(dof_u,dof_a);
    scale = Zeval * ip.weight / _mu0;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    // -- this contribution is also defined block-wise, so loop
    for (int k = 0; k < dim; k++){
      Vector col;
      gshape_a.GetColumnReference( k, col );
      MultVWt(shape_u, col, tempuA);
      elmats(0,3)->AddMatrix( scale, tempuA, dof_u*k, 0 );
    }




    //***********************************************************************
    // z,A block
    //***********************************************************************
    // Mixed Stiffness (eventually rescaled by dt) --------------------------
    scale = ip.weight / detJ;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    AddMult_a_ABt( scale, gshape_z, gshape_a, *(elmats(2,3)) );

    // NB: integration by parts should give an extra boundary term -int_dO grad(A)n y, however:
    // - for Dirichlet nodes on A, its contribution is killed (since dA=0 there)
    // - for Neumann nodes on A, its contribution is also killed (since grad(A)n=0 there, and
    //    its contribution goes to the rhs)



  }


  //***********************************************************************
  // u,p block - reprise
  //***********************************************************************
  // Negative Gradient (eventually rescaled by dt) ------------------------
  elmats(0,1)->Transpose( *(elmats(1,0)) );


}













void IncompressibleMHD2DIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvecs){

  int dof_u = el[0]->GetDof();
  int dof_p = el[1]->GetDof();
  int dof_z = el[2]->GetDof();
  int dof_a = el[3]->GetDof();

  int dim = el[0]->GetDim();

  elvecs[0]->SetSize(dof_u*dim);
  elvecs[1]->SetSize(dof_p    );
  elvecs[2]->SetSize(dof_z    );
  elvecs[3]->SetSize(dof_a    );

  *(elvecs[0]) = 0.;
  *(elvecs[1]) = 0.;
  *(elvecs[2]) = 0.;
  *(elvecs[3]) = 0.;


  shape_u.SetSize(dof_u);
  shape_p.SetSize(dof_p);
  shape_z.SetSize(dof_z);
  shape_a.SetSize(dof_a);
  shape_Du.SetSize(dof_u, dim);
  shape_Dz.SetSize(dof_z, dim);
  shape_Da.SetSize(dof_a, dim);

  gshape_u.SetSize(dof_u, dim);
  gshape_z.SetSize(dof_z, dim);
  gshape_a.SetSize(dof_a, dim);

  uCoeff.UseExternalData((elfun[0])->GetData(), dof_u, dim);  // coefficients of basis functions for u
  // pCoeff.UseExternalData((elfun[1])->GetData(), dof_p);       // coefficients of basis functions for p
  // zCoeff.UseExternalData((elfun[2])->GetData(), dof_z);       // coefficients of basis functions for z
  // aCoeff.UseExternalData((elfun[3])->GetData(), dof_a);       // coefficients of basis functions for A


  const IntegrationRule& ir = GetRule(el, Tr);

  for (int i = 0; i < ir.GetNPoints(); ++i){
    // compute quantities specific to this integration point
    // - Jacobian-related
    const IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    const DenseMatrix adJ( Tr.AdjugateJacobian() );
    const double detJ = Tr.Weight();
    // - basis functions evaluations (on reference element)
    el[0]->CalcShape(  ip, shape_u  );
    el[1]->CalcShape(  ip, shape_p  );
    el[2]->CalcShape(  ip, shape_z  );
    el[3]->CalcShape(  ip, shape_a  );
    el[0]->CalcDShape( ip, shape_Du );
    el[2]->CalcDShape( ip, shape_Dz );
    el[3]->CalcDShape( ip, shape_Da );
    // - basis functions gradient evaluations (on physical element)
    Mult(shape_Du, adJ, gshape_u);  // NB: this way they are all multiplied by detJ already!
    Mult(shape_Dz, adJ, gshape_z);
    Mult(shape_Da, adJ, gshape_a);
    // - function evaluations
    Vector Ueval(dim), gAeval(dim), ugu(dim);
    DenseMatrix gUeval(dim);

    uCoeff.MultTranspose(shape_u, Ueval);          // u
    double Peval = (elfun[1])->operator*(shape_p); // p
    double Zeval = (elfun[2])->operator*(shape_z); // z
    double Aeval = (elfun[3])->operator*(shape_a); // A
    MultAtB(uCoeff, gshape_u, gUeval);             // ∇u
    gshape_a.MultTranspose( *(elfun[3]), gAeval ); // ∇A
    double gAu   = gAeval * Ueval;                 // u·∇A
    gUeval.Mult(Ueval, ugu);                       // (u·∇)u
    double du    = gUeval.Trace();                 // ∇·u

    double scale =0.;


    //***********************************************************************
    // u block
    //***********************************************************************
    // Mass (eventually rescaled by 1/dt) -----------------------------------
    scale = ip.weight * detJ;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
    // -- this contribution is made block-wise: loop over physical dimensions
    for (int k = 0; k < dim; k++){
      for ( int j = 0; j < dof_u; ++j ){
        elvecs[0]->operator()(j+k*dof_u) += shape_u(j) * scale * Ueval(k);
      }
    }



    // Stiffness (eventually rescaled by dt) --------------------------------
    DenseMatrix tempuu(dof_u,dim);
    scale = ip.weight / detJ * _mu;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    MultABt( gshape_u, gUeval, tempuu ); // compute gU:gV for each V
    tempuu *= scale;
    // -- this contribution is made block-wise: loop over physical dimensions
    for (int k = 0; k < dim; k++){
      for ( int j = 0; j < dof_u; ++j ){
        elvecs[0]->operator()(j+k*dof_u) += tempuu(j,k);
      }
    }



    // Non-linear convection (eventually rescaled by dt) --------------------
    scale = ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    // -- this contribution is made block-wise: loop over physical dimensions
    for (int k = 0; k < dim; k++){
      for ( int j = 0; j < dof_u; ++j ){
        elvecs[0]->operator()(j+k*dof_u) += shape_u(j) * scale * ugu(k);
      }
    }



    // Negative divergence (eventually rescaled by dt) ----------------------
    Vector dshape_u(dim*dof_u);
    gshape_u.GradToDiv(dshape_u); // get the divergence of the basis functions
    scale = - ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    elvecs[0]->Add( scale*Peval, dshape_u );

    

    // Lorentz (eventually rescaled by dt) ----------------------------------
    scale = ip.weight / _mu0;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    // -- this contribution is defined block-wise: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      for ( int j = 0; j < dof_u; ++j ){
        elvecs[0]->operator()(j+k*dof_u) += shape_u(j) * scale * Zeval * gAeval(k);
      }
    }




    //***********************************************************************
    // p block
    //***********************************************************************
    // Negative Divergence (eventually rescaled by dt) ----------------------
    scale = - ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    elvecs[1]->Add( scale*du, shape_p );
    



    //***********************************************************************
    // z block
    //***********************************************************************
    // Mass (eventually rescaled by dt) -------------------------------------
    scale = ip.weight * detJ;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    elvecs[2]->Add( scale*Zeval, shape_z );


    // Mixed Stiffness (eventually rescaled by dt) --------------------------
    Vector tempzA(dof_z);
    gshape_z.Mult( gAeval, tempzA );
    scale = ip.weight / detJ;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    elvecs[2]->Add( scale, tempzA );




    //***********************************************************************
    // A block
    //***********************************************************************
    // Mass (eventually rescaled by 1/dt) -----------------------------------
    scale = ip.weight * detJ;
// #ifndef MULT_BY_DT
//     scale *= 1./_dt;
// #endif
    elvecs[3]->Add( scale*Aeval, shape_a );
 

    // Stiffness (eventually rescaled by dt) --------------------------------
    Vector tempAA(dof_a);
    gshape_a.Mult(gAeval, tempAA);
    scale = ip.weight / detJ * _eta;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    elvecs[3]->Add( scale, tempAA );
  

    // Semi-linear convection (eventually rescaled by dt) -------------------
    scale = ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    elvecs[3]->Add( scale*gAu, shape_a );

  }


}




const IntegrationRule& IncompressibleMHD2DIntegrator::GetRule(const Array<const FiniteElement *> &el,
                                                                     ElementTransformation &Tr){

  // Find an integration rule which is accurate enough
  const int ordU  = el[0]->GetOrder();
  // const int ordP  = el[1]->GetOrder();
  const int ordZ  = el[2]->GetOrder();
  const int ordA  = el[3]->GetOrder();
  const int ordGU = Tr.OrderGrad(el[0]);
  // const int ordGZ = Tr.OrderGrad(el[2])
  const int ordGA = Tr.OrderGrad(el[3]);

  // I check every term in the equations, see which poly order is reached, and take the max. These give, in order
  // - 2*ordU, 2*ordU + (ordGU), 2*(ordGU), ordGU + ordP, ordZ + ordGA + ordU,
  // - ordP + ordGU
  // - ordGA+ordGZ, 2*ordZ
  // - 2*ordA, ordU+(ordGA)+ordA,2*(ordGA)
  //  and it's likely that the non-linear terms will give the biggest headaches, so:
  Array<int> ords(3);
  ords[0] = 2*ordU + ordGU;       // ( (u·∇)u, v )
  ords[1] = ordZ + ordGA + ordU;  // (   z ∇A, v )
  ords[2] = ordU + ordGA + ordA;  // ( (u·∇A), B )

  // TODO: this is overkill. I should prescribe different accuracies for each component of the integrator!
  return IntRules.Get( el[0]->GetGeomType(), ords.Max() );

}





} // namespace mfem