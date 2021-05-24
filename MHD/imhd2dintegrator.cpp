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

  elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);  *elmats(0,0) = 0.0;
  elmats(0,1)->SetSize(dof_u*dim, dof_p);      *elmats(0,1) = 0.0;
  elmats(0,2)->SetSize(dof_u*dim, dof_z);      *elmats(0,2) = 0.0;
  elmats(0,3)->SetSize(dof_u*dim, dof_a);      *elmats(0,3) = 0.0;
  elmats(1,0)->SetSize(dof_p,     dof_u*dim);  *elmats(1,0) = 0.0;
  elmats(1,1)->SetSize(dof_p,     dof_p);      *elmats(1,1) = 0.0;
  elmats(1,2)->SetSize(dof_p,     dof_z);      *elmats(1,2) = 0.0;
  elmats(1,3)->SetSize(dof_p,     dof_a);      *elmats(1,3) = 0.0;
  elmats(2,0)->SetSize(dof_z,     dof_u*dim);  *elmats(2,0) = 0.0;
  elmats(2,1)->SetSize(dof_z,     dof_p);      *elmats(2,1) = 0.0;
  elmats(2,2)->SetSize(dof_z,     dof_z);      *elmats(2,2) = 0.0;
  elmats(2,3)->SetSize(dof_z,     dof_a);      *elmats(2,3) = 0.0;
  elmats(3,0)->SetSize(dof_a,     dof_u*dim);  *elmats(3,0) = 0.0;
  elmats(3,1)->SetSize(dof_a,     dof_p);      *elmats(3,1) = 0.0;
  elmats(3,2)->SetSize(dof_a,     dof_z);      *elmats(3,2) = 0.0;
  elmats(3,3)->SetSize(dof_a,     dof_a);      *elmats(3,3) = 0.0;

  DenseMatrix adJ;
  adJ.SetSize(dim);    

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
    CalcAdjugate(Tr.Jacobian(), adJ);

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
    //  This is just the transpose of elmats(1,0), so we simply copy at the end
    //  (Unless stabilisation is included)






    //***********************************************************************
    // z,z block
    //***********************************************************************
    // Mass (eventually rescaled by dt) -------------------------------------
    scale = ip.weight * detJ;
// #ifdef MULT_BY_DT
    // scale *= _dt;
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
    scale = ip.weight / detJ * _eta/_mu0;
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
    // scale *= _dt;
// #endif
    AddMult_a_ABt( scale, gshape_z, gshape_a, *(elmats(2,3)) );

    // NB: integration by parts gives an extra boundary term -int_dO grad(A)n y.
    //     While its contribution is included in the rhs for the Neumann part of
    //     the boundary, thiw needs to be explicitly computed for the Dirichlet part.
    //     This is done in AssembleFaceGrad()







    //***********************************************************************
    //***********************************************************************
    // IF STABILIZATION IS INCLUDED, I NEED EXTRA TERMS!
    //***********************************************************************
    //***********************************************************************
    if ( _stab ){

      double tu=0., ta=0.;
      GetTaus( Ueval, gAeval, Tr.InverseJacobian(), tu, ta );

      // - basis functions evaluations (on reference element)
      shape_Dp.SetSize(dof_p, dim);
      gshape_p.SetSize(dof_p, dim);
      // Vector dgshape_u(dof_u);
      // Vector dgshape_A(dof_A);
      el[1]->CalcDShape( ip, shape_Dp );
      // el[0]->CalcPhysLaplacian( ip, dgshape_u ); // TODO: should I scale by detJ?
      // el[3]->CalcPhysLaplacian( ip, dgshape_A ); // TODO: should I scale by detJ?
      Mult(shape_Dp, adJ, gshape_p);  // NB: this way they are all multiplied by detJ already!



      // After linearisation, the skew-symmetric terms are (w·∇)u, ∇p, and  w·∇A?????????????



      //***********************************************************************
      // u,u block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*(w·∇)v
      // dtu term -------------------------------------------------------------
      scale = tu * ip.weight;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
      // gshape_u.Mult(Ueval, wgu); //reuse it
      MultVWt(wgu, shape_u, tempuu);
      tempuu *= scale;
      // -- this contribution is block-diagonal: for each physical dimension, I can just copy it
      for (int k = 0; k < dim; k++){
        elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
      }

      // (w·∇)u term -------------------------------------------------------
      scale = tu * ip.weight / detJ;
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      MultVVt(wgu, tempuu);
      tempuu *= scale;
      // -- this contribution is block-diagonal: for each physical dimension, I can just copy it
      for (int k = 0; k < dim; k++){
        elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
      }

      // (u·∇)w term -------------------------------------------------------
      scale = tu * ip.weight / detJ;
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      MultVWt(wgu, shape_u, tempuu);
      tempuu *= scale;
      // -- this is not, so we must loop over its components
      for (int k = 0; k < dim; k++){
        for (int j = 0; j < dim; j++){
          elmats(0,0)->AddMatrix(gUeval(k, j), tempuu, dof_u*k, dof_u*j );
        }
      }

      // -\mu ∇·∇u term -------------------------------------------------------
//       // -- this contribution is block-diagonal
//       scale = - tu * _mu * ip.weight / (detJ*detJ); // TODO: scale by detJ?
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       MultVWt(wgu, dgshape_u, tempuu);
//       for (int k = 0; k < dim; k++){
//         elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
//       }

      //***********************************************************************
      // u,p block
      //***********************************************************************
      // - p∇·v term ----------------------------------------------------------
      // This is the B^T term appearing in the non-stabilised version too: I
      //  need to compute it explicitly this time because with stabilisation
      //  I'm breaking symmetry
      // Vector dshape_u(dim*dof_u); // reuse     
      // gshape_u.GradToDiv(dshape_u); // get the divergence of the basis functions
      scale = - ip.weight;
  // #ifdef MULT_BY_DT
      scale*= _dt;
  // #endif
      AddMult_a_VWt( scale, dshape_u, shape_p, *(elmats(0,1)) );

      // ∇p·((w·∇)v) term ---------------------------------------------------
      // can be simplified to - p ∇·((w·∇)v) = - p dvi/dxj dwj/dxi
  //     scale = - tu * ip.weight / detJ;
  // // #ifdef MULT_BY_DT
  //     scale*= _dt;
  // // #endif
  //     for (int k = 0; k < dim; k++){
  //       Vector tempdwj(dim);
  //       gUeval.GetColumn(k,tempdwj);
  //       tempdwj *= scale;
  //       Vector tempdiwjdjvi = dshape_u * tempdwj;
  //       DenseMatrix tempup(dof_u,dof_p);
  //       MultVWt( tempdiwjdjvi, shape_p, tempup );        
  //       elmats(0,1)->AddMatrix( tempup, dof_u*k, 0 );
  //     }
      // but actually dunno how to handle rhs in that case, so let's stick to the original
      scale = tu * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale*= _dt;
  // #endif
      for (int k = 0; k < dim; k++){
        Vector tempgpi(dof_p);
        DenseMatrix tempup(dof_u,dof_p);
        gshape_p.GetColumn(k,tempgpi);
        MultVWt(wgu, tempgpi, tempup);
        elmats(0,1)->AddMatrix( scale, tempup, dof_u*k, 0 );
      }


      //***********************************************************************
      // u,z block
      //***********************************************************************
      // TODO: should I include something here?
      // The point is, this would dramatically change the shape of the Jacobian,
      //  adding non-zero blocks to Z. Now, I think this isn't done in Cyr's paper,
      //  the reason being (I assume) because the basis functions are picked so
      //  that the Laplacian of A is zero in the element interior. Here I'm
      //  considering a different formulation, which actually should include some
      //  terms, but hey, whatever: if the eq for j is solved exactly, it should
      //  still be 0?
      //***********************************************************************
      // u,A block
      //***********************************************************************
      // same here


      
      //***********************************************************************
      // p,p block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*∇q ***
      // ∇p term (ie, stiffness) ----------------------------------------------
      scale = tu * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale *= _dt;
  // #endif
      AddMult_a_AAt( scale, gshape_p, *(elmats(1,1)));


      //***********************************************************************
      // p,u block
      //***********************************************************************
      // dtu term -------------------------------------------------------------
      // -- this contribution is block-diagonal
      scale = tu * ip.weight;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
      for (int k = 0; k < dim; k++){
        Vector tempgpi(dof_p);
        DenseMatrix tempgpu(dof_p,dof_u);
        gshape_p.GetColumn(k,tempgpi);
        MultVWt(tempgpi, shape_u, tempgpu);
        elmats(1,0)->AddMatrix( scale, tempgpu, 0, dof_u*k );
      }

//       // -\mu ∇·∇u term -------------------------------------------------------
//       // -- this contribution is block-diagonal
//       scale = - tu * _mu * ip.weight / (detJ*detJ); // TODO: scale by detJ?
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       for (int k = 0; k < dim; k++){
//         Vector tempgpi(dof_p);
//         DenseMatrix tempgpdgu(dof_p,dof_u);
//         gshape_p.GetColumn(k,tempgpi);
//         MultVWt(tempgpi, dgshape_u, tempgpdgu);
//         tempgpdgu *= scale;
//         elmats(1,0)->AddMatrix( tempgpdgu, 0, dof_u*k );
//       }

      // (w·∇)u term -------------------------------------------------------
      // I'm breaking symmetry, but that doesn't mean I can't reuse the same code
      //  for the symmetric bits ;)
      scale = tu * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale*= _dt;
  // #endif
      for (int k = 0; k < dim; k++){
        Vector tempgpi(dof_p);
        DenseMatrix temppu(dof_p,dof_u);
        gshape_p.GetColumn(k,tempgpi);
        MultVWt(tempgpi, wgu, temppu);
        elmats(1,0)->AddMatrix( scale, temppu, 0, dof_u*k );
      }

      // (u·∇)w term -------------------------------------------------------
      scale = tu * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale*= _dt;
  // #endif
      DenseMatrix tempgpgw( dof_p, dim );
      Mult(gshape_p,gUeval,tempgpgw);
      for (int k = 0; k < dim; k++){
        Vector tempgpgwi(dof_p);
        tempgpgw.GetColumn(k,tempgpgwi);
        DenseMatrix temppu( dof_p, dof_u );
        MultVWt(tempgpgwi, shape_u, temppu);
        elmats(1,0)->AddMatrix( scale, temppu, 0, dof_u*k );
      }





      //***********************************************************************
      // p,z block
      //***********************************************************************
      // same considerations as for u,z
      //***********************************************************************
      // p,A block
      //***********************************************************************
      // same here


      //***********************************************************************
      // z,z block
      //***********************************************************************
      // Ignore
      //***********************************************************************
      // z,A block
      //***********************************************************************
      // Ignore



      //***********************************************************************
      // A,A block
      //***********************************************************************
      // I need to test the residual of the vector potential equation against ta*(w·∇b)
      // dtA term -------------------------------------------------------------
      // Vector gAu(dof_a);
      // gshape_a.Mult( Ueval, gAu ); // reuse values
      scale = ta * ip.weight;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
      AddMult_a_VWt(scale, gAu, shape_a, *(elmats(3,3)) );

      // w·∇A term ------------------------------------------------------------
      scale = ta * ip.weight / detJ; // TODO: scale by detJ?
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      AddMult_a_VVt(scale, gAu, *(elmats(3,3)) );

//       // -\eta/\mu0 ∇·∇A term ---------------------------------------------------
//       scale = - ta * _eta/_mu0 * ip.weight / (detJ*detJ); // TODO: scale by detJ?
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       AddMult_a_VWt(scale, gAu, dgshape_A, *(elmats(3,3)) );



      //***********************************************************************
      // A,u block
      //***********************************************************************
      // u·∇c term ------------------------------------------------------------
      MultVWt(gAu, shape_u, tempAu);
      scale = ta * ip.weight / detJ; // TODO: scale by detJ?
  // #ifdef MULT_BY_DT
      scale *= _dt;
  // #endif    
      // -- this contribution is defined block-wise
      for (int k = 0; k < dim; k++){
        elmats(3,0)->AddMatrix( gAeval(k)*scale, tempAu, 0, dof_u*k );
      }



    }


  }





  //***********************************************************************
  // u,p block - reprise
  //***********************************************************************
  // Negative Gradient (eventually rescaled by dt) ------------------------
  // NB: including stabilisation breaks symmetry!! So this can be done only if no stabilisation is prescribed
  if ( !_stab ){
    elmats(0,1)->Transpose( *(elmats(1,0)) );
  }



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

  elvecs[0]->SetSize(dof_u*dim);    *(elvecs[0]) = 0.;
  elvecs[1]->SetSize(dof_p    );    *(elvecs[1]) = 0.;
  elvecs[2]->SetSize(dof_z    );    *(elvecs[2]) = 0.;
  elvecs[3]->SetSize(dof_a    );    *(elvecs[3]) = 0.;

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

    double scale=0.;


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
    // scale *= _dt;
// #endif
    elvecs[2]->Add( scale*Zeval, shape_z );


    // Mixed Stiffness (eventually rescaled by dt) --------------------------
    Vector tempzA(dof_z);
    gshape_z.Mult( gAeval, tempzA );
    scale = ip.weight / detJ;
// #ifdef MULT_BY_DT
    // scale *= _dt;
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
    scale = ip.weight / detJ * _eta/_mu0;
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








    //***********************************************************************
    //***********************************************************************
    // IF STABILIZATION IS INCLUDED, I NEED EXTRA TERMS!
    //***********************************************************************
    //***********************************************************************
    if ( _stab ){

      double tu=0., ta=0.;
      GetTaus( Ueval, gAeval, Tr.InverseJacobian(), tu, ta );

      // - basis functions evaluations (on reference element)
      DenseMatrix shape_Dp(dof_p, dim);
      DenseMatrix gshape_p(dof_p, dim);
      Vector dgshape_u(dof_u);
      Vector dgshape_A(dof_a);
      el[1]->CalcDShape( ip, shape_Dp );
      el[0]->CalcPhysLinLaplacian( Tr, dgshape_u ); // this is the laplacian in *physical* coords, so no need to rescale by detJ
      el[3]->CalcPhysLinLaplacian( Tr, dgshape_A );
      double lAeval = (elfun[3])->operator*(dgshape_A); // ∇·∇A
      Vector lueval(dim);
      uCoeff.MultTranspose( dgshape_u, lueval);                  // ∇·∇u
      Mult(shape_Dp, adJ, gshape_p);  // NB: this way they are all multiplied by detJ already!
      Vector gpeval(dim);
      gshape_p.MultTranspose( *(elfun[1]), gpeval ); // ∇p

      // rhs evaluations
      Vector fEval(dim);
      _f->Eval(fEval, Tr, ip);
      double hEval = _h->Eval(Tr, ip);

      //***********************************************************************
      // u block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*(u·∇)v
      Vector tempu(dof_u);
      gshape_u.Mult( Ueval, tempu ); // this is (u·∇)v

      Vector resU(dim);
      for ( int k = 0; k < dim; ++k ){
// #ifdef MULT_BY_DT
        //                          dtu       +       (u·∇)u      -  mu ∇·∇u      + ∇p             + z∇A/mu0                   - f
        resU(k) = tu * ip.weight * ( Ueval(k) + _dt*( ugu(k)/detJ - _mu*lueval(k) + gpeval(k)/detJ + Zeval*gAeval(k)/_mu0/detJ - fEval(k) ));
// #else
        // resU(k) = tu * ip.weight * ( Ueval(k)/_dt +   ugu(k)/detJ - _mu*lUeval(k) + gpeval(k)/detJ + Zeval*gAeval(k)/_mu0/detJ - fEval(k)  );
// #endif
      }


      // -- this contribution is made block-wise: loop over physical dimensions
      for (int k = 0; k < dim; k++){
        for ( int j = 0; j < dof_u; ++j ){
          elvecs[0]->operator()(j+k*dof_u) += tempu(j) * resU(k);
        }
      }


//       // dtu ------------------------------------------------------------------
//       scale = tu * ip.weight;
// //   #ifndef MULT_BY_DT
// //       scale*= 1./_dt;
// //   #endif
//       // -- this contribution is made block-wise: loop over physical dimensions
//       for (int k = 0; k < dim; k++){
//         for ( int j = 0; j < dof_u; ++j ){
//           elvecs[0]->operator()(j+k*dof_u) += tempu(j) * scale * Ueval(k);
//         }
//       }

//       // (u·∇)u ---------------------------------------------------------------
//       scale = tu * ip.weight / detJ;
// //   #ifdef MULT_BY_DT
//       scale*= _dt;
// //   #endif
//       // -- this contribution is made block-wise: loop over physical dimensions
//       for (int k = 0; k < dim; k++){
//         for ( int j = 0; j < dof_u; ++j ){
//           elvecs[0]->operator()(j+k*dof_u) += tempu(j) * scale * ugu(k);
//         }
//       }

// //       // -\mu ∇·∇u ------------------------------------------------------------
// //       scale = -_mu * tu * ip.weight / (detJ*detJ);
// // //   #ifdef MULT_BY_DT
// //       scale*= _dt;
// // //   #endif
// //       // -- this contribution is made block-wise: loop over physical dimensions
// //       for (int k = 0; k < dim; k++){
// //         for ( int j = 0; j < dof_u; ++j ){
// //           elvecs[0]->operator()(j+k*dof_u) += tempu(j) * scale * lueval(k);
// //         }
// //       }

//       // ∇p -------------------------------------------------------------------
//       scale = tu * ip.weight / detJ;
// //   #ifdef MULT_BY_DT
//       scale*= _dt;
// //   #endif
//       // -- this contribution is made block-wise: loop over physical dimensions
//       for (int k = 0; k < dim; k++){
//         for ( int j = 0; j < dof_u; ++j ){
//           elvecs[0]->operator()(j+k*dof_u) += tempu(j) * scale * gpeval(k);
//         }
//       }

//       // -f -------------------------------------------------------------------
//       scale = tu * ip.weight;
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       for (int k = 0; k < dim; k++){
//         for ( int j = 0; j < dof_u; ++j ){
//           elvecs[0]->operator()(j+k*dof_u) += tempu(j) * scale * (-fEval(k));
//         }
//       }



      //***********************************************************************
      // p block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*∇q ***
      Vector tempp(dof_p);
      gshape_p.Mult( resU, tempp );
      elvecs[1]->operator+=(tempp);


//       // dtu ------------------------------------------------------------------
//       scale = tu * ip.weight;
// // #ifndef MULT_BY_DT
// //       scale *= 1/_dt;
// // #endif
//       elvecs[1]->Add( scale, tempp );

//       // (u·∇)u ---------------------------------------------------------------
//       scale = tu * ip.weight / detJ;
// // #ifdef MULT_BY_DT
//       scale *= _dt;
// // #endif
//       gshape_p.Mult( ugu, tempp );
//       elvecs[1]->Add( scale, tempp );
      
//       // -\mu ∇·∇u ------------------------------------------------------------
// //       scale = -_mu * tu * ip.weight / (detJ*detJ);
// // // #ifdef MULT_BY_DT
// //       scale *= _dt;
// // // #endif
// //       Vector tempp(dof_p);
// //       gshape_p.Mult( lueval, tempp );
// //       elvecs[1]->Add( scale, tempp );

//       // ∇p -------------------------------------------------------------------
//       scale = tu * ip.weight / detJ;
// // #ifdef MULT_BY_DT
//       scale *= _dt;
// // #endif
//       gshape_p.Mult( gpeval, tempp );
//       elvecs[1]->Add( scale, tempp );

//       // -f -------------------------------------------------------------------
//       scale = tu * ip.weight;
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       gshape_p.Mult( fEval,   tempp );
//       elvecs[1]->Add( -scale, tempp );



      //***********************************************************************
      // z block
      //***********************************************************************
      // No stabilisation    





      //***********************************************************************
      // A block
      //***********************************************************************
      // I need to test the residual of the vector potential equation against ta*u·∇b
      gshape_a.Mult( Ueval, tempAA); // this is u·∇b
// #ifdef MULT_BY_DT
      //                              dtA    +       u·∇A     -  eta/ mu0 ∇·∇A   - h
      double resA = ta * ip.weight * ( Aeval + _dt*( gAu/detJ - _eta/_mu0*lAeval - hEval ));
// #else
      // double res = ta * ip.weight * ( Aeval/_dt   + gAu/detJ - _eta/_mu0*lAeval - hEval );
// #endif
      elvecs[3]->Add( resA, tempAA );
      // std::cout<<res<<"\t";



//       scale = ta * ip.weight;
// // #ifndef MULT_BY_DT
// //       scale *= 1/_dt;
// // #endif
//       elvecs[3]->Add( scale*Aeval, tempAA );

//       // u·∇A -----------------------------------------------------------------
//       scale = ta * ip.weight / detJ;
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       elvecs[3]->Add( scale*gAu, tempAA );

//       // -\eta/\mu0 ∇·∇A ------------------------------------------------------
//       scale = - ta * _eta/_mu0 * ip.weight;
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       elvecs[3]->Add( scale*lAeval, tempAA );
//       // -h -------------------------------------------------------------------
//       scale = ta * ip.weight;
// // #ifdef MULT_BY_DT
//       scale*= _dt;
// // #endif
//       elvecs[3]->Add( -scale*hEval, tempAA );


      // std::cout<<"ResU=("<<resU(0)<<","<<resU(1)<<")\t|ResU|="<<resU.Norml2()<<", \tresA="<<resA<<std::endl;

      // std::cout<<"(x,y)=("<<ip.x<<","<<ip.y<<"),\tgAu="<<gAu<<"\t-dgA="<<-_eta/_mu0*lAeval<<"\th="<<hEval<<"\tdetJ="<<detJ<<",\t Res="<<res<<std::endl;
      // std::cin>>res;
    }



  }

  // double sum = 0.;
  // for ( int k = 0; k < dof_a; ++k ){
  //   sum+=(elvecs[3])->operator()(k);
  // }
  // std::cout<<"Sum="<<sum<<"\n";
}






void IncompressibleMHD2DIntegrator::AssembleFaceGrad(
  const Array< const FiniteElement * > &  el1,
  const Array< const FiniteElement * > &  el2,
  FaceElementTransformations &  Trans,
  const Array< const Vector * > &   elfun,
  const Array2D< DenseMatrix * > &  elmats ){

  if (Trans.Elem2No >= 0){
    MFEM_ABORT("AssembleFaceGrad only works for boundary faces");
  }

  int dof_u = el1[0]->GetDof();
  int dof_p = el1[1]->GetDof();
  int dof_z = el1[2]->GetDof();
  int dof_a = el1[3]->GetDof();

  int dim = el1[0]->GetDim();

  elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);   *elmats(0,0) = 0.0;
  elmats(0,1)->SetSize(dof_u*dim, dof_p);       *elmats(0,1) = 0.0;
  elmats(0,2)->SetSize(dof_u*dim, dof_z);       *elmats(0,2) = 0.0;
  elmats(0,3)->SetSize(dof_u*dim, dof_a);       *elmats(0,3) = 0.0;
  elmats(1,0)->SetSize(dof_p,     dof_u*dim);   *elmats(1,0) = 0.0;
  elmats(1,1)->SetSize(dof_p,     dof_p);       *elmats(1,1) = 0.0;
  elmats(1,2)->SetSize(dof_p,     dof_z);       *elmats(1,2) = 0.0;
  elmats(1,3)->SetSize(dof_p,     dof_a);       *elmats(1,3) = 0.0;
  elmats(2,0)->SetSize(dof_z,     dof_u*dim);   *elmats(2,0) = 0.0;
  elmats(2,1)->SetSize(dof_z,     dof_p);       *elmats(2,1) = 0.0;
  elmats(2,2)->SetSize(dof_z,     dof_z);       *elmats(2,2) = 0.0;
  elmats(2,3)->SetSize(dof_z,     dof_a);       *elmats(2,3) = 0.0;
  elmats(3,0)->SetSize(dof_a,     dof_u*dim);   *elmats(3,0) = 0.0;
  elmats(3,1)->SetSize(dof_a,     dof_p);       *elmats(3,1) = 0.0;
  elmats(3,2)->SetSize(dof_a,     dof_z);       *elmats(3,2) = 0.0;
  elmats(3,3)->SetSize(dof_a,     dof_a);       *elmats(3,3) = 0.0;

  shape_z.SetSize(dof_z);
  shape_Da.SetSize(dof_a, dim);
  shape_Dan.SetSize(dof_a);

  nor.SetSize(dim);
  ni.SetSize(dim);
  nh.SetSize(dim);

  // a simple choice for the integration order; is this OK?
  int order = el1[2]->GetOrder() + el1[3]->GetOrder();
  const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);
  // const IntegrationRule *ir = IntRule;
  // if (ir == NULL){
  //   // a simple choice for the integration order; is this OK?
  //   int order = el1[2]->GetOrder() + el1[3]->GetOrder();
  //   // ir = &IntRules.Get(Trans.GetGeometryType(), order);
  //   ir = &IntRules.Get(Trans.FaceGeom, order);
  // }

  for (int p = 0; p < ir->GetNPoints(); p++){
    const IntegrationPoint &ip = ir->IntPoint(p);
    // compute quantities specific to this integration point
    // - Jacobian-related
    IntegrationPoint eip1;
    Trans.Loc1.Transform(ip, eip1);
    Trans.Face->SetIntPoint(&ip);
    Trans.Elem1->SetIntPoint(&eip1);
    double scale = ip.weight/Trans.Elem1->Weight();
// #ifdef MULT_BY_DT
    // scale *= _dt;
// #endif
    const DenseMatrix adjJ( Trans.Elem1->AdjugateJacobian() );
    // CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
    // - normal to face
    if (dim == 1){
      nor(0) = 2*eip1.x - 1.0;
    }else{
      // CalcOrtho(Trans.Jacobian(), nor);
      CalcOrtho(Trans.Face->Jacobian(), nor);
    }
    ni.Set(scale, nor);
    adjJ.Mult(ni, nh);

    // - basis functions evaluations (on reference element)
    el1[3]->CalcDShape(eip1, shape_Da);
    el1[2]->CalcShape(eip1,  shape_z);


    //***********************************************************************
    // z,A block
    //***********************************************************************
    // Neumann boundary term (eventually rescaled by dt) --------------------
    // \int_\Gamma_N < ∇A·n,z >
    shape_Da.Mult(nh, shape_Dan);
    AddMult_a_VWt( -1., shape_z, shape_Dan, *elmats(2,3) );

  }

}








void IncompressibleMHD2DIntegrator::AssembleFaceVector(
  const Array< const FiniteElement * > &  el1,
  const Array< const FiniteElement * > &  el2,
  FaceElementTransformations &  Trans,
  const Array< const Vector * > &   elfun,
  const Array< Vector * > &   elvecs ){

  if (Trans.Elem2No >= 0){
    MFEM_ABORT("AssembleFaceVector only works for boundary faces");
  }

  int dof_u = el1[0]->GetDof();
  int dof_p = el1[1]->GetDof();
  int dof_z = el1[2]->GetDof();
  int dof_a = el1[3]->GetDof();

  int dim = el1[0]->GetDim();

  elvecs[0]->SetSize(dof_u*dim);  *(elvecs[0]) = 0.;
  elvecs[1]->SetSize(dof_p    );  *(elvecs[1]) = 0.;
  elvecs[2]->SetSize(dof_z    );  *(elvecs[2]) = 0.;
  elvecs[3]->SetSize(dof_a    );  *(elvecs[3]) = 0.;

  shape_z.SetSize(dof_z);
  shape_Da.SetSize(dof_a, dim);
  shape_Dan.SetSize(dof_a);

  nor.SetSize(dim);
  ni.SetSize(dim);
  nh.SetSize(dim);


  // a simple choice for the integration order; is this OK?
  int order = el1[2]->GetOrder() + el1[3]->GetOrder();
  const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);
  // const IntegrationRule *ir = IntRule;
  // if (ir == NULL){
  //   // a simple choice for the integration order; is this OK?
  //   int order = el1[2]->GetOrder() + el1[3]->GetOrder();
  //   // ir = &IntRules.Get(Trans.GetGeometryType(), order);
  //   ir = &IntRules.Get(Trans.FaceGeom, order);
  // }

  for (int p = 0; p < ir->GetNPoints(); p++){
    const IntegrationPoint &ip = ir->IntPoint(p);
    // compute quantities specific to this integration point
    // - Jacobian-related
    IntegrationPoint eip1;
    Trans.Loc1.Transform(ip, eip1);
    Trans.Face->SetIntPoint(&ip);
    Trans.Elem1->SetIntPoint(&eip1);
    double scale = ip.weight/Trans.Elem1->Weight();
// #ifdef MULT_BY_DT
//    scale *= _dt;
// #endif
    const DenseMatrix adjJ( Trans.Elem1->AdjugateJacobian() );
    // CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
    // - normal to face
    if (dim == 1){
      nor(0) = 2*eip1.x - 1.0;
    }else{
      // CalcOrtho(Trans.Jacobian(), nor);
      CalcOrtho(Trans.Face->Jacobian(), nor);
    }
    ni.Set(scale, nor);
    adjJ.Mult(ni, nh);

    // - basis functions evaluations (on reference element)
    el1[3]->CalcDShape(eip1, shape_Da);
    el1[2]->CalcShape(eip1,  shape_z);
    Vector gAeval(dim);
    shape_Da.MultTranspose( *(elfun[3]), gAeval ); // ∇A

    //***********************************************************************
    // z,A block
    //***********************************************************************
    // Neumann boundary term (eventually rescaled by dt) --------------------
    // \int_\Gamma_N < ∇A·n,z >
    double gAn = gAeval*nh;
    elvecs[2]->Add( -gAn, shape_z );
  }


}











void IncompressibleMHD2DIntegrator::GetTaus(const Vector& u, const Vector& gA,
                                            const DenseMatrix& Jinv, double& tauU, double& tauA) const{

  double B2 = gA*gA; // B is [dyA, -dxA], so |B| and |gradA| coincide

  DenseMatrix GC;
  GetGC( Jinv, GC );
  double GCn = GC.FNorm();

  // with rho = 1, I could simplify...
  tauU = 4./(_dt*_dt) + GC.InnerProduct(u.GetData(),u.GetData()) + C1*C1* _mu* _mu*GCn*GCn + C2*C2/_mu0*B2*GCn;
  tauA = 4./(_dt*_dt) + GC.InnerProduct(u.GetData(),u.GetData()) + C1*C1*_eta*_eta*GCn*GCn;
  tauU = 1./sqrt(tauU);
  tauA = 1./sqrt(tauA);

}






void IncompressibleMHD2DIntegrator::GetGC( const DenseMatrix& Jinv, DenseMatrix& GC ) const{
  // I need to assemble the "contravariant metric tensor Gc of the transformation from local element coordinates {ζα} to physical coordinates {xi}".
  //  Now, if I got everything right:
  //  - Tr.Jacobian() gives the jacobian of the transformation from {ζα}->{xi}
  //  - Its inverse contains the components dζα/dxi (α varies down rows, i along columns)
  //  - GC_ij is given by dζα/dxi dζα/dxj
  //  - Assuming they're using einstein notation, this means \sum_α [ dζα/dxi dζα/dxj ]
  // So that's what I'm going to assemble, hoping for the best
  GC.SetSize( Jinv.Height(), Jinv.Width() );
  GC = 0.;
  for ( int i = 0; i < Jinv.Width(); ++i ){
    for ( int j = i; j < Jinv.Width(); ++j ){
      Vector colI( Jinv.Height() );
      Jinv.GetColumn(i,colI);
      if ( i==j ){
        GC.Elem(i,i) = colI * colI;
      }else{
        Vector colJ( Jinv.Height() );
        Jinv.GetColumn(j,colJ);
        GC.Elem(i,j) = colI * colJ;
        GC.Elem(j,i) = GC.Elem(i,j);
      }
    }
  }
  // // same stuff, but slower?
  // for ( int i = 0; i < Jinv.Width(); ++i ){
  //   for ( int j = 0; j < Jinv.Width(); ++j ){
  //     for ( int k = 0; k < Jinv.Height(); ++k ){
  //       GC.Elem(i,j) += Jinv.Elem(k,i) * Jinv.Elem(k,j);
  //     }
  //   }
  // }


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
  if ( !_stab ){
    ords[0] = 2*ordU + ordGU;         // ( (u·∇)u, v )
    ords[1] =   ordU + ordGA + ordZ;  // (   z ∇A, v )
    ords[2] =   ordU + ordGA + ordA;  // ( (u·∇A), B )
  }else{
    ords[0] = 2*ordU + 2*ordGU;                  // ( (u·∇)u, (w·∇)v )
    ords[1] =   ordU +   ordGU +   ordGA + ordZ; // (   z ∇A, (w·∇)v )
    ords[2] = 2*ordU           + 2*ordGA;        // ( (u·∇A),  w·∇B )    
  }



  // std::cout<<"Selecting integrator of order "<<ords.Max()<<std::endl;

  // TODO: this is overkill. I should prescribe different accuracies for each component of the integrator!
  return IntRules.Get( el[0]->GetGeomType(), ords.Max() );

}





} // namespace mfem