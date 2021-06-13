#include "imhd2dspaceintegrator.hpp"

namespace mfem{

void IncompressibleMHD2DSpaceIntegrator::AssembleElementGrad(
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
    // Stiffness (eventually rescaled by dt) --------------------------------
    scale = ip.weight / detJ * _mu;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    DenseMatrix gugv(dof_u);
    MultAAt( gshape_u, gugv );

    // this contribution is block-diagonal: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,0)->AddMatrix( scale, gugv, dof_u*k, dof_u*k );
    }


    // Non-linear convection (eventually rescaled by dt) --------------------
    scale = ip.weight;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif
    // - term ((w·∇)u,v)
    Vector ugu(dof_u);  
    gshape_u.Mult(Ueval, ugu);
    DenseMatrix uguv(dof_u);
    MultVWt(shape_u, ugu, uguv);
    // -- this contribution is block-diagonal: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,0)->AddMatrix( scale, uguv, dof_u*k, dof_u*k );
    }

    // - term ((u·∇)w,v)
    DenseMatrix uv(dof_u);
    MultVVt(shape_u, uv);
    // -- this is not, so we must loop over its components
    for (int k = 0; k < dim; k++){
      for (int j = 0; j < dim; j++){
        elmats(0,0)->AddMatrix(gUeval(k, j)*scale, uv, dof_u*k, dof_u*j );
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
    // Stiffness (eventually rescaled by dt) --------------------------------
    scale = ip.weight / detJ * (_eta/_mu0);
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
//     // Linearised Lorentz (eventually rescaled by dt) -----------------------
//     DenseMatrix tempuz(dof_u,dof_z);
//     MultVWt(shape_u, shape_z, tempuz);
//     scale = ip.weight / _mu0;
// // #ifdef MULT_BY_DT
//     scale *= _dt;
// // #endif    
//     // -- this contribution is defined block-wise: for each physical dimension, I can just copy it
//     for (int k = 0; k < dim; k++){
//       elmats(0,2)->AddMatrix( gAeval(k)*scale, tempuz, dof_u*k, 0 );
//     }



    //***********************************************************************
    // u,A block
    //***********************************************************************
//     // Linearised Lorentz (eventually rescaled by dt) -----------------------
//     DenseMatrix tempuA(dof_u,dof_a);
//     scale = Zeval * ip.weight / _mu0;
// // #ifdef MULT_BY_DT
//     scale *= _dt;
// // #endif    
//     // -- this contribution is also defined block-wise, so loop
//     for (int k = 0; k < dim; k++){
//       Vector col;
//       gshape_a.GetColumnReference( k, col );
//       MultVWt(shape_u, col, tempuA);
//       elmats(0,3)->AddMatrix( scale, tempuA, dof_u*k, 0 );
//     }


    // This ignores Z:
    // < ∇·∇A ∇B, v > (z is fixed) ------------------------------------------
    Vector dgshape_a(dof_a);
    el[3]->CalcPhysLinLaplacian( Tr, dgshape_a );
    double lAeval = (elfun[3])->operator*(dgshape_a); // ∇·∇A

    DenseMatrix tempuA(dof_u,dof_a);
    scale = lAeval * ip.weight / _mu0;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    // -- this contribution is defined block-wise, so loop
    for (int k = 0; k < dim; k++){
      Vector col;
      gshape_a.GetColumnReference( k, col );
      MultVWt(shape_u, col, tempuA);
      elmats(0,3)->AddMatrix( scale, tempuA, dof_u*k, 0 );
    }

    // < ∇·∇B ∇A, v > (A is fixed) ------------------------------------------
    scale = ip.weight / _mu0;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    MultVWt( shape_u, dgshape_a, tempuA );
    for (int k = 0; k < dim; k++){
      elmats(0,3)->AddMatrix( gAeval(k)*scale, tempuA, dof_u*k, 0 );
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

    // // TODO: change this to the above!!
    // Vector dgshape_a(dof_a);
    // el[3]->CalcPhysLinLaplacian( Tr, dgshape_a );
    // scale = -ip.weight * detJ;
    // AddMult_a_VWt( scale, shape_z, dgshape_a, *(elmats(2,3)) );

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
      // - function evaluations
      Vector wEval(dim);
      _w->Eval(wEval, Tr, ip);
      Vector gcEval(dim);
      _c->GetGridFunction()->GetGradient( Tr, gcEval );
      // - stabilisation parameters
      double tu=0., ta=0.;
      GetTaus( wEval, gcEval, Tr.InverseJacobian(), _dt, _mu, _mu0, _eta, tu, ta );

      // - basis functions evaluations (on reference element)
      shape_Dp.SetSize(dof_p, dim);
      gshape_p.SetSize(dof_p, dim);
      Vector dgshape_u(dof_u);
      Vector dgshape_a(dof_a);
      el[1]->CalcDShape( ip, shape_Dp );
      if ( _include2ndOrd ){
        el[0]->CalcPhysLinLaplacian( Tr, dgshape_u );
        el[3]->CalcPhysLinLaplacian( Tr, dgshape_a );
      }else{
        Zeval     = 0.;
        dgshape_u = 0.;
        dgshape_a = 0.;
      }
      Mult(shape_Dp, adJ, gshape_p);  // NB: this way they are all multiplied by detJ already!



      //***********************************************************************
      // u,u block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*(w·∇)v
      Vector wgv(dof_u);
      gshape_u.Mult(wEval, wgv);  // this is (w·∇)v

      // (w·∇)u term -------------------------------------------------------
      scale = tu * ip.weight / detJ;
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      // -- this contribution is block-diagonal: for each physical dimension, I can just copy it
      DenseMatrix uguwgv(dof_u);
      MultVWt(wgv, ugu, uguwgv);
      for (int k = 0; k < dim; k++){
        elmats(0,0)->AddMatrix( scale, uguwgv, dof_u*k, dof_u*k );
      }
      // (u·∇)w term -------------------------------------------------------
      DenseMatrix uwgv(dof_u);
      MultVWt(wgv, shape_u, uwgv);
      // -- this is not, so we must loop over all of its components
      for (int k = 0; k < dim; k++){
        for (int j = 0; j < dim; j++){
          elmats(0,0)->AddMatrix( scale*gUeval(k, j), uwgv, dof_u*k, dof_u*j );
        }
      }

      // -\mu ∇·∇u term -------------------------------------------------------
      scale = - tu * _mu * ip.weight;
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      // -- this contribution is block-diagonal
      DenseMatrix luwgv(dof_u);
      MultVWt(wgv, dgshape_u, luwgv);
      for (int k = 0; k < dim; k++){
        elmats(0,0)->AddMatrix( scale, luwgv, dof_u*k, dof_u*k );
      }



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
        Vector gpi;
        gshape_p.GetColumnReference(k,gpi);
        DenseMatrix gpwgv(dof_u,dof_p);
        MultVWt(wgv, gpi, gpwgv);
        elmats(0,1)->AddMatrix( scale, gpwgv, dof_u*k, 0 );
      }


      //***********************************************************************
      // u,z block
      //***********************************************************************
      // TODO: should I include anything here?
      // The point is, this would dramatically change the shape of the Jacobian,
      //  adding non-zero blocks to Z. Now, I think this isn't done in Cyr's paper,
      //  the reason being (I assume) because the basis functions are picked so
      //  that the Laplacian of A is zero in the element interior. Here I'm
      //  considering a different formulation, which actually should include some
      //  terms, but hey, whatever: if the eq for j is solved exactly, it should
      //  still be 0?
    //   if ( _include2ndOrd ){
    //     // z ∇C·((w·∇)v) term -------------------------------------------------
    //     scale = tu * ip.weight / _mu0 / detJ;
    // // #ifdef MULT_BY_DT
    //     scale *= _dt;
    //     DenseMatrix zwgv(dof_u,dof_z);
    //     MultVWt(wgv, shape_z, zwgv);
    //     // -- this contribution is defined block-wise
    //     for (int k = 0; k < dim; k++){
    //       elmats(0,2)->AddMatrix( gAeval(k)*scale, zwgv, dof_u*k, 0 );
    //     }

    //   }

      //***********************************************************************
      // u,A block
      //***********************************************************************
      // same here
      if ( _include2ndOrd ){
        // y ∇A·((w·∇)v) term -------------------------------------------------
        scale = tu * ip.weight / _mu0 / detJ;
// #ifdef MULT_BY_DT
        scale *= _dt;
// #endif    
        // -- this contribution is also defined block-wise, so loop
        for (int k = 0; k < dim; k++){
          Vector gAi;
          gshape_a.GetColumnReference( k, gAi );
          DenseMatrix gAwgv(dof_u,dof_a);
          MultVWt(wgv, gAi, gAwgv);
          elmats(0,3)->AddMatrix( lAeval*scale, gAwgv, dof_u*k, 0 );
        }

        // z ∇C·((w·∇)v) term -------------------------------------------------
        scale = tu * ip.weight / _mu0 / detJ;
    // #ifdef MULT_BY_DT
        scale *= _dt;
        DenseMatrix zwgv(dof_u,dof_a);
        MultVWt(wgv, dgshape_a, zwgv);
        // -- this contribution is defined block-wise
        for (int k = 0; k < dim; k++){
          elmats(0,3)->AddMatrix( gAeval(k)*scale, zwgv, dof_u*k, 0 );
        }
      }



      
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
      // (w·∇)u term -------------------------------------------------------
      // I'm breaking symmetry, but that doesn't mean I can't reuse the same code
      //  for the symmetric bits ;)
      scale = tu * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale*= _dt;
  // #endif
      for (int k = 0; k < dim; k++){
        Vector gqi;
        gshape_p.GetColumnReference(k,gqi);
        DenseMatrix ugugq(dof_p,dof_u);
        MultVWt(gqi, ugu, ugugq);
        elmats(1,0)->AddMatrix( scale, ugugq, 0, dof_u*k );
      }

      // (u·∇)w term -------------------------------------------------------
      scale = tu * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale*= _dt;
  // #endif
      DenseMatrix gqgu( dof_p, dim );
      Mult(gshape_p,gUeval,gqgu);
      for (int k = 0; k < dim; k++){
        Vector gqgui;
        gqgu.GetColumnReference(k,gqgui);
        DenseMatrix gqguu( dof_p, dof_u );
        MultVWt(gqgui, shape_u, gqguu);
        elmats(1,0)->AddMatrix( scale, gqguu, 0, dof_u*k );
      }

      // -\mu ∇·∇u term -------------------------------------------------------
      // -- this contribution is block-diagonal
      scale = - tu * _mu * ip.weight; // TODO: scale by detJ?
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      for (int k = 0; k < dim; k++){
        Vector gqi;
        gshape_p.GetColumnReference(k,gqi);
        DenseMatrix gqlu(dof_p,dof_u);
        MultVWt(gqi, dgshape_u, gqlu);
        elmats(1,0)->AddMatrix( scale, gqlu, 0, dof_u*k );
      }




      //***********************************************************************
      // p,z block
      //***********************************************************************
    //   // same considerations as for u,z
    //   if ( _include2ndOrd ){
    //     // z ∇C·∇q term -------------------------------------------------------
    //     scale = tu * ip.weight / _mu0 / detJ;
    // // #ifdef MULT_BY_DT
    //     scale *= _dt;
    //     Vector gAgq(dof_p);
    //     gshape_p.Mult( gAeval, gAgq );
    //     AddMult_a_VWt( scale, gAgq, shape_z, *(elmats(1,2)) );
    //   }

      //***********************************************************************
      // p,A block
      //***********************************************************************
      // same here
      if ( _include2ndOrd ){
        // y ∇A·∇q term -------------------------------------------------------
        scale = tu * ip.weight / _mu0 / detJ;
// #ifdef MULT_BY_DT
        scale *= _dt;
// #endif    
        AddMult_a_ABt( lAeval*scale, gshape_p, gshape_a, *(elmats(1,3)) );


        // z ∇C·∇q term -------------------------------------------------------
        scale = tu * ip.weight / _mu0 / detJ;
    // #ifdef MULT_BY_DT
        scale *= _dt;
    // #endif    
        Vector gAgq(dof_p);
        gshape_p.Mult( gAeval, gAgq );
        AddMult_a_VWt( scale, gAgq, dgshape_a, *(elmats(1,3)) );
      }



      //***********************************************************************
      // z,z block
      //***********************************************************************
      // No stab here
      //***********************************************************************
      // z,A block
      //***********************************************************************
      // No stab here



      //***********************************************************************
      // A,A block
      //***********************************************************************
      // I need to test the residual of the vector potential equation against ta*(w·∇b)
      Vector wgB(dof_a);
      gshape_a.Mult( wEval, wgB ); // this is (w·∇b)

      // w·∇A term ------------------------------------------------------------
      scale = ta * ip.weight / detJ;
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      AddMult_a_VWt(scale, wgB, gAu, *(elmats(3,3)) );

      // -\eta/\mu0 ∇·∇A term ---------------------------------------------------
      scale = - ta * (_eta/_mu0) * ip.weight;
// #ifdef MULT_BY_DT
      scale*= _dt;
// #endif
      AddMult_a_VWt(scale, wgB, dgshape_a, *(elmats(3,3)) );



      //***********************************************************************
      // A,u block
      //***********************************************************************
      // u·∇c term ------------------------------------------------------------
      scale = ta * ip.weight / detJ;
  // #ifdef MULT_BY_DT
      scale *= _dt;
      DenseMatrix ugcwgb(dof_a,dof_u);
      MultVWt(wgB, shape_u, ugcwgb);
  // #endif    
      // -- this contribution is defined block-wise
      for (int k = 0; k < dim; k++){
        elmats(3,0)->AddMatrix( gAeval(k)*scale, ugcwgb, 0, dof_u*k );
      }



    }


  }


//   // finally rescale everything by dt
// // #ifdef MULT_BY_DT
//   elmats(0,0)->operator*=(_dt);
//   elmats(0,2)->operator*=(_dt);
//   elmats(0,3)->operator*=(_dt);
//   elmats(1,0)->operator*=(_dt);
//   elmats(3,0)->operator*=(_dt);
//   elmats(3,3)->operator*=(_dt);
//   if ( _stab ){
//     elmats(0,1)->operator*=(_dt);
//     if ( _include2ndOrd ){
//       elmats(1,2)->operator*=(_dt);
//       elmats(1,3)->operator*=(_dt);
//     }
//   }
// // #endif



  //***********************************************************************
  // u,p block - reprise
  //***********************************************************************
  // Negative Gradient (eventually rescaled by dt) ------------------------
  // NB: including stabilisation breaks symmetry!! So this can be done only if no stabilisation is prescribed
  if ( !_stab ){
    elmats(0,1)->Transpose( *(elmats(1,0)) );
  }



}













void IncompressibleMHD2DSpaceIntegrator::AssembleElementVector(
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
    MultAtB(uCoeff, gshape_u, gUeval);             // ∇u
    gshape_a.MultTranspose( *(elfun[3]), gAeval ); // ∇A
    double gAu   = gAeval * Ueval;                 // u·∇A
    gUeval.Mult(Ueval, ugu);                       // (u·∇)u
    double du    = gUeval.Trace();                 // ∇·u

    double scale=0.;


    //***********************************************************************
    // u block
    //***********************************************************************
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
//     scale = ip.weight / _mu0;
// // #ifdef MULT_BY_DT
//     scale *= _dt;
// // #endif    
//     // -- this contribution is defined block-wise: for each physical dimension, I can just copy it
//     for (int k = 0; k < dim; k++){
//       for ( int j = 0; j < dof_u; ++j ){
//         elvecs[0]->operator()(j+k*dof_u) += shape_u(j) * scale * Zeval * gAeval(k);
//       }
//     }

    // attempt 2: without considering Z - SAVE THIS!
    Vector dgshape_a(dof_a);
    el[3]->CalcPhysLinLaplacian( Tr, dgshape_a );
    double lAeval = (elfun[3])->operator*(dgshape_a); // ∇·∇A
    scale = ip.weight / _mu0;
// #ifdef MULT_BY_DT
    scale *= _dt;
// #endif    
    // -- this contribution is defined block-wise: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      for ( int j = 0; j < dof_u; ++j ){
        elvecs[0]->operator()(j+k*dof_u) += shape_u(j) * scale * lAeval * gAeval(k);
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

    // TODO: change me to the part above
    // Vector dgshape_a(dof_a);
    // el[3]->CalcPhysLinLaplacian( Tr, dgshape_a );
    // double lAeval = (elfun[3])->operator*(dgshape_a); // ∇·∇A
    // scale = ip.weight * detJ;
    // elvecs[2]->Add( -scale*lAeval, shape_z );




    //***********************************************************************
    // A block
    //***********************************************************************
    // Stiffness (eventually rescaled by dt) --------------------------------
    Vector tempAA(dof_a);
    gshape_a.Mult(gAeval, tempAA);
    scale = ip.weight / detJ * (_eta/_mu0);
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

      // - function evaluations
      Vector wEval(dim);
      _w->Eval(wEval, Tr, ip);
      Vector gcEval(dim);
      _c->GetGridFunction()->GetGradient( Tr, gcEval );
      //  -- for rhs
      Vector fEval(dim);
      _f->Eval(fEval, Tr, ip);
      double hEval = _h->Eval(Tr, ip);

      // - stabilisation parameters
      double tu=0., ta=0.;
      GetTaus( wEval, gcEval, Tr.InverseJacobian(), _dt, _mu, _mu0, _eta, tu, ta );

      // - basis functions evaluations (on reference element)
      DenseMatrix shape_Dp(dof_p, dim);
      DenseMatrix gshape_p(dof_p, dim);
      Vector dgshape_u(dof_u);
      Vector dgshape_a(dof_a);
      el[1]->CalcDShape( ip, shape_Dp );
      el[0]->CalcPhysLinLaplacian( Tr, dgshape_u ); // this is the laplacian in *physical* coords, so no need to rescale by detJ
      el[3]->CalcPhysLinLaplacian( Tr, dgshape_a );
      double lAeval;
      Vector lueval(dim);
      if ( _include2ndOrd ){
        uCoeff.MultTranspose( dgshape_u, lueval);  // ∇·∇u
        lAeval = (elfun[3])->operator*(dgshape_a); // ∇·∇A
      }else{
        Zeval  = 0.;
        lAeval = 0.;
        lueval = 0.;
      }
      Mult(shape_Dp, adJ, gshape_p);  // NB: this way they are all multiplied by detJ already!
      Vector gpeval(dim);
      gshape_p.MultTranspose( *(elfun[1]), gpeval ); // ∇p


      //***********************************************************************
      // u block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*(w·∇)v
      Vector wgv(dof_u);
      gshape_u.Mult( wEval, wgv ); // this is (w·∇)v

      Vector resU(dim);
      for ( int k = 0; k < dim; ++k ){
        //                         (u·∇)u       -  mu ∇·∇u      + ∇p             + z∇A/mu0                   - f
        // resU(k) = tu * ip.weight *( ugu(k)/detJ - _mu*lueval(k) + gpeval(k)/detJ + Zeval*gAeval(k)/_mu0/detJ - fEval(k) );
        resU(k) = tu * ip.weight *( ugu(k)/detJ - _mu*lueval(k) + gpeval(k)/detJ + lAeval*gAeval(k)/_mu0/detJ - fEval(k) );
      }
// #ifdef MULT_BY_DT
      resU *= _dt;
// #endif


      // -- this contribution is made block-wise: loop over physical dimensions
      for (int k = 0; k < dim; k++){
        for ( int j = 0; j < dof_u; ++j ){
          elvecs[0]->operator()(j+k*dof_u) += wgv(j) * resU(k);
        }
      }



      //***********************************************************************
      // p block
      //***********************************************************************
      // I need to test the residual of the momentum equation against tu*∇q ***
      Vector tempp(dof_p);
      gshape_p.Mult( resU, tempp );
      elvecs[1]->operator+=(tempp);



      //***********************************************************************
      // A block
      //***********************************************************************
      // I need to test the residual of the vector potential equation against ta*w·∇b
      gshape_a.Mult( wEval, tempAA); // this is w·∇b
      //                              u·∇A     -   eta/ mu0  ∇·∇A   - h
      double resA = ta * ip.weight *( gAu/detJ - (_eta/_mu0)*lAeval - hEval );
// #ifdef MULT_BY_DT
      resA *= _dt;
// #endif

      elvecs[3]->Add( resA, tempAA );

    }


  }

}






void IncompressibleMHD2DSpaceIntegrator::AssembleFaceGrad(
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








void IncompressibleMHD2DSpaceIntegrator::AssembleFaceVector(
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











void IncompressibleMHD2DSpaceIntegrator::GetTaus( const Vector& u, const Vector& gA, const DenseMatrix& Jinv,
                                                  double dt, double mu, double mu0, double eta,
                                                  double& tauU, double& tauA){

  double B2 = gA*gA; // B is [dyA, -dxA], so |B| and |gradA| coincide

  DenseMatrix GC;
  GetGC( Jinv, GC );
  double GCn = GC.FNorm();

  // with rho = 1, I could simplify...
  // tauU = 4./(dt*dt) + GC.InnerProduct(u.GetData(),u.GetData()) + C1*C1* mu* mu*GCn*GCn + C2*C2/mu0*B2*GCn; // 1/dt^2 since 1st order method is used
  // tauA = 4./(dt*dt) + GC.InnerProduct(u.GetData(),u.GetData()) + C1*C1*eta*eta*GCn*GCn;
  tauU = 1./(dt*dt) + GC.InnerProduct(u.GetData(),u.GetData()) + C1*C1* mu* mu*GCn*GCn + C2*C2/mu0*B2*GCn;
  tauA = 1./(dt*dt) + GC.InnerProduct(u.GetData(),u.GetData()) + C1*C1*eta*eta*GCn*GCn;
  tauU = 1./sqrt(tauU);
  tauA = 1./sqrt(tauA);

}






void IncompressibleMHD2DSpaceIntegrator::GetGC( const DenseMatrix& Jinv, DenseMatrix& GC ){
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










const IntegrationRule& IncompressibleMHD2DSpaceIntegrator::GetRule(const Array<const FiniteElement *> &el,
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
  // if ( !_stab ){
  //   ords[0] = 2*ordU + ordGU;         // ( (u·∇)u, v )
  //   ords[1] =   ordU + ordGA + ordZ;  // (   z ∇A, v )
  //   ords[2] =   ordU + ordGA + ordA;  // ( (u·∇A), B )
  // }else{
    ords[0] = 2*ordU + 2*ordGU;                  // ( (u·∇)u, (w·∇)v )
    ords[1] =   ordU +   ordGU +   ordGA + ordZ; // (   z ∇A, (w·∇)v )
    ords[2] = 2*ordU           + 2*ordGA;        // ( (u·∇A),  w·∇B )    
  // }



  // std::cout<<"Selecting integrator of order "<<ords.Max()<<std::endl;

  // TODO: this is overkill. I should prescribe different accuracies for each component of the integrator!
  return IntRules.Get( el[0]->GetGeomType(), ords.Max() );

}





} // namespace mfem