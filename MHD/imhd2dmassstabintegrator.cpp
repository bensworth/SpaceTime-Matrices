#include "imhd2dmassstabintegrator.hpp"

namespace mfem{

void IncompressibleMHD2DMassStabIntegrator::AssembleElementGrad(
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
  shape_a.SetSize(dof_a);
  shape_Du.SetSize(dof_u, dim);
  shape_Dp.SetSize(dof_p, dim);
  shape_Da.SetSize(dof_a, dim);

  gshape_u.SetSize(dof_u, dim);
  gshape_p.SetSize(dof_p, dim);
  gshape_a.SetSize(dof_a, dim);


  const IntegrationRule& ir = GetRule(el, Tr);

  for (int i = 0; i < ir.GetNPoints(); ++i){

    // compute quantities specific to this integration point
    // - Jacobian-related
    const IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    CalcAdjugate(Tr.Jacobian(), adJ);

    // - basis functions evaluations (on reference element)
    el[0]->CalcShape(  ip, shape_u  );
    el[3]->CalcShape(  ip, shape_a  );
    el[0]->CalcDShape( ip, shape_Du );
    el[1]->CalcDShape( ip, shape_Dp );
    el[3]->CalcDShape( ip, shape_Da );
    // - basis functions gradient evaluations (on physical element)
    Mult(shape_Du, adJ, gshape_u);  // NB: this way they are all multiplied by detJ already!
    Mult(shape_Dp, adJ, gshape_p);
    Mult(shape_Da, adJ, gshape_a);
    // - function evaluations
    Vector wEval(dim);
    _w->Eval(wEval, Tr, ip);
    Vector gcEval(dim);
    _c->GetGridFunction()->GetGradient( Tr, gcEval );
    // gshape_a.MultTranspose( *(elfun[3]), gAeval ); // ∇A

    double tu=0., ta=0.;
    GetTaus( wEval, gcEval, Tr.InverseJacobian(), tu, ta );

    double scale =0.;




    //***********************************************************************
    // u,u block
    //***********************************************************************
    // I need to test the residual of the momentum equation against tu*(w·∇)v
    // dtu term -------------------------------------------------------------
    scale = tu * ip.weight;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
    DenseMatrix tempuu(dof_u);
    Vector wgu(dof_u);
    gshape_u.Mult(wEval, wgu);
    MultVWt(wgu, shape_u, tempuu);
    tempuu *= scale;
    // -- this contribution is block-diagonal: for each physical dimension, I can just copy it
    for (int k = 0; k < dim; k++){
      elmats(0,0)->AddMatrix( tempuu, dof_u*k, dof_u*k );
    }


    //***********************************************************************
    // p,u block
    //***********************************************************************
    // I need to test the residual of the momentum equation against tu*∇q ***
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

    //***********************************************************************
    // A,A block
    //***********************************************************************
    // I need to test the residual of the vector potential equation against ta*(w·∇b)
    // dtA term -------------------------------------------------------------
    Vector gAw(dof_a);
    gshape_a.Mult( wEval, gAw );
    scale = ta * ip.weight;
// #ifndef MULT_BY_DT
//     scale*= 1./_dt;
// #endif
    AddMult_a_VWt(scale, gAw, shape_a, *(elmats(3,3)) );



  }


  // multiply everything by -1.
  for ( int i = 0; i < 4; ++i ){
    for ( int j = 0; j < 4; ++j ){
      elmats(i,j)->Neg();
    }
  }

}













void IncompressibleMHD2DMassStabIntegrator::AssembleElementVector(
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
  shape_a.SetSize(dof_a);
  shape_Du.SetSize(dof_u, dim);
  shape_Dp.SetSize(dof_p, dim);
  shape_Da.SetSize(dof_a, dim);

  gshape_u.SetSize(dof_u, dim);
  gshape_p.SetSize(dof_p, dim);
  gshape_a.SetSize(dof_a, dim);

  uCoeff.UseExternalData((elfun[0])->GetData(), dof_u, dim);  // coefficients of basis functions for u


  const IntegrationRule& ir = GetRule(el, Tr);

  for (int i = 0; i < ir.GetNPoints(); ++i){
    // compute quantities specific to this integration point
    // - Jacobian-related
    const IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    const DenseMatrix adJ( Tr.AdjugateJacobian() );
    // - basis functions evaluations (on reference element)
    el[0]->CalcShape(  ip, shape_u  );
    el[3]->CalcShape(  ip, shape_a  );
    el[0]->CalcDShape( ip, shape_Du );
    el[1]->CalcDShape( ip, shape_Dp );
    el[3]->CalcDShape( ip, shape_Da );
    // - basis functions gradient evaluations (on physical element)
    Mult(shape_Du, adJ, gshape_u);  // NB: this way they are all multiplied by detJ already!
    Mult(shape_Dp, adJ, gshape_p);
    Mult(shape_Da, adJ, gshape_a);
    // - function evaluations
    Vector wEval(dim), gcEval(dim);
    _w->Eval(wEval, Tr, ip);
    _c->GetGridFunction()->GetGradient( Tr, gcEval );

    Vector uEval(dim);
    uCoeff.MultTranspose(shape_u, uEval);          // u
    double AEval = (elfun[3])->operator*(shape_a); // A

    double tu=0., ta=0.;
    GetTaus( wEval, gcEval, Tr.InverseJacobian(), tu, ta );

    double scale =0.;



    //***********************************************************************
    // u block
    //***********************************************************************
    Vector tempu(dof_u);
    // I need to test the residual of the momentum equation against tu*(w·∇)v
    gshape_u.Mult( wEval, tempu ); // this is (w·∇)v
    // dtu ------------------------------------------------------------------
    scale = tu * ip.weight;
//   #ifndef MULT_BY_DT
//       scale*= 1./_dt;
//   #endif
    // -- this contribution is made block-wise: loop over physical dimensions
    for (int k = 0; k < dim; k++){
      for ( int j = 0; j < dof_u; ++j ){
        elvecs[0]->operator()(j+k*dof_u) += tempu(j) * scale * uEval(k);
      }
    }

    //***********************************************************************
    // p block
    //***********************************************************************
    // I need to test the residual of the momentum equation against tu*∇q ***
    Vector tempp(dof_p);
    // dtu ------------------------------------------------------------------
    scale = tu * ip.weight;
// #ifndef MULT_BY_DT
//       scale *= 1/_dt;
// #endif
    gshape_p.Mult( uEval,  tempp );
    elvecs[1]->Add( scale, tempp );



    //***********************************************************************
    // A block
    //***********************************************************************
    // I need to test the residual of the vector potential equation against ta*w·∇b
    Vector tempAA(dof_a);
    gshape_a.Mult( wEval, tempAA); // this is w·∇b
    // dtA ------------------------------------------------------------------
    scale = ta * ip.weight;
// #ifndef MULT_BY_DT
//       scale *= 1/_dt;
// #endif
    elvecs[3]->Add( scale*AEval, tempAA );

  }


  // multiply everything by -1.
  for ( int i = 0; i < 4; ++i ){
    elvecs[i]->Neg();
  }


}














void IncompressibleMHD2DMassStabIntegrator::GetTaus(const Vector& u, const Vector& gA,
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






void IncompressibleMHD2DMassStabIntegrator::GetGC( const DenseMatrix& Jinv, DenseMatrix& GC ) const{
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










const IntegrationRule& IncompressibleMHD2DMassStabIntegrator::GetRule(const Array<const FiniteElement *> &el,
                                                                      ElementTransformation &Tr){

  // Find an integration rule which is accurate enough
  const int ordU  = el[0]->GetOrder();
  const int ordA  = el[3]->GetOrder();
  const int ordGU = Tr.OrderGrad(el[0]);
  const int ordGP = Tr.OrderGrad(el[1]);
  const int ordGA = Tr.OrderGrad(el[3]);

  // I check every term in the equations, see which poly order is reached, and take the max. These give, in order
  // - 2*ordU, 2*ordU + (ordGU), 2*(ordGU), ordGU + ordP, ordZ + ordGA + ordU,
  // - ordP + ordGU
  // - ordGA+ordGZ, 2*ordZ
  // - 2*ordA, ordU+(ordGA)+ordA,2*(ordGA)
  //  and it's likely that the non-linear terms will give the biggest headaches, so:
  Array<int> ords(3);
  ords[0] = 2*ordU + ordGU;        // ( u, (w·∇)v )
  ords[1] = 2*ordU + ordGP;        // ( u,    ∇q  )
  ords[2] =   ordU + ordGA + ordA; // ( A,  w·∇B  )    



  // std::cout<<"Selecting integrator of order "<<ords.Max()<<std::endl;

  // TODO: this is overkill. I should prescribe different accuracies for each component of the integrator!
  return IntRules.Get( el[0]->GetGeomType(), ords.Max() );

}





} // namespace mfem