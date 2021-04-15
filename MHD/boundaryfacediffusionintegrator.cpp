#include "boundaryfacediffusionintegrator.hpp"
#include <iostream>

namespace mfem{



void BoundaryFaceDiffusionIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
  											  const FiniteElement &test_fe1, const FiniteElement &test_fe2,
  											  FaceElementTransformations &Trans, DenseMatrix &elmat){

  if (Trans.Elem2No >= 0){
    MFEM_ABORT("AssembleFaceMatrix (mixed form) only works for boundary faces");
  }

  int dim, dof_a, dof_z;
  double w;

  dof_a = trial_fe.GetDof();
  dim   = trial_fe.GetDim();
  dof_z = test_fe1.GetDof();

  nor.SetSize(dim);
  nh.SetSize(dim);
  ni.SetSize(dim);
  adjJ.SetSize(dim);
  if (MQ){
    mq.SetSize(dim);
  }

  shape_z.SetSize(dof_z);
  shape_Da.SetSize(dof_a, dim);
  shape_Dan.SetSize(dof_a);

  elmat.SetSize(dof_z,dof_a);
  elmat = 0.0;


  const IntegrationRule *ir = IntRule;
  if (ir == NULL){
    // a simple choice for the integration order; is this OK?
    int order = trial_fe.GetOrder() + test_fe1.GetOrder();
    // ir = &IntRules.Get(Trans.GetGeometryType(), order);
    ir = &IntRules.Get(Trans.FaceGeom, order);
  }

  // assemble: < {(Q \nabla A).n},z >
  for (int p = 0; p < ir->GetNPoints(); p++){
    const IntegrationPoint &ip = ir->IntPoint(p);

    IntegrationPoint eip1;
    Trans.Loc1.Transform(ip, eip1);
    Trans.Face->SetIntPoint(&ip);


    // // Set the integration point in the face and the neighboring elements
    // Trans.SetAllIntPoints(&ip);

    // // Access the neighboring elements' integration points
    // const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

    if (dim == 1){
      nor(0) = 2*eip1.x - 1.0;
    }else{
      // CalcOrtho(Trans.Jacobian(), nor);
      CalcOrtho(Trans.Face->Jacobian(), nor);
    }

    trial_fe.CalcDShape(eip1, shape_Da);
    test_fe1.CalcShape(eip1,  shape_z);

    Trans.Elem1->SetIntPoint(&eip1);
    w = ip.weight/Trans.Elem1->Weight();
    if (!MQ){
      if (Q){
        w *= Q->Eval(*Trans.Elem1, eip1);
      }
      ni.Set(w, nor);
    }else{
      nh.Set(w, nor);
      MQ->Eval(mq, *Trans.Elem1, eip1);
      mq.MultTranspose(nh, ni);
    }
    CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
    adjJ.Mult(ni, nh);

    shape_Da.Mult(nh, shape_Dan);
    AddMult_a_VWt( 1., shape_z, shape_Dan, elmat );

    // for (int i = 0; i < ndof1; i++){
    //   for (int j = 0; j < ndof1; j++){
    //     elmat(i, j) += shape1(i) * dshape1dn(j);
    //   }
    // }


  }



  // // elmat := -elmat + sigma*elmat^t + jmat
  // for (int i = 0; i < ndofs; i++){
  //   for (int j = 0; j < i; j++){
  //     double aij = elmat(i,j), aji = elmat(j,i);
  //     elmat(i,j) = sigma*aji - aij;
  //     elmat(j,i) = sigma*aij - aji;
  //   }
  //   elmat(i,i) *= (sigma - 1.);
  // }
}













// void BoundaryFaceDiffusionIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,const FiniteElement &test_fe,
// 																	   											   ElementTransformation &Trans, DenseMatrix &elmat){


//   int dim, dof_a, dof_z;
//   double w;

//   dim   = trial_fe.GetDim();
//   dof_a = trial_fe.GetDof();
//   dof_z =  test_fe.GetDof();

//   nor.SetSize(dim);
//   nh.SetSize(dim);
//   ni.SetSize(dim);
//   adjJ.SetSize(dim);
//   if (MQ){
//     mq.SetSize(dim);
//   }

//   shape_z.SetSize(dof_z);
//   shape_Da.SetSize(dof_a, dim);
//   shape_Dan.SetSize(dof_a);

//   elmat.SetSize(dof_z,dof_a);
//   elmat = 0.0;


//   const IntegrationRule *ir = IntRule;
//   if (ir == NULL){
//     // a simple choice for the integration order; is this OK?
//     int order = trial_fe.GetOrder() + test_fe.GetOrder();
//     ir = &IntRules.Get(Trans.GetGeometryType(), order);
//   }

//   // assemble: < {(Q \nabla A).n},z >
//   for (int p = 0; p < ir->GetNPoints(); p++){
//     const IntegrationPoint &ip = ir->IntPoint(p);
//     Trans.SetIntPoint(&ip);

//     const DenseMatrix adJ( Trans.AdjugateJacobian() );
//     const double detJ = Trans.Weight();
//     CalcOrtho(Trans.Jacobian(), nor);

//     trial_fe.CalcDShape(ip, shape_Da);
//     test_fe.CalcShape(ip,   shape_z);


//     w = ip.weight;
//     // if (!MQ){
//       // if (Q){
//       //   w *= Q->Eval(*Trans.Elem1, ip);
//       // }
//       ni.Set(w, nor);
//     // }else{
//     //   nh.Set(w, nor);
//     //   MQ->Eval(mq, *Trans.Elem1, eip1);
//     //   mq.MultTranspose(nh, ni);
//     // }
//     CalcAdjugate(Trans.Jacobian(), adjJ);
//     adjJ.Mult(ni, nh);

//     shape_Da.Mult(nh, shape_Dan);
//     AddMult_a_VWt( 1., shape_z, shape_Dan, elmat );


//     std::cout<<"Point "<< p <<std::endl;
//     std::cout<<"nor "; nor.Print(); std::cout<<std::endl;
//     std::cout<<"ni  "; ni.Print(); std::cout<<std::endl;
//     std::cout<<"nh  "; nh.Print(); std::cout<<std::endl;
//     std::cout<<"det "<<detJ<<std::endl;
//     std::cout<<"Da  "; shape_Da.Print(); std::cout<<std::endl;
//     std::cout<<"Dan "; shape_Dan.Print(); std::cout<<std::endl;
//     std::cout<<"z   "; shape_z.Print(); std::cout<<std::endl;


//     // for (int i = 0; i < ndof1; i++){
//     //   for (int j = 0; j < ndof1; j++){
//     //     elmat(i, j) += shape1(i) * dshape1dn(j);
//     //   }
//     // }


//   }

//   std::cout<<"elmat  "; elmat.Print(mfem::out,dof_a); std::cout<<std::endl;



//   // // elmat := -elmat + sigma*elmat^t + jmat
//   // for (int i = 0; i < ndofs; i++){
//   //   for (int j = 0; j < i; j++){
//   //     double aij = elmat(i,j), aji = elmat(j,i);
//   //     elmat(i,j) = sigma*aji - aij;
//   //     elmat(j,i) = sigma*aij - aji;
//   //   }
//   //   elmat(i,i) *= (sigma - 1.);
//   // }
// }





// void BoundaryFaceDiffusionIntegrator::AssembleFaceMatrix(const FiniteElement &trial_face_fe,
//   											  const FiniteElement &test_fe1, const FiniteElement &test_fe2,
//   											  FaceElementTransformations &Trans, DenseMatrix &elmat){

//   if (Trans.Elem2No >= 0){
//     MFEM_ABORT("AssembleFaceMatrix (mixed form) only works for boundary faces");
//   }

//   int dim, dof_a, dof_z;
//   double w;

//   dim   = trial_face_fe.GetDim();
//   dof_a = trial_face_fe.GetDof();
//   dof_z = test_fe1.GetDof();

//   nor.SetSize(dim);
//   nh.SetSize(dim);
//   ni.SetSize(dim);
//   adjJ.SetSize(dim);
//   if (MQ){
//     mq.SetSize(dim);
//   }

//   shape_z.SetSize(dof_z);
//   shape_Da.SetSize(dof_a, dim);
//   shape_Dan.SetSize(dof_a);

//   elmat.SetSize(dof_z,dof_a);
//   elmat = 0.0;


//   const IntegrationRule *ir = IntRule;
//   if (ir == NULL){
//     // a simple choice for the integration order; is this OK?
//     int order = trial_face_fe.GetOrder() + test_fe1.GetOrder();
//     // ir = &IntRules.Get(Trans.GetGeometryType(), order);
//     ir = &IntRules.Get(Trans.FaceGeom, order);
//   }

//   // assemble: < {(Q \nabla A).n},z >
//   for (int p = 0; p < ir->GetNPoints(); p++){
//     const IntegrationPoint &ip = ir->IntPoint(p);

//     IntegrationPoint eip1;
//     Trans.Loc1.Transform(ip, eip1);
//     Trans.Face->SetIntPoint(&ip);


//     // // Set the integration point in the face and the neighboring elements
//     // Trans.SetAllIntPoints(&ip);

//     // // Access the neighboring elements' integration points
//     // const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

//     if (dim == 1){
//       nor(0) = 2*eip1.x - 1.0;
//     }else{
//       // CalcOrtho(Trans.Jacobian(), nor);
//       CalcOrtho(Trans.Face->Jacobian(), nor);
//     }

//     trial_face_fe.CalcDShape(eip1, shape_Da);
//     test_fe1.CalcShape(eip1, shape_z);

//     Trans.Elem1->SetIntPoint(&eip1);
//     w = ip.weight/Trans.Elem1->Weight();
//     if (!MQ){
//       if (Q){
//         w *= Q->Eval(*Trans.Elem1, eip1);
//       }
//       ni.Set(w, nor);
//     }else{
//       nh.Set(w, nor);
//       MQ->Eval(mq, *Trans.Elem1, eip1);
//       mq.MultTranspose(nh, ni);
//     }
//     CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
//     adjJ.Mult(ni, nh);

//     shape_Da.Mult(nh, shape_Dan);
//     AddMult_a_VWt( 1., shape_z, shape_Dan, elmat );


//     std::cout<<"Point "<< p <<std::endl;
//     std::cout<<"ni  "; ni.Print(); std::cout<<std::endl;
//     std::cout<<"nh  "; nh.Print(); std::cout<<std::endl;
//     std::cout<<"Da  "; shape_Da.Print(); std::cout<<std::endl;
//     std::cout<<"Dan "; shape_Dan.Print(); std::cout<<std::endl;
//     std::cout<<"z   "; shape_z.Print(); std::cout<<std::endl;


//     // for (int i = 0; i < ndof1; i++){
//     //   for (int j = 0; j < ndof1; j++){
//     //     elmat(i, j) += shape1(i) * dshape1dn(j);
//     //   }
//     // }


//   }

//   std::cout<<"elmat  "; elmat.Print(mfem::out,dof_a); std::cout<<std::endl;



//   // // elmat := -elmat + sigma*elmat^t + jmat
//   // for (int i = 0; i < ndofs; i++){
//   //   for (int j = 0; j < i; j++){
//   //     double aij = elmat(i,j), aji = elmat(j,i);
//   //     elmat(i,j) = sigma*aji - aij;
//   //     elmat(j,i) = sigma*aij - aji;
//   //   }
//   //   elmat(i,i) *= (sigma - 1.);
//   // }
// }


}