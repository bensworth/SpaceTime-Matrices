function mats = readOperators( filename )
% ReadMatrices
myfilename =strcat( filename,'Mz.dat' );
mats.Mz = spconvert( load( myfilename ) );

myfilename =strcat( filename,'K.dat' );
mats.K = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Mp.dat' );
mats.Mp = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Ap.dat' );
mats.Ap = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Aa.dat' );
mats.Aa = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Mu.dat' );
mats.Mu = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Ma.dat' );
mats.Ma = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Mps.dat' );
mats.Mps = spconvert( load( myfilename ) );

myfilename =strcat( filename,'B.dat' );
mats.B = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Bt.dat' );
mats.Bt = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Cs.dat' );
mats.Cs = spconvert( load( myfilename ) );

myfilename =strcat( filename,'X1.dat' );
mats.X1 = spconvert( load( myfilename ) );

myfilename =strcat( filename,'X2.dat' );
mats.X2 = spconvert( load( myfilename ) );


myfilename =strcat( filename,'Fu.dat' );
mats.Fu = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Fa.dat' );
mats.Fa = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Z1.dat' );
mats.Z1 = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Z2.dat' );
mats.Z2 = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Y.dat' );
mats.Y = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Wp.dat' );
mats.Wp = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Wa.dat' );
mats.Wa = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Cp.dat' );
mats.Cp = spconvert( load( myfilename ) );


% Dirichlet nodes
mats.essU =load(strcat( filename,'essU.dat' ));
mats.essP =load(strcat( filename,'essP.dat' ));
mats.essA =load(strcat( filename,'essA.dat' ));


mats.zz00 = sparse(size(mats.Fu,1), size(mats.Fu,2));
mats.zz01 = sparse(size(mats.Bt,1), size(mats.Bt,2));
mats.zz02 = sparse(size(mats.Z1,1), size(mats.Z1,2));
mats.zz03 = sparse(size(mats.Z2,1), size(mats.Z2,2));
mats.zz10 = sparse(size(mats.B,1),  size(mats.B,2));
mats.zz11 = sparse(size(mats.Cs,1), size(mats.Cs,2));
mats.zz12 = sparse(size(mats.X1,1), size(mats.X1,2));
mats.zz13 = sparse(size(mats.X2,1), size(mats.X2,2));
mats.zz20 = sparse(size(mats.Mz,1), size(mats.Fu,2));
mats.zz21 = sparse(size(mats.Mz,1), size(mats.Bt,2));
mats.zz22 = sparse(size(mats.Mz,1), size(mats.Mz,2));
mats.zz23 = sparse(size(mats.K,1),  size(mats.K,2));
mats.zz30 = sparse(size(mats.Y,1),  size(mats.Y,2));
mats.zz31 = sparse(size(mats.Fa,1), size(mats.Bt,2));
mats.zz32 = sparse(size(mats.Fa,1), size(mats.Z1,2));
mats.zz33 = sparse(size(mats.Fa,1), size(mats.Fa,2));

mats.A=[ mats.Fu,   mats.Bt,   mats.Z1,   mats.Z2; ...
         mats.B,    mats.Cs,   mats.X1,   mats.X2; ...
         mats.zz20, mats.zz21, mats.Mz,   mats.K; ...
         mats.Y,    mats.zz32, mats.zz32, mats.Fa];



mats.At= [ mats.Fu,   mats.zz00, mats.Bt,   mats.zz01, mats.Z1,   mats.zz02, mats.Z2,   mats.zz03 ;...
           mats.Mu,   mats.Fu,   mats.zz01, mats.Bt,   mats.zz02, mats.Z1,   mats.zz03, mats.Z2   ;...
           mats.B,    mats.zz10, mats.Cs,   mats.zz11, mats.X1,   mats.zz12, mats.X2,   mats.zz13 ; ...
           mats.Mps,  mats.B,    mats.zz11, mats.Cs,   mats.zz12, mats.X1,   mats.zz13, mats.X2   ; ...
           mats.zz20, mats.zz20, mats.zz21, mats.zz21, mats.Mz,   mats.zz22, mats.K,    mats.zz23 ; ...
           mats.zz20, mats.zz20, mats.zz21, mats.zz21, mats.zz22  mats.Mz,   mats.zz23, mats.K    ; ...
           mats.Y,    mats.zz30, mats.zz31, mats.zz31, mats.zz32, mats.zz32, mats.Fa,   mats.zz33 ; ...
           mats.zz30, mats.Y,    mats.zz31, mats.zz31, mats.zz32, mats.zz32, mats.Ma,   mats.Fa  ];

mats.Att=[ mats.Fu,   mats.zz00, mats.zz00, mats.Bt,   mats.zz01, mats.zz01, mats.Z1,   mats.zz02, mats.zz02, mats.Z2,   mats.zz03, mats.zz03 ; ...
           mats.Mu,   mats.Fu,   mats.zz00, mats.zz01, mats.Bt,   mats.zz01, mats.zz02, mats.Z1,   mats.zz02, mats.zz03, mats.Z2,   mats.zz03 ; ...
           mats.zz00, mats.Mu,   mats.Fu,   mats.zz01, mats.zz01, mats.Bt,   mats.zz02, mats.zz02, mats.Z1,   mats.zz03, mats.zz03, mats.Z2   ; ...
           mats.B,    mats.zz10, mats.zz10, mats.Cs,   mats.zz11, mats.zz11, mats.X1,   mats.zz12, mats.zz12, mats.X2,   mats.zz13, mats.zz13 ; ...
           mats.Mps,  mats.B,    mats.zz10, mats.zz11, mats.Cs,   mats.zz11, mats.zz12, mats.X1,   mats.zz12, mats.zz13, mats.X2,   mats.zz13 ; ...
           mats.zz10, mats.Mps,  mats.B,    mats.zz11, mats.zz11, mats.Cs,   mats.zz12, mats.zz12, mats.X1,   mats.zz13, mats.zz13, mats.X2   ; ...  
           mats.zz20, mats.zz20, mats.zz20, mats.zz21, mats.zz21, mats.zz21, mats.Mz,   mats.zz22, mats.zz22, mats.K,    mats.zz23, mats.zz23 ; ...
           mats.zz20, mats.zz20, mats.zz20, mats.zz21, mats.zz21, mats.zz21, mats.zz22, mats.Mz,   mats.zz22, mats.zz23, mats.K,    mats.zz23 ; ...
           mats.zz20, mats.zz20, mats.zz20, mats.zz21, mats.zz21, mats.zz21, mats.zz22, mats.zz22, mats.Mz,   mats.zz23, mats.zz23, mats.K    ; ...
           mats.Y,    mats.zz30, mats.zz30, mats.zz31, mats.zz31, mats.zz31, mats.zz32, mats.zz32, mats.zz32, mats.Fa,   mats.zz33, mats.zz33 ; ...
           mats.zz30, mats.Y,    mats.zz30, mats.zz31, mats.zz31, mats.zz31, mats.zz32, mats.zz32, mats.zz32, mats.Ma,   mats.Fa,   mats.zz33 ; ...
           mats.zz30, mats.zz30, mats.Y,    mats.zz31, mats.zz31, mats.zz31, mats.zz32, mats.zz32, mats.zz32, mats.zz33, mats.Ma,   mats.Fa   ];

