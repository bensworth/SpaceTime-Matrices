% ReadMatrices
filename = '_ugaOrig_';


myfilename =strcat( filename,'Mz.dat' );
matsOr.Mz = spconvert( load( myfilename ) );

myfilename =strcat( filename,'K.dat' );
matsOr.K = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Mp.dat' );
matsOr.Mp = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Ap.dat' );
matsOr.Ap = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Aa.dat' );
matsOr.Aa = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Fu.dat' );
matsOr.Mu = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Ma.dat' );
matsOr.Ma = spconvert( load( myfilename ) );

myfilename =strcat( filename,'B.dat' );
matsOr.B = spconvert( load( myfilename ) );

matsOr.Mps = sparse(size(matsOr.B,1),size(matsOr.B,2));

myfilename =strcat( filename,'Bt.dat' );
matsOr.Bt = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Cs.dat' );
matsOr.Cs = spconvert( load( myfilename ) );

myfilename =strcat( filename,'X1.dat' );
matsOr.X1 = spconvert( load( myfilename ) );

myfilename =strcat( filename,'X2.dat' );
matsOr.X2 = spconvert( load( myfilename ) );


myfilename =strcat( filename,'Fu.dat' );
matsOr.Fu = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Fa.dat' );
matsOr.Fa = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Z1.dat' );
matsOr.Z1 = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Z2.dat' );
matsOr.Z2 = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Y.dat' );
matsOr.Y = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Wp.dat' );
matsOr.Wp = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Wa.dat' );
matsOr.Wa = spconvert( load( myfilename ) );

myfilename =strcat( filename,'Cp.dat' );
matsOr.Cp = spconvert( load( myfilename ) );


% Dirichlet nodes
matsOr.essU =load(strcat( filename,'essU.dat' ));
matsOr.essP =load(strcat( filename,'essP.dat' ));
matsOr.essA =load(strcat( filename,'essA.dat' ));


matsOr.zz00 = sparse(size(matsOr.Fu,1), size(matsOr.Fu,2));
matsOr.zz01 = sparse(size(matsOr.Bt,1), size(matsOr.Bt,2));
matsOr.zz02 = sparse(size(matsOr.Z1,1), size(matsOr.Z1,2));
matsOr.zz03 = sparse(size(matsOr.Z2,1), size(matsOr.Z2,2));
matsOr.zz10 = sparse(size(matsOr.B,1),  size(matsOr.B,2));
matsOr.zz11 = sparse(size(matsOr.Cs,1), size(matsOr.Cs,2));
matsOr.zz12 = sparse(size(matsOr.X1,1), size(matsOr.X1,2));
matsOr.zz13 = sparse(size(matsOr.X2,1), size(matsOr.X2,2));
matsOr.zz20 = sparse(size(matsOr.Mz,1), size(matsOr.Fu,2));
matsOr.zz21 = sparse(size(matsOr.Mz,1), size(matsOr.Bt,2));
matsOr.zz22 = sparse(size(matsOr.Mz,1), size(matsOr.Mz,2));
matsOr.zz23 = sparse(size(matsOr.K,1),  size(matsOr.K,2));
matsOr.zz30 = sparse(size(matsOr.Y,1),  size(matsOr.Y,2));
matsOr.zz31 = sparse(size(matsOr.Fa,1), size(matsOr.Bt,2));
matsOr.zz32 = sparse(size(matsOr.Fa,1), size(matsOr.Z1,2));
matsOr.zz33 = sparse(size(matsOr.Fa,1), size(matsOr.Fa,2));

matsOr.A=[ matsOr.Fu,   matsOr.Bt,   matsOr.Z1,   matsOr.Z2; ...
         matsOr.B,    matsOr.Cs,   matsOr.X1,   matsOr.X2; ...
         matsOr.zz20, matsOr.zz21, matsOr.Mz,   matsOr.K; ...
         matsOr.Y,    matsOr.zz32, matsOr.zz32, matsOr.Fa];



matsOr.At= [ matsOr.Fu,   matsOr.zz00, matsOr.Bt,   matsOr.zz01, matsOr.Z1,   matsOr.zz02, matsOr.Z2,   matsOr.zz03 ;...
           matsOr.Mu,   matsOr.Fu,   matsOr.zz01, matsOr.Bt,   matsOr.zz02, matsOr.Z1,   matsOr.zz03, matsOr.Z2   ;...
           matsOr.B,    matsOr.zz10, matsOr.Cs,   matsOr.zz11, matsOr.X1,   matsOr.zz12, matsOr.X2,   matsOr.zz13 ; ...
           matsOr.Mps,  matsOr.B,    matsOr.zz11, matsOr.Cs,   matsOr.zz12, matsOr.X1,   matsOr.zz13, matsOr.X2   ; ...
           matsOr.zz20, matsOr.zz20, matsOr.zz21, matsOr.zz21, matsOr.Mz,   matsOr.zz22, matsOr.K,    matsOr.zz23 ; ...
           matsOr.zz20, matsOr.zz20, matsOr.zz21, matsOr.zz21, matsOr.zz22  matsOr.Mz,   matsOr.zz23, matsOr.K    ; ...
           matsOr.Y,    matsOr.zz30, matsOr.zz31, matsOr.zz31, matsOr.zz32, matsOr.zz32, matsOr.Fa,   matsOr.zz33 ; ...
           matsOr.zz30, matsOr.Y,    matsOr.zz31, matsOr.zz31, matsOr.zz32, matsOr.zz32, matsOr.Ma,   matsOr.Fa  ];

matsOr.Att=[ matsOr.Fu,   matsOr.zz00, matsOr.zz00, matsOr.Bt,   matsOr.zz01, matsOr.zz01, matsOr.Z1,   matsOr.zz02, matsOr.zz02, matsOr.Z2,   matsOr.zz03, matsOr.zz03 ; ...
           matsOr.Mu,   matsOr.Fu,   matsOr.zz00, matsOr.zz01, matsOr.Bt,   matsOr.zz01, matsOr.zz02, matsOr.Z1,   matsOr.zz02, matsOr.zz03, matsOr.Z2,   matsOr.zz03 ; ...
           matsOr.zz00, matsOr.Mu,   matsOr.Fu,   matsOr.zz01, matsOr.zz01, matsOr.Bt,   matsOr.zz02, matsOr.zz02, matsOr.Z1,   matsOr.zz03, matsOr.zz03, matsOr.Z2   ; ...
           matsOr.B,    matsOr.zz10, matsOr.zz10, matsOr.Cs,   matsOr.zz11, matsOr.zz11, matsOr.X1,   matsOr.zz12, matsOr.zz12, matsOr.X2,   matsOr.zz13, matsOr.zz13 ; ...
           matsOr.Mps,  matsOr.B,    matsOr.zz10, matsOr.zz11, matsOr.Cs,   matsOr.zz11, matsOr.zz12, matsOr.X1,   matsOr.zz12, matsOr.zz13, matsOr.X2,   matsOr.zz13 ; ...
           matsOr.zz10, matsOr.Mps,  matsOr.B,    matsOr.zz11, matsOr.zz11, matsOr.Cs,   matsOr.zz12, matsOr.zz12, matsOr.X1,   matsOr.zz13, matsOr.zz13, matsOr.X2   ; ...  
           matsOr.zz20, matsOr.zz20, matsOr.zz20, matsOr.zz21, matsOr.zz21, matsOr.zz21, matsOr.Mz,   matsOr.zz22, matsOr.zz22, matsOr.K,    matsOr.zz23, matsOr.zz23 ; ...
           matsOr.zz20, matsOr.zz20, matsOr.zz20, matsOr.zz21, matsOr.zz21, matsOr.zz21, matsOr.zz22, matsOr.Mz,   matsOr.zz22, matsOr.zz23, matsOr.K,    matsOr.zz23 ; ...
           matsOr.zz20, matsOr.zz20, matsOr.zz20, matsOr.zz21, matsOr.zz21, matsOr.zz21, matsOr.zz22, matsOr.zz22, matsOr.Mz,   matsOr.zz23, matsOr.zz23, matsOr.K    ; ...
           matsOr.Y,    matsOr.zz30, matsOr.zz30, matsOr.zz31, matsOr.zz31, matsOr.zz31, matsOr.zz32, matsOr.zz32, matsOr.zz32, matsOr.Fa,   matsOr.zz33, matsOr.zz33 ; ...
           matsOr.zz30, matsOr.Y,    matsOr.zz30, matsOr.zz31, matsOr.zz31, matsOr.zz31, matsOr.zz32, matsOr.zz32, matsOr.zz32, matsOr.Ma,   matsOr.Fa,   matsOr.zz33 ; ...
           matsOr.zz30, matsOr.zz30, matsOr.Y,    matsOr.zz31, matsOr.zz31, matsOr.zz31, matsOr.zz32, matsOr.zz32, matsOr.zz32, matsOr.zz33, matsOr.Ma,   matsOr.Fa   ];

