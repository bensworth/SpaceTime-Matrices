function [] = importOperators()
Pb   = 3;
Prec = 0;
STsolveU = 0;
STsolveA = 0;
oU = 2;
oP = 1;
oZ = 1;
oA = 2;

petscOpt = 'rc_SpaceTimeIMHD2D';

r  = 5;
NT = 4;

mu  = 1;

rng(1)


path = strcat('Prec',int2str(Prec),'_STsolveU',int2str(STsolveU),'_STsolveA',int2str(STsolveA),...
              '_oU',int2str(oU),'_oP',int2str(oP),'_oZ',int2str(oZ),'_oA',int2str(oA),...
              '_Pb',int2str(Pb),'_',petscOpt,'/NP',int2str(NT),'_r',int2str(r),'/Nit');


dt = 1./NT;



% apply inverse of Lub
function y = invertLub(x,FFui,MMzi,YY,ZZ1)
  nU = size(ZZ1,1);
  nZ = size(ZZ1,2);
  nA = size(YY, 1);

  y = x;

  tmp = x(1:nU) - ZZ1*( MMzi(x((end-nA-nZ+1):(end-nA))) );
  y((end-nA+1):end) = x((end-nA+1):end) - YY*( FFui(tmp) );

end

% apply inverse of Uub (modified to include the Fu factor, too)
function y = invertUub(x,ZZ1,ZZ2,MMzi,KK,ASi)
  nU = size(ZZ1,1);
  nZ = size(ZZ1,2);
  nA = size(ZZ2,2);

  y = x;

  y((end-nA+1):end) = ASi( x((end-nA+1):end) );
  y((end-nA-nZ+1):(end-nA)) = MMzi( x((end-nA-nZ+1):(end-nA)) - KK*y((end-nA+1):end));
  y(1:nU) = x(1:nU) - ZZ1*y((end-nA-nZ+1):(end-nA)) - ZZ2*y((end-nA+1):end);

end

% apply inverse of Lup
function y = invertLup(x,FFui,BB)
  nP = size(BB,1);
  nU = size(BB,2);

  y = x;

  y((nU+1):(nU+nP)) = x((nU+1):(nU+nP)) - BB*( FFui(y(1:nU)) );

end

% apply inverse of Uup
function y = invertUup(x,FFui,pSi,BB)
  nP = size(BB,1);
  nU = size(BB,2);

  y = x;

  y((nU+1):(nU+nP)) = pSi( x((nU+1):(nU+nP)) );
  y(1:nU) = FFui( x(1:nU) - BB'*y((nU+1):(nU+nP)) );

end

% apply approximate pressure Schur complement inverse
function y = invertSp(x,Api,Mpi,Wp,dt,mu,ess)
  nP = Api.MatrixSize(1);
  nT = length(x) / nP;

  Apix = zeros(size(x));
  Mpix = zeros(size(x));

  % invert pressure matrix
  for ii =1:nT
    Apix((1:nP)+nP*(ii-1)) = ( Api\x((1:nP)+nP*(ii-1)) );
  end

  % invert mass matrix
  for ii =1:nT
    if isempty(Wp{ii})
      tmp = x((1:nP)+nP*(ii-1));
      tmp(ess) = 0.;
      tmp = mu*( Mpi\tmp );
      tmp(ess) = 0.;
      Mpix((1:nP)+nP*(ii-1)) = tmp;
    else
      tmp = Apix((1:nP)+nP*(ii-1));
      tmp(ess) = 0.;
      tmp = Wp{ii}*tmp;
      tmp(ess) = 0.;
      tmp = Mpi\tmp;
      tmp(ess) = 0.;
      Mpix((1:nP)+nP*(ii-1)) = tmp;
    end
  end

  % include lower diagonal in space-time matrix
  for ii =nT:-1:2
    tmp = Apix((1:nP)+nP*(ii-2));
    tmp(ess) = 0.;
    Apix((1:nP)+nP*(ii-1)) = Apix((1:nP)+nP*(ii-1)) - tmp;
  end

  % combine pressure and mass contributions
  y = - (Apix + dt* Mpix);

end

% apply approximate magnetic Schur complement inverse
function y = invertSA(x,CCi,Mai,Wa,dt,ess)
  nA = Mai.MatrixSize(1);
  nT = length(x) / nA;

  % Apply FaMa^-1
  % compute spatial part
  WMaix = zeros(size(x));
  for ii =1:nT
    tmp = Mai\x((1:nA)+nA*(ii-1));
    tmp(ess) = 0.;
    tmp = Wa{ii}*tmp;
    tmp(ess) = 0.;
    WMaix((1:nA)+nA*(ii-1)) = tmp;
    
%     myfileID = fopen( strcat( path(1:end-3), "Maix_", int2str(ii-1), ".dat" ),'r');
%     outMaix = fscanf(myfileID,'%f');
%     fclose(myfileID);
%     myfileID = fopen( strcat( path(1:end-3), "WMaix_", int2str(ii-1), ".dat" ),'r');
%     outWMaix = fscanf(myfileID,'%f');
%     fclose(myfileID);

    
  end
  % compute temporal part
  tmp = x;
  for ii =nT:-1:2
    tmp2 = tmp((1:nA)+nA*(ii-2));
    tmp2(ess) = 0.;
    tmp((1:nA)+nA*(ii-1)) = tmp((1:nA)+nA*(ii-1)) - tmp2;
%     myfileID = fopen( strcat( path(1:end-3), "temp_", int2str(ii-1), ".dat" ),'r');
%     outTmp = fscanf(myfileID,'%f');
%     fclose(myfileID);

  end
  % combine spatial and temporal contributions
  y1 = tmp + dt*WMaix;
    
  % Apply CCa^-1
  y = CCi(y1);

end






%% Import matrices which remain constant throughout the Newton iterations
filename = strcat(path, int2str(0),'__','Mu.dat');
Mu   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','Mz.dat');
Mz   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','Ma.dat');
Ma   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','B.dat');
B    = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','K.dat');
K    = spconvert(load(filename));

filename = strcat(path, int2str(0),'__','Ap.dat');
Ap   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','Mp.dat');
Mp   = spconvert(load(filename));
  
filename = strcat(path, int2str(0),'__','MaNZ.dat');
MaNZ = spconvert(load(filename));

fileID = fopen( strcat( path, int2str(0), "_essU.dat" ),'r');
essU = fscanf(fileID,'%d') + 1; % adjust zero-indexed values
fclose(fileID);

fileID = fopen( strcat( path, int2str(0), "_essP.dat" ),'r');
essP = fscanf(fileID,'%d') + 1; % adjust zero-indexed values
fclose(fileID);

fileID = fopen( strcat( path, int2str(0), "_essA.dat" ),'r');
essA = fscanf(fileID,'%d') + 1; % adjust zero-indexed values
fclose(fileID);
  

% initialise inverses of operators which remain constant throughout the Newton iterations
% if isempty(essP) % if Ap is singular, fix the last node
%   Ap(end,:)   = sparse(1,size(Ap,2));
%   Ap(:,end)   = sparse(size(Ap,1),1);
%   Ap(end,end) = 1;
% end
Apinv = decomposition(Ap,   'chol');
Mpinv = decomposition(Mp,   'chol');
Mainv = decomposition(MaNZ, 'chol');
Mzinv = decomposition(Mz,   'chol');

% useful constants
NU = size(Mu,1);
NP = size(B,1);
NZ = size(Mz,1);
NA = size(Ma,1);



for newtIt = 0:0
  % import matrices which vary throughout the Newton iterations
  Fu = cell(1,NT);
  Fa = cell(1,NT);
  Z1 = cell(1,NT);
  Z2 = cell(1,NT);
  Y  = cell(1,NT);
  Wp = cell(1,NT);
  Wa = cell(1,NT);
  Cp = cell(1,NT);
  C0 = cell(1,NT);
  Cm = cell(1,NT);

  for pp = 1:NT
    filename = strcat(path, int2str(newtIt),'__','Fu_', int2str(pp-1),'.dat');
    Fu{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Fa_', int2str(pp-1),'.dat');
    Fa{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Z1_', int2str(pp-1),'.dat');
    Z1{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Z2_', int2str(pp-1),'.dat');
    Z2{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Y_',  int2str(pp-1),'.dat');
    Y{pp}  = spconvert(load(filename));
    
    filename = strcat(path, int2str(newtIt),'__','Wp_', int2str(pp-1),'.dat');
    Wp{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Wa_', int2str(pp-1),'.dat');
    Wa{pp} = spconvert(load(filename));

    filename = strcat(path, int2str(newtIt),'__','Cp_', int2str(pp-1),'.dat');
    Cp{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','C0_', int2str(pp-1),'.dat');
    C0{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Cm_', int2str(pp-1),'.dat');
    Cm{pp} = spconvert(load(filename));

  end
  Cp{1} = 2*Cp{1};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp

	% initialise relevant space-time operators
  FFu = sparseBlockLowTri( [ { Fu; repmat({Mu},1,NT-1)}; repmat({[{[]}]},NT-2,1) ] );
  FFa = sparseBlockLowTri( [ { Fa; repmat({Ma},1,NT-1)}; repmat({[{[]}]},NT-2,1) ] );
  MMz = sparseBlockLowTri( [ {     repmat({Mz},1,NT)  }; repmat({[{[]}]},NT-1,1) ] );
  BB  = sparseBlockLowTri( [ {     repmat({B}, 1,NT)  }; repmat({[{[]}]},NT-1,1) ] );
  ZZ1 = sparseBlockLowTri( [ { Z1                     }; repmat({[{[]}]},NT-1,1) ] );
  ZZ2 = sparseBlockLowTri( [ { Z2                     }; repmat({[{[]}]},NT-1,1) ] );
  KK  = sparseBlockLowTri( [ {     repmat({K}, 1,NT)  }; repmat({[{[]}]},NT-1,1) ] );
  YY  = sparseBlockLowTri( [ { Y                      }; repmat({[{[]}]},NT-1,1) ] );
  CC  = sparseBlockLowTri( [ { Cp; C0; Cm };             repmat({[{[]}]},NT-3,1) ] );

  % combine them to create the jacobian
  J = [ FFu,                 BB',                 ZZ1,                 ZZ2                ;...
        BB,                  sparse(NT*NP,NT*NP), sparse(NT*NP,NT*NZ), sparse(NT*NP,NT*NA);...
        sparse(NT*NZ,NT*NU), sparse(NT*NZ,NT*NP), MMz,                 KK                 ;...
        YY,                  sparse(NT*NA,NT*NP), sparse(NT*NA,NT*NZ), FFa                ];


  % initialise inverse of operators which vary throughout the Newton iterations
  Fuinv = cell(1,NT);
  Cpinv = cell(1,NT);
  for pp=1:NT
    Fuinv{pp} = decomposition(Fu{pp}, 'lu');
    Cpinv{pp} = decomposition(Cp{pp}, 'chol');
  end



  %% Initialise preconditioner
  FFui = @(x) invertBlockLowTri(x,{Fuinv,{Mu}},essU);
  MMzi = @(x) invertBlockLowTri(x,{Mzinv});
  CCi  = @(x) invertBlockLowTri(x,{Cpinv,C0,Cm},essA);

  pSi  = @(x) invertSp(x,Apinv,Mpinv,Wp,dt,mu,essP);
  ASi  = @(x) invertSA(x,CCi,Mainv,Wa,dt,essA);

  Lubi = @(x) invertLub(x,FFui,MMzi,YY,ZZ1);
  Uubi = @(x) invertUub(x,ZZ1,ZZ2,MMzi,KK,ASi);
  Lupi = @(x) invertLup(x,FFui,BB);
  Uupi = @(x) invertUup(x,FFui,pSi,BB);

  switch Prec
    case 0
      precOp = @(x) Uupi(Lupi(Uubi(Lubi(x))));
    case 1
      precOp = @(x) Uupi(Lupi(Uubi(x)));
    otherwise
        error('Preconditioner not recognised')
  end



  %% Start testing!
  % - read and assemble intermediate results for this newton iteration
  FFuix  = zeros( NU, NT );
  pSix   = zeros( NP, NT );
  MMzix  = zeros( NZ, NT );
  aSix   = zeros( NA, NT );
  
  Lubixu = zeros( NU, NT );
  Lubixp = zeros( NP, NT );
  Lubixz = zeros( NZ, NT );
  Lubixa = zeros( NA, NT );
  Uubixu = zeros( NU, NT );
  Uubixp = zeros( NP, NT );
  Uubixz = zeros( NZ, NT );
  Uubixa = zeros( NA, NT );
  Lupixu = zeros( NU, NT );
  Lupixp = zeros( NP, NT );
  Lupixz = zeros( NZ, NT );
  Lupixa = zeros( NA, NT );
  Uupixu = zeros( NU, NT );
  Uupixp = zeros( NP, NT );
  Uupixz = zeros( NZ, NT );
  Uupixa = zeros( NA, NT );
  Pixu   = zeros( NU, NT );
  Pixp   = zeros( NP, NT );
  Pixz   = zeros( NZ, NT );
  Pixa   = zeros( NA, NT );

  rhsu  = zeros(NU,NT);
  rhsp  = zeros(NP,NT);
  rhsz  = zeros(NZ,NT);
  rhsa  = zeros(NA,NT);
  
  dsolu = zeros(NU,NT);
  dsolp = zeros(NP,NT);
  dsolz = zeros(NZ,NT);
  dsola = zeros(NA,NT);
  
  for pp=1:NT
    fileID = fopen( strcat( path, int2str(newtIt), "_rhs",      int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    rhsu(:,pp) = temp((1:NU)         );
    rhsp(:,pp) = temp((1:NP)+NU      );
    rhsz(:,pp) = temp((1:NZ)+NU+NP   );
    rhsa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_FFuix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    FFuix(:,pp) = temp((1:NU)         );
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_pSix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    pSix(:,pp) = temp((1:NP)         );
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_MMzix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    MMzix(:,pp) = temp((1:NZ)         );
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_aSix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    aSix(:,pp) = temp((1:NA)         );
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_Lubix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    Lubixu(:,pp) = temp((1:NU)         );
    Lubixp(:,pp) = temp((1:NP)+NU      );
    Lubixz(:,pp) = temp((1:NZ)+NU+NP   );
    Lubixa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_Uubix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    Uubixu(:,pp) = temp((1:NU)         );
    Uubixp(:,pp) = temp((1:NP)+NU      );
    Uubixz(:,pp) = temp((1:NZ)+NU+NP   );
    Uubixa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_Lupix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    Lupixu(:,pp) = temp((1:NU)         );
    Lupixp(:,pp) = temp((1:NP)+NU      );
    Lupixz(:,pp) = temp((1:NZ)+NU+NP   );
    Lupixa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_Uupix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    Uupixu(:,pp) = temp((1:NU)         );
    Uupixp(:,pp) = temp((1:NP)+NU      );
    Uupixz(:,pp) = temp((1:NZ)+NU+NP   );
    Uupixa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_Pix", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    Pixu(:,pp) = temp((1:NU)         );
    Pixp(:,pp) = temp((1:NP)+NU      );
    Pixz(:,pp) = temp((1:NZ)+NU+NP   );
    Pixa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);

    fileID = fopen( strcat( path, int2str(newtIt), "_deltaSol", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    dsolu(:,pp) = temp((1:NU)         );
    dsolp(:,pp) = temp((1:NP)+NU      );
    dsolz(:,pp) = temp((1:NZ)+NU+NP   );
    dsola(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);
  end
  FFuix = FFuix(:);
  pSix  =  pSix(:);
  MMzix = MMzix(:);
  aSix  =  aSix(:);
  rhsu  =  rhsu(:);
  rhsp  =  rhsp(:);
  rhsz  =  rhsz(:);
  rhsa  =  rhsa(:);
  rhs   = [   rhsu(:);   rhsp(:);   rhsz(:);   rhsa(:) ];
  Lubix = [ Lubixu(:); Lubixp(:); Lubixz(:); Lubixa(:) ];
  Uubix = [ Uubixu(:); Uubixp(:); Uubixz(:); Uubixa(:) ];
  Lupix = [ Lupixu(:); Lupixp(:); Lupixz(:); Lupixa(:) ];
  Uupix = [ Uupixu(:); Uupixp(:); Uupixz(:); Uupixa(:) ];
  Pix   = [   Pixu(:);   Pixp(:);   Pixz(:);   Pixa(:) ];
  dsol  = [  dsolu(:);  dsolp(:);  dsolz(:);  dsola(:) ];

  


  % - compute equivalent variables internally
  myFFuix = FFui(   rhsu ); disp( strcat('Error for FFuix: ', num2str(norm( myFFuix - FFuix ))) );
  mypSix  = pSi(    rhsp ); disp( strcat('Error for pSix:  ', num2str(norm( mypSix  - pSix  ))) );
  myMMzix = MMzi(   rhsz ); disp( strcat('Error for MMzix: ', num2str(norm( myMMzix - MMzix ))) );
  myaSix  = ASi(    rhsa ); disp( strcat('Error for aSix:  ', num2str(norm( myaSix  - aSix  ))) );
  myLubix = Lubi(   rhs  ); disp( strcat('Error for Lubix: ', num2str(norm( myLubix - Lubix ))) );
  myUubix = Uubi(   rhs  ); disp( strcat('Error for Uubix: ', num2str(norm( myUubix - Uubix ))) );
  myLupix = Lupi(   rhs  ); disp( strcat('Error for Lupix: ', num2str(norm( myLupix - Lupix ))) );
  myUupix = Uupi(   rhs  ); disp( strcat('Error for Uupix: ', num2str(norm( myUupix - Uupix ))) );
  myPix   = precOp( rhs  ); disp( strcat('Error for Pix:   ', num2str(norm( myPix   - Pix   ))) );

  [ mydsol, err, it ] = GMRESrp( J, rhs, 1e-10, 100, zeros(size(rhs)), precOp );
  disp( strcat('Error for dsol:  ', num2str(norm( mydsol   - dsol   ))) );


end

end
