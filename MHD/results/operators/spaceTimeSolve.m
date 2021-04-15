function [] = spaceTimeSolve()
Pb   = 3;
Prec = 0;
STsolveU = 0;
STsolveA = 2;
oU = 2;
oP = 1;
oZ = 1;
oA = 2;

petscOpt = 'rc_SpaceTimeIMHD2D';

r  = 3;
NT = 4;

mu  = 1;


rng(1)


path = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),'_STsolveU',int2str(STsolveU),'_STsolveA',int2str(STsolveA),...
              '_oU',int2str(oU),'_oP',int2str(oP),'_oZ',int2str(oZ),'_oA',int2str(oA),...
              '_',petscOpt,'/NP',int2str(NT),'_r',int2str(r),'/Nit');


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






%% Import matrices which remain constant throughout the Newton iterations
filename = strcat(path, int2str(0),'__','Mu.dat');
Mu   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','Mz.dat');
Mz   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','Ma.dat');
Ma   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','Aa.dat');
Aa   = spconvert(load(filename));
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

% useful constants
NU = size(Mu,1);
NP = size(B,1);
NZ = size(Mz,1);
NA = size(Ma,1);


NLrhsu = zeros(NU,NT);
NLrhsp = zeros(NP,NT);
NLrhsz = zeros(NZ,NT);
NLrhsa = zeros(NA,NT);
for pp=1:NT
  fileID = fopen( strcat( path, int2str(0), "__NLrhsU_",      int2str(pp-1), ".dat" ),'r');
  NLrhsu(:,pp) = fscanf(fileID,'%f');
  fclose(fileID);
  fileID = fopen( strcat( path, int2str(0), "__NLrhsP_",      int2str(pp-1), ".dat" ),'r');
  NLrhsp(:,pp) = fscanf(fileID,'%f');
  fclose(fileID);
  fileID = fopen( strcat( path, int2str(0), "__NLrhsZ_",      int2str(pp-1), ".dat" ),'r');
  NLrhsz(:,pp) = fscanf(fileID,'%f');
  fclose(fileID);
  fileID = fopen( strcat( path, int2str(0), "__NLrhsA_",      int2str(pp-1), ".dat" ),'r');
  NLrhsa(:,pp) = fscanf(fileID,'%f');
  fclose(fileID);
end
NLrhsu = NLrhsu(:);
NLrhsp = NLrhsp(:);
NLrhsz = NLrhsz(:);
NLrhsa = NLrhsa(:);


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



for newtIt = 0:3
  % import matrices which vary throughout the Newton iterations
  Fu = cell(1,NT);
  Fa = cell(1,NT);
  Z1 = cell(1,NT);
  Z2 = cell(1,NT);
  Y  = cell(1,NT);
  Wp = cell(1,NT);
  Wa = cell(1,NT);
  dtuWa = cell(1,NT);
  Cp = cell(1,NT);
  C0 = cell(1,NT);
  Cm = cell(1,NT);
  B0 = cell(1,NT);

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
    filename = strcat(path, int2str(newtIt),'__','dtuWa_', int2str(pp-1),'.dat');
    dtuWa{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','B0_', int2str(pp-1),'.dat');
    B0{pp} = load(filename);

    filename = strcat(path, int2str(newtIt),'__','Cp_', int2str(pp-1),'.dat');
    Cp{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','C0_', int2str(pp-1),'.dat');
    C0{pp} = spconvert(load(filename));
    filename = strcat(path, int2str(newtIt),'__','Cm_', int2str(pp-1),'.dat');
    Cm{pp} = spconvert(load(filename));

  end
  

  
	% initialise relevant space-time operators
  FFu = sparseBlockLowTri( [ { Fu; repmat({Mu},1,NT-1)}; repmat({[{[]}]},NT-2,1) ] );
  FFa = sparseBlockLowTri( [ { Fa; repmat({Ma},1,NT-1)}; repmat({[{[]}]},NT-2,1) ] );
  MMz = sparseBlockLowTri( [ {     repmat({Mz},1,NT)  }; repmat({[{[]}]},NT-1,1) ] );
  MMa = sparseBlockLowTri( [ {     repmat({MaNZ},1,NT)}; repmat({[{[]}]},NT-1,1) ] );
  BB  = sparseBlockLowTri( [ {     repmat({B}, 1,NT)  }; repmat({[{[]}]},NT-1,1) ] );
  ZZ1 = sparseBlockLowTri( [ { Z1                     }; repmat({[{[]}]},NT-1,1) ] );
  ZZ2 = sparseBlockLowTri( [ { Z2                     }; repmat({[{[]}]},NT-1,1) ] );
  KK  = sparseBlockLowTri( [ {     repmat({K}, 1,NT)  }; repmat({[{[]}]},NT-1,1) ] );
  YY  = sparseBlockLowTri( [ { Y                      }; repmat({[{[]}]},NT-1,1) ] );

  % combine them to create the jacobian
  J = [ FFu,                 BB',                 ZZ1,                 ZZ2                ;...
        BB,                  sparse(NT*NP,NT*NP), sparse(NT*NP,NT*NZ), sparse(NT*NP,NT*NA);...
        sparse(NT*NZ,NT*NU), sparse(NT*NZ,NT*NP), MMz,                 KK                 ;...
        YY,                  sparse(NT*NA,NT*NP), sparse(NT*NA,NT*NZ), FFa                ];


  %% Initialise preconditioner
  % initialise inverse of operators which vary throughout the Newton iterations
  Fuinv = cell(1,NT);
  for pp=1:NT
    Fuinv{pp} = decomposition(Fu{pp}, 'lu');
  end
  
  FFui = @(x) invertBlockLowTri(x,{Fuinv,{Mu}},essU);
  MMzi = @(x) invertBlockLowTri(x,{{Mzinv}});

%   magSchurComps = [0,1,2,10,11,12,13,14,20,21,22,23,24,30,31,32];
  magSchurComps = 1;
  for magSchurComp = magSchurComps

  % experiment with different operators for the magnetic schur complement
  CCi = assembleCCinv( magSchurComp, Fa, MaNZ, Wa, dtuWa, Aa, B0, dt, essA );


  pSi  = @(x) invertSp(x,Apinv,Mpinv,Wp,dt,mu,essP);
  ASi  = @(x) invertSA(x,CCi,  Mainv,Wa,dt,essA);

  Lubi = @(x) invertLub(x,FFui,MMzi,YY,ZZ1);
  Uubi = @(x) invertUub(x,ZZ1,ZZ2,MMzi,KK,ASi);
  Lupi = @(x) invertLup(x,FFui,BB);
  Uupi = @(x) invertUup(x,FFui,pSi,BB);

  switch Prec
    case 0
      precOp = @(x) Uupi(Lupi(Uubi(Lubi(x))));
    case 1
      precOp = @(x) Uupi(Lupi(Uubi(x)));
    case 2
      precOp = @(x) Uupi(Uubi(x));  % this actually seems to be giving the (almost) exact same results!
    otherwise
      error('Preconditioner not recognised')
  end



  %% Start testing!
  resu   = zeros(NU,NT);
  resp   = zeros(NP,NT);
  resz   = zeros(NZ,NT);
  resa   = zeros(NA,NT);
  dsolu  = zeros(NU,NT);
  dsolp  = zeros(NP,NT);
  dsolz  = zeros(NZ,NT);
  dsola  = zeros(NA,NT);
  for pp=1:NT
    fileID = fopen( strcat( path, int2str(newtIt), "_rhs",      int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    resu(:,pp)  = temp((1:NU)         );
    resp(:,pp)  = temp((1:NP)+NU      );
    resz(:,pp)  = temp((1:NZ)+NU+NP   );
    resa(:,pp)  = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);
    
    fileID = fopen( strcat( path, int2str(newtIt), "_deltaSol", int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    dsolu(:,pp) = temp((1:NU)         );
    dsolp(:,pp) = temp((1:NP)+NU      );
    dsolz(:,pp) = temp((1:NZ)+NU+NP   );
    dsola(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);
  end
  res  = [  resu(:);  resp(:);  resz(:);  resa(:) ];
  dsol = [ dsolu(:); dsolp(:); dsolz(:); dsola(:) ];


  


  % - compute equivalent variables internally
  [ mydsol, err, it ] = GMRESrp( J, res, 1e-10, 100, zeros(size(res)), precOp );
  semilogy(err);
  legend(num2str(magSchurComps'));
  hold on
  disp(norm(mydsol-dsol));
  
  % - print c++ convergence result
  filename = strcat('../',path(1:end-3), 'GMRESconv_Nit',int2str(newtIt), '.txt');
  tempfile = strcat('../',path(1:end-3),'temp.txt');
  command = ['tac ', filename,' | sed ''/Residual norms/q'' | tac | tee ', tempfile ];
  [dummy, dummier] = unix(command);
  % read from table
  T = readtable( tempfile );
  % extract relevant info (its and residual norm)
  res = table2array(T(:,5));
  semilogy(res,'--');
  pause(0.2)

  
  end
end

end
