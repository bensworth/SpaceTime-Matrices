function [] = timeStep()
Pb   = 4;
Prec = 0;
STsolveU = 0;
STsolveA = 0;
oU = 2;
oP = 1;
oZ = 1;
oA = 2;

petscOpt = 'rc_SpaceTimeIMHD2D';

r  = 4;
NT = 4;

mu  = 1;

rng(1)


path = strcat('Prec',int2str(Prec),'_STsolveU',int2str(STsolveU),'_STsolveA',int2str(STsolveA),...
              '_oU',int2str(oU),'_oP',int2str(oP),'_oZ',int2str(oZ),'_oA',int2str(oA),...
              '_Pb',int2str(Pb),'_',petscOpt,'/NP',int2str(NT),'_r',int2str(r),'/Nit');


dt = 1./NT;


%% Bunch of auxiliary functions useful in preconditioner definitions
% apply inverse of Lub
function y = invertLub(x,FFui,MMzi,YY,ZZ1)
  nU = size(ZZ1,1);
  nZ = size(ZZ1,2);
  nA = size(YY, 1);

  y = x;

  tmp = x(1:nU) - ZZ1*( MMzi\x((end-nA-nZ+1):(end-nA)) );
  y((end-nA+1):end) = x((end-nA+1):end) - YY*( FFui\tmp );

end

% apply inverse of Uub (modified to include the Fu factor, too)
function y = invertUub(x,ZZ1,ZZ2,MMzi,KK,ASi)
  nU = size(ZZ1,1);
  nZ = size(ZZ1,2);
  nA = size(ZZ2,2);

  y = x;

  y((end-nA+1):end) = ASi( x((end-nA+1):end) );
  y((end-nA-nZ+1):(end-nA)) = MMzi\( x((end-nA-nZ+1):(end-nA)) - KK*y((end-nA+1):end));
  y(1:nU) = x(1:nU) - ZZ1*y((end-nA-nZ+1):(end-nA)) - ZZ2*y((end-nA+1):end);

end

% apply inverse of Lup
function y = invertLup(x,FFui,BB)
  nP = size(BB,1);
  nU = size(BB,2);

  y = x;

  y((nU+1):(nU+nP)) = x((nU+1):(nU+nP)) - BB*( FFui\y(1:nU) );

end

% apply inverse of Uup
function y = invertUup(x,FFui,pSi,BB)
  nP = size(BB,1);
  nU = size(BB,2);

  y = x;

  y((nU+1):(nU+nP)) = pSi( x((nU+1):(nU+nP)) );
  y(1:nU) = FFui\( x(1:nU) - BB'*y((nU+1):(nU+nP)) );

end

% apply inverse of Fa*Mai*Fa = Fai*Ma*Fai
function y = invertCC(x,FFai,Ma,ess)
  y = FFai\(Ma*(FFai\x));
  y(ess) = x(ess);  % leave dirichlet nodes untouched

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
filename = strcat(path, int2str(0),'__','Aa.dat');
Aa   = spconvert(load(filename));
filename = strcat(path, int2str(0),'__','MaNZL.dat');
MaNZL= spconvert(load(filename));

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
MLinv = spdiags( 1./sum( MaNZ, 2 ), 0, size(MaNZ,1),size(MaNZ,2) ); % lumped mass matrix for Ma

% useful constants
NU = size(Mu,1);
NP = size(B,1);
NZ = size(Mz,1);
NA = size(Ma,1);

newtIt = 0;

for pp = 1:NT
  % import matrices which vary throughout the Newton iterations
  filename = strcat(path, int2str(newtIt),'__','Fu_', int2str(pp-1),'.dat');
  Fu = spconvert(load(filename));
  filename = strcat(path, int2str(newtIt),'__','Fa_', int2str(pp-1),'.dat');
  Fa = spconvert(load(filename));
  filename = strcat(path, int2str(newtIt),'__','Z1_', int2str(pp-1),'.dat');
  Z1 = spconvert(load(filename));
  filename = strcat(path, int2str(newtIt),'__','Z2_', int2str(pp-1),'.dat');
  Z2 = spconvert(load(filename));
  filename = strcat(path, int2str(newtIt),'__','Y_',  int2str(pp-1),'.dat');
  Y  = spconvert(load(filename));
  
  filename = strcat(path, int2str(newtIt),'__','Wp_', int2str(pp-1),'.dat');
  Wp = spconvert(load(filename));
  filename = strcat(path, int2str(newtIt),'__','Wa_', int2str(pp-1),'.dat');
  Wa = spconvert(load(filename));

  % filename = strcat(path, int2str(newtIt),'__','Cp_', int2str(pp-1),'.dat');
  % Cp = spconvert(load(filename));
  % filename = strcat(path, int2str(newtIt),'__','C0_', int2str(pp-1),'.dat');
  % C0 = spconvert(load(filename));
  % filename = strcat(path, int2str(newtIt),'__','Cm_', int2str(pp-1),'.dat');
  % Cm = spconvert(load(filename));



  % combine them to create the jacobian
  J = [ Fu,            B',            Z1,            Z2           ;...
        B,             sparse(NP,NP), sparse(NP,NZ), sparse(NP,NA);...
        sparse(NZ,NU), sparse(NZ,NP), Mz,            K            ;...
        Y,             sparse(NA,NP), sparse(NA,NZ), Fa          ];


  % initialise inverse of operators which vary throughout the Newton iterations
  Fuinv = decomposition(Fu, 'lu');
  Fainv = decomposition(Fa, 'lu');
  CCinv = @(x) invertCC(x,Fainv,Ma,essA);



  % initialise preconditioner
  pSi  = @(x) invertSp(x,Apinv,Mpinv,{Wp},dt,mu,essP);
  ASi  = @(x) invertSA(x,CCinv,Mainv,{Wa},dt,essA);

  Lubi = @(x) invertLub(x,Fuinv,Mzinv,Y,Z1);
  Uubi = @(x) invertUub(x,Z1,Z2,Mzinv,K,ASi);
  Lupi = @(x) invertLup(x,Fuinv,B);
  Uupi = @(x) invertUup(x,Fuinv,pSi,B);

  switch Prec
    case 0
      precOp = @(x) Uupi(Lupi(Uubi(Lubi(x))));
    case 1
      precOp = @(x) Uupi(Lupi(Uubi(x)));
    case 2
      precOp = @(x) Uupi(Uubi(x));
    otherwise
        error('Preconditioner not recognised')
  end



  fileID = fopen( strcat( path, int2str(newtIt), "_rhs",      int2str(pp-1), ".dat" ),'r');
  temp = fscanf(fileID,'%f');
  rhsu = temp((1:NU)         );
  rhsp = temp((1:NP)+NU      );
  rhsz = temp((1:NZ)+NU+NP   );
  rhsa = temp((1:NA)+NU+NP+NZ);
  fclose(fileID);
  rhs  = [ rhsu; rhsp; rhsz; rhsa ];


  
  % solve
  [ mydsol, err, it ] = GMRESrp( J, rhs, 1e-10, 100, zeros(size(rhs)), precOp );
  semilogy(err)
  hold on

end

end
