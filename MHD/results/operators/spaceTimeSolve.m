function [] = spaceTimeSolve()
Pb   = 2;
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


%% Bunch of auxiliary functions useful in preconditioner definitions
% Solution for block diagonal
% - assumes diagonal is constant!
function y = invertDD(x,Mi)
  n  = Mi.MatrixSize(1);
  nT = length(x)/n;

  y = zeros(n*nT,1);
  
  for ii=1:nT
    y((1:n)+n*(ii-1)) = Mi\( x((1:n)+n*(ii-1)) );
  end
end

% Forward substitution for block-bidiagonal
% - assumes lower diagonal is constant!
function y = invertFF(x,Fi,M)
  nT = length(Fi);
  n  = length(x)/nT;
  
  y      = zeros(n*nT,1);
  
  yprev  = Fi{1}\x(1:n);
  y(1:n) = yprev;

  for ii=2:nT
    yprev = Fi{ii}\( x((1:n)+n*(ii-1)) - M*yprev );
    y((1:n)+n*(ii-1)) = yprev;
  end
end

% Forward substitution for block-tridiagonal
function y = invertCC(x,Ci,C0,Cm,ess)
  nT = length(Ci);
  n  = length(x)/nT;

  y      = zeros(n*nT,1);
  yprev2 = Ci{1}\x(1:n);
  y(1:n) = yprev2;
  yprev  = Ci{2}\( x((n+1):(2*n)) - C0{1}*yprev2 );
  y((n+1):(2*n)) = yprev;

  for ii=3:nT
    tmp = Ci{ii}\( x((1:n)+n*(ii-1)) - C0{ii-1}*yprev - Cm{ii-2}*yprev2 );
    y((1:n)+n*(ii-1)) = tmp;
    yprev2 = yprev;
    yprev  = tmp;
  end
  
  % leave dirichlet nodes untouched
  for ii=1:nT
    y(ess+1+n*(ii-1)) = x(ess+1+n*(ii-1));
  end  
end

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

fileID = fopen( strcat( path, int2str(0), "_essP.dat" ),'r');
essP = fscanf(fileID,'%d');
fclose(fileID);

fileID = fopen( strcat( path, int2str(0), "_essA.dat" ),'r');
essA = fscanf(fileID,'%d');
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
  dtuWa = cell(1,NT);
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
    filename = strcat(path, int2str(newtIt),'__','dtuWa_', int2str(pp-1),'.dat');
    dtuWa{pp} = spconvert(load(filename));

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
  
  FFui = @(x) invertFF(x,Fuinv,Mu);
  MMzi = @(x) invertDD(x,Mzinv);    

  magSchurComps = [0];
  for magSchurComp = magSchurComps

  % experiment with different operators for the magnetic schur complement
  CCpinv = cell(1,NT);
  CC0    = cell(1,NT);
  CCm    = cell(1,NT);
  for pp = 1:NT   % first of all, kill all contributions to Dirichlet nodes
    Wa{pp}(essA+1,:) = sparse(length(essA),size(Wa{1},2));
    Wa{pp}(:,essA+1) = sparse(size(Wa{1},1),length(essA));
  end
  switch magSchurComp
    % -- Inverse of whole Fa*Mai*Fa = Fai*Ma*Fai  -- HOLDS ONLY IF |B0|=0!!
    case 0
      Fainv = cell(1,NT);
      for pp=1:NT
        Fainv{pp} = decomposition(Fa{pp}, 'lu');
      end
      FFai = @(x) invertFF(x,Fainv,Ma);
      CCi  = @(x) FFai( MMa*FFai(x) );
    % -- Pure wave equation:
    case 1
      for pp=1:NT
        if pp == 1
          CCpinv{pp} = decomposition(Cp{pp}, 'chol');  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp
        else
          CCpinv{pp} = decomposition(Cp{pp}, 'chol');
        end
        CC0{pp} = C0{pp};
        CCm{pp} = Cm{pp};
      end
      CCi  = @(x) invertCC(x,CCpinv,CC0,CCm,essA);
    % -- Wave equation + terms in dt coming from space-time Fa*Mai*Fa
    case 2
      for pp = 1:NT
        if pp == 1
          temp = Cp{pp} + 2*dt*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
          CCpinv{pp} = decomposition(temp, 'lu');
        else
          temp =   Cp{pp} + 2*dt*Wa{pp};
          CCpinv{pp} = decomposition(temp, 'lu');
        end
        if pp < NT
          CC0{pp} =  C0{pp} -   dt*(Wa{pp}+Wa{pp+1});
        end
        CCm{pp} =  Cm{pp}; % maybe "symmetrise" this?
      end
      CCi  = @(x) invertCC(x,CCpinv,CC0,CCm,essA);
    % -- Wave equation + terms in dt, but organised as if it was the discretisation of a dissipative wave eq
    case 3
      for pp = 1:NT
        if pp == 1
          temp = Cp{pp} + 2*dt*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
          CCpinv{pp} = decomposition(temp, 'lu');
        else
          temp =   Cp{pp} + 2*dt*Wa{pp};
          CCpinv{pp} = decomposition(temp, 'lu');
        end
        CC0{pp} =  C0{pp};
        CCm{pp} =  Cm{pp} - 2*dt*Wa{pp};
      end
      CCi  = @(x) invertCC(x,CCpinv,CC0,CCm,essA);
    % -- Wave equation + *ALL* terms in dt which would come from the continuous operator: also dtu gradA
    case 4
      for pp = 1:NT
        temp =   Cp{pp} + 2*dt*Wa{pp} + dt*dtuWa{pp}/4;
        CCpinv{pp} = decomposition(temp, 'lu');
        CC0{pp} =  C0{pp} + 2*dt*dtuWa{pp}/4;
        CCm{pp} =  Cm{pp} - 2*dt*Wa{pp} + dt*dtuWa{pp}/4;
      end
      CCi  = @(x) invertCC(x,CCpinv,CC0,CCm,essA);
    % -- Wave equation + terms in dt and terms in dt^2 with inverse of diagonal of mass matrix
    case 5
      for pp = 1:NT
        if pp == 1
          temp = Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
          CCpinv{pp} = decomposition(temp, 'lu');
        else
          temp =   Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
          CCpinv{pp} = decomposition(temp, 'lu');
        end
        if pp < NT
          CC0{pp} =  C0{pp} -   dt*(Wa{pp}+Wa{pp+1});
        end
        CCm{pp} =  Cm{pp}; % maybe "symmetrise" this?
      end
      CCi  = @(x) invertCC(x,CCpinv,CC0,CCm,essA);
    % -- Wave equation + terms in dt and dt^2, but as if they came from discretising a continuous operator
    case 6
      for pp = 1:NT
        temp =      Cp{pp} + 2*dt*Wa{pp} + 4*dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp}/4;  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
        CCpinv{pp} = decomposition(temp, 'lu');
        if pp < NT
          CC0{pp} = C0{pp}               + 8*dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp}/4;
        end
        CCm{pp} =   Cm{pp} - 2*dt*Wa{pp} + 4*dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp}/4;
      end
      CCi  = @(x) invertCC(x,CCpinv,CC0,CCm,essA);
  end
  


  pSi  = @(x) invertSp(x,Apinv,Mpinv,Wp,dt,mu,essP);
  ASi  = @(x) invertSA(x,CCi,  Mainv,Wa,dt,essA);

  Lubi = @(x) invertLub(x,FFui,MMzi,YY,ZZ1);
  Uubi = @(x) invertUub(x,ZZ1,ZZ2,MMzi,KK,ASi);
  Lupi = @(x) invertLup(x,FFui,BB);
  Uupi = @(x) invertUup(x,FFui,pSi,BB);

  switch Prec
    case 0
%       precOp = @(x) Uupi(Lupi(Uubi(Lubi(x))));
      precOp = @(x) Uupi(Uubi(Lubi(x)));
    case 1
      precOp = @(x) Uupi(Lupi(Uubi(x)));
    otherwise
        error('Preconditioner not recognised')
  end



  %% Start testing!
  rhsu  = zeros(NU,NT);
  rhsp  = zeros(NP,NT);
  rhsz  = zeros(NZ,NT);
  rhsa  = zeros(NA,NT);
  for pp=1:NT
    fileID = fopen( strcat( path, int2str(newtIt), "_rhs",      int2str(pp-1), ".dat" ),'r');
    temp = fscanf(fileID,'%f');
    rhsu(:,pp) = temp((1:NU)         );
    rhsp(:,pp) = temp((1:NP)+NU      );
    rhsz(:,pp) = temp((1:NZ)+NU+NP   );
    rhsa(:,pp) = temp((1:NA)+NU+NP+NZ);
    fclose(fileID);
  end
  rhs = [rhsu(:); rhsp(:); rhsz(:); rhsa(:) ];


  


  % - compute equivalent variables internally
  [ mydsol, err, it ] = GMRESrp( J, rhs, 1e-10, 100, zeros(size(rhs)), precOp );
  semilogy(err);
  legend(num2str(magSchurComps'));
  pause(0.1)
  hold on
  end
end

end
