function sigma = computeSingVals()

% NB: ONLY WORKS FOR ENCLOSED FLOWS!
Pb = 1;
Prec = 1;
STsolve = 0;
oU = 2;
oP = 1;

r  = 6;
mu = 1;


% -> CC'= B*Finv'*B' * ( Minv*Fp*Ainv )'*( Minv*Fp*Ainv ) * B*Finv*B'
function y = precOp( x, Mpinv, Apinv, Fuinv, B, Mu, dt, Np, Nu, NT, dirIdx )
    
  % apply B'
  y0 = zeros(Nu*NT,1);
  for t=1:NT
    y0((1:Nu)+(t-1)*Nu) = B'*x((1:Np)+(t-1)*Np);
  end
  % solve for Fu via time stepping
  yprev = zeros(Nu,1);
  for t=1:NT
    y0((1:Nu)+(t-1)*Nu) = Fuinv\( y0((1:Nu)+(t-1)*Nu) - Mu*(yprev/dt) );
    yprev = y0((1:Nu)+(t-1)*Nu);
  end
  % apply B
  y1 = zeros(Np*NT,1);
  for t=1:NT
    y1((1:Np)+(t-1)*Np) = B*y0((1:Nu)+(t-1)*Nu);
  end
  
  % Minv*Fp*Ainv is block bi-diag, with blocks Apinv/dt+mu*Mpinv and -Apinv/dt
  yAprev = zeros(Np,1);
  for t=1:NT
    yA = Apinv\y1((1:Np)+(t-1)*Np);
    yM = Mpinv\y1((1:Np)+(t-1)*Np);
    y1((1:Np)+(t-1)*Np) = yA/dt+mu*yM - yAprev/dt;  % yAprev is zero at first it
    yAprev = yA;
    yAprev(dirIdx) = 0; % kill contributions from dirichlet
  end
  % - once again, but it's transposed this time, so backward sub:
  yAprev = zeros(Np,1);
  for t=NT:-1:1
    yA = Apinv\y1((1:Np)+(t-1)*Np);
    yM = Mpinv\y1((1:Np)+(t-1)*Np);
    y1((1:Np)+(t-1)*Np) = yA/dt+mu*yM - yAprev/dt;  % yAprev is zero at first it
    yAprev = yA;
    yAprev(dirIdx) = 0; % kill contributions from dirichlet
  end
  
  % Once again:
  % apply B'
  y2 = zeros(Nu*NT,1);
  for t=1:NT
    y2((1:Nu)+(t-1)*Nu) = B'*y1((1:Np)+(t-1)*Np);
  end
  % solve for Fu via time stepping - but it's transposed this time, so backward sub:
  yprev = zeros(Nu,1);
  for t=NT:-1:1
    yprev = Fuinv\( y2((1:Nu)+(t-1)*Nu) - Mu*(yprev/dt) );
    y2((1:Nu)+(t-1)*Nu) = yprev;
  end
  % apply B
  y = zeros(Np*NT,1);
  for t=1:NT
    y((1:Np)+(t-1)*Np) = B*y2((1:Nu)+(t-1)*Nu);
  end

end




refLvls = 0:-1:-2;
sigma = cell(1,length(refLvls));

for i = refLvls
  dt = 10^i;
  path = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),'_STsolve',int2str(STsolve),...
               '_oU',int2str(oU),'_oP',int2str(oP),...
               '/dt',num2str(dt,'%8.6f'),'_r',int2str(r),'_');

  filename = strcat(path, 'B.dat');
  B  = spconvert(load(filename)) / dt;    % B is assembled as if multiplied by dt, so rescale it
  filename = strcat(path, 'Ap.dat');
  Ap = spconvert(load(filename));
  filename = strcat(path, 'Mp.dat');
  Mp = spconvert(load(filename));
  filename = strcat(path, 'Mu.dat');
  Mu = -spconvert(load(filename))/ dt;    % Mu is assembled as if multiplied by - dt, so rescale it
  filename = strcat(path, 'Fu.dat');
  Fu = spconvert(load(filename)) / dt;    % Fu is assembled as if multiplied by dt, so rescale it
  
  % this is to find eventual dirichlet BC imposed on pressure
  % TODO: should this go *after* I set one value on pressure to avoid singular operator?
  dirIdx = [];
  for kk=1:size(Ap,1)
    if (nnz(Ap(:,kk)) == 1)
      dirIdx =[dirIdx,kk];
    end
  end

  if Pb == 1          % Ap is singular in that case: tweak last row/col to make it invertible
    Ap(:,end) = sparse(size(Ap,1),1);
    Ap(end,:) = sparse(1,size(Ap,2));
    Ap(end,end) = 1;
  end
  
  Fuinv = decomposition(Fu, 'chol');
  Apinv = decomposition(Ap, 'chol');
  Mpinv = decomposition(Mp, 'chol');

  Np = size(Mp,1);
  Nu = size(Mu,1);
  NT = 10^(-i);

  currentPrecOp = @(x) precOp( x, Mpinv, Apinv, Fuinv, B, Mu, dt, Np, Nu, NT, dirIdx );
  
  sigma{i+length(refLvls)} = sqrt( eigs( currentPrecOp, NT*Np, ceil(Np/4), 'bothendsreal' ) );


  scatter(sigma{i+length(refLvls)},zeros(size((sigma{i+length(refLvls)}))))
  set(gca,'xscale','log')
  pause(0.01)
  hold on
  
end

out = [sigma{3}, sigma{2}, sigma{1}];

filename = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),...
                  '_STsolve',int2str(STsolve),...
                  '_oU',int2str(oU),'_oP',int2str(oP),...
                  '/singvals_r',int2str(r),'.dat');
format = [ repmat(' %20.18f', [1,size(out,2)] ), '\n' ];
fileID = fopen(filename,'w');
fprintf(fileID,format,out');
fclose(fileID);


end