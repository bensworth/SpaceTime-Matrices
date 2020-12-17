function sigma = computeSingVals()
Pb = 2;
Prec = 1;
STsolve = 0;
oU = 2;
oP = 1;

r  = 7;
mu = 1;

% C = B*Finv*B'*( 1/dt*Ainv + mu*Minv )
% -> CC'= B*Finv*B'* ( 1/dt*Ainv + mu*Minv )^2 * B*Finv*B'
% B*Finv\B'*(      Ainv\x + dt*mu*Minv\x ); % if matrices are assembled scaled by dt
function y = precOp(x, Minv, Ainv, B, dt, Finv)
  y0 = B'*x;
  y0 = Finv\y0;
  y0 = B*y0;
  for kk=1:2
    y1 = Minv\y0;
    y2 = Ainv\y0;
    y0 = y2/dt + mu*y1;
  %   y0 = y2 + dt*mu*y1;
  end
  y = B'*y0;
  y = Finv\y;
  y = B*y;
end

sigma = cell(1,4);

for i = -3:0
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
%   filename = strcat(path, 'Mu.dat');
%   Mu = spconvert(load(filename));
  filename = strcat(path, 'Fu.dat');
  Fu = spconvert(load(filename)) / dt;    % Fu is assembled as if multiplied by dt, so rescale it

  if Pb == 1          % Ap is singular in that case: tweak last row/col to make it invertible
    Ap(:,end) = sparse(size(Ap,1),1);
    Ap(end,:) = sparse(1,size(Ap,2));
    Ap(end,end) = 1;
  end
  
  Finv = decomposition(Fu, 'chol');
  Ainv = decomposition(Ap, 'chol');
  Minv = decomposition(Mp, 'chol');

  N = size(Mp,1);

  currentPrecOp = @(x) precOp(x, Minv, Ainv, B, dt, Finv);
  
  sigma{i+4} = sqrt( eigs( currentPrecOp, N, N ) );


  scatter(sigma{i+4},zeros(size((sigma{i+4}))))
  set(gca,'xscale','log')
  pause(0.01)
  hold on
  
end

out = [sigma{1},sigma{2},sigma{3},sigma{4}];

filename = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),...
                  '_STsolve',int2str(STsolve),...
                  '_oU',int2str(oU),'_oP',int2str(oP),...
                  '/singvals_r',int2str(r),'.dat');
format = [ repmat(' %20.18f', [1,size(out,2)] ), '\n' ];
fileID = fopen(filename,'w');
fprintf(fileID,format,out');
fclose(fileID);


end