function lambda = computeEigs()
Pb = 1;
Prec = 1;
STsolve = 0;
oU = 2;
oP = 1;

r  = 6;
mu = 1;

% precOp = @(x) B*Finv\B'*( 1/dt*Ainv\x + mu*Minv\x );
function y = precOp(x, Minv, Ainv, B, dt, Finv)
  y1 = Minv\x;
  y2 = Ainv\x;
  y3 = B'*(y2/dt + mu*y1);
  y = Finv\y3;
  y = B*y;
end

lambda = cell(1,4);

for i = -3:0
  dt = 10^i;
  path = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),'_STsolve',int2str(STsolve),...
               '_oU',int2str(oU),'_oP',int2str(oP),...
               '/dt',num2str(dt,'%8.6f'),'_r',int2str(r),'_');

  filename = strcat(path, 'B.dat');
  B  = spconvert(load(filename));
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
  
  lambda{i+4} = eigs( currentPrecOp, N, N );


  scatter(real(lambda{i+4}),imag(lambda{i+4}))
  set(gca,'xscale','log')
  pause(0.01)
  hold on
  
end

out = [lambda{1},lambda{2},lambda{3},lambda{4}];

filename = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),...
                  '_STsolve',int2str(STsolve),...
                  '_oU',int2str(oU),'_oP',int2str(oP),...
                  '/eigs_r',int2str(r),'.dat');
format = [ repmat(' %20.18f', [1,size(out,2)] ), '\n' ];
fileID = fopen(filename,'w');
fprintf(fileID,format,out');
fclose(fileID);


end