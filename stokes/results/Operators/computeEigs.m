function lambda = computeEigs(r,Pb)
% Pb = 4;
Prec = 1;
STsolve = 0;
oU = 2;
oP = 1;
Pe = 1000;

% r  = 6;
mu = 1;

rng(1)

% B*Finv\B'*( 1/dt*Ainv\x + mu*Minv\x );
% B*Finv\B'*(      Ainv\x + dt*mu*Minv\x ); % if matrices are assembled scaled by dt
function y = precOp(x, Minv, Ainv, B, dt, Finv, W)
  yA = Ainv\x;
  yM = Minv\( mu*x + W*yA );
  y3 = B'*(yA/dt + yM);
  y = Finv\y3;
  y = B*y;
end

refLvls = -2:-1:-5; % must be an array of descending, consecutive numbers
lambda = cell(1,length(refLvls));
colors = {'blue', 'red', 'green', 'black'}; % 4 should suffice?

path0 = strcat('Pb',int2str(Pb),'_Prec',int2str(Prec),'_STsolve',int2str(STsolve),...
             '_oU',int2str(oU),'_oP',int2str(oP));
if Pb==4
  path0 = strcat(path0, '_Pe', num2str(Pe,'%8.6f'));
end

for i = 1:length(refLvls)
  dt = 2^refLvls(i);
  path = strcat(path0,'/dt',num2str(dt,'%8.6f'),'_r',int2str(r),'_');

  filename = strcat(path, 'B.dat');
  B  = spconvert(load(filename)) / dt;    % B is assembled as if multiplied by dt, so rescale it
  filename = strcat(path, 'Ap.dat');
  Ap = spconvert(load(filename));
  filename = strcat(path, 'Mp.dat');
  Mp = spconvert(load(filename));

  if Pb == 1 || Pb == 4          % Ap is singular in that case: tweak last row/col to make it invertible
    Ap(:,end) = sparse(size(Ap,1),1);
    Ap(end,:) = sparse(1,size(Ap,2));
    Ap(end,end) = 1;
  end
  
  Ainv = decomposition(Ap, 'chol');
  Minv = decomposition(Mp, 'chol');

  N = size(Mp,1);

  if Pb==4
    Nk = 2^(-refLvls(i));
  else
    Nk = 1;
  end
  
  lambda{i} = zeros(N*Nk,1);

  for k = 1:Nk
    filename = strcat(path, 'Fu_', int2str(k-1), '.dat');
    Fu = spconvert(load(filename)) / dt;    % Fu is assembled as if multiplied by dt, so rescale it
    if Pb==4
      Finv = decomposition(Fu, 'lu');
      filename = strcat(path, 'Wp_', int2str(k-1), '.dat');
      W = spconvert(load(filename));    % don't rescale!
    else
      Finv = decomposition(Fu, 'chol');
      W = sparse(N,N);    % W is not used: leave it as zero
    end

    currentPrecOp = @(x) precOp(x, Minv, Ainv, B, dt, Finv, W);
  
    lambdapart = eigs( currentPrecOp, N, N );

    scatter(real(lambdapart),imag(lambdapart), colors{i})
    set(gca,'xscale','log')
    pause(0.01)
    hold on
    lambda{i}((1:N)+N*(k-1)) = lambdapart;
  end
end

out  = ones(length(lambda{end}),2*length(refLvls))*NaN;
nsubsmpl = min(floor(N/2),100); %must be even
out2 = zeros(nsubsmpl,2*length(refLvls));
for i=1:length(refLvls)
  out(1:length(lambda{i}),(1:2) + 2*(i-1)) = [real(lambda{i}),imag(lambda{i})];
  % perform a smart subsampling as well:
  % - pick nsubsmpl eigs (spread apart so that I don't have two consecutive ones)
  idx = 2*randperm(floor(length(lambda{i})/2), nsubsmpl);
  % - store them
  ii = 1;
  while ii<=length(idx)
    out2(ii,(1:2) + 2*(i-1)) = [real(lambda{i}(idx(ii))), imag(lambda{i}(idx(ii)))];
    % making sure we preserve the whole eventual complex pair
    if lambda{i}(idx(ii)) ~= real(lambda{i}(idx(ii)))
      % eventually killing a real eig if I'm right at the end (there must be one, since numsubsmpl is even!)
      if ii==length(idx)
        newidx = find( out2(:,2+2*(i-1)) == 0.0, 1 );
        out2(newidx:end-1,(1:2)+2*(i-1)) = out2(newidx+1:end,(1:2)+2*(i-1));
        out2(ii,(1:2) + 2*(i-1)) = [real(lambda{i}(idx(ii))),-imag(lambda{i}(idx(ii)))];
      else
        out2(ii+1,(1:2) + 2*(i-1)) = [real(lambda{i}(idx(ii))),-imag(lambda{i}(idx(ii)))];
      end
      ii = ii+1;
    end
    ii = ii + 1;
  end
end

filename = strcat(path0,'/eigs_r',int2str(r),'.dat');
format = [ repmat(' %20.18f', [1,size(out,2)] ), '\n' ];
fileID = fopen(filename,'w');
fprintf(fileID,format,out');
fclose(fileID);

filename = strcat(path0,'/eigsSub_r',int2str(r),'.dat');
format = [ repmat(' %20.18f', [1,size(out2,2)] ), '\n' ];
fileID = fopen(filename,'w');
fprintf(fileID,format,out2');
fclose(fileID);


end