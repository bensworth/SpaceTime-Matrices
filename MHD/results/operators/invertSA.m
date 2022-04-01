function y = invertSA(x,CCi,Mai,Wa,dt,ess)
%invertSA Inverts approximate Magnetic Schur complement for 2D IMHD
% 
% Implements preconditioner (3.32) in [1]. Depending on how its components
% are defined, it can work both for single time-step and for space-time
% operators.
% 
% [1] Eric C. Cyr, John N. Shadid, Raymond S. Tuminaro, Roger P. Pawlowski,
%     and Luis Chacon. "A new approximate block factorization preconditioner
%     for two-dimensional incompressible (reduced) resistive MHD".
%
% Syntax:  y = invertSA(x,CCi,Mai,Wa,dt,ess)
% 
%   Input:
%    x       - Vector. Right-hand side of the system
%    CCi     - Function handle. Inverts the right-most matrix in (3.32).
%               Depending on how it's defined, different operators can be
%               implemented (P_CSC, P_Comm, P_Diag)
%    Mai     - (Decomposition of) matrix. Mai\x computes the inverse of the
%               mass matrix for vector potential A
%    Wa      - Cell of matrices. Spatial parts of vector potential
%               operator (one per time-step)
%    dt      - Scalar. Size of time step (must be constant).
%    ess     - Vector. Indices of dirichlet nodes for A
%
%
%  Output:
%    y       - Vector. Result from applying the approximate Schur
%               complement to x
%
%
% Author: Federico Danieli, Numerical Analysis Group
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% Mar 2021; Last revision: Mar-2021


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