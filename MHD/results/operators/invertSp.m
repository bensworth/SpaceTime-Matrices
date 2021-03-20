function y = invertSp(x,Api,Mpi,Wp,dt,mu,ess)
%invertSp Inverts approximate Pressure Schur complement for 2D IMHD
%
% Implements PCD in [1]. Depending on how its components are defined, it
% can work both for single time-step and for space-time operators.
%
% [1] H. Elman, D. Silvester, and A. Wathen. "Finite elements and fast
%  iterative solvers: with applications in incompressible fluid dynamics".
%
% Syntax:  y = invertSp(x,Api,Mpi,Wp,dt,mu,ess)
% 
%   Input:
%    x       - Vector. Right-hand side of the system
%    Api     - (Decomposition of) matrix. Api\x computes the inverse of the
%               stiffness matrix for pressure p
%    Mpi     - (Decomposition of) matrix. Mpi\x computes the inverse of the
%               stiffness matrix for pressure p
%    Wp      - Cell of matrices. Spatial parts of pressure operator (one
%               per time-step)
%    dt      - Scalar. Size of time step (must be constant).
%    mu      - Scalar. Fluid viscosity (must be constant).
%    ess     - Vector. Indices of dirichlet nodes for p
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

% ess = [];

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
    tmp(ess+1) = 0.;
    tmp = mu*( Mpi\tmp );
    tmp(ess+1) = 0.;
    Mpix((1:nP)+nP*(ii-1)) = tmp;
  else
    tmp = Apix((1:nP)+nP*(ii-1));
    tmp(ess+1) = 0.;
    tmp = Wp{ii}*tmp;
    tmp(ess+1) = 0.;
    tmp = Mpi\tmp;
    tmp(ess+1) = 0.;
    Mpix((1:nP)+nP*(ii-1)) = tmp;
  end
end

% include lower diagonal in space-time matrix
for ii =nT:-1:2
  tmp = Apix((1:nP)+nP*(ii-2));
  tmp(ess+1) = 0.;
  Apix((1:nP)+nP*(ii-1)) = Apix((1:nP)+nP*(ii-1)) - tmp;
end

% combine stiffness and mass contributions
y = - (Apix + dt* Mpix);

end