function [ x, err, it ] = GMRESrp( A, b, tol, maxIT, x0, R )
%GMRESrp Own implementation of GMRES with right-preconditioning
% 
% Shamelessly copied from Wikipedia:
% https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
%
% Syntax:  [ x, err, it ] = GMRESrp( A, b, tol, maxIT, x0, R )
% 
%   Input:
%    A       - Can be a Matrix or a function handle
%               - Matrix: the actual system to solve.
%               - Function: A(x) returns the action of A on x, A*x.
%    b       - Vector. Right-hand side of linear system.
%    tol     - Scalar. Tolerance on relative error to achieve convergence.
%    maxIT   - Integer. Maximum number of iterations.
%    x0      - Vector. Initial guess (optional, default: zero vector).
%    R       - Can be a Matrix or a function handle (optional):
%               - Matrix: the right preconditioner itself.
%               - Function: R(b) returns the action of the inverse of the
%                  right-preconditioner on b, R\b.
%
%  Output:
%    x       - Vector. (Approximate) solution.
%    err     - Vector. Relative error (based on residual).
%    it      - Integer. Number of iterations before convergence.
%
%
% Author: Federico Danieli, Numerical Analysis Group
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% July 2019; Last revision: Jul-2019


if ( nargin<2 || isempty(A) || isempty(b) )
	error('GMRESrp: need to provide at least matrix and right-hand side')
end
if (nargin<3 || isempty(tol))
	tol = 1e-8;
end
b = b(:);				% make b a column vector
if (nargin<4 || isempty(maxIT))
	maxIT = size(b,1);
end
if (nargin<5 || isempty(x0))
	x0 = zeros(size(b));
end
if (nargin<6 || isempty(R))
	R = speye(size(b,1),size(b,1));
end

if ~( isnumeric(A) || isa(A, 'function_handle') )
	error('Representation of system not recognised')
end
if ~( isnumeric(R) || isa(R, 'function_handle') )
	error('Format of right-preconditioner not recognised')
end


%% Initialization
N   = length(b);			% size of matrix
nIt = min(N, maxIT);  % maximum number of iterations

if isnumeric(A)				% residual of initial iteration
	r  = b - A*x0;				
elseif isa(A,'function_handle')
	r  = b - A(x0);
end

bn = norm(b);

e1    = zeros(nIt,1);	% rhs of minimisation problem
e1(1) = 1;
beta  = norm(r)*e1;

H = zeros(nIt+1,nIt); % upper Hessenberg matrix
Q = zeros(N,nIt);     % Arnoldi matrix
Q(:,1) = r/norm(r);   

C = zeros(nIt,1);			% sines and cosines of Givens' rotations
S = zeros(nIt,1);

err = e1;

for k = 1:nIt
	% Update base of Krylov subspace
	% - compute new direction in expanded Krylov subspace
	% temp = A * ( R \ Q(:,k) );
	if isnumeric(R)
		temp = R \ Q(:,k); 
	elseif isa(R,'function_handle')
		temp = R( Q(:,k) );
	end
	
	if isnumeric(A)
		temp = A*temp; 
	elseif isa(A,'function_handle')
		temp = A( temp );
	end
	
	[ H(1:k+1,k), Q(:,k+1) ]  = arnoldi( temp, Q(:,1:k), k );
	% Rotate to get rid of last element of H
	[ H(1:k+1,k), C(k), S(k)] = givensRotation( H(1:k+1,k), C, S, k );
	
	% Rotate rhs too
	beta(k+1) = -S(k)*beta(k);
  beta(k)   =  C(k)*beta(k);
	err(k+1)  = abs( beta(k+1) ) / bn;
  
	if ( err(k+1) <= tol )
		break;
	end
	

end

% Solve minimisation problem
y = H(1:k,1:k) \ beta(1:k);

% Recover solution
if isnumeric(R)
	x = x0 + R \ ( Q(:,1:k)*y ); 
elseif isa(R,'function_handle')
	x = x0 + R( Q(:,1:k)*y ); 
end

it = k;


end

