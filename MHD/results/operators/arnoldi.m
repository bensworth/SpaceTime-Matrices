function [ h, q ] = arnoldi( q, Q, k )
%arnoldi Arnoldi method to extract orthonormal basis
%
% Quite the classic, straightforward implementation, using stabilised
% Gram-Schmidt to build the basis.
%
%
%   Input:
%    q       - Vector. New vector in the basis.
%    Q       - Matrix whose columns correspond to the base built so far.
%    k       - Integer. Current dimension of the subspace.
%
%		Output:
%    h       - Vector. New column of the upper Hessenberg matrix for the
%               subspace.
%    q       - Vector. New orthonormalised basis vector for the expanded
%               subspace.
%
%
% Author: Federico Danieli, Numerical Analysis Group
% (Shamelessly copied from Wikipedia:
% https://en.wikipedia.org/wiki/Arnoldi_iteration)
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% July 2019; Last revision: Jul-2019


% initialise new column of Hessenberg matrix
h = zeros( k+1, 1 );


% for each element in the basis
for i = 1:k
	% store projection in h
	h(i)= q'*Q(:,i);
	% orthogonise new element in the basis
	q = q - h(i)*Q(:,i);
end

% normalise
h(k+1) = norm(q);
q = q / h(k+1);

end