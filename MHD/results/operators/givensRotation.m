function [ h, ck, sk ] = givensRotation( h, c, s, k )
%givensRotation Applies Givens' rotations to a vector
%
%   Input:
%    h       - Vector to rotate
%    c       - Vector. Cosines of rotation angles.
%    s       - Vector. Sines of rotation angles.
%    k       - Integer. Current dimension of Krylov subspace (just to know
%               where to stop rotating).
%
%		Output:
%    h       - Vector. Rotated h.
%    ck      - Scalar. Cosine of angle for next rotation.
%    sk      - Scalar. Sine of angle for next rotation.
%
%
% Author: Federico Danieli, Numerical Analysis Group
% (Shamelessly copied from Wikipedia:
% https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% July 2019; Last revision: Jul-2019


% apply rotation to each column
for i = 1:k-1                              
	temp   =  c(i)*h(i) + s(i)*h(i+1);
	h(i+1) = -s(i)*h(i) + c(i)*h(i+1);
	h(i)   = temp;
end

% find sin/cos values for next rotation
if ( h(k)==0 )
	ck = 0;
	sk = 1;
else
	t = sqrt( h(k)^2 + h(k+1)^2 );
	ck = abs( h(k) ) / t;
	sk = ck * h(k+1) / h(k);
end


%eliminate H(i+1,i)
h(k) = ck*h(k) + sk*h(k+1);
h(k+1) = 0.0;

end