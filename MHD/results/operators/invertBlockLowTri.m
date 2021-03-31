function y = invertBlockLowTri( x, diags, ess )
%invertBlockLowTri Inverts block lower triangular matrix
% 
% This function is implemented with block diagonal/bidiagonal/tridiagonal matrices
% stemming from time-discretisations in mind, but can support any number of diagonals.
% For this reason, input is provided diagonal-wise.
%
%
% Syntax:  y = invertBlockLowTri( x, diags )
% 
%   Input:
%    x       - Vector. Right-hand side of the system
%    diags   - Cell of (cells of) matrices. Each entry represents a block diagonal:
%               if a diagonal consists of only one matrix, then it's consider constant
%               down the diagonal, otherwise each block is taken in order. The main
%               diagonal (first entry) should contain the decomposed inverse operators.
%               Diagonals should be provided in order, and there should be no empty
%               matrices (provide a sparse one, in case).
%    ess     - Array of ints. Nodes identifying Dirichlet BC (for single time-step).
%               (Default is empty).
%
%
%  Output:
%    y       - Vector. Results from inversion.
%
%
% Author: Federico Danieli, Numerical Analysis Group
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% Mar 2021; Last revision: Mar-2021

if (nargin<3)
	ess = [];
end

% size of spatial unknowns, number of time steps, number of sub-diagonals
NX = diags{1}{1}.MatrixSize(1); % you must have something on the main diagonal
NT = length(x)/NX;
ND = length(diags)-1;

% if only one matrix is provided in a diag, that diag is considered constant
isConst = (cellfun(@length,diags) == 1);


% solution at previous iterations (for time-stepping)
yprev = zeros(NX,ND);

% initialize solution
y = zeros(size(x));

% Time step
for ii=1:NT
  currX = x((1:NX)+NX*(ii-1));
  
  % for each subdiag
  for dd = 1:ND
    % if I'm ahead enough with the time-steps
    if ii>dd
      % include its contribution to the rhs
      % - pick the right index in the diag depending on whether it's constant or not
      currX = currX - diags{dd+1}{ isConst(dd+1) + (1-isConst(dd+1))*(ii-dd) } * yprev(:,dd);
    end
  end

  % solve for current time-step
  y((1:NX)+NX*(ii-1)) = diags{1}{ isConst(1) + (1-isConst(1))*ii } \ currX;

  % store solutions at previous time-steps
  if ND>0
    if ND>1
      yprev(:,2:end) = yprev(:,1:end-1);
    end
    yprev(:,1) = y((1:NX)+NX*(ii-1));
  end

  % adjust dirichlet contributions
  y((ess)+NX*(ii-1)) = x((ess)+NX*(ii-1));
  yprev(ess,:) = 0.;

end




end