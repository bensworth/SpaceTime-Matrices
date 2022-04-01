function [ T ] = sparseBlockLowTri( c )
%sparseBlockLowTri Generates sparse block-lower-triangular matrix with blocks specified diagonal-wise
%
% NB: this function assumes that all blocks have the same size, and that
% the main diagonal c{1} is always filled
%
% Syntax:  [ T ] = sparseBlockLowTri( c )
%
%   Input:
%    c       - Cell Vector of cell vectors of sparse matrices. Each entry
%               corresponds to a diagonal (starting from the main one,
%               moving downwards). Diagonals are filled starting from the
%               leftmost column, taking elements of c in order (extra blocks
%               in c are ignored). Its length defines the size of the matrix.
%
%   Output:
%    T       - Sparse Matrix. Block-lower-triangular matrix.
%
%
% Author: Federico Danieli, Numerical Analysis Group
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% Mar 2021; Last revision: Mar-2021
%

% Block sizes
[Nr,Nc] = size(c{1}{1});
% Total number of nonzero elements per each block diagonal
totNZ = zeros(length(c),1);

for dd=1:length(c)
	totNZ(dd) = sum(cellfun(@nnz,c{dd}(1:(min(end,length(c)-dd+1)))));
end

% Initialise arrays for final sparse matrix T (that is, i,j, and vals) 
ii = zeros(sum(totNZ),1);
jj = zeros(sum(totNZ),1);
ss = zeros(sum(totNZ),1);


% For each diagonal
for dd=1:length(c)
	% compute the number of non-zero elements per each block
	lclNZ = cellfun(@nnz,c{dd}(1:(min(end,length(c)-dd+1))));
	% initialise new data to insert
	ddi  = zeros(sum(lclNZ),1);
	ddj  = zeros(sum(lclNZ),1);
	dds  = zeros(sum(lclNZ),1);

	% for each block in this diagonal
	for bb = 1:(min(length(c{dd}),length(c)-dd+1))
		% decompose it into its indices
		[curri,currj,currs] = find( c{dd}{bb} );
		% include them in the global diagonal indices
		currIdx = sum(lclNZ(1:(bb-1)));
		ddi((currIdx+1):(currIdx+lclNZ(bb))) = curri+Nr*(bb+dd-2);
		ddj((currIdx+1):(currIdx+lclNZ(bb))) = currj+Nc*(bb-1);
		dds((currIdx+1):(currIdx+lclNZ(bb))) = currs;
	end

	% include the indices in the global block lower triangular matrix
	currIdx = sum(totNZ(1:(dd-1)));
	ii((currIdx+1):(currIdx+totNZ(dd))) = ddi;
	jj((currIdx+1):(currIdx+totNZ(dd))) = ddj;
	ss((currIdx+1):(currIdx+totNZ(dd))) = dds;


end



% Finally assemble matrix
T = sparse(ii,jj,ss,Nr*length(c),Nc*length(c));


end
