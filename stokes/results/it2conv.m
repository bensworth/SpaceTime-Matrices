% Handy script to collect number of iterations necessary to convergence,
%  varying the desired convergence tolerance, from files containing
%  printouts of residual output after each GMRES iteration

close all
clear all
clc

% since this is a remote location, update list of matlab known files
rehash path


% Problem parameters
prec    = 1;	% preconditioner used (0: diag, 1:triangular)
STsolve = 3;  % whether the space-time velocity block is solved via time-stepping (0) or with AMG (1)
PBtype  = 4;  % type of problem (1:cavity, 2:poiseuille, 3:step, 4:glazing)
Pe      = 1;  % peclet numer (only used if PBtype==4)
opts    = 'rc_SpaceTimeStokes_approx2';  % list of petsc options used

peString = '';
if ( PBtype == 4 && Pe >= 0 )
	peString = strcat('_Pe',num2str(Pe,'%10.6f'));
end
 
tol = 1e-6;	% required tolerance to decide whether convergence was reached or not

path = ['./convergence_results_Prec', num2str(prec),  ...
	      '_STsolve', num2str(STsolve), '_oU2_oP1_Pb',...
        num2str(PBtype),peString,'_',opts];

if( ~exist(path,'dir') )
	disp('No results for specified experiment were found');
else

	its2conv = zeros(7);
	
	for pow2 = 1:7
		for r = 2:8
			np = 2^pow2;
			
			filename = [path,'/NP',num2str(np),'_r',num2str(r),'.txt'];
			tempfile = [path,'/temp.txt'];
			if( ~exist( filename, 'file' ) )
				disp(['Result file not found for NP=',num2str(np),' and r=',num2str(r)]);
				
			else
				% this next command cleans the convergence file in case it was
				%  restarted and multiple convergence histories were collated
				%  together
				command = ['tac ', filename,' | sed ''/Residual norms/q'' | tac | tee ', tempfile ];
				[dummy, dummier] = unix(command);
				% read from table
				T = readtable( tempfile );
				% extract relevant info (its and residual norm)
				try
					it  = table2array(T(:,1));
					res = table2array(T(:,5));
				
					% simply take last value of it if tolerance is min
					if tol == 1e-10
						its2conv( r-1,pow2 ) = it(end);
					else
						% otherwise find first occurrence in residual which is smaller than tol
						its2conv( r-1,pow2 ) = it( find( res<tol, 1 ) );
					end
				catch
					its2conv( r-1,pow2 ) = 0;
					disp(strcat('Something went wrong with file NP', int2str(np), '_r', int2str(r)));
				end
			end	
		end
	end
	
end

its2conv