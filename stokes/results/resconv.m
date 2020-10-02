% Handy script to print residual behaviour from files containing
%  printouts of residual output after each GMRES iteration

% close all

% since this is a remote location, update list of matlab known files
rehash path


% Problem parameters
prec    = 1;	% preconditioner used (0: diag, 1:triangular)
STsolve = 2;  % solver for space-time velocity block: time-stepping 0, AMG 1, GMRES+AMG 2, parareal 3
PBtype  = 4;  % type of problem (1:cavity, 2:poiseuille, 3:step, 4:glazing)
Pe      = 10;  % peclet numer (only used if PBtype==4)
opts    = '_FGMRES_approx2';  % list of petsc options used: direct: '', iterative: '_approx2'

peString = '';
if ( PBtype == 4 && Pe >= 0 )
	peString = strcat('_Pe',num2str(Pe,'%10.6f'));
end
 
path = ['./convergence_results_Prec', num2str(prec),  ...
	      '_STsolve', num2str(STsolve), '_oU2_oP1_Pb',...
        num2str(PBtype),peString,'_rc_SpaceTimeStokes', opts];
if ( PBtype == 4 || PBtype == 1 )
	path = [path, '_SingAp'];
end

colors = {'-b', '-k', '-r', '-m', '-g', '-c', '-y'};

figure
			
if( ~exist(path,'dir') )
	disp('No results for specified experiment were found');
else
	
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
					res = res / res(1);
% 					errU = table2array(T(:,6));
% 					errP = table2array(T(:,7));
					
% 					subplot(1,3,1);
					semilogy( it, res, colors{r-1} );
% 					hold on
% 					subplot(1,3,2);
% 					semilogy( it, errU, colors{pow2} );
% 					hold on
% 					subplot(1,3,3);
% 					semilogy( it, errP, colors{pow2} );
					hold on
					
				catch
					disp(strcat('Something went wrong with file NP', int2str(np), '_r', int2str(r)));
				end
			end	
		end
	end
	hold off
	legend({'r2','r3','r4','r5','r6','r7','r8'})
end

