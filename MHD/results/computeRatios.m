function [res] = computeRatios()
% pathTS = './Pb6_Prec2_STsolveU9_STsolveA9_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D_test/';
% path   = './Pb6_Prec2_STsolveU0_STsolveA3_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D_test/';
pathTS = './Pb7_Prec2_STsolveU9_STsolveA9_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D_test/';
path   = './Pb7_Prec2_STsolveU0_STsolveA3_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D/';

filename = 'Newton_convergence_results.txt';

tableTS = table2array(readtable([pathTS,filename]));
table   = table2array(readtable([path,filename]));

% - just take average as is
% avgNewtit  =   tableTS(:,7);
% avgGMRESit =   tableTS(:,10);

% - adjust average for "stalling"
NT          = tableTS(:,2);
newtNonConv = tableTS(:,6);
totNewt     = tableTS(:,5);
totGMRES    = tableTS(:,8);
avgNewtit   = totNewt  ./(NT-newtNonConv);
avgGMRESit  = totGMRES ./(NT-newtNonConv);

newtit    = table(:,5);
GMRESit   = table(:,6).*newtit;


NL = round(newtit./avgNewtit, 2);
L  = round(GMRESit./avgGMRESit, 2);

numrows = 7;
NL = reshape( NL, [numrows,length(L)/numrows] );
L  = reshape(  L, [numrows,length(L)/numrows] );

res = [NL;L];
res = reshape(res,[numrows,numel(res)/numrows]);
end