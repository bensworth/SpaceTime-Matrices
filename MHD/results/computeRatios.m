function [res] = computeRatios()
% pathTS = './Pb6_Prec2_STsolveU9_STsolveA9_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D/';
% path   = './Pb6_Prec2_STsolveU0_STsolveA3_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D_test/';
pathTS = './Pb7_Prec2_STsolveU9_STsolveA9_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D/';
path   = './Pb7_Prec2_STsolveU0_STsolveA3_oU3_oP2_oZ1_oA1_rc_SpaceTimeIMHD2D/';

filename = 'Newton_convergence_results.txt';

tableTS = table2array(readtable([pathTS,filename]));
table   = table2array(readtable([path,filename]));

avgNewtit  = tableTS(:,7);
avgGMRESit = tableTS(:,10);
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