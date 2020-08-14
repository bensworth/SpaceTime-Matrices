close all
% clear all
% clc


filename = 'out_original_B.dat';
Bo  = spconvert(load(filename));
filename = 'out_original_F.dat';
Fo  = spconvert(load(filename));
filename = 'out_original_M.dat';
Mo  = spconvert(load(filename));

filename = 'out_final_Mp.dat';
Mp  = spconvert(load(filename));
filename = 'out_final_Ap.dat';
Ap  = spconvert(load(filename));
filename = 'out_final_Wp.dat';
Wp  = spconvert(load(filename));

filename = 'out_final_B.dat';
Bd  = spconvert(load(filename));
filename = 'out_final_F.dat';
Fd  = spconvert(load(filename));
filename = 'out_final_M.dat';
Md  = spconvert(load(filename));

nu = size(Fo,1);
np = size(Bo,1);
NT = 3;
Nu = nu*NT;
Np = np*NT;

Zuu = sparse(nu,nu);
Zpp = sparse(np,np);
Zup = sparse(nu,np);
Zpu = sparse(np,nu);


FF = [Fd, Zuu, Zuu;...
	    Md,  Fd, Zuu;...
			Zuu, Md, Fd ];
BB = [Bd,  Zpu, Zpu;...
	    Zpu,  Bd, Zpu;...
			Zpu, Zpu, Bd ];

A = [FF, BB';...
	   BB, sparse(Np,Np)];
	 
FFo = [Fo, Zuu, Zuu;...
	     Mo,  Fo, Zuu;...
			Zuu,  Mo, Fo ];
BBo = [Bo,  Zpu, Zpu;...
	     Zpu,  Bo, Zpu;...
			 Zpu, Zpu, Bo ];

Ao = [FFo, BBo';...
	    BBo, sparse(Np,Np)];

		
u0 = zeros( NT*nu, 1 );
f0 = zeros( NT*nu, 1 );
p0 = zeros( NT*np, 1 );
g0 = zeros( NT*np, 1 );
for j = 0:NT-1
	temp = load(['IGu.dat.', num2str(j)]);
	u0(nu*j+(1:nu),1) = temp(2:end);
	temp = load(['RHSu.dat.', num2str(j)]);
	f0(nu*j+(1:nu),1) = temp(2:end);
	temp = load(['IGp.dat.', num2str(j)]);
	p0(np*j+(1:np),1) = temp(2:end);
	temp = load(['RHSp.dat.', num2str(j)]);
	g0(np*j+(1:np),1) = temp(2:end);
end

rhs = [ f0; g0 ];
x0  = [ u0; p0 ];


% check that dirichlet modifications to rhs are exact
% - here's the output from the rhs before modifications
rhs0 = [0.433333 -0.0111111 0.433333 -0.0111111 0.266667 0.0888889 0.133333 0.0888889 1.91111 0.444444 -1.66533e-16 -0.444444 1.66533e-16 0 0.888889 0 -0.888889 0 0.444444 0 0.444444 0 0 0 0 0 1.77778 0.444444 -1.66533e-16 -0.444444 1.66533e-16 0 0.888889 0 -0.888889 0 0.444444 0 0.444444 0 0 0 0 0 1.77778 0.444444 -1.66533e-16 -0.444444 1.66533e-16 0 0.888889 0 -0.888889 0 0 0 0 0 0 0 0 0 0 0 0 0]';

essNodesV = [1:4,6,8:13,15,17,18];
essNodesQ = [2,4];
uBC = zeros(nu,1); uBC(9) = 1;

uga = rhs0;

diagFo = full(diag(Fo));

for i=0:NT-1
	if i==0
		uga( (1:nu) + (i*nu) ) = uga( (1:nu) + (i*nu) )  - Fo*uBC;
	else
		uga( (1:nu) + (i*nu) ) = uga( (1:nu) + (i*nu) )  - Fo*uBC - Mo*uBC;	
	end
	uga( essNodesV + (i*nu) ) = diagFo(essNodesV).*uBC(essNodesV);
	
	uga( nu*NT + i*np + (1:np) )  = uga( nu*NT + i*np + (1:np) ) - Bo*uBC;
end

max(abs(uga-rhs))



%% Solve
% GMRES
prec = @(b) fakePrecon( b, Fd, Md, Bd, Ap, Mp, Wp, essNodesQ );
[ x, err, it ] = GMRESrp( A, rhs, 1e-10, 50, x0, prec );
err
