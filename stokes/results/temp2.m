close all
clear all
clc

frhs  = 'rhsu';
fuig  = 'IGu';
fusol = 'usol';
nFile = 7;
rhsu = load([frhs,  '0.dat']);
igu  = load([fuig,  '0.dat']);
usol = load([fusol, '0.dat.0']);
NX = length(usol) - 1;
NT = length(rhsu) / NX;
rhsu = zeros( NT*NX, nFile );
igu  = zeros( NT*NX, nFile );
usol = zeros( NT*NX, nFile );
for i = 0:nFile-1
	rhsu(:,i+1) = load([frhs, num2str(i), '.dat']);
	igu(:,i+1)  = load([fuig, num2str(i), '.dat']);
	for j = 0:NT-1
		temp = load([fusol, num2str(i), '.dat.', num2str(j)]);
		usol(NX*j+(1:NX),i+1) = temp(2:end);
	end
end

u0 = zeros( NT*NX, 1 );
f0 = zeros( NT*NX, 1 );
for j = 0:NT-1
	temp = load(['IGu.dat.', num2str(j)]);
	u0(NX*j+(1:NX),1) = temp(2:end);
	temp = load(['RHSu.dat.', num2str(j)]);
	f0(NX*j+(1:NX),1) = temp(2:end);
end


