filename = 'original_Fu.dat';
Fu   = spconvert(load(filename));
filename = 'dirichlet_Fu.dat';
Fud  = spconvert(load(filename));

filename = 'original_B.dat';
B   = spconvert(load(filename));
filename = 'dirichlet_B.dat';
Bd  = spconvert(load(filename));

filename = 'original_f.dat';
f   = load(filename);
filename = 'dirichlet_f.dat';
fd  = load(filename);

filename = 'original_g.dat';
g   = load(filename);
filename = 'dirichlet_g.dat';
gd  = load(filename);

filename = 'dirichlet_uIG.dat';
uIGd= load(filename);

filename = 'essentialNodesQ.dat';
tmp= load(filename);
essQ=zeros(size(B,1),1);
essQ(tmp+1)=1;
filename = 'essentialNodesV.dat';
tmp= load(filename);
essV=zeros(size(Fu,1),1);
essV(tmp+1)=1;

figure
spy(Fu)
figure
spy(Fud)
figure
spy(B)
figure
spy(Bd)

