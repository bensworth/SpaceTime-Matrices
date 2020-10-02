filename = 'out_original_F6.dat';
F6   = spconvert(load(filename));
filename = 'out_original_F0.dat';
F0   = spconvert(load(filename));
filename = 'out_original_M.dat';
Mu  = spconvert(load(filename));
filename = 'out_original_B.dat';
B   = spconvert(load(filename));

filename = 'out_final_F.dat';
Fud  = spconvert(load(filename));
filename = 'out_final_M.dat';
Mud = spconvert(load(filename));
filename = 'out_final_B.dat';
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

