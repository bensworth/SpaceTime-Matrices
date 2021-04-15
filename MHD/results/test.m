path = './operators/MFEM_';
filename = strcat(path, 'Fu.dat');
Fu   = spconvert(load(filename));
filename = strcat(path, 'Mz.dat');
Mz   = spconvert(load(filename));
filename = strcat(path, 'B.dat');
B    = spconvert(load(filename));
filename = strcat(path, 'K.dat');
K    = spconvert(load(filename));
filename = strcat(path, 'Y.dat');
Y    = spconvert(load(filename));
filename = strcat(path, 'Z1.dat');
Z1   = spconvert(load(filename));
filename = strcat(path, 'Z2.dat');
Z2   = spconvert(load(filename));
filename = strcat(path, 'Fa.dat');
Fa   = spconvert(load(filename));

path = './operators/my_';
filename = strcat(path, 'Fu.dat');
myFu = spconvert(load(filename));
filename = strcat(path, 'Mz.dat');
myMz = spconvert(load(filename));
filename = strcat(path, 'B.dat');
myB  = spconvert(load(filename));
filename = strcat(path, 'K.dat');
myK  = spconvert(load(filename));
filename = strcat(path, 'Y.dat');
myY  = spconvert(load(filename));
filename = strcat(path, 'Z1.dat');
myZ1 = spconvert(load(filename));
filename = strcat(path, 'Z2.dat');
myZ2 = spconvert(load(filename));
filename = strcat(path, 'Fa.dat');
myFa = spconvert(load(filename));

dFu = ( myFu - Fu );
dMz = ( myMz - Mz );
dB  = ( myB  - B  );
dK  = ( myK  - K  );
dY  = ( myY  - Y  );
dZ1 = ( myZ1 - Z1 );
dZ2 = ( myZ2 - Z2 );
dFa = ( myFa - Fa );

disp( max(max(abs( dFu ))) );
disp( max(max(abs( dMz ))) );
disp( max(max(abs( dB  ))) );
disp( max(max(abs( dK  ))) );
disp( max(max(abs( dY  ))) );
disp( max(max(abs( dZ1 ))) );
disp( max(max(abs( dZ2 ))) );
disp( max(max(abs( dFa ))) );





