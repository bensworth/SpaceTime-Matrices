function plotAisland(delta,eps,beta)
x = linspace(0,1,20);
y = linspace(0,1,20)';

Afun = @(x,y) delta*log(cosh(y/delta)+eps*cos(x/delta)) + beta*cos(pi*y/2).*cos(pi*x);
figure
surf(Afun(x,y));

Ax = @(x,y) -eps*sin(x/delta)./(cosh(y/delta)+eps*cos(x/delta)) - pi  *beta*cos(pi*y/2).*sin(pi*x);
Ay = @(x,y) eps*sinh(y/delta)./(cosh(y/delta)+eps*cos(x/delta)) + pi/2*beta*sin(pi*y/2).*cos(pi*x);
figure
quiver(Ax(x,y),Ay(x,y))

lapA = @(x,y) (1-eps^2)./(delta *(cosh(y/delta)+eps*cos(x/delta)) ) - 5./4.*beta*pi^2*cos(pi*y/2).*cos(pi*x);

figure
surf(lapA(x,y))

figure
quiver(Ax(x,y).*lapA(x,y),Ay(x,y).*lapA(x,y));
end