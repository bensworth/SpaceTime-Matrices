function plotAtearing(lambda,beta,L)
x = linspace(0,L,L*20);
y = linspace(0,1,20)';
Afun = @(x,y) log(cosh(lambda*(y-0.5)))/lambda + beta*sin(pi*y).*cos(2*pi*x/L);
figure
surf(Afun(x,y));

Ax = @(x,y)                      - 2*pi/L*beta*sin(pi*y).*sin(2*pi*x/L);
Ay = @(x,y) tanh(lambda*(y-0.5)) +   pi  *beta*cos(pi*y).*cos(2*pi*x/L);
figure
quiver(Ax(x,y),Ay(x,y))

lapA = @(x,y) lambda * cosh(lambda*(y-0.5)).^(-2) - beta*sin(pi*y).*cos(2*pi*x/L) * pi^2*(1+4/L^2);

figure
surf(lapA(x,y))

figure
quiver(Ax(x,y).*lapA(x,y),Ay(x,y).*lapA(x,y));
end