[x,y] = meshgrid(0:0.05:1,0:0.05:1);
u = (sin(2*pi*x).^2).*sin(4*pi*y);
v = -sin(4*pi*x).*(sin(2*pi*y).^2);

figure
quiver(x,y,u,v)
