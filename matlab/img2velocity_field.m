A = imread('../vel_2d_output.png')
A = imresize(A, 0.1);
[dx,dy,c] = size(A)
vx = A(:,:,1)
vy = A(:,:,2)

vx = vx.*2 - 1
vy = vy.*2 - 1

[x, y] = meshgrid(linspace(0, 1, dx), linspace(0, 1, dy))

figure
quiver(x,y,vx,vy)