%% Uniform Spherical Distribution as in Matlab documantation OK
rng(0,'twister');
rvals = 2*rand(1000,1)-1;
elevation = asin(rvals);
azimuth = 2*pi*rand(1000,1);
radii=1;
[x,y,z] = sph2cart(azimuth,elevation,radii);
figure
plot3(x,y,z,'.');hold on;
axis equal
%% sector of USD OK
rng(0,'twister');
rvals = (2*rand(1000,1)-1)*sind(8.3);
elevation = asin(rvals);
azimuth = 2*pi*rand(1000,1)/10;
radii=1;
[x,y,z] = sph2cart(azimuth,elevation,radii);
figure
plot3(x,y,z,'.');hold on;
axis equal
%% as in Mariusz's PhD thesis OK
rng(0,'twister');
rvals = rand(1000,1);
elevation = 2*asin(sqrt(rvals))-pi/2;
% elevation2 =cat(elevation, -elevation);
azimuth = 2*pi*rand(1000,1);
radii=1;
[x,y,z] = sph2cart(azimuth,elevation,radii);
figure
plot3(x,y,z,'.')
axis equal
%% Uniform Grid Distribution
rvals = 0:0.1:.5;
% elevation = asin(2*rvals-1);
elevation = 2*asin(sqrt(rvals))+pi/2;
rvals2 =  0:0.05:1; % 2 times denser
azimuth = 2*pi*rvals2-pi;
[elevationgrid azimuthgrid] = meshgrid(elevation, azimuth);
radii=1;
[x,y,z] = sph2cart(azimuthgrid,elevationgrid,radii);
figure
plot3(x,y,z,'.b')
axis equal
%% sector of uniform grid distribution - no sqrt - OK
rvals2 =  0:0.0025:.1 ; % 2 times denser
azimuth = 2*pi*rvals2-pi;
rvalsrange = sind(8.3);
rvals = -rvalsrange:rvalsrange/10:rvalsrange;
elevation = asin(rvals);
[elevationgrid azimuthgrid] = meshgrid(elevation, azimuth);
radii=1;
[x,y,z] = sph2cart(azimuthgrid,elevationgrid,radii);
figure
plot3(x,y,z,'.b')
axis equal
%% sector of uniform grid distribution w/sqrt
rvals2 =  0:0.01:.1 ; % sholud be 2 times denser than for elevation
azimuth = 2*pi*rvals2-pi;
rvalsrange = (sind(45))^2;%8.3
rvals = 0:rvalsrange/10:rvalsrange;
elevation = 2*asin(sqrt(rvals))+pi/2;
[elevationgrid azimuthgrid] = meshgrid(elevation, azimuth);
radii=1;
[x,y,z] = sph2cart(azimuthgrid,elevationgrid,radii);
figure
plot3(x,y,z,'.b')
axis equal
