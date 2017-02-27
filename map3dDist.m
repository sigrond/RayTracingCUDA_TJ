function [ output_args ] = map3dDist( pointsr,pointsb )
%map3d wyrysowanie zale¿noœci funkcji celu od 2 zmiennych
%   Detailed explanation goes here
%xZ
%yZ
%zZ
%XZ
%YZ
map2dxZ=zeros(20,20);
map2dyZ=zeros(20,20);
map2dzZ=zeros(20,20);
map2dXZ=zeros(20,20);
map2dYZ=zeros(20,20);
y=zeros(20);
z=zeros(20);
y=linspace(-1,1,20);
z=linspace(83.5,85.5,20);
for i=1:20
    for j=1:20
        %xZ
        point=[(j-10)*0.1,0,0,0,0,84.5+(i-10)*0.1];
        map2dxZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
        
        %yZ
        point=[0,(j-10)*0.1,0,0,0,84.5+(i-10)*0.1];
        map2dyZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
        
        %zZ
        point=[0,0,(j-10)*0.1,0,0,84.5+(i-10)*0.1];
        map2dzZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
        
        %XZ
        point=[0,0,0,(j-10)*0.1,0,84.5+(i-10)*0.1];
        map2dXZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
        
        %YZ
        point=[0,0,0,0,(j-10)*0.1,84.5+(i-10)*0.1];
        map2dYZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
    end
    str=sprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\t\t%6.3f%%\n',((i-1)/0.2));
    display(str);
end

figure('Name','xZ')
mesh(y,z,map2dxZ);

figure('Name','yZ')
mesh(y,z,map2dyZ);

figure('Name','zZ')
mesh(y,z,map2dzZ);

figure('Name','XZ')
mesh(y,z,map2dXZ);

figure('Name','YZ')
mesh(y,z,map2dYZ);

end

