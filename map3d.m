function [ output_args ] = map3d( Frame,a1,a2 )
%map3d wyrysowanie zale¿noœci funkcji celu od 2 zmiennych
%   Detailed explanation goes here
map2d=zeros(200,200);
y=zeros(200);
z=zeros(200);
y=linspace(-1,1,200);
z=linspace(84.2,86.2,200);
for i=1:200
    for j=1:200
        point=[0,0,(j-100)*0.01,0,0,85.2+(i-100)*0.01];
        map2d(i,j)=BrightnesScalarization(Frame,a1,a2,point);
    end
    str=sprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\t\t%6.3f%%\n',((i-1)/2));
    display(str);
end

figure
mesh(y,z,map2d);

end

