function [ output_args ] = map2d( Frame,a1,a2 )
%map2d wyrysowanie zale¿noœci funkcji celu od jednej zmiennej
%   Detailed explanation goes here
for i=1:300
    point=[0,0,0,0,0,84.4+(i-100)*0.01];
    z(i)=84.4+(i-100)*0.01;
    map2d(i)=BrightnesScalarization(Frame,a1,a2,point);
end

figure
plot(map2d,z)

end

