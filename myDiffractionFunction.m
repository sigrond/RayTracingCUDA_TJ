function [ y ] = myDiffractionFunction( x )
%myDiffractionFunction Summary of this function goes here
%   Detailed explanation goes here

y=fresnelc(x);
y=y.*(y>=0)+atan(x*4)./atan(Inf).*0.5.*(y<0);

