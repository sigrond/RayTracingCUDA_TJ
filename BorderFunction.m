function [ X Y ] = BorderFunction( PkX,PkY,PkZ,shX,shY,lCCD,lambda )
%BorderFunction funkcja generuj¹ca ramkê
%   Detailed explanation goes here
global efDr efDg efDb GSystem;

r=658;
g=532;
b=458;
try
    lambdas=evalin('base', 'lambdas');
catch
    lambdas=[r,g,b];
end
r=lambdas(1);
g=lambdas(2);
b=lambdas(3);

%handles.S=SetSystem;
handles.S=GSystem;
if(exist('lambda','var'))
    handles.S.lambda=lambda;
    if lambda==r
        efD  = efDr;
    elseif lambda==g
        efD  = efDg;
    elseif lambda==b
        efD  = efDb;
    else
        efD  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,handles.S.lambda,25);
    end
    handles.S.efD  = efD;
end
handles.S.m2 = Calculate_m(25,handles.S.lambda, 'BK7');

handles.S.Pk(1)=PkX;
handles.S.Pk(2)=PkY;
handles.S.Pk(3)=PkZ;
handles.shX=shX;
handles.shY=shY;
handles.S.lCCD=lCCD;

horizontalPoints=10;
verticalPoints=30;

%Br = zeros(4*handles.S.N,3);       % vector of border points
Br = zeros(horizontalPoints*2+verticalPoints*2,3);
% calculation of position for the 4 outer points
alpha = asin(handles.S.dW/2/handles.S.R_midl_El);
[X(1),Y(1)] = pol2cart(alpha,handles.S.R_midl_El);
[X(2),Y(2)] = pol2cart(-alpha,handles.S.R_midl_El);
Z(1) = -handles.S.dH/2;
Z(2) = handles.S.dH/2;
P(1,:) = [X(1),Y(1),Z(1)];
P(2,:) = [X(1),Y(1),Z(2)];
P(3,:) = [X(2),Y(2),Z(2)];
P(4,:) = [X(2),Y(2),Z(1)];

V = P(4,:) - handles.S.Pk; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*handles.S.Pk(1)+V(2)*handles.S.Pk(2));
C = handles.S.Pk(1)^2 + handles.S.Pk(2)^2 - (handles.S.R_dis_Ring)^2;
D = B^2-4*A*C;
t = (-B+sqrt(D) )/2/A;
P(5,:) = [ handles.S.Pk(1)+V(1)*t handles.S.Pk(2)+V(2)*t -handles.S.dH/2];

V = P(1,:) - handles.S.Pk; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*handles.S.Pk(1)+V(2)*handles.S.Pk(2));
C = handles.S.Pk(1)^2 + handles.S.Pk(2)^2 - (handles.S.R_dis_Ring)^2;
D = B^2-4*A*C;
t = (-B+sqrt(D) )/2/A;
P(6,:) = [ handles.S.Pk(1)+V(1)*t handles.S.Pk(2)+V(2)*t -handles.S.dH/2];

P(7,1:2) = P(6,1:2);
P(7,3)   = handles.S.dH/2;

P(8,1:2) = P(5,1:2);
P(8,3)   = handles.S.dH/2;
%
%Br(1:handles.S.N,1) = ones(1,handles.S.N)*P(1,1);
Br(1:horizontalPoints,1) = ones(1,horizontalPoints)*P(1,1);
%Br(1:handles.S.N,2) = ones(1,handles.S.N)*P(1,2);
Br(1:horizontalPoints,2) = ones(1,horizontalPoints)*P(1,2);
%Br(1:handles.S.N,3) = linspace(P(2,3),P(1,3),handles.S.N);
Br(1:horizontalPoints,3) = linspace(P(2,3),P(1,3),horizontalPoints);

V1 = P(6,:); % directing vector from origin to point 6
V2 = P(5,:); % directing vector from origin to point 5
Bt1 = acos( dot( V1(1:2) ,[1,0])/norm( V1(1:2) ));
Bt2 = acos( dot(V2(1:2),[1,0])/norm(V2(1:2)));
%VBt = linspace(Bt1,-Bt2,handles.S.N);
VBt = linspace(Bt1,-Bt2,verticalPoints);
%
[X,Y] = pol2cart(VBt,handles.S.R_dis_Ring);

%Br((handles.S.N+1):2*handles.S.N,1) = X;
Br((horizontalPoints+1):horizontalPoints+verticalPoints,1) = X;
%Br((handles.S.N+1):2*handles.S.N,2) = Y;
Br((horizontalPoints+1):horizontalPoints+verticalPoints,2) = Y;
%Br((handles.S.N+1):2*handles.S.N,3) = -ones(1,handles.S.N)*handles.S.dH/2;
Br((horizontalPoints+1):horizontalPoints+verticalPoints,3) = -ones(1,verticalPoints)*handles.S.dH/2;

%Br((2*handles.S.N+1):3*handles.S.N,1) = ones(1,handles.S.N)*P(4,1);
Br((horizontalPoints+verticalPoints+1):horizontalPoints*2+verticalPoints,1) = ones(1,horizontalPoints)*P(4,1);
%Br((2*handles.S.N+1):3*handles.S.N,2) = ones(1,handles.S.N)*P(4,2);
Br((horizontalPoints+verticalPoints+1):horizontalPoints*2+verticalPoints,2) = ones(1,horizontalPoints)*P(4,2);
%Br((2*handles.S.N+1):3*handles.S.N,3) = linspace(P(4,3),P(3,3),handles.S.N);
Br((horizontalPoints+verticalPoints+1):horizontalPoints*2+verticalPoints,3) = linspace(P(4,3),P(3,3),horizontalPoints);

%VBt = linspace(-Bt2,Bt1,handles.S.N);
VBt = linspace(-Bt2,Bt1,verticalPoints);
%
[X,Y] = pol2cart(VBt,handles.S.R_dis_Ring);
%Br((3*handles.S.N+1):4*handles.S.N,1) = X;
Br((horizontalPoints*2+verticalPoints+1):horizontalPoints*2+verticalPoints*2,1) = X;
%Br((3*handles.S.N+1):4*handles.S.N,2) = Y;
Br((horizontalPoints*2+verticalPoints+1):horizontalPoints*2+verticalPoints*2,2) = Y;
%Br((3*handles.S.N+1):4*handles.S.N,3) = ones(1,handles.S.N)*handles.S.dH/2;
Br((horizontalPoints*2+verticalPoints+1):horizontalPoints*2+verticalPoints*2,3) = ones(1,verticalPoints)*handles.S.dH/2;

handles.Br=Br;

if exist('RayTracing_MEX', 'file') == 3
    RayTracing_MEX_exist=1;
else
    RayTracing_MEX_exist=0;
end

for i = 1:size(handles.Br,1)
       Pd = [ handles.Br(i,1), handles.Br(i,2), handles.Br(i,3) ]; % Points on the diaphragm plane 
       %P = RayTracing(Pd,handles.S);
       if RayTracing_MEX_exist
           Ppom=RayTracing_MEX(Pd,handles);
           P=double(Ppom');
       else
           P = RayTracing(Pd,handles.S);
       end
       %if P~=Ppom'
       %    display(P);
       %end
       if size(P,1) ~= 11
           if(i==1)
               W1(i)=0;
               Hi1(i)=0;
           elseif i>2%==3
               W1(i) = 2*W1(i-1)-W1(i-2); % liniowa aproksymacja nastêpnego punktu, jeœli nie uda³o siê go lepiej wyliczyæ
               Hi1(i) = 2*Hi1(i-1)-Hi1(i-2);
           elseif i>3
               a=(W1(i-3)-W1(i-2))/(Hi1(i-3)^2-Hi1(i-2)^2);
               b=Hi1(i-1)-a*W1(i-1)^2;
               %d=((W1(i-1)-W1(i-2))^2+(Hi1(i-1)-Hi1(i-2))^2)^0.5;
               xw = 2*W1(i-1)-W1(i-2);
               yw = a*xw+b;
               yh = 2*Hi1(i-1)-Hi1(i-2);
               xh = ((yh-b)/a)^0.5;
               if xw^2+yw^2 < xh^2+yh^2
                   W1(i) = xw;
                   Hi1(i) = yw;
               else
                   W1(i) = xh;
                   Hi1(i) = yh;
               end
           else
               W1(i) = W1(i-1);
               Hi1(i) = Hi1(i-1);
           end
       else
%              if size(P,1) == 7
                 W1(i) = P(7,2);
                 Hi1(i) = P(7,3);
%              else 
              % Terminate the rays that don't hit the CCD element   
%                  W1(i) = NaN;
%                  Hi1(i) = NaN;
       end
 end
% Recalculation meters to pixels
% shifting the  origin to middle of the image.
% The center of image isn't placed  on [0,0] point, but on [240,320] point
 handles.R1(1,:) = (handles.S.CCDW/2 + W1)/handles.S.PixSize;  % [ Pix ]
 handles.R1(2,:) = (handles.S.CCDH/2 + Hi1)/handles.S.PixSize; % [ Pix ]
  

X = handles.R1(1,:) +  handles.shX;
Y = handles.R1(2,:) + handles.shY;

end

