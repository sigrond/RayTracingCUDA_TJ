function [ distance ] = MeanSquaredDistance( PositionR, PositionB, Args )
%MEANSQUAREDDISTANCE Ta funkcja ma byæ minimalizowan¹ funkcj¹ celu
%   Detailed explanation goes here
r=658;
g=532;
b=458;

Px=Args(1);
Py=Args(2);
Pz=Args(3);
ShX=Args(4);
ShY=Args(5);
lCCD=Args(6);

distance=0;
notused=0;

for i=1:length(PositionR(:,1))
    [X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,r);
    tmp=(BorderDistance(X,Y,PositionR(i,1),PositionR(i,2)))^2;
    if(tmp==NaN || tmp==Inf || isnan(tmp) || isinf(tmp))

        tmp=10;

        notused=notused-1;
    end
    distance=distance+tmp;
end

for i=1:length(PositionB(:,1))
    [X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,b);
    tmp=(BorderDistance(X,Y,PositionB(i,1),PositionB(i,2)))^2;
    if(tmp==NaN || tmp==Inf || isnan(tmp) || isinf(tmp))

        tmp=10;

        notused=notused-1;
    end
    distance=distance+tmp;
end

distance=distance/((length(PositionR(:,1))+length(PositionB(:,1))-notused)^2);
%[1,1,1,1.5,1.5,Args(6)+3]
if abs(Args(1))>1 || abs(Args(2))>1 || abs(Args(3))>1 || abs(Args(4))>1.5 || abs(Args(5))>1.5 || Args(6)<81 ||Args(6)>87
    distance=distance+10000;
end

if(distance==NaN || distance==Inf || isnan(distance) || isinf(distance))
    distance=10000;
end

end

