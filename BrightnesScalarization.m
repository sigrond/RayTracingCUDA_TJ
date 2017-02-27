function [ score ] = BrightnesScalarization( Frame,a1,a2, Args )
%BrightnesScalarization funkcja skalaryzujaca jasnoœæ na zewn¹trz i
%wewn¹trz
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

[X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,r);

alpha=a1;%0.105;

Bw1 = roipoly(Frame(:,:,1),X,Y);
f=Frame(:,:,1)./max(max(Frame(:,:,1)));
fge=f>alpha;
FBr=fge.*Bw1;
%Bright=sum(sum(FB,2),1);
Bw2=~Bw1;
fle=f<alpha;
FDr=fle.*Bw2;
%Dim=sum(sum(FD,2),1);

%[Br, Dr]=BrightInDimOut(Frame(:,:,1),X,Y);

[X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,b);
%[Bb, Db]=BrightInDimOut(Frame(:,:,3),X,Y);

alpha2=a2;%0.017;

Bw1 = roipoly(Frame(:,:,1),X,Y);
f=Frame(:,:,3)./max(max(Frame(:,:,3)));
fge=f>alpha2;
FBb=fge.*Bw1;
Bw2=~Bw1;
fle=f<alpha2;
FDb=fle.*Bw2;

FB=FBr|FBb;
FD=FDr&FDb;

B=sum(sum(FB,2),1);
D=sum(sum(FD,2),1);

a=3;
b=1;
score=-(a*B+b*D);

end

