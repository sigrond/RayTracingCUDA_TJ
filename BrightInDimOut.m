function [ Bright, Dim ] = BrightInDimOut( Frame, X, Y )
%BRIGHTINDIMOUT ile jest punkt�w jasnych w �rodku kszta�tu, a ile ciemnych
%na zewn�trz
%   Detailed explanation goes here
Bw1 = roipoly(Frame,X,Y);
f=Frame./max(max(Frame));
fge=f>0.105;
FB=fge.*Bw1;
Bright=sum(sum(FB,2),1);

Bw2=~Bw1;
fle=f<0.105;
FD=fle.*Bw2;
Dim=sum(sum(FD,2),1);

end

