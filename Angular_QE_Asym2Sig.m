function Angular_QEofPikeCCD = Angular_QE_Asym2Sig( angle )
% Angular distribution of Quantum Efficiency for Pike KAI-0340 sensor
%   Asymmetric double Sigmoidal function fitted to distribution obtained in
%   experiment by GD
% angle in [deg]
y0=0.12597;
xc=2.04687;
A=0.86527;
w1=18.65631;
w2=1.91634;
w3=1.75922;
Angular_QEofPikeCCD = y0+ A*(1./(1+exp(-(angle-xc+w1/2)/w2))).*(1-1./(1+exp(-(angle-xc-w1/2)/w3)));