function Angular_QEofPikeCCD = Angular_QE( angle )
% Angular distribution of Quantum Efficiency for Pike KAI-0340 sensor
%   Sigmoidal Boltzman function fitted to distribution given by the
%   manufacturer for monochrome sensor w/microlenses - good fit until ~18
%   deg, farther a bit off
% angle in [deg]
A1=0.67411;
A2=105.55422;
x0=12.75482;
dx=-4.30795;
Angular_QEofPikeCCD = zeros(length(angle),1);
for counter=1 : length(angle)
if angle(counter) >= 0
    Angular_QEofPikeCCD(counter) = 0.01*((A1-A2) ./ (1 + exp((angle(counter)-x0)./dx)) + A2);
else
    Angular_QEofPikeCCD(counter) = 0.01*((A1-A2) ./ (1 + exp((-angle(counter)-x0)./dx)) + A2);
end
end

