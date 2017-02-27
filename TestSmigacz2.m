function TestSmigacz2

% Radius ewolution

%%

load evol 
load fraction
load Tp
load Ts

x1 = [1 : 1 : length(evol)]';
x2 = [1 : 5 : length(evol)]';

%evol = interp1(x1, evol, x2);

fraction = interp1(x2, fraction, x1);

[ref_indx1, ref_indx2]  = maxGarnettTest( fraction);

Ipp = zeros(length(evol), length(Tp));
Iss = zeros(length(evol), length(Ts));

%theta = (70:0.1:110)*pi/180;


for i = 1:length(evol)

    waves.wavelength = 532;

    waves.polarization = 0; % Ivv Iss

    Ipp(i,:) = GeneratePattern(evol(i), ref_indx1(i), Tp, waves); 

    waves.wavelength = 632; 

    waves.polarization = 1; % Ihh Ipp

    Iss(i,:) = GeneratePattern(evol(i), ref_indx2(i), Ts, waves); 

    %plot(Tp,Ivv(i,:),Ts,Ihh(i,:));

    %pause(0.1)

    disp(i); 

end

save Ipp Ipp
save Iss Iss


