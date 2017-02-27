function TestSmigacz

% Radius ewolution

%%

figure;

axes;

grid on;

hold on;

% plot('ExpRadiu7s')

t_end = 100;           % time endpoint

a_start = 30e-6;       % initial radius

xlim([0, t_end]);

xlabel('t[sek]')

ylim([0,a_start]);

ylabel('a[\mum]');

[ xp yp ] = getpts;

X =linspace(0,500,500);

Y = interp1(xp,yp,X,'cubic');

plot(xp,yp,'+','color','r');

plot(X,Y); hold off;

grid on;

waves.wavelength = 532;

waves.theta = 0;

waves.polarization = 1;

r = Y*1e9;

m = ones(size(r))*1.93;

theta = (70:0.1:110)*pi/180;

for i = 1:length(r)

     waves.wavelength = 532;

waves.polarization = 0; % Ivv Ipp

Ivv(i,:) = GeneratePattern(r(i), m(i), theta, waves); waves.wavelength = 632; waves.polarization = 1; % Ihh Iss

Ihh(i,:) = GeneratePattern(r(i), m(i), theta, waves); plot(theta,Ivv(i,:),theta,Ihh(i,:));

pause(0.1)

end