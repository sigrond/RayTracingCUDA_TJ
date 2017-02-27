function koef = gentab_angle(fi_diafragmy,hccd_max,lambda) 

% Generacja tablicy wi¹¿¹cej alfa0 i hccd unormowane do hccdmax;
% tylko w tym programie trzeba jawnie wprowadziæ fi_diafragmy i hccd_max,
% oba w [m]

% wyznaczanie alpha_0_max ze œrednicy diafragmy w [m]
alpha_0_max=atan(fi_diafragmy/(2*0.01686));
% a=16.768*pi/180; % fi=10.16;

ap=alpha_0_max/1000;
% fid=fopen('tablica.txt','wt');
tablica = zeros(2,1400);
for i=1:1400
    hccdnorm = aberracja1(  (i-200)*ap, hccd_max,fi_diafragmy,lambda);
%     tablica(1,i) = tan((i-200)*ap)/tan(alpha_0_max); 
    tablica(1,i) = (i-200)*ap; 
    tablica(2,i) = hccdnorm;
%     fprintf( fid,'%10.5f       %1.8f\n',tan( i*ap )/tan( alpha_0_max ),hccdnorm );
end;
% fclose(fid);

[v ind ] = find(tablica(1,:));
tablica = tablica(:,ind);

cfun = fit(tablica(2,:)',tablica(1,:)','rat43'); % dorobic fitoptions( Algorithm, 'Levenberg-Marquardt' )
koef = coeffvalues( cfun );
fitt=@(x,k) ((k(1)*x^4+k(2)*x^3+k(3)*x^2+k(4)*x+k(5))/(x^3+k(6)*x^2+k(7)*x+k(8)));
%==========================================================================

figure;
plot(tablica(2,:),tablica(1,:));
hold on;
for kk = 1:length(tablica(2,:))
    temp(kk) = fitt(tablica(2,kk),koef);
end;
plot(tablica(2,:),temp,'color','r')
grid on;