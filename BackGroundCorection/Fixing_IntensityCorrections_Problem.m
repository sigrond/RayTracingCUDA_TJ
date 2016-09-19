%% loading data 
load('C:\Our_soft\RayTracingCUDA_TJ\BackGroundCorection\SumSpecleFram_and_RaytrasingCorrection.mat');
%% E:\Justice_Archer\50mM SDS_DEG_SiO2 suspension\50mM suspension\01.09.2016V0.avi
figure;
plot(mean(double(Frame(:,:,1)),1)./max(mean(double(Frame(:,:,1)))));
hold on;
plot(mean(I_CR,1)./max(mean(I_CR,1)),'r');
hold off;
%%
figure;
% FR = Fr11(:,:,1)-60;
 FR = Fr3705(:,:,1);
surf(FR,'linestyle','non')
%%
FR = Fr3705(:,:,1)-60;
iz = FR<=0;
FR(iz)=0;
surf(FR(Vx,Vy)./(I_CR(Vx,Vy)-0.1),'linestyle','non');view([0,90])
%%
figure;
FR = Fr3705(:,:,1);
surf(FR(Vx,Vy),'linestyle','non');view([0,90])
%%
% figure;
plot(median(FR(Vx,Vy),1));hold on;
plot(median( (FR(Vx,Vy)-60)./(I_CR(Vx,Vy)-.0),1));hold off;
%%
surf((I_CR(Vx,Vy)),'linestyle','non')
%%
FR = mean(double(Frame(:,:,1)),1)./max(mean(double(Frame(:,:,1))));
FB = (FB-min(FB))./max(FB-min(FB));
plot(FB);
hold on;
I_CBn = mean(I_CR,1)./max(mean(I_CR,1));
I_CBn = (I_CBn-0.4)./max((I_CBn-0.4));
plot(I_CBn,'r');
plot(FB./I_CBn,'m')
hold off;
%%
% figure;
FR = mean(double(Frame(:,:,1)),1)-2.13e5;
plot(FR);
hold on;
I_CRn = mean(I_CR,1);
I_CRn = I_CRn./max(I_CRn);
plot(FR./I_CRn,'r');
plot(FR,'m');
hold off;
%%
figure;
Vx = 200:300;
Vy = 100:500;
surf(Frame(Vx,Vy,1)./I_CR(Vx,Vy),'linestyle','non');
%% 
figure;
plot(mean(Frame(Vx,Vy,1)./I_CR(Vx,Vy),1))
%%
figure;
surf(I_CR,'linestyle','non');
%% 
figure;
FB = mean(double(Frame(:,:,3)),1)./max(mean(double(Frame(:,:,1))));
FB = (FB-min(FB))./max(FB-min(FB));
plot(FB);
hold on;
I_CBn = mean(I_CB,1)./max(mean(I_CB,1));
I_CBn = (I_CBn-0.4)./max((I_CBn-0.4));
plot(I_CBn,'r');
hold off;
%% 
figure;
Red = Frame(:,:,1);
surf(Red,'linestyle','non');
figure;
Vx = 200:300;
Vy = 100:500;
surf(Red(Vx,Vy)./I_CR(Vx,Vy),'linestyle','non');
%% Odejmujemy tlo
figure;
Red = Frame(:,:,1)-2e5;
surf(Red,'linestyle','non');
figure;
Vx = 200:300;
Vy = 100:500;
surf(Red(Vx,Vy)./I_CR(Vx,Vy)-0.18,'linestyle','non');
%% Testing Justice data
figure;
Vx = 200:300;
Vy = 100:400;
surf(F299(Vx,Vy,1),'linestyle','none');
title('Raw data')
figure;
surf(F299(Vx,Vy,1)./I_CR(Vx,Vy),'linestyle','none');
title('Corected data')
%% 
figure;
R = F299(:,:,1)-100;
id_z = (R<=0);
R(id_z) = 0;
plot((Theta_R(:)+90),R(:),'.');hold on;
R2 = R./I_CR;
plot((Theta_R(:)+90),R2(:),'.','color','r')
ylim([0 2e3])
%%
figure;
R = F299(:,:,1);
id_z = (R<=0);
R(id_z) = 0;
plot((Theta_R(1:step:end)+90),R(1:step:end),'.');
ylim([0 2e3])
%% 

