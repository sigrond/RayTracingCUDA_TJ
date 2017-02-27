
clear all; close all;

load rExp
time = [1:length(rExp)]';

ft = fit(time,rExp,'splineinterp');

figure
plot(time,rExp,'.',time,ft(time));

