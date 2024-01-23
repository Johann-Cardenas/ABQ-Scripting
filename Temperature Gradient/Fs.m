% Sinusoidal
function [Fs]=Fs(s,t)
global Tmax Tmin
Fs=(Tmax+Tmin)/2/s+6*pi*(Tmax-Tmin)/(pi^2+144*s^2);
