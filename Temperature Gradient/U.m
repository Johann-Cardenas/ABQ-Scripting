function [U]=U(z,s,t)
global aa1 aa2 ll1 ll2 H c
r1=sqrt(s/aa1);
r2=sqrt(s/aa2);
llrp=1+ll1*r1/ll2/r2;
llrm=1-ll1*r1/ll2/r2;
L=llrp/llrm;
A=(Fs(s,t)-c/s)/(1-L*exp(2*H*r1));
U=A*(exp(r1*z)-L*exp(r1*(2*H-z)));
