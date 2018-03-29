%Nonlinear regression
%@Zhaozheng Yin, Missouri S&T, Spring 2017

clc; clear all; close all;

%generate data
x = -20:1:20; I = length(x);
phi = [3 2 0 5 7]; 
w = zeros(1,I);
for ii=1:length(phi)
    w = w+phi(ii)*x.^(ii-1);
end
w = w + 100000*rand(1,I);
w = w';

%Nonlinear regression by polynomial regression
Z = [];
for ii = 1:length(phi)
    Z = [Z; x.^(ii-1)];
end
Phi_poly = (Z*Z')\(Z*w)

%nonlinear regression by radial basis function
nRadBasis = 6;
minx = min(x); maxx = max(x);
alpha_rad = minx:(maxx-minx)/(nRadBasis-1):maxx;
lambda_rad = 100;
Z = ones(1,I);
for ii = 1:nRadBasis
    Z = [Z; exp(-(x-alpha_rad(ii)).^2/lambda_rad)];
end
Phi_rad = (Z*Z')\(Z*w);

%nonlinear regression by arc tangent function
nArcTanBasis = 7;
minx = min(x); maxx = max(x);
alpha_arc = minx:(maxx-minx)/(nArcTanBasis-1):maxx;
lambda_arc = .1;
Z = [];
for ii = 1:nArcTanBasis
    Z = [Z; atan(lambda_arc*x-alpha_arc(ii))];
end
Phi_arc = (Z*Z')\(Z*w);

%visualization
figure; hold on;
plot(x,w,'k+','Markersize',10);
xx = -20:.1:20;
Z_poly = [];
for ii = 1:length(phi)
    Z_poly = [Z_poly; xx.^(ii-1)];
end
ww_poly = Z_poly'*Phi_poly;
plot(xx,ww_poly,'r','Linewidth',4);

Z_rad = ones(1,length(xx));
for ii = 1:nRadBasis
    Z_rad = [Z_rad; exp(-(xx-alpha_rad(ii)).^2/lambda_rad)];
end
ww_rad = Z_rad'*Phi_rad;
plot(xx,ww_rad,'g','Linewidth',2);

Z_arc = [];
for ii = 1:nArcTanBasis
    Z_arc = [Z_arc; atan(lambda_arc*xx-alpha_arc(ii))];
end
ww_arc = Z_arc'*Phi_arc;
plot(xx,ww_arc,'b','Linewidth',2);

legend('Data','Poly','Radial','Arctan');
