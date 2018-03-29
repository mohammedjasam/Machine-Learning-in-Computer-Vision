%Dual Linear regression

clc; clear all; close all;
dir_training = 'training\';
dir_testing = 'testing\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir([dir_training '*.jpg']);
X = []; w = [];
for ii = 1:size(files,1)
    filename = files(ii).name;
    w = [w; str2double(filename(1:4))];
    im = imread([dir_training filename]);
    im = im(:,:,1);
    X = [X im(:)];
end
X = double(X); %every column in X is one vectorized input image
X = [ones(1,size(X,2)); X];

psi=(X'*X)\w;   

phi=X*psi;

I = size(X,2);
temp = w - X'*phi;
prior_var = (temp' * temp) / I;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir([dir_testing '*.jpg']);
X_t = []; w_t = [];
for ii = 1:size(files,1)
    filename = files(ii).name;
    w_t = [w_t; str2double(filename(1:4))];
    im = imread([dir_testing filename]);
    im = im(:,:,1);
    X_t = [X_t im(:)];
end
X_t = double(X_t); %every column in X is one vectorized input image
X_t = [ones(1,size(X_t,2)); X_t];

X_t_var=var(X_t,0,2);

I = size(X,2);
lamda=1000; % lamda range is:10000','1000','100','10','1','.1','.01','.001','.0001','.00001','.000001'
% Compute A_inv.   
A_inv = inv ((X')*(X)*(X')*(X) + (lamda*eye(I)));

psi_t=(A_inv)*(X')*(X)*(w);
phi_t=X*psi_t;

W=(X_t')*(phi_t);

evaluation_temp = 0;
evaluation = 0;
for i=1:size(files,1)
   evaluation_temp = abs(W(i)-w_t(i));
   evaluation = evaluation+ evaluation_temp;
end
evaluation = evaluation/size(files,1);
disp(evaluation);
figure; plot(w_t,W,'r');




