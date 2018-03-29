%Linear regression

% clc; clear all; %close all;
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
% return %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
X = double(X); %every column in X is one vectorized input image
X = [ones(1,size(X,2)); X];

% Compute phi
%  phi=(X*X')\(X*w);  we can cancel X from numerator and denominator
phi=pinv(X')*(w);  
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

files = dir([dir_testing '*.jpg']);
X_t = []; w_t = [];
for ii = 1:size(files,1)
    filename = files(ii).name;
    w_t = [w; str2double(filename(1:4))];
    im =imread([dir_testing filename]);
    im = im(:,:,1);
    X_t = [X_t im(:)];
end
X_t = double(X_t); %every column in X is one vectorized input image
X_t = [ones(1,size(X_t,2)); X_t];
   
%X_newtest = X_test;
% w_test = phi.'*X_newtest;
   
W=(X_t')*(phi);
evaluation_temp = 0;
evaluation = 0;

for i=1:size(files,1)
   evaluation_temp = abs(W(i)-w_t(i));
   evaluation = evaluation+ evaluation_temp;
end
evaluation = evaluation/size(files,1);
disp(evaluation);
figure; plot(w_t,W,'r');
