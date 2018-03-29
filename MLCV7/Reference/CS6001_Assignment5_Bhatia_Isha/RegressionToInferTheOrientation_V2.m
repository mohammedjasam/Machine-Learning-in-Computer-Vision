%Linear regression with Feature Selection

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

X_feature=var(X,0,2);

for i=2:size(X_feature,1) %first value is zero
    if X_feature(i)<51 % threshold for eliminating values and assigning zero value to the values
        X(i,:)=0;      % below threshold   
    end        
end
X(all(X==0,2),:)=[]; %only selecting non-zero values

% a=find(X_feature~=0);
% feature_new = X_feature(a);
% b= find(X_feature==0);
 
% Compute phi
%  phi=(X*X')\(X*w);  
phi=pinv(X*X')*(X*w);  

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

X_t_feature=var(X_t,0,2);
for i=2:size(X_t_feature,1)
    if X_t_feature(i)<51% threshold for eliminating values and assigning zero value to the values
                         % below threshold   
        X_t(i,:)=0;
    end        
end
X_t(all(X_t==0,2),:)=[];
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






