%Bayesian or Regularized Linear regression 

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
I = size(X,2);

X_feature=var(X,0,2);

for i=2:size(X_feature,1) %first value is zero
    if X_feature(i)<20
         X(i,:)=0;
    end        
end

X(all(X==0,2),:)=[];
 
% Compute phi
%  phi=(X*X')\(X*w);  we can cancel X from numerator and denominator
phi=pinv(X*X')*(X*w); 

prior_var=var(phi);
D = size(X,1) - 1;
I = size(X,2);
% I_test = size(X_test,2);
lambda=1000; % lamda range is:10000','1000','100','10','1','.1','.01','.001','.0001','.00001','.000001'

 % Compute A_inv.    
   
    A_inv = eye(D+1) - X*inv(X'*X + lambda*eye(I))*X';
    A_inv = prior_var * A_inv;
    
    
    phi=pinv(X*X'+A_inv)*(X*w);
    
    

% calculating variance of prior phi values



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

X_test_feature=var(X_t,0,2);
for i=2:size(X_test_feature,1)
    if X_test_feature(i)<20
        X_t(i,:)=0;
    end        
end
X_t(all(X_t==0,2),:)=[];

D = size(X,1) - 1;
I = size(X,2);
lambda=1000; % lamda range is:10000','1000','100','10','1','.1','.01','.001','.0001','.00001','.000001'

 % Compute A_inv.    
   
    A_inv = eye(D+1) - X*inv(X'*X+ lambda*eye(I))*X';
    A_inv = prior_var * A_inv;
    
    
    phi_t=pinv(X*X'+A_inv)*(X*w);
    
    


W=(X_t')*(phi_t);

evaluation_temp = 0;
evaluation = 0;
for i=1:size(files,1)
   evaluation_temp = abs(W(i)-w_t(i));
   evaluation = evaluation+ evaluation_temp;
end
evaluation = evaluation/size(files,1);
disp(evaluation);
figure; plot(w_t);
hold on 
plot(W);
