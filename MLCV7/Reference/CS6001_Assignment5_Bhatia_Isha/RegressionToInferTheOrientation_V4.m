%Non Linear regression

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

phi=pinv(X')*(w);
prior_var = var(phi);

X_var=var(X,0,2);
     
for i=1:size(X_var,1) %first value is zero
    if X_var(i)<100    % threshold for eliminating values and assigning zero value to the values
        X(i,:)=0;     % below threshold   
    end        
end
X(all(X==0,2),:)=[];   %only selecting non-zero values
X = [ones(1,size(X,2)); X];

Z = [];
for ii = 1:2
    Z = [Z; X.^(ii)];
end
% Z = [ones(1,size(Z,2)); Z];

phi_Z=pinv(Z')*w;   
% prior_var_Z = var(phi_Z);
I = size(Z,2);
temp = w - Z'*phi_Z;
prior_var_Z = (temp' * temp) / I;


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

for i=1:size(X_t_var,1)
    if X_t_var(i)<100
        X_t(i,:)=0;
    end        
end
X_t(all(X_t==0,2),:)=[];

Z_t = [];
for ii = 1:2
    Z_t = [Z_t; X_t.^(ii)];
end
%  Z_t = [ones(1,size(Z_t,2)); Z_t];

D=size(Z_t,1);
lamda=1000; % lamda range is:10000','1000','100','10','1','.1','.01','.001','.0001','.00001','.000001'
phi_t=((Z*Z')+ lamda*eye(D))\(Z*w);
W=(Z_t')*(phi_t);

evaluation_temp = 0;
evaluation = 0;
for i=1:size(files,1)
   evaluation_temp = abs(W(i)-w_t(i));
   evaluation = evaluation+ evaluation_temp;
end
evaluation = evaluation/size(files,1);
disp(evaluation);
figure; plot(w_t);
hold on;
plot(W)


