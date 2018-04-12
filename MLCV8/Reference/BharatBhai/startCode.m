clc; clear all; close all;
dir_training_w0 = 'trainingImages\background_resized\';
dir_training_w1 = 'trainingImages\face_resized\';
dir_testing_w0 = 'testingImages\background_resized\';
dir_testing_w1 = 'testingImages\face_resized\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

colorSpace = 'RGB';
X_w0 = getAllIms(dir_training_w0,colorSpace)';
X_w1 = getAllIms(dir_training_w1,colorSpace)';

X_test_w0 = getAllIms(dir_testing_w0,colorSpace)';
X_test_w1 = getAllIms(dir_testing_w1,colorSpace)';

X = [[ones(1,size(X_w0,2)); X_w0], [ones(1,size(X_w1,2)); X_w1]];
w = [zeros(size(X_w0,2),1); ones(size(X_w1,2),1)];

X_test = [[ones(1,size(X_test_w0,2)); X_test_w0], [ones(1,size(X_test_w1,2)); X_test_w1]];
w_test = [zeros(size(X_test_w0,2),1); ones(size(X_test_w1,2),1)];

miss_detection = zeros(6,1);
false_alarm = zeros(6,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

initial_phi = pinv(X(2:end,:)')*w;
initial_phi = [1;initial_phi];

% initial_phi = rand(241,1);
% initial_psi = rand(433,1);
var_prior = var(initial_phi(:));
%var_prior = [];

[predictions, phi] = fit_logr (X, w, var_prior, X_test, initial_phi);

infer = predictions;

infer(predictions>0.5) = 1;
infer(predictions<0.5) = 0;

%Performance evaluation by mean absolute error method
%Miss detection
face_index_start = size(X_test_w0,2)+1;
face_index_end = size(X_test,2);
sum=0;
for i = face_index_start:face_index_end
    sum = sum + abs(w_test(i) - infer(i));
end
miss_detection(1) = sum/i;

%False Alarm
bg_index_start = 1;
bg_index_end = size(X_test_w0,2);
sum=0;
for j = bg_index_start:bg_index_end
    sum = sum + abs(w_test(j) - infer(j));
end
false_alarm(1) = sum/j;
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Bayesian Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

[predictions, phi] = fit_blogr (X, w, var_prior, X_test, initial_phi);

infer = predictions;

infer(predictions>0.5) = 1;
infer(predictions<0.5) = 0;

%Performance evaluation by mean absolute error method
%Miss detection
face_index_start = size(X_test_w0,2)+1;
face_index_end = size(X_test,2);
sum=0;
for i = face_index_start:face_index_end
    sum = sum + abs(w_test(i) - infer(i));
end
miss_detection(2) = sum/i;

%False Alarm
bg_index_start = 1;
bg_index_end = size(X_test_w0,2);
sum=0;
for j = bg_index_start:bg_index_end
    sum = sum + abs(w_test(j) - infer(j));
end
false_alarm(2) = sum/j;
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dual Logistic Regression method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
lambda = 1;
initial_psi = zeros(433,1);
var_prior = [];
[predictions, psi] = fit_dlogr (X, w, var_prior, X_test, initial_psi);

infer = predictions;

infer(predictions>0.5) = 1;
infer(predictions<0.5) = 0;
%Performance evaluation by mean absolute error method
%Miss detection
face_index_start = size(X_test_w0,2)+1;
face_index_end = size(X_test,2);
sum=0;
for i = face_index_start:face_index_end
    sum = sum + abs(w_test(i) - infer(i));
end
miss_detection(3) = sum/i;

%False Alarm
bg_index_start = 1;
bg_index_end = size(X_test_w0,2);
sum=0;
for j = bg_index_start:bg_index_end
    sum = sum + abs(w_test(j) - infer(j));
end
false_alarm(3) = sum/j;
return
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dual Bayesian Logistic Regression method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
var_prior=6;
[predictions, psi] = fit_dblogr (X, w, var_prior, X_test, initial_psi);

infer = predictions;

infer(predictions>0.5) = 1;
infer(predictions<0.5) = 0;
%Performance evaluation by mean absolute error method
%Miss detection
face_index_start = size(X_test_w0,2)+1;
face_index_end = size(X_test,2);
sum=0;
for i = face_index_start:face_index_end
    sum = sum + abs(w_test(i) - infer(i));
end
miss_detection(4) = sum/i;

%False Alarm
bg_index_start = 1;
bg_index_end = size(X_test_w0,2);
sum=0;
for j = bg_index_start:bg_index_end
    sum = sum + abs(w_test(j) - infer(j));
end
false_alarm(4) = sum/j;
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kernel Logistic Regression method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
var_prior = 6;
lambda = 1;
[predictions, psi] = fit_klogr (X, w, var_prior, X_test, ...
    initial_psi, @kernel_gauss, lambda);

infer = predictions;

infer(predictions>0.5) = 1;
infer(predictions<0.5) = 0;
%Performance evaluation by mean absolute error method
%Miss detection
face_index_start = size(X_test_w0,2)+1;
face_index_end = size(X_test,2);
sum=0;
for i = face_index_start:face_index_end
    sum = sum + abs(w_test(i) - infer(i));
end
miss_detection(5) = sum/i;

%False Alarm
bg_index_start = 1;
bg_index_end = size(X_test_w0,2);
sum=0;
for j = bg_index_start:bg_index_end
    sum = sum + abs(w_test(j) - infer(j));
end
false_alarm(5) = sum/j;
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Relevance Vector Logistic Regression method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
nu = 1;
[predictions, relevant_points] = fit_rvc (X, w, nu, X_test,...
    initial_psi, @kernel_gauss, lambda);

infer = predictions;

infer(predictions>0.5) = 1;
infer(predictions<0.5) = 0;
%Performance evaluation by mean absolute error method
%Miss detection
face_index_start = size(X_test_w0,2)+1;
face_index_end = size(X_test,2);
sum=0;
for i = face_index_start:face_index_end
    sum = sum + abs(w_test(i) - infer(i));
end
miss_detection(6) = sum/i;

%False Alarm
bg_index_start = 1;
bg_index_end = size(X_test_w0,2);
sum=0;
for j = bg_index_start:bg_index_end
    sum = sum + abs(w_test(j) - infer(j));
end
false_alarm(6) = sum/j;
%%