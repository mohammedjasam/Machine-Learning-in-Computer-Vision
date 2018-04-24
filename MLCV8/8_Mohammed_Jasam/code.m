%% Preprocessing
% Clear Everything
clc;
clear;
close all;

% Importing functions from a sub folder
addpath('./models');

% Colorspace
colorSpace = 'RGB'; % RGB, HSV, YCbCr, Gray, Gradient

% Extracting Data
[XTrain, WTrain, XTest, WTest, ... 
    NumTrainFace, NumTrainBackground, ...
    NumTestFace, NumTestBackground] = getData(colorSpace);

%% Task 1: Logistic Regression
InitialPhi = pinv(XTrain') * WTrain; % Phi
% InitialPhi = pinv(XTrain(2 : end, :)') * WTrain; % Phi
% InitialPhi = [1; InitialPhi];
VarPrior = var(InitialPhi); % Variance

% Inference
Inference1 = fit_logr (XTrain, WTrain, VarPrior, XTest, InitialPhi);

% Evaluation
[MissDetection1, FalseAlarm1] = evaluate(WTest, Inference1, NumTestFace, NumTestBackground);

%% Task 2: Bayesian Logistic Regression
% Inference
Inference2 = fit_blogr (XTrain, WTrain, VarPrior, XTest, InitialPhi);

% Evaluation
[MissDetection2, FalseAlarm2] = evaluate(WTest, Inference2, NumTestFace, NumTestBackground);

%% Task 3: Dual Logistic Regression
InitialPsi = zeros((NumTrainFace + NumTrainBackground), 1); % Psi
VarPrior = []; % Variance

% Inference
Inference3 = fit_dlogr (XTrain, WTrain, VarPrior, XTest, InitialPsi);

% Evaluation
[MissDetection3, FalseAlarm3] = evaluate(WTest, Inference3, NumTestFace, NumTestBackground);

%% Task 4: Dual Bayesian Logistic Regression
VarPrior = var(InitialPhi); % Variance
InitialPsi = zeros((NumTrainFace + NumTrainBackground), 1); % Psi

% Inference
Inference4 = fit_dblogr (XTrain, WTrain, VarPrior, XTest, InitialPsi);

% Evaluation
[MissDetection4, FalseAlarm4] = evaluate(WTest, Inference4, NumTestFace, NumTestBackground);

%% Task 5: Kernel Logistic Regression
Lambda = 1; % Lambda 

% Inference
Inference5 = fit_klogr (XTrain, WTrain, VarPrior, XTest, InitialPsi, @kernel_gauss, Lambda);

% Evaluation
[MissDetection5, FalseAlarm5] = evaluate(WTest, Inference5, NumTestFace, NumTestBackground);

%% Task 6: Relevance Vector Logistic Regression
Nu = 1; % Nu

% Inference
Inference6 = fit_rvc (XTrain, WTrain, Nu, XTest, InitialPsi, @kernel_gauss, Lambda);

% Evaluation
[MissDetection6, FalseAlarm6] = evaluate(WTest, Inference6, NumTestFace, NumTestBackground);

%% Results
fprintf('\n\n\n\nResults:\n');
fprintf('_____________________________________________________\n');
fprintf('1. Logistic Regression:\n')
fprintf('   --------------------\n')
fprintf('   Misdetection: %f\n   False  Alarm: %f\n\n', MissDetection1, FalseAlarm1)
fprintf('_____________________________________________________\n');
fprintf('2. Bayesian Logistic Regression:\n')
fprintf('   -----------------------------\n')
fprintf('   Misdetection: %f\n   False  Alarm: %f\n\n', MissDetection2, FalseAlarm2)
fprintf('_____________________________________________________\n');
fprintf('3. Dual Logistic Regression:\n')
fprintf('   -------------------------\n')
fprintf('   Misdetection: %f\n   False  Alarm: %f\n\n', MissDetection3, FalseAlarm3)
fprintf('___________________________OKM,N LUO/980P__________________________\n');
fprintf('4. Dual Bayesian Logistic Regression:\n')
fprintf('   ----------------------------------\n')
fprintf('   Misdetection: %f\n   False  Alarm: %f\n\n', MissDetection4, FalseAlarm4)
fprintf('_____________________________________________________\n');
fprintf('5. Kernel Logistic Regression:\n')
fprintf('   ---------------------------\n')
fprintf('   Misdetection: %f\n   False  Alarm: %f\n\n', MissDetection5, FalseAlarm5)
fprintf('_____________________________________________________\n');
fprintf('6. Relevance Vector Logistic Regression:\n')
fprintf('   -------------------------------------\n')
fprintf('   Misdetection: %f\n   False  Alarm: %f\n\n', MissDetection6, FalseAlarm6)
fprintf('_____________________________________________________\n\n\n');