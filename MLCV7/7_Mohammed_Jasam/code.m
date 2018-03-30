%% Preprocessing

% Clear Everything
clc;
clear;
close all;

% Directories
TrainingPath = 'training\';
TestingPath = 'testing\';

% Extracting Training Data
[XTrain, WTrain] = getData(TrainingPath);

% Extracting the Testing Data
[XTest, GT] = getData(TestingPath);

% Calculating Phi to Infer Rotation of the Image
phi1 = pinv(XTrain') * WTrain;



%% Task 1: Linear Regression

% Inference of Rotation Angle on Testing Data
Inference1 = XTest' * phi1;

% Evaluation of accuracy
LinearRegressionResult = sum(abs(Inference1(:) - GT(:)))/size(GT, 1);

% Displaying Linear Regression Result
fprintf('Linear Regression: %f\n', LinearRegressionResult);

% Graph
draw(GT, Inference1, 'Linear Regression');



%% Task 2: Feature Selection to reduce the Dimensionality for faster computation

% Copying vars
WTrain2 = WTrain;

% Calculating Variance on Training Data
TrainingVariance = var(XTrain, 0, 2);

% Threshold value to reduce the number of values
Threshold = 40;

% Selecting Features in Training Data
XTrain2 = XTrain(TrainingVariance > Threshold, :);

% Selecting Features in Testing Data
XTest2 = XTest(TrainingVariance > Threshold, :);

% Calculating Phi to Infer Rotation of the Image
phi2 = pinv(XTrain2') * WTrain2;

% Inference of Rotation Angle on Testing Data
Inference2 = XTest2' * phi2;

% Evaluation of accuracy
FeatureSelection = sum(abs(Inference2(:) - GT(:)))/size(GT, 1);

% Displaying Linear Regression Result
fprintf('Feature Selection: %f\n', FeatureSelection);

% Graph
draw(GT, Inference2, 'LR with Feature Selection');



%% Task 3: Bayesian Solution based on Linear Regression and Selected Features

% Copying Variables from Earlier Code to Eliminate Confusion
XTrain3 = XTrain2; % Using the Feature Selected Training Data
WTrain3 = WTrain; % Using the original W
XTest3 = XTest2; % Using the Feature Selected Testing Data
phi3 = phi2; % Using the Phi after Feature Selection

% Variance in Phi
VarPrior = var(phi3);

[Inference3, VarianceTest3, VarianceTrain3, AInverse] = fit_blr (XTrain3, WTrain3, VarPrior, XTest3);

% Inference of Rotation Angle on Testing Data
BayesianSolution = sum(abs(Inference3(:) - GT(:)))/size(GT,1);

% Displaying Linear Regression Result
fprintf('Bayesian Solution: %f\n', BayesianSolution);

% Graph
draw(GT, Inference3, 'Bayesian Solution');



%% Task 4: Non-Linear Regression with Regularization

