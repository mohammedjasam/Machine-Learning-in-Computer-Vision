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

%% Task 2: Feature Selection

% Calculating Variance on Training Data
TrainingVariance = var(XTrain, 0, 2);

% Selecting Features in Training Data
XTrain = XTrain(TrainingVariance > 100, :);

% Selecting Features in Testing Data
XTest = XTest(TrainingVariance > 100, :);

% Calculating Phi to Infer Rotation of the Image
phi2 = pinv(XTrain') * WTrain;

% Inference of Rotation Angle on Testing Data
Inference2 = XTest' * phi2;

% Evaluation of accuracy
FeatureSelection = sum(abs(Inference2(:) - GT(:)))/size(GT, 1);

% Displaying Linear Regression Result
fprintf('Feature Selection: %f\n', FeatureSelection);

% Graph
draw(GT, Inference2, 'Feature Selection');

%% Task 3: Bayesian Solution










