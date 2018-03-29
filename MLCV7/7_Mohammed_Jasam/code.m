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
phi = pinv(XTrain') * WTrain;
%% Task 1: Linear Regression

% Inference of Rotation Angle on Testing Data
Inference1 = XTest' * phi;

% Evaluation of accuracy
LinearRegressionResult = sum(abs(Inference1(:) - GT(:)))/size(GT, 1);

% Displaying Linear Regression Result
fprintf('Linear Regression: %f\n', LinearRegressionResult);

%% Task 2: Feature Selection


