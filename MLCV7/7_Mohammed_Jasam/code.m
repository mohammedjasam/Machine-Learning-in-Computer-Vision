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
TaskName1 = 'Linear Regression';

% Inference of Rotation Angle on Testing Data
Inference1 = XTest' * phi1;

% Evaluation of accuracy
LinearRegressionResult = sum(abs(Inference1(:) - GT(:)))/size(GT, 1);

% Displaying Linear Regression Result
fprintf('%s: %f\n', TaskName1, LinearRegressionResult);

% Graph
draw(GT, Inference1, TaskName1);

%% Task 2: Feature Selection to reduce the Dimensionality for faster computation
TaskName2 = 'Feature Selection';

% Copying Variables from Earlier Code
WTrain2 = WTrain;

% Calculating Variance on Training Data
TrainingVariance = var(XTrain, 0, 2);

% Threshold value to reduce the number of values
Threshold = 100;

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
fprintf('%s: %f\n', TaskName2, FeatureSelection);

% Graph
draw(GT, Inference2, TaskName2);

%% Task 3: Bayesian Solution based on Linear Regression and Feature Selection
TaskName3 = 'Bayesian Solution';

% Copying Variables from Earlier Code
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
fprintf('%s: %f\n', TaskName3, BayesianSolution);

% Graph
draw(GT, Inference3, 'Bayesian Solution');

%% Task 4: Non-Linear Regression with Regularization and Feature Selection
TaskName4 = 'Non Linear Solution';

% Copying Variable from Earlier Code
XTrain4 = XTrain2;
XTest4 = XTest2;
WTrain4 = WTrain;
VarianceTrain4 = VarianceTrain3;

Lambda = VarianceTrain4 / var(phi3);

% Select the Number of Polynomials
n = 2;

% Creating the Z Train matrix
ZTrain = [];
ZTest = [];
for i = 1 : n
    NewXTrain = XTrain4 .^ i;
    ZTrain = [ZTrain; NewXTrain];
    NewXTest = XTest4 .^ i;
    ZTest = [ZTest; NewXTest];
end

phi4 = (ZTrain * ZTrain' + Lambda * eye(size(ZTrain, 1))) \ ZTrain * WTrain2;
Inference4 = phi4' * ZTest;

% Inference of Rotation Angle on Testing Data
NonLinearSolution = sum(abs(Inference4(:) - GT(:)))/size(GT,1);

fprintf('%s: %f\n', TaskName4, NonLinearSolution);

% Graph
draw(GT, Inference4, TaskName4);

%% Task 5: Dual Non Linear Regression with Regularization and without Feature Selection
TaskName5 = 'Dual Non-L.R.';

