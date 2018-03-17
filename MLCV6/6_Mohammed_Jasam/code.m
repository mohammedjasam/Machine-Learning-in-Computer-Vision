%% Clear Everything
clc; 
clear all;

%% Directories
TrainingDataset = 'DatasetsForFaceRecognition\Training\';

%% Parameters 
colorSpace = 'RGB'; %RGB, Gray, HSV, YCbCr, HSVYCbCr, Gradient

% Number of Eigen Vectors
K = 100;

% Number of different faces in the training dataset
NumTrainingClasses = 28;

%% Step 1: Reading Dataset and creating the matrix
[TrainingMatrix, nrows, ncols, np, TrainingContents] = getDataMatrix(TrainingDataset, colorSpace, NumTrainingClasses);

%% Step 2: Compute the mean
Train_Mean = mean(TrainingMatrix, 2);

%% Step 3: Subtract Mean from each pixel of the TrainingMatrix
TrainingMatrixOriginal = TrainingMatrix; % Copy of original training matrix

% Subtracting Mean of each row from each row of the Training Matrix
for i = 1 : size(TrainingMatrix,1)
    TrainingMatrix(i,:) = TrainingMatrix(i,:) - Train_Mean(i);
end

%% 


