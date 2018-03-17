%% Clear Everything
clc; 
clear all;

%% Directories
TrainingDataset = 'DatasetsForFaceRecognition\Training\';

%     for iSubject = 1:28
%     %     [FaceIms,nrows,ncols,np] = getAllIms(sprintf('%s%02d\\',facedir_training,iSubject),colorSpace);
%         disp(sprintf('%s%02d\\',TrainingDataset,iSubject));
%     %     return
% 
%     %     if isempty(FaceIms), continue; end
%     %     allFaceIms_training = [allFaceIms_training; FaceIms];    
%     %     labels = [labels; [iSubject*ones(size(FaceIms,1),1) (1:size(FaceIms,1))']]; 
%     end

%% Parameters 
colorSpace = 'RGB'; %RGB, Gray, HSV, YCbCr, HSVYCbCr, Gradient

% Number of Eigen Vectors
K = 100;

% Number of different faces in the training dataset
NumTrainingClasses = 28;

%% Reading Dataset and creating the matrix
[TrainingMatrix, nrows, ncols, np, TrainingContents] = getDataMatrix(TrainingDataset, colorSpace, NumTrainingClasses);






