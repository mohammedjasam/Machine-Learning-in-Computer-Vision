clc;
clear;
clear all;

% Image directories
train_images = 'DatasetsForFaceRecognition\Training\';
test_images = 'DatasetsForFaceRecognition\Testing\';

% Different ColorSpaces
colorSpace = 'RGB';
% colorSpace = 'HSV';
% colorSpace = 'YCbCr';
% colorSpace = 'HSVYCbCr';
% colorSpace = 'Gradient';
% colorSpace = 'Gray';

% Number of different faces in the training dataset
n_diff_faces = 28;


%% This module is used to test the accuracy with respect to changes in K
% n_train_images = 173;
% 
% for K = 1 : n_train_images
%     what_goes_here = run_algo(train_images, test_images, colorSpace, n_diff_faces, K);
% end


%% This module is used to run the eigen faces algorithm for single K

% This helps us select the number of features from the matrix
K = 75;

what_goes_here = run_algo(train_images, test_images, colorSpace, n_diff_faces, K);