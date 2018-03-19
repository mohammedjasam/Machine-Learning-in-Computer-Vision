clc;
clear;
close all;

%% Image directories
train_images = 'DatasetsForFaceRecognition\Training\';
test_images = 'DatasetsForFaceRecognition\Testing\';

%% Different ColorSpaces
colorSpace = 'RGB';
% colorSpace = 'HSV';
% colorSpace = 'YCbCr';
% colorSpace = 'HSVYCbCr';
% colorSpace = 'Gradient';
% colorSpace = 'Gray';

%% Flags to be set
% Number of different faces in the training dataset
n_diff_faces = 28;

% This helps us select the number of features from the matrix
K = 100;

% Flag to display matched images for all 3 measures
display_images = 1;

% Flag to run the Variable K section
VariableK = 0;

%% Main run

if VariableK == 0
    %% This module is used to run the eigen faces algorithm for single K
    AccMat = run_algo(train_images, test_images, colorSpace, n_diff_faces, K, display_images);
    
    fprintf('Eigen Faces Algorithm\n');
    fprintf('------------------------------------\n');
    fprintf('Value of K: %d\n', K);
    fprintf('------------------------------------\n');
    fprintf('Euclidean Image Accuracy: %.2f \n', AccMat(2));
    fprintf('Euclidean Subject Accuracy: %.2f \n', AccMat(5));
    fprintf('------------------------------------\n');
    fprintf('Manhattan Image Accuracy: %.2f \n', AccMat(3));
    fprintf('Manhattan Subject Accuracy: %.2f \n', AccMat(6));
    fprintf('------------------------------------\n');
    fprintf('Mahalanobis Image Accuracy: %.2f \n', AccMat(4));
    fprintf('Mahalanobis Subject Accuracy: %.2f \n', AccMat(7));
    fprintf('------------------------------------\n');
    
    
else
    %% This module is used to test the accuracy with respect to changes in K
    n_train_images = 173;

    AccData = [];
    for K = 1 : 30
        AccMat = run_algo(train_images, test_images, colorSpace, n_diff_faces, K*5, display_images);
        AccData = [AccData; AccMat.'];
    end

    % Plotting the Accuracies against K values
    figure();
    X = AccData(:,1);
    Euc = AccData(:,2);
    Man = AccData(:,3);
    Mah = AccData(:,4);

    plot(X, Euc);
    plot(X, Man);
    plot(X, Mah);
end



