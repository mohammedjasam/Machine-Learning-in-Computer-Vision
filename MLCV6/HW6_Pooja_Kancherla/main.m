clc;
clear;
close all;

%% Image directories
train_images = 'DatasetsForFaceRecognition\Training\';
test_images = 'DatasetsForFaceRecognition\Testing\';

%% Different ColorSpaces
% colorSpace = 'RGB';
% colorSpace = 'HSV';
colorSpace = 'YCbCr';
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

% Flag to run the variable K section
var_k = 0;

%% Main run

if var_k == 0
    %% This module is used to run the eigen faces algorithm for single K
    [acc_matrix, sub_mat] = run_algo(train_images, test_images, colorSpace, n_diff_faces, K, display_images);
    
    fprintf('Eigen Faces Algorithm\n');
    fprintf('------------------------------------\n');
    fprintf('Value of K: %d\n', K);
    fprintf('------------------------------------\n');
    fprintf('Euclidean Image Accuracy: %.2f \n', acc_matrix(2));
    fprintf('Euclidean Subject Accuracy: %.2f \n', sub_mat(1));
    fprintf('------------------------------------\n');
    fprintf('Manhattan Image Accuracy: %.2f \n', acc_matrix(3));
    fprintf('Manhattan Subject Accuracy: %.2f \n', sub_mat(2));
    fprintf('------------------------------------\n');
    fprintf('Mahalanobis Image Accuracy: %.2f \n', acc_matrix(4));
    fprintf('Mahalanobis Subject Accuracy: %.2f \n', sub_mat(3));
    fprintf('------------------------------------\n');
    
    
else
    %% This module is used to test the accuracy with respect to changes in K
    n_train_images = 173;
    
    % Displaying images is set to 0 by default to reduce cpu usage
    display_images = 0;
    
    acc_data = [];
    for K = 1 : 30
        [acc_matrix, sub_mat] = run_algo(train_images, test_images, colorSpace, n_diff_faces, K*5, display_images);
        acc_data = [acc_data; acc_matrix.'];
    end

    % Plotting the Accuracies against K values
    figure();
    X = acc_data(:,1);
    euc = acc_data(:,2);
    man = acc_data(:,3);
    mah = acc_data(:,4);

    plot(X, euc);
    plot(X, man);
    plot(X, mah);
end



