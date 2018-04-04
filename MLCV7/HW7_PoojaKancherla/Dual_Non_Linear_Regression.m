function [print, result, ground_truth] = Dual_Non_Linear_Regression()
    %% Clear
    %clc; 
    %clear all; 
%     close all;

    %% Data
    training_directory = 'training\';
    testing_directory = 'testing\';

    %% Training
    training_files = dir(training_directory);

    w_train = []; % Given w
    X_train = [];


    % Creating the image column matrix
    for i = 3 : size(training_files)
        w_train = [w_train; str2double(training_files(i).name(1:4))]; % Extracting the rotation angle from filename
        image = imread([training_directory training_files(i).name]);
        image = image(:,:,1);
        X_train = [X_train image(:)];
    end

    X_train = double(X_train);
    ones_row = ones(1, size(X_train, 2));
    X_train = [ones_row; X_train]; % Adding the 1s row at the beginning

    %% Calculating phi to learn parameters
    % M = X_train * X_train';
    % [r,~] = size(M);
    % M(1:r+1:end) = M(1:r+1:end) + 0.001; % Adding a small value to the diagonal to avoid singularity
    % 
    % X_train_new = M;
    % 
    % z = inv(X_train_new);
    % y = X_train * w_train;
    % phi = z * y;

    % Since DxD is huge we do the opposite to get IxI
    M = (X_train' * X_train);

    % Calculating Psi
    psi =  M \ w_train;

    % Calculating Phi
    phi = X_train * psi;

    %% Testing
    testing_files = dir(testing_directory);

    ground_truth = []; % Given w
    X_test = [];

    % Creating the image column matrix
    for i = 3 : size(testing_files)
        ground_truth = [ground_truth; str2double(testing_files(i).name(1:4))]; % Extracting the rotation angle from filename
        image = imread([testing_directory testing_files(i).name]);
        image = image(:,:,1);
        X_test = [X_test image(:)];
    end

    X_test = double(X_test);
    ones_row = ones(1, size(X_test, 2));
    X_test = [ones_row; X_test]; % Adding the 1s row at the beginning

    %% Dual Non-Linear Regression
    num_train = size(X_train, 2); % Number of training images
    num_test = size(X_test,2); % Number of training images

%     t = w_train - ((X_train' * X_train) * psi);
%     variance_temp = ((t)' * t) \ num_train;

    % Calculating mean for train data
    temp = 0;
    for i = 1 : size(w_train, 1)
        temp = temp + w_train(i);
    end
    mean_train = temp / num_train;

    % Calculating variance for train data
    temp = 0;
    variance_train = 0;
    for i = 1 : size(w_train, 1)
        temp = w_train(i) - mean_train;
        temp = temp ^ 2;
        variance_train = variance_train + temp;
    end
    variance_train = variance_train / num_train;

    % Calculating the min value for variance
    variance = fminbnd (@(variance) calc_cost (variance, X_train, num_train, w_train, var(phi)), 0, variance_train);

    lambda = variance / var(phi);
    X_test_var = var(X_test, 0, 2);

    % Compute A_inv.   
    A = ((X_train' * X_train) * (X_train' * X_train)) + (lambda * eye(num_train));
    A_inv = inv(A);

    psi_test = A_inv * (X_train' * X_train) * w_train;
    phi_test = X_train * psi_test; 
    
    %% Inferring the rotation on test files
    w_inferred = phi_test' * X_test;

    %% Calculating the diff
    diff_sum = 0;
    for i = 1 : size(testing_files, 1) - 2
       diff_sum = diff_sum + abs(w_inferred(i) - ground_truth(i));
    end
    diff_sum = diff_sum / (size(testing_files, 1) - 2);
%     disp(sprintf('Dual Non Linear Regression = %f', diff_sum));
    print = sprintf('Dual Non Linear Regression = %f', diff_sum);
    
    %% Visualization    
    bar(w_inferred - ground_truth');
    hold on
    bar(ground_truth' - ground_truth');    
    legend('Inference','Ground Truth');
    title(sprintf('Dual Non Linear Regression: %f', diff_sum));
    
    figure();
    plot(w_inferred);
    hold on;
    plot(ground_truth); 
    legend('Inference','Ground Truth');
    title(sprintf('Dual Non Linear Regression: %f', diff_sum));
    
result = w_inferred;