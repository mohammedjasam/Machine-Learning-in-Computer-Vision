%% Clear
clc; 
clear all; 
close all;

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
M = X_train * X_train';
[r,~] = size(M);
M(1:r+1:end) = M(1:r+1:end) + 0.001; % Adding a small value to the diagonal to avoid singularity

X_train_new = M;

z = inv(X_train_new);
y = X_train * w_train;
phi = z * y;


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

%% Inferring the rotation on test files
w_inferred = phi' * X_test;

%% Calculating the accuracy
diff_sum = 0;
for i = 1 : size(testing_files, 1) - 2
   diff_sum = diff_sum + abs(w_inferred(i) - ground_truth(i));
end
diff_sum = diff_sum / (size(testing_files, 1) - 2);
disp(diff_sum);

%% Visualization
plot(w_inferred);
hold on
plot(ground_truth);
legend('Inference','Ground Truth');
title('Linear Regression');