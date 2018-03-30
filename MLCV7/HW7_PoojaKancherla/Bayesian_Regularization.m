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
var_mat = var(X_train, 0, 2);
temp = var_mat > 100;
X_train_temp = X_train .* temp;
X_train_temp(all(X_train_temp == 0, 2), :) = [];
X_train_temp = [ones_row; X_train_temp]; % Adding the 1s row at the beginning

%% Calculating phi to learn parameters
M = X_train_temp * X_train_temp';
[r,~] = size(M);
M(1:r+1:end) = M(1:r+1:end) + 0.001; % Adding a small value to the diagonal to avoid singularity

X_train_new = M;

z = inv(X_train_new);
y = X_train_temp * w_train;
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
var_mat = var(X_train, 0, 2);
temp = var_mat > 100;
X_test_temp = X_test .* temp;
X_test_temp(all(X_test_temp == 0, 2), :) = [];
X_test_temp = [ones_row; X_test_temp]; % Adding the 1s row at the beginning


%% Bayesian Linear Regression
num_train = size(X_train_temp, 2); % Number of training images
dim_train = size(X_train_temp, 1) - 1; % Dimensionality of the training images
num_test = size(X_test_temp,2); % Number of training images

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
variance = fminbnd (@(variance) calc_cost (variance, X_train_temp, num_train, w_train, var(phi)), 0, variance_train);

% Calculating A inverse
A_inverse = 0;    
if dim_train > num_train
    M = X_train_temp' * X_train_temp + (variance/var(phi))*eye(num_train);
    inv_train_temp = inv(M) * X_train_temp';
    A_inverse = eye(dim_train+1) - X_train_temp * inv_train_temp;
    A_inverse = var(phi) * A_inverse;    
else    
    A_inverse = inv ((X_train_temp*X_train_temp') ./ variance + eye(dim_train+1) ./ var(phi));
end

% Calculating the mean for test data
temp = X_test_temp' * A_inverse;
mean_test = (temp * X_train_temp * w_train) ./ variance;

% Calculating the variance for test data
variance_test = repmat(variance, num_train, 1);
for i = 1 : num_train
    variance_test(i) = variance_test(i) + temp(i,:) * X_test_temp(:, i);
end

%% Inferring the rotation on test files and calculating the diff
w_inferred = mean_test;

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
title('Bayesian using Regularization');