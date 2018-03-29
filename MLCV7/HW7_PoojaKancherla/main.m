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
    a = training_files(i).name;
    w_train = [w_train; str2double(training_files(i).name(1:4))]; % Extracting the rotation angle from filename
    image = imread([training_directory training_files(i).name]);
    image = image(:,:,1);
    X_train = [X_train image(:)];
end

X_train = double(X_train);
ones_row = ones(1, size(X_train, 2));
X_train = [ones_row; X_train]; % Adding the 1s row at the beginning

% % Calculate phi
% phi_old = pinv(X)' * w_train;

M = X_train * X_train';
[r,~] = size(M);
M(1:r+1:end) = M(1:r+1:end) + 0.001; % Adding a small value to the diagonal to avoid singularity

X_train_new = M;
z = inv(X_train_new);
y = X_train * w_train;
phi_new = z * y;
phi = phi_new;
%% Testing
testing_files = dir(testing_directory);

Gr_truth = []; % Given w
X_test = [];

% Creating the image column matrix
for i = 3 : size(testing_files)
    a = testing_files(i).name;
    Gr_truth = [Gr_truth; str2double(testing_files(i).name(1:4))]; % Extracting the rotation angle from filename
    image = imread([testing_directory testing_files(i).name]);
    image = image(:,:,1);
    X_test = [X_test image(:)];
end

X_test = double(X_test);
ones_row = ones(1, size(X_test
2));
X_test = [ones_row; X_test]; % Adding the 1s row at the beginning

w_inferred = phi' * X_test;

Eval_linReg = sum(abs(w_inferred(:) - Gr_truth(:)))/size(Gr_truth,1);
