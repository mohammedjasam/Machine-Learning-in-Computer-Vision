%% Clear
clc; 
clear all; 
close all;

%% Data
training_directory = 'training\';
testing_directory = 'testing\';

%% Training
training_files = dir(training_directory);

w = []; % Given w
X = [];

% Creating the image column matrix
for i = 3 : size(training_files)
    a = training_files(i).name;
    w = [w; str2double(training_files(i).name(1:4))]; % Extracting the rotation angle from filename
    image = imread([training_directory training_files(i).name]);
    image = image(:,:,1);
    X = [X image(:)];
end

X = double(X);
ones_row = ones(1, size(X, 2));
X = [ones_row; X]; % Adding the 1s row at the beginning

% Calculate phi
phi = pinv(X)' * w;

phi = inv(X*X');


