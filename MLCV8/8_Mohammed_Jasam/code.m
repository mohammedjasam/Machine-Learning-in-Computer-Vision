%% Preprocessing
% Clear Everything
clc;
clear;
close all;

% Importing functions from a sub folder
addpath('./models');

colorSpace = 'RGB';
[ZXTrain, ZWTrain, ZXTest, ZWTest] = getData(colorSpace);