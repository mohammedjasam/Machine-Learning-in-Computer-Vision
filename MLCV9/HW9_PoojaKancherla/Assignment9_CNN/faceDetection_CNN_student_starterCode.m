clc; clear all;

% Load the data as an |ImageDatastore| object.
trainDatasetPath = 'trainingImages\';
testDatasetPath = 'testingImages\';
trainData = imageDatastore(trainDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
testData = imageDatastore(testDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% Define the convolutional neural network architecture. 
layers = [imageInputLayer([40 30 3]);
          convolution2dLayer(3,20);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(2);
          softmaxLayer();
          classificationLayer()];  

% Set the options to default settings for the stochastic gradient descent with momentum. 
options = trainingOptions('sgdm');  

% Train the network. 
tic
convnet = trainNetwork(trainData,layers,options);
toc

%% 
% Run the trained network on the test set and predict the image labels.
tic
YTest = classify(convnet,testData);
TTest = testData.Labels;
toc

% Calculate the accuracy. 
accuracy = sum(YTest == TTest)/numel(TTest)   