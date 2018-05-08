clc; clear all;

% Load the data as an |ImageDatastore| object.
trainDatasetPath = 'trainingImages\';
testDatasetPath = 'testingImages\';
trainData = imageDatastore(trainDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
testData = imageDatastore(testDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% Parameters
DataAug = 'randcrop';
dropOut = 0.3;
Batch = 32;
LearnRate = 0.001;
Epochs = 20;

% Define the convolutional neural network architecture. 
layers = [imageInputLayer([40 30 3], 'DataAugmentation', DataAug);
          convolution2dLayer(3,20);
          reluLayer();
          crossChannelNormalizationLayer(3);
          maxPooling2dLayer(2,'Stride',2);
          
          convolution2dLayer(3,40);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          
          convolution2dLayer(3,80);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          
          dropoutLayer(dropOut, 'Name', 'drop1')
          fullyConnectedLayer(2);
          softmaxLayer();
          classificationLayer()];  

% Set the options to default settings for the stochastic gradient descent with momentum. 
options = trainingOptions('sgdm',...
                          'MiniBatchSize', Batch,...
                          'InitialLearnRate', LearnRate,...                          
                          'MaxEpochs', Epochs);  

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
accuracy = sum(YTest == TTest)/numel(TTest);

% Output
% fprintf('\nDataAugmentation = %s\nDropOut = %f\nMiniBatchSize = %d\nInitialLearnRate = %f\nMaxEpochs = %d\n', DataAug, dropOut, Batch, LearnRate, Epochs);
fprintf('\nMiniBatchSize = %d\nInitialLearRate = %f\nMaxEpochs = %d\nDropOut = %f\nDataAugmentation = %s\n', Batch, LearnRate, Epochs, dropOut, DataAug);
fprintf('\nAccuracy = %f\n', accuracy);

