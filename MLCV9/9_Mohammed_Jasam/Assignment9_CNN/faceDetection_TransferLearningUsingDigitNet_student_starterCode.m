% Transfer learning using a pre-trained network

clc; clear all;

% Load the data as an |ImageDatastore| object.
trainDatasetPath = 'trainingImages\';
testDatasetPath = 'testingImages\';
trainData = imageDatastore(trainDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames','ReadFcn',@myreader);
testData = imageDatastore(testDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames','ReadFcn',@myreader);

% Load a pretrained network.
load('convnet_from_digitData.mat','convnet');

% Parameters
dropOut = 0.3;
WeightLearnRate = 3;
BiasLearnRate = 3;
Batch = 32;
LearnRate = 0.0001;
Epochs = 40;

% Create the layer array by combining the transferred layers with the new layers.
layers = [convnet.Layers(1:end-3)
    %change the last three layers of the pre-trained network below
    fullyConnectedLayer(2, 'WeightLearnRateFactor', WeightLearnRate, 'BiasLearnRateFactor', BiasLearnRate);
    softmaxLayer();
    classificationLayer();
    ];

% Create the training options. 
options = trainingOptions('sgdm',...
                          'MiniBatchSize', Batch,...
                          'InitialLearnRate', LearnRate,...
                          'MaxEpochs', Epochs); 

% Fine-tune the network using thes new layer array.
netTransfer = trainNetwork(trainData,layers,options);

% Classify the test images using |classify|.
predictedLabels = classify(netTransfer,testData);

% Calculate the classification accuracy.
testLabels = testData.Labels;

accuracy = sum(predictedLabels==testLabels)/numel(predictedLabels);

fprintf('\nMiniBatchSize = %d\nInitialLearRate = %f\nMaxEpochs = %d\nWeightLearnRateFactor = %d\nBiasLearnRateFactor = %d\n', Batch, LearnRate, Epochs, WeightLearnRate, BiasLearnRate);
fprintf('\nAccuracy = %f\n', accuracy);
%since the image sizes do not match, we write a new image reader here 
function data = myreader(filename)
    data = rgb2gray(imresize(imread(filename),[28 28]));
end