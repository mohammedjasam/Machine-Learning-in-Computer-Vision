function [XTrain, WTrain, XTest, WTest] = getData(colorSpace)

colorSpace = 'RGB';

% Directories
TrainingFacePath = 'trainingImages\face_resized\';
TrainingBackgroundPath = 'trainingImages\background_resized\';
TestingFacePath = 'testingImages\face_resized\';
TestingBackgroundPath = 'testingImages\background_resized\';

% Read Data
[XFaceTrain, RowsTrainFace, ColsTrainFace, ChannelsTrainFace] = getAllIms(TrainingFacePath, colorSpace);
[XBackgroundTrain, RowsTrainBackground, ColsTrainBackground, ChannelsTrainBackground] = getAllIms(TrainingBackgroundPath, colorSpace);
[XFaceTest, RowsTestFace, ColsTestFace, ChannelsTestFace] = getAllIms(TestingFacePath, colorSpace);
[XBackgroundTest, RowsTestBackground, ColsTestBackground, ChannelsTestBackground] = getAllIms(TestingBackgroundPath, colorSpace);

XFaceTrain = XFaceTrain';
XBackgroundTrain = XBackgroundTrain';
XFaceTest = XFaceTest';
XBackgroundTest = XBackgroundTest';

NumTrainFace = size(XFaceTrain, 2);
NumTrainBackground = size(XBackgroundTrain, 2);
NumTestFace = size(XFaceTest, 2);
NumTestBackground = size(XBackgroundTest, 2);

XTrain = [(ones(1, (NumTrainFace + NumTrainBackground))); [XFaceTrain, XBackgroundTrain]];
WTrain = [ones(NumTrainFace, 1); zeros(NumTrainBackground, 1)];
XTest = [(ones(1, (NumTestFace + NumTestBackground))); [XFaceTest, XBackgroundTest]];
WTest = [ones(NumTestFace, 1); zeros(NumTestBackground, 1)];


end
