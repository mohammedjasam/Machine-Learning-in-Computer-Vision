function [XTrain, WTrain, XTest, WTest, NumTrainFace, NumTrainBackground, NumTestFace, NumTestBackground] = getData(colorSpace)
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
    
    % Transforming Data
    XFaceTrain = XFaceTrain';
    XBackgroundTrain = XBackgroundTrain';
    XFaceTest = XFaceTest';
    XBackgroundTest = XBackgroundTest';
    
    % Lengths of data
    NumTrainFace = size(XFaceTrain, 2);
    NumTrainBackground = size(XBackgroundTrain, 2);
    NumTestFace = size(XFaceTest, 2);
    NumTestBackground = size(XBackgroundTest, 2);

    % Face first then Background
    XTrain = [(ones(1, (NumTrainFace + NumTrainBackground))); [XFaceTrain, XBackgroundTrain]];
    WTrain = [ones(NumTrainFace, 1); zeros(NumTrainBackground, 1)];
    XTest = [(ones(1, (NumTestFace + NumTestBackground))); [XFaceTest, XBackgroundTest]];
    WTest = [ones(NumTestFace, 1); zeros(NumTestBackground, 1)];

%     % Background first then Face
%     XTrain = [(ones(1, (NumTrainFace + NumTrainBackground))); [XBackgroundTrain, XFaceTrain]];
%     WTrain = [zeros(NumTrainBackground, 1); ones(NumTrainFace, 1)];
%     XTest = [(ones(1, (NumTestFace + NumTestBackground))); [XBackgroundTest, XFaceTest]];
%     WTest = [zeros(NumTestBackground, 1); ones(NumTestFace, 1)];
end
