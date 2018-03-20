%% Clear Everything
clc; 
clear all;

%% Directories
TrainingDataset = 'DatasetsForFaceRecognition\Training\';
TestingDataset = 'DatasetsForFaceRecognition\Testing\';

%% Parameters 
colorSpace = 'RGB'; %RGB, HSV, YCbCr, HSVYCbCr, Gray, Gradient

% Number of Eigen Vectors
K = 100;

% Number of different faces in the training dataset
NumSubjects = 28;

%% TRAINING

% Step 1: Reading Dataset and creating the matrix
[TrainingMatrix, nrows, ncols, np, TrainingLabels] = getDataMatrix(TrainingDataset, colorSpace, NumSubjects);

% Step 2: Compute the mean
Train_Mean = mean(TrainingMatrix, 2);

% Step 3: Subtract Mean from each pixel of the TrainingMatrix
TrainingMatrixOriginal = TrainingMatrix; % Copy of original training matrix
% Subtracting Mean of each row from each cell of the Training Matrix
for i = 1 : size(TrainingMatrix,1)
    TrainingMatrix(i,:) = TrainingMatrix(i,:) - Train_Mean(i);
end

% Step 4: Defining A and performing Eigen Decomposition A'A i,e A'A.v = Lambda.v
A = TrainingMatrix;
% Computing the Square Matrix
A_A = A' * A;
% Performing the Eigen Decomposition
[V, Lambda] = eig(A_A);

% Step 5: Finding the top K EigenVectors
TopKEigenVectors = V(:,end-K+1:end);

% Step 6: Computing Uk where Uk = A.Vk
Uk = A * TopKEigenVectors;

% Step 7: Computing Alpha = Uk'. ImageMatrix
TrainingAlpha = Uk' * TrainingMatrix;

% Step 8: Generating the FeatureVector for Reference Images
TrainingFeatureVector = TrainingAlpha';

%% TESTING

% Step 1: Reading the Testing dataset
[TestingMatrix, nrows, ncols, np, TestingLabels] = getDataMatrix(TestingDataset, colorSpace, NumSubjects);
% 
% % Step 2: Compute the mean
% Test_Mean = mean(TestingMatrix, 2);

% Step 3: Subtract Mean from each pixel of the TestingMatrix
for i = 1 : size(TestingMatrix,1)
    TestingMatrix(i,:) = TestingMatrix(i,:) - Train_Mean(i);
end

% Step 4: Projecting it to subspace to obtain feature vector
TestingAlpha = Uk' * TestingMatrix;

% Step 5: Generating Testing Feature Vector and comparing it with Training
TestingFeatureVector = TestingAlpha';

% Step 6: Measuring the distance between Training and Testing Features
[MinDistance, Indices] = pdist2(TrainingFeatureVector, TestingFeatureVector, 'euclidean', 'Smallest', 1); % euclidean, minkowski, cityblock, hamming, jaccard, mahalanobis
[AllMinDistances, AllIndices] = pdist2(TrainingFeatureVector, TestingFeatureVector, 'euclidean', 'Smallest', 173); % euclidean, minkowski, cityblock, hamming, jaccard, mahalanobis
ForSubjects = pdist2(TrainingFeatureVector, TestingFeatureVector, 'euclidean'); % euclidean, minkowski, cityblock, hamming, jaccard, mahalanobis

% Step 7: Calculating Accuracy of Faces
[AccuracyFaces, AccuracySubjects] = checkAccuracy(Indices, size(TestingMatrix, 2), TrainingLabels, TestingLabels, NumSubjects);

ArraySub = [];
for i1 = 1 : NumSubjects
    tempTest = [];
    for j1 = 1 : size(TestingFeatureVector, 1)
        if (TestingLabels(j1,1) == i1)
            tempTest = [tempTest; TestingFeatureVector(TestingLabels(j1,3),:)];
        end        
    end
    minD = [];
    for i = 1 : NumSubjects
        tempTrain = [];        
        for j = 1 : size(TrainingFeatureVector, 1)
            if (TrainingLabels(j,1) == i)
                tempTrain = [tempTrain; TrainingFeatureVector(TrainingLabels(j,3),:)];
            end        
        end
        subD = [];
        for i2 = 1 : size(tempTest,1)
            for j2 = 1 : size(tempTrain, 1)
                D = norm(tempTest(i2) - tempTrain(j2));
                subD = [subD; D];
            end
        end
        minD = [minD; min(subD)];
    end
    [minddistance, Indexx] = min(minD,[],1);
    ArraySub = [ArraySub; Indexx];
end

ArraySUBS = 0;

for i = 1 : NumSubjects
    if ismember(i, ArraySub)
        ArraySUBS = ArraySUBS + 1;
    end
end

fprintf('Accuracy of Subjects: %.02f \n', (ArraySUBS/NumSubjects));

