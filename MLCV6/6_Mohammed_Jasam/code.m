%% Clear Everything
clc; 
clear all;
close all;

%% Directories
TrainingDataset = 'DatasetsForFaceRecognition\Training\';
TestingDataset = 'DatasetsForFaceRecognition\Testing\';

%% Parameters 
colorSpace = 'RGB'; %RGB, HSV, YCbCr, HSVYCbCr, Gray, Gradient

% Number of Eigen Vectors
K = 100;

% Number of different faces in the training dataset
NumSubjects = 28;

FA1 = [];
FA2 = [];
FA3 = [];
FA4 = [];
SA = [];

DisplayAccuracies = 1; %% ENABLE THIS TO SHOW END RESULT OF ACCURACIES

%%%%%% ENABLE BELOW FOR CORRESPONDING END TO RUN FOR MULTIPLE K VALUES %%%%%%
Plot = 1; %% ENABLE THIS TO PLOT THE GRAPHS WHEN USING MULTIPLE K VALUES
for K = 1 : 10 : 180 
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

    % Visualizing Eigen Values %%%%%%%%%%%%% UNCOMMENT TO DISPLAY THE GRAPH    
    bar(diag(Lambda));
    title('Eigen Values');
    
    % Step 5: Finding the top K EigenVectors
    TopKEigenVectors = V(:,end-K+1:end);

    % Step 6: Computing Uk where Uk = A.Vk
    Uk = A * TopKEigenVectors;
    
    % Visualizing the Eigen Faces
%     figure;
%     imshow(reshape(Uk(:,end-3), 40, 30, 3));
    
    % Step 7: Computing Alpha = Uk'. ImageMatrix
    TrainingAlpha = Uk' * TrainingMatrix;

    % Step 8: Generating the FeatureVector for Reference Images
    TrainingFeatureVector = TrainingAlpha';

    %% TESTING

    % Step 1: Reading the Testing dataset
    [TestingMatrix, nrows, ncols, np, TestingLabels] = getDataMatrix(TestingDataset, colorSpace, NumSubjects);

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
    [MinDistance1, Indices1] = pdist2(TrainingFeatureVector, TestingFeatureVector, 'euclidean', 'Smallest', 1); % euclidean, minkowski, cityblock, mahalanobis
    [MinDistance2, Indices2] = pdist2(TrainingFeatureVector, TestingFeatureVector, 'minkowski', 'Smallest', 1); % euclidean, minkowski, cityblock, mahalanobis
    [MinDistance3, Indices3] = pdist2(TrainingFeatureVector, TestingFeatureVector, 'cityblock', 'Smallest', 1); % euclidean, minkowski, cityblock, mahalanobis
    [MinDistance4, Indices4] = pdist2(TrainingFeatureVector, TestingFeatureVector, 'mahalanobis', 'Smallest', 1); % euclidean, minkowski, cityblock, mahalanobis
    
    % Step 7: Calculating Accuracy of Faces
    [AccuracyFaces1, AccuracySubjects1] = checkAccuracy(colorSpace, Indices1, size(TestingMatrix, 2), TrainingLabels, TestingLabels, NumSubjects, TestingFeatureVector, TrainingFeatureVector);
    [AccuracyFaces2, AccuracySubjects2] = checkAccuracy(colorSpace, Indices2, size(TestingMatrix, 2), TrainingLabels, TestingLabels, NumSubjects, TestingFeatureVector, TrainingFeatureVector);
    [AccuracyFaces3, AccuracySubjects3] = checkAccuracy(colorSpace, Indices3, size(TestingMatrix, 2), TrainingLabels, TestingLabels, NumSubjects, TestingFeatureVector, TrainingFeatureVector);
    [AccuracyFaces4, AccuracySubjects4] = checkAccuracy(colorSpace, Indices4, size(TestingMatrix, 2), TrainingLabels, TestingLabels, NumSubjects, TestingFeatureVector, TrainingFeatureVector);
    

    % Displaying Results
%     fprintf('Accuracy of Face Images: %.02f \nAccuracy of Subjects: %.02f \n', AccuracyFaces, AccuracySubjects);
    FA1 = [FA1; [K AccuracyFaces1 AccuracySubjects1]];
    FA2 = [FA2; [K AccuracyFaces2 AccuracySubjects2]];
    FA3 = [FA3; [K AccuracyFaces3 AccuracySubjects3]];
    FA4 = [FA4; [K AccuracyFaces4 AccuracySubjects4]];
    
end %%%%%%%%%%%%%%%%%%%%%%%%%%% THIS IS THE END TO BE UNCOMMENTED TO
% ENABLE MULTIPLE K EXECUTION

if (Plot == 1)
    subplot(2,2,1);
    plot(FA1(:,1),FA1(:,2),FA1(:,1),FA1(:,3))
    legend('Face Accuracy','Subject Accuracy')
    title('Euclidean distance')
    subplot(2,2,2);
    plot(FA2(:,1),FA2(:,2),FA2(:,1),FA2(:,3))
    legend('Face Accuracy','Subject Accuracy')
    title('Minkowski distance')
    subplot(2,2,3);
    plot(FA3(:,1),FA3(:,2),FA3(:,1),FA3(:,3))
    legend('Face Accuracy','Subject Accuracy')
    title('CityBlock Distance')
    subplot(2,2,4);
    plot(FA4(:,1),FA4(:,2),FA4(:,1),FA4(:,3))
    legend('Face Accuracy','Subject Accuracy')
    title('Mahalanobis Distance')

end

if (DisplayAccuracies == 1)
    disp('Eigen Faces Project:');
    fprintf('\nEuclidean\nAccuracy of Face Images: %.02f \nAccuracy of Subjects: %.02f \n', AccuracyFaces1, AccuracySubjects1);
    fprintf('\nMinkowski\nAccuracy of Face Images: %.02f \nAccuracy of Subjects: %.02f \n', AccuracyFaces2, AccuracySubjects2);
    fprintf('\nCityBlock\nAccuracy of Face Images: %.02f \nAccuracy of Subjects: %.02f \n', AccuracyFaces3, AccuracySubjects3);
    fprintf('\nMahalanobis\nAccuracy of Face Images: %.02f \nAccuracy of Subjects: %.02f \n', AccuracyFaces4, AccuracySubjects4);
end
