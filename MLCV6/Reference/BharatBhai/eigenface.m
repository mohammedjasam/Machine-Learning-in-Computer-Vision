clc; clear all; close all;

facedir_training = 'training\';
facedir_testing = 'testing\';

colorSpace = 'RGB'; %RGB, Gray, HSV, YCbCr, HSVYCbCr, Gradient
nSubject = 33; %There are 33 persons in the training dataset. There are only 27 in the testing dataset
K = 100; %the number of eigen vectors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%111111111111 read in all the images
allFaceIms_training = [];
labels1 = []; %each row represents [folder, image in the folder] i.e., [person, which image of the person].

for iSubject = 1:nSubject
    [FaceIms,nrows,ncols,np] = getAllIms(sprintf('%s%02d\\',facedir_training,iSubject),colorSpace);
    if isempty(FaceIms), continue; end
    allFaceIms_training = [allFaceIms_training; FaceIms];    
    labels1 = [labels1; [iSubject*ones(size(FaceIms,1),1) (1:size(FaceIms,1))']]; 
end
allFaceIms_training = allFaceIms_training'; %every column in allFaceIms_training is one face image

%222222222 compute the average face vector
I_mean = mean(allFaceIms_training,2);
bbbbb = allFaceIms_training;

%333333333 Subtract the mean face from each face so each face retains only special
% characteristics peculiar to it
allFaceIms = bsxfun(@minus, allFaceIms_training, I_mean);

%Compute eigen vectors for these faces after substracting mean and their associated eigen values
[V,D] = eig(allFaceIms'*allFaceIms);
return
%Plot histogram of the eigen values
bar(diag(D))

%Compute eigen faces by taking product of the above computed eigen vector
%matrix and the natural face images. Effectively, we have reduced the
%dimention of the training data from 184 to say the value of K=20.
U = allFaceIms*V;

%Select best few eigen faces to represent the training images
U = U(:,end-K+1:end);

%An example image displayed for eigen face second best
figure;
imshow(reshape(U(:,end-1), [40, 30, 3]));

%Compute feature vectors (wieghts) for each face image so it can be
%recovered by taking product of alpha and eigen face matrix.
alpha = U'*allFaceIms;

%Inference


allFaceIms_testing = [];
labels2 = []; %each row represents [folder, image in the folder] i.e., [person, which image of the person].
for iSubject = 1:nSubject
    [FaceIms_test,nrows,ncols,np] = getAllIms(sprintf('%s%02d\\',facedir_testing,iSubject),colorSpace);
    
    if isempty(FaceIms), continue; end
    allFaceIms_testing = [allFaceIms_testing; FaceIms_test];    
    labels2 = [labels2; [iSubject*ones(size(FaceIms_test,1),1) (1:size(FaceIms_test,1))']]; 
end

allFaceIms_testing = allFaceIms_testing';

% Substract the mean of the original training dataset from the test image
% dataset so only those characteristics are retained that are unique to the
% image under consideration
allFaceIms_test = bsxfun(@minus, allFaceIms_testing, I_mean);

%Determine the feature vector(weights) that uniquely identify the
%underlying characteristics of an image under study.
alpha_testing = U'*allFaceIms_test;

%Number of testing images
nTest = size(allFaceIms_testing,2);


% compute euclidean (norm 2) distance
Dis = pdist2(alpha', alpha_testing','euclidean');


%Accuracy
[acc_Eucl,min_dist_Eul,I_Eul, comp_Eul] = comAccu(Dis,nTest,labels1,labels2);

%compute norm 1 distance. CityBlock is another name for norm 1.

dis = pdist2(alpha', alpha_testing','cityblock');

%Accuracy
[acc_CityBlo,min_dist_CityBlo,I_CityBlo, comp_CityBlo] = comAccu(dis,nTest,labels1,labels2);


% compute Mahalanobis distance
%D2 = mahal(alpha_testing', alpha');

[R, N] = size(alpha);
[R, P] = size(alpha_testing);

C = [alpha, alpha_testing];
invcov = inv(cov(C'));

for i=1:N
    diff = repmat(alpha(:,i), 1, P) - alpha_testing;
    dsq(i,:) = sum((invcov*diff).*diff , 1);
end

d = sqrt(dsq);
%Accuracy
[acc_Mahal,min_dist_mahal,I_mahal, comp_mahal] = comAccu(d,nTest,labels1,labels2);

