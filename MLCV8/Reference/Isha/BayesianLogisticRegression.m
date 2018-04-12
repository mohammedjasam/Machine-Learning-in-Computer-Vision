%Bayesian Logistic Regression method

clc; clear all; close all;

facedir_training = 'face_resized\';
bgdir_training='background_resized\';
colorspace = 'Gray'; %RGB, Gray, HSV, YCbCr, HSVYCbCr, Gradient

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[faceImageFiles,nrows_1,ncols_1,np_1]=getAllIms(facedir_training,colorspace);
[bgImageFiles,nrows_0,ncols_0,np_0]=getAllIms(bgdir_training,colorspace);

X_face_files=size(faceImageFiles,1);
X_bg_files=size(bgImageFiles,1);

X=[bgImageFiles' faceImageFiles'];
% X=faceImageFiles';
% X=[X,bgImageFiles'];

X = [ones(1,size(X,2)); X];

w=[];
for i=1:X_bg_files
    w = [w; 0];
end

for i=1:X_face_files
    w = [w; 1];
end

initial_phi = 0.3*ones(size(X,1),1);
var_prior=6;
% sig=sum(var(X));      
% var_prior=sig/100;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

facedir_testing = 'testingImages\face_resized\';
bgdir_testing='testingImages\background_resized\';

[faceIm_test,nrows_test1,ncols_test1,np_test1]=getAllIms(facedir_testing,colorspace);
[bgIm_test,nrows_test0,ncols_test0,np_test0]=getAllIms(bgdir_testing,colorspace);

X_test_face_files=size(faceIm_test,1);
X_test_bg_files=size(bgIm_test,1);

X_test=[bgIm_test' faceIm_test'];
X_test=[ones(1,size(X_test,2)); X_test];

w_test=[zeros(X_test_bg_files,1); ones(X_test_face_files,1)];
% for i=1:X_test_bg_files
%     w_test = [w; 0];
% end
% 
% for i=1:X_test_face_files
%     w_test = [w; 1];
% end

[predictions, phi] = fit_blogr (X, w, var_prior, X_test, initial_phi);
y_cap=predictions;

for i=1:size(y_cap,2)
    if y_cap(i)>0.5
        y_cap(i)=1;        
    else        
        y_cap(i)=0;    
    end
end

absError= abs(y_cap-w_test');

%Calculate Miss detection for face
%Testing face files are after 1 row of ones and total number of background
%images
Miss_detection=sum(absError(X_test_bg_files+1:size(X_test,2)))/X_test_face_files;

%Calculate False alarm for background
False_alarm= sum(absError(1:X_test_bg_files))/X_test_bg_files;

%Plotting Miss detection for face
plot((1:232),w_test(1:232),'r+');
title('Miss Detection');
xlabel('Images');
ylabel('Image 0-1');
hold on;
plot((1:232),y_cap(1:232),'ko');
legend('Ground truth','Prediction');

%Plotting False Alarm for background
figure;
plot((233:796),w_test(233:796),'r+');
title('False Alarm');
xlabel('Images');
ylabel('Image 0-1');
hold on;
plot((233:796),y_cap(233:796),'ko');
legend('Ground truth','Prediction');

%Plotting Predictions
figure;
plot((1:796),predictions,'r.')
title('Predictions');
xlabel('Images');
ylabel('Image 0-1');
legend('Predictions');


