%Linear regression

%@Zhaozheng Yin, spring 2017

clc; clear all; close all;
dir_training = 'training\';
dir_testing = 'testing\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir([dir_training '*.jpg']);
X = []; w = [];
for ii = 1:size(files,1)
    filename = files(ii).name;
    w = [w; str2double(filename(1:4))];
    im = imread([dir_training filename]);
    im = im(:,:,1);
    X = [X im(:)];
end

X = double(X); %every column in X is one vectorized input image
X = [ones(1,size(X,2)); X];


%Find phi to learn the parameters and infer the rotation of the images in
%the training set
% phi = (X*X')\X*w;

%phi = regress(w,X');
phi = pinv(X')*w;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_files = dir([dir_testing '*.jpg']);
X_test = []; Gr_truth = [];
for jj = 1:size(test_files,1)
    Testfilename = test_files(jj).name;
    Gr_truth = [Gr_truth; str2double(Testfilename(1:4))];
    im_test = imread([dir_testing Testfilename]);
    im_test = im_test(:,:,1);
    X_test = [X_test im_test(:)];
end
X_test = double(X_test); %every column in X is one vectorized input image
X_test = [ones(1,size(X_test,2)); X_test];

%infer the rotation of the images using the parameters(phi) learned above
%on test images
w_inferred = phi'*X_test;
plot(Gr_truth)
hold on;
plot(w_inferred)

%Check the accuracy of the model by comparing the inferred with the actual
%values
Eval_linReg = sum(abs(w_inferred(:) - Gr_truth(:)))/size(Gr_truth,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%feature selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%By visualizing the images it can be intuitively inferred that pixels at
%certain locations(outside and inside the ring) are same across all the 
%images and hence noniniformative. It is safe to ignore these pixels for  
%computational benifits. Hence, pixel values across a particular location  
%in the input image matrix are tested for existance of any variance among 
%them and are determined to be kept if variance exist or drop otherwise.
variance = var(X,0,2);

%Choose the pixel locations where the varaince among the values is atleast
%100. Call this new matrix X_new
%X_new = X; X_new(variance<100,:) = 0;
X_new = X(variance>100,:);


%phi_feat_sele = (X_new*X_new')\(X_new*w);

%Test the accuracy for inference following the reduction of pixel locations

%variance_test = var(X_test,0,2); 
%X_new_test = X_test; X_new_test(variance_test<100,:) = 0;
X_new_test = X_test(variance>100,:);

phi123 = pinv(X_new') * w;
w_inferred123 = phi123'*X_new_test;
Eval_Feat = sum(abs(w_inferred123(:) - Gr_truth(:)))/size(Gr_truth,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Add regularization to the above model or Bayesian solution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We get the inference, 
[w_infer_reg, var_test, vari, A_inv] = fit_blr (X_new, w, var(phi), X_new_test);



lambda = vari/var(phi);

%phi_reg = (X_new*X_new' + lambda*eye(size(X_new,1)))\(X_new*w);

%phi_reg = lassoglm(X_new',w,'normal', 'Lambda', lambda);
hold on;
% plot(Gr_truth)
% hold on;
%w_infer_reg = phi_reg'*X_new_test;
plot(w_infer_reg)

Eval_Regu = sum(abs(w_infer_reg(:) - Gr_truth(:)))/size(Gr_truth,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Based on the selected features, Implement the nonlinear regression 
%(e.g., polynomial regression) using the regularization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
asdsd = 1;
[D,N] = size(X_new);
d = 2; %change the value of d to the desired number of polynomial powers 

Z = zTransform(d, D, N, X_new);

abc = Z*Z';
phi_reg_nonLinear = (abc + lambda*eye(size(Z,1)))\(Z*w);

[D,N] = size(X_new_test);
Z_test = zTransform(d,D,N,X_new_test);
w_infer_nonLinear = phi_reg_nonLinear'*Z_test;


hold on
plot(w_infer_nonLinear)
Eval_NonL = sum(abs(w_infer_nonLinear(:) - Gr_truth(:)))/size(Gr_truth,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Without using the feature selection, implement the dual nonlinear 
%regression with regularization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Since (X*X') is of (D+1)*(D+1) and can be computationally expense as D
%grows in size. So, instead if the problem is formulated as (X'*X) it is of
%size I*I and it is of our interest to recast the problem so we deal with
%(X'*X) instead. Now this is represented as psi.



[w_dual_infer, var_test] = fit_dr (X, w, var(phi), X_test);
% psi_dual_reg = (X'*X*X'*X + lambda*eye(size(X,2)))\X'*X*w;
% phi_dual_reg = X*psi_dual_reg;
% w_dual_infer = phi_dual_reg'*X_test;
hold on
plot(w_dual_infer)

legend('Ground Truth', 'Lin Reg','Regul feature selection', 'Non-linear feature selection','Dual lin Reg')
Eval_Dual = sum(abs(w_dual_infer(:) - Gr_truth(:)))/size(Gr_truth,1);

xlabel('Ground Truth rotations of the ring in the test images');
ylabel('Inferred rotations of the ring in the corresponding images');
disp(Eval_linReg);
disp(Eval_Feat);
disp(Eval_Regu);
disp(Eval_NonL);
disp(Eval_Dual);

return