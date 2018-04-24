% clc; 
clear; 
close all;
warning ('off','all');

train_face_dir = 'trainingImages\face_resized\';
train_back_dir = 'trainingImages\background_resized\';
test_face_dir = 'testingImages\face_resized\';
test_back_dir = 'testingImages\background_resized\';

colorSpace = 'RGB';

%% Training
X_face_train = getAllIms(train_face_dir, colorSpace);
X_back_train = getAllIms(train_back_dir, colorSpace);
X_train = [ones(1, (size(X_face_train, 1) + size(X_back_train, 1))); [X_face_train' X_back_train']];
w_train = [ones(size(X_face_train, 1), 1); zeros(size(X_back_train, 1), 1)];

X_face_test = getAllIms(test_face_dir, colorSpace);
X_back_test = getAllIms(test_back_dir, colorSpace);
X_test = [ones(1, (size(X_face_test, 1) + size(X_back_test, 1))); [X_face_test' X_back_test']];
w_test = [ones(size(X_face_test, 1), 1); zeros(size(X_back_test, 1), 1)];

% Initial Phi value
initial_phi = pinv(X_train') * w_train;

%% Testing
lambda = 1;
initial_psi = zeros((size(X_face_train, 1) + size(X_back_train, 1)), 1);
var_prior = var(initial_phi);

[predictions, phi] = fit_klogr (X_train, w_train, var_prior, X_test, initial_psi, @kernel_gauss, lambda);

for i = 1 : length(predictions)
    if predictions(i) >= 0.5
       predictions(i) = 1;
    else
       predictions(i) = 0;
    end
end

miss_detection_rate = 0;
false_alarm_rate = 0;

for i = 1 : size(X_face_test, 1)
    miss_detection_rate = miss_detection_rate + abs(w_test(i) - predictions(i));
end
miss_detection_rate = miss_detection_rate / size(X_face_test, 1);

for i = (size(X_face_test, 1) + 1) : (size(X_face_test, 1) + size(X_back_test, 1))
    false_alarm_rate = false_alarm_rate + abs(w_test(i) - predictions(i));
end
false_alarm_rate = false_alarm_rate / size(X_back_test, 1);

fprintf('\n\nKernel Logistic Regression\n');
fprintf('--------------------------\n');
fprintf('Miss Detection Rate: %f\nFalse Alarm Rate: %f\n\n\n', miss_detection_rate, false_alarm_rate);



