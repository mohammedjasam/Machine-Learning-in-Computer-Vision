% Visualize all the methods

clc;
clear;
close all;

%% Extract inferences from each method

[lr, gt] = Linear_Regression();
[fs, gt] = Feature_Selection();
[blr, gt] = Bayesian_Regularization();
[nlr, gt] = Non_Linear_Regression();
[dnlr, gt] = Dual_Non_Linear_Regression();

close all;
figure();
plot(gt); 
hold on;
plot(lr);
hold on;
plot(fs);
hold on;
plot(blr);
hold on;
plot(nlr);
hold on;
plot(dnlr);

legend('Ground Truth', 'Linear Regression', 'Feature Selection',...
        'Bayesian LR + FS', 'Non Linear Regression', 'Dual Non-LR');
hold off;
title('Ground Truth vs Various Methods');