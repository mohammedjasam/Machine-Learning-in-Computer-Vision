% Visualize all the methods

clc;
clear;
close all;

%% Extract inferences from each method

[msg1, lr, gt] = Linear_Regression();
[msg2, fs, gt] = Feature_Selection();
[msg3, blr, gt] = Bayesian_Regularization();
[msg4, nlr, gt] = Non_Linear_Regression();
[msg5, dnlr, gt] = Dual_Non_Linear_Regression();

disp('Mean Absolute Error:');
disp('----------------------------------------------------------')
disp(sprintf('| 1. %s                        |', msg1));
disp(sprintf('| 2. %s |', msg2));
disp(sprintf('| 3. %s                  |', msg3));
disp(sprintf('| 4. %s                    |', msg4));
disp(sprintf('| 5. %s               |', msg5));
disp('----------------------------------------------------------')


close all;
diff = [];
diff = [diff (lr - gt')' (fs - gt')' (blr - gt')' (nlr - gt')' (dnlr - gt')'];

figure();
plot(diff(:,1))
hold on
plot(diff(:,2))
hold on
plot(diff(:,3))
hold on 
plot(diff(:,4))
hold on
plot(diff(:,5))
hold on
plot(gt' - gt')

title('Deviation Chart')
legend('Ground Truth', 'Linear Regression', 'Feature Selection',...
        'Bayesian LR + FS', 'Non Linear Regression', 'Dual Non-LR');

figure()
bar(diff(:,1))
hold on
bar(diff(:,2))
hold on
bar(diff(:,3))
hold on
bar(diff(:,4))
hold on
bar(diff(:,5))
hold on
bar(gt' - gt')

title('Deviation Chart')
legend('Ground Truth', 'Linear Regression', 'Feature Selection',...
        'Bayesian LR + FS', 'Non Linear Regression', 'Dual Non-LR');