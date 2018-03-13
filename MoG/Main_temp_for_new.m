clc;
clear all;
close all;
warning('off');

disp(' ## Program for Face Classification by MOG ## ');
 
ColorGamut = 'RGB';


[face_matrix, fc_image_num] = getData('trainingDataset\face_resized\', ColorGamut);
[background_matrix, bg_image_num] = getData('trainingDataset\background_resized\', ColorGamut);
[test_face_matrix, fc_test_image_num] = getData('testingDataset\face_resized\', ColorGamut);
[test_background_matrix, bg_test_image_num] = getData('testingDataset\background_resized\', ColorGamut);


%%
[lambda_face, mean_face, sigma_face] = MOG_FCN(face_matrix, 3, 100, fc_image_num, 900); % For 3 Gaussian mixture curvature.
[lambda_bg, mean_bg, sigma_bg] = MOG_FCN(background_matrix, 3, 100, bg_image_num, 900); % For 3 Gaussian mixture curvature.

% INFERENCE ALGORITHM:


True_fc_num=0;
False_fc_num=0;

disp('computing inference for test face images: ');

for k = 1 : 3
    
    fc_img_test(:,k)=lambda_face(k)*mvgd1(test_face_matrix, mean_face(k,:), sigma_face{k}, fc_test_image_num, 900);
end
 fc_img_test=double(fc_img_test);
 
 daffuq = 0
 for  iFile = 1:fc_test_image_num -1;
     daffuq = daffuq + 1;
     fc_img_test1(iFile,1)=sum(fc_img_test(iFile,:));
     
 end
 disp(daffuq);
 
 for k = 1 : 3
    
    fc_img_test(:,k)=lambda_bg(k)*mvgd1(test_face_matrix, mean_bg(k,:), sigma_bg{k}, fc_test_image_num, 900);
end
 fc_img_test=double(fc_img_test);
 
 for  iFile = 1:fc_test_image_num-1;
     
     fc_img_test2(iFile,1)=sum(fc_img_test(iFile,:));
     
 end
 
 for iFile = 1:fc_test_image_num-1;
    
    if (fc_img_test1(iFile,1)) > (fc_img_test2(iFile,1));
        
        True_fc_num=True_fc_num+1;
        True_face_images(True_fc_num,1)=iFile; 
        
    else
        
        False_fc_num=False_fc_num+1;
        False_face_images(False_fc_num,1)=iFile;
    
    end
 end

True_bg_num=0;
False_bg_num=0;

disp('computing inference for test background images: ');

for k = 1 : 3
    
    bg_img_test(:,k)=lambda_bg(k)*mvgd1(test_background_matrix, mean_bg(k,:), sigma_bg{k}, bg_test_image_num, 900);
end
 bg_img_test=double(bg_img_test);
 
 for  iFile = 1:bg_test_image_num-1;
     
     bg_img_test1(iFile,1)=sum(bg_img_test(iFile,:));
     
 end
 
 
 for k = 1 : 3
    
    bg_img_test(:,k)=lambda_face(k)*mvgd1(test_background_matrix, mean_face(k,:), sigma_face{k}, bg_test_image_num, 900);
end
 bg_img_test=double(bg_img_test);
 
 for  iFile = 1:bg_test_image_num-1;
     
     bg_img_test2(iFile,1)=sum(bg_img_test(iFile,:));
     
 end
 
 
for iFile = 1:bg_test_image_num-1;
    
    if (bg_img_test1(iFile,1)) > (bg_img_test2(iFile,1));
        
        True_bg_num=True_bg_num+1;
        True_bg_images(True_bg_num,1)=iFile;
        
    else
        
        False_bg_num=False_bg_num+1;
        False_bg_images(False_bg_num,1)=iFile;
    
    end
end

% Calculation of accuracy:

disp('computing accuracy: ');

face_accuracy=(True_fc_num*100)/(fc_test_image_num-1);
background_accuracy=(True_bg_num*100)/(bg_test_image_num-1);
total_accuracy=((True_fc_num+True_bg_num)*100)/((fc_test_image_num-1)+(bg_test_image_num-1));

disp("Results");
disp("Face: " + face_accuracy);
disp("Background: " + background_accuracy);
disp("Total Accuracy: " + total_accuracy);


