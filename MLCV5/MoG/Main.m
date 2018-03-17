clc;
clear all;
close all;
warning('off');

disp('Face vs. Background using MoG');
disp("--------------------------------------- ");
 
colorSpace = 'Gradient'; %'RGB', 'HSV', 'Gradient', 'Gray'

disp('Reading Data');

[train_faces, Train_Num_Faces] = getData('trainingDataset\face_resized\', colorSpace);
[train_background, Train_Num_Backgrounds] = getData('trainingDataset\background_resized\', colorSpace);
[test_faces, Test_Num_Faces] = getData('testingDataset\face_resized\', colorSpace);
[test_background, Test_Num_Backgrounds] = getData('testingDataset\background_resized\', colorSpace);

% Performing the EM Steps
disp("Executing EM Algorithm");
[Lambda_Faces, Mean_Faces, Sigma_Faces] = applyMoG(train_faces, 3, 100, Train_Num_Faces, 900); % For 3 Gaussian mixture curvature.
[Lambda_Backgrounds, Mean_Backgrounds, Sigma_Backgrounds] = applyMoG(train_background, 3, 100, Train_Num_Backgrounds, 900); % For 3 Gaussian mixture curvature.

True_fc_num=0;
False_fc_num=0;

disp('Inference: Test Face Images');

for k = 1 : 3    
    test_face_image(:,k)=Lambda_Faces(k)*mvn_pdf(test_faces, Mean_Faces(k,:), Sigma_Faces{k}, Test_Num_Faces, 900);
end
 test_face_image=double(test_face_image);

 for  iFile = 1:Test_Num_Faces -1;
     test_face_image1(iFile,1)=sum(test_face_image(iFile,:));     
 end
 
 for k = 1 : 3    
    test_face_image(:,k)=Lambda_Backgrounds(k)*mvn_pdf(test_faces, Mean_Backgrounds(k,:), Sigma_Backgrounds{k}, Test_Num_Faces, 900);
end
 test_face_image=double(test_face_image);
 
 for  iFile = 1:Test_Num_Faces-1;
     
     test_face_image2(iFile,1)=sum(test_face_image(iFile,:));
     
 end
 
 for iFile = 1:Test_Num_Faces-1;    
    if (test_face_image1(iFile,1)) > (test_face_image2(iFile,1));        
        True_fc_num=True_fc_num+1;
        True_face_images(True_fc_num,1)=iFile;         
    else        
        False_fc_num=False_fc_num+1;
        False_face_images(False_fc_num,1)=iFile;    
    end
 end


True_bg_num=0;
False_bg_num=0;

disp('Inference: Test Background Images');

for k = 1 : 3    
    test_back_image(:,k)=Lambda_Backgrounds(k)*mvn_pdf(test_background, Mean_Backgrounds(k,:), Sigma_Backgrounds{k}, Test_Num_Backgrounds, 900);
end
 test_back_image=double(test_back_image);
 
 for  iFile = 1:Test_Num_Backgrounds-1;     
     test_back_image1(iFile,1)=sum(test_back_image(iFile,:));     
 end
 
 
 for k = 1 : 3    
    test_back_image(:,k)=Lambda_Faces(k)*mvn_pdf(test_background, Mean_Faces(k,:), Sigma_Faces{k}, Test_Num_Backgrounds, 900);
end
 test_back_image=double(test_back_image);
 
 for  iFile = 1:Test_Num_Backgrounds-1     
     test_back_image2(iFile,1)=sum(test_back_image(iFile,:));     
 end
 
 
for iFile = 1:Test_Num_Backgrounds-1    
    if (test_back_image1(iFile,1)) > (test_back_image2(iFile,1))        
        True_bg_num=True_bg_num+1;
        True_bg_images(True_bg_num,1)=iFile;        
    else        
        False_bg_num=False_bg_num+1;
        False_bg_images(False_bg_num,1)=iFile;    
    end
end

% Performance metrics

t_p_face = True_fc_num;
f_p_face = False_bg_num;
f_n_face = False_fc_num;

t_p_back = True_bg_num;
f_p_back = False_fc_num;
f_n_back = False_bg_num;

p_face = t_p_face / (t_p_face + f_p_face);
r_face = t_p_face / (t_p_face + f_n_face);
f1_face = (2 * p_face * r_face) / (p_face + r_face);

p_back = t_p_back / (t_p_back + f_p_back);
r_back = t_p_back / (t_p_back + f_n_back);
f1_back = (2 * p_back * r_back) / (p_back + r_back);

% Calculation of accuracy:
face_accuracy=(True_fc_num*100)/(Test_Num_Faces-1);
background_accuracy=(True_bg_num*100)/(Test_Num_Backgrounds-1);
total_accuracy=((True_fc_num+True_bg_num)*100)/((Test_Num_Faces-1)+(Test_Num_Backgrounds-1));

% Results
disp(" ");
disp("Results");
disp("--------------------------------------- ");
disp("Face Accuracy: " + face_accuracy);
disp("Face Precision: " + p_face);
disp("Face Recall: " + r_face);
disp("Face F-1 Score: " + f1_face);
disp("--------------------------------------- ");
disp("Background Accuracy: " + background_accuracy);
disp("Background Precision: " + p_back);
disp("Background Recall: " + r_back);
disp("Background F-1 Score: " + f1_back);
disp("--------------------------------------- ");
disp("Total Accuracy: " + total_accuracy);
disp("--------------------------------------- ");

