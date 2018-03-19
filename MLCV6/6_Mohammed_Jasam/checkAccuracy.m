function[AccuracyFaces, AccuracySubjects] = checkAccuracy(Indices, NumTestImages, TrainingLabels, TestingLabels, NumSubjects)
    
    CountImages = 0;
    comp = zeros(NumTestImages, 2);
    
    % Counting the image Accuracy
    for i = 1 : NumTestImages
        comp(i,:) = [TestingLabels(i,1), TrainingLabels(Indices(i),1)];
        if(TestingLabels(i,1) == TrainingLabels(Indices(i),1))
            CountImages = CountImages +1;
        end
    end
    
    % Counting the subject Accuracy
    CountSubjects = 0;
    for i = 1 : NumSubjects
        if ismember(i, comp(:,2))
            CountSubjects = CountSubjects + 1;
        end
    end
    
    %Accuracy of Images from Total Test Images
    AccuracyFaces = CountImages/NumTestImages;
    
    %Accuracy of Subjects from Total Subjects
    AccuracySubjects = CountSubjects/NumSubjects;

end