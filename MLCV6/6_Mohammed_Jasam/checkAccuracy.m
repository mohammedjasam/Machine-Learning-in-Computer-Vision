function[AccuracyFaces, AccuracySubjects] = checkAccuracy(colorSpace, Indices, NumTestImages, TrainingLabels, TestingLabels, NumSubjects, TestingFeatureVector, TrainingFeatureVector, ShowMatch)
    
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
    
    % Calculating Subject accuracy
    ArraySub = [];
    for i1 = 1 : NumSubjects
        tempTest = [];
        for j1 = 1 : size(TestingFeatureVector, 1)
            if (TestingLabels(j1,1) == i1)
                tempTest = [tempTest; TestingFeatureVector(TestingLabels(j1,3),:)];
            end        
        end
        minD = [];
        for i = 1 : NumSubjects
            tempTrain = [];        
            for j = 1 : size(TrainingFeatureVector, 1)
                if (TrainingLabels(j,1) == i)
                    tempTrain = [tempTrain; TrainingFeatureVector(TrainingLabels(j,3),:)];
                end        
            end
            subD = [];
            for i2 = 1 : size(tempTest,1)
                for j2 = 1 : size(tempTrain, 1)
                    D = norm(tempTest(i2) - tempTrain(j2));
                    subD = [subD; D];
                end
            end
            minD = [minD; min(subD)];
        end
        [minddistance, Indexx] = min(minD,[],1);
        ArraySub = [ArraySub; Indexx];
    end

    ArraySUBS = 0;

    for i = 1 : NumSubjects
        if ismember(i, ArraySub)
            ArraySUBS = ArraySUBS + 1;
        end
    end
     
    %Accuracy of Images from Total Test Images
    AccuracyFaces = CountImages/NumTestImages;
    
    %Accuracy of Subjects from Total Subjects
    AccuracySubjects = ArraySUBS/NumSubjects;

end