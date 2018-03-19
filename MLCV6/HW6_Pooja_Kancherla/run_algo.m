function AccMat = run_algo(train_images, test_images, colorSpace, n_diff_faces, K, display_images)
what_goes_here = [];
    %% Training
    train_data = [];
    
    for face = 1 : n_diff_faces
        path = sprintf('%s%02d\\', train_images, face);
        [data, nrows, ncols, np] = getAllIms(path, colorSpace);
        train_data = [train_data; data];                
    end
    train_data = train_data';
    
    % Computing Mean of the data
    train_mean = zeros(size(train_data, 1), 1);
    
    for i = 1 : size(train_data, 2)
        train_mean = train_mean + train_data(:, i);
    end
    train_mean = train_mean./ size(train_data, 2);
       
    % Subtracting mean from datamatrix
    train_datamatrix = bsxfun(@minus, train_data, train_mean);
    
    % Eigen Decomposition
    [V, DiagonalMatrix] = eig(train_datamatrix.' * train_datamatrix);
    
    % Selecting top K
    V = V(:, end-K+1 : end);
    
    % Computing Eigen Faces
    U = train_datamatrix * V;
    
    % Computing training alpha
    train_alpha = U.' * train_datamatrix;
    
    % Feature vectr generation
    train_feature_vector = train_alpha.';
    
    
    %% Testing
    test_data = [];
    
    for face = 1 : n_diff_faces
        path = sprintf('%s%02d\\', test_images, face);
        [data, nrows, ncols, np] = getAllIms(path, colorSpace);
        test_data = [test_data; data];                
    end
    test_data = test_data';
    
    % Computing Mean of the data
    test_mean = zeros(size(test_data, 1), 1);
    
    for i = 1 : size(test_data, 2)
        test_mean = test_mean + test_data(:, i);
    end
    test_mean = test_mean./ size(test_data, 2);
    
    % Subtracting mean from datamatrix
    test_datamatrix = bsxfun(@minus, test_data, test_mean);
    
    % Projecting to subspace in order to obtain features
    test_alpha = U' * test_datamatrix;
    
    %% Similarity Metrics
    num_train_images = size(train_datamatrix, 2);
    num_test_images = size(test_datamatrix, 2);
    
    % Extracting the min distance and the corresponding indices
    [distance_matrix_euc, min_distance_euc, index_euc] = calc_distance('euclidean', K, num_train_images, num_test_images, train_alpha, test_alpha);
    [distance_matrix_man, min_distance_man, index_man] = calc_distance('manhattan', K, num_train_images, num_test_images, train_alpha, test_alpha);
    [distance_matrix_mah, min_distance_mah, index_mah] = calc_distance('mahalanobis', K, num_train_images, num_test_images, train_alpha, test_alpha);

    %% Calculating accuracy
    AccMat = K;
    face_acc = [];
    for i = 1 : 3
        faces = [];
        Predicted = 0;
        if i == 1
            index = index_euc;
        end
        if i == 2
            index = index_man;
        end
        if i == 3
            index = index_mah;
        end
        
        
        for i = 1 : num_test_images 
            [test_image_pathh, person, bucket_test, imagepath_test] = getFileNameFromIndex('testing', i, n_diff_faces);
            [matchedFilePath, person, bucket_train, imagepath_train] = getFileNameFromIndex('training', round(index(i)), n_diff_faces);
            
            if bucket_test == bucket_train
                faces = [faces; str2num(bucket_test)];

                Predicted = Predicted + 1;
                if display_images == 1
                    if rand() < 0.1
                        figure();
                        subplot(1,2,1); imshow(imread(test_image_pathh)); title('Image from testing data');
                        subplot(1,2,2); imshow(imread(matchedFilePath)); title('Match from the training data');                
                    end
                end
            end
            
        end
        
        acc = Predicted/num_test_images;        
        AccMat = [AccMat; acc];
        
        face_count = 0;
        for i = 1 : n_diff_faces
            if ismember(i, faces)
                face_count = face_count + 1;
            end
        end
        
        face_acc = [face_acc; (face_count/n_diff_faces)];
%         fprintf("Subject Accuracy: %.2f\n", (face_count/n_diff_faces));
    end
    AccMat = [AccMat; face_acc];
