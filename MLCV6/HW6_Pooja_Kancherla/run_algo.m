function what_goes_here = run_algo(train_images, test_images, colorSpace, n_diff_faces, K)
    
    train_data = [];
    
    for face = 1 : n_diff_faces
        path = sprintf('%s%02d\\', train_images, face);
        [data, nrows, ncols, np] = getAllIms(path, colorSpace);
%         
%         % Extracting the image count in each sub folder
%         NumImagesInFolder = dir([path '*.jpg']);    
% 
%         % Creating the contents of each subfolder
%         for i = 1 : size(NumImagesInFolder,1)
%             row = [str2num(sprintf('%02d', face)) i];
%             train_labels = [train_labels; row];
%         end
%         
        train_data = [train_data; data];
                
    end
    train_data = train_data';
    
    % Computing Mean of the data
    train_mean = zeros(size(train_data,1),1);
    
    for i = 1 : size(train_data,2)
        train_mean = train_mean + train_data(:,i);
    end
    train_mean = train_mean./size(train_data,2);
       
    % Subtracting mean from datamatrix
    train_datamatrix = bsxfun(@minus, train_data, train_mean);
    
    % Eigen Decomposition
    [V, DiagonalMatrix] = eig(train_datamatrix.' * train_datamatrix);
    
    % Selecting top K
    V = V(:, end-K+1:end);
    
    % Computing Eigen Faces
    U = train_datamatrix * V;
    
    % Computing training alpha
    train_alpha = U.' * train_datamatrix;
    
    % Feature vectr generation
    train_feature_vector = train_alpha.';
    
    
    
    test_data = [];
    for face = 1 : n_diff_faces
        path = sprintf('%s%02d\\', test_images, face);
        [data, nrows, ncols, np] = getAllIms(path, colorSpace);
%         
%         % Extracting the image count in each sub folder
%         NumImagesInFolder = dir([path '*.jpg']);    
% 
%         % Creating the contents of each subfolder
%         for i = 1 : size(NumImagesInFolder,1)
%             row = [str2num(sprintf('%02d', face)) i];
%             train_labels = [train_labels; row];
%         end
%         
        test_data = [test_data; data];
                
    end
    test_data = test_data';
    
    % Computing Mean of the data
    test_mean = zeros(size(test_data,1),1);
    
    for i = 1 : size(test_data,2)
        test_mean = test_mean + test_data(:,i);
    end
    test_mean = test_mean./size(test_data,2);
    
    % Subtracting mean from datamatrix
    test_datamatrix = bsxfun(@minus, test_data, test_mean);
    
    % Projecting to subspace in order to obtain features
    test_alpha = U'*test_datamatrix;
    
    %% Similarity Metrics
    num_train_images = size(train_datamatrix,2);
    num_test_images = size(test_datamatrix,2);
    
    % Extracting the min distance and the corresponding indices
    [min_distance_euc, index_euc] = calc_distance('euclidean', K, num_train_images, num_test_images, train_alpha, test_alpha);
    [min_distance_man, index_man] = calc_distance('manhattan', K, num_train_images, num_test_images, train_alpha, test_alpha);
    [min_distance_mah, index_mah] = calc_distance('mahalanobis', K, num_train_images, num_test_images, train_alpha, test_alpha);

    
  
    what_goes_here = 1;
    return