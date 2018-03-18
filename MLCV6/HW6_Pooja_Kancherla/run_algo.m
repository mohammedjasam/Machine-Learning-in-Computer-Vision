function what_goes_here = run_algo(train_images, test_images, colorSpace, n_diff_faces, K)
    
    train_data = [];
    train_labels = [];
    
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
    XXX = train_mean;
    train_mean = train_mean./size(train_data,2);
       
    what_goes_here = 1;