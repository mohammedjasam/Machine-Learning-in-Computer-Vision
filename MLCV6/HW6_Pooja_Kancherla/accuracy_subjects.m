function subject_acc = accuracy_subjects(train_images, test_images, n_diff_faces, train_feature_vector, test_feature_vector)
    cur_index_test = 0;
    
    smallest_match_between_subs = [];

    for i = 1 : n_diff_faces
        each_match_min_distance = [];
        path = sprintf('%s%02d\\', test_images, i);
        files_in_dir_test = dir(path);
        no_files_in_dir_test = size(files_in_dir_test, 1);        
        slice_test = test_feature_vector(cur_index_test+1:cur_index_test+(no_files_in_dir_test-2),:);

        for a = 1 : no_files_in_dir_test - 2
            cur_index_test = cur_index_test + 1; 
        end
        
        cur_index_train = 0;
        for j = 1 : n_diff_faces
            path = sprintf('%s%02d\\', train_images, j);
            files_in_dir_train = dir(path);
            no_files_in_dir_train = size(files_in_dir_train, 1);                
            slice_train = train_feature_vector(cur_index_train+1:cur_index_train+(no_files_in_dir_train-2),:);
            
%             disp(no_files_in_dir_train - 2)
%             disp(cur_index_train+1);
%             disp(cur_index_train+(no_files_in_dir_train-2));
%             disp('===============================');
            for b = 1 : no_files_in_dir_train - 2
                cur_index_train = cur_index_train + 1; 
            end
            
            
            min_distance = find_best_match(slice_test, slice_train);
            each_match_min_distance = [each_match_min_distance; min_distance];
            
        end
        [Y, I] = min(each_match_min_distance,[],1);
        smallest_match_between_subs = [smallest_match_between_subs; I];
       


    end
    count = 0;
    for i = 1 : n_diff_faces
        if ismember(i, smallest_match_between_subs)
            count = count + 1;
        end
    end
    
subject_acc = count/n_diff_faces;