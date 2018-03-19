function [pathToFile, person, bucket, onlypath] = getFileNameFromIndex(dataset, index, n_diff_faces)
    cur_index = 0;
    
    if strcmpi(dataset, 'training') == 1
        root = 'DatasetsForFaceRecognition\Training\';
    else
        root = 'DatasetsForFaceRecognition\Testing\';
    end
    
    for i = 1 : n_diff_faces
        path = sprintf('%s%02d\\', root, i);
        files_in_dir = dir(path);
        no_files_in_dir = size(files_in_dir, 1);
        
        for j = 3 : no_files_in_dir
            
            cur_index = cur_index + 1;
            if  cur_index == index
                bucket = sprintf('%02d', i);
                onlypath = sprintf('%02d\\%s', i, (files_in_dir(j).name));
                pathToFile = sprintf('%s%02d\\%s', root, i, (files_in_dir(j).name));
                person = i;
                break;
            end
            
        end
    end
end