function visualize_predictions(num_of_op, test_face_dir, test_back_dir, predictions, num_test_face)

    random_nums = randperm(170); % randomly picking images to viz
    i = 0;
    figure();
    
    % Visualizing the images predicted as Background
    for r = random_nums(1 : num_of_op)
        i = i + 1;
        images = dir(test_back_dir);
        images(1:3) = []; % eliminating the first two dirs ('.', '..')
        im = imread(strcat(test_back_dir, images(r).name));        
        subplot(2,num_of_op,i), imshow(im);
        title(sprintf('%f', predictions(num_test_face + r)))
    end    
       
    % Visualizing the images predicted as Faces
    for r = random_nums(num_of_op + 1 : num_of_op * 2)
        i = i + 1;
        images = dir(test_face_dir);
        images(1:3) = []; % eliminating the first two dirs ('.', '..')
        im = imread(strcat(test_face_dir, images(r).name));
        subplot(2,num_of_op,i), imshow(im);
        title(sprintf('%f', predictions(r)))
    end


end