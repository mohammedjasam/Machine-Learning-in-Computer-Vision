function visualize_predictions(num_of_op, test_face_dir, test_back_dir, predictions, num_test_face)
   
    num_test_face = 199;
    
    random_nums = randperm(200);
    rand_set = random_nums(1 : num_of_op);
    
    for f = rand_set
        images = dir(test_back_dir);
        im = imread(strcat(test_back_dir, images(f).name));
        figure; imshow(im)
        title(sprintf('%f', predictions(num_test_face + f)))
    end    
    
    random_nums = randperm(200);    
    rand_set = random_nums(1 : num_of_op);
    
    for f = rand_set
        images = dir(test_face_dir);
        im = imread(strcat(test_face_dir, images(f).name));
        figure; imshow(im)
        title(sprintf('%f', predictions(f)))
    end


end