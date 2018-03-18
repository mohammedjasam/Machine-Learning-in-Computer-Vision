function [min_distance, index] = calc_distance(sim_measure, K, num_train_images, num_test_images, train_alpha, test_alpha) 
    
    if strcmpi(sim_measure, 'euclidean') == 1
        a = train_alpha;
        b = test_alpha;
        aa=sum(a.*a,1); bb=sum(b.*b,1); ab=a'*b; 
        distance_matrix = sqrt(abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab));
        [min_distance, index] = min(distance_matrix);
    end
    
    if strcmpi(sim_measure, 'manhattan') == 1
        a = train_alpha;
        b = test_alpha;
        for i = 1:173
            for j = 1:186
                distance_matrix(i,j) = sum(abs(a(:,i) - b(:,j)));
            end
        end
        [min_distance, index] = min(distance_matrix);
        
    end
    
    if strcmpi(sim_measure, 'mahalanobis') == 1
        C = [train_alpha, test_alpha];
        invcov = inv(cov(C'));

        for i=1:num_train_images
            diff = repmat(train_alpha(:,i), 1, num_test_images) - test_alpha;
            dsq(i,:) = sum((invcov*diff).*diff , 1);
        end

        distance_matrix = sqrt(dsq);
        [min_distance, index] = min(distance_matrix);
    end
    
    