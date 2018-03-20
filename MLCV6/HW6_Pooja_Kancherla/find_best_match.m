function match = find_best_match(slice_test, slice_train)

distances = [];
for i = 1 : size(slice_test, 1)
    for j = 1 : size(slice_train, 1)
        distance = norm(slice_test(i) - slice_train(j));
        distances = [distances; distance];
    end    
end

match = min(distances);