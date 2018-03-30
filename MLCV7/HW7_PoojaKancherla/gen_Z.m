function Z = gen_Z(d, X) 
    % Dimensions of X
    rows = size(X, 1);
    cols = size(X, 2);  
    
    Z = [];
    for pow = 1 : d
        % Empty matrix X
        temp = zeros(rows, cols);
        for i = 1 : rows
            for j = 1 : cols
                temp(i, j) = X(i, j) ^ pow;
            end
        end
        Z = [Z; temp];
    end
    ones_row = ones(1, size(Z, 2));
    Z = [ones_row; Z];

end