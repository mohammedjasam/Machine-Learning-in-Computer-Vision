function [mu_test, var_test, var, A_inv] = compute (method, X, w, var_prior, X_test)
    D = size(X,1) - 1;
    I = size(X,2);
    I_test = size(X_test,2);
    
    if strcmp(method, 'BLR')
        % Compute the variance. Use the range [0,variance of world values].
        % Constrain var to be positive, by expressing it as var=sqrt(var)^2,
        % that is, the standard deviation squared.
        mu_world = sum(w) / I;
        var_world = sum((w - mu_world) .^ 2) / I;
        var = fminbnd (@(var) cost(method, var, X, w, var_prior), 0, var_world);

        % Compute A_inv.    
        A_inv = 0;    
        if D < I
            A_inv = inv ((X*X') ./ var + eye(D+1) ./ var_prior);
        else    
            A_inv = eye(D+1) - X*inv(X'*X + (var/var_prior)*eye(I))*X';
            A_inv = var_prior * A_inv;
        end

        % Compute the mean for each test example.
        temp = X_test' * A_inv;
        mu_test = (temp * X * w) ./ var;

        % Compute the variance for each test example.    
        var_test = repmat(var,I_test,1);
        for i = 1 : I_test
            var_test(i) = var_test(i) + temp(i,:) * X_test(:,i);
        end
    else
        % Compute K[X,X].    
        K = zeros(I,I);
        for i=1:I
            for j=1:I
                x1 = X(:,i);
                x2 = X(:,j);
                temp = ((1 + x1(:)'*x2(:))^2);
                K(i,j) = temp;%kernel(X(:,i), X(:,j));
            end
        end

        % Compute K[X_test,X].
        K_test = zeros(I_test, I);
        for i=1:I_test
            for j=1:I
                x1 = X_test(:,i);
                x2 = X(:,j);
                temp = ((1 + x1(:)'*x2(:))^2);
                K_test(i,j) = temp;%kernel(X_test(:,i), X(:,j));
            end
        end

        % Compute the variance. Use the range [0,variance of world values].
        % Constrain var to be positive, by expressing it as var=sqrt(var)^2,
        % that is, the standard deviation squared.
        mu_world = sum(w) / I;
        var_world = sum((w - mu_world) .^ 2) / I;
        var = fminbnd (@(var) cost (method, var, K, w, var_prior), 0, var_world);

        % Compute A_inv.
        A_inv = inv (K*K/var + eye(I)/var_prior);

        % Compute mu_test.
        temp = K_test * A_inv;    
        mu_test = (temp*K*w)/var;

        % Compute var_test.
        var_test = repmat(var,I_test,1);
        for i = 1 : I_test
            var_test(i) = var_test(i) + temp(i,:) * K_test(i,:)';
        end
    end
end