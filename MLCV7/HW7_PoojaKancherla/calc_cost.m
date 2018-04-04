function cost = calc_cost (variance, X_train_temp, num_train, w_train, var_prior)
    covariance = var_prior * (X_train_temp' * X_train_temp) + (sqrt(variance) ^ 2) * eye(num_train);
    cost = mvnpdf (w_train, zeros(num_train, 1), covariance);
    cost = -log(cost);
end