function f = cost (method, var, X, w, var_prior)
    if strcmp(method, 'BLR')
        I = size(X,2);
        covariance = var_prior*(X'*X) + (sqrt(var)^2)*eye(I);        
    else        
        I = size(X,1);
        covariance = var_prior*X*X + var*eye(I);
    end
    f = mvnpdf (w, zeros(I,1), covariance);
    f = -log(f);
end