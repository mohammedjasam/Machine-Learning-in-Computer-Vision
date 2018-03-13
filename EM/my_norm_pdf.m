
% Description: Calculates PDF
% Input: x       - each row is one datapoint.
%        K       - number of Gaussians in the mixture.
%        precision - the algorithm stops when the difference between
%                    the previous and the new likelihood is < precision.
%                    Typically this is a small number like 0.01.
% Output:
%        lambda  - lambda(k) is the weight for the k-th Gaussian.
%        mu      - mu(k,:) is the mean for the k-th Gaussian.
%        sig     - sig{k} is the covariance matrix for the k-th Gaussian.
function [pdf] = my_norm_pdf (x, mu, sigma)
    D = size(sigma);
   
    %x
    %size(x)
    pdf = mvnpdf(x, mu, sigma);
    return;
    %diff = (x-mu);
    diff = bsxfun(@minus,x,mu);
    
    power = -0.5 * ( (diff)' * inv(sigma) * (diff) );
  % power = -0.5 * (diff' / sigma) * diff;
    denominator = sqrt( ((2 * pi)^D)*(det(sigma)) );
    pdf = (1/denominator) * exp(power);
    
end