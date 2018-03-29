%Transform into non-linear vector

function[Z] = zTransform(d, D, N, X) 

    Z = zeros(d*D+1, N);
    %transform X to Z
    for j = 1:N
        for k=1:d
            for i = (k-1)*D+2:k*D+1
                Z(i,j) = X(i-(k-1)*D-1,j)^(k);
            end
        end
    end

    Z(1,:) = ones(1,N);
end