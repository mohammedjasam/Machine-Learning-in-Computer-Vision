x = 10*rand(100,1); %x \in [0,10]
w = 2 + 3 * x + 0.01*rand(100,1); %n ~ norm[0,1];

X = [ones(1,100); x'];
% phi = inv(X*X')*(X*w)
phi = (X*X')\(X*w)

figure; plot(x,w,'r+'); axis equal;