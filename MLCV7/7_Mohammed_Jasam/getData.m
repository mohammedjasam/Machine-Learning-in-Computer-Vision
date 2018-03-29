function [X, W] = getData(Path)

Files = dir([Path '*.jpg']);

X = [];
W = [];

for ii = 1:size(Files,1)
    Filename = Files(ii).name;
    W = [W; str2double(Filename(1:4))];
    Im = imread([Path Filename]);
    Im = Im(:,:,1);
    X = [X Im(:)];
end

X = double(X);
X = [ones(1, size(X, 2)); X];