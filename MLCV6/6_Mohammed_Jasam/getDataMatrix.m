function [dataMatrix, nrows, ncols, np, contents]  = getDataMatrix(Dataset, colorSpace, NumClasses)

% Get a list of all files and folders in this folder.
files = dir(Dataset);

% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];

% Extract only those that are directories.
subFolders = files(dirFlags);

allImages = [];
contents = [];
j = 1;
% Print folder names to command window.
for k = 3 : length(subFolders)
    
    % Name of the subfolder
    sub = subFolders(k).name;
    
    % Creating the subfolder directory path
    dirString = strcat(Dataset, sub, '\');
    
    % Extracting the image count in each sub folder
    NumImagesInFolder = dir([dirString '*.jpg']);    
    
    
    % Creating the contents of each subfolder
    for i = 1 : size(NumImagesInFolder,1)
        contents = [contents; [str2num(sub) i j]];
        j = j + 1;
    end
        
    % Pulling the images
    [Images,nrows,ncols,np] = getAllIms(dirString,colorSpace);
    
    if isempty(Images), continue; end
    
    % Appending the rows in allImages
    allImages = [allImages; Images];  
    
end

dataMatrix = allImages';


    