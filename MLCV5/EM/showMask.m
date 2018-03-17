clear all; clc; close all;

fileList = dir('*.jpg');

for iFile = 1:length(fileList);
    im = imread(fileList(iFile).name);
    load([fileList(iFile).name '_Labels.mat'],'labels');
    figure; imshow([im repmat(labels*120,[1 1 3])]);
    
    
end