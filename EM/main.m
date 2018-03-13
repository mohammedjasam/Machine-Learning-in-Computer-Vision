clc;
clear all;
close all;

testImageList = zeros(5, 256);
testImageList = {'ButlerCarlton2.jpg'; 'FultonHall.jpg'; 'HavenerCenter2.jpg'; 'McNutt.jpg'; 'RollaBuilding.jpg'};

Scene_pixel_classification(testImageList);