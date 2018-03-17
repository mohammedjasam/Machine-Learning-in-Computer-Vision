function [dataMatrix,fc_image_num]  = getData(dir, colorgamut)

face_matrix = [];
[face_matrix, NRows, NCols, Channels] = getAllIms(dir, colorgamut);
fc_image_num = size(face_matrix,1);
face_matrix=normalizeIm1((face_matrix'),fc_image_num);
dataMatrix=face_matrix';