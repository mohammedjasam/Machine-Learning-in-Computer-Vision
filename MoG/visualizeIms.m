function visualizeIms(allIms, nrows, ncols, np, nImPerLine, titleStr)
%allIms: each row is an image (nrows*ncols*np)

showIm = zeros(nrows*nImPerLine, ncols*nImPerLine, np);
for ii = 1:size(allIms,1)
    im = reshape(allIms(ii,:),[nrows, ncols, np]);    
    [iRow, iCol] = ind2sub([nImPerLine, nImPerLine],ii);
    showIm((iRow-1)*nrows+1:iRow*nrows, (iCol-1)*ncols+1:iCol*ncols, :) = im;
end
figure; imagesc(showIm); title(titleStr);
