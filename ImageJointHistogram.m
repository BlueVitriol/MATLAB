function [jointHistogram, normJointHistogram] = ImageJointHistogram(image1, image2, numBins)

% SYNOPSIS:
% Given two images (2D or 3D) of the same size with values from 0 to 255 and the number of bins,
% this function returns both the joint histogram and normalised joint histogram of the images.
%
% INPUTS:
% image1 - [2|3D matrix] image matrix 1
% image2 - [2|3D matrix] image matrix 2 (same size as image1)
% numBins - [+ve integer] number of bins desired
%
% OUTPUTS:
% jointHistogram - [2D matrix] joint histogram
% normJointHistogram - [2D matrix] normalised joint histogram
%
% AUTHOR: Sarah Li - Version 1.0, 16-08-2011

% Size of bins
sizeBins = (255 + 1 - 0) / numBins;
offset = (sizeBins - 1)/2;

% Centres of bins
binCentres = (1:numBins) * sizeBins - offset;

% Initialise joint histogram
jointHistogram = zeros(numBins,numBins);

% Vectorise image matrices for speed
array1 = image1(:);
array2 = image2(:);

for i = 1:length(array1),
    % Find the correspoding bin count index for both images
    ind1 = find(hist(array1(i), binCentres) == 1);
    ind2 = find(hist(array2(i), binCentres) == 1);
    
    % Increment the location in the Joint Histogram
    jointHistogram(ind1,ind2) = jointHistogram(ind1,ind2) + 1;
end

% Normalised joint histogram
normJointHistogram = jointHistogram / sum( jointHistogram(:) );

% Verify by plotting Graph
figure; imagesc(normJointHistogram); axis(’square’); colorbar;
title([’Normalised Joint Histogram of two Images’, 10, ’with values ranging between 0 to 255 using ’, ...
    num2str(numBins), ’ bins of size ’, num2str(sizeBins)]);
ylabel(’Image 1 Bins’); xlabel(’Image 2 Bins’);

end
