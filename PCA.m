
function [z, V, PC] = PCA(data, dim)

% SYNOPOSIS:
% Principal Component Analysis (PCA) is a statistical procedure that uses 
% orthogonal transformation to convert a set of observations of possibly 
% correlated variables into a set of values of linearly uncorrelated variables 
% called principal components.
%
% REFERENCE: en.wikipedia.org/wiki/Principal_component_analysis
%
% AUTHOR: Sarah Li - Version 1.0, 02-04-2011

[row, column] = size(data);
covariance_matrix = cov(data);

% Find the eigenvectors and eigenvalues
[PC, V] = eig(covariance_matrix);

% Extract diagonal of matrix as vector
V = diag(V);

% Sort the variances in decreasing order
[V, rindices] = sort(V,'descend');
PC = PC(:, rindices);

% Apply dimension
w = PC(:,1:dim);

% Project the original data set
m = mean(data, 1);
for i = 1:row
    data(i, :) = data(i, :) - m;
end
z = data * w;

end
