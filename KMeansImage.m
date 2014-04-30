function [centres, flag] = KMeansImage(filename, k, feature_length, normalise, MAX_Iter, MAX_Error)

% SYNOPSIS:
% This function runs K-Means Clustering on RGB images.
% INPUTS:
% filename       - name of input image
% k              - number of clusters (4,8,16,32)
% feature_length - feature vector length (3 or 5)
% normalise      - boolean option to normalise feature vectors
% MAX_Iter       - maximum number of iterations
% MAX_Error      - maximum error threshold
%
% OUTPUTS:
% centres        - final cluster centres
% flag           - return: 1 (hits below MAX_Error), 2 (hits MAX_Iter)
%
% AUTHOR: Sarah Li - Version 1.0, 13-05-2011

% Read in image and store dimensions
f = imread(filename); 
f = double(f);
[row, column, rgb_dim] = size(f);

% Rearrange image into a matrix of size feature length by number of pixels
f_rearrange = zeros(2 + rgb_dim, row * column);
point = 1;
for i = 1:row,
    for j = 1:column,
        % feature vector = [ R; G; B; X; Y] 
        f_rearrange(:,point) = [f(i,j,1); f(i,j,2); f(i,j,3); j; i];
        point = point + 1;
    end
end

% Call main K-Means M-file
[centres, output, clusters, flag] = KMeansCluster(k, f_rearrange, feature_length, normalise, MAX_Iter, MAX_Error);
    
% Assign cluster centre colours to corresponding pixels in the new image
g = zeros(size(f));
for i = 1:size(output,1)
    g(output(i,5), output(i,4), 1) = centres(clusters(i), 1); % R
    g(output(i,5), output(i,4), 2) = centres(clusters(i), 2); % G
    g(output(i,5), output(i,4), 3) = centres(clusters(i), 3); % B
end

% Normalise new image in order to use IMSHOW
for i = 1:3,
    g(:,:,i) = g(:,:,i) / max(max(g(:,:,i)));
end

% Display new image
figure; 
imshow(g);
if feature_length == 3,
    title(['K-Means Clustering on RGB Colour Coordinates K=', num2str(k)]);
elseif feature_length == 5,
    title(['K-Means Clustering on RGB Colour and Pixel Coordinates K=', num2str(k)]);
end

end



function [C_true, A, clusters, flag] = KMeansCluster(k, A, feature_length, normalise, MAX_Iter, MAX_Error)

% SYNOPSIS:
% This function runs K-Means Clustering on rearranged RGB images.
%
% INPUTS:
% k              - number of clusters (4,8,16,32)
% A              - input rearranged image
% feature_length - feature vector length (3 or 5)
% normalise      - boolean option to normalise feature vectors
% MAX_Iter       - maximum number of iterations
% MAX_Error      - maximum error threshold
%
% OUTPUTS:
% C_true         - final cluster centres
% A              - an error check; same as input A
% clusters       - cluster group assignments of instances
% flag           - return: 1 (hits below MAX_Error), 2 (hits MAX_Iter)
%
% AUTHOR: Sarah Li - Version 1.0, 13-05-2011

% Matrix of size number of pixels by feature length
A = A'; [row, column] = size(A);  

% Choice of Normalisation
if normalise == true,
    % Gaussian normalisation Z = (X - mean) / s.t.d.
    meanIm = repmat(mean(A,1), row, 1);
    stdIm = repmat(std(A,1,1), row, 1);
    B = (A - meanIm)./ stdIm;
else
    B = A;
end

% Assign random initial normalised cluster centres
rand_row = randperm(row);
C = zeros(k, feature_length);
for num_k = 1:k,
    C(num_k,:) = B(rand_row(num_k),1:feature_length);
end

% Initialise other important vectors
C_true = zeros(k,feature_length);  % True cluster centres
C_prev = zeros(size(C));           % Previous normalised cluster centres
d = zeros(row,k);                  % Distances between instances to cluster centres
b = zeros(k,row);                  % Binary Cluster Group Storage

% K-Means Algorithm
error = MAX_error;
iter = 1;
while iter <= MAX_Iter,   
    if error >= MAX_Error,
    
        % Update previous cluster centres and cluster group storage
        C_prev = C;
        b = zeros(k, row);
        
        % Calculate euclidean distance from instance to all cluster centres
        for num_k = 1:k,
            rep_k = repmat(C(num_k,:),row,1);
            delta = B(:,1:feature_length) - rep_k;   
            d(:,num_k) = sum(delta.^2, 2);
        end     
        
        % Store the cluster centre linked to the minimum distance to instance
        [min_value, min_index] = min(d, [], 2);     
        for i = 1:k, 
        
            % Find instances belonging to cluster group k
            cluster_group = find(min_index == i);
            
            % Store instances of cluster group k in b
            b(i, cluster_group') = 1;
            
            % Update kth normalised cluster centre
            C(i, :) = (b(i,:) * B(:,1:feature_length)) / sum(b(i,:));
            
            % Update kth true cluster centre
            C_true(i,:) = (b(i,:) * A(:,1:feature_length)) / sum(b(i,:));
        end     
        
        % Calculate euclidean distance between new and previous centres.
        % As algorithm converges, error (euclidean distance) decreases.
        % Error is close to 0 when algorthim converges because
        % cluster centre locations barely change.
        error = sum(sum((C - C_prev).^2,2));
        iter = iter + 1;
    else
        % ERROR hits below MAX_ERROR
        flag = 1;
        break;
    end
end

% ITER reaches MAX_ITER
if iter >= MAX_Iter, 
    flag = 2;
end

% Cluster group assignments of instances in their vector order
clusters = min_index(:);

end
