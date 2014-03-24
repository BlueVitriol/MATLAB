
function m = kmeans(data, k)

% SYNOPSIS:
% K-means clustering aims to partition n observations into k clusters 
% in which each observation belongs to the cluster with the nearest mean, 
% serving as a prototype of the cluster.
%
% REFERENCE: en.wikipedia.org/wiki/K-means_clustering
%
% AUTHOR: Sarah Li - Version 1.0, 19-04-2011

[row, column] = size(data);

% Randomly choose K Cluster Centres
rand_row = randperm(row);
m = zeros(k, column);
for num_k = 1:k,
    m(num_k,:) = data(rand_row(num_k),:);
end
m_prev = zeros(k, column);

% Plot Data and Initial K Cluster Centres
figure;
plot(data(:,1), data(:,2), 'gx');
hold on;
plot(m(:,1), m(:,2),'b*');
title(['Plot of Data and Initial K=', num2str(k), ' Cluster Centres']);

% K Means
while m ~= m_prev, 
    % Update Previous Cluster Centres
    m_prev = m;
    b = zeros(k, row);
    for i = 1:row
        d = zeros(k,1);
        for num_k = 1:k,
            % Compute Distance from Data Point to Cluster Centres
            d(num_k) = pdist( [data(i, :); m(num_k, :)] );
        end
        % Update b Matrix
        [min_value, min_index] = min(d);
        b(min_index, i) = 1;
    end
    % Update New Cluster Centres
    for j = 1:k,
        m(j, :) = (b(j,:) * data) / sum(b(j,:));
    end
end

% Colour and Marker Combinations
all_colours = ['r', 'g', 'y', 'b', 'c', 'm', 'r', 'g', 'y', 'b', 'c', 'm'];
all_markers = ['+', 'x', 'o', '^', 'v', '<', '>', '+', 'x', 'o', '^', 'v'];

% Plot Data and Converged K Cluster Centres
figure; hold on;
for num_k = 1:k,
    signal_index = find(~isnan(b(num_k,:) ./ 0)); 
    cluster = data(signal_index, :);
    plot(cluster(:, 1), cluster(:, 2), strcat(all_colours(num_k), all_markers(num_k)));
end
plot(m(:,1), m(:,2),'k*');
title(['Plot of Data and Converged K=', num2str(k), ' Cluster Centres']);

end
