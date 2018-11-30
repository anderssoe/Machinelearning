% exercise 10.2.1

%% Hierarchical clustering

LinkageType = {'average', 'centroid', 'complete', 'single', 'ward', 'median', 'weighted'};
for ii = 1:7
    
    
    % Maximum number of clusters
    Maxclust = bestK; %11;
    
    % Compute hierarchical clustering
    Z = linkage(X, LinkageType(ii), 'euclidean');
    
    % Compute clustering by thresholding the dendrogram
    i = cluster(Z, 'Maxclust', Maxclust) - 1;
    
    % Estimate the cluster validity
    [Rand(ii), Jaccard(ii), NMI(ii)] = clusterval(y, i);
    
    
end

figure('Name', 'Hierarchical clustering validity')
H = bar([Rand' , Jaccard', NMI'])
legend('Rand','Jaccard','NMI');
xticklabels(LinkageType);
xtickangle(45);

%xticks()
%% Plot results
% Plot dendrogram
mfig('Dendrogram'); clf;
dendrogram(Z,Maxclust);

% Plot data
mfig('Hierarchical'); clf;
clusterplot(X, y, i);
