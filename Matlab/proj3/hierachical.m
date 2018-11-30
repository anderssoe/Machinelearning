% exercise 10.2.1

%% Hierarchical clustering

% Maximum number of clusters
Maxclust = 11;

% Compute hierarchical clustering
Z = linkage(X, 'single', 'euclidean');

% Compute clustering by thresholding the dendrogram
i = cluster(Z, 'Maxclust', Maxclust);

%% Plot results

% Plot dendrogram
mfig('Dendrogram'); clf;
dendrogram(Z,Maxclust);

% Plot data
mfig('Hierarchical'); clf; 
clusterplot(X, y, i);
