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
    [h_Rand(ii), h_Jaccard(ii), h_NMI(ii)] = clusterval(y, i);
    
    if ii > 1 && (h_Jaccard(ii) >= h_Jaccard(ii-1))
        Z_best = Z;
    end
end




figure('Name', 'Hierarchical clustering validity', 'Position', [500 500 800 500])
H = bar([h_Rand' , h_Jaccard', h_NMI'])
legend({'Rand','Jaccard','NMI'}, 'Location', 'best');
set(gca, 'FontSize', 16)
title('Hierarchical clustering validity', 'FontSize', 20)

xticklabels(LinkageType);
xtickangle(45);

saveas(gcf, 'Plots/Hierichical_validity', 'epsc')
%xticks()
%% Plot results

% Plot dendrogram
figure('Name', 'Dendrogram of best hierarchical clustering', 'Position', [500 500 800 500]);
plt = dendrogram(Z_best,Maxclust);
set(plt, 'LineWidth', 2)
set(gca, 'FontSize', 16)
title('Dendogram of best hierarchical clustering', 'FontSize', 20)

saveas(gcf, 'Plots/Hierichical_dendogram', 'epsc')

% Plot data
i = cluster(Z_best, 'Maxclust', Maxclust) - 1;

figure('Name', 'Best hierarchical clustering', 'Position', [500 500 1100 700]); clf;
clusterplot(X, y, i);
set(gca, 'FontSize', 16)
title('Best hierarchical clustering', 'FontSize', 20)

saveas(gcf, 'Plots/Hierichical_clustering', 'epsc')
