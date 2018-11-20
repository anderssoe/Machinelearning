% exercise 10.1.3

% Load data
cdir = fileparts(mfilename('fullpath')); 
load(fullfile(cdir,'../Data/synth1'));

%% K-means clustering

% Maximum number of clusters
K = 10;

% Allocate variables
Rand = nan(K,1);
Jaccard = nan(K,1);
NMI = nan(K,1);

for k = 1:K    
    % Run k-means
    [i, Xc] = kmeans(X, k);
    
    % Compute cluster validities
    [Rand(k), Jaccard(k), NMI(k)] = clusterval(y, i);
end
%% Plot results

mfig('Cluster validity'); clf; hold all;
plot(1:K, Rand);
plot(1:K, Jaccard);
plot(1:K, NMI);

legend({'Rand', 'Jaccard','NMI'});