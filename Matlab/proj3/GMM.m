% exercise 11.1.5

%% Gaussian mixture model

% Range of K's to try
KRange = 1:30;
T = length(KRange);

% Allocate variables
BIC = nan(T,1);
AIC = nan(T,1);
CVE = zeros(T,1);
Gbest = [];
bestK = [];
% For each model order
for t = 1:T
    % Get the current K
    K = KRange(t);
    
    % Display information
    fprintf('Fitting model for K=%d\n', K);
    
    % Fit model
    G = gmdistribution.fit(X, K, 'Replicates', 10);
    
    % Get BIC and AIC
    BIC(t) = G.BIC;
    AIC(t) = G.AIC;
    
    
    % For each crossvalidation fold
    for k = 1:CV.NumTestSets
        % Extract the training and test set
        X_train = X(CV.training(k), :);
        X_test = X(CV.test(k), :);
        
        % Fit model to training set
        G = gmdistribution.fit(X_train, K, 'Replicates', 10);
        
        % Evaluation crossvalidation error
        [~, NLOGL] = posterior(G, X_test);
        CVE(t) = CVE(t)+NLOGL;
    end
    
    % Save the best fitting model
    if CVE(t) <= min(CVE(1:t))
        Gbest = G;
        bestK = K;
    end
end


%% Plot results

figure('Name', 'GMM: Number of clusters')

hold all
plot(KRange, BIC, 'LineWidth', 2);
plot(KRange, AIC, 'LineWidth', 2);
plot(KRange, 2*CVE, 'LineWidth', 2);

ylabel('Error');
xlabel('K');
set(gca, 'FontSize', 12)
title('GMM: Number of clusters', 'FontSize', 16)
legend('BIC', 'AIC', 'Crossvalidation');%, 'FontSize', 12)

saveas(gcf, 'Plots/GMM_NumberOfClusters', 'epsc')



%% Extract cluster centers
X_c = Gbest.mu;
Sigma_c=Gbest.Sigma;
i = cluster(Gbest, X)-1;
%% Plot results

% Plot clustering
figure('Name', 'GMM: Clustering', 'Position', [500 500 1100 800])

clusterplot(X, y, i, X_c, Sigma_c);

set(gca, 'FontSize', 18)

title('GMM: Clustering', 'FontSize', 24)

saveas(gcf, 'Plots/GMM_Clustering', 'epsc')




fprintf('Best K value according to Xval: %d \n',bestK');

[gmm_Rand, gmm_Jaccard, gmm_NMI] = clusterval(y, i)
