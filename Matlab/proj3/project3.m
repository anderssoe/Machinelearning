%project 3


%ex12_1_3: binary encoding
%% Initialize

clear all
close all
clc

% Load the train data into Matlab
TRAIN = csvread('Data/vowel_train.csv');
TEST = csvread('Data/vowel_test.csv');

% Extract the rows and columns corresponding to the sensor data, and
% transpose the matrix to have rows correspond to data items
X_TRAIN = TRAIN(:,3:12);
X_TEST = TEST(:,3:12);

% Extract attribute names from the first column
attributeNames = cellstr(["x.1", "x.2",	"x.3",	"x.4",	"x.5",	"x.6",	"x.7",	"x.8",	"x.9",	"x.10"]);

% FOR TRAINING
% Extract unique class names from the first row
classLabels_TRAIN = cellstr(num2str(TRAIN(1:end,2)));
classNames = unique(classLabels_TRAIN);
% Extract class labels that match the class names
[y__TRAIN,y_TRAIN] = ismember(classLabels_TRAIN, classNames); y_TRAIN = y_TRAIN-1;

% Create 1ofK outputs instead
% y_1ofk_TRAIN = zeros(length(y_TRAIN),11);
% for ii = 1:length(y_TRAIN)
%     for jj = 1:11
%         if y_TRAIN(ii,1) == jj-1
%             y_1ofk_TRAIN(ii,jj) = 1;
%         end
%     end
% end
% y_TRAIN = y_1ofk_TRAIN;

% FOR TESTING
classLabels_TEST = cellstr(num2str(TEST(1:end,2)));
% Extract class labels that match the class names
[y__TEST,y_TEST] = ismember(classLabels_TEST, classNames); y_TEST = y_TEST-1;


% MAKE Y 1OFK
% y_1ofk_TEST = zeros(length(y_TEST),11);
% for ii = 1:length(y_TEST)
%     for jj = 1:11
%         if y_TEST(ii,1) == jj-1
%             y_1ofk_TEST(ii,jj) = 1;
%         end
%     end
% end
% y_TEST = y_1ofk_TEST;

% Set classNames to correct vowels
%classNames = {'i';'I';'E';'A';'a:';'Y';'O';'C:';'U';'u:';'3:'};
% Create full data matrix
X = [X_TRAIN;X_TEST];
y = [y_TRAIN;y_TEST];


N = length(X); %number of observation
M = length(X(1,:)); %input variables
%% K-fold initialization

% Create crossvalidation partition for evaluation
K = 5;
CV = cvpartition(N, 'Kfold', K);

K2 = 10;
CV2 = cvpartition(CV.TrainSize(1), 'Kfold', K2);


% Initialize variables
Features = nan(K,M);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_fs = nan(K,1);
Error_test_fs = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);

%% Clustering

% ex_10_2_1 = Hierachical clustering
run('GMM.m');
%%
% Hierachical
run('hierachical.m');


title('Hierarchical clustering validity', 'FontSize', 20)

xticklabels(LinkageType);
xtickangle(45);

%% plot evaluation of GMM & hierical

[temp, BestHierachical] = max(h_Jaccard);

figure('Name', 'Clustering validity', 'Position', [500 500 800 500])
H = bar([h_Rand(BestHierachical), gmm_Rand;
    h_Jaccard(BestHierachical), gmm_Jaccard;
    h_NMI(BestHierachical), gmm_NMI]);
legend('Hierachical','GMM');
set(gca, 'FontSize', 16)

xticklabels({'Rand','Jaccard','NMI'});
xtickangle(45);
saveas(gcf, 'Plots/Eval_GMM_Hierachical', 'epsc')




%% Anomaly detection
%    Rank all the observations in terms of the Gaussian Kernel density (using leaveone-out), KNN density, KNN average relative density (ARD).
%(If the scale of each attribute in your data are very di?erent it may turn useful to normalize the data prior to the analysis).
%2. Discuss whether it seems there may be outliers in your data according to the three scoring methods.

% ex11_4_1 = KNN density,
%% GKN using leave-one-out
widths = max(var(X))*(2.^[-10:2]); % evaluate for a range of kernel widths
for w = 1:length(widths)
    [density,log_density] = gausKernelDensity(X,widths(w));
    logP(w) = sum(log_density);
end
[val,ind]=max(logP);
width = widths(ind);
display(['Optimal kernel width is ' num2str(width)])
% evaluate density for estimated width
density = gausKernelDensity(X,width);

% Sort the densities
[sorted_GKN_density,GKN_density_index] = sort(density);

% Plot outlier scores
figure('Name', 'KDE','Position',[0 0, 950,800])
bar(sorted_GKN_density(1:20));
ylabel('Density');
xticks([1:20]);
xticklabels(GKN_density_index(1:20));
xtickangle(45);
xlabel('Sorted KDE densities');
set(gca, 'FontSize', 12)
title(sprintf('KDE\n width = 0.021582'), 'FontSize', 16)

saveas(gcf, 'Plots/KDE', 'epsc')

%% K-nearest neighbor density estimator

% Number of neighbors
K = 5;

% Find the k nearest neighbors
[idx, D] = knnsearch(X, X, 'K', K+1);

% Compute the density
density = 1./(sum(D(:,2:end),2)/K);

% Sort the densities
[sorted_KNN_density, KNN_density_index] = sort(density);

% Plot outlier scores
figure('Name','KNN: Lowest 20 outlier score','Position',[0 0, 950,800]);
bar(sorted_KNN_density(1:20));
ylabel('Score');
xticks([1:20]);
xticklabels(KNN_density_index(1:20));
xlabel('Sorted KNN score index');
xtickangle(45);
set(gca, 'FontSize', 12)
title(sprintf('KNN: Lowest 20 outlier score\n K = %d',K), 'FontSize', 16)
saveas(gcf, 'Plots/KNN', 'epsc')

% AAAAAAAARD
avg_rel_density=density./(sum(density(idx(:,2:end)),2)/K);

% Sort the densities
[sorted_avg_rel_density,avg_rel_density_index] = sort(avg_rel_density);

% Plot outlier scores
%mfig('KNN average relative density: outlier score'); clf;
%bar(sorted_avg_rel_density(1:990));


figure('Name','ARD: Lowest 20 outlier score','Position',[0 0, 950,800]);
bar(sorted_avg_rel_density(1:20));
ylabel('Score');
xticks([1:20]);
xticklabels(avg_rel_density_index(1:20));
xtickangle(45);
xlabel('Sorted ARD score index');
set(gca, 'FontSize', 12)
title(sprintf('ARD: Lowest 20 outlier score\n K = %d',K), 'FontSize', 16)
saveas(gcf, 'Plots/ARD', 'epsc')
% Plot possible outliers
%mfig('KNN average relative density: Possible outliers'); clf;
%for k = 1:20
%    subplot(4,5,k);
%    imagesc(reshape(X(i(k),:), 16, 16));
%    title(k);
%    colormap(1-gray);
%    axis image off;
%end

%% Comparing outlier detection
Outlier_detection_table = [GKN_density_index(1:20)';
                            KNN_density_index(1:20)';
                            avg_rel_density_index(1:20)'];

% number of repetitions
for i = 1:990
    if sum(sum(Outlier_detection_table == i)) > 1
        fprintf('Index %d is repeated %d times\n', i, sum(sum(Outlier_detection_table == i)));
    end
  
end


%% %% ASSOCIATION MINING
%Include y as a potential association. But first, onehot encoding
y_onehot = zeros(N,max(y)+1);
for ii = 1:N
    y_onehot(ii,y(ii)+1) = 1;
end
%
%binarize data
M = size(X, 2);
N = size(X, 1);
%
[Xbin,attributeNamesBin] = binarize(X, 2*ones(1,M), attributeNames);

Xbin = [Xbin, y_onehot];
attributeNamesBin = [attributeNamesBin,cellstr(["y.1","y.2","y.3","y.4","y.5","y.6","y.7","y.8","y.9","y.10","y.11"])];
%%
% from ex12_1_6
minSup = .09; % minimum support
minConf = 0.95; % minmum confidence
nRules = 100; % Max rules
sortFlag = 1; % sorting of found rules (see doc)
[rules, frequentItemSets] = findRules(Xbin, minSup, minConf, nRules, sortFlag);
disp('Rules found:')
print_apriori_rules(rules,attributeNamesBin)

