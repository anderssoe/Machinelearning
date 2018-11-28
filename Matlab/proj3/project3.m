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


%% ex11_1_5 = GMM








%% Anomaly detection
%    Rank all the observations in terms of the Gaussian Kernel density (using leaveone-out), KNN density, KNN average relative density (ARD). 
%(If the scale of each attribute in your data are very di?erent it may turn useful to normalize the data prior to the analysis).
%2. Discuss whether it seems there may be outliers in your data according to the three scoring methods.

% ex11_4_1 = KNN density,
%% GKN using leave-one-out
widths=max(var(X))*(2.^[-10:2]); % evaluate for a range of kernel widths
for w=1:length(widths)
   [density,log_density]=gausKernelDensity(X,widths(w));
   logP(w)=sum(log_density);
end
[val,ind]=max(logP);
width=widths(ind);
display(['Optimal kernel width is ' num2str(width)])
% evaluate density for estimated width
density=gausKernelDensity(X,width);

% Sort the densities
[y,i] = sort(density);

% Plot outlier scores
mfig('Gaussian Kernel Density: outlier score'); clf;
bar(y(1:990));

% Plot possible outliers
mfig('Gaussian Kernel Density: Possible outliers'); clf;
for k = 1:990
    subplot(4,5,k);
    imagesc(reshape(X(i(k),:), 16, 16)); 
    title(k);
    colormap(1-gray); 
    axis image off;
end


%% K-nearest neighbor density estimator - *** Missing crossvalidation

% Number of neighbors
K = 5;

% Find the k nearest neighbors
[idx, D] = knnsearch(X, X, 'K', K+1);

% Compute the density
density = 1./(sum(D(:,2:end),2)/K);

% Sort the densities
[y,i] = sort(density);

% Plot outlier scores
mfig('KNN density: outlier score'); clf;
bar(y(1:20));

% Plot possible outliers
mfig('KNN density: Possible outliers'); clf;
for k = 1:20
    subplot(4,5,k);
    imagesc(reshape(X(i(k),:), 16, 16)); 
    title(k);
    colormap(1-gray); 
    axis image off;
end

%% K-nearest neigbor average relative density
% Compute the average relative density
avg_rel_density=density./(sum(density(idx(:,2:end)),2)/K);

% Sort the densities
[y,i] = sort(avg_rel_density);

% Plot outlier scores
mfig('KNN average relative density: outlier score'); clf;
bar(y(1:20));

% Plot possible outliers
mfig('KNN average relative density: Possible outliers'); clf;
for k = 1:20
    subplot(4,5,k);
    imagesc(reshape(X(i(k),:), 16, 16)); 
    title(k);
    colormap(1-gray); 
    axis image off;
end



%% %% ASSOCIATION MINING
%binarize data
d = X;
M = ceil(max(d(:)));
N = size(d,1);
X_ass = zeros(N,M);
disp(d);
% Binary encode the dataset
for i=1:N
    X_ass(i,d(i, d(i,:) >0) ) = 1;
end
disp(X_ass);



