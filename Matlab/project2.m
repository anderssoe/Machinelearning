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

y = X(:,2);
X(:,2) = [];
attributeNames = cellstr(["x.1", "x.3",	"x.4",	"x.5",	"x.6",	"x.7",	"x.8",  "x.9",	"x.10"]);


N = length(X); %number of observation
M = length(X(1,:)); %input variables

%% K-fold initialization

% Create crossvalidation partition for evaluation
K = 5;
CV = cvpartition(N, 'Kfold', K);

% Initialize variables
Features = nan(K,M);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_fs = nan(K,1);
Error_test_fs = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);




 %% REGRESSION
 

 
 %% Use ex_6_2_1.m for linear regression with forward selection
 run('ex6_2_1');
 run('LinearRegressionPlot.m')
 %% ANN
 run('ann_reg');
 
 %% AVG Output
 

 for zz = 1:5
    baseline(zz) = sum((y(CV.test(zz)) - mean(y(CV.training(zz)))).^2); %least squares baseline
 end
 baseline = baseline';

ANN_error_test = [19.8261959639922;19.7530728644934;17.1528333032427;18.8975654052411;19.0888622151111]';
Lin_error_test = [55.0487754387538
65.8627173707434
52.9219877263876
62.1475686904868
65.0792500989336]';

% Determine if classifiers are significantly different.
% The function ttest computes credibility interval 
mfig('Error rates');
boxplot([(ANN_error_test./CV.TestSize)*100; (Lin_error_test./CV.TestSize)*100; (baseline./CV.TestSize)*100]', ...
    'labels', {'ANN', 'LinReg','Baseline'});
ylabel('Error rate');


fprintf('ANN vs LinReg\n');
nu = K-1; 
z = ANN_error_test - Lin_error_test; 
zb = mean(z);
sig = sqrt( mean( (z-zb).^2) / (K-1));
alpha = 0.05; 
[zLH] = zb + sig * tinv([alpha/2, 1-alpha/2], nu)
if zLH(1) < 0 && zLH(2) > 0, 
    disp('Classifiers are NOT significantly different');
else
    disp('Classifiers are significantly different');    
end

fprintf('ANN vs baseline\n');
nu = K-1; 
z = ANN_error_test - baseline; 
zb = mean(z);
sig = sqrt( mean( (z-zb).^2) / (K-1));
alpha = 0.05; 
[zLH] = zb + sig * tinv([alpha/2, 1-alpha/2], nu)
if zLH(1) < 0 && zLH(2) > 0, 
    disp('Classifiers are NOT significantly different');
else
    disp('Classifiers are significantly different');    
end

fprintf('LinReg vs baseline\n');
nu = K-1; 
z = Lin_error_test - baseline; 
zb = mean(z);
sig = sqrt( mean( (z-zb).^2) / (K-1));
alpha = 0.05; 
[zLH] = zb + sig * tinv([alpha/2, 1-alpha/2], nu)
if zLH(1) < 0 && zLH(2) > 0, 
    disp('Classifiers are NOT significantly different');
else
    disp('Classifiers are significantly different');    
end




 return;

 
%% CLASSIFICATION
% Try to classify the different vowels

X = [X_TRAIN;X_TEST];
y = [y_TRAIN;y_TEST];
attributeNames = cellstr(["x.1", "x.2",	"x.3",	"x.4",	"x.5",	"x.6",	"x.7",	"x.8",	"x.9",	"x.10"]);

%% Decision tree - missing 2layer

% exercise 5.1.6
minparent = [20 40 60 80 100]; % Minimum number of observations per branch before stopping

% Fit classification tree


for k = 1:length(minparent);
    
 


    
    T = fitctree(X, classNames(y+1), ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'PredictorNames', attributeNames, ...
        'prune', 'off', ... % What does prune do??
        'minparent', minparent(k));%, 'CrossVal','on');%, 'CrossVal','on');
    % View the tree
    view(T, 'mode','graph')
    title(sprintf('minparent %d',minparent(k)));
    cvmodel = crossval(T);
    L = kfoldLoss(cvmodel)
end

%% K-nearest

% exercise 7.1.2

% Load data
%ex4_1_1



% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 20; % Maximum number of neighbors


% Leave-one-out crossvalidation
CV = cvpartition(length(X_TRAIN), 'Leaveout');
KK = CV.NumTestSets;
% Variable for classification error
Error = nan(KK,L);

for kk = 1:KK % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', kk, CV.NumTestSets);

    % Extract training and test set
    X_train2 = X_TRAIN(CV.training(kk), :);
    y_train2 = y_TRAIN(CV.training(kk));
    X_test2 = X_TRAIN(CV.test(kk), :);
    y_test2 = y_TRAIN(CV.test(kk));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        % old code:
        % y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        % new code:
        knn=fitcknn(X_train2, y_train2, 'NumNeighbors', l, 'Distance', Distance);
        y_test_est=predict(knn, X_test2);
        
        % Compute number of classification errors
        Error(kk,l) = sum(y_test2~=y_test_est); % Count the number of errors
    end
end

% Plot the classification error rate
mfig('Error rate');
plot(sum(Error)./sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');


%% ANN - Missing 2layer
edit('ex8_3_1');

