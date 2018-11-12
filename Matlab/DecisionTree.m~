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

X = [X_TRAIN;X_TEST];
y = [y_TRAIN;y_TEST];
attributeNames = cellstr(["x.1", "x.2",	"x.3",	"x.4",	"x.5",	"x.6",	"x.7",	"x.8",	"x.9",	"x.10"]);

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

%% Decision tree - missing 2layer

% exercise 5.1.6
minparent = [20 40 60 80 100]; % Minimum number of observations per branch before stopping

% Fit classification tree

% Outer loop
for k = 1:5
    
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    for kk = 1:10
        
        inner_X_train = X_train(CV2.training(kk), :);
        inner_y_train = y_train(CV2.training(kk));
        inner_X_test = X_train(CV2.test(kk), :);
        inner_y_test = y_train(CV2.test(kk));
        
        
        for Ms = 1:5
            
            
        
            T = fitctree(inner_X_train, classNames(y+1), ...
                'splitcriterion', 'gdi', ...
                'categorical', [], ...
                'PredictorNames', attributeNames, ...
                'prune', 'off', ... % What does prune do??
                'minparent', minparent(Ms));%, 'CrossVal','on');%, 'CrossVal','on');
            % View the tree
            view(T, 'mode','graph')
            title(sprintf('minparent %d',minparent(k)));
            cvmodel = crossval(T);
            L = kfoldLoss(cvmodel)
        end
    end
    
    
end