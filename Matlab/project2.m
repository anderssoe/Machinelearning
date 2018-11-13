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




 %% REGRESSION
 

 
 %% Use ex_6_2_1.m for linear regression with forward selection
 run('ex6_2_1');
 run('LinearRegressionPlot.m')
 %% ANN
 %run('ann_reg');
 
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





 
%% CLASSIFICATION
% Try to classify the different vowels

X = [X_TRAIN;X_TEST];
y = [y_TRAIN;y_TEST];
attributeNames = cellstr(["x.1", "x.2",	"x.3",	"x.4",	"x.5",	"x.6",	"x.7",	"x.8",	"x.9",	"x.10"]);

%% Decision tree - missing 2layer

run('DecisionTree.m')

%% K-nearest

run('K_Nearest.m')


%% ANN - Missing 2layer
edit('ex8_3_1');

