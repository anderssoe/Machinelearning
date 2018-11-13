%% K-nearest

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
load('CV1_2.mat')
K2 = 10;


% Initialize variables
Features = nan(K,M);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_fs = nan(K,1);
Error_test_fs = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
% exercise 7.1.2

% Load data
%ex4_1_1



% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
% Leave-one-out crossvalidation
%CV = cvpartition(length(X_TRAIN), 'Leaveout');
%KK = CV.NumTestSets;
% Variable for classification error
%Error = nan(KK,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
    
    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    BestError = inf;
    
    
    for kk = 1:K2 % For each number of neighbors
        
        inner_X_train = X_train(CV2.training(kk), :);
        inner_y_train = y_train(CV2.training(kk));
        inner_X_test = X_train(CV2.test(kk), :);
        inner_y_test = y_train(CV2.test(kk));
        
        % Use knnclassify to find the l nearest neighbors
        % old code:
        % y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        % new code:
        
        for Ms = 1:10
            
            knn = fitcknn(inner_X_train, inner_y_train, 'NumNeighbors', Ms, 'Distance', Distance);
            
            y_test_est = predict(knn, inner_X_test);
            
            % Compute number of classification errors
            Error_val = sum(inner_y_test ~= y_test_est) / length(inner_y_test); % Count the number of errors
            if  Error_val <= BestError
                BestError = Error_val;
                BestModel = Ms;
            end
        end
    end
    
    fprintf('Best model for K-fold %d = %d \n',k, BestModel);
    
    knn = fitcknn(X_train, y_train, 'NumNeighbors', BestModel, 'Distance', Distance);
    
    y_test_est = predict(knn, X_test);
    
    % Array of right (hits)
    Error_test(k)  = sum(y_test ~= y_test_est) / length(y_test); % Count the number of errors
    
    
end
fprintf('K-Nearest Generalization Error = %d \n',mean(Error_test));



 k = 4; 

 X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

for Ms = 1:10
    knn = fitcknn(X_train, y_train, 'NumNeighbors', Ms, 'Distance', Distance);
    
    y_test_est = predict(knn, X_test);
    
    % Array of right (hits)
    Error_N(Ms)  = sum(y_test ~= y_test_est) / length(y_test); % Count the number of errors
    
    
end

NumNeighbours = 1:10;
% Plot the classification error rate
figure('Name', 'Error rate for number of neighbours');

plot(NumNeighbours, Error_N, 'LineWidth', 2);
title('Error rate for number of neighbors')
xlabel('Number of neighbors');
ylabel('Test error');
xticks(1:10)
xlim([0.5 10.5])

set(gca, 'FontSize', 16)
saveas(gcf, 'Plots/Project2/KNearestErrorNumNeighbours', 'epsc')
