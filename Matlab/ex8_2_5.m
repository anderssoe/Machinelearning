% exercise 8.2.5
% Last updated 17 nov. 2015
p = path();
cdir = fileparts(mfilename('fullpath')); 
rmpath(fullfile(cdir,'/Tools/nc_multiclass'));
addpath(fullfile(cdir,'/Tools/nc_binclass'));
% Load data
%load(fullfile(cdir,'../Data/wine2'))

% predict white vs. red wine type. 

% K-fold crossvalidation
K = 990;
CV = cvpartition(N,'Kfold', K);

% Parameters for neural network classifier
NHiddenUnits = 10;  % Number of hidden units
NTrain = 5; % Number of re-trains of neural network

% Variable for classification error
Error_train = nan(K,1);
Error_test = nan(K,1);
bestnet=cell(K,1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
    
    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    % Fit neural network to training set
    MSEBest = inf;
    for t = 1:NTrain
        netwrk = nc_main(X_train, y_train, X_test, y_test, NHiddenUnits);
        if netwrk.Etrain(end)<MSEBest, 
            bestnet{k} = netwrk; 
        %    MSEBest=netwrk.Etrain(end); 
            MSEBest=netwrk.Etrain(end); 
        end
    end
    
    % Predict model on test and training data    
    y_train_est = bestnet{k}.t_est_train;    
    y_test_est = bestnet{k}.t_est_test;        
    
    % Compute number of committed errors
    Error_train(k) = sum((y_train~=y_train_est));
    Error_test(k) = sum((y_test~=y_test_est));            
end

% Print the least squares errors
%% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error rate:  %.1f%%\n', sum(Error_train)/sum(CV.TrainSize)*100);
fprintf('- Test error rate:      %.1f%%\n', sum(Error_test)/sum(CV.TestSize)*100);

path(p); % reset path.
% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold
displayNetworkClassification(bestnet{k});

% Display the decition boundary (use only for two class classification problems)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkClassification(X_train, y_train, X_test, y_test, bestnet{k});
end