% exercise 8.2.2
% Last updated 17 nov. 2015
% Load data
p = path();
cdir = fileparts(mfilename('fullpath')); 
rmpath(fullfile(cdir,'/Tools/nc_multiclass'));
addpath(fullfile(cdir,'/Tools/nc_binclass'));

%load(fullfile(cdir,'../Data/xor'))
 
% K-fold crossvalidation
K = 10; 
CV = cvpartition(y, 'Kfold', K);

% Parameters for neural network classifier
NHiddenUnits = 50;  % Number of hidden units
NTrain = 10; % Number of re-trains of neural network

% Variable for classification error
Error = nan(K,1);
bestnet = cell(K,1); 

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
        if netwrk.Etrain(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.Etrain(end); end
    end
    
    % Predict model on test data    
    y_test_est = bestnet{k}.t_est_test>.5;    
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end


% Print the error rate
fprintf('Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);

% Display the trained network 
mfig('Trained Network');
clf;
k=1; % cross-validation fold
displayNetworkClassification(bestnet{k});


% Display the decision boundary (use only for two class classification problems)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkClassification(X_train, y_train, X_test, y_test, bestnet{k});
end

path(p); %reset path.