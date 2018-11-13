% exercise 8.3.1
%clear all;
% Load data
cdir = fileparts(mfilename('fullpath')); 
%load(fullfile(cdir,'../Data/synth1'))
%load(fullfile(cdir,'../Data/xor'))
addpath(fullfile(cdir,'/Tools/nc_binclass'));
rmpath(fullfile(cdir,'/Tools/nc_binclass'));
addpath(fullfile(cdir,'/Tools/nc_multiclass'));

% Parameters for neural network classifier
NHiddenUnits = [1 3 5];  % Number of hidden units
NTrain = 1;
N_test = length(X_TEST);
KK = 10; % inner folds amount
besthidden = [];
%% OUTER
for k = 1:K
    X_train = X(CV.training(k),:);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k),:);
    y_test = y(CV.test(k));
    
    fprintf('# Outer cross-fold %d / %d \n',k,K);
    
    CV2 = cvpartition(CV.TrainSize(k),'Kfold', 10);
    inner_error_test = [];
    best_inner_error = [];
    bestnet = cell(10,1);
    inner_bestnet = cell(10,1);
% Inner

    for kk = 1:KK % For each crossvalidation fold
        fprintf('## Inner cross-fold %d / %d \n',kk,KK);
        % Extract training and test set
        X_inner_train = X(CV2.training(kk), :);
        y_inner_train = y(CV2.training(kk));
        X_inner_test = X(CV2.test(kk), :);
        y_inner_test = y(CV2.test(kk));
        
        for s = 1:length(NHiddenUnits) %test each number of hidden units
        fprintf('### Hidden neurons: %d / %d \n',NHiddenUnits(s),max(NHiddenUnits));
            CBest = inf;
            for t = 1:NTrain %Retrain network
                % Fit multiclass neural network to training set
                net = nc_main(X_inner_train, y_inner_train+1, X_inner_test, y_inner_test+1, NHiddenUnits(s));
                if net.Ctrain(end)<CBest, bestnet{kk} = net;CBest=net.Ctrain(end);end
            end
            % Compute results on inner data
            % Get the predicted output for the test data
            Y_inner_train_est = nc_eval(net,X_inner_train);
            Y_inner_test_est = nc_eval(net, X_inner_test);       

            % Compute the class index by finding the class with highest probability from the neural
            % network
            y_inner_train_est = max_idx(Y_inner_train_est);
            y_inner_test_est = max_idx(Y_inner_test_est);
            % Subtract one to have y_test_est between 0 and C-1
            y_inner_train_est = y_inner_train_est-1;
            y_inner_test_est = y_inner_test_est-1;

            inner_error_test(kk) = sum(y_inner_test~=y_inner_test_est)/length(y(CV2.test(kk)));
            if(inner_error_test(kk) <= min(inner_error_test))
                inner_bestnet = bestnet{kk};
                best_inner_error(k) = inner_error_test(kk);
                besthidden = NHiddenUnits(s);
                fprintf('besthidden(%d): %d \n',kk,besthidden);
            end
        end %end find model
    end %end inner fold
    
    outer_net(k) = nc_main(X_train, y_train+1, X_test, y_test+1, NHiddenUnits(s)); %retrain on new training data with optimized units
    % Get the predicted output for the test data
    Y_train_est = nc_eval(outer_net(k),X_train);
    Y_test_est = nc_eval(outer_net(k), X_test);      
    
    % Compute the class index by finding the class with highest probability from the neural
    % network
    y_train_est = max_idx(Y_train_est);
    y_test_est = max_idx(Y_test_est);
    % Subtract one to have y_test_est between 0 and C-1
    y_train_est = y_train_est-1;
    y_test_est = y_test_est-1;  
    ErrorRate(k) = sum(y_test~=y_test_est)/length(y_test);
    Train_ErrorRate(k) = sum(y_train~=y_train_est)/length(y_test);
fprintf('Test error rates: %.0f%%\n', ErrorRate(k)*100);
end %end outer





 %% Plot results
% % Display trained network
% mfig('Trained network'); clf;   
% displayNetworkClassification(net)
%  
% % Display decision boundaries
% mfig('Decision Boundaries'); clf;   
% dbplot(X_TEST, y_TEST, @(X) max_idx(nc_eval(net, X))-1);
% xlabel(attributeNames(1));
% ylabel(attributeNames(2));

