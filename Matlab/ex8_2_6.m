% exercise 8.2.6
% Last updated 17 nov. 2015
% Load data
cdir = fileparts(mfilename('fullpath')); 


% Parameters for neural network classifier
NHiddenUnits = [1 2 3 4 5];  % Number of hidden units
NTrain = 5; % Number of re-trains of neural network

besthidden = zeros(5, 1);
E_test = [];
bestnet=cell(K,1);

%% OUTER
for kk = 1:K

    %% INNER
    % K-fold crossvalidation
    CV2 = cvpartition(CV.TrainSize(kk),'Kfold', 10); %*** made in main
    for k = 1:10 % For each crossvalidation fold
    %    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

        % Extract training and test set
        X_train = X(CV2.training(k), :);
        y_train = y(CV2.training(k));
        X_test = X(CV2.test(k), :);
        y_test = y(CV2.test(k));

        % Fit neural network to training set
        MSEBest = inf;
        for t = 1:NTrain
           % fprintf('Number of hidden units: %d \n',NHiddenUnits(t));% ***
            netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits(t));
            if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end);
                    fprintf('Current best number of hidden units: %d \n',NHiddenUnits(t)); % ***
                    besthidden(k) = NHiddenUnits(t);
            end
        end

        % Predict model on test and training data    
        y_train_est = bestnet{k}.t_pred_train;    
        y_test_est = bestnet{k}.t_pred_test;        

        % Compute least squares error
        Error_train(k) = sum((y_train-y_train_est).^2);
        Error_test(k) = sum((y_test-y_test_est).^2); 

        E_test(k) = mean(y_test-y_test_est);
         fprintf('E_test for fold %d = %d \n',k,E_test(k)); %***   

        % Compute least squares error when predicting output to be mean of
        % training data
        Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
        Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  

    end

end

% Print the least squares errors


%% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV2.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV2.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));

fprintf('- Best number of hidden units per fold: %d, %d, %d, %d, %d \n', besthidden(1:5)); %***
fprintf('- E_test for each fold: %d, %d, %d, %d, %d \n', Error_test_nofeatures(1:k));

% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold
displayNetworkRegression(bestnet{k});

% Display how network predicts (only for when there are two attributes)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkRegression(X_train, y_train, X_test, y_test, bestnet{k});
end