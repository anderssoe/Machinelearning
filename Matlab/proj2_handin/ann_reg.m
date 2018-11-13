% exercise 8.2.6
% Last updated 17 nov. 2015
% Load data
cdir = fileparts(mfilename('fullpath')); 


% Parameters for neural network classifier
NHiddenUnits = [1 5 10 20 50];  % Number of hidden units
NTrain = 2; % Number of re-trains of neural network

besthidden = [];
E_test = [];

absolute_net=cell(K,1);
absolute_inner_test = [];

%% OUTER
for kk = 1:K

    X_train = X(CV.training(kk),:);
    y_train = y(CV.training(kk));
    X_test = X(CV.test(kk),:);
    y_test = y(CV.test(kk));
    
    fprintf('## Outer cross-fold %d / %d \n',kk,K);
    
    
    bestnet=cell(10,1);
    %% INNER
     %K-fold crossvalidation
    CV2 = cvpartition(CV.TrainSize(kk),'Kfold', 10); 
    for k = 1:10 % For each crossvalidation fold
        % Extract training and test set
        X_inner_train = X(CV2.training(k), :);
        y_inner_train = y(CV2.training(k));
        X_inner_test = X(CV2.test(k), :);
        y_inner_test = y(CV2.test(k));
       
       for s = 1:length(NHiddenUnits) % Test each number of hidden units
        % Fit neural network to inner training set 
              MSEBest = inf;
            for t = 1:NTrain  %retrain network
                netwrk = nr_main(X_inner_train, y_inner_train, X_inner_test, y_inner_test, NHiddenUnits(s));
                if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
            end    

            % Predict model on test and training data    
            y_train_est = bestnet{k}.t_pred_train;    
            y_test_est = bestnet{k}.t_pred_test;        

            % Compute least squares error
            inner_Error_train(k) = sum((y_inner_train-y_train_est).^2);
            inner_Error_test(k) = sum((y_inner_test-y_test_est).^2); 

            if inner_Error_test(k) == min(inner_Error_test)
              absolute_net(kk) = bestnet(k);
              absolute_inner_test(kk) = inner_Error_test(k);
              besthidden(kk) = NHiddenUnits(s)
            end

        
%         % Compute least squares error when predicting output to be mean of
%         % training data
%         Error_train_nofeatures(k) = sum((y_inner_train-mean(y_inner_train)).^2);
%         Error_test_nofeatures(k) = sum((y_inner_test-mean(y_inner_train)).^2);  
       end
    end % end inner

    inner_Error_test = [];
    netwrk = nr_main(X_train, y_train, X_test, y_test, besthidden(kk)); %Retrain network
    % Predict model on test and training data    
    y_out_train_est = netwrk.t_pred_train;    
    y_out_test_est = netwrk.t_pred_test;        

    % Compute least squares error
    Error_train(kk) = sum((y_train-y_out_train_est).^2);
    Error_test(kk) = sum((y_test-y_out_test_est).^2); 

    % Compute least squares error when predicting output to be mean of
    % training data    
    Error_train_nofeatures(kk) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(kk) = sum((y_test-mean(y_train)).^2); 

end % end outer
%% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));

fprintf('- Best number of hidden units per fold: %d,%d,%d,%d,%d \n', besthidden(1:s)); %***
fprintf('- E_test for each fold: %d,%d,%d,%d,%d \n',Error_test(1:kk));

fprintf('-- gen error = %d \n',sum(Error_test)/K);
% Display the trained network 
%mfig('Trained Network');
(Error_test == min(Error_test)) % cross-validation fold
displayNetworkRegression(absolute_net{3});

% Display how network predicts (only for when there are two attributes)
if size(X_inner_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkRegression(X_inner_train, y_train, X_inner_test, y_inner_test, bestnet{k});
end