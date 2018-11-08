% exercise 6.3.1

% Load the wine data set
cdir = fileparts(mfilename('fullpath')); 
load(fullfile(cdir,'../Data/wine2'))

%% Crossvalidation

% Create 10-fold crossvalidation partition for evaluation
K = 10;
CV = cvpartition(N, 'Kfold', K);

% Initialize variables
Error_LogReg = nan(1,K);
Error_DecTree = nan(1,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :); 
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :); 
    y_test = y(CV.test(k));

    % Logistic regression 
    w_est = glmfit(X_train, y_train, 'binomial');
    y_est = glmval(w_est, X_test, 'logit');
    Error_LogReg(k) = sum(y_test~=(y_est>.5));

    % Decision tree with pruning
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', 'splitcriterion', 'gdi', ...
        'categorical', []);
    Error_DecTree(k) = sum(~strcmp(classNames(y_test+1), eval(T, X_test)));
    
end

%% Determine if classifiers are significantly different.
% The function ttest computes credibility interval 
mfig('Error rates');
boxplot([Error_LogReg./CV.TestSize; Error_DecTree./CV.TestSize]'*100, ...
    'labels', {'Logistic regression', 'Decision tree'});
ylabel('Error rate (%)');
%%
% test if the classifiers are significantly different by computing the
% credibility interval using the methods of section 9.3.3
% The following can also be accomplished by calling ttest(Error_LogReg,
% Error_DecTree). 
nu = K-1; 
z = Error_LogReg - Error_DecTree; 
zb = mean(z);
sig = sqrt( mean( (z-zb).^2) / (K-1));
alpha = 0.05; 
[zLH] = zb + sig * tinv([alpha/2, 1-alpha/2], nu);
%%
if zLH(1) < 0 && zLH(2) > 0, 
    disp('Classifiers are NOT significantly different');
else
    disp('Classifiers are significantly different');    
end