%% exercise 2.3.1
%% Load data
cdir = fileparts(mfilename('fullpath')); 
load(fullfile(cdir,'../Data/zipdata.mat'));

% Extract digits (training set)
X = traindata(:,2:end);
y = traindata(:,1);

% Extract digits (test set)
Xtest = testdata(:,2:end);
ytest = testdata(:,1);

% Subtract the mean from the data
Y = bsxfun(@minus, X, mean(X));
Ytest = bsxfun(@minus, Xtest, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y, 'econ');

% Number of principal components to use, i.e. the reduced dimensionality
Krange = [8,10,15,20,30,40,50,60,100,150];
errorRate = zeros(length(Krange),1);
for i = 1:length(Krange)
    K=Krange(i);
    % Compute the projection onto the principal components
    Z = Y*V(:,1:K);
    Ztest = Ytest*V(:,1:K);

    % Classify digits using a K-nearest neighbour classifier
    model=fitcknn(Z,y,'NumNeighbors',1);
    yest = predict(model,Ztest);
    
    errorRate(i) = nnz(ytest~=yest)/length(ytest);

    % Display results
    fprintf('Error rate %.1f%%\n',errorRate(i)*100);
end

%% Visualize error rates vs. number of principal components considered
mfig('Variance explained by principal components'); clf;
plot(Krange,errorRate, 'o-');
xlabel('Number of principal components K')
ylabel('Error rate [%]')
