
K_fold = [1 2 3 4 5];

figure('Name', 'Squared error for each fold')
h = bar(Error_test_fs);
xlabel('K-fold number')
ylabel('Squared Error')
