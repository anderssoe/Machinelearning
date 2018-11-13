
K_fold = [1 2 3 4 5];

figure('Name', 'Squared error for each fold')
h = bar(Error_test_fs);
xlabel('K-fold number', 'FontSize', 14)
ylabel('Squared Error', 'FontSize', 14)
title('Squared error for each fold', 'FontSize', 20)
saveas(gcf, 'Plots/Project2/LGerrorPerFold', 'epsc')
