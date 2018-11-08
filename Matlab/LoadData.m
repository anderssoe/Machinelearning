clear all
close all
clc

% Load the data into Matlab
NUMERIC = csvread('Data/vowel_test.csv');

% Extract the rows and columzzzns corresponding to the sensor data, and
% transpose the matrix to have rows correspond to data items
X = NUMERIC(:,3:12);

% Extract attribute names from the first column
attributeNames = cellstr(["x.1", "x.2",	"x.3",	"x.4",	"x.5",	"x.6",	"x.7",	"x.8",	"x.9",	"x.10"]);

% Extract unique class names from the first row
classLabels = cellstr(num2str(NUMERIC(1:end,2)));
classNames = unique(classLabels);

% Extract class labels that match the class names
[y_,y] = ismember(classLabels, classNames); y = y-1;


%% Histogram with norma fit
% for i = 1:10
% figure(10+i)
% h = histfit(X(:,i),100);
% title(['Histogram of attribute ' int2str(i) ' with normal distribution'])
% hold on
% %plot(frequency_axis, mag_12(1:fft_length/2), 'linewidth', width2)
% set(h(2),'linewidth', 2.5);
% legend({['Distribution of attribute ' int2str(i)], 'Normal distribution fit'}, 'FontSize', 12 )
% set(gca,'fontsize',15)
% saveas(gcf,['./Plots/normaldist' int2str(i)],'epsc')
% end
%% Calculate correlation of attributes.
R = corrcoef(X);
R = round(R,2);
csvwrite('Correlation',R)


%% Subtract the mean from the data
Y = bsxfun(@minus, X, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y);

% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);

% Plot variance explained
figure(21)
plot(rho, 'o-', 'LineWidth', 4);
title('Variance explained by principal components');
set(gca,'fontsize',15)
xlabel('Principal component');
ylabel('Variance explained value');
saveas(gcf,'./Plots/varianceexplained','epsc')


%% Compute the projection onto the principal components
Z = U*S;

% Plot PCA of data
figure('Position',[0 0 1250 900])
C = length(classNames);
hold all;
for c = 0:C-1
    plot(Z(y==c,1), Z(y==c,2), 'o','linewidth',3);
end

%legend(classNames);
%legend('x.1','x.2','x.3','x.4','x.5','x.6','x.7','x.8','x.9','x.10');
legend('i','I','E','A','a:','Y','O','C:','U','u:', '3:');
xlabel(sprintf('PCA %d', 1));
ylabel(sprintf('PCA %d', 2));
title('PCA of Vowel data');
set(gca,'fontsize',15)
saveas(gcf,'./Plots/PCA1vsPCA2','epsc')

%% Compute pca, vowel vs output

y = y+1;
figure('Position',[0 0 1250 1000])
subplot(2,1,1)
plot(Z(:,1),y,'o','linewidth',2);
xlabel('PCA 1');
ylabel('Vowel');
title('PCA 1 vs Output vowel');
ylim([0 12]);
yticks([1:1:11]);
subplot(2,1,2)
plot(Z(:,2),y,'o','linewidth',2);
xlabel('PCA 2');
ylabel('Vowel');
title('PCA 2 vs Output vowel');
ylim([0 12]);
yticks([1:1:11]);
saveas(gcf,'./Plots/PCA1PCA2vsOutput','epsc')
%% boxplot
figure(34)
boxplot(X)
xlabel('Attribute no.');
ylabel('Value');
title('Boxplot of attributes');
set(gca,'fontsize',15)
saveas(gcf,'./Plots/boxplot','epsc')

%% statistics

Std_div = nanstd(X)
Skewness = skewness(X)
meanX = mean(X)
