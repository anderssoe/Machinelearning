%% exercise 2.1.4

% Index of the principal components
i = 1;
j = 2;

% Compute the projection onto the principal components
Z = U*S;

% Plot PCA of data
mfig('NanoNose: PCA'); clf; hold all; 
C = length(classNames);
for c = 1:C
    plot(Z(y==c,i), Z(y==c,j), 'o');
end
legend(classNames);
xlabel(sprintf('PC %d', i));
ylabel(sprintf('PC %d', j));
title('PCA of NanoNose data');
