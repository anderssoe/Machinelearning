%% exercise 2.1.5
disp('2nd principal component')
V(:,2)
% Projection of water class onto the 2nd principal component.
Y(y==4,:) * V(:,2)
% E.i. large in V(:,2) determine the important features of Y and their
% weighting. 