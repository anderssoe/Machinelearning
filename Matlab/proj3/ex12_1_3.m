cdir = fileparts(mfilename('fullpath')); 
d = csvread(fullfile(cdir,'../Data/courses.txt'));
M = max(d(:));
N = size(d,1);
X = zeros(N,M);
disp(d)
% Binary encode the dataset
for i=1:N
    X(i,d(i, d(i,:) >0) ) = 1;
end
disp(X);
