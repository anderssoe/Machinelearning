%% ex12_1_5
cdir = fileparts(mfilename('fullpath')); 
load(fullfile(cdir,'../Data/wine2'));
% Try changing the 2 to a 3 and notice what happens to the output (see
% attributeNamesBin)
[Xbin,attributeNamesBin] = binarize(X, 2*ones(1,M), attributeNames);