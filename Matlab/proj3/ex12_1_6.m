%% ex12_1_6
ex12_1_5; % load data
minSup = .2; % minimum support
minConf = .6; % minmum confidence
nRules = 100; % Max rules
sortFlag = 1; % sorting of found rules (see doc)
[rules, ~] = findRules(Xbin, minSup, minConf, nRules, sortFlag);
disp('Rules found:')
print_apriori_rules(rules,attributeNamesBin)