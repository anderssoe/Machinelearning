ex12_1_3; % load data
minSup = .8; % minimum support
minConf = 1; % minmum confidence
nRules = 100; % Max rules
sortFlag = 1; % sorting of found rules (see doc)
[Rules, ~] = findRules(X, minSup, minConf, nRules, sortFlag);
% Inspect rules and note output is of the form Rules{1}(i) -> Rules{2}{i}
% We can also pretty-print the found rules using this function (see
% definition below)
disp('Rules found:');
print_apriori_rules(Rules)

labels = {'02322', '02450', '02453', '02454', '02457', '02459', '02582'};
disp('The same rules but with labels:')
print_apriori_rules(Rules,labels)

