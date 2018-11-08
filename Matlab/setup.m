
function setup()
disp('Resetting path...');
restoredefaultpath();
disp('Setting up the 02450 Toolbox...');
cdir = pwd();
path = fileparts(mfilename('fullpath'));
% Setup path
if nargin < 1, nntoolbox_type = 'mul'; end
if strcmp(nntoolbox_type,'mul'), 
    nn_bin = false;
elseif strcmp(nntoolbox_type,'bin'),
    nn_bin = true;
else
    disp('Wrong type of neural network toolbox selected')
    nntoolbox_type = false;
    asser(false);
end

if strcmp(path(1),'/') % linux system
    addpath([path '/Tools/TMG_6.0R7'])
    addpath([path '/Tools/MBox'])   
    if ~nn_bin, 
        addpath([path '/Tools/nr'])
        addpath([path '/Tools/nc_multiclass'])
    else
        addpath([path '/Tools/nc_binclass'])        
    end
    addpath([path '/Tools/Apriori'])    
    addpath([path '/Tools/02450Tools'])
    addpath([path '/Scripts'])
else % Windows system   
    addpath([path '\Tools\TMG_6.0R7'])
    addpath([path '\Tools\MBox'])    
    if ~nn_bin, 
        addpath([path '\Tools\nr'])
        addpath([path '\Tools\nc_multiclass'])
    else
        addpath([path '\Tools\nc_binclass'])        
    end    
    addpath([path '\Tools\Apriori'])
    addpath([path '\Tools\02450Tools'])
    addpath([path '\Scripts'])
end
clear path;

% Chech version number
v = ver('MATLAB');
verVec = sscanf(v.Version, '%d.%d.%d');
verNum = sum(verVec.*logspace(0, 1-length(verVec), length(verVec))');
fprintf('Running Matlab version %s\n',v.Version);
if verNum<7.7
	fprintf('WARNING: We recommend using Matlab version 7.7 or newer!\n');
end

% Disable glmfit warning
warning('off', 'stats:glmfit:IllConditioned'); 
warning('off', 'stats:glmfit:IterationLimit');
% Disable xlsread warning
warning('off', 'MATLAB:xlsread:ActiveX'); 
warning('off', 'MATLAB:xlsread:Mode'); 
% Disable NaiveBayes warning
warning('off', 'stats:NaiveBayes:BadDataforMVMN');
warning('off', 'stats:NaiveBayes:BadDataforMN');
% Disable neural network warining
warning('off', 'MATLAB:nearlySingularMatrix');
% Disable GMM warning
warning('off', 'stats:gmdistribution:IllCondCov');
warning('off', 'stats:gmdistribution:FailedToConverge');

% Set execute permission to apriori
if isunix, system('chmod +x Tools/Apriori/apriori'); end;

% Done
disp('Setup of the 02450 Toolbox completed.');
cd(cdir);
rehash path

end