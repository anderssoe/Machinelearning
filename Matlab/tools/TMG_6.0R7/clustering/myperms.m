function v=myperms(p, l)
% MYPERMS - computes all possible combinations of the input
%   V=MYPERMS[P, L] returns all possible combinations of the 
%   input vector of integers with L numbers.
%
% Copyright 2011 Dimitrios Zeimpekis, Eugenia Maria Kontopoulou, Efstratios Gallopoulos

error(nargchk(2, 2, nargin));
v=myperms_p(p, l);