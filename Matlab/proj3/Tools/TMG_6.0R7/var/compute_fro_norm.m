function s=compute_fro_norm(A, W, H)
% COMPUTE_FRO_NORM - returns the frobenius norm of a rank-l 
% matrix A - W * H
% 
% Copyright 2011 Dimitrios Zeimpekis, Eugenia Maria Kontopoulou, Efstratios Gallopoulos

error(nargchk(3, 3, nargin));
s=compute_fro_norm_p(A, W, H);