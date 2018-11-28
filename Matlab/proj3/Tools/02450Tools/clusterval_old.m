function [Entropy, Purity, Rand, Jaccard] = clusterval(y, i)
% CLUSTERVAL Estimate cluster validity using entroy, purity, rand index,
% and Jaccard coefficient.
%
% Usage:
%   [Entropy, Purity, Rand, Jaccard] = clusterval(y, i);
%
% Input:
%    y         N-by-1 vector of class labels 
%    i         N-by-1 vector of cluster indices
%
% Output:
%   Entropy    Entropy measure.
%   Purity     Purity measure.
%   Rand       Rand index.
%   Jaccard    Jaccard coefficient.
%
% Copyright 2011, Mikkel N. Schmidt, Morten Mørup, Technical University of Denmark
% Updated 3 November 2015 to not count self pairs in Rand and Jaccard

N = length(y);
[d1_,d2_,jy] = unique(y);
Zy = sparse(jy, 1:N, ones(1,N));
Ay = Zy'*Zy;

[d1_,d2_,ji] = unique(i);
Zi = sparse(ji, 1:N, ones(1,N));
Ai = Zi'*Zi;

f11 = full(sum(sum(triu(Ai.*Ay,1))));
f00 = full(sum(sum(triu((1-Ai).*(1-Ay),1))));
mZi = sum(Zi,2);
P = diag(1./mZi)*Zi*Zy';
e = -sum(P.*log(P+eps),2);

Entropy = full(sum(e.*mZi)/sum(mZi));
Purity = full(sum(max(P,[],2).*mZi)/sum(mZi));
Rand = (f11+f00)/(N*(N-1)/2);
Jaccard = f11/(N*(N-1)/2-f00);



