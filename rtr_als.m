function [cores, varargout] = rtr_als(X, ranks, embedding_dims, varargin)
%rTRD Projection based randomized tensor ring decomposition
%
%cores = rTRD(X, ranks, embedding_dims, varargin) computes a tensor ring
%(TR) decomposition of the input N-dimensional array X. ranks is a length-N
%vector containing the target ranks. The output cores is a cell containing
%the N core tensors, each represented as a 3-way array. The embedding_dims
%vector controls how much each side of X is compressed.
%
%Optional arguments can be passed, and these will be passed on to tr_als
%which is used as the solver after compressing.
%
%This method was proposed by [Yu19].

sz = size(X);
N = length(sz);
Q = cell(1,N);
P = tensor(X);
for n = 1:N
    Xn = classical_mode_unfolding(X, n);
    K = embedding_dims(n);
    if K < sz(n) % Only compress if embedding dim is smaller than dim size
        M = randn(prod(sz)/sz(n), K);
        [U,~] = qr(Xn*M,0);
        Q{n} = U.';
        P = ttm(P, Q{n}, n);
    end
end
P = double(P);
if nargout == 1
    cores = tr_als(P, ranks, varargin{:});
else
    [cores, conv_vec] = tr_als(P, ranks, varargin{:});
    varargout{1} = conv_vec; 
end
for n = 1:N
    K = embedding_dims(n);
    if K < sz(n) % Only uncompress compressed dims
        cores{n} = double(ttm(tensor(cores{n}), Q{n}.', 2));
    end
end

end