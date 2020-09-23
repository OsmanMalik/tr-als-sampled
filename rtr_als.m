function cores = rtr_als(X, ranks, embedding_dims, varargin)
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
for n = 1:N
    Xn = classical_mode_unfolding(X, n);
    J = embedding_dims(n);
    M = randn(prod(sz)/sz(n), J);
    [U,~] = qr(Xn*M,0);
    Q{n} = U.';
end
P = double(ttm(tensor(X), Q));
cores = tr_als(P, ranks, varargin{:});
for n = 1:N
    cores{n} = double(ttm(tensor(cores{n}), Q{n}.', 2));
end

end