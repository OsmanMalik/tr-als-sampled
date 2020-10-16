function [X, cores] = generate_sparse_low_rank_tensor(sz, ranks, density, noise)
%generate_sparse_low_rank_tensor Generate sparse (TR) low-rank tensor
%
%X = generate_sparse_low_rank_tensor(sz, ranks, density, noise) generates a
%sparse tensor of size sz with TR-ranks ranks, with each core having a
%density of about the number specified in density. Moreover, Gaussian noise
%with standard deviation specified in noise is added to all nonzero
%entries.
%
%[X, cores] = generate_sparse_low_rank_tensor(___) also returns the
%TR-cores.

N = length(sz);
cores = cell(N,1);
for n = 1:N
    R0 = ranks(mod(n-2, N)+1);
    R1 = ranks(n);
    cores{n} = sptenrand([R0 sz(n) R1], density);
end

X = cores{n};
for n = 2:N-1
    X = ttt(X, cores{n}, 3, 1);
    X = reshape(X, [size(X,1), size(X,2)*size(X,3), size(X,4)]);
end
X = ttt(X, cores{N}, [1 3], [3 1]);
X = reshape(X, sz);
X(X.subs) = X(X.subs) + noise*randn(size(X.vals));

for n = 1:N
    cores{n} = double(tensor(cores{n}));
end

end
