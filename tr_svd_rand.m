function cores = tr_svd_rand(X, ranks, oversamp)
%tr_svd_rand Implementation of randomized TR-SVD in [Ah20]
%
%cores = tr_svd_rand(X, ranks, oversamp) computes a tensor ring (TR)
%decomposition of the input N-dimensional array X. Ranks is a length-N
%vector containing the target ranks. oversamp is the oversampling used in
%the randomized range finder steps. Overall, the algorithm is essentially a
%variant of the standard TR-SVD algorithm, but incorporating randomized
%range finding instead of the standard SVD to make computations cheaper.
%The algorithm itself follows the pseudocode in Alg. 7 of [Ah20].
%
%REFERENCES
%
%[Ah20] Salman Ahmadi-Asl et al 2020 Mach. Learn.: Sci. Technol. in press
%       https://doi.org/10.1088/2632-2153/abad87 

%% Initialize cores

sz = size(X);
N = length(sz);
cores = cell(N, 1);

%% Main algorithm
% Roughly, capital letters correspond to matrices and the addition "_ten"
% means it's a tensor, which is used when there's an under bar in the
% algorithm in [Ah20].

C = reshape(X, sz(1), prod(sz(2:N)));
Z = C * randn(prod(sz(2:N)), ranks(N)*ranks(1) + oversamp);
%Q = qr(Z, 0);
[Q,~,~] = svd(Z, 'econ');
Q = Q(:, 1:ranks(N)*ranks(1));
cores{1} = permute(reshape(Q, [sz(1), ranks(N), ranks(1)]), [2 1 3]);
C_ten = reshape(Q.' * C, [size(Q,2) sz(2:N)]);
C_ten = reshape(C_ten, [ranks(N), ranks(1), prod(sz(2:N))]);
C_ten = permute(C_ten, [2 3 1]);
C_ten = reshape(C_ten, [ranks(1)*sz(2), prod(sz(3:N)), ranks(N)]);

for n = 2:N-1
    C = reshape(C_ten, [ranks(n-1)*sz(n), prod(sz(n+1:N))*ranks(N)]);
    Z = C * randn(prod(sz(n+1:N))*ranks(N), ranks(n) + oversamp);
    %Q = qr(Z, 0);
    [Q,~,~] = svd(Z, 'econ');
    Q = Q(:, 1:ranks(n));
    cores{n} = reshape(Q, [ranks(n-1), sz(n), ranks(n)]);
    C_ten = reshape(Q.' * C, [ranks(n) prod(sz(n+1:N)) ranks(N)]);
end

cores{N} = C_ten;

end