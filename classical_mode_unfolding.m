function Xn = classical_mode_unfolding(X, n)
%classical_mode_unfolding Does classical mode-n unfolding of input tensor X

sz = size(X);
N = length(sz);
perm_vec = [n 1:n-1 n+1:N];
X_perm = permute(X, perm_vec);
Xn = reshape(X_perm, [sz(n), prod(sz)/sz(n)]);

if isa(Xn, 'sptensor')
    Xn = sptensor_mat_2_sparse(Xn);
end

end
