function Xn = classical_mode_unfolding(X, n)
%classical_mode_unfolding Does classical mode-n unfolding of input tensor X

N = max(ndims(X), n);
szn = size(X,n);
perm_vec = [n 1:n-1 n+1:N];
X_perm = permute(X, perm_vec);
Xn = reshape(X_perm, [szn, prod(size(X))/szn]);

if isa(Xn, 'sptensor')
    Xn = sptensor_mat_2_sparse(Xn);
end

end
