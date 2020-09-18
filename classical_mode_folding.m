function X = classical_mode_folding(Xn, n, sz)
%classical_mode_folding The inverse of classical_mode_unfolding
%
%X = classical_mode_folding(Xn, n, sz) returns a tensor X of size sz. The
%function is the inverse of classical_mode_unfolding.

N = length(sz);
perm_vec = [2:n 1 n+1:N];
perm_vec_sz = [n 1:n-1 n+1:N];
X = permute(reshape(Xn, sz(perm_vec_sz)), perm_vec);

end