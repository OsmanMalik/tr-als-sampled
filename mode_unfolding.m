function Xn = mode_unfolding(X, n)
%mode_unfolding Does mode-n unfolding of input tensor X
%
%Xn = mode_unfolding(X, n) computes the mode-n unfolding of X, where X can 
%be either a multiway array or an sptensor. In the former case, Xn will be
%returned as a regular matrix, and in the latter case it will be returned
%as a sparse matrix.

sz = size(X);
N = length(sz);
perm_vec = [n n+1:N 1:n-1];
Xn = reshape(permute(X, perm_vec), [sz(n), prod(size(X))/sz(n)]);

if isa(Xn, 'sptensor')
    Xn = sptensor_mat_2_sparse(Xn);
end

end
