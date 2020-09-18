function Y = sptensor_mat_2_sparse(X)
%sptensor_mat_2_sparse Convert 2D sptensors to sparse matrices

Y = sparse(X.subs(:,1), X.subs(:,2), X.vals, size(X,1), size(X,2));

end