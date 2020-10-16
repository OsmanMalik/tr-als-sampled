function U = col(A)
%col Compute matrix whos columns form orthonormal basis for range of A

R = rank(A);
[U, ~, ~] = svd(A, 'econ');
U = U(:,1:R);

end
