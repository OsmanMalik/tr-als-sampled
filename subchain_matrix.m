function subchain_unfolding = subchain_matrix(cores, n)
%subchain_matrix Compute mode-2 unfolding of subchain excluding n-th core
%
%The output is the mode-2 unfolding of the subchain consisting of all cores
%except the n-th. 

N = length(cores);
idx = [n+1:N 1:n-1];
for j = 1:length(idx)
    if j == 1
        M = classical_mode_unfolding(cores{idx(j)}, 3).';
    else
        M = M * classical_mode_unfolding(cores{idx(j)}, 1);
        R = size(cores{idx(j)}, 3);
        M = reshape(M, numel(M)/R, R);
    end
end

[R0, ~, R1] = size(cores{n});
M = reshape(M, R1, numel(M)/(R0*R1), R0);
subchain_unfolding = mode_unfolding(M, 2);

end