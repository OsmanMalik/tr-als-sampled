function X = cores_2_tensor(cores)
%cores_2_tensor Convertes TR tensor given by cores to full dense tensor

N = length(cores);
sz = zeros(1,N);
for n = 1:N
    sz(n) = size(cores{n},2);
end
if isa(cores{1}, 'double') 
    G = subchain_matrix(cores, N);
    X = G * classical_mode_unfolding(cores{N}, 2).';
    X = reshape(X, sz);
elseif isa(cores{1}, 'sptensor')
    
end

end