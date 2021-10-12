function X = cores_2_tensor(cores, varargin)
%cores_2_tensor Convertes TR tensor given by cores to full dense tensor
%
%X = cores_2_tensor(cores) takes a cell array containing TR cores and
%outputs the corresponding full dense tensor. The input TR cores should be
%3-way standard Matlab arrays. The output will also be a standard Matlab
%array.

% Optional parameters
params = inputParser;
addParameter(params, 'permute_for_speed', false);
parse(params, varargin{:});
permute_for_speed = params.Results.permute_for_speed;

N = length(cores);
sz = cellfun(@(x) size(x,2), cores);

if permute_for_speed
    [~, max_idx] = max(sz);
    perm_vec = mod((max_idx+1 : max_idx+N) - 1, N) + 1;
    inv_perm_vec(perm_vec) = 1:N;
    cores = cores(perm_vec);
    sz = sz(perm_vec);
end

if isa(cores{1}, 'double') 
    G = subchain_matrix(cores, N);
    X = G * classical_mode_unfolding(cores{N}, 2).';
    X = reshape(X, sz(:)');
elseif isa(cores{1}, 'sptensor')
    
end

if permute_for_speed
    X = permute(X, inv_perm_vec);
end

end
