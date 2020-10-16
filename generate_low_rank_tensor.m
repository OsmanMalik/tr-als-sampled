function [X, cores] = generate_low_rank_tensor(sz, ranks, noise, varargin)
%generate_low_rank_tensor Generate dense (TR) low-rank tensor
%
%X = generate_low_rank_tensor(sz, ranks, noise) returns a full dense tensor 
%of size specified in sz and TR-ranks specified in ranks. Moreover,
%Gaussian noise with standard deviation specified in noise is added.
%
%X = generate_low_rank_tensor(___, 'large_elem', val) is an optional input
%that can be used to randomly draw an entry in each TR-core and set it to
%val. In our paper, we do this with a value of val=20.
%
%[X, cores] = generate_low_rank_tensor(___) additionally returns the true
%TR-cores in cores.

% Handle optional inputs
params = inputParser;
addParameter(params, 'large_elem', 0);
parse(params, varargin{:});
large_elem = params.Results.large_elem;

% Construct tensor
N = length(sz);
cores = cell(1,N);
for n = 1:N
    R0 = ranks(mod(n-2, N)+1);
    R1 = ranks(n);
    cores{n} = randn(R0, sz(n), R1);
    if large_elem > 0
        r0 = randsample(R0, 1);
        i = randsample(sz(n), 1);
        r1 = randsample(R1, 1);
        cores{n}(r0,i,r1) = large_elem;
    end
end

X = cores_2_tensor(cores);
X = X + noise*randn(sz);

end
