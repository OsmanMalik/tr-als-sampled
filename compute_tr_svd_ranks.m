% This script was used to compute the TR-SVD ranks in required to achieve a
% certain accuracy for TR-SVD; see Remark S9 in the supplement of our
% paper for details.

I = 100:100:500; % Tensor sizes
N = 3; % Tensor dimensionality
ranks = 10*ones(1, N); % Target ranks
noise = 1e-1; % Amount of Gaussian noise added to each entry
large_elem = 20;

extra_rank = 10; % Additional rank added
acc = .94;

TR_SVD_ranks = zeros(5,3);
for m = 1:length(I)
    sz = I(m)*ones(1, N);   
    X = generate_low_rank_tensor(sz, ranks + extra_rank, noise, 'large_elem', large_elem);
    
    cores = TRdecomp(X, acc);
    TR_SVD_ranks(m, 1) = size(cores{1},3);
    TR_SVD_ranks(m, 2) = size(cores{2},3);
    TR_SVD_ranks(m, 3) = size(cores{3},3);
    m
end

TR_SVD_ranks
