% Compute SNR
% Formula is SNR_db = 20*log_10(|| Signal(:) ||_2 / || Epsilon(:) ||_2)

I = 100:100:500; % Tensor sizes
N = 3; % Tensor dimensionality
ranks = 10*ones(1, N); % Target ranks
extra_rank = 0; % Additional rank added
noise = 1e+1; % Amount of Gaussian noise added to each entry
large_elem = 20;

SNR = zeros(1,5);

for m = 1:length(I)
    sz = I(m)*ones(1, N);   
    X = generate_low_rank_tensor(sz, ranks + extra_rank, 0, 'large_elem', large_elem);
    E = noise*randn(sz);
    
    SNR(m) = 20*log10(norm(X(:)) / norm(E(:)));
end

SNR

