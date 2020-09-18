function X = generate_low_rank_tensor(sz, ranks, noise)
%generate_low_rank_tensor Generate dense (TR) low-rank tensor

N = length(sz);
cores = cell(1,N);
for n = 1:N
    R0 = ranks(mod(n-2, N)+1);
    R1 = ranks(n);
    cores{n} = randn(R0, sz(n), R1);
    cores{n}(1,1,1) = 10;
end

X = cores_2_tensor(cores);
X = X + noise*randn(sz);

end