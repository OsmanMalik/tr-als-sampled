function cores = initialize_cores(sz, ranks)
%initialize_cores Initializes cores using Gaussian distribution
%
%cores = initialize_cores(sz, ranks) returns a length-N cell with N cores,
%each with entires drawn iid from the standard Gaussian distribution. sz is
%a length-N vector with the sizes, and ranks is a length-N vector with the
%outgoing ranks.

N = length(sz);
cores = cell(N,1);
for n = 1:N
    R1 = ranks(n);
    if n == 1
        R0 = ranks(end);
    else
        R0 = ranks(n-1);
    end
    cores{n} = randn(R0, sz(n), R1);
end

end