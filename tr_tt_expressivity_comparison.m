% This script was used for computing the TT and TR ranks mentioned in
% Remark 1 of our paper

cores = cell(4,1);
cores{1} = reshape(1:8,2,2,2);
cores{2} = reshape(1:8,2,2,2);
cores{3} = reshape(1:8,2,2,2);
cores{4} = reshape(1:8,2,2,2);
N = length(cores);

no_tr_param = 0;
for k = 1:N
    no_tr_param = no_tr_param + numel(cores{k});
end

X = cores_2_tensor(cores);
sz = size(X);

tt_ranks = zeros(N-1,1);
no_tt_param = 0;
for k = 1:N-1
    tt_ranks(k) = rank(reshape(X, prod(sz(1:k)), prod(sz(k+1:end))));
    svd(reshape(X, prod(sz(1:k)), prod(sz(k+1:end))));
    if k > 1
        no_tt_param = no_tt_param + sz(k)*tt_ranks(k)*tt_ranks(k-1);
    else
        no_tt_param = no_tt_param + sz(k)*tt_ranks(k);
    end
end
no_tt_param = no_tt_param + sz(end)*tt_ranks(end);
no_tr_param
no_tt_param
