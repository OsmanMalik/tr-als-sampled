% This experiment is meant to be for sparse real world data

% Settings
dataset = "crime-comm";
J = 20000;
R = 10;
no_it = 20;
tol = 1e-1;
alpha = 0;

%% Load and preprocess

if strcmp(dataset, 'synthetic')
    sz = [100 100 100];
    N = length(sz);
    density = .01;
    X = sptenrand(sz, density);
    breakup = [1, 1, 1];
else
    if strcmp(dataset, 'uber')
        tensor_path = "D:\data_sets\tensors\Uber Pickups\uber.tns";
        breakup = [1, 1, 1, 1];
    elseif strcmp(dataset, 'nips')
        tensor_path = "D:\data_sets\tensors\NIPS Publications\nips.tns";
        breakup = [3, 3, 14, 1];
    elseif strcmp(dataset, 'crime-comm') % Size: 6186 x 24 x 77 x 32
        tensor_path = "D:\data_sets\tensors\Chicago Crime\chicago-crime-comm.tns";
        breakup = [6, 1, 1, 1];
    elseif strcmp(dataset, 'crime-geo') % Size: 6185 x 24 x 380 x 395 x 32
        tensor_path = "D:\data_sets\tensors\Chicago Crime\chicago.tns";
        breakup = [1, 1, 1, 1, 1];
    end
    mat = importdata(tensor_path);
    N = size(mat, 2) - 1;
    X = sptensor(mat(:, 1:N), mat(:, end));
    sz = size(X);
end

%% Clear out zeros
%{
nnz_idx = cell(1,N);
col_cell = cell(1,N);
for n = 1:N
    col_cell{n} = ':';
end
for n = 1:N
    nnz_idx{n} = [];
    a = col_cell;
    for k = 1:sz(n)
        a{n} = k;
        if nnz(X(a{:})) > 0
            nnz_idx{n} = [nnz_idx{n} k];
        end
    end
    n
end
%}

%% Do decomposition

sketch_dims = J*ones(1,N);
ranks = R*ones(1,N);
my_tic = tic;
[cores, conv_vec] = tr_als_sampled(X, ranks, sketch_dims, 'tol', tol, 'maxiters', no_it, 'resample', true, 'verbose', true, 'conv_crit', 'norm', 'breakup', breakup, 'alpha', alpha); 
my_toc = toc(my_tic);

%% Save

fname = "experiment3_" + dataset + "_R" + num2str(R) + "_J" + num2str(J);
save(fname)