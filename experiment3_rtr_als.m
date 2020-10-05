% This experiment is meant to be for sparse real world data, for the
% rTR-ALS method.

% Settings
dataset = "nell-mini";
R = 10;
no_it = 20;
tol = 1e-1;

%% Load and preprocess

if strcmp(dataset, 'synthetic')
    sz = [100 100 100];
    N = length(sz);
    density = .01;
    X = sptenrand(sz, density);
else
    if strcmp(dataset, 'uber')
        tensor_path = "D:\data_sets\tensors\Uber Pickups\uber.tns";
    elseif strcmp(dataset, 'nips')
        tensor_path = "D:\data_sets\tensors\NIPS Publications\nips.tns";
    elseif strcmp(dataset, 'crime-comm') % Size: 6186 x 24 x 77 x 32
        tensor_path = "D:\data_sets\tensors\Chicago Crime\chicago-crime-comm.tns";
    elseif strcmp(dataset, 'crime-geo') % Size: 6185 x 24 x 380 x 395 x 32
        tensor_path = "D:\data_sets\tensors\Chicago Crime\chicago.tns";
    elseif strcmp(dataset, 'nell-mini')
        tensor_path = "D:\data_sets\tensors\NELL-2\nell-2.tns";
        mini_size = 1000;
    end
    mat = importdata(tensor_path);
    N = size(mat, 2) - 1;
    X = sptensor(mat(:, 1:N), mat(:, end));
    if strcmp(dataset, 'nell-mini')
        X = X(1:mini_size, 1:mini_size, 1:mini_size);
    end
    sz = size(X);
end

%% Compress 

tic_compress = tic;
if strcmp(dataset, 'uber') % Compress to 150 x 24 x 150 x 150
    K = 150;
    
    % Compute dim-1 factor matrix
    Xn = classical_mode_unfolding(X, 1);
    XnM = zeros(size(Xn,1), K);
    for k = 1:K
        M = randn(prod(sz)/sz(1), 1);
        XnM(:,k) = Xn*M;
        fprintf('Dim 1, k = %d\n', k)
    end
    [U,~] = qr(XnM,0);
    Q1 = U.';
    
    % Compute dim-3 factor matrix
    Xn = classical_mode_unfolding(X, 3);
    XnM = zeros(size(Xn,1), K);
    for k = 1:K
        M = randn(prod(sz)/sz(3), 1);
        XnM(:,k) = Xn*M;
        fprintf('Dim 3, k = %d\n', k)
    end
    [U,~] = qr(XnM,0);
    Q3 = U.';
    
    % Compute dim-4 factor matrix
    Xn = classical_mode_unfolding(X, 4);
    XnM = zeros(size(Xn,1), K);
    for k = 1:K
        M = randn(prod(sz)/sz(4), 1);
        XnM(:,k) = Xn*M;
        fprintf('Dim 4, k = %d\n', k)
    end
    [U,~] = qr(XnM,0);
    Q4 = U.';
    
    % Compress X, in careful order to avoid memory blow-up
    X = ttm(X, Q4, 4);
    X = ttm(X, Q3, 3);
    X = ttm(X, Q1, 1);
    
elseif strcmp(dataset, 'nell-mini') % Compress to 500 x 500 x 500 
    K = 500;
    
    % Compute dim-1 factor matrix
    Xn = classical_mode_unfolding(X, 1);
    XnM = zeros(size(Xn,1), K);
    for k = 1:K
        M = randn(prod(sz)/sz(1), 1);
        XnM(:,k) = Xn*M;
        fprintf('Dim 1, k = %d\n', k)
    end
    [U,~] = qr(XnM,0);
    Q1 = U.';
    
    % Compute dim-2 factor matrix
    Xn = classical_mode_unfolding(X, 2);
    XnM = zeros(size(Xn,1), K);
    for k = 1:K
        M = randn(prod(sz)/sz(2), 1);
        XnM(:,k) = Xn*M;
        fprintf('Dim 2, k = %d\n', k)
    end
    [U,~] = qr(XnM,0);
    Q2 = U.';
    
    % Compute dim-3 factor matrix
    Xn = classical_mode_unfolding(X, 3);
    XnM = zeros(size(Xn,1), K);
    for k = 1:K
        M = randn(prod(sz)/sz(3), 1);
        XnM(:,k) = Xn*M;
        fprintf('Dim 3, k = %d\n', k)
    end
    [U,~] = qr(XnM,0);
    Q3 = U.';
    
    % Compress
    X = ttm(X, Q1, 1);
    X = ttm(X, Q2, 2);
    X = ttm(X, Q3, 3);
end
X = double(tensor(X));
toc_compress = toc(tic_compress);

%% Run TR-ALS on compressed tensor

ranks = R*ones(1,N);
tic_decompose = tic;
[cores, conv_vec] = tr_als(X, ranks, 'tol', tol, 'maxiters', no_it, 'verbose', true, 'conv_crit', 'norm');
toc_decompose = toc(tic_decompose);

%% Compute TR cores for original tensor

tic_uncompress = tic;
cores{1} = double(ttm(tensor(cores{1}), Q1.', 2));
cores{3} = double(ttm(tensor(cores{3}), Q3.', 2));
cores{4} = double(ttm(tensor(cores{4}), Q4.', 2));
toc_uncompress = toc(tic_uncompress);

toc_total = toc_compress + toc_decompose + toc_uncompress;

%% Save stuff

fname = "experiment3_rtr_als_" + dataset + "_R" + num2str(R) + "_J" + num2str(J);
save(fname)

