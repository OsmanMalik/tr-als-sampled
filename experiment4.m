% Experiment 4
% The purpose of this script is to essentially repeat what we did in
% experiment 1, but on real data.

%include_toolboxes

% Settings: General experiment 
dataset = "nell-mini";
R = 10;
no_it = 100;

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
        mini_size = 500;
    end
    mat = importdata(tensor_path);
    N = size(mat, 2) - 1;
    X = sptensor(mat(:, 1:N), mat(:, end));
    if strcmp(dataset, 'nell-mini')
        X = X(1:mini_size, 1:mini_size, 1:mini_size);
    end
    sz = size(X);
    X = double(X);
end
normX = norm(tensor(X));

%% 
ranks = R*ones(1, N); % Target ranks
target_acc = 1.20; % Target accuracy (1+epsilon)
no_trials = 1; % Number of experiment trials for averaging
verbose = true;

% Settings: TR-ALS 
tol = 1e-3; % Tolerance to pass to tr_als when determining number of iterations

% Settings: TR-ALS-Sampled
J_init = 2*max(ranks)^2;
J_inc = 1000;

% Settings: TR-SVD-Rand
oversamp = 10; % Following Remark 1 in their paper

% Create structures for storing results
len_I = 1;
NO_IT = nan(no_trials, len_I);
rel_error_TR_ALS = nan(no_trials, len_I);
rel_error_TR_ALS_Sampled = cell(no_trials, len_I);
rel_error_rTR_ALS = cell(no_trials, len_I);
rel_error_TR_SVD = nan(no_trials, len_I);
rel_error_TR_SVD_Rand = nan(no_trials, len_I);
time_TR_ALS = nan(no_trials, len_I);
time_TR_ALS_Sampled = cell(no_trials, len_I);
time_rTR_ALS = cell(no_trials, len_I);
time_TR_SVD = nan(no_trials, len_I);
time_TR_SVD_Rand = nan(no_trials, len_I);
fname = "experiment4_results_" + dataset + ".mat";

m = 1;
% Inner loop: Go through trials
for tr = 1:no_trials     
    fprintf('\n\tTRIAL %d\n', tr);

    % Determine number of iterations
    fprintf('\tDeterminig no. iterations for trial = %d...', tr);
    [~, conv_vec] = tr_als(X, ranks, 'tol', tol, 'maxiters', 1000, 'verbose', verbose, 'conv_crit', 'relative error');
    tr_als_it = length(conv_vec);
    no_it = 2*tr_als_it;
    NO_IT(tr, m) = no_it;
    fprintf(' Done! No. iterations to use: %d\n', no_it);

    % Run TR-ALS
    fprintf('\tRunning TR-ALS for trial = %d...', tr);
    tic_exp = tic;
    cores = tr_als(X, ranks, 'tol', 0, 'maxiters', no_it, 'verbose', verbose); 
    time_TR_ALS(tr, m) = toc(tic_exp);
    Y = cores_2_tensor(cores);
    rel_error_TR_ALS(tr, m) = norm(Y(:) - X(:)) / normX;
    fprintf(' Done!\n')

    % Run TR-ALS-Sampled
    fprintf('\tRunning TR-ALS-Sampled for trial = %d', tr)
    J = J_init;
    while true
        tic_exp = tic;
        cores = tr_als_sampled(X, ranks, J*ones(size(sz)), 'tol', 0, 'maxiters', no_it, 'resample', true, 'verbose', verbose); 
        time_TR_ALS_Sampled{tr, m} = [time_TR_ALS_Sampled{tr, m}; toc(tic_exp)];
        Y = cores_2_tensor(cores);
        rel_error_TR_ALS_Sampled{tr, m} = [rel_error_TR_ALS_Sampled{tr, m}; norm(Y(:)-X(:))/normX];
        if rel_error_TR_ALS_Sampled{tr, m}(end)/rel_error_TR_ALS(tr, m) < target_acc
            break
        end
        J = J + J_inc;
        fprintf('.');
    end
    fprintf(' Done!\n');

    % Run rTR-ALS
    fprintf('\tRunning rTR-ALS for trial = %d', tr)
    K = round(sz/10);
    while true
        tic_exp = tic; 
        cores = rtr_als(X, ranks, K, 'tol', 0, 'maxiters', no_it, 'verbose', verbose);
        time_rTR_ALS{tr, m} = [time_rTR_ALS{tr, m}; toc(tic_exp)];
        Y = cores_2_tensor(cores);
        rel_error_rTR_ALS{tr, m} = [rel_error_rTR_ALS{tr, m}; norm(Y(:)-X(:))/normX];
        if rel_error_rTR_ALS{tr, m}(end)/rel_error_TR_ALS(tr, m) < target_acc
            break
        end
        K = K + round(sz/20);
        fprintf('.');
    end
    fprintf(' Done!\n');

    % Run TR-SVD
    fprintf('\tRunning TR-SVD for trial = %d', tr)
    tic_exp = tic;
    cores = TRdecomp_ranks(X, ranks);
    time_TR_SVD(tr, m) = toc(tic_exp);
    Y = cores_2_tensor(cores);
    rel_error_TR_SVD(tr, m) = norm(Y(:)-X(:))/normX;
    fprintf(' Done!\n');

    % Run TR-SVD-Rand
    fprintf('\tRunning TR-SVD-Rand for trial = %d', tr)
    tic_exp = tic;
    cores = tr_svd_rand(X, ranks, oversamp);
    time_TR_SVD_Rand(tr, m) = toc(tic_exp);
    Y = cores_2_tensor(cores);
    rel_error_TR_SVD_Rand(tr, m) = norm(Y(:)-X(:))/normX;
    fprintf(' Done!\n');
end

fprintf('\n');

% Save stuff
save(fname, 'NO_IT', ...
    'rel_error_TR_ALS', ...
    'rel_error_TR_ALS_Sampled', ...
    'rel_error_rTR_ALS', ...
    'rel_error_TR_SVD', ...
    'rel_error_TR_SVD_Rand', ...
    'time_TR_ALS', ...
    'time_TR_ALS_Sampled', ...
    'time_rTR_ALS', ...
    'time_TR_SVD', ...
    'time_TR_SVD_Rand')
