% Experiment 1
% The purpose of this script is to run synthetic experiments on our
% proposed TR-ALS-Sampled, as well as four other methods that we compare
% against.

% Settings: General experiment 
I = 100:100:500; % Tensor sizes
N = 3; % Tensor dimensionality
ranks = 10*ones(1, N); % Target ranks
extra_rank = 0; % Additional rank added
noise = 1e-1; % Amount of Gaussian noise added to each entry
target_acc = 1.2; % Target accuracy (1+epsilon)
no_trials = 10; % Number of experiment trials for averaging
verbose = false;
large_elem = 20;

% Settings: TR-ALS 
tol = 1e-6; % Tolerance to pass to tr_als when determining number of iterations

% Settings: TR-ALS-Sampled
J_init = 2*max(ranks)^2;
J_inc = 100;

% Settings: TR-SVD-Rand
oversamp = 10; % Following Remark 1 in their paper

% Create structures for storing results
len_I = length(I);
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
fname = 'experiment1_results.mat';

% Outer loop: Go through sizes
for m = 1:length(I)
    fprintf('Running experiment for I = %d...\n', I(m));
    
    % Set size of tensor
    sz = I(m)*ones(1, N);    
    
    % Inner loop: Go through trials
    for tr = 1:no_trials     
        fprintf('\n\tTRIAL %d\n', tr);
        
        % Generate tensor   
        fprintf('\tGenerating tensor for I = %d, trial = %d...', I(m), tr);
        X = generate_low_rank_tensor(sz, ranks + extra_rank, noise, 'large_elem', large_elem);
        normX = norm(X(:));
        fprintf(' Done!\n');
        
        % Determine number of iterations
        fprintf('\tDeterminig no. iterations for I = %d, trial = %d...', I(m), tr);
        [~, tr_als_it] = tr_als(X, ranks, 'tol', tol, 'maxiters', 1000, 'verbose', verbose);
        no_it = 2*tr_als_it;
        NO_IT(tr, m) = no_it;
        fprintf(' Done! No. iterations to use: %d\n', no_it);
        
        % Run TR-ALS
        fprintf('\tRunning TR-ALS for I = %d, trial = %d...', I(m), tr);
        tic_exp = tic;
        cores = tr_als(X, ranks, 'tol', 0, 'maxiters', no_it, 'verbose', verbose); 
        time_TR_ALS(tr, m) = toc(tic_exp);
        Y = cores_2_tensor(cores);
        rel_error_TR_ALS(tr, m) = norm(Y(:) - X(:)) / normX;
        fprintf(' Done!\n')
        
        % Run TR-ALS-Sampled
        fprintf('\tRunning TR-ALS-Sampled for I = %d, trial = %d', I(m), tr)
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
        fprintf('\tRunning rTR-ALS for I = %d, trial = %d', I(m), tr)
        K = I(m)/10;
        while true
            tic_exp = tic; 
            cores = rtr_als(X, ranks, K*ones(size(sz)), 'tol', 0, 'maxiters', no_it, 'verbose', verbose);
            time_rTR_ALS{tr, m} = [time_rTR_ALS{tr, m}; toc(tic_exp)];
            Y = cores_2_tensor(cores);
            rel_error_rTR_ALS{tr, m} = [rel_error_rTR_ALS{tr, m}; norm(Y(:)-X(:))/normX];
            if rel_error_rTR_ALS{tr, m}(end)/rel_error_TR_ALS(tr, m) < target_acc
                break
            end
            K = K + I(m)/20;
            fprintf('.');
        end
        fprintf(' Done!\n');
        
        % Run TR-SVD
        fprintf('\tRunning TR-SVD for I = %d, trial = %d', I(m), tr)
        tic_exp = tic;
        cores = TRdecomp_ranks(X, ranks);
        time_TR_SVD(tr, m) = toc(tic_exp);
        Y = cores_2_tensor(cores);
        rel_error_TR_SVD(tr, m) = norm(Y(:)-X(:))/normX;
        fprintf(' Done!\n');
        
        % Run TR-SVD-Rand
        fprintf('\tRunning TR-SVD-Rand for I = %d, trial = %d', I(m), tr)
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
end
