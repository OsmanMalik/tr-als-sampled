% Experiment 1b: Varying rank
%
% The purpose of this script is to run synthetic experiments on our
% proposed TR-ALS-Sampled, as well as four other methods that we compare
% against. 
%
% In this experiment, we vary the rank.

% Settings: General experiment 
I = 100; % Tensor sizes
N = 3; % Tensor dimensionality
sz = I*ones(1, N);
R = 10:5:35;
true_rank = 30;
%ranks = 10*ones(1, N); % Target ranks
extra_rank = 0; % Additional rank added
noise = 1e-1; % Amount of Gaussian noise added to each entry
target_acc = 1.02; % Target accuracy (1+epsilon)
no_trials = 5; % Number of experiment trials for averaging
verbose = false;
large_elem = 20;

% Settings: TR-ALS 
tol = 1e-3; % Tolerance to pass to tr_als when determining number of iterations

% Settings: TR-SVD-Rand
oversamp = 10; % Following Remark 1 in their paper

% Create structures for storing results
len_R = length(R);
NO_IT = nan(no_trials, len_R);
rel_error_TR_ALS = nan(no_trials, len_R);
rel_error_TR_ALS_Sampled = cell(no_trials, len_R);
rel_error_rTR_ALS = cell(no_trials, len_R);
rel_error_TR_SVD = nan(no_trials, len_R);
rel_error_TR_SVD_Rand = nan(no_trials, len_R);
time_TR_ALS = nan(no_trials, len_R);
time_TR_ALS_Sampled = cell(no_trials, len_R);
time_rTR_ALS = cell(no_trials, len_R);
time_TR_SVD = nan(no_trials, len_R);
time_TR_SVD_Rand = nan(no_trials, len_R);
fname = 'experiment1b_results.mat';

% Outer loop: Go through trials
for tr = 1:no_trials   
    fprintf('Running experiment for tr = %d...\n', tr);

    % Generate tensor   
    fprintf('\tGenerating tensor for R = %d, trial = %d...', true_rank, tr);
    X = generate_low_rank_tensor(sz, true_rank*ones(1,N), noise, 'large_elem', large_elem);
    normX = norm(X(:));
    fprintf(' Done!\n');
        
    % Inner loop: Go through ranks
    for m = 1:len_R
        fprintf('\n\tRANK %d\n', R(m)); 
        
        % Set ranks
        ranks = R(m)*ones(1, N);

        % Settings: TR-ALS-Sampled
        J_init = 2*max(R(m))^2;
        J_inc = 1000;
        
        % Determine number of iterations
        fprintf('\tDeterminig no. iterations for R = %d, trial = %d...', R(m), tr);
        [~, conv_vec] = tr_als(X, ranks, 'tol', tol, 'maxiters', 500, 'verbose', verbose, 'conv_crit', 'relative error');
        tr_als_it = length(conv_vec);
        no_it = 2*tr_als_it;
        NO_IT(tr, m) = no_it;
        fprintf(' Done! No. iterations to use: %d\n', no_it);
        
        % Run TR-ALS
        fprintf('\tRunning TR-ALS for R = %d, trial = %d...', R(m), tr);
        tic_exp = tic;
        cores = tr_als(X, ranks, 'tol', 0, 'maxiters', no_it, 'verbose', verbose); 
        time_TR_ALS(tr, m) = toc(tic_exp);
        Y = cores_2_tensor(cores);
        rel_error_TR_ALS(tr, m) = norm(Y(:) - X(:)) / normX;
        fprintf(' Done!\n')
        
        % Run TR-ALS-Sampled
        fprintf('\tRunning TR-ALS-Sampled for R = %d, trial = %d', R(m), tr)
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
        fprintf('\tRunning rTR-ALS for R = %d, trial = %d', R(m), tr)
        K = I/10;
        while true
            tic_exp = tic; 
            cores = rtr_als(X, ranks, K*ones(size(sz)), 'tol', 0, 'maxiters', no_it, 'verbose', verbose);
            time_rTR_ALS{tr, m} = [time_rTR_ALS{tr, m}; toc(tic_exp)];
            Y = cores_2_tensor(cores);
            rel_error_rTR_ALS{tr, m} = [rel_error_rTR_ALS{tr, m}; norm(Y(:)-X(:))/normX];
            if rel_error_rTR_ALS{tr, m}(end)/rel_error_TR_ALS(tr, m) < target_acc
                break
            end
            K = K + I/20;
            fprintf('.');
        end
        fprintf(' Done!\n');
        
        % Run TR-SVD
        if R(m)^2 < I
            fprintf('\tRunning TR-SVD for R = %d, trial = %d', R(m), tr)
            tic_exp = tic;
            cores = TRdecomp_ranks(X, ranks);
            time_TR_SVD(tr, m) = toc(tic_exp);
            Y = cores_2_tensor(cores);
            rel_error_TR_SVD(tr, m) = norm(Y(:)-X(:))/normX;
            fprintf(' Done!\n');
        else
            time_TR_SVD(tr, m) = nan;
            rel_error_TR_SVD(tr, m) = nan;
        end
        
        % Run TR-SVD-Rand
        if R(m)^2 < I
            fprintf('\tRunning TR-SVD-Rand for R = %d, trial = %d', R(m), tr)
            tic_exp = tic;
            cores = tr_svd_rand(X, ranks, oversamp);
            time_TR_SVD_Rand(tr, m) = toc(tic_exp);
            Y = cores_2_tensor(cores);
            rel_error_TR_SVD_Rand(tr, m) = norm(Y(:)-X(:))/normX;
            fprintf(' Done!\n');
        else
            time_TR_SVD_Rand(tr, m) = nan;
            rel_error_TR_SVD_Rand(tr, m) = nan;
        end
    end
    
    fprintf('\n');
    
    % Save stuff
    save(fname)  % Just save everything...
    %save(fname, 'NO_IT', ...
    %    'rel_error_TR_ALS', ...
    %    'rel_error_TR_ALS_Sampled', ...
    %    'rel_error_rTR_ALS', ...
    %    'rel_error_TR_SVD', ...
    %    'rel_error_TR_SVD_Rand', ...
    %    'time_TR_ALS', ...
    %    'time_TR_ALS_Sampled', ...
    %    'time_rTR_ALS', ...
    %    'time_TR_SVD', ...
    %    'time_TR_SVD_Rand')
end
