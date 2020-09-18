function cores = tr_als_sampled(X, ranks, embedding_dims, varargin)
%tr_als_sampled Compute tensor ring decomposition via sampled ALS
%
%cores = tr_als(X, ranks, embedding_dims) computes a tensor ring (TR) 
%decomposition of the input N-dimensional array X by sampling the LS
%problems using sketch sizes for each dimension given in embedding_dims.
%Ranks is a length-N vector containing the target ranks. The output cores
%is a cell containing the N cores tensors, each represented as a 3-way
%array.
%
%cores = tr_als(___, 'tol', tol) is an optional argument that controls the
%termination tolerance. If the change in the relative error is less than
%tol at the conclusion of a main loop iteration, the algorithm terminates.
%Default is 1e-3.
%
%cores = tr_als(___, 'maxiters', maxiters) is an optional argument that
%controls the maximum number of main loop iterations. The default is 50.
%
%cores = tr_als(___, 'verbose', verbose) is an optional argument that
%controls the amount of information printed to the terminal during
%execution. Setting verbose to true will result in more print out. Default
%is false.

%% Add relevant paths

addpath('help_functions\mtimesx\mtimesx_20110223')
mtimesx('SPEED');

%% Handle optional inputs

params = inputParser;
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'resample', true, @isscalar);
addParameter(params, 'verbose', false, @isscalar);
parse(params, varargin{:});

tol = params.Results.tol;
maxiters = params.Results.maxiters;
resample = params.Results.resample;
verbose = params.Results.verbose;

%% Initialize cores, sampling probabilities and sampled cores

sz = size(X);
cores = initialize_cores(sz, ranks);
N = length(sz);
sampling_probs = cell(1, N);
for n = 2:N
    U = col(classical_mode_unfolding(cores{n}, 2));
    sampling_probs{n} = sum(U.^2, 2)/size(U, 2);
end
core_samples = cell(1, N);

slow_idx = cell(1,N);
sz_shifted = [1 sz(1:end-1)];
idx_prod = cumprod(sz_shifted);
for n = 1:N
    J = embedding_dims(n);
    samples_lin_idx_2 = prod(sz_shifted(1:n))*(0:sz(n)-1).';
    slow_idx{n} = repelem(samples_lin_idx_2, J, 1);
end

%% Main loop
% Iterate until convergence, for a maximum of maxiters iterations

if ~resample
    J = embedding_dims(1); % Always use same embedding dim
    samples = nan(J, N);
    
    for m = 2:N-1
        samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
        core_samples{m} = cores{m}(:, samples(:,m), :);
    end
end

er_old = Inf;
for it = 1:maxiters
    for n = 1:N
        
        % Construct sketch and sample cores
        if resample
            % Resample all cores, except nth which will be updated
            J = embedding_dims(n);
            samples = nan(J, N);
            for m = 1:N
                if m ~= n
                    samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
                    core_samples{m} = cores{m}(:, samples(:,m), :);
                end
            end
        else
            % Only resample the core that was updated in last iteration
            m = mod(n-2,N)+1;
            samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
            core_samples{m} = cores{m}(:, samples(:,m), :);
        end
        
        % Compute the row rescaling factors
        rescaling = ones(J, 1);
        for m = 1:N
            if m ~= n
                rescaling = rescaling ./ sqrt(sampling_probs{m}(samples(:, m)));
            end
            rescaling = rescaling ./ sqrt(J);
        end
        
        % Construct sketched design matrix
        idx = [n+1:N 1:n-1]; % Order in which to multiply cores
        G_sketch = permute(core_samples{idx(1)}, [1 3 2]);
        for m = 2:N-1
            permuted_core = permute(core_samples{idx(m)}, [1 3 2]);
            G_sketch = mtimesx(G_sketch, permuted_core);
        end
        G_sketch = permute(G_sketch, [3 2 1]);
        G_sketch = reshape(G_sketch, J, numel(G_sketch)/J);
        G_sketch = rescaling .* G_sketch;
        
        % Construct sketched right hand side
        %sz_shifted = [1 sz(1:end-1)];
        %idx_prod = cumprod(sz_shifted);
        samples_lin_idx_1 = 1 + (samples(:, idx)-1) * idx_prod(idx).';
        %samples_lin_idx_2 = prod(sz_shifted(1:n))*(0:sz(n)-1).';
        
        samples_lin_idx = repmat(samples_lin_idx_1, sz(n), 1) + slow_idx{n};
        X_sampled = X(samples_lin_idx);
        Xn_sketch = reshape(X_sampled, J, sz(n));
        %Xn_sketch = reshape(X_sampled, sz(n), J);
        %Xn_sketch = permute(Xn_sketch, [2 1]);
        Xn_sketch = rescaling .* Xn_sketch;
        
        % Below is old code for constructing sketched RHS, which is slower
        % than the above block.
        %{
        Xn = permute(mode_unfolding(X, n), [2 1]);
        samples_prod = 1 + (samples(:, idx)-1) * [1 cumprod(sz(idx(1:end-1)))].'; 
        Xn_sketch = Xn(samples_prod, :);      
        if isa(Xn_sketch, 'sptensor')
            Xn_sketch = sparse(Xn_sketch.subs(:,1), Xn_sketch.subs(:,2), Xn_sketch.vals, size(Xn_sketch,1), size(Xn_sketch,2));
        end
        %}
        
        % Solve sketched LS problem and update core
        Z = (G_sketch \ Xn_sketch).';
        cores{n} = classical_mode_folding(Z, 2, size(cores{n}));
        
        % Update sampling distribution for core
        U = col(classical_mode_unfolding(cores{n}, 2));
        sampling_probs{n} = sum(U.^2, 2)/size(U, 2);
    end
    
    if tol > 0
        % Compute full tensor corresponding to cores
        Y = cores_2_tensor(cores);

        % Compute current relative error
        er = norm(X(:)-Y(:))/norm(X(:));
        if verbose
            fprintf('\tRelative error after iteration %d: %.8f\n', it, er);
        end

        % Break if change in relative error below threshold
        if abs(er - er_old) < tol
            if verbose
                fprintf('\tRelative error change below tol; terminating...\n');
            end
            break
        end

        % Update old error
        er_old = er;
    else
        if verbose
            fprintf('\tIteration %d complete\n', it);
        end
    end
    
end

end