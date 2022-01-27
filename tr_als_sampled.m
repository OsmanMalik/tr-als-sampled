function [cores, varargout] = tr_als_sampled(X, ranks, embedding_dims, varargin)
%tr_als_sampled Compute tensor ring decomposition via sampled ALS
%
%For loading from file: It is assumed that the tensor is stored in a
%variable Y in the mat file.
%
%cores = tr_als(X, ranks, embedding_dims) computes a tensor ring (TR) 
%decomposition of the input N-dimensional array X by sampling the LS
%problems using sketch sizes for each dimension given in embedding_dims.
%Ranks is a length-N vector containing the target ranks. The output cores
%is a cell containing the N cores tensors, each represented as a 3-way
%array.
%
%cores = tr_als(___, 'conv_crit', conv_crit) is an optional parameter used
%to control which convergence criterion is used. Set to either 'relative
%error' or 'norm' to terminate when change in relative error or norm of
%TR-tensor is below the tolerance in tol. Default is that no convergence
%criterion is used.
%
%cores = tr_als(___, 'tol', tol) is an optional argument that controls the
%termination tolerance. If the change in the relative error is less than
%tol at the conclusion of a main loop iteration, the algorithm terminates.
%Default is 1e-3.
%
%cores = tr_als(___, 'maxiters', maxiters) is an optional argument that
%controls the maximum number of main loop iterations. The default is 50.
%
%cores = tr_als(___, 'resample', resample) can be used to avoid resampling
%those factor tensor that haven't been updated. This means that theoretical
%guarantees no longer formally apply for the LS problems. We always run
%this with the default of true, which means that full resampling is always
%done.
%
%cores = tr_als(___, 'verbose', verbose) is an optional argument that
%controls the amount of information printed to the terminal during
%execution. Setting verbose to true will result in more print out. Default
%is false.
%
%cores = tr_als(___, 'no_mat_inc', no_mat_inc) is used to control how
%input tensors read from file are sliced up to save RAM. We never use this
%in our experiments, and I may eventually remove this functionality.
%
%cores = tr_als(___, 'breakup', breakup) is an optional length-N vector
%input that can be used to break up the LS problems with multiple right
%hand sides that come up into pieces so that not all problems are solved at
%the same time. This is useful when a tensor dimension is particularly
%large.
%
%cores = tr_als(___, 'alpha', alpha) alpha is an optional parameter which
%controls how much Tikhonov regularization is added in LS problems. We
%found that this helped avoid ill-conditioning on certain datasets.

%% Handle inputs 

% Optional inputs
params = inputParser;
addParameter(params, 'conv_crit', 'none');
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'resample', true, @isscalar);
addParameter(params, 'verbose', false, @isscalar);
addParameter(params, 'no_mat_inc', false);
addParameter(params, 'breakup', false);
addParameter(params, 'alpha', 0);
addParameter(params, 'uniform_sampling', false);
parse(params, varargin{:});

conv_crit = params.Results.conv_crit;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
resample = params.Results.resample;
verbose = params.Results.verbose;
no_mat_inc = params.Results.no_mat_inc;
breakup = params.Results.breakup;
alpha = params.Results.alpha;
uniform_sampling = params.Results.uniform_sampling;

% Check if X is path to mat file on disk
%   X_mat_flag is a flag that keeps track of if X is an array or path to
%   a mat file on disk. In the latter case, X_mat will be a matfile that
%   can be used to access elements of the mat file.
if isa(X, 'char') || isa(X, 'string')
    X_mat_flag = true;
    X_mat = matfile(X, 'Writable', false);
else
    X_mat_flag = false;
end

%% Initialize cores, sampling probabilities and sampled cores

if X_mat_flag
    sz = size(X_mat, 'Y');
    N = length(sz);
    col_cell = cell(1,N);
    for n = 1:N
        col_cell{n} = ':';
    end
    
    % If value for no_mat_inc is provided, make sure it is a properly shaped
    % vector.
    if no_mat_inc(1)
        if ~(size(no_mat_inc,1)==1 && size(no_mat_inc,2)==N)
            no_mat_inc = no_mat_inc(1)*ones(1,N);
        end
    end
else
    sz = size(X);
    N = length(sz);
end
cores = initialize_cores(sz, ranks);

sampling_probs = cell(1, N);
for n = 2:N
    if uniform_sampling
        sampling_probs{n} = ones(sz(n), 1)/sz(n);
    else
        U = col(classical_mode_unfolding(cores{n}, 2));
        sampling_probs{n} = sum(U.^2, 2)/size(U, 2);
    end
end
core_samples = cell(1, N);
if ~breakup(1)
    breakup = ones(1,N);
end

slow_idx = cell(1,N);
sz_shifted = [1 sz(1:end-1)];
idx_prod = cumprod(sz_shifted);
sz_pts = cell(1,N);
for n = 1:N
    sz_pts{n} = round(linspace(0, sz(n), breakup(n)+1));
    slow_idx{n} = cell(1,breakup(n));
    for brk = 1:breakup(n)
        J = embedding_dims(n);
        samples_lin_idx_2 = prod(sz_shifted(1:n))*(sz_pts{n}(brk):sz_pts{n}(brk+1)-1).';
        slow_idx{n}{brk} = repelem(samples_lin_idx_2, J, 1);
    end
end

if nargout > 1 && tol > 0 && (strcmp(conv_crit, 'relative error') || strcmp(conv_crit, 'norm'))
    conv_vec = zeros(1, maxiters);
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
    
    % Inner for loop
    for n = 1:N
        
        % Construct sketch and sample cores
        if resample % Resample all cores, except nth which will be updated
            J = embedding_dims(n);
            samples = nan(J, N);
            for m = 1:N
                if m ~= n
                    samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
                    core_samples{m} = cores{m}(:, samples(:,m), :);
                end
            end
        else % Only resample the core that was updated in last iteration
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
        end
        rescaling = rescaling ./ sqrt(J);
        
        % Construct sketched design matrix
        idx = [n+1:N 1:n-1]; % Order in which to multiply cores
        G_sketch = permute(core_samples{idx(1)}, [1 3 2]);
        for m = 2:N-1
            permuted_core = permute(core_samples{idx(m)}, [1 3 2]);
            G_sketch = pagemtimes(G_sketch, permuted_core);
        end
        G_sketch = permute(G_sketch, [3 2 1]);
        G_sketch = reshape(G_sketch, J, numel(G_sketch)/J);
        G_sketch = rescaling .* G_sketch;
        if breakup(n) > 1
            if alpha > 0
                [L, U, p] = lu(G_sketch.'*G_sketch + alpha*eye(size(G_sketch,2)), 'vector');
            else 
                [L, U, p] = lu(G_sketch, 'vector');
            end
            ZT = zeros(size(G_sketch,2), sz(n));
        end
        
        % Sample right hand side
        for brk = 1:breakup(n)
        %{
        if X_mat_flag % X in mat file -- load slicewise
            Xn_sketch = zeros(J, sz(n));
            for j = 1:J
                slice_arg = cell(1,N);
                for m = 1:N
                    if m == n
                        slice_arg{m} = ':';
                    else
                        slice_arg{m} = samples(j, m);
                    end
                end
                row = X_mat.Y(slice_arg{:});
                Xn_sketch(j, :) = row(:).';
                if mod(j,3000) == 0
                    fprintf('\tDone with j = %d\n', j);
                end
            end
            
            % Old below
            %{
            slice_arg = col_cell;
            inc_pts = linspace(0, sz(n), no_mat_inc(n)+1);
            Xn_sketch = zeros(J, sz(n));
            for m = 1:no_mat_inc(n)
                slice_arg{n} = inc_pts(m)+1:inc_pts(m+1);
                df = inc_pts(m+1)-inc_pts(m);
                sz_slice = sz;
                sz_slice(n) = df;
                sz_slice_shifted = [1 sz_slice(1:end-1)];
                idx_slice_prod = cumprod(sz_slice_shifted);
                samples_lin_idx_1 = 1 + (samples(:, idx)-1) * idx_slice_prod(idx).';
                samples_lin_idx_2 = idx_slice_prod(n)*(0:df-1).';
                samples_lin_idx = repmat(samples_lin_idx_1, df, 1) + repelem(samples_lin_idx_2, J, 1);
                X_slice = X_mat.Y(slice_arg{:});
                Xn_sketch(:, inc_pts(m)+1:inc_pts(m+1)) = reshape(X_slice(samples_lin_idx), J, df);
                if verbose
                    fprintf('\tFinished slice %d of %d\n', m, no_mat_inc(n));
                end
            end
            %}
        else % X array in RAM -- use linear indexing
            samples_lin_idx_1 = 1 + (samples(:, idx)-1) * idx_prod(idx).';
            samples_lin_idx = repmat(samples_lin_idx_1, sz(n), 1) + slow_idx{n};
            X_sampled = X(samples_lin_idx);
            Xn_sketch = reshape(X_sampled, J, sz(n));
            %if isa(X, 'sptensor')
            %    Xn_sketch = sparse(Xn_sketch);
            %end
        end
        %}
            
            no_cols = sz_pts{n}(brk+1)-sz_pts{n}(brk);
            samples_lin_idx_1 = 1 + (samples(:, idx)-1) * idx_prod(idx).';
            samples_lin_idx = repmat(samples_lin_idx_1, no_cols, 1) + slow_idx{n}{brk};
            X_sampled = X(samples_lin_idx);
            Xn_sketch = reshape(X_sampled, J, no_cols);

            % Rescale right hand side
            Xn_sketch = rescaling .* Xn_sketch;
            
            if breakup(n) > 1
                if alpha > 0
                    ZT(:, sz_pts{n}(brk)+1:sz_pts{n}(brk+1)) = U \ (L \ G_sketch(:,p).'*Xn_sketch);
                else
                    ZT(:, sz_pts{n}(brk)+1:sz_pts{n}(brk+1)) = U \ (L \ Xn_sketch(p, :));
                end
            end
        end
        if breakup(n) > 1
            Z = ZT.';
        else
            if alpha > 0
                Z = (( G_sketch.'*G_sketch + alpha*eye(size(G_sketch,2)) ) \ ( G_sketch.'*Xn_sketch )).';
            else
                Z = (G_sketch \ Xn_sketch).';
            end
        end

        cores{n} = classical_mode_folding(Z, 2, size(cores{n}));
        
        % Update sampling distribution for core
        if uniform_sampling
            sampling_probs{n} = ones(sz(n), 1)/sz(n);
        else
            U = col(classical_mode_unfolding(cores{n}, 2));
            sampling_probs{n} = sum(U.^2, 2)/size(U, 2);
        end
    end
    
    
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
        
        % Compute full tensor corresponding to cores
        Y = cores_2_tensor(cores);

        % Compute current relative error
        if X_mat_flag
            XX = X_mat.Y;
            er = norm(XX(:)-Y(:))/norm(XX(:));
        else
            er = norm(X(:)-Y(:))/norm(X(:));
        end        
        if verbose
            fprintf('\tRelative error after iteration %d: %.8f\n', it, er);
        end

        % Save current error to conv_vec if required
        if nargout > 1
            conv_vec(it) = er;
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
        
        
    % Check convergence: Norm change 
    elseif tol > 0 && strcmp(conv_crit, 'norm')
        
        % Compute norm of TR tensor and change if it > 1

        norm_new = normTR(cores);
        if it == 1
            norm_change = Inf;
        else
            norm_change = abs(norm_new - norm_old);
            if verbose
                fprintf('\tNorm change after iteration %d: %.8f\n', it, norm_change);
            end
        end
        
        % Save current norm_change to conv_vec
        if nargout > 1
            conv_vec(it) = norm_change;
        end
        
        % Break if change in relative error below threshold
        if norm_change < tol
            if verbose
                fprintf('\tNorm change below tol; terminating...\n');
            end
            break
        end
        
        % Update old norm
        norm_old = norm_new;
    
    % Just print iteration count
    else
        if verbose
            fprintf('\tIteration %d complete\n', it);
        end  
    end
    
end

if nargout > 1 && exist('conv_vec', 'var')
    varargout{1} = conv_vec(1:it);
else
    varargout{1} = nan;
end

end
