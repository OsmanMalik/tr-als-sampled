function [cores, varargout] = tr_als(X, ranks, varargin)
%tr_als Compute tensor ring decomposition via alternating least squares
%
%cores = tr_als(X, ranks) computes a tensor ring (TR) decomposition of the
%input N-dimensional array X. ranks is a length-N vector containing the
%target ranks. The output cores is a cell containing the N cores tensors,
%each represented as a 3-way array.
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

%% Handle optional inputs

params = inputParser;
addParameter(params, 'conv_crit', 'none');
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', false, @isscalar);
parse(params, varargin{:});

conv_crit = params.Results.conv_crit;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;

%% Initialize cores

sz = size(X);
cores = initialize_cores(sz, ranks);

if nargout > 1 && tol > 0 && (strcmp(conv_crit, 'relative error') || strcmp(conv_crit, 'norm'))
    conv_vec = zeros(1, maxiters);
end

%% Main loop
% Iterate until convergence, for a maximum of maxiters iterations

N = length(sz);
er_old = Inf;
for it = 1:maxiters
    for n = 1:N
        % Compute G_{[2]}^{\neq n} from cores
        G = subchain_matrix(cores, n);
        
        % Compute relevant unfolding of data tensor X
        XnT = mode_unfolding(X, n).';
        
        % Solve LS problem and update core
        Z = (G \ XnT).';
        cores{n} = classical_mode_folding(Z, 2, size(cores{n}));        
    end
    
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
        
        % Compute full tensor corresponding to cores
        Y = cores_2_tensor(cores);

        % Compute current relative error
        er = norm(X(:)-Y(:))/norm(X(:));
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