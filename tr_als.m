function cores = tr_als(X, ranks, varargin)
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
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', false, @isscalar);
parse(params, varargin{:});

tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;

%% Initialize cores

sz = size(X);
cores = initialize_cores(sz, ranks);

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