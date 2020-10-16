function rel_error = rel_error_TR_sptensor(cores, X_sparse, varargin)
%rel_error_TR_mat Computes relative error between TR tensor and sptensor
%
%rel_error = rel_error_TR_mat(cores, X_sparse) computes the relative
%error between the TR tensor defined by the TR-cores in cores, and the
%sptensor X_sparse. The sptensor is treated as ground truth, i.e., used
%in the denominator in the computation. By default, it slices the first
%dimension into 10 pieces, loads them separately and computes the norm for
%one slice at a time. 
%
%rel_error = rel_error_TR_mat(___, 'no_inc', no_inc) is an optional
%parameter that can control how many slices the sptensor is sliced
%into. Default is 10.
%
%rel_error = rel_error_TR_mat(___, 'slice_dim', slice_dim) is an optional
%parameter than can control which dimension is sliced. Default is the first
%dimension.

%% Handle optional inputs

params = inputParser;
addParameter(params, 'no_inc', 10);
addParameter(params, 'slice_dim', 1);
parse(params, varargin{:});

no_inc = params.Results.no_inc;
slice_dim = params.Results.slice_dim;

%% Compute norm via slices

sz = size(X_sparse);
N = length(sz);
inc_pts = round(linspace(0, sz(slice_dim), no_inc+1));
slice_arg = cell(1,N);
for n = 1:N
    slice_arg{n} = ':';
end
cores_slice = cores;
nrm2 = 0;
nrmX2 = 0;
for n = 1:no_inc
    slice_arg{slice_dim} = inc_pts(n)+1:inc_pts(n+1);
    X_slice = double(full(X_sparse(slice_arg{:})));
    cores_slice{slice_dim} = cores{slice_dim}(:, inc_pts(n)+1:inc_pts(n+1), :);
    Y_slice = cores_2_tensor(cores_slice);
    nrm2 = nrm2 + norm(X_slice(:) - Y_slice(:))^2;
    nrmX2 = nrmX2 + norm(X_slice(:))^2;
    fprintf('Finished computing norm for slice %d out of %d\n', n, no_inc)
end

rel_error = sqrt(nrm2/nrmX2);

end
