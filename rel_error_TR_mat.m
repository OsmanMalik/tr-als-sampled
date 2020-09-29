function rel_error = rel_error_TR_mat(cores, X_mat_file, varargin)
%rel_error_TR_mat Computes relative error between TR tensor and mat tensor
%
%rel_error = rel_error_TR_mat(cores, X_mat_file) computes the relative
%error between the TR tensor defined by the TR-cores in cores, and the full
%tensor stored as a matfile, with path given in X_mat_file. The full tensor
%is treated as ground truth, i.e., used in the denominator in the
%computation. By default, it slices the first dimension into 10 pieces,
%loads them separately and computes the norm for one slice at a time.
%
%rel_error = rel_error_TR_mat(___, 'no_mat_inc', no_mat_inc) is an optional
%parameter than can control how many slices the matfile tensor is sliced
%into. Default is 10.
%
%rel_error = rel_error_TR_mat(___, 'slice_dim', slice_dim) is an optional
%parameter than can control which dimension is sliced. Default is the first
%dimension.

%% Handle optional inputs

params = inputParser;
addParameter(params, 'no_mat_inc', 10);
addParameter(params, 'slice_dim', 1);
parse(params, varargin{:});

no_mat_inc = params.Results.no_mat_inc;
slice_dim = params.Results.slice_dim;

%% Compute norm via slices

X_mat = matfile(X_mat_file, 'Writable', false);
sz = size(X_mat, 'Y');
N = length(sz);
inc_pts = round(linspace(0, sz(slice_dim), no_mat_inc+1));
slice_arg = cell(1,N);
for n = 1:N
    slice_arg{n} = ':';
end
cores_slice = cores;
nrm2 = 0;
nrmX2 = 0;
for n = 1:no_mat_inc
    slice_arg{slice_dim} = inc_pts(n)+1:inc_pts(n+1);
    X_slice = X_mat.Y(slice_arg{:});
    cores_slice{slice_dim} = cores{slice_dim}(:, inc_pts(n)+1:inc_pts(n+1), :);
    Y_slice = cores_2_tensor(cores_slice);
    nrm2 = nrm2 + norm(X_slice(:) - Y_slice(:))^2;
    nrmX2 = nrmX2 + norm(X_slice(:))^2;
end

rel_error = sqrt(nrm2/nrmX2);

end