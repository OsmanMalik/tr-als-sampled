% Experiment 4
%
% The purpose of this script is to essentially repeat what we did in
% experiment 1, but on real data.
%
% This script was used to generate the results on real data in the paper.

%include_toolboxes

% Settings: General experiment 
dataset = "pavia"; % Which dataset to use
R = 10;
no_it = 100;
save_snap = false; % We set this to true for the coil dataset to save intermediate images of the Red Truck to be able to show visually the difference between the decompositions. 
run_TR_SVD = false;
run_TR_SVD_Rand = false;

%% Load and preprocess

if strcmp(dataset, 'synthetic')
    sz = [100 100 100];
    N = length(sz);
    density = .01;
    X = sptenrand(sz, density);
else
    if strcmp(dataset, 'uber')
        tensor_path = "D:\data_sets\tensors\Uber Pickups\uber.tns";
        tensor_type = 'sparse';
    elseif strcmp(dataset, 'nips')
        tensor_path = "D:\data_sets\tensors\NIPS Publications\nips.tns";
        tensor_type = 'sparse';
    elseif strcmp(dataset, 'crime-comm') % Size: 6186 x 24 x 77 x 32
        tensor_path = "D:\data_sets\tensors\Chicago Crime\chicago-crime-comm.tns";
        tensor_type = 'sparse';
    elseif strcmp(dataset, 'crime-geo') % Size: 6185 x 24 x 380 x 395 x 32
        tensor_path = "D:\data_sets\tensors\Chicago Crime\chicago.tns";
        tensor_type = 'sparse';
    elseif strcmp(dataset, 'nell-mini')
        tensor_path = "D:\data_sets\tensors\NELL-2\nell-2.tns";
        mini_size = 500;
        tensor_type = 'sparse';
    elseif strcmp(dataset, 'pavia')
        tensor_path = "D:\data_sets\hyperspectral_imaging\PaviaU.mat";
        load(tensor_path)
        X = paviaU;
        tensor_type = 'dense';
    elseif strcmp(dataset, 'dc')
        % See this post for info on loading data: https://www.mathworks.com/matlabcentral/answers/449291-how-to-display-hyperspectral-image-washington-dc
        tensor_path = "D:\data_sets\hyperspectral_imaging\dc.tif";
        data1 = imread(tensor_path);
        X = double(data1);
        tensor_type = 'dense';
    elseif strcmp(dataset, 'bench')
        tensor_path = 'D:\data_sets\videos\Man Sitting on a Bench\bench.mat';
        load(tensor_path);
        %tensor_path = 'D:\data_sets\videos\Man Sitting on a Bench\Man Sitting On a Bench.mp4';
        %vid = VideoReader(tensor_path);
        %X = zeros(1080, 1920, 364);
        %for k = 1:size(X,3)
        %    X(:,:,k) = mean(read(vid, k),3);
        %end
        tensor_type = 'dense';
    elseif strcmp(dataset, 'cat')
        tensor_path = 'D:\data_sets\videos\Cat\tabby_cat.mat';
        load(tensor_path);
        tensor_type = 'dense';
    elseif strcmp(dataset, 'coil') || strcmp(dataset, 'coil reshape')
        path = "D:\data_sets\images\coil-100";
        flist = dir(path);
        no_obj = 1;
        pic_per_obj = 72;
        strt = 5;
        X = zeros(128, 128, 3, no_obj*pic_per_obj);
        cnt = 0;
        for obj = 1:no_obj
            for pic = 1:pic_per_obj
                X(:, :, :, (obj-1)*pic_per_obj + pic) = imread(string(path) + "/" + flist(5+cnt).name);
                cnt = cnt+1;
            end
            obj;
        end
        tensor_type = 'dense';
        X = double(X);
        if strcmp(dataset, 'coil reshape')
            newsize = cell(16, 1);
            for j = 1:14
                newsize{j} = 2;
            end
            newsize{15} = 3;
            newsize{16} = 72;
        end
        X = reshape(X, newsize{:});
        X = permute(X, [1 8 2 9 3 10 4 11 5 12 6 13 7 14 15 16]);
    elseif strcmp(dataset, 'coil compressed')
        path = "D:\data_sets\images\coil-100-downsampled\compressed_coil_100.mat";
        load(path)
        X = img_array;
        tensor_type = 'dense';
    elseif strcmp(dataset, 'sin')
        x = linspace(-1,1,4^10);
        y = (x+1).*sin(100*(x+1).^2);
        X = reshape(y, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
		tensor_type = 'dense';
    elseif strcmp(dataset, 'airy')
        x = linspace(.01,100,4^10);
        y = x.^(-1/4).*sin(2/3 * x.^(3/2));
        X = reshape(y, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
		tensor_type = 'dense';
    elseif strcmp(dataset, 'chirp')
        x = linspace(0.01,1,4^10);
        y = sin(4./x).*cos(x.^2);
        X = reshape(y, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
		tensor_type = 'dense';
    elseif strcmp(dataset, 'weight')
        path = "D:\data_sets\neural_network_weights\weight.mat";
        load(path);
        dict = {4, 4, 4, 4, 4, 4, 4, 4, 6, 3};
        X = double(reshape(X, dict{:}));
        tensor_type = 'dense';
    end
    if strcmp(tensor_type, 'sparse')
        mat = importdata(tensor_path);
        N = size(mat, 2) - 1;
        X = sptensor(mat(:, 1:N), mat(:, end));
        if strcmp(dataset, 'nell-mini')
            X = X(1:mini_size, 1:mini_size, 1:mini_size);
        end
        sz = size(X);
        X = double(X);
    elseif strcmp(tensor_type, 'dense')
        sz = size(X);
        N = length(sz);
    end
end
normX = norm(tensor(X));

%% 
ranks = R*ones(1, N); % Target ranks
target_acc = 1.10; % Target accuracy (1+epsilon)
no_trials = 1; % Number of experiment trials for averaging
verbose = true;

% Settings: TR-ALS 
tol = 1e-3; % Tolerance to pass to tr_als when determining number of iterations

% Settings: TR-ALS-Sampled
J_init = 2*max(ranks)^2;
J_inc = 100;

% Settings: rTR-ALS
%K_init = round(max(sz)/10);
K_init = 2;
%K_inc = round(max(sz)/20);
K_inc = 1;        

% Settings: TR-SVD-Rand
oversamp = 10; % Following Remark 1 by Ahmadi-Asl et al. (2020)

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
    if save_snap && tr == 1
        Y_TR_ALS = Y;
    end
        
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
    if save_snap && tr == 1
        Y_TR_ALS_Sampled = Y;
    end
    fprintf(' Done!\n');

    % Run rTR-ALS
    fprintf('\tRunning rTR-ALS for trial = %d', tr)
    K = K_init;
    while true
        tic_exp = tic; 
        cores = rtr_als(X, ranks, K*ones(size(sz)), 'tol', 0, 'maxiters', no_it, 'verbose', verbose);
        time_rTR_ALS{tr, m} = [time_rTR_ALS{tr, m}; toc(tic_exp)];
        Y = cores_2_tensor(cores);
        rel_error_rTR_ALS{tr, m} = [rel_error_rTR_ALS{tr, m}; norm(Y(:)-X(:))/normX];
        if rel_error_rTR_ALS{tr, m}(end)/rel_error_TR_ALS(tr, m) < target_acc
            break
        end
        K = K + K_inc;
        fprintf('.');
    end
    if save_snap && tr == 1
        Y_rTR_ALS = Y;
    end
    fprintf(' Done!\n');

    % Run TR-SVD
    if run_TR_SVD
        fprintf('\tRunning TR-SVD for trial = %d', tr)
        tic_exp = tic;
        cores = TRdecomp_ranks(X, ranks);
        time_TR_SVD(tr, m) = toc(tic_exp);
        Y = cores_2_tensor(cores);
        rel_error_TR_SVD(tr, m) = norm(Y(:)-X(:))/normX;
        if save_snap && tr == 1
            Y_TR_SVD = Y;
        end
        fprintf(' Done!\n');
    else
        rel_error_TR_SVD = nan;
        time_TR_SVD = nan;
    end

    % Run TR-SVD-Rand
    if run_TR_SVD_Rand
        fprintf('\tRunning TR-SVD-Rand for trial = %d', tr)
        tic_exp = tic;
        cores = tr_svd_rand(X, ranks, oversamp);
        time_TR_SVD_Rand(tr, m) = toc(tic_exp);
        Y = cores_2_tensor(cores);
        rel_error_TR_SVD_Rand(tr, m) = norm(Y(:)-X(:))/normX;
        if save_snap && tr == 1
            Y_TR_SVD_Rand = Y;
        end
        fprintf(' Done!\n');
    else
        rel_error_TR_SVD_Rand = nan;
        time_TR_SVD_Rand = nan;
    end
end

fprintf('\n');

% Save stuff
save(fname)  % Just save everything...
% save(fname, 'NO_IT', ...
%     'rel_error_TR_ALS', ...
%     'rel_error_TR_ALS_Sampled', ...
%     'rel_error_rTR_ALS', ...
%     'rel_error_TR_SVD', ...
%     'rel_error_TR_SVD_Rand', ...
%     'time_TR_ALS', ...
%     'time_TR_ALS_Sampled', ...
%     'time_rTR_ALS', ...
%     'time_TR_SVD', ...
%     'time_TR_SVD_Rand')
