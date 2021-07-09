% The purpose of this script is to demonstrate how the TR decomposition can
% be used for rapid feature extraction in a simple classification task.
%
% This script was used to run the rapid feature extraction experiments in
% Section 5.2 of our paper.

method = "tr_als";

path = "D:\data_sets\images\coil-100-downsampled\compressed_coil_100.mat";
load(path)
X = img_array;
ranks = [5 5 5 5];
tol = 1e-4;
maxiters = 20;
sz = size(X);
exp5_tic = tic;
switch method
    case "tr_als"
        cores = tr_als(X, ranks, 'tol', tol, 'conv_crit', 'relative error', 'maxiters', maxiters, 'verbose', true);
    case "rtr_als"
        K = [20 20 3 200];
        cores = rtr_als(X, ranks, K, 'tol', tol, 'conv_crit', 'relative error', 'maxiters', maxiters, 'verbose', true);
    case "tr_als_sampled"
        J = 2000;
        cores = tr_als_sampled(X, ranks, J*ones(size(sz)), 'tol', tol, 'conv_crit', 'relative error', 'maxiters', maxiters, 'resample', true, 'verbose', true);
    case "tr_svd"
        cores = TRdecomp_ranks(X, ranks);
    case "tr_svd_rand"
        oversamp = 10;
        cores = tr_svd_rand(X, ranks, oversamp);
end
exp5_toc = toc(exp5_tic);

feat_mat = reshape(permute(cores{4}, [2 1 3]), 7200, 25);
Mdl = fitcknn(feat_mat, class_array, 'numneighbors', 1);
cvmodel = crossval(Mdl);
loss = kfoldLoss(cvmodel, 'lossfun', 'classiferror');
accuracy = 1 - loss;
fprintf('Accuracy is %.4f\n', accuracy);
fprintf('Decomposition time was %.4f\n', exp5_toc);
save('experiment5_results.mat', 'accuracy', 'exp5_toc');
