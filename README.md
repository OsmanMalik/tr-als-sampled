# A Sampling-Based Method for Tensor Ring Decomposition
This repo provides code used in the experiments of our paper 

> Osman Asif Malik, Stephen Becker. *A Sampling-Based Method for Tensor Ring Decomposition*. Proceedings of the 38th International Conference on Machine Learning, PMLR 139:7400-7411, 2021.

Our paper is available in the [Proceedings of Machine Learning Research](http://proceedings.mlr.press/v139/malik21b.html).

## Referencing this code
If you use our code in any of your own work, please reference our paper:

```
@InProceedings{pmlr-v139-malik21b,
  title = {A Sampling-Based Method for Tensor Ring Decomposition},
  author = {Malik, Osman Asif and Becker, Stephen},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {7400--7411},
  year = {2021},
  volume = {139},
  series = {Proceedings of Machine Learning Research},
  month = {18--24 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v139/malik21b/malik21b.pdf},
  url = {http://proceedings.mlr.press/v139/malik21b.html},
}
```

We have done our best to include relevant references in our code, so please also cite those works as appropriate.

## Some further details
The script **tr_als_sampled.m** is a Matlab implementation of our proposed TR-ALS-Sampled method for sampled tensor ring decomposition. 
The experiments that appear in our paper were conducted using the following scripts:
- **experiment1.m:** Was used to run the experiments on synthetic data in Section 5.1.1.
- **experiment4.m:** Was used to run the experiments in Sections 5.1.2 and 5.1.3. 
- **experiment5.m:** Was used to run the rapid feature extraction experiment in Section 5.2. 

There are other "experiment" scripts in this repo, but we did not use results from those scripts in our paper. However, they may still may be of interest:
- **experiment1b.m:** This script runs synthetic experiments similar to those in experiment1.m, but with the *rank* varying instead of the size of each mode.
- **experiment1c.m:** This script runs synthetic experiments similar to those in experiment1.m, but with the *number of modes* varying instead of the size of each mode.
- **experiment3.m, experiment3_rtr_als.m:** These two scripts were used to do some preliminary testing on sparse data. Separate scripts for TR-ALS-Sampled and rTR-ALS were necessary since they required different preprocessing. 

Implementations of the methods we compare to in our paper are available in the following files:
- **tr_als.m**: The standard TR-ALS algorithm.
- **rtr_als.m**: The rTR-ALS algorithm.
- **TRdecomp_ranks.m**: This is what we call TR-SVD in our paper. This a modified version of TRdecomp.m available at https://github.com/oscarmickelin/tensor-ring-decomposition.
- **tr_svd_rand.m**: This is the randomized variant of TR-SVD which we call TR-SVD-Rand in our paper.

## Requirements
This code requires Matlab version R2020b or later; see note below.

Portions of this code also require the following:
- The Statistics and Machine Learning Toolbox which is an official Matlab toolbox from Mathworks.
- Tensor Toolbox for MATLAB. It is available at https://www.tensortoolbox.org/ and at https://gitlab.com/tensors/tensor_toolbox.
- The function **normTR.m** for efficient computation of the norm of tensors in TR format, available at https://github.com/oscarmickelin/tensor-ring-decomposition.

Previous versions of this repo required [**mtimesx**](https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support) for doing batched matrix multiplication.
Since Matlab introduced its own function [**pagemtimes**](https://www.mathworks.com/help/matlab/ref/pagemtimes.html) for doing batched matrix multiplication in version R2020b we have now switched over to using **pagemtimes** instead of **mtimesx**.
In addition to being easier to use since it does not require compilation, **pagemtimes** also usually ran faster than **mtimesx** in some tests we did.
The last version of this repo that did use **mtimesx** is the latest version committed on Oct 12, 2021.

## Installation
The Statistics and Machine Learning Toolbox can be installed via Matlab's Add-Ons interface.
The remaining dependencies just need to be downloaded and made accessible from within Matlab, e.g. via the Set Path function.

## Author contact information
Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on our paper. I can be reached at osman.malik@colorado.edu.
