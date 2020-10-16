# A Sampling Based Method for Tensor Ring Decomposition
This repo provides code used in the experiments of our paper

[ADD PAPER TITLE]

It is available at [ADD LINK]

## Some further details
The script **tr_als_sampled.m** is a Matlab implementation of our proposed TR-ALS-Sampled method for sampled tensor ring decomposition. The script **experiment1.m** was used to run the experiments on synthetic data, and **experiment4.m** was used to run experiments on real data. Implementations of the methods we compare to in our paper are available in the following files:
- **tr_als.m**: The standard TR-ALS algorithm.
- **rtr_als.m**: The rTR-ALS algorithm.
- **TRdecomp_ranks.m**: This is what we call TR-SVD in our paper. This a modified version of TRdecomp.m available at https://github.com/oscarmickelin/tensor-ring-decomposition.
- **tr_svd_rand.m**: This is the randomized variant of TR-SVD which we call TR-SVD-Rand in our paper.

## Requirements
Our **tr_als_sampled.m** requires mtimesx, which is available at https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support. We also provide a copy of this software in the folder help_functions/mtimesx/ of this repo. Please see the license file in that folder for license details on mtimesx.

Portions of this code also require the following code:
- Tensor Toolbox for MATLAB. It is available at https://www.tensortoolbox.org/.
- The function **normTR.m** for efficient computation of the norm of tensors in TR format, available at https://github.com/oscarmickelin/tensor-ring-decomposition.

## Installation
How to install mtimesx depends on if you are on a Windows or Linux machine. The script **compile_script.m** in help_functions/mtimesx has code for compiling on a Windows machine. The comments in that script also contain pointers for compiling on a Linux machine.

The other dependencies listed under Requirements should require no installation. The rest of the code in this repo should work as is. We ran this code in Matlab 2017a, so this version and newer should work fine.

## Referencing this code
If you use our code in any of your own work, please reference our paper:

[ADD BIB CODE]

## Author contact information
Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on our paper. I can be reached at osman.malik@colorado.edu.
