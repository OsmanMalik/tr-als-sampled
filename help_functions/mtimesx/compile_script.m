% This script contains the suggestion by Ville-Veikko Wettenhovi on the File
% Exchange page for mtimesx for compiling on a Windows machine. See
% https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support?s_tid=answers_rc2-2_p5_MLT
% for details. Copy this into the folder containing all the other code and
% execute.
%
% For compilation on Linux, I had to use the following commands instead:
% >> blas_lib = '/usr/local/MATLAB/R2017a/bin/glnxa64/libmwblas.so'
% >> mex('-DDEFINEUNIX','-largeArrayDims', 'mtimesx.c',blas_lib)

libdir = 'mingw64';
comp = computer;
mext = mexext;
lc = length(comp);
lm = length(mext);
cbits = comp(max(1:lc-1):lc);
mbits = mext(max(1:lm-1):lm);
if( isequal(cbits,'64') || isequal(mbits,'64') )
compdir = 'win64';
largearraydims = '-largeArrayDims';
else
compdir = 'win32';
largearraydims = '';
end
lib_blas = [matlabroot '\extern\lib\' compdir '\' libdir '\libmwblas.lib'];
d = dir('mtimesx.m');
mname = [d.folder '/' d.name];
cname = [mname(1:end-2) '.c'];
mex(cname,largearraydims,lib_blas,'COMPFLAGS="$COMPFLAGS /openmp"');
