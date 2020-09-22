function cores = TRdecomp_ranks(A, ranks)
%%% Converts a tensor A in full format into TR-format with target ranks
%%% specified in the vector ranks.

% This code is a modification of the function TRdecomp.m accompanying the
% paper "On Algorithms for and Computing with the Tensor Ring
% Decomposition" by Oscar Mickelin and Sertac Karaman. Their code is
% available at https://github.com/oscarmickelin/tensor-ring-decomposition

    s = size(A);
    d = length(s);
    cores = cell(d,1);
    C = A;
    rold = 1;
    
    %initial step
    n = s(1);
    C = reshape(C, [rold*n, numel(C)/(rold*n)]);
    [U,S,V] = svd(C,'econ');
    r0 = ranks(d);
    rnew = r0*ranks(1);
    

    U=U(:,1:rnew);
    S=S(1:rnew, 1:rnew);
    V = V(:, 1:rnew);

    Unew = zeros([rold*r0, n, rnew/r0]);
    for index = 1:r0
        Unew(index, :, :) = U(:, ( (index-1)*rnew/r0 + 1):(index*rnew/r0));
    end
    cores{1} = Unew;
    C = (S*V');

    %reshape C into a 3-tensor
    Cnew = zeros([rnew/r0, numel(C)/(rnew), r0]);
    for index = 1:r0
        Cnew(:, :, index) = C(( (index-1)*(rnew/r0) + 1):(index*(rnew/r0)), :);
    end
    C = Cnew;
    %%%reshape to merge i_d and alpha_0
    %make it full
    C = reshape(C, [rnew/r0 s(2:end) r0]);
    %merge last two indices
    C = reshape(C, [rnew/r0 s(2:end-1) s(end)*r0]);
    %and take first unfolding of this
    C = reshape(C, [rnew/r0, numel(C)/(rnew/r0)]);
    rold = rnew/r0;    
    %%%
    %intermediate steps
    for k =2:d-1
        n = s(k);
        C = reshape(C, [rold*n, numel(C)/(rold*n)]);        
        [U,S,V] = svd(C,'econ');
        rnew = ranks(k);
        U=U(:,1:rnew);
        S=S(1:rnew, 1:rnew);
        V = V(:, 1:rnew);
        cores{k} = reshape(U, [rold, n, rnew]);
        C = (S*V');
        rold = rnew;
    end
    
    C = reshape(C, [rold, s(end), r0]);
    cores{d} = C;
   
end