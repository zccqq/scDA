function Zi = initialize_Z(X, k, method, is_diag)
% This matlab code implements initialization of representation graph
%
% inputs:
%        X -- G*N data matrix
%        k -- positive integer specifying the number of nearest neigbors
%        method -- distance metrics in knnsearch. default is 'euclidean'
%        is_diag -- logical value. default is false
% outputs:
%        Zi -- N*N initialized representation matrix
%
% created by Qianqian Shi on 07/07/2019, qqshi@mail.hzau.edu.cn

if nargin < 3
    method = 'euclidean';
end
if nargin < 4
    is_diag = false;
end
n = size(X,2);
Zi = zeros(n,n);

window_id = knnsearch(X',X','k',k,'distance',method);
for i = 1:size(window_id)
    if (~is_diag)
        Zi(i, window_id(i,:)) = 1; % the diag values equal one when is_diag == false
    else
        Zi(i, window_id(i,2:k)) = 1; % the diag values equal one when is_diag == true
    end
end

end

