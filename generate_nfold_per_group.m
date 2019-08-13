function res = generate_nfold_per_group(label, nfold)
% This matlab code implements the random sampling per group
%
% inputs:
%        label -- vector with length L specifying labels of data set
%        nfold -- positive integer N specifying the number of folds to
%        separate the samples
%        
% outputs:
%        res -- ~(L/N)*N index matrix specifying spliting results
%
% created by Qianqian Shi on 07/07/2019, qqshi@mail.hzau.edu.cn

ulabel = unique(label);
counts = hist(label,ulabel)';

if nargin < 2
    nfold = 5;
    warning('Default nfold is 5.');
end

if min(counts) < nfold
    nfold = min(counts);
    warning(['The valud of nfold is forced to ',int2str(nfold),'.']);
end 

res = [];
flag = 0;

for i = 1:length(ulabel)
    temp_index = find(label == ulabel(i));
    num = max(floor(counts(i)/nfold),1);
    temp_index_rand = randsample(temp_index,counts(i));
    
    for j = 1:num
        tt = temp_index_rand((nfold*(j-1)+1):(nfold*j));
        res = [res; tt'];
    end
    if num*nfold ~= counts(i)
        temp_res = zeros(1,nfold);
        temp_res(1:(length(temp_index)-nfold*num)) = temp_index((nfold*num+1):end);
        if flag == 0
            res = [res; temp_res];
            flag = 1;
        else
            res = [res; fliplr(temp_res)];
            flag = 0;
        end
    end
end

