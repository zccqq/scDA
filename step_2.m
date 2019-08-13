function label_validation_predict = step_2(P, X_discovery, label_discovery, X_validation, dim)
% This matlab code implements the prediction of new samples in UDA
%
% inputs:
%        P -- G*N discrimination matrix identified from discovery data set
%        X_discovery -- G*N data matrix of discovery cohort
%        label_discovery -- vector specifying labels of discovery cohort
%        X_validation -- G*M data matrix of validation cohort
%        dim -- positive integer specifying the number of discriminants for
%        label prediction
%        
% outputs:
%        label_validation_predict -- vector specifying predicted labels of validation cohort
%
% created by Qianqian Shi on 07/07/2019, qqshi@mail.hzau.edu.cn

if nargin < 5
    dim = size(X_discovery, 2)*0.05;
end

dim = min(size(P,2), dim);
mdl = ClassificationKNN.fit(X_discovery'*P(:,1:dim),label_discovery','NumNeighbors',1);
label_validation_predict= predict(mdl,X_validation'*P(:,1:dim));
  
end

