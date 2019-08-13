function [Zf, Pf] = step_1(X, lambda, Zi, maxIter)
% This matlab code implements the main objective function of UDA model
%------------------------------
% min |Z|_*+lambda*|P^T*X - P^T*XZ|_2,1
%        P*P^T = I; 
% s.t.   Z^T*1 = 1; 
%        Z(i,j) = 0 for the complement of Zi;
%--------------------------------
% inputs:
%        X -- G*N data matrix
%        lambda -- numeric specifying tuning parameter
%        Zi -- N*N initialized representation matrix
%        maxIter -- positive integer specifying the maximum times of iterations
%        
% outputs:
%        Zf -- optimal N*N representation matrix
%        Pf -- optimal G*D discrimination matrix
%
% created by Qianqian Shi on 07/07/2019, qqshi@mail.hzau.edu.cn

if nargin < 4
    maxIter = 100;
end

%% Initializing optimization variables
Zf = [];
Pf = [];
obj_final = inf;
P = eye(size(X,1));

for iter =1:maxIter
    
    %% fixing P, solve Z
    X_input = P'*X;
    [Z, E] = solve_Z(X_input, lambda, Zi);    
    
    %% fixing Z, solve P
    P = solve_P(X, Z, P); 
    
    %% Is or not Coveraged
    M = P'*(X-X*Z);
    obj = sum(sqrt(sum((M.*M),1))); 
%     if abs(obj-obj_final)/obj < 0.01 && norm(P-Pf) < 0.01
    if abs(obj-obj_final)/obj < 0.001        
        break;
    else
        Zf = Z;
        Pf = P;
        obj_final = obj;        
    end
    
    iter = iter + 1;
    
end
end