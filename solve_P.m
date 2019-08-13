function Pu = solve_P(X, Z, P)
% This matlab code implements l2,1 minimization method for solving
% discrimination matrix
%------------------------------
% min |P^T*X - P^T*XZ|_2,1
% s.t.    P*P^T = I; 
%--------------------------------
% inputs:
%        X -- G*N data matrix
%        Z -- N*N representation matrix
%        P -- G*N discrimination matrix
%        
% outputs:
%        Pu -- updated D*G discrimination matrix
%
% created by Qianqian Shi on 07/07/2019, qqshi@mail.hzau.edu.cn
    temp = P'*(X-X*Z);
    D = diag(0.5./sqrt(sum(temp.*temp))+eps);
    S = (X-X*Z)*D*(X-X*Z)';
    S = (S+S')/2;
    % St=Xtrain*Xtrain';
    % St=St+0.001*eye(size(St));
    [Pall, DS] = eig(S);
    [ds, ind] = sort(diag(DS), 'ascend');
    [ind2] = find(ds>10^-3);    
%     [ind2]=find(ds>0);  
    Pu = Pall(:,ind2);

end