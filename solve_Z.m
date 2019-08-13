function [Z,E] = solve_Z(X, lambda, Zi, maxIter)
% This matlab code implements linearized ADM method for solving
% representation graph
%------------------------------
% min |Z|_*+lambda*|E|_2,1
%        P^T*X = P^T*XZ+E; 
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
%        Z -- N*N representation matrix
%        E -- G*N sparse error matrix
%
% created by Qianqian Shi on 07/07/2019, qqshi@mail.hzau.edu.cn
%

%clear global;

addpath PROPACK;
if nargin < 4
    maxIter = 1000;
end

rho = 1.1;
normfX = norm(X,'fro');
tol1 = 1e-4;
tol2 = 1e-5;
[d n] = size(X);
max_mu = 1e30;
mu = 1e-6;
norm2X = norm(X,2);
norm21 = norm(ones(1,n),2);
eta = norm2X*norm2X + norm21*norm21 + 1;
% eta = norm2X*norm2X + 1;

opt.tol = tol2;% precision for computing the partial SVD
opt.p0 = ones(n,1);
%% Initializing optimization variables
% intialize
J = zeros(n,n);
Z = zeros(n,n);
E = sparse(d,n);

Y1 = zeros(d,n);
Y2 = zeros(1,n);
Y3 = zeros(n,n);

sv = 5;
svp = sv;

%% Start main loop
convergenced = 0;
iter = 0;

for iter = 1:maxIter
    Em = E;
    Zm = Z;
      
    temp = Z + Y3/mu;
    %[U,sigma,V] = svd(temp,'econ');
    [U,sigma,V] = lansvd(temp,n,n,sv,'L',opt);
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    temp = X-X*Z+Y1/mu;
    E = solve_l1l2(temp,lambda/mu);
    
    H = -X'*(X-X*Z-E+Y1/mu)-ones(n,1)*(ones(1,n)-ones(1,n)*Z+Y2/mu)+(Z-J+Y3/mu);
% H = -X'*(X-X*Z-E+Y1/mu)+(Z-J+Y3/mu);
    M = Z - H/eta;
    Z = Zi.*M;
    
    xmaz = X-X*Z;
    leq1 = xmaz-E;
    leq2 = ones(1,n)-ones(1,n)*Z;
    leq3 = Z-J;
    relChgZ = norm(Z-Zm,'fro')/normfX;
    relChgE = norm(E-Em,'fro')/normfX;
    relChg = max(relChgE,relChgZ);    
    recErr = norm(leq1,'fro')/normfX;
    
    %% Is or not Coveraged
    convergenced = recErr < tol1 && relChg < tol2;
    if convergenced
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end


function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end