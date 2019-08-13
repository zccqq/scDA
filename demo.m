%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% data preparation
clear;clc
load('data.mat');
K = 100; % free parameter: the number of nearest neighbors for adjacency graph
r1_lambda = 0.1; r2_lambda = 1; % tuning parameter: r1 and r2 specifying base and exponent respectively
d_percentage = 0.05; % the number of discriminants / the number of involved cells

%% Initializing variables
n = size(res_data,2);
lambda = r1_lambda*10^(r2_lambda-1);

%% Clustering with whole data set
Zi = initialize_Z(res_data, K); % initializing the representation graph Z
% [Z,E] = solve_Z(res_data, lambda, Zi); % identifying Z by LRR approach
[Z, P] = step_1(res_data, lambda, Zi); % identifying Z though alternate iteractions
W = (abs(Z)+abs(Z)')./2; % making symmetric matrix of Z
[K1, K2, K12, K22] = estimate_number_of_clusters(W, [2:25]); % estimating the number of clusters
group = spectralClustering(W,K1); % clustering
clustering_ari = adjrand(res_label, group); % calculating clustering accuracy by ARI

%% Clustering and classification with discovery and validation data sets
% Initializing variables
nfold = 5;
dim = max(floor(n/nfold*(nfold-1)*d_percentage),3);
% randomly sampling for cross validation
rng(1,'twister');
res = generate_nfold_per_group(res_label, nfold);
% clustering_aris = [];
classification_aris = [];

for i = 1:nfold
    % spliting data sets into discovery and validation cohorts
    temp = sort(res((find(res(:,i) > 0)),i));
    data_validation = res_data(:, temp);
    label_validation = res_label(temp);
    data_discovery = res_data(:,setdiff((1:n)',temp));
    label_discovery = res_label(setdiff((1:n)',temp));
    
    % main process: step 1
    Zi_discovery = initialize_Z(data_discovery,K);
    [Zf_discovery, Pf] = step_1(data_discovery, lambda, Zi_discovery);
%     W_discovery = (abs(Zf_discovery)+abs(Zf_discovery)')./2;
%     [K1, K2, K12, K22] = estimate_number_of_clusters(W_discovery, [2:25]);
%     group_discovery = spectralClustering(W_discovery, K1);
%     clustering_aris = [clustering_aris; adjrand(label_discovery, group_discovery)];

    % main process: step 2
    label_validation_predict = step_2(Pf, data_discovery, label_discovery, data_validation, dim);
    classification_aris = [classification_aris; adjrand(label_validation',label_validation_predict)];

end
% clustering_ari = mean(clustering_aris);
classification_ari = mean(classfication_aris);
