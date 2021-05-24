# -*- coding: utf-8 -*-

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.sparse

from sklearn.neighbors import KNeighborsClassifier

import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scipy.io as sio


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )
    return g


def initialize_Z(X, n_neighbors, metric='euclidean', is_diag=False):
    
    U = PCA(n_components=20, random_state=0).fit_transform(X.T)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(U)
    
    if is_diag:
        Zi = nbrs.kneighbors_graph(U)
    else:
        Zi = nbrs.kneighbors_graph(U) - scipy.sparse.eye(U.shape[0])
    
    return Zi.toarray()


def solve_l1l2(W, lambda1):
    
    NW = torch.norm(W, 2, dim=0, keepdim=True)
    E = (NW > lambda1) * (NW - lambda1) / NW
    
    return E * W


def solve_Z(X, lambda1, Zi, dev, maxIter=1000):
    
    device = torch.device(dev)
    
    rho = 1.1
    normfX = torch.norm(X, 'fro')
    tol1 = 1e-4
    tol2 = 1e-5
    d, n = X.shape
    n = torch.tensor(n).to(device)
    max_mu = 1e30
    mu = 1e-6
    norm2X = torch.norm(X, 2)
    eta = norm2X ** 2 + n + 1
    zero_tensor = torch.Tensor([0]).to(device)
    
    # Initializing optimization variables
    # intialize
    J = torch.zeros(n, n).to(device)
    Z = torch.zeros(n, n).to(device)
    E = torch.zeros(d, n).to(device)
    
    Y1 = torch.zeros(d, n).to(device)
    Y2 = torch.zeros(1, n).to(device)
    Y3 = torch.zeros(n, n).to(device)
    
    sv = 5
    svp = sv
    
    # Start main loop
    converged = 0
    Iter = 0
    
    for Iter in range(maxIter):
        Em = E
        Zm = Z
        
        temp = Z + Y3 / mu
        
        U, sigma, V = torch.svd(temp)
        svp = torch.sum(sigma > 1 / mu)
        
        if svp < sv:
            sv = torch.minimum(svp + 1, n)
        else:
            sv = torch.minimum(svp + torch.round(0.05 * n), n)
        
        if svp >= 1:
            sigma = sigma[:svp] - 1 / mu
        else:
            svp = 1
            sigma = zero_tensor
        
        J = torch.matmul(torch.matmul(U[:, :svp], torch.diag(sigma)), V[:, :svp].T)
        
        temp = X - torch.matmul(X, Z) + Y1 / mu
        E = solve_l1l2(temp, lambda1 / mu)
        
        H = - torch.matmul(X.T, (X - torch.matmul(X, Z) - E + Y1 / mu)) - (1 - torch.sum(Z, axis=0, keepdims=True) + Y2 / mu) + (Z - J + Y3 / mu)
        M = Z - H / eta
        Z = Zi * M
        
        xmaz = X - torch.matmul(X, Z)
        leq1 = xmaz - E
        leq2 = 1 - torch.sum(Z, axis=0, keepdims=True)
        leq3 = Z - J
        relChgZ = torch.norm(Z - Zm, 'fro') / normfX
        relChgE = torch.norm(E - Em, 'fro') / normfX
        relChg = torch.max(relChgE, relChgZ)
        recErr = torch.norm(leq1, 'fro') / normfX
        
        print("Iter", Iter, "relChg", relChg.item(), "recErr", recErr.item(), flush=True)
        
        converged = relChg < tol1 and recErr < tol2
        
        if converged:
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            Y3 = Y3 + mu * leq3
            mu = min(max_mu, mu * rho)
    
    return Z, E


def solve_P(X, Z, P):
    
    eps = 1e-14
    temp = torch.matmul(P.T, X - torch.matmul(X, Z))
    D = torch.diag(0.5 / torch.norm(temp, 2, dim=0) + eps)
    S = torch.matmul(torch.matmul(X - torch.matmul(X, Z), D), (X - torch.matmul(X, Z)).T)
    S = (S + S.T) / 2
    DS, Pall = torch.eig(S, eigenvectors=True)
    Pu = Pall[:, DS[:, 0] > 10^-3]
    
    return Pu


def step_1(X, lambda1, Zi, maxIter, dev):
    
    if dev is None or dev == "cuda":
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    
    device = torch.device(dev)
    
    X = torch.Tensor(X).type(torch.float32).to(device)
    Zi = torch.Tensor(Zi).type(torch.float32).to(device)
    
    obj_final = float('inf')
    P = torch.eye(X.shape[0]).to(device)
    
    for Iter in range(maxIter):
        
        # fixing P, solve Z
        X_input = torch.matmul(P.T, X)
        Z, E = solve_Z(X_input, lambda1, Zi, dev)  
        
        # fixing Z, solve P
        P = solve_P(X, Z, P)
        
        # Is or not converged
        M = torch.matmul(P.T, X - torch.matmul(X, Z))
        obj = torch.sum(torch.norm(M, 2, dim=0))
        
        print("Iter", Iter, "obj", obj.item(), flush=True)
        
        if torch.abs(obj - obj_final) / obj < 0.001:
            break
        else:
            Zf = Z
            Pf = P
            obj_final = obj
    
    return Zf.cpu().numpy(), Pf.cpu().numpy()


def step_2(P, X_discovery, label_discovery, X_validation, dim=None):
    
    if dim is None:
        dim = X_discovery.shape[1] // 20
    
    dim = min(P.shape[1], dim)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(np.matmul(X_discovery.T, P[:, :dim]), label_discovery.T)
    label_validation_predict = neigh.predict(np.matmul(X_validation.T, P[:, :dim]))
    
    return label_validation_predict


if __name__ == '__main__':
    
    res_data = sio.loadmat("../data.mat")['res_data']
    res_label = sio.loadmat("../data.mat")['res_label'][:,0]
    
    Zi = initialize_Z(res_data, n_neighbors=100, metric='euclidean', is_diag=False)
    
    Zf, Pf = step_1(X=res_data, lambda1=0.1, Zi=Zi, maxIter=2, dev='cuda')
    
    Z = (np.abs(Zf) + np.abs(Zf.T)) / 2
    
    Z_resolution = 2.5
    
    g_Z = get_igraph_from_adjacency(Z)
    part = la.find_partition(g_Z, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter = Z_resolution, seed=0)
    y_pred = np.array(part.membership)
    
    print("predicted cluster number:", len(set(y_pred)))
    print("ARI:", adjusted_rand_score(res_label, y_pred))
    print("NMI:", normalized_mutual_info_score(res_label, y_pred))



















