
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .helper_functions import x2fx

def MRE(A, X, Y):
    """
    Multivariate Richardson Extrapolation (PyTorch version).

    Parameters:
        A : array-like, shape (m, d)   – sparse basis (binary / integer exponents)
        X : array-like, shape (n, d)   – training inputs
        Y : array-like, shape (n,) or (n,1) – training outputs

    Returns:
        torch.Tensor (0-dim scalar): predicted f(0)
    """

    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64).flatten()
    A = torch.as_tensor(A, dtype=torch.float64)

    n_train, d = X.shape
    m = A.shape[0]

    nbrs = NearestNeighbors(n_neighbors=m).fit(X.cpu().numpy())
    idx = nbrs.kneighbors(np.zeros((1, d)), return_distance=False).flatten()

    X_sel = X[idx]      # (m, d)
    Y_sel = Y[idx]      # (m,)

    ep = 1e-16
    nX = (X_sel.max(dim=0).values - X_sel.min(dim=0).values) + ep  # (d,)
    nY = (Y_sel.max() - Y_sel.min()) + ep                          # 标量

    Xn = X_sel / nX
    Yn = Y_sel / nY

    V = x2fx(Xn, A)                                           # (m, m)
    eval_point = x2fx(torch.zeros((1, d), dtype=torch.float64), A)  # (1, m)

    sol = torch.linalg.lstsq(V, Yn.unsqueeze(1), driver='gelsd')
    coeffs = sol.solution  # (m, 1)

    mu = nY * (eval_point @ coeffs)  # (1,1)

    return mu.view(())   # 返回标量 tensor
