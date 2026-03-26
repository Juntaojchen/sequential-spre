import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .basis import x2fx


def mre(
    A: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:

    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64).flatten()
    A = torch.as_tensor(A, dtype=torch.float64)

    n_train, d = X.shape
    m          = A.shape[0]

    nbrs = NearestNeighbors(n_neighbors=m).fit(X.cpu().numpy())
    idx  = nbrs.kneighbors(
        np.zeros((1, d)), return_distance=False
    ).flatten()                                                  # (m,)

    X_sel = X[idx]    # (m, d)
    Y_sel = Y[idx]    # (m,)

    ep  = 1e-16
    nX  = (X_sel.max(dim=0).values - X_sel.min(dim=0).values) + ep  # (d,)
    nY  = float((Y_sel.max() - Y_sel.min()).item()) + ep

    Xn  = X_sel / nX
    Yn  = Y_sel / nY

    V          = x2fx(Xn, A)                                         # (m, m)
    eval_point = x2fx(torch.zeros(1, d, dtype=torch.float64), A)     # (1, m)

    sol    = torch.linalg.lstsq(V, Yn.unsqueeze(1), driver="gelsd")
    coeffs = sol.solution                                             # (m, 1)

    mu = nY * (eval_point @ coeffs)   # (1, 1)
    return mu.view(())                # scalar
