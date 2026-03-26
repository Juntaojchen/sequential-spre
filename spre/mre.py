"""
Multivariate Richardson Extrapolation (MRE).

Classical deterministic polynomial extrapolation to the origin using the
*m* nearest training points, where m = |A| is the polynomial basis size.
Provides a deterministic baseline (no GP, no hyperparameter tuning).

Algorithm
---------
1. Find the m points in X closest to the origin (ℓ₂ distance).
2. Normalise those m points and their corresponding observations.
3. Solve the square polynomial system  V @ c = y_norm  via least-squares.
4. Evaluate the polynomial at the normalised origin to get μ = ŷ(0).

References
----------
Richardson, L. F. (1911). The approximate arithmetical solution by finite
differences of physical problems involving differential equations.
Phil. Trans. R. Soc. London A, 210, 307–357.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .basis import x2fx


def mre(
    A: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """
    Predict f(0) by Multivariate Richardson Extrapolation.

    Parameters
    ----------
    A : (m, d)  multi-index set defining the polynomial basis.
    X : (n, d)  training inputs (raw, not normalised).
    Y : (n,) or (n, 1)  training outputs (raw).

    Returns
    -------
    mu : scalar tensor  predicted f(0) in original scale.

    Notes
    -----
    Requires m ≤ n (more data points than basis terms).
    The least-squares solve uses LAPACK's divide-and-conquer SVD driver
    (``gelsd``) for numerical robustness when the system is ill-conditioned.
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64).flatten()
    A = torch.as_tensor(A, dtype=torch.float64)

    n_train, d = X.shape
    m          = A.shape[0]

    # ── 1. Select m nearest neighbours to the origin ──────────────────────────
    nbrs = NearestNeighbors(n_neighbors=m).fit(X.cpu().numpy())
    idx  = nbrs.kneighbors(
        np.zeros((1, d)), return_distance=False
    ).flatten()                                                  # (m,)

    X_sel = X[idx]    # (m, d)
    Y_sel = Y[idx]    # (m,)

    # ── 2. Normalise (max-min, matching the JAX reference) ────────────────────
    ep  = 1e-16
    nX  = (X_sel.max(dim=0).values - X_sel.min(dim=0).values) + ep  # (d,)
    nY  = float((Y_sel.max() - Y_sel.min()).item()) + ep

    Xn  = X_sel / nX
    Yn  = Y_sel / nY

    # ── 3. Build polynomial system and solve ──────────────────────────────────
    V          = x2fx(Xn, A)                                         # (m, m)
    eval_point = x2fx(torch.zeros(1, d, dtype=torch.float64), A)     # (1, m)

    sol    = torch.linalg.lstsq(V, Yn.unsqueeze(1), driver="gelsd")
    coeffs = sol.solution                                             # (m, 1)

    # ── 4. Evaluate at origin in original scale ───────────────────────────────
    mu = nY * (eval_point @ coeffs)   # (1, 1)
    return mu.view(())                # scalar
