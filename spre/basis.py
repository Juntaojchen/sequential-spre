"""
Polynomial basis utilities for sparse polynomial Richardson extrapolation.

Three core primitives:

softplus
    Smooth, numerically stable positive-valued activation used to map
    unconstrained optimisation variables to positive hyperparameters.

x2fx
    Polynomial design matrix  V[i, j] = ∏_k X[i, k]^A[j, k].
    Used to build the trend (mean) function of the universal kriging model.

stepwise
    Generate the set of candidate polynomial monomials of the next total
    degree from the current multi-index set A.  Used in greedy basis
    selection (see selection.py).
"""

import torch


def softplus(x) -> torch.Tensor:
    """
    Numerically stable softplus activation: log(1 + exp(x)).

    Maps an unconstrained scalar or tensor to the positive reals.
    Used to ensure kernel hyperparameters (amplitude σ², lengthscale ℓ)
    remain positive during gradient-based optimisation.

    Parameters
    ----------
    x : float or torch.Tensor

    Returns
    -------
    torch.Tensor
        softplus(x) ∈ (0, ∞)
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64)
    return torch.nn.functional.softplus(x)


def x2fx(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float64)
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float64, device=X.device)

    base = X.unsqueeze(1)               # (n, 1, d)
    exp  = A.unsqueeze(0).to(X.dtype)  # (1, m, d)
    return torch.prod(torch.pow(base, exp), dim=-1)  # (n, m)


def stepwise(A: torch.Tensor, order: int) -> torch.Tensor:

    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.int64)

    if A.numel() == 0:
        d = A.shape[1] if A.dim() == 2 else 0
        return torch.empty((0, d), dtype=torch.int64, device=A.device)

    _n_models, d = A.shape
    device = A.device

    mask      = (A.sum(dim=1) == (order - 1))
    A_filtered = A[mask]                                       # (k, d)

    if A_filtered.shape[0] == 0:
        return torch.empty((0, d), dtype=A.dtype, device=device)

    eye_d     = torch.eye(d, dtype=A.dtype, device=device)
    expanded  = A_filtered.unsqueeze(1) + eye_d.unsqueeze(0)  # (k, d, d)
    candidates = expanded.reshape(-1, d)                       # (k·d, d)
    return torch.unique(candidates, dim=0)
