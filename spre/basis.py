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
    """
    Polynomial design matrix (monomial feature expansion).

    Computes:
        V[i, j] = ∏_{k=1}^{d}  X[i, k]^{A[j, k]}

    This is the ``x2fx`` function from MATLAB's Statistics Toolbox, adapted
    for arbitrary multi-index sets.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        Design points (may be normalised).
    A : torch.Tensor, shape (m, d)
        Multi-index set; each row α = (α₁, ..., α_d) defines the monomial
        x₁^α₁ · x₂^α₂ · … · x_d^α_d.

    Returns
    -------
    V : torch.Tensor, shape (n, m)
        Polynomial design matrix.

    References
    ----------
    Le Maître & Knio (2010), *Spectral Methods for Uncertainty Quantification*.
    """
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float64)
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float64, device=X.device)

    # Broadcasting: (n,1,d)^(1,m,d) → (n,m,d), then product over d
    base = X.unsqueeze(1)               # (n, 1, d)
    exp  = A.unsqueeze(0).to(X.dtype)  # (1, m, d)
    return torch.prod(torch.pow(base, exp), dim=-1)  # (n, m)


def stepwise(A: torch.Tensor, order: int) -> torch.Tensor:
    """
    Generate candidate monomials of total degree ``order``.

    Starting from the current multi-index set A, the function identifies all
    rows whose total degree equals (order − 1) and increments each coordinate
    by 1.  Duplicates are removed via ``torch.unique``.

    This is the incremental step used in greedy forward stepwise selection:
    at each iteration we only test monomials of the next total degree.

    Parameters
    ----------
    A     : torch.Tensor, shape (m, d), dtype int64
        Current multi-index set.
    order : int
        Total degree of monomials to generate.

    Returns
    -------
    candidates : torch.Tensor, shape (k, d)
        New candidate monomials of total degree ``order``.
        May be empty (shape (0, d)) if no rows in A have degree order−1.

    Notes
    -----
    The algorithm mirrors the JAX implementation in the companion MATLAB/JAX
    SPRE codebase.  It expands only from rows with sum == order−1 to avoid
    generating lower-degree duplicates.
    """
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.int64)

    if A.numel() == 0:
        d = A.shape[1] if A.dim() == 2 else 0
        return torch.empty((0, d), dtype=torch.int64, device=A.device)

    _n_models, d = A.shape
    device = A.device

    # Select rows with total degree == order − 1
    mask      = (A.sum(dim=1) == (order - 1))
    A_filtered = A[mask]                                       # (k, d)

    if A_filtered.shape[0] == 0:
        return torch.empty((0, d), dtype=A.dtype, device=device)

    # Increment each coordinate by 1: (k, 1, d) + (1, d, d) → (k, d, d)
    eye_d     = torch.eye(d, dtype=A.dtype, device=device)
    expanded  = A_filtered.unsqueeze(1) + eye_d.unsqueeze(0)  # (k, d, d)
    candidates = expanded.reshape(-1, d)                       # (k·d, d)
    return torch.unique(candidates, dim=0)
