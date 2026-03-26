"""
GP posterior prediction at an arbitrary query point.

Provides two public functions:

predict_normalised
    Low-level: given pre-built kernel / design-matrix blocks, return
    (mu, var) in normalised space.  No data normalisation is applied here.

predict_at_zero
    High-level: builds all required kernel and basis vectors, calls
    predict_normalised, then de-normalises the result to original scale.
    Used by the SPRE class to extrapolate to h = 0.

References
----------
Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*,
  Chapter 2 (posterior GP equations).
"""

from typing import Callable, Tuple

import torch

from .basis import x2fx


def predict_normalised(
    K_nn: torch.Tensor,
    V_n:  torch.Tensor,
    Y_n:  torch.Tensor,
    k_sn: torch.Tensor,
    k_ss: torch.Tensor,
    v_s:  torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Universal kriging posterior (mean, variance) at a single query point.

    Implements the closed-form formulae:

        K_inv = K(X, X)⁻¹
        r     = v_s − Vᵀ K_inv k_sn               (residual vector)
        B     = (Vᵀ K_inv V)⁻¹

        μ     = k_snᵀ K_inv y  +  rᵀ B (Vᵀ K_inv y)
        σ²    = k(s,s) − k_snᵀ K_inv k_sn  +  rᵀ B r

    Parameters
    ----------
    K_nn : (n, n)  training kernel matrix K(X, X)
    V_n  : (n, m)  polynomial design matrix evaluated at training points
    Y_n  : (n,)    normalised training observations
    k_sn : (1, n)  cross-covariance vector k(s, X)
    k_ss : (1, 1)  prior variance k(s, s)
    v_s  : (m, 1)  polynomial basis evaluated at query point s

    Returns
    -------
    mu  : (1, 1)  posterior mean in normalised space
    var : (1, 1)  posterior variance in normalised space (≥ 0)
    """
    Y_n = torch.as_tensor(Y_n, dtype=torch.float64).flatten()

    K_inv  = torch.linalg.inv(K_nn)               # (n, n)
    VtKinv = V_n.T @ K_inv                         # (m, n)
    r      = v_s - VtKinv @ k_sn.T                 # (m, 1)
    B      = torch.linalg.inv(VtKinv @ V_n)        # (m, m)

    mu = (
        k_sn @ K_inv @ Y_n.unsqueeze(1)
        + r.T @ B @ (VtKinv @ Y_n.unsqueeze(1))
    )  # (1, 1)

    var = k_ss - k_sn @ K_inv @ k_sn.T + r.T @ B @ r  # (1, 1)
    var = torch.clamp(var, min=0.0)

    return mu, var


def predict_at_zero(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x:         torch.Tensor,
    X_norm:    torch.Tensor,
    Y_norm:    torch.Tensor,
    A:         torch.Tensor,
    nY:        float,
    Y_mean:    float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extrapolate the GP + polynomial model to the origin (query point = 0).

    Builds all kernel and design-matrix blocks, calls predict_normalised,
    then de-normalises the result.

    Parameters
    ----------
    kernel_fn : callable(X1, X2, x) → (n1, n2) Tensor
        Kernel function matching the SPRE.kernel signature.
    x         : (p,) raw kernel hyperparameters.
    X_norm    : (n, d)  normalised training inputs.
    Y_norm    : (n,)    normalised training outputs.
    A         : (m, d)  polynomial multi-index set.
    nY        : float   Y scale factor from normalisation.
    Y_mean    : float   Y location (0.0 for max-min; mean(Y) for MAD).

    Returns
    -------
    mu  : scalar tensor  posterior mean in original scale.
    var : scalar tensor  posterior variance in original scale (≥ 0).
    """
    d  = X_norm.shape[1]
    Xs = torch.zeros(1, d, dtype=torch.float64)        # query at origin

    K_nn = kernel_fn(X_norm, X_norm, x)               # (n, n)
    k_sn = kernel_fn(Xs,    X_norm,  x)               # (1, n)
    k_ss = kernel_fn(Xs,    Xs,      x)               # (1, 1)
    V_n  = x2fx(X_norm, A)                            # (n, m)
    v_s  = x2fx(Xs, A).T                              # (m, 1)

    mu_norm, var_norm = predict_normalised(K_nn, V_n, Y_norm, k_sn, k_ss, v_s)

    mu  = mu_norm  * nY + Y_mean
    var = var_norm * (nY ** 2)

    return mu, var
