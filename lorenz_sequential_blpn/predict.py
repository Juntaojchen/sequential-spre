"""
GP posterior predictive at h → 0  (Richardson limit).

Uses the same kernel and nugget as the REML fitting (gp_utils.py) to ensure
numerical consistency across all kernels, including Matern5/2 whose smoother
covariance matrix can become ill-conditioned without a regularising nugget.
"""

import math
from typing import Tuple

import torch

from .constants import DTYPE, EPSILON, NOISE_VAR, JITTER
from .gp_utils   import kernel_matrix, build_design_matrix, _chol_safe


def _k_star(
    r:           torch.Tensor,   # (n,) distances from 0 to normalised training X
    amp:         float,
    ell:         float,
    kernel_spec: str,
) -> torch.Tensor:
    """k(x*=0, X_norm) as a 1-D tensor of shape (n,)."""
    s = r / (ell + EPSILON)
    if kernel_spec == 'Matern1/2':
        return amp * torch.exp(-s)
    elif kernel_spec == 'Matern3/2':
        u = math.sqrt(3.0) * s
        return amp * (1.0 + u) * torch.exp(-u)
    elif kernel_spec == 'Matern5/2':
        u = math.sqrt(5.0) * s
        return amp * (1.0 + u + u ** 2 / 3.0) * torch.exp(-u)
    elif kernel_spec == 'Gaussian':
        return amp * torch.exp(-(s ** 2))
    else:
        raise ValueError(f"Unknown kernel_spec: {kernel_spec!r}")


def predict_at_zero(
    h_vals:      object,    # array-like (n,)
    y_vals:      object,    # array-like (n,)
    amplitude:   float,
    lengthscale: float,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> Tuple[float, float]:
    """
    Universal-kriging posterior (mean, variance) at h = 0.

    Applies the same MAD normalisation and nugget-regularised kernel matrix
    as the REML fitting, ensuring numerical stability for all kernels.

    Parameters
    ----------
    h_vals      : (n,) step-size grid (raw)
    y_vals      : (n,) Euler observations (raw)
    amplitude   : GP kernel amplitude in MAD-normalised space
    lengthscale : GP kernel lengthscale in MAD-normalised space
    A           : polynomial multi-index, shape (m, 1)
    kernel_spec : 'Matern1/2' | 'Matern3/2' | 'Matern5/2' | 'Gaussian'

    Returns
    -------
    mu  : posterior mean at h = 0 (original scale)
    var : posterior variance at h = 0 (original scale, ≥ 0)
    """
    X_raw = torch.as_tensor(h_vals, dtype=DTYPE)
    Y_raw = torch.as_tensor(y_vals, dtype=DTYPE)

    nX     = float((X_raw.max() - X_raw.min()).item()) + EPSILON
    X_norm = (X_raw / nX).unsqueeze(1)          # (n, 1)

    Y_mean   = float(Y_raw.mean().item())
    Y_median = float(Y_raw.median().item())
    mad      = float(torch.median(torch.abs(Y_raw - Y_median)).item())
    nY       = mad + EPSILON
    Y_norm   = (Y_raw - Y_mean) / nY            # (n,)

    n   = X_norm.shape[0]
    amp = amplitude
    ell = lengthscale
    Y   = Y_norm.view(-1, 1).double()

    K_nn = kernel_matrix(X_norm, amp, ell, kernel_spec)
    K_nn = K_nn + (NOISE_VAR + JITTER) * torch.eye(n, dtype=DTYPE)
    L_K  = _chol_safe(K_nn)

    r_star = X_norm.squeeze().double()           # (n,)
    k_sn   = _k_star(r_star, amp, ell, kernel_spec)   # (n,)

    k_ss   = amp

    V_n  = build_design_matrix(X_norm, A)        # (n, m)
    m    = V_n.shape[1]
    Xs   = torch.zeros(1, 1, dtype=DTYPE)
    v_s  = build_design_matrix(Xs, A).view(-1, 1)  # (m, 1) — basis at 0

    K_inv_Y   = torch.cholesky_solve(Y,                  L_K)  # (n, 1)
    K_inv_V   = torch.cholesky_solve(V_n,                L_K)  # (n, m)
    K_inv_kns = torch.cholesky_solve(k_sn.view(-1, 1),   L_K)  # (n, 1)

    VtKinvV = V_n.T @ K_inv_V + JITTER * torch.eye(m, dtype=DTYPE)  # (m, m)
    L_C     = _chol_safe(VtKinvV)

    VtKinvY   = V_n.T @ K_inv_Y              # (m, 1)
    VtKinv_ks = K_inv_V.T @ k_sn.view(-1, 1) # (m, 1) = V^T K^{-1} k_s

    r = v_s - VtKinv_ks                      # (m, 1)

    C_inv_r         = torch.cholesky_solve(r,         L_C)  # (m, 1)
    C_inv_VtKinvY   = torch.cholesky_solve(VtKinvY,   L_C)  # (m, 1)

    mu_data    = float((K_inv_kns.T @ Y).item())
    mu_poly    = float((r.T @ C_inv_VtKinvY).item())
    mu_norm    = mu_data + mu_poly

    var_red    = float((K_inv_kns.T @ k_sn.view(-1, 1)).item())
    var_poly   = float((r.T @ C_inv_r).item())
    var_norm   = max(0.0, k_ss - var_red + var_poly)

    mu  = mu_norm * nY + Y_mean
    var = var_norm * (nY ** 2)

    return float(mu), max(0.0, float(var))
