"""
GP utilities: kernel matrices, polynomial basis matrix, and REML log-likelihood.

Supported kernels (kernel_spec strings match SPRE's canonical names):
    'Matern1/2'  — exponential / Ornstein–Uhlenbeck
    'Matern3/2'  — once mean-square differentiable   (default)
    'Matern5/2'  — twice mean-square differentiable
    'Gaussian'   — RBF / squared-exponential

All computations use torch.float64 for numerical stability.
"""

import math
from typing import Optional, Tuple

import torch

from .constants import DTYPE, EPSILON, JITTER, NOISE_VAR


def matern12_matrix(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """Matérn 1/2: k(r) = amp · exp(−r/ℓ)"""
    r = torch.cdist(X.double(), X.double())
    return amp * torch.exp(-r / (ell + EPSILON))


def matern32_matrix(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """Matérn 3/2: k(r) = amp · (1 + √3 r/ℓ) exp(−√3 r/ℓ)"""
    r = torch.cdist(X.double(), X.double())
    s = math.sqrt(3.0) * r / (ell + EPSILON)
    return amp * (1.0 + s) * torch.exp(-s)


def matern52_matrix(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """Matérn 5/2: k(r) = amp · (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(−√5 r/ℓ)"""
    r = torch.cdist(X.double(), X.double())
    s = math.sqrt(5.0) * r / (ell + EPSILON)
    return amp * (1.0 + s + s ** 2 / 3.0) * torch.exp(-s)


def gaussian_matrix(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """Gaussian RBF: k(r) = amp · exp(−r²/ℓ²)"""
    r = torch.cdist(X.double(), X.double())
    return amp * torch.exp(-(r ** 2) / (ell + EPSILON) ** 2)


_KERNEL_FNS = {
    'Matern1/2': matern12_matrix,
    'Matern3/2': matern32_matrix,
    'Matern5/2': matern52_matrix,
    'Gaussian':  gaussian_matrix,
}

def kernel_matrix(
    X:           torch.Tensor,
    amp:         float,
    ell:         float,
    kernel_spec: str = 'Matern3/2',
) -> torch.Tensor:
    """
    Evaluate a kernel matrix by name.

    Parameters
    ----------
    X           : (n, 1)  normalised design points
    amp         : kernel amplitude  (> 0)
    ell         : kernel lengthscale (> 0)
    kernel_spec : one of 'Matern1/2', 'Matern3/2', 'Matern5/2', 'Gaussian'

    Returns
    -------
    K : (n, n) symmetric positive-definite kernel matrix
    """
    fn = _KERNEL_FNS.get(kernel_spec)
    if fn is None:
        raise ValueError(
            f"Unknown kernel_spec {kernel_spec!r}. "
            f"Choose from: {list(_KERNEL_FNS)}"
        )
    return fn(X, amp, ell)


def _dmatern12_dlogell(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """∂K/∂log(ℓ) for Matérn 1/2:  amp · s · exp(−s),  s = r/ℓ"""
    r = torch.cdist(X.double(), X.double())
    s = r / (ell + EPSILON)
    return amp * s * torch.exp(-s)


def _dmatern32_dlogell(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """∂K/∂log(ℓ) for Matérn 3/2:  amp · s² · exp(−s),  s = √3 r/ℓ"""
    r = torch.cdist(X.double(), X.double())
    s = math.sqrt(3.0) * r / (ell + EPSILON)
    return amp * s ** 2 * torch.exp(-s)


def _dmatern52_dlogell(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """∂K/∂log(ℓ) for Matérn 5/2:  amp · s²(1+s)/3 · exp(−s),  s = √5 r/ℓ"""
    r = torch.cdist(X.double(), X.double())
    s = math.sqrt(5.0) * r / (ell + EPSILON)
    return amp * s ** 2 * (1.0 + s) / 3.0 * torch.exp(-s)


def _dgaussian_dlogell(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    """∂K/∂log(ℓ) for Gaussian RBF:  amp · 2s² · exp(−s²),  s = r/ℓ
    (uses convention k = amp·exp(−r²/ℓ²), so ∂k/∂log(ℓ) = 2s²·k/amp · amp)"""
    r = torch.cdist(X.double(), X.double())
    s = r / (ell + EPSILON)
    return amp * 2.0 * s ** 2 * torch.exp(-(s ** 2))


_DKERNEL_DLOGELL_FNS = {
    'Matern1/2': _dmatern12_dlogell,
    'Matern3/2': _dmatern32_dlogell,
    'Matern5/2': _dmatern52_dlogell,
    'Gaussian':  _dgaussian_dlogell,
}


def dkernel_dlogell_matrix(
    X:           torch.Tensor,
    amp:         float,
    ell:         float,
    kernel_spec: str = 'Matern3/2',
) -> torch.Tensor:
    """Analytical ∂K/∂log(ℓ) matrix for the specified kernel."""
    fn = _DKERNEL_DLOGELL_FNS.get(kernel_spec)
    if fn is None:
        raise ValueError(
            f"Unknown kernel_spec {kernel_spec!r}. "
            f"Choose from: {list(_DKERNEL_DLOGELL_FNS)}"
        )
    return fn(X, amp, ell)


def build_design_matrix(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Polynomial design matrix  V[i, j] = ∏_d  X[i, d]^A[j, d].

    For 1-D Euler error (A = [[0],[1]]):
        V = [1, X]   (intercept + linear term, capturing O(h) truncation)

    Parameters
    ----------
    X : (n, d)   normalised design points
    A : (m, d)   multi-index; rows = polynomial monomials

    Returns
    -------
    V : (n, m) design matrix
    """
    n = X.shape[0]
    m = A.shape[0]
    V = torch.ones((n, m), dtype=DTYPE)
    for j in range(m):
        for d in range(A.shape[1]):
            exp = int(A[j, d].item())
            if exp != 0:
                V[:, j] = V[:, j] * (X[:, d] ** exp)
    return V


def _chol_safe(M: torch.Tensor, fallback_jitter: float = 1e-4) -> torch.Tensor:
    """Cholesky decomposition with automatic jitter fallback."""
    try:
        return torch.linalg.cholesky(M)
    except RuntimeError:
        n = M.shape[0]
        return torch.linalg.cholesky(M + fallback_jitter * torch.eye(n, dtype=DTYPE))


def reml_log_likelihood(
    X_norm:      torch.Tensor,
    Y_norm:      torch.Tensor,
    log_amp:     float,
    log_ell:     float,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> float:
    
    amp = math.exp(log_amp)
    ell = math.exp(log_ell)
    n   = X_norm.shape[0]
    Y   = Y_norm.view(-1, 1).double()

    K   = kernel_matrix(X_norm, amp, ell, kernel_spec)
    K   = K + (NOISE_VAR + JITTER) * torch.eye(n, dtype=DTYPE)
    L_K = _chol_safe(K)

    V = build_design_matrix(X_norm, A)        # (n, m)
    m = V.shape[1]

    K_inv_Y = torch.cholesky_solve(Y, L_K)    # (n, 1)
    K_inv_V = torch.cholesky_solve(V, L_K)    # (n, m)

    VtKinvV = V.T @ K_inv_V                   # (m, m)
    VtKinvV = VtKinvV + JITTER * torch.eye(m, dtype=DTYPE)
    L_VKV   = _chol_safe(VtKinvV)

    log_det_K   = 2.0 * torch.log(torch.diag(L_K)).sum()
    log_det_VKV = 2.0 * torch.log(torch.diag(L_VKV)).sum()

    VtKinvY  = V.T @ K_inv_Y                  # (m, 1)
    ytKinvy  = (Y.T @ K_inv_Y).squeeze()      # yᵀ K⁻¹ y
    ytPy     = ytKinvy - (VtKinvY.T @ torch.cholesky_solve(VtKinvY, L_VKV)).squeeze()

    log_reml = (
        -0.5 * (n - m) * math.log(2.0 * math.pi)
        - 0.5 * log_det_K
        - 0.5 * log_det_VKV
        - 0.5 * ytPy
    )
    return float(log_reml.item())


def reml_log_likelihood_and_grad(
    X_norm:      torch.Tensor,
    Y_norm:      torch.Tensor,
    log_amp:     float,
    log_ell:     float,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> Tuple[float, float, float]:
   
    amp = math.exp(log_amp)
    ell = math.exp(log_ell)
    n   = X_norm.shape[0]
    Y   = Y_norm.view(-1, 1).double()

    K_kern = kernel_matrix(X_norm, amp, ell, kernel_spec)           # (n, n) no nugget
    K      = K_kern + (NOISE_VAR + JITTER) * torch.eye(n, dtype=DTYPE)
    L_K    = _chol_safe(K)

    V = build_design_matrix(X_norm, A)                              # (n, m)
    m = V.shape[1]

    K_inv_Y = torch.cholesky_solve(Y, L_K)                         # (n, 1)
    K_inv_V = torch.cholesky_solve(V, L_K)                         # (n, m)
    K_inv   = torch.cholesky_solve(torch.eye(n, dtype=DTYPE), L_K) # (n, n)

    VtKinvV  = V.T @ K_inv_V + JITTER * torch.eye(m, dtype=DTYPE)  # (m, m)
    L_VKV    = _chol_safe(VtKinvV)
    VtKinvY  = V.T @ K_inv_Y                                        # (m, 1)

    log_det_K   = 2.0 * torch.log(torch.diag(L_K)).sum()
    log_det_VKV = 2.0 * torch.log(torch.diag(L_VKV)).sum()
    C_inv_VtKinvY = torch.cholesky_solve(VtKinvY, L_VKV)           # (m, 1)
    ytKinvy  = (Y.T @ K_inv_Y).squeeze()
    ytPy     = ytKinvy - (VtKinvY.T @ C_inv_VtKinvY).squeeze()

    log_reml = float((
        -0.5 * (n - m) * math.log(2.0 * math.pi)
        - 0.5 * log_det_K
        - 0.5 * log_det_VKV
        - 0.5 * ytPy
    ).item())

    alpha = (K_inv_Y - K_inv_V @ C_inv_VtKinvY).squeeze()          # (n,)

    def _grad(dK: torch.Tensor) -> float:
        quad      = float((alpha @ dK @ alpha).item())
        trK_inv   = float(torch.trace(K_inv @ dK).item())
        U         = K_inv_V                                         # (n, m)
        C_inv_Ut  = torch.cholesky_solve(U.T, L_VKV)               # (m, n)
        trC_inv   = float(torch.trace(C_inv_Ut @ dK @ U).item())
        tr_Pa_dK  = trK_inv - trC_inv
        return 0.5 * (quad - tr_Pa_dK)

    d_log_amp = _grad(K_kern)

    dK_ell    = dkernel_dlogell_matrix(X_norm, amp, ell, kernel_spec)
    d_log_ell = _grad(dK_ell)

    return log_reml, d_log_amp, d_log_ell
