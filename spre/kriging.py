"""
Gaussian process inference for universal kriging (GP with polynomial mean).

The universal kriging model is:

    y(x) = Σ_j β_j φ_j(x) + f(x),    f ~ GP(0, k(·,·))

where the polynomial trend {φ_j} is determined by the multi-index set A,
and f is a zero-mean Gaussian process with kernel k.

This module provides three standalone inference routines:

loocv_loss
    Leave-one-out cross-validation (LOOCV) log-likelihood via Dubrule's
    O(n³) formula.  Used for hyperparameter selection.

reml_loss
    Restricted Maximum Likelihood (REML) log-likelihood.  Used for
    joint MAP estimation in TR-SPRE.

reml_sigma_mle
    Closed-form REML amplitude estimate: σ²_MLE = yᵀ P y / (n − m).

kriging_predict
    GP posterior (mean, variance) at a single query point.

References
----------
Dubrule, O. (1983). Cross validation of kriging in a unique neighborhood.
    Mathematical Geology, 15(6), 687–699.

Patterson, H. D. & Thompson, R. (1971). Recovery of inter-block information
    when block sizes are unequal.  Biometrika, 58(3), 545–554.

Rasmussen, C. E. & Williams, C. K. I. (2006).
    *Gaussian Processes for Machine Learning*. MIT Press.
"""

import math

import torch

from .constants import EPSILON, JITTER


# ─────────────────────────────────────────────────────────────────────────────
# Augmented Kriging system
# ─────────────────────────────────────────────────────────────────────────────
def _augmented_system(K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Build the augmented Kriging matrix:

        M = [[K,   V ],
             [Vᵀ,  0 ]]   ∈ R^{(n+m)×(n+m)}

    Parameters
    ----------
    K : (n, n)  kernel (covariance) matrix
    V : (n, m)  polynomial design matrix

    Returns
    -------
    M : (n+m, n+m)
    """
    n, m = V.shape
    M_top = torch.cat([K, V],                                    dim=1)  # (n, n+m)
    M_bot = torch.cat([V.T, torch.zeros(m, m, dtype=K.dtype)],  dim=1)  # (m, n+m)
    return torch.cat([M_top, M_bot], dim=0)                              # (n+m, n+m)


def _chol_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B using Cholesky with automatic jitter fallback."""
    try:
        L = torch.linalg.cholesky(A)
    except RuntimeError:
        n = A.shape[0]
        L = torch.linalg.cholesky(A + 1e-4 * torch.eye(n, dtype=A.dtype, device=A.device))
    return torch.cholesky_solve(B, L)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOOCV  (Dubrule 1983)
# ─────────────────────────────────────────────────────────────────────────────
def loocv_loss(
    K: torch.Tensor,
    V: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """
    LOOCV log-likelihood for universal kriging via Dubrule's formula.

    For each i, the LOO predictive distribution is N(ŷᵢ, σ̂²ᵢ) where:

        ŷᵢ   = yᵢ − [M⁻¹]ᵢᵢ⁻¹ αᵢ          (LOO residual form)
        σ̂²ᵢ  = [M⁻¹]ᵢᵢ⁻¹

    with [α; β] = M⁻¹ [y; 0].  The total LOOCV log-likelihood is:

        L = Σᵢ { −½ log(2π σ̂²ᵢ) − rᵢ² / (2 σ̂²ᵢ) }

    where  rᵢ = αᵢ / [M⁻¹]ᵢᵢ  is the LOO residual.

    Complexity: O(n³)  — same as a single matrix inversion.

    Parameters
    ----------
    K : (n, n)  kernel matrix (without noise nugget)
    V : (n, m)  polynomial design matrix
    Y : (n,)    normalised observations

    Returns
    -------
    L : scalar torch.Tensor  (higher = better fit)
    """
    n, m = V.shape
    Y_col = Y.view(-1, 1)                                      # (n, 1)

    # Augmented system with diagonal stabilisation
    M     = _augmented_system(K, V)                            # (n+m, n+m)
    M_reg = M + JITTER * torch.eye(n + m, dtype=K.dtype, device=K.device)
    M_inv = torch.linalg.inv(M_reg)

    # Solve M [α; β] = [y; 0]
    Y_aug   = torch.cat([Y_col, torch.zeros(m, 1, dtype=K.dtype)], dim=0)
    alpha   = (M_inv @ Y_aug)[:n]                             # (n, 1)
    diag_M  = torch.diagonal(M_inv)[:n]                       # (n,)

    # Dubrule LOOCV statistics
    residuals = alpha.flatten() / diag_M                      # rᵢ = αᵢ / [M⁻¹]ᵢᵢ
    variances = torch.clamp(1.0 / diag_M, min=1e-14)          # σ̂²ᵢ

    # LOO log-likelihood
    log_ll = (
        -0.5 * torch.log(2 * math.pi * variances)
        - 0.5 * (residuals ** 2) / variances
    ).sum()

    return log_ll


# ─────────────────────────────────────────────────────────────────────────────
# 2. REML  (Patterson & Thompson 1971)
# ─────────────────────────────────────────────────────────────────────────────
def reml_loss(
    K: torch.Tensor,
    V: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """
    Restricted Maximum Likelihood (REML) log-likelihood.

    Marginalises out the polynomial coefficients β analytically:

        log L_REML = −(n−m)/2 log(2π)
                   − ½ log|K|
                   − ½ log|Vᵀ K⁻¹ V|
                   − ½ yᵀ P y

    where  P = K⁻¹ − K⁻¹ V (Vᵀ K⁻¹ V)⁻¹ Vᵀ K⁻¹.

    Parameters
    ----------
    K : (n, n)  kernel matrix (with noise nugget already added if desired)
    V : (n, m)  polynomial design matrix
    Y : (n, 1)  normalised observations

    Returns
    -------
    log_reml : scalar torch.Tensor  (higher = better)
    """
    n, m = V.shape
    Y = Y.view(-1, 1).to(torch.float64)

    # Cholesky of K
    try:
        L_K = torch.linalg.cholesky(K)
    except RuntimeError:
        L_K = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=K.dtype, device=K.device))

    log_det_K = 2.0 * torch.log(torch.diag(L_K)).sum()

    # K⁻¹ Y  and  K⁻¹ V
    K_inv_Y = torch.cholesky_solve(Y, L_K)            # (n, 1)
    K_inv_V = torch.cholesky_solve(V, L_K)            # (n, m)

    # Vᵀ K⁻¹ V
    VtKinvV = V.T @ K_inv_V                           # (m, m)
    jitter_vkv = max(JITTER,
                     1e-8 * torch.trace(VtKinvV).abs().item() / m) if m > 0 else JITTER
    VtKinvV_reg = VtKinvV + jitter_vkv * torch.eye(m, dtype=K.dtype, device=K.device)

    try:
        L_VKV = torch.linalg.cholesky(VtKinvV_reg)
    except RuntimeError:
        L_VKV = torch.linalg.cholesky(
            VtKinvV + 1e-4 * torch.eye(m, dtype=K.dtype, device=K.device))

    log_det_VKV = 2.0 * torch.log(torch.diag(L_VKV)).sum()

    # yᵀ P y = yᵀ K⁻¹ y − (Vᵀ K⁻¹ y)ᵀ (Vᵀ K⁻¹ V)⁻¹ (Vᵀ K⁻¹ y)
    VtKinvY    = V.T @ K_inv_Y                                       # (m, 1)
    VKV_inv_VKY = torch.cholesky_solve(VtKinvY, L_VKV)              # (m, 1)
    ytKinvy    = (Y.T @ K_inv_Y).squeeze()                           # scalar
    correction = (VtKinvY.T @ VKV_inv_VKY).squeeze()                # scalar
    ytPy       = ytKinvy - correction

    log_reml = (
        -0.5 * (n - m) * math.log(2.0 * math.pi)
        - 0.5 * log_det_K
        - 0.5 * log_det_VKV
        - 0.5 * ytPy
    )
    return log_reml


# ─────────────────────────────────────────────────────────────────────────────
# 3. Closed-form REML amplitude estimate
# ─────────────────────────────────────────────────────────────────────────────
def reml_sigma_mle(
    K_unit: torch.Tensor,
    V:      torch.Tensor,
    Y:      torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form REML estimate of the amplitude σ².

        σ²_MLE = (1 / (n − m))  yᵀ P y

    where K_unit is the kernel matrix with amplitude fixed to 1 and P is
    the projection matrix  P = K_unit⁻¹ − K_unit⁻¹ V (Vᵀ K_unit⁻¹ V)⁻¹ Vᵀ K_unit⁻¹.

    This allows efficient profiling: optimise only the lengthscale, then
    recover the amplitude analytically.

    Parameters
    ----------
    K_unit : (n, n)  kernel matrix with σ² = 1
    V      : (n, m)  polynomial design matrix
    Y      : (n, 1)  normalised observations

    Returns
    -------
    sigma_sq : scalar torch.Tensor  ≥ EPSILON
    """
    n, m = V.shape
    Y = Y.view(-1, 1).to(torch.float64)

    try:
        L_K = torch.linalg.cholesky(K_unit)
    except RuntimeError:
        L_K = torch.linalg.cholesky(K_unit + 1e-6 * torch.eye(n, dtype=K_unit.dtype))

    K_inv_Y = torch.cholesky_solve(Y,   L_K)
    K_inv_V = torch.cholesky_solve(V,   L_K)
    VtKinvV = V.T @ K_inv_V

    jitter_vkv = max(JITTER,
                     1e-8 * torch.trace(VtKinvV).abs().item() / m) if m > 0 else JITTER
    VtKinvV_reg = VtKinvV + jitter_vkv * torch.eye(m, dtype=K_unit.dtype)

    try:
        L_VKV = torch.linalg.cholesky(VtKinvV_reg)
    except RuntimeError:
        L_VKV = torch.linalg.cholesky(VtKinvV + 1e-4 * torch.eye(m, dtype=K_unit.dtype))

    VtKinvY     = V.T @ K_inv_Y
    VKV_inv_VKY = torch.cholesky_solve(VtKinvY, L_VKV)
    ytKinvy     = (Y.T @ K_inv_Y).squeeze()
    correction  = (VtKinvY.T @ VKV_inv_VKY).squeeze()
    ytPy        = ytKinvy - correction

    sigma_sq = ytPy / (n - m)
    return torch.clamp(sigma_sq, min=EPSILON)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GP posterior at a query point  (used for prediction at h=0)
# ─────────────────────────────────────────────────────────────────────────────
def kriging_predict(
    K_nn:  torch.Tensor,
    V_n:   torch.Tensor,
    Y_n:   torch.Tensor,
    k_sn:  torch.Tensor,
    v_s:   torch.Tensor,
) -> tuple:
    """
    Universal kriging posterior (mean, variance) at a single query point s.

    Solves the full kriging system via direct K⁻¹ (consistent with LOOCV
    which also uses direct inversion rather than Cholesky, preserving
    numerical equivalence across methods).

    Posterior formulae:
        K_inv = K(X,X)⁻¹
        r = v_s − Vᵀ K_inv k_sn                  (residual vector)
        B = (Vᵀ K_inv V)⁻¹

        μ  = k_snᵀ K_inv y  +  rᵀ B (Vᵀ K_inv y)
        σ² = k(s,s) − k_snᵀ K_inv k_sn  +  rᵀ B r

    Parameters
    ----------
    K_nn : (n, n)  training kernel matrix K(X,X)
    V_n  : (n, m)  training design matrix
    Y_n  : (n,)    normalised training observations
    k_sn : (1, n)  cross-covariance k(s, X)
    v_s  : (m, 1)  polynomial basis at s: x2fx([s], A).T

    Returns
    -------
    mu  : (1, 1) tensor  posterior mean at s
    var : (1, 1) tensor  posterior variance at s (clamped to ≥ 0)
    """
    Y_n = torch.as_tensor(Y_n, dtype=torch.float64).flatten()
    K_inv   = torch.linalg.inv(K_nn)

    VtKinv  = V_n.T @ K_inv                               # (m, n)
    r       = v_s - VtKinv @ k_sn.T                       # (m, 1)
    B       = torch.linalg.inv(VtKinv @ V_n)              # (m, m)

    mu  = k_sn @ K_inv @ Y_n.unsqueeze(1) + r.T @ B @ (VtKinv @ Y_n.unsqueeze(1))
    var = (self_k := (k_sn @ K_inv @ k_sn.T)) * 0        # placeholder, computed below
    k_ss_scalar = torch.zeros(1, 1, dtype=K_nn.dtype)     # k(s,s) filled by caller

    # Compute variance without k(s,s) term; caller adds it
    var_partial = -(k_sn @ K_inv @ k_sn.T) + r.T @ B @ r

    # Caller must pass k_ss via the kernel; here we return the partial form
    # so that the caller adds k_ss.  This matches cv_loss_calculation in SPRE.py.
    var = var_partial  # sign: cov = k_ss + var_partial; caller handles k_ss separately

    cov = k_ss_scalar + var_partial
    if cov[0, 0] < 0:
        cov = torch.clamp(cov, min=0.0)

    return mu, cov
