

import math
from typing import Tuple

import numpy as np
import scipy.linalg as sla
import torch

from .constants import DTYPE, EPSILON, JITTER, NOISE_VAR


def matern32_matrix(X: torch.Tensor, amp: float, ell: float) -> torch.Tensor:
    r = torch.cdist(X.double(), X.double())
    s = math.sqrt(3.0) * r / (ell + EPSILON)
    return amp * (1.0 + s) * torch.exp(-s)


def build_design_matrix(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
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
    try:
        return torch.linalg.cholesky(M)
    except RuntimeError:
        n = M.shape[0]
        return torch.linalg.cholesky(M + fallback_jitter * torch.eye(n, dtype=DTYPE))


def _to_np(t) -> np.ndarray:
    """Convert torch tensor or array-like to float64 numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().double().numpy()
    return np.asarray(t, dtype=np.float64)


def _chol_safe_np(M: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky with jitter fallback."""
    try:
        return sla.cholesky(M, lower=True)
    except sla.LinAlgError:
        return sla.cholesky(M + 1e-4 * np.eye(M.shape[0]), lower=True)


def _build_design_np(X_np: np.ndarray, A_np: np.ndarray) -> np.ndarray:
    """Polynomial design matrix V[i,j] = prod_d X[i,d]^A[j,d]."""
    n, m = X_np.shape[0], A_np.shape[0]
    V = np.ones((n, m), dtype=np.float64)
    for j in range(m):
        for d in range(A_np.shape[1]):
            e = int(A_np[j, d])
            if e != 0:
                V[:, j] *= X_np[:, d] ** e
    return V


def _gp_core_np(X_np, Y_np, amp, ell, A_np):
    """
    Core GP computation shared by REML, REML+grad, and LOOCV.

    Returns (n, s, exp_s, K_base, L_K, V, m_, K_inv_Y, K_inv_V,
             L_VKV, K_inv, P, Py)
    where all arrays are numpy float64.
    """
    n = X_np.shape[0]

    diff  = X_np[:, np.newaxis, :] - X_np[np.newaxis, :, :]
    r     = np.sqrt(np.sum(diff ** 2, axis=-1))
    s     = math.sqrt(3.0) * r / (ell + EPSILON)
    exp_s = np.exp(-s)
    K_base = amp * (1.0 + s) * exp_s
    K      = K_base + (NOISE_VAR + JITTER) * np.eye(n)
    L_K    = _chol_safe_np(K)

    V  = _build_design_np(X_np, A_np)
    m_ = V.shape[1]

    K_inv_Y = sla.cho_solve((L_K, True), Y_np)   # (n, 1)
    K_inv_V = sla.cho_solve((L_K, True), V)       # (n, m)

    VtKinvV = V.T @ K_inv_V + JITTER * np.eye(m_)
    L_VKV   = _chol_safe_np(VtKinvV)

    K_inv = sla.cho_solve((L_K, True), np.eye(n))       # (n, n)
    C     = sla.cho_solve((L_VKV, True), np.eye(m_))    # (m, m)
    P     = K_inv - K_inv_V @ C @ K_inv_V.T             # (n, n)
    Py    = P @ Y_np                                      # (n, 1)

    return n, s, exp_s, K_base, L_K, V, m_, K_inv_Y, K_inv_V, L_VKV, K_inv, P, Py


def reml_log_likelihood(
    X_norm:  torch.Tensor,
    Y_norm:  torch.Tensor,
    log_amp: float,
    log_ell: float,
    A:       torch.Tensor,
) -> float:
    """
    REML log-likelihood for a GP with Matérn 3/2 kernel.

        log L_REML = -(n-m)/2 log(2pi)
                   - 1/2 log|K|
                   - 1/2 log|V^T K^{-1} V|
                   - 1/2 y^T P y
    """
    amp  = math.exp(log_amp)
    ell  = math.exp(log_ell)
    X_np = _to_np(X_norm)
    Y_np = _to_np(Y_norm).reshape(-1, 1)
    A_np = _to_np(A).astype(np.int64)

    n, s, exp_s, K_base, L_K, V, m_, K_inv_Y, K_inv_V, L_VKV, K_inv, P, Py = \
        _gp_core_np(X_np, Y_np, amp, ell, A_np)

    log_det_K   = 2.0 * np.sum(np.log(np.diag(L_K)))
    log_det_VKV = 2.0 * np.sum(np.log(np.diag(L_VKV)))

    VtKinvY = V.T @ K_inv_Y
    ytKinvy = float(Y_np.T @ K_inv_Y)
    ytPy    = ytKinvy - float(VtKinvY.T @ sla.cho_solve((L_VKV, True), VtKinvY))

    return float(
        -0.5 * (n - m_) * math.log(2.0 * math.pi)
        - 0.5 * log_det_K
        - 0.5 * log_det_VKV
        - 0.5 * ytPy
    )


def reml_log_likelihood_and_grad(
    X_norm:  torch.Tensor,
    Y_norm:  torch.Tensor,
    log_amp: float,
    log_ell: float,
    A:       torch.Tensor,
) -> Tuple[float, np.ndarray]:
    """
    REML log-likelihood and analytic gradient w.r.t. (log_amp, log_ell).

    Gradient formula:
        d L / d theta = 0.5 (Py)^T dK/dtheta (Py) - 0.5 tr(P dK/dtheta)

    dK/d(log_amp) = K_base
    dK/d(log_ell) = amp * s^2 * exp(-s)
    """
    amp  = math.exp(log_amp)
    ell  = math.exp(log_ell)
    X_np = _to_np(X_norm)
    Y_np = _to_np(Y_norm).reshape(-1, 1)
    A_np = _to_np(A).astype(np.int64)

    n, s, exp_s, K_base, L_K, V, m_, K_inv_Y, K_inv_V, L_VKV, K_inv, P, Py = \
        _gp_core_np(X_np, Y_np, amp, ell, A_np)

    log_det_K   = 2.0 * np.sum(np.log(np.diag(L_K)))
    log_det_VKV = 2.0 * np.sum(np.log(np.diag(L_VKV)))
    VtKinvY = V.T @ K_inv_Y
    ytKinvy = float(Y_np.T @ K_inv_Y)
    ytPy    = ytKinvy - float(VtKinvY.T @ sla.cho_solve((L_VKV, True), VtKinvY))

    log_reml = float(
        -0.5 * (n - m_) * math.log(2.0 * math.pi)
        - 0.5 * log_det_K
        - 0.5 * log_det_VKV
        - 0.5 * ytPy
    )

    dK_damp = K_base                   # dK / d(log_amp)
    dK_dell = amp * (s ** 2) * exp_s   # dK / d(log_ell)

    def _grad_term(dK: np.ndarray) -> float:
        quad  = float(Py.T @ dK @ Py)
        trace = float(np.sum(P * dK))   # tr(P dK) = Frobenius inner product
        return 0.5 * quad - 0.5 * trace

    return log_reml, np.array([_grad_term(dK_damp), _grad_term(dK_dell)],
                               dtype=np.float64)


def loocv_log_score(
    X_norm:  torch.Tensor,
    Y_norm:  torch.Tensor,
    log_amp: float,
    log_ell: float,
    A:       torch.Tensor,
) -> float:
    """
    Leave-one-out CV log-score (predictive criterion, used for lambda selection).

        L_LOO = sum_i [ -0.5 log(2pi/P_ii) - 0.5 (Py)_i^2 * P_ii ]

    Preferable to REML for lambda selection: REML always favours lambda=0
    (training criterion), LOOCV can reward regularisation.
    """
    amp  = math.exp(log_amp)
    ell  = math.exp(log_ell)
    X_np = _to_np(X_norm)
    Y_np = _to_np(Y_norm).reshape(-1, 1)
    A_np = _to_np(A).astype(np.int64)

    _, s, exp_s, K_base, L_K, V, m_, K_inv_Y, K_inv_V, L_VKV, K_inv, P, Py = \
        _gp_core_np(X_np, Y_np, amp, ell, A_np)

    Py_flat = Py.ravel()
    P_diag  = np.maximum(np.diag(P), 1e-10)

    loo = np.sum(
        -0.5 * math.log(2.0 * math.pi)
        + 0.5 * np.log(P_diag)
        - 0.5 * Py_flat ** 2 * P_diag
    )
    return float(loo)
