import math
import torch
from .constants import EPSILON, JITTER
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
def loocv_loss(
    K: torch.Tensor,
    V: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:

    n, m = V.shape
    Y_col = Y.view(-1, 1)                                      # (n, 1)

    M     = _augmented_system(K, V)                            # (n+m, n+m)
    M_reg = M + JITTER * torch.eye(n + m, dtype=K.dtype, device=K.device)
    M_inv = torch.linalg.inv(M_reg)

    Y_aug   = torch.cat([Y_col, torch.zeros(m, 1, dtype=K.dtype)], dim=0)
    alpha   = (M_inv @ Y_aug)[:n]                             # (n, 1)
    diag_M  = torch.diagonal(M_inv)[:n]                       # (n,)

    residuals = alpha.flatten() / diag_M                      # rᵢ = αᵢ / [M⁻¹]ᵢᵢ
    variances = torch.clamp(1.0 / diag_M, min=1e-14)          # σ̂²ᵢ

    log_ll = (
        -0.5 * torch.log(2 * math.pi * variances)
        - 0.5 * (residuals ** 2) / variances
    ).sum()

    return log_ll


def reml_loss(
    K: torch.Tensor,
    V: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:

    n, m = V.shape
    Y = Y.view(-1, 1).to(torch.float64)

    try:
        L_K = torch.linalg.cholesky(K)
    except RuntimeError:
        L_K = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=K.dtype, device=K.device))

    log_det_K = 2.0 * torch.log(torch.diag(L_K)).sum()

    K_inv_Y = torch.cholesky_solve(Y, L_K)            # (n, 1)
    K_inv_V = torch.cholesky_solve(V, L_K)            # (n, m)

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


def reml_sigma_mle(
    K_unit: torch.Tensor,
    V:      torch.Tensor,
    Y:      torch.Tensor,
) -> torch.Tensor:

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


def kriging_predict(
    K_nn:  torch.Tensor,
    V_n:   torch.Tensor,
    Y_n:   torch.Tensor,
    k_sn:  torch.Tensor,
    v_s:   torch.Tensor,
) -> tuple:

    Y_n = torch.as_tensor(Y_n, dtype=torch.float64).flatten()
    K_inv   = torch.linalg.inv(K_nn)

    VtKinv  = V_n.T @ K_inv                               # (m, n)
    r       = v_s - VtKinv @ k_sn.T                       # (m, 1)
    B       = torch.linalg.inv(VtKinv @ V_n)              # (m, m)

    mu  = k_sn @ K_inv @ Y_n.unsqueeze(1) + r.T @ B @ (VtKinv @ Y_n.unsqueeze(1))
    var = (self_k := (k_sn @ K_inv @ k_sn.T)) * 0       
    k_ss_scalar = torch.zeros(1, 1, dtype=K_nn.dtype)    

    var_partial = -(k_sn @ K_inv @ k_sn.T) + r.T @ B @ r

    var = var_partial  

    cov = k_ss_scalar + var_partial
    if cov[0, 0] < 0:
        cov = torch.clamp(cov, min=0.0)

    return mu, cov
