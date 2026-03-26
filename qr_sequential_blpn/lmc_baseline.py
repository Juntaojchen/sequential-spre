

import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from .constants import EPSILON
from .metrics import evaluate

JITTER = 1e-6



def _kernel(X1: np.ndarray, X2: np.ndarray,
            amp: float, ell: float,
            kernel_spec: str = 'Matern3/2') -> np.ndarray:
    """Kernel matrix k(X1_i, X2_j) for the selected kernel."""
    r = np.abs(X1[:, None] - X2[None, :]) / (ell + EPSILON)
    if kernel_spec == 'Matern1/2':
        return amp ** 2 * np.exp(-r)
    elif kernel_spec == 'Matern3/2':
        s = np.sqrt(3.0) * r
        return amp ** 2 * (1.0 + s) * np.exp(-s)
    elif kernel_spec == 'Matern5/2':
        s = np.sqrt(5.0) * r
        return amp ** 2 * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)
    elif kernel_spec == 'Gaussian':
        return amp ** 2 * np.exp(-0.5 * r ** 2)
    else:
        raise ValueError(f"Unknown kernel_spec: {kernel_spec!r}")


def _poly_basis(X: np.ndarray, order: int) -> np.ndarray:
    """Polynomial feature matrix  Φ[i,k] = X[i]^k,  k = 0, …, order."""
    return np.column_stack([X ** k for k in range(order + 1)])



def _normalise_x(X: np.ndarray):
    nX = float(X.max() - X.min()) + EPSILON
    return X / nX, nX


def _normalise_y(y: np.ndarray):
    """MAD normalisation for a single task."""
    mean_y   = float(np.mean(y))
    median_y = float(np.median(y))
    mad      = float(np.median(np.abs(y - median_y))) + EPSILON
    return (y - mean_y) / mad, mad, mean_y



class LMCBaseline:
   

    def __init__(self,
                 rank:        int   = 2,
                 poly_order:  int   = 1,
                 n_restarts:  int   = 5,
                 jitter:      float = JITTER,
                 kernel_spec: str   = 'Matern3/2'):
        self.rank        = rank
        self.poly_order  = poly_order
        self.n_restarts  = n_restarts
        self.jitter      = jitter
        self.kernel_spec = kernel_spec
        self._fitted     = False


    def _preprocess(self, X: np.ndarray, Y: np.ndarray):
        """Normalise and polynomial-demean.  Returns residuals in norm space."""
        T, n = Y.shape

        X_norm, nX = _normalise_x(X)

        nY_arr     = np.zeros(T)
        Y_mean_arr = np.zeros(T)
        Y_norm     = np.zeros_like(Y)
        for i in range(T):
            Y_norm[i], nY_arr[i], Y_mean_arr[i] = _normalise_y(Y[i])

        Phi    = _poly_basis(X_norm, self.poly_order)   # (n, d)
        betas  = np.zeros((T, self.poly_order + 1))
        Y_res  = np.zeros_like(Y_norm)
        for i in range(T):
            beta_i, _, _, _ = np.linalg.lstsq(Phi, Y_norm[i], rcond=None)
            betas[i]  = beta_i
            Y_res[i]  = Y_norm[i] - Phi @ beta_i

        return X_norm, Y_res, betas, nX, nY_arr, Y_mean_arr

    def _nll(self, params: np.ndarray,
             X_norm: np.ndarray, Y_res: np.ndarray) -> float:
       
        T, n = Y_res.shape

        amp   = float(np.exp(params[0]))
        ell   = float(np.exp(params[1]))
        B     = params[2:].reshape(T, self.rank)
        W     = B @ B.T + self.jitter * np.eye(T)   # PSD by construction

        K_base    = _kernel(X_norm, X_norm, amp, ell, self.kernel_spec) + self.jitter * np.eye(n)
        lam_W, Q_W = eigh(W)
        lam_K, Q_K = eigh(K_base)

        lam_joint = np.outer(lam_W, lam_K).ravel()   # (T*n,)
        if np.any(lam_joint <= 0):
            return 1e12

        Y_eig  = (Q_W.T @ Y_res @ Q_K).ravel()
        log_det = np.sum(np.log(lam_joint))
        quad    = np.sum(Y_eig ** 2 / lam_joint)
        return 0.5 * (log_det + quad + T * n * np.log(2.0 * np.pi))


    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'LMCBaseline':
       
        T, n = Y.shape
        X_norm, Y_res, betas, nX, nY_arr, Y_mean_arr = self._preprocess(X, Y)

        bounds = (
            [(-4.0, 4.0)]    +   # log_amp
            [(-4.0, 4.0)]    +   # log_ell
            [(None, None)] * (T * self.rank)   # B entries
        )

        best_nll    = np.inf
        best_params = None

        for s in range(self.n_restarts):
            rng      = np.random.RandomState(s)
            log_amp0 = rng.uniform(-1.0,  1.0)
            log_ell0 = np.log(float(np.std(X_norm)) + EPSILON) + rng.uniform(-1.0, 1.0)
            B0       = rng.randn(T, self.rank) * 0.1

            p0 = np.concatenate([
                [log_amp0, log_ell0],
                B0.ravel(),
            ])

            try:
                res = minimize(
                    self._nll, p0,
                    args=(X_norm, Y_res),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-13},
                )
                if np.isfinite(res.fun) and res.fun < best_nll:
                    best_nll    = res.fun
                    best_params = res.x
            except Exception as exc:
                warnings.warn(f"LMC restart {s} failed: {exc}")

        if best_params is None:
            raise RuntimeError("LMC fitting failed on all restarts.")

        amp   = float(np.exp(best_params[0]))
        ell   = float(np.exp(best_params[1]))
        B     = best_params[2:].reshape(T, self.rank)
        W     = B @ B.T + self.jitter * np.eye(T)

        K_base     = _kernel(X_norm, X_norm, amp, ell, self.kernel_spec) + self.jitter * np.eye(n)
        lam_W, Q_W = eigh(W)
        lam_K, Q_K = eigh(K_base)

        self._store = dict(
            amp=amp, ell=ell, noise=0.0,
            B=B, W=W, betas=betas,
            lam_W=lam_W, Q_W=Q_W,
            lam_K=lam_K, Q_K=Q_K,
            X_norm=X_norm, Y_res=Y_res,
            nX=nX, nY_arr=nY_arr, Y_mean_arr=Y_mean_arr,
            best_nll=best_nll,
        )
        self._fitted = True
        return self

    def predict_at_zero(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior predictive (mean, std) at x* = 0 for every task.

        Uses the Kronecker formula:
            μᵢ(0) = [W α k*]ᵢ  +  polynomial mean at 0
            Var_i(0) = W_{ii} k(0,0) − (W[i,:]⊗k*)ᵀ K_joint⁻¹ (W[i,:]⊗k*)

        where α = K_joint⁻¹ Y_res (computed via Kronecker eigenbasis).

        Returns
        -------
        mu  : (T,)  predictive means  (original scale)
        std : (T,)  predictive std    (original scale)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_at_zero().")

        s          = self._store
        amp, ell   = s['amp'], s['ell']
        noise      = s['noise']
        W, betas   = s['W'], s['betas']
        lam_W, Q_W = s['lam_W'], s['Q_W']
        lam_K, Q_K = s['lam_K'], s['Q_K']
        X_norm     = s['X_norm']
        Y_res      = s['Y_res']
        nY_arr     = s['nY_arr']
        Y_mean_arr = s['Y_mean_arr']

        T, n = Y_res.shape

        k_star = _kernel(np.array([0.0]), X_norm, amp, ell, self.kernel_spec)[0]  # (n,)
        k_ss   = amp ** 2                                           # k(0, 0)

        mean_star_norm = betas[:, 0]   # (T,)

        lam_joint = np.outer(lam_W, lam_K).ravel()   # (T*n,)

        Y_eig     = Q_W.T @ Y_res @ Q_K                           # (T, n)
        alpha_eig = Y_eig / lam_joint.reshape(T, n)               # (T, n)
        alpha     = Q_W @ alpha_eig @ Q_K.T                       # (T, n)

        mu_res_norm = W @ alpha @ k_star                           # (T,)
        mu_norm     = mu_res_norm + mean_star_norm                 # (T,)

        mu = mu_norm * nY_arr + Y_mean_arr                        # (T,)

        k_star_eig = Q_K.T @ k_star   # (n,)

        var_norm = np.zeros(T)
        for i in range(T):
            w_i_eig  = lam_W * Q_W[i, :]                          # (T,)
            outer_sq = np.outer(w_i_eig ** 2, k_star_eig ** 2)    # (T, n)
            quad_i   = np.sum(outer_sq / lam_joint.reshape(T, n))
            var_norm[i] = max(W[i, i] * k_ss - quad_i, 1e-12)

        std = np.sqrt(var_norm) * nY_arr                           # (T,)

        return mu, std



def run_lmc_baseline(
    X_vals:      np.ndarray,
    q_matrix:    np.ndarray,
    exact:       np.ndarray,
    rank:        int  = 2,
    poly_order:  int  = 1,
    n_restarts:  int  = 5,
    kernel_spec: str  = 'Matern3/2',
    verbose:     bool = True,
) -> Dict:
   
    Y = q_matrix.T   # (T, W)

    if verbose:
        T = Y.shape[0]
        print(f"[LMC] Fitting rank-{rank} LMC  ({kernel_spec}, T={T} tasks, "
              f"n={Y.shape[1]} obs, restarts={n_restarts}) …")

    lmc = LMCBaseline(rank=rank, poly_order=poly_order, n_restarts=n_restarts,
                      kernel_spec=kernel_spec)
    lmc.fit(X_vals, Y)
    mu, std = lmc.predict_at_zero()

    metrics = evaluate(mu, std, exact)

    if verbose:
        s = lmc._store
        print(f"[LMC] amp={s['amp']:.4g}  ell={s['ell']:.4g}  "
              f"noise=0 (fixed)  NLL={s['best_nll']:.4f}")
        print(f"[LMC] Cov±2σ={metrics['cov_2s']:.3f}  "
              f"z_mean={metrics['z_mean']:+.3f}  "
              f"z_std={metrics['z_std']:.3f}  "
              f"KS={metrics['ks_stat']:.3f}")

    return dict(
        mu=mu, std=std, metrics=metrics,
        amp=lmc._store['amp'],
        ell=lmc._store['ell'],
        noise=lmc._store['noise'],
        B=lmc._store['B'],
    )
