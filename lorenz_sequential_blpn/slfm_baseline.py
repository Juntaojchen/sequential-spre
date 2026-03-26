"""
SLFM (Semiparametric Latent Factor Model) baseline for the Lorenz experiment.

Standalone implementation — does not depend on qr_sequential_BLPN.

Model
-----
Each time point t is a linear combination of p independent latent GPs:

    f_t(x) = sum_{j=1}^p  c_{tj} g_j(x),   g_j ~ GP(0, k_j)

Each factor g_j has its own Matern kernel with (sigma_j, ell_j).

Usage
-----
    from lorenz_sequential_BLPN.slfm_baseline import run_slfm_lorenz
    slfm = run_slfm_lorenz(datasets, truth_np, rank=2, verbose=True)
    mu, std = slfm['mu'], slfm['std']
"""
import warnings
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .constants import EPSILON
from .metrics import evaluate

JITTER = 1e-6



def _kernel(X1: np.ndarray, X2: np.ndarray,
            amp: float, ell: float,
            kernel_spec: str = 'Matern3/2') -> np.ndarray:
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
    return np.column_stack([X ** k for k in range(order + 1)])


def _normalise_x(X: np.ndarray):
    nX = float(X.max() - X.min()) + EPSILON
    return X / nX, nX


def _normalise_y(y: np.ndarray):
    mean_y   = float(np.mean(y))
    median_y = float(np.median(y))
    mad      = float(np.median(np.abs(y - median_y))) + EPSILON
    return (y - mean_y) / mad, mad, mean_y



class SLFMBaseline:
    def __init__(self, rank=2, poly_order=1, n_restarts=5,
                 jitter=JITTER, kernel_spec='Matern3/2'):
        self.rank        = rank
        self.poly_order  = poly_order
        self.n_restarts  = n_restarts
        self.jitter      = jitter
        self.kernel_spec = kernel_spec
        self._fitted     = False

    def _preprocess(self, X, Y):
        T, n = Y.shape
        X_norm, nX = _normalise_x(X)
        nY_arr     = np.zeros(T)
        Y_mean_arr = np.zeros(T)
        Y_norm     = np.zeros_like(Y)
        for i in range(T):
            Y_norm[i], nY_arr[i], Y_mean_arr[i] = _normalise_y(Y[i])
        Phi   = _poly_basis(X_norm, self.poly_order)
        betas = np.zeros((T, self.poly_order + 1))
        Y_res = np.zeros_like(Y_norm)
        for i in range(T):
            beta_i, _, _, _ = np.linalg.lstsq(Phi, Y_norm[i], rcond=None)
            betas[i]  = beta_i
            Y_res[i]  = Y_norm[i] - Phi @ beta_i
        return X_norm, Y_res, betas, nX, nY_arr, Y_mean_arr

    def _build_joint_cov(self, X_norm, amps, ells, C, noise):
        T, n = self.T_, self.n_
        K = np.zeros((T * n, T * n))
        for j in range(self.rank):
            K_j = (_kernel(X_norm, X_norm, amps[j], ells[j], self.kernel_spec)
                   + self.jitter * np.eye(n))
            c_j = C[:, j]
            K  += np.kron(np.outer(c_j, c_j), K_j)
        K += (noise + self.jitter) * np.eye(T * n)
        return K

    def _nll(self, params, X_norm, Y_res_flat):
        T, n, p = self.T_, self.n_, self.rank
        log_amps = params[:p]
        log_ells = params[p:2 * p]
        C        = params[2 * p:].reshape(T, p)
        amps = np.exp(np.clip(log_amps, -10.0, 10.0))
        ells = np.exp(np.clip(log_ells, -10.0, 10.0))
        K = self._build_joint_cov(X_norm, amps, ells, C, noise=0.0)
        try:
            L, lower = cho_factor(K, lower=True, check_finite=False)
            alpha   = cho_solve((L, lower), Y_res_flat)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            quad    = float(Y_res_flat @ alpha)
            nll     = 0.5 * (log_det + quad + T * n * np.log(2.0 * np.pi))
            return float(nll) if np.isfinite(nll) else 1e12
        except Exception:
            return 1e12

    def fit(self, X, Y):
        T, n = Y.shape
        self.T_, self.n_ = T, n
        p = self.rank
        X_norm, Y_res, betas, nX, nY_arr, Y_mean_arr = self._preprocess(X, Y)
        Y_res_flat = Y_res.ravel()
        bounds = (
            [(-4.0, 4.0)] * p       +   # log_sigma per factor
            [(-4.0, 4.0)] * p       +   # log_ell   per factor
            [(None, None)] * (T * p)    # C mixing matrix
        )
        try:
            U_svd, S_svd, _ = np.linalg.svd(Y_res, full_matrices=False)
            C_svd = U_svd[:, :p] * S_svd[:p]
        except Exception:
            C_svd = None
        best_nll    = np.inf
        best_params = None
        for s in range(self.n_restarts):
            rng        = np.random.RandomState(s)
            log_amps0  = rng.uniform(-1.0,  1.0, p)
            log_ells0  = (np.log(float(np.std(X_norm)) + EPSILON)
                          + rng.uniform(-1.0, 1.0, p))
            log_noise0 = rng.uniform(-8.0, -3.0)
            C0 = C_svd if (s == 0 and C_svd is not None) else rng.randn(T, p) * 0.5
            p0 = np.concatenate([log_amps0, log_ells0, C0.ravel()])
            try:
                res = minimize(
                    self._nll, p0,
                    args=(X_norm, Y_res_flat),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-13},
                )
                if np.isfinite(res.fun) and res.fun < best_nll:
                    best_nll    = res.fun
                    best_params = res.x
            except Exception as exc:
                warnings.warn(f"SLFM restart {s} failed: {exc}")
        if best_params is None:
            raise RuntimeError("SLFM fitting failed on all restarts.")
        log_amps = best_params[:p]
        log_ells = best_params[p: 2 * p]
        C        = best_params[2 * p:].reshape(T, p)
        amps  = np.exp(np.clip(log_amps, -10, 10))
        ells  = np.exp(np.clip(log_ells, -10, 10))
        K      = self._build_joint_cov(X_norm, amps, ells, C, noise=0.0)
        L, low = cho_factor(K, lower=True, check_finite=False)
        self._store = dict(
            amps=amps, ells=ells, noise=0.0, C=C, betas=betas,
            L=L, low=low,
            X_norm=X_norm, Y_res_flat=Y_res_flat,
            nX=nX, nY_arr=nY_arr, Y_mean_arr=Y_mean_arr,
            best_nll=best_nll,
        )
        self._fitted = True
        return self

    def predict_at_zero(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_at_zero().")
        s                  = self._store
        amps, ells, C      = s['amps'], s['ells'], s['C']
        betas              = s['betas']
        L, low             = s['L'], s['low']
        X_norm             = s['X_norm']
        Y_res_flat         = s['Y_res_flat']
        nY_arr, Y_mean_arr = s['nY_arr'], s['Y_mean_arr']
        T, n, p            = self.T_, self.n_, self.rank
        alpha  = cho_solve((L, low), Y_res_flat)
        x_star = np.array([0.0])
        k_cross = np.zeros((T, T * n))
        k_ss    = np.zeros((T, T))
        for j in range(p):
            k_j_star = _kernel(x_star, X_norm, amps[j], ells[j], self.kernel_spec)[0]
            k_j_ss   = float(_kernel(x_star, x_star, amps[j], ells[j], self.kernel_spec)[0, 0])
            c_j      = C[:, j]
            c_j_k_j_star = np.concatenate([c_j[t] * k_j_star for t in range(T)])
            k_cross += np.outer(c_j, c_j_k_j_star)
            k_ss    += np.outer(c_j, c_j) * k_j_ss
        mean_star_norm = betas[:, 0]
        mu_res_norm = k_cross @ alpha
        mu_norm     = mu_res_norm + mean_star_norm
        mu          = mu_norm * nY_arr + Y_mean_arr
        var_norm = np.zeros(T)
        for t in range(T):
            k_t         = k_cross[t]
            v           = cho_solve((L, low), k_t)
            var_norm[t] = max(float(k_ss[t, t] - k_t @ v), 1e-12)
        std = np.sqrt(var_norm) * nY_arr
        return mu, std



def run_slfm_lorenz(
    datasets:    List[Dict],
    truth_np:    np.ndarray,
    rank:        int  = 2,
    poly_order:  int  = 1,
    n_restarts:  int  = 5,
    kernel_spec: str  = 'Matern3/2',
    verbose:     bool = True,
) -> Dict:
    """
    SLFM baseline for Lorenz extrapolation (single coordinate).

    Parameters
    ----------
    datasets    : list of Nt dicts with keys 'X_raw' (h_vals) and 'Y_raw' (Euler obs)
    truth_np    : (Nt,) RK4 ground-truth values at each time point
    rank        : number of latent factors p
    poly_order  : polynomial mean degree
    n_restarts  : random restarts for L-BFGS-B
    kernel_spec : 'Matern1/2' | 'Matern3/2' | 'Matern5/2' | 'Gaussian'
    verbose     : print progress

    Returns
    -------
    dict with keys: mu, std, metrics, amps, ells, noise, C
    """
    if len(datasets) == 0:
        raise ValueError("datasets is empty")

    h_vals   = np.asarray(datasets[0]['X_raw'].numpy(), dtype=np.float64)
    n_h      = len(h_vals)
    n_tasks  = len(datasets)

    q_matrix = np.zeros((n_h, n_tasks), dtype=np.float64)
    for t, d in enumerate(datasets):
        y_t = np.asarray(d['Y_raw'].numpy(), dtype=np.float64)
        if len(y_t) != n_h:
            raise ValueError(
                f"dataset {t} has {len(y_t)} observations but h_vals has {n_h}"
            )
        q_matrix[:, t] = y_t

    Y = q_matrix.T   # (Nt, n_h)

    if verbose:
        print(f"[SLFM] Fitting rank-{rank} SLFM  ({kernel_spec}, T={n_tasks} tasks, "
              f"n={n_h} obs, restarts={n_restarts}) ...")

    slfm = SLFMBaseline(rank=rank, poly_order=poly_order,
                        n_restarts=n_restarts, kernel_spec=kernel_spec)
    slfm.fit(h_vals, Y)
    mu, std = slfm.predict_at_zero()
    metrics = evaluate(mu, std, truth_np)

    if verbose:
        st = slfm._store
        print(f"[SLFM] amps={np.round(st['amps'], 4)}  "
              f"ells={np.round(st['ells'], 4)}  "
              f"noise=0 (fixed)  NLL={st['best_nll']:.4f}")
        print(f"[SLFM] Cov±2σ={metrics['cov_2s']:.3f}  "
              f"z_mean={metrics['z_mean']:+.3f}  "
              f"z_std={metrics['z_std']:.3f}  "
              f"KS={metrics['ks_stat']:.3f}")

    return dict(
        mu=mu, std=std, metrics=metrics,
        amps=slfm._store['amps'],
        ells=slfm._store['ells'],
        noise=slfm._store['noise'],
        C=slfm._store['C'],
    )
