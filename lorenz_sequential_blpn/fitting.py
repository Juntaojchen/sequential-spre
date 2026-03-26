
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from .constants import EPSILON
from .gp_utils import reml_log_likelihood, reml_log_likelihood_and_grad

try:
    from spre import SPRE as _SPRE
    HAS_SPRE = True
except ImportError:
    try:
        from sparse_pre.SPRE import SPRE as _SPRE
        HAS_SPRE = True
    except ImportError:
        HAS_SPRE = False
        warnings.warn("spre (or sparse_pre) not found — fit_loocv_single unavailable.")


def fit_loocv_single(
    X_raw:       torch.Tensor,
    Y_raw:       torch.Tensor,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> Tuple[float, float]:

    if not HAS_SPRE:
        raise RuntimeError("spre (or sparse_pre) library is required for fit_loocv_single.")

    from .constants import DTYPE
    X_t = X_raw.unsqueeze(1).to(DTYPE)   # (n, 1)
    Y_t = Y_raw.unsqueeze(1).to(DTYPE)   # (n, 1)

    sp = _SPRE(kernel_spec=kernel_spec, dimension=1)
    sp.set_normalised_data_mad(X_t, Y_t)

    try:
        fit_result = sp.perform_extrapolation_optimization(A)
        x_opt = fit_result['x']
        hp    = sp.extract_hyperparameters(x_opt)
        return float(hp['amplitude']), float(hp['lengthscale'])
    except Exception:
        return 1.0, 0.5   # safe fallback


def fit_reml_single(
    X_norm:      torch.Tensor,
    Y_norm:      torch.Tensor,
    A:           torch.Tensor,
    n_restarts:  int = 8,
    rng:         Optional[np.random.Generator] = None,
    kernel_spec: str = 'Matern3/2',
) -> Tuple[float, float]:

    if rng is None:
        rng = np.random.default_rng(0)

    def neg_reml(params: np.ndarray) -> float:
        try:
            return -reml_log_likelihood(
                X_norm, Y_norm, params[0], params[1], A, kernel_spec)
        except Exception:
            return 1e9

    def neg_reml_grad(params: np.ndarray) -> np.ndarray:
        eps = 1e-6
        f0  = neg_reml(params)
        g   = np.zeros(2)
        for i in range(2):
            p    = params.copy()
            p[i] += eps
            g[i] = (neg_reml(p) - f0) / eps
        return g

    best_val = np.inf
    best_x   = np.array([0.0, 0.0])

    starts = [np.zeros(2)] + [rng.normal(0.0, 1.0, 2) for _ in range(n_restarts - 1)]
    for x0 in starts:
        try:
            res = minimize(
                neg_reml, x0,
                jac=neg_reml_grad,
                method='L-BFGS-B',
                bounds=[(-8.0, 8.0), (-8.0, 8.0)],
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x   = res.x
        except Exception:
            pass

    return math.exp(best_x[0]), math.exp(best_x[1])


def _map_neg_objective(
    params:      np.ndarray,
    datasets:    List[Dict],
    lambda_s:    float,
    lambda_l:    float,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> float:

    T        = len(datasets)
    log_amps = params[:T]
    log_ells = params[T:]

    total_ll = 0.0
    for t, d in enumerate(datasets):
        try:
            total_ll += reml_log_likelihood(
                d['X_norm'], d['Y_norm'], log_amps[t], log_ells[t], A,
                kernel_spec)
        except Exception:
            total_ll -= 1e6    # penalise diverged evaluations

    d_amp = np.diff(log_amps)
    d_ell = np.diff(log_ells)
    penalty = (lambda_s * float(np.dot(d_amp, d_amp))
             + lambda_l * float(np.dot(d_ell, d_ell)))

    return -(total_ll - penalty)            # negate for minimisation


def _map_neg_gradient(
    params:      np.ndarray,
    datasets:    List[Dict],
    lambda_s:    float,
    lambda_l:    float,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> np.ndarray:
   
    eps  = 1e-5
    f0   = _map_neg_objective(params, datasets, lambda_s, lambda_l, A, kernel_spec)
    grad = np.zeros(len(params))
    for i in range(len(params)):
        p    = params.copy()
        p[i] += eps
        fi   = _map_neg_objective(p, datasets, lambda_s, lambda_l, A, kernel_spec)
        grad[i] = (fi - f0) / eps
    return grad


def fit_sequential(
    datasets:    List[Dict],
    lambda_s:    float,
    lambda_l:    float,
    A:           torch.Tensor,
    init_amps:   np.ndarray,
    init_ells:   np.ndarray,
    max_iter:    int = 500,
    kernel_spec: str = 'Matern3/2',
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    
    T      = len(datasets)
    x0     = np.concatenate([np.log(init_amps + EPSILON),
                              np.log(init_ells  + EPSILON)])
    bounds = [(-8.0, 8.0)] * (2 * T)

    result = minimize(
        fun=_map_neg_objective,
        x0=x0,
        args=(datasets, lambda_s, lambda_l, A, kernel_spec),
        jac=_map_neg_gradient,               # gradient supplied
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False},
    )

    amps_opt = np.exp(result.x[:T])
    ells_opt = np.exp(result.x[T:])

    return amps_opt, ells_opt, {
        'success': result.success,
        'message': result.message,
        'n_iter':  result.nit,
        'n_feval': result.nfev,
        'map_obj': -result.fun,
    }
