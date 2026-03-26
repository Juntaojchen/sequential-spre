

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize as sp_minimize

from .constants import DTYPE
from .normalise import normalise_mad
from .gp_utils  import reml_log_likelihood, reml_log_likelihood_and_grad

SEED = 42   # default seed for multi-restart randomisation


def fit_reml_single(
    X_raw:      torch.Tensor,
    Y_raw:      torch.Tensor,
    A:          torch.Tensor,
    n_restarts: int = 8,
    rng:        Optional[np.random.Generator] = None,
) -> Tuple[float, float]:

    if rng is None:
        rng = np.random.default_rng(0)

    X_norm, Y_norm, _, _, _ = normalise_mad(X_raw, Y_raw)
    X_norm_2d = X_norm.unsqueeze(1)   # (W, 1)

    def neg_reml_and_grad(params: np.ndarray):
        try:
            val, grad = reml_log_likelihood_and_grad(
                X_norm_2d, Y_norm, params[0], params[1], A)
            return -val, -grad
        except Exception:
            return 1e9, np.zeros(2)

    best_val = np.inf
    best_x   = np.zeros(2)
    starts   = [np.zeros(2)] + [rng.normal(0.0, 1.0, 2) for _ in range(n_restarts - 1)]

    for x0 in starts:
        try:
            res = sp_minimize(
                neg_reml_and_grad, x0, method='L-BFGS-B', jac=True,
                bounds=[(-8.0, 8.0), (-8.0, 8.0)],
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x   = res.x
        except Exception:
            pass

    return math.exp(best_x[0]), math.exp(best_x[1])


try:
    from sparse_pre.SPRE import SPRE as _SPRE
    HAS_SPRE = True
except ImportError:
    try:
        from spre import SPRE as _SPRE
        HAS_SPRE = True
    except ImportError:
        HAS_SPRE = False
        warnings.warn("sparse_pre (or spre) not found — fitting unavailable.")


def _make_spre(X_raw: np.ndarray, Y_raw: np.ndarray,
               kernel_spec: str = 'Matern3/2') -> '_SPRE':
    """Create and normalise a SPRE object from raw arrays.

    IMPORTANT: set_normalised_data_mad must receive X as (n, 1) numpy and
    Y as (n,) numpy (1-D).  Passing Y as a 2-D tensor (n, 1) suppresses
    the mu_cv / var_cv outputs from perform_extrapolation.
    """
    X_np = np.asarray(X_raw, dtype=np.float64).reshape(-1, 1)  # (n, 1)
    Y_np = np.asarray(Y_raw, dtype=np.float64).flatten()        # (n,)
    sp   = _SPRE(kernel_spec=kernel_spec, dimension=1)
    sp.set_normalised_data_mad(X_np, Y_np)
    return sp


def fit_loocv_single(
    X_raw: torch.Tensor,
    Y_raw: torch.Tensor,
    A:     torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> Tuple[float, float]:
    """
    Fit GP hyperparameters for a single task using SPRE's LOOCV.

    Parameters
    ----------
    X_raw : (W,)   raw GP inputs  (torch tensor, NOT normalised)
    Y_raw : (W,)   raw eigenvalue estimates  (torch tensor)
    A     : (m, 1) polynomial multi-index

    Returns
    -------
    (amplitude, lengthscale)
    """
    if not HAS_SPRE:
        raise RuntimeError("spre (or sparse_pre) library is required.")

    sp = _make_spre(X_raw.numpy(), Y_raw.numpy(), kernel_spec=kernel_spec)
    try:
        fit_result = sp.perform_extrapolation_optimization(A)
        hp = sp.extract_hyperparameters(fit_result['x'])
        return float(hp['amplitude']), float(hp['lengthscale'])
    except Exception:
        return 1.0, 0.5


def fit_independent_tasks(
    X_vals:      np.ndarray,
    q_matrix:    np.ndarray,
    A:           torch.Tensor,
    kernel_spec: str = 'Matern3/2',
    num_restarts: int = 10,
    seed:         int = SEED,
) -> List[Dict]:

    if not HAS_SPRE:
        raise RuntimeError("spre (or sparse_pre) library is required.")

    W, n_tasks = q_matrix.shape
    results: List[Dict] = []

    for i in range(n_tasks):
        sp = _make_spre(X_vals, q_matrix[:, i], kernel_spec=kernel_spec)
        try:
            opt = sp.perform_extrapolation_optimization(
                A, num_restarts=num_restarts, seed=seed + i,
            )
            hp     = sp.extract_hyperparameters(opt['x'])
            extrap = sp.perform_extrapolation(opt['x'], A, return_mu_and_var=True)

            mu_cv  = extrap['mu_cv'].numpy().flatten()  if extrap.get('mu_cv')  is not None else None
            var_cv = extrap['var_cv'].numpy().flatten() if extrap.get('var_cv') is not None else None

            results.append({
                'mu':          float(extrap['mu'].flatten()[0]),
                'var':         float(extrap['var'].flatten()[0]),
                'amplitude':   float(hp['amplitude']),
                'lengthscale': float(hp['lengthscale']),
                'cv':          float(opt['cv']),
                'mu_cv':       mu_cv,
                'var_cv':      var_cv,
                'x_opt':       opt['x'].numpy(),
                'success':     True,
            })
        except Exception as e:
            warnings.warn(f"Task {i} fit failed: {e}")
            results.append({
                'mu': np.nan, 'var': np.nan,
                'amplitude': np.nan, 'lengthscale': np.nan,
                'cv': np.nan, 'mu_cv': None, 'var_cv': None,
                'x_opt': None, 'success': False,
            })
    return results
