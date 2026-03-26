

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize as sp_minimize

from .constants import DTYPE
from .fitting   import _make_spre
from .normalise import normalise_mad
from .gp_utils  import reml_log_likelihood_and_grad, loocv_log_score

LOG_MIN   = -8.0    # hard floor for log(sigma), log(ell)
LOG_MAX   =  8.0    # hard ceiling
SAFE_LOSS = 1e30    # sentinel for non-finite objective


def compute_regularised_grw(
    X_vals:       np.ndarray,
    q_matrix:     np.ndarray,
    lam_sig:      float,
    A:            torch.Tensor,
    kernel_spec:  str = 'Matern3/2',
    num_restarts: int = 5,
    init_sigmas:  Optional[np.ndarray] = None,
    init_ells:    Optional[np.ndarray] = None,
    verbose:      bool = False,
    lam_ell:      Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    
    if lam_ell is None:
        lam_ell = lam_sig

    W, n_tasks = q_matrix.shape

    X_raw_t = torch.tensor(X_vals, dtype=DTYPE)
    datasets = []
    nY_scales = np.zeros(n_tasks)          # GRW normalisation scale per task
    for i in range(n_tasks):
        Y_raw_t = torch.tensor(q_matrix[:, i], dtype=DTYPE)
        X_norm, Y_norm, _, nY, _ = normalise_mad(X_raw_t, Y_raw_t)
        nY_scales[i] = nY
        datasets.append({'X_norm': X_norm.unsqueeze(1), 'Y_norm': Y_norm})

    if init_sigmas is not None and init_ells is not None:
        log_sigs_init = np.array([
            float(np.clip(np.log(max(s, 1e-30)), LOG_MIN, LOG_MAX))
            for s in init_sigmas
        ])
        log_ells_init = np.array([
            float(np.clip(np.log(max(e, 1e-30)), LOG_MIN, LOG_MAX))
            for e in init_ells
        ])
    else:
        log_sigs_init = np.zeros(n_tasks)          # sigma = 1
        log_ells_init = np.full(n_tasks, -0.693)   # ell   = 0.5

    theta0 = np.concatenate([log_sigs_init, log_ells_init])
    n_nonfinite = [0]

    def objective_and_grad(theta: np.ndarray):
        log_sigs = np.clip(theta[:n_tasks], LOG_MIN, LOG_MAX)
        log_ells = np.clip(theta[n_tasks:], LOG_MIN, LOG_MAX)

        REML_FLOOR = -1e4
        total_ll  = 0.0
        grad_sigs = np.zeros(n_tasks)
        grad_ells = np.zeros(n_tasks)
        for i in range(n_tasks):
            try:
                val, grad_i = reml_log_likelihood_and_grad(
                    datasets[i]['X_norm'], datasets[i]['Y_norm'],
                    log_sigs[i], log_ells[i], A,
                )
                if val < REML_FLOOR:          # degenerate task — zero gradient
                    total_ll += REML_FLOOR
                else:
                    total_ll     += val
                    grad_sigs[i]  = grad_i[0]
                    grad_ells[i]  = grad_i[1]
            except Exception:
                n_nonfinite[0] += 1
                return SAFE_LOSS, np.zeros(2 * n_tasks)

        diff_sigs = log_sigs[1:] - log_sigs[:-1]
        diff_ells = log_ells[1:] - log_ells[:-1]
        grw = (lam_sig * float(np.sum(diff_sigs ** 2)) +
               lam_ell * float(np.sum(diff_ells ** 2)))

        grw_grad_sigs = np.zeros(n_tasks)
        grw_grad_sigs[:-1] -= 2.0 * lam_sig * diff_sigs
        grw_grad_sigs[1:]  += 2.0 * lam_sig * diff_sigs

        grw_grad_ells = np.zeros(n_tasks)
        grw_grad_ells[:-1] -= 2.0 * lam_ell * diff_ells
        grw_grad_ells[1:]  += 2.0 * lam_ell * diff_ells

        total = -(total_ll - grw)   # negate for minimisation
        if not np.isfinite(total):
            n_nonfinite[0] += 1
            return SAFE_LOSS, np.zeros(2 * n_tasks)

        grad_out = np.concatenate([
            -(grad_sigs - grw_grad_sigs),
            -(grad_ells - grw_grad_ells),
        ])
        return float(total), grad_out

    bounds = [(LOG_MIN, LOG_MAX)] * (2 * n_tasks)
    rng    = np.random.RandomState(42)

    best_fun = np.inf
    best_x   = theta0.copy()
    best_res = None

    for r_idx in range(max(1, num_restarts)):
        x0 = (theta0.copy() if r_idx == 0
               else rng.uniform(LOG_MIN / 2, LOG_MAX / 2, 2 * n_tasks))
        try:
            res = sp_minimize(
                objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_fun:
                best_fun = res.fun
                best_x   = res.x.copy()
                best_res = res
        except Exception:
            pass

    if best_res is None:
        best_res = sp_minimize(objective_and_grad, theta0, method='L-BFGS-B',
                               jac=True, bounds=bounds, options={'maxiter': 500})
        best_x = best_res.x.copy()

    if verbose:
        print(f"  [GRW optim] nfev={best_res.nfev}, success={best_res.success},"
              f" n_nonfinite={n_nonfinite[0]}")

    final_sigs = np.exp(np.clip(best_x[:n_tasks], LOG_MIN, LOG_MAX))
    final_ells = np.exp(np.clip(best_x[n_tasks:], LOG_MIN, LOG_MAX))

    diag = dict(
        optim_success=bool(best_res.success),
        optim_message=str(best_res.message),
        n_nonfinite=n_nonfinite[0],
        nY_scales=nY_scales,   # GRW normalisation scale per task (for prediction rescaling)
    )
    return final_sigs, final_ells, float(best_fun), diag


def select_regularisation_grw(
    X_vals:        np.ndarray,
    q_matrix:      np.ndarray,
    lam_sig_grid:  np.ndarray,
    lam_ell_grid:  np.ndarray,
    A:             torch.Tensor,
    kernel_spec:   str  = 'Matern3/2',
    init_sigmas:   Optional[np.ndarray] = None,
    init_ells:     Optional[np.ndarray] = None,
    verbose:       bool = True,
) -> Dict:

    n_tasks = q_matrix.shape[1]
    n_sig   = len(lam_sig_grid)
    n_ell   = len(lam_ell_grid)
    scores  = np.full((n_sig, n_ell), -np.inf)
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    total = n_sig * n_ell
    count = 0

    for i, ls in enumerate(lam_sig_grid):
        for j, ll in enumerate(lam_ell_grid):
            count += 1
            if verbose:
                print(f"\r  [{count:3d}/{total}] lam_sig={ls:.2e}, lam_ell={ll:.2e}",
                      end='', flush=True)
            try:
                sigmas, ells, _, _ = compute_regularised_grw(
                    X_vals, q_matrix, float(ls), A,
                    kernel_spec=kernel_spec, num_restarts=2,
                    init_sigmas=init_sigmas, init_ells=init_ells,
                    lam_ell=float(ll),
                )
            except Exception:
                continue

            if not (np.all(np.isfinite(sigmas)) and np.all(np.isfinite(ells))):
                continue

            X_raw_t  = torch.tensor(X_vals, dtype=DTYPE)
            loocv_sum = 0.0
            ok = True
            for k in range(n_tasks):
                Y_raw_t = torch.tensor(q_matrix[:, k], dtype=DTYPE)
                X_norm, Y_norm, _, _, _ = normalise_mad(X_raw_t, Y_raw_t)
                try:
                    val = loocv_log_score(
                        X_norm.unsqueeze(1), Y_norm,
                        float(np.log(max(sigmas[k], 1e-30))),
                        float(np.log(max(ells[k],   1e-30))),
                        A,
                    )
                    if not np.isfinite(val):
                        ok = False; break
                    loocv_sum += float(np.clip(val, -1e4, 0.0))
                except Exception:
                    ok = False; break

            if ok:
                scores[i, j] = loocv_sum / n_tasks
                cache[(i, j)] = (sigmas.copy(), ells.copy())

    if verbose:
        print()

    if np.any(np.isfinite(scores)):
        best_i, best_j = np.unravel_index(np.nanargmax(scores), scores.shape)
    else:
        best_i, best_j = 0, 0

    best_lam_sig = float(lam_sig_grid[best_i])
    best_lam_ell = float(lam_ell_grid[best_j])

    if verbose:
        print(f"  GRW selected: lam_sig={best_lam_sig:.4e}, lam_ell={best_lam_ell:.4e},"
              f" score={scores[best_i, best_j]:.4f}")

    return dict(
        best_lam_sig=best_lam_sig,
        best_lam_ell=best_lam_ell,
        best_score=float(scores[best_i, best_j]),
        scores=scores,
        lam_sig_grid=lam_sig_grid,
        lam_ell_grid=lam_ell_grid,
        cached_hyperparams=cache.get((best_i, best_j)),
    )


def predict_with_hyperparams(
    X_vals:    np.ndarray,
    q_matrix:  np.ndarray,
    sigmas:    np.ndarray,
    ells:      np.ndarray,
    A:         torch.Tensor,
    kernel_spec: str = 'Matern3/2',
) -> List[Dict]:
    
    n_tasks = q_matrix.shape[1]
    results: List[Dict] = []

    for i in range(n_tasks):
        sp = _make_spre(X_vals, q_matrix[:, i], kernel_spec=kernel_spec)
        try:
            x_raw  = sp.hyperparams_to_raw(float(sigmas[i]), float(ells[i]))
            extrap = sp.perform_extrapolation(
                torch.as_tensor(x_raw, dtype=DTYPE), A, return_mu_and_var=True,
            )
            mu_cv  = extrap['mu_cv'].numpy().flatten()  if extrap.get('mu_cv')  is not None else None
            var_cv = extrap['var_cv'].numpy().flatten() if extrap.get('var_cv') is not None else None
            results.append({
                'mu':          float(extrap['mu'].flatten()[0]),
                'var':         float(extrap['var'].flatten()[0]),
                'amplitude':   float(sigmas[i]),
                'lengthscale': float(ells[i]),
                'mu_cv':       mu_cv,
                'var_cv':      var_cv,
                'success':     True,
            })
        except Exception as e:
            warnings.warn(f"GRW prediction task {i} failed: {e}")
            results.append({
                'mu': np.nan, 'var': np.nan,
                'amplitude': float(sigmas[i]), 'lengthscale': float(ells[i]),
                'mu_cv': None, 'var_cv': None, 'success': False,
            })
    return results
