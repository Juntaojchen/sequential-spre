"""
Cross-validation-based selection of Sequential SPRE regularisation strengths
(lambda_sigma, lambda_ell).

select_lambda_cv
    k-fold CV over time points.  For each candidate (λ_σ, λ_ℓ) pair:
      1. Split T datasets into k folds.
      2. Train Sequential SPRE on the k-1 training folds.
      3. Interpolate optimised hyperparameters to held-out time points.
      4. Evaluate held-out REML log-likelihood.
    Select the pair with the highest mean CV score.

Typical usage
-------------
    lambda_s, lambda_l = select_lambda_cv(
        datasets, A, init_amps, init_ells,
        lambda_s_grid=np.logspace(-1, 3, 10),
        lambda_l_grid=np.logspace(-1, 3, 10),
        n_folds=5,
        verbose=True,
    )
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .fitting    import fit_sequential
from .init_utils import smooth_init
from .gp_utils   import reml_log_likelihood
from .constants  import EPSILON


def select_lambda_cv(
    datasets:      List[Dict],
    A:             torch.Tensor,
    init_amps:     np.ndarray,
    init_ells:     np.ndarray,
    lambda_s_grid: np.ndarray = None,
    lambda_l_grid: np.ndarray = None,
    n_folds:       int        = 5,
    max_iter:      int        = 300,
    seed:          int        = 42,
    verbose:       bool       = True,
    kernel_spec:   str        = 'Matern3/2',
) -> Tuple[float, float, Dict]:
    """
    Select (lambda_sigma, lambda_ell) via k-fold cross-validation.

    The CV score for a candidate pair (λ_σ, λ_ℓ) is:

        score = mean over folds of { mean over held-out t of log p_REML(D_t | φ_t) }

    where φ_t is obtained by linearly interpolating the Sequential SPRE solution
    trained on the other folds to the held-out time points.

    Parameters
    ----------
    datasets      : list of T dicts (from experiment.py Step 1),
                    each with 'X_norm', 'Y_norm', 'T'
    A             : polynomial multi-index tensor
    init_amps     : (T,) initial amplitude values (e.g. from independent fits)
    init_ells     : (T,) initial lengthscale values
    lambda_s_grid : 1-D array of candidate λ_σ values.
                    Default: np.logspace(-1, 3, 8)
    lambda_l_grid : 1-D array of candidate λ_ℓ values.
                    Default: np.logspace(-1, 3, 8)
    n_folds       : number of CV folds (use T for leave-one-out)
    max_iter      : max L-BFGS-B iterations per Sequential SPRE fit
    seed          : random seed for fold assignment
    verbose       : print progress

    Returns
    -------
    best_ls   : float, selected lambda_sigma
    best_ll   : float, selected lambda_ell
    cv_info   : dict with full grid of scores for inspection
        {
            'scores': 2-D array (n_ls, n_ll),
            'lambda_s_grid': 1-D array,
            'lambda_l_grid': 1-D array,
            'best_lambda_s': float,
            'best_lambda_l': float,
        }
    """
    if lambda_s_grid is None:
        lambda_s_grid = np.logspace(-1, 3, 8)
    if lambda_l_grid is None:
        lambda_l_grid = np.logspace(-1, 3, 8)

    T   = len(datasets)
    rng = np.random.default_rng(seed)

    n_folds = min(n_folds, T)
    idx     = np.arange(T)
    rng.shuffle(idx)
    folds   = [list(fold) for fold in np.array_split(idx, n_folds)]

    T_vals = np.array([d['T'] for d in datasets])

    if verbose:
        n_pairs = len(lambda_s_grid) * len(lambda_l_grid)
        print(f"  [λ CV]  grid {len(lambda_s_grid)}×{len(lambda_l_grid)} = {n_pairs} pairs"
              f",  {n_folds}-fold CV  (T={T})")

    scores = np.full((len(lambda_s_grid), len(lambda_l_grid)), np.nan)

    for i, ls in enumerate(lambda_s_grid):
        for j, ll in enumerate(lambda_l_grid):

            fold_scores = []

            for fold_idx, test_list in enumerate(folds):
                train_list = [t for t in range(T) if t not in test_list]

                if len(train_list) < 2:
                    continue

                train_datasets = [datasets[t] for t in train_list]
                tr_amps_init, tr_ells_init = smooth_init(
                    init_amps[np.array(train_list)],
                    init_ells[np.array(train_list)],
                    window=5,
                )

                try:
                    amps_opt, ells_opt, _ = fit_sequential(
                        train_datasets, ls, ll, A,
                        tr_amps_init, tr_ells_init,
                        max_iter=max_iter,
                        kernel_spec=kernel_spec,
                    )
                except Exception:
                    fold_scores.append(-1e6)
                    continue

                T_train = T_vals[np.array(train_list)]
                if len(train_list) >= 2:
                    amps_test = np.interp(T_vals[np.array(test_list)],
                                          T_train, amps_opt)
                    ells_test = np.interp(T_vals[np.array(test_list)],
                                          T_train, ells_opt)
                else:
                    amps_test = np.full(len(test_list), amps_opt[0])
                    ells_test = np.full(len(test_list), ells_opt[0])

                held_lls = []
                for k, t in enumerate(test_list):
                    try:
                        ll_val = reml_log_likelihood(
                            datasets[t]['X_norm'],
                            datasets[t]['Y_norm'],
                            float(np.log(amps_test[k] + EPSILON)),
                            float(np.log(ells_test[k] + EPSILON)),
                            A,
                            kernel_spec,
                        )
                        held_lls.append(ll_val)
                    except Exception:
                        held_lls.append(-1e6)

                fold_scores.append(float(np.mean(held_lls)))

            scores[i, j] = float(np.mean(fold_scores)) if fold_scores else -np.inf

            if verbose:
                print(f"    ls={ls:.3g}  ll={ll:.3g}  cv_score={scores[i,j]:.4f}")

    best_i, best_j = np.unravel_index(np.nanargmax(scores), scores.shape)
    best_ls = float(lambda_s_grid[best_i])
    best_ll = float(lambda_l_grid[best_j])

    if verbose:
        print(f"  [λ CV]  best  λ_σ={best_ls:.4g}  λ_ℓ={best_ll:.4g}"
              f"  score={scores[best_i, best_j]:.4f}")

    cv_info = {
        'scores':        scores,
        'lambda_s_grid': lambda_s_grid,
        'lambda_l_grid': lambda_l_grid,
        'best_lambda_s': best_ls,
        'best_lambda_l': best_ll,
    }

    return best_ls, best_ll, cv_info
