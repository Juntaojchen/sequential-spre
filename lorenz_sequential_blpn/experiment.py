import math
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .lorenz       import LorenzSystem
from .normalise    import normalise_mad
from .fitting      import fit_reml_single, fit_sequential
from .init_utils   import smooth_init, geom_means
from .lambda_cv    import select_lambda_cv
from .predict      import predict_at_zero
from .metrics      import evaluate
from .lmc_baseline  import run_lmc_lorenz
from .slfm_baseline import run_slfm_lorenz
from .constants    import DTYPE, EPSILON, resolve_kernel


def run_lorenz_experiment(
    T_min:         float      = 0.05,
    T_max:         float      = 1.0,
    n_time_points: int        = 40,
    h_base:        float      = 0.01,
    h_factors:     np.ndarray = None,   
    ref_h_factor:  float      = 0.01,  
    lambdas:       Dict       = None,  
    lambda_s_grid: np.ndarray = None,  
    lambda_l_grid: np.ndarray = None,  
    n_folds:       int        = 5,     
    kernel:        str        = 'Matern3/2',  # 'Matern1/2'|'Matern3/2'|'Matern5/2'|'Gaussian'
    n_restarts:    int        = 8,
    output_dir:    str        = './lorenz_sequential_results',
    verbose:       bool       = True,
) -> Dict:
    """
    Full Sequential SPRE experiment on the Lorenz system.

    Parameters
    ----------
    T_min, T_max    : time range
    n_time_points   : number of evaluation times in [T_min, T_max]
    h_base          : base step size
    h_factors       : multipliers for h_base; default np.arange(0.4, 1.6, 0.1)
    ref_h_factor    : ref_h = h_base * ref_h_factor * min(h_vals)
    lambdas         : {coord: (lambda_s, lambda_l)}.
                      If None, lambdas are selected per coordinate via CV.
    lambda_s_grid   : candidate λ_σ values for CV grid search.
                      Default: np.logspace(-1, 3, 8)
    lambda_l_grid   : candidate λ_ℓ values for CV grid search.
                      Default: np.logspace(-1, 3, 8)
    n_folds         : number of CV folds (used only when lambdas=None)
    output_dir      : directory for saved figures
    verbose         : print progress

    Returns
    -------
    results : dict keyed by coord ∈ {0, 1, 2}
        results[coord] = {
            'T'      : (N,) time points,
            'truth'  : (N,) RK4 ground truth,
            'lambda' : (lambda_s, lambda_l) used for Sequential SPRE,
            'cv_info': CV grid scores (None if lambdas was provided),
            'indep'  : {'amp', 'ell', 'mu', 'std', 'metrics'},
            'geomn'  : {'amp', 'ell', 'mu', 'std', 'metrics'},
            'sequential': {'amp', 'ell', 'mu', 'std', 'metrics'},
            'lmc'    : {'mu', 'std', 'metrics', 'amp', 'ell', 'noise', 'B'},
        }
    """
    if h_factors is None:
        h_factors = np.arange(0.4, 1.6, 0.1)

    kernel_spec = resolve_kernel(kernel)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h_vals = h_base * h_factors
    ref_h  = h_base * ref_h_factor * float(h_factors.min())
    T_list = np.linspace(T_min, T_max, n_time_points)
    lorenz = LorenzSystem()

    A = torch.tensor([[0], [1]], dtype=torch.int64)

    results = {}

    for coord in range(3):
        cname = ['x', 'y', 'z'][coord]

        if verbose:
            print(f"\n{'='*65}")
            print(f"Coordinate {coord} ({cname})")
            print('=' * 65)

        datasets  = []
        truth_arr = []
        T_kept    = []

        for T in T_list:
            obs_all   = lorenz.euler_batch(h_vals, T)       # (n_h, 3)
            truth_vec = lorenz.rk4_reference(ref_h, T)      # (3,)

            if not (np.isfinite(obs_all).all() and np.isfinite(truth_vec).all()):
                continue

            y_raw   = obs_all[:, coord]
            X_raw_t = torch.tensor(h_vals, dtype=DTYPE)
            Y_raw_t = torch.tensor(y_raw,  dtype=DTYPE)

            X_norm, Y_norm, nX, nY, Y_mean = normalise_mad(X_raw_t, Y_raw_t)

            datasets.append({
                'X_norm': X_norm.unsqueeze(1),   # (n_h, 1)
                'Y_norm': Y_norm,                # (n_h,)
                'X_raw':  X_raw_t,
                'Y_raw':  Y_raw_t,
                'nX': nX, 'nY': nY, 'Y_mean': Y_mean, 'T': T,
            })
            truth_arr.append(float(truth_vec[coord]))
            T_kept.append(T)

        T_arr    = np.array(T_kept)
        truth_np = np.array(truth_arr)
        Nt       = len(datasets)

        if verbose:
            print(f"  Valid time points: {Nt}/{n_time_points}")

        if verbose:
            print("  [1/4] Independent REML fits …")

        amps_indep = np.zeros(Nt)
        ells_indep = np.zeros(Nt)
        for t, d in enumerate(datasets):
            amps_indep[t], ells_indep[t] = fit_reml_single(
                d['X_norm'], d['Y_norm'], A, n_restarts=n_restarts,
                kernel_spec=kernel_spec,
            )

        if verbose:
            print(f"    amp ∈ [{amps_indep.min():.4g}, {amps_indep.max():.4g}]")
            print(f"    ell ∈ [{ells_indep.min():.4g}, {ells_indep.max():.4g}]")

        amp_gm, ell_gm = geom_means(amps_indep, ells_indep)
        amps_gm = np.full(Nt, amp_gm)
        ells_gm = np.full(Nt, ell_gm)
        if verbose:
            print(f"  [1b/4] Geometric Means: amp={amp_gm:.4f}  ell={ell_gm:.4f}")

        init_amps = np.full(Nt, amp_gm)
        init_ells = np.full(Nt, ell_gm)

        if lambdas is not None and coord in lambdas:
            lam_s, lam_l = lambdas[coord]
            cv_info = None
            if verbose:
                print(f"  [2/4] Using supplied λ_σ={lam_s}  λ_ℓ={lam_l}  (CV skipped)")
        else:
            if verbose:
                print(f"  [2/4] Selecting λ via {n_folds}-fold CV …")
            lam_s, lam_l, cv_info = select_lambda_cv(
                datasets, A, init_amps, init_ells,
                lambda_s_grid=lambda_s_grid,
                lambda_l_grid=lambda_l_grid,
                n_folds=n_folds,
                verbose=verbose,
                kernel_spec=kernel_spec,
            )
            if verbose:
                print(f"    → λ_σ={lam_s:.4g}  λ_ℓ={lam_l:.4g}")

        if verbose:
            print(f"  [3/4] Sequential SPRE joint MAP  λ_σ={lam_s:.4g}  λ_ℓ={lam_l:.4g} …")

        amps_seq, ells_seq, opt_info = fit_sequential(
            datasets, lam_s, lam_l, A, init_amps, init_ells,
            kernel_spec=kernel_spec,
        )

        if verbose:
            s = opt_info
            print(f"    Converged={s['success']}  iters={s['n_iter']}"
                  f"  fevals={s['n_feval']}  MAP={s['map_obj']:.4f}")
            print(f"    amp ∈ [{amps_seq.min():.4f}, {amps_seq.max():.4f}]")
            print(f"    ell ∈ [{ells_seq.min():.4f}, {ells_seq.max():.4f}]")

        if verbose:
            print("  [4/4] Predicting at h → 0 …")

        def predict_sequence(amps_seq, ells_seq):
            mus, stds = [], []
            for t, d in enumerate(datasets):
                mu, var = predict_at_zero(
                    d['X_raw'].numpy(), d['Y_raw'].numpy(),
                    amps_seq[t], ells_seq[t], A,
                    kernel_spec=kernel_spec,
                )
                mus.append(mu)
                stds.append(math.sqrt(max(var, EPSILON)))
            return np.array(mus), np.array(stds)

        mu_ind, std_ind = predict_sequence(amps_indep, ells_indep)
        mu_gm,  std_gm  = predict_sequence(amps_gm,   ells_gm)
        mu_seq,  std_seq  = predict_sequence(amps_seq,   ells_seq)

        metrics_ind = evaluate(mu_ind, std_ind, truth_np)
        metrics_gm  = evaluate(mu_gm,  std_gm,  truth_np)
        metrics_seq  = evaluate(mu_seq,  std_seq,  truth_np)

        if verbose:
            print("  [LMC] Fitting LMC baseline ...")

        try:
            lmc_res = run_lmc_lorenz(
                datasets, truth_np, rank=2, poly_order=1,
                n_restarts=5, kernel_spec=kernel_spec, verbose=verbose,
            )
            lmc_dict = dict(
                mu=lmc_res['mu'], std=lmc_res['std'],
                metrics=lmc_res['metrics'],
                amp=lmc_res['amp'], ell=lmc_res['ell'],
                noise=lmc_res['noise'], B=lmc_res['B'],
            )
            metrics_lmc = lmc_res['metrics']
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [LMC] FAILED for coord {coord}: {e}")
            lmc_dict    = None
            metrics_lmc = None

        if verbose:
            print("  [SLFM] Fitting SLFM baseline ...")

        try:
            slfm_res = run_slfm_lorenz(
                datasets, truth_np, rank=2, poly_order=1,
                n_restarts=5, kernel_spec=kernel_spec, verbose=verbose,
            )
            slfm_dict = dict(
                mu=slfm_res['mu'], std=slfm_res['std'],
                metrics=slfm_res['metrics'],
                amps=slfm_res['amps'], ells=slfm_res['ells'],
                noise=slfm_res['noise'], C=slfm_res['C'],
            )
            metrics_slfm = slfm_res['metrics']
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [SLFM] FAILED for coord {coord}: {e}")
            slfm_dict    = None
            metrics_slfm = None

        if verbose:
            for tag, m in [('Independent', metrics_ind),
                           ('Geom. Means', metrics_gm),
                           ('Sequential',  metrics_seq)]:
                print(f"    {tag:<14}  Cov±2σ={m['cov_2s']:.3f}"
                      f"  z̄={m['z_mean']:+.3f}  sz={m['z_std']:.3f}"
                      f"  KS={m['ks_stat']:.3f}")
            for tag, m in [('LMC', metrics_lmc), ('SLFM', metrics_slfm)]:
                if m is not None:
                    print(f"    {tag:<14}  Cov±2σ={m['cov_2s']:.3f}"
                          f"  z̄={m['z_mean']:+.3f}  sz={m['z_std']:.3f}"
                          f"  KS={m['ks_stat']:.3f}")

        results[coord] = dict(
            T=T_arr, truth=truth_np,
            lambda_used=(lam_s, lam_l),
            cv_info=cv_info,
            indep=dict(amp=amps_indep, ell=ells_indep,
                       mu=mu_ind, std=std_ind, metrics=metrics_ind),
            geomn=dict(amp=amps_gm,   ell=ells_gm,
                       mu=mu_gm,  std=std_gm,  metrics=metrics_gm),
            sequential=dict(amp=amps_seq,  ell=ells_seq,
                            mu=mu_seq, std=std_seq, metrics=metrics_seq),
            lmc=lmc_dict,
            slfm=slfm_dict,
        )

    return results
