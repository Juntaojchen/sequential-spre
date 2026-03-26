
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .matrix       import build_poisson_2d, poisson_eigenvalues_exact
from .qr           import qr_iteration, extract_observations, \
                          estimate_convergence_rate, compute_gp_inputs
from .normalise    import normalise_mad
from .fitting      import fit_reml_single
from .init_utils   import geom_means, smooth_init
from .predict      import predict_at_zero
from .metrics      import evaluate
from .lmc_baseline  import run_lmc_baseline
from .slfm_baseline import run_slfm_baseline
from .grw_fitting  import compute_regularised_grw, select_regularisation_grw, \
                          predict_with_hyperparams
from .constants    import DTYPE, EPSILON


def run_qr_experiment(
    l:          int        = 5,
    m:          int        = 2,
    w_max:      int        = 20,
    w_values:   Optional[List[int]] = None,   # default: 1, …, w_max
    w_min_start: int       = 3,               # skip first few for rate estimation
    poly_order: int        = 1,
    lam_sig:    float      = 1.0,
    lam_ell:    float      = 1.0,
    run_lambda_cv: bool    = False,           # grid search for λ (slow)
    lam_grid:   Optional[np.ndarray] = None, # λ grid (default: logspace(-1,3,8))
    grw_restarts: int      = 5,
    lmc_rank:      int  = 2,
    lmc_restarts:  int  = 5,
    slfm_rank:     int  = 2,
    slfm_restarts: int  = 5,
    kernel_spec:   str  = 'Matern3/2',
    verbose:    bool       = True,
    output_dir: str        = './qr_bbpn_results',
) -> Dict:
   
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if w_values is None:
        w_values = list(range(1, w_max + 1))

    A = torch.tensor([[k] for k in range(poly_order + 1)], dtype=torch.int64)

    if verbose:
        print(f"{'='*65}")
        print(f"QR-BBPN experiment: Poisson({l=}, {m=}), n={l*m}")
        print(f"  QR iterations: w in {{{w_values[0]}, ..., {w_values[-1]}}}")
        print(f"  poly_order = {poly_order}")
        print('=' * 65)

    A_mat  = build_poisson_2d(l, m)
    exact  = poisson_eigenvalues_exact(l, m)
    n      = len(exact)

    if verbose:
        print(f"\n[1/5] Matrix built: n={n}, "
              f"lam in [{exact.min():.4f}, {exact.max():.4f}]")

    if verbose:
        print(f"[2/5] Running QR iteration to w={w_max} ...")

    iterates = qr_iteration(A_mat, w_max=max(w_values))
    w_arr, q_matrix = extract_observations(iterates, w_values)

    if verbose:
        last_err = np.max(np.abs(q_matrix[-1, :] - exact))
        print(f"      q_matrix shape: {q_matrix.shape}")
        print(f"      max|q(w={w_values[-1]}) - lam_exact| = {last_err:.2e}")

    if verbose:
        print("[3/5] Estimating convergence rate ...")

    c_rate, c_per_task = estimate_convergence_rate(
        w_values, q_matrix, w_min_start=w_min_start,
    )
    X_vals = compute_gp_inputs(w_values, c_rate=c_rate)

    if verbose:
        n_valid = int(np.sum(np.isfinite(c_per_task)))
        print(f"      c_global = {c_rate:.4f}  (r = exp(-c) = {np.exp(-c_rate):.4f})")
        print(f"      {n_valid}/{n} tasks had valid per-task rates")
        print(f"      X = exp(-{c_rate:.3f}*w): [{X_vals.min():.6f}, {X_vals.max():.6f}]")

    if verbose:
        print("[4/5] Independent REML fits ...")

    amps_indep = np.zeros(n)
    ells_indep = np.zeros(n)
    rng = np.random.default_rng(42)

    for i in range(n):
        X_t = torch.tensor(X_vals, dtype=DTYPE)
        Y_t = torch.tensor(q_matrix[:, i], dtype=DTYPE)
        amps_indep[i], ells_indep[i] = fit_reml_single(X_t, Y_t, A, rng=rng)

    if verbose:
        print(f"      amp in [{amps_indep.min():.4g}, {amps_indep.max():.4g}]")
        print(f"      ell in [{ells_indep.min():.4g}, {ells_indep.max():.4g}]")

    amp_gm, ell_gm = geom_means(amps_indep, ells_indep)
    amps_gm = np.full(n, amp_gm)
    ells_gm = np.full(n, ell_gm)

    if verbose:
        print(f"      Geometric means: amp = {amp_gm:.4g},  ell = {ell_gm:.4g}")

    if verbose:
        print("[4c/5] Sequential SPRE (GRW regularisation) ...")

    amps_init = np.full(n, amp_gm)
    ells_init = np.full(n, ell_gm)

    if run_lambda_cv:
        grid = lam_grid if lam_grid is not None else np.logspace(-1, 3, 8)
        if verbose:
            print(f"       λ grid search ({len(grid)}×{len(grid)} = "
                  f"{len(grid)**2} fits) ...")
        cv_res = select_regularisation_grw(
            X_vals, q_matrix,
            lam_sig_grid=grid, lam_ell_grid=grid, A=A,
            init_sigmas=amps_init, init_ells=ells_init,
            verbose=verbose,
        )
        lam_sig_used = cv_res['best_lam_sig']
        lam_ell_used = cv_res['best_lam_ell']
        if cv_res['cached_hyperparams'] is not None:
            amps_seq, ells_seq = cv_res['cached_hyperparams']
        else:
            amps_seq, ells_seq, _, _ = compute_regularised_grw(
                X_vals, q_matrix, lam_sig_used, A,
                init_sigmas=amps_init, init_ells=ells_init,
                num_restarts=grw_restarts, lam_ell=lam_ell_used,
            )
    else:
        lam_sig_used, lam_ell_used = lam_sig, lam_ell
        amps_seq, ells_seq, _, _ = compute_regularised_grw(
            X_vals, q_matrix, lam_sig_used, A,
            init_sigmas=amps_init, init_ells=ells_init,
            num_restarts=grw_restarts, lam_ell=lam_ell_used,
            verbose=verbose,
        )

    if verbose:
        print(f"       λ_σ={lam_sig_used:.3g}, λ_ℓ={lam_ell_used:.3g}")
        print(f"       amp ∈ [{amps_seq.min():.4g}, {amps_seq.max():.4g}]")
        print(f"       ell ∈ [{ells_seq.min():.4g}, {ells_seq.max():.4g}]")

    if verbose:
        print(f"[4d/5] LMC baseline (rank={lmc_rank}) ...")

    lmc_results = run_lmc_baseline(
        X_vals, q_matrix, exact,
        rank=lmc_rank,
        poly_order=poly_order,
        n_restarts=lmc_restarts,
        kernel_spec=kernel_spec,
        verbose=verbose,
    )

    if verbose:
        print(f"[4e/5] SLFM baseline (rank={slfm_rank}) ...")

    slfm_results = run_slfm_baseline(
        X_vals, q_matrix, exact,
        rank=slfm_rank,
        poly_order=poly_order,
        n_restarts=slfm_restarts,
        kernel_spec=kernel_spec,
        verbose=verbose,
    )

    if verbose:
        print("[5/5] Predicting eigenvalues at X -> 0 ...")

    X_raw_t = torch.tensor(X_vals, dtype=DTYPE)
    nY_grw  = np.zeros(n)
    nY_spre = np.zeros(n)
    for i in range(n):
        Y_raw_t = torch.tensor(q_matrix[:, i], dtype=DTYPE)
        _, _, _, nY_grw[i], _ = normalise_mad(X_raw_t, Y_raw_t)   # with floor
        Y_np = q_matrix[:, i]
        mad  = float(np.median(np.abs(Y_np - np.median(Y_np))))
        nY_spre[i] = mad + EPSILON                                  # SPRE (no floor)

    scale_ratio = nY_grw / np.maximum(nY_spre, EPSILON)   # (n,)

    def predict_sequence(amps, ells, rescale=False):
        mus, stds = [], []
        for i in range(n):
            amp_i = float(amps[i] * scale_ratio[i]) if rescale else float(amps[i])
            mu, var = predict_at_zero(
                X_vals, q_matrix[:, i],
                amp_i, ells[i], A,
            )
            mus.append(mu)
            stds.append(math.sqrt(max(var, EPSILON)))
        return np.array(mus), np.array(stds)

    mu_ind, std_ind = predict_sequence(amps_indep, ells_indep, rescale=True)
    mu_gm,  std_gm  = predict_sequence(amps_gm,   ells_gm,   rescale=False)
    mu_seq, std_seq = predict_sequence(amps_seq,   ells_seq,  rescale=True)

    metrics_ind = evaluate(mu_ind, std_ind, exact)
    metrics_gm  = evaluate(mu_gm,  std_gm,  exact)
    metrics_seq = evaluate(mu_seq, std_seq, exact)

    if verbose:
        print()
        for tag, m_dict in [
            ('Independent',  metrics_ind),
            ('Geom. Means',  metrics_gm),
            ('Sequential',   metrics_seq),
            ('LMC',          lmc_results['metrics']),
            ('SLFM',         slfm_results['metrics']),
        ]:
            print(f"  {tag:<14}  Cov+-2s={m_dict['cov_2s']:.3f}"
                  f"  z_mean={m_dict['z_mean']:+.3f}  z_std={m_dict['z_std']:.3f}"
                  f"  KS={m_dict['ks_stat']:.3f}")

    return dict(
        l=l, m=m, n=n,
        exact=exact,
        c_rate=c_rate,
        c_per_task=c_per_task,
        w_values=w_values,
        X_vals=X_vals,
        q_matrix=q_matrix,
        indep=dict(amp=amps_indep, ell=ells_indep,
                   mu=mu_ind, std=std_ind, metrics=metrics_ind),
        geomn=dict(amp=amps_gm,   ell=ells_gm,
                   mu=mu_gm,  std=std_gm,  metrics=metrics_gm),
        sequential=dict(amp=amps_seq, ell=ells_seq,
                        mu=mu_seq, std=std_seq, metrics=metrics_seq,
                        lam_sig=lam_sig_used, lam_ell=lam_ell_used),
        lmc=dict(mu=lmc_results['mu'], std=lmc_results['std'],
                 metrics=lmc_results['metrics'],
                 amp=lmc_results['amp'], ell=lmc_results['ell'],
                 noise=lmc_results['noise'], B=lmc_results['B']),
        slfm=dict(mu=slfm_results['mu'], std=slfm_results['std'],
                  metrics=slfm_results['metrics'],
                  amps=slfm_results['amps'], ells=slfm_results['ells'],
                  noise=slfm_results['noise'], C=slfm_results['C']),
    )
