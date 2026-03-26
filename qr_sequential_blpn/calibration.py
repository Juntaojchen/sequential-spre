
from typing import Dict, List, Optional

import numpy as np
import torch

ALPHA_CAP = 100.0   # maximum inflation factor for holdout replacement


def apply_z_calibration(
    results:     List[Dict],
    X_vals:      np.ndarray,
    q_matrix:    np.ndarray,
    A:           torch.Tensor,
    k_holdout:   int   = 5,
    kernel_spec: str   = 'Matern3/2',
    alpha_min:   float = 0.05,
    alpha_max:   float = 10.0,
    label:       str   = '',
    exact:       Optional[np.ndarray] = None,
) -> float:
   
    try:
        from sparse_pre.SPRE import SPRE as _SPRE_local
    except ImportError:
        try:
            from spre import SPRE as _SPRE_local
        except ImportError:
            print(f"  {label}Z-calibration: SPRE not available, skipping.")
            return 1.0

    W, n_tasks = q_matrix.shape
    k = min(k_holdout, W - 3)
    if k <= 0:
        print(f"  {label}Z-calibration: not enough data, skipping.")
        return 1.0

    fit_idx  = list(range(W - k))
    hold_idx = list(range(W - k, W))

    X_np     = np.asarray(X_vals, dtype=np.float64).reshape(-1, 1)
    A_t      = torch.as_tensor(A, dtype=torch.float64)
    Xs_zero  = torch.zeros((1, 1), dtype=torch.float64)
    Ys_dummy = torch.zeros(1,      dtype=torch.float64)

    task_vf:  List[float] = []   # var_fit_at_X0_i  (0.0 when collapsed)
    task_mse: List[float] = []   # holdout_mse_i
    task_idx: List[int]   = []   # results index

    for i, r in enumerate(results):
        if not r.get('success', False):
            continue
        amp_i     = r.get('amplitude')
        ell_i     = r.get('lengthscale')
        mu_full_i = r.get('mu')
        if (amp_i is None or ell_i is None or mu_full_i is None
                or not np.isfinite(amp_i) or not np.isfinite(ell_i)
                or not np.isfinite(mu_full_i)
                or amp_i <= 0 or ell_i <= 0):
            continue

        Y_np = np.asarray(q_matrix[:, i], dtype=np.float64).flatten()
        sp   = _SPRE_local(kernel_spec=kernel_spec, dimension=1)
        sp.set_normalised_data_mad(X_np, Y_np)

        try:
            x_raw = sp.hyperparams_to_raw(float(amp_i), float(ell_i))
        except Exception:
            continue

        X_norm_fit  = sp.X_normalised[fit_idx]    # (W-k, 1)
        Y_norm_fit  = sp.Y_normalised[fit_idx]    # (W-k,)
        X_norm_hold = sp.X_normalised[hold_idx]   # (k,   1)

        use_mad = getattr(sp, '_use_mad', True)
        Y_mean  = getattr(sp, 'Y_mean', torch.tensor(0.0, dtype=torch.float64))
        nY      = float(sp.nY)

        vf = 0.0
        try:
            _, var_norm_zero = sp.cv_loss_calculation(
                A_t, X_norm_fit, Y_norm_fit,
                Xs_zero, Ys_dummy, x_raw,
                return_mu_cov=True,
            )
            _vf = float((nY ** 2) * float(var_norm_zero[0, 0]))
            if np.isfinite(_vf) and _vf >= 0.0:
                vf = _vf
        except Exception:
            pass

        all_errs_sq = []
        for j in range(k):
            Xs = X_norm_hold[j : j + 1, :]
            Ys = torch.zeros(1, dtype=torch.float64)
            try:
                mu_norm, _ = sp.cv_loss_calculation(
                    A_t, X_norm_fit, Y_norm_fit, Xs, Ys, x_raw,
                    return_mu_cov=True,
                )
                if use_mad:
                    mu_j = float(nY * float(mu_norm[0, 0]) + float(Y_mean))
                else:
                    mu_j = float(nY * float(mu_norm[0, 0]))
                q_true = float(q_matrix[hold_idx[j], i])
                err_sq = (q_true - mu_j) ** 2
                if np.isfinite(err_sq):
                    all_errs_sq.append(err_sq)
            except Exception:
                continue

        mse = float(np.mean(all_errs_sq)) if all_errs_sq else 0.0

        if vf == 0.0 and not all_errs_sq:
            continue   # nothing useful for this task

        task_vf.append(vf)
        task_mse.append(mse)
        task_idx.append(i)

    if len(task_idx) < 2:
        print(f"  {label}Z-calibration: too few valid tasks ({len(task_idx)}), alpha=1")
        return 1.0

    vf_arr = np.array(task_vf)

    pos_vf      = vf_arr[vf_arr > 0]
    vf_median   = float(np.median(pos_vf)) if len(pos_vf) > 0 else 1.0
    vf_thresh   = vf_median * 1e-4
    collapsed   = vf_arr < vf_thresh   # boolean mask, len = len(task_idx)
    n_collapsed = int(np.sum(collapsed))
    n_good      = len(task_idx) - n_collapsed

    use_exact = (exact is not None)
    z_list: List[float] = []
    for mask_i, (tidx, vf) in enumerate(zip(task_idx, task_vf)):
        if collapsed[mask_i] or vf <= 0.0:
            continue
        q_target  = float(exact[tidx]) if use_exact else float(q_matrix[-1, tidx])
        mu_full_i = float(results[tidx]['mu'])
        z_i       = (q_target - mu_full_i) / np.sqrt(vf)
        if np.isfinite(z_i):
            z_list.append(z_i)

    if len(z_list) < 2:
        print(f"  {label}Z-calibration: too few non-collapsed tasks "
              f"({len(z_list)} good, {n_collapsed} collapsed), alpha=1")
        for mask_i, tidx in enumerate(task_idx):
            if collapsed[mask_i]:
                results[tidx]['var'] = task_mse[mask_i]
        return 1.0

    z_np   = np.array(z_list)
    z_std  = float(np.std(z_np))
    z_mean = float(np.mean(z_np))
    alpha  = float(np.clip(z_std, alpha_min, alpha_max))

    for mask_i, (tidx, vf, mse) in enumerate(zip(task_idx, task_vf, task_mse)):
        r = results[tidx]
        if collapsed[mask_i]:
            r['var']           = mse          # data floor for collapsed tasks
            r['z_calib_alpha'] = None
        else:
            r['var']           = vf * (alpha ** 2)
            r['z_calib_alpha'] = alpha
        r['var_fit_at_X0'] = vf
        r['holdout_mse']   = mse

    src = 'exact' if use_exact else 'q_last'
    print(f"  {label}Z-calibration (k={k}, z_src={src}): "
          f"z_mean={z_mean:+.3f}, z_std={z_std:.4f} "
          f"→ alpha={alpha:.4f}  (n_good={n_good}, n_collapsed={n_collapsed})")
    return alpha


def apply_last_k_calibration(
    results:   List[Dict],
    X_vals:    np.ndarray,
    q_matrix:  np.ndarray,
    k:         Optional[int] = None,
    label:     str = '',
    alpha_cap: float = ALPHA_CAP,
) -> np.ndarray:
    
    W = len(X_vals)
    if k is None or k >= W:
        calib_idx = np.arange(W)
    else:
        calib_idx = np.argsort(X_vals)[:k]

    n_tasks = q_matrix.shape[1]
    alphas  = np.full(n_tasks, np.nan)

    for i, r in enumerate(results):
        if not r.get('success', False):
            continue
        if r.get('mu_cv') is None or r.get('var_cv') is None:
            continue

        Y      = q_matrix[:, i]
        mu_cv  = np.asarray(r['mu_cv']).flatten()
        var_cv = np.asarray(r['var_cv']).flatten()

        resid_sq = (Y[calib_idx] - mu_cv[calib_idx]) ** 2
        var_k    = var_cv[calib_idx]
        loocv_mse = float(np.mean(resid_sq))

        var_max   = float(np.max(var_k)) if len(var_k) > 0 else 0.0
        var_floor = max(var_max * 1e-4, 1e-30)
        pos_mask  = var_k > var_floor

        if np.sum(pos_mask) < 2:
            alpha = 1.0
        else:
            alpha = float(np.mean(resid_sq[pos_mask] / var_k[pos_mask]))
            alpha = float(np.clip(alpha, 0.0, alpha_cap))

        r['var'] = max(r['var'] * alpha, loocv_mse)
        r['calibration_alpha'] = alpha
        alphas[i] = alpha

    valid = alphas[np.isfinite(alphas)]
    k_str = 'all' if (k is None or k >= W) else str(k)
    if len(valid) > 0:
        print(f"  {label}Calibration (k={k_str}): "
              f"median={np.median(valid):.2f}, "
              f"range=[{valid.min():.2f}, {valid.max():.2f}], "
              f"n={len(valid)}/{n_tasks}")
    else:
        print(f"  {label}Calibration: no tasks calibrated.")

    return alphas


def apply_holdout_calibration(
    results:     List[Dict],
    X_vals:      np.ndarray,
    q_matrix:    np.ndarray,
    A:           torch.Tensor,
    k_holdout:   int   = 5,
    kernel_spec: str   = 'Matern3/2',
    alpha_cap:   float = ALPHA_CAP,
    label:       str   = '',
) -> np.ndarray:
   
    try:
        from sparse_pre.SPRE import SPRE as _SPRE_local
    except ImportError:
        try:
            from spre import SPRE as _SPRE_local
        except ImportError:
            print(f"  {label}Holdout calibration: SPRE not available, skipping.")
            return np.full(q_matrix.shape[1], np.nan)

    W       = len(X_vals)
    n_tasks = q_matrix.shape[1]
    k       = min(k_holdout, W - 3)   # need ≥ 3 fit points

    if k <= 0:
        print(f"  {label}Holdout calibration: not enough data "
              f"(W={W}, k_holdout={k_holdout}), skipping.")
        return np.full(n_tasks, np.nan)

    fit_idx  = list(range(W - k))
    hold_idx = list(range(W - k, W))

    A_t      = torch.as_tensor(A, dtype=torch.float64)
    alphas   = np.full(n_tasks, np.nan)
    Xs_zero  = torch.zeros((1, 1), dtype=torch.float64)
    Ys_dummy = torch.zeros(1,      dtype=torch.float64)

    for i, r in enumerate(results):
        if not r.get('success', False):
            continue
        amp_i = r.get('amplitude')
        ell_i = r.get('lengthscale')
        if (amp_i is None or ell_i is None
                or not np.isfinite(amp_i) or not np.isfinite(ell_i)
                or amp_i <= 0 or ell_i <= 0):
            continue

        Y_np = np.asarray(q_matrix[:, i], dtype=np.float64).flatten()
        X_np = np.asarray(X_vals, dtype=np.float64).reshape(-1, 1)
        sp   = _SPRE_local(kernel_spec=kernel_spec, dimension=1)
        sp.set_normalised_data_mad(X_np, Y_np)

        try:
            x_raw = sp.hyperparams_to_raw(float(amp_i), float(ell_i))
        except Exception:
            continue

        X_norm_fit  = sp.X_normalised[fit_idx]    # (W-k, 1)
        Y_norm_fit  = sp.Y_normalised[fit_idx]    # (W-k,)
        X_norm_hold = sp.X_normalised[hold_idx]   # (k,   1)

        use_mad = getattr(sp, '_use_mad', True)
        Y_mean  = getattr(sp, 'Y_mean', torch.tensor(0.0, dtype=torch.float64))
        nY      = float(sp.nY)

        var_fit_at_X0 = None
        try:
            _, var_norm_zero = sp.cv_loss_calculation(
                A_t, X_norm_fit, Y_norm_fit,
                Xs_zero, Ys_dummy, x_raw,
                return_mu_cov=True,
            )
            vf = float((nY ** 2) * float(var_norm_zero[0, 0]))
            if np.isfinite(vf) and vf >= 0.0:
                var_fit_at_X0 = vf
        except Exception:
            pass

        all_errs_sq = []   # used for holdout_mse floor
        errs_sq     = []   # used for alpha diagnostic
        vars_pred   = []

        for j in range(k):
            Xs = X_norm_hold[j : j + 1, :]           # (1, 1)
            Ys = torch.zeros(1, dtype=torch.float64)
            try:
                mu_norm, var_norm = sp.cv_loss_calculation(
                    A_t, X_norm_fit, Y_norm_fit, Xs, Ys, x_raw,
                    return_mu_cov=True,
                )
                if use_mad:
                    mu_j = float(nY * float(mu_norm[0, 0]) + float(Y_mean))
                else:
                    mu_j = float(nY * float(mu_norm[0, 0]))
                var_j  = float((nY ** 2) * float(var_norm[0, 0]))
                q_true = float(q_matrix[hold_idx[j], i])
                err_sq = (q_true - mu_j) ** 2

                if np.isfinite(err_sq):
                    all_errs_sq.append(err_sq)          # always record
                if var_j > 1e-30 and np.isfinite(err_sq) and np.isfinite(var_j):
                    errs_sq.append(err_sq)
                    vars_pred.append(var_j)
            except Exception:
                continue

        if var_fit_at_X0 is None and not all_errs_sq:
            continue   # nothing useful to calibrate with

        holdout_mse = float(np.mean(all_errs_sq)) if all_errs_sq else 0.0

        old_var = float(r['var'])
        if var_fit_at_X0 is not None:
            new_var = max(var_fit_at_X0, holdout_mse)
        else:
            new_var = holdout_mse                       # fallback (amp collapse)

        r['var'] = new_var

        denom = max(old_var, 1e-300)
        alpha = float(np.clip(new_var / denom, 0.0, alpha_cap))
        r['holdout_alpha'] = alpha
        alphas[i]          = new_var          # store new_var (for median print)

    valid = alphas[np.isfinite(alphas)]
    if len(valid) > 0:
        print(f"  {label}Holdout calibration (k={k}): "
              f"median new_var={np.median(valid):.3e}, "
              f"range=[{valid.min():.2e}, {valid.max():.2e}], "
              f"n={len(valid)}/{n_tasks}")
    else:
        print(f"  {label}Holdout calibration: no tasks calibrated.")

    return alphas
