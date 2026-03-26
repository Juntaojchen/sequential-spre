"""
QR iteration, eigenvalue extraction, and convergence rate estimation.

The unshifted QR algorithm produces a sequence of unitarily similar matrices:

    A_0 = A,    A_{w-1} = Q_{w-1} R_{w-1},    A_w = R_{w-1} Q_{w-1}

For a symmetric matrix the diagonal entries converge to the eigenvalues:

    q_i(w) = diag(A_w)_i  →  λ_i   as  w → ∞

Convergence is exponential in w (not polynomial in h = 1/w):

    q_i(w) ≈ λ_i + C_i · r_i^w,    r_i = |λ_{i+1} / λ_i|

Exponential variable change
---------------------------
We estimate a global rate c = median_i(c_i) and set the GP input

    X = exp(−c · w)  ∈ (0, 1),   X → 0  as  w → ∞.

This transforms the model to q_i(X) ≈ λ_i + C_i · X^{c/c_i}, which is
well-approximated by a linear mean β_0 + β_1 X when c ≈ c_i.
The extrapolation target X = 0 then recovers the exact eigenvalue.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


def qr_iteration(A: np.ndarray, w_max: int) -> Dict[int, np.ndarray]:

    iterates: Dict[int, np.ndarray] = {}
    A_w = A.copy()
    for w in range(1, w_max + 1):
        Q, R = np.linalg.qr(A_w)
        A_w  = R @ Q
        if not np.all(np.isfinite(A_w)):
            warnings.warn(f"QR iteration produced non-finite values at w={w}.")
            break
        iterates[w] = A_w.copy()
    return iterates


def extract_observations(
    iterates: Dict[int, np.ndarray],
    w_values: List[int],
) -> Tuple[np.ndarray, np.ndarray]:

    W = len(w_values)
    n = next(iter(iterates.values())).shape[0]
    q_matrix = np.zeros((W, n), dtype=np.float64)
    for j, w in enumerate(w_values):
        q_matrix[j, :] = np.sort(np.diag(iterates[w]))
    return np.array(w_values, dtype=np.float64), q_matrix


def estimate_convergence_rate(
    w_values:    List[int],
    q_matrix:    np.ndarray,
    w_min_start: int = 3,
) -> Tuple[float, np.ndarray]:
   
    ws      = np.array(w_values, dtype=np.float64)
    n_tasks = q_matrix.shape[1]

    c_per_task   = np.full(n_tasks, np.nan)
    valid_rates: List[float] = []

    for i in range(n_tasks):
        diffs = np.abs(np.diff(q_matrix[:, i]))
        w_mid = ws[1:]

        mask = (w_mid >= w_min_start) & (diffs > 1e-15)
        if np.sum(mask) < 3:
            continue

        log_diffs = np.log(diffs[mask])
        w_valid   = w_mid[mask]

        A_reg = np.vstack([w_valid, np.ones(len(w_valid))]).T
        coefs = np.linalg.lstsq(A_reg, log_diffs, rcond=None)[0]
        slope = float(coefs[0])

        if slope < -0.01:
            c_per_task[i] = -slope
            valid_rates.append(-slope)

    if len(valid_rates) == 0:
        c_global = 0.3
        warnings.warn(f"Could not estimate convergence rate; using fallback c={c_global}")
    else:
        c_global = float(np.median(valid_rates))

    return c_global, c_per_task


def compute_gp_inputs(
    w_values: List[int],
    c_rate:   Optional[float],
) -> np.ndarray:
   
    ws = np.array(w_values, dtype=np.float64)
    if c_rate is not None and c_rate > 0.0:
        return np.exp(-c_rate * ws)
    return 1.0 / ws
