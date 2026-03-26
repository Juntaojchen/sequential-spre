"""
Hyperparameter initialisation utilities for Sequential SPRE.

smooth_init
    Moving-average smoothing of independent fits in log-space,
    used to warm-start the joint MAP optimisation.

geom_means
    Geometric mean aggregation of independent fits — §3.1 baseline.
"""

from typing import Tuple

import numpy as np

from .constants import EPSILON


def smooth_init(
    amps_indep: np.ndarray,
    ells_indep: np.ndarray,
    window:     int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Moving-average smoothing in log-space of independently fitted
    hyperparameters, used to warm-start the Sequential SPRE optimisation.

    Smoothing in log-space ensures scale invariance (multiplicative
    smoothness) and prevents negative values.

    Parameters
    ----------
    amps_indep : (T,) amplitude values from independent fits
    ells_indep : (T,) lengthscale values from independent fits
    window     : number of neighbouring points to average (should be odd)

    Returns
    -------
    amps_smooth, ells_smooth : (T,) smoothed arrays
    """
    T  = len(amps_indep)
    hw = window // 2
    la = np.log(amps_indep + EPSILON)
    le = np.log(ells_indep + EPSILON)
    la_s = np.array([
        np.mean(la[max(0, t - hw):min(T, t + hw + 1)]) for t in range(T)
    ])
    le_s = np.array([
        np.mean(le[max(0, t - hw):min(T, t + hw + 1)]) for t in range(T)
    ])
    return np.exp(la_s), np.exp(le_s)


def geom_means(
    amps_indep: np.ndarray,
    ells_indep: np.ndarray,
) -> Tuple[float, float]:
    """
    Geometric mean aggregation of independently fitted hyperparameters.

    Corresponds to arithmetic mean in log-space (§3.1):
        log ā² = (1/T) Σ_t log σ_t²
        log l̄  = (1/T) Σ_t log ℓ_t

    A single global (amp, ell) pair is then used for ALL time points,
    enforcing complete temporal consistency at the cost of local adaptivity.

    Parameters
    ----------
    amps_indep : (T,) amplitude values from independent fits
    ells_indep : (T,) lengthscale values from independent fits

    Returns
    -------
    amp_global, ell_global : geometric-mean amplitude and lengthscale
    """
    amp_global = float(np.exp(np.mean(np.log(amps_indep + EPSILON))))
    ell_global = float(np.exp(np.mean(np.log(ells_indep + EPSILON))))
    return amp_global, ell_global
