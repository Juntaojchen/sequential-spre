

from typing import Tuple

import numpy as np

from .constants import EPSILON


def geom_means(
    amps_indep: np.ndarray,
    ells_indep: np.ndarray,
) -> Tuple[float, float]:
 
    amp_global = float(np.exp(np.mean(np.log(amps_indep + EPSILON))))
    ell_global = float(np.exp(np.mean(np.log(ells_indep + EPSILON))))
    return amp_global, ell_global


def smooth_init(
    amps_indep: np.ndarray,
    ells_indep: np.ndarray,
    window:     int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Moving-average smoothing in log-space of independently fitted
    hyperparameters, used to warm-start joint MAP optimisations.

    For QR the 'index' axis is the eigenvalue index i (not time).

    Parameters
    ----------
    amps_indep : (n,) amplitude values from independent fits
    ells_indep : (n,) lengthscale values from independent fits
    window     : number of neighbouring indices to average

    Returns
    -------
    amps_smooth, ells_smooth : (n,) smoothed arrays
    """
    N  = len(amps_indep)
    hw = window // 2
    la = np.log(amps_indep + EPSILON)
    le = np.log(ells_indep + EPSILON)
    la_s = np.array([
        np.mean(la[max(0, i - hw):min(N, i + hw + 1)]) for i in range(N)
    ])
    le_s = np.array([
        np.mean(le[max(0, i - hw):min(N, i + hw + 1)]) for i in range(N)
    ])
    return np.exp(la_s), np.exp(le_s)
