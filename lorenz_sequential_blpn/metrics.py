"""
Calibration metrics for a sequence of predictive Gaussians.
"""

from typing import Dict

import numpy as np
from scipy.stats import kstest
from scipy.stats import norm as sp_norm

from .constants import EPSILON


def evaluate(
    mus:    np.ndarray,
    stds:   np.ndarray,
    truths: np.ndarray,
) -> Dict:
    """
    Calibration metrics for a sequence of N(μ_t, σ_t²) predictive distributions.

    Parameters
    ----------
    mus    : (N,)  posterior means
    stds   : (N,)  posterior standard deviations (> 0)
    truths : (N,)  ground-truth values

    Returns
    -------
    dict with:
        z       : (N,) standardised residuals  (truth − μ) / σ
        pit     : (N,) probability integral transform values  Φ(z)
        cov_2s  : float, empirical coverage at ±2σ  (≈ 95.45 % for N(0,1))
        z_mean  : float, mean of z
        z_std   : float, std of z  (ddof=1; NaN for N=1)
        ks_stat : float, KS statistic vs N(0,1)  (NaN for N < 4)
        ks_pval : float, KS p-value              (NaN for N < 4)
        n       : int,   number of evaluation points
    """
    z      = (truths - mus) / (stds + EPSILON)
    pit    = sp_norm.cdf(z)
    cov_2s = float(np.mean(np.abs(z) <= 2.0))   # ±2σ ≈ 95.45 %
    z_mean = float(np.mean(z))
    z_std  = float(np.std(z, ddof=1)) if len(z) > 1 else float('nan')

    if len(z) >= 4:
        ks_stat, ks_pval = kstest(z, 'norm')
        ks_stat, ks_pval = float(ks_stat), float(ks_pval)
    else:
        ks_stat = ks_pval = float('nan')

    return dict(
        z=z, pit=pit,
        cov_2s=cov_2s,
        z_mean=z_mean, z_std=z_std,
        ks_stat=ks_stat, ks_pval=ks_pval,
        n=len(z),
    )
