

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
    
    z      = (truths - mus) / (stds + EPSILON)
    pit    = sp_norm.cdf(z)
    cov_2s = float(np.mean(np.abs(z) <= 2.0))
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
