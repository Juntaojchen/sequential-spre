"""
Global numerical constants shared across all modules.
"""

import torch
from scipy.stats import norm as sp_norm

EPSILON   = 1e-10          # guard against division by zero
JITTER    = 1e-8           # diagonal perturbation for Cholesky stability
NOISE_VAR = 1e-6           # small observation noise / nugget
DTYPE     = torch.float64

COV_2SIGMA = float(2 * sp_norm.cdf(2) - 1)
