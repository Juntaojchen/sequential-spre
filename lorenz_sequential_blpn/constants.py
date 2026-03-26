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

KERNEL_ALIASES = {
    'matern12':  'Matern1/2',
    'matern32':  'Matern3/2',
    'matern52':  'Matern5/2',
    'rbf':       'Gaussian',
    'gaussian':  'Gaussian',
    'Matern1/2': 'Matern1/2',
    'Matern3/2': 'Matern3/2',
    'Matern5/2': 'Matern5/2',
    'Gaussian':  'Gaussian',
}

def resolve_kernel(name: str) -> str:
    """Return the canonical SPRE kernel_spec string for a user-supplied name."""
    spec = KERNEL_ALIASES.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown kernel {name!r}. "
            f"Choose from: {list(KERNEL_ALIASES)}"
        )
    return spec
