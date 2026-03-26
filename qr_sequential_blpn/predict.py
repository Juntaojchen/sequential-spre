

import math
import warnings
from typing import Tuple

import numpy as np
import torch

from .constants import DTYPE, EPSILON

try:
    from spre import SPRE as _SPRE
    HAS_SPRE = True
except ImportError:
    try:
        from sparse_pre.SPRE import SPRE as _SPRE
        HAS_SPRE = True
    except ImportError:
        HAS_SPRE = False
        warnings.warn("spre (or sparse_pre) not found — predict_at_zero unavailable.")


def predict_at_zero(
    X_vals:      np.ndarray,
    y_vals:      np.ndarray,
    amplitude:   float,
    lengthscale: float,
    A:           torch.Tensor,
) -> Tuple[float, float]:
   
    if not HAS_SPRE:
        raise RuntimeError("spre (or sparse_pre) library is required.")

    X_t = torch.tensor(X_vals, dtype=DTYPE).unsqueeze(1)   # (W, 1)
    Y_t = torch.tensor(y_vals, dtype=DTYPE).unsqueeze(1)   # (W, 1)

    sp = _SPRE(kernel_spec='Matern3/2', dimension=1)
    sp.set_normalised_data_mad(X_t, Y_t)
    out = sp.perform_extrapolation_fixed_hyperparams(
        amplitude=amplitude, lengthscale=lengthscale,
        A=A, return_mu_and_var=True,
    )

    mu  = float(out['mu'].item())
    var = max(0.0, float(out['var'].item()))
    return mu, var
