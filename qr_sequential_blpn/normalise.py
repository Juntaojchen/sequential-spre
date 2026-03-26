
from typing import Tuple

import numpy as np
import torch

from .constants import EPSILON


def normalise_mad(
    X_raw: torch.Tensor,
    Y_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
   
    nX     = float((X_raw.max() - X_raw.min()).item()) + EPSILON
    X_norm = X_raw / nX

    Y_mean  = float(Y_raw.mean().item())
    Y_np    = Y_raw.detach().cpu().numpy()
    Y_med   = float(np.median(Y_np))                 # numpy median (matches SPRE)
    mad     = float(np.median(np.abs(Y_np - Y_med))) # numpy MAD   (matches SPRE)
    y_scale = float(torch.abs(Y_raw).mean().item()) + EPSILON
    nY      = max(mad, 1e-3 * y_scale) + EPSILON
    Y_norm  = (Y_raw - Y_mean) / nY

    return X_norm, Y_norm, nX, nY, Y_mean
