"""
Data normalisation — MAD-based standardisation.

Mirrors SPRE.set_normalised_data_mad exactly so that hyperparameters
learnt here can be passed directly to
SPRE.perform_extrapolation_fixed_hyperparams.
"""

from typing import Tuple

import torch

from .constants import EPSILON


def normalise_mad(
    X_raw: torch.Tensor,
    Y_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Normalise design points and observations:

        X_norm = X_raw / (max(X) − min(X))
        Y_norm = (Y_raw − mean(Y)) / MAD(Y)

    Parameters
    ----------
    X_raw : (n,)  raw step sizes
    Y_raw : (n,)  raw observations

    Returns
    -------
    X_norm  : normalised step sizes
    Y_norm  : normalised observations
    nX      : X scale factor  (max − min)
    nY      : Y scale factor  (MAD)
    Y_mean  : Y location      (arithmetic mean)
    """
    nX     = float((X_raw.max() - X_raw.min()).item()) + EPSILON
    X_norm = X_raw / nX

    Y_mean   = float(Y_raw.mean().item())
    Y_median = float(Y_raw.median().item())
    mad      = float(torch.median(torch.abs(Y_raw - Y_median)).item())
    nY       = mad + EPSILON
    Y_norm   = (Y_raw - Y_mean) / nY

    return X_norm, Y_norm, nX, nY, Y_mean
