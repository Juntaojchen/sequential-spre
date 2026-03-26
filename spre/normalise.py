"""
Data normalisation for SPRE.

Two normalisation strategies are provided:

normalise_maxmin
    X scaled by  (max − min),  Y scaled by (max − min).
    Default behaviour matching the original SPRE implementation.

normalise_mad
    X scaled by (max − min),  Y centred by mean and scaled by MAD.
    More robust to outliers; used when observations span different scales.

Both return inverse-transform parameters (nX, nY, Y_mean) so that
de-normalised predictions can be recovered in the original scale:

    μ_original = μ_norm × nY  +  Y_mean
    σ²_original = σ²_norm × nY²
"""

from typing import Tuple

import torch

from .constants import EPSILON


def normalise_maxmin(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Max-min normalisation.

        X_norm = X / (max(X) − min(X) + ε)     [column-wise for multi-dim X]
        Y_norm = Y / (max(Y) − min(Y) + ε)

    Parameters
    ----------
    X : (n, d)  raw design points
    Y : (n, 1) or (n,)  raw observations

    Returns
    -------
    X_norm : (n, d)  normalised design points
    Y_norm : (n,)    normalised observations
    nX     : (d,)    X scale factors (one per dimension)
    nY     : float   Y scale factor
    Y_mean : float   Y location (0 for max-min, used for de-normalisation)
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64).flatten()

    nX     = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON  # (d,)
    X_norm = X / nX

    nY     = float((Y.max() - Y.min()).item()) + EPSILON
    Y_mean = 0.0
    Y_norm = Y / nY

    return X_norm, Y_norm, nX, nY, Y_mean


def normalise_mad(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    MAD-based (median absolute deviation) normalisation.

        X_norm = X / (max(X) − min(X) + ε)
        Y_norm = (Y − mean(Y)) / (MAD(Y) + ε)

    where MAD(Y) = median(|Y − median(Y)|).

    This normalisation is more robust to outliers than max-min and is
    the default used in the TR-SPRE lorenz experiment.

    Parameters
    ----------
    X : (n, d)  raw design points
    Y : (n, 1) or (n,)  raw observations

    Returns
    -------
    X_norm : (n, d)  normalised design points
    Y_norm : (n,)    normalised, centred observations
    nX     : (d,)    X scale factors
    nY     : float   Y scale factor (MAD)
    Y_mean : float   Y location (arithmetic mean)
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64).flatten()

    nX     = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
    X_norm = X / nX

    Y_mean   = float(Y.mean().item())
    Y_median = float(Y.median().item())
    mad      = float(torch.median(torch.abs(Y - Y_median)).item())
    nY       = mad + EPSILON
    Y_norm   = (Y - Y_mean) / nY

    return X_norm, Y_norm, nX, nY, Y_mean


def denormalise(
    mu_norm:  torch.Tensor,
    var_norm: torch.Tensor,
    nY:       float,
    Y_mean:   float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse normalisation for GP predictions.

        μ_original  = μ_norm  × nY + Y_mean
        σ²_original = σ²_norm × nY²

    Parameters
    ----------
    mu_norm  : posterior mean in normalised space
    var_norm : posterior variance in normalised space (≥ 0)
    nY       : Y scale factor from normalisation
    Y_mean   : Y location from normalisation (0 for max-min)

    Returns
    -------
    mu, var  : tensors in original scale
    """
    mu  = mu_norm  * nY + Y_mean
    var = var_norm * (nY ** 2)
    return mu, var
