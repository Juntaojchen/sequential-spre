"""
Covariance (kernel) functions for Gaussian process regression.

All kernels operate on float64 tensors.  Hyperparameters are supplied as a
raw (unconstrained) parameter vector ``x`` and mapped to the positive reals
via ``softplus`` before use.

Available kernels
-----------------
gaussian       Radial basis function / squared-exponential kernel
gaussian_ard   ARD variant with per-dimension lengthscales
matern12       Matérn ν = 1/2  (exponential kernel)
matern32       Matérn ν = 3/2  (once-differentiable)
matern52       Matérn ν = 5/2  (twice-differentiable)
white_noise    White noise (diagonal) kernel
gre            GRE kernel with polynomial rate-function modulation

Helper
------
cdist          Pairwise Euclidean distances (numerically safe)

Dispatcher
----------
eval_kernel    Select and evaluate a kernel by name
default_params Return default raw parameter vector for a kernel by name

References
----------
Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*,
  Chapter 4.
"""

import math
from typing import Callable, List, Optional, Union

import torch

from .constants import EPSILON
from .basis     import softplus, x2fx
from .utils     import cellsum


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise distance
# ─────────────────────────────────────────────────────────────────────────────
def cdist(XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Euclidean distances between rows of XA and XB.

    Uses the numerically stable identity  ‖a − b‖² = ‖a‖² + ‖b‖² − 2 aᵀb
    with clamping to avoid negative values from floating-point cancellation.

    Parameters
    ----------
    XA : (m, d)
    XB : (n, d)

    Returns
    -------
    D : (m, n)  distances ≥ 0
    """
    XA = XA.to(torch.float64)
    XB = XB.to(torch.float64)
    XA_sq = (XA ** 2).sum(dim=1, keepdim=True)  # (m, 1)
    XB_sq = (XB ** 2).sum(dim=1)                 # (n,)
    cross  = XA @ XB.T                            # (m, n)
    return torch.sqrt(torch.clamp(XA_sq - 2 * cross + XB_sq, min=0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Individual kernel functions  (pure functions, no state)
# ─────────────────────────────────────────────────────────────────────────────
def gaussian(X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Radial basis function (RBF / squared-exponential) kernel.

        k(r) = σ² exp(−r² / ℓ²)

    Raw parameters: x = [amp_raw, ell_raw]
        σ² = ε + softplus(x[0]),  ℓ = ε + softplus(x[1])

    Reference: Rasmussen & Williams (2006), Eq. (4.9).

    Note: uses ℓ² denominator (not 2ℓ²) matching the original implementation.
    """
    amp = EPSILON + softplus(x[0])
    ell = EPSILON + softplus(x[1])
    r   = cdist(X1, X2)
    return amp * torch.exp(-(r ** 2) / (ell ** 2))


def gaussian_ard(X1: torch.Tensor, X2: torch.Tensor,
                 x: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Automatic Relevance Determination (ARD) Gaussian kernel.

        k(x, x') = σ² exp(−Σ_i (x_i − x'_i)² / ℓ_i²)

    Raw parameters: x = [amp_raw, ell_1_raw, ..., ell_d_raw]

    Reference: Rasmussen & Williams (2006), Section 5.1.
    """
    amp          = EPSILON + softplus(x[0])
    squared_dists = []
    for i in range(dimension):
        ell_i = EPSILON + softplus(x[i + 1])
        d_i   = cdist(X1[:, [i]], X2[:, [i]]) ** 2 / (ell_i ** 2)
        squared_dists.append(d_i)
    sq_dist = cellsum(squared_dists)
    return amp * torch.exp(-sq_dist)


def matern12(X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Matérn covariance with ν = 1/2 (exponential / Ornstein–Uhlenbeck kernel).

        k(r) = σ² exp(−r / ℓ)

    Produces sample paths that are continuous but nowhere differentiable.

    Raw parameters: x = [amp_raw, ell_raw]

    Reference: Rasmussen & Williams (2006), Eq. (4.14).
    """
    amp = EPSILON + softplus(x[0])
    ell = EPSILON + softplus(x[1])
    r   = cdist(X1, X2)
    return amp * torch.exp(-r / ell)


def matern32(X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Matérn covariance with ν = 3/2.

        k(r) = σ² (1 + √3 r / ℓ) exp(−√3 r / ℓ)

    Produces sample paths that are once (mean-square) differentiable.

    Raw parameters: x = [amp_raw, ell_raw]

    Reference: Rasmussen & Williams (2006), Eq. (4.14).
    """
    amp        = EPSILON + softplus(x[0])
    ell        = EPSILON + softplus(x[1])
    r          = cdist(X1, X2)
    sqrt3_r_l  = math.sqrt(3.0) * r / ell
    return amp * (1.0 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)


def matern52(X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Matérn covariance with ν = 5/2.

        k(r) = σ² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(−√5 r/ℓ)

    Produces sample paths that are twice (mean-square) differentiable.

    Raw parameters: x = [amp_raw, ell_raw]

    Reference: Rasmussen & Williams (2006), Eq. (4.14).
    """
    amp        = EPSILON + softplus(x[0])
    ell        = EPSILON + softplus(x[1])
    r          = cdist(X1, X2)
    sqrt5_r_l  = math.sqrt(5.0) * r / ell
    return amp * (1.0 + sqrt5_r_l + sqrt5_r_l ** 2 / 3.0) * torch.exp(-sqrt5_r_l)


def white_noise(X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    White noise (diagonal) kernel.

        k(x, x') = σ² δ(x, x')

    where δ is the Kronecker delta.  Produces uncorrelated observations.

    Raw parameters: x = [amp_raw]
    """
    amp = EPSILON + softplus(x[0])
    matches = (X1.unsqueeze(1) == X2.unsqueeze(0)).all(dim=-1)
    return amp * matches.to(dtype=X1.dtype)


def gre(X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor,
        B: torch.Tensor, base_kernel_fn: Callable) -> torch.Tensor:
    """
    Gauss-Richardson Extrapolation (GRE) kernel with polynomial rate-function
    modulation.

        k_GRE(x, x') = σ²_GRE · b(x) · k_base(x, x') · b(x')

    where  b(x) = Σ_j φ_j(x)  is a polynomial rate function evaluated using
    the multi-index set B.

    Raw parameters: x = [gre_amp_raw, *base_kernel_params]

    Parameters
    ----------
    X1, X2    : (n1, d), (n2, d)  design points
    x         : raw hyperparameters
    B         : (r, d)  multi-index set for the rate function b(·)
    base_kernel_fn : callable(X1, X2, x_base) → (n1, n2) kernel matrix

    Reference: companion GRE paper (see project README).
    """
    amp_gre = EPSILON + softplus(x[0])

    # Polynomial rate function: b(x_i) = Σ_j φ_j(x_i)
    b_X1 = x2fx(X1, B).sum(dim=1)   # (n1,)
    b_X2 = x2fx(X2, B).sum(dim=1)   # (n2,)

    K_base = base_kernel_fn(X1, X2, x[1:])

    return amp_gre * b_X1.unsqueeze(1) * K_base * b_X2.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Default raw parameters by kernel name
# ─────────────────────────────────────────────────────────────────────────────
def default_params(kernel_spec: str, dimension: int = 1,
                   gre_base: Optional[torch.Tensor] = None) -> List[float]:
    """
    Default raw (unconstrained) parameter vector for a given kernel.

    The amplitude raw value is chosen so that softplus(raw) ≈ 1, which is
    natural after data standardisation.

    Parameters
    ----------
    kernel_spec : str
        One of "Gaussian", "GaussianARD", "Matern1/2", "Matern3/2",
        "white", "GRE".
    dimension   : int
        Input dimensionality d  (relevant only for "GaussianARD" and "GRE").
    gre_base    : torch.Tensor or None
        B matrix for GRE.  Required when kernel_spec == "GRE".

    Returns
    -------
    list[float]
        Raw parameter initialisation.
    """
    from .constants import AMP_RAW_FOR_SIGMA_1
    A = AMP_RAW_FOR_SIGMA_1

    if kernel_spec == "Gaussian":
        return [A, 0.1]
    elif kernel_spec == "GaussianARD":
        return [A] + [0.1] * dimension
    elif kernel_spec in ("Matern1/2", "Matern3/2", "Matern5/2"):
        return [A, 1.0]
    elif kernel_spec == "white":
        return [A]
    elif kernel_spec == "GRE":
        # One extra amplitude for GRE on top of the base kernel parameters
        base_spec = "Gaussian"  # default base; caller can override
        base = default_params(base_spec, dimension)
        return [A] + base
    else:
        raise ValueError(f"Unknown kernel specification: '{kernel_spec}'")


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def eval_kernel(
    kernel_spec: str,
    X1: torch.Tensor,
    X2: torch.Tensor,
    x:  torch.Tensor,
    dimension: int         = 1,
    gre_base: Optional[torch.Tensor] = None,
    gre_base_spec: str     = "Gaussian",
) -> torch.Tensor:
    """
    Evaluate a kernel matrix by name.

    Parameters
    ----------
    kernel_spec   : str   kernel name
    X1, X2        : (n1, d), (n2, d)  design points
    x             : raw hyperparameter vector
    dimension     : int  input dimensionality (for ARD / GRE)
    gre_base      : (r, d) or None   rate-function multi-index set (GRE only)
    gre_base_spec : str   base kernel for GRE (default "Gaussian")

    Returns
    -------
    K : (n1, n2) kernel matrix
    """
    X1 = torch.as_tensor(X1, dtype=torch.float64)
    X2 = torch.as_tensor(X2, dtype=torch.float64)
    x  = torch.as_tensor(x,  dtype=torch.float64)

    if kernel_spec == "Gaussian":
        return gaussian(X1, X2, x)
    elif kernel_spec == "GaussianARD":
        return gaussian_ard(X1, X2, x, dimension)
    elif kernel_spec == "Matern1/2":
        return matern12(X1, X2, x)
    elif kernel_spec == "Matern3/2":
        return matern32(X1, X2, x)
    elif kernel_spec == "Matern5/2":
        return matern52(X1, X2, x)
    elif kernel_spec == "white":
        return white_noise(X1, X2, x)
    elif kernel_spec == "GRE":
        if gre_base is None:
            raise ValueError("eval_kernel: gre_base must be provided for GRE kernel.")

        def _base(X1_, X2_, x_):
            return eval_kernel(gre_base_spec, X1_, X2_, x_, dimension)

        return gre(X1, X2, x, gre_base, _base)
    else:
        raise ValueError(f"Unknown kernel specification: '{kernel_spec}'")
