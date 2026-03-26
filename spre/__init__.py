"""
spre — Sparse Polynomial Richardson Extrapolation.

This package provides a clean, modular implementation of the SPRE algorithm
for GP-based Richardson extrapolation with greedy polynomial basis selection.

Orchestrator
------------
SPRE
    Stateful class combining data normalisation, kernel evaluation, LOOCV
    optimisation, and posterior prediction at the origin.

Deterministic baseline
-----------------------
mre
    Multivariate Richardson Extrapolation (polynomial least-squares, no GP).

Functional building blocks
---------------------------
eval_kernel       Evaluate a kernel matrix by name.
default_params    Default raw hyperparameter vector for a named kernel.
loocv_loss        LOOCV log-likelihood via Dubrule's O(n³) formula.
reml_loss         REML log-likelihood (Patterson & Thompson 1971).
reml_sigma_mle    Closed-form REML amplitude estimate.
optimise_loocv    Multi-start L-BFGS optimiser for LOOCV.
predict_at_zero   GP posterior at the origin in original scale.
normalise_maxmin  Max-min normalisation.
normalise_mad     MAD-based normalisation.
denormalise       Inverse normalisation.
x2fx              Polynomial design matrix (monomial expansion).
softplus          Numerically stable positive activation.
"""

from .spre       import SPRE
from .mre        import mre
from .kernels    import eval_kernel, default_params
from .kriging    import loocv_loss, reml_loss, reml_sigma_mle, kriging_predict
from .optimise   import optimise_loocv
from .extrapolate import predict_at_zero, predict_normalised
from .normalise  import normalise_maxmin, normalise_mad, denormalise
from .basis      import x2fx, softplus, stepwise
from .selection  import stepwise_selection, check_unisolvent

__all__ = [
    "SPRE",
    "mre",
    "eval_kernel",
    "default_params",
    "loocv_loss",
    "reml_loss",
    "reml_sigma_mle",
    "kriging_predict",
    "optimise_loocv",
    "predict_at_zero",
    "predict_normalised",
    "normalise_maxmin",
    "normalise_mad",
    "denormalise",
    "x2fx",
    "softplus",
    "stepwise",
    "stepwise_selection",
    "check_unisolvent",
]
