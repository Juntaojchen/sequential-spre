"""
Numerical stability constants shared across all SPRE modules.
"""

# Guard against division by zero in kernel and normalisation formulas
EPSILON: float = 1e-10

# Diagonal perturbation added to the augmented Kriging system M before
# inversion to prevent numerical singularity.
JITTER: float = 1e-12

# Default number of random restarts for multi-start hyperparameter
# optimisation (see optimise.py).
DEFAULT_NUM_RESTARTS: int = 10

# Raw parameter value such that softplus(AMP_RAW_FOR_SIGMA_1) ≈ 1.0.
# Used as the default amplitude initialisation.
#   softplus_inv(1.0) = log(exp(1) - 1) ≈ 0.5413
AMP_RAW_FOR_SIGMA_1: float = 0.5413
