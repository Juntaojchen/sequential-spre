"""
lorenz_sequential — Sequential SPRE for the Lorenz system.

Public API
----------
run_lorenz_experiment   Full experiment (data → CV → fits → predictions → metrics)
select_lambda_cv        k-fold CV grid search for (lambda_sigma, lambda_ell)
fit_sequential          Joint MAP optimisation of Sequential SPRE hyperparameters
fit_reml_single         Independent per-time-point REML fit (baseline)
fit_loocv_single        Independent per-time-point LOOCV fit (kept for reference)
geom_means              Geometric-mean aggregation of independent fits (§3.1)
predict_at_zero         GP posterior predictive at h → 0
evaluate                Calibration metrics for predictive Gaussians
plot_comparison         Error-bar + z-score comparison figure
plot_hyperparams        Hyperparameter trajectory figure
plot_pit                PIT histogram figure
LorenzSystem            Lorenz attractor integrator (Euler / RK4)
"""

from .experiment    import run_lorenz_experiment
from .lambda_cv     import select_lambda_cv
from .fitting       import fit_sequential, fit_reml_single, fit_loocv_single
from .init_utils    import geom_means, smooth_init
from .predict       import predict_at_zero
from .metrics       import evaluate
from .plotting      import plot_comparison, plot_hyperparams, plot_pit
from .lorenz        import LorenzSystem
from .lmc_baseline  import run_lmc_lorenz
from .slfm_baseline import run_slfm_lorenz

__all__ = [
    "run_lorenz_experiment",
    "select_lambda_cv",
    "fit_sequential",
    "fit_reml_single",
    "fit_loocv_single",
    "geom_means",
    "smooth_init",
    "predict_at_zero",
    "evaluate",
    "plot_comparison",
    "plot_hyperparams",
    "plot_pit",
    "LorenzSystem",
    "run_lmc_lorenz",
    "run_slfm_lorenz",
]
