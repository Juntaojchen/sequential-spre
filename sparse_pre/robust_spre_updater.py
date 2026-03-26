
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class RobustSPREConfig:
    """Configuration for Robust SPRE Updater."""

    base_ridge: float = 1e-4           
    adaptive_ridge_scale: float = 0.01  
    max_ridge: float = 1.0              

    min_sigma: float = 1e-4            
    max_sigma: float = 100.0            

    temperature: float = 1.5           

    residual_window: int = 10           
    residual_weight: float = 0.5      

    ewma_alpha: float = 0.1             
    slew_rate_limit: float = 0.15       
    stability_threshold: float = 0.3    

    default_lengthscale: float = 1.0


class RobustSPREUpdater:


    def __init__(self, config: Optional[RobustSPREConfig] = None):
        """
        Initialize the robust updater.

        Parameters
        ----------
        config : RobustSPREConfig, optional
            Configuration object. If None, uses default settings.
        """
        self.config = config or RobustSPREConfig()

        self._prev_sigma: Optional[float] = None
        self._sigma_history: List[float] = []
        self._raw_sigma_history: List[float] = []

        self._residual_buffer: deque = deque(maxlen=self.config.residual_window)

        self._diagnostics: Dict = {}

    def reset(self):
        """Reset all state variables."""
        self._prev_sigma = None
        self._sigma_history = []
        self._raw_sigma_history = []
        self._residual_buffer.clear()
        self._diagnostics = {}


    def _compute_kernel_matrix(
        self,
        X: np.ndarray,
        lengthscale: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute Gaussian RBF kernel matrix.

        K(x, x') = exp(-||x - x'||^2 / (2 * l^2))
        """
        if lengthscale is None:
            lengthscale = self.config.default_lengthscale

        n = X.shape[0]

        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        dist_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
        dist_sq = np.maximum(dist_sq, 0)  # Numerical stability

        K = np.exp(-dist_sq / (2 * lengthscale ** 2 + 1e-10))

        return K

    def _compute_adaptive_ridge(self, K: np.ndarray) -> float:
        """
        Compute adaptive ridge coefficient based on matrix condition.

        The ridge is scaled by the trace of K to be scale-invariant.
        """
        n = K.shape[0]
        trace_K = np.trace(K)

        ridge = self.config.base_ridge + \
                self.config.adaptive_ridge_scale * (trace_K / n)

        ridge = min(ridge, self.config.max_ridge)

        return ridge

    def _stable_cholesky_solve(
        self,
        K: np.ndarray,
        y: np.ndarray,
        ridge: float
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve K @ alpha = y using Cholesky decomposition with ridge.

        Returns (alpha, success_flag).
        """
        n = K.shape[0]
        K_reg = K + ridge * np.eye(n)

        try:
            L = np.linalg.cholesky(K_reg)

            z = np.linalg.solve(L, y)

            alpha = np.linalg.solve(L.T, z)

            return alpha, True

        except np.linalg.LinAlgError:
            warnings.warn(f"Cholesky failed with ridge={ridge:.2e}, using fallback")

            K_reg = K + self.config.max_ridge * np.eye(n)
            try:
                alpha = np.linalg.solve(K_reg, y)
                return alpha, False
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_reg, y, rcond=None)[0]
                return alpha, False

    def _compute_regularized_mle_sigma(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lengthscale: Optional[float] = None
    ) -> Tuple[float, Dict]:
       
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y).flatten()
        n = len(Y)

        if n < 2:
            return self.config.min_sigma, {'status': 'insufficient_data'}

        X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-10
        Y_mean, Y_std = Y.mean(), Y.std() + 1e-10

        X_norm = (X - X_mean) / X_std
        Y_norm = (Y - Y_mean) / Y_std

        K = self._compute_kernel_matrix(X_norm, lengthscale)

        ridge = self._compute_adaptive_ridge(K)

        alpha, cholesky_success = self._stable_cholesky_solve(K, Y_norm, ridge)

        sigma_sq_normalized = np.dot(Y_norm, alpha) / n

        sigma_sq_normalized = max(sigma_sq_normalized, 1e-10)

        sigma_sq = sigma_sq_normalized * (Y_std ** 2)
        sigma = np.sqrt(sigma_sq)

        sigma = np.clip(sigma, self.config.min_sigma, self.config.max_sigma)

        diagnostics = {
            'ridge_used': ridge,
            'cholesky_success': cholesky_success,
            'sigma_sq_normalized': sigma_sq_normalized,
            'Y_std': Y_std,
            'condition_number': np.linalg.cond(K) if n < 100 else None,
        }

        return sigma, diagnostics


    def _update_residual_buffer(self, error: float):
        """Add prediction error to the rolling buffer."""
        self._residual_buffer.append(abs(error))

    def _get_rolling_error(self) -> float:

        if len(self._residual_buffer) == 0:
            return 0.0

        return np.mean(list(self._residual_buffer))

    def _apply_residual_feedback(
        self,
        sigma_mle: float,
        rolling_error: float
    ) -> float:

        sigma_inflated = sigma_mle * np.sqrt(self.config.temperature)

        sigma_residual = self.config.residual_weight * rolling_error

        sigma_combined = max(sigma_inflated, sigma_residual)

        return sigma_combined


    def _apply_ewma(self, sigma_new: float) -> float:

        alpha = self.config.ewma_alpha

        if self._prev_sigma is None:
            return sigma_new

        return alpha * sigma_new + (1 - alpha) * self._prev_sigma

    def _apply_slew_rate_limit(self, sigma_new: float) -> float:

        if self._prev_sigma is None:
            return sigma_new

        max_change = self.config.slew_rate_limit * self._prev_sigma

        delta = sigma_new - self._prev_sigma
        delta_clamped = np.clip(delta, -max_change, max_change)

        return self._prev_sigma + delta_clamped


    def update(
        self,
        X_window: np.ndarray,
        Y_window: np.ndarray,
        current_prediction_error: Optional[float] = None,
        lengthscale: Optional[float] = None,
    ) -> Tuple[float, bool]:

        sigma_mle, mle_diagnostics = self._compute_regularized_mle_sigma(
            X_window, Y_window, lengthscale
        )
        self._raw_sigma_history.append(sigma_mle)

        if current_prediction_error is not None:
            self._update_residual_buffer(current_prediction_error)

        rolling_error = self._get_rolling_error()
        sigma_feedback = self._apply_residual_feedback(sigma_mle, rolling_error)

        sigma_smoothed = self._apply_ewma(sigma_feedback)

        sigma_limited = self._apply_slew_rate_limit(sigma_smoothed)

        sigma_final = np.clip(
            sigma_limited,
            self.config.min_sigma,
            self.config.max_sigma
        )

        is_stable = True
        if self._prev_sigma is not None:
            relative_change = abs(sigma_final - self._prev_sigma) / (self._prev_sigma + 1e-10)
            is_stable = relative_change < self.config.stability_threshold

        is_stable = is_stable and mle_diagnostics.get('cholesky_success', True)

        self._prev_sigma = sigma_final
        self._sigma_history.append(sigma_final)

        self._diagnostics = {
            'sigma_mle': sigma_mle,
            'sigma_feedback': sigma_feedback,
            'sigma_smoothed': sigma_smoothed,
            'sigma_limited': sigma_limited,
            'sigma_final': sigma_final,
            'rolling_error': rolling_error,
            'is_stable': is_stable,
            **mle_diagnostics
        }

        return sigma_final, is_stable

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information from the last update."""
        return self._diagnostics.copy()

    def get_history(self) -> Tuple[np.ndarray, np.ndarray]:
       
        return np.array(self._raw_sigma_history), np.array(self._sigma_history)



def create_robust_updater(
    max_sigma: float = 100.0,
    temperature: float = 1.5,
    ewma_alpha: float = 0.1,
    slew_rate_limit: float = 0.15,
    residual_window: int = 10,
) -> RobustSPREUpdater:
   
    config = RobustSPREConfig(
        max_sigma=max_sigma,
        temperature=temperature,
        ewma_alpha=ewma_alpha,
        slew_rate_limit=slew_rate_limit,
        residual_window=residual_window,
    )
    return RobustSPREUpdater(config)



if __name__ == "__main__":
    print("Robust SPRE Updater - Example")
    print("=" * 50)

    np.random.seed(42)
    n_points = 100
    t = np.linspace(0, 5, n_points)
    y_true = np.sin(2 * np.pi * t) * np.exp(-0.3 * t)
    y_obs = y_true + 0.1 * np.random.randn(n_points)

    updater = create_robust_updater(
        max_sigma=10.0,
        temperature=1.5,
        ewma_alpha=0.1,
        slew_rate_limit=0.15,
    )

    window_size = 15
    sigmas = []
    errors = []
    stability_flags = []

    for i in range(window_size, n_points):
        X_window = t[i-window_size:i].reshape(-1, 1)
        Y_window = y_obs[i-window_size:i]

        pred = y_obs[i-1]
        error = y_true[i] - pred
        errors.append(abs(error))

        sigma, is_stable = updater.update(
            X_window, Y_window,
            current_prediction_error=error
        )
        sigmas.append(sigma)
        stability_flags.append(is_stable)

    sigmas = np.array(sigmas)
    errors = np.array(errors)

    raw_sigmas, final_sigmas = updater.get_history()

    print(f"\nResults:")
    print(f"  Mean raw sigma:   {raw_sigmas.mean():.4f}")
    print(f"  Mean final sigma: {final_sigmas.mean():.4f}")
    print(f"  Max sigma jump:   {np.abs(np.diff(final_sigmas)).max():.4f}")
    print(f"  Stability rate:   {np.mean(stability_flags)*100:.1f}%")

    corr = np.corrcoef(errors, sigmas)[0, 1]
    print(f"  Error-sigma corr: {corr:.3f}")

    print("\nExample completed.")
