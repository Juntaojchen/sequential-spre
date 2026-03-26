# -*- coding: utf-8 -*-
"""
Dynamic MLE Parameter Estimation with Variance Inflation

This module implements:
1. Local MLE estimation with sliding window
2. Variance Inflation (Cold Posterior) for overconfidence correction
3. Stability constraints with smoothing filters

References:
- arXiv:2001.10965 - MLE bias in GP regression
- arXiv:2008.05912 - Cold Posteriors

Author: SPRE Research Team
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# Numerical stability constants
EPSILON = 1e-10
JITTER = 1e-8


@dataclass
class DynamicMLEConfig:
    """Configuration for Dynamic MLE estimation."""
    window_size: int = 20           # Sliding window size
    temperature: float = 1.5        # Variance inflation factor T (Cold Posterior)
    smoothing_alpha: float = 0.3    # Exponential smoothing factor (0 < alpha < 1)
    use_standardization: bool = True
    min_sigma: float = 1e-6         # Minimum allowed sigma
    max_sigma: float = 1e6          # Maximum allowed sigma
    use_cholesky: bool = True       # Use Cholesky decomposition for stability


class DynamicMLEEstimator:
    """
    Dynamic MLE estimator for SPRE with variance inflation and smoothing.

    Implements three core functionalities:
    1. Local MLE estimation: σ²_MLE = (1/n) y^T K_φ^{-1} y
    2. Variance inflation: σ²_adj = σ²_MLE × T (Cold Posterior)
    3. Smoothing filter: σ_t = α × σ_new + (1-α) × σ_{t-1}

    Example:
        estimator = DynamicMLEEstimator(config)
        for t in range(len(trajectory)):
            X_window, Y_window = get_window(t)
            sigma = estimator.update_parameters(X_window, Y_window, kernel_func)
    """

    def __init__(self, config: Optional[DynamicMLEConfig] = None):
        """
        Initialize the estimator.

        Parameters
        ----------
        config : DynamicMLEConfig, optional
            Configuration object. If None, uses default settings.
        """
        self.config = config or DynamicMLEConfig()

        # State variables for smoothing
        self._prev_sigma: Optional[float] = None
        self._sigma_history: List[float] = []
        self._raw_sigma_history: List[float] = []  # Before smoothing

        # Standardization parameters (updated each window)
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._Y_mean: Optional[float] = None
        self._Y_std: Optional[float] = None

    def reset(self):
        """Reset the estimator state."""
        self._prev_sigma = None
        self._sigma_history = []
        self._raw_sigma_history = []

    def _standardize(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize input data to prevent numerical instability.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input features
        Y : np.ndarray, shape (n,)
            Target values

        Returns
        -------
        X_std, Y_std : standardized arrays
        """
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + EPSILON
        self._Y_mean = Y.mean()
        self._Y_std = Y.std() + EPSILON

        X_standardized = (X - self._X_mean) / self._X_std
        Y_standardized = (Y - self._Y_mean) / self._Y_std

        return X_standardized, Y_standardized

    def _compute_kernel_matrix(
        self,
        X: np.ndarray,
        lengthscale: float = 1.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Compute Gaussian (RBF) kernel matrix.

        K(x, x') = amplitude * exp(-||x - x'||² / (2 * lengthscale²))

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input points
        lengthscale : float
            Kernel lengthscale
        amplitude : float
            Kernel amplitude (set to 1 for MLE estimation of σ²)

        Returns
        -------
        K : np.ndarray, shape (n, n)
            Kernel matrix
        """
        n = X.shape[0]

        # Compute pairwise squared distances
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        dist_sq = X_sq + X_sq.T - 2 * X @ X.T
        dist_sq = np.maximum(dist_sq, 0)  # Numerical stability

        # RBF kernel
        K = amplitude * np.exp(-dist_sq / (2 * lengthscale ** 2 + EPSILON))

        return K

    def _stable_inverse(self, K: np.ndarray) -> np.ndarray:
        """
        Compute stable matrix inverse using Cholesky or regularization.

        Parameters
        ----------
        K : np.ndarray, shape (n, n)
            Positive semi-definite matrix

        Returns
        -------
        K_inv : np.ndarray, shape (n, n)
            Inverse of K
        """
        n = K.shape[0]

        # Add jitter for numerical stability
        K_reg = K + JITTER * np.eye(n)

        if self.config.use_cholesky:
            try:
                # Cholesky decomposition: K = L @ L.T
                L = np.linalg.cholesky(K_reg)
                # Solve L @ L.T @ K_inv = I
                L_inv = np.linalg.solve(L, np.eye(n))
                K_inv = L_inv.T @ L_inv
            except np.linalg.LinAlgError:
                # Fallback to regularized inverse
                K_reg = K + 1e-4 * np.eye(n)
                K_inv = np.linalg.inv(K_reg)
        else:
            K_inv = np.linalg.inv(K_reg)

        return K_inv

    def compute_mle_sigma(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lengthscale: float = 1.0
    ) -> float:
        """
        Compute MLE estimate of σ² using closed-form solution.

        σ²_MLE = (1/n) y^T K_φ^{-1} y

        where K_φ is the kernel matrix with unit amplitude.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input features (will be standardized if config.use_standardization)
        Y : np.ndarray, shape (n,)
            Target values (will be standardized if config.use_standardization)
        lengthscale : float
            Kernel lengthscale parameter

        Returns
        -------
        sigma_sq : float
            MLE estimate of σ²
        """
        # Ensure correct shapes
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y).flatten()
        n = len(Y)

        if n < 2:
            return self.config.min_sigma ** 2

        # Standardization
        if self.config.use_standardization:
            X, Y = self._standardize(X, Y)

        # Compute kernel with unit amplitude
        K = self._compute_kernel_matrix(X, lengthscale=lengthscale, amplitude=1.0)

        # Stable inverse
        K_inv = self._stable_inverse(K)

        # MLE closed-form: σ²_MLE = (1/n) y^T K^{-1} y
        sigma_sq = (Y @ K_inv @ Y) / n

        # Clamp to valid range
        sigma_sq = np.clip(sigma_sq, self.config.min_sigma ** 2, self.config.max_sigma ** 2)

        return float(sigma_sq)

    def apply_variance_inflation(self, sigma_sq: float) -> float:
        """
        Apply Cold Posterior variance inflation.

        σ²_adjusted = σ²_MLE × T

        Parameters
        ----------
        sigma_sq : float
            Raw MLE sigma squared

        Returns
        -------
        sigma_sq_inflated : float
            Inflated sigma squared
        """
        return sigma_sq * self.config.temperature

    def apply_smoothing(self, sigma: float) -> float:
        """
        Apply exponential smoothing filter for stability.

        σ_t = α × σ_new + (1-α) × σ_{t-1}

        This prevents wild jumps in error bars between adjacent time points.

        Parameters
        ----------
        sigma : float
            New sigma estimate

        Returns
        -------
        sigma_smoothed : float
            Smoothed sigma
        """
        alpha = self.config.smoothing_alpha

        if self._prev_sigma is None:
            sigma_smoothed = sigma
        else:
            sigma_smoothed = alpha * sigma + (1 - alpha) * self._prev_sigma

        self._prev_sigma = sigma_smoothed
        return sigma_smoothed

    def update_parameters(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lengthscale: float = 1.0,
        apply_inflation: bool = True,
        apply_smoothing: bool = True
    ) -> Dict[str, float]:
        """
        Main update function: compute optimized σ from window data.

        This is the core function to be called in the SPRE prediction loop.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Current window input features
        Y : np.ndarray, shape (n,)
            Current window target values
        lengthscale : float
            Kernel lengthscale (can be optimized separately)
        apply_inflation : bool
            Whether to apply variance inflation
        apply_smoothing : bool
            Whether to apply smoothing filter

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'sigma': final sigma estimate
            - 'sigma_sq': final sigma squared
            - 'sigma_raw': raw MLE sigma (before inflation/smoothing)
            - 'sigma_inflated': sigma after inflation (before smoothing)
        """
        # Step 1: Compute raw MLE sigma
        sigma_sq_mle = self.compute_mle_sigma(X, Y, lengthscale)
        sigma_raw = np.sqrt(sigma_sq_mle)

        # Store raw sigma for diagnostics
        self._raw_sigma_history.append(sigma_raw)

        # Step 2: Apply variance inflation (Cold Posterior)
        if apply_inflation:
            sigma_sq_inflated = self.apply_variance_inflation(sigma_sq_mle)
        else:
            sigma_sq_inflated = sigma_sq_mle
        sigma_inflated = np.sqrt(sigma_sq_inflated)

        # Step 3: Apply smoothing filter
        if apply_smoothing:
            sigma_final = self.apply_smoothing(sigma_inflated)
        else:
            sigma_final = sigma_inflated

        # Store final sigma
        self._sigma_history.append(sigma_final)

        return {
            'sigma': sigma_final,
            'sigma_sq': sigma_final ** 2,
            'sigma_raw': sigma_raw,
            'sigma_inflated': sigma_inflated,
        }

    def get_sigma_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get history of sigma estimates.

        Returns
        -------
        raw_history : np.ndarray
            Raw MLE sigma values
        smoothed_history : np.ndarray
            Final (smoothed + inflated) sigma values
        """
        return np.array(self._raw_sigma_history), np.array(self._sigma_history)


class DynamicMLEVisualizer:
    """
    Visualization tools for Dynamic MLE estimation results.
    """

    @staticmethod
    def plot_predictions_with_ci(
        t: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma: np.ndarray,
        title: str = "Predictions with Confidence Interval",
        n_std: float = 2.0,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot predictions with confidence intervals using fill_between.

        Parameters
        ----------
        t : np.ndarray
            Time points
        y_true : np.ndarray
            True trajectory
        y_pred : np.ndarray
            Predicted values
        sigma : np.ndarray
            Standard deviation at each point
        title : str
            Plot title
        n_std : float
            Number of standard deviations for CI (default: 2 for ~95%)
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure

        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot confidence interval
        lower = y_pred - n_std * sigma
        upper = y_pred + n_std * sigma
        ax.fill_between(t, lower, upper, alpha=0.3, color='red',
                       label=f'{int(n_std*2)}σ CI (Inflated)')

        # Plot predictions and true values
        ax.plot(t, y_true, 'b-', lw=1.5, label='True trajectory', alpha=0.8)
        ax.plot(t, y_pred, 'r--', lw=1.5, label='Prediction')

        # Highlight points outside CI
        outside = np.abs(y_true - y_pred) > n_std * sigma
        if outside.any():
            ax.scatter(t[outside], y_true[outside], c='orange', s=50,
                      marker='x', linewidths=2, zorder=5,
                      label=f'Outside CI ({outside.sum()}/{len(t)})')

        # Coverage statistics
        coverage = 1 - outside.mean()
        ax.set_title(f'{title}\nCoverage: {coverage*100:.1f}%')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_sigma_evolution(
        t: np.ndarray,
        sigma_raw: np.ndarray,
        sigma_smoothed: np.ndarray,
        title: str = "Sigma Evolution",
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the evolution of sigma estimates over time.

        Parameters
        ----------
        t : np.ndarray
            Time points
        sigma_raw : np.ndarray
            Raw MLE sigma values
        sigma_smoothed : np.ndarray
            Smoothed sigma values
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Raw vs Smoothed
        ax1 = axes[0]
        ax1.plot(t, sigma_raw, 'b-', alpha=0.5, lw=1, label='Raw MLE')
        ax1.plot(t, sigma_smoothed, 'r-', lw=2, label='Smoothed + Inflated')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Sigma')
        ax1.set_title('Sigma: Raw vs Smoothed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Sigma jumps (derivative)
        ax2 = axes[1]
        if len(t) > 1:
            jumps_raw = np.abs(np.diff(sigma_raw))
            jumps_smoothed = np.abs(np.diff(sigma_smoothed))
            ax2.plot(t[1:], jumps_raw, 'b-', alpha=0.5, lw=1, label='Raw jumps')
            ax2.plot(t[1:], jumps_smoothed, 'r-', lw=2, label='Smoothed jumps')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('|Δσ|')
            ax2.set_title('Sigma Stability (smaller = more stable)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_error_vs_confidence(
        errors: np.ndarray,
        sigmas: np.ndarray,
        title: str = "Error vs Confidence",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Scatter plot of prediction error vs model confidence.

        Ideal: positive correlation
        Overconfident: points in bottom-right (high error, low sigma)
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(errors, sigmas, c='blue', alpha=0.6, s=50)

        # Correlation
        if len(errors) > 2:
            corr = np.corrcoef(errors, sigmas)[0, 1]

            # Trend line
            z = np.polyfit(errors, sigmas, 1)
            p = np.poly1d(z)
            x_line = np.linspace(errors.min(), errors.max(), 50)
            ax.plot(x_line, p(x_line), 'k--', lw=2, alpha=0.7,
                   label=f'Trend (slope={z[0]:.3f})')

            # Mark overconfident region
            med_err = np.median(errors)
            med_sig = np.median(sigmas)
            overconf = (errors > med_err) & (sigmas < med_sig)

            if overconf.any():
                ax.scatter(errors[overconf], sigmas[overconf], c='red',
                          s=80, marker='x', linewidths=2,
                          label=f'Overconfident ({overconf.sum()})')

            ax.axvline(med_err, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(med_sig, color='gray', linestyle=':', alpha=0.5)

            ax.set_title(f'{title}\nCorrelation: {corr:.3f}')
        else:
            ax.set_title(title)

        ax.set_xlabel('Prediction Error |y_true - y_pred|')
        ax.set_ylabel('Model Sigma')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


# ============================================================
# Integration with SPRE
# ============================================================

def create_spre_dynamic_estimator(
    window_size: int = 20,
    temperature: float = 1.5,
    smoothing_alpha: float = 0.3
) -> DynamicMLEEstimator:
    """
    Factory function to create a configured DynamicMLEEstimator for SPRE.

    Recommended settings based on experiments:
    - window_size=20: Balance between local adaptation and stability
    - temperature=1.5: Moderate inflation to cover ~95% of points
    - smoothing_alpha=0.3: Strong smoothing to prevent jumps

    Parameters
    ----------
    window_size : int
        Number of recent points to use for local estimation
    temperature : float
        Variance inflation factor (T > 1 for wider CI)
    smoothing_alpha : float
        Smoothing factor (0 < alpha < 1, smaller = more smoothing)

    Returns
    -------
    estimator : DynamicMLEEstimator
    """
    config = DynamicMLEConfig(
        window_size=window_size,
        temperature=temperature,
        smoothing_alpha=smoothing_alpha,
        use_standardization=True,
        use_cholesky=True,
    )
    return DynamicMLEEstimator(config)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    print("Dynamic MLE Estimator - Example Usage")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_points = 100
    t = np.linspace(0, 5, n_points)
    y_true = np.sin(2 * np.pi * t) * np.exp(-0.3 * t)
    noise = 0.1 * np.random.randn(n_points)
    y_obs = y_true + noise

    # Create estimator
    estimator = create_spre_dynamic_estimator(
        window_size=15,
        temperature=1.5,
        smoothing_alpha=0.3
    )

    # Simulate prediction loop with sliding window
    window_size = 15
    predictions = []
    sigmas = []

    for i in range(window_size, n_points):
        # Get window data
        X_window = t[i-window_size:i].reshape(-1, 1)
        Y_window = y_obs[i-window_size:i]

        # Update parameters
        result = estimator.update_parameters(X_window, Y_window, lengthscale=0.5)

        # Simple prediction (use last value as proxy)
        predictions.append(y_obs[i-1])
        sigmas.append(result['sigma'])

    predictions = np.array(predictions)
    sigmas = np.array(sigmas)
    t_pred = t[window_size:]
    y_true_pred = y_true[window_size:]

    # Visualize
    viz = DynamicMLEVisualizer()

    # Plot 1: Predictions with CI
    fig1 = viz.plot_predictions_with_ci(
        t_pred, y_true_pred, predictions, sigmas,
        title="Dynamic MLE with Variance Inflation",
        save_path="dynamic_mle_example.png"
    )

    # Plot 2: Sigma evolution
    sigma_raw, sigma_smoothed = estimator.get_sigma_history()
    fig2 = viz.plot_sigma_evolution(
        t_pred, sigma_raw, sigma_smoothed,
        title="Sigma Evolution",
        save_path="dynamic_mle_sigma.png"
    )

    plt.show()
    print("\nExample completed. Figures saved.")
