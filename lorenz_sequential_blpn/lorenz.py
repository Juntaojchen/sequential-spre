"""
Lorenz (1963) attractor: numerical integration utilities.

    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = x y − β z

Default parameters: σ=10, ρ=28, β=8/3, u₀=(−5, 5, 20).
"""

import numpy as np
from scipy.interpolate import CubicSpline


class LorenzSystem:
    """Forward Euler and RK4 integrators for the Lorenz attractor."""

    def __init__(
        self,
        sigma: float = 10.0,
        rho:   float = 28.0,
        beta:  float = 8.0 / 3.0,
        u0: np.ndarray = np.array([-5., 5., 20.]),
    ):
        self.sigma = sigma
        self.rho   = rho
        self.beta  = beta
        self.u0    = np.asarray(u0, dtype=np.float64).copy()

    def _rhs(self, u: np.ndarray) -> np.ndarray:
        x, y, z = u
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ], dtype=np.float64)

    def euler(self, h: float, T: float) -> np.ndarray:
        """
        Integrate with forward Euler at step size h up to time T.
        Returns u(T) ∈ R³ via linear interpolation; NaN on divergence.
        """
        h, T = float(h), float(T)
        t = np.arange(0.0, T + h + 1e-14, h)
        u = np.empty((len(t), 3), dtype=np.float64)
        u[0] = self.u0
        for k in range(len(t) - 1):
            u[k + 1] = u[k] + h * self._rhs(u[k])
            if not np.isfinite(u[k + 1]).all():
                return np.full(3, np.nan)
        k = int(np.clip(np.searchsorted(t, T) - 1, 0, len(t) - 2))
        w = (T - t[k]) / (t[k + 1] - t[k] + 1e-16)
        return (1.0 - w) * u[k] + w * u[k + 1]

    def rk4_reference(self, h: float, T: float) -> np.ndarray:
        """
        Integrate with classical RK4 at step size h, then fit a not-a-knot
        cubic spline and evaluate at T.  Used as the ground-truth reference.
        """
        h, T = float(h), float(T)
        t = np.arange(0.0, T + h + 1e-14, h)
        u = np.empty((len(t), 3), dtype=np.float64)
        u[0] = self.u0
        for k in range(len(t) - 1):
            k1 = self._rhs(u[k])
            k2 = self._rhs(u[k] + 0.5 * h * k1)
            k3 = self._rhs(u[k] + 0.5 * h * k2)
            k4 = self._rhs(u[k] +       h * k3)
            u[k + 1] = u[k] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(u[k + 1]).all():
                return np.full(3, np.nan)
        return np.array([
            CubicSpline(t, u[:, i], bc_type='not-a-knot')(T)
            for i in range(3)
        ])

    def euler_batch(self, h_vals: np.ndarray, T: float) -> np.ndarray:
        """Returns shape (n_h, 3); row i = euler(h_vals[i], T)."""
        return np.stack([self.euler(h, T) for h in h_vals])
