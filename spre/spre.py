"""
Sparse Polynomial Richardson Extrapolation — orchestrator class.

The ``SPRE`` class wires together the functional sub-modules
(normalise, kernels, kriging, optimise, extrapolate, selection) into
a stateful API that is backward-compatible with ``sparse_pre.SPRE``.

Typical workflow
----------------
    sp = SPRE(kernel_spec='Matern3/2', dimension=1)
    sp.set_normalised_data(X, Y)           # max-min normalisation
    sp.set_normalised_data_mad(X, Y)       # MAD normalisation

    result = sp.stepwise_selection()

    result = sp.perform_extrapolation_optimization(A)
    out    = sp.perform_extrapolation(result['x'], A, return_mu_and_var=True)

    out = sp.perform_extrapolation_fixed_hyperparams(
              amplitude=σ², lengthscale=ℓ, A=A)

References
----------
Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*.
"""

import math
from typing import Optional, Union

import torch
from tqdm import tqdm

from .constants  import EPSILON, DEFAULT_NUM_RESTARTS
from .basis      import softplus, stepwise, x2fx
from .kernels    import eval_kernel, default_params as _kernel_default_params
from .normalise  import normalise_maxmin, normalise_mad
from .kriging    import loocv_loss
from .optimise   import optimise_loocv
from .extrapolate import predict_at_zero as _predict_at_zero
from .selection  import stepwise_selection, check_unisolvent


class SPRE:
    """
    Sparse Polynomial Richardson Extrapolation.

    Parameters
    ----------
    kernel_spec : str
        Kernel name: ``"Gaussian"``, ``"GaussianARD"``, ``"Matern1/2"``,
        ``"Matern3/2"``, ``"Matern5/2"``, ``"white"``, or ``"GRE"``.
    dimension   : int
        Input space dimensionality *d*.
    gre_base    : torch.Tensor or None, shape (r, d)
        Multi-index set *B* for the GRE rate function.  Enables GRE mode.
    """

    def __init__(
        self,
        kernel_spec: str,
        dimension:   int,
        gre_base:    Optional[torch.Tensor] = None,
    ):
        self.dimension = dimension
        self._set_kernel(kernel_spec, gre_base)

        self.X_normalised: Optional[torch.Tensor] = None
        self.Y_normalised: Optional[torch.Tensor] = None
        self.nX:           Optional[torch.Tensor] = None
        self.nY:           float = 1.0
        self.Y_mean:       float = 0.0
        self._use_mad:     bool  = False


    def _set_kernel(
        self,
        kernel_spec: str,
        gre_base:    Optional[torch.Tensor],
    ) -> None:
        if gre_base is None:
            self.kernel_spec = kernel_spec
            self.kernel_base = None
        else:
            self.kernel_spec = "GRE"
            self.kernel_base = kernel_spec

        self.gre_base = gre_base

        spec_for_defaults = "GRE" if gre_base is not None else kernel_spec
        self.default_kernel_parameters: list = _kernel_default_params(
            spec_for_defaults,
            dimension=self.dimension,
            gre_base=gre_base,
        )

    def set_kernel_spec(
        self,
        kernel_spec: str,
        gre_base:    Optional[torch.Tensor] = None,
    ) -> None:
        """Change kernel specification (used during GRE stepwise selection)."""
        self._set_kernel(kernel_spec, gre_base)

    def kernel(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        x:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate K(X1, X2 | x) using the current kernel specification."""
        if x is None:
            x = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
        return eval_kernel(
            self.kernel_spec,
            X1, X2, x,
            dimension=self.dimension,
            gre_base=self.gre_base,
            gre_base_spec=self.kernel_base or "Gaussian",
        )


    def set_normalised_data(self, X, Y, use_mad: bool = False) -> None:
        """
        Normalise and store training data.

        Parameters
        ----------
        X       : (n, d) array-like  raw design points.
        Y       : (n,) array-like    raw observations.
        use_mad : bool               use MAD-based normalisation for Y.
        """
        if use_mad:
            X_norm, Y_norm, nX, nY, Y_mean = normalise_mad(X, Y)
            self._use_mad = True
        else:
            X_norm, Y_norm, nX, nY, Y_mean = normalise_maxmin(X, Y)
            self._use_mad = False

        self.X_normalised = X_norm
        self.Y_normalised = Y_norm
        self.nX           = nX
        self.nY           = float(nY)
        self.Y_mean       = float(Y_mean)

    def set_normalised_data_mad(self, X, Y) -> None:
        """Convenience wrapper: MAD-based normalisation."""
        self.set_normalised_data(X, Y, use_mad=True)


    def extract_hyperparameters(self, x: torch.Tensor) -> dict:
        """Return interpretable hyperparameters extracted from raw vector *x*."""
        x = torch.as_tensor(x, dtype=torch.float64)
        result: dict = {"raw": x.clone(), "kernel_spec": self.kernel_spec}

        if self.kernel_spec in ("Gaussian", "Matern1/2", "Matern3/2", "Matern5/2"):
            result["amplitude"]  = float(EPSILON + softplus(x[0]))
            result["lengthscale"] = float(EPSILON + softplus(x[1]))

        elif self.kernel_spec == "GaussianARD":
            result["amplitude"]  = float(EPSILON + softplus(x[0]))
            result["lengthscale"] = [
                float(EPSILON + softplus(x[i + 1]))
                for i in range(self.dimension)
            ]

        elif self.kernel_spec == "white":
            result["amplitude"]   = float(EPSILON + softplus(x[0]))
            result["lengthscale"] = None

        elif self.kernel_spec == "GRE":
            result["amplitude_gre"] = float(EPSILON + softplus(x[0]))
            result["amplitude"]     = float(EPSILON + softplus(x[1]))
            result["lengthscale"]   = (
                float(EPSILON + softplus(x[2])) if len(x) > 2 else None
            )

        return result

    def hyperparams_to_raw(
        self,
        amplitude:   float,
        lengthscale: Union[float, list],
    ) -> torch.Tensor:
        """
        Convert interpretable hyperparameters to raw (unconstrained) space.

        Inverse of ``softplus``:  ``softplus_inv(y) = log(exp(y − ε) − 1)``.
        """
        def _inv(y: float) -> float:
            y_safe = max(float(y) - EPSILON, 1e-10)
            if y_safe < 20:
                return math.log(math.exp(y_safe) - 1.0 + EPSILON)
            return y_safe                              # softplus ≈ identity for large x

        amp_raw = _inv(amplitude)

        if self.kernel_spec == "white":
            return torch.tensor([amp_raw], dtype=torch.float64)

        elif self.kernel_spec == "GaussianARD":
            if isinstance(lengthscale, (list, tuple)):
                ell_raws = [_inv(l) for l in lengthscale]
            else:
                ell_raws = [_inv(lengthscale)] * self.dimension
            return torch.tensor([amp_raw] + ell_raws, dtype=torch.float64)

        else:  # Gaussian, Matern1/2, Matern3/2, Matern5/2
            return torch.tensor([amp_raw, _inv(lengthscale)], dtype=torch.float64)


    def cv_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        LOOCV log-likelihood via Dubrule's O(n³) formula (higher = better).

        Parameters
        ----------
        x : (p,) raw kernel hyperparameters.
        A : (m, d) polynomial multi-index set.
        """
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)
        K = self.kernel(self.X_normalised, self.X_normalised, x)
        V = x2fx(self.X_normalised, A)
        return loocv_loss(K, V, self.Y_normalised)


    def check_unisolvent(self, A: torch.Tensor) -> int:
        """Return 1 if A is unisolvent on the stored training data, -1 otherwise."""
        return 1 if check_unisolvent(self.X_normalised, A) else -1


    def perform_extrapolation_optimization(
        self,
        A:             torch.Tensor,
        do_jit:        bool                    = False,   # legacy, unused
        x0:            Optional[torch.Tensor]  = None,
        num_restarts:  int                     = DEFAULT_NUM_RESTARTS,
        restart_scale: float                   = 0.5,
        seed:          int                     = 0,
    ) -> dict:
        """
        Optimise LOOCV hyperparameters for polynomial basis *A*.

        Returns
        -------
        dict
            ``'x'``  : (p,) optimal raw hyperparameter vector.
            ``'cv'`` : scalar tensor — *negative* LOOCV log-likelihood
                       (lower is better, consistent with sparse_pre convention).
        """
        A = torch.as_tensor(A, dtype=torch.float64)
        x_base = (
            torch.as_tensor(x0, dtype=torch.float64).flatten()
            if x0 is not None
            else torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
        )

        def loocv_fn(x_raw: torch.Tensor) -> torch.Tensor:
            K = self.kernel(self.X_normalised, self.X_normalised, x_raw)
            V = x2fx(self.X_normalised, A)
            return loocv_loss(K, V, self.Y_normalised)

        best_x, best_val = optimise_loocv(
            loocv_fn, x_base, num_restarts, restart_scale, seed
        )

        return {
            "x":  best_x,
            "cv": torch.tensor(-best_val, dtype=torch.float64),   # neg-ll
        }


    def perform_extrapolation(
        self,
        x:                 torch.Tensor,
        A:                 torch.Tensor,
        return_mu_and_var: bool = False,
    ) -> dict:
        """
        Evaluate LOOCV and optionally extrapolate to x = 0.

        Parameters
        ----------
        x                 : (p,) raw hyperparameters.
        A                 : (m, d) polynomial multi-index set.
        return_mu_and_var : if True, also compute (mu, var) at the origin.

        Returns
        -------
        dict
            ``'cv'`` : LOOCV log-likelihood tensor (higher = better).
            ``'mu'``, ``'var'`` : (if requested) posterior in original scale.
        """
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        cv   = self.cv_loss(x, A)
        out: dict = {"cv": cv}

        if return_mu_and_var:
            mu, var = _predict_at_zero(
                self.kernel, x,
                self.X_normalised, self.Y_normalised, A,
                self.nY, self.Y_mean,
            )
            out["mu"]  = mu
            out["var"] = var

        return out

    def perform_extrapolation_fixed_hyperparams(
        self,
        amplitude:         float,
        lengthscale:       Union[float, list],
        A:                 torch.Tensor,
        return_mu_and_var: bool = True,
    ) -> dict:
        """
        Extrapolate with FIXED (pre-optimised) hyperparameters.

        Converts ``(amplitude, lengthscale)`` to raw space and delegates to
        :meth:`perform_extrapolation`.

        Returns
        -------
        dict
            Same as :meth:`perform_extrapolation` plus ``'x'``,
            ``'amplitude'``, ``'lengthscale'``.
        """
        x_raw = self.hyperparams_to_raw(amplitude, lengthscale)
        out   = self.perform_extrapolation(x_raw, A, return_mu_and_var=return_mu_and_var)
        out.update({"x": x_raw, "amplitude": amplitude, "lengthscale": lengthscale})
        return out


    def stepwise_selection(self) -> dict:
        """
        Greedy forward selection of polynomial basis and hyperparameters.

        Delegates to the standard SPRE algorithm for non-GRE kernels or the
        GRE variant (which expands the rate-function basis *B*) otherwise.

        Returns
        -------
        dict
            ``'cv'``, ``'mu'``, ``'var'`` (original scale),
            ``'A'`` (selected basis), ``'x'`` (optimal raw hyperparameters).
            GRE mode additionally returns ``'B'`` (selected rate basis).
        """
        n_train = self.X_normalised.shape[0]

        if self.kernel_base is not None:
            return self._gre_stepwise()

        def optimise_fn(A_: torch.Tensor):
            res = self.perform_extrapolation_optimization(A_)
            return res["x"], float(res["cv"].item())      # (x_opt, neg_ll)

        A_opt, fit = stepwise_selection(
            optimise_fn, self.X_normalised, self.dimension, n_train
        )

        x_opt = fit["x"]
        out   = self.perform_extrapolation(x_opt, A_opt, return_mu_and_var=True)
        out["A"] = A_opt
        out["x"] = x_opt
        return out

    def _gre_stepwise(self) -> dict:
        """
        GRE stepwise selection: expand rate-function basis *B* with fixed *A*.
        """
        A = torch.zeros(1, self.dimension, dtype=torch.int64)
        B = torch.zeros(1, self.dimension, dtype=torch.int64)

        self.set_kernel_spec(self.kernel_base, B)
        res  = self.perform_extrapolation_optimization(A)
        cv   = float(res["cv"].item())
        fit  = res

        order    = 0
        carry_on = True

        while carry_on:
            order  += 1
            B_extra = stepwise(B, order)
            n_extra = B_extra.shape[0]

            if n_extra == 0:
                break

            print(f"Fitting GRE rate interactions of order {order}...")
            to_include = torch.zeros(n_extra, dtype=torch.bool)

            for i in tqdm(range(n_extra), desc="GRE stepwise progress"):
                B_new = torch.cat([B, B_extra[i : i + 1, :]], dim=0)
                self.set_kernel_spec(self.kernel_base, B_new)
                res_new = self.perform_extrapolation_optimization(A)
                cv_new  = float(res_new["cv"].item())
                if cv_new < cv:
                    to_include[i] = True

            if to_include.any():
                B_updated = torch.cat([B, B_extra[to_include, :]], dim=0)
                self.set_kernel_spec(self.kernel_base, B_updated)
                res_up = self.perform_extrapolation_optimization(A)
                cv_up  = float(res_up["cv"].item())

                if cv_up >= cv:
                    carry_on = False
                else:
                    B, cv, fit = B_updated, cv_up, res_up
            else:
                carry_on = False

        x_opt = fit["x"]
        self.set_kernel_spec(self.kernel_base, B)
        out       = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)
        out["B"]  = B
        out["A"]  = A
        out["x"]  = x_opt
        return out
