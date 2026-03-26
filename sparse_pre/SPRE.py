import math
import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm

from .helper_functions import (
    x2fx,
    softplus,
    cellsum,
    white,
    remove_row,
    stepwise,
)

# ═══════════════════════════════════════════════════════════════════════
# Numerical Stability Constants
# ═══════════════════════════════════════════════════════════════════════

EPSILON = 1e-10        # Safeguard against division by zero
JITTER = 1e-12           # Diagonal perturbation for matrix inversion
DEFAULT_NUM_RESTARTS = 10  # Multi-start optimization attempts


class SPRE:
    def __init__(self, kernel_spec: str, dimension: int, gre_base: torch.Tensor | None = None):
        """
        Initialize SPRE model.

        Parameters
        ----------
        kernel_spec : str
            Name of the covariance kernel: "Gaussian", "GaussianARD",
            "Matern1/2", "Matern3/2", "white".
            If gre_base is not None, SPRE operates in GRE mode.
        dimension : int
            Spatial dimension d of the input space.
        gre_base : torch.Tensor or None, shape (m, d)
            Polynomial basis for GRE (Gauss-Richardson Extrapolation) compatibility.
            If provided, enables GRE mode with rate function modulation.

        Notes
        -----
        Data normalization parameters (X_norm, Y_norm, sigma_X, sigma_Y) are
        set later via set_normalised_data().
        """
        self.dimension = dimension

        # Initialize kernel specification (supports GRE mode via basis injection)
        self.set_kernel_spec(kernel_spec, gre_base)

        # Data containers (populated by set_normalised_data)
        self.X_normalised: torch.Tensor | None = None  # Normalized design points
        self.Y_normalised: torch.Tensor | None = None  # Normalized responses
        self.nX: torch.Tensor | None = None  # Normalization scale factors per dimension
        self.nY: torch.Tensor | None = None  # Normalization scale factor for response



    def cdist_torch(self, XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between rows of XA and XB.

        Implements ||x_i - x_j'|| for all pairs (i, j), equivalent to
        scipy.spatial.distance.cdist(XA, XB, 'euclidean').

        Parameters
        ----------
        XA : torch.Tensor, shape (m, d)
            First set of points
        XB : torch.Tensor, shape (n, d)
            Second set of points

        Returns
        -------
        distances : torch.Tensor, shape (m, n)
            Pairwise Euclidean distances
        """
        XA = XA.to(torch.float64)
        XB = XB.to(torch.float64)

        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        XA_sq = (XA ** 2).sum(dim=1, keepdim=True)  # (m,1)
        XB_sq = (XB ** 2).sum(dim=1)                # (n,)
        cross = XA @ XB.T                           # (m,n)

        nums = XA_sq - 2 * cross + XB_sq           # (m,n)
        nums = torch.clamp(nums, min=0.0)
        return torch.sqrt(nums)

    def set_kernel_spec(self, kernel_spec: str, gre_base: torch.Tensor | None = None):
        """
        设置 kernel 规格（包括 GRE 模式）。
        """
        if gre_base is None:
            self.kernel_spec = kernel_spec
            self.kernel_base = None
        else:
            # 兼容 GRE：对外暴露 "GRE"，内部保留 base kernel 的名字和 B 矩阵
            self.kernel_spec = "GRE"
            self.kernel_base = kernel_spec

        self.gre_base = gre_base  # B 矩阵（或 None）
        self.set_kernel_default_parameters()

    def set_kernel_default_parameters(self):
        """
        设置 kernel 的初始超参数。

        Raw parameters are transformed via softplus: param = ε + softplus(raw)

        For amplitude σ² = 1.0 (natural after standardization):
            softplus_inv(1.0) = log(exp(1.0) - 1) ≈ 0.5413

        This initialization prevents per-T optimization from pulling σ² → 0.
        """
        # Raw value for σ² = 1.0: softplus_inv(1.0) ≈ 0.5413
        AMP_RAW_FOR_SIGMA_1 = 0.5413

        match self.kernel_spec:
            case "Gaussian":
                # [amp_raw, lengthscale_raw]
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1, 0.1]
            case "GaussianARD":
                # [amp_raw, l1_raw, l2_raw, ..., ld_raw]
                params = [AMP_RAW_FOR_SIGMA_1]
                params.extend([0.1] * self.dimension)
                self.default_kernel_parameters = params
            case "white":
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1]
            case "Matern1/2":
                # [amp_raw, lengthscale_raw]
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1, 1.0]
            case "Matern3/2":
                # [amp_raw, lengthscale_raw]
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1, 1.0]
            case "GRE":
                # 先为 base kernel 设置 default，然后在前面加上一个 GRE amp
                base_spec = self.kernel_base
                self.kernel_spec = base_spec
                self.set_kernel_default_parameters()
                base_defaults = self.default_kernel_parameters
                self.kernel_spec = "GRE"
                self.kernel_base = base_spec

                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1]
                self.default_kernel_parameters.extend(base_defaults)
            case _:
                raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

    def kernel(self, X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """
        Evaluate the kernel K(X1, X2 | x).

        X1: (n1, d)
        X2: (n2, d)
        x:  (p,) hyperparameters
        """
        X1 = torch.as_tensor(X1, dtype=torch.float64)
        X2 = torch.as_tensor(X2, dtype=torch.float64)

        if x is None:
            x = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
        else:
            x = torch.as_tensor(x, dtype=torch.float64)

        spec = self.kernel_spec

        if spec == "Gaussian":
            # Radial Basis Function (RBF) / Squared Exponential kernel
            # Reference: Rasmussen & Williams (2006), Equation 4.9
            #
            # Definition: κ(r) = σ² exp(-r² / 2ℓ²)
            #
            # where:
            #   σ² (amplitude): signal variance
            #   ℓ (lengthscale): characteristic length scale
            #   r = ||x - x'||: Euclidean distance
            amp = EPSILON + softplus(x[0])
            ell = EPSILON + softplus(x[1])
            dist = self.cdist_torch(X1, X2)
            return amp * torch.exp(-(dist ** 2) / (ell ** 2))

        elif spec == "GaussianARD":
            # Automatic Relevance Determination (ARD) Gaussian kernel
            # Reference: Rasmussen & Williams (2006), Section 5.1
            #
            # Definition: κ(x, x') = σ² exp(-Σ_i (x_i - x'_i)² / 2ℓ_i²)
            #
            # Uses dimension-specific lengthscales ℓ_1, ..., ℓ_d to automatically
            # determine the relevance of each input dimension
            amp = EPSILON + softplus(x[0])
            length_terms = []
            for i in range(self.dimension):
                d_i = self.cdist_torch(
                    X1[:, [i]], X2[:, [i]]
                ) ** 2 / (EPSILON + softplus(x[i + 1])) ** 2
                length_terms.append(d_i)
            # Sum squared distances across all dimensions (element-wise)
            sq_dist = cellsum(length_terms)
            return amp * torch.exp(-sq_dist)

        elif spec == "white":
            # White noise kernel (diagonal covariance)
            #
            # Definition: κ(x, x') = σ² δ(x, x')
            #
            # where δ is the Kronecker delta (1 if x = x', 0 otherwise)
            amp = EPSILON + softplus(x[0])
            return amp * white(X1, X2)

        elif spec == "Matern1/2":
            # Matérn covariance with ν = 1/2 (equivalent to exponential kernel)
            # Reference: Rasmussen & Williams (2006), Equation 4.14
            #
            # Definition: κ(r) = σ² exp(-r / ℓ)
            #
            # This kernel produces non-differentiable sample paths
            amp = EPSILON + softplus(x[0])
            ell = EPSILON + softplus(x[1])
            dist = self.cdist_torch(X1, X2)
            return amp * torch.exp(-dist / ell)

        elif spec == "Matern3/2":
            # Matérn covariance with ν = 3/2
            # Reference: Rasmussen & Williams (2006), Equation 4.14
            #
            # Definition: κ(r) = σ² (1 + √3·r/ℓ) exp(-√3·r/ℓ)
            #
            # This kernel produces once-differentiable sample paths
            amp = EPSILON + softplus(x[0])
            ell = EPSILON + softplus(x[1])
            dist = self.cdist_torch(X1, X2)
            r_l = dist / ell
            sqrt3_r_l = math.sqrt(3.0) * r_l
            return amp * (1.0 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)

        elif spec == "GRE":
            # Gauss-Richardson Extrapolation (GRE) kernel with polynomial modulation
            #
            # Definition: κ_GRE(x, x') = σ²_GRE · b(x) · κ_base(x, x') · b(x')
            #
            # where:
            #   b(x) = Σ_j φ_j(x): polynomial rate function
            #   κ_base: underlying base kernel (Gaussian, Matérn, etc.)
            #   σ²_GRE: GRE-specific amplitude parameter
            #
            # This allows the kernel to adaptively model rate functions in
            # Richardson extrapolation
            if self.gre_base is None:
                raise RuntimeError("GRE kernel requires gre_base (B matrix).")
            amp_gre = EPSILON + softplus(x[0])

            # Evaluate polynomial rate function: b(x) = Σ_j φ_j(x)
            base_X1 = x2fx(X1, self.gre_base).sum(dim=1)  # (n1,)
            base_X2 = x2fx(X2, self.gre_base).sum(dim=1)  # (n2,)

            # Temporarily switch to base kernel for evaluation
            base_spec = self.kernel_base
            old_spec = self.kernel_spec
            self.kernel_spec = base_spec
            K_base = self.kernel(X1, X2, x=x[1:])  # Use base kernel hyperparameters
            self.kernel_spec = old_spec

            # Apply polynomial modulation: b(x_i) · K_base · b(x_j)
            return amp_gre * base_X1.unsqueeze(1) * K_base * base_X2.unsqueeze(0)

        else:
            raise ValueError(f"Unknown kernel specification: {spec}")

    # ═══════════════════════════════════════════════════════════════════════
    # III. DATA NORMALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    def set_normalised_data(self, X, Y, use_mad: bool = False):
        """
        Normalize data using max-min scaling or MAD-based scaling.

        Normalization Formula (max-min, default)
        ----------------------------------------
        For inputs X and outputs Y, compute scale factors:
            σ_X = max(X) - min(X) + ε
            σ_Y = max(Y) - min(Y) + ε

        Then normalize:
            X_norm = X / σ_X
            Y_norm = Y / σ_Y

        MAD-based Normalization (use_mad=True)
        --------------------------------------
        For outputs Y, use median absolute deviation:
            Y_norm = (Y - mean(Y)) / (ε + MAD(Y))

        where MAD(Y) = median(|Y - median(Y)|)

        The epsilon term prevents division by zero for constant data.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Input design points
        Y : array-like, shape (n,)
            Output responses
        use_mad : bool, default=False
            If True, use MAD-based standardization for Y

        Notes
        -----
        Normalization improves numerical conditioning for GP inference and
        makes lengthscale hyperparameters more interpretable.
        MAD-based normalization is more robust to outliers.
        """
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64).flatten()

        # Compute normalization scale factors for X (always max-min)
        self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
        self.X_normalised = X / self.nX

        if use_mad:
            # MAD-based standardization for Y
            Y_mean = Y.mean()
            Y_median = Y.median()
            mad = torch.median(torch.abs(Y - Y_median))
            self.nY = mad + EPSILON  # scale factor is MAD
            self.Y_mean = Y_mean  # store mean for de-normalization
            self.Y_normalised = (Y - Y_mean) / self.nY
            self._use_mad = True
        else:
            # Max-min normalization for Y (original behavior)
            self.nY = (Y.max() - Y.min()) + EPSILON
            self.Y_mean = torch.tensor(0.0, dtype=torch.float64)  # no centering
            self.Y_normalised = Y / self.nY
            self._use_mad = False

    def set_normalised_data_mad(self, X, Y):
        """
        Convenience method for MAD-based standardization.

        Formula: Y_norm = (Y - mean(Y)) / (ε + MAD(Y))
        where MAD(Y) = median(|Y - median(Y)|)

        Parameters
        ----------
        X : array-like, shape (n, d)
            Input design points
        Y : array-like, shape (n,)
            Output responses
        """
        self.set_normalised_data(X, Y, use_mad=True)

    # ═══════════════════════════════════════════════════════════════════════
    # III-B. HYPERPARAMETER EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════

    def extract_hyperparameters(self, x: torch.Tensor) -> dict:
        """
        Extract interpretable hyperparameters from raw (unconstrained) parameters.

        The raw parameters are transformed via softplus to ensure positivity:
            param = ε + softplus(x)

        Parameters
        ----------
        x : torch.Tensor
            Raw hyperparameters from optimization

        Returns
        -------
        dict
            Dictionary containing:
            - 'amplitude': kernel amplitude σ²
            - 'lengthscale': kernel lengthscale ℓ (or list for ARD)
            - 'raw': original raw parameters
            - 'kernel_spec': kernel specification name
        """
        x = torch.as_tensor(x, dtype=torch.float64)

        result = {
            'raw': x.clone(),
            'kernel_spec': self.kernel_spec
        }

        if self.kernel_spec == "Gaussian":
            result['amplitude'] = float(EPSILON + softplus(x[0]))
            result['lengthscale'] = float(EPSILON + softplus(x[1]))

        elif self.kernel_spec == "GaussianARD":
            result['amplitude'] = float(EPSILON + softplus(x[0]))
            result['lengthscale'] = [float(EPSILON + softplus(x[i+1]))
                                     for i in range(self.dimension)]

        elif self.kernel_spec in ["Matern1/2", "Matern3/2"]:
            result['amplitude'] = float(EPSILON + softplus(x[0]))
            result['lengthscale'] = float(EPSILON + softplus(x[1]))

        elif self.kernel_spec == "white":
            result['amplitude'] = float(EPSILON + softplus(x[0]))
            result['lengthscale'] = None

        elif self.kernel_spec == "GRE":
            result['amplitude_gre'] = float(EPSILON + softplus(x[0]))
            # Base kernel parameters start at index 1
            result['amplitude'] = float(EPSILON + softplus(x[1]))
            if len(x) > 2:
                result['lengthscale'] = float(EPSILON + softplus(x[2]))
            else:
                result['lengthscale'] = None

        else:
            raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

        return result

    def hyperparams_to_raw(self, amplitude: float, lengthscale: float | list) -> torch.Tensor:
        """
        Convert interpretable hyperparameters back to raw (unconstrained) space.

        Inverse of softplus: softplus_inv(y) = log(exp(y) - 1)

        Parameters
        ----------
        amplitude : float
            Kernel amplitude σ²
        lengthscale : float or list
            Kernel lengthscale ℓ (or list for ARD kernels)

        Returns
        -------
        torch.Tensor
            Raw hyperparameters suitable for kernel evaluation
        """
        def softplus_inv(y):
            """Inverse softplus: log(exp(y - ε) - 1), clamped for stability"""
            y_safe = max(y - EPSILON, 1e-10)
            if y_safe < 20:  # Avoid overflow for large values
                return math.log(math.exp(y_safe) - 1.0 + EPSILON)
            else:
                return y_safe  # softplus(x) ≈ x for large x

        amp_raw = softplus_inv(amplitude)

        if self.kernel_spec == "white":
            return torch.tensor([amp_raw], dtype=torch.float64)

        elif self.kernel_spec == "GaussianARD":
            if isinstance(lengthscale, (list, tuple)):
                ell_raw = [softplus_inv(l) for l in lengthscale]
            else:
                ell_raw = [softplus_inv(lengthscale)] * self.dimension
            return torch.tensor([amp_raw] + ell_raw, dtype=torch.float64)

        else:
            # Gaussian, Matern1/2, Matern3/2
            ell_raw = softplus_inv(lengthscale)
            return torch.tensor([amp_raw, ell_raw], dtype=torch.float64)

    # ═══════════════════════════════════════════════════════════════════════
    # IV. GAUSSIAN PROCESS POSTERIOR INFERENCE & LOOCV
    # ═══════════════════════════════════════════════════════════════════════

    def cv_local_loss(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        row_num: int,
        return_mu_cov: bool = False,
    ):
 
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        X_full = self.X_normalised
        Y_full = self.Y_normalised

        X = remove_row(X_full, row_num)
        Y = remove_row(Y_full.unsqueeze(1), row_num).flatten()
        Xs = X_full[row_num : row_num + 1, :]
        Ys = Y_full[row_num : row_num + 1]

        return self.cv_loss_calculation(A, X, Y, Xs, Ys, x, return_mu_cov=return_mu_cov)

    def check_unisolvent(self, A: torch.Tensor) -> int:
        """
        检查 A 是否生成 unisolvent 集（rank == m）。
        """
        A = torch.as_tensor(A, dtype=torch.float64)
        m = A.shape[0]
        VA = x2fx(self.X_normalised, A)  # (n,m)
        rank = torch.linalg.matrix_rank(VA)
        if rank == m:
            return 1
        else:
            # 和 JAX 版一样，用返回 -1 而不是抛异常
            return -1

    def cv_loss_calculation(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        Xs: torch.Tensor,
        Ys: torch.Tensor,
        x: torch.Tensor,
        return_mu_cov: bool = False,
    ):
        """
        计算 (Xs, Ys) 对应的 GP+多项式模型的
        - 若 return_mu_cov=True: 返回 mu, cov
        - 否则返回 log-likelihood (local contribution)
        """
        A = torch.as_tensor(A, dtype=torch.float64)
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64).flatten()
        Xs = torch.as_tensor(Xs, dtype=torch.float64)
        Ys = torch.as_tensor(Ys, dtype=torch.float64)
        x = torch.as_tensor(x, dtype=torch.float64)

        # K_inv = K(X,X)^{-1}
        K = self.kernel(X, X, x)
        K_inv = torch.linalg.inv(K)

        kernel_Xs_Xs = self.kernel(Xs, Xs, x)
        kernel_X_Xs = self.kernel(X, Xs, x)

        if self.kernel_spec != "GRE":
            kernel_Xs_X = kernel_X_Xs.T
        else:
            kernel_Xs_X = self.kernel(Xs, X, x)

        # Basis VA
        VA = x2fx(X, A)        # (n,m)
        vAT = x2fx(Xs, A).T    # (m,1)

        VA_T_at_K_inv = VA.T @ K_inv          # (m,n)
        residual_X_Xs = vAT - VA_T_at_K_inv @ kernel_X_Xs  # (m,1)

        inv_VA_T_K_inv_VA = torch.linalg.inv(VA_T_at_K_inv @ VA)  # (m,m)

        cov_val = (
            kernel_Xs_Xs
            - kernel_Xs_X @ K_inv @ kernel_X_Xs
            + residual_X_Xs.T @ inv_VA_T_K_inv_VA @ residual_X_Xs
        )  # (1,1)

        beta_X_Y = inv_VA_T_K_inv_VA @ (VA_T_at_K_inv @ Y)  # (m,)

        mu_val = kernel_Xs_X @ K_inv @ Y + residual_X_Xs.T @ beta_X_Y  # (1,)

        if return_mu_cov:
            cov_scalar = cov_val[0, 0]
            if cov_scalar < 0:
                cov_scalar = torch.tensor(0.0, dtype=torch.float64)
                cov_val = cov_val.clone()
                cov_val[0, 0] = cov_scalar
            return mu_val.view(1, 1), cov_val

        # 否则返回 log-likelihood
        diff = Ys.view(-1, 1) - mu_val.view(1, 1)
        inv_cov = torch.linalg.inv(cov_val)
        term1 = -0.5 * torch.log(torch.det(2 * math.pi * cov_val))
        term2 = -0.5 * (diff.T @ inv_cov @ diff)
        return (term1 + term2).view(())

    # def cv_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    #     """
    #     总 LOOCV loss：sum_i local_loss_i
    #     """
    #     x = torch.as_tensor(x, dtype=torch.float64)
    #     A = torch.as_tensor(A, dtype=torch.float64)
    #     n = self.X_normalised.shape[0]
    #     total = 0.0
    #     for i in range(n):
    #         total = total + self.cv_local_loss(x, A, i, return_mu_cov=False)
    #     return torch.as_tensor(total, dtype=torch.float64)
    def cv_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
 
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        X = self.X_normalised
        Y = self.Y_normalised.view(-1, 1)  # (n, 1)
        n = X.shape[0]
        m = A.shape[0]  # Number of polynomial basis terms
        # ───────────────────────────────────────────────────────────────────
        # Step 1: Compute kernel (covariance) matrix K ∈ R^{n×n}
        # ───────────────────────────────────────────────────────────────────
        K = self.kernel(X, X, x)
        # ───────────────────────────────────────────────────────────────────
        # Step 2: Compute polynomial design matrix Φ ∈ R^{n×m}
        # ───────────────────────────────────────────────────────────────────
        # Φ[i,j] = ∏_k X[i,k]^{A[j,k]}
        V = x2fx(X, A)
        # ───────────────────────────────────────────────────────────────────
        # Step 3: Construct augmented matrix M = [[K, Φ], [Φᵀ, 0]]
        # ───────────────────────────────────────────────────────────────────
        # This is the Kriging system with polynomial drift
        # M ∈ R^{(n+m)×(n+m)}
        M_top = torch.cat([K, V], dim=1)  # (n, n+m)
        zeros_m = torch.zeros((m, m), dtype=torch.float64, device=X.device)
        M_bot = torch.cat([V.T, zeros_m], dim=1)  # (m, n+m)
        M = torch.cat([M_top, M_bot], dim=0)  # (n+m, n+m)

        # ───────────────────────────────────────────────────────────────────
        # Step 4: Compute M^{-1} with diagonal jitter for stability
        # ───────────────────────────────────────────────────────────────────
        M_inv = torch.linalg.inv(M + torch.eye(n + m, device=X.device) * JITTER)

        # ───────────────────────────────────────────────────────────────────
        # Step 5: Construct augmented response vector [y; 0] ∈ R^{n+m}
        # ───────────────────────────────────────────────────────────────────
        zeros_y = torch.zeros((m, 1), dtype=torch.float64, device=X.device)
        Y_aug = torch.cat([Y, zeros_y], dim=0)  # (n+m, 1)

        # ───────────────────────────────────────────────────────────────────
        # Step 6: Apply Dubrule's formula for LOOCV statistics
        # ───────────────────────────────────────────────────────────────────
        # Solve: M @ [α; β] = [y; 0]
        # Extract first n coefficients (corresponding to data points)
        alpha = (M_inv @ Y_aug)[:n]  # (n, 1)
        diag_inv = torch.diagonal(M_inv)[:n]  # (n,)

        # LOOCV residuals: r_i = α_i / [M^{-1}]_{ii}
        residuals = alpha.flatten() / diag_inv

        # LOOCV variances: σ²_i = 1 / [M^{-1}]_{ii}
        variances = 1.0 / diag_inv

        # ───────────────────────────────────────────────────────────────────
        # Step 7: Compute log-likelihood L = Σ_i log p(y_i | y_{-i})
        # ───────────────────────────────────────────────────────────────────
        # For Gaussian predictive density:
        # log p(y_i | y_{-i}) = -½ log(2πσ²_i) - ½(r²_i/σ²_i)

        # Clamp variances to prevent numerical underflow
        variances = torch.clamp(variances, min=1e-14)

        # Compute log-likelihood components
        term1 = -0.5 * torch.log(2 * math.pi * variances)  # Normalization term
        term2 = -0.5 * (residuals ** 2) / variances  # Squared error term

        # Sum over all n data points
        total_ll = (term1 + term2).sum()

        return total_ll

    # ═══════════════════════════════════════════════════════════════════════
    # IV-B. MAXIMUM LIKELIHOOD ESTIMATION (MLE)
    # Reference: Arxiv 2001.10965 - "Maximum Likelihood Estimation and
    #            Overconfidence in Gaussian Process Regression"
    # ═══════════════════════════════════════════════════════════════════════

    def mle_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the Restricted Maximum Likelihood (REML) for GP with polynomial mean.

        For GP: y ~ N(V @ beta, K)
        The REML marginal likelihood (integrating out beta with improper prior):

        log p(y|θ) = -0.5*(n-m)*log(2π) - 0.5*log|K| - 0.5*log|V^T K^{-1} V|
                     - 0.5 * y^T P y

        where P = K^{-1} - K^{-1} V (V^T K^{-1} V)^{-1} V^T K^{-1}

        Parameters
        ----------
        x : torch.Tensor
            Kernel hyperparameters [σ², ℓ, ...]
        A : torch.Tensor
            Polynomial basis multi-index set

        Returns
        -------
        log_ml : torch.Tensor
            Log marginal likelihood (higher is better)
        """
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        X = self.X_normalised
        Y = self.Y_normalised.view(-1, 1)  # (n, 1)
        n = X.shape[0]
        m = A.shape[0]

        # Step 1: Compute kernel matrix K with jitter
        K = self.kernel(X, X, x)
        K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * JITTER

        # Step 2: Cholesky decomposition of K
        try:
            L_K = torch.linalg.cholesky(K_reg)
        except RuntimeError:
            K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * 1e-6
            L_K = torch.linalg.cholesky(K_reg)

        # Step 3: Compute log|K|
        log_det_K = 2.0 * torch.log(torch.diag(L_K)).sum()

        # Step 4: Compute K^{-1} @ Y and K^{-1} @ V
        K_inv_Y = torch.cholesky_solve(Y, L_K)  # (n, 1)
        V = x2fx(X, A)  # (n, m)
        K_inv_V = torch.cholesky_solve(V, L_K)  # (n, m)

        # Step 5: Compute V^T K^{-1} V and its Cholesky
        VtKinvV = V.T @ K_inv_V  # (m, m)

        # Use larger jitter for small matrices (more prone to numerical issues)
        jitter_vkv = max(JITTER, 1e-8 * torch.trace(VtKinvV).abs().item() / m) if m > 0 else JITTER
        VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * jitter_vkv

        try:
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)
        except RuntimeError:
            # Aggressive fallback with relative jitter
            VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * 1e-4
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)

        # Step 6: Compute log|V^T K^{-1} V|
        log_det_VKV = 2.0 * torch.log(torch.diag(L_VKV)).sum()

        # Step 7: Compute y^T P y where P = K^{-1} - K^{-1} V (V^T K^{-1} V)^{-1} V^T K^{-1}
        # y^T P y = y^T K^{-1} y - y^T K^{-1} V (V^T K^{-1} V)^{-1} V^T K^{-1} y
        VtKinvY = V.T @ K_inv_Y  # (m, 1)
        VKV_inv_VKY = torch.cholesky_solve(VtKinvY, L_VKV)  # (m, 1)

        ytKinvy = (Y.T @ K_inv_Y).squeeze()  # scalar
        correction = (VtKinvY.T @ VKV_inv_VKY).squeeze()  # scalar
        ytPy = ytKinvy - correction

        # Step 8: REML log marginal likelihood
        log_ml = -0.5 * (n - m) * math.log(2 * math.pi) - 0.5 * log_det_K \
                 - 0.5 * log_det_VKV - 0.5 * ytPy

        return log_ml

    def compute_sigma_mle(self, x_no_amp: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the closed-form REML estimate for σ² (amplitude).

        For GP with polynomial mean, the REML estimate is:
            σ²_MLE = (1/(n-m)) * y^T P y

        where P = K_φ^{-1} - K_φ^{-1} V (V^T K_φ^{-1} V)^{-1} V^T K_φ^{-1}
        and K_φ is the kernel matrix with unit amplitude.

        Parameters
        ----------
        x_no_amp : torch.Tensor
            Kernel hyperparameters WITHOUT amplitude (e.g., just lengthscale)
        A : torch.Tensor
            Polynomial basis multi-index set

        Returns
        -------
        sigma_sq : torch.Tensor
            REML estimate of σ²
        """
        X = self.X_normalised
        Y = self.Y_normalised.view(-1, 1)
        n = X.shape[0]
        m = A.shape[0]

        # Build kernel with amplitude = 1 (unit kernel)
        # softplus^{-1}(1) = log(exp(1) - 1) ≈ 0.5413
        amp_raw_for_unit = math.log(math.exp(1.0) - 1.0)
        x_unit = torch.cat([torch.tensor([amp_raw_for_unit], dtype=torch.float64), x_no_amp])

        K = self.kernel(X, X, x_unit)  # K with amp=1
        K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * JITTER

        # Cholesky of K
        try:
            L_K = torch.linalg.cholesky(K_reg)
        except RuntimeError:
            K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * 1e-6
            L_K = torch.linalg.cholesky(K_reg)

        # Compute K^{-1} Y and K^{-1} V
        K_inv_Y = torch.cholesky_solve(Y, L_K)  # (n, 1)
        V = x2fx(X, A)  # (n, m)
        K_inv_V = torch.cholesky_solve(V, L_K)  # (n, m)

        # Compute V^T K^{-1} V
        VtKinvV = V.T @ K_inv_V  # (m, m)

        # Adaptive jitter for numerical stability
        jitter_vkv = max(JITTER, 1e-8 * torch.trace(VtKinvV).abs().item() / m) if m > 0 else JITTER
        VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * jitter_vkv

        try:
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)
        except RuntimeError:
            VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * 1e-4
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)

        # Compute y^T P y = y^T K^{-1} y - y^T K^{-1} V (V^T K^{-1} V)^{-1} V^T K^{-1} y
        VtKinvY = V.T @ K_inv_Y  # (m, 1)
        VKV_inv_VKY = torch.cholesky_solve(VtKinvY, L_VKV)  # (m, 1)

        ytKinvy = (Y.T @ K_inv_Y).squeeze()
        correction = (VtKinvY.T @ VKV_inv_VKY).squeeze()
        ytPy = ytKinvy - correction

        # σ²_MLE = (1/(n-m)) * y^T P y
        sigma_sq = ytPy / (n - m)

        return torch.clamp(sigma_sq, min=EPSILON)

    def perform_extrapolation_optimization_mle(
        self,
        A: torch.Tensor,
        x0: torch.Tensor | None = None,
        num_restarts: int = DEFAULT_NUM_RESTARTS,
        restart_scale: float = 0.5,
        seed: int = 0,
        use_closed_form_sigma: bool = True,
    ) -> dict:
        """
        MLE-based hyperparameter optimization (alternative to CV-based).

        Parameters
        ----------
        A : torch.Tensor
            Polynomial basis multi-index set
        x0 : torch.Tensor, optional
            Initial hyperparameters
        num_restarts : int
            Number of random restarts for multi-start optimization
        restart_scale : float
            Scale of Gaussian perturbations for restarts
        seed : int
            Random seed for reproducibility
        use_closed_form_sigma : bool
            If True, use closed-form solution for σ² and only optimize other params

        Returns
        -------
        result : dict
            - 'x': optimal hyperparameters
            - 'mle': log marginal likelihood at optimum (higher is better)

        Notes
        -----
        Unlike CV which returns negative log-likelihood (lower is better),
        MLE returns log marginal likelihood (higher is better).
        For compatibility, we also store 'cv' = -mle.
        """
        A = torch.as_tensor(A, dtype=torch.float64)

        default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)

        if x0 is None:
            x_base = default_params
        else:
            x_base = torch.as_tensor(x0, dtype=torch.float64).flatten()
            if x_base.numel() != default_params.numel():
                raise ValueError(
                    f"x0 has wrong length: got {x_base.numel()}, expected {default_params.numel()}."
                )

        # Multi-start setup
        gen = torch.Generator()
        gen.manual_seed(int(seed))

        starting_points = [x_base.clone()]
        for _ in range(int(num_restarts)):
            noise = torch.randn(x_base.shape, dtype=x_base.dtype, generator=gen) * float(restart_scale)
            starting_points.append((x_base + noise).clone())

        best_mle = float("-inf")  # maximize log marginal likelihood
        best_x = x_base.clone()

        for start_x in starting_points:
            try:
                if use_closed_form_sigma:
                    x_opt, mle_val = self._optimize_mle_with_closed_form_sigma(start_x, A)
                else:
                    x_opt, mle_val = self._optimize_mle_torch(start_x, A)

                if float(mle_val) > best_mle:
                    best_mle = float(mle_val)
                    best_x = x_opt
            except Exception as e:
                print(f"[WARNING] MLE optimization failed for starting point: {e}")
                continue

        if best_mle == float("-inf"):
            raise RuntimeError("All MLE optimization starting points failed.")

        return {
            "x": best_x,
            "mle": torch.tensor(best_mle, dtype=torch.float64),
            "cv": torch.tensor(-best_mle, dtype=torch.float64),  # For compatibility
        }

    def _optimize_mle_torch(self, init_x: torch.Tensor, A: torch.Tensor):
        """
        Internal: L-BFGS optimization for MLE (maximize log marginal likelihood).
        """
        x_param = init_x.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [x_param],
            lr=1.0,
            max_iter=100,
            max_eval=120,
            tolerance_grad=1e-5,
            tolerance_change=1e-5,
            history_size=10,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            log_ml = self.mle_loss(x_param, A)
            loss = -log_ml  # Minimize negative log marginal likelihood
            if loss.requires_grad:
                loss.backward()
            return loss

        optimizer.step(closure)

        final_mle = self.mle_loss(x_param, A)
        return x_param.detach(), final_mle.item()

    def _optimize_mle_with_closed_form_sigma(self, init_x: torch.Tensor, A: torch.Tensor):
        """
        Internal: Optimize only lengthscale params, use closed-form for σ².

        This is more efficient and often more stable than optimizing all params.
        """
        # Separate amplitude from other parameters
        # init_x = [amp_raw, lengthscale_raw, ...]
        x_no_amp = init_x[1:].clone().detach().requires_grad_(True)

        if x_no_amp.numel() == 0:
            # Only amplitude parameter (e.g., white noise kernel)
            # Just compute closed-form sigma
            sigma_sq = self.compute_sigma_mle(x_no_amp, A)
            amp_raw = math.log(math.exp(float(sigma_sq)) - 1.0 + EPSILON)
            x_opt = torch.tensor([amp_raw], dtype=torch.float64)
            mle_val = self.mle_loss(x_opt, A)
            return x_opt, mle_val.item()

        optimizer = torch.optim.LBFGS(
            [x_no_amp],
            lr=1.0,
            max_iter=100,
            max_eval=120,
            tolerance_grad=1e-5,
            tolerance_change=1e-5,
            history_size=10,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()

            # Compute closed-form sigma for current lengthscale
            with torch.no_grad():
                sigma_sq = self.compute_sigma_mle(x_no_amp.detach(), A)

            # Convert sigma_sq to raw amplitude parameter
            # softplus(amp_raw) = sigma_sq => amp_raw = softplus^{-1}(sigma_sq)
            amp_raw = torch.log(torch.exp(sigma_sq) - 1.0 + EPSILON)

            # Full parameter vector
            x_full = torch.cat([amp_raw.unsqueeze(0), x_no_amp])

            log_ml = self.mle_loss(x_full, A)
            loss = -log_ml

            if loss.requires_grad:
                loss.backward()

            return loss

        optimizer.step(closure)

        # Final computation with optimized lengthscale
        with torch.no_grad():
            sigma_sq = self.compute_sigma_mle(x_no_amp.detach(), A)
            amp_raw = torch.log(torch.exp(sigma_sq) - 1.0 + EPSILON)
            x_opt = torch.cat([amp_raw.unsqueeze(0), x_no_amp.detach()])
            final_mle = self.mle_loss(x_opt, A)

        return x_opt, final_mle.item()

    # ═══════════════════════════════════════════════════════════════════════
    # V. PREDICTION & EXTRAPOLATION
    # ═══════════════════════════════════════════════════════════════════════

    def perform_extrapolation(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        return_mu_and_var: bool = False,
    ) -> dict:
        """
        Perform SPRE extrapolation given hyperparameters and polynomial basis.

        Extrapolates the GP + polynomial model to x = 0 (the Richardson limit)
        and optionally computes LOOCV predictions at all training points.

        Parameters
        ----------
        x : torch.Tensor, shape (p,)
            Kernel hyperparameters [σ², ℓ, ...]
        A : torch.Tensor, shape (m, d)
            Polynomial basis multi-index set
        return_mu_and_var : bool, default=False
            If True, also compute predictions (mu, var) at x=0 and LOOCV
            predictions (mu_cv, var_cv) at all training points

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'cv': LOOCV log-likelihood criterion
            - 'mu': (if return_mu_and_var) extrapolation mean at x=0
            - 'var': (if return_mu_and_var) extrapolation variance at x=0
            - 'mu_cv': (if return_mu_and_var) LOOCV means at training points
            - 'var_cv': (if return_mu_and_var) LOOCV variances at training points

        Notes
        -----
        All returned mu/var values are in original (unnormalized) units.
        """
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        cv = self.cv_loss(x, A)
        out = {"cv": cv}

        # Check if MAD-based normalization was used
        use_mad = getattr(self, '_use_mad', False)
        Y_mean = getattr(self, 'Y_mean', torch.tensor(0.0, dtype=torch.float64))

        if return_mu_and_var:
            n_train = self.X_normalised.shape[0]
            mu_cv = torch.zeros(n_train, dtype=torch.float64)
            var_cv = torch.zeros(n_train, dtype=torch.float64)

            for i in range(n_train):
                mu_val, cov_val = self.cv_local_loss(x, A, i, return_mu_cov=True)
                # De-normalize: if MAD, mu = mu_norm * nY + Y_mean; else mu = mu_norm * nY
                if use_mad:
                    mu_cv[i] = self.nY * mu_val[0, 0] + Y_mean
                else:
                    mu_cv[i] = self.nY * mu_val[0, 0]
                var_cv[i] = (self.nY ** 2) * cov_val[0, 0]

            # f(0) 处的预测（Xs=0）
            Xs0 = torch.zeros((1, self.dimension), dtype=torch.float64)
            Ys0 = torch.zeros((1,), dtype=torch.float64)  # 不重要
            mu_val0, cov_val0 = self.cv_loss_calculation(
                A,
                self.X_normalised,
                self.Y_normalised,
                Xs0,
                Ys0,
                x,
                return_mu_cov=True,
            )

            # De-normalize predictions
            if use_mad:
                mu = self.nY * mu_val0 + Y_mean
            else:
                mu = self.nY * mu_val0
            var = (self.nY ** 2) * cov_val0

            out.update(
                {
                    "mu": mu,
                    "var": var,
                    "mu_cv": mu_cv,
                    "var_cv": var_cv,
                }
            )

        return out

    def perform_extrapolation_fixed_hyperparams(
        self,
        amplitude: float,
        lengthscale: float | list,
        A: torch.Tensor,
        return_mu_and_var: bool = True,
    ) -> dict:
        """
        Perform SPRE extrapolation with FIXED hyperparameters (no optimization).

        This method is used when global hyperparameters have been learned across
        multiple time points and should be reused without further optimization.

        Parameters
        ----------
        amplitude : float
            Fixed kernel amplitude σ²
        lengthscale : float or list
            Fixed kernel lengthscale ℓ (or list for ARD kernels)
        A : torch.Tensor, shape (m, d)
            Polynomial basis multi-index set
        return_mu_and_var : bool, default=True
            If True, compute predictions (mu, var) at x=0 and LOOCV predictions

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'cv': LOOCV log-likelihood criterion
            - 'mu': extrapolation mean at x=0
            - 'var': extrapolation variance at x=0
            - 'mu_cv': LOOCV means at training points
            - 'var_cv': LOOCV variances at training points
            - 'x': raw hyperparameters used
            - 'amplitude': the fixed amplitude
            - 'lengthscale': the fixed lengthscale

        Notes
        -----
        No hyperparameter optimization is performed. The provided values are
        converted to raw space and used directly for prediction.
        """
        # Convert to raw hyperparameters
        x_raw = self.hyperparams_to_raw(amplitude, lengthscale)

        # Perform extrapolation with fixed hyperparameters
        out = self.perform_extrapolation(x_raw, A, return_mu_and_var=return_mu_and_var)

        # Add hyperparameter info to output
        out['x'] = x_raw
        out['amplitude'] = amplitude
        out['lengthscale'] = lengthscale

        return out

    def objective(self, x_np: np.ndarray, A: np.ndarray) -> float:
        """
        SciPy 用的 objective：返回负的 cv（因为要最小化）。
        """
        x_t = torch.as_tensor(x_np, dtype=torch.float64)
        A_t = torch.as_tensor(A, dtype=torch.float64)
        val = self.cv_loss(x_t, A_t)
        return float((-val).detach().cpu().numpy())
    
    
    def perform_extrapolation_optimization(
        self,
        A: torch.Tensor,
        do_jit: bool = True,
        x0: torch.Tensor | None = None,
        num_restarts: int = DEFAULT_NUM_RESTARTS,
        restart_scale: float = 0.5,
        seed: int = 0,
    ) -> dict:
        """
        Patched version:
          - Supports user-provided initialisation x0 (raw/unconstrained space).
          - Uses additive Gaussian perturbations in raw space for multi-start (more appropriate with softplus).
          - Keeps your original LBFGS optimisation and "best loss" selection logic.

        Notes:
          - cv_loss returns total log-likelihood (higher is better).
          - _optimize_torch returns (x_opt, loss_val) where loss_val = -LL (lower is better).
          - We therefore select the smallest loss_val.
        """
        A = torch.as_tensor(A, dtype=torch.float64)

        # Default initial point in raw space
        default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)

        # Base initial point
        if x0 is None:
            x_base = default_params
        else:
            x_base = torch.as_tensor(x0, dtype=torch.float64).flatten()
            if x_base.numel() != default_params.numel():
                raise ValueError(
                    f"x0 has wrong length: got {x_base.numel()}, expected {default_params.numel()}."
                )

        # Multi-start: base + additive Gaussian noise in raw space
        gen = torch.Generator()
        gen.manual_seed(int(seed))

        starting_points = [x_base.clone()]
        for _ in range(int(num_restarts)):
            noise = torch.randn(x_base.shape, dtype=x_base.dtype, generator=gen) * float(restart_scale)
            starting_points.append((x_base + noise).clone())

        best_loss = float("inf")   # minimise -LL
        best_x = x_base.clone()

        # Optimise from each starting point
        for start_x in starting_points:
            try:
                x_opt, loss_val = self._optimize_torch(start_x, A)
                if float(loss_val) < best_loss:
                    best_loss = float(loss_val)
                    best_x = x_opt
            except Exception as e:
                # Skip unstable starts
                print(f"[WARNING] Optimization failed for starting point: {e}")
                continue

        # Fallback if all starting points failed
        if best_loss == float("inf"):
            raise RuntimeError("All optimization starting points failed. Check input data or parameters.")

        return {"x": best_x, "cv": torch.tensor(best_loss, dtype=torch.float64)}

    # def perform_extrapolation_optimization(self, A: torch.Tensor, do_jit: bool = True) -> dict:

    #     A = torch.as_tensor(A, dtype=torch.float64)

    #     # ───────────────────────────────────────────────────────────────────
    #     # Step 1: Generate multi-start initial points
    #     # ───────────────────────────────────────────────────────────────────
    #     default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)

    #     # Create list of starting points: default + random perturbations
    #     num_restarts = DEFAULT_NUM_RESTARTS
    #     starting_points = [default_params]
    #     for _ in range(num_restarts):
    #         # Perturb in log-space to maintain positivity
    #         noise = torch.randn_like(default_params) * 0.5
    #         perturbed = default_params * torch.exp(noise)
    #         starting_points.append(perturbed)

    #     best_cv = float('inf')
    #     best_x = default_params

    #     # ───────────────────────────────────────────────────────────────────
    #     # Step 2: Optimize from each starting point
    #     # ───────────────────────────────────────────────────────────────────
    #     for start_x in starting_points:
    #         try:
    #             x_opt, cv_val = self._optimize_torch(start_x, A)

    #             # Track best result across all restarts
    #             if cv_val < best_cv:
    #                 best_cv = cv_val
    #                 best_x = x_opt
    #         except Exception as e:
    #             # Skip starting points that cause numerical instability
    #             continue

    #     return {"x": best_x, "cv": torch.tensor(best_cv, dtype=torch.float64)}

    def _optimize_torch(self, init_x: torch.Tensor, A: torch.Tensor):
        """
        内部辅助函数：使用 L-BFGS 优化单个起点
        修复核心逻辑：最小化 Negative Log-Likelihood
        """
        # 复制参数并开启梯度
        x_param = init_x.clone().detach().requires_grad_(True)
        
        # L-BFGS 设置 (保持不变)
        optimizer = torch.optim.LBFGS(
            [x_param], 
            lr=1.0, 
            max_iter=100, 
            max_eval=120, 
            tolerance_grad=1e-5, 
            tolerance_change=1e-5,
            history_size=10,
            line_search_fn="strong_wolfe"
        )
        def closure():
            optimizer.zero_grad()
    
            raw_ll = self.cv_loss(x_param, A)
            loss = -raw_ll
            if loss.requires_grad:
                loss.backward()
    
            return loss

        # def closure():
        #     optimizer.zero_grad()
            
        #     # 1.  Log-Likelihood (LL)
        #     # LL 越大代表模型越好
        #     raw_ll = self.cv_loss(x_param, A)
        #     amp = EPSILON + torch.nn.functional.softplus(x_param[0])
        #     prior_mean = -2.0  # log(0.13) 
        #     prior_std = 2.0
        #     log_amp = torch.log(amp)
        #     prior_penalty = 0.5 * ((log_amp - prior_mean) / prior_std)**2
       
        #     loss = -raw_ll + prior_penalty
            
       
        #     if loss.requires_grad:
        #         loss.backward()
            
        #     return loss

       
        optimizer.step(closure)
        
        # 4. 计算最终结果
        # 我们返回 loss (即 -LL)，这样你的 stepwise_selection 逻辑
        # (if cv_new < cv) 依然成立：我们希望 Loss 越小越好。
        final_ll = self.cv_loss(x_param, A)

        return x_param.detach(), (-final_ll).item()
    # ═══════════════════════════════════════════════════════════════════════
    # VII. GREEDY BASIS SELECTION ALGORITHM
    # ═══════════════════════════════════════════════════════════════════════

    def stepwise_selection(self) -> dict:

        n_train = self.X_normalised.shape[0]

        # ───────────────────────────────────────────────────────────────────
        # Initialize with intercept-only basis: α₀ = {0}
        # ───────────────────────────────────────────────────────────────────
        A = torch.zeros((1, self.dimension), dtype=torch.int64)

        # Delegate to GRE-specific stepwise if in GRE mode
        if self.kernel_base is not None:
            return self._GRE_stepwise_selection(A)

        # ───────────────────────────────────────────────────────────────────
        # Standard SPRE stepwise selection
        # ───────────────────────────────────────────────────────────────────
        do_jit = False  # Legacy parameter (no longer used)

        # Optimize hyperparameters for initial basis
        order = 0
        fit = self.perform_extrapolation_optimization(A, do_jit)
        cv = fit["cv"]

        carry_on = True

        while carry_on:
            m = A.shape[0]
            order += 1

            # Generate candidate terms of current order
            A_extra = stepwise(A, order)
            n_extra = A_extra.shape[0]
            to_include = torch.zeros(n_extra, dtype=torch.bool)

            print(f"Fitting interactions of order {order}:")

            # Test each candidate term
            for i in tqdm(range(n_extra), desc="Stepwise progress"):
                A_new = torch.cat([A, A_extra[i : i + 1, :]], dim=0)

                # Check if augmented basis is unisolvent (full rank)
                if self.check_unisolvent(A_new) > 0:
                    fit_new = self.perform_extrapolation_optimization(A_new, do_jit)
                    cv_new = fit_new["cv"]

                    # Accept if LOOCV improves (note: cv is negative log-likelihood)
                    if cv_new < cv:
                        to_include[i] = True

            # Add accepted terms if within size limit (n-1)
            if to_include.any() and (m + int(to_include.sum().item()) < (n_train - 1)):
                A_updated = torch.cat([A, A_extra[to_include, :]], dim=0)
                fit_updated = self.perform_extrapolation_optimization(A_updated, do_jit)
                cv_updated = fit_updated["cv"]

                # Check if joint acceptance improves LOOCV
                if cv_updated >= cv:
                    carry_on = False  # No improvement, terminate
                else:
                    A = A_updated
                    fit = fit_updated
                    cv = cv_updated
            else:
                carry_on = False  # No terms accepted or size limit reached

        # Final extrapolation with optimal basis and hyperparameters
        x_opt = fit["x"]
        out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)
        return out

    def _GRE_stepwise_selection(self, A: torch.Tensor) -> dict:
        """
        Stepwise selection for GRE (Gauss-Richardson Extrapolation).

        A: basis for mean (fixed in GRE)
        We add terms to B (rate function) stored in gre_base.
        """
        # 初始 B：仅截距
        B = torch.zeros((1, self.dimension), dtype=torch.int64)

        order = 0
        # 设定当前 GRE kernel
        self.set_kernel_spec(self.kernel_base, B)
        fit = self.perform_extrapolation_optimization(A, do_jit=False)
        cv = fit["cv"]

        carry_on = True
        while carry_on:
            order += 1
            B_extra = stepwise(B, order)
            n_extra = B_extra.shape[0]
            if n_extra == 0:
                break

            print(f"Fitting GRE rate interactions of order {order}...")
            to_include = torch.zeros(n_extra, dtype=torch.bool)

            for i in tqdm(range(n_extra), desc="GRE stepwise progress"):
                B_new = torch.cat([B, B_extra[i : i + 1, :]], dim=0)
                self.set_kernel_spec(self.kernel_base, B_new)
                fit_new = self.perform_extrapolation_optimization(A, do_jit=False)
                cv_new = fit_new["cv"]
                if cv_new < cv:
                    to_include[i] = True

            if to_include.any():
                B_updated = torch.cat([B, B_extra[to_include, :]], dim=0)
                self.set_kernel_spec(self.kernel_base, B_updated)
                fit_updated = self.perform_extrapolation_optimization(A, do_jit=False)
                cv_updated = fit_updated["cv"]
                if cv_updated >= cv:
                    carry_on = False
                else:
                    B = B_updated
                    fit = fit_updated
                    cv = cv_updated
            else:
                carry_on = False

        # 最终用 B、best x 做一次完整外推
        x_opt = fit["x"]
        self.set_kernel_spec(self.kernel_base, B)
        out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)
        out["B"] = B
        return out
