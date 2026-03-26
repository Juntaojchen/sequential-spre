
# import math
# import numpy as np
# import torch
# from scipy.optimize import minimize
# from tqdm import tqdm

# from .helper_functions import (
#     x2fx,
#     softplus,
#     cellsum,
#     white,
#     remove_row,
#     stepwise,
# )

# # ═══════════════════════════════════════════════════════════════════════
# # Numerical Stability Constants
# # ═══════════════════════════════════════════════════════════════════════

# EPSILON = 1e-12          # Safeguard against division by zero
# JITTER = 1e-12       # Diagonal perturbation for matrix inversion
# DEFAULT_NUM_RESTARTS = 10  # Multi-start optimization attempts


# class SPRE:
#     """
#     Sparse Probabilistic Richardson Extrapolation (SPRE)
#     ====================================================

#     A sparse Gaussian Process framework for Richardson extrapolation that learns
#     multivariate polynomial bases through greedy forward selection.

#     Mathematical Framework
#     ----------------------
#     Given observations {(x_i, y_i)}^n_{i=1} where x_i ∈ R^d and y_i ∈ R, we model:

#         y(x) = Σ_j β_j φ_j(x) + GP(0, κ(·,·))

#     where:
#         - φ_j(x) = ∏_k x_k^{α_jk} are polynomial basis functions
#         - α = {α_j}^m_{j=1} ⊂ N^d is the multi-index set (learned via stepwise selection)
#         - κ(·,·) is a covariance kernel (Gaussian, Matérn, etc.)
#         - β ∈ R^m are coefficients (marginalized analytically)

#     The goal is to extrapolate to x = 0 to estimate the limit f(0).

#     Key Features
#     ------------
#     1. **Vectorized LOOCV**: Efficient leave-one-out cross-validation using the
#        Dubrule (1983) formula, avoiding explicit n-fold model retraining.

#     2. **Automatic Basis Selection**: Greedy forward stepwise algorithm that
#        adaptively builds sparse polynomial bases α.

#     3. **Multi-start L-BFGS Optimization**: Robust hyperparameter optimization
#        with random restarts to avoid local minima.

#     References
#     ----------
#     .. [Dubrule1983] Dubrule, O. (1983). Cross validation of kriging in a unique
#        neighborhood. Journal of the International Association for Mathematical
#        Geology, 15(6), 687-699.

#     .. [Rasmussen2006] Rasmussen, C. E., & Williams, C. K. I. (2006).
#        Gaussian Processes for Machine Learning. MIT Press.

#     Implementation Notes
#     --------------------
#     - All computations use float64 precision for numerical stability
#     - Covariance kernels use softplus transformations to ensure positivity
#     - Normalization uses max-min scaling with epsilon-jitter for robustness
#     """

#     def __init__(self, kernel_spec: str, dimension: int, gre_base: torch.Tensor | None = None):
#         """
#         Initialize SPRE model.

#         Parameters
#         ----------
#         kernel_spec : str
#             Name of the covariance kernel: "Gaussian", "GaussianARD",
#             "Matern1/2", "Matern3/2", "white".
#             If gre_base is not None, SPRE operates in GRE mode.
#         dimension : int
#             Spatial dimension d of the input space.
#         gre_base : torch.Tensor or None, shape (m, d)
#             Polynomial basis for GRE (Gauss-Richardson Extrapolation) compatibility.
#             If provided, enables GRE mode with rate function modulation.

#         Notes
#         -----
#         Data normalization parameters (X_norm, Y_norm, sigma_X, sigma_Y) are
#         set later via set_normalised_data().
#         """
#         self.dimension = dimension

#         # Initialize kernel specification (supports GRE mode via basis injection)
#         self.set_kernel_spec(kernel_spec, gre_base)

#         # Data containers (populated by set_normalised_data)
#         self.X_normalised: torch.Tensor | None = None  # Normalized design points
#         self.Y_normalised: torch.Tensor | None = None  # Normalized responses
#         self.nX: torch.Tensor | None = None  # Normalization scale factors per dimension
#         self.nY: torch.Tensor | None = None  # Normalization scale factor for response

#     # ═══════════════════════════════════════════════════════════════════════
#     # II. COVARIANCE KERNEL FAMILY κ(·, ·; θ)
#     # ═══════════════════════════════════════════════════════════════════════
#     #
#     # This section implements various kernel functions for Gaussian Process
#     # regression. Each kernel defines a different prior assumption about the
#     # smoothness and correlation structure of the unknown function f(·).
#     #
#     # Standard kernels: Gaussian (RBF), Matérn-ν, ARD
#     # Special case: GRE (Gauss-Richardson) with polynomial modulation
#     # ───────────────────────────────────────────────────────────────────────

#     def cdist_torch(self, XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
#         """
#         Compute pairwise Euclidean distances between rows of XA and XB.

#         Implements ||x_i - x_j'|| for all pairs (i, j), equivalent to
#         scipy.spatial.distance.cdist(XA, XB, 'euclidean').

#         Parameters
#         ----------
#         XA : torch.Tensor, shape (m, d)
#             First set of points
#         XB : torch.Tensor, shape (n, d)
#             Second set of points

#         Returns
#         -------
#         distances : torch.Tensor, shape (m, n)
#             Pairwise Euclidean distances
#         """
#         XA = XA.to(torch.float64)
#         XB = XB.to(torch.float64)

#         # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
#         XA_sq = (XA ** 2).sum(dim=1, keepdim=True)  # (m,1)
#         XB_sq = (XB ** 2).sum(dim=1)                # (n,)
#         cross = XA @ XB.T                           # (m,n)

#         nums = XA_sq - 2 * cross + XB_sq           # (m,n)
#         nums = torch.clamp(nums, min=0.0)
#         return torch.sqrt(nums)

#     def set_kernel_spec(self, kernel_spec: str, gre_base: torch.Tensor | None = None):
#         """
#         设置 kernel 规格（包括 GRE 模式）。
#         """
#         if gre_base is None:
#             self.kernel_spec = kernel_spec
#             self.kernel_base = None
#         else:
#             # 兼容 GRE：对外暴露 "GRE"，内部保留 base kernel 的名字和 B 矩阵
#             self.kernel_spec = "GRE"
#             self.kernel_base = kernel_spec

#         self.gre_base = gre_base  # B 矩阵（或 None）
#         self.set_kernel_default_parameters()

#     def set_kernel_default_parameters(self):
#         """
#         设置 kernel 的初始超参数
#         """
#         match self.kernel_spec:
#             case "Gaussian":
#                 # [amp, lengthscale]
#                 self.default_kernel_parameters = [1.0, 0.1]
#             case "GaussianARD":
#                 # [amp, l1, l2, ..., ld]
#                 params = [1.0]
#                 params.extend([0.1] * self.dimension)
#                 self.default_kernel_parameters = params
#             case "white":
#                 self.default_kernel_parameters = [1.0]
#             case "Matern1/2":
#                 # [amp, lengthscale]
#                 self.default_kernel_parameters = [1.0, 1.0]
#             case "Matern3/2":
#                 self.default_kernel_parameters = [1.0, 1.0]
#             case "GRE":
#                 # 先为 base kernel 设置 default，然后在前面加上一个 GRE amp
#                 base_spec = self.kernel_base
#                 self.kernel_spec = base_spec
#                 self.set_kernel_default_parameters()
#                 base_defaults = self.default_kernel_parameters
#                 self.kernel_spec = "GRE"
#                 self.kernel_base = base_spec

#                 self.default_kernel_parameters = [1.0]
#                 self.default_kernel_parameters.extend(base_defaults)
#             case _:
#                 raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

#     def kernel(self, X1: torch.Tensor, X2: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
#         """
#         Evaluate the kernel K(X1, X2 | x).

#         X1: (n1, d)
#         X2: (n2, d)
#         x:  (p,) hyperparameters
#         """
#         X1 = torch.as_tensor(X1, dtype=torch.float64)
#         X2 = torch.as_tensor(X2, dtype=torch.float64)

#         if x is None:
#             x = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
#         else:
#             x = torch.as_tensor(x, dtype=torch.float64)

#         spec = self.kernel_spec

#         if spec == "Gaussian":
#             # Radial Basis Function (RBF) / Squared Exponential kernel
#             # Reference: Rasmussen & Williams (2006), Equation 4.9
#             #
#             # Definition: κ(r) = σ² exp(-r² / 2ℓ²)
#             #
#             # where:
#             #   σ² (amplitude): signal variance
#             #   ℓ (lengthscale): characteristic length scale
#             #   r = ||x - x'||: Euclidean distance  maybe we can try different combine method ,like....  
        
#             amp = EPSILON + softplus(x[0])
#             ell = EPSILON + softplus(x[1])
#             dist = self.cdist_torch(X1, X2)
#             return amp * torch.exp(-(dist ** 2) / (2.0 *ell ** 2))

#         elif spec == "GaussianARD":
#             # Automatic Relevance Determination (ARD) Gaussian kernel
#             # Reference: Rasmussen & Williams (2006), Section 5.1
#             #
#             # Definition: κ(x, x') = σ² exp(-Σ_i (x_i - x'_i)² / 2ℓ_i²)
#             #
#             # Uses dimension-specific lengthscales ℓ_1, ..., ℓ_d to automatically
#             # determine the relevance of each input dimension
#             amp = EPSILON + softplus(x[0])
#             length_terms = []
#             for i in range(self.dimension):
#                 d_i = self.cdist_torch(
#                     X1[:, [i]], X2[:, [i]]
#                 ) ** 2 / (EPSILON + softplus(x[i + 1])) ** 2
#                 length_terms.append(d_i)
#             # Sum squared distances across all dimensions (element-wise)
#             sq_dist = cellsum(length_terms)
#             return amp * torch.exp(-sq_dist)

#         elif spec == "white":
#             # White noise kernel (diagonal covariance)
#             #
#             # Definition: κ(x, x') = σ² δ(x, x')
#             #
#             # where δ is the Kronecker delta (1 if x = x', 0 otherwise) we proposed
#             amp = EPSILON + softplus(x[0])
#             return amp * white(X1, X2)

#         elif spec == "Matern1/2":
#             # Matérn covariance with ν = 1/2 (equivalent to exponential kernel)
#             # Reference: Rasmussen & Williams (2006), Equation 4.14
#             #
#             # Definition: κ(r) = σ² exp(-r / ℓ)
#             #
#             # This kernel produces non-differentiable sample paths
#             amp = EPSILON + softplus(x[0])
#             ell = EPSILON + softplus(x[1])
#             dist = self.cdist_torch(X1, X2)
#             return amp * torch.exp(-dist / ell)

#         elif spec == "Matern3/2":
#             # Matérn covariance with ν = 3/2
#             # Reference: Rasmussen & Williams (2006), Equation 4.14
#             #
#             # Definition: κ(r) = σ² (1 + √3·r/ℓ) exp(-√3·r/ℓ)
#             #
#             # This kernel produces once-differentiable sample paths
#             amp = EPSILON + softplus(x[0])
#             ell = EPSILON + softplus(x[1])
#             dist = self.cdist_torch(X1, X2)
#             r_l = dist / ell
#             sqrt3_r_l = math.sqrt(3.0) * r_l
#             return amp * (1.0 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)

#         elif spec == "GRE":
#             # Gauss-Richardson Extrapolation (GRE) kernel with polynomial modulation
#             #
#             # Definition: κ_GRE(x, x') = σ²_GRE · b(x) · κ_base(x, x') · b(x')
#             #
#             # where:
#             #   b(x) = Σ_j φ_j(x): polynomial rate function
#             #   κ_base: underlying base kernel (Gaussian, Matérn, etc.)
#             #   σ²_GRE: GRE-specific amplitude parameter
#             #
#             # This allows the kernel to adaptively model rate functions in
#             # Richardson extrapolation
#             if self.gre_base is None:
#                 raise RuntimeError("GRE kernel requires gre_base (B matrix).")
#             amp_gre = EPSILON + softplus(x[0])

#             # Evaluate polynomial rate function: b(x) = Σ_j φ_j(x)
#             base_X1 = x2fx(X1, self.gre_base).sum(dim=1)  # (n1,)
#             base_X2 = x2fx(X2, self.gre_base).sum(dim=1)  # (n2,)

#             # Temporarily switch to base kernel for evaluation
#             base_spec = self.kernel_base
#             old_spec = self.kernel_spec
#             self.kernel_spec = base_spec
#             K_base = self.kernel(X1, X2, x=x[1:])  # Use base kernel hyperparameters
#             self.kernel_spec = old_spec

#             # Apply polynomial modulation: b(x_i) · K_base · b(x_j)
#             return amp_gre * base_X1.unsqueeze(1) * K_base * base_X2.unsqueeze(0)

#         else:
#             raise ValueError(f"Unknown kernel specification: {spec}")

#     # ═══════════════════════════════════════════════════════════════════════
#     # III. DATA NORMALIZATION
#     # ═══════════════════════════════════════════════════════════════════════

#     def set_normalised_data(self, X, Y):
#         """
#         Normalize data using max-min scaling with epsilon-jitter.

#         Normalization Formula
#         --------------------
#         For inputs X and outputs Y, compute scale factors:
#             σ_X = max(X) - min(X) + ε
#             σ_Y = max(Y) - min(Y) + ε

#         Then normalize:
#             X_norm = X / σ_X
#             Y_norm = Y / σ_Y

#         The epsilon term prevents division by zero for constant data.

#         Parameters
#         ----------
#         X : array-like, shape (n, d)
#             Input design points
#         Y : array-like, shape (n,)
#             Output responses

#         Notes
#         -----
#         Normalization improves numerical conditioning for GP inference and
#         makes lengthscale hyperparameters more interpretable.
#         """
#         X = torch.as_tensor(X, dtype=torch.float64)
#         Y = torch.as_tensor(Y, dtype=torch.float64).flatten()

#         # Compute normalization scale factors (max-min with ε-jitter)
#         self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
#         self.nY = (Y.max() - Y.min()) + EPSILON

#         # Apply normalization
#         self.X_normalised = X / self.nX
#         self.Y_normalised = Y / self.nY

#     # ═══════════════════════════════════════════════════════════════════════
#     # IV. GAUSSIAN PROCESS POSTERIOR INFERENCE & LOOCV
#     # ═══════════════════════════════════════════════════════════════════════

#     def cv_local_loss(
#         self,
#         x: torch.Tensor,
#         A: torch.Tensor,
#         row_num: int,
#         return_mu_cov: bool = False,
#     ):
#         """
#         Compute Leave-One-Out Cross-Validation (LOOCV) statistics for a single sample.

#         Parameters
#         ----------
#         x : torch.Tensor, shape (p,)
#             Kernel hyperparameters
#         A : torch.Tensor, shape (m, d)
#             Polynomial basis multi-index set
#         row_num : int
#             Index of the sample to leave out (0 ≤ row_num < n)
#         return_mu_cov : bool, default=False
#             If True, return (mu, cov) predictions; otherwise return log-likelihood

#         Returns
#         -------
#         If return_mu_cov=True:
#             mu : torch.Tensor, shape (1, 1)
#                 Posterior mean prediction
#             cov : torch.Tensor, shape (1, 1)
#                 Posterior covariance (variance)

#         If return_mu_cov=False:
#             log_likelihood : torch.Tensor, scalar
#                 Log-predictive density at the left-out point
#         """
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         X_full = self.X_normalised
#         Y_full = self.Y_normalised

#         X = remove_row(X_full, row_num)
#         Y = remove_row(Y_full.unsqueeze(1), row_num).flatten()
#         Xs = X_full[row_num : row_num + 1, :]
#         Ys = Y_full[row_num : row_num + 1]

#         return self.cv_loss_calculation(A, X, Y, Xs, Ys, x, return_mu_cov=return_mu_cov)

#     def check_unisolvent(self, A: torch.Tensor) -> int:
#         """
#         检查 A 是否生成 unisolvent 集（rank == m）。
#         """
#         A = torch.as_tensor(A, dtype=torch.float64)
#         m = A.shape[0]
#         VA = x2fx(self.X_normalised, A)  # (n,m)
#         rank = torch.linalg.matrix_rank(VA)
#         if rank == m:
#             return 1
#         else:
            
#             return -1

#     def cv_loss_calculation(
#         self,
#         A: torch.Tensor,
#         X: torch.Tensor,
#         Y: torch.Tensor,
#         Xs: torch.Tensor,
#         Ys: torch.Tensor,
#         x: torch.Tensor,
#         return_mu_cov: bool = False,
#     ):
#         """
#         计算 (Xs, Ys) 对应的 GP+多项式模型的
#         - 若 return_mu_cov=True: 返回 mu, cov
#         - 否则返回 log-likelihood (local contribution)
#         """
#         A = torch.as_tensor(A, dtype=torch.float64)
#         X = torch.as_tensor(X, dtype=torch.float64)
#         Y = torch.as_tensor(Y, dtype=torch.float64).flatten()
#         Xs = torch.as_tensor(Xs, dtype=torch.float64)
#         Ys = torch.as_tensor(Ys, dtype=torch.float64)
#         x = torch.as_tensor(x, dtype=torch.float64)

#         # K_inv = K(X,X)^{-1}
#         K = self.kernel(X, X, x)
#         K_inv = torch.linalg.inv(K)

#         kernel_Xs_Xs = self.kernel(Xs, Xs, x)
#         kernel_X_Xs = self.kernel(X, Xs, x)

#         if self.kernel_spec != "GRE":
#             kernel_Xs_X = kernel_X_Xs.T
#         else:
#             kernel_Xs_X = self.kernel(Xs, X, x)

#         # Basis VA
#         VA = x2fx(X, A)        # (n,m)
#         vAT = x2fx(Xs, A).T    # (m,1)

#         VA_T_at_K_inv = VA.T @ K_inv          # (m,n)
#         residual_X_Xs = vAT - VA_T_at_K_inv @ kernel_X_Xs  # (m,1)

#         inv_VA_T_K_inv_VA = torch.linalg.inv(VA_T_at_K_inv @ VA)  # (m,m)

#         cov_val = (
#             kernel_Xs_Xs
#             - kernel_Xs_X @ K_inv @ kernel_X_Xs
#             + residual_X_Xs.T @ inv_VA_T_K_inv_VA @ residual_X_Xs
#         )  # (1,1)

#         beta_X_Y = inv_VA_T_K_inv_VA @ (VA_T_at_K_inv @ Y)  # (m,)

#         mu_val = kernel_Xs_X @ K_inv @ Y + residual_X_Xs.T @ beta_X_Y  # (1,)

#         if return_mu_cov:
#             cov_scalar = cov_val[0, 0]
#             if cov_scalar < 0:
#                 cov_scalar = torch.tensor(0.0, dtype=torch.float64)
#                 cov_val = cov_val.clone()
#                 cov_val[0, 0] = cov_scalar
#             return mu_val.view(1, 1), cov_val

#         # 否则返回 log-likelihood
#         diff = Ys.view(-1, 1) - mu_val.view(1, 1)
#         inv_cov = torch.linalg.inv(cov_val)
#         term1 = -0.5 * torch.log(torch.det(2 * math.pi * cov_val))
#         term2 = -0.5 * (diff.T @ inv_cov @ diff)
#         return (term1 + term2).view(())

#     # def cv_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     总 LOOCV loss：sum_i local_loss_i
#     #     """
#     #     x = torch.as_tensor(x, dtype=torch.float64)
#     #     A = torch.as_tensor(A, dtype=torch.float64)
#     #     n = self.X_normalised.shape[0]
#     #     total = 0.0
#     #     for i in range(n):
#     #         total = total + self.cv_local_loss(x, A, i, return_mu_cov=False)
#     #     return torch.as_tensor(total, dtype=torch.float64)
#     def cv_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
#         """
#         Vectorized Leave-One-Out Cross-Validation (LOOCV) log-likelihood.

#         Uses the closed-form Dubrule (1983) formula to compute LOOCV statistics
#         without explicit n-fold retraining, achieving O(n³) instead of O(n⁴).

#         Mathematical Formulation
#         -----------------------
#         Given the augmented linear system for GP with polynomial mean:

#             [K   Φ] [α]   [y]
#             [Φᵀ  0] [β] = [0]

#         where:
#             K ∈ R^{n×n}: kernel (covariance) matrix
#             Φ ∈ R^{n×m}: polynomial design matrix
#             α ∈ R^n: GP weight coefficients
#             β ∈ R^m: polynomial basis coefficients

#         The LOOCV residuals and variances can be computed via:

#             LOO residual:  r_i = α_i / [M^{-1}]_{ii}
#             LOO variance:  σ²_i = 1 / [M^{-1}]_{ii}

#         where M = [[K, Φ], [Φᵀ, 0]] ∈ R^{(n+m)×(n+m)} and α = M^{-1}[y; 0].

#         Algorithm Complexity
#         -------------------
#         O(n³): dominated by single (n+m)×(n+m) matrix inversion
#         Compare to naive LOOCV: O(n⁴) from n separate n×n inversions

#         Parameters
#         ----------
#         x : torch.Tensor, shape (p,)
#             Kernel hyperparameters [σ², ℓ, ...], transformed via softplus
#         A : torch.Tensor, shape (m, d)
#             Multi-index set defining polynomial basis exponents
#             Entry A[j, k] is the exponent of dimension k in basis j

#         Returns
#         -------
#         log_likelihood : torch.Tensor, scalar
#             Sum of LOOCV log-predictive densities:
#             L = Σ_i [-½ log(2πσ²_i) - ½(r²_i/σ²_i)]

#         References
#         ----------
#         .. [Dubrule1983] Dubrule, O. (1983). Cross validation of kriging
#            in a unique neighborhood. Mathematical Geology, 15(6), 687-699.

#         Notes
#         -----
#         - Uses diagonal jitter (JITTER = 1e-12) for numerical stability
#         - Clamps variances to minimum of 1e-14 to prevent underflow
#         - Returns total log-likelihood (higher is better)
#         """
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         X = self.X_normalised
#         Y = self.Y_normalised.view(-1, 1)  # (n, 1)
#         n = X.shape[0]
#         m = A.shape[0]  # Number of polynomial basis terms

#         # ───────────────────────────────────────────────────────────────────
#         # Step 1: Compute kernel (covariance) matrix K ∈ R^{n×n}
#         # ───────────────────────────────────────────────────────────────────
#         K = self.kernel(X, X, x)

#         # ───────────────────────────────────────────────────────────────────
#         # Step 2: Compute polynomial design matrix Φ ∈ R^{n×m}
#         # ───────────────────────────────────────────────────────────────────
#         # Φ[i,j] = ∏_k X[i,k]^{A[j,k]}
#         V = x2fx(X, A)

#         # ───────────────────────────────────────────────────────────────────
#         # Step 3: Construct augmented matrix M = [[K, Φ], [Φᵀ, 0]]
#         # ───────────────────────────────────────────────────────────────────
#         # This is the Kriging system with polynomial drift
#         # M ∈ R^{(n+m)×(n+m)}
#         M_top = torch.cat([K, V], dim=1)  # (n, n+m)
#         zeros_m = torch.zeros((m, m), dtype=torch.float64, device=X.device)
#         M_bot = torch.cat([V.T, zeros_m], dim=1)  # (m, n+m)
#         M = torch.cat([M_top, M_bot], dim=0)  # (n+m, n+m)

#         # ───────────────────────────────────────────────────────────────────
#         # Step 4: Compute M^{-1} with diagonal jitter for stability
#         # ───────────────────────────────────────────────────────────────────
#         M_inv = torch.linalg.inv(M + torch.eye(n + m, device=X.device) * JITTER)

#         # ───────────────────────────────────────────────────────────────────
#         # Step 5: Construct augmented response vector [y; 0] ∈ R^{n+m}
#         # ───────────────────────────────────────────────────────────────────
#         zeros_y = torch.zeros((m, 1), dtype=torch.float64, device=X.device)
#         Y_aug = torch.cat([Y, zeros_y], dim=0)  # (n+m, 1)

#         # ───────────────────────────────────────────────────────────────────
#         # Step 6: Apply Dubrule's formula for LOOCV statistics
#         # ───────────────────────────────────────────────────────────────────
#         # Solve: M @ [α; β] = [y; 0]
#         # Extract first n coefficients (corresponding to data points)
#         alpha = (M_inv @ Y_aug)[:n]  # (n, 1)
#         diag_inv = torch.diagonal(M_inv)[:n]  # (n,)

#         # LOOCV residuals: r_i = α_i / [M^{-1}]_{ii}
#         residuals = alpha.flatten() / diag_inv

#         # LOOCV variances: σ²_i = 1 / [M^{-1}]_{ii}
#         variances = 1.0 / diag_inv

#         # ───────────────────────────────────────────────────────────────────
#         # Step 7: Compute log-likelihood L = Σ_i log p(y_i | y_{-i})
#         # ───────────────────────────────────────────────────────────────────
#         # For Gaussian predictive density:
#         # log p(y_i | y_{-i}) = -½ log(2πσ²_i) - ½(r²_i/σ²_i)

#         # Clamp variances to prevent numerical underflow
#         variances = torch.clamp(variances, min=1e-14)

#         # Compute log-likelihood components
#         term1 = -0.5 * torch.log(2 * math.pi * variances)  # Normalization term
#         term2 = -0.5 * (residuals ** 2) / variances  # Squared error term

#         # Sum over all n data points
#         total_ll = (term1 + term2).sum()

#         return total_ll

#     # ═══════════════════════════════════════════════════════════════════════
#     # V. PREDICTION & EXTRAPOLATION
#     # ═══════════════════════════════════════════════════════════════════════

#     def perform_extrapolation(
#         self,
#         x: torch.Tensor,
#         A: torch.Tensor,
#         return_mu_and_var: bool = False,
#     ) -> dict:
#         """
#         Perform SPRE extrapolation given hyperparameters and polynomial basis.

#         Extrapolates the GP + polynomial model to x = 0 (the Richardson limit)
#         and optionally computes LOOCV predictions at all training points.

#         Parameters
#         ----------
#         x : torch.Tensor, shape (p,)
#             Kernel hyperparameters [σ², ℓ, ...]
#         A : torch.Tensor, shape (m, d)
#             Polynomial basis multi-index set
#         return_mu_and_var : bool, default=False
#             If True, also compute predictions (mu, var) at x=0 and LOOCV
#             predictions (mu_cv, var_cv) at all training points

#         Returns
#         -------
#         result : dict
#             Dictionary containing:
#             - 'cv': LOOCV log-likelihood criterion
#             - 'mu': (if return_mu_and_var) extrapolation mean at x=0
#             - 'var': (if return_mu_and_var) extrapolation variance at x=0
#             - 'mu_cv': (if return_mu_and_var) LOOCV means at training points
#             - 'var_cv': (if return_mu_and_var) LOOCV variances at training points

#         Notes
#         -----
#         All returned mu/var values are in original (unnormalized) units.
#         """
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         cv = self.cv_loss(x, A)
#         out = {"cv": cv}

#         if return_mu_and_var:
#             n_train = self.X_normalised.shape[0]
#             mu_cv = torch.zeros(n_train, dtype=torch.float64)
#             var_cv = torch.zeros(n_train, dtype=torch.float64)

#             for i in range(n_train):
#                 mu_val, cov_val = self.cv_local_loss(x, A, i, return_mu_cov=True)
#                 mu_cv[i] = self.nY * mu_val[0, 0]
#                 var_cv[i] = (self.nY ** 2) * cov_val[0, 0]

#             # f(0) 处的预测（Xs=0）
#             Xs0 = torch.zeros((1, self.dimension), dtype=torch.float64)
#             Ys0 = torch.zeros((1,), dtype=torch.float64)  # 不重要
#             mu_val0, cov_val0 = self.cv_loss_calculation(
#                 A,
#                 self.X_normalised,
#                 self.Y_normalised,
#                 Xs0,
#                 Ys0,
#                 x,
#                 return_mu_cov=True,
#             )

#             mu = self.nY * mu_val0
#             var = (self.nY ** 2) * cov_val0

#             out.update(
#                 {
#                     "mu": mu,
#                     "var": var,
#                     "mu_cv": mu_cv,
#                     "var_cv": var_cv,
#                 }
#             )

#         return out

#     def objective(self, x_np: np.ndarray, A: np.ndarray) -> float:
#         """
#         SciPy 用的 objective：返回负的 cv（因为要最小化）。
#         """
#         x_t = torch.as_tensor(x_np, dtype=torch.float64)
#         A_t = torch.as_tensor(A, dtype=torch.float64)
#         val = self.cv_loss(x_t, A_t)
#         return float((-val).detach().cpu().numpy())


#     # ═══════════════════════════════════════════════════════════════════════
#     # VI. HYPERPARAMETER OPTIMIZATION
#     # ═══════════════════════════════════════════════════════════════════════

#     def perform_extrapolation_optimization(self, A: torch.Tensor, do_jit: bool = True) -> dict:
#         """
#         Optimize kernel hyperparameters via multi-start L-BFGS.

#         Uses PyTorch's native L-BFGS optimizer with automatic differentiation
#         to maximize the LOOCV log-likelihood. Multiple random restarts help
#         avoid local minima.

#         Algorithm
#         ---------
#         1. Generate starting points: default + random perturbations
#         2. For each start point:
#             a. Run L-BFGS optimization with automatic gradients
#             b. Track best result across all starts
#         3. Return optimal hyperparameters and LOOCV criterion

#         Parameters
#         ----------
#         A : torch.Tensor, shape (m, d)
#             Polynomial basis multi-index set (fixed during optimization)
#         do_jit : bool, default=True
#             Deprecated parameter (kept for API compatibility)

#         Returns
#         -------
#         result : dict
#             Dictionary containing:
#             - 'x': optimal hyperparameters (torch.Tensor)
#             - 'cv': LOOCV log-likelihood at optimum (torch.Tensor, scalar)

#         Notes
#         -----
#         - Uses DEFAULT_NUM_RESTARTS = 10 random starting points
#         - Each L-BFGS run has max 100 iterations with strong Wolfe line search
#         - Hyperparameters are transformed via softplus to ensure positivity
#         """
#         A = torch.as_tensor(A, dtype=torch.float64)

#         # ───────────────────────────────────────────────────────────────────
#         # Step 1: Generate multi-start initial points
#         # ───────────────────────────────────────────────────────────────────
#         default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)

#         # Create list of starting points: default + random perturbations
#         num_restarts = DEFAULT_NUM_RESTARTS
#         starting_points = [default_params]
#         for _ in range(num_restarts):
#             # Perturb in log-space to maintain positivity
#             noise = torch.randn_like(default_params) * 0.5
#             perturbed = default_params * torch.exp(noise)
#             starting_points.append(perturbed)

#         best_cv = float('inf')
#         best_x = default_params

#         # ───────────────────────────────────────────────────────────────────
#         # Step 2: Optimize from each starting point
#         # ───────────────────────────────────────────────────────────────────
#         for start_x in starting_points:
#             try:
#                 x_opt, cv_val = self._optimize_torch(start_x, A)

#                 # Track best result across all restarts
#                 if cv_val < best_cv:
#                     best_cv = cv_val
#                     best_x = x_opt
#             except Exception as e:
#                 # Skip starting points that cause numerical instability
#                 continue

#         return {"x": best_x, "cv": torch.tensor(best_cv, dtype=torch.float64)}

#     def _optimize_torch(self, init_x: torch.Tensor, A: torch.Tensor):
#         """
#         内部辅助函数：使用 L-BFGS 优化单个起点
#         修复核心逻辑：最小化 Negative Log-Likelihood
#         """
#         # 复制参数并开启梯度
#         x_param = init_x.clone().detach().requires_grad_(True)
        
#         # L-BFGS 设置 (保持不变)
#         optimizer = torch.optim.LBFGS(
#             [x_param], 
#             lr=1.0, 
#             max_iter=100, 
#             max_eval=120, 
#             tolerance_grad=1e-5, 
#             tolerance_change=1e-5,
#             history_size=10,
#             line_search_fn="strong_wolfe"
#         )

#         def closure():
#             optimizer.zero_grad()
            
#             # 1. 计算原始 Log-Likelihood (LL)
#             # LL 越大代表模型越好
#             raw_ll = self.cv_loss(x_param, A)
            
#             # 2. 关键修复：取负号！
#             # Loss = -LL
#             # 因为 L-BFGS 会尝试把 Loss 变得越小越好，
#             # 这等价于把 LL 变得越大越好。
#             loss = -raw_ll
            
#             # 3. 反向传播
#             if loss.requires_grad:
#                 loss.backward()
            
#             return loss

#         # 执行优化
#         optimizer.step(closure)
        
#         # 4. 计算最终结果
#         # 我们返回 loss (即 -LL)，这样你的 stepwise_selection 逻辑
#         # (if cv_new < cv) 依然成立：我们希望 Loss 越小越好。
#         final_ll = self.cv_loss(x_param, A)
#         final_loss = -final_ll
        
#         return x_param.detach(), final_loss.item()
#     # ═══════════════════════════════════════════════════════════════════════
#     # VII. GREEDY BASIS SELECTION ALGORITHM
#     # ═══════════════════════════════════════════════════════════════════════

#     def stepwise_selection(self) -> dict:
#         """
#         Greedy forward stepwise polynomial basis selection.

#         Iteratively adds polynomial terms that improve the LOOCV criterion
#         until no improvement is observed. Automatically uses GRE-specific
#         stepwise selection if gre_base was provided during initialization.

#         Algorithm (Forward Stepwise Selection)
#         --------------------------------------
#         1. Initialize: α₀ = {0} (intercept only)
#         2. For polynomial order k = 1, 2, ...:
#             a. Generate candidate terms C_k of order k
#             b. For each c ∈ C_k:
#                 - Test α ∪ {c} via LOOCV
#                 - Accept if L(α ∪ {c}) > L(α)
#             c. If no terms accepted: STOP
#             d. Update α ← α ∪ {accepted terms}
#         3. Optimize hyperparameters for final α
#         4. Return extrapolation to x = 0

#         Returns
#         -------
#         result : dict
#             Dictionary containing:
#             - 'cv': LOOCV log-likelihood
#             - 'mu': extrapolation mean at x=0
#             - 'var': extrapolation variance at x=0
#             - 'mu_cv': LOOCV means at training points
#             - 'var_cv': LOOCV variances at training points

#         Complexity
#         ----------
#         O(d² × n × n³) per iteration, dominated by matrix inversion

#         Notes
#         -----
#         - Ensures unisolvent bases (full rank design matrix)
#         - Limits basis size to n-1 to avoid overfitting
#         - Uses multi-start L-BFGS for hyperparameter optimization
#         """
#         n_train = self.X_normalised.shape[0]

#         # ───────────────────────────────────────────────────────────────────
#         # Initialize with intercept-only basis: α₀ = {0}
#         # ───────────────────────────────────────────────────────────────────
#         A = torch.zeros((1, self.dimension), dtype=torch.int64)

#         # Delegate to GRE-specific stepwise if in GRE mode
#         if self.kernel_base is not None:
#             return self._GRE_stepwise_selection(A)

#         # ───────────────────────────────────────────────────────────────────
#         # Standard SPRE stepwise selection
#         # ───────────────────────────────────────────────────────────────────
#         do_jit = False  # Legacy parameter (no longer used)

#         # Optimize hyperparameters for initial basis
#         order = 0
#         fit = self.perform_extrapolation_optimization(A, do_jit)
#         cv = fit["cv"]

#         carry_on = True

#         while carry_on:
#             m = A.shape[0]
#             order += 1

#             # Generate candidate terms of current order
#             A_extra = stepwise(A, order)
#             n_extra = A_extra.shape[0]
#             to_include = torch.zeros(n_extra, dtype=torch.bool)

#             print(f"Fitting interactions of order {order}:")

#             # Test each candidate term
#             for i in tqdm(range(n_extra), desc="Stepwise progress"):
#                 A_new = torch.cat([A, A_extra[i : i + 1, :]], dim=0)

#                 # Check if augmented basis is unisolvent (full rank)
#                 if self.check_unisolvent(A_new) > 0:
#                     fit_new = self.perform_extrapolation_optimization(A_new, do_jit)
#                     cv_new = fit_new["cv"]

#                     # Accept if LOOCV improves (note: cv is negative log-likelihood)
#                     if cv_new < cv:
#                         to_include[i] = True

#             # Add accepted terms if within size limit (n-1)
#             if to_include.any() and (m + int(to_include.sum().item()) < (n_train - 1)):
#                 A_updated = torch.cat([A, A_extra[to_include, :]], dim=0)
#                 fit_updated = self.perform_extrapolation_optimization(A_updated, do_jit)
#                 cv_updated = fit_updated["cv"]

#                 # Check if joint acceptance improves LOOCV
#                 if cv_updated >= cv:
#                     carry_on = False  # No improvement, terminate
#                 else:
#                     A = A_updated
#                     fit = fit_updated
#                     cv = cv_updated
#             else:
#                 carry_on = False  # No terms accepted or size limit reached

#         # Final extrapolation with optimal basis and hyperparameters
#         x_opt = fit["x"]
#         out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)

#         self.current_A = A.detach().clone()
#         self.current_x = x_opt.detach().clone()
#         out["A"] = self.current_A
#         out["x"] = self.current_x

#         return out

#     def _GRE_stepwise_selection(self, A: torch.Tensor) -> dict:
#         """
#         Stepwise selection for GRE (Gauss-Richardson Extrapolation).

#         A: basis for mean (fixed in GRE)
#         We add terms to B (rate function) stored in gre_base.
#         """
#         # 初始 B：仅截距
#         B = torch.zeros((1, self.dimension), dtype=torch.int64)

#         order = 0
#         # 设定当前 GRE kernel
#         self.set_kernel_spec(self.kernel_base, B)
#         fit = self.perform_extrapolation_optimization(A, do_jit=False)
#         cv = fit["cv"]

#         carry_on = True
#         while carry_on:
#             order += 1
#             B_extra = stepwise(B, order)
#             n_extra = B_extra.shape[0]
#             if n_extra == 0:
#                 break

#             print(f"Fitting GRE rate interactions of order {order}...")
#             to_include = torch.zeros(n_extra, dtype=torch.bool)

#             for i in tqdm(range(n_extra), desc="GRE stepwise progress"):
#                 B_new = torch.cat([B, B_extra[i : i + 1, :]], dim=0)
#                 self.set_kernel_spec(self.kernel_base, B_new)
#                 fit_new = self.perform_extrapolation_optimization(A, do_jit=False)
#                 cv_new = fit_new["cv"]
#                 if cv_new < cv:
#                     to_include[i] = True

#             if to_include.any():
#                 B_updated = torch.cat([B, B_extra[to_include, :]], dim=0)
#                 self.set_kernel_spec(self.kernel_base, B_updated)
#                 fit_updated = self.perform_extrapolation_optimization(A, do_jit=False)
#                 cv_updated = fit_updated["cv"]
#                 if cv_updated >= cv:
#                     carry_on = False
#                 else:
#                     B = B_updated
#                     fit = fit_updated
#                     cv = cv_updated
#             else:
#                 carry_on = False

#         # 最终用 B、best x 做一次完整外推
#         # x_opt = fit["x"]
#         # self.set_kernel_spec(self.kernel_base, B)
#         # out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)
#         # out["B"] = B
#         # Final extrapolation with optimal basis and hyperparameters
# # 最终用 B、best x 做一次完整外推
#         x_opt = fit["x"]
#         self.set_kernel_spec(self.kernel_base, B)
#         out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)


#         self.current_A = A.detach().clone()
#         self.current_x = x_opt.detach().clone()
#         self.current_B = B.detach().clone()


#         out["A"] = self.current_A
#         out["x"] = self.current_x
#         out["B"] = self.current_B

#         return out






# import math
# import numpy as np
# import torch
# from tqdm import tqdm

# from .helper_functions import (
#     x2fx,
#     softplus,
#     cellsum,
#     white,
#     remove_row,
#     stepwise,
# )

# # ═══════════════════════════════════════════════════════════════════════
# # Numerical Stability Constants
# # ═══════════════════════════════════════════════════════════════════════

# EPSILON = 1e-16
# JITTER = 1e-10
# MID_JITTER = 1e-14
# DEFAULT_NUM_RESTARTS = 10


# class SPRE:
#     """
#     Sparse Probabilistic Richardson Extrapolation (SPRE)
#     ====================================================

#     Universal kriging (CJO / Matlab style):

#       cov = k_ss
#             - k_sX K^{-1} k_Xs
#             + r^T (V^T K^{-1} V)^{-1} r

#     param_mode:
#       - "opt"    : raw theta -> softplus -> positive (for optimisation)
#       - "direct" : x already positive (for Matlab alignment)
#     """

#     # ────────────────────────────────────────────────────────────────
#     # I. INITIALISATION
#     # ────────────────────────────────────────────────────────────────

#     def __init__(
#         self,
#         kernel_spec: str,
#         dimension: int,
#         gre_base: torch.Tensor | None = None,
#         param_mode: str = "opt",
#     ):
#         self.dimension = int(dimension)
#         self.param_mode = param_mode

#         self.set_kernel_spec(kernel_spec, gre_base)

#         self.X_normalised = None
#         self.Y_normalised = None
#         self.nX = None
#         self.nY = None

#     # ────────────────────────────────────────────────────────────────
#     # II. KERNELS
#     # ────────────────────────────────────────────────────────────────

#     @staticmethod
#     def cdist_torch(XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
#         XA = torch.as_tensor(XA, dtype=torch.float64)
#         XB = torch.as_tensor(XB, dtype=torch.float64)
#         XA_sq = (XA ** 2).sum(dim=1, keepdim=True)
#         XB_sq = (XB ** 2).sum(dim=1)
#         cross = XA @ XB.T
#         nums = XA_sq - 2 * cross + XB_sq
#         nums = torch.clamp(nums, min=0.0)
#         return torch.sqrt(nums)

#     def set_kernel_spec(self, kernel_spec: str, gre_base: torch.Tensor | None = None):
#         if gre_base is None:
#             self.kernel_spec = kernel_spec
#             self.kernel_base = None
#         else:
#             self.kernel_spec = "GRE"
#             self.kernel_base = kernel_spec

#         self.gre_base = gre_base
#         self.set_kernel_default_parameters()

#     def set_kernel_default_parameters(self):
#         match self.kernel_spec:
#             case "Gaussian":
#                 self.default_kernel_parameters = [1.0, 0.1]
#             case "GaussianARD":
#                 self.default_kernel_parameters = [1.0] + [0.1] * self.dimension
#             case "white":
#                 self.default_kernel_parameters = [1.0]
#             case "Matern1/2":
#                 self.default_kernel_parameters = [1.0, 1.0]
#             case "Matern3/2":
#                 self.default_kernel_parameters = [1.0, 1.0]
#             case "GRE":
#                 base_spec = self.kernel_base
#                 self.kernel_spec = base_spec
#                 self.set_kernel_default_parameters()
#                 base_defaults = self.default_kernel_parameters
#                 self.kernel_spec = "GRE"
#                 self.kernel_base = base_spec
#                 self.default_kernel_parameters = [1.0] + base_defaults
#             case _:
#                 raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

#     # -------- parameter transform --------

#     def _pos_param(self, t: torch.Tensor) -> torch.Tensor:
#         if self.param_mode == "opt":
#             return 1e-2 + softplus(t)
#         if self.param_mode == "direct":
#             return EPSILON + t
#         raise ValueError("param_mode must be 'opt' or 'direct'")

#     # -------- kernel --------

#     def kernel(self, X1, X2, x=None):
#         X1 = torch.as_tensor(X1, dtype=torch.float64)
#         X2 = torch.as_tensor(X2, dtype=torch.float64)

#         if x is None:
#             x = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
#         else:
#             x = torch.as_tensor(x, dtype=torch.float64)

#         spec = self.kernel_spec

#         if spec == "Gaussian":
#             amp = self._pos_param(x[0])
#             ell = self._pos_param(x[1])
#             dist = self.cdist_torch(X1, X2)
#             return amp * torch.exp(-(dist ** 2) / (ell ** 2))

#         elif spec == "GaussianARD":
#             amp = self._pos_param(x[0])
#             length_terms = []
#             for i in range(self.dimension):
#                 ell_i = self._pos_param(x[i + 1])
#                 d_i = (self.cdist_torch(X1[:, [i]], X2[:, [i]]) ** 2) / (ell_i ** 2)
#                 length_terms.append(d_i)
#             sq_dist = cellsum(length_terms)
#             return amp * torch.exp(-sq_dist)

#         elif spec == "white":
#             amp = self._pos_param(x[0])
#             return amp * white(X1, X2)

#         elif spec == "Matern1/2":
#             amp = self._pos_param(x[0])
#             ell = self._pos_param(x[1])
#             dist = self.cdist_torch(X1, X2)
#             return amp * torch.exp(-dist / ell)

#         elif spec == "Matern3/2":
#             amp = self._pos_param(x[0])
#             ell = self._pos_param(x[1])
#             dist = self.cdist_torch(X1, X2)
#             r_l = dist / ell
#             sqrt3_r_l = math.sqrt(3.0) * r_l
#             return amp * (1.0 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)

#         elif spec == "GRE":
#             if self.gre_base is None:
#                 raise RuntimeError("GRE kernel requires gre_base.")
#             amp_gre = self._pos_param(x[0])
#             base_X1 = x2fx(X1, self.gre_base).sum(dim=1)
#             base_X2 = x2fx(X2, self.gre_base).sum(dim=1)

#             base_spec = self.kernel_base
#             old_spec = self.kernel_spec
#             self.kernel_spec = base_spec
#             K_base = self.kernel(X1, X2, x=x[1:])
#             self.kernel_spec = old_spec

#             return amp_gre * base_X1.unsqueeze(1) * K_base * base_X2.unsqueeze(0)

#         else:
#             raise ValueError(f"Unknown kernel specification: {spec}")

#     # ────────────────────────────────────────────────────────────────
#     # III. DATA NORMALISATION
#     # ────────────────────────────────────────────────────────────────

#     def set_normalised_data(self, X, Y):
#         X = torch.as_tensor(X, dtype=torch.float64)
#         Y = torch.as_tensor(Y, dtype=torch.float64).view(-1, 1)

#         self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
#         self.nY = (Y.max() - Y.min()) + EPSILON

#         self.X_normalised = X / self.nX
#         self.Y_normalised = (Y / self.nY).view(-1)

#     # ────────────────────────────────────────────────────────────────
#     # IV. CORE KRIGING
#     # ────────────────────────────────────────────────────────────────

#     def cv_local_loss(self, x, A, row_num, return_mu_cov=False):
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         X_full = self.X_normalised
#         Y_full = self.Y_normalised

#         X = remove_row(X_full, row_num)
#         Y = remove_row(Y_full.unsqueeze(1), row_num).view(-1, 1)

#         Xs = X_full[row_num: row_num + 1, :]
#         Ys = Y_full[row_num: row_num + 1].view(-1, 1)

#         return self.cv_loss_calculation(A, X, Y, Xs, Ys, x, return_mu_cov)

#     def check_unisolvent(self, A):
#         A = torch.as_tensor(A, dtype=torch.float64)
#         VA = x2fx(self.X_normalised, A)
#         return 1 if torch.linalg.matrix_rank(VA) == A.shape[0] else -1

#     # -------- main kriging routine --------

#     def cv_loss_calculation(self, A, X, Y, Xs, Ys, x, return_mu_cov=False):
#         A = torch.as_tensor(A, dtype=torch.float64)
#         X = torch.as_tensor(X, dtype=torch.float64)
#         Y = torch.as_tensor(Y, dtype=torch.float64).view(-1, 1)
#         Xs = torch.as_tensor(Xs, dtype=torch.float64)
#         Ys = torch.as_tensor(Ys, dtype=torch.float64).view(-1, 1)
#         x = torch.as_tensor(x, dtype=torch.float64)

#         # ---- kernel matrix ----
#         K = self.kernel(X, X, x)
#         n = K.shape[0]
#         K = K + JITTER * torch.eye(n, dtype=torch.float64, device=K.device)
#         L = torch.linalg.cholesky(K)

#         def solveK(B):
#             return torch.cholesky_solve(B, L)

#         # ---- kernel blocks ----
#         k_ss = self.kernel(Xs, Xs, x)    # (1,1)
#         k_sX = self.kernel(Xs, X, x)     # (1,n)
#         k_Xs = k_sX.T                    # (n,1)

#         # ---- basis ----
#         VA = x2fx(X, A)                 # (n,m)
#         vAT = x2fx(Xs, A).T             # (m,1)

#         # ---- mid = V^T K^{-1} V ----
#         KiVA = solveK(VA)
#         mid = VA.T @ KiVA
#         mid = mid + MID_JITTER * torch.eye(mid.shape[0], dtype=torch.float64, device=mid.device)

#         # ---- beta_hat ----
#         KiY = solveK(Y)
#         rhs = VA.T @ KiY
#         beta_hat = torch.linalg.solve(mid, rhs)

#         # ---- r term ----
#         Kik = solveK(k_Xs)
#         r = vAT - VA.T @ Kik

#         # ---- mean ----
#         mu_val = (k_sX @ KiY) + (r.T @ beta_hat)

#         # ---- covariance ----
#         q = (k_sX @ Kik)
#         base_cov = k_ss - q
#         infl = (r.T @ torch.linalg.solve(mid, r))
#         cov_val = base_cov + infl

#         if return_mu_cov:
#             diag = {
#                 "k_ss": k_ss.detach().cpu().numpy(),
#                 "base_cov": base_cov.detach().cpu().numpy(),
#                 "infl": infl.detach().cpu().numpy(),
#             }
#             return mu_val, cov_val, diag

#         # ---- scalar loglik ----
#         cov_scalar = cov_val[0, 0]
#         diff = (Ys - mu_val)[0, 0]
#         cov_safe = torch.clamp(cov_scalar, min=1e-30)
#         ll = -0.5 * (math.log(2.0 * math.pi) + torch.log(cov_safe) + (diff * diff) / cov_safe)
#         return ll.view(())

#     # ────────────────────────────────────────────────────────────────
#     # V. LOOCV
#     # ────────────────────────────────────────────────────────────────

#     def cv_loss(self, x, A):
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         total = torch.tensor(0.0, dtype=torch.float64)
#         for i in range(self.X_normalised.shape[0]):
#             total = total + self.cv_local_loss(x, A, i, False)
#         return total

#     # ────────────────────────────────────────────────────────────────
#     # VI. EXTRAPOLATION
#     # ────────────────────────────────────────────────────────────────

#     def perform_extrapolation(self, x, A, return_mu_and_var=False):
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         cv = self.cv_loss(x, A)
#         out = {"cv": cv}

#         if return_mu_and_var:
#             n = self.X_normalised.shape[0]
#             mu_cv = torch.zeros(n, dtype=torch.float64)
#             var_cv = torch.zeros(n, dtype=torch.float64)

#             for i in range(n):
#                 mu_i, cov_i = self.cv_local_loss(x, A, i, True)[:2]
#                 mu_cv[i] = self.nY * mu_i[0, 0]
#                 var_cv[i] = (self.nY ** 2) * cov_i[0, 0]

#             # ---- prediction at x = 0 ----
#             Xs0 = torch.zeros((1, self.dimension), dtype=torch.float64)
#             Ys0 = torch.zeros((1,), dtype=torch.float64)

#             mu0, cov0, diag0 = self.cv_loss_calculation(
#                 A, self.X_normalised, self.Y_normalised, Xs0, Ys0, x, True
#             )

#             # ---- diagnostics ----
#             print("---- DIAGNOSTICS at x=0 ----")
#             print(f"k_ss      = {diag0['k_ss'].ravel()[0]:.3e}")
#             print(f"base_cov  = {diag0['base_cov'].ravel()[0]:.3e}")
#             print(f"inflation = {diag0['infl'].ravel()[0]:.3e}")
#             print("--------------------------------")

#             mu = self.nY * mu0
#             var = (self.nY ** 2) * cov0

#             out.update({"mu": mu, "var": var, "mu_cv": mu_cv, "var_cv": var_cv})

#         return out

#     # ────────────────────────────────────────────────────────────────
#     # VII. OPTIMISATION
#     # ────────────────────────────────────────────────────────────────

#     def objective(self, x_np, A):
#         x_t = torch.as_tensor(x_np, dtype=torch.float64)
#         A_t = torch.as_tensor(A, dtype=torch.float64)
#         val = self.cv_loss(x_t, A_t)
#         return float((-val).detach().cpu().numpy())

#     def perform_extrapolation_optimization(self, A, do_jit=True):
#         A = torch.as_tensor(A, dtype=torch.float64)

#         default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
#         starts = [default_params]

#         for _ in range(DEFAULT_NUM_RESTARTS):
#             noise = torch.randn_like(default_params) * 0.5
#             starts.append(default_params * torch.exp(noise))

#         best_cv = float("inf")
#         best_x = default_params

#         for s in starts:
#             try:
#                 x_opt, cv_val = self._optimize_torch(s, A)
#                 if cv_val < best_cv:
#                     best_cv = cv_val
#                     best_x = x_opt
#             except Exception:
#                 continue

#         return {"x": best_x, "cv": torch.tensor(best_cv, dtype=torch.float64)}

#     def _optimize_torch(self, init_x, A):
#         x_param = init_x.clone().detach().requires_grad_(True)

#         optimizer = torch.optim.LBFGS(
#             [x_param],
#             lr=1.0,
#             max_iter=100,
#             max_eval=120,
#             tolerance_grad=1e-5,
#             tolerance_change=1e-5,
#             history_size=10,
#             line_search_fn="strong_wolfe",
#         )

#         def closure():
#             optimizer.zero_grad()
#             raw_ll = self.cv_loss(x_param, A)
#             loss = -raw_ll
#             if loss.requires_grad:
#                 loss.backward()
#             return loss

#         optimizer.step(closure)

#         final_ll = self.cv_loss(x_param, A)
#         final_loss = -final_ll
#         return x_param.detach(), final_loss.item()

#     # ────────────────────────────────────────────────────────────────
#     # VIII. STEPWISE (unchanged)
#     # ────────────────────────────────────────────────────────────────

#     def stepwise_selection(self):
#         n_train = self.X_normalised.shape[0]
#         A = torch.zeros((1, self.dimension), dtype=torch.int64)

#         if self.kernel_base is not None:
#             return self._GRE_stepwise_selection(A)

#         order = 0
#         fit = self.perform_extrapolation_optimization(A)
#         cv = fit["cv"]

#         carry_on = True
#         while carry_on:
#             order += 1
#             A_extra = stepwise(A, order)
#             to_include = torch.zeros(A_extra.shape[0], dtype=torch.bool)

#             print(f"Fitting interactions of order {order}:")
#             for i in tqdm(range(A_extra.shape[0]), desc="Stepwise progress"):
#                 A_new = torch.cat([A, A_extra[i: i + 1]], dim=0)
#                 if self.check_unisolvent(A_new) > 0:
#                     fit_new = self.perform_extrapolation_optimization(A_new)
#                     if fit_new["cv"] < cv:
#                         to_include[i] = True

#             if to_include.any():
#                 A_updated = torch.cat([A, A_extra[to_include]], dim=0)
#                 fit_updated = self.perform_extrapolation_optimization(A_updated)
#                 if fit_updated["cv"] < cv:
#                     A, cv, fit = A_updated, fit_updated["cv"], fit_updated
#                 else:
#                     carry_on = False
#             else:
#                 carry_on = False

#         out = self.perform_extrapolation(fit["x"], A, True)
#         self.current_A = A
#         self.current_x = fit["x"]
#         out["A"] = A
#         out["x"] = fit["x"]
#         return out

#     def _GRE_stepwise_selection(self, A):
#         B = torch.zeros((1, self.dimension), dtype=torch.int64)

#         order = 0
#         self.set_kernel_spec(self.kernel_base, B)
#         fit = self.perform_extrapolation_optimization(A)
#         cv = fit["cv"]

#         carry_on = True
#         while carry_on:
#             order += 1
#             B_extra = stepwise(B, order)
#             if B_extra.shape[0] == 0:
#                 break

#             to_include = torch.zeros(B_extra.shape[0], dtype=torch.bool)

#             for i in tqdm(range(B_extra.shape[0]), desc="GRE stepwise"):
#                 B_new = torch.cat([B, B_extra[i: i + 1]], dim=0)
#                 self.set_kernel_spec(self.kernel_base, B_new)
#                 fit_new = self.perform_extrapolation_optimization(A)
#                 if fit_new["cv"] < cv:
#                     to_include[i] = True

#             if to_include.any():
#                 B_updated = torch.cat([B, B_extra[to_include]], dim=0)
#                 self.set_kernel_spec(self.kernel_base, B_updated)
#                 fit_updated = self.perform_extrapolation_optimization(A)
#                 if fit_updated["cv"] < cv:
#                     B, cv, fit = B_updated, fit_updated["cv"], fit_updated
#                 else:
#                     carry_on = False
#             else:
#                 carry_on = False

#         self.set_kernel_spec(self.kernel_base, B)
#         out = self.perform_extrapolation(fit["x"], A, True)
#         self.current_A = A
#         self.current_x = fit["x"]
#         self.current_B = B
#         out["A"] = A
#         out["x"] = fit["x"]
#         out["B"] = B
#         return out


# import math
# import numpy as np
# import torch
# from tqdm import tqdm

# from .helper_functions import (
#     x2fx,
#     softplus,
#     cellsum,
#     white,
#     remove_row,
#     stepwise,
# )

# # ═══════════════════════════════════════════════════════════════════════
# # Numerical Stability Constants
# # ═══════════════════════════════════════════════════════════════════════

# EPSILON = 1e-16
# JITTER = 1e-10      # 用于扩展矩阵 M 的对角扰动
# MID_JITTER = 1e-14  # 用于内部矩阵计算的扰动
# DEFAULT_NUM_RESTARTS = 10


# class SPRE:
#     """
#     Sparse Probabilistic Richardson Extrapolation (SPRE)
#     ====================================================
#     升级版：采用 Dubrule (1983) 闭式公式实现向量化 LOOCV 计算。
#     """

#     # ────────────────────────────────────────────────────────────────
#     # I. INITIALISATION
#     # ────────────────────────────────────────────────────────────────

#     def __init__(
#         self,
#         kernel_spec: str,
#         dimension: int,
#         gre_base: torch.Tensor | None = None,
#         param_mode: str = "opt",
#     ):
#         self.dimension = int(dimension)
#         self.param_mode = param_mode
#         self.set_kernel_spec(kernel_spec, gre_base)

#         self.X_normalised = None
#         self.Y_normalised = None
#         self.nX = None
#         self.nY = None

#     # ────────────────────────────────────────────────────────────────
#     # II. KERNELS
#     # ────────────────────────────────────────────────────────────────

#     @staticmethod
#     def cdist_torch(XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
#         XA = torch.as_tensor(XA, dtype=torch.float64)
#         XB = torch.as_tensor(XB, dtype=torch.float64)
#         XA_sq = (XA ** 2).sum(dim=1, keepdim=True)
#         XB_sq = (XB ** 2).sum(dim=1)
#         cross = XA @ XB.T
#         nums = XA_sq - 2 * cross + XB_sq
#         nums = torch.clamp(nums, min=0.0)
#         return torch.sqrt(nums)

#     def set_kernel_spec(self, kernel_spec: str, gre_base: torch.Tensor | None = None):
#         if gre_base is None:
#             self.kernel_spec = kernel_spec
#             self.kernel_base = None
#         else:
#             self.kernel_spec = "GRE"
#             self.kernel_base = kernel_spec

#         self.gre_base = gre_base
#         self.set_kernel_default_parameters()

#     def set_kernel_default_parameters(self):
#         match self.kernel_spec:
#             case "Gaussian":
#                 self.default_kernel_parameters = [1.0, 0.1]
#             case "GaussianARD":
#                 self.default_kernel_parameters = [1.0] + [0.1] * self.dimension
#             case "white":
#                 self.default_kernel_parameters = [1.0]
#             case "Matern1/2":
#                 self.default_kernel_parameters = [1.0, 1.0]
#             case "Matern3/2":
#                 self.default_kernel_parameters = [1.0, 1.0]
#             case "GRE":
#                 base_spec = self.kernel_base
#                 self.kernel_spec = base_spec
#                 self.set_kernel_default_parameters()
#                 base_defaults = self.default_kernel_parameters
#                 self.kernel_spec = "GRE"
#                 self.kernel_base = base_spec
#                 self.default_kernel_parameters = [1.0] + base_defaults
#             case _:
#                 raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

#     def _pos_param(self, t: torch.Tensor) -> torch.Tensor:
#         if self.param_mode == "opt":
#             return 1e-2 + softplus(t)
#         if self.param_mode == "direct":
#             return EPSILON + t
#         raise ValueError("param_mode must be 'opt' or 'direct'")

#     def kernel(self, X1, X2, x=None):
#         X1 = torch.as_tensor(X1, dtype=torch.float64)
#         X2 = torch.as_tensor(X2, dtype=torch.float64)
#         if x is None:
#             x = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
#         else:
#             x = torch.as_tensor(x, dtype=torch.float64)

#         spec = self.kernel_spec
#         if spec == "Gaussian":
#             amp = self._pos_param(x[0])
#             ell = self._pos_param(x[1])
#             dist = self.cdist_torch(X1, X2)
#             return amp * torch.exp(-(dist ** 2) / (ell ** 2))
#         elif spec == "GaussianARD":
#             amp = self._pos_param(x[0])
#             length_terms = []
#             for i in range(self.dimension):
#                 ell_i = self._pos_param(x[i + 1])
#                 d_i = (self.cdist_torch(X1[:, [i]], X2[:, [i]]) ** 2) / (ell_i ** 2)
#                 length_terms.append(d_i)
#             sq_dist = cellsum(length_terms)
#             return amp * torch.exp(-sq_dist)
#         elif spec == "white":
#             amp = self._pos_param(x[0])
#             return amp * white(X1, X2)
#         elif spec == "Matern1/2":
#             amp = self._pos_param(x[0])
#             ell = self._pos_param(x[1])
#             dist = self.cdist_torch(X1, X2)
#             return amp * torch.exp(-dist / ell)
#         elif spec == "Matern3/2":
#             amp = self._pos_param(x[0])
#             ell = self._pos_param(x[1])
#             dist = self.cdist_torch(X1, X2)
#             r_l = dist / ell
#             sqrt3_r_l = math.sqrt(3.0) * r_l
#             return amp * (1.0 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)
#         elif spec == "GRE":
#             if self.gre_base is None: raise RuntimeError("GRE kernel requires gre_base.")
#             amp_gre = self._pos_param(x[0])
#             base_X1 = x2fx(X1, self.gre_base).sum(dim=1)
#             base_X2 = x2fx(X2, self.gre_base).sum(dim=1)
#             base_spec = self.kernel_base
#             old_spec = self.kernel_spec
#             self.kernel_spec = base_spec
#             K_base = self.kernel(X1, X2, x=x[1:])
#             self.kernel_spec = old_spec
#             return amp_gre * base_X1.unsqueeze(1) * K_base * base_X2.unsqueeze(0)
#         else:
#             raise ValueError(f"Unknown kernel specification: {spec}")

#     # ────────────────────────────────────────────────────────────────
#     # III. DATA NORMALISATION
#     # ────────────────────────────────────────────────────────────────

#     def set_normalised_data(self, X, Y):
#         X = torch.as_tensor(X, dtype=torch.float64)
#         Y = torch.as_tensor(Y, dtype=torch.float64).view(-1, 1)
#         self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
#         self.nY = (Y.max() - Y.min()) + EPSILON
#         self.X_normalised = X / self.nX
#         self.Y_normalised = (Y / self.nY).view(-1)

#     # ────────────────────────────────────────────────────────────────
#     # IV. VECTORIZED LOOCV (Dubrule 1983)
#     # ────────────────────────────────────────────────────────────────

#     def cv_loss(self, x, A) -> torch.Tensor:
#         """
#         升级版：基于 M 矩阵逆的对角线元素快速计算 LOOCV 似然和。
#         复杂度 O(n^3)，优于原始 O(n^4)。
#         """
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         X = self.X_normalised
#         Y = self.Y_normalised.view(-1, 1)  # (n, 1)
#         n = X.shape[0]
#         m = A.shape[0]

#         # 1. 构造扩展系统矩阵 M = [[K, V], [V.T, 0]]
#         K = self.kernel(X, X, x)
#         V = x2fx(X, A)
        
#         M_top = torch.cat([K, V], dim=1)  # (n, n+m)
#         zeros_m = torch.zeros((m, m), dtype=torch.float64, device=X.device)
#         M_bot = torch.cat([V.T, zeros_m], dim=1)  # (m, n+m)
#         M = torch.cat([M_top, M_bot], dim=0)  # (n+m, n+m)

#         # 2. 矩阵求逆 (加入 JITTER 保证稳定性)
#         M_inv = torch.linalg.inv(M + torch.eye(n + m, device=X.device) * JITTER)

#         # 3. 计算增强响应向量 [y; 0] 的解
#         zeros_y = torch.zeros((m, 1), dtype=torch.float64, device=X.device)
#         Y_aug = torch.cat([Y, zeros_y], dim=0)  # (n+m, 1)
        
#         # 解方程 M @ [alpha; beta] = [y; 0]
#         # 根据 Dubrule 1983 公式：
#         # 留一残差 r_i = alpha_i / M_inv[i,i]
#         # 留一方差 sigma2_i = 1 / M_inv[i,i]
#         sol_aug = M_inv @ Y_aug
#         alpha = sol_aug[:n]
#         diag_inv = torch.diagonal(M_inv)[:n]

#         residuals = alpha.flatten() / diag_inv
#         variances = 1.0 / diag_inv

#         # 4. 数值保护并计算总对数似然
#         variances = torch.clamp(variances, min=1e-14)
        
#         term1 = -0.5 * torch.log(2 * math.pi * variances)
#         term2 = -0.5 * (residuals ** 2) / variances
        
#         return (term1 + term2).sum()

#     # ────────────────────────────────────────────────────────────────
#     # V. KRIGING CORE (For Prediction at x=0)
#     # ────────────────────────────────────────────────────────────────

#     def cv_loss_calculation(self, A, X, Y, Xs, Ys, x, return_mu_cov=False):
#         """保持原有计算逻辑，用于生成最终外推点"""
#         A = torch.as_tensor(A, dtype=torch.float64)
#         X = torch.as_tensor(X, dtype=torch.float64)
#         Y = torch.as_tensor(Y, dtype=torch.float64).view(-1, 1)
#         Xs = torch.as_tensor(Xs, dtype=torch.float64)
#         Ys = torch.as_tensor(Ys, dtype=torch.float64).view(-1, 1)
#         x = torch.as_tensor(x, dtype=torch.float64)

#         K = self.kernel(X, X, x)
#         n = K.shape[0]
#         K = K + JITTER * torch.eye(n, dtype=torch.float64, device=K.device)
#         L = torch.linalg.cholesky(K)

#         def solveK(B): return torch.cholesky_solve(B, L)

#         k_ss = self.kernel(Xs, Xs, x)
#         k_sX = self.kernel(Xs, X, x)
#         k_Xs = k_sX.T

#         VA = x2fx(X, A)
#         vAT = x2fx(Xs, A).T

#         KiVA = solveK(VA)
#         mid = VA.T @ KiVA
#         mid = mid + MID_JITTER * torch.eye(mid.shape[0], dtype=torch.float64)

#         KiY = solveK(Y)
#         beta_hat = torch.linalg.solve(mid, VA.T @ KiY)

#         Kik = solveK(k_Xs)
#         r = vAT - VA.T @ Kik

#         mu_val = (k_sX @ KiY) + (r.T @ beta_hat)
#         q = (k_sX @ Kik)
#         base_cov = k_ss - q
#         infl = (r.T @ torch.linalg.solve(mid, r))
#         cov_val = base_cov + infl

#         if return_mu_cov:
#             diag = {"k_ss": k_ss.detach().cpu().numpy(), "base_cov": base_cov.detach().cpu().numpy(), "infl": infl.detach().cpu().numpy()}
#             return mu_val, cov_val, diag
        
#         cov_safe = torch.clamp(cov_val[0,0], min=1e-30)
#         diff = (Ys - mu_val)[0,0]
#         ll = -0.5 * (math.log(2.0 * math.pi) + torch.log(cov_safe) + (diff * diff) / cov_safe)
#         return ll.view(())

#     def check_unisolvent(self, A):
#         A = torch.as_tensor(A, dtype=torch.float64)
#         VA = x2fx(self.X_normalised, A)
#         return 1 if torch.linalg.matrix_rank(VA) == A.shape[0] else -1

#     # ────────────────────────────────────────────────────────────────
#     # VI. EXTRAPOLATION & POSTERIOR
#     # ────────────────────────────────────────────────────────────────

#     def perform_extrapolation(self, x, A, return_mu_and_var=False):
#         x = torch.as_tensor(x, dtype=torch.float64)
#         A = torch.as_tensor(A, dtype=torch.float64)

#         cv = self.cv_loss(x, A)
#         out = {"cv": cv}

#         if return_mu_and_var:
#             # 此处保持显式循环用于生成各训练点的 mu/var 结果输出（仅在最终展示时调用，不影响优化速度）
#             n = self.X_normalised.shape[0]
#             mu_cv = torch.zeros(n, dtype=torch.float64)
#             var_cv = torch.zeros(n, dtype=torch.float64)
#             X_full = self.X_normalised
#             Y_full = self.Y_normalised

#             for i in range(n):
#                 X_sub = remove_row(X_full, i)
#                 Y_sub = remove_row(Y_full.unsqueeze(1), i).view(-1, 1)
#                 Xs_i = X_full[i:i+1, :]
#                 Ys_i = Y_full[i:i+1].view(-1, 1)
#                 m_i, c_i = self.cv_loss_calculation(A, X_sub, Y_sub, Xs_i, Ys_i, x, True)[:2]
#                 mu_cv[i] = self.nY * m_i[0,0]
#                 var_cv[i] = (self.nY**2) * c_i[0,0]

#             # 外推到 x=0
#             Xs0 = torch.zeros((1, self.dimension), dtype=torch.float64)
#             Ys0 = torch.zeros((1,), dtype=torch.float64)
#             mu0, cov0, diag0 = self.cv_loss_calculation(A, self.X_normalised, self.Y_normalised, Xs0, Ys0, x, True)

#             print("---- DIAGNOSTICS at x=0 ----")
#             print(f"k_ss = {diag0['k_ss'].ravel()[0]:.3e} | base_cov = {diag0['base_cov'].ravel()[0]:.3e} | infl = {diag0['infl'].ravel()[0]:.3e}")

#             out.update({"mu": self.nY * mu0, "var": (self.nY**2) * cov0, "mu_cv": mu_cv, "var_cv": var_cv})
#         return out

#     # ────────────────────────────────────────────────────────────────
#     # VII. OPTIMISATION (L-BFGS)
#     # ────────────────────────────────────────────────────────────────

#     def perform_extrapolation_optimization(self, A):
#         A = torch.as_tensor(A, dtype=torch.float64)
#         default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)
#         starts = [default_params]
#         for _ in range(DEFAULT_NUM_RESTARTS):
#             starts.append(default_params * torch.exp(torch.randn_like(default_params) * 0.5))

#         best_cv, best_x = -float("inf"), default_params
#         for s in starts:
#             try:
#                 x_opt, cv_val = self._optimize_torch(s, A)
#                 if cv_val > best_cv: # 现在 cv_loss 返回的是似然，越大越好
#                     best_cv, best_x = cv_val, x_opt
#             except Exception: continue
#         return {"x": best_x, "cv": torch.tensor(best_cv, dtype=torch.float64)}

#     def _optimize_torch(self, init_x, A):
#         x_param = init_x.clone().detach().requires_grad_(True)
#         optimizer = torch.optim.LBFGS([x_param], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
#         def closure():
#             optimizer.zero_grad()
#             loss = -self.cv_loss(x_param, A) # 最小化负对数似然
#             if loss.requires_grad: loss.backward()
#             return loss
#         optimizer.step(closure)
#         return x_param.detach(), -closure().item()

#     # ────────────────────────────────────────────────────────────────
#     # VIII. STEPWISE SELECTION
#     # ────────────────────────────────────────────────────────────────

#     def stepwise_selection(self):
#         n_train = self.X_normalised.shape[0]
#         A = torch.zeros((1, self.dimension), dtype=torch.int64)
#         if self.kernel_base is not None: return self._GRE_stepwise_selection(A)

#         order = 0
#         fit = self.perform_extrapolation_optimization(A)
#         cv = fit["cv"]

#         carry_on = True
#         while carry_on:
#             order += 1
#             A_extra = stepwise(A, order)
#             to_include = torch.zeros(A_extra.shape[0], dtype=torch.bool)
#             print(f"Fitting interactions of order {order}:")
#             for i in tqdm(range(A_extra.shape[0]), desc="Stepwise"):
#                 A_new = torch.cat([A, A_extra[i: i + 1]], dim=0)
#                 if self.check_unisolvent(A_new) > 0:
#                     if self.perform_extrapolation_optimization(A_new)["cv"] > cv:
#                         to_include[i] = True

#             if to_include.any():
#                 A_updated = torch.cat([A, A_extra[to_include]], dim=0)
#                 fit_updated = self.perform_extrapolation_optimization(A_updated)
#                 if fit_updated["cv"] > cv:
#                     A, cv, fit = A_updated, fit_updated["cv"], fit_updated
#                 else: carry_on = False
#             else: carry_on = False

#         out = self.perform_extrapolation(fit["x"], A, True)
#         out.update({"A": A, "x": fit["x"]})
#         return out

#     def _GRE_stepwise_selection(self, A):
#         B = torch.zeros((1, self.dimension), dtype=torch.int64)
#         order = 0
#         self.set_kernel_spec(self.kernel_base, B)
#         fit = self.perform_extrapolation_optimization(A)
#         cv = fit["cv"]

#         carry_on = True
#         while carry_on:
#             order += 1
#             B_extra = stepwise(B, order)
#             if B_extra.shape[0] == 0: break
#             to_include = torch.zeros(B_extra.shape[0], dtype=torch.bool)
#             for i in tqdm(range(B_extra.shape[0]), desc="GRE Stepwise"):
#                 B_new = torch.cat([B, B_extra[i: i + 1]], dim=0)
#                 self.set_kernel_spec(self.kernel_base, B_new)
#                 if self.perform_extrapolation_optimization(A)["cv"] > cv: to_include[i] = True
#             if to_include.any():
#                 B_updated = torch.cat([B, B_extra[to_include]], dim=0)
#                 self.set_kernel_spec(self.kernel_base, B_updated)
#                 fit_updated = self.perform_extrapolation_optimization(A)
#                 if fit_updated["cv"] > cv: B, cv, fit = B_updated, fit_updated["cv"], fit_updated
#                 else: carry_on = False
#             else: carry_on = False
        
#         self.set_kernel_spec(self.kernel_base, B)
#         out = self.perform_extrapolation(fit["x"], A, True)
#         out.update({"A": A, "x": fit["x"], "B": B})
#         return out






































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
        设置 kernel 的初始超参数（与 JAX 逻辑一致）。
        """
        match self.kernel_spec:
            case "Gaussian":
                # [amp, lengthscale]
                self.default_kernel_parameters = [1.0, 0.1]
            case "GaussianARD":
                # [amp, l1, l2, ..., ld]
                params = [1.0]
                params.extend([0.1] * self.dimension)
                self.default_kernel_parameters = params
            case "white":
                self.default_kernel_parameters = [1.0]
            case "Matern1/2":
                # [amp, lengthscale]
                self.default_kernel_parameters = [1.0, 1.0]
            case "Matern3/2":
                self.default_kernel_parameters = [1.0, 1.0]
            case "GRE":
                # 先为 base kernel 设置 default，然后在前面加上一个 GRE amp
                base_spec = self.kernel_base
                self.kernel_spec = base_spec
                self.set_kernel_default_parameters()
                base_defaults = self.default_kernel_parameters
                self.kernel_spec = "GRE"
                self.kernel_base = base_spec

                self.default_kernel_parameters = [1.0]
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

    def set_normalised_data(self, X, Y):
        """
        Normalize data using max-min scaling with epsilon-jitter.

        Normalization Formula
        --------------------
        For inputs X and outputs Y, compute scale factors:
            σ_X = max(X) - min(X) + ε
            σ_Y = max(Y) - min(Y) + ε

        Then normalize:
            X_norm = X / σ_X
            Y_norm = Y / σ_Y

        The epsilon term prevents division by zero for constant data.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Input design points
        Y : array-like, shape (n,)
            Output responses

        Notes
        -----
        Normalization improves numerical conditioning for GP inference and
        makes lengthscale hyperparameters more interpretable.
        """
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64).flatten()

        # Compute normalization scale factors (max-min with ε-jitter)
        self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
        self.nY = (Y.max() - Y.min()) + EPSILON

        # Apply normalization
        self.X_normalised = X / self.nX
        self.Y_normalised = Y / self.nY

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

        if return_mu_and_var:
            n_train = self.X_normalised.shape[0]
            mu_cv = torch.zeros(n_train, dtype=torch.float64)
            var_cv = torch.zeros(n_train, dtype=torch.float64)

            for i in range(n_train):
                mu_val, cov_val = self.cv_local_loss(x, A, i, return_mu_cov=True)
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

    def objective(self, x_np: np.ndarray, A: np.ndarray) -> float:
        """
        SciPy 用的 objective：返回负的 cv（因为要最小化）。
        """
        x_t = torch.as_tensor(x_np, dtype=torch.float64)
        A_t = torch.as_tensor(A, dtype=torch.float64)
        val = self.cv_loss(x_t, A_t)
        return float((-val).detach().cpu().numpy())


    def perform_extrapolation_optimization(self, A: torch.Tensor, do_jit: bool = True) -> dict:

        A = torch.as_tensor(A, dtype=torch.float64)

        # ───────────────────────────────────────────────────────────────────
        # Step 1: Generate multi-start initial points
        # ───────────────────────────────────────────────────────────────────
        default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)

        # Create list of starting points: default + random perturbations
        num_restarts = DEFAULT_NUM_RESTARTS
        starting_points = [default_params]
        for _ in range(num_restarts):
            # Perturb in log-space to maintain positivity
            noise = torch.randn_like(default_params) * 0.5
            perturbed = default_params * torch.exp(noise)
            starting_points.append(perturbed)

        best_cv = float('inf')
        best_x = default_params

        # ───────────────────────────────────────────────────────────────────
        # Step 2: Optimize from each starting point
        # ───────────────────────────────────────────────────────────────────
        for start_x in starting_points:
            try:
                x_opt, cv_val = self._optimize_torch(start_x, A)

                # Track best result across all restarts
                if cv_val < best_cv:
                    best_cv = cv_val
                    best_x = x_opt
            except Exception as e:
                # Skip starting points that cause numerical instability
                continue

        return {"x": best_x, "cv": torch.tensor(best_cv, dtype=torch.float64)}


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
            
            # 1. 计算原始 Log-Likelihood (LL)
            # LL 越大代表模型越好
            raw_ll = self.cv_loss(x_param, A)
            amp = EPSILON + torch.nn.functional.softplus(x_param[0])
            prior_mean = -2.0  # log(0.13) 左右
            prior_std = 2.0
            log_amp = torch.log(amp)
            prior_penalty = 0.5 * ((log_amp - prior_mean) / prior_std)**2
            # 2. 关键修复：取负号！
            # Loss = -LL
            # 因为 L-BFGS 会尝试把 Loss 变得越小越好，
            # 这等价于把 LL 变得越大越好。
            loss = -raw_ll + prior_penalty
            
            # 3. 反向传播
            if loss.requires_grad:
                loss.backward()
            
            return loss

        # 执行优化
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
