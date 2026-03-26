





































        



























            




































































        

            
            
            
            

        
        



































































































































































        


        


        
        











        





















        






































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

        self.set_kernel_spec(kernel_spec, gre_base)

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
                self.default_kernel_parameters = [1.0, 0.1]
            case "GaussianARD":
                params = [1.0]
                params.extend([0.1] * self.dimension)
                self.default_kernel_parameters = params
            case "white":
                self.default_kernel_parameters = [1.0]
            case "Matern1/2":
                self.default_kernel_parameters = [1.0, 1.0]
            case "Matern3/2":
                self.default_kernel_parameters = [1.0, 1.0]
            case "GRE":
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
            amp = EPSILON + softplus(x[0])
            ell = EPSILON + softplus(x[1])
            dist = self.cdist_torch(X1, X2)
            return amp * torch.exp(-(dist ** 2) / (ell ** 2))

        elif spec == "GaussianARD":
            amp = EPSILON + softplus(x[0])
            length_terms = []
            for i in range(self.dimension):
                d_i = self.cdist_torch(
                    X1[:, [i]], X2[:, [i]]
                ) ** 2 / (EPSILON + softplus(x[i + 1])) ** 2
                length_terms.append(d_i)
            sq_dist = cellsum(length_terms)
            return amp * torch.exp(-sq_dist)

        elif spec == "white":
            amp = EPSILON + softplus(x[0])
            return amp * white(X1, X2)

        elif spec == "Matern1/2":
            amp = EPSILON + softplus(x[0])
            ell = EPSILON + softplus(x[1])
            dist = self.cdist_torch(X1, X2)
            return amp * torch.exp(-dist / ell)

        elif spec == "Matern3/2":
            amp = EPSILON + softplus(x[0])
            ell = EPSILON + softplus(x[1])
            dist = self.cdist_torch(X1, X2)
            r_l = dist / ell
            sqrt3_r_l = math.sqrt(3.0) * r_l
            return amp * (1.0 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)

        elif spec == "GRE":
            if self.gre_base is None:
                raise RuntimeError("GRE kernel requires gre_base (B matrix).")
            amp_gre = EPSILON + softplus(x[0])

            base_X1 = x2fx(X1, self.gre_base).sum(dim=1)  # (n1,)
            base_X2 = x2fx(X2, self.gre_base).sum(dim=1)  # (n2,)

            base_spec = self.kernel_base
            old_spec = self.kernel_spec
            self.kernel_spec = base_spec
            K_base = self.kernel(X1, X2, x=x[1:])  # Use base kernel hyperparameters
            self.kernel_spec = old_spec

            return amp_gre * base_X1.unsqueeze(1) * K_base * base_X2.unsqueeze(0)

        else:
            raise ValueError(f"Unknown kernel specification: {spec}")


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

        self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
        self.nY = (Y.max() - Y.min()) + EPSILON

        self.X_normalised = X / self.nX
        self.Y_normalised = Y / self.nY


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

        K = self.kernel(X, X, x)
        K_inv = torch.linalg.inv(K)

        kernel_Xs_Xs = self.kernel(Xs, Xs, x)
        kernel_X_Xs = self.kernel(X, Xs, x)

        if self.kernel_spec != "GRE":
            kernel_Xs_X = kernel_X_Xs.T
        else:
            kernel_Xs_X = self.kernel(Xs, X, x)

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

        diff = Ys.view(-1, 1) - mu_val.view(1, 1)
        inv_cov = torch.linalg.inv(cov_val)
        term1 = -0.5 * torch.log(torch.det(2 * math.pi * cov_val))
        term2 = -0.5 * (diff.T @ inv_cov @ diff)
        return (term1 + term2).view(())

    def cv_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
 
        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        X = self.X_normalised
        Y = self.Y_normalised.view(-1, 1)  # (n, 1)
        n = X.shape[0]
        m = A.shape[0]  # Number of polynomial basis terms
        K = self.kernel(X, X, x)
        V = x2fx(X, A)
        M_top = torch.cat([K, V], dim=1)  # (n, n+m)
        zeros_m = torch.zeros((m, m), dtype=torch.float64, device=X.device)
        M_bot = torch.cat([V.T, zeros_m], dim=1)  # (m, n+m)
        M = torch.cat([M_top, M_bot], dim=0)  # (n+m, n+m)

        M_inv = torch.linalg.inv(M + torch.eye(n + m, device=X.device) * JITTER)

        zeros_y = torch.zeros((m, 1), dtype=torch.float64, device=X.device)
        Y_aug = torch.cat([Y, zeros_y], dim=0)  # (n+m, 1)

        alpha = (M_inv @ Y_aug)[:n]  # (n, 1)
        diag_inv = torch.diagonal(M_inv)[:n]  # (n,)

        residuals = alpha.flatten() / diag_inv

        variances = 1.0 / diag_inv


        variances = torch.clamp(variances, min=1e-14)

        term1 = -0.5 * torch.log(2 * math.pi * variances)  # Normalization term
        term2 = -0.5 * (residuals ** 2) / variances  # Squared error term

        total_ll = (term1 + term2).sum()

        return total_ll


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

        default_params = torch.tensor(self.default_kernel_parameters, dtype=torch.float64)

        num_restarts = DEFAULT_NUM_RESTARTS
        starting_points = [default_params]
        for _ in range(num_restarts):
            noise = torch.randn_like(default_params) * 0.5
            perturbed = default_params * torch.exp(noise)
            starting_points.append(perturbed)

        best_cv = float('inf')
        best_x = default_params

        for start_x in starting_points:
            try:
                x_opt, cv_val = self._optimize_torch(start_x, A)

                if cv_val < best_cv:
                    best_cv = cv_val
                    best_x = x_opt
            except Exception as e:
                continue

        return {"x": best_x, "cv": torch.tensor(best_cv, dtype=torch.float64)}


    def _optimize_torch(self, init_x: torch.Tensor, A: torch.Tensor):
        """
        内部辅助函数：使用 L-BFGS 优化单个起点
        修复核心逻辑：最小化 Negative Log-Likelihood
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
            
            raw_ll = self.cv_loss(x_param, A)
            amp = EPSILON + torch.nn.functional.softplus(x_param[0])
            prior_mean = -2.0  # log(0.13) 左右
            prior_std = 2.0
            log_amp = torch.log(amp)
            prior_penalty = 0.5 * ((log_amp - prior_mean) / prior_std)**2
            loss = -raw_ll + prior_penalty
            
            if loss.requires_grad:
                loss.backward()
            
            return loss

        optimizer.step(closure)
        
        final_ll = self.cv_loss(x_param, A)

        return x_param.detach(), (-final_ll).item()

    def stepwise_selection(self) -> dict:

        n_train = self.X_normalised.shape[0]

        A = torch.zeros((1, self.dimension), dtype=torch.int64)

        if self.kernel_base is not None:
            return self._GRE_stepwise_selection(A)

        do_jit = False  # Legacy parameter (no longer used)

        order = 0
        fit = self.perform_extrapolation_optimization(A, do_jit)
        cv = fit["cv"]

        carry_on = True

        while carry_on:
            m = A.shape[0]
            order += 1

            A_extra = stepwise(A, order)
            n_extra = A_extra.shape[0]
            to_include = torch.zeros(n_extra, dtype=torch.bool)

            print(f"Fitting interactions of order {order}:")

            for i in tqdm(range(n_extra), desc="Stepwise progress"):
                A_new = torch.cat([A, A_extra[i : i + 1, :]], dim=0)

                if self.check_unisolvent(A_new) > 0:
                    fit_new = self.perform_extrapolation_optimization(A_new, do_jit)
                    cv_new = fit_new["cv"]

                    if cv_new < cv:
                        to_include[i] = True

            if to_include.any() and (m + int(to_include.sum().item()) < (n_train - 1)):
                A_updated = torch.cat([A, A_extra[to_include, :]], dim=0)
                fit_updated = self.perform_extrapolation_optimization(A_updated, do_jit)
                cv_updated = fit_updated["cv"]

                if cv_updated >= cv:
                    carry_on = False  # No improvement, terminate
                else:
                    A = A_updated
                    fit = fit_updated
                    cv = cv_updated
            else:
                carry_on = False  # No terms accepted or size limit reached

        x_opt = fit["x"]
        out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)
        return out

    def _GRE_stepwise_selection(self, A: torch.Tensor) -> dict:
        """
        Stepwise selection for GRE (Gauss-Richardson Extrapolation).

        A: basis for mean (fixed in GRE)
        We add terms to B (rate function) stored in gre_base.
        """
        B = torch.zeros((1, self.dimension), dtype=torch.int64)

        order = 0
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

        x_opt = fit["x"]
        self.set_kernel_spec(self.kernel_base, B)
        out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)
        out["B"] = B
        return out
