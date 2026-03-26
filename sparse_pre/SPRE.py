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


EPSILON = 1e-10    
JITTER = 1e-12           
DEFAULT_NUM_RESTARTS = 10  


class SPRE:
    def __init__(self, kernel_spec: str, dimension: int, gre_base: torch.Tensor | None = None):

        self.dimension = dimension

        self.set_kernel_spec(kernel_spec, gre_base)

        self.X_normalised: torch.Tensor | None = None  # Normalized design points
        self.Y_normalised: torch.Tensor | None = None  # Normalized responses
        self.nX: torch.Tensor | None = None  # Normalization scale factors per dimension
        self.nY: torch.Tensor | None = None  # Normalization scale factor for response



    def cdist_torch(self, XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:

        XA = XA.to(torch.float64)
        XB = XB.to(torch.float64)

        XA_sq = (XA ** 2).sum(dim=1, keepdim=True)  # (m,1)
        XB_sq = (XB ** 2).sum(dim=1)                # (n,)
        cross = XA @ XB.T                           # (m,n)

        nums = XA_sq - 2 * cross + XB_sq           # (m,n)
        nums = torch.clamp(nums, min=0.0)
        return torch.sqrt(nums)

    def set_kernel_spec(self, kernel_spec: str, gre_base: torch.Tensor | None = None):
 
        if gre_base is None:
            self.kernel_spec = kernel_spec
            self.kernel_base = None
        else:
            self.kernel_spec = "GRE"
            self.kernel_base = kernel_spec

        self.gre_base = gre_base  # B 矩阵（或 None）
        self.set_kernel_default_parameters()

    def set_kernel_default_parameters(self):

        AMP_RAW_FOR_SIGMA_1 = 0.5413

        match self.kernel_spec:
            case "Gaussian":
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1, 0.1]
            case "GaussianARD":
                params = [AMP_RAW_FOR_SIGMA_1]
                params.extend([0.1] * self.dimension)
                self.default_kernel_parameters = params
            case "white":
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1]
            case "Matern1/2":
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1, 1.0]
            case "Matern3/2":
                self.default_kernel_parameters = [AMP_RAW_FOR_SIGMA_1, 1.0]
            case "GRE":
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


    def set_normalised_data(self, X, Y, use_mad: bool = False):

        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64).flatten()

        self.nX = (X.max(dim=0).values - X.min(dim=0).values) + EPSILON
        self.X_normalised = X / self.nX

        if use_mad:
            Y_mean = Y.mean()
            Y_median = Y.median()
            mad = torch.median(torch.abs(Y - Y_median))
            self.nY = mad + EPSILON  # scale factor is MAD
            self.Y_mean = Y_mean  # store mean for de-normalization
            self.Y_normalised = (Y - Y_mean) / self.nY
            self._use_mad = True
        else:
            self.nY = (Y.max() - Y.min()) + EPSILON
            self.Y_mean = torch.tensor(0.0, dtype=torch.float64)  # no centering
            self.Y_normalised = Y / self.nY
            self._use_mad = False

    def set_normalised_data_mad(self, X, Y):

        self.set_normalised_data(X, Y, use_mad=True)


    def extract_hyperparameters(self, x: torch.Tensor) -> dict:

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
            result['amplitude'] = float(EPSILON + softplus(x[1]))
            if len(x) > 2:
                result['lengthscale'] = float(EPSILON + softplus(x[2]))
            else:
                result['lengthscale'] = None

        else:
            raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

        return result

    def hyperparams_to_raw(self, amplitude: float, lengthscale: float | list) -> torch.Tensor:

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
            ell_raw = softplus_inv(lengthscale)
            return torch.tensor([amp_raw, ell_raw], dtype=torch.float64)


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


    def mle_loss(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:

        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        X = self.X_normalised
        Y = self.Y_normalised.view(-1, 1)  # (n, 1)
        n = X.shape[0]
        m = A.shape[0]

        K = self.kernel(X, X, x)
        K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * JITTER

        try:
            L_K = torch.linalg.cholesky(K_reg)
        except RuntimeError:
            K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * 1e-6
            L_K = torch.linalg.cholesky(K_reg)

        log_det_K = 2.0 * torch.log(torch.diag(L_K)).sum()

        K_inv_Y = torch.cholesky_solve(Y, L_K)  # (n, 1)
        V = x2fx(X, A)  # (n, m)
        K_inv_V = torch.cholesky_solve(V, L_K)  # (n, m)

        VtKinvV = V.T @ K_inv_V  # (m, m)

        jitter_vkv = max(JITTER, 1e-8 * torch.trace(VtKinvV).abs().item() / m) if m > 0 else JITTER
        VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * jitter_vkv

        try:
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)
        except RuntimeError:
            VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * 1e-4
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)

        log_det_VKV = 2.0 * torch.log(torch.diag(L_VKV)).sum()

        VtKinvY = V.T @ K_inv_Y  # (m, 1)
        VKV_inv_VKY = torch.cholesky_solve(VtKinvY, L_VKV)  # (m, 1)

        ytKinvy = (Y.T @ K_inv_Y).squeeze()  # scalar
        correction = (VtKinvY.T @ VKV_inv_VKY).squeeze()  # scalar
        ytPy = ytKinvy - correction

        log_ml = -0.5 * (n - m) * math.log(2 * math.pi) - 0.5 * log_det_K \
                 - 0.5 * log_det_VKV - 0.5 * ytPy

        return log_ml

    def compute_sigma_mle(self, x_no_amp: torch.Tensor, A: torch.Tensor) -> torch.Tensor:

        X = self.X_normalised
        Y = self.Y_normalised.view(-1, 1)
        n = X.shape[0]
        m = A.shape[0]

        amp_raw_for_unit = math.log(math.exp(1.0) - 1.0)
        x_unit = torch.cat([torch.tensor([amp_raw_for_unit], dtype=torch.float64), x_no_amp])

        K = self.kernel(X, X, x_unit)  # K with amp=1
        K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * JITTER

        try:
            L_K = torch.linalg.cholesky(K_reg)
        except RuntimeError:
            K_reg = K + torch.eye(n, device=X.device, dtype=torch.float64) * 1e-6
            L_K = torch.linalg.cholesky(K_reg)

        K_inv_Y = torch.cholesky_solve(Y, L_K)  # (n, 1)
        V = x2fx(X, A)  # (n, m)
        K_inv_V = torch.cholesky_solve(V, L_K)  # (n, m)

        VtKinvV = V.T @ K_inv_V  # (m, m)

        jitter_vkv = max(JITTER, 1e-8 * torch.trace(VtKinvV).abs().item() / m) if m > 0 else JITTER
        VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * jitter_vkv

        try:
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)
        except RuntimeError:
            VtKinvV_reg = VtKinvV + torch.eye(m, device=X.device, dtype=torch.float64) * 1e-4
            L_VKV = torch.linalg.cholesky(VtKinvV_reg)

        VtKinvY = V.T @ K_inv_Y  # (m, 1)
        VKV_inv_VKY = torch.cholesky_solve(VtKinvY, L_VKV)  # (m, 1)

        ytKinvy = (Y.T @ K_inv_Y).squeeze()
        correction = (VtKinvY.T @ VKV_inv_VKY).squeeze()
        ytPy = ytKinvy - correction

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

        x_no_amp = init_x[1:].clone().detach().requires_grad_(True)

        if x_no_amp.numel() == 0:
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

            with torch.no_grad():
                sigma_sq = self.compute_sigma_mle(x_no_amp.detach(), A)

            amp_raw = torch.log(torch.exp(sigma_sq) - 1.0 + EPSILON)

            x_full = torch.cat([amp_raw.unsqueeze(0), x_no_amp])

            log_ml = self.mle_loss(x_full, A)
            loss = -log_ml

            if loss.requires_grad:
                loss.backward()

            return loss

        optimizer.step(closure)

        with torch.no_grad():
            sigma_sq = self.compute_sigma_mle(x_no_amp.detach(), A)
            amp_raw = torch.log(torch.exp(sigma_sq) - 1.0 + EPSILON)
            x_opt = torch.cat([amp_raw.unsqueeze(0), x_no_amp.detach()])
            final_mle = self.mle_loss(x_opt, A)

        return x_opt, final_mle.item()


    def perform_extrapolation(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        return_mu_and_var: bool = False,
    ) -> dict:

        x = torch.as_tensor(x, dtype=torch.float64)
        A = torch.as_tensor(A, dtype=torch.float64)

        cv = self.cv_loss(x, A)
        out = {"cv": cv}

        use_mad = getattr(self, '_use_mad', False)
        Y_mean = getattr(self, 'Y_mean', torch.tensor(0.0, dtype=torch.float64))

        if return_mu_and_var:
            n_train = self.X_normalised.shape[0]
            mu_cv = torch.zeros(n_train, dtype=torch.float64)
            var_cv = torch.zeros(n_train, dtype=torch.float64)

            for i in range(n_train):
                mu_val, cov_val = self.cv_local_loss(x, A, i, return_mu_cov=True)
                if use_mad:
                    mu_cv[i] = self.nY * mu_val[0, 0] + Y_mean
                else:
                    mu_cv[i] = self.nY * mu_val[0, 0]
                var_cv[i] = (self.nY ** 2) * cov_val[0, 0]

            Xs0 = torch.zeros((1, self.dimension), dtype=torch.float64)
            Ys0 = torch.zeros((1,), dtype=torch.float64)  
            mu_val0, cov_val0 = self.cv_loss_calculation(
                A,
                self.X_normalised,
                self.Y_normalised,
                Xs0,
                Ys0,
                x,
                return_mu_cov=True,
            )

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
      
        x_raw = self.hyperparams_to_raw(amplitude, lengthscale)

        out = self.perform_extrapolation(x_raw, A, return_mu_and_var=return_mu_and_var)

        out['x'] = x_raw
        out['amplitude'] = amplitude
        out['lengthscale'] = lengthscale

        return out

    def objective(self, x_np: np.ndarray, A: np.ndarray) -> float:
      
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

        gen = torch.Generator()
        gen.manual_seed(int(seed))

        starting_points = [x_base.clone()]
        for _ in range(int(num_restarts)):
            noise = torch.randn(x_base.shape, dtype=x_base.dtype, generator=gen) * float(restart_scale)
            starting_points.append((x_base + noise).clone())

        best_loss = float("inf")   # minimise -LL
        best_x = x_base.clone()

        for start_x in starting_points:
            try:
                x_opt, loss_val = self._optimize_torch(start_x, A)
                if float(loss_val) < best_loss:
                    best_loss = float(loss_val)
                    best_x = x_opt
            except Exception as e:
                print(f"[WARNING] Optimization failed for starting point: {e}")
                continue

        if best_loss == float("inf"):
            raise RuntimeError("All optimization starting points failed. Check input data or parameters.")

        return {"x": best_x, "cv": torch.tensor(best_loss, dtype=torch.float64)}









    def _optimize_torch(self, init_x: torch.Tensor, A: torch.Tensor):

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
            loss = -raw_ll
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

        do_jit = False  

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
                    carry_on = False 
                else:
                    A = A_updated
                    fit = fit_updated
                    cv = cv_updated
            else:
                carry_on = False  
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
