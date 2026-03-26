

        

        



        
        

        
        
        
            

import torch
import gpytorch
from .helper_functions import x2fx

class PolynomialMean(gpytorch.means.Mean):
    """
    自定义均值函数: mu(x) = V(x) * weights
    用于模拟 Richardson Extrapolation 中的多项式部分。
    """
    def __init__(self, A, input_dim):
        super().__init__()
        self.register_buffer("A", A)
        
        m = A.shape[0]
        self.weights = torch.nn.Parameter(torch.zeros(m, 1, dtype=torch.float64))

    def forward(self, x):
        
        if x.dim() == 3:
            res = []
            for i in range(x.size(0)):
                V = x2fx(x[i], self.A) # (n, m)
                res.append(V @ self.weights) # (n, 1)
            return torch.stack(res).squeeze(-1)
        else:
            V = x2fx(x, self.A) # (n, m)
            return (V @ self.weights).squeeze(-1) # (n,)

class GREKernel(gpytorch.kernels.Kernel):
    """
    GRE 专用核函数: k(x1, x2) = b(x1) * k_base(x1, x2) * b(x2)
    """
    def __init__(self, base_kernel, B, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.register_buffer("B", B) 

    def _compute_b(self, x):
        if x.dim() > 2: 
            batch_shape = x.shape[:-1]
            d = x.shape[-1]
            x_flat = x.reshape(-1, d)
            basis = x2fx(x_flat, self.B) # (N_total, m_B)
            b_flat = basis.sum(dim=-1)   # (N_total,)
            return b_flat.reshape(*batch_shape)
        else:
            return x2fx(x, self.B).sum(dim=-1)

    def forward(self, x1, x2, diag=False, **params):
        b_x1 = self._compute_b(x1) # (..., n1)
        b_x2 = self._compute_b(x2) # (..., n2)
        
        covar = self.base_kernel(x1, x2, diag=diag, **params)
        
        if diag:
            return b_x1 * covar * b_x1
        else:
            return b_x1.unsqueeze(-1) * covar * b_x2.unsqueeze(-2)

class SPREModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, A, k_name="Matern1/2", gre_base=None):
        super(SPREModel, self).__init__(train_x, train_y, likelihood)
        
        d = train_x.shape[-1]
        
        self.mean_module = PolynomialMean(A, input_dim=d)
        
        if k_name == "Gaussian" or k_name == "RBF":
            base = gpytorch.kernels.RBFKernel()
        elif k_name == "GaussianARD":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        elif k_name == "Matern1/2":
            base = gpytorch.kernels.MaternKernel(nu=0.5)
        elif k_name == "Matern3/2":
            base = gpytorch.kernels.MaternKernel(nu=1.5)
        elif k_name == "white":
            base = gpytorch.kernels.RBFKernel()
            base.lengthscale = 1e-4
        else:
            raise ValueError(f"Unknown kernel: {k_name}")
            
        if gre_base is not None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                GREKernel(base, gre_base)
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)