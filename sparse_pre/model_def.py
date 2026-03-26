# import torch
# import gpytorch
# from .helper_functions import x2fx

# class PolynomialMean(gpytorch.means.Mean):
#     """
#     自定义均值函数: mu(x) = V(x) * weights

#     """
#     def __init__(self, A, input_dim):
#         super().__init__()
#         # 将 A 注册为 buffer (它是模型状态的一部分，但不参与梯度更新)
#         self.register_buffer("A", A)
        
#         # weights 对应原代码中的 beta 系数
#         # 我们将其初始化为可学习的参数，让优化器自动寻找最优解
#         m = A.shape[0]
#         self.weights = torch.nn.Parameter(torch.zeros(m, 1))

#     def forward(self, x):
#         # x shape: (batch_size, n, d) or (n, d)
        
#         # 处理 GPyTorch 的 batch 模式
#         if x.dim() == 3:
#             res = []
#             for i in range(x.size(0)):
#                 V = x2fx(x[i], self.A) # (n, m)
#                 res.append(V @ self.weights) # (n, 1)
#             return torch.stack(res).squeeze(-1)
#         else:
#             V = x2fx(x, self.A)
#             return (V @ self.weights).squeeze(-1)

# class GREKernel(gpytorch.kernels.Kernel):
#     """
#     GRE 专用核函数: k(x1, x2) = b(x1) * k_base(x1, x2) * b(x2)
#     其中 b(x) 是由 B 决定的多项式基函数。
#     """
#     def __init__(self, base_kernel, B, **kwargs):
#         super().__init__(**kwargs)
#         self.base_kernel = base_kernel
#         self.register_buffer("B", B) 

#     def forward(self, x1, x2, diag=False, **params):
#         # 辅助函数：计算缩放因子 b(x)
#         # 逻辑：base(X) = sum(x2fx(X, B), axis=1)
#         def compute_b(x):
#             if x.dim() == 3: # (batch, n, d)
#                 # 展平处理再变回来
#                 basis = x2fx(x.reshape(-1, x.size(-1)), self.B)
#                 return basis.sum(dim=-1).reshape(x.size(0), x.size(1))
#             else:
#                 return x2fx(x, self.B).sum(dim=-1)

#         b_x1 = compute_b(x1) # (n1,)
#         b_x2 = compute_b(x2) # (n2,)
        
#         # 计算基础核矩阵
#         covar = self.base_kernel(x1, x2, diag=diag, **params)
        
#         # 应用缩放: b(x1) * Cov * b(x2)^T
#         if diag:
#             return b_x1 * covar * b_x1
#         else:
#             # 利用广播机制进行外积乘法
#             # unsqueeze(-1) 变成列向量, unsqueeze(-2) 变成行向量
#             return b_x1.unsqueeze(-1) * covar * b_x2.unsqueeze(-2)

# class SPREModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, A, k_name="Matern1/2", gre_base=None):
#         super(SPREModel, self).__init__(train_x, train_y, likelihood)
        
#         d = train_x.shape[-1]
        
#         # 1. 设置均值模块 (Richardson Extrapolation 的多项式部分)
#         self.mean_module = PolynomialMean(A, input_dim=d)
        
#         # 2. 设置协方差模块 (Kernel)
#         if k_name == "Gaussian" or k_name == "RBF":
#             base = gpytorch.kernels.RBFKernel()
#         elif k_name == "GaussianARD":
#             base = gpytorch.kernels.RBFKernel(ard_num_dims=d)
#         elif k_name == "Matern1/2":
#             base = gpytorch.kernels.MaternKernel(nu=0.5)
#         elif k_name == "Matern3/2":
#             base = gpytorch.kernels.MaternKernel(nu=1.5)
#         elif k_name == "white":
#             # GPyTorch 通常在 Likelihood 层处理白噪声
#             # 这里用一个极短长度尺度的 RBF 作为占位
#             base = gpytorch.kernels.RBFKernel()
#             base.lengthscale = 1e-4 
#         else:
#             raise ValueError(f"Unknown kernel: {k_name}")
            
#         # 3. 如果是 GRE 模式 (提供了 gre_base)，则包裹一层 GREKernel
#         if gre_base is not None:
#             # ScaleKernel 提供 amplitude (sigma_f^2)
#             self.covar_module = gpytorch.kernels.ScaleKernel(
#                 GREKernel(base, gre_base)
#             )
#         else:
#             # 标准 SPRE 模式
#             self.covar_module = gpytorch.kernels.ScaleKernel(base)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
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
        # 注册 A 为 buffer，这样它会被保存到 state_dict 但不会被优化器更新
        self.register_buffer("A", A)
        
        # weights 初始化为 0，作为可学习参数
        m = A.shape[0]
        self.weights = torch.nn.Parameter(torch.zeros(m, 1, dtype=torch.float64))

    def forward(self, x):
        # x shape: (batch_size, n, d) 或 (n, d)
        # GPyTorch 的均值函数通常期望返回 shape (..., n) 而不是 (..., n, 1)
        
        # 1. 计算多项式特征 V(x)
        # x2fx 需要处理 batch 维度，如果 helper_functions.x2fx 不支持 batch，我们需要手动处理
        if x.dim() == 3:
            # Batch mode: (b, n, d)
            res = []
            for i in range(x.size(0)):
                V = x2fx(x[i], self.A) # (n, m)
                res.append(V @ self.weights) # (n, 1)
            # Stack -> (b, n, 1) -> squeeze -> (b, n)
            return torch.stack(res).squeeze(-1)
        else:
            # Single mode: (n, d)
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
        # 计算缩放因子 b(x) = sum(x^B)
        # x: (..., n, d)
        if x.dim() > 2: 
            # 处理 batch 维度
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
        
        # 计算基础核矩阵
        covar = self.base_kernel(x1, x2, diag=diag, **params)
        
        if diag:
            return b_x1 * covar * b_x1
        else:
            # 利用广播机制: (..., n1, 1) * (..., n1, n2) * (..., 1, n2)
            return b_x1.unsqueeze(-1) * covar * b_x2.unsqueeze(-2)

class SPREModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, A, k_name="Matern1/2", gre_base=None):
        super(SPREModel, self).__init__(train_x, train_y, likelihood)
        
        d = train_x.shape[-1]
        
        # 1. 均值模块
        self.mean_module = PolynomialMean(A, input_dim=d)
        
        # 2. 协方差模块 (Kernel)
        if k_name == "Gaussian" or k_name == "RBF":
            base = gpytorch.kernels.RBFKernel()
        elif k_name == "GaussianARD":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        elif k_name == "Matern1/2":
            base = gpytorch.kernels.MaternKernel(nu=0.5)
        elif k_name == "Matern3/2":
            base = gpytorch.kernels.MaternKernel(nu=1.5)
        elif k_name == "white":
            # 极短长度尺度的 RBF 模拟 White Noise 行为 (除了对角线外几乎为0)
            base = gpytorch.kernels.RBFKernel()
            base.lengthscale = 1e-4
        else:
            raise ValueError(f"Unknown kernel: {k_name}")
            
        # 3. 组合 ScaleKernel (Amplitude) 和 GRE Logic
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