
# ##gemini##
# import torch
# import numpy as np

# def cellsum(arrays: list) -> torch.Tensor:
#     """
#     Pointwise addition for a collection of arrays stored in a list.

#     Parameters:
#         arrays : list of torch.Tensor
#             List of tensors to sum element-wise.

#     Returns:
#         out : torch.Tensor
#             Element-wise sum of all tensors.
#     """
#     if not arrays:
#         raise ValueError("Input list 'arrays' cannot be empty.")
    
#     out = arrays[0]
#     for arr in arrays[1:]:
#         out = torch.add(out, arr)
#     return out

# def remove_row(arr: torch.Tensor, index: int) -> torch.Tensor:
#     """
#     Remove a row from a 2D Tensor.

#     Parameters:
#         arr : torch.Tensor
#             Input tensor.
#         index : int
#             Index of the row to remove.

#     Returns:
#         torch.Tensor
#             Tensor with the specified row removed.
#     """
#     # PyTorch does not have a direct 'delete' method like numpy/jax
#     # Use concatenation of slices instead
#     return torch.cat([arr[:index], arr[index+1:]])

# def softplus(x: float) -> torch.Tensor:
#     """
#     Computes softplus: log(1 + exp(x)).
    
#     Parameters:
#        x : float or torch.Tensor

#     Returns:
#        torch.Tensor
#     """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x)
#     return torch.nn.functional.softplus(x)

# def stepwise(A: torch.Tensor, order: int) -> torch.Tensor:
#     """
#     Compute which high-order interactions to consider next.

#     Parameters:
#         A : torch.Tensor
#             n_models x d tensor of current interactions.
#         order : int
#             current order to consider.

#     Returns:
#         torch.Tensor
#             tensor of new interactions to consider.
#     """
#     if not torch.is_tensor(A):
#         A = torch.tensor(A)
        
#     n_models, d = A.shape
    
#     # Mask for rows where the sum equals (order - 1)
#     mask = (A.sum(dim=1) == (order - 1))
#     A_filtered = A[mask]  # Shape: (k, d)

#     if A_filtered.shape[0] == 0:
#         return torch.empty((0, d), dtype=A.dtype, device=A.device)

#     # Create identity matrix
#     eye_d = torch.eye(d, dtype=A.dtype, device=A.device)  # Shape: (d, d)
    
#     # Expand dims for broadcasting: (k, 1, d) + (1, d, d) -> (k, d, d)
#     # This adds 1 to each column position for each row in A_filtered
#     expanded = A_filtered.unsqueeze(1) + eye_d.unsqueeze(0)

#     # Reshape to 2D (k*d, d)
#     candidates = expanded.reshape(-1, d)
    
#     # Remove duplicates
#     out = torch.unique(candidates, dim=0)
#     return out

# def white(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
#     """
#     White noise kernel.

#     Parameters:
#         X1 : torch.Tensor
#             of shape (n1, d)
#         X2 : torch.Tensor
#             of shape (n2, d)

#     Returns:
#         torch.Tensor
#             of shape (n1, n2)
#     """
#     if not torch.is_tensor(X1): X1 = torch.tensor(X1)
#     if not torch.is_tensor(X2): X2 = torch.tensor(X2)

#     # Compare all rows in X1 with all rows in X2 efficiently
#     # X1[:, None, :] shape -> (n1, 1, d)
#     # X2[None, :, :] shape -> (1, n2, d)
#     # Broadcasting equality gives (n1, n2, d)
#     # Check if all elements in last dim are equal -> (n1, n2)
#     matches = (X1[:, None, :] == X2[None, :, :]).all(dim=-1)
    
#     # Convert boolean to float (1.0 for match, 0.0 otherwise)
#     return matches.float()

# def x2fx(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
#     """
#     Generate polynomial basis terms for each row in X using exponents in A.
    
#     Parameters:
#         X : torch.Tensor
#             predictor matrix (n, d)
#         A : torch.Tensor (m, d)
#             powers to use for predictor variables

#     Returns:
#         torch.Tensor
#             (n, m) design matrix where V[i,j] = prod_k X[i,k]^A[j,k]
#     """
#     if not torch.is_tensor(X): X = torch.tensor(X)
#     if not torch.is_tensor(A): A = torch.tensor(A, device=X.device)

#     # X: (n, d) -> (n, 1, d)
#     # A: (m, d) -> (1, m, d)
#     X_unsqueezed = X.unsqueeze(1)
#     A_unsqueezed = A.unsqueeze(0)
    
#     # Power operation: X^A -> (n, m, d)
#     # Note: Ensure X and A are compatible types (float for X usually)
#     term = torch.pow(X_unsqueezed, A_unsqueezed)
    
#     # Product over the last dimension (d) -> (n, m)
#     V = torch.prod(term, dim=-1)
    
#     return V

# import torch

# def cellsum(arrays):
#     """
#     Sum of a list of arrays/tensors.
#     """
#     if isinstance(arrays, torch.Tensor):
#         if arrays.numel() == 0:
#             return torch.tensor(0.0, dtype=arrays.dtype, device=arrays.device)
#         # 如果是 stacked tensor (k, d)，求和 dim 0
#         if arrays.dim() > 1:
#             return torch.sum(arrays, dim=0)
#         return torch.sum(arrays)

#     if not arrays: # List check
#         return torch.tensor(0.0)
    
#     out = arrays[0].clone()
#     for arr in arrays[1:]:
#         out = out + arr
#     return out

# def remove_row(tensor, index):
#     """
#     Removes a row at a specific index. 
#     """
#     if tensor.shape[0] <= 1:
#         return torch.empty((0, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
#     return torch.cat([tensor[:index], tensor[index+1:]])

# def softplus(x):
#     """
#     Computes log(1 + exp(x))
#     """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float64)
#     return torch.nn.functional.softplus(x)

# def stepwise(A, order):
#     """
#     Generates stepwise terms based on order.
#     """
#     if A.numel() == 0:
#         # Handle empty case specifically to avoid dimensionality errors
#         # Assuming d=2 roughly based on context, or return empty
#         return torch.empty((0, 2), dtype=torch.int64)

#     k, d = A.shape
#     device = A.device
    
#     # Create identity matrix for expansion
#     eye_d = torch.eye(d, dtype=A.dtype, device=device)
    
#     # Expand: (k, 1, d) + (1, d, d) -> (k, d, d)
#     expanded = A.unsqueeze(1) + eye_d.unsqueeze(0)
    
#     # Reshape to (k*d, d)
#     candidates = expanded.view(-1, d)
    
#     # Filter based on order
#     sums = candidates.sum(dim=1)
#     mask = sums <= order
    
#     valid_candidates = candidates[mask]
    
#     if valid_candidates.numel() == 0:
#          return torch.empty((0, d), dtype=A.dtype, device=device)
    
#     # PyTorch unique
#     unique_candidates = torch.unique(valid_candidates, dim=0)
    
#     return unique_candidates

# def white(A, B):
#     """
#     Check for matching rows.
#     """
#     matches = (A.unsqueeze(1) == B.unsqueeze(0)).all(dim=-1).double()
#     return matches

# def x2fx(X, A):
#     """
#     Polynomial feature expansion. X^A
#     """
#     # X: (n, d) -> (n, 1, d)
#     # A: (k, d) -> (1, k, d)
#     # Result: (n, k)
    
#     # Numerical stability: 0^0 = 1
#     # We use exp(A * log(X)) approach or simpler power if X is safe
#     # Direct power is usually fine in float64
#     base = X.unsqueeze(1)
#     exp = A.unsqueeze(0).to(X.dtype)
    
#     term_values = torch.pow(base, exp)
#     return torch.prod(term_values, dim=-1)

import torch

def cellsum(arrays):
    """
    点对点相加：
      - 如果传入的是 2D tensor，行为等价于 jax 版本对行求和：arrays.sum(axis=0)
      - 如果传入的是 list[Tensor]，则逐个 tensor 做逐元素相加
    """
    # 情况 1：直接是一个 tensor（例如 test_cellsum 中的二维数组）
    if isinstance(arrays, torch.Tensor):
        if arrays.numel() == 0:
            return torch.tensor(0.0, dtype=arrays.dtype, device=arrays.device)
        if arrays.dim() > 1:
            return arrays.sum(dim=0)
        return arrays.sum()

    # 情况 2：是一个 list[Tensor]
    if not arrays:
        return torch.tensor(0.0, dtype=torch.float64)

    out = arrays[0].clone()
    for arr in arrays[1:]:
        out = out + arr
    return out


def remove_row(tensor, index):
    """
    删除第 index 行（axis=0）
    """
    if tensor.shape[0] <= 1:
        return torch.empty((0, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor[:index], tensor[index+1:]], dim=0)


def softplus(x):
    """
    计算 softplus(x) = log(1 + exp(x))
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64)
    return torch.nn.functional.softplus(x)


def stepwise(A, order):
    """
    计算下一阶可选的高阶交互项，行为严格对齐 JAX 版本。
    """
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.int64)

    if A.numel() == 0:
        return torch.empty((0, A.shape[1] if A.dim() == 2 else 0),
                           dtype=torch.int64, device=A.device)

    n_models, d = A.shape
    device = A.device

    # 只从 sum == order-1 的行扩展
    mask = (A.sum(dim=1) == (order - 1))
    A_filtered = A[mask]
    if A_filtered.shape[0] == 0:
        return torch.empty((0, d), dtype=A.dtype, device=device)

    eye_d = torch.eye(d, dtype=A.dtype, device=device)      # (d, d)
    expanded = A_filtered.unsqueeze(1) + eye_d.unsqueeze(0) # (k, d, d)
    candidates = expanded.reshape(-1, d)
    unique_candidates = torch.unique(candidates, dim=0)
    return unique_candidates


def white(X1, X2):
    """
    White kernel：相同的点给 1，不同的点给 0
    """
    if not torch.is_tensor(X1):
        X1 = torch.tensor(X1)
    if not torch.is_tensor(X2):
        X2 = torch.tensor(X2, device=X1.device, dtype=X1.dtype)

    matches = (X1.unsqueeze(1) == X2.unsqueeze(0)).all(dim=-1)
    return matches.to(dtype=X1.dtype)


def x2fx(X, A):
    """
    生成多项式基：V[i,j] = ∏_k X[i,k] ** A[j,k]
      X: (n, d)
      A: (m, d)
      V: (n, m)
    """
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float64)
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float64, device=X.device)

    base = X.unsqueeze(1)              # (n, 1, d)
    exp  = A.unsqueeze(0).to(X.dtype)  # (1, m, d)
    term_values = torch.pow(base, exp) # (n, m, d)
    return torch.prod(term_values, dim=-1)  # (n, m)
