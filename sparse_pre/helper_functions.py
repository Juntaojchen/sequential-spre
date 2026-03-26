
import torch

def cellsum(arrays):
   
    if isinstance(arrays, torch.Tensor):
        if arrays.numel() == 0:
            return torch.tensor(0.0, dtype=arrays.dtype, device=arrays.device)
        if arrays.dim() > 1:
            return arrays.sum(dim=0)
        return arrays.sum()

    if not arrays:
        return torch.tensor(0.0, dtype=torch.float64)

    out = arrays[0].clone()
    for arr in arrays[1:]:
        out = out + arr
    return out


def remove_row(tensor, index):
  
    if tensor.shape[0] <= 1:
        return torch.empty((0, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor[:index], tensor[index+1:]], dim=0)


def softplus(x):

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64)
    return torch.nn.functional.softplus(x)


def stepwise(A, order):
    
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.int64)

    if A.numel() == 0:
        return torch.empty((0, A.shape[1] if A.dim() == 2 else 0),
                           dtype=torch.int64, device=A.device)

    n_models, d = A.shape
    device = A.device

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

    if not torch.is_tensor(X1):
        X1 = torch.tensor(X1)
    if not torch.is_tensor(X2):
        X2 = torch.tensor(X2, device=X1.device, dtype=X1.dtype)

    matches = (X1.unsqueeze(1) == X2.unsqueeze(0)).all(dim=-1)
    return matches.to(dtype=X1.dtype)


def x2fx(X, A):

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float64)
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float64, device=X.device)

    base = X.unsqueeze(1)              # (n, 1, d)
    exp  = A.unsqueeze(0).to(X.dtype)  # (1, m, d)
    term_values = torch.pow(base, exp) # (n, m, d)
    return torch.prod(term_values, dim=-1)  # (n, m)
