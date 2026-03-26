"""
General-purpose tensor utilities used across SPRE modules.
"""

import torch


def cellsum(arrays) -> torch.Tensor:
    """
    Element-wise sum of a collection of tensors.

    Two calling conventions are supported to match the original JAX interface:

    * ``cellsum(list_of_tensors)`` — successive element-wise addition.
    * ``cellsum(2D_tensor)``       — equivalent to ``tensor.sum(dim=0)``.

    Parameters
    ----------
    arrays : list[torch.Tensor] or torch.Tensor

    Returns
    -------
    torch.Tensor
        Element-wise sum.
    """
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


def remove_row(tensor: torch.Tensor, index: int) -> torch.Tensor:
    """
    Remove a single row from a 2-D tensor.

    Parameters
    ----------
    tensor : torch.Tensor, shape (n, d)
    index  : int
        Zero-based row index to remove.

    Returns
    -------
    torch.Tensor, shape (n-1, d)
    """
    if tensor.shape[0] <= 1:
        return torch.empty((0, tensor.shape[1]),
                           dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor[:index], tensor[index + 1:]], dim=0)
