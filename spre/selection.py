

from typing import Callable, Tuple

import torch
from tqdm import tqdm

from .basis import stepwise, x2fx


def check_unisolvent(X_norm: torch.Tensor, A: torch.Tensor) -> bool:
    """
    Return True iff the design matrix V = x2fx(X_norm, A) has full column rank.

    This ensures the polynomial trend is identifiable given the training data.

    Parameters
    ----------
    X_norm : (n, d)  normalised training inputs.
    A      : (m, d)  multi-index set.
    """
    A  = torch.as_tensor(A, dtype=torch.float64)
    VA = x2fx(X_norm, A)                                 # (n, m)
    return int(torch.linalg.matrix_rank(VA).item()) == A.shape[0]


def stepwise_selection(
    optimise_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, float]],
    X_norm:      torch.Tensor,
    dimension:   int,
    n_train:     int,
) -> Tuple[torch.Tensor, dict]:

    A       = torch.zeros(1, dimension, dtype=torch.int64)
    x_opt, neg_ll = optimise_fn(A)
    fit     = {"x": x_opt, "cv": torch.tensor(neg_ll, dtype=torch.float64)}
    cv      = neg_ll                                       # lower is better

    order    = 0
    carry_on = True

    while carry_on:
        m       = A.shape[0]
        order  += 1
        A_extra = stepwise(A, order)                       # candidates of degree `order`
        n_extra = A_extra.shape[0]

        if n_extra == 0:
            break

        to_include = torch.zeros(n_extra, dtype=torch.bool)
        print(f"Fitting interactions of order {order}:")

        for i in tqdm(range(n_extra), desc="Stepwise progress"):
            A_new = torch.cat([A, A_extra[i : i + 1, :]], dim=0)

            if check_unisolvent(X_norm, A_new):
                _, neg_ll_new = optimise_fn(A_new)
                if neg_ll_new < cv:
                    to_include[i] = True

        n_accepted = int(to_include.sum().item())

        if to_include.any() and (m + n_accepted < n_train - 1):
            A_updated              = torch.cat([A, A_extra[to_include, :]], dim=0)
            x_opt_new, neg_ll_new  = optimise_fn(A_updated)

            if neg_ll_new >= cv:
                carry_on = False          # joint fit did not improve
            else:
                A     = A_updated
                cv    = neg_ll_new
                x_opt = x_opt_new
                fit   = {
                    "x":  x_opt,
                    "cv": torch.tensor(neg_ll_new, dtype=torch.float64),
                }
        else:
            carry_on = False              # no candidates accepted or size limit

    return A, fit
