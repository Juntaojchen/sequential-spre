"""
Multi-start L-BFGS hyperparameter optimisation for SPRE.

Provides a single public entry-point:

    optimise_loocv(loocv_fn, x0, ...)

which maximises a LOOCV log-likelihood (or any scalar function of the raw
parameter vector x) via PyTorch L-BFGS with multiple Gaussian random restarts
in unconstrained (raw) space.

The internal helper ``_lbfgs_step`` performs a single optimisation run from
one starting point and returns the detached optimum together with the achieved
log-likelihood value.
"""

from typing import Callable, Tuple

import torch

from .constants import DEFAULT_NUM_RESTARTS


def optimise_loocv(
    loocv_fn:     Callable[[torch.Tensor], torch.Tensor],
    x0:           torch.Tensor,
    num_restarts: int   = DEFAULT_NUM_RESTARTS,
    restart_scale: float = 0.5,
    seed:         int   = 0,
) -> Tuple[torch.Tensor, float]:

    x0 = torch.as_tensor(x0, dtype=torch.float64)
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    starting_points = [x0.clone()]
    for _ in range(int(num_restarts)):
        noise = torch.randn(x0.shape, dtype=x0.dtype, generator=gen) * float(restart_scale)
        starting_points.append((x0 + noise).clone())

    best_val = float("-inf")
    best_x   = x0.clone()

    for start in starting_points:
        try:
            x_opt, val = _lbfgs_step(loocv_fn, start)
            if val > best_val:
                best_val = val
                best_x   = x_opt
        except Exception as exc:                           # noqa: BLE001
            print(f"[optimise_loocv] restart failed: {exc}")
            continue

    if best_val == float("-inf"):
        raise RuntimeError(
            "All optimisation restarts failed. "
            "Check data quality or kernel parameters."
        )

    return best_x, best_val


def _lbfgs_step(
    loocv_fn: Callable[[torch.Tensor], torch.Tensor],
    init_x:   torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """
    Single L-BFGS run from ``init_x``, maximising ``loocv_fn(x)``.

    Uses PyTorch's built-in L-BFGS with strong Wolfe line search.

    Returns
    -------
    (x_opt, final_val) : detached tensor and float.
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
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        ll   = loocv_fn(x_param)
        loss = -ll                        # minimise negative log-likelihood
        if loss.requires_grad:
            loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_val = float(loocv_fn(x_param.detach()).item())

    return x_param.detach(), final_val
