"""Learning rate scheduler (linear warmup + constant)."""

from __future__ import annotations

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_constant_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a scheduler with linear warmup then constant LR.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer to schedule.
    warmup_steps : int
        Number of steps for linear warmup from 0 to base LR.
    last_epoch : int
        Used for resuming. Default -1 (start from beginning).

    Returns
    -------
    LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
