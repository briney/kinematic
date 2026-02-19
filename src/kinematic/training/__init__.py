"""Training utilities for Kinematic."""

from kinematic.training.losses import TrajectoryLoss
from kinematic.training.scheduler import get_warmup_constant_scheduler

__all__ = [
    "TrajectoryLoss",
    "get_warmup_constant_scheduler",
    "TrainConfig",
    "train",
]


def __getattr__(name: str):
    if name in ("TrainConfig", "train"):
        from kinematic.training.trainer import TrainConfig, train

        globals()["TrainConfig"] = TrainConfig
        globals()["train"] = train
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
