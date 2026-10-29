from dataclasses import dataclass
from typing import MutableMapping, Protocol

import torch

from records.track_3_optimization.optimizers.ringed.momentum.constants import HISTORY_LENGTH


@dataclass(frozen=True)
class MomentumKernel:
    """Newest-first taps; opt-in startup mass correction, off for record EMA."""
    weights: torch.Tensor
    normalize_available: bool = False

    def __post_init__(self):
        weights = torch.as_tensor(self.weights, dtype=torch.float32).detach().clone()
        if weights.shape != (HISTORY_LENGTH,) or not torch.isfinite(weights).all():
            raise ValueError("momentum requires exactly 256 finite weights, newest first")
        if self.normalize_available and (weights < 0).any():
            raise ValueError("available-mass correction requires nonnegative weights")
        object.__setattr__(self, "weights", weights)


class MomentumScheduler(Protocol):
    def __call__(self, iteration: int) -> MomentumKernel: ...



@dataclass
class GradientRing:
    """Explicit view of saved tensor/integer state; no second history buffer."""
    state: MutableMapping
    prefix: str = "gradient"


def push_and_apply(history: GradientRing, sample: torch.Tensor,
                   kernel: MomentumKernel) -> torch.Tensor:
    """Advance the explicit saved ring; the sole public update operation."""
    if not isinstance(history, GradientRing):
        raise TypeError("history must be an explicit GradientRing")
    state, prefix = history.state, history.prefix
    key = prefix + "_history"
    if key not in state:
        state[key] = torch.zeros(HISTORY_LENGTH, sample.numel(),
                                 dtype=torch.float32, device=sample.device)
        state[prefix + "_newest"] = HISTORY_LENGTH - 1
        state[prefix + "_filled"] = 0
    buffer = state[key]
    if buffer.shape != (HISTORY_LENGTH, sample.numel()):
        raise ValueError("checkpoint history shape does not match the sample")
    newest = (state[prefix + "_newest"] + 1) % HISTORY_LENGTH
    buffer[newest].copy_(sample.detach().flatten().float())
    filled = min(HISTORY_LENGTH, state[prefix + "_filled"] + 1)
    state[prefix + "_newest"] = newest
    state[prefix + "_filled"] = filled
    weights = kernel.weights.to(device=sample.device, dtype=torch.float32)
    ages = (newest - torch.arange(HISTORY_LENGTH, device=sample.device)) % HISTORY_LENGTH
    value = weights[ages] @ buffer
    if kernel.normalize_available:
        mass = weights[:filled].sum()
        if mass <= 0:
            raise ValueError("the available kernel mass is not positive")
        value = value / mass
    return value.view_as(sample)
