from dataclasses import dataclass
from typing import Callable
import math

import torch

from records.track_3_optimization.optimizers.ringed.momentum.constants import HISTORY_LENGTH
from records.track_3_optimization.optimizers.ringed.momentum.kernel import MomentumKernel, MomentumScheduler


@dataclass(frozen=True)
class ConstantSchedule:
    value: float
    def __call__(self, iteration: int) -> float:
        return self.value


@dataclass(frozen=True)
class LinearSchedule:
    start: float
    end: float
    start_step: int
    end_step: int

    def __post_init__(self):
        if self.end_step <= self.start_step:
            raise ValueError("end_step must exceed start_step")

    def __call__(self, iteration: int) -> float:
        fraction = min(1.0, max(0.0, (iteration - self.start_step) /
                                (self.end_step - self.start_step)))
        return self.start + fraction * (self.end - self.start)


@dataclass(frozen=True)
class FixedKernel:
    kernel: MomentumKernel
    def __call__(self, iteration: int) -> MomentumKernel:
        return self.kernel


@dataclass(frozen=True)
class EMA:
    """Exact finite truncation of a possibly time-varying EMA recurrence.

    w_i(t) = (1-beta(t-i)) product_{j=0}^{i-1} beta(t-j).
    This is NOT the stationary kernel (1-beta(t))*beta(t)**i when beta varies.
    No renormalization: the omitted tail is the explicitly accepted truncation.
    With nesterov_mix=m, returns (1-m)*current + m*EMA via the same dot product.
    """
    decay: Callable[[int], float]
    nesterov_mix: float | None = None

    def __call__(self, iteration: int) -> MomentumKernel:
        product = 1.0
        weights = []
        for age in range(HISTORY_LENGTH):
            beta = float(self.decay(max(0, iteration - age)))
            if not math.isfinite(beta) or not 0 <= beta < 1:
                raise ValueError("EMA decay must be finite and in [0,1)")
            weights.append((1 - beta) * product)
            product *= beta
        if self.nesterov_mix is not None:
            if not 0 <= self.nesterov_mix <= 1:
                raise ValueError("Nesterov blend must be in [0,1]")
            weights = [self.nesterov_mix * w for w in weights]
            weights[0] += 1 - self.nesterov_mix
        return MomentumKernel(torch.tensor(weights))


@dataclass(frozen=True)
class HeavyBall:
    decay: float
    def __call__(self, iteration: int) -> MomentumKernel:
        if not math.isfinite(self.decay) or not 0 <= self.decay < 1:
            raise ValueError("heavy-ball decay must be in [0,1)")
        return MomentumKernel(self.decay ** torch.arange(HISTORY_LENGTH))


@dataclass(frozen=True)
class Mixture:
    schedulers: tuple
    coefficients: tuple
    def __call__(self, iteration: int) -> MomentumKernel:
        if not self.schedulers or len(self.schedulers) != len(self.coefficients):
            raise ValueError("one coefficient scheduler per component is required")
        kernels = [s(iteration) for s in self.schedulers]
        if any(k.normalize_available for k in kernels):
            raise ValueError("mixtures require unnormalized component kernels")
        weights = sum(float(c(iteration)) * k.weights
                      for c, k in zip(self.coefficients, kernels))
        return MomentumKernel(weights)


def bimaxwell(fast_decay: float, slow_decay: float, fast_weight: float,
              nesterov_mix: float | None = None) -> Mixture:
    if not 0 <= fast_weight <= 1:
        raise ValueError("Bi-Maxwell mixture weight must be in [0,1]")
    return Mixture((EMA(ConstantSchedule(fast_decay), nesterov_mix),
                    EMA(ConstantSchedule(slow_decay), nesterov_mix)),
                   (ConstantSchedule(fast_weight), ConstantSchedule(1 - fast_weight)))


@dataclass(frozen=True)
class KernelInterpolation:
    """Interpolate solved endpoint taps, not the endpoint constraints."""
    start: MomentumScheduler
    end: MomentumScheduler
    fraction: Callable[[int], float]

    def __call__(self, iteration: int) -> MomentumKernel:
        fraction = float(self.fraction(iteration))
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("kernel interpolation fraction must be in [0,1]")
        first, last = self.start(iteration), self.end(iteration)
        if first.normalize_available or last.normalize_available:
            raise ValueError("kernel interpolation requires unnormalized taps")
        return MomentumKernel(first.weights.lerp(last.weights, fraction))


@dataclass(frozen=True)
class ScheduleSwitch:
    """Select complete schedules at a configured optimizer iteration."""
    before: MomentumScheduler
    after: MomentumScheduler
    start_step: int

    def __call__(self, iteration: int) -> MomentumKernel:
        return (self.before if iteration < self.start_step else self.after)(iteration)
