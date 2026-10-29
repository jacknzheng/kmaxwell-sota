"""Explicit tied-beta/current-mix intervention; delegates taps to deployed EMA.

Do not replace the entire schedule with a constant post-intervention EMA:
the historical beta factors before optimizer iteration2009 remain0.95.
This changes Muon's beta and current Nesterov readout mix; Adam and every LR
remain unchanged. The existing gradient ring is preserved, never reset.
"""
from dataclasses import dataclass

from records.track_3_optimization.optimizers.ringed.momentum.schedulers import EMA

BETA_KAPPA_HALF = 0.8990666066208468
BETA_KAPPA_DOUBLE = 0.9750961447125188


@dataclass(frozen=True)
class StepBeta:
    after: float
    before: float = 0.95
    start_step: int = 2009

    def __post_init__(self):
        if not 0 <= self.before < 1 or not 0 <= self.after < 1:
            raise ValueError("Beta must be in[0,1)")

    def __call__(self, iteration):
        return self.before if iteration < self.start_step else self.after


@dataclass(frozen=True)
class TiedStepNesterov:
    beta_after: float
    beta_before: float = 0.95
    start_step: int = 2009

    def __call__(self, iteration):
        decay = StepBeta(self.beta_after, self.beta_before, self.start_step)
        # EMA reads decay(iteration-age), preserving the true beta history.
        # The Nesterov blend is the CURRENT beta, shared across all taps.
        return EMA(decay=decay, nesterov_mix=decay(iteration))(iteration)
