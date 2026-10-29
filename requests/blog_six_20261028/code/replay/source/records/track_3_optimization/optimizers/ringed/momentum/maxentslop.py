from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
import torch
from records.track_3_optimization.optimizers.ringed.momentum.constants import HISTORY_LENGTH
from records.track_3_optimization.optimizers.ringed.momentum.kernel import MomentumKernel

@dataclass(frozen=True)
class MomentumDesiderata:
    """We numerically solve for momentum kernels that have certain properties. Just a bookkeeping class to track the desiderata.

    mean_lag:             sum_k k * kernel[k], the mean age in steps of the gradients the direction averages over.
    log_moment:           sum_k log(1 + k) * kernel[k], the mean log-age; together with mean_lag it fixes how
                          the weight is spread between recent and old gradients.
    newest_weight:        kernel[0], the weight of the current gradient.
    second_newest_weight: kernel[1], the weight of the previous gradient.
    length:               number of kernel entries (ages 0 .. length-1).
    Always also imposed: the weights sum to 1, and sum_k (-1)**k * kernel[k] = 0, so a gradient that flips sign
    every step (a period-2 oscillation) contributes nothing to the direction.
    """
    mean_lag: float
    log_moment: float
    newest_weight: float
    second_newest_weight: float
    length: int = HISTORY_LENGTH

def solve_for_momentum_kernel_given_desiderata(d: MomentumDesiderata) -> torch.Tensor:
    """Maximum-entropy momentum kernel meeting the desiderata exactly.

    kernel[0] and kernel[1] are set to the two requested weights. Entries for ages k >= 2 maximise the entropy
    -sum(w log w) subject to the four equalities (sum, zero period-2 response, mean_lag, log_moment). The maximiser
    has the exponential-family form w[k] = exp(a + b*(-1)**k + c*k + d*log(1+k)); its four multipliers are found by
    Newton's method on the convex dual with a backtracking line search, to machine precision in a few dozen steps.
    """
    k = np.arange(2, d.length, dtype=float)
    feats = np.vstack([(-1.0) ** k, k, np.log1p(k)])                       # 3 x (length - 2)
    w0, w1 = d.newest_weight, d.second_newest_weight
    mass = 1.0 - w0 - w1                                                     # weight left for ages >= 2
    target = np.array([-(w0 - w1), d.mean_lag - w1, d.log_moment - w1 * np.log(2.0)]) / mass
    scale = np.ptp(feats, axis=1)                                            # condition the features
    F_, t = feats / scale[:, None], target / scale
    def dual(lam):
        z = F_.T @ lam
        return z.max() + np.log(np.exp(z - z.max()).sum()) - lam @ t
    lam = np.zeros(3)
    for _ in range(500):
        z = F_.T @ lam
        p = np.exp(z - z.max()); p /= p.sum()                                # tail weights, normalised to 1
        grad = F_ @ p - t
        if np.max(np.abs(grad)) < 1e-14:
            break
        hess = (F_ * p) @ F_.T - np.outer(F_ @ p, F_ @ p)
        step = -np.linalg.solve(hess, grad)
        f0, a = dual(lam), 1.0
        while dual(lam + a * step) > f0 + 1e-4 * a * (grad @ step) and a > 1e-10:
            a *= 0.5
        lam = lam + a * step
    z = F_.T @ lam
    p = np.exp(z - z.max()); p /= p.sum()
    w = np.concatenate([[w0, w1], mass * p])
    ages = np.arange(d.length)
    check = np.array([w.sum(), ((-1.0) ** ages) @ w, ages @ w, np.log1p(ages) @ w])
    assert np.allclose(check, [1.0, 0.0, d.mean_lag, d.log_moment], rtol=0.0, atol=1e-9), check
    assert w.min() > 0.0 and len(w) == d.length, (w.min(), len(w))
    return torch.as_tensor(w, dtype=torch.float32)


@dataclass(frozen=True)
class MaxEntSlop:
    """Schedule constraints, then run the approved PR369 solver."""
    mean_lag: Callable[[int], float]
    log_moment: Callable[[int], float]
    newest_weight: Callable[[int], float]
    second_newest_weight: Callable[[int], float]

    def __call__(self, iteration: int) -> MomentumKernel:
        d = MomentumDesiderata(self.mean_lag(iteration), self.log_moment(iteration),
                               self.newest_weight(iteration), self.second_newest_weight(iteration))
        validate_desiderata(d)
        return MomentumKernel(solve_for_momentum_kernel_given_desiderata(d))


def validate_desiderata(d: MomentumDesiderata):
    if d.length != HISTORY_LENGTH:
        raise ValueError("MaxEntSlop history is fixed at 256")
    values = (d.mean_lag, d.log_moment, d.newest_weight, d.second_newest_weight)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("MaxEntSlop constraints must be finite")
    if min(d.newest_weight, d.second_newest_weight) <= 0 or d.newest_weight + d.second_newest_weight >= 1:
        raise ValueError("newest weights must be positive and leave positive tail mass")
