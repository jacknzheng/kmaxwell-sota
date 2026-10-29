from dataclasses import dataclass
import math

from records.track_3_optimization.optimizers.ringed.momentum.schedulers import (
    ConstantSchedule, LinearSchedule, EMA, HeavyBall, Mixture, FixedKernel,
    KernelInterpolation, ScheduleSwitch)
from records.track_3_optimization.optimizers.ringed.momentum.maxentslop import MaxEntSlop


def value_scheduler(spec):
    if isinstance(spec, (int, float)):
        if not math.isfinite(spec): raise ValueError("scheduled values must be finite")
        return ConstantSchedule(float(spec))
    if not isinstance(spec, dict): raise ValueError("a value scheduler requires a number or mapping")
    kind = spec.get("kind")
    if kind == "constant" and set(spec) == {"kind", "value"}:
        return value_scheduler(spec["value"])
    if kind == "linear" and set(spec) == {"kind", "start", "end", "start_step", "end_step"}:
        if not all(math.isfinite(spec[k]) for k in ("start", "end")):
            raise ValueError("linear endpoints must be finite")
        return LinearSchedule(spec["start"], spec["end"], spec["start_step"], spec["end_step"])
    raise ValueError("unknown value scheduler or unexpected fields")


@dataclass(frozen=True)
class DecayCoordinate:
    value: object
    coordinate: str

    def __call__(self, iteration):
        value = self.value(iteration)
        if self.coordinate == "beta": return value
        if self.coordinate == "log_mean_age": value = math.exp(value)
        if value < 0: raise ValueError("mean age must be nonnegative")
        return value / (1 + value)


def momentum_scheduler(spec):
    """No legacy optimizer alias, gamma generator, or hidden literal replacement."""
    if not isinstance(spec, dict): raise ValueError("momentum scheduler requires a mapping")
    kind = spec.get("kind")
    if kind == "ema":
        if set(spec) - {"kind", "decay", "coordinate", "nesterov_mix"}:
            raise ValueError("unexpected EMA fields")
        coordinate = spec.get("coordinate", "beta")
        if coordinate not in ("beta", "mean_age", "log_mean_age"):
            raise ValueError("unknown decay coordinate")
        return EMA(DecayCoordinate(value_scheduler(spec["decay"]), coordinate), spec.get("nesterov_mix"))
    if kind == "heavyball" and set(spec) == {"kind", "decay"}:
        return HeavyBall(spec["decay"])
    if kind in ("mixture", "bimaxwell") and set(spec) == {"kind", "components", "coefficients"}:
        components = tuple(momentum_scheduler(s) for s in spec["components"])
        coefficients = tuple(value_scheduler(s) for s in spec["coefficients"])
        if len(components) != len(coefficients) or not components:
            raise ValueError("one scheduled coefficient per component is required")
        if kind == "bimaxwell" and (len(components) != 2 or not all(isinstance(s, EMA) for s in components)):
            raise ValueError("Bi-Maxwell requires exactly two EMA components")
        return Mixture(components, coefficients)
    if kind == "maxentslop" and set(spec) == {"kind", "mean_lag", "log_moment", "newest_weight", "second_newest_weight"}:
        return MaxEntSlop(*(value_scheduler(spec[key]) for key in
            ("mean_lag", "log_moment", "newest_weight", "second_newest_weight")))
    if kind == "fixed" and set(spec) == {"kind", "schedule"}:
        return FixedKernel(momentum_scheduler(spec["schedule"])(0))
    if kind == "interpolate" and set(spec) == {"kind", "start", "end", "fraction"}:
        return KernelInterpolation(momentum_scheduler(spec["start"]),
                                   momentum_scheduler(spec["end"]), value_scheduler(spec["fraction"]))
    if kind == "switch" and set(spec) == {"kind", "before", "after", "start_step"}:
        return ScheduleSwitch(momentum_scheduler(spec["before"]),
                              momentum_scheduler(spec["after"]), spec["start_step"])
    raise ValueError(f"unsupported momentum scheduler {kind!r}")
