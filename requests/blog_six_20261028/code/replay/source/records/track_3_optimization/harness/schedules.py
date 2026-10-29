import math


def scheduled_learning_rate(spec, step, train_steps, initial_lr):
    """Assign absolute power-cooldown rates without a divide/multiply round trip."""
    if spec["kind"] == "power_cooldown":
        if spec["initial_lr"] != initial_lr:
            raise ValueError("power-cooldown initial rate disagrees with optimizer")
        if initial_lr <= 0 or spec["power_c"] < 0 or spec["power"] <= 0:
            raise ValueError("invalid power-cooldown parameters")
        end = train_steps - (spec.get("phase_advance", 0)
                             if step >= spec.get("phase_switch_step", train_steps) else 0)
        return min(initial_lr, spec["power_c"] * max(0.0, end - step) ** spec["power"])
    return initial_lr * learning_rate_factor(spec, step, train_steps)


def learning_rate_factor(spec, step, train_steps):
    kind = spec["kind"]
    if kind == "power_cooldown":
        # Absolute rate divided by the group's original rate, not by its stop
        # boundary. PR341 advances its schedule end at1400 but stops at2600.
        initial = spec["initial_lr"]
        return scheduled_learning_rate(spec, step, train_steps, initial) / initial
    if kind == "cooldown":
        progress = step / train_steps
        fraction = spec["fraction"]
        if not 0 < fraction <= 1:
            raise ValueError("cooldown fraction must lie in (0,1]")
        return 1.0 if progress < 1 - fraction else (1 - progress) / fraction
    if kind == "piecewise_constant":
        boundaries, factors = spec["boundaries"], spec["factors"]
        if (len(factors) != len(boundaries) + 1
                or any(not 0 < boundary < train_steps for boundary in boundaries)
                or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
                or any(factor <= 0 for factor in factors)):
            raise ValueError("invalid piecewise constant LR schedule")
        return factors[sum(step >= boundary for boundary in boundaries)]
    if kind == "piecewise_linear":
        knots = spec["knots"]
        if (len(knots) < 2
                or any(len(knot) != 2 or not isinstance(knot[0], int)
                       or knot[0] < 0 or knot[0] >= train_steps or knot[1] <= 0
                       for knot in knots)
                or any(left[0] >= right[0] for left, right in zip(knots, knots[1:]))):
            raise ValueError("invalid piecewise linear LR schedule")
        if step <= knots[0][0]:
            return knots[0][1]
        for left, right in zip(knots, knots[1:]):
            if step <= right[0]:
                fraction = (step - left[0]) / (right[0] - left[0])
                return left[1] + (right[1] - left[1]) * fraction
        return knots[-1][1]
    if kind == "fast_slow":
        warm, plateau, fast = spec["warmup_end"], spec["plateau_end"], spec["fast_decay_end"]
        peak, floor, minimum = spec["peak_lr"], spec["floor_lr"], spec["min_lr"]
        if not 0 < warm <= plateau < fast < train_steps:
            raise ValueError("invalid fast-slow phase boundaries")
        if step < warm:
            return peak * (step + 1) / warm
        if step < plateau:
            return peak
        if step < fast:
            progress = (step - plateau) / (fast - plateau)
            exponent = spec["fast_decay_exponent"]
            fraction = progress if exponent == 1 else (0 if progress <= 0 else progress ** exponent)
            return peak + (floor - peak) * fraction
        progress = min(max((step - fast) / (train_steps - fast), 0.0), 1.0)
        shape = spec["slow_decay_schedule"]
        if shape == "linear":
            coefficient = 1 - progress
        elif shape == "cosine":
            coefficient = 0.5 * (1 + math.cos(math.pi * progress))
        elif shape == "minus_sqrt":
            coefficient = 1 - math.sqrt(progress)
        else:
            raise ValueError("unknown slow-decay schedule")
        return minimum + coefficient * (floor - minimum)
    raise ValueError(f"unknown learning-rate policy {kind!r}")
