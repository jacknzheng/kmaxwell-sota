import argparse
import math
import re


def decay_shift(weight_decay, normalization_tokens=1, include_weight_decay=True):
    if type(include_weight_decay) is not bool:
        raise ValueError("include_weight_decay must be boolean")
    if (
        type(normalization_tokens) is not int
        or normalization_tokens < 1
        or isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("invalid decay coefficient/normalization tokens")
    return float(weight_decay) / normalization_tokens if include_weight_decay else 0.0


def add_decay_flag(parser):
    parser.add_argument(
        "--include-weight-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include decoupled decay in the frozen response "
        "(default); --no-include-weight-decay restores the polar-only operator",
    )


def checkpoint_decay(checkpoint, names):
    """Use actual saved scheduled settings, not YAML defaults or Adam decay.

    Existing capture settings bind one group to a fullmatch pattern. Multiple
    groups without per-parameter ownership cannot safely be guessed.
    """
    try:
        tokens = (
            checkpoint["training_batch_tokens"]
            if checkpoint.get("format") == 3
            else checkpoint["batch_receipts"]["used"]["batch_tokens"]
        )
        settings = checkpoint["settings"]
    except KeyError as error:
        raise ValueError(
            "decay requires saved optimizer settings and training batch tokens"
        ) from error
    values = {}
    for name in names:
        matches = [s for s in settings if re.fullmatch(s["pattern"], name)]
        if (
            len(matches) != 1
            or matches[0]["optimizer"] not in ("muon", "muonh")
            or len(matches[0]["groups"]) != 1
            or "weight_decay" not in matches[0]["groups"][0]
        ):
            raise ValueError(f"decay requires an unambiguous saved Muon group for {name}")
        value = matches[0]["groups"][0]["weight_decay"]
        decay_shift(value, tokens)
        values[name] = float(value)
    if not values:
        raise ValueError("nonempty Muon names required for decay")
    return values, tokens


def decay_readout(mu, shifts):
    """Preserve the original response eigenvalue for a uniform identity shift.

    A nonuniform diagonal need not commute with T: no scalar subtraction then.
    The polar component refers to the selected total-response eigenmode; it need
    not be a positive polar mode if decay moves a negative mode above zero.
    """
    unique = set(shifts)
    shift = next(iter(unique)) if len(unique) == 1 else None
    return dict(
        weight_decay_shift=shift,
        mu_polar_component=mu - shift if mu is not None and shift is not None else None,
    )
