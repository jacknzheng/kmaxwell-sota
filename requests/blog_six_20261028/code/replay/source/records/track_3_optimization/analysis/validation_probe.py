import glob
from pathlib import Path

import numpy as np
import torch


def fixed_validation_batch(config, device, tokens=524288, offset=0):
    """Deterministic contiguous, non-wrapping validation probe; nested by size.

    Offset counts input tokens, not bytes. Disjoint probes leave one boundary
    token between intervals so their input/target tokens do not overlap.
    """
    if (
        type(tokens) is not int
        or tokens < 1024
        or tokens % 1024
        or type(offset) is not int
        or offset < 0
    ):
        raise ValueError("validation tokens must be a positive multiple of 1024")
    files = sorted(Path(p) for p in glob.glob(config["val_data"]))
    if not files:
        raise ValueError("fixed validation Hessian needs a validation shard")
    path = files[0]
    header = np.fromfile(path, dtype=np.int32, count=256)
    if (
        len(header) != 256
        or header[0] != 20240520
        or header[1] != 1
        or header[2] < offset + tokens + 1
    ):
        raise ValueError("invalid fixed validation shard")
    raw = np.fromfile(path, dtype=np.uint16, count=tokens + 1, offset=1024 + 2 * offset)
    if len(raw) != tokens + 1:
        raise ValueError("incomplete fixed validation probe")
    return (
        torch.from_numpy(raw[:-1].copy())
        .to(device=device, dtype=torch.int32)
        .view(-1, 1024),
        torch.from_numpy(raw[1:].copy())
        .to(device=device, dtype=torch.int64)
        .view(-1, 1024),
    )
