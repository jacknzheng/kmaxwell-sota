from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch import Tensor


def _load_data_shard(file: Path) -> Tensor:
    """Reads one fineweb .bin shard into pinned uint16 tokens (anchor, verbatim)."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern: str, batch_size: int,
                               seq_len: int = 1024, start_batch: int = 0) -> Iterator[tuple[Tensor, Tensor]]:
    """The anchor's sharded token stream, verbatim: each rank slices its
    batch_size/world_size contiguous tokens per step and yields CUDA
    (inputs, targets) sequence batches."""
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    f = next(file_iter); tokens, pos = _load_data_shard(f), 0
    if dist.get_rank() == 0: print(f"data shard loaded: {f} ({len(tokens)} tokens; batch {batch_size} tokens)", flush=True)
    skipped = 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            f = next(file_iter); tokens, pos = _load_data_shard(f), 0
            if dist.get_rank() == 0: print(f"data shard loaded: {f} ({len(tokens)} tokens)", flush=True)
        if skipped < start_batch:
            pos += batch_size
            skipped += 1
            continue
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)
