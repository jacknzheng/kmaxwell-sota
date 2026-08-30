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
                               seq_len: int = 1024) -> Iterator[tuple[Tensor, Tensor]]:
    """The anchor's sharded token stream, verbatim: each rank slices its
    batch_size/world_size contiguous tokens per step and yields CUDA
    (inputs, targets) sequence batches."""
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


def iterate_batches_single_process(filename_pattern: str, total_tokens: int,
                                   microbatch_sequences: int, seq_len: int = 1024,
                                   shard_rank: int = 0,
                                   shard_world: int = 1) -> Iterator[tuple[Tensor, Tensor]]:
    """Yields (inputs, targets) CUDA microbatches from the .bin shards without the
    training loop's dist machinery.

    Serves the offline diagnostics: large-batch gradients, HVPs, held-out-shard
    gradients. total_tokens bounds the GLOBAL stream; with shard_world > 1 the
    stream is strided so worker shard_rank sees microbatches shard_rank,
    shard_rank + shard_world, ... -- every worker walks the identical global
    stream, so partial sums reduced across workers equal the single-worker sums.

    Raises:
        AssertionError: When the glob matches no files.
    """
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no data files match {filename_pattern}"
    served = 0
    index = 0
    for file in files:
        tokens = _load_data_shard(file)
        pos = 0
        chunk = microbatch_sequences * seq_len
        while pos + chunk + 1 < len(tokens) and served < total_tokens:
            if index % shard_world == shard_rank:
                buf = tokens[pos:pos + chunk + 1]
                inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True).view(microbatch_sequences, seq_len)
                targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True).view(microbatch_sequences, seq_len)
                yield inputs, targets
            pos += chunk
            served += chunk
            index += 1
        if served >= total_tokens:
            return
