from __future__ import annotations

import torch.distributed as dist
from torch import Tensor, nn

from .hooks import Config, bind_sites, inject


def accumulate_global_gradient(model: nn.Module, batch: tuple[Tensor, Tensor],
                               microbatch_sequences: int) -> None:
    """The anchor's training section, verbatim: microbatched token-SUM backward,
    then a SUM all-reduce per parameter -- gradients end up summed over the full
    batch. Exactly one forward/backward pass per optimizer step (Track-3 rule)."""
    inputs, targets = batch
    assert len(inputs) % microbatch_sequences == 0
    for i in range(len(inputs) // microbatch_sequences):
        model(inputs[i*microbatch_sequences:(i+1)*microbatch_sequences],
              targets[i*microbatch_sequences:(i+1)*microbatch_sequences]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def run(config: Config) -> None:
    """The Track-3 training loop: gradient accumulation plus four hook sites.

    The state is an open dict, deliberately: this is research code, and what a run
    records changes faster than a schema would. The loop reads exactly three keys
    -- "model", "train_batches", "optimizer" (all installed by setup hooks) -- and
    writes one, "step"; a setup that omits one crashes at step 0 rather than
    training something unconfigured. Everything else (logging, validation clocks,
    recorder buffers) is a contract between hooks, documented where the hooks live.

    Args:
        config: The fully resolved YAML config dict.

    Raises:
        ValueError: On a world size other than config["require_world_size"] --
            rank count sets the effective batch, so it is part of the protocol.
    """
    if dist.get_world_size() != config["require_world_size"]:
        raise ValueError(f"this config's reproduction protocol requires world_size"
                         f" {config['require_world_size']}, got {dist.get_world_size()}")
    hooks = bind_sites(config)   # all four sites resolve before step 0
    state = inject(hooks["setup"], config, {})
    for step in range(config.get("start_step", 0), config["train_steps"]):
        state["step"] = step
        accumulate_global_gradient(state["model"], next(state["train_batches"]),
                                   config["microbatch_sequences"])
        state = inject(hooks["pre_optimizer"], config, state)
        state["optimizer"].apply_updates()
        state["model"].zero_grad(set_to_none=True)
        state = inject(hooks["post_optimizer"], config, state)
        if step == config.get("stop_after_step", -1):
            break
    inject(hooks["teardown"], config, state)
