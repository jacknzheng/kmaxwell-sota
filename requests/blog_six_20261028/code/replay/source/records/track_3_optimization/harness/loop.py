from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from records.track_3_optimization.harness.hooks import Config, bind_sites, inject
from records.track_3_optimization.harness.capture_hooks import install


def accumulate_global_gradient(model: nn.Module, batch: tuple[Tensor, Tensor],
                               microbatch_sequences: int,
                               local_gradient_callback=None) -> None:
    """The anchor's training section, verbatim: microbatched token-SUM backward,
    then a SUM all-reduce per parameter -- gradients end up summed over the full
    batch. Exactly one forward/backward pass per optimizer step (Track-3 rule).

    local_gradient_callback (optional, installed by
    capture_local_gradient_snapshots) runs between the backward loop and the
    all-reduce, while p.grad still holds this rank's local sum; it must be
    read-only on the gradients, so the reduced values are bit-identical with or
    without it."""
    inputs, targets = batch
    assert len(inputs) % microbatch_sequences == 0
    for i in range(len(inputs) // microbatch_sequences):
        loss = model(inputs[i*microbatch_sequences:(i+1)*microbatch_sequences],
                     targets[i*microbatch_sequences:(i+1)*microbatch_sequences])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
    if local_gradient_callback is not None:
        local_gradient_callback()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def run(config: Config) -> None:
    """The shared Track-3 loop, with an optional per-update pre-forward site.

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
    if config.get("capture") or config.get("resume_anchor"):
        install(config, hooks)
    state = inject(hooks["setup"], config, {})
    # stop_step (optional): end the loop early (short same-state branches); the LR schedule still uses train_steps
    for step in range(config.get("start_step", 0), min(config["train_steps"], int(config.get("stop_step", config["train_steps"])))):
        state["step"] = step
        state = inject(hooks["pre_forward"], config, state)
        accumulate_global_gradient(state["model"], next(state["train_batches"]),
                                   config["microbatch_sequences"],
                                   state.get("capture_local_gradients"))
        state = inject(hooks["pre_optimizer"], config, state)
        state["optimizer"].step()
        state["model"].zero_grad(set_to_none=True)
        state = inject(hooks["post_optimizer"], config, state)
        if step == config.get("stop_after_step", -1):
            break
    inject(hooks["teardown"], config, state)
