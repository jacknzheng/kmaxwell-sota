from __future__ import annotations
import math
import os
import time
import uuid
from typing import Any, Callable
import torch
import torch.distributed as dist
import yaml
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from records.track_3_optimization.harness.soap_record_protocol import initialize_soap341
from records.track_3_optimization.harness.schedules import learning_rate_factor, scheduled_learning_rate
from records.track_3_optimization.harness.soap_record_protocol import soap341_mu
from records.track_3_optimization.harness.checkpoint import save_checkpoint
from records.track_3_optimization.harness.checkpoint import load_checkpoint
from records.track_3_optimization.harness import soap_record_hooks
from records.track_3_optimization.harness.optimizers import GroupedOptimizers
from records.track_3_optimization.harness.data_fineweb import distributed_data_generator
from records.track_3_optimization.harness.model_gpt import GPT, initialize_parameters_like_record
Config = dict[str, Any]
State = dict[str, Any]
Hook = Callable[[Config, State], State]
SITES = ("setup", "pre_forward", "pre_optimizer", "post_optimizer", "teardown")


def step_is_due(every: int, step: int, train_steps: int) -> bool:
    """The single step-cadence predicate every periodic hook routes through:
    fires on multiples of `every` and always on the final step; 0 never fires."""
    if every == 0:
        return False
    return step % every == 0 or step == train_steps - 1


def read_git_head() -> str:
    """The current commit hash by reading .git directly -- never a subprocess:
    forking a process that holds a live NCCL communicator (as every rank does by
    the time setup hooks run) can hang later collectives."""
    try:
        with open(".git/HEAD") as f:
            ref = f.read().strip()
        if not ref.startswith("ref: "):
            return ref
        ref_path = os.path.join(".git", ref[5:])
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return f.read().strip()
        with open(".git/packed-refs") as f:
            for line in f:
                if line.strip().endswith(ref[5:]):
                    return line.split()[0]
        return "unknown"
    except OSError:
        return "unknown"


def open_rank_zero_log() -> Hook:
    """Installs state["print_log"] (a rank-0 file+console writer) plus rank/world/
    master, and logs the provenance preamble: the resolved config text and the git
    commit stand in for the single-file artifacts' habit of logging their own
    source code."""
    def hook(config: Config, state: State) -> State:
        rank = dist.get_rank()
        state["rank"] = rank
        state["world_size"] = dist.get_world_size()
        state["master"] = rank == 0
        logfile = None
        if rank == 0:
            os.makedirs(config["output_dir"], exist_ok=True)
            logfile = os.path.join(config["output_dir"], "train.log")
            print(logfile)

        def print_log(s: str, console: bool = False, log: bool = True) -> None:
            if rank == 0:
                if console:
                    print(s)
                if log:
                    with open(logfile, "a") as f:
                        print(s, file=f)

        state["print_log"] = print_log
        state["logfile"] = logfile
        print_log(yaml.safe_dump(config, sort_keys=False))
        print_log("=" * 100)
        print_log(f"git HEAD: {read_git_head()}")
        print_log(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
                  + f" on {torch.cuda.get_device_name()} with world_size {dist.get_world_size()}")
        print_log("=" * 100)
        if rank == 0 and config.get("wandb_offline", False):
            if wandb is None:
                raise RuntimeError("wandb_offline requires the wandb package")
            state["wandb"] = wandb.init(mode="offline", dir=config["output_dir"],
                                        project="track3-reproductions", name=config["run_id"], config=config)
        return state
    return hook


def load_validation_tokens() -> Hook:
    """Loads the fixed validation batch into state["validation_batch"]."""
    def hook(config: Config, state: State) -> State:
        state["validation_batch"] = next(
            distributed_data_generator(config["val_data"], config["val_tokens"]))
        return state
    return hook


def build_compiled_gpt() -> Hook:
    """Builds the Track-3 GPT on CUDA, compiled like the anchor, into state["model"]."""
    def hook(config: Config, state: State) -> State:
        model = GPT(**config["model"]).cuda()
        model.compile(dynamic=False)
        state["model"] = model
        return state
    return hook


def seed_then_initialize_parameters() -> Hook:
    """Seeds the global RNG with 1337 + config seed, then runs the record's init
    loop. RNG contract: nothing between this hook and broadcast_initial_parameters
    may consume the global RNG, or paired-seed trials break."""
    def hook(config: Config, state: State) -> State:
        if config["reseed_before_initialization"]:
            torch.manual_seed(config["seed_offset"] + config["seed"])
        initialize_parameters_like_record(state["model"], **config["initialization"])
        return state
    return hook


def seed_process(*, rank_offset: bool = True) -> Hook:
    def hook(config: Config, state: State) -> State:
        seed = config["seed"] + (dist.get_rank() if rank_offset else 0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return state
    return hook


def initialize_soap_record() -> Hook:
    """PR341's constructor-preserving initialization, not the Muon reset loop."""
    def hook(config: Config, state: State) -> State:
        initialize_soap341(state["model"], seed=config["seed"])
        return state
    return hook


def schedule_optimizer_groups() -> Hook:
    def hook(config: Config, state: State) -> State:
        for row, (_, optimizer) in zip(config["optimizer_groups"], state["optimizer"].groups):
            factor = learning_rate_factor(row["lr_schedule"], state["step"], config["train_steps"])
            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(row["lr_schedule"], state["step"],
                                                       config["train_steps"], group["initial_lr"])
                if row.get("decay_follows_lr", False):
                    group["weight_decay"] = row["hyperparams"]["weight_decay"] * factor
                if row.get("momentum_schedule") == "soap341":
                    group["mu"] = soap341_mu(state["step"])
        return state
    return hook


def assemble_grouped_optimizer(*, outer_nesterov: dict | None = None) -> Hook:
    """Builds the regex-grouped hybrid optimizer into state["optimizer"]."""
    def hook(config: Config, state: State) -> State:
        state["optimizer"] = GroupedOptimizers(
            state["model"], config["optimizer_groups"])
        if outer_nesterov is not None:
            state["optimizer"].install_outer_nesterov(state["model"], outer_nesterov)
        return state
    return hook


def optimizer_before_forward() -> Hook:
    def hook(config: Config, state: State) -> State:
        state["optimizer"].before_forward()
        return state
    return hook


def open_training_batches() -> Hook:
    """Opens the sharded training-token stream into state["train_batches"]."""
    def hook(config: Config, state: State) -> State:
        state["train_batches"] = distributed_data_generator(
            config["train_data"], config["batch_tokens"], start_batch=config.get("start_step", 0))
        return state
    return hook


def checkpoint_state() -> Hook:
    def hook(config: Config, state: State) -> State:
        boundary = state["step"] + 1
        if boundary % config["checkpoint_every"] == 0 or boundary == min(config["train_steps"], config.get("stop_step", config["train_steps"])):
            save_checkpoint(config, state, boundary)
        return state
    return hook


def restore_checkpoint() -> Hook:
    def hook(config: Config, state: State) -> State:
        if config.get("resume_checkpoint"):
            config["start_step"] = load_checkpoint(config, state)
        return state
    return hook


def broadcast_initial_parameters() -> Hook:
    """Broadcasts rank 0's initialized parameters to the fleet."""
    def hook(config: Config, state: State) -> State:
        for p in state["model"].parameters():
            dist.broadcast(p.detach(), 0)
        return state
    return hook


def mark_log_finished() -> Hook:
    """Teardown: writes the closing line so finished logs are distinguishable."""
    def hook(config: Config, state: State) -> State:
        state["print_log"](f"run finished: {state.get('step', -1) + 1} executed steps,"
                           + f" total train_time {state.get('training_time', 0.0):.3f}s")
        if state.get("wandb") is not None:
            state["wandb"].finish()
        return state
    return hook


def boundary_is_due(boundary: int, train_steps: int, every: int, final_tenth_every: int,
                    dense_window: list[int] | None, dense_every: int) -> bool:
    """The record artifacts' validation cadence, verbatim in boundary terms: every
    125 steps (25 in the last tenth), always at the end, plus the optional dense
    eval-only window. Module-level so tests can compare it against the artifacts'
    inline predicates over every boundary."""
    if boundary == train_steps:
        return True
    if every:
        frequency = every if boundary / train_steps < 0.9 else (final_tenth_every or every)
        if boundary % frequency == 0:
            return True
    if dense_window is not None and dense_window[0] <= boundary <= dense_window[1] \
            and boundary % dense_every == 0:
        return True
    return False


def validate_at_step_boundaries(*, every: int = 0, final_tenth_every: int = 0,
                                dense_window: list[int] | None = None,
                                dense_every: int = 10) -> Hook:
    """Runs the artifact's validation protocol at due step boundaries (or, at the
    setup site, at boundary 0): stop the clock, eval in 64-sequence chunks with
    token-SUM + all-reduce, print the artifact's exact log line, restart the clock."""
    def run_validation(config: Config, state: State, boundary: int) -> None:
        train_steps = config["train_steps"]
        mbs = config["microbatch_sequences"]
        model = state["model"]
        # stop the clock
        dist.barrier()
        elapsed = time.perf_counter() - state["clock_start"]
        completed = boundary - state["last_validated_boundary"]
        step_avg = elapsed / completed if completed > 0 else float("nan")
        state["last_validated_boundary"] = boundary
        state["training_time"] += elapsed
        model.eval()
        val_inputs, val_targets = state["validation_batch"]
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= config["val_tokens"]
        if not torch.isfinite(val_loss):
            raise RuntimeError("non-finite validation loss")
        state["print_log"](f"step:{boundary}/{train_steps} val_loss:{val_loss:.5f}"
                           + f" train_time:{state['training_time']:.3f}s"
                           + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        if state.get("wandb") is not None:
            state["wandb"].log({"step": boundary, "val_loss": val_loss.item(),
                                "objective": val_loss.item()})
        model.train()
        # start the clock again
        dist.barrier()
        state["clock_start"] = time.perf_counter()

    def hook(config: Config, state: State) -> State:
        if "step" not in state:
            # setup site: establish the clock, then validate boundary 0
            state["training_time"] = 0.0
            state["last_validated_boundary"] = config.get("start_step", 0)
            dist.barrier()
            state["clock_start"] = time.perf_counter()
            run_validation(config, state, config.get("start_step", 0))
            return state
        boundary = state["step"] + 1
        if boundary_is_due(boundary, config["train_steps"],
                           every, final_tenth_every, dense_window, dense_every):
            run_validation(config, state, boundary)
        return state
    return hook


def print_training_progress(*, every: int = 1) -> Hook:
    """The artifacts' console-only per-step progress line (never written to the log)."""
    def hook(config: Config, state: State) -> State:
        step = state["step"]
        if every and step % every == 0:
            approx_training_time = state["training_time"] + (time.perf_counter() - state["clock_start"])
            state["print_log"](f"step:{step+1}/{config['train_steps']}"
                               + f" train_time:{approx_training_time:.3f}s"
                               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms",
                               console=True, log=False)
        return state
    return hook


def inject(hooks: list[Hook], config: Config, state: State) -> State:
    """Threads state through the site's hooks, in YAML list order."""
    for hook in hooks:
        state = hook(config, state)
    return state

def hook_factories():
    # Keep the PR341 initializer separate from the original Muon reset hook.
    factories = {'initialize_soap_record': initialize_soap_record,
                 'optimizer_before_forward': optimizer_before_forward}
    factories.update(_base_hook_factories())
    for name in ('install_soap_record_state', 'enable_output_kfac', 'apply_output_kfac',
                 'update_soap_tail', 'close_soap_record_state', 'validate_soap_record'):
        factories[name] = getattr(soap_record_hooks, name)
    return factories

def _base_hook_factories():
    return {'open_rank_zero_log': open_rank_zero_log, 'load_validation_tokens': load_validation_tokens, 'build_compiled_gpt': build_compiled_gpt, 'seed_process': seed_process, 'seed_then_initialize_parameters': seed_then_initialize_parameters, 'assemble_grouped_optimizer': assemble_grouped_optimizer, 'open_training_batches': open_training_batches, 'broadcast_initial_parameters': broadcast_initial_parameters, 'mark_log_finished': mark_log_finished, 'schedule_optimizer_groups': schedule_optimizer_groups, 'restore_checkpoint': restore_checkpoint, 'checkpoint_state': checkpoint_state, 'validate_at_step_boundaries': validate_at_step_boundaries, 'print_training_progress': print_training_progress}

def bind_sites(config: Config) -> dict[str, list[Hook]]:
    factories = hook_factories()
    unknown = sorted({entry["name"] for site in SITES for entry in config.get(site, [])
                      if entry["name"] not in factories})
    if unknown:
        raise ValueError(f"unsupported selected lifecycle hooks: {unknown}")
    return {site: [factories[entry["name"]](**entry.get("hyperparams", {}))
                   for entry in config.get(site, [])] for site in SITES}
