from __future__ import annotations

import math
import os
import time
import uuid
from typing import Any, Callable

import torch
import torch.distributed as dist
import yaml

from optimizers import GroupedOptimizers, find_muon_family_group, parse_group_specs
from optimizers.secant_gmres_muon import SecantGmresMuon
from optimizers.muon import attach_newton_muon_activation_stats, owned_param_indices
from secant_gmres_solver import serialization
from secant_gmres_solver.diagnostics import measure_symmetry_defect
from secant_gmres_solver.solve import solve_secant_least_squares
from secant_gmres_solver.streams import (GRAM_CHUNK, accumulate_secant_gram,
                                         advance_or_first_touch, build_decay_ladder,
                                         build_one_minus_betas)

from .data_fineweb import distributed_data_generator, iterate_batches_single_process
from .model_gpt import GPT, initialize_parameters_like_record

Config = dict[str, Any]
State = dict[str, Any]
Hook = Callable[[Config, State], State]

SITES = ("setup", "pre_optimizer", "post_optimizer", "teardown")


def step_is_due(every: int, step: int, train_steps: int) -> bool:
    """The single step-cadence predicate every periodic hook routes through:
    fires on multiples of `every` and always on the final step; 0 never fires."""
    if every == 0:
        return False
    return step % every == 0 or step == train_steps - 1


# ---------------------------------------------------------------------------
# lifecycle: setup and teardown hooks. Every public function in this module is a
# factory over keyword-only hyperparameters returning a hook
# (config, state) -> state; private memory lives in the closure. Hooks read
# config and mutate/return state; they never write config.
# ---------------------------------------------------------------------------


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
            os.makedirs("logs", exist_ok=True)
            logfile = f"logs/{uuid.uuid4()}.txt"
            print(logfile)

        def print_log(s: str, console: bool = False, log: bool = True) -> None:
            if rank == 0:
                if console:
                    print(s)
                if log:
                    with open(logfile, "a") as f:
                        print(s, file=f)

        state["print_log"] = print_log
        print_log(yaml.safe_dump(config, sort_keys=False))
        print_log("=" * 100)
        print_log(f"git HEAD: {read_git_head()}")
        print_log(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
                  + f" on {torch.cuda.get_device_name()} with world_size {dist.get_world_size()}")
        print_log("=" * 100)
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
        torch.manual_seed(1337 + config["seed"])
        initialize_parameters_like_record(state["model"])
        return state
    return hook


def assemble_grouped_optimizer() -> Hook:
    """Builds the regex-grouped hybrid optimizer into state["optimizer"]."""
    def hook(config: Config, state: State) -> State:
        state["optimizer"] = GroupedOptimizers(
            state["model"], parse_group_specs(config["optimizer_groups"]))
        return state
    return hook


def attach_newton_muon_activation_stats() -> Hook:
    """Install activation-covariance hooks only for a nonzero-alpha Newton arm."""
    def hook(config: Config, state: State) -> State:
        enabled = any(entry["optimizer"].startswith("newton_") and
                      float(entry.get("hyperparams", {}).get("newton_alpha", 0)) > 0
                      for entry in config["optimizer_groups"])
        if enabled:
            state["newton_activation_hook_handles"] = attach_newton_muon_activation_stats(
                state["model"])
        return state
    return hook


def open_training_batches() -> Hook:
    """Opens the sharded training-token stream into state["train_batches"]."""
    def hook(config: Config, state: State) -> State:
        state["train_batches"] = distributed_data_generator(
            config["train_data"], config["batch_tokens"])
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
        state["print_log"](f"run finished: {config['train_steps']} steps,"
                           + f" total train_time {state.get('training_time', 0.0):.3f}s")
        return state
    return hook


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def set_learning_rate_stairs(*, stairs: list) -> Hook:
    """Override the base schedule with fixed multipliers after named steps.

    ``stairs`` is ``[[start_step, multiplier], ...]``.  Before the first
    start step this hook leaves the learning rates unchanged, which lets a
    preceding schedule hook reproduce the shared pre-fork trajectory.  At and
    after the first stair, every optimizer group's learning rate is its base
    rate times the most recent multiplier.
    """
    ordered = sorted((int(step), float(multiplier))
                     for step, multiplier in stairs)
    if not ordered:
        raise ValueError("stairs must contain at least one [step, multiplier]")
    if len({step for step, _ in ordered}) != len(ordered):
        raise ValueError("stairs must have distinct start steps")
    if any(multiplier < 0 for _, multiplier in ordered):
        raise ValueError("learning-rate multipliers must be nonnegative")

    def hook(config: Config, state: State) -> State:
        active = [multiplier for start, multiplier in ordered
                  if state["step"] >= start]
        if active:
            state["optimizer"].scale_learning_rates(active[-1])
        return state
    return hook


def cool_down_learning_rate(*, cooldown_frac: float,
                            shape: str = "linear",
                            schedule_step_min: int | None = None,
                            schedule_step_max: int | None = None,
                            eta_scale: float = 1.0,
                            fixed_eta_after_step: int | None = None,
                            fixed_eta_after: float | None = None) -> Hook:
    """The record's stable-then-decay schedule, pulling the one grouped lever:
    every group's lr = its base lr times eta. shape selects the decay curve
    inside the cooldown window: "linear" (the record's) or "cosine"
    (eta = 0.5*(1 + cos(pi * u)), u = position inside the cooldown).

    schedule_step_min / schedule_step_max clamp the step the schedule is
    evaluated at (intervention experiments): schedule_step_max = S holds eta
    at its step-S value from step S on; schedule_step_min = S applies the
    step-S eta from the start until the true step catches up. eta_scale
    multiplies the resulting eta (values above 1 exceed the peak rate)."""
    assert shape in ("linear", "cosine")
    if (fixed_eta_after_step is None) != (fixed_eta_after is None):
        raise ValueError("fixed_eta_after_step and fixed_eta_after must be set together")
    if fixed_eta_after_step is not None and fixed_eta_after_step < 0:
        raise ValueError("fixed_eta_after_step must be nonnegative")
    if fixed_eta_after is not None and fixed_eta_after < 0:
        raise ValueError("fixed_eta_after must be nonnegative")

    def hook(config: Config, state: State) -> State:
        step = state["step"]
        if fixed_eta_after_step is not None and step >= fixed_eta_after_step:
            state["optimizer"].scale_learning_rates(fixed_eta_after)
            return state
        if schedule_step_max is not None:
            step = min(step, schedule_step_max)
        if schedule_step_min is not None:
            step = max(step, schedule_step_min)
        progress = step / config["train_steps"]
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            u = (progress - (1 - cooldown_frac)) / cooldown_frac
            eta = (1 - u) if shape == "linear" else 0.5 * (1 + math.cos(math.pi * u))
        state["optimizer"].scale_learning_rates(eta * eta_scale)
        return state
    return hook


def log_learning_rates_at_steps(*, steps: list[int]) -> Hook:
    """Logs every resolved optimizer-group LR after schedule hooks run."""
    pinned = {int(step) for step in steps}

    def hook(config: Config, state: State) -> State:
        if state["step"] not in pinned:
            return state
        values = []
        for spec, built in state["optimizer"].groups:
            values.append({"pattern": spec.pattern,
                           "lr": [float(group["lr"])
                                  for group in built.param_groups]})
        state["print_log"](f"learning_rates step:{state['step']} {values}",
                           console=True)
        return state
    return hook


# ---------------------------------------------------------------------------
# validation and progress. The artifacts validate at the top of iteration t, i.e.
# at parameter state x_t; in the harness that same state exists after the
# optimizer ran at step t-1, so the post_optimizer instance evaluates *boundary*
# b = step + 1 with the artifact's exact cadence predicate, and the same
# registered name placed at the setup site (no cadence hyperparams) validates
# boundary 0 and starts the clock. The two instances are distinct closures, which
# is why the clock lives in state (training_time, clock_start,
# last_validated_boundary), not in a closure. Wallclock is unscored on Track 3;
# the stop-the-clock protocol is kept so logs stay line-comparable.
# ---------------------------------------------------------------------------


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
        step_avg = elapsed / (boundary - state["last_validated_boundary"]) if boundary > 0 else float("nan")
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
        state["print_log"](f"step:{boundary}/{train_steps} val_loss:{val_loss:.5f}"
                           + f" train_time:{state['training_time']:.3f}s"
                           + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        state["clock_start"] = time.perf_counter()

    def hook(config: Config, state: State) -> State:
        if "step" not in state:
            # setup site: establish the clock, then validate boundary 0
            state["training_time"] = 0.0
            state["last_validated_boundary"] = 0
            dist.barrier()
            state["clock_start"] = time.perf_counter()
            run_validation(config, state, 0)
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


# ---------------------------------------------------------------------------
# recording: the hooks that make an instrumented run a YAML. record_paired_averages
# advances the paired EMA streams IN MEMORY every step at (x_t, g_t) -- exactly as
# the secant optimizer itself would maintain them -- and dump_secant_state_at_steps
# additionally copies them to disk at the pinned analysis steps. There is
# deliberately no rolling disk dump. Ordering contract (list order in the YAML is
# the contract): these run at the pre_optimizer site, after the gradient
# all-reduce and before any optimizer kernel mutates parameters or gradients (the
# record's Muon kernels mutate p.grad in place), so everything recorded is the
# true (x_t, g_t); record_paired_averages must precede the probe/dump hooks.
# ---------------------------------------------------------------------------


def _format_solve_statistics(stats: dict[str, Any]) -> str:
    """The shared field segment of every solve log line."""
    return (f"rank_disp:{stats['rank_displacements']}"
            f" rank_gmres:{stats['rank_gmres']}"
            f" res_ratio:{stats['res_ratio']:.4f}"
            f" delta_dot_g:{stats['delta_dot_g']:.3e}"
            f" coeff_norm:{stats['coefficient_norm']:.3e}")


def _require_recorder_state(state: State, hook_name: str) -> None:
    if "secant_buffers" not in state:
        raise ValueError(f"{hook_name} needs record_paired_averages earlier in the pre_optimizer list")


def record_paired_averages(*, num_buffers: int, min_decay: float = 0.8,
                           max_decay: float = 0.995, pin_exact_rates: bool = True) -> Hook:
    """Every step, advances the paired EMA buffers for this rank's owned Muon
    parameters at (x_t, g_t), sharing the Muon group's sorted order and rank
    ownership so shard sorted_index matches the optimizer's exactly."""
    ladder = build_decay_ladder(num_buffers, min_decay, max_decay,
                                (0.85, 0.98) if pin_exact_rates else None)
    one_minus_betas: torch.Tensor | None = None

    def hook(config: Config, state: State) -> State:
        nonlocal one_minus_betas
        _, muon = find_muon_family_group(state["optimizer"])
        if isinstance(muon, SecantGmresMuon):
            raise ValueError("record_paired_averages duplicates secant_gmres_muon's own"
                             " paired-EMA buffers; record on a bimaxwell_muon/muon run,"
                             " or read the optimizer's state directly")
        params = muon.sorted_params()
        if "secant_buffers" not in state:
            state["secant_buffers"] = {}
            state["secant_ladder"] = ladder
            state["secant_recipe"] = dict(num_buffers=num_buffers, min_decay=min_decay,
                                          max_decay=max_decay, pin_exact_rates=pin_exact_rates)
            muon_numel = sum(p.numel() for p in params)
            total_gb = 2 * num_buffers * muon_numel * 4 / 1e9
            state["print_log"](f"secant recorder: {muon_numel} muon params x 2 x k={num_buffers}"
                               + f" fp32 = {total_gb:.1f} GB total,"
                               + f" {total_gb / state['world_size']:.1f} GB/GPU;"
                               + f" each pinned dump copies the same again to disk", console=True)
        if one_minus_betas is None:
            one_minus_betas = build_one_minus_betas(ladder, params[0].device)
        for idx in owned_param_indices(len(params), state["rank"], state["world_size"]):
            p = params[idx]
            advance_or_first_touch(state["secant_buffers"].setdefault(idx, {}),
                                   p.grad.flatten(), p.detach().flatten(),
                                   one_minus_betas, num_buffers)
        return state
    return hook


def probe_secant_solve_at_cadence(*, every: int,
                                  truncated_rank_accumulated_parameter_displacements: int,
                                  truncated_rank_gmres: int,
                                  flush_every: int = 500,
                                  dump_dir: str = "secant_dumps") -> Hook:
    """At the cadence, closes the global Gram (one tiny fp64 all_reduce), runs the
    passive solve, logs its statistics, and (rank 0) appends (step, full Gram, w,
    stats) to state["probe_history"], periodically flushed to probe_history.pt."""
    def hook(config: Config, state: State) -> State:
        step = state["step"]
        train_steps = config["train_steps"]
        if not step_is_due(every, step, train_steps):
            return state
        _require_recorder_state(state, "probe_secant_solve_at_cadence")
        _, muon = find_muon_family_group(state["optimizer"])
        params = muon.sorted_params()
        k = state["secant_recipe"]["num_buffers"]
        state.setdefault("secant_solve_recipe", dict(
            truncated_rank_accumulated_parameter_displacements=
                truncated_rank_accumulated_parameter_displacements,
            truncated_rank_gmres=truncated_rank_gmres))
        G = torch.zeros(2 * k + 1, 2 * k + 1, dtype=torch.float64, device=params[0].device)
        for idx in owned_param_indices(len(params), state["rank"], state["world_size"]):
            buffers = state["secant_buffers"][idx]
            p = params[idx]
            accumulate_secant_gram(G, buffers["grad_avgs"], buffers["param_avgs"],
                                   p.grad.flatten(), p.detach().flatten(), GRAM_CHUNK)
        dist.all_reduce(G)
        G_cpu = G.cpu()
        w, stats = solve_secant_least_squares(
            G_cpu, k, truncated_rank_accumulated_parameter_displacements, truncated_rank_gmres)
        epsilon_sym = measure_symmetry_defect(G_cpu, k)
        state["print_log"](f"secant_probe muon_step:{step} {_format_solve_statistics(stats)}"
                           + f" eps_sym:{epsilon_sym:.4f}"
                           + f" fallback:{stats['fallback_reason']}")
        state["last_secant_probe"] = dict(muon_step=step, G=G_cpu, w=w, stats=stats)
        if state["master"]:
            history = state.setdefault("probe_history", [])
            history.append(dict(muon_step=step, G=G_cpu,
                                w=(w.clone() if w is not None else None),
                                stats=dict(stats, epsilon_sym=epsilon_sym)))
            if step_is_due(flush_every, step, train_steps):
                serialization.write_probe_history(dump_dir, dict(
                    format_version=2, k=k, betas=state["secant_ladder"],
                    ladder=state["secant_recipe"], solve=state["secant_solve_recipe"],
                    probes=history))
        return state
    return hook


def dump_secant_state_at_steps(*, steps: list[int], dump_dir: str = "secant_dumps") -> Hook:
    """At each pinned step, copies this rank's in-memory buffers (plus grads and
    parameters) to a shard on disk, and (rank 0) the matching model checkpoint --
    the checkpoints the offline analysis consumes."""
    pinned = set(int(s) for s in steps)

    def hook(config: Config, state: State) -> State:
        step = state["step"]
        if step not in pinned:
            return state
        _require_recorder_state(state, "dump_secant_state_at_steps")
        spec, muon = find_muon_family_group(state["optimizer"])
        params = muon.sorted_params()
        k = state["secant_recipe"]["num_buffers"]
        probe = state.get("last_secant_probe")
        if probe is not None and probe["muon_step"] != step:
            probe = None
        names = {id(p): name for name, p in state["model"].named_parameters()}
        payload: dict[str, Any] = dict(
            format_version=2, muon_step=step, k=k, betas=state["secant_ladder"].clone(),
            rank=state["rank"], world_size=state["world_size"],
            ladder=state["secant_recipe"], solve=state.get("secant_solve_recipe"),
            optimizer_recipe=dict(optimizer=spec.optimizer, hyperparams=dict(spec.hyperparams)),
            w=(probe["w"].clone() if probe is not None and probe["w"] is not None else None),
            stats=(dict(probe["stats"]) if probe is not None else None),
            params={})
        for idx in owned_param_indices(len(params), state["rank"], state["world_size"]):
            buffers = state["secant_buffers"][idx]
            p = params[idx]
            payload["params"][idx] = dict(
                sorted_index=idx, name=names[id(p)], shape=tuple(p.shape),
                m=buffers["grad_avgs"].cpu(), q=buffers["param_avgs"].cpu(),
                grad=p.grad.detach().cpu(), x=p.detach().cpu())
        path = serialization.write_secant_shard(dump_dir, payload)
        if state["master"]:
            serialization.write_model_checkpoint(dump_dir, step, state["model"])
        dist.barrier()
        state["print_log"](f"secant_dump muon_step:{step} wrote {path} and model checkpoint")
        return state
    return hook


def checkpoint_model_at_cadence(*, every: int, dump_dir: str = "secant_dumps",
                                dense_windows: list[list[int]] | None = None) -> Hook:
    """Rank 0 writes a model-only checkpoint at the cadence (at x_t, pre-change).

    dense_windows: optional [[lo, hi], ...] inclusive step ranges inside which a
    checkpoint is written at EVERY step (for per-step trajectory chords), on top
    of the cadence.
    """
    windows = [(int(lo), int(hi)) for lo, hi in (dense_windows or [])]

    def hook(config: Config, state: State) -> State:
        step = state["step"]
        due = step_is_due(every, step, config["train_steps"]) or any(
            lo <= step <= hi for lo, hi in windows)
        if due and state["master"]:
            serialization.write_model_checkpoint(dump_dir, step, state["model"])
        return state
    return hook


def dump_training_state_at_steps(*, steps: list[int],
                                 dump_dir: str = "secant_dumps") -> Hook:
    """At each pinned step S (pre-update, params = x_S), writes this rank's full
    optimizer state plus (rank 0) the model, so a run can be forked from S.

    The per-rank file carries every group's optimizer state_dict — Muon-family
    momentum lives only on the owning rank, so all ranks' files are needed to
    resume — plus the Muon-family _muon_steps_seen counter, which state_dict does
    not cover and which drives the Bi-Maxwell switch.
    """
    pinned = set(int(s) for s in steps)

    def hook(config: Config, state: State) -> State:
        step = state["step"]
        if step not in pinned:
            return state
        os.makedirs(dump_dir, exist_ok=True)
        _, muon = find_muon_family_group(state["optimizer"])
        payload = dict(step=step, rank=state["rank"], world_size=state["world_size"],
                       optimizer_state_dicts=[built.state_dict()
                                              for _, built in state["optimizer"].groups],
                       muon_steps_seen=getattr(muon, "_muon_steps_seen", 0))
        path = os.path.join(dump_dir,
                            f"train_state_step{step:06d}_rank{state['rank']}"
                            f"of{state['world_size']}.pt")
        serialization.write_file_atomically(payload, path)
        if state["master"]:
            serialization.write_file_atomically(
                {k: v.cpu() for k, v in state["model"].state_dict().items()},
                os.path.join(dump_dir, f"train_state_model_step{step:06d}.pt"))
        dist.barrier()
        state["print_log"](f"train_state dump muon_step:{step} wrote {path}")
        return state
    return hook


def load_training_state(*, state_dir: str, step: int,
                        skip_batches: int = 0) -> Hook:
    """Setup hook (place after broadcast_initial_parameters): restores a full
    training state written by dump_training_state_at_steps, then fast-forwards
    the training-token stream by skip_batches so forks can diverge in data.

    The loop must be launched with start_step equal to the dumped step.
    """
    def hook(config: Config, state: State) -> State:
        model_sd = torch.load(
            os.path.join(state_dir, f"train_state_model_step{step:06d}.pt"),
            map_location="cpu", weights_only=True)
        state["model"].load_state_dict(model_sd)
        shard = torch.load(
            os.path.join(state_dir, f"train_state_step{step:06d}_rank{state['rank']}"
                                    f"of{state['world_size']}.pt"),
            map_location="cpu", weights_only=False)
        assert shard["world_size"] == state["world_size"]
        for (_, built), sd in zip(state["optimizer"].groups,
                                  shard["optimizer_state_dicts"]):
            built.load_state_dict(sd)
        _, muon = find_muon_family_group(state["optimizer"])
        muon._muon_steps_seen = shard["muon_steps_seen"]
        for _ in range(skip_batches):
            next(state["train_batches"])
        state["print_log"](f"resumed training state from step {step} in {state_dir},"
                           f" skipped {skip_batches} batches", console=True)
        return state
    return hook


def record_reference_gradients_in_windows(*, windows: list[list[int]], tokens: int,
                                          dump_dir: str = "secant_dumps",
                                          extra_param_names: list[str] = ()) -> Hook:
    """Inside each [start, end] step window (inclusive), computes the reference
    gradient of the Muon-family parameters at x_t over a fixed val-token set and
    writes this rank's owned slices to dump_dir/gradient_sequence/.

    The token set is identical at every step, so step-to-step differences in the
    stored gradients reflect the trajectory alone. The pass is sharded across
    ranks and all-reduced; the budget is split into two disjoint halves (a, b)
    whose difference gives a per-step noise floor. Microbatches reuse the
    training sequence count so the compiled model sees a single shape.

    Args:
        windows: List of [start, end] step spans, inclusive.
        tokens: Global reference-token budget per step; must split evenly into
            an even number of microbatches per rank.
        dump_dir: Parent directory for the gradient_sequence/ shard files.

    Raises:
        ValueError: On malformed windows or a token budget that does not split.
    """
    spans = [(int(a), int(b)) for a, b in windows]
    if not spans or any(b < a for a, b in spans):
        raise ValueError(f"windows must be nonempty [start, end] pairs, got {windows}")

    def hook(config: Config, state: State) -> State:
        step = state["step"]
        if not any(a <= step <= b for a, b in spans):
            return state
        t_start = time.time()
        microbatch_tokens = config["microbatch_sequences"] * 1024
        per_rank = tokens // (microbatch_tokens * state["world_size"])
        if per_rank < 2 or per_rank % 2 or tokens % (microbatch_tokens * state["world_size"]):
            raise ValueError(f"tokens={tokens} must give an even number (>=2) of"
                             f" {microbatch_tokens}-token microbatches per rank")
        _, muon = find_muon_family_group(state["optimizer"])
        params = muon.sorted_params()
        device = params[0].device
        named = dict(state["model"].named_parameters())
        extras = []
        for n in extra_param_names:
            p = named.get(n)
            if p is None:
                p = named.get("_orig_mod." + n)
            assert p is not None, f"extra param {n} not found"
            extras.append((n, p))
        all_params = list(params) + [p for _, p in extras]
        halves = [[torch.zeros(p.numel(), dtype=torch.float32, device=device)
                   for p in all_params] for _ in range(2)]
        for j, (inputs, targets) in enumerate(iterate_batches_single_process(
                config["val_data"], tokens, config["microbatch_sequences"],
                shard_rank=state["rank"], shard_world=state["world_size"])):
            loss = state["model"](inputs, targets)
            accs = halves[0 if j < per_rank // 2 else 1]
            for acc, g in zip(accs, torch.autograd.grad(loss, all_params)):
                acc += g.detach().float().flatten()
        half_tokens = tokens // 2
        for accs in halves:
            for acc in accs:
                dist.all_reduce(acc)
                acc *= config["batch_tokens"] / half_tokens
        names = {id(p): name for name, p in state["model"].named_parameters()}
        payload: dict[str, Any] = dict(
            format_version=1, muon_step=step, rank=state["rank"],
            world_size=state["world_size"], tokens_per_half=half_tokens,
            batch_tokens=config["batch_tokens"], params={})
        for idx in owned_param_indices(len(params), state["rank"], state["world_size"]):
            p = params[idx]
            payload["params"][idx] = dict(
                sorted_index=idx, name=names[id(p)], shape=tuple(p.shape),
                a=halves[0][idx].cpu(), b=halves[1][idx].cpu())
        if extras:
            payload["extra"] = {}
            for k, (n, p) in enumerate(extras):
                rows0 = halves[0][len(params) + k].view(p.shape)[state["rank"]::state["world_size"]]
                rows1 = halves[1][len(params) + k].view(p.shape)[state["rank"]::state["world_size"]]
                payload["extra"][n] = dict(name=n, shape=tuple(p.shape),
                                           row_stride=state["world_size"],
                                           a=rows0.flatten().cpu(), b=rows1.flatten().cpu())
        seq_dir = os.path.join(dump_dir, "gradient_sequence")
        os.makedirs(seq_dir, exist_ok=True)
        path = os.path.join(
            seq_dir, f"gradseq_step{step:06d}_rank{state['rank']}of{state['world_size']}.pt")
        serialization.write_file_atomically(payload, path)
        state["print_log"](f"gradseq muon_step:{step} {tokens} reference tokens"
                           f" in {time.time() - t_start:.2f}s -> {os.path.basename(path)}")
        return state
    return hook


def log_solve_statistics_at_cadence(*, every: int) -> Hook:
    """Logs the SecantGmresMuon's last solve statistics at the cadence.

    Raises:
        ValueError: When the Muon-family group is not a secant_gmres_muon.
    """
    def hook(config: Config, state: State) -> State:
        if not step_is_due(every, state["step"], config["train_steps"]):
            return state
        _, muon = find_muon_family_group(state["optimizer"])
        if not isinstance(muon, SecantGmresMuon):
            raise ValueError("log_solve_statistics_at_cadence requires the secant_gmres_muon optimizer")
        stats = muon.last_solve_statistics
        if stats is None:
            return state   # before solve_start
        state["print_log"](f"secant_solve muon_step:{stats['muon_step']} {_format_solve_statistics(stats)}"
                           + f" delta_norm:{stats['delta_norm']:.3e}"
                           + f" fallbacks:{stats['fallback_count']}")
        return state
    return hook


# Registry name == the YAML name; one entry per hook, no aliases -- a hook name is
# part of every reproduction's config, so renaming one silently breaks a YAML
# (lock-tested). Every value is a factory over keyword-only hyperparameters.

def log_gradient_autocorrelation(*, every: int = 8, halflife: int = 32) -> Hook:
    """Every step, dots the current global gradient of this rank's owned Muon
    parameters against the previous two, all-reduces the three scalars, and on
    rank 0 logs exponentially smoothed lag-1 and lag-2 autocorrelation estimates
    ("gradcorr" lines). Runs before any kernel mutates grads. The online
    oscillation readout an adaptive kernel would consume."""
    decay = 0.5 ** (1.0 / halflife)
    prev: dict[int, list[torch.Tensor]] = {}
    ema = {"c0": 0.0, "c1": 0.0, "c2": 0.0, "w": 0.0}

    def hook(config: Config, state: State) -> State:
        _, muon = find_muon_family_group(state["optimizer"])
        params = muon.sorted_params()
        c = torch.zeros(3, device=params[0].device)
        for idx in owned_param_indices(len(params), state["rank"], state["world_size"]):
            g = params[idx].grad.flatten().float()
            hist = prev.setdefault(idx, [])
            c[0] += g @ g
            if len(hist) >= 1:
                c[1] += g @ hist[-1]
            if len(hist) >= 2:
                c[2] += g @ hist[-2]
            hist.append(g.clone())
            if len(hist) > 2:
                hist.pop(0)
        dist.all_reduce(c)
        c0, c1, c2 = (float(v) for v in c)
        for key, val in (("c0", c0), ("c1", c1), ("c2", c2)):
            ema[key] = decay * ema[key] + (1 - decay) * val
        ema["w"] = decay * ema["w"] + (1 - decay)
        if state["master"] and ema["w"] > 0 and state["step"] % every == 0:
            r1 = ema["c1"] / ema["c0"] if ema["c0"] > 0 else 0.0
            r2 = ema["c2"] / ema["c0"] if ema["c0"] > 0 else 0.0
            state["print_log"](f"gradcorr step:{state['step']} c0:{c0:.4e}"
                               + f" r1_ema:{r1:+.4f} r2_ema:{r2:+.4f}")
        return state
    return hook


_HOOKS: dict[str, Callable[..., Hook]] = {
    "open_rank_zero_log": open_rank_zero_log,
    "load_validation_tokens": load_validation_tokens,
    "build_compiled_gpt": build_compiled_gpt,
    "seed_then_initialize_parameters": seed_then_initialize_parameters,
    "attach_newton_muon_activation_stats": attach_newton_muon_activation_stats,
    "assemble_grouped_optimizer": assemble_grouped_optimizer,
    "open_training_batches": open_training_batches,
    "broadcast_initial_parameters": broadcast_initial_parameters,
    "validate_at_step_boundaries": validate_at_step_boundaries,
    "cool_down_learning_rate": cool_down_learning_rate,
    "log_learning_rates_at_steps": log_learning_rates_at_steps,
    "set_learning_rate_stairs": set_learning_rate_stairs,
    "print_training_progress": print_training_progress,
    "record_paired_averages": record_paired_averages,
    "probe_secant_solve_at_cadence": probe_secant_solve_at_cadence,
    "dump_secant_state_at_steps": dump_secant_state_at_steps,
    "checkpoint_model_at_cadence": checkpoint_model_at_cadence,
    "dump_training_state_at_steps": dump_training_state_at_steps,
    "load_training_state": load_training_state,
    "record_reference_gradients_in_windows": record_reference_gradients_in_windows,
    "log_gradient_autocorrelation": log_gradient_autocorrelation,
    "log_solve_statistics_at_cadence": log_solve_statistics_at_cadence,
    "mark_log_finished": mark_log_finished,
}


def _check_pinned_steps_get_probed(config: Config) -> None:
    """A pinned dump only carries w/stats when a probe ran that same step; make a
    misalignment of the two cadences fail at launch instead of silently writing
    solve-less shards.

    Raises:
        ValueError: When a pinned dump step is not a probe step.
    """
    entries = {entry["name"]: entry.get("hyperparams") or {}
               for entry in (config.get("pre_optimizer") or [])}
    if "dump_secant_state_at_steps" not in entries or "probe_secant_solve_at_cadence" not in entries:
        return
    probe_every = entries["probe_secant_solve_at_cadence"]["every"]
    unprobed = [s for s in entries["dump_secant_state_at_steps"]["steps"]
                if probe_every == 0 or int(s) % probe_every != 0]
    if unprobed:
        raise ValueError(f"pinned dump steps {unprobed} are not multiples of the probe"
                         f" cadence ({probe_every}); their shards would carry no solve")


def bind_sites(config: Config) -> dict[str, list[Hook]]:
    """Resolves every configured hook at every site, binding YAML hyperparams --
    all four sites resolve before step 0, so a YAML typo fails at launch rather
    than after a long run.

    Raises:
        ValueError: On an unregistered hook name or misaligned recording cadences.
    """
    unknown = sorted({entry["name"] for site in SITES
                      for entry in (config.get(site) or [])
                      if entry["name"] not in _HOOKS})
    if unknown:
        raise ValueError(f"unknown injected functions {unknown}; available: {sorted(_HOOKS)}")
    bound = {site: [_HOOKS[entry["name"]](**(entry.get("hyperparams") or {}))
                    for entry in (config.get(site) or [])]
             for site in SITES}
    _check_pinned_steps_get_probed(config)   # after binding: missing hyperparams TypeError first
    return bound


def inject(hooks: list[Hook], config: Config, state: State) -> State:
    """Threads state through the site's hooks, in YAML list order."""
    for hook in hooks:
        state = hook(config, state)
    return state
