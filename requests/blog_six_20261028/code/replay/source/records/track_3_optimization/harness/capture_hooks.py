from __future__ import annotations

import json
import hashlib
from records.track_3_optimization.harness.capture_store import file_sha
from records.track_3_optimization.harness.model_gpt import GPT
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from records.track_3_optimization.harness.capture_store import CaptureWriter
from records.track_3_optimization.harness.constants import BLOCK_PREFIX, ANCHOR_FORMAT
from records.track_3_optimization.optimizers import Muon
from records.track_3_optimization.optimizers.ringed.muon import MuonH
from records.track_3_optimization.optimizers.ringed.momentum.constants import HISTORY_LENGTH

# Ported from frozen capture_fd_20261001_r2_final/src/harness/capture_hooks.py.
# Schema 2 uses fixed256 history/newest/filled and scheduler iteration. Historical
# schema-1 captures still require their pinned replay source; do not relabel them.


def capture_spec(config: dict[str, Any]) -> dict[str, Any] | None:
    spec = config.get("capture")
    if not spec:
        return None
    spec = dict(spec)
    spec["windows"] = [(int(a), int(b)) for a, b in spec.get("windows", [])]
    spec["anchors"] = sorted(int(a) for a in spec.get("anchors", []))
    spec["root"] = os.environ.get("CAPTURE_ROOT", spec.get("root", "./capture_out"))
    spec.setdefault("ring_length", None)
    spec["ringless_anchors"] = sorted(int(a) for a in spec.get("ringless_anchors", []))
    assert set(spec["ringless_anchors"]) <= set(spec["anchors"]), "ringless_anchors must be anchors"
    return spec


def ring_length_from_config(config: dict[str, Any]) -> int:
    for g in config["optimizer_groups"]:
        if g["optimizer"] in ("muon", "muonh"):
            return HISTORY_LENGTH
    return 0


def gradient_steps(spec: dict[str, Any], ring_length: int) -> set[int]:
    steps: set[int] = set()
    for a, b in spec["windows"]:
        if spec.get("window_gradient_only", False):
            # Posthoc boundary analysis needs g_t, but no earlier ring files
            # unless a ringless restart anchor is requested.
            steps.update(range(a, b + 1))
        else:
            # boundary a's ring holds insertions a-L .. a-1 (pre-boundary convention)
            steps.update(range(max(0, a - ring_length), b + 1))
    for n in spec.get("ringless_anchors", []):
        # a ringless anchor N rebuilds its ring from insertions N-L .. N-1, so they must be stored
        if n - ring_length < 0:
            raise ValueError(f"ringless anchor {n} needs {ring_length} earlier insertions")
        steps.update(range(n - ring_length, n))
    return steps


def window_steps(spec: dict[str, Any]) -> set[int]:
    return {t for a, b in spec["windows"] for t in range(a, b + 1)}


@torch.no_grad()
def momentum_restoring_metrics(weight, gradient, momentum, polar, kernel, filled, polar_fn):
    """Read-only, per-matrix update-t scalars, before decay/displacement.

    M_t includes g_t. c_t,0 is the coefficient actually applied to g_t,
    including opt-in available-mass correction. Both signs use the trainer's
    BF16 Newton-Schulz approximation, BEFORE rectangular-shape scaling.
    This is a subtraction from the recorded M_t, not a fresh ring reduction.
    """
    if not 1 <= int(filled) <= HISTORY_LENGTH:
        raise ValueError("restoring metrics require a post-insertion history")
    taps = kernel.weights.to(device=momentum.device, dtype=torch.float32)
    c0 = taps[0]
    if kernel.normalize_available:
        mass = taps[:int(filled)].sum()
        if not bool(mass > 0):
            raise ValueError("the available kernel mass is not positive")
        c0 = c0 / mass
    remainder = momentum - c0 * gradient.float()
    remainder_polar = polar_fn(remainder)
    w, g, m, r, q, qr = (x.detach().double() for x in
                         (weight, gradient, momentum, remainder, polar, remainder_polar))
    values = dict(
        momentum_norm=float(m.norm()),
        momentum_without_lag0_norm=float(r.norm()),
        gradient_norm=float(g.norm()), lag0_coefficient=float(c0),
        msign_momentum_norm=float(q.norm()),
        msign_momentum_without_lag0_norm=float(qr.norm()),
        msign_difference_norm=float((q - qr).norm()),
        weight_norm=float(w.norm()),
        weight_dot_msign_momentum=float((w * q).sum()),
        gradient_dot_msign_momentum=float((g * q).sum()))
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("non-finite momentum restoring metrics")
    return values


class CountingBatches:
    """Wraps the training-batch iterator; ``consumed`` is the data-loader position
    (batch index b = consumed - 1 is the batch the current update used)."""

    def __init__(self, inner, consumed: int = 0) -> None:
        self.inner, self.consumed, self.last = inner, consumed, None

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.inner)
        self.consumed += 1
        self.last = batch
        return batch


def batch_digest(batch) -> str:
    h = hashlib.sha256()
    for x in batch:
        h.update(str(tuple(x.shape)).encode()); h.update(str(x.dtype).encode())
        h.update(x.detach().cpu().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


@lru_cache(maxsize=None)
def shard_sha256(path: str) -> str:
    return file_sha(path)


def pin_eval_batches(val_pattern: str | None) -> dict[str, Any]:
    """The four fixed evaluation batches of dense/kprobe_dense.py: first validation
    shard, tokens [b*2048, (b+1)*2048+1) for b = 0..3, inputs/targets 2x1024."""
    files = sorted(Path.cwd().glob(val_pattern)) if val_pattern else []
    if not files:
        return dict(available=False, note="no validation shard matched; pin at launch")
    path = files[0]
    tokens = np.fromfile(path, dtype=np.uint16, count=4 * 2048 + 1, offset=1024).astype(np.int64)
    rows = []
    for b in range(4):
        buf = torch.from_numpy(tokens[b * 2048:(b + 1) * 2048 + 1].copy())
        batch = (buf[:-1].to(torch.int32).reshape(2, 1024), buf[1:].reshape(2, 1024))
        rows.append(dict(batch=b, input_token_start=b * 2048, target_token_start=b * 2048 + 1, tokens=2048,
                         raw_tokens_sha256=hashlib.sha256(tokens[b * 2048:(b + 1) * 2048 + 1].astype(np.uint16).tobytes()).hexdigest(),
                         batch_sha256=batch_digest(batch)))
    return dict(available=True, source=str(path.resolve()), source_sha256=shard_sha256(str(path.resolve())),
                header_bytes=1024, batches=rows)


def optimizer_state_digest(state) -> dict[str, Any]:
    """sha256 of every optimizer state tensor except Muon rings (rebuilt from gradient
    files), keyed group/param-name/key; the direct boundary receipt that offline
    reconstructions are checked against."""
    names = {id(p): n for n, p in state["model"].named_parameters()}
    out = {}
    for spec_, built in state["optimizer"].groups:
        for p, st in built.state.items():
            for k, v in st.items():
                if k == "gradient_history":
                    continue
                key = f"{spec_.optimizer}:{spec_.pattern}|{names[id(p)]}|{k}"
                if isinstance(v, torch.Tensor):
                    t = v.detach().cpu().contiguous()
                    out[key] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dim() else t.reshape(1).view(torch.uint8).numpy().tobytes()).hexdigest()
                else:
                    out[key] = repr(v)
    return out


@lru_cache(maxsize=None)
def shard_sizes(pattern: str) -> list[tuple[str, int]]:
    """(resolved path, header token count) for the sorted shards the loader walks."""
    rows = []
    for f in sorted(Path.cwd().glob(pattern)):
        header = np.fromfile(f, dtype=np.int32, count=256)
        assert header[0] == 20240520 and header[1] == 1, f"bad shard header {f}"
        rows.append((str(f.resolve()), int(header[2])))
    return rows


def batch_receipt(pattern: str | None, batch_tokens: int, b: int, rank: int, world: int) -> dict[str, Any]:
    """Where distributed_data_generator's batch b comes from, by replaying its
    arithmetic: shard switch when pos + batch_size + 1 >= len(shard); rank r reads
    tokens [pos + r*local, pos + (r+1)*local + 1) of that shard (inputs = first
    local tokens, targets = the same range shifted by one)."""
    out = dict(batch_index=b, rank=rank, world_size=world, batch_tokens=batch_tokens)
    shards = shard_sizes(pattern) if pattern else []
    if not shards:
        out.update(shard=None, note="no shard files matched (synthetic data); position only")
        return out
    local = batch_tokens // world
    k, pos = 0, 0
    for _ in range(b + 1):
        if pos + batch_tokens + 1 >= shards[k][1]:
            k, pos = k + 1, 0
        here = pos
        pos += batch_tokens
    start = here + rank * local
    out.update(shard=shards[k][0], shard_name=Path(shards[k][0]).name, shard_sha256=shard_sha256(shards[k][0]),
               shard_index=k, batch_start_in_shard=here,
               rank_token_start=start, rank_token_stop=start + local + 1,
               header_bytes=1024, byte_offset=1024 + 2 * start)
    return out


def rng_state() -> dict[str, Any]:
    """RNG states as weights_only-loadable primitives and tensors (no NumPy objects)."""
    version, internal, gauss = random.getstate()
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    out = dict(torch=torch.get_rng_state(),
               python=dict(version=int(version), internal=torch.tensor(internal, dtype=torch.int64),
                           gauss=None if gauss is None else float(gauss)),
               numpy=dict(kind=str(kind), keys=torch.from_numpy(keys.astype(np.int64)), pos=int(pos),
                          has_gauss=int(has_gauss), cached_gaussian=float(cached)))
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng(saved: dict[str, Any]) -> None:
    torch.set_rng_state(saved["torch"])
    py = saved["python"]
    random.setstate((py["version"], tuple(int(x) for x in py["internal"].tolist()), py["gauss"]))
    npy = saved["numpy"]
    np.random.set_state((npy["kind"], npy["keys"].numpy().astype(np.uint32), npy["pos"],
                         npy["has_gauss"], npy["cached_gaussian"]))
    if "cuda" in saved and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda"])


def _muon(state):
    groups = [built for _, built in state["optimizer"].groups if isinstance(built, Muon)]
    if len(groups) != 1 or len(groups[0].param_groups) != 1:
        raise ValueError("dense capture requires exactly one single-group Muon/MuonH")
    return groups[0]


def _index_append(root: Path, row: dict[str, Any]) -> None:
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def ring_digest(ring: torch.Tensor, ring_next: int) -> str:
    """sha256 over (format tag, ring length, cursor, shape, then every ring byte as
    contiguous row-major float32 in stored slot order). Equal digests mean equal
    rings and cursors up to sha256 collisions; copied to CPU in row chunks."""
    h = hashlib.sha256()
    L = int(ring.shape[0])
    h.update(f"capture-ring-v2|L={L}|cursor={int(ring_next)}|shape={tuple(ring.shape)}|float32|".encode())
    step = max(1, (1 << 26) // max(1, ring[0].numel()))   # ~256 MB per chunk
    for i in range(0, L, step):
        h.update(ring[i:i + step].detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def muon_ring_digests(state) -> dict[str, str]:
    muon = _muon(state)
    names = {id(p): n for n, p in state["model"].named_parameters()}
    return {names[id(p)]: ring_digest(st["gradient_history"], (st["gradient_newest"] + 1) % HISTORY_LENGTH)
            for p, st in muon.state.items() if "gradient_history" in st}


def nonfinite_tensors(obj, prefix: str = "") -> list[str]:
    """Paths of every floating tensor inside obj (dicts/lists/tuples) with NaN or Inf."""
    bad = []
    if isinstance(obj, torch.Tensor):
        if (obj.is_floating_point() or obj.is_complex()) and obj.numel() and not bool(torch.isfinite(obj).all()):
            bad.append(prefix or "<tensor>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += nonfinite_tensors(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += nonfinite_tensors(v, f"{prefix}[{i}]")
    return bad


def build_anchor(config, state, next_update: int, ringless: bool = False,
                 stored: set[int] | None = None) -> dict[str, Any]:
    """ringless=True drops gradient_history (keeps newest/filled) and records the
    stored gradient range that rebuilds it; used only when that range is on disk."""
    muon = _muon(state)
    if int(muon.param_groups[0]["iteration"]) != next_update:
        raise ValueError("anchor boundary differs from scheduler clock")
    if state["train_batches"].consumed != next_update:
        raise ValueError("anchor boundary differs from data-loader position")
    names = {id(p): n for n, p in state["model"].named_parameters()}
    sds = [built.state_dict() for _, built in state["optimizer"].groups]
    ring_rebuild = None
    if ringless:
        L = HISTORY_LENGTH
        need = set(range(next_update - L, next_update))
        if stored is None or not need <= stored:
            raise ValueError(f"anchor {next_update}: gradients {next_update - L}..{next_update - 1} not all stored")
        gi = [i for i, (_, b) in enumerate(state["optimizer"].groups) if b is muon][0]
        order = [names[id(p)] for p in muon.param_groups[0]["params"]]
        idx = sds[gi]["param_groups"][0]["params"]
        sd_state = dict(sds[gi]["state"])
        params = {}
        for i, name in zip(idx, order):
            if i in sd_state and "gradient_history" in sd_state[i]:
                entry = dict(sd_state[i]); del entry["gradient_history"]; sd_state[i] = entry
                params[int(i)] = name
        sds[gi] = dict(sds[gi], state=sd_state)
        ring_rebuild = dict(group_index=gi, first_update=next_update - L, last_update=next_update - 1,
                            ring_length=L, params=params,
                            rule="slot (gradient_newest + 1 - k) % L holds gradient_{N-k}.pt[name], k=1..L")
    return dict(ring_rebuild=ring_rebuild,
                live_ring_sha256=muon_ring_digests(state),   # exact digest, verified after load or rebuild
        format=ANCHOR_FORMAT, anchor_update=next_update,
        convention="state after update N-1 applied; executes update N",
        rank=state["rank"], world_size=state["world_size"],
        model={k: v for k, v in state["model"].state_dict().items()},
        optimizer_state_dicts=sds,
        group_settings=[[{k: v for k, v in g.items() if k != "params"} for g in built.param_groups]
                        for _, built in state["optimizer"].groups],
        muon_steps_seen=int(muon.param_groups[0]["iteration"]),
        batches_consumed=int(state["train_batches"].consumed),
        rng=rng_state(), config_run_id=config.get("run_id"),
        lr_schedule_note="cool_down_learning_rate is a pure function of the step; no state",
    )


def _validate_restore_boundary(shard, state) -> int:
    """Check schema-2 identity before mutating model, optimizer, loader or RNG.

    A partially advanced loader may catch up to N, but cannot rewind. Its
    consumed count is trusted to describe its iterator; this is not a hash
    check of the dataset or a qualification of distributed/CUDA replay.
    """
    def counter(value, label):
        if type(value) is not int or value < 0:
            raise ValueError(f"anchor {label} must be a nonnegative integer")
        return value

    boundary = counter(shard.get("anchor_update"), "boundary")
    if counter(shard.get("muon_steps_seen"), "scheduler clock") != boundary:
        raise ValueError("anchor boundary differs from scheduler clock")
    if counter(shard.get("batches_consumed"), "data-loader position") != boundary:
        raise ValueError("anchor boundary differs from data-loader position")
    if shard.get("rank") != state["rank"] or shard.get("world_size") != state["world_size"]:
        raise ValueError("anchor rank or world size differs")
    consumed = counter(state["train_batches"].consumed, "current data-loader position")
    if consumed > boundary:
        raise ValueError("data-loader position is ahead of anchor boundary; cannot rewind")
    saved_groups = shard.get("optimizer_state_dicts", [])
    if len(saved_groups) != len(state["optimizer"].groups):
        raise ValueError("anchor optimizer group count differs")
    muon = _muon(state)
    for (_, built), saved in zip(state["optimizer"].groups, saved_groups):
        if built is not muon:
            continue
        groups = saved.get("param_groups", [])
        if len(groups) != 1 or counter(groups[0].get("iteration"), "optimizer iteration") != boundary:
            raise ValueError("anchor boundary differs from optimizer iteration")
        # Fixed256 advances once per optimizer iteration, including zero
        # samples when a gradient is absent. Ringless anchors retain these.
        for entry in saved.get("state", {}).values():
            if (entry.get("gradient_newest") != (boundary - 1) % HISTORY_LENGTH
                    or entry.get("gradient_filled") != min(boundary, HISTORY_LENGTH)):
                raise ValueError("anchor history cursor/fill differs from boundary")
    return boundary


def load_anchor_into(config, state, path: str) -> int:
    """Restore schema 2; refuse inconsistent boundary metadata before loading.

    Later payload/digest failures are not transactional rollback. Callers must
    discard the destination state after any failed restore.
    """
    p = Path(path)
    if p.is_dir():   # rank directory root: pick this rank's file
        raise ValueError("pass the anchor file of rank 0; other ranks are derived by name")
    rank_path = Path(str(p).replace("/rank0/", f"/rank{state['rank']}/"))
    shard = torch.load(rank_path, map_location="cpu", weights_only=True)
    if shard.get("format") != ANCHOR_FORMAT:
        raise ValueError("anchor is not the fixed256 scheduler schema; use its historical replay source")
    bad = nonfinite_tensors(shard)
    if bad:
        raise ValueError(f"anchor {rank_path} holds non-finite tensors: {bad[:5]} ({len(bad)} total); refusing to resume")
    _validate_restore_boundary(shard, state)
    state["model"].load_state_dict(shard["model"])
    for (_, built), sd in zip(state["optimizer"].groups, shard["optimizer_state_dicts"]):
        built.load_state_dict(sd)
    muon = _muon(state)
    if muon.param_groups[0]["iteration"] != shard["muon_steps_seen"]:
        raise ValueError("anchor scheduler clock mismatch")
    rb = shard.get("ring_rebuild")
    if rb:
        gradient_dir = rank_path.parent.parent / "rank0"
        manifest = {}
        for line in (gradient_dir / "manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "gradient":
                manifest[row["step"]] = row
        N, L = shard["anchor_update"], rb["ring_length"]
        grads = {}
        for s in range(rb["first_update"], rb["last_update"] + 1):
            f = gradient_dir / manifest[s]["file"]
            if file_sha(f) != manifest[s]["sha256"]:
                raise ValueError(f"gradient {s} hash mismatch")
            grads[s] = torch.load(f, map_location="cpu", weights_only=True)
        params = muon.param_groups[0]["params"]
        for i, name in rb["params"].items():
            p = params[int(i)]
            st = muon.state[p]
            write = (int(st["gradient_newest"]) + 1) % L
            ring = torch.zeros(L, p.numel(), dtype=torch.float32, device=p.device)
            for k in range(1, L + 1):
                ring[(write - k) % L].copy_(grads[N - k][name].flatten().float())
            st["gradient_history"] = ring
    expected = shard.get("live_ring_sha256")
    if expected is None:
        raise ValueError("anchor lacks live_ring_sha256; written by an older hook, refuse")
    names_by_id = {id(p): n for n, p in state["model"].named_parameters()}
    got = muon_ring_digests(state)
    bad = sorted(n for n in set(expected) | set(got) if expected.get(n) != got.get(n))
    if bad:
        raise ValueError(f"anchor {shard['anchor_update']}: restored ring differs from the live ring"
                         f" for {len(bad)} matrices, e.g. {bad[:3]}")
    rings = {names_by_id[id(p)]: st["gradient_history"] for p, st in muon.state.items() if "gradient_history" in st}
    bad = nonfinite_tensors(rings)
    if bad:   # a rebuilt ring reads gradient files, which the shard scan above did not cover
        raise ValueError(f"restored ring holds non-finite values: {bad[:5]}; refusing to resume")
    state["ring_verified"] = len(got)
    for _ in range(shard["batches_consumed"] - state["train_batches"].consumed):
        next(state["train_batches"])
    restore_rng(shard["rng"])
    config["start_step"] = shard["anchor_update"]
    return shard["anchor_update"]


def install(config: dict[str, Any], hooks: dict[str, list]) -> None:
    """Attach recording to the existing loop; never create a second trainer.

    setup wraps the batch iterator and optionally restores an anchor; pre/post
    bracket optimizer.step() to record its inputs and realized outputs; teardown
    closes receipts. No capture block and no resume anchor means no hooks.
    """
    spec = capture_spec(config)
    resume = config.get("resume_anchor")
    if spec is None and not resume:
        return
    ctx: dict[str, Any] = {}

    def setup(config, state):
        state["train_batches"] = CountingBatches(state["train_batches"], config.get("start_step", 0))
        if resume:
            n = load_anchor_into(config, state, resume)
            state["print_log"](f"capture: resumed anchor {resume} -> start_step {n};"
                               f" ring sha256 verified for {state.get('ring_verified', 0)} matrices", console=True)
        if spec is None:
            return state
        ctx["store"] = CaptureWriter(spec["root"], state["rank"])
        ctx["root"] = Path(spec["root"])
        muon = _muon(state)
        L = HISTORY_LENGTH
        if spec["ring_length"] not in (None, L):
            raise ValueError("capture requires a fixed256 history")
        ctx["L"] = L
        ctx["grad_steps"] = gradient_steps(spec, L)
        ctx["win_steps"] = window_steps(spec)
        ctx["anchors"] = set(spec["anchors"])
        ctx["ringless"] = set(spec["ringless_anchors"])
        ctx["window_starts"] = {a for a, _ in spec["windows"]}
        ctx["names"] = {id(p): n for n, p in state["model"].named_parameters()}
        ctx["params"] = dict(state["model"].named_parameters())
        muon.capture_pre_polar = None
        muon.capture_post_polar = None
        if isinstance(muon, MuonH) and muon.radial_to_tangent_ratio is not None:
            raise ValueError("lag-0 subtraction requires untransformed linear momentum")
        if muon.capture_polar_observer is not None:
            raise ValueError("capture would overwrite an existing polar observer")
        ctx["local_muon_spec"] = spec.get("local_muon_operator")
        if ctx["local_muon_spec"]:
            from records.track_3_optimization.analysis.local_muon_capture import validate_spec
            ctx["local_muon_spec"] = validate_spec(ctx["local_muon_spec"], muon, ctx["names"])
            if type(muon) is not Muon:
                raise ValueError("local Muon operator requires plain untransformed Muon")
            if muon.momentum_observer is not None:
                raise ValueError("local Muon readout would overwrite a momentum observer")
        if state["rank"] == 0:
            ctx["root"].mkdir(parents=True, exist_ok=True)
            (ctx["root"] / "capture_config.json").write_text(json.dumps(
                dict(windows=spec["windows"], anchors=spec["anchors"], ring_length=L,
                     window_gradient_only=bool(spec.get("window_gradient_only", False)),
                     gradient_steps=[min(ctx["grad_steps"], default=None), max(ctx["grad_steps"], default=None)],
                     eval_batches=spec.get("eval_batches"), eval_batches_pinned=pin_eval_batches(config.get("val_data")),
                     train_shards=[dict(name=Path(f).name, path=f, tokens=n) for f, n in shard_sizes(config.get("train_data"))]
                     if config.get("train_data") else [],
                     directions_file=spec.get("directions_file"),
                     local_muon_operator=ctx["local_muon_spec"],
                     momentum_restoring_metrics=dict(
                         clock="update t: W before update, reduced g_t, post-insertion M_t",
                         coefficient="actual newest-first kernel tap; available-mass correction if enabled",
                         msign="trainer BF16 12-step Newton-Schulz, before shape scaling",
                         coverage="each owned Muon matrix, dense windows only; no cross-matrix aggregation"),
                     determinism_policy_installed=bool(getattr(torch, "_muoff_determinism_installed", False)),
                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                     conventions=__doc__), indent=1))
            ctx["norms"] = (ctx["root"] / "norms_rank0.jsonl").open("a", encoding="utf-8")
        return state

    def pre(config, state):
        if spec is None:
            return state
        t = state["step"]
        muon = _muon(state)
        if muon.param_groups[0]["iteration"] != t:
            raise ValueError("loop update and scheduler iteration differ")
        kernel = muon.momentum_scheduler(t)
        ctx["kernel_used"] = kernel.weights.detach().clone()
        ctx["normalize_available"] = bool(kernel.normalize_available)
        in_win = t in ctx["win_steps"]
        muon.capture_pre_polar = {} if in_win else None
        muon.capture_post_polar = {} if in_win else None
        ctx["restoring_metrics"] = {}
        def observe_polar(p, gradient, momentum, polar, actual_kernel, iteration, saved_state):
            if iteration != t:
                raise ValueError("restoring metric and optimizer clocks differ")
            ctx["restoring_metrics"][ctx["names"][id(p)]] = momentum_restoring_metrics(
                p, gradient, momentum, polar, actual_kernel,
                saved_state["gradient_filled"], muon.polar)
        muon.capture_polar_observer = observe_polar if in_win else None
        ctx["before"] = {n: p.detach().clone() for n, p in ctx["params"].items()} if in_win else None
        if ctx["local_muon_spec"]:
            from records.track_3_optimization.analysis.local_muon_capture import prepare_readout, readout_due
            muon.momentum_observer = prepare_readout(
                config, state, muon, ctx, in_win and readout_due(ctx["local_muon_spec"], t))
        ctx["anneal_frac_used"] = float(muon._anneal_frac()) if hasattr(muon, "_anneal_frac") else None
        # whole-run cheap per-matrix scalars: keep W_before (rank 0) and the gradient norm
        if state["rank"] == 0:
            ctx["norm_before"] = {n: (p.detach().clone() if ctx["before"] is None else ctx["before"][n],
                                      float(p.grad.detach().double().norm()))
                                  for n, p in ctx["params"].items() if p.ndim == 2}
        if in_win:
            tb = state["train_batches"]
            b = tb.consumed - 1
            args = (config.get("train_data"), int(config["batch_tokens"]))
            ctx["batch_receipts"] = dict(
                used=dict(batch_receipt(*args, b, state["rank"], state["world_size"]),
                          sha256_this_rank=batch_digest(tb.last)),
                next=dict(batch_receipt(*args, b + 1, state["rank"], state["world_size"]),
                          note="determined by loader arithmetic; digest is in the next boundary if captured"),
                loader_position_after_draw=tb.consumed)
        if t in ctx["grad_steps"] and state["rank"] == 0:
            payload = {n: p.grad.detach() for n, p in ctx["params"].items() if n.startswith(BLOCK_PREFIX) and p.ndim == 2}
            if (t + 1) in ctx["window_starts"]:
                # Adam replay into the window: anchor w0-1 + these non-block gradients + settings -> boundary w0
                payload.update({f"aux::{n}": p.grad.detach() for n, p in ctx["params"].items()
                                if not (n.startswith(BLOCK_PREFIX) and p.ndim == 2)})
                payload["aux_settings"] = [dict(pattern=sp.pattern, optimizer=sp.optimizer,
                                                lr=[float(g["lr"]) for g in b.param_groups],
                                                weight_decay=[float(g["weight_decay"]) for g in b.param_groups])
                                           for sp, b in state["optimizer"].groups]
            r = ctx["store"].save("gradient", t, payload)
            _index_append(ctx["root"], dict(update=t, kind="gradient", file=f"rank0/{r['file']}", sha256=r["sha256"], bytes=r["bytes"]))
        if in_win:
            ctx["aux_grads"] = {n: p.grad.detach().clone() for n, p in ctx["params"].items()
                                if not (n.startswith(BLOCK_PREFIX) and p.ndim == 2)}
            ctx["settings"] = [dict(pattern=spec_.pattern, optimizer=spec_.optimizer,
                                    groups=[{k: v for k, v in g.items() if k in ("lr", "weight_decay", "initial_lr", "mu", "betas", "eps")}
                                            for g in built.param_groups])
                               for spec_, built in state["optimizer"].groups]
        return state

    def post(config, state):
        if spec is None:
            return state
        t = state["step"]
        muon = _muon(state)
        rank = state["rank"]
        if rank == 0:
            row = dict(update=t, matrices={})
            for spec_, built in state["optimizer"].groups:
                g = built.param_groups[0]
                lr, wd = float(g["lr"]), float(g.get("weight_decay", 0.0))
                decay = 1.0 - lr * wd
                for p in g["params"]:
                    n = ctx["names"][id(p)]
                    if p.ndim != 2:
                        continue
                    wb, gnorm = ctx["norm_before"][n]
                    wb64 = wb.double(); delta = wb64 - p.detach().double()   # executed: subtracted, decay included
                    # displacement_minus_decay_norm = ||dW_exec + (1 - decay) * W_before|| with dW_exec = after - before
                    # (equivalently ||(before - after) - (1 - decay) * W_before||). This is the realized displacement
                    # with the decay term removed, NOT the optimizer's nominal pre-decay step (rounding differs).
                    # The exact nominal step lr*Q_t is not logged per update: the muon.py seam is active only inside
                    # windows and only on the owning rank; there lr*||Q_t|| follows from the boundary files.
                    disp_minus_decay = delta - (1.0 - decay) * wb64
                    row["matrices"][n] = dict(
                        group=f"{spec_.optimizer}:{spec_.pattern}", lr=lr, weight_decay=wd,
                        decay_factor=decay, weight_norm_before=float(wb64.norm()),
                        weight_norm_after=float(p.detach().double().norm()), grad_norm=gnorm,
                        executed_update_norm=float(delta.norm()),
                        displacement_minus_decay_norm=float(disp_minus_decay.norm()),
                        executed_update_dot_weight_before=float((delta * wb64).sum()))
            ctx["norms"].write(json.dumps(row) + "\n"); ctx["norms"].flush()
            ctx["norm_before"] = None
        if t in ctx["win_steps"]:
            names = ctx["names"]
            payload = dict(update=t, rank=rank,
                           momentum_restoring_metrics=ctx["restoring_metrics"],
                           local_muon_operator=ctx.get("local_muon_results", {}),
                           # POST-insertion: M_t already contains gradient t (not the boundary-t ring)
                           post_insertion_prepolar_momentum={names[k]: v for k, v in muon.capture_pre_polar.items()},
                           polar_output={names[k]: v for k, v in muon.capture_post_polar.items()},
                           format=ANCHOR_FORMAT,
                           muon_steps_seen_before=int(muon.param_groups[0]["iteration"]) - 1,
                           kernel_weights_used=ctx["kernel_used"],
                           normalize_available=ctx["normalize_available"],
                           anneal_frac_used=ctx["anneal_frac_used"],   # fraction the saved update used (pre-hook)
                           anneal_frac_next=float(muon._anneal_frac()) if hasattr(muon, "_anneal_frac") else None,
                           batch_receipts=ctx["batch_receipts"],
                           gradient_file=f"gradient_{t:06d}.pt (rank0)")
            if rank == 0:
                before = ctx["before"]
                payload.update(
                    model_before={k: v for k, v in before.items()},
                    buffers={k: v for k, v in state["model"].named_buffers()},
                    executed_update={n: before[n] - p.detach() for n, p in ctx["params"].items()},
                    aux_gradients=ctx["aux_grads"], settings=ctx["settings"])
            r = ctx["store"].save("boundary", t, payload)
            _index_append(ctx["root"], dict(update=t, kind="boundary", file=f"rank{rank}/{r['file']}", sha256=r["sha256"], bytes=r["bytes"]))
            muon.capture_pre_polar = muon.capture_post_polar = None
            muon.capture_polar_observer = None
            if ctx["local_muon_spec"]:
                from records.track_3_optimization.analysis.local_muon_capture import finish_readout
                finish_readout(ctx, t, rank)
                muon.momentum_observer = None
            metrics_row = dict(update=t, rank=rank, matrices=ctx["restoring_metrics"])
            with (ctx["root"] / f"momentum_restoring_rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics_row, sort_keys=True) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            ctx["before"] = None
        if (t + 1) in ctx["window_starts"]:
            # direct receipt of the optimizer state at the window's first boundary
            _index_append(ctx["root"], dict(
                update=t + 1, kind="optimizer_receipt", rank=rank, digests=optimizer_state_digest(state)))
        if (t + 1) in ctx["anchors"]:
            anchor = build_anchor(
                config, state, t + 1, ringless=(t + 1) in ctx["ringless"], stored=ctx["grad_steps"])
            bad = nonfinite_tensors(anchor)
            if bad:   # refuse before anything is written
                raise ValueError(f"anchor {t + 1} rank {rank}: non-finite tensors {bad[:5]} ({len(bad)} total); not written")
            r = ctx["store"].save("anchor", t + 1, anchor)
            _index_append(ctx["root"], dict(update=t + 1, kind="anchor", file=f"rank{rank}/{r['file']}", sha256=r["sha256"], bytes=r["bytes"]))
            state["print_log"](f"capture: anchor {t + 1} written", console=True)
        return state

    def teardown(config, state):
        if ctx.get("local_muon_spec"):
            _muon(state).momentum_observer = None
        if spec is not None and state.get("rank") == 0 and "norms" in ctx:
            ctx["norms"].close()
        if dist.is_initialized():
            dist.barrier()   # all ranks finished writing before exit (stop_step or full run)
        return state

    hooks["setup"].append(setup)
    hooks["teardown"].append(teardown)
    hooks["pre_optimizer"].append(pre)
    hooks["post_optimizer"].insert(0, post)


# ---------------------------------------------------------------------------
# dry-size
# ---------------------------------------------------------------------------

def dry_size(config: dict[str, Any]) -> dict[str, Any]:
    """Bytes per window / anchor / total, from parameter shapes on the meta device."""
    spec = capture_spec(config)
    with torch.device("meta"):
        model = GPT(**config["model"])
    shapes = {n: (p.numel(), p.element_size()) for n, p in model.named_parameters()}
    world = int(config["require_world_size"])
    L = ring_length_from_config(config)
    blocks = {n: v for n, v in shapes.items() if n.startswith(BLOCK_PREFIX) and len(model.get_parameter(n).shape) == 2}
    block_fp32 = sum(k * 4 for k, _ in blocks.values())
    full_param = sum(k * e for k, e in shapes.values())
    nonblock = full_param - sum(k * e for k, e in blocks.values())
    grad_file = block_fp32                                  # reduced block gradients, fp32
    boundary_rank0 = full_param * 2 + nonblock + 2 * block_fp32   # model_before + executed + aux grads + (M,Q) share
    boundary_all_ranks = boundary_rank0 + (world - 1) * (2 * block_fp32 / world)
    # anchor: every rank: model + Adam m,v (non-block) + full ring (owned block matrices, L rows fp32)
    ring_all = L * block_fp32
    adam = 2 * nonblock * 2  # m, v in param dtype (fp32 except embed bf16); bound with 2x
    anchor_all_ranks = world * full_param + world * adam + ring_all
    out = dict(ring_length=L, world_size=world, block_matrices=len(blocks),
               bytes_block_gradient_file=grad_file, bytes_boundary_per_update_all_ranks=int(boundary_all_ranks),
               bytes_anchor_all_ranks=int(anchor_all_ranks), windows=[])
    total = 0
    counted_grads: set[int] = set()
    for a, b in spec["windows"]:
        g = set(range(max(0, a - L), b + 1)) - counted_grads
        counted_grads |= g
        w = len(g) * grad_file + (b - a + 1) * boundary_all_ranks
        out["windows"].append(dict(window=[a, b], gradient_files=len(g), boundaries=b - a + 1, bytes=int(w)))
        total += w
    out["aux_gradients_before_each_window"] = dict(count=len(spec["windows"]), bytes_each=int(nonblock))
    total += len(spec["windows"]) * nonblock
    extra = set(gradient_steps(spec, L)) - counted_grads   # ring ranges of ringless anchors outside windows
    out["extra_gradient_files_for_ringless_anchors"] = dict(count=len(extra), updates=sorted(extra), bytes=len(extra) * grad_file)
    total += len(extra) * grad_file
    ringless = set(spec["ringless_anchors"])
    out["anchors"] = dict(
        with_ring=dict(updates=[a for a in spec["anchors"] if a not in ringless], bytes_each=int(anchor_all_ranks)),
        ringless=dict(updates=sorted(ringless), bytes_each=int(anchor_all_ranks - ring_all)))
    anchors_bytes = sum(anchor_all_ranks - (ring_all if a in ringless else 0) for a in spec["anchors"])
    out["anchors"]["bytes"] = int(anchors_bytes)
    total += anchors_bytes
    local = spec.get("local_muon_operator")
    if local:
        # Conservative all-block upper bound, two FP32 eigenvector components;
        # serialized scalars/JSON and Arnoldi workspace are extra, not tensors.
        cadence = int(local.get("every", 1))
        if cadence < 1:
            raise ValueError("local Muon cadence must be positive")
        bursts = local.get("windows")
        updates = {t for a, b in spec["windows"] for t in range(a, b + 1)
                   if t % cadence == 0 and (bursts is None or any(x <= t <= y for x, y in bursts))}
        extra_bytes = len(updates) * 2 * block_fp32
        out["local_muon_vectors_upper_bound"] = dict(updates=len(updates), bytes=extra_bytes,
             note="all blocks, real+imag FP32; metadata and Krylov workspace excluded")
        total += extra_bytes
    out["bytes_total"] = int(total)
    out["gib_total"] = round(total / 2**30, 1)
    return out
