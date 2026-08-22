#!/usr/bin/env python3
"""Staged K-Maxwell sweeps. Default stages 1-2; 3-7 are follow-ups.

Logs go to logs/kmaxwell/<tag>/ (torchrun uuid log plus a tagged tee).

Examples (from repo root):
  python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/kmaxwell_sweep.py --stage 0
  python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/kmaxwell_sweep.py --stage 1
  python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/kmaxwell_sweep.py --stage 2 --k 4
  python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/kmaxwell_sweep.py --stage 3 --k 4
  python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/kmaxwell_sweep.py --stage 6 --k 4
  python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/kmaxwell_sweep.py --stage 7 --k 4 --weights 0.35,0.25,0.25,0.15 --delta 0.05
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kmaxwell_kernel import (
    endpoint_mass_family,
    format_weights,
    pairwise_mass_family,
    parse_weights,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TRAIN = HERE / "train_gpt_kmaxwell.py"

BIMAXWELL_TAU_MIN = 0.85 / (1.0 - 0.85)  # 5.666...
BIMAXWELL_TAU_MAX = 0.98 / (1.0 - 0.98)  # 49.0

STAGE1_K = (2, 3, 4, 6, 8, 12, 16)
STAGE2_TAU_MAX = (25, 49, 100, 200, 400)
STAGE3_SIGMA = (0.4, 0.8, 1.2, 1.6)
STAGE4_START = (0, 500, 1000)


def nproc_default():
    env = os.environ.get("NPROC_PER_NODE")
    if env:
        return int(env)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = len([ln for ln in out.splitlines() if ln.strip()])
        return n if n else 1
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 1


def run_one(extra, log_dir: Path, nproc: int, dry_run: bool):
    tag = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok == "--bimaxwell-exact":
            tag.append("exact")
            i += 1
            continue
        if tok.startswith("--"):
            tag.append(tok[2:] + str(extra[i + 1]))
            i += 2
            continue
        tag.append(tok)
        i += 1
    name = "_".join(tag).replace(".", "p") or "run"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "--", str(TRAIN), *map(str, extra),
    ]
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    tee_path = log_dir / f"{name}.stdout"
    with tee_path.open("w") as tee:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=tee, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"run failed ({proc.returncode}): {name}")


def _weight_cfg(seed, k, tau_min, tau_max, start, weights):
    return ["--seed", seed, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
            "--start", start, "--weights", format_weights(weights)]


def configs_for(stage: int, k: int, tau_max: float, sigma: float, start: int,
                seeds: list[int], weights=None, delta=0.1, steps=2, pairwise=False):
    """Return a list of argv tails for train_gpt_kmaxwell.py."""
    tau_min = BIMAXWELL_TAU_MIN
    if stage == 0:
        return [["--seed", 0, "--k", 2, "--bimaxwell-exact"]]
    if stage == 1:
        # denser ticks in the bi-Maxwell window; Gaussian weights (not exact)
        out = []
        for kk in STAGE1_K:
            out.append(["--seed", 0, "--k", kk, "--tau-min", tau_min, "--tau-max", BIMAXWELL_TAU_MAX,
                        "--sigma", 1.0, "--start", 1000])
        return out
    if stage == 2:
        out = []
        for tmax in STAGE2_TAU_MAX:
            out.append(["--seed", 0, "--k", k, "--tau-min", tau_min, "--tau-max", tmax,
                        "--sigma", 1.0, "--start", 1000])
        return out
    if stage == 3:
        # follow-up: Gaussian blend width at a chosen (K, tau_max)
        out = []
        for s in STAGE3_SIGMA:
            out.append(["--seed", 0, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
                        "--sigma", s, "--start", 1000])
        return out
    if stage == 4:
        # follow-up: switch step at a chosen kernel
        out = []
        for st in STAGE4_START:
            out.append(["--seed", 0, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
                        "--sigma", sigma, "--start", st])
        return out
    if stage == 5:
        # follow-up: multi-seed confirm of a chosen kernel
        out = []
        extra = ["--weights", format_weights(weights)] if weights is not None else ["--sigma", sigma]
        for seed in seeds:
            out.append(["--seed", seed, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
                        "--start", start, *extra])
        return out
    if stage == 6:
        # K=4 default: even mix, then move n*delta between fastest and slowest
        families = endpoint_mass_family(k, delta, steps)
        if pairwise:
            seen = {tuple(round(w, 8) for w in fam) for fam in families}
            for nxt in pairwise_mass_family([1.0 / k] * k, delta):
                key = tuple(round(w, 8) for w in nxt)
                if key not in seen:
                    seen.add(key)
                    families.append(nxt)
        return [_weight_cfg(0, k, tau_min, tau_max, start, w) for w in families]
    if stage == 7:
        # refine around a chosen mix with a smaller delta (default: pairwise ±delta)
        base = weights if weights is not None else [1.0 / k] * k
        families = pairwise_mass_family(base, delta) if pairwise or weights is not None else endpoint_mass_family(k, delta, steps)
        return [_weight_cfg(0, k, tau_min, tau_max, start, w) for w in families]
    raise ValueError(f"unknown stage {stage}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--k", type=int, default=4, help="K for stages 2-7 (stage 1 grids K)")
    p.add_argument("--tau-max", type=float, default=BIMAXWELL_TAU_MAX, help="tau_max for stages 3-7")
    p.add_argument("--sigma", type=float, default=1.0, help="sigma for stages 3-5 Gaussian kernels")
    p.add_argument("--start", type=int, default=1000, help="start for stages 5-7")
    p.add_argument("--seeds", default="0,1,2", help="comma seeds for stage 5 (use 0,1,2,3,4,5,6,7 to confirm)")
    p.add_argument("--weights", default=None, help="comma mix weights for stages 5 and 7")
    p.add_argument("--delta", type=float, default=0.1, help="mass-transfer step for stages 6-7")
    p.add_argument("--steps", type=int, default=2, help="how many ±n*delta endpoint shifts (stage 6-7)")
    p.add_argument("--pairwise", action="store_true",
                   help="stage 6/7: also try every ordered-pair ±delta transfer")
    p.add_argument("--log-dir", default="logs/kmaxwell")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    nproc = nproc_default()
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO / log_dir
    weights = parse_weights(args.weights) if args.weights else None
    cfgs = configs_for(args.stage, args.k, args.tau_max, args.sigma, args.start, seeds,
                       weights=weights, delta=args.delta, steps=args.steps, pairwise=args.pairwise)
    print(f"stage {args.stage}: {len(cfgs)} run(s), nproc={nproc}")
    for cfg in cfgs:
        run_one(cfg, log_dir / f"stage{args.stage}", nproc, args.dry_run)


if __name__ == "__main__":
    main()
