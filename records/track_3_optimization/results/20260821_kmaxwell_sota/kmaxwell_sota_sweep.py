#!/usr/bin/env python3
"""Staged K-Maxwell sweeps on the #46 SOTA trainer.

Logs go to logs/kmaxwell_sota/<tag>/ (torchrun uuid log plus a tagged tee).
Stage 0 is the PR #339 identity (--k 2 --bimaxwell-exact).

Examples (from repo root):
  python3 records/track_3_optimization/results/20260821_kmaxwell_sota/kmaxwell_sota_sweep.py --stage 0
  python3 records/track_3_optimization/results/20260821_kmaxwell_sota/kmaxwell_sota_sweep.py --stage 1
  python3 records/track_3_optimization/results/20260821_kmaxwell_sota/kmaxwell_sota_sweep.py --stage 2 --k 4
  python3 records/track_3_optimization/results/20260821_kmaxwell_sota/kmaxwell_sota_sweep.py --stage 3 --k 4 --tau-max 200
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TRAIN = HERE / "train_gpt_kmaxwell_sota.py"

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


def configs_for(stage: int, k: int, tau_max: float, sigma: float, start: int, seeds: list[int]):
    """Return a list of argv tails for train_gpt_kmaxwell_sota.py."""
    tau_min = BIMAXWELL_TAU_MIN
    if stage == 0:
        return [["--seed", 0, "--k", 2, "--bimaxwell-exact"]]
    if stage == 1:
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
        out = []
        for s in STAGE3_SIGMA:
            out.append(["--seed", 0, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
                        "--sigma", s, "--start", 1000])
        return out
    if stage == 4:
        out = []
        for st in STAGE4_START:
            out.append(["--seed", 0, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
                        "--sigma", sigma, "--start", st])
        return out
    if stage == 5:
        out = []
        for seed in seeds:
            out.append(["--seed", seed, "--k", k, "--tau-min", tau_min, "--tau-max", tau_max,
                        "--sigma", sigma, "--start", start])
        return out
    raise ValueError(f"unknown stage {stage}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3, 4, 5])
    p.add_argument("--k", type=int, default=4, help="K for stages 2-5 (stage 1 grids K)")
    p.add_argument("--tau-max", type=float, default=BIMAXWELL_TAU_MAX, help="tau_max for stages 3-5")
    p.add_argument("--sigma", type=float, default=1.0, help="sigma for stages 4-5")
    p.add_argument("--start", type=int, default=1000, help="start for stage 5")
    p.add_argument("--seeds", default="0,1,2", help="comma seeds for stage 5 (use 0,1,2,3,4,5,6,7 to confirm)")
    p.add_argument("--log-dir", default="logs/kmaxwell_sota")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    nproc = nproc_default()
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO / log_dir
    cfgs = configs_for(args.stage, args.k, args.tau_max, args.sigma, args.start, seeds)
    print(f"stage {args.stage}: {len(cfgs)} run(s), nproc={nproc}")
    for cfg in cfgs:
        run_one(cfg, log_dir / f"stage{args.stage}", nproc, args.dry_run)


if __name__ == "__main__":
    main()
