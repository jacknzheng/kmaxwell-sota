#!/usr/bin/env python3
"""Regroup wave/stage logs by k / tau-window / mean-age. Dry-run unless --apply."""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/Users/jackzheng/ML/modded-nanogpt")
LOGS = ROOT / "logs"
KM = LOGS / "kmaxwell"
sys.path.insert(0, str(ROOT / "records/track_3_optimization/results/20260715_bimaxwell_baseline_3210"))
from kmaxwell_kernel import build_kmaxwell_kernel  # noqa: E402

HEADER = re.compile(
    r"--k\s+(\d+)|--tau-min\s+([0-9.]+)|--tau-max\s+([0-9.]+)|--weights\s+([0-9.,]+)|--taus\s+([0-9.,]+)|--start\s+(\d+)|--bimaxwell-exact"
)
FNAME_K = re.compile(r"_k(\d+)_")
FNAME_TMIN = re.compile(r"tau-min([0-9p]+)")
FNAME_TMAX = re.compile(r"tau-max([0-9p]+)")
FNAME_W = re.compile(r"weights([0-9p,]+)")


def pfloat(s: str) -> float:
    return float(s.replace("p", "."))


def pretty_tau(x: float) -> str:
    if abs(x - 5.666666666666666) < 0.03:
        return "5.67"
    if abs(x - 1.9270521156385474) < 0.03:
        return "1.93"
    if abs(x - 2.5) < 0.05:
        return "2.5"
    known = [2, 3, 4, 8, 16, 25, 49, 56, 64, 72, 80]
    for k in known:
        if abs(x - k) < 0.05:
            return str(int(k))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def pretty_age(age: float) -> str:
    return f"a{int(round(age))}"


def parse_header(path: Path) -> dict:
    info = {}
    try:
        text = path.read_text(errors="replace")[:4000]
    except Exception:
        return info
    for line in text.splitlines()[:8]:
        if "--k" in line or "tau-min" in line or "bimaxwell" in line or "taus" in line:
            km = re.search(r"--k\s+(\d+)", line)
            if km:
                info["k"] = int(km.group(1))
            tm = re.search(r"--tau-min\s+([0-9.]+)", line)
            if tm:
                info["tmin"] = float(tm.group(1))
            tx = re.search(r"--tau-max\s+([0-9.]+)", line)
            if tx:
                info["tmax"] = float(tx.group(1))
            w = re.search(r"--weights\s+([0-9.eE,-]+)", line)
            if w:
                info["weights"] = [float(x) for x in w.group(1).split(",") if x]
            taus = re.search(r"--taus\s+([0-9.,]+)", line)
            if taus:
                info["taus"] = [float(x) for x in taus.group(1).split(",") if x]
            if "--bimaxwell-exact" in line or "k2_exact" in path.name:
                info["exact"] = True
            st = re.search(r"--start\s+(\d+)", line)
            if st:
                info["start"] = int(st.group(1))
    return info


def parse_fname(path: Path) -> dict:
    n = path.name
    info = {}
    m = FNAME_K.search(n)
    if m:
        info["k"] = int(m.group(1))
    m = FNAME_TMIN.search(n)
    if m:
        info["tmin"] = pfloat(m.group(1))
    m = FNAME_TMAX.search(n)
    if m:
        info["tmax"] = pfloat(m.group(1))
    m = FNAME_W.search(n)
    if m:
        info["weights"] = [pfloat(x) for x in m.group(1).split(",") if x]
    if "k2_exact" in n or "bimaxwell-exact" in n:
        info["exact"] = True
        info["k"] = 2
    return info


# Explicit wave recipes when the filename is a short alias.
WAVE = {
    "A36_age36.stdout": dict(k=4, tmin=5.666666666666666, tmax=49.0, age=36, tag="a36"),
    "A42_age42.stdout": dict(k=4, tmin=5.666666666666666, tmax=49.0, age=42, tag="a42"),
    "A48_age48.stdout": dict(k=4, tmin=5.666666666666666, tmax=49.0, age=48, tag="a48"),
    "B56_tmax56.stdout": dict(k=4, tmin=5.666666666666666, tmax=56.0, weights=[0.05, 0.20, 0.25, 0.50]),
    "B64_tmax64.stdout": dict(k=4, tmin=5.666666666666666, tmax=64.0, weights=[0.05, 0.20, 0.25, 0.50]),
    "DEC_tmin3_tmax56.stdout": dict(k=4, tmin=3.0, tmax=56.0, weights=[0.05, 0.20, 0.25, 0.50]),
    "K6a33.stdout": dict(k=6, tmin=5.666666666666666, tmax=49.0, age=33, tag="a33"),
    "K6a37.stdout": dict(k=6, tmin=5.666666666666666, tmax=49.0, age=37, tag="a37"),
    "MIX_midheavy_tmax56.stdout": dict(k=4, tmin=5.666666666666666, tmax=56.0, weights=[0.05, 0.25, 0.40, 0.30], tag="mix_midheavy"),
    "DEC2.stdout": dict(k=4, tmin=2.0, tmax=56.0, weights=[0.05, 0.20, 0.25, 0.50]),
    "DEC4.stdout": dict(k=4, tmin=4.0, tmax=56.0, weights=[0.05, 0.20, 0.25, 0.50]),
    "INT_spread.stdout": dict(k=4, taus=[3, 4.5, 23.8, 56], weights=[0.05, 0.20, 0.25, 0.50], age=35, tag="INT_spread"),
    "INT_mid.stdout": dict(k=4, taus=[3, 11, 18.6, 56], weights=[0.05, 0.20, 0.25, 0.50], age=35, tag="INT_mid"),
    "INT_bunch.stdout": dict(k=4, taus=[3, 13.5, 16.6, 56], weights=[0.05, 0.20, 0.25, 0.50], age=35, tag="INT_bunch"),
    "K6_a35.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=35, tag="a35"),
    "START800.stdout": dict(k=4, tmin=3.0, tmax=56.0, weights=[0.05, 0.20, 0.25, 0.50], start=800, tag="start800"),
    "SOAP_s0.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=35, tag="SOAP_s0", stack=True),
    "SOAP_DEC_s0.stdout": dict(k=4, tmin=3.0, tmax=56.0, weights=[0.05, 0.20, 0.25, 0.50], tag="SOAP_DEC_s0", stack=True),
    "aurora_s0.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=35, tag="aurora_s0", stack=True),
    "adam_s0.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=35, tag="adam_s0", stack=True),
    "K6_a32.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=32, tag="a32"),
    "K6_a36.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=36, tag="a36"),
    "K6_a37.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=37, tag="a37"),
    "K6_a38.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=38, tag="a38"),
    "K6_a39.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=39, tag="a39"),
    "K6_ws40.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=35, tag="mix_ws40"),
    "K6_ws50.stdout": dict(k=6, tmin=3.0, tmax=56.0, age=35, tag="mix_ws50"),
    "K6_tmin2.stdout": dict(k=6, tmin=2.0, tmax=56.0, age=35, tag="a35"),
    "K6_a38_tmin2.stdout": dict(k=6, tmin=2.0, tmax=56.0, age=38, tag="a38"),
}


def mean_age(info: dict) -> float | None:
    if "age" in info:
        return float(info["age"])
    if info.get("exact"):
        _, _, _, age = build_kmaxwell_kernel(2, 5.666666666666666, 49.0, 1.0, bimaxwell_exact=True)
        return age
    k = info.get("k")
    w = info.get("weights")
    if info.get("taus") and w and k:
        tau = info["taus"]
        ww = [x / sum(w) for x in w]
        return sum(a * b for a, b in zip(ww, tau))
    tmin, tmax = info.get("tmin"), info.get("tmax")
    if k and tmin and tmax:
        _, _, _, age = build_kmaxwell_kernel(k, tmin, tmax, 1.0, weights=w)
        return age
    return None


def dest_for(path: Path, info: dict) -> Path:
    k = info.get("k")
    if not k:
        return KM / "_unsorted" / path.name
    if info.get("stack"):
        age = pretty_age(mean_age(info) or 0)
        win = f"t{pretty_tau(info['tmin'])}-{pretty_tau(info['tmax'])}"
        return KM / "stacks" / f"k{k}_{win}_{age}" / path.name
    if info.get("taus"):
        age = pretty_age(mean_age(info) or 35)
        return KM / f"k{k}" / f"interiors_{age}" / path.name
    if info.get("exact"):
        return KM / "k2" / "exact_bimaxwell" / path.name
    tmin, tmax = info.get("tmin"), info.get("tmax")
    if tmin is None or tmax is None:
        return KM / f"k{k}" / "_unknown_window" / path.name
    win = f"t{pretty_tau(tmin)}-{pretty_tau(tmax)}"
    age = mean_age(info)
    age_dir = pretty_age(age) if age is not None else "a?"
    tag = info.get("tag")
    # mix shots that hold age but change shape
    if tag and tag.startswith("mix_"):
        return KM / f"k{k}" / win / tag / path.name
    if tag == "start800":
        return KM / f"k{k}" / win / f"{age_dir}_start800" / path.name
    return KM / f"k{k}" / win / age_dir / path.name


def collect_moves():
    moves = []
    skip_parts = {"wr", "anneal_sweeps", "_hunts", "stacks"}
    for path in sorted(LOGS.rglob("*.stdout")):
        rel = path.relative_to(LOGS).parts
        if rel[0] in skip_parts or (rel[0] == "kmaxwell" and len(rel) > 1 and rel[1] in skip_parts):
            continue
        if rel[0] not in {"wave1", "wave2", "wave3", "wave4", "wave4_kill", "wave5", "kmaxwell"}:
            continue
        if rel[0] == "kmaxwell" and rel[1] not in {f"stage{i}" for i in range(9)}:
            continue
        info = dict(WAVE.get(path.name, {}))
        for src in (parse_fname(path), parse_header(path)):
            for k, v in src.items():
                info.setdefault(k, v)
        dst = dest_for(path, info)
        if dst.resolve() != path.resolve():
            moves.append((path, dst, info))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    moves = collect_moves()
    for src, dst, info in moves:
        age = mean_age(info)
        print(f"{src.relative_to(LOGS)}  ->  {dst.relative_to(LOGS)}  k={info.get('k')} age={age and round(age,2)}")
    docs = []
    for wave in ("wave1", "wave2", "wave3", "wave4", "wave4_kill", "wave5"):
        d = LOGS / wave
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix in {".txt", ".sh", ".md"} or p.name in {"PLAN.txt", "DECISION.txt"}:
                docs.append((p, KM / "_hunts" / wave / p.name))
    print(f"\n{len(moves)} log moves, {len(docs)} hunt docs")
    if not args.apply:
        print("dry-run; pass --apply to move")
        return
    for src, dst, _ in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(src.parent.name + "_" + dst.name)
        shutil.move(str(src), str(dst))
    for src, dst in docs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    # drop empty dirs
    for d in sorted(LOGS.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            if d.name.startswith("stage") or d.name.startswith("wave") or d.name == "kmaxwell_sota":
                d.rmdir()
                print("rmdir", d.relative_to(LOGS))
    for wave in ("wave1", "wave2", "wave3", "wave4", "wave4_kill", "wave5", "kmaxwell_sota"):
        d = LOGS / wave
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            print("rmdir", wave)
        elif d.is_dir():
            print("left", wave, list(d.iterdir())[:8])


if __name__ == "__main__":
    main()
