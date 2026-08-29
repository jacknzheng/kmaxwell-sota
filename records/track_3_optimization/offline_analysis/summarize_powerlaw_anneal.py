"""Summarize scheduled-power-law runs from the harness's UUID-named logs.

The screen is same-state and deterministic, so the primary comparison is final
validation loss against the bi-Maxwell fork. We also report the mean paired
difference over the dense cooldown window (steps >= 2900), which distinguishes
a persistent trajectory advantage from a lucky final checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RUN = re.compile(r"^run_id:\s*['\"]?([^'\"\s]+)", re.MULTILINE)
VAL = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)")


def parse_logs(log_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in log_dir.glob("*.txt"):
        text = path.read_text(errors="replace")
        match = RUN.search(text)
        if not match or not match.group(1).startswith(
                ("plann_", "pl_anneal_", "expann_")):
            continue
        run_id = match.group(1)
        vals = {int(step): float(loss) for step, _, loss in VAL.findall(text)}
        # A retry may leave a shorter log for the same run_id. Prefer the log
        # with the furthest validation boundary, then the most measurements.
        candidate = {"path": str(path), "values": vals,
                     "finished": "\nrun finished:" in text.lower()}
        old = runs.get(run_id)
        score = (max(vals, default=-1), len(vals), candidate["finished"])
        old_score = ((max(old["values"], default=-1), len(old["values"]),
                      old["finished"]) if old else (-1, -1, False))
        if score > old_score:
            runs[run_id] = candidate
    return runs


def paired_mean_delta(values: dict[int, float], control: dict[int, float],
                      first_step: int = 2900) -> tuple[float | None, int]:
    steps = sorted(set(values) & set(control))
    steps = [step for step in steps if step >= first_step]
    if not steps:
        return None, 0
    return sum(values[s] - control[s] for s in steps) / len(steps), len(steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/tmp/plann_configs/plann_manifest.json"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    runs = parse_logs(args.log_dir)
    controls = {
        "bimaxwell": runs.get("plann_bimaxwell_control", {}).get("values", {}),
        "pr357": runs.get("plann_pr357_kernel_control", {}).get("values", {}),
        "pr359": runs.get("plann_pr359_kernel_control", {}).get("values", {}),
    }
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {"runs": {}}
    rows = []
    for run_id, item in runs.items():
        values = item["values"]
        final_step = max(values, default=None)
        final_loss = values.get(final_step) if final_step is not None else None
        final_controls = {
            name: vals.get(final_step) if final_step is not None else None
            for name, vals in controls.items()
        }
        late_delta, late_n = paired_mean_delta(values, controls["bimaxwell"])
        key = run_id.removeprefix("plann_")
        kernel = manifest.get("runs", {}).get(key, {})
        rows.append({
            "run_id": run_id,
            "final_step": final_step,
            "final_loss": final_loss,
            # Positive means the candidate power-law run is better.
            **{
                f"improvement_vs_{name}": (
                    baseline - final_loss
                    if final_loss is not None and baseline is not None else None
                )
                for name, baseline in final_controls.items()
            },
            "late_mean_delta_vs_bimaxwell": late_delta,
            "late_paired_points": late_n,
            "gamma_1000": kernel.get("gamma_1000"),
            "gamma_end": kernel.get("gamma_end"),
            "finished": item["finished"],
            "log": item["path"],
        })
    rows.sort(key=lambda row: (row["final_loss"] is None,
                               row["final_loss"] if row["final_loss"] is not None else 99,
                               row["run_id"]))
    fields = list(rows[0]) if rows else ["run_id"]
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields,
                            delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
