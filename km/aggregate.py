#!/usr/bin/env python3
"""Central-commander aggregation. Pulls per-run ledger dirs from every box,
merges into a single local INDEX.tsv, and renders SCOREBOARD.md.

  python3 aggregate.py pull          # scp ledger/<name>/ from all boxes -> local
  python3 aggregate.py render        # rebuild INDEX.tsv + SCOREBOARD.md from local dirs

Boxes are read from BOXES below. Reproducibility anchor per run is the
content-hash pin recorded in each spec.json (trainer_sha256/kernel_sha256).
"""
import os, sys, json, subprocess, glob

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/jackzhengretardruns
LEDGER = os.path.join(HOME, "ledger")
BOXES = {
    "A": "training-job-wlmz82q-0.ssh.baseten.co",
    "B": "training-job-3y51enw-0.ssh.baseten.co",
    "C": "training-job-wxgj90q-0.ssh.baseten.co",
    "D": "training-job-q974r03-0.ssh.baseten.co",
    "E": "training-job-qrgxkr3-0.ssh.baseten.co",
    "F": "training-job-qz7dvew-0.ssh.baseten.co",
}
TARGET = 3.28  # val_loss @ step 3150

SSH = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def pull():
    os.makedirs(LEDGER, exist_ok=True)
    for tag, host in BOXES.items():
        # list run dirs that have a verdict
        r = subprocess.run(["ssh", *SSH, host,
                            "ls /root/km/ledger/*/verdict.json 2>/dev/null"],
                           capture_output=True, text=True)
        names = [p.split("/")[-2] for p in r.stdout.split()]
        if not names:
            print(f"[{tag} {host}] no verdicts")
            continue
        print(f"[{tag} {host}] {len(names)} runs: {' '.join(names)}")
        for n in names:
            dst = os.path.join(LEDGER, n)
            os.makedirs(dst, exist_ok=True)
            for fn in ("spec.json", "verdict.json", "cmd.txt", "train.log"):
                subprocess.run(["scp", "-q", *SSH, f"{host}:/root/km/ledger/{n}/{fn}", dst + "/"],
                               capture_output=True)


def load_runs():
    runs = []
    for vpath in glob.glob(os.path.join(LEDGER, "*", "verdict.json")):
        try:
            v = json.load(open(vpath))
            sp = os.path.join(os.path.dirname(vpath), "spec.json")
            v["_spec"] = json.load(open(sp)) if os.path.exists(sp) else {}
        except Exception as e:
            print("skip", vpath, e); continue
        runs.append(v)
    return runs


def render():
    runs = load_runs()
    scored = [r for r in runs if r.get("val_3150") is not None]
    scored.sort(key=lambda r: r["val_3150"])
    cols = ["name", "k", "mean_age", "shape", "tmin", "tmax", "noise_gain",
            "val_3150", "val_3250", "seed", "status"]
    with open(os.path.join(LEDGER, "INDEX.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(runs, key=lambda r: (r.get("val_3150") or 9)):
            f.write("\t".join(str(r.get(c, r.get("_spec", {}).get(c, ""))) for c in cols) + "\n")
    lines = ["# kmaxwell scoreboard", "",
             f"Target: **val_loss < {TARGET} @ step 3150**. "
             f"Pin: trainer sha256 `0d97226d…`, kernel `38c1d65d…` (content-hash reproducible).",
             f"{len(scored)} scored runs.", "",
             "| rank | name | k | age | shape | noise_gain | **val@3150** | val@3250 | Δ vs target |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(scored, 1):
        d = r["val_3150"] - TARGET
        mark = "✅" if d < 0 else ""
        lines.append(f"| {i} | {r['name']} | {r['k']} | {r.get('mean_age','')} | {r.get('shape','')} "
                     f"| {r.get('noise_gain','')} | **{r['val_3150']:.5f}** {mark} "
                     f"| {r.get('val_3250','')} | {d:+.5f} |")
    best = scored[0] if scored else None
    if best:
        lines += ["", f"**Best:** {best['name']} — {best['val_3150']:.5f} @3150 "
                  f"({'UNDER' if best['val_3150']<TARGET else 'over'} target by {abs(best['val_3150']-TARGET):.5f}), "
                  f"{best.get('val_3250','?')} @3250."]
    open(os.path.join(HOME, "SCOREBOARD.md"), "w").write("\n".join(lines) + "\n")
    print(f"rendered {len(scored)} runs -> SCOREBOARD.md; best="
          f"{best['name']} {best['val_3150']:.5f}" if best else "no scored runs")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    if cmd == "pull":
        pull(); render()
    else:
        render()
