#!/usr/bin/env python3
"""Single reproducible kmaxwell run: pin-check -> resolve spec -> torchrun ->
parse -> ledger. One run == one directory under /root/km/ledger/<name>/ with
spec.json (inputs + derived weights/stats + provenance), train.log, verdict.json.

Usage:
  kmrun.py --name K8_a38 --seed 0 --k 8 --tmin 3 --tmax 56 --age 38 --shape linear
  kmrun.py --spec '{"name":"K8_a38","seed":0,"k":8,"tmin":3,"tmax":56,"age":38,"shape":"linear"}'
"""
import sys, os, json, argparse, subprocess, socket, datetime, hashlib

KM = os.environ.get("KM_HOME", "/root/km")
REPO = os.environ.get("KM_REPO", "/root/modded-nanogpt")
VENV = os.environ.get("KM_VENV", "/root/venv")
RESDIR = "records/track_3_optimization/results/20260715_bimaxwell_baseline_3210"
TRAINER = f"{RESDIR}/train_gpt_kmaxwell.py"
KERNEL = f"{RESDIR}/kmaxwell_kernel.py"
PINS = json.load(open(f"{KM}/pins.json"))
sys.path.insert(0, KM)
from solve import resolve_spec, parse_log  # noqa: E402


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw).stdout.strip()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--name"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int); ap.add_argument("--tmin", type=float)
    ap.add_argument("--tmax", type=float); ap.add_argument("--age", type=float)
    ap.add_argument("--shape"); ap.add_argument("--weights"); ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--port", type=int, default=29511)
    a = ap.parse_args()

    spec = json.loads(a.spec) if a.spec else {
        "name": a.name, "seed": a.seed, "k": a.k, "tmin": a.tmin, "tmax": a.tmax,
        "age": a.age, "shape": a.shape, "weights": a.weights, "start": a.start}
    spec.setdefault("seed", 0); spec.setdefault("start", 1000)
    name = spec["name"]

    # 1. reproducibility gate: the two files that define the update must be
    #    byte-identical to the pinned versions (content hash, not git state)
    th, kh = sha256(f"{REPO}/{TRAINER}"), sha256(f"{REPO}/{KERNEL}")
    if th != PINS["trainer_sha256"]:
        sys.exit(f"PIN MISMATCH: trainer {th} != {PINS['trainer_sha256']}")
    if kh != PINS["kernel_sha256"]:
        sys.exit(f"PIN MISMATCH: kernel {kh} != {PINS['kernel_sha256']}")

    # 2. resolve weights + filter stats from the 5 knobs (derived, not typed)
    rs = resolve_spec(spec)

    rundir = f"{KM}/ledger/{name}"
    os.makedirs(rundir, exist_ok=True)
    now = datetime.datetime.utcnow().isoformat() + "Z"
    provenance = {"git_sha": PINS["git_sha"], "trainer_sha256": th, "kernel_sha256": kh,
                  "host": socket.gethostname(),
                  "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader -i 0"),
                  "torch": sh(f"{VENV}/bin/python -c 'import torch;print(torch.__version__)'"),
                  "started": now, "trainer": TRAINER}
    full = {**rs, **provenance}
    with open(f"{rundir}/spec.json", "w") as f:
        json.dump(full, f, indent=2)

    # 3. dispatch the exact torchrun command
    cmd = (f"cd {REPO} && {VENV}/bin/torchrun --standalone --nproc_per_node={a.nproc} "
           f"--master_port={a.port} -- {TRAINER} --seed {rs['seed']} --start {rs['start']} "
           f"--k {rs['k']} --tau-min {rs['tmin']} --tau-max {rs['tmax']} --weights {rs['weights']}")
    with open(f"{rundir}/cmd.txt", "w") as f:
        f.write(cmd + "\n")
    logpath = f"{rundir}/train.log"
    with open(logpath, "w") as lf:
        lf.write(f"==== {now} {name} seed={rs['seed']} trainer_sha={th[:8]} "
                 f"mean_age={rs['mean_age']} noise={rs['noise_gain']} ====\n")
        lf.write(cmd + "\n")
        lf.flush()
        rc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT).returncode

    # 4. verdict
    v = parse_log(logpath)
    verdict = {"name": name, "seed": rs["seed"], "exit_code": rc,
               "val_3150": v["val_3150"], "val_3250": v["val_3250"],
               "train_time_3150": v["train_time_3150"], "n_val_points": v["n_val_points"],
               "mean_age": rs["mean_age"], "noise_gain": rs["noise_gain"], "k": rs["k"],
               "tmin": rs["tmin"], "tmax": rs["tmax"], "shape": rs["shape"],
               "finished": datetime.datetime.utcnow().isoformat() + "Z",
               "status": "ok" if rc == 0 and v["val_3150"] is not None else "FAILED"}
    with open(f"{rundir}/verdict.json", "w") as f:
        json.dump({**verdict, "curve": v["curve"]}, f, indent=2)
    print("VERDICT " + json.dumps(verdict))
    return 0 if verdict["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
