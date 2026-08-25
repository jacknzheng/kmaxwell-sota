#!/usr/bin/env python3
"""Annealed-mix run: interpolate the K-EMA mix weights from a start-age vector
to an end-age vector over training. Reproducible: pins the original kernel +
the anneal trainer by content hash; weights derived from (k,window,age0,age1).

spec: {name,seed,k,tmin,tmax,age_start,age_end,shape?,anneal_frac?,start?}
"""
import sys, os, json, argparse, subprocess, socket, datetime, hashlib, re, shutil
KM = os.environ.get("KM_HOME", "/root/km")
REPO = os.environ.get("KM_REPO", "/root/modded-nanogpt")
VENV = os.environ.get("KM_VENV", "/root/venv")
RESDIR = "records/track_3_optimization/results/20260715_bimaxwell_baseline_3210"
KERNEL = f"{RESDIR}/kmaxwell_kernel.py"
TRAINER = f"{RESDIR}/train_gpt_kmaxwell_anneal.py"
PINS = json.load(open(f"{KM}/pins.json"))
PINS_A = json.load(open(f"{KM}/pins_anneal.json"))
sys.path.insert(0, KM)
from solve import solve_weights, stats_for, parse_log

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def sh(c): return subprocess.run(c, shell=True, text=True, capture_output=True).stdout.strip()

a = argparse.ArgumentParser(); a.add_argument("--spec"); a.add_argument("--specfile")
ar = a.parse_args(); spec = json.loads(open(ar.specfile).read() if ar.specfile else ar.spec)
name = spec["name"]; seed = spec.get("seed", 0); start = spec.get("start", 1000)
k = int(spec["k"]); tmin = float(spec["tmin"]); tmax = float(spec["tmax"])
shape = spec.get("shape", "linear"); af = float(spec.get("anneal_frac", 1.0))
a0 = float(spec["age_start"]); a1 = float(spec["age_end"])

# reproducibility gate
assert sha(f"{REPO}/{KERNEL}") == PINS["kernel_sha256"], "kernel pin mismatch"
assert sha(f"{REPO}/{TRAINER}") == PINS_A["anneal_trainer_sha256"], "anneal trainer pin mismatch"

w0 = solve_weights(k, tmin, tmax, a0, shape)
w1 = solve_weights(k, tmin, tmax, a1, shape)
st0 = stats_for(k, tmin, tmax, w0); st1 = stats_for(k, tmin, tmax, w1)
w0s = ",".join(f"{x:.6f}" for x in w0); w1s = ",".join(f"{x:.6f}" for x in w1)

rundir = f"{KM}/ledger/{name}"; os.makedirs(rundir, exist_ok=True)
now = datetime.datetime.utcnow().isoformat() + "Z"
spec_out = {**spec, "weights_start": w0s, "weights_end": w1s,
            "age_start": a0, "age_end": a1, "anneal_frac": af,
            "git_sha": PINS["git_sha"], "anneal_trainer_sha256": PINS_A["anneal_trainer_sha256"],
            "host": socket.gethostname(), "started": now, "trainer": TRAINER,
            "noise_gain_start": round(st0["noise_gain"], 6), "noise_gain_end": round(st1["noise_gain"], 6)}
json.dump(spec_out, open(f"{rundir}/spec.json", "w"), indent=2)

cmd = (f"cd {REPO} && {VENV}/bin/torchrun --standalone --nproc_per_node=8 --master_port=29511 -- "
       f"{TRAINER} --seed {seed} --start {start} --k {k} --tau-min {tmin} --tau-max {tmax} "
       f"--weights {w0s} --weights-end {w1s} --anneal-frac {af}")
open(f"{rundir}/cmd.txt", "w").write(cmd + "\n")
logpath = f"{rundir}/train.log"
with open(logpath, "w") as lf:
    lf.write(f"==== {now} {name} ANNEAL seed={seed} age {a0}->{a1} frac={af} ====\n{cmd}\n"); lf.flush()
    rc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT).returncode
v = parse_log(logpath)
# durable named logs (Track 3 layout): stdout + trainer uuid source-dump logfile
fleet = spec.get("fleet", "ablation_anneal_n8")
fleetdir = f"{REPO}/logs/kmaxwell/{fleet}"; os.makedirs(fleetdir, exist_ok=True)
uuid_path = None
for line in open(logpath, errors="replace"):
    if re.fullmatch(r"logs/[0-9a-f-]+\.txt", line.strip()):
        uuid_path = line.strip(); break
shutil.copyfile(logpath, f"{fleetdir}/{name}_seed{seed}.stdout")
uuid_ok = False
if uuid_path and os.path.exists(f"{REPO}/{uuid_path}"):
    shutil.copyfile(f"{REPO}/{uuid_path}", f"{fleetdir}/{name}_seed{seed}.txt"); uuid_ok = True
verdict = {"name": name, "seed": seed, "exit_code": rc, "val_3150": v["val_3150"], "val_3250": v["val_3250"],
           "uuid_logfile": uuid_path, "uuid_copied": uuid_ok,
           "k": k, "tmin": tmin, "tmax": tmax, "shape": f"anneal{a0:.0f}-{a1:.0f}", "mean_age": (a0+a1)/2,
           "noise_gain": round((st0["noise_gain"]+st1["noise_gain"])/2, 6),
           "anneal": True, "age_start": a0, "age_end": a1,
           "status": "ok" if rc == 0 and v["val_3150"] is not None else "FAILED"}
json.dump({**verdict, "curve": v["curve"]}, open(f"{rundir}/verdict.json", "w"), indent=2)
print("VERDICT " + json.dumps(verdict))
