#!/usr/bin/env python3
"""WR-stack runner v2 — CWD (#46 stack) frozen K-Maxwell + PR #339 bi-Maxwell.

Records the FULL val/val_ema curve (the record metric is a statsig mean over
seeds on a 5-step grid, NOT first-cross; first-cross is kept as a diagnostic
only). Produces durable named logs matching the Track 3 logs/kmaxwell layout:
  <repo>/logs/kmaxwell/<fleet>/seed<N>.stdout   (torchrun stdout, every step)
  <repo>/logs/kmaxwell/<fleet>/seed<N>.txt      (trainer uuid logfile: source dump + steps)
plus the km ledger entry (spec.json, cmd.txt, train.log, verdict.json).

spec: {name, seed, trainer: "cwd"|"bm339", fleet, k?, tmin?, tmax?, age?, start?}
"""
import sys, os, json, argparse, subprocess, socket, datetime, re, hashlib, shutil

KM = os.environ.get("KM_HOME", "/root/km")
REPO = os.environ.get("KM_REPO", "/root/modded-nanogpt")
VENV = os.environ.get("KM_VENV", "/root/venv")
WRDIR = "records/track_3_optimization/results/20260823_kmaxwell_wr_stack"
BMDIR = "records/track_3_optimization/results/20260713_bimaxwell_2635"
TRAINERS = {
    "cwd":   f"{WRDIR}/train_gpt_cwd_kmaxwell.py",
    "bm339": f"{BMDIR}/train_gpt_bimaxwell_st1000.py",
}
PIN_KEYS = {"cwd": "cwd_trainer_sha256", "bm339": "bm339_trainer_sha256"}
TAIL_GRID = list(range(2580, 2725, 5))   # the 5-step scoring grid

sys.path.insert(0, KM)
from solve import solve_weights

def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

a = argparse.ArgumentParser()
a.add_argument("--spec"); a.add_argument("--specfile"); a.add_argument("--stop", default="2720")
ar = a.parse_args()
spec = json.loads(open(ar.specfile).read() if ar.specfile else ar.spec)
name = spec["name"]; seed = int(spec.get("seed", 0)); tname = spec.get("trainer", "cwd")
fleet = spec.get("fleet", "cwd_frozen_n8" if tname == "cwd" else "bimaxwell339_n8")
trainer = TRAINERS[tname]

# ---- pin gate: refuse to launch on hash mismatch ----
pins = json.load(open(f"{KM}/pins_wr.json"))
got = sha256(f"{REPO}/{trainer}")
want = pins[PIN_KEYS[tname]]
if got != want:
    print(f"PIN MISMATCH {trainer}: {got} != {want}", file=sys.stderr); sys.exit(74)
if tname == "cwd":
    kgot = sha256(f"{REPO}/{WRDIR}/kmaxwell_kernel.py")
    if kgot != pins["kernel_sha256"]:
        print(f"PIN MISMATCH kernel: {kgot}", file=sys.stderr); sys.exit(74)

# ---- build command ----
if tname == "cwd":
    k = int(spec.get("k", 6)); tmin = float(spec.get("tmin", 3)); tmax = float(spec.get("tmax", 56))
    age = float(spec.get("age", 35)); start = int(spec.get("start", 1000))
    w = solve_weights(k, tmin, tmax, age, "linear"); ws = ",".join(f"{x:.6f}" for x in w)
    train_args = (f"--seed {seed} --start {start} --k {k} --tau-min {tmin} "
                  f"--tau-max {tmax} --weights {ws}")
else:
    ws = None
    train_args = f"--seed {seed}"
cmd = (f"cd {REPO} && STOP_STEP={ar.stop} {VENV}/bin/torchrun --standalone --nproc_per_node=8 "
       f"--master_port=29513 -- {trainer} {train_args}")

rundir = f"{KM}/ledger/{name}"; os.makedirs(rundir, exist_ok=True)
fleetdir = f"{REPO}/logs/kmaxwell/{fleet}"; os.makedirs(fleetdir, exist_ok=True)
now = datetime.datetime.utcnow().isoformat() + "Z"
json.dump({**spec, "weights": ws, "trainer_sha256": got, "host": socket.gethostname(),
           "started": now, "cmd": cmd}, open(f"{rundir}/spec.json", "w"), indent=2)
open(f"{rundir}/cmd.txt", "w").write(cmd + "\n")

logp = f"{rundir}/train.log"
with open(logp, "w") as lf:
    lf.write(f"==== {now} {name} trainer={tname} seed={seed} ====\n{cmd}\n"); lf.flush()
    rc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT).returncode

# ---- parse the full trace ----
VAL_RE = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+)(?: val_ema_loss:([\d.]+))? train_time")
STEP_RE = re.compile(r"step:(\d+)/\d+ train_time")
uuid_path = None
val_curve, ema_curve = {}, {}
n_step_lines = 0; max_step = 0
cross_v = cross_e = None
for line in open(logp, errors="replace"):
    line = line.strip()
    if uuid_path is None and re.fullmatch(r"logs/[0-9a-f-]+\.txt", line):
        uuid_path = line
    m = VAL_RE.search(line)
    if m:
        st = int(m.group(1)); vl = float(m.group(2))
        val_curve[st] = vl
        if m.group(3):
            ve = float(m.group(3)); ema_curve[st] = ve
            if cross_e is None and ve < 3.28: cross_e = st
        if cross_v is None and vl < 3.28: cross_v = st
        max_step = max(max_step, st)
    elif STEP_RE.search(line):
        n_step_lines += 1
        max_step = max(max_step, int(STEP_RE.search(line).group(1)))

# ---- copy durable named logs ----
stable_txt = f"{fleetdir}/seed{seed}.txt"
stable_out = f"{fleetdir}/seed{seed}.stdout"
uuid_ok = False
if uuid_path and os.path.exists(f"{REPO}/{uuid_path}"):
    shutil.copyfile(f"{REPO}/{uuid_path}", stable_txt); uuid_ok = True
shutil.copyfile(logp, stable_out)

# ---- validity per protocol ----
missing_tail = [s for s in TAIL_GRID if s not in ema_curve]
problems = []
if rc != 0: problems.append(f"exit_code={rc}")
if max_step < int(ar.stop): problems.append(f"stopped_early_at_{max_step}")
if missing_tail: problems.append(f"missing_ema_tail:{missing_tail[:5]}{'...' if len(missing_tail)>5 else ''}")
if n_step_lines < int(ar.stop) - 5: problems.append(f"step_lines_only_{n_step_lines}")
if not uuid_ok: problems.append("uuid_logfile_missing")

verdict = {"name": name, "trainer": tname, "fleet": fleet, "seed": seed, "exit_code": rc,
           "cross_step_val": cross_v, "cross_step_ema": cross_e,   # diagnostics ONLY, not the metric
           "ema_tail": {s: ema_curve[s] for s in TAIL_GRID if s in ema_curve},
           "ema_curve": ema_curve, "val_curve": val_curve,
           "uuid_logfile": uuid_path, "stable_txt": stable_txt, "stable_stdout": stable_out,
           "status": "ok" if not problems else "INVALID", "problems": problems}
json.dump(verdict, open(f"{rundir}/verdict.json", "w"), indent=2)
print("VERDICT " + json.dumps({k: verdict[k] for k in
      ("name", "trainer", "seed", "exit_code", "status", "problems", "cross_step_ema")}))
