#!/usr/bin/env python3
"""Back-fill a ledger entry from an already-captured stdout log (a run that
predates the harness). Same spec resolution + parser, so the ledger row is
identical in shape to a native run. Marks provenance.backfilled=true."""
import sys, os, json, hashlib, socket, datetime
KM = os.environ.get("KM_HOME", "/root/km")
REPO = os.environ.get("KM_REPO", "/root/modded-nanogpt")
RESDIR = "records/track_3_optimization/results/20260715_bimaxwell_baseline_3210"
PINS = json.load(open(f"{KM}/pins.json"))
sys.path.insert(0, KM)
from solve import resolve_spec, parse_log

spec = json.loads(sys.argv[1]); logfile = sys.argv[2]
name = spec["name"]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
th = sha(f"{REPO}/{RESDIR}/train_gpt_kmaxwell.py")
kh = sha(f"{REPO}/{RESDIR}/kmaxwell_kernel.py")
assert th == PINS["trainer_sha256"] and kh == PINS["kernel_sha256"], "pin mismatch at backfill"
rs = resolve_spec(spec)
rundir = f"{KM}/ledger/{name}"; os.makedirs(rundir, exist_ok=True)
prov = {"git_sha": PINS["git_sha"], "trainer_sha256": th, "kernel_sha256": kh,
        "host": socket.gethostname(), "backfilled": True,
        "backfill_ts": datetime.datetime.utcnow().isoformat() + "Z", "source_log": logfile}
json.dump({**rs, **prov}, open(f"{rundir}/spec.json", "w"), indent=2)
v = parse_log(logfile)
verdict = {"name": name, "seed": rs.get("seed", 0), "exit_code": 0,
           "val_3150": v["val_3150"], "val_3250": v["val_3250"],
           "train_time_3150": v["train_time_3150"], "n_val_points": v["n_val_points"],
           "mean_age": rs["mean_age"], "noise_gain": rs["noise_gain"], "k": rs["k"],
           "tmin": rs["tmin"], "tmax": rs["tmax"], "shape": rs["shape"],
           "status": "ok" if v["val_3150"] is not None else "FAILED", "backfilled": True}
json.dump({**verdict, "curve": v["curve"]}, open(f"{rundir}/verdict.json", "w"), indent=2)
print(f"backfilled {name}: val3150={v['val_3150']} val3250={v['val_3250']}")
