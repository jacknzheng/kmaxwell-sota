#!/usr/bin/env python3
"""Sequential queue driver. Reads a queue file (one JSON spec per line, '#'
comments ok), runs each via kmrun.py, appends a row to ledger/INDEX.tsv, and
touches ledger/<queue>.DONE when drained. Idempotent: a run whose verdict.json
already exists with status ok is skipped (safe to re-run a queue)."""
import sys, os, json, subprocess, datetime

KM = os.environ.get("KM_HOME", "/root/km")
LEDGER = f"{KM}/ledger"
INDEX = f"{LEDGER}/INDEX.tsv"
COLS = ["ts", "name", "seed", "k", "tmin", "tmax", "shape", "mean_age",
        "noise_gain", "val_3150", "val_3250", "exit_code", "status"]


def append_index(v):
    os.makedirs(LEDGER, exist_ok=True)
    new = not os.path.exists(INDEX)
    with open(INDEX, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(v.get(c, "")) for c in COLS) + "\n")


def already_done(name):
    p = f"{LEDGER}/{name}/verdict.json"
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("status") == "ok"
    except Exception:
        return False


def main():
    qfile = sys.argv[1]
    qname = os.path.basename(qfile).replace(".jsonl", "").replace(".txt", "")
    specs = []
    for line in open(qfile):
        line = line.strip()
        if line and not line.startswith("#"):
            specs.append(json.loads(line))
    print(f"[queue {qname}] {len(specs)} specs", flush=True)
    for i, spec in enumerate(specs, 1):
        name = spec["name"]
        if already_done(name):
            print(f"[queue {qname}] ({i}/{len(specs)}) {name} SKIP (done)", flush=True)
            continue
        print(f"[queue {qname}] ({i}/{len(specs)}) {name} START {datetime.datetime.utcnow().isoformat()}Z", flush=True)
        runner = "kmwr.py" if "trainer" in spec else ("kmanneal.py" if "age_end" in spec else "kmrun.py")
        r = subprocess.run([sys.executable, f"{KM}/{runner}", "--spec", json.dumps(spec)],
                           capture_output=True, text=True)
        vline = [l for l in r.stdout.splitlines() if l.startswith("VERDICT ")]
        if vline:
            v = json.loads(vline[-1][len("VERDICT "):])
            v["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
            append_index(v)
            print(f"[queue {qname}] ({i}/{len(specs)}) {name} DONE "
                  f"val3150={v.get('val_3150')} val3250={v.get('val_3250')} status={v.get('status')}", flush=True)
        else:
            print(f"[queue {qname}] ({i}/{len(specs)}) {name} NO-VERDICT rc={r.returncode}\n{r.stdout[-500:]}\n{r.stderr[-500:]}", flush=True)
    open(f"{LEDGER}/{qname}.DONE", "w").write(datetime.datetime.utcnow().isoformat() + "Z\n")
    print(f"[queue {qname}] DRAINED", flush=True)


if __name__ == "__main__":
    main()
