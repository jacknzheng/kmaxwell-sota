"""REQ-056: install KMaxwellAdam into the optimizer package and register it as 'kmaxwell_adam'.
Idempotent. Run from repo root (records/track_3_optimization on sys.path via CWD). Reads the module source
from /root/kmaxwell_adam_src.py (scp'd there) and copies it into optimizers/kmaxwell_adam.py."""
import os, shutil, sys

OPT = "records/track_3_optimization/optimizers"
SRC = "/root/kmaxwell_adam_src.py"
DST = f"{OPT}/kmaxwell_adam.py"

assert os.path.isfile(SRC), f"missing {SRC} (scp kmaxwell_adam.py first)"
shutil.copyfile(SRC, DST)

init = f"{OPT}/__init__.py"
txt = open(init).read()
changed = False
if "from .kmaxwell_adam import KMaxwellAdam" not in txt:
    anchor = "_REGISTRY"
    i = txt.index(anchor)
    txt = txt[:i] + "from .kmaxwell_adam import KMaxwellAdam\n\n" + txt[i:]
    changed = True
if '"kmaxwell_adam":' not in txt:
    key = '"adamw": build_record_adamw,'
    assert key in txt, "could not find adamw registry line to anchor insertion"
    txt = txt.replace(key, key + '\n    "kmaxwell_adam": KMaxwellAdam,', 1)
    changed = True
if changed:
    open(init, "w").write(txt)
    print("patched __init__.py (import + registry)")
else:
    print("already patched")

# --- patch hooks.py so the fork dump/load tolerate configs with NO Muon-family group ---
HK = "records/track_3_optimization/harness/hooks.py"
h = open(HK).read()
hchanged = False
dump_old = ('        _, muon = find_muon_family_group(state["optimizer"])\n'
            '        payload = dict(step=step, rank=state["rank"], world_size=state["world_size"],')
dump_new = ('        try:\n'
            '            _, muon = find_muon_family_group(state["optimizer"])\n'
            '        except ValueError:\n'
            '            muon = None\n'
            '        payload = dict(step=step, rank=state["rank"], world_size=state["world_size"],')
if dump_old in h:
    h = h.replace(dump_old, dump_new, 1); hchanged = True
load_old = ('        _, muon = find_muon_family_group(state["optimizer"])\n'
            '        muon._muon_steps_seen = shard["muon_steps_seen"]')
load_new = ('        try:\n'
            '            _, muon = find_muon_family_group(state["optimizer"])\n'
            '            muon._muon_steps_seen = shard["muon_steps_seen"]\n'
            '        except ValueError:\n'
            '            pass')
if load_old in h:
    h = h.replace(load_old, load_new, 1); hchanged = True
if hchanged:
    open(HK, "w").write(h)
    print("patched hooks.py (dump/load tolerate zero Muon groups)")
else:
    print("hooks.py already patched (or anchors absent)")

# sanity import
sys.path.insert(0, "records/track_3_optimization")
from optimizers import _REGISTRY  # noqa
assert "kmaxwell_adam" in _REGISTRY, "registration failed"
print("OK: kmaxwell_adam registered ->", _REGISTRY["kmaxwell_adam"].__name__)
