"""REQ-063 B6 verified control pilot config generator. Reuses the REQ-058 base/continuation machinery
(base K-Maxwell to the switch with a state dump; continuations fork via load_training_state and run 64
updates). Adds the explicitly-named **ordinary Muon, mu=0.95** control (distinct from nomom mu=0, the
eight-stream mixture, and the exact-age single-EMA control).

Arms (all fork at the switch step, 64 updates, disjoint val at offsets 0/8/../64):
  a1all      : K-Maxwell eight-stream mixture (a=1 baseline)
  nomom      : no-momentum Muon (mu=0)   <-- with the REQ-063 restore fix this actually runs mu=0
  ordmuon095 : ordinary Muon, mu=0.95    <-- named control; before the fix nomom collapses onto this
  matched    : exact-age single-EMA control

Run the harness WITH impl/apply_req063_restorefix.py applied so the restored nomom arm keeps mu=0.
Usage: make_req063_pilot_configs.py --out configs/req063 --seed S --fork F
"""
import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0, "/root/kmaxwell-sota/logs/kmaxwell/req058_layerwise_momentum_response/impl")
sys.path.insert(0, "logs/kmaxwell/req058_layerwise_momentum_response/impl")
import make_req058_configs as m


def ordmuon_group():  # named ordinary Muon control, mu=0.95
    return {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon",
            "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fork", type=int, default=2000)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    s, f = a.seed, a.fork
    # base: reuse REQ-058 base (dumps state at the switch = fork), but rename the state dir to req063
    base = m.base_config(s, f)
    for h in base["pre_optimizer"]:
        if h["name"] == "dump_training_state_at_steps":
            h["hyperparams"]["dump_dir"] = f"req063_state_s{s}_f{f}"
    base["run_id"] = f"req063_base_s{s}_f{f}"
    (a.out / f"base_s{s}_f{f}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))

    arms = [("a1all", m.blocks_group("kmax", f, f + 750), None),
            ("nomom", m.blocks_group("nomom", f, f + 750), None),
            ("ordmuon095", ordmuon_group(), None),
            ("matched", m.blocks_group("matched", f, f + 750), None)]
    for label, blocks, tgt in arms:
        c = m.cont_config(s, f, label, blocks, tgt)
        # point the restore at the req063 state dir
        for h in c["setup"]:
            if h.get("name") == "load_training_state":
                h["hyperparams"]["state_dir"] = f"req063_state_s{s}_f{f}"
        c["run_id"] = f"req063_{label}_s{s}_f{f}"
        (a.out / f"{label}_s{s}_f{f}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"wrote base + {len(arms)} pilot arms (incl. named ordinary-Muon mu=0.95) for seed{s} fork{f} -> {a.out}")


if __name__ == "__main__":
    main()
