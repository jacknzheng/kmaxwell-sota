"""REQ-064 config + allocation generator. Reuses the REQ-058 base/continuation machinery and the REQ-059
AllocatedAnnealedWeightsMuon (per-matrix a-map). 39 arms/base around the frozen global a_star=0.5:
  - 3 globals: all_reference (a=0.5), all_shorter (0.25), all_longer (1.0)
  - 36 selective: each of 18 sentinels {blocks 0,6,11} x {attn.q,k,v,proj, mlp.fc,proj} set to 0.25 or 1.0,
    the other 71 matrices at 0.5.
Each continuation: fork at 2000, 256 completed updates (stop_after 2256), selection loss at 2000/2064/2128/2256.
Base: plain-Muon to the switch (2000) with a state dump (shared by all arms of that seed).
Usage: make_req064_configs.py --out DIR --seed S [--pilot]   (--pilot = seed-0 block-6 6 types x2 + 3 globals = 15 arms)
"""
import argparse, json, sys, yaml
from pathlib import Path
sys.path.insert(0, "/root/kmaxwell-sota/logs/kmaxwell/req058_layerwise_momentum_response/impl")
sys.path.insert(0, "logs/kmaxwell/req058_layerwise_momentum_response/impl")
import make_req058_configs as m

A_STAR = 0.5
A_SHORT, A_LONG = 0.25, 1.0
A_VALUES = [0.25, 0.5, 1.0]
TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]
BLOCKS = list(range(12))
SENTINEL_BLOCKS = [0, 6, 11]
FORK, UPDATES = 2000, 256
ALL72 = [f"blocks.{b}.{t}.weight" for b in BLOCKS for t in TYPES]


def build_alloc(pilot: bool) -> dict:
    """arm -> {matrix_name: a}. Every arm lists all 72 (unlisted would default to a=1.0)."""
    arms = {}
    arms["all_reference"] = {n: A_STAR for n in ALL72}
    arms["all_shorter"] = {n: A_SHORT for n in ALL72}
    arms["all_longer"] = {n: A_LONG for n in ALL72}
    sent_blocks = [6] if pilot else SENTINEL_BLOCKS
    for b in sent_blocks:
        for t in TYPES:
            sent = f"blocks.{b}.{t}.weight"; tag = f"b{b}_{t.replace('.', '')}"
            for suffix, a in (("short", A_SHORT), ("long", A_LONG)):
                arm = {n: A_STAR for n in ALL72}; arm[sent] = a
                arms[f"sel_{tag}_{suffix}"] = arm
    return arms


def base_config(seed):
    c = m.base_config(seed, FORK)
    for h in c["pre_optimizer"]:
        if h["name"] == "dump_training_state_at_steps":
            h["hyperparams"]["dump_dir"] = f"req064_state_s{seed}"
    c["run_id"] = f"req064_base_s{seed}"
    return c


def arm_config(seed, arm, alloc_file):
    blocks = {"pattern": r"^blocks\..*\.weight$", "optimizer": "allocated_annealed_weights_muon",
              "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95, "decays": m.DECAYS,
                              "start_weights": m.SW, "end_weights": m.EW, "switch_step": FORK,
                              "anneal_end_step": FORK + 750, "a_values": A_VALUES}}
    c = m.cont_config(seed, FORK, arm, blocks, None)  # reuse load_training_state + validate wiring
    for h in c["setup"]:
        if h.get("name") == "load_training_state":
            h["hyperparams"]["state_dir"] = f"req064_state_s{seed}"
    # replace the (absent) perturb-target tag with the REQ-059 allocation tag
    c["setup"].insert(-1, {"name": "tag_req059_allocation",
                           "hyperparams": {"alloc_file": alloc_file, "arm": arm}})
    c.update(start_step=FORK, stop_after_step=FORK + UPDATES)
    win = [FORK, FORK + UPDATES]
    for h in c["setup"]:
        if h.get("name") == "validate_at_step_boundaries":
            h["hyperparams"] = {"dense_window": win, "dense_every": 64}
    c["post_optimizer"] = [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"dense_window": win, "dense_every": 64}}]
    c["run_id"] = f"req064_{arm}_s{seed}"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    alloc = build_alloc(a.pilot)
    alloc_file = str(a.out / f"alloc_s{a.seed}.json")
    json.dump(alloc, open(alloc_file, "w"), indent=0)
    (a.out / f"base_s{a.seed}.yaml").write_text(yaml.safe_dump(base_config(a.seed), sort_keys=False))
    for arm in alloc:
        (a.out / f"{arm}_s{a.seed}.yaml").write_text(yaml.safe_dump(arm_config(a.seed, arm, alloc_file), sort_keys=False))
    print(f"wrote base + {len(alloc)} arms (seed {a.seed}, {'pilot' if a.pilot else 'full'}) -> {a.out}")
    # sanity: every arm lists exactly 72 matrices
    bad = {k: len(v) for k, v in alloc.items() if len(v) != 72}
    print("alloc sizes all 72:", not bad, ("" if not bad else bad))


if __name__ == "__main__":
    main()
