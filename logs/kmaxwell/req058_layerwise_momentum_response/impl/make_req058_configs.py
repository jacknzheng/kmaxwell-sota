"""REQ-058 config generator: base (K-Maxwell kernel, cloned-buffer switch at fork) + continuations
(41 full; 17 pilot). Continuations fork via load_training_state, swap the blocks optimizer to the memory
intervention, run 64 updates, and record selection-loss on a disjoint 131072-tok val set at offsets 0/8/../64.
Usage: make_req058_configs.py --out configs/req058 --seed S --fork F [--pilot]
"""
import argparse, yaml
from pathlib import Path

DECAYS = [0.75, 0.822852439855, 0.877930338626, 0.917598547218,
          0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
SW = [0.005093975, 0.010187949, 0.015281924, 0.020375898, 0.025469873, 0.030563847, 0.035657822, 0.857368713]
EW = [0.032261839, 0.064523678, 0.096785516, 0.129047355, 0.161309194, 0.193571033, 0.225832871, 0.096668514]
TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]
PILOT_BLOCK = 6  # one matrix of each type at mid-depth block 6
UPDATES = 64
TRAIN_STEPS = 3250  # keep the original cool_down horizon


def kmax_group(switch, anneal_end):
    return {"pattern": r"^blocks\..*\.weight$", "optimizer": "annealed_weights_muon",
            "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95, "decays": DECAYS,
                            "start_weights": SW, "end_weights": EW,
                            "switch_step": switch, "anneal_end_step": anneal_end}}


def blocks_group(kind, switch, anneal_end, perturb_a=None, perturb_all=False):
    if kind == "kmax":       # a=1 baseline mixture
        return kmax_group(switch, anneal_end)
    if kind == "perturbed":
        hp = {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95, "decays": DECAYS, "start_weights": SW,
              "end_weights": EW, "switch_step": switch, "anneal_end_step": anneal_end,
              "perturb_a": perturb_a, "perturb_all": perturb_all}
        return {"pattern": r"^blocks\..*\.weight$", "optimizer": "perturbed_annealed_weights_muon", "hyperparams": hp}
    if kind == "matched":
        return {"pattern": r"^blocks\..*\.weight$", "optimizer": "exact_age_matched_muon",
                "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95, "decays": DECAYS,
                                "start_weights": SW, "end_weights": EW, "switch_step": switch,
                                "anneal_end_step": anneal_end, "nu": 0.95, "n_steps": TRAIN_STEPS}}
    if kind == "nomom":      # no-momentum Muon = baseline Muon with mu=0
        return {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon",
                "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.0}}
    raise ValueError(kind)


def common(run_id, seed):
    return {"loop": "gpt_record", "run_id": run_id, "seed": seed, "require_world_size": 8,
            "train_steps": TRAIN_STEPS, "batch_tokens": 524288, "microbatch_sequences": 64,
            "train_data": "data/fineweb10B/fineweb_train_*.bin",
            "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
            "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
            "setup": [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                      {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                      {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                      {"name": "broadcast_initial_parameters"}, {"name": "validate_at_step_boundaries"}],
            "post_optimizer": [{"name": "print_training_progress"}],
            "teardown": [{"name": "mark_log_finished"}]}


def std_groups(blocks):
    return [{"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            blocks,
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.001}}]


def base_config(seed, fork):
    anneal_end = fork + 750
    c = common(f"req058_base_s{seed}_f{fork}", seed)
    c["optimizer_groups"] = std_groups(kmax_group(fork, anneal_end))
    c.update(stop_after_step=fork)
    c["pre_optimizer"] = [
        {"name": "dump_training_state_at_steps", "hyperparams": {"steps": [fork], "dump_dir": f"req058_state_s{seed}_f{fork}"}},
        {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    return c


def cont_config(seed, fork, label, blocks, perturb_target=None):
    anneal_end = fork + 750
    run_id = f"req058_{label}_s{seed}_f{fork}"
    c = common(run_id, seed)
    c["optimizer_groups"] = std_groups(blocks)
    c.update(start_step=fork, stop_after_step=fork + UPDATES)
    # selection-loss on a disjoint 131072-tok val set at offsets fork..fork+64 every 8
    c["val_data"] = "data/fineweb10B/fineweb_val_*.bin"
    c["setup"].insert(-1, {"name": "load_training_state",
                           "hyperparams": {"state_dir": f"req058_state_s{seed}_f{fork}", "step": fork, "skip_batches": fork}})
    if perturb_target is not None:
        c["setup"].insert(-1, {"name": "tag_req058_perturb_target", "hyperparams": {"target_name": perturb_target}})
    c["setup"][-1] = {"name": "validate_at_step_boundaries",
                      "hyperparams": {"dense_window": [fork, fork + UPDATES], "dense_every": 8}}
    c["pre_optimizer"] = [{"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    c["post_optimizer"] = [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"dense_window": [fork, fork + UPDATES], "dense_every": 8}}]
    c["val_tokens"] = 524288  # min feasible = world*mbs*seq = 8*64*1024; 131072 infeasible at mbs64/world8
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fork", type=int, required=True); ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    s, f = a.seed, a.fork
    (a.out / f"base_s{s}_f{f}.yaml").write_text(yaml.safe_dump(base_config(s, f), sort_keys=False))
    conts = []
    # shared a=1-all control
    conts.append(("a1all", blocks_group("kmax", f, f + 750), None))
    sentinel_blocks = [PILOT_BLOCK] if a.pilot else [0, 6, 11]
    for blk in sentinel_blocks:
        for t in TYPES:
            mname = f"blocks.{blk}.{t}.weight"; tt = f"b{blk}_{t.replace('.', '')}"
            conts.append((f"a05_{tt}", blocks_group("perturbed", f, f + 750, perturb_a=0.5), mname))
            conts.append((f"a2_{tt}", blocks_group("perturbed", f, f + 750, perturb_a=2.0), mname))
    conts.append(("a05all", blocks_group("perturbed", f, f + 750, perturb_a=0.5, perturb_all=True), "all"))
    conts.append(("a2all", blocks_group("perturbed", f, f + 750, perturb_a=2.0, perturb_all=True), "all"))
    conts.append(("nomom", blocks_group("nomom", f, f + 750), None))
    conts.append(("matched", blocks_group("matched", f, f + 750), None))
    for label, blocks, tgt in conts:
        (a.out / f"{label}_s{s}_f{f}.yaml").write_text(yaml.safe_dump(cont_config(s, f, label, blocks, tgt), sort_keys=False))
    print(f"wrote base + {len(conts)} continuations for seed{s} fork{f} -> {a.out}")


if __name__ == "__main__":
    main()
