"""REQ-072 config generator: 6 optimizer arms, each capturing 3 streams for blocks.{0,5,11}.attn.proj.weight.
AdamW/SGD run over the full model; Muon runs on hidden matrices + auxiliary (plain) AdamW on embed/head/
scalars. Weight decay = 0 in all arms: the conditioned direction u_t (the spectrogram source) is captured
BEFORE lr and decay, so decay cannot affect the primary analysis; wd=0 removes the coupled/decoupled confound.
LR per optimizer is set from a short stability pilot (--lr_*). 3250 steps, standard cooldown, forks @500/1500/2500.

Usage: make_req072_configs.py --out_dir DIR --dur DUR [--lr_adamw 0.002 --lr_sgd 0.02 --lr_muon 0.025] [--pilot_steps N]
"""
import argparse, yaml
from pathlib import Path

SELECTED = [f"blocks.{b}.attn.proj.weight" for b in (0, 5, 11)]
FORKS = [500, 1500, 2500]


def common(run_id, seed, train_steps):
    return {"loop": "gpt_record", "run_id": run_id, "seed": seed, "require_world_size": 8,
            "train_steps": train_steps, "batch_tokens": 524288, "microbatch_sequences": 64,
            "train_data": "data/fineweb10B/fineweb_train_*.bin",
            "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
            "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768}}


def groups_for(arm, lr_adamw, lr_sgd, lr_muon):
    if arm.startswith("adamw"):
        b1 = 0.9 if arm.endswith("mom") and "nomom" not in arm else 0.0
        return [{"pattern": ".*", "optimizer": "capture_adamw",
                 "hyperparams": {"lr": lr_adamw, "weight_decay": 0.0, "betas": [b1, 0.95], "eps": 1e-10}}]
    if arm.startswith("sgd"):
        mom = 0.9 if "nomom" not in arm else 0.0
        return [{"pattern": ".*", "optimizer": "sgd_blocks",
                 "hyperparams": {"lr": lr_sgd, "weight_decay": 0.0, "momentum": mom}}]
    if arm.startswith("muon"):
        mu = 0.95 if "nomom" not in arm else 0.0
        return [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.0}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.0}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon", "hyperparams": {"lr": lr_muon, "weight_decay": 0.0, "mu": mu}},
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.0}},
        ]
    raise ValueError(arm)


def build(arm, seed, dur, lr_adamw, lr_sgd, lr_muon, train_steps, capture):
    c = common(f"req072_{arm}_s{seed}", seed, train_steps)
    c["optimizer_groups"] = groups_for(arm, lr_adamw, lr_sgd, lr_muon)
    c["setup"] = [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                  {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                  {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                  {"name": "broadcast_initial_parameters"}]
    if capture:
        a = f"{dur}/{arm}"
        c["setup"].append({"name": "tag_req072_capture",
                           "hyperparams": {"names": SELECTED, "grad_dir": f"{a}/grad",
                                           "uT_dir": f"{a}/uT", "disp_dir": f"{a}/disp", "chunk_steps": 500}})
    c["setup"].append({"name": "validate_at_step_boundaries"})
    c["pre_optimizer"] = ([{"name": "req072_pre"}] if capture else []) + \
                         [{"name": "dump_training_state_at_steps", "hyperparams": {"steps": FORKS, "dump_dir": f"{dur}/{arm}/forks"}}] * (1 if capture else 0) + \
                         [{"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    c["post_optimizer"] = ([{"name": "req072_post"}] if capture else []) + \
                          [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries",
                            "hyperparams": {"every": 125, "final_tenth_every": 25}}]
    c["teardown"] = ([{"name": "finalize_req072"}] if capture else []) + [{"name": "mark_log_finished"}]
    return c


ARMS = ["adamw-mom", "adamw-nomom", "sgd-mom", "sgd-nomom", "muon-mom", "muon-nomom"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--dur", required=True)
    ap.add_argument("--lr_adamw", type=float, default=0.002); ap.add_argument("--lr_sgd", type=float, default=0.02)
    ap.add_argument("--lr_muon", type=float, default=0.025); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot_steps", type=int, default=0, help=">0 => short no-capture pilot configs for LR screening")
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    ts = a.pilot_steps if a.pilot_steps > 0 else 3250
    capture = a.pilot_steps == 0
    for arm in ARMS:
        cfg = build(arm, a.seed, a.dur, a.lr_adamw, a.lr_sgd, a.lr_muon, ts, capture)
        (a.out_dir / f"{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {len(ARMS)} arm configs ({'pilot '+str(ts)+'-step no-capture' if not capture else '3250-step capture'}) "
          f"lr adamw/sgd/muon={a.lr_adamw}/{a.lr_sgd}/{a.lr_muon} -> {a.out_dir}")


if __name__ == "__main__":
    main()
