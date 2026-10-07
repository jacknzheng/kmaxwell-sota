"""REQ-076 config generator: step-1000 dumper, 256-update 5-arm pilot, and 3250-update 5-arm x 3-seed screen.

Common recipe = verified REQ-073 small-batch B (524,288 tok/update, Muon lr 0.025, cool_down 0.7, aux AdamW
embed 0.7 / head 0.004 / catch-all 0.015, wd=0). Filter arms overwrite the raw hidden-matrix gradient
(req076_filter) with momentum OFF (mu=0); standard-mom keeps mu=0.95; nomom-halfLR is mu=0 at half Muon LR.

Modes:
  dump1000 : standard-momentum baseline from scratch to step 1000, dump full state (for the pilot).
  pilot    : 5 arms from the step-1000 state (load_training_state), 256 updates, filter from the 8-update
             startup, capture every update (raw/filtered/muon-input/post-polar/disp) for the 3 diagnostics.
  screen   : 5 arms x 3 seeds from scratch, 3250 updates, filter from startup, windowed capture
             (500-755/1500-1755/2500-2755). Seeds are independently initialized full runs.

Usage: make_req076_configs.py --mode {dump1000,pilot,screen} --out_dir DIR --dur DUR [--fork_dir STATE1000]
"""
import argparse, yaml
from pathlib import Path

B = 524288
SEL = ["blocks.5.attn.q.weight", "blocks.5.attn.k.weight", "blocks.5.mlp.proj.weight"]
WINDOWS = [[500, 755], [1500, 1755], [2500, 2755]]
# arm: (mu, filter_kind, muon_lr_mult, clear_buffer_on_load)
ARMS = {
    "nomom":        (0.0,  "none",      1.0, True),
    "pair":         (0.0,  "pair",      1.0, True),
    "fir9":         (0.0,  "fir9",      1.0, True),
    "nomom-halfLR": (0.0,  "none",      0.5, True),
    "standard-mom": (0.95, "none",      1.0, False),
}
MUON_LR = 0.025


def groups(mu, lr):
    return [{"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.0}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.0}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon", "hyperparams": {"lr": lr, "weight_decay": 0.0, "mu": mu}},
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.0}}]


def common(run_id, seed, start_step, train_steps, mu, lr):
    return {"loop": "gpt_record", "run_id": run_id, "seed": seed, "require_world_size": 8,
            "start_step": start_step, "train_steps": train_steps, "batch_tokens": B, "microbatch_sequences": 64,
            "train_data": "data/fineweb10B/fineweb_train_*.bin", "val_data": "data/fineweb10B/fineweb_val_*.bin",
            "val_tokens": 10485760, "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
            "optimizer_groups": groups(mu, lr)}


def build_dump1000(dur):
    c = common("req076_dump1000", 0, 0, 1001, 0.95, MUON_LR)
    c["setup"] = [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                  {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                  {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                  {"name": "broadcast_initial_parameters"},
                  {"name": "validate_at_step_boundaries"}]
    c["pre_optimizer"] = [{"name": "dump_training_state_at_steps", "hyperparams": {"steps": [1000], "dump_dir": f"{dur}/state1000"}},
                          {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    c["post_optimizer"] = [{"name": "print_training_progress"}]
    c["teardown"] = [{"name": "mark_log_finished"}]
    return c


def build_pilot(arm, dur, fork_dir):
    mu, kind, lrm, clear = ARMS[arm]
    a = f"{dur}/pilot/{arm}"
    c = common(f"req076_pilot_{arm}", 0, 1000, 1256, mu, MUON_LR * lrm)
    c["setup"] = [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                  {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                  {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                  {"name": "broadcast_initial_parameters"},
                  {"name": "load_training_state", "hyperparams": {"state_dir": fork_dir, "step": 1000, "skip_batches": 1000}},
                  {"name": "req075_set_muon_group", "hyperparams": {"mu": mu, "base_lr": MUON_LR * lrm, "clear_buffer": clear}},
                  {"name": "tag_req073_capture", "hyperparams": {"names": SEL, "out_base": a, "chunk_steps": 256}},
                  {"name": "validate_at_step_boundaries"}]
    c["capture_filtered"] = (kind != "none")
    c["pre_optimizer"] = [{"name": "req073_pre"},
                          {"name": "req076_filter", "hyperparams": {"kind": kind, "selected": SEL}},
                          {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7, "schedule_step_max": 1000}}]
    c["post_optimizer"] = [{"name": "req073_post"}, {"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"every": 64}}]
    c["teardown"] = [{"name": "finalize_req073"}, {"name": "mark_log_finished"}]
    return c


def build_screen(arm, seed, dur):
    mu, kind, lrm, _ = ARMS[arm]
    a = f"{dur}/screen/{arm}_s{seed}"
    c = common(f"req076_screen_{arm}_s{seed}", seed, 0, 3250, mu, MUON_LR * lrm)
    c["capture_windows"] = WINDOWS
    c["capture_filtered"] = (kind != "none")
    c["setup"] = [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                  {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                  {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                  {"name": "broadcast_initial_parameters"},
                  {"name": "tag_req073_capture", "hyperparams": {"names": SEL, "out_base": a, "chunk_steps": 256}},
                  {"name": "validate_at_step_boundaries"}]
    c["pre_optimizer"] = [{"name": "req073_pre"},
                          {"name": "req076_filter", "hyperparams": {"kind": kind, "selected": SEL}},
                          {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    c["post_optimizer"] = [{"name": "req073_post"}, {"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125, "final_tenth_every": 25}}]
    c["teardown"] = [{"name": "finalize_req073"}, {"name": "mark_log_finished"}]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump1000", "pilot", "screen"], required=True)
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--dur", required=True)
    ap.add_argument("--fork_dir", default="")
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.mode == "dump1000":
        (a.out_dir / "dump1000.yaml").write_text(yaml.safe_dump(build_dump1000(a.dur), sort_keys=False)); n = 1
    elif a.mode == "pilot":
        for arm in ARMS:
            (a.out_dir / f"{arm}.yaml").write_text(yaml.safe_dump(build_pilot(arm, a.dur, a.fork_dir), sort_keys=False))
        n = len(ARMS)
    else:
        n = 0
        for arm in ARMS:
            for s in (0, 1, 2):
                (a.out_dir / f"{arm}_s{s}.yaml").write_text(yaml.safe_dump(build_screen(arm, s, a.dur), sort_keys=False)); n += 1
    print(f"wrote {n} {a.mode} configs -> {a.out_dir}")


if __name__ == "__main__":
    main()
