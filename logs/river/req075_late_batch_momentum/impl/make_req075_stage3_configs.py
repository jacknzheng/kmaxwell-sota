"""REQ-075 Stage 3 config generator: equal-TOKEN performance comparison from the B-mom @2500 fork.

4 configs (B/16B x Muon momentum on/off) x 3 paired continuation data-order seeds, common token budget =
736B = 385,875,968 tokens (736 B-updates == 46 16B-updates). The original TOKEN-indexed LR schedule is
preserved via cool_down_by_tokens (B and 16B get identical eta at equal tokens); the endpoint is truncated at
token-step 3236/3250 (736 rounded to a multiple of 16B) -- recorded, NOT stretched to 736 optimizer updates.

Paired seeds: seed s starts at B-batch 2512 + s*736 (== 16B-batch 157 + s*46), so within a seed the B and 16B
arms read the SAME ordered token stream (16B groups 16 B-blocks) and the three seeds occupy DISJOINT 386M
windows. All configs restore the identical @2500 branch state; nomom arms clear the Muon buffer + set mu=0.
Baseline wd=0 retained (zero decay-shrinkage to match). 4-stage capture every update on the main 3-seed runs.

LR search (--mode lrsearch): Muon base-LR multipliers {0.5,1,2} on the tuning seed only, tuning-validation =
a held-out chunk (fineweb_train_000099), no capture; select per config, then evaluate on the 3 paired seeds
with reporting-validation = fineweb_val (the main runs).

Usage: make_req075_stage3_configs.py --out_dir DIR --dur DUR --fork_dir FORK --mode {main3seed,lrsearch}
"""
import argparse, yaml
from pathlib import Path

B = 524288
FORK_STEP = 2500
TOTAL_STEPS = 3250
B_UPDATES = 736                       # 736 B-updates == 46 16B-updates == 385,875,968 tokens
CD_FRAC = 0.7
SELECTED = ["blocks.5.attn.q.weight", "blocks.5.attn.k.weight", "blocks.5.mlp.proj.weight"]
# config: (batch_tokens, mu, clear_buffer, scale_factor, updates, skip_base_per_seed_unit)
CONFIGS = {
    "B-mom":     (B,      0.95, False, 1.0,      B_UPDATES,      ("B", 2512, 736)),
    "B-nomom":   (B,      0.0,  True,  1.0,      B_UPDATES,      ("B", 2512, 736)),
    "16B-mom":   (16 * B, 0.95, False, 1.0 / 16, B_UPDATES // 16, ("16B", 157, 46)),
    "16B-nomom": (16 * B, 0.0,  True,  1.0 / 16, B_UPDATES // 16, ("16B", 157, 46)),
}
MUON_BASE_LR = 0.025


def build(cfg, seed, lr_mult, fork_dir, dur, capture, val_data):
    batch, mu, clear, scale, updates, (_, skip0, skipstep) = CONFIGS[cfg]
    skip = skip0 + seed * skipstep
    base_lr = MUON_BASE_LR * lr_mult
    tag = f"{cfg}_s{seed}" + ("" if lr_mult == 1.0 else f"_lr{lr_mult}")
    a = f"{dur}/{tag}"
    setup = [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
             {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
             {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
             {"name": "broadcast_initial_parameters"},
             {"name": "load_training_state", "hyperparams": {"state_dir": fork_dir, "step": FORK_STEP, "skip_batches": skip}},
             {"name": "req075_set_muon_group", "hyperparams": {"mu": mu, "base_lr": base_lr, "clear_buffer": clear}}]
    if capture:
        setup.append({"name": "tag_req073_capture", "hyperparams": {"names": SELECTED, "out_base": a, "chunk_steps": updates}})
    setup.append({"name": "validate_at_step_boundaries"})
    return {
        "loop": "gpt_record", "run_id": f"req075_s3_{tag}", "seed": 0, "require_world_size": 8,
        "start_step": FORK_STEP, "train_steps": FORK_STEP + updates,
        "batch_tokens": batch, "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin", "val_data": val_data,
        "val_tokens": 10485760, "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.0}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.0}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon", "hyperparams": {"lr": base_lr, "weight_decay": 0.0, "mu": mu}},
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.0}}],
        "setup": setup,
        "pre_optimizer": ([{"name": "req073_pre"}] if capture else []) +
                         [{"name": "scale_grad_for_4B", "hyperparams": {"factor": scale}},
                          {"name": "cool_down_by_tokens", "hyperparams": {"cooldown_frac": CD_FRAC,
                           "total_steps": TOTAL_STEPS, "tokens_per_step_base": B, "fork_step": FORK_STEP}}],
        "post_optimizer": ([{"name": "req073_post"}] if capture else []) +
                          [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"every": max(1, updates // 8)}}],
        "teardown": ([{"name": "finalize_req073"}] if capture else []) + [{"name": "mark_log_finished"}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--dur", required=True)
    ap.add_argument("--fork_dir", required=True); ap.add_argument("--mode", choices=["main3seed", "lrsearch"], required=True)
    ap.add_argument("--lr_mult", type=float, default=1.0, help="Muon base-LR multiplier for main3seed (selected tuned LR)")
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if a.mode == "main3seed":
        for cfg in CONFIGS:
            for seed in (0, 1, 2):
                c = build(cfg, seed, a.lr_mult, a.fork_dir, a.dur, capture=True,
                          val_data="data/fineweb10B/fineweb_val_*.bin")  # reporting-validation
                (a.out_dir / f"{cfg}_s{seed}.yaml").write_text(yaml.safe_dump(c, sort_keys=False)); n += 1
    else:  # lrsearch: tuning seed 0, mults 0.5/1/2, tuning-validation = held-out chunk, no capture
        for cfg in CONFIGS:
            for mult in (0.5, 1.0, 2.0):
                c = build(cfg, 0, mult, a.fork_dir, a.dur, capture=False,
                          val_data="data/fineweb10B/fineweb_train_000099.bin")  # tuning-validation (held out)
                nm = f"{cfg}_lr{mult}".replace(".0", "")
                (a.out_dir / f"{nm}.yaml").write_text(yaml.safe_dump(c, sort_keys=False)); n += 1
    print(f"wrote {n} Stage-3 {a.mode} configs -> {a.out_dir} (budget {B_UPDATES}B tokens = 385,875,968)")


if __name__ == "__main__":
    main()
