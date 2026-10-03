"""REQ-068 amendment: window fork-capture config. Forks from a retained training state theta_F
(load_training_state, from the full run's forks) and runs 64 updates F..F+63, capturing BOTH streams
(g_t gradient + delta_t displacement) plus probe states at offsets 32/48/63 within the window.

Windows: F in {500, 1500, 2500} -> steps F..F+63 (zero-based), 64 observations, 192 total.
The forked g_t cross-checks the full-run capture; delta_t is the new mandatory stream.

Usage: make_req068_window_config.py --out CONFIG.yaml --fork 500 --forks_dir <dur>/forks \\
         --grad_dir <dur>/win500/grads --disp_dir <dur>/win500/disp --probe_dir <dur>/win500/probes [--seed 0]
"""
import argparse, yaml
from pathlib import Path
import make_req068_config as base  # reuse the verified baseline recipe (optimizer_groups, model, data)

WINDOW = 64
PROBE_OFFSETS = [32, 48, 63]


def build(fork, seed, forks_dir, grad_dir, disp_dir, probe_dir):
    c = base.build(seed, grad_dir, "unused", chunk_steps=WINDOW, train_steps=3250, fork_steps=[])
    c["run_id"] = f"req068_win{fork}_s{seed}"
    c.update(start_step=fork, stop_after_step=fork + WINDOW)
    # restore theta_F + optimizer + data cursor (skip_batches aligns the token stream to F)
    c["setup"].insert(-1, {"name": "load_training_state",
                           "hyperparams": {"state_dir": forks_dir, "step": fork, "skip_batches": fork}})
    # pre_optimizer: capture g_t, snapshot theta_t (for delta), dump offset probe states, cooldown
    c["pre_optimizer"] = [
        {"name": "capture_full_gradients", "hyperparams": {"out_dir": grad_dir, "chunk_steps": WINDOW}},
        {"name": "snapshot_weights_pre_update"},
        {"name": "dump_training_state_at_steps",
         "hyperparams": {"steps": [fork + o for o in PROBE_OFFSETS], "dump_dir": probe_dir}},
        {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
    ]
    # post_optimizer: capture delta_t, progress
    c["post_optimizer"] = [
        {"name": "capture_displacement", "hyperparams": {"out_dir": disp_dir, "chunk_steps": WINDOW}},
        {"name": "print_training_progress"},
    ]
    c["teardown"] = [
        {"name": "finalize_gradient_capture"},
        {"name": "finalize_displacement_capture"},
        {"name": "mark_log_finished"},
    ]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--fork", type=int, required=True)
    ap.add_argument("--forks_dir", required=True); ap.add_argument("--grad_dir", required=True)
    ap.add_argument("--disp_dir", required=True); ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = build(a.fork, a.seed, a.forks_dir, a.grad_dir, a.disp_dir, a.probe_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {a.out} (fork {a.fork}, steps {a.fork}..{a.fork+WINDOW-1}, probes {[a.fork+o for o in PROBE_OFFSETS]})")


if __name__ == "__main__":
    main()
