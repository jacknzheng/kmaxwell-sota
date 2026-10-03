"""REQ-069: branch config generator. Forks the MIDDLE fork (step 1500) into branch b with IDENTICAL model
weights + full optimizer state + schedule position, but an INDEPENDENT data-sampling sequence: branch b
consumes a disjoint 64-batch corpus window via skip_batches = 1500 + b*64 (the data generator is
deterministic-sequential, so disjoint skip offsets = ten independent minibatch sequences; token windows do
not overlap). Runs 64 updates (steps 1500..1563) capturing BOTH streams (g_t + delta_t) per REQ-068 schema
+ probe states at offsets 32/48/63. Model/optimizer are byte-identical across branches; only the data differs.

Usage: make_req069_branch_config.py --out CONFIG.yaml --branch b --forks_dir <dur>/forks \\
         --grad_dir <d>/br{b}/grads --disp_dir <d>/br{b}/disp --probe_dir <d>/br{b}/probes [--fork 1500]
"""
import argparse, yaml
from pathlib import Path
import make_req068_config as base

WINDOW = 64
PROBE_OFFSETS = [32, 48, 63]


def build(branch, fork, forks_dir, grad_dir, disp_dir, probe_dir, seed=0):
    c = base.build(seed, grad_dir, "unused", chunk_steps=WINDOW, train_steps=3250, fork_steps=[])
    c["run_id"] = f"req069_br{branch}_fork{fork}"
    c["river_branch"] = branch  # provenance: which data-seed branch
    c.update(start_step=fork, stop_after_step=fork + WINDOW)
    # restore theta_fork + optimizer + counters, then advance the data iterator to the branch's disjoint window
    skip = fork + branch * WINDOW
    c["setup"].insert(-1, {"name": "load_training_state",
                           "hyperparams": {"state_dir": forks_dir, "step": fork, "skip_batches": skip}})
    c["pre_optimizer"] = [
        {"name": "capture_full_gradients", "hyperparams": {"out_dir": grad_dir, "chunk_steps": WINDOW}},
        {"name": "snapshot_weights_pre_update"},
        {"name": "dump_training_state_at_steps",
         "hyperparams": {"steps": [fork + o for o in PROBE_OFFSETS], "dump_dir": probe_dir}},
        {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
    ]
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
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--branch", type=int, required=True)
    ap.add_argument("--fork", type=int, default=1500); ap.add_argument("--forks_dir", required=True)
    ap.add_argument("--grad_dir", required=True); ap.add_argument("--disp_dir", required=True)
    ap.add_argument("--probe_dir", required=True)
    a = ap.parse_args()
    cfg = build(a.branch, a.fork, a.forks_dir, a.grad_dir, a.disp_dir, a.probe_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    skip = a.fork + a.branch * WINDOW
    print(f"wrote {a.out} (branch {a.branch}, fork {a.fork}, skip_batches={skip}, "
          f"data window batches {skip}..{skip+WINDOW-1})")


if __name__ == "__main__":
    main()
