"""REQ-074 batch plotter: render time x period x amplitude heatmaps for every recompute .npz in a directory.
One PNG per (npz, matrix) at the primary window (128, hop 1); x=training time (window-end update), y=oscillation
period (log), color=log10 summed power, with a period-2 reference line. Title derived from the filename
(arm + stream/stage). Also emits a small per-npz detrended-128 panel set under a detrend/ subdir.

Usage: req074_plot_all.py --arrays_dir <dir> --out_dir <dir> [--window 128] [--detrend_too]
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(npz_path, out_dir, window, pkey_suffix, title_prefix):
    d = np.load(npz_path, allow_pickle=True)
    made = []
    keys = sorted(k for k in d.files if k.endswith(pkey_suffix + "__P"))
    for k in keys:
        base = k[:-len(pkey_suffix + "__P")]
        P = np.asarray(d[k])                                   # (n_cols, n_freq)
        # step_end keyed by the actual window size (the detrend variant reuses the w128 axis)
        se_key = f"{base}{pkey_suffix}__step_end"
        if se_key not in d.files:
            se_key = f"{base}__w{window}__step_end"
        step_end = np.asarray(d[se_key])
        freqs = np.asarray(d[f"{base}__freqs"])
        keep = freqs > 0
        per = 1.0 / freqs[keep]
        Pk = P[:, keep]
        order = np.argsort(per); per = per[order]; Pk = Pk[:, order]
        fig, ax = plt.subplots(figsize=(10, 4))
        amp = np.log10(Pk.T + 1e-12)
        im = ax.pcolormesh(step_end, per, amp, shading="auto", cmap="magma")
        ax.set_yscale("log"); ax.set_ylabel("oscillation period (updates/cycle)")
        ax.set_xlabel("training time (optimizer update, window end)")
        verify = int(d["verify_ok"]) if "verify_ok" in d.files else -1
        ax.set_title(f"{title_prefix}  {base}\n[window {window}, hop 1, verify_ok={verify}]  log10 summed power",
                     fontsize=9)
        fig.colorbar(im, ax=ax, label="log10 power")
        ax.axhline(2.0, color="cyan", lw=0.7, ls="--", alpha=0.7)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{os.path.basename(npz_path)[:-4]}__{base}{pkey_suffix}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); made.append(out)
    return made


def title_from_name(stem):
    # req072_adamw-mom_grad  /  req073_4B-nomom_s3_postpolar
    parts = stem.split("_", 1)
    return f"{parts[0].upper()} {parts[1]}" if len(parts) == 2 else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=128); ap.add_argument("--detrend_too", action="store_true")
    a = ap.parse_args()
    npzs = sorted(glob.glob(os.path.join(a.arrays_dir, "*.npz")))
    total = 0
    for p in npzs:
        stem = os.path.basename(p)[:-4]
        tp = title_from_name(stem)
        made = render(p, a.out_dir, a.window, f"__w{a.window}", tp)
        total += len(made)
        if a.detrend_too:
            made_d = render(p, os.path.join(a.out_dir, "detrend"), 128, "__w128_detrend", tp + " [detrended]")
            total += len(made_d)
        print(f"  {stem}: {len(made)} heatmaps", flush=True)
    print(f"WROTE {total} PNGs from {len(npzs)} npz into {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
