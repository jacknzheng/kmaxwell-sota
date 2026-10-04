"""REQ-074 spectrogram plotter: from a recompute .npz, render the requested time-period-amplitude spectrograms
(training time on x = step_end, oscillation period on y, amplitude as color). One PNG per matrix (primary
window 128). Runs locally (matplotlib Agg). Period y-axis capped at the window (periods > window are the DC/
near-DC bin); log color for dynamic range.

Usage: req074_plot.py --npz <recompute.npz> --title_prefix "REQ-073 4B-mom s3_postpolar" --out_dir <dir>
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--title_prefix", default="")
    ap.add_argument("--out_dir", required=True); ap.add_argument("--window", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = np.load(a.npz, allow_pickle=True)
    keys = [k for k in d.files if k.endswith(f"__w{a.window}__P")]
    for k in keys:
        base = k[:-len(f"__w{a.window}__P")]
        P = d[k]                              # (n_cols, n_freq)
        step_end = d[f"{base}__w{a.window}__step_end"]
        periods = d[f"{base}__freqs"]; periods = np.where(np.array(periods) > 0, 1.0/np.array(periods), np.inf)
        # y = period (updates/cycle), ascending period 2..window; drop the inf (DC) bin for the plot
        freqs = np.array(d[f"{base}__freqs"])
        keep = freqs > 0
        per = 1.0 / freqs[keep]
        Pk = P[:, keep]                       # (n_cols, n_freq-1)
        # order by period ascending (2 .. window)
        order = np.argsort(per); per = per[order]; Pk = Pk[:, order]
        fig, ax = plt.subplots(figsize=(10, 4))
        amp = np.log10(Pk.T + 1e-12)          # period (y) x time (x), log amplitude
        im = ax.pcolormesh(step_end, per, amp, shading="auto", cmap="magma")
        ax.set_yscale("log"); ax.set_ylabel("oscillation period (updates/cycle)")
        ax.set_xlabel("training time (optimizer update, window end)")
        ax.set_title(f"{a.title_prefix} {base}  [window {a.window}, hop 1]  log10 amplitude")
        fig.colorbar(im, ax=ax, label="log10 summed power")
        ax.axhline(2.0, color="cyan", lw=0.6, ls="--", alpha=0.6)  # period-two reference
        out = os.path.join(a.out_dir, f"{base}__w{a.window}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
