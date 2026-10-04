# REQ-074 — Time × Period × Amplitude spectrograms

**Status: DONE.** Delivers the training-time-resolved spectrograms that the REQ-072/073 analyzers
discarded (they saved only the time-averaged spectrum). No training rerun — recomputed entirely from the
REQ-072/073 raw stage histories already on the durable FS.

## What was delivered

- **84 primary heatmaps** (`heatmaps/*.png`) + **84 detrended** (`heatmaps/detrend/*.png`): one per
  (series × matrix). x = training time (optimizer update at the window's end), y = oscillation period
  (updates/cycle, log), color = log10 summed power. Period-2 reference line drawn.
- **28 time-resolved arrays** (`arrays/*.npz`): `P[step_end, freq]` for windows 128 (primary, hop 1),
  32, 256, plus a 128-detrended variant; freqs/periods/step_end axes; `verify_ok` per source history.
  All 28 pass integrity (verify_ok=1, finite, 3 matrices each).
- Coverage: REQ-072 (6 arms × {raw grad, conditioned u_t}) + REQ-073 (4 arms × {s1 raw grad,
  s2 post-momentum, s3 post-orthogonalization, s4 displacement}), 3 matrices each.

The first 127 updates of every series have insufficient window history and are **not** plotted (never faked);
columns begin at step_end = W−1.

## Method

`req074_spectrogram.time_resolved_power` (5/5 CPU tests) is the reference: per-coordinate windowed STFT
(Hann, hop 1), |FFT|² summed over all coordinates, normalized by fs·window-energy, retaining the time axis.
The recompute (`req074_recompute_gpu.py`) runs this on an H100 via `torch.fft` and is **parity-exact to the
numpy reference at ~1e-15** (`--selfcheck`, verified on-GPU). Each capture chunk is read once and all three
matrices extracted together. Box qvkdpj3 (1×8 H100, within the 2-node cap) was stopped after the pull.

## Findings (band-power fractions, early vs late training quartile; averaged over 3 matrices)

The spectrograms localize the fast→slow transition **in training time** and attribute it to **momentum**:

| stream | period-2 frac (early→late) | slow (period≥16) frac (early→late) |
|---|---|---|
| raw gradient `s1`/`grad` | high, **falls** (e.g. B-mom 0.12→0.03) | low, rises modestly (0.04→0.12) |
| post-momentum `s2_premom` / `u_t` (mom arms) | falls hard (B-mom 0.07→0.008) | **slow power surges** (0.40→0.68; adamw u_t 0.69→0.79) |
| post-orthogonalization `s3`/`s4` (mom arms) | nearly gone (0.015→0.011) | highest + stable (0.57→0.63) |
| **no-momentum** arms (`s1`/`s2`) | **stays high** (4B-nomom 0.31→0.37) | **stays low** (0.006→0.03) |

1. **Momentum is the low-pass that builds the slow component — and it builds progressively.** The post-
   momentum stage (s2 / conditioned u_t) shows slow-band power *growing over training* (e.g. B-mom s2
   0.40→0.68, 4B-mom s2 0.17→0.58) while the raw gradient stays broadband/period-2-heavy. The transition is
   not instantaneous; the heatmaps show it emerging over roughly the first training quartile.
2. **Orthogonalization (Newton-Schulz) is a secondary low-pass.** s3/s4 push slow-band fraction higher still
   and nearly erase period-2 (0.015→0.011). s4 (displacement) equals s3 in every fraction (displacement =
   −lr·s3; scale cancels) — a built-in consistency check that holds.
3. **Batch size is NOT the mechanism.** Quadrupling the batch (B→4B) does *not* remove the period-2
   oscillation in the no-momentum arms — 4B-nomom keeps the *highest* period-2 fraction (0.31→0.37) and the
   lowest slow fraction. This decisively refutes "period-2 = minibatch noise": the oscillation is
   deterministic, and only momentum (not more data per step) converts fast structure into slow structure.
   Consistent with and now time-resolving the REQ-073 conclusion.

### Caveats
- `sgd-mom` degenerates late in training (band fractions collapse to ~0) — an SGD+momentum instability on
  this harness, visible directly in its heatmaps; read those two series with care.
- `sgd-nomom` grad == u_t exactly (no momentum ⇒ u_t = grad), as expected.
- Muon `u_t` in REQ-072 was owner-sharded (None at the master); those two arms use the displacement proxy
  (`*_uT_fromdisp`, wd=0 ⇒ δ=−lr·u_t), which preserves all power *fractions*. REQ-073 captured every stage
  via all_reduce, so its Muon stages are complete.

## Files
- `impl/req074_spectrogram.py` — time-resolved STFT core (+ `test_req074_spectrogram.py`, 5/5)
- `impl/req074_recompute_gpu.py` — GPU recompute, numpy-parity `--selfcheck`
- `impl/req074_recompute.py` — CPU reference recompute (same schema)
- `impl/req074_plot.py` / `impl/req074_plot_all.py` — single / batch heatmap renderers
- `impl/run_req074_box.sh` — on-box verify + recompute orchestration
- `arrays/*.npz` (28) — time-resolved arrays · `heatmaps/*.png` (168) — the spectrograms

Raw stage tensors remain on the durable FS (`/root/.cache/user_artifacts/req07{2,3}`); only derived arrays
and images are committed.
