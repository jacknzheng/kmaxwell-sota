# REQ-072 — six optimizer ablations: where does the fast→slow transition come from? — **MOMENTUM**

**Harness `365c392d`, node w7vln1w (1×8 H100), stopped after delivery.** Six arms (AdamW/SGD/Muon ×
first-moment momentum on/off), 3250 updates each, identical init/data/eval, one seed. For the three selected
matrices `blocks.{0,5,11}.attn.proj.weight` (768×768), captured every step: raw g_t, conditioned direction
u_t (after momentum/bias-correction/adaptive-scale/orthogonalization + shape scaling, before lr/decay),
displacement δ_t. wd=0 all arms (u_t is pre-decay, so decay is irrelevant to the spectra). Temporal STFT per
coordinate, power summed over coordinates (`req072_spectrogram.py`, 5/5 CPU tests). LR from a 150-step pilot.

## Result — temporal spectrum of the conditioned direction u_t (blocks.5, window 64)

| arm | final val | **u_t period-two frac** (f≈0.5) | u_t low-f frac | g_t period-two frac |
|:--|--:|--:|--:|--:|
| adamw-mom (β1=0.9) | 3.549 | **0.009** | 0.738 | 0.148 |
| adamw-nomom (β1=0) | 3.848 | **0.233** | 0.080 | 0.287 |
| sgd-mom (0.9) | 8.205 | **0.000** | 0.999 | 0.121 |
| sgd-nomom (0) | 30.35 | **0.121** | 0.121 | 0.121 |
| muon-mom (mu=0.95) | 3.296 | **0.040** | 0.680 | 0.331 |
| muon-nomom (mu=0) | 3.444 | **0.185** | 0.105 | 0.391 |

**1. First-moment momentum is the mechanism that turns the oscillating gradient into a smooth, low-frequency
conditioned direction — consistently across ALL THREE optimizers.** Turning momentum OFF raises the
period-two (f≈0.5) power fraction of u_t by 4–26× and collapses the low-frequency fraction:
AdamW 0.009→0.233, SGD 0.000→0.121, Muon 0.040→0.185. With momentum, u_t power sits at low f (0.68–0.999).

**2. Neither AdamW's second-moment (√v) adaptivity nor Muon's orthogonalization suppresses the oscillation by
itself.** adamw-nomom keeps the √v adaptive scale yet shows strong period-two (0.233); muon-nomom keeps the
orthogonalization + shape scaling yet shows 0.185. Only the first-moment averaging (momentum) low-passes the
period-two bounce. The raw gradient g_t carries period-two power in every arm (0.12–0.39); momentum is what
removes it from the *conditioned* direction.

**3. This explains REQ-068's "slow parameter movement".** The persistent step-to-step displacement (REQ-068
lag1 ≈ +0.79) is momentum's low-pass filtering of the period-two-oscillating gradient — not a property of
Muon's geometry or a reproducible loss-descent "river" (cf. REQ-069/070 negative). The fast→slow transition
= first-moment momentum, full stop.

Secondary: final val shows the optimizer ranking (Muon 3.30/3.44 < AdamW 3.55/3.85 < SGD 8.2/30.3; full-model
SGD conditions transformer gradients poorly, especially without momentum). Momentum helps loss in every pair.

## Honesty / caveats
- **Muon u_t recovered from the displacement stream** (`analysis/muon_disp_spectra.json`): Muon shards
  param ownership across ranks, so `compute_polar_input` (and thus the direct u_t stash) runs only on each
  matrix's owning rank — master captured u_t only for rank-0-owned matrices (None otherwise). Since wd=0,
  δ_t = −lr·u_t exactly, and the period-two *fraction* is scale-invariant, so the displacement spectrum
  equals u_t's. AdamW/SGD are elementwise (every rank has all u_t) → captured directly. A future precise
  Muon u_t capture needs per-rank writing (noted in `apply_req072_capture.py`); the spectral conclusion is
  unaffected.
- One seed, exploratory paired comparison (not a replicated performance claim). LR per optimizer from a
  150-step pilot; different optimizers use different LRs by design. g_t period-two differs by arm because
  each optimizer follows its own trajectory (not a shared-gradient filter-transfer comparison).

## Files
`analysis/{adamw-mom,adamw-nomom,sgd-mom,sgd-nomom}.json` (direct u_t+g_t spectra), `muon_disp_spectra.json`
(Muon u_t via δ). `impl/req072_capture_adamw.py` (+test), `apply_req072_capture.py`, `make_req072_configs.py`,
`req072_spectrogram.py` (+test 5/5), `req072_analyze.py`, `run_req072.sh`. Raw 3-stream tensors off-Git on
durable FS (`req072/<arm>/{grad,uT,disp}`). Node w7vln1w stopped. No secrets/weights/tensors committed.
