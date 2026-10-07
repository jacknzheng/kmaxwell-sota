# REQ-076 + REQ-077 — causal low-pass gradient filtering: shared plan, forecast, filter responses

**Status: RUNNING (2026-10-07), session 01YKPwGqPuEuP2wzzHuCEhok.** Both requests test filtering the raw
Muon hidden-matrix gradient (before clipping/Muon) and are tightly coupled; REQ-077 reuses REQ-076's
compatible arms. This doc is the pre-launch plan + resource forecast + the **published filter responses**
(required before training). Filter core built + CPU-validated (`test_req07x_filters.py` 12/12). Training hook
and node pending.

## Filters (registered, fixed; `impl/req07x_filters.py`, responses in `filter_responses.json`)
Amplitude |H(f)| at periods (cycles/update = 1/period):

| filter | p2 | p3 | p4 | p8 | p16 | p32 | notes |
|---|---|---|---|---|---|---|---|
| pair `½,½` | 0.000 | 0.500 | 0.707 | 0.924 | 0.981 | 0.995 | cancels period-2 exactly; age ½ |
| fir9 (fc=0.20, Hann) | 0.005 | 0.077 | 0.309 | 0.785 | 0.944 | 0.986 | strong p2–p4 attenuation; 4-upd delay; taps have negatives |
| ema-⅓ `v=⅔g+⅓v₋` | 0.500 | 0.555 | 0.632 | 0.834 | 0.947 | 0.986 | retains ½ p2 amplitude (¼ power); age ½ |

**FIR9 taps**: [−0.000, −0.0092, 0.0472, 0.2605, 0.4032, 0.2605, 0.0472, −0.0092, −0.000] (Σ=1).
**Matched (pair vs ema-⅓)**: Σh=1, Σk·h=½ (mean age / low-freq delay), Σh²=½ (independent-noise output
variance) — all equal → equal constant gain, mean age, ideal-noise variance; they differ in the spectrum
(pair cancels p2, ema retains ½), so a win is a temporal-filter contrast, not unique p2 causation.
Startup: first 8 updates unfiltered (record genuine history), activate on the 9th. CPU checks confirm
constant gain=1, pure-p2 → pair=s / ema=s±½a / fir9≈s, sinusoidal amplitude == published, no input aliasing.

## Arms
- **REQ-076** (5): nomom, pair, fir9, nomom-halfLR, standard-mom. Filter the raw grad, momentum OFF for the
  filtered/nomom arms (mu=0, Nesterov off), standard momentum for standard-mom. Common Muon LR except halfLR.
- **REQ-077** (4): nomom, pair, ema-third, standard-mom. Primary contrast pair vs ema-third.
- **Shared arms** (nomom, pair, standard-mom): run once under REQ-076; REQ-077 reuses them, counting shared
  execution once (per REQ-077's instruction). REQ-077 adds only ema-third + its frozen-noise probes.

## Resource forecast (ONE node, 8 node-hours cap EACH request)
Per-run estimate (3250-update from-scratch B run, capture only in windows 500–755/1500–1755/2500–2755 =
768 captured steps): ~24 min (from REQ-075 Stage-3: 736 captured updates ≈ 6 min ⇒ 3250 upd ≈ 26 min, fewer
captured ⇒ ~24; +small filter overhead). Pilot (256 upd) ≈ 2–3 min/arm.

| phase | runs | est | cumulative |
|---|---|---|---|
| REQ-076 Stage-1 CPU checks | — | free (done) | 0 |
| REQ-076 Stage-1 pilot (5 arms, 256 upd, step-1000 state) | 5 | ~0.25 h | 0.25 |
| REQ-076 Stage-2 fixed-LR screen (5×3 = 15 runs, 3250 upd) | 15 | ~6.0 h | ~6.25 |
| REQ-076 LR selection (nomom/standard-mom/best-filtered × {0.5,2}, 1 tuning seed) | ~6 | ~2.4 h | **~8.6 h → over cap** |

⚠ **REQ-076 cannot fit Stage-1 + full Stage-2 screen + full LR selection in 8 node-h.** Plan: run Stage-1 +
the 15-run fixed-LR screen (~6.25 h, the primary deliverable), then spend the remaining budget on as much LR
selection as fits and **report the rest as unrun** (the request explicitly permits this: "report any
confirmation left unrun"). No horizon shortening is labeled a full replication.

REQ-077 marginal (reusing REQ-076's nomom/pair/standard-mom ×3 seeds): ema-third ×3 seeds (~1.2 h) + Stage-1
pilot/ema + frozen-noise probes (~0.4 h) + LR selection for ema-third (~fits). Well within REQ-077's own 8 h.

Storage: 768 captured steps × 4 stages × 3 diagnostic matrices (native dtype) ≈ a few GB/run durable
(off-Git). History buffers: 8 tensors × all hidden matrices ≈ ~1.4 GB live (bf16). I/O like REQ-073.

## Staged execution (≤1 node per request, stop after)
1. Build + CPU-validate the training filter hook (overwrite p.grad pre-Muon; per-matrix history; startup;
   owner-rank replicated raw grad). Verify distributed parity + that the ACTUAL displacement shows the
   intended spectral change (not just the filter input).
2. Provision 1 node, bootstrap (reuse REQ-075 recipe: harness 365c392d, venv019, fineweb). Need verified
   step-1000 state (REQ-073 forks are @500/1500/2500 — use @1500? spec says step-1000; forks lack 1000.
   RESOLVE: either dump a fresh step-1000 state via a short baseline run, or use the nearest verified fork
   @1500 and declare it. Decide on-box.) Stage-2 is from-scratch so needs no fork.
3. REQ-076 Stage-1 pilot → Stage-2 screen → partial LR selection (budget-bounded).
4. REQ-077 ema-third runs + probes, reusing REQ-076 shared arms.
5. Spectra (REQ-074 readers, bands p2 0.45–0.50 and broad 0.25–0.50), loss/token/time tables, reports.

Raw tensors off-Git; derived arrays/responses/configs/readers/figures committed. Findings separate from
interpretation; no outcome-driven cutoff/kernel/LR expansion.
