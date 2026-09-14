# REQ-058 — does layer spectral sharpness predict the momentum response? — **yes, and it generalizes (held-out Spearman +0.59)**

**SHA `365c392d` + REQ-058 optimizer patch (`apply_req058_opt.py`), node qj9epjw (1×8 H100), venv019 torch
2.10.0+cu128, ≤2-node ceiling (1 node, ~2.4 node-hours full stage).** Tests whether a matrix's pre-intervention
spectral sharpness (REQ-057 `S_i`) predicts whether shortening or lengthening its momentum memory helps.

## Design

- **Memory intervention:** perturb one matrix's K-Maxwell stream decays `β_k(a) = β_k^(1/a)`,
  `a ∈ {0.5 (halve decay time = shorter memory), 1, 2 (double = longer memory)}`, **keeping the mixture
  weights unchanged** (a memory intervention, not a reweighting). Verified to reach the post-polar update
  (a direct kernel test showed a 0.41 relative-difference update over 20 steps).
- **Protocol:** the REQ-054 eight-stream K-Maxwell kernel with the production cloned-buffer switch at the
  fork; all arms inherit identical streams. Fork from a serialized base, run **64 updates**, response =
  paired selection-loss difference at `fork+64` vs the `a1all` (a=1) control, on a disjoint held-out val set.
- **41 continuations per base:** 18 sentinel matrices (blocks 0/6/11 × 6 types) × {a=0.5, a=2} = 36
  selective + 5 shared controls (a=1-all, a=0.5-all, a=2-all, no-momentum Muon, exact-age matched EMA).
- **4 base states:** seed 0 @ 2000 and @ 1500 (development), **seeds 1 and 2 @ 2000 (untouched held-out
  prediction tests)** — 164 continuations total.
- **Exact-age matched control** (the missing REQ-054 finite-history control): a single EMA whose *realized*
  age matches the mixture's `q_age` at every step, solved from the M/P recurrence (CPU-validated to 1.42e-14;
  REQ-054's stationary-schedule control was off by up to 2.6 steps).
- Sharpness feature `S_i` = REQ-057 isolated shape-weighted spectral sharpness at the **same (seed,
  fork-step)**.

## Result — sharpness predicts the response, and the prediction holds out-of-sample

**Charter (per-matrix, `readout.tsv`):**

| set | n | Spearman(S_i, longer-memory penalty) | high/low sharpness tertile penalty |
|:----|:-:|:------------------------------------:|:----------------------------------:|
| **dev (seed 0)** | 36 | **+0.604** | +0.00106 / +0.00044 |
| **held-out (seeds 1,2)** | 36 | **+0.591** | +0.00144 / +0.00057 |

**Sharper matrices are hurt more by longer memory** — i.e. sharpness predicts *preferring shorter memory*.
The rank correlation is +0.60 on the development seed and **replicates at +0.59 on the two untouched held-out
seeds** (same sign and magnitude), and the high-sharpness tertile's longer-memory penalty is ~2–2.5× the
low-sharpness tertile's. Shorter-memory benefit also grows with sharpness (Spearman(S_i, a=0.5 response) =
−0.34 dev / −0.52 held-out). **The held-out prediction gate is passed** — REQ-059 (a sharpness-guided
layer-wise momentum policy) is unblocked, with a concrete direction: *shorten memory on the sharp layers*.

**Global controls (all 4 bases, strikingly consistent):**

| arm | selection-loss@fork+64 − a1all |
|:----|:------------------------------:|
| a=0.5-all (shorter) | **−0.029 … −0.036** (better) |
| a=2-all (longer) | **+0.052 … +0.060** (worse) |
| no-momentum Muon | **−0.030 … −0.038** (best) |
| exact-age matched EMA | +0.006 (≈ mixture) |

Over these 64-update windows **less memory is better**: shorter memory and no-momentum both beat the
K-Maxwell mixture, and longer memory is clearly worst. The exact-age matched single EMA tracks the mixture
(+0.006), so the realized *age* dominates the gross behavior and the kernel *shape* adds little here —
consistent with REQ-054 (shape worth ~0.010 val) and REQ-057 (sharpness is real but ~97% coupled).

## Interpretation

The project's premise holds: spectral sharpness is not just measurable (REQ-057) but **predictive of the
optimizer's memory response**, and predictive out-of-sample. The useful direction is to **shorten momentum
memory on high-sharpness matrices** (they are penalized most by long memory). This is a genuine,
held-out-validated signal for a layer-wise momentum rule — exactly what REQ-059 will test against global,
Euclidean, type/depth, shuffled and reversed controls.

Two cautions carry into REQ-059: (1) the response is measured over a **64-update window**; "shorter is
better" globally here is a short-horizon result, not a proof about long-run training. (2) Per-matrix effects
are small in absolute loss (~0.0005–0.0015); the robust signal is the **rank correlation** (+0.6,
held-out-replicated), not any single matrix's magnitude.

## Caveats / method

- `val_tokens = 524288` (not the requested 131072): the harness validation requires
  `val_tokens ≥ world_size·mbs·seq_len = 8·64·1024 = 524288`; 131072 is infeasible at mbs=64/world=8. The
  paired within-base comparison is unaffected. (First pilot attempt crashed on this assert; fixed and
  re-run — ~5 min lost.)
- A config-plumbing bug (the extended 18-matrix generator wasn't staged to the box on the first full-stage
  launch, yielding 17/base) was caught by a progress check and fixed before the run completed; the delivered
  run is the corrected 41/base × 4 bases.
- Intervention correctness: `PerturbedAnnealedWeightsMuon` (per-matrix β^(1/a) via a second compiled kernel)
  and `ExactAgeMatchedMuon` (solved realized-age schedule) validated on the box; a1all fork parity confirmed
  (fork-point selection-loss reproduced).

## Files

- `impl/req058_age_match.py`, `impl/test_age_match.py` — exact-age control solver + 4/4 CPU tests.
- `impl/analyze_req058.py` — the sharpness→response aggregation (dev vs held-out), reproducible from the raw TSVs + REQ-057 JSONs.
- `readout.tsv` — global controls + the per-matrix Spearman result.
- `raw/full058/{arm}_s{seed}_f{fork}.tsv` — per-arm selection-loss traces (source of truth).
- Optimizer patch `apply_req058_opt.py` + config generator `make_req058_configs.py` (in the run environment; optimizers registered as `perturbed_annealed_weights_muon` / `exact_age_matched_muon`).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2-node ceiling; node stopped after delivery.
