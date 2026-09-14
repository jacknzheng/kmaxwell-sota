# REQ-059 — sharpness-guided layer-wise momentum policy — **no practical win (negative primary outcome)**

**SHA `365c392d` + REQ-058/059 optimizer patches, node w7r58o3 (1×8 H100), venv019 torch 2.10.0+cu128,
≤2-node ceiling (1 node, ~4.4 node-hours: 1 pilot + 3 held-out seeds).** The direct project-success test: does
*correctly* assigning momentum memory to layers (by spectral sharpness) beat the best global memory setting
and the cheaper feature/heuristic controls, under equal memory resources?

## Design

- **Frozen rule (from REQ-058):** sharper matrices prefer shorter memory. The **guided** allocation gives, per
  type (12 block-matrices), the 4 sharpest `a=0.5` (shorten), the 4 least-sharp `a=2` (lengthen), the middle 4
  `a=1` — a **balanced 24/24/24** allocation (equal memory resources, `a_k(a)=β_k^{1/a}`, weights unchanged).
- **Fresh seeds 3,4,5,6** (seed 3 pilot, 4/5/6 held out); each seed's own step-2000 base → **2750** (750
  updates), 1× batch. Features (spectral `S_i` + Euclidean Lanczos `λ_i` + type/depth) measured once at the
  fork; each matrix's `a` held fixed through the continuation.
- **9 arms/seed:** global `a=0.5`/`a=1`/`a=2`; **guided** (spectral `S_i`); **euclidean** (rank by `λ_i`, same
  constraint); **typedepth** (rank by the dev-mean `S_i(type,block)` prior — no this-seed measurement);
  **shuffled** (guided permuted within type, one perm/seed); **reversed** (swap `a=0.5`↔`a=2`); **nomom** (the
  mandatory 9th arm, since no-momentum beat all global mixtures in REQ-058's window).
- Final validation on the held-out **10,485,760-token** set at 2000/2125/…/2750; endpoint 2750 primary, mean
  of the last three secondary. Paired across seeds, 95% CI (t, n=4), **Holm-corrected** across the 8
  comparisons. Practical win = guided ≥0.0005 lower than **every** control with CI excluding 0.

## Result — the policy does not deliver a practical win (`readout.tsv`)

Guided − control, endpoint 2750 (n=4 seeds; negative = guided better):

| contrast | mean diff | 95% CI | verdict |
|:---------|----------:|:-------|:--------|
| guided − **global_a05** | **+0.00914** | [+0.0091,+0.0092] | **guided WORSE** (all-shorter global wins) |
| guided − global_a1 | +0.00061 | [+0.0004,+0.0008] | guided slightly worse |
| guided − global_a2 | −0.01293 | [−0.0132,−0.0126] | beats |
| guided − **euclidean** | **−0.00012** | [−0.0004,+0.0001] | **tie** (spectral adds nothing over Euclidean) |
| guided − **typedepth** | **−0.00018** | [−0.0002,−0.0002] | below 0.0005 margin (barely beats the prior) |
| guided − shuffled | −0.00057 | [−0.0008,−0.0003] | beats |
| guided − reversed | −0.00169 | [−0.0020,−0.0014] | beats |
| guided − nomom | −0.00480 | [−0.0050,−0.0046] | beats |

The secondary metric (mean of last three) gives the identical verdict (guided−global_a05 +0.0109;
guided−euclidean −0.00001, tie; guided−typedepth −0.00020).

**PRACTICAL WIN: NO.** Three findings:

1. **The balanced allocation loses to global shorter-memory.** Forcing 24 matrices to long memory (`a=2`) to
   keep the resource budget balanced costs ~+0.009 vs simply shortening memory everywhere (`a=0.5`-all). The
   equal-resource framing is the wrong one here — REQ-058 already showed shorter-everywhere is best, and this
   confirms it at the 750-update horizon.
2. **No value from *spectral-specific* assignment.** Guided is statistically indistinguishable from the
   **Euclidean**-feature allocation (−0.00012, CI includes 0) and beats the **type/depth** prior by only
   −0.00018 (below the 0.0005 margin). Per the request's own criterion — *"failure to beat shuffle/type-depth
   means no demonstrated value from sharpness-specific assignment"* — guided beats shuffle but **not**
   type/depth or Euclidean at the practical margin. The expensive spectral sharpness (REQ-057) does not earn
   its cost over cheaper features for this policy.
3. **The assignment *direction* is real but tiny.** Guided > shuffled > reversed monotonically
   (−0.00057 / −0.00169), so a sharpness-informed assignment does beat scrambling or inverting it — the
   REQ-058 signal is present — but the magnitude is ~10× below the global-vs-global spread and is captured by
   cheaper features.

**Bottom line:** REQ-057 (measurement) and REQ-058 (held-out prediction) both held, but they **do not convert
into a useful static layer-wise momentum policy** under equal memory resources. The project's central bet —
that correctly allocating memory by *measured spectral sharpness* beats the best global setting — is **not
supported**. This is a valid negative result; the favored arm was not retuned after seeing the test seeds.

## What this does (and does not) rule out

- Static, balanced-resource, spectral-guided allocation: no practical win (this experiment). 
- It does **not** rule out an *online* feedback controller (explicitly a later experiment, not run here), nor
  a non-balanced allocation (e.g. shorten-most-and-shift-budget), nor gains at other horizons/LRs (out of
  scope — no LR retune authorized). But any future variant must clear the same bar: beat global `a=0.5`-all
  and beat cheaper Euclidean/type-depth features.

## Caveats / method

- Within a seed the arms share the exact base state and data cursor, so their differences are
  near-deterministic → very tight CIs; the paired signal is across the **4 seeds** (n=4).
- 750-update horizon, 1× batch; final val held out from all fitting/selection.
- Euclidean feature = 8-iteration Lanczos `λ_i` per matrix (`measure_per_matrix_curvature`); spectral feature
  = REQ-057 isolated shape-weighted `S_i` (cheap budget). Type/depth prior = mean `S_i(type,block)` over
  REQ-057 seeds 0,1,2.

## Files

- `impl/make_req059_alloc.py` — allocation generator (guided/euclidean/typedepth/shuffled/reversed, balanced 24/24/24).
- `impl/make_req059_configs.py` — base + 9-arm config generator. `impl/apply_req059_opt.py` — `AllocatedAnnealedWeightsMuon` (per-matrix a-map) + `tag_req059_allocation` hook.
- `impl/analyze_req059.py` — paired Holm-corrected guided-vs-control stats (endpoint + last-3), reproducible from the raw TSVs.
- `readout.tsv` — per-seed endpoints + the paired comparison + verdict.
- `raw/full059/{arm}_s{seed}.tsv` — per-arm validation traces (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2-node ceiling; node stopped after delivery.
