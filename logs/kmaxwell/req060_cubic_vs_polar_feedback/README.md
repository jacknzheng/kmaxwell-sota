# REQ-060 — loss-cubic feedback vs Muon polar-map nonlinearity — **both contribute; loss-cubic not purely cubic**

**SHA `365c392d`, node 3ypd5k3 (1×8 H100), venv019 torch 2.10.0+cu128, ≤2-node ceiling (1 node, well under 8
node-hours).** When memory changes curvature, is the change consistent with the loss's third derivatives, with
Muon's nonlinear normalization, or both? (Secondary mechanism follow-up to REQ-057/058; independent of
REQ-059's negative policy outcome, per the request.)

## Method

Reused the matched **a=0.5/1/2 trajectories** from REQ-058 (regenerated: 3 bases + 9 continuations, seeds
0,1,2), retaining full-state windows at **offsets 32 (step 2032) and 64 (step 2064)**. On 3 fixed diagnostic
val minibatches (mean-per-token gradients), along the **spectral-polar sharp direction** `δ` per Muon matrix:

- **Loss-cubic feedback:** `e_loss(c,δ) = [g(c+δ)+g(c−δ)]/2 − g(c)` (even 2nd-difference of the gradient =
  ½·∇³L[δ,δ] + O(δ⁴)), with the **scale test** `δ·{1,0.5,0.25}` (cubic ideal ratio 0.5/1 = 0.25).
- **Separation of the two nonlinearities** at the gradient-polar center (`Φ` = `zeropower_via_newtonschulz5`
  × shape scale, `Hd` = odd part ≈ `H[δ]`):
  `E_total = [Φ(g(c+δ))+Φ(g(c−δ))]/2 − Φ(g(c))`; `E_map = [Φ(g₀+Hd)+Φ(g₀−Hd)]/2 − Φ(g₀)` (the polar-map
  nonlinearity, which exists on a purely quadratic loss); `E_residual = E_total − E_map` (the extra curvature
  from loss-cubic feedback). `map_frac = ‖E_map‖/‖E_total‖`, `residual_frac = ‖E_residual‖/‖E_total‖`.

The separation math was CPU-validated on float64 quadratic/cubic toys first (`test_cubic_diag.py`, 4/4):
`e_loss` is 0 on a quadratic and exactly s²-scaling matching ½T[·,d,d] on a cubic; the separation gives
`E_residual≈0` / `E_map≠0` on a quadratic and `E_residual>0` on a cubic. This is a **frozen-buffer diagnostic
at the gradient-polar center** (`c0=1` simplification), not a full evolving-buffer replay.

## Result (`readout.tsv`, mean over seeds 0,1,2)

| arm | offset | e_loss 0.5/1 | map_frac | residual_frac |
|:---:|:------:|-------------:|---------:|--------------:|
| a05 | 2032/2064 | 0.314 / 0.314 | 0.788 / 0.792 | 0.660 / 0.664 |
| a1  | 2032/2064 | 0.342 / 0.305 | 0.770 / 0.796 | 0.683 / 0.670 |
| a2  | 2032/2064 | 0.313 / 0.360 | 0.772 / 0.795 | 0.686 / 0.688 |

**1. Both mechanisms contribute; the polar map is the larger single contributor.** At every state the
polar-map share (`map_frac ≈ 0.77–0.80`) and the loss-cubic residual share (`residual_frac ≈ 0.66–0.69`) are
both large (they are not orthogonal, so the fractions do not sum to 1). Neither nonlinearity alone explains
the memory-induced curvature change — the answer to the request's question is **both**, with Muon's polar-map
nonlinearity somewhat larger by norm.

**2. The loss-cubic feedback is real but not purely cubic.** The `e_loss` scale ratio (0.5/1) is **~0.30–0.36**
across all states, consistently above the pure-cubic ideal of 0.25 → the finite-scale `e_loss`/`E_residual`
carries **higher-order (>cubic) contamination**. Per the request, this residual must **not** be labeled a pure
third-derivative force; a clean pure-cubic isolation is **not** established.

**3. Memory length weakly modulates the cubic feedback.** `e_loss_norm` and `residual_frac` rise slightly with
longer memory (a2 ≳ a1 ≳ a05; ~5–10% in `e_loss_norm`, `residual_frac` 0.66→0.69). The direction is
consistent (longer memory → marginally more loss-cubic feedback) but the magnitude is small and within the
higher-order-contamination uncertainty.

## The 16-update replay was not run (documented)

The request gates the bounded even-gradient-removal replay on the signal being *resolved*. It is not cleanly
resolved: the scale-ratio contamination (~0.31 vs 0.25) and the small/weak memory modulation are explicit
INCONCLUSIVE conditions the request lists ("a large Taylor residual permits an INCONCLUSIVE result"). The
**both-mechanisms** attribution is the resolved qualitative answer; the replay would test a pure-cubic force
that the scale test does not support, so it was skipped rather than over-interpreted.

## Caveats

- Frozen-buffer diagnostic at the gradient-polar center (`c0=1`), not the exact evolving-mixture center; base
  displacement scale 0.01; joint over the 72 Muon matrices. We do **not** claim momentum multiplies the
  third-derivative tensor at fixed weights.
- Secondary experiment: REQ-059 showed the memory-guided policy has no practical win, so this characterizes
  the mechanism of a marginal effect.

## Files

- `impl/req060_cubic_diag.py` + `impl/test_cubic_diag.py` — separation primitives + 4/4 CPU toy validation.
- `impl/measure_req060_diag.py` — real-model diagnostic (e_loss scale test + E_map/E_residual separation).
- `impl/make_req060_traj.py` — a=0.5/1/2 trajectory configs with state windows at 2032/2064.
- `impl/analyze_req060.py` — aggregation, reproducible from the raw JSONs.
- `readout.tsv` — per-(arm,offset) results + verdict. `raw/diag/{arm}_s{seed}_step{2032,2064}.json` — 18 profiles (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2-node ceiling; node stopped after delivery.
