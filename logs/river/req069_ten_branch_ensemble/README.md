# REQ-069 — ten data-seed branches: do independent minibatch sequences reproduce the slow component? — **NEGATIVE confirmation**

**Harness `365c392d` + REQ-068 capture patches. Node qz0mpk3 (1×8 H100), stopped after delivery.** Ten
branches forked from the retained middle fork **θ_1500** with byte-identical model weights + full optimizer
state + schedule position, varying only the data-sampling sequence (branch b consumes disjoint corpus
window batches 1500+b·64 … via `skip_batches`; the loader is deterministic-sequential, so disjoint offsets =
ten independent minibatch sequences). 64 updates/branch, both streams (g_t + δ_t) + probe states at offsets
32/48/63 (27/branch). Ensemble analysis streams all 10 branches (`req069_ensemble.py` / `_analyze.py`).

## Result (`analysis/ensemble_{grad,disp}.json`)

| stream | indiv lag1/2/4/8 | ensemble lag1/2/4/8 | two-group agree (mean; **first→last step**) | ens-mean norm n=1/2/5/10 | dispersion | mean/rms | ens-mean family |
|:--|:--|:--|:--|:--|--:|--:|:--|
| gradient | −.64/+.46/+.23/+.05 | −.64/+.48/+.22/+.01 | +0.159; **0.94 → −0.03** | 0.12/0.08/0.04/0.02 | 4.05 | 0.24 | mlp.proj .41, attn.v .31 |
| displacement | +.79/+.63/+.40/+.14 | +.83/+.70/+.50/+.26 | +0.148; **0.90 → +0.04** | 1002/718/467/344 | 2.76 | 0.34 | **embed 1.0** |

**1. The within-branch signatures reproduce qualitatively.** Every branch shows REQ-068's pattern: gradient
anti-alignment (lag1 ≈ −0.64) and displacement persistence (lag1 ≈ +0.79). The phenomena are robust to the
data seed.

**2. But there is NO reproducible common whole-model direction across independent data.** The decisive test
— two predeclared independent groups of five — shows agreement **collapsing from ~0.9 at the fork point to
~0 within 64 steps** (grad 0.944→−0.029; disp 0.899→+0.042). At step 1500 all branches share θ_1500 (and
nearly the same first minibatch structure), so they agree; as independent data drives them apart, the
whole-model direction becomes **unreproducible** (mean agreement +0.15–0.16). The ensemble-mean norm
**shrinks** as branches are added (grad 0.12→0.02; disp 1002→344) — incoherent averaging, not a reinforcing
shared direction — and mean-to-rms is low (0.24/0.34), i.e. most of each branch's motion is branch-specific.

**3. The ensemble mean retains alternation and the persistent δ — but from shared initial conditions, not
data-reproducibility.** Ensemble averaging does **not** cancel the gradient's period-two (ensemble lag1
−0.637 ≈ individual −0.639): the bounce is a **shared phase** inherited from the common θ_1500 (exactly the
flagged caveat). The displacement's ensemble persistence (lag1 +0.83) is likewise the **embedding-dominated**
(family share 1.0, AdamW lr0.7) shared-initial-condition component, not a direction that independent data
converges onto.

## Conclusion

**Independent minibatch sequences do NOT reproduce a common slow whole-model "river" direction.** They
reproduce the qualitative signatures (anti-aligned gradients, persistent displacements), but genuine
data-independent agreement decays to ≈0 within the 64-step window, the ensemble averages incoherently
(shrinking mean norm, high dispersion), and the surviving persistent/alternating structure is a shared
initial-condition + embedding artifact. This **tempers REQ-068's candidate river**: the locally persistent
displacement is largely shared-IC/embedding, not a reproducible low-dimensional descent direction — fully
consistent with REQ-068's "bending river, not a fixed low-rank valley floor." A negative/ambiguous result
is a valid outcome per the request; **no expansion to more forks or full training is warranted on this
evidence**, and the 500/2500 windows were not run (the spec gates them on positive middle-window evidence).

## Caveats / honesty
- Whole-model Euclidean geometry is embedding-dominated for δ (AdamW lr0.7 vs Muon lr0.025) — a
  layer-reweighted view (separately labeled) would be needed to probe a hidden-matrix river.
- 10 branches = limited; dispersion reported, not p-values. Branches share the θ_1500 initial bounce phase.
- Mean gradient at different weights ≠ gradient at mean weights (branches diverge within the window).

## Files
`analysis/ensemble_{grad,disp}.json` (lag profiles, two-group agreement per step, convergence, per-family,
64×64 ensemble-mean C). `impl/{make_req069_branch_config,req069_ensemble,req069_ensemble_analyze,run_req069}.py`
(+ `test_req069_ensemble.py` 4/4). Raw per-branch tensors + probe states off-Git on durable FS
(`req069/br{0..9}/{grads,disp,probes}`). Node qz0mpk3 stopped. No secrets/weights/tensors committed.
