# REQ-047 — per-token backward statistics — **both alignment questions answered (n=4)**

**SHA `ebf53cd` (EoS serialized-fork-state), 1 node (3m7r5k3, 8×H100), venv019 torch 2.10.0+cu128.**
An increment to the REQ-038/043 probe (`measure_activation_backward_v2.py`): the per-token `acts`/`grads`
tensors already in memory are reduced four more ways — `d_token_norms` / `a_token_norms` (mean, sd,
participation), `da_cos_mean` (adjacent-token backward cosine), `grad_rank1_frac` (σ₁²/Σσᵢ² of
`Σₜ dₜaₜᵀ = W.grad`). Run on Arm A's four fork-1500 seeds (0–3) → **n=4** immediately; one forward+backward
per seed. Base val@1500 = 3.519 / 3.522 / 3.521 / 3.522 (matches REQ-043's 3.52124). **B=8, T=1024.**

> **Correctness detail honoured:** `da_cos_mean` is computed **along the sequence axis within each batch
> row** (`d[:, :-1]` vs `d[:, 1:]` on the `[B, T, F]` tensor), never across the flattened token axis — so
> no spurious decorrelation is injected at sequence boundaries.

## The three registered checks

| check | result | verdict |
|:------|:-------|:-------:|
| **1. `da_cos_mean` q,k < v/proj** (band-14 same-shape attn set) | q,k ≈ **+0.02**, v/proj ≈ **+0.25**, in **4/4 seeds** | **PASS** (≥3/4) |
| **2. `grad_rank1_frac` ~ `align_ratio`** (\|r\|>0.5) | **r = +0.656** over 288 matrix-seed obs | **PASS** |
| **3. corr(a_part, d_part) across depth** (band 27) | near-zero/positive for 5 of 6 types | **scale, not support** |

## Question (a) — why does the softmax Jacobian align less across tokens? → **token-coherence**

Per-type means (n=4, `readout.tsv`):

| type | `da_cos_mean` | `grad_rank1_frac` | `d` participation (tokens) |
|:-----|-------------:|------------------:|---------------------------:|
| **attn.v** | **+0.417** | **0.543** | **456** |
| attn.proj | +0.085 | 0.218 | 3157 |
| **attn.q** | **+0.060** | 0.238 | 941 |
| **attn.k** | **−0.018** | 0.183 | 715 |
| mlp.fc | +0.036 | 0.270 | 2751 |
| mlp.proj | +0.091 | 0.227 | 3320 |

**v's backward vectors are highly coherent token-to-token (cos 0.42) and concentrated on few tokens
(participation 456, the fewest), so its outer products add coherently → high rank-1 (0.54). q and k's
backward vectors are near-orthogonal token-to-token (cos ≈ 0.02; k is even slightly anti-coherent) and
spread over more tokens, so their outer products add incoherently → low rank-1 (0.18–0.24).** The
alignment deficit band 25 measured in aggregate (−0.190 dex) is, mechanistically, **a token-coherence
deficit**: q,k's per-token gradients point in nearly unrelated directions from one token to the next,
while v's reinforce. Check 2 closes the loop — `grad_rank1_frac` (how coherently the outer products
accumulate) correlates with the aggregate `align_ratio` at r = +0.66. **The deficit is token structure,
not sparsity** (q,k are not *more* concentrated than v — the opposite; v is the concentrated one).

## Question (b) — why do ‖a‖ and ‖d‖ trade off across depth? → **scale, not support**

Band 27 found `corr(log‖a‖, log‖d‖) = −0.87…−0.99` across depth with the product 2–4× flatter than
either factor. REQ-047 distinguishes the two readings via the **participation** correlation across depth
(check 3): a *support* trade-off (same total signal spread over more/fewer tokens) predicts strongly
**negative** `corr(a_part, d_part)`; a *scale* trade-off (both tensors rescale in magnitude, support
unchanged) predicts **near-zero**. The measured correlations are near-zero-to-positive for five of six
types (attn.q +0.59, k +0.22, proj +0.32, mlp.fc +0.30, mlp.proj +0.11), negative only for attn.v
(−0.36). **So band 27's forward/backward depth trade-off is predominantly a *scale* effect** — the norms
rescale across depth while token support stays roughly fixed — not a redistribution of signal over tokens.
Note that q/k/v share `a_participation` = 8138 exactly (they read the same input — consistent with band 21).

## Honest scoping

Per the request, this does **not** attempt to revive the aggregate alignment ratio as a predictor of C
(iteration 141 showed it adds nothing once the shared-λ component is removed). It targets only the two
mechanism questions, and both are now measured at n=4: the deficit's origin (token-coherence) and the
depth trade-off's nature (scale). Neither blocks the account of C, which is complete at n=4.

## Files

- `readout.tsv` — the 3 checks + per-type means of the 4 new fields.
- `analyze_req047.py` — the checks, reproducible from the raw JSONs.
- `measure_activation_backward_v2.py` — the augmented probe (adds the 4 per-token reductions; `da_cos_mean` along the sequence axis).
- `raw_json/req047_seed{0..3}.json` — per-matrix stats incl. `d_token_norms`/`a_token_norms`/`da_cos_mean`/`grad_rank1_frac` (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling (one box).
