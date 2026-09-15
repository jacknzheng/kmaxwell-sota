# REQ-064 pilot gate evaluation — ALL GATES PASS → expansion authorized

Node wlv5j0q (1×8 H100, venv019 torch 2.10.0+cu128), seed-0 base @ step 2000 (REQ-063 restore fix applied).
Pilot = 18 sentinels feature probe (FW K20/50 × 5 starts × 3 subsets + nested) + 15 continuations (block-6
sentinels ×2 + 3 globals, 256 updates). `raw/feat_s0/features_req064_state_s0_step002000.json`.

## Measurement/budget gates (pre-registered)

| gate | requirement | result | verdict |
|:--|:--|:--|:--:|
| FW budget K20→K50 | ≥90% of resolved sentinels change ≤5% | 18/18 (100%) on subset0/1/2 | **PASS** |
| S_i/G_i rank stability | median pairwise Spearman across subsets ≥0.8 | ρ = +0.988 (0.988/0.988/0.979) | **PASS** |
| Lanczos λ_i convergence | 8 iters + diagnostics | 10/18 converged, median residual 8.9e-4, max 9.8e-3 | usable + documented |
| nested-token sensitivity | report rank vs absolute separately | S_i rank-corr subset0↔nested +0.986; median abs ratio 0.614 | rank-stable (abs shifts, as REQ-057) |

## Intervention gate (continuations, step-2256 selection loss)

Global ordering is monotone and measurable: **all_shorter 3.37894 < all_reference (a_star=0.5) 3.39287 <
all_longer 3.41064**. Single-matrix (selective) effects are small but present (~±0.0006), e.g. loss(long)−
loss(short) per block-6 sentinel: attn.q +0.00019, attn.k +0.00059, attn.v +0.00050, attn.proj +0.00058,
mlp.fc +0.00076, mlp.proj −0.00008. Most matrices marginally prefer shorter memory; mlp.proj marginally
prefers longer. These per-matrix signals are what M1/M2/Euclidean will be tested to predict on held-out seeds.

## Decision

Intervention + measurement + budget checks **PASS** → expand to the pre-registered **39 arms × 4 bases =
156 continuations** (dev seeds 0,1; prospective test seeds 7,8), then run the registered H1 analysis. The
Lanczos residual caveat (8/18 above the strict 1e-3 threshold, all ≤9.8e-3) is the documented convergence
diagnostic for the Euclidean comparator; it does not block the comparison. Budget: pilot consumed ~1 node
(feature probe ~40 min + 15 continuations); expansion within the 24-node-hour ceiling.
