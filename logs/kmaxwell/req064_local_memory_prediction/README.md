# REQ-064 — predict local memory improvements beyond a strong global setting — **NEGATIVE (H1 FAIL)**

**Harness `365c392d` + REQ-058/059 patches + the REQ-063 restore fix. Node wlv5j0q (1×8 H100, venv019
torch 2.10.0+cu128), stopped after delivery.** Around the calibrated global memory (a_star=0.5), does
measured spectral sharpness predict which individual matrices benefit from changing memory **better than
type/depth and Euclidean features**? Registered prospective-prediction pass; see
[PREREGISTRATION.md](PREREGISTRATION.md) (frozen before any test-seed outcome) and
[PILOT_GATES.md](PILOT_GATES.md) (all measurement/budget gates PASS).

## Design (as pre-registered)

- Global reference **a_star = 0.5** (best of the clean REQ-059 global sweep: a05 3.32943 < a1 3.33795 <
  a2 3.35149; the a-arms declare mu=0.95 so are unaffected by the nomom restore bug REQ-063 found).
- **18 sentinels** (types {attn.q,k,v,proj, mlp.fc,proj} × blocks {0,6,11}); each set alone to a_star/2
  (0.25) or 2·a_star (1.0), other 71 at 0.5. **39 arms/base** (36 selective + 3 global), **256 updates**,
  selection loss at 2000/2064/2128/2256. **4 bases: dev seeds 0,1; prospective test seeds 7,8** (verified
  unused for this task). **156 continuations** total, all completed.
- Per-sentinel primary response r_i = loss(2·a_star arm) − loss(a_star/2 arm) at step 2256. Feature models
  fit on the 18 dev sentinels (seeds 0,1 pooled), predicted on test seeds 7,8 individually.

## Result (`H1_readout.txt`) — sharpness does NOT beat the free type/depth prior

Per-seed RMSE of each 1-D model (predicting r_i on held-out seeds):

| model | seed 7 RMSE | seed 8 RMSE | sign recovery |
|:--|--:|--:|:--|
| **M1 type/depth (measurement-free)** | **0.000256** | **0.000336** | 17/18, 16/18 |
| M2 S_i/G_i sharpness (primary) | 0.000394 | 0.000443 | 17/18, 16/18 |
| Euclidean λ_i/‖g‖_F² | 0.000406 | 0.000471 | 17/18, 16/18 |

**H1 (pre-registered): FAIL.** The gate required M2 to have **lower per-seed RMSE than both M1 and
Euclidean on both test seeds**. Instead **M1 (type/depth) predicts best on both seeds**; M2 and Euclidean
are worse. All three recover the *sign* of the benefit well (≥16/18), but the pre-registration explicitly
states **correlation sign is not the gate** — and on the actual predictive-precision gate, the expensive
spectral-sharpness measurement adds **no value over knowing a matrix's type and depth**.

## Conclusion

**Measured sharpness does not help choose different momentum memories for different matrices beyond a free
type/depth prior, around the strong global a_star=0.5.** This is consistent with and strengthens REQ-059's
negative policy outcome (a guided allocation lost to a good global). Per the pre-registration, **REQ-065's
unconstrained frozen policy trial is NOT authorized** ("no policy trial on a correlation-only pass").

Secondary observations: the global sweep at the 256-update horizon reproduces shorter-is-better around
a_star (all_shorter 3.379 < a_star 3.393 < all_longer 3.411); single-matrix effects are ~±0.0006 and mostly
favor shorter memory. The type/depth structure (not the per-matrix spectral measurement) carries what little
predictable signal exists.

## Caveats

- 1-D least-squares per feature (the pre-registered model form); a multivariate fit was not pre-registered.
- Lanczos λ_i is an 8-iteration estimate (10/18 fully converged, residuals ≤9.8e-3; documented in
  PILOT_GATES.md) — the Euclidean comparator is approximate, but it loses to M1 regardless.
- Per-matrix responses are small (~10⁻⁴) relative to run-to-run noise; the effect being predicted is itself
  marginal, which is itself part of the negative conclusion.

## Files
- `PREREGISTRATION.md` (frozen predictions), `PILOT_GATES.md` (gate pass), `H1_readout.txt` (per-seed RMSE).
- `impl/req064_features.py`+`test_req064_features.py` (Euclidean Lanczos + actual-direction, CPU-validated),
  `impl/measure_req064_features.py` (probe driver), `impl/make_req064_configs.py` (39-arm generator),
  `impl/analyze_req064.py` (H1 test, self-test PASS), `impl/run_req064_pilot.sh`/`run_req064_expand.sh`.
- `raw/feat_all/features_*.json` (4 seeds), `raw/full064/*.tsv` (156 endpoints). No weights/tensors/secrets.
