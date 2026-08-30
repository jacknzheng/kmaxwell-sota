# REQ-019 — generalized-sharpness (Frank–Wolfe) calibration

**Calibrates the paper-faithful global block-spectral generalized sharpness before adding it to the fleet.** Two valid checkpoints from a corrected run (`kmaxwell-sota @ ebf53cd`, the `s100` ×1.0-control fork-1500 arm), one early (step 1500 = the fork point) and one late (step 2750), on the same fixed 131072-token set as the Euclidean-curvature measurement. Node `qvgl1eq` (8×H100), torch 2.10.0+cu128.

## What the measurement is

S(x) = max_v vᵀ H v over the **product of spectral-norm balls** {‖v_m‖₂ ≤ 1} across all **72 Muon block matrices**, solved by Frank–Wolfe. Each FW iteration uses exactly **one joint Hessian–vector product** that couples all blocks (a single scalar dot Σ_n⟨g_n, v_n⟩ over every block, then one second backward → (Hv)_m = Σ_n H_mn v_n); the per-block linear-minimization oracle is the exact spectral-ball polar `r·UVᵀ` from each block's SVD. **Cross-block Hessian terms are present** — this is a single global quantity, not a collection of independent diagonal-block maxima. Parallelism is data-parallel token sharding with an all-reduce of the joint Hv each iteration (matrices cannot be split across ranks because they are coupled). Implementation + tests are under `impl/` (see below).

## Convergence over the iteration budget (one shared restart-0 trajectory)

| step | q@K=5 | q@K=10 | q@K=20 | q@K=50 | Δ 5→10 | Δ 10→20 | Δ 20→50 |
|-----:|------:|-------:|-------:|-------:|-------:|--------:|--------:|
| 1500 | 3.833e8 | 3.993e8 | 4.089e8 | 4.165e8 | +4.17% | +2.40% | +1.85% |
| 2750 | 4.144e8 | 4.255e8 | 4.319e8 | 4.368e8 | +2.68% | +1.49% | +1.13% |

The objective rises with the budget and the relative gains shrink monotonically to ~1–2% over the last (20→50) interval, so **K=50 is reasonably converged** (the late checkpoint converges slightly faster than the early one).

## Restart sensitivity at K=50 (one restart vs five)

| step | single (restart 0) | 5-restart mean | std | min | max | std/mean |
|-----:|-------------------:|---------------:|----:|----:|----:|---------:|
| 1500 | 4.165e8 | 4.072e8 | 7.16e6 | 3.971e8 | 4.165e8 | 1.76% |
| 2750 | 4.368e8 | 4.172e8 | 1.30e7 | 3.995e8 | 4.368e8 | 3.12% |

Five independent initializations (restart 0 = gradient polar; restarts 1–4 = random spectral-unit) agree to within ~2–3% at K=50, and every restart converges to the same order despite very different starts (e.g. q(v₀) ranging 2.5e5 → 1.3e8). The single (canonical restart-0, gradient-seeded) run lands at the top of the spread, so a **single gradient-seeded restart is a good, slightly optimistic estimator**; the 5-restart mean is the conservative central value.

## Cost

Peak CUDA memory **40.8 GiB/rank** (of 80), node wall **~1397 s (~23 min) per checkpoint** for 5 restarts × 51 joint HVPs at 131072 tokens. Linear in restarts × (max_iters+1) × token passes.

## Euclidean Ritz reference (scale only — NOT expected to agree)

Retained for scale from the fork-1500 curvature deliverable (`../req019_eos_state_dependence/`): the per-matrix **diagonal-block** top Hessian eigenvalue for the same `s100` ×1.0 arm is max ≈ 3.96e5 (mean ≈ 2.22e4) at step 2750. The FW **joint spectral-ball** maximum (≈4.37e8 at 2750) is ~10³× larger, as expected — it aggregates coupled curvature over all 72 blocks under unit-spectral perturbations, a fundamentally different observable from any single diagonal-block eigenvalue. No agreement between the two is claimed. (The early-checkpoint Euclidean scale was not separately computed; the fork-1500 curvature window was 2250–2750.)

## Files

- `summary.tsv` — per checkpoint: q@{5,10,20,50}, K=50 restart spread (mean/std/min/max), peak CUDA bytes, wall seconds.
- `objective_trace.tsv` — `step, restart, iter, objective` for every FW iteration of every restart (both checkpoints, 5 restarts × 51 iters).
- `raw/req019_fw_generalized_sharpness.json` — full merged result incl. `relative_changes`, `restart_traces`, `spread_at_k50`, gradient block norms.
- `configs/req019_fw_generalized_sharpness.json` — the exact CLI/config used.
- `fw-cal-console.log` — run console (per-restart q(v₀)→q(v₅₀), timings).
- `impl/measure_generalized_sharpness_fw.py` — the tool (joint cross-block HVP, spectral-ball polar LMO, γ=2/(k+2) one-HVP-per-iter FW, data-parallel token sharding).
- `impl/test_generalized_sharpness_fw.py` — 8 CPU tests: polar-LMO/nuclear duality, **cross-block coupling guard** (joint vs diagonal HVP), FW monotonicity + closed-form-max checks, objective=⟨v,Hv⟩, and init device/dtype contract.

Per the request, this authorizes only the two-checkpoint calibration; no generalized-sharpness fleet is launched pending review of this convergence/restart/cost evidence. Checkpoints and env dumps are not committed.
