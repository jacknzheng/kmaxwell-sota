# REQ-023 — wide per-matrix learning-rate interventions at two fork states

**SHA `25d3208` (codex/per-matrix-lr-public, adds `PerMatrixLrMuon`), node q4g2jyq (8×H100, released), torch 2.10.0+cu128.** Three replicated, type-balanced assignments give each of the 72 Muon matrices each multiplier ∈ {0.6, 1.0, 1.7} exactly once, repeated identically from fork states 1500 and 2000, to ask: does changing one matrix's LR affect **only that matrix** (local thermostat) or does curvature move into **untreated matrices** (collective cross-layer sharpness budget)?

## Gates (all PASS)

- **Offline tests:** 7/7 (`test_per_matrix_lr_muon`, `test_per_matrix_lr_config_generator`, registry); the broad `test_hook_registry_is_locked` failure is pre-existing at parent `ebf53cd` and unrelated.
- **Config integrity:** regenerating from the frozen code byte-matched the committed `requests/req023/` configs.
- **LR-trace gate** (`runtime-lr-trace.tsv`), both forks: loaded pre-update state == serialized source (model exact; optimizer **tensors** 0.000e+00 abs-diff — the file-hash differs only by `PerMatrixLrMuon`'s `lr_multipliers` metadata, momentum `m_fast`/`m_slow` byte-identical); the `learning_rates` row lists all **72 entries in `sorted_index` order**; every `sorted_index → name → multiplier` matches `assignments.tsv`; every effective Muon LR == `0.025 × multiplier`.
- **Shared-state gate** (`shared-state-check.tsv`): the three assignments at each fork load the identical source state (model + optimizer tensors, max abs-diff 0.000e+00).

## ⚠️ Checkpoint-grid defect (frozen config)

The request asks for curvature at {1850,1975,2100,2225,2350} (f1500) / {2350,2475,2600,2725,2850} (f2000). **These are not producible by the committed configs:** `checkpoint_model_at_cadence(every=125)` routes through `step_is_due`, which fires at `step % 125 == 0` **or** `step == train_steps−1 = 3249` (never reached, since the runs stop at 2350/2850). The actual dumps are therefore {1500,1625,1750,1875,2000,2125,2250} / {2000,…,2750}. Fix: give the checkpoint hook `dense_windows`/explicit steps for the requested grid, or set `stop_after_step` so the cadence lands on it. **I ran curvature on the closest-available grid** — f1500 {1750,1875,2000,2125,2250}, f2000 {2250,2375,2500,2625,2750} (within 25 steps of 4 of the 5 requested each; the 5th, the stop step, is absent). All other measurement settings are exactly as REQ-019: 72 Muon matrices, `--iters 8 --tokens 131072`, raw Lanczos alphas/off-diagonals preserved.

## Analysis (`summary.tsv`)

Response = log λ_top [raw] and log(λ_top·‖W‖²_F) [gauge-normalized] at the late checkpoint. **Direct effect** = the own-multiplier slope β from a within-matrix (matrix fixed-effects) regression on x = log(own multiplier), identified from each matrix's 3 assignments. **Cross-talk** = slope of the within-matrix residual (own-multiplier removed) on the mean log-multiplier of neighbor groups (same-block, adjacent-block, same-type, all-other). n = 3 assignments is the replication unit — a discovery experiment, not a significance claim.

### Direct own-multiplier effect — strong, negative, replicated (local EoS law at the matrix level)

| fork | β raw | β gauge | per-assignment slopes (raw) |
|-----:|------:|--------:|:----|
| 1500 | −1.29 (se 0.10) | −0.57 | −1.14, −1.09, −1.63 |
| 2000 | −1.38 (se 0.10) | −0.59 | −1.36, −1.08, −1.70 |

Raising a matrix's own LR multiplier lowers its own curvature with slope ≈ −1.3 (raw) / −0.58 (gauge), consistent across **both forks** and **all three assignments**, and squarely inside the per-matrix slope range REQ-019/022 reported (≈ −2.1 … −1.0). The own-LR → own-curvature control is real and state-independent.

### Cross-talk — no robust signal into untreated matrices

| fork | same-block | adjacent-block | same-type | all-other |
|-----:|-----------:|---------------:|----------:|----------:|
| 1500 raw | +0.00 | **+1.56** | 0.0 (se 1.1) | 0.0 (se 7.0) |
| 2000 raw | −0.02 | −0.09 | 0.0 (se 1.1) | 0.0 (se 7.0) |

- **same-type** and **all-other** are uninformative *by design*: every assignment holds a fixed multiplier histogram (24 matrices at each of 0.6/1.0/1.7; 4 per type), so the mean multiplier over ~all others is ≈ constant across assignments (SE 1.1–7.0). This experiment cannot estimate those channels.
- **same-block** ≈ 0 at both forks (se 0.24): no detectable same-block redistribution.
- **adjacent-block** shows a large positive slope at fork 1500 (+1.56) but **does not replicate** at fork 2000 (−0.09). With only 3 assignment-level configurations the reported SE understates the true uncertainty, and a signal that flips to ~0 at the paired fork is consistent with noise, not a collective law.

## Verdict

**The evidence favors a LOCAL response, with cross-talk unresolved (underpowered).** The direct own-multiplier effect is strong, negative, and reproduces across both fork states and all three assignments (β ≈ −1.3 raw / −0.58 gauge) — changing a matrix's learning rate primarily moves *its own* curvature. The informative cross-talk channel (same-block) is ≈ 0 at both forks; the only large cross-talk estimate (adjacent-block at fork 1500) fails to replicate at fork 2000 and is consistent with n=3 noise; the same-type and all-other channels are non-identifiable under this balanced design. So there is **no robust evidence of a collective cross-layer sharpness budget** at three assignments, but the design has limited power to exclude a weak effect — most sharply for the adjacent-block channel, which merits more assignments before the redistribution hypothesis can be ruled out. The per-matrix rows are preserved (each `<run_id>/per_matrix_curvature.json`, plus `weight_norms.tsv`) so later assignments can extend the regression without rerunning these six continuations.

## Files

- `summary.tsv` — per fork/response: val_loss, direct β (raw & gauge, se), the four cross-talk group estimates (se), per-assignment slopes; leading comments carry the grid caveat.
- `shared-state-check.tsv`, `runtime-lr-trace.tsv` — the two gates.
- `assignments.tsv` / `assignments.json` / `manifest.tsv` — the frozen design.
- `weight_norms.tsv` — per-matrix ‖W‖_F at every curvature checkpoint (for the gauge normalization).
- `configs/` — the six committed configs.
- `req023_f{1500,2000}_a{0,1,2}/` — per run: `command.txt`, `console.log`, `train-log.txt`, `per_matrix_curvature.json` (74 matrices — 72 Muon + embed/proj — × 5 checkpoints, full Lanczos).

Checkpoints, optimizer shards, `eos_shared_state`, FineWeb data, and env dumps are **not** committed.
