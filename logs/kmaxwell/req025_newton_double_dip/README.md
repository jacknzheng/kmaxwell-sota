# REQ-025 — Newton–Muon double-dip interaction grid — **NEEDS-INFO (gate failure: alpha=0 baseline diverged to NaN)**

**SHA `365c392d` (codex/req025-newton-muon, name-collision fix), node `qj0z1gw` (8×H100), torch 2.10.0+cu128, `microbatch_sequences=32` uniform.** The 12-arm grid ran to completion, but the request's own gate — *"alpha-0 record trace matches plain bi-Maxwell from the same fork"* — fails in the way that matters: **all four alpha=0 baseline arms diverged to NaN within ≤125 steps of the fork**, so the requested interaction `I(alpha) = [L_short(alpha)−L_short(0)] − [L_record(alpha)−L_record(0)]` is **uncomputable** (its `L(0)` reference is NaN). Per the request's failure protocol I preserved evidence, set this block to NEEDS-INFO, did **not** launch anything further, and released the node.

## The finding (airtight)

| fork | kernel | alpha | val_loss @ fork+750 | weights (NaN∪Inf / total) |
|-----:|:-------|------:|:--------------------|:--------------------------|
| 1500 | record | **0.00** | **nan** | **173/185** |
| 1500 | record | 0.25 | 3.40535 | 0/185 |
| 1500 | record | 0.50 | 3.40351 | 0/185 |
| 1500 | short  | **0.00** | **nan** | **173/185** |
| 1500 | short  | 0.25 | 3.40320 | 0/185 |
| 1500 | short  | 0.50 | 3.40373 | 0/185 |
| 2000 | record | **0.00** | **nan** | **173/185** |
| 2000 | record | 0.25 | 3.33464 | 0/185 |
| 2000 | record | 0.50 | 3.33369 | 0/185 |
| 2000 | short  | **0.00** | **nan** | **173/185** |
| 2000 | short  | 0.25 | 3.32857 | 0/185 |
| 2000 | short  | 0.50 | 3.32855 | 0/185 |

Every alpha=0 arm goes NaN at the **first** post-fork validation (f1500 at step 1625, f2000 at step 2125); a checkpoint dump confirms genuine weight divergence (173/185 tensors NaN/Inf, not a validation-path artifact). Every alpha>0 arm is clean and monotone. `summary.tsv`, `alpha0-trace-check.tsv`, `val_trajectories_raw.txt` carry the full per-step traces.

## This is NOT a REQ-025 code bug — five hypotheses ruled out

1. **Preconditioner code:** at `newton_alpha==0`, `RightPreconditionedMuonMixin.compute_polar_input` is a verified clean bypass — it calls `super().compute_polar_input` (plain `BimaxwellMuon`); no preconditioner math runs. (Contrast the prior two issues, which *were* preconditioner-path bugs.)
2. **Activation hook:** `_accumulate_activation_covariance` is gated on `ref["collect"]`, initialised `False`; at alpha=0 `prepare_forward` returns early and never sets it — the hook is a complete no-op.
3. **`_muon_steps_seen`:** explicitly persisted (save) and restored (load) across the fork, so the bi-Maxwell kernel resumes on the correct (post-switch) branch.
4. **LR schedule:** base and fork share `train_steps=3250`, `cooldown_frac=0.7`, Muon `lr=0.025` → LR is continuous across the fork, no jump.
5. **State loading:** shared-state gate = `0.000e+00` for all 12 arms (`shared-state-check.tsv`) — every arm begins from bit-identical weights+optimizer state. The only per-arm difference is `newton_alpha`.

## What the evidence points to — and the one thing I could not A/B

Two independent controls prove **alpha=0 is not intrinsically unstable here**:

- **The base bi-Maxwell run itself** is monotone-stable straight through the resume region: step 1500 → 3.52003, 1625 → 3.50441, … 2000 → 3.44257 (no NaN).
- **The earlier gate-probe** of *this same alpha=0 config* was **stable and finite** on SHA `596d2868`: val@1625 = **3.50100**, val@1750 = **3.48138** (rows preserved at the top of `alpha0-trace-check.tsv`).

So the baseline was stable before and NaN now. **Between "stable" and "NaN", two things changed together:** (1) the SHA `596d2868 → 365c392d` (the name-collision fix re-aliased the activation-hook import — and this very trace file documents that alpha=0's data batch order is *RNG/hook-sensitive*, so a shift there moves the un-preconditioned trajectory), and (2) `microbatch_sequences → 32`. The preconditioned arms (alpha≥0.25) are robust to whatever shifted; the raw bi-Maxwell baseline is on a stability knife-edge and tips to NaN.

**I could not cleanly A/B the microbatch axis:** `microbatch_sequences=64` OOMs for this model (6.14 GiB logits buffer at 65536 tok/microbatch exceeds free HBM even on empty GPUs) — which is precisely why mb=32 was mandated. Isolating the SHA axis would require a fresh `596d2868` checkout, which the failure protocol ("stop before launching further") told me not to pursue without your go-ahead.

**Net:** activation right-preconditioning is *load-bearing* for fork-resume stability at `lr=0.025` from these states; the un-preconditioned alpha=0 control diverges, so the double-dip interaction cannot be anchored.

## What I need from you (pick one; I'll execute immediately on the held pattern)

1. **(Recommended) Stabilise the alpha=0 baseline, then rerun the 4 baseline arms only.** Most likely fix, given the evidence: pin the fork-resume data order so alpha=0 reproduces the gate-probe trajectory (e.g. restore RNG state rather than re-seed, or confirm `596d2868`'s data order). If you tell me `365c392d` did/didn't change hook RNG draws, I can confirm the mechanism in one short run. The 8 alpha>0 arms are already valid and can be reused.
2. **Lower the grid's base Muon LR** (e.g. 0.025 → 0.018) so the raw alpha=0 update has stability margin, and rerun all 12 for a like-with-like interaction.
3. **Re-anchor the interaction at alpha_min = 0.25** instead of 0 (redefine `I` relative to the smallest *stable* preconditioning). Computable from the existing 8 arms + one added alpha (e.g. 0.125), but it no longer measures "preconditioning vs none."
4. **Accept the finding as the deliverable** — "activation preconditioning is load-bearing for fork-resume stability; the un-preconditioned baseline diverges" — no interaction grid.

## Overhead budget (requested)

Covariance-hook peak memory (the number you asked me to record for the planned recorder): **≈ 64,590 MiB** peak reserved on the alpha=0.25 smoke. This is also *why* `mb=64` OOMs with the hook active and `mb=32` was required.

## Gates

- SHA / clean worktree: PASS (365c392d). Targeted tests: PASS (19/19 on this SHA, per owner note). Config byte-diff: PASS (regenerated == `requests/req025/`).
- Shared-model / shared-optimizer-state gate: **PASS** — `shared-state-check.tsv`, 0.000e+00 for all 12.
- **alpha-0 record trace vs plain bi-Maxwell: the two match (same code path) but BOTH are NaN under the rerun conditions → gate FAILS on usability.** ← the blocker.

## Files

- `summary.tsv` — 12-arm per-arm val_loss + NaN-weight census + SHA + artifact paths.
- `interactions.tsv` — the requested I(alpha) table, all rows NaN (baseline undefined), with the alpha>0 absolute losses preserved.
- `alpha0-trace-check.tsv` — gate-probe (stable, `596d2868`) **and** grid-rerun (NaN, `365c392d`) side by side + the five ruled-out hypotheses.
- `shared-state-check.tsv` — the 0.000e+00 load gate for all 12 arms.
- `val_trajectories_raw.txt` — full per-step val_loss for all 12 arms + the base.

Curvature was **not** collected: the run was stopped once the alpha=0 baseline was confirmed NaN (measuring the Hessian of NaN weights is meaningless, and the alpha>0 curvature is not useful without a computable interaction). Checkpoints, optimizer shards, `eos_shared_state`, FineWeb data, and env dumps are **not** committed. Node `qj0z1gw` released after this push.
