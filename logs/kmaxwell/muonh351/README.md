# REQ-013 — K-Maxwell on PR #351 MuonH fast-slow decay: **CONFIRMED RECORD**

**Result: K-Maxwell's first-moment kernel transfers to MuonH fast-slow decay and sets a new per-optimizer record — first statsig-passing boundary at step 3065, vs PR #351's 3125 (60 steps earlier), confirmed at n=8 (paired t=21 @3125).**

## Which hyperparameter mattered

The gain is driven by **annealing the K-EMA mixture toward short memory (mean age 50→22) with an early onset (km_start=750)**. K itself (4/6/8) and μ (0.95) are secondary — μ=0.95 is optimal and K∈{4,6,8} are within seed-0 noise. The winning config:

```
train_gpt_muonh_kmaxwell.py \
  --warmup_end 100 --plateau_end 200 --fast_decay_end 1750 --peak_lr 0.030 \
  --floor_lr 0.006 --fast_decay_exponent 0.6 --slow_decay_schedule linear \
  --min_lr 0.0 --train_steps 3125 --seed <s> \
  --k 6 --tau-min 3 --tau-max 64 --km-start 750 --mu 0.95 --anneal-frac 1.0 \
  --age 50 --age-end 22
```

Only MuonH's first moment changes (single EMA → deterministic log-spaced K-EMA convex mixture, lazy-init switch bit-identical to baseline); NS direction, `scale_invariant_update_`, hyperball, param groups, and the fast-slow LR schedule are untouched. Minimal diff vs PR #351: `muonh_kmaxwell.diff`.

## Reproduction gate (before any sweep)

MuonH is not bit-deterministic across runs (NCCL/cuBLAS atomics). The no-K-Maxwell path (`--k 1`) reproduces PR #351 **within** that nondeterminism — closer than two base runs reproduce each other (|base1−base2|@3125=0.00051 vs |base1−variant_k1|=0.00028). The lazy-init switch is bit-identical by construction (every K-buffer seeded from the just-advanced single-EMA momentum; the switch-step update calls the identical `muon_update`). So the variant is code-faithful to PR #351.

## Seed-0 staged screen (`sweep.tsv`)

Every screened config beat the control; the improvement grows monotonically toward younger anneal end-age and earlier onset. Winner K=6/50→22/km750/μ0.95: seed-0 −0.00344 @3125 (~17σ over the ~0.0002 seed-0 noise), crosses 3.28 at 3025.

## n=8 confirmation (`summary.tsv`)

Candidate (winner) + exact PR #351 control, seeds 0–7, same hardware, Track-3 protocol (`margin=(3.28−mean)·√8`, pass ≥0.004):

| quantity | candidate | control (PR #351) |
|----------|-----------|-------------------|
| 8-seed mean val_loss @3125 | **3.27627** | 3.27912 |
| Track-3 first-passing boundary | **3065** (margin +0.00473) | 3125 (n=8 margin +0.00250, fails at 3125) |
| per-seed cross of 3.28 | 3025–3070 | mostly 3125; 2 seeds never cross by 3125 |

- **First-passing boundary 3065 < 3125 → success / new record** (60 steps earlier).
- **Paired Δ(ctrl−cand)@3125 = +0.00284, sd 0.00038, t = +21.2** (df=7, p ≪ 0.001). All 8 seeds improve; not outlier-driven.
- **vs PR #351 published n=20 mean (3.278994):** equal-step statistic `(3.278994 − 3.27627)/√(1/20+1/8) = +0.00650` (≫ 0.004).
- 3065 is 12 five-step notches below 3125 (not within one notch), so no seeds 0–19 extension needed.

## Artifacts

- `summary.tsv` — n=8 per-seed candidate + control, means, margins, first-pass, paired stats.
- `sweep.tsv` — full seed-0 staged screen (stages 1–5).
- `muonh_kmaxwell.diff` — minimal diff of the K-Maxwell variant vs PR #351 `train_gpt_muonh_fast_slow_decay.py`.
- `n8logs/` — raw per-seed logs (8 candidate + 8 control).
- Seed-0 screen logs: `logs/muonh351/`.

**Verdict: K-Maxwell beats PR #351's MuonH per-optimizer SOTA — 3065 vs 3125 steps, statistically confirmed. Recommend cutting a record folder/PR from the winning config.**
