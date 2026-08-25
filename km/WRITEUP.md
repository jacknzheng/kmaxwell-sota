# K-Maxwell momentum: annealing study + world-record test — full write-up

_Author: automated run (lagrange-2). Date: 2026-08-24. All artifacts under `~/jackzhengretardruns/`._

---

## 0. TL;DR

1. **Goal (as given):** get val_loss **< 3.28 at step 3150** on the modded-nanogpt
   Track-3 **K-Maxwell ablation trainer**. **Achieved, robustly** — via a new
   idea (annealed momentum mix), an **8-seed mean of 3.27885** (best single 3.27761),
   vs a frozen-tuning plateau of ~3.2815.
2. **The idea that worked:** the K-Maxwell momentum mix should **not be frozen**.
   Anneal its mean-age from **old (~58, heavy averaging early) → young (~26,
   responsive late)** over training. Direction is load-bearing (old→young wins;
   young→old is worse); wider is monotonically better.
3. **World-record test (the important caveat):** the ablation trainer is a *weak*
   trainer. On the **actual current record** (#46, the SOAP + Cautious-Weight-Decay
   + tail-EMA stack, 2690 steps), **annealing does NOT help** — its tail-EMA already
   provides the late-stage averaging annealing was buying. There, the **frozen**
   K-Maxwell mix is best.
4. **A dangling, unverified lead:** frozen K-Maxwell stacked on the SOAP record
   trainer crosses 3.28 at a **~2658-step mean (n=7)**, *below* the published 2690
   record. This is **not yet a validated record** — my crossing metric may not match
   the leaderboard's acceptance protocol. Verifying this is the #1 open task.

---

## 1. Two different trainers — do not conflate them

This is the single most important thing to understand.

| | **Ablation trainer** | **WR trainer** |
|---|---|---|
| file | `records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py` | `records/track_3_optimization/results/20260823_kmaxwell_wr_stack/train_gpt_cwd_kmaxwell.py` |
| what it is | bi-Maxwell baseline (leaderboard #13, 3210 steps); plain SDPA, no FP8 — a **weak** trainer for studying Muon momentum in isolation | the **#46 record stack**: SOAP-Muon + tail-EMA + Cautious Weight Decay + row-floor |
| schedule | 3250 steps | 2900 steps (stop-early at 2720) |
| metric used | **val_loss @ step 3150** (lower better) | **step at which tail-EMA val_loss crosses < 3.28** (fewer better; record = 2690) |
| K-Maxwell default | k=2 exact (bimaxwell) | k=6, frozen mix, mean age ≈ 35 |

"Under 3.28 @3150" was the goal **on the ablation trainer**. That number is **not**
comparable to the 2690-step world record — different trainer, different metric.
The WR test in §5 is the separate, harder question.

Also present (WR-class, same K-Maxwell hooks): `train_gpt_aurora_kmaxwell.py`,
`train_gpt_adamh_kmaxwell.py`. AdamH does not reach 3.28 in its 4875-step budget
here; Aurora only got a partial run (its frozen control was killed mid-run).

---

## 2. The K-Maxwell mix (what the knobs mean)

The Muon optimizer's single-EMA momentum is replaced, from step `start` (=1000)
onward, by a **mix of K fixed-rate EMA buffers** whose ages τ are log-spaced in
`[tmin, tmax]` (β = τ/(τ+1)). The mix weights are a ramp, solved so the **mean age
Σ(w·τ)** hits a target. A run is fully specified by:

`{k, tmin, tmax, age (or explicit weights), shape, seed, start}`

`shape` ∈ linear|quad|cub|geo|sqrt|flat (+ reversed rlinear/rquad, + native gaussian).
Weights are **derived on the box** from these knobs by the repo's own
`kmaxwell_kernel.py` — never hand-typed (anti-hallucination anchor).

---

## 3. Reproducible harness & documentation system (`~/jackzhengretardruns/`)

Built to make every run reproducible and auditable at scale.

```
km/
  solve.py          spec -> weights + filter stats (reuses repo kernel); log parser
  kmrun.py          one ablation run: pin-check -> resolve -> torchrun -> verdict.json
  kmanneal.py       annealed ablation run (start-age -> end-age weight lerp)
  kmwr.py           WR-trainer run (cwd/aurora/adamh/muon); parses crossing step
  kmqueue.py        sequential queue driver over JSONL specs (dispatches the right runner)
  kmbackfill.py     reconstruct a ledger entry from a pre-harness stdout log
  pins.json         content-hash pins for train_gpt_kmaxwell.py + kmaxwell_kernel.py
  pins_anneal.json  content-hash pin for the annealed ablation trainer
  aggregate.py      pull ledgers from all boxes -> INDEX.tsv + SCOREBOARD.md
  queues/           JSONL queue files per box per stage
  train_gpt_kmaxwell_anneal.py   the annealed ablation trainer (new file, pinned)
ledger/<name>/      per run: spec.json, verdict.json, cmd.txt, train.log
SCOREBOARD.md       auto-rendered ablation leaderboard
README.md           running log of findings (this study's scratch history)
STATE.md            live box->queue map for crash recovery
WRITEUP.md          this document
```

**Reproducibility anchor:** the code that computes the update is pinned by
**sha256 content hash** (not git state, because boxes ran from a tarball). Every run
refuses to launch unless the trainer + kernel hashes match `pins.json`. Logical git
ref: branch `track3-kmaxwell-wr-stack` @ `e4c5624` (local; origin tip had diverged).
A run reproduces exactly from `(pins.json, spec)`.

**Annealed trainers** are separate files with their own hashes; the frozen code path
is byte-identical to baseline when no `--weights-end` is given.

---

## 4. The ablation study (goal: < 3.28 @ 3150)

### 4a. Frozen sweep → a plateau
~50 runs mapping the frozen mix. Every axis is flat at its optimum:
- **age:** best k8≈38, k10≈36. Off-optimum punishes (+1 past optimum ≈ +0.0009).
- **shape:** linear beats every convex/concave/reversed/gaussian variant.
- **buffers k:** k8 ≈ k10 (at their own best age); k12+ worse.
- **window [tmin,tmax]:** [3,56] best; widening/narrowing loses.
- **start:** 1000 best; 800/1200 lose.
- **noise_gain** (closed-form variance) does **not** predict val_loss — falsified.

Best frozen single seed **K8_a38 = 3.28003**; but the **8-seed mean ≈ 3.2815**
(above target). Two analysis sub-agents established:
- **Ridge law: optimal_age ≈ 46 − k** (k8→38, k10→36 anchor it).
- **The seed dominates the config** (crossing r = 0.995, Spearman ρ = 1.0 across
  configs). The variance is a **rigid trajectory offset set early and held to the
  end** (constant ~0.0028 gap best-vs-worst at every step) — NOT eval noise.
  → No frozen config's *mean* clears 3.28; and the offset being set *early* is the
  clue that the optimal mix is **time-varying**.

### 4b. The annealing breakthrough
Made the mix time-varying: interpolate the weight vector from a start-age vector to
an end-age vector over the post-`start` steps (one-line change in `Muon.step`:
`weights = km_weights.lerp(km_weights_end, frac)`).

Results (val_loss @ 3150, ablation trainer):
- First hit: **old2young 44→30 = 3.27931** (seed 0, deterministic).
- **Wider is monotonically better**; direction essential (young2old 3.2831 = worse).
- **Champion: k8 [3,64], anneal 58→26.**
  - **8-seed mean = 3.27885** (under 3.28 by 0.00115); seeds
    [3.27761, 3.27771, 3.27805, 3.27857, 3.27891, 3.27901, 3.28045, 3.28051].
  - **Best single run = 3.27761.** Every seed beats the frozen best-ever single (3.28003).
  - `xxwide` 54→28 (k8 [3,56]): 7-seed mean 3.27931 (also robustly under).
- **38 of 47 annealed configs landed under 3.28.**

**Why it works:** early training (large, noisy gradients) wants heavy averaging = old
momentum; late training (fine-tuning) wants responsiveness = young. The frozen ridge
(age≈46−k) was the *time-average* of an optimum that moves. Annealing tracks it.

---

## 5. World-record test (the harder, honest question)

Ported the annealing to the **actual #46 record trainer** (`train_gpt_cwd_kmaxwell.py`,
SOAP + CWD + tail-EMA). Metric = step where tail-EMA val_loss crosses 3.28 (record = 2690).

- **Aggressive anneals hurt:** 50→24 / 54→22 / 50→26 didn't even cross by 2720;
  46→28 = 2705; 44→26 = 2680 (frozen seed-0 was 2670).
- **Mild anneal 38→34 looked like a 2655 win on one seed — but it was noise.** The
  **frozen n=7 mean is 2658** (range 2630–2680), so 2655 is a mid-distribution draw.
- **Conclusion: annealing does not beat frozen on SOAP.** Almost certainly because
  SOAP's **tail-EMA already does late-stage averaging** — annealing's "responsive-late"
  benefit is already captured, and its "heavy-early" half just slows the crossing.
- **Frozen-age sweep on SOAP (seed 0):** ages 33–37 all cross in 2645–2680 (inside
  seed noise); age38 = 2705 (worse); k8 ≈ k6. So frozen age ~34–35 is optimal and flat.

### The unverified lead
Frozen K-Maxwell on SOAP crosses at **~2658 mean (n=7)**, *below* the published
**2690** record. **This is not a validated claim.** My crossing metric = "first eval
with EMA-loss < 3.28 on stop-step-2720 runs." The leaderboard uses a statsig protocol
at a "reported_step" (n=8/10 with p-values). Until re-scored under the *exact*
leaderboard metric, treat "2658 vs 2690" as promising-but-unconfirmed.

---

## 6. Key numbers

| context | config | metric | value |
|---|---|---|---|
| ablation @3150 | frozen best single (K8_a38) | val_loss | 3.28003 |
| ablation @3150 | frozen 8-seed mean | val_loss | ~3.2815 |
| ablation @3150 | **anneal 58→26 champion, 8-seed mean** | val_loss | **3.27885** ✅ |
| ablation @3150 | anneal 58→26 best single | val_loss | 3.27761 |
| WR (SOAP) | published #46 record | cross step | 2690 |
| WR (SOAP) | **frozen K-Maxwell, n=7 mean** | cross step | **~2658** (unverified) |
| WR (SOAP) | annealed (any) | cross step | ≥ frozen (no gain) |

---

## 7. Exact reproducible recipes

**Ablation champion (under 3.28 @3150):** annealed trainer, k8, window [3,64],
mean-age 58→26 over training:
```
torchrun --standalone --nproc_per_node=8 -- \
  records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell_anneal.py \
  --seed <s> --start 1000 --k 8 --tau-min 3 --tau-max 64 \
  --weights <age58 vec> --weights-end <age26 vec> --anneal-frac 1.0
```
(weight vectors are solved by `km/solve.py solve_weights(8,3,64,58,'linear')` and `(…,26,…)`).

**Best no-anneal setup on the WR (SOAP) trainer:** frozen K-Maxwell, k6, [3,56], age ~34–35:
```
STOP_STEP=2720 torchrun --standalone --nproc_per_node=8 -- \
  records/track_3_optimization/results/20260823_kmaxwell_wr_stack/train_gpt_cwd_kmaxwell.py \
  --seed <s> --k 6 --tau-min 3 --tau-max 56 --weights <age35 vec>
```

Both are driven reproducibly via `km/kmrun.py` / `km/kmwr.py` from a spec file.

---

## 8. Open tasks (ranked)

1. **Validate the SOAP frozen-K-Maxwell lead against the real record metric.** Pin
   down the leaderboard's exact acceptance (reported_step + statsig protocol from
   `records/track_3_optimization/README.md` and the #46 PR), re-score frozen
   K-Maxwell (age ~34–35) at n=8/10 the same way, and compute p-value vs 2690. This
   is the difference between "promising" and "a submittable record."
2. If it holds: submit frozen K-Maxwell-on-SOAP as a record; write the PR.
3. Aurora/AdamH: complete a clean frozen-vs-anneal n=8 (Aurora's control was killed;
   AdamH doesn't reach 3.28 in budget — may be out of the record regime).
4. Ablation side is done; the annealing result is a clean academic finding
   ("time-varying momentum-mix age beats any frozen age on a trainer without tail-EMA").

---

## 9. Infra notes / gotchas (for whoever runs this next)

- **6 GPU boxes** (8×H100 truss workstations), currently **idle**: job-ids
  `3mpryz3, qed6y1w, qvg5eeq, qj0gkgw, wdp8knw, 3mpk7o3`
  (`training-job-<id>-0.ssh.baseten.co`). Repo at `/root/modded-nanogpt`, venv
  `/root/venv` (torch 2.10.0+cu128), harness at `/root/km`, 20 fineweb chunks present.
  **Stop them when done:** `truss train stop --job-id <id>`.
- **Launch detached with plain `nohup python3 … & ` — NOT `(setsid nohup … &)`.**
  The subshell/setsid form silently failed on these boxes; plain nohup persists.
- **Never `pkill -f kmqueue` in the same shell that then launches kmqueue** — the
  launcher's own command line contains "kmqueue.py", so pkill kills itself before
  launching. Kill in a separate ssh pass.
- **`kmqueue.py` hangs after a WR run** (its `subprocess.run(capture_output=True)`
  waits on an orphaned torchrun pipe). For WR runs, launch one `kmwr.py` per box
  directly instead of via the queue, OR fix kmqueue to not capture stdout.
- zsh does not word-split unquoted vars: write ssh flags inline, not `$SF`.
- Local background pollers get culled frequently; on-box detached runs survive, so
  treat box state as source of truth and re-poll. `km/aggregate.py pull` re-syncs.
- WR trainer speed ≈ 400 ms/step (SOAP is heavy) → ~18 min to stop-2720; AdamH's
  4875-step schedule ≈ 2× longer.

---

# Part 2 — CWD n=8 validation, #339 pairwise, anneal vs #340 (2026-08-25)

_Prompt: ~/Downloads/AGENT_PROMPT_CWD_N8.md. Fresh 6-box 8×H100 fleet
(wlmz82q, 3y51enw, wxgj90q, q974r03, qrgxkr3, qz7dvew — all STOPPED after harvest).
Scoring = the Track 3 acceptance rule everywhere: at each grid step S,
margin = (3.28 − mean over 8 consecutive seeds) × √8; reported step = smallest
S with margin ≥ 0.004. First-cross is recorded as a diagnostic only._

## Fleet A — frozen K-Maxwell on the #46 CWD stack: VALIDATED, beats #46

k6 [3,56] age 35 linear start 1000, seeds 0–7, `train_gpt_cwd_kmaxwell.py`
(pin 77e74b7e…, logging-only diffs vs the #46 stack).

- **First statsig-passing step = 2680** (margin 0.00432; 2675 fails at 0.00343).
- **Beats the published #46 record (2690) by 10 steps** under the same rule.
- At 2690 our mean = 3.277818 vs #46's published 3.278329 — but pairwise vs
  their published mean, lhs = (3.278329−3.277818)/0.5 = **0.00102 < 0.004: NOT
  a significant pairwise win at equal step**; the win is the 10-step earlier pass.
- Part 1's "~2658 (n=7 first-cross)" was indeed the first-cross trap: properly
  scored the number is 2680.
- Full grid: `logs/kmaxwell/cwd_frozen_n8/summary.tsv` (+ seed{0..7}.stdout/.txt).

## Fleet B — PR #339 reproduction: REPRODUCES, and it beats us

`train_gpt_bimaxwell_st1000.py` byte-identical from orange4664@track3-bimaxwell-2635
(upstream sha 07e064a0…, logging-only diffs), seeds 0–7 on our H100s.

- **First-passing step = 2640** (published H100 claim: 2645 → reproduces, one
  5-step notch faster on our boxes). The baseline is NOT broken.
- **Pairwise #339 vs frozen K-Maxwell at every grid step: lhs ≈ −0.0043…−0.0057**
  (bm_mean below km_mean by ~2.2–2.8 mloss) → **#339 is significantly better
  than frozen K-Maxwell on our hardware**. Step gap at first-pass: 40.
- Verdict: frozen K-Maxwell-on-CWD beats the *merged* leaderboard (#46 2690)
  but NOT the unmerged #339 claim. If #339 merges, K-Maxwell is behind.
- Grids: `logs/kmaxwell/bimaxwell339_n8/summary.tsv`, `logs/kmaxwell/pairwise_km_vs_339.tsv`.

## Fleet C — annealed mix vs PR #340 (Muon family): BEATS #340

Ablation trainer + anneal (58→26, k8 [3,64], frac 1.0), seeds 0–7 re-run from
pins (the Part-1 logs died with the old fleet). Raw val_loss, 10-step tail grid.

- **First-passing step = 3160** (margin 0.00584). 3150 fails (mean 3.278726,
  margin 0.00360) — exactly as Part 1's champion mean predicted.
- **#340's record is 3210 → annealed K-Maxwell passes 50 steps earlier.**
- Pairwise at equal steps: lhs = **0.00748 @3200** and **0.00729 @3210** (≥0.004,
  significant; their means 3.27881/3.27817 vs ours 3.275070/3.274527).
- C1 probes around the champion (seed-0, 15 runs): flat plateau 3.2775–3.2791.
  Best C1_s58e22_t64 = 3.27746 ≈ champion band; [3,72] ties [3,64]; end-age
  22–28 ties; start 54–58 ties, 62 slightly worse, 44–48 worse. No further
  lever found — 58→2x in [3,64] IS the optimum of this family.
- Note: 6 originally-authored probes (age_start 58/62 in [3,56]) were
  infeasible (mean age > τmax → solver correctly refuses, negative weights);
  re-run in [3,64]/[3,72] as `*_t64/_t72`. The pin/solve gate caught it.
- Grid: `logs/kmaxwell/ablation_anneal_n8/summary.tsv` (+ per-run stdout/.txt).

## Score table (all under the 0.004 statsig rule, n=8)

| fleet | config | first-pass S | reference | verdict |
|---|---|---|---|---|
| A | frozen KM k6 a35 on CWD | **2680** | #46 = 2690 | **beats merged record** |
| B | #339 bi-Maxwell (as published) | **2640** | their claim 2645 | reproduces; beats KM pairwise |
| C | anneal 58→26 k8 [3,64] | **3160** | #340 = 3210 | **beats #340 by 50 steps** |

## Infra notes (new fleet, delta vs Part 1 §9)

- Image `nvidia/cuda:12.8.1-devel-ubuntu24.04` lacks **python3-dev**: Triton JIT
  dies on Python.h at first compile. bootstrap_box.sh now installs it.
- **Parallel ssh/scp to DIFFERENT baseten jobs races on the shared signed cert**
  (~/.ssh/baseten/id_ed25519-cert.pub) → random "Permission denied/Connection
  closed". Serialize all fleet ssh.
- ssh sessions that launch `nohup X &` sometimes never close AND the harness
  may kill them — the remote launch usually still lands; verify by probe, and
  never re-issue the same launch without checking (double-driver on box A cost
  a 20-min redo of CWDf_s0/s4).
- kmwr.py v2 pin-gates (pins_wr.json), stores the full ema curve in verdict.json,
  and copies uuid + stdout logs to stable names; kmqueue's WR hang is moot.
