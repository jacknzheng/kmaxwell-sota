# Record: Track 3 Optimization — frozen K-Maxwell momentum on SOAP-CWD — 2680 steps (n=8)

## TL;DR

Not a new optimizer stack: this is record **#46** (PR
[#328](https://github.com/KellerJordan/modded-nanogpt/pull/328): SOAP-Muon +
Tail-EMA readout + RowFloor + post-pin Cautious Weight Decay) with **one
change** to Muon momentum. From step 1000 onward, the single-EMA first moment is
replaced by a **frozen** mix of **K=6 log-spaced EMA buffers** on τ ∈ [3, 56]
with mean age 35 (a **K-Maxwell** kernel):

```python
# per Muon 2-D param, step >= 1000:
for k, beta in enumerate(betas):              # 6 log-spaced τ in [3, 56]
    m[k].lerp_(g, 1 - beta)
m_eff = sum(w[k] * m[k] for k in range(6))    # rising mix, mean age 35
update = g.lerp_(m_eff, mu)                   # then SOAP / NS / RowFloor / CWD
```

SOAP still sees the **raw** gradient (the kernel does not mutate `g` before the
preconditioner). RowFloor, post-pin CWD, Tail-EMA readout, radius pin, and the
#46 schedule are unchanged. At the switch step all six buffers lazy-initialize
to the current single-EMA momentum, so that step's update is bit-identical to
#46's.

On **n = 8 seeds (0–7, 8×H100)** the eval model with the Tail-EMA readout
reaches 3.28 in **2680 steps**: mean `val_ema = 3.278472`, significance
`(3.28 − mean)·√8 = 0.00432 ≥ 0.004` (2675 fails at 0.00343). Net vs the
published A40 #46 record: **−10** (formal, 2690 → 2680).

This is a **thin** record in the sense of the pairwise rule (same situation as
result #16 vs #14): it attains statsig for <3.28 earlier than #46, but is **not
pairwise significant** vs #46 at equal step (2690: LHS `0.0010 < 0.004`).
Per-seed first val_ema crossings average **2661** vs #46's **2667**. A
same-hardware n=8 reproduction of the unmerged bi-Maxwell SOAP claim
(PR [#339](https://github.com/KellerJordan/modded-nanogpt/pull/339)) first-passes
at **2640** and **beats this kernel pairwise** — if #339 merges, K6_a35 is
behind it.

Annealing the mix (the lever that won on plain Muon,
[3160](../20260824_kmaxwell_3160/README.md)) **does not help** this stack:
Tail-EMA already averages the cooldown tail, and old→young annealing regresses.

![full val_ema descent vs #46](figure.png)

![target zone (zoom)](zoomed_figure.png)

## Changes (vs the #46 base)

One isolated block; omitting the kernel recovers #46. The mix is a linear ramp
on the five fast ticks with leftover on the slow tick, solved so `Σ wᵢ τᵢ = 35`:

```text
w = 0.036726, 0.073452, 0.110179, 0.146905, 0.183631, 0.449107
```

**Unchanged from #46 (kept, load-bearing):** SOAP on all hidden matrices
(`freq=1`, `β2=0.90`, `denom_power=0.50`), RowFloor `TARGET_UW=0.3825`, radial
dampening + rescale-to-radius, post-pin CWD `0.025`, Tail-EMA (`τ=150`, `λ=0.6`,
`[2400, 2900]`, token embedding excluded), attn trust gate + early-trust-floor,
EMA-Nesterov (`0.3 / 0.99 / 300 / rest−950`), PowerCool LR (`power=1.2`,
`t_end=2900`), `MU=0.95`, `MUON_LR=0.0375`.

## Configuration

| field | value (was, in #46) |
|---|---|
| K-Maxwell `k` / `[τ_min, τ_max]` / mean age | `6` / `[3, 56]` / `35` (new) |
| mix weights | `0.036726,0.073452,0.110179,0.146905,0.183631,0.449107` |
| enable at | step 1000 (lazy-init identity) |
| `TAILEMA_*` / `ROWFLOOR` / `CWD` | unchanged |
| `MUON_LR` / `MU` | `0.0375` / `0.95` (unchanged) |
| `FINAL_TRAIN_STEPS` / `FINAL_SCHEDULE_STEPS` / `FINAL_LR_POWER` | `2900` / `2900` / `1.2` (unchanged; runs stopped at 2720, identical to a full run on `[0, 2720]`) |

## Result

**8×H100, n = 8 non-cherry-picked seeds (0–7).** Reported with the **Tail-EMA
readout** (`val_ema`), the readout this configuration ships (#45/#46). Dense
eval every 5 steps on the tail; per-seed first val_ema crossing of 3.28:
`{2655, 2660, 2660, 2660, 2675, 2645, 2675, 2655}` (mean 2661).

| seed | step 2675 | step **2680** | step 2685 | step 2690 | step 2700 |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.27850 | 3.27818 | 3.27786 | 3.27753 | 3.27697 |
| 1 | 3.27892 | 3.27865 | 3.27828 | 3.27796 | 3.27740 |
| 2 | 3.27868 | 3.27837 | 3.27804 | 3.27771 | 3.27716 |
| 3 | 3.27866 | 3.27832 | 3.27798 | 3.27768 | 3.27709 |
| 4 | 3.27981 | 3.27948 | 3.27915 | 3.27884 | 3.27827 |
| 5 | 3.27780 | 3.27749 | 3.27715 | 3.27684 | 3.27626 |
| 6 | 3.27974 | 3.27941 | 3.27906 | 3.27877 | 3.27820 |
| 7 | 3.27819 | 3.27788 | 3.27755 | 3.27721 | 3.27665 |
| **mean** | **3.27879** | **3.27847** | **3.27813** | **3.27782** | **3.27725** |
| **(3.28 − mean)·√8** | **0.00343 (✗)** | **0.00432 (✓)** | **0.00528 (✓)** | **0.00617 (✓)** | **0.00778 (✓)** |

**First-passing step = 2680** (val_ema; 2675 fails at 0.00343, 2680 clears at 0.00432).

Pairwise vs published #46 at equal step 2690 (their mean `3.278329`):
`(3.278329 − 3.277818) / √(1/8+1/8) = 0.00102 < 0.004` — **not pairwise**.
Hardware is also not matched (#46 logs are 2×A40; these are 8×H100).

## Reproducing

```bash
STOP_STEP=2720 torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/results/20260824_kmaxwell_2680/train_gpt_cwd_kmaxwell.py \
    --seed 0
```

All hyperparameters are hardcoded as defaults; only `--seed` varies. Dataset /
batch / architecture / validation match `records/track_3_optimization/train_gpt_simple.py`;
one forward-backward per step; no third-party optimizer import; the stopping
rule (smallest 5-step boundary with `(3.28−mean)·√n ≥ 0.004`) is fixed in
advance. `STOP_STEP=2720` only drops the unused `[2721, 2900]` tail (prefix-identical
to a 2900-step run).

## Files

- `train_gpt_cwd_kmaxwell.py` — #46 submission script + the frozen K6_a35
  kernel (`kmaxwell_kernel.py` sibling).
- `H100_seed{0..7}.txt` — the n=8 H100 logs the numbers above are computed from,
  each embedding its full source. These embed the research trainer launched with
  `--k 6 --tau-min 3 --tau-max 56 --weights 0.036726,0.073452,0.110179,0.146905,0.183631,0.449107`
  (equivalently, `train_gpt_cwd_kmaxwell.py` hardcodes exactly these).
- `figure.png` — full descent vs the published #46 A40 n=8 fleet.
  `zoomed_figure.png` — target-zone zoom. Rebuild:
  `python3 plot_cwd_compare.py`.

## Credits

Idea and direction: [Jeffrey Cheng](https://github.com/jeffreyscheng)
([@jeffreyscheng](https://github.com/jeffreyscheng)). Help:
[Jerry Hong](https://github.com/jerryhong21)
([@jerryhong21](https://github.com/jerryhong21)).

#46 stack: PR [#328](https://github.com/KellerJordan/modded-nanogpt/pull/328) by
[@ypwang61](https://github.com/ypwang61) (SOAP-Muon #321, Tail-EMA #325,
RowFloor, post-pin CWD). Related momentum-mixture prior work: AggMo (Lucas et
al., 2018), QHM (Ma & Yarats, 2019), AdEMAMix (Pagliardini et al., 2024), and
the two-timescale bi-Maxwell kernel on this stack
(PR [#339](https://github.com/KellerJordan/modded-nanogpt/pull/339)).
