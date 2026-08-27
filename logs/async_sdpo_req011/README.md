# REQ-011 async-SDPO 4+4 fleet — result & post-mortem

**Bottom line:** the 4+4 async-SDPO stack works end-to-end and produces a strong, unambiguous SDPO training signal. Two of three arms (diligence `answer_free`, `answer_bearing`) trained to steps 114 and 176 with large, consistently-negative teacher−student gaps (mean ≈ −0.10 nats, **0% dead steps**). No arm reached the requested `total_steps=200`: every arm eventually crashed on an **uncaught API transient** (the glm-5.3-flash / OpenRouter auxiliary-LLM path is very flaky), and there is **no trainer checkpointing**, so a crash forfeits all progress.

## What was delivered

- **The stack is proven.** Init → rollout → compile → backward → NCCL weight-sync → optimizer step all work at 4 vLLM + 4 FSDP2 on 8×H100, Qwen3-8B, mini_batch=2. (The REQ-009 blockers — deadlock, TP-rendezvous, embedding DTensor, grad-clip — are all closed.)
- **Real SDPO signal.** `teacher_minus_student_logp` (hinted teacher vs bare student) is large and stable:
  - `answer_free`: 114 steps, mean gap −0.106 (−0.37…−0.043), staleness 0.95/max 3
  - `answer_bearing`: 176 steps, mean gap −0.101 (−0.27…−0.039), staleness 1.08/max 3
  - `tau2 gold`: reached step 64, gaps −0.04…−0.06 before being stopped (log lost when box released)
  - `frac(|gap|<1e-3) = 0` on every arm → not a no-op; the hint genuinely moves the teacher.
- Per-step trajectories: `dilfree_extract.txt`, `dilbear_extract.txt`. Full metrics: `summary.tsv`.

## Why no arm hit 200 (post-mortem)

1. **The auxiliary LLM is flaky.** `stealth/ox-alpha` was retired mid-run (404) and replaced by its permanent successor `z-ai/glm-5.3-flash` (same model, per OpenRouter). GLM-5.3-flash via OpenRouter/Z.AI throws transient errors at high volume — **3598 (free) / 4188 (bearing) hint drops**, plus intermittent `402 Insufficient credits` dips when the balance momentarily hit zero.
2. **Only the hint path is hardened.** `generate_hint` catches transients and drops the rollout (so `episodes_empty=0` and training continues). But the **eval path** (`evaluate_pass1` / in-train eval) and **tau2's multi-turn user-simulator** have uncaught call sites. Over a long run (thousands of API calls) an uncaught transient is eventually hit and it propagates to top level → `wandb.finish()` → torchrun tears down all ranks → whole-arm crash.
   - tau2 was the most exposed (multi-turn user-sim) and crashed 4× at step 1–64 (402, then a 16384-context overflow — fixed with `max_model_len=32768` — then more 402s). A hardening patch was written+applied to `run_tau2_episode` (drop-and-count like the hint path), but tau2's 4th crash was in the *pass@1 eval* path, which that patch doesn't cover.
   - the diligence arms are single-turn and far more robust, but still crashed at 114/176 on an uncaught async transient in the eval path.
3. **No checkpointing.** `runs/` holds no trainer checkpoints, so none of this is resumable — each crash means restart-from-0.

## What a clean 200 needs

- **Stable OpenRouter balance** (a comfortable buffer, not intermittent top-ups) — removes the 402 class entirely.
- **Harden every auxiliary-LLM call site** (eval `evaluate_pass1`, tau2 user-sim), not just hints — wrap each in the same drop/skip-and-continue guard so no single transient can crash a rank. (Partial patch for `run_tau2_episode` already applied on the box.)
- Optionally, **trainer checkpointing** every N steps so a crash costs minutes, not the whole run.
- Optionally, a **more reliable provider** for the auxiliary LLM (the flakiness is provider-side; a hosted/dedicated GLM-5.3-flash endpoint would cut the drop rate dramatically).

With stable credits + call-site hardening, the arms will reach 200 (they were healthy and advancing between transients). The scientific conclusion — **async-SDPO trains with a strong, non-degenerate teacher−student gap** — is already established by the partial runs.
