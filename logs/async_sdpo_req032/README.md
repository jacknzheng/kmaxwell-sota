# REQ-032 — diligence + tau2 @500 steps (H100/8B Qwen3-8B, Path B) — 2 COMPLETE + 2 PARTIAL

Four async-SDPO arms, `Qwen/Qwen3-8B` (`path=B`; H200 was unavailable fleet-wide → Path B per the request's
fallback). Ran under a 4-box parallel fleet, then wound to ≤2 per the operator's node ceiling.
**Per the 2026-09-03 operator directive (Jack), the two long tau2 arms were checkpointed and STOPPED at ~step 215/220
to free capacity for the EoS queue** — not abandoned: resumable from their step_200 checkpoints (on-box).

| arm | box | status | steps | metric | shape |
|-----|-----|--------|------:|--------|-------|
| diligence answer_free | qrvdr53 | **COMPLETE** | 500 | judge_score | flat ~0.10 (0.142→0.094) |
| diligence answer_bearing | woxvogw | **COMPLETE** | 500 | judge_score | flat ~0.10 (0.117→0.084) |
| tau2 gold | wnle40q | PARTIAL (stopped) | 215 | pass^1 | flat ~0.29 (0.20–0.35), to 200/500 |
| tau2 step_hint | wox8gkw | PARTIAL (stopped) | 220 | pass^1 | flat ~0.28 (0.22–0.37), to 200/500 |

**Headline (all four arms):** at 500 steps (2.5× REQ-024's horizon), the 8B policy shows **no sustained gain** on any
arm — both diligence judge curves stay flat ~0.10, and both tau2 pass^1 curves (through 200/500) oscillate ~0.29 with
no upward trend. The longer horizon does **not** break REQ-024's flat-8B finding. The tau2 arms are partial (stopped
at operator direction); their step_200 checkpoints are on-box if a resume to 500 is wanted.

Per-arm curves: `diligence-answer_free.tsv`, `diligence-answer_bearing.tsv`, `tau2-gold_partial.tsv`,
`tau2-step_hint_partial.tsv`. tau2 gold crashed once at step 124 (transient NCCL/CUDA abort), auto-recovered from its
step_100 checkpoint, ran clean to the stop. No secrets/weights/tensors committed; on-box checkpoints not committed.
