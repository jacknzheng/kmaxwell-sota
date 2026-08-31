# REQ-024 — 8-GPU 4+4 OpenRouter DeepSeek SDPO fleet (diligence + tau2 gold)

**scaling-sdpo @ `ecf6fd8`, node wgmy26w (8×H100: 4 vLLM rollout + 4 FSDP2 trainer), Qwen3-8B, torch cu128, vLLM 0.26.0.** Async SDPO with OpenRouter DeepSeek hints + Nemotron judge/user-sim. Nodes released.

## Delivered

### Diligence arms — both to step 200 ✅
- **answer_free** (`diligence-answer-free/`, run 043221): full artifacts, **256 eval rows** (8 boundaries × 30 held-out tasks + envelopes). judge_score ~0.10, roughly flat across 25→200 (0.099 → 0.101; sub-scores factual-accuracy ~0.19, analytical-reasoning ~0.02, judge_errors 0). rollouts/training/vllm/api_failures jsonl gzipped.
- **answer_bearing** (`diligence-answer-bearing/console.log`, run 071952): reached step 200; teacher-student gap trajectory healthy (−0.08…−0.15), **hint_drop 0.0%** throughout. Its in-run tau2 evals wrote **0 evaluations.jsonl** because they hit the tau2-bench `a2c0247` signature bug (fixed below) — the training trajectory is intact in the console; the tau2 eval numbers for this arm were never produced.
- **Hint gate PASS** (`hint-fix-validation.tsv`): 186 attempts, **0 drops (0.00%)**, 0 openrouter_length, api_failures empty — the DeepSeek OpenRouter hint-output-budget fix is validated (the original REQ-018 concern).

### tau2 harness fix — 3 coupled bugs, delivered as a patch ✅
`tau2-solo_mode-fix.patch` (against pinned tau2-bench `a2c0247`; **push blocked** — this box's token is read-only on `jacknzheng/scaling-sdpo`, so `git am` + push to `fix/hint-output-budget` is the client's step):
1. `evaluate_simulation()` missing required positional `solo_mode` → add `solo_mode=False`.
2. `evaluate_simulation()` forwards `solo_mode` to the domain env constructor, so the passed `env_kwargs` must not also carry it (`get_environment() got multiple values for 'solo_mode'`) → strip it at the eval call site (rollout `make_env` path unchanged).
3. The user-simulator call re-raised on any exception → a transient upstream 429 killed the whole run → add bounded retry-with-backoff (`_retry_transient`).
Plus an `inspect.signature` guard test. Also cloned the **missing tau2-bench data** at `a2c0247` (retail/airline `tasks.json`).

## tau2 gold — NOT delivered: blocked on OpenRouter provider capacity ⚠️
`tau2-gold-attempt/` (run 112840, retail+airline, 200 steps, max_model_len 32768, hints skipped). Cleared four blockers (the two signature bugs, the missing data, and the `:free`-tier 20-req/min limit → switched to paid `nemotron-3-super-120b`). With retry-hardening the run **survives** the 429s instead of crashing, but the **paid model's upstream shared pool is chronically overloaded** ("temporarily rate-limited upstream / engine_overloaded", DeepInfra + DigitalOcean): it reached **step 25/200 then deadlocked at the first synchronous eval boundary** — `evaluation_started=1` at 12:53Z with **0 held-out tasks completed 37+ min later**, 443× 429 absorbed (`DEADLOCK-EVIDENCE.txt`). At that throughput even the first eval won't complete, let alone 200 steps, so the node was stopped and released rather than burn it at ~zero output.

**To resume gold:** a dedicated Nemotron provider key (BYOK, per OpenRouter's own remedy hint) or a less-contended user-sim/judge model. Re-provision + relaunch with `bash scripts/run_taubench.sh gold trainer.total_steps=200 "data.domains=[retail,airline]" generator.engine.max_model_len=32768` after applying the patch; the config is preserved in `tau2-gold-attempt/gold-config.yaml`.

No `.env`, keys, weights, or checkpoint tensors are committed. Large jsonl are gzipped.
