# REQ-031 — tau2 gold (retail + airline), DeepSeek user-sim, no sandbox — **COMPLETE**

**scaling-sdpo @ `ac07c90c1590823427a0a01f66bef6f69f0c3cf4` (fix/hint-output-budget), node `qv16djq` (8×H200-class 8×H100 workstation), torch 2.11.0+cu128, vLLM 0.26.0.** Full 200-step tau2 `gold` run on the official retail+airline test set, scored at step 0 (baseline), every 25, and final@200. Supersedes REQ-024's gold and withdrawn REQ-030. **Node released.**

## Result — the full pass^1 curve (`summary.tsv`; W&B below)

| phase | step | pass^1 | retail | airline |
|:------|-----:|-------:|-------:|--------:|
| baseline | 0 | 0.2833 | 0.400 | 0.050 |
| interval | 25 | **0.4333** | 0.600 | 0.100 |
| interval | 50 | 0.3333 | 0.400 | 0.200 |
| interval | 75 | 0.2500 | 0.350 | 0.050 |
| interval | 100 | 0.3333 | 0.425 | 0.150 |
| interval | 125 | 0.3333 | 0.375 | 0.250 |
| interval | 150 | 0.3333 | 0.400 | 0.200 |
| interval | 175 | 0.2833 | 0.350 | 0.150 |
| **final** | **200** | **0.3000** | 0.375 | 0.150 |

Every eval is the full official held-out set (n=60 = retail 40 + airline 20), **0 rollout errors** at every boundary. Shape (the deliverable, no interpretation asked): the 8B policy **peaks at step 25 (0.433) then oscillates around the 0.283 baseline**, ending at 0.300 — no sustained gain over 200 steps (airline improves 0.05→0.15; retail flat/slightly down 0.40→0.375). This matches REQ-024's flat-8B finding and motivates REQ-032's larger-model test.

**W&B:** https://wandb.ai/jacknzheng-united-states-department-of-state/sdpo-tau2/runs/8xt6ktgo — `eval/pass1`, `eval/pass1_retail`, `eval/pass1_airline` on the default `_step` axis (baseline at step 0, final at step 200).

## Why this completed when REQ-024's gold could not

REQ-024's gold deadlocked on the Nemotron user-sim's OpenRouter shared-pool 429 storm (0 tasks completed). Here, three things fixed it, all confirmed:
- **DeepSeek user-sim** (`openrouter/deepseek/deepseek-v4-flash`, temperature 0, reasoning off) — funded, preflight HTTP 200 at 2026-09-02T05:55:55Z.
- **No sandbox** — banking/`bwrap`/`srt` removed at ac07c90; retail+airline use in-process DB tools only. No `--privileged`.
- **Retry-hardening** — ac07c90 already carries my REQ-024 tau2 patch (solo_mode, env_kwargs strip, `_retry_transient` backoff). The run absorbed **2231 rate-limits with api_failures = 0**.

## Preflight / bootstrap (recorded)

- Checkout `ac07c90` (clean); offline `pytest -m 'not network'` = **228 passed, 2 deselected** (matches the expected 228p/2s).
- `uv sync --extra tau2`; then **manual `uv pip install vllm==0.26.0`** (vLLM is not a declared dep). torch 2.11.0+cu128 has no matching cu128 torch**vision**/torch**audio**, whose fatal `_check_cuda_version()` aborts a text-only import → **patched both to no-op** (safe: no vision/audio CUDA ops used), exactly as the REQ-031 spec anticipates.
- tau2 data sparse-cloned at `a2c0247` (retail/airline `tasks.json`) → `TAU2_DATA_DIR`.
- 8 GPUs confirmed visible before launch; 4+4 map as specified.

## Files

- `summary.tsv` — SHA, CLI, GPU count, steps, wall time, per-phase overall+per-domain pass^1, 429 count, failure counters, W&B URL.
- `tau2-gold/` — `config.yaml`, `args.txt`, `ARTIFACTS.txt`, `console.log.gz`, `train.log.gz`, `rank{1,2,3}.log`, `evaluations.jsonl.gz` (source of truth), `rollouts.jsonl.gz`, `training.jsonl.gz`, `vllm.jsonl.gz`, `api_failures.jsonl.gz` (empty — 0 failures).

No secrets, env dumps, weights, or checkpoint tensors are committed. Large text is gzipped losslessly.
