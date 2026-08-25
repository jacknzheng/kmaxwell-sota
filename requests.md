# requests.md — request queue for Jerry's agent

This branch is watched by an autonomous agent (Jerry's Claude session). It
pulls every **10 minutes**. To ask for work:

1. Append a request block below (copy the template), commit, and push to
   **this branch** (`jerry-agent`).
2. The agent picks it up on the next poll, flips `status` to `RUNNING`,
   executes, then commits results back under your block (plus any artifacts —
   run logs land in `logs/kmaxwell/<fleet>/`, scored tables as `summary.tsv`).
3. `status` meanings: `OPEN` (yours, not yet seen) → `RUNNING` → `DONE` /
   `FAILED` / `NEEDS-INFO` (agent will say exactly what's missing).

What it can do tonight: launch modded-nanogpt runs on fresh 8×H100 Baseten
workstations (K-Maxwell frozen/annealed specs, the #46 CWD stack, #339
bi-Maxwell, n=8 seed fleets), score them under the Track 3 statsig protocol
(margin=(3.28−mean)×√n ≥ 0.004), make code changes on this branch, and answer
analysis questions from the existing ledgers. Keep one request per block;
include seeds/steps/configs if you care, otherwise it will pick sane defaults
(k6 [3,56] age35 for CWD; 58→26 k8 [3,64] for anneal).

Reference results already on this branch: CWD frozen KM passes @2680 (#46 =
2690); #339 reproduces @2640 and pairwise-beats KM; anneal 58→26 beats #340
@3160. See `km/WRITEUP.md`.

---

## Template

```
## REQ-<nnn>: <short title>
- status: OPEN
- requested: <your name / time, optional>

<what you want, freeform. configs, seeds, questions — anything.>
```

---

<!-- append requests below this line -->

## REQ-001: async-sdpo post-training fleet (NOT nanogpt)

- status: OPEN
- requested: Jack / 2026-08-25

This is **not** a modded-nanogpt / K-Maxwell / Track 3 job. Do not launch CWD, anneal, or seed fleets. Clone a different repo and run our off-policy SDPO post-training stack on fresh 8×H100 Baseten workstations, one box per ablation, in parallel.

### Repo

https://github.com/jacknzheng/async-sdpo.git (branch `main`)

Read `README.md` first. It is the source of truth for install, launch scripts, GPU split, secrets, sandbox, and the two failure modes below. Launch via `scripts/run_taubench.sh` and `scripts/run_diligencebench.sh`, not raw `python run.py`, unless you are iterating on a code fix.

### Secrets

These keys are intentionally included so you can run without `NEEDS-INFO`. Put them in `.env` at the async-sdpo repo root on every box. Do not commit `.env` back to `async-sdpo`.

```

OPENROUTER_API_KEY=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
WANDB_API_KEY=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
HF_TOKEN=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
PARALLEL_API_KEY=GZOauTxsbmXpP_gIL2f2hyZPL4DPl_PsrMXVbhAg

```

### What success looks like

1. Env actually works on Baseten (not RunPod). Smoke pass, then real runs.
2. All listed ablations launched; logs + scored tables committed back under this block.
3. Training signal is alive: `teacher_minus_student_logp` clearly nonzero, not ~0.
4. Tau2 banking sandbox does not kill or silently starve rollouts.
5. You **fix the code** when something is broken. Do not stop at a diagnosis. Commit fixes to a branch on `async-sdpo` (or a patch attached here) and rerun the affected arm.

### Box setup (every 8×H100)

Follow the README "Run it (8×H100)" section. Short version:

```bash
git clone https://github.com/jacknzheng/async-sdpo.git
cd async-sdpo
uv sync --extra knowledge
bash scripts/setup_tau2_sandbox.sh
which srt rg bwrap socat
# write the four keys above into .env at repo root; never commit it
```

Baseten is not RunPod. `run.py` only auto-points caches at `/workspace` if that dir exists — set equivalent persistent cache dirs on the Baseten disk (`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `HF_HOME`). Large `/dev/shm`. Pin **vLLM 0.26.x exactly**. Python 3.12.

GPU split is 4 vLLM rollout + 4 FSDP trainer. Default model is `Qwen/Qwen3.8-27B`. If that OOMs after a real shrink of `mini_batch_size`, fall back to `Qwen/Qwen3-8B` and say so in the writeup. Do not silently change the SDPO loss sign, clip window (must contain 1.0), or `group_size=1` / `keep_failures=True`.

### Launch order

**Phase 0 — one box, prove the stack (mandatory before the fleet).**

- `bash scripts/run_taubench.sh smoke`
- `bash scripts/run_diligencebench.sh smoke`
  If smoke fails, **fix it** (deps, sandbox, paths, vLLM pin, shm, cache dirs) and rerun. Do not scale out a broken image.

**Phase 1 — baselines (can share a box, sequential is fine).**

- `bash scripts/run_taubench.sh baseline`
- `bash scripts/run_diligencebench.sh baseline`
  These are the numbers to beat.

**Phase 2 — full ablation fleet, one 8×H100 per arm, in parallel.**

Tau2 (`scripts/run_taubench.sh`):

- `gold` — teacher sees Sierra gold docs / canonical tool trajectory. Main tau arm.
- `step_hint` — OpenRouter names the single next correct action.
- `gold_banking` — gold, `banking_knowledge` only (sandbox-stress arm).

Diligence (`scripts/run_diligencebench.sh`):

- `answer_free` — main arm; hint must not state figures/conclusions.
- `answer_bearing` — hint may cite missed rubric facts.
- `mixture` — 50/50 of the two teachers.

Sane defaults if you need them: `trainer.total_steps=200`, keep `judge.eval_interval=25`, `trainer.algorithm.max_staleness=3`. If a box still has time after 200 and the curve is moving, continue to 500. Log to each box's disk and copy run dirs back; wandb projects `sdpo-tau2` / `sdpo-diligence`.

### Autonomous debugging (this is the job, not a nice-to-have)

You own two known failure modes. Fix them in code, then rerun the arm that failed. Keep iterating until the fleet is green or you have a hard blocker that is not code (missing key, no GPUs). README section "What to watch (and what to fix)" has the same list.

**A. Training signal collapse ("reward variance" / gap ~0).**
SDPO has **no reward model in the gradient**. The whole update is the teacher−student logp gap (`teacher_minus_student_logp` in `train/trainer.py`). If `|gap| < 1e-3`, training is a no-op even if loss looks healthy.

Watch, in order: (1) gap clearly nonzero, (2) clip fractions a modest minority, (3) staleness ≤ 3, (4) held-out score above baseline. Also watch `store_hint_dropped_percent` — an unhinted teacher is identical to the student.

If gap is dead: inspect hint prompts in `data/prompts/`, hint drop rate, OpenRouter failures, tau2 `gold_suffix` actually injecting gold, diligence `answer_free` being too timid, SOD + loss_mask eating all tokens, sandbox errors producing empty transcripts. Plausible fixes: stronger hints, fewer silent hint drops, retries, logging that shows *why* a hint failed. `answer_bearing` is the stronger teacher; a win there is closer to distillation and must be labeled as such. Do **not** add GRPO-style whitening or a baseline term (the paper's point).

Tau2 episode `reward` is binary pass^1 and is **eval-only**. Low variance there is expected; do not confuse it with the SDPO gap. Still: if almost every banking episode scores 0 because the sandbox exploded, the _eval_ is garbage and gold hints may be meaningless — fix the sandbox first.

**B. Tau2 sandbox breakage.**
`banking_knowledge` `shell` runs inside Anthropic `sandbox-runtime` (`srt`) and needs `rg`, `bwrap`, `socat` on the host. Missing any of them raises `SandboxRuntimeError` at env construction (`scripts/setup_tau2_sandbox.sh`). Runtime breakage is also likely (bwrap permissions, no nested userns on the workstation image, srt crashing mid-episode).

Make this robust: install the tools in the image, fail loud at startup if they're missing (don't discover it 40 steps in), catch sandbox errors per-episode so one broken shell tool doesn't kill the training process, retry or skip that episode, log a `sandbox_fail` counter. Retail/airline do not need the sandbox — if banking is uniquely toxic, isolate it (`gold_banking` as the canary) rather than taking down `gold` on all three domains. Prefer fixing banking over dropping it.

**Other likely Baseten issues to just fix:** `/workspace` cache assumption, `/log` not writable (scripts already fall back to `./log`), vLLM version, torch.compile + FSDP first-step hangs (`compile_trainer=false` is the debug lever; `--smoke` already runs uncompiled), NCCL weight-sync deadlock (receive RPC must be in flight before `trainer_send_weights` — do not "simplify" that order).

### Do not

- Launch nanogpt / K-Maxwell / seed fleets.
- Flip the SDPO sign in `loss.py` (`A = log π_teacher − log π_student`). Tests pin this.
- Commit `.env` or huge weight checkpoints back to `async-sdpo` — commit logs, `summary.tsv`, config dumps, and code patches. Using the keys in this request to write `.env` on the box is expected.
- Wait for us on obvious infra/code fixes. If you need a product decision, `NEEDS-INFO` with one concrete question.

### Write back under this block

- Table: arm × box × steps × baseline vs final held-out metric (tau2 `pass1` overall + per domain; diligence `judge_score` + section scores).
- For each arm: mean `teacher_minus_student_logp`, hint-drop rate, sandbox-fail count, any code you changed and why.
- Pointers to logs (console + `train.log`) and wandb URLs.
- If you branched `async-sdpo`, the branch URL and a 5-line summary of the diff.
