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
