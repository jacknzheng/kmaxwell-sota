# Standalone zeta solver (blog six, fixed 16M protocol)

This is the production eigensolver, unchanged, with a one-command launcher. It measures the
leading response eigenvalue zeta of the frozen coupled Muon operator on one saved checkpoint
("capture") using a fixed 16,777,216-token validation probe.

## What you need

- 8 CUDA GPUs on one host (the solver requires exactly 8 ranks). Production used H200 NVL
  (microbatch 16) and H100 NVL 80 GB (microbatch 4; the launcher falls back 16 -> 8 -> 4 on OOM).
- Python 3.12, `torch==2.11.0+cu128`, `numpy==2.4.3`, `wandb==0.31.0` (offline mode; still imported).
  See `requirements.txt`; `requirements-freeze.txt` is the full production freeze.
- `fineweb_val_000000.bin` (200,001,024 bytes,
  sha256 `5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640`).
- One capture: `rank0..rank7/boundary_NNNNNN.pt` (starter total about 1.89 GB: rank0 1.44 GB, ranks1–7 about 64 MB each; 8 shards required),
  ideally with the `index.jsonl` that pins each shard's sha256 and byte count.
- Optional warm start: a finished output directory of a nearby checkpoint
  (its `current_vector_rank0..7.pt`, about 42 MB each).

## Run

```
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
pip install -r requirements.txt
./launch.sh --capture /data/captures --validation /data/fineweb_val_000000.bin \
            --output /runs/t002041 --update 2041 [--warm /runs/t002040]
```

`launch.sh` verifies the inputs on the CPU (`check_inputs.py`), launches `torchrun` with the
production arguments and environment, retries with a smaller microbatch only on CUDA OOM, and
writes `result.json` (`read_result.py`). A killed run is resumed by re-running the same command.
`DRY_RUN=1 ./launch.sh ...` prints the exact command without touching a GPU.

Protocol as in production: single warm Arnoldi search (seed 921 Gaussian when no warm vector),
stop when the largest positive Ritz value is stable to 0.1% over 9 consecutive dimensions
(`DENSE_PREFIX_WINDOW=9`), dimensions up to 128, weight decay included, no residual veto, no
routine independent start. `--independent` (fresh Gaussian start, seed 77193) exists only for
protocol audits of extreme outliers; the warm result stays the point.

## Outputs (in `--output`)

- `progress.json`: full record (operator hashes, precision, tuning, probes, Arnoldi history).
- `result.json`: zeta, dimension, qualified flag, residual (diagnostic), timings.
- `current_vector_rank{0..7}.pt`: last Ritz vector; this is what `--warm` of the next point reads.
- `leading_N16777216_o0_rank{0..7}.pt`: the same vector saved at completion.
- `arnoldi_N16777216_search01_seed921.json`, `convergence.csv`, `heartbeat.json`, `solve.log`, `wandb/`.

Runtime varies by checkpoint and warm start. The supplied PR323 t=1300 reference took 377 seconds on 8×H200. Compilation, transfers and a colder start can add time.

## Provenance and license

`source/` is byte-identical to the audited production archive (`source.tgz`, sha256
`a3e212ec312e0921a9875807ad5a820c1912564b72b6f8bb93fada3ca452a56f`), restricted to the solver's
import closure; `fullrun_solve.py` and the four helpers are the production wrappers verbatim.
`sha256sum -c source_sha256.txt` verifies every file. The code derives from Keller Jordan's
modded-nanogpt (MIT, see `LICENSE`); the solver and harness extensions are by the muoff project
and are shared under the same MIT terms.

## Actual-kernel prediction

From the package root, `python3 solver/predict.py --capture data/captures --update 1300`
reads the actual saved kernel, rates and decay, and prints bubbleforce and predicted ζ.
It uses the same frozen_prediction.py as production (NumPy/SciPy on CPU).
The formula includes the training-batch normalization; do not normalize curves to fit.
The helper requires uniform Muon LR/decay across this checkpoint's groups, as in
the assigned PR323 run. It refuses to reduce heterogeneous groups to a misleading scalar.
