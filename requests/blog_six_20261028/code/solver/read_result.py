"""CPU-only readout of a finished solve directory, in the same shape the production queue records."""
import json, sys
from pathlib import Path


def read(path):
    progress = json.loads((path / "progress.json").read_text())
    if progress.get("stage") != "fixed_probes_complete":
        raise SystemExit(f"solve not complete: stage={progress.get('stage')!r}")
    probe = progress["probes"][-1]
    rows = [s["history"][-1] for s in probe["searches"]]
    winner = max(rows, key=lambda r: r["mu"] or float("-inf"))
    return dict(
        update=progress["update"], zeta=probe["mu"], tokens=probe["tokens"], validation_offset=probe["offset"],
        dimension=winner["dimension"], residual=winner.get("residual"),
        dimension_qualified=probe["qualified"], stopping_rule=probe.get("stopping_rule"),
        queue_initialization=probe.get("queue_initialization"),
        independent_start_checked=probe["independent_start_checked"],
        start_relative_spread=probe["seed_relative_spread"],
        solver_status="dimension_agreement" if probe["qualified"] else "unresolved_dimension",
        seconds=probe["seconds"], operator_calls=probe["operator_calls"], operator_seconds=probe["operator_seconds"],
        precision=progress["precision"], include_weight_decay=progress["include_weight_decay"],
        weight_decay_shift=probe.get("weight_decay_shift"), mu_polar_component=probe.get("mu_polar_component"),
        training_batch_tokens=progress["training_batch_tokens"],
        analysis_gpu=progress.get("gpu"), analysis_microbatch=progress.get("microbatch"), torch=progress.get("torch"),
        warm_start=progress.get("warm_start"), study_source_sha256=progress.get("study_source_sha256"),
        operator_hashes=progress.get("operator_hashes"), output=str(path),
    )


if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    row = read(out)
    (out / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ("update", "zeta", "dimension", "dimension_qualified", "residual", "seconds", "analysis_gpu", "analysis_microbatch")}))
    print(f"wrote {out / 'result.json'}")
