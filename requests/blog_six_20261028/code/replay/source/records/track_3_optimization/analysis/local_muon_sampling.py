"""Token-count convergence on fixed trained checkpoints, independently of Arnoldi.

The nested probe is a deterministic validation prefix, NOT training-token input.
Compare operators defined by mean loss per token: averaging eigenvalues of small
batch operators is not equivalent. Independent corpus intervals quantify observed
sampling variation; these are empirical diagnostics, not a population guarantee.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .response_decay import add_decay_flag


MATRICES = ["blocks.0.attn.q.weight", "blocks.6.attn.k.weight",
            "blocks.10.mlp.fc.weight", "blocks.11.mlp.fc.weight",
            "blocks.4.mlp.fc.weight", "blocks.9.mlp.proj.weight",
            "blocks.11.attn.v.weight", "blocks.11.mlp.proj.weight"]


def sampling_verdict(rows, tolerance=.05, minimum_tokens=524288):
    """Predeclared 5% eigenvalue tolerance, with tighter per-probe solver gates.

    Require two successive size comparisons and two disjoint replicas at the
    final size. Never turn a missing/null or solver-failed result into convergence.
    """
    nested = sorted((r for r in rows if r["validation_offset"] == 0), key=lambda r: r["tokens"])
    eligible = len(nested) >= 3 and nested[-1]["tokens"] >= minimum_tokens
    replicas = ([r for r in rows if r["tokens"] == nested[-1]["tokens"] and
                 r["validation_offset"] > 0] if nested else [])
    checked = nested[-3:] + replicas
    valid = eligible and len(replicas) >= 2 and all(
        r.get("convergence", {}).get("empirical_convergence") and
        r["convergence"].get("mu_estimate") is not None for r in checked)
    changes, spread = [], None
    if valid:
        values = [r["convergence"]["mu_estimate"] for r in nested[-3:]]
        changes = [abs(b-a) / max(abs(a), abs(b)) for a, b in zip(values, values[1:])]
        final = [r["convergence"]["mu_estimate"] for r in [nested[-1]] + replicas]
        spread = (max(final) - min(final)) / max(abs(v) for v in final)
    passed = valid and max(changes) <= tolerance and spread <= tolerance
    return dict(schema="local_muon_sampling_convergence_v1", sampling_converged=passed,
                status="empirical_sampling_convergence" if passed else "sampling_not_established",
                tolerance=tolerance, final_tokens=nested[-1]["tokens"] if nested else None,
                successive_relative_changes=changes, disjoint_probe_relative_spread=spread,
                probe_count=len(rows), population_accuracy_certified=False)


def main():
    parser = argparse.ArgumentParser()
    add_decay_flag(parser)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--update", type=int, default=1250)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768, 131072, 524288, 1048576])
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Use highest FP32 after separately recorded precision qualification")
    parser.add_argument("--resume-from", type=Path,
                        help="Reuse only complete, matching-probe results from a preserved study")
    parser.add_argument("--benchmark-warm", action="store_true",
                        help="Compare warm/cold solves at adjacent actual checkpoints on 8192 tokens")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import wandb
    run = wandb.init(project="local-muon-convergence", mode="offline",
                     dir=str(args.output), name=f"token-convergence-t{args.update}-rank{args.rank}")
    matrix = MATRICES[args.rank]
    rows = []
    def execute(tokens, offset, precision, compare=False, convergence=True):
        output = args.output / f"n{tokens}_o{offset}_{precision}"
        if args.resume_from is not None:
            prior_path = args.resume_from / output.name / f"pilot_rank{args.rank}.json"
            if prior_path.exists():
                prior = json.loads(prior_path.read_text())
                if (prior.get("stage") == "complete" and prior.get("matrix") == matrix and
                    prior.get("update") == args.update and prior.get("tokens") == tokens and
                    prior.get("validation_offset") == offset and prior.get("loss") == "fixed_validation" and
                    prior.get("loss_normalization") == "mean" and prior.get("matmul_precision") == precision and
                    prior.get("derivative_precision") == "FP32 Hessian; BF16 NS primals, linear FP32 tangents" and
                    prior.get('include_weight_decay', False) == args.include_weight_decay and
                    prior.get("convergence", {}).get("empirical_convergence") and
                    max(prior.get("additivity_relative_error", 1.), prior.get("homogeneity_relative_error", 1.)) < .0001):
                    output.mkdir(parents=True, exist_ok=True)
                    (output / prior_path.name).write_bytes(prior_path.read_bytes())
                    return prior
        command = [sys.executable, "-m", "records.track_3_optimization.analysis.local_muon_pilot",
                   "--capture", str(args.capture), "--output", str(output),
                   "--rank", str(args.rank), "--update", str(args.update),
                   "--matrix", matrix, "--fixed-validation", "--tokens", str(tokens),
                   "--offset", str(offset), "--microbatch", str(args.microbatch),
                   "--normalization", "mean", "--matmul-precision", precision]
        command.append('--include-weight-decay' if args.include_weight_decay else '--no-include-weight-decay')
        if convergence:
            command.append("--convergence")
        if compare:
            command.append("--compare-precision")
        with (args.output / "workers.log").open("a") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row = json.loads((output / f"pilot_rank{args.rank}.json").read_text())
        if completed.returncode:
            raise RuntimeError(f"probe failed: {row}")
        result = row.get("convergence", {})
        run.log(dict(tokens=tokens, offset=offset, objective=result.get("mu_estimate"),
                     residual=result.get("leading_positive_residual"),
                     matvec_seconds=row["matvec_seconds"],
                     additivity=row["additivity_relative_error"],
                     precision_error=row.get("precision_response_relative_error")))
        return row
    try:
        if args.benchmark_warm:
            benchmark = []
            initial = args.output / "warm_benchmark/previous"
            for label, update, warm_from in (
                    ("previous", args.update, None),
                    ("warm", args.update + 1, initial),
                    ("cold", args.update + 1, None)):
                output = args.output / "warm_benchmark" / label
                command = [sys.executable, "-m", "records.track_3_optimization.analysis.local_muon_pilot",
                           "--capture", str(args.capture), "--output", str(output), "--rank", str(args.rank),
                           "--update", str(update), "--matrix", matrix, "--fixed-validation", "--tokens", "8192",
                           "--microbatch", str(args.microbatch), "--normalization", "mean", "--convergence"]
                command.append('--include-weight-decay' if args.include_weight_decay else '--no-include-weight-decay')
                if warm_from is not None:
                    previous = json.loads((initial / f"pilot_rank{args.rank}.json").read_text())
                    if not previous.get("convergence", {}).get("empirical_convergence"):
                        break
                    command += ["--warm-start-from", str(warm_from)]
                with (args.output / "workers.log").open("a") as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                benchmark.append(dict(label=label, **json.loads((output / f"pilot_rank{args.rank}.json").read_text())))
            (args.output / "warm_benchmark.json").write_text(json.dumps(benchmark, indent=2, allow_nan=False))
        # TF32/high is not used just because it is faster: first compare full
        # response, linearity and the dominant eigenvalue against highest FP32.
        precision = "highest"
        if not args.skip_calibration:
            high = execute(8192, 0, "high", compare=True)
            highest = execute(8192, 0, "highest")
            hs, ss = high["convergence"], highest["convergence"]
            acceptable = (high["precision_response_relative_error"] < .001 and
                          high["additivity_relative_error"] < .0001 and
                          hs["empirical_convergence"] and ss["empirical_convergence"] and
                          abs(hs["mu_estimate"]-ss["mu_estimate"]) / abs(ss["mu_estimate"]) < .001)
            precision = "high" if acceptable else "highest"
            (args.output / "precision.json").write_text(json.dumps(dict(
                chosen=precision, high=high, highest=highest), indent=2, allow_nan=False))
        if args.calibration_only:
            return
        for size in args.sizes:
            rows.append(execute(size, 0, precision))
            (args.output / "sampling.json").write_text(json.dumps(dict(
                matrix=matrix, update=args.update, probes=rows,
                verdict=sampling_verdict(rows)), indent=2, allow_nan=False))
        size = args.sizes[-1]
        for offset in (size + 1, 2 * (size + 1)):
            rows.append(execute(size, offset, precision))
            (args.output / "sampling.json").write_text(json.dumps(dict(
                matrix=matrix, update=args.update, probes=rows,
                verdict=sampling_verdict(rows)), indent=2, allow_nan=False))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
