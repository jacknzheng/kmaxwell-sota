"""Per-matrix token-convergence CSV and figure, including incomplete results."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input.glob("rank*/n*/pilot_rank*.json")):
        data = json.loads(path.read_text())
        result = data.get("convergence", {})
        rows.append(dict(matrix=data["matrix"], update=data["update"], tokens=data["tokens"],
                         offset=data["validation_offset"], mu_mean=result.get("mu_estimate"),
                         candidate_mean=result.get("leading_positive_candidate"),
                         solver_converged=result.get("empirical_convergence", False),
                         residual=result.get("leading_positive_residual"),
                         matvec_calls=result.get("matvec_calls"),
                         matvec_seconds=data.get("matvec_seconds"),
                         probe_sha256=data["batch_sha256"], stage=data["stage"]))
    if not rows:
        raise ValueError("no measured probe results")
    with (args.output / "mu_token_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({row["matrix"] for row in rows})
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        subset = sorted((row for row in rows if row["matrix"] == name), key=lambda row: row["tokens"])
        qualified = [row for row in subset if row["solver_converged"] and row["offset"] == 0]
        axis.plot([row["tokens"] for row in qualified], [row["mu_mean"] for row in qualified], "o-",
                  label="Nested validation prefix; solver qualified")
        replicas = [row for row in subset if row["solver_converged"] and row["offset"] > 0]
        axis.scatter([row["tokens"] for row in replicas], [row["mu_mean"] for row in replicas],
                     marker="s", label="Disjoint equal-size probe")
        failures = [row for row in subset if not row["solver_converged"] and row["candidate_mean"] is not None]
        axis.scatter([row["tokens"] for row in failures], [row["candidate_mean"] for row in failures],
                     marker="x", color="red", label="UNQUALIFIED Ritz candidate")
        axis.set_title(name); axis.set_xscale("log", base=2)
        axis.set_xlabel("Validation probe tokens"); axis.set_ylabel("μ: mean loss per token")
        axis.grid(alpha=.25); axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(names):]:
        axis.set_visible(False)
    fig.suptitle("Validation-token convergence: one panel per matrix, no layer aggregation\n"
                 "Solver-qualified points do NOT imply sampling convergence")
    fig.savefig(args.output / "mu_token_convergence.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
