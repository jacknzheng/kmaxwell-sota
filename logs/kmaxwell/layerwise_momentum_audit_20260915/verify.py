"""CPU audit of delivered scalar artifacts, plus the pinned Muon state-load issue."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import statistics
import subprocess

PIN = "365c392d695f95dc9a4fb89095e85a6a7b5d551e"


def read_curve(path):
    pairs = [line.split() for line in path.read_text().splitlines() if line.strip()]
    steps = [int(step) for step, _ in pairs]
    assert len(steps) == len(set(steps)), f"duplicate steps: {path}"
    return {int(step): float(loss) for step, loss in pairs}


def t3_two_sided(t):
    """Exact Student-t survival probability for three degrees of freedom."""
    if math.isinf(t):
        return 0.0
    theta = math.atan(abs(t) / math.sqrt(3))
    return max(0.0, 1 - 2 / math.pi * (theta + math.sin(theta) * math.cos(theta)))


def audit(repo, harness_repo):
    root = repo / "logs/kmaxwell"
    missing = []
    coverage = []
    for seed, fork in [(0, 1500), (0, 2000), (1, 2000), (2, 2000)]:
        arms = ["a1all", "a05all", "a2all", "nomom", "matched"]
        arms += [f"{a}_b{b}_{typ}" for a in ["a05", "a2"] for b in [0, 6, 11]
                 for typ in ["attnq", "attnk", "attnv", "attnproj", "mlpfc", "mlpproj"]]
        for arm in arms:
            path = root / "req058_layerwise_momentum_response/raw/full058" / f"{arm}_s{seed}_f{fork}.tsv"
            curve = read_curve(path)
            if fork + 64 not in curve:
                missing.append({"file": str(path.relative_to(repo)), "expected": fork + 64,
                                "last_logged": max(curve)})
        coverage.append({"seed": seed, "fork": fork, "arms": len(arms)})

    vals = {}
    for path in (root / "req059_sharpness_guided_momentum/raw/full059").glob("*.tsv"):
        arm, seed = path.stem.rsplit("_s", 1)
        curve = read_curve(path)
        assert 2750 in curve, f"missing endpoint: {path}"
        vals.setdefault(arm, {})[int(seed)] = curve[2750]
    assert len(vals) == 9 and all(set(v) == {3, 4, 5, 6} for v in vals.values())
    contrasts = {}
    for arm in vals:
        if arm == "guided":
            continue
        diff = [vals["guided"][s] - vals[arm][s] for s in [3, 4, 5, 6]]
        mean = statistics.mean(diff)
        se = statistics.stdev(diff) / 2
        t = mean / se if se else (math.inf if mean else 0.0)
        half = 3.182446305284263 * se
        contrasts[arm] = {"differences": diff, "mean": mean,
                          "ordinary_paired_ci95": [mean - half, mean + half],
                          "p_t_df3": t3_two_sided(t)}
    adjusted = 0.0
    for i, arm in enumerate(sorted(contrasts, key=lambda a: contrasts[a]["p_t_df3"])):
        adjusted = max(adjusted, min(1.0, (len(contrasts) - i) * contrasts[arm]["p_t_df3"]))
        contrasts[arm]["holm_p"] = adjusted
    assert math.isclose(contrasts["global_a05"]["mean"], 0.009135, abs_tol=1e-12)
    assert 0.18 < contrasts["euclidean"]["holm_p"] < 0.20
    assert 0.01 < contrasts["shuffled"]["holm_p"] < 0.011

    coupling = []
    for path in sorted((root / "req057_layerwise_spectral_sharpness/raw/full").glob("*.json")):
        data = json.loads(path.read_text())["subset0"]
        c = data["cross_layer_at_gradpolar"]
        assert math.isclose(c["c_diag"] + c["c_cross"], c["c"], rel_tol=1e-12)
        coupling.append({"file": path.name, "cross_fraction_at_gradpolar": c["c_cross"] / abs(c["c"]),
                         "sum_isolated_over_joint_witness": sum(x["S_i"] for x in data["isolated"].values()) / data["S_joint"]})

    reload_result = {"status": "not run; supply --harness-repo"}
    if harness_repo:
        import torch
        source = subprocess.check_output(
            ["git", "show", f"{PIN}:records/track_3_optimization/optimizers/muon.py"],
            cwd=harness_repo, text=True)
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in {"Muon", "order_params_like_record"}]
        assert len(nodes) == 2
        # Extract only the production class and its constructor's sorting helper.
        # No import-time GPU or compilation operations from the full module are run.
        code = "from __future__ import annotations\n" + "\n\n".join(ast.get_source_segment(source, n) for n in nodes)
        ns = {"torch": torch}
        exec(compile(code, "pinned_muon_extract", "exec"), ns)
        p = torch.nn.Parameter(torch.ones(2, 2))
        base = ns["Muon"]([p], mu=0.95)
        control = ns["Muon"]([p], mu=0.0)
        before = control.param_groups[0]["mu"]
        control.load_state_dict(base.state_dict())
        after = control.param_groups[0]["mu"]
        assert before == 0.0 and after == 0.95
        reload_result = {"status": "overwrite reproduced", "source_sha": PIN,
                         "torch": torch.__version__, "mu_before_load": before, "mu_after_load": after}

    return {"audit_source": "dabff19", "req058_coverage": coverage, "req058_missing_requested_endpoints": missing,
            "req059_mean_losses": {a: statistics.mean(v.values()) for a, v in vals.items()},
            "req059_corrected_contrasts": contrasts, "req059_primary_practical_win": False,
            "req057_coupling": coupling, "nomom_reload": reload_result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--harness-repo", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = audit(args.repo, args.harness_repo)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    print(json.dumps({"req058_missing_endpoints": len(result["req058_missing_requested_endpoints"]),
                      "req059_guided_minus_shorter": result["req059_corrected_contrasts"]["global_a05"]["mean"],
                      "nomom_reload": result["nomom_reload"]}, indent=2))
