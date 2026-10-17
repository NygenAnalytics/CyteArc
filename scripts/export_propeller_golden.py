#!/usr/bin/env python
"""Regenerate tests/speckle_1_10_0_propeller_golden.json.

Builds deterministic categories x samples count tables, hands them to
scripts/export_propeller_golden.R, and records speckle::propeller's table and
the limma prior it fitted. Normal CI does not need R. Provide an R
installation carrying speckle 1.10.0 and limma 3.66.0, for example:

    uv run python scripts/export_propeller_golden.py --rscript Rscript
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO / "tests/speckle_1_10_0_propeller_golden.json"
R_SCRIPT = Path(__file__).resolve().parent / "export_propeller_golden.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--seed", type=int, default=20261015)
    return parser.parse_args()


def input_fingerprint(inputs: object) -> str:
    encoded = json.dumps(inputs, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def simulate_counts(
    rng: np.random.Generator,
    *,
    n_categories: int,
    group_sizes: list[int],
    shifted: dict[int, list[float]],
    depth: tuple[int, int] = (300, 900),
    zero_category: int | None = None,
) -> list[list[int]]:
    """Draw Dirichlet-multinomial counts with group-specific category shifts."""
    base = rng.dirichlet(np.full(n_categories, 4.0))
    columns = []
    for group, size in enumerate(group_sizes):
        expected = base.copy()
        for category, factors in shifted.items():
            expected[category] *= factors[group]
        expected /= expected.sum()
        for _ in range(size):
            proportions = rng.dirichlet(expected * 60.0)
            total = int(rng.integers(depth[0], depth[1]))
            column = rng.multinomial(total, proportions)
            columns.append(column)
    counts = np.column_stack(columns)
    if zero_category is not None:
        counts[zero_category, ::2] = 0
    return counts.astype(int).tolist()


def build_cases(seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    specs: list[dict[str, object]] = [
        {
            "name": "two_groups_logit_robust",
            "n_categories": 7,
            "group_sizes": [4, 4],
            "shifted": {1: [1.0, 2.5], 4: [1.0, 0.5]},
            "transform": "logit",
            "robust": True,
            "trend": False,
        },
        {
            "name": "two_groups_logit",
            "n_categories": 7,
            "group_sizes": [4, 4],
            "shifted": {2: [1.0, 2.0]},
            "transform": "logit",
            "robust": False,
            "trend": False,
        },
        {
            "name": "two_groups_asin_robust",
            "n_categories": 6,
            "group_sizes": [3, 5],
            "shifted": {0: [1.0, 1.8]},
            "transform": "asin",
            "robust": True,
            "trend": False,
        },
        {
            "name": "three_groups_anova_robust",
            "n_categories": 8,
            "group_sizes": [3, 3, 3],
            "shifted": {3: [1.0, 2.0, 0.6]},
            "transform": "logit",
            "robust": True,
            "trend": False,
        },
        {
            "name": "three_groups_anova",
            "n_categories": 5,
            "group_sizes": [3, 4, 3],
            "shifted": {1: [1.0, 1.0, 2.5]},
            "transform": "asin",
            "robust": False,
            "trend": False,
        },
        {
            "name": "two_categories",
            "n_categories": 2,
            "group_sizes": [3, 3],
            "shifted": {0: [1.0, 1.5]},
            "transform": "logit",
            "robust": True,
            "trend": False,
        },
        {
            "name": "zero_counts_robust_trend",
            "n_categories": 12,
            "group_sizes": [5, 5],
            "shifted": {5: [1.0, 3.0], 7: [1.0, 0.4]},
            "transform": "logit",
            "robust": True,
            "trend": True,
            "zero_category": 9,
        },
        {
            "name": "trend",
            "n_categories": 12,
            "group_sizes": [4, 4],
            "shifted": {2: [1.0, 2.0]},
            "transform": "logit",
            "robust": False,
            "trend": True,
        },
    ]
    group_names = ["ctrl", "covid", "flu"]
    cases = []
    for spec in specs:
        group_sizes = list(spec["group_sizes"])  # type: ignore[call-overload]
        counts = simulate_counts(
            rng,
            n_categories=int(spec["n_categories"]),  # type: ignore[call-overload]
            group_sizes=group_sizes,
            shifted=dict(spec["shifted"]),  # type: ignore[call-overload]
            zero_category=spec.get("zero_category"),  # type: ignore[arg-type]
        )
        n_categories = len(counts)
        groups = [
            group_names[group]
            for group, size in enumerate(group_sizes)
            for _ in range(size)
        ]
        cases.append(
            {
                "name": spec["name"],
                "categories": [f"c{idx:02d}" for idx in range(n_categories)],
                "samples": [f"s{idx:02d}" for idx in range(len(groups))],
                "groups": groups,
                "counts": counts,
                "transform": spec["transform"],
                "robust": spec["robust"],
                "trend": spec["trend"],
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    cases = build_cases(args.seed)
    with tempfile.TemporaryDirectory(prefix="cytearc-propeller-golden-") as raw:
        work = Path(raw)
        (work / "cases.json").write_text(json.dumps(cases))
        subprocess.run([str(args.rscript), str(R_SCRIPT), str(work)], check=True)
        output = json.loads((work / "results.json").read_text())
    versions = output["versions"]
    fixture = {
        "provenance": {
            "package": "speckle",
            "packageVersion": versions["speckle"],
            "limmaVersion": versions["limma"],
            "rVersion": versions["r"],
            "sourceRepository": "https://bioconductor.org/packages/speckle",
            "referenceFunction": "speckle::propeller",
            "generator": "scripts/export_propeller_golden.py",
            "seed": int(args.seed),
        },
        "inputs": {"cases": cases, "sha256": input_fingerprint(cases)},
        "speckle": output["results"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
