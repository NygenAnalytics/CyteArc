#!/usr/bin/env python
"""Regenerate tests/milor_2_6_0_golden.json.

Simulates reduced dimensions with a population that expands in one
condition, computes an exact Euclidean kNN graph, draws index vertices with
``cytearc.abundance.milo.sample_index_cells``, and hands them to
scripts/export_milo_golden.R, so a datastore run with the recorded seed
samples the same vertices. The fixture records miloR's refined index cells,
neighbourhoods, counts, edgeR intermediates, test table, spatial FDR, and
annotations. Normal CI does not need R. Provide an R installation carrying
miloR 2.6.0, for example:

    uv run python scripts/export_milo_golden.py --rscript Rscript

``--wilk`` instead mounts the Wilk et al. 2020 COVID-19 PBMC dataset from
its CyteBase URI, builds the graph with :func:`wilk_graph`, runs
``DataStore.abundance`` on it, and asks R to test CyteArc's kNN graph and
refined index cells with miloR, and the same cell-type counts with speckle.
The fixture is derived from a public dataset but is not published: it goes to
the git-ignored ``tests/datasets`` directory, which
``tests/test_abundance_live_wilk.py`` reads:

    uv run python scripts/export_milo_golden.py --wilk
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cytearc.abundance.milo import sample_index_cells

if TYPE_CHECKING:
    from cytearc import DataStore
    from cytearc.storage.refs import ArtifactRef

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO / "tests/milor_2_6_0_golden.json"
WILK_FIXTURE = REPO / "tests/datasets/milor_2_6_0_wilk_golden.json.gz"
WILK_WORK = REPO / "tests/datasets/wilk_milo"
R_SCRIPT = Path(__file__).resolve().parent / "export_milo_golden.R"

# Replication-dependent histones, exact ribosomal proteins, MHC, and sex-linked
# genes, from skills/cytearc-single-cell/references/gene-blacklists.md.
HUMAN_BLACKLIST = "|".join(
    [
        r"^MT-",
        r"^RP[SL]\d+[AXY]?\d*$|^RPLP[0-2]$|^RPSA$|^MRP[SL]\d+[A-C]?$",
        r"^HLA-",
        r"^HIST[1-4]H|^H1-[1-6]$|^H1F[1-6]$|^H2AC\d+$|^H2BC\d+$|^H3C\d+$|^H4C\d+$"
        r"|^H2AW$|^H3-4$|^H4-16$",
        r"^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$",
    ]
)

# One recipe serves the fixture and the live test. Donors are the replicates:
# the dataset has no sample column, and donor C1 contributed two samples
# (ventilated and not), which are pooled. The ANN index is built with high
# effort so that its graph is close to miloR's exact kNN. ``search`` is the
# CyteBase catalog query that found WILK_CYTEBASE_ID in the release.
WILK: dict[str, Any] = {
    "release": "2610",
    "search": "wilk sars cov 2",
    "hvgTopN": 2000,
    "pcaDims": 30,
    "annEfc": 400,
    "annEf": 400,
    "annM": 64,
    "k": 30,
    "prop": 0.05,
    "samplingSeed": 20261015,
    "sampleBy": "donor_id",
    "conditionBy": "disease",
    "conditionLevels": ["normal", "COVID-19"],
    "labelsBy": "cell.type.coarse",
}
WILK_CYTEBASE_ID = (
    "wilk_2020_single_cell_atlas_peripheral_sars_cov_2_infection_456e8b9b"
)
WILK_URI = (
    f"hf://buckets/Nygen/cytebase-{WILK['release']}/datasets/"
    f"{WILK_CYTEBASE_ID}/data.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--wilk",
        action="store_true",
        help="build the unpublished Wilk COVID-19 fixture from CyteBase instead",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=WILK_WORK,
        help="local directory that holds the Wilk mount (with --wilk)",
    )
    return parser.parse_args()


def input_fingerprint(inputs: object) -> str:
    encoded = json.dumps(inputs, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def exact_knn(points: np.ndarray, k: int) -> np.ndarray:
    """Return each point's k nearest other points by Euclidean distance."""
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(squared, np.inf)
    order = np.argsort(squared, axis=1, kind="stable")
    return order[:, :k]


def simulate(
    rng: np.random.Generator,
    *,
    samples: list[str],
    conditions: list[str],
    batches: list[str],
    cells_per_sample: int,
    expanded: str,
    dims: int,
) -> dict[str, object]:
    centres = rng.normal(scale=4.0, size=(4, dims))
    labels = np.array(["b", "mono", "nk", "t"])
    points = []
    cell_samples = []
    cell_labels = []
    for sample, condition in zip(samples, conditions, strict=True):
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        if condition == expanded:
            weights = np.array([0.15, 0.3, 0.45, 0.1])
        weights = weights / weights.sum()
        population = rng.choice(4, size=cells_per_sample, p=weights)
        offsets = rng.normal(scale=1.0, size=(cells_per_sample, dims))
        points.append(centres[population] + offsets)
        cell_samples.extend([sample] * cells_per_sample)
        cell_labels.extend(labels[population].tolist())
    coordinates = np.round(np.vstack(points), 6)
    order = rng.permutation(coordinates.shape[0])
    return {
        "coordinates": coordinates[order],
        "sample": [cell_samples[i] for i in order],
        "label": [cell_labels[i] for i in order],
        "designDf": {
            "sample": samples,
            "condition": conditions,
            "batch": batches,
        },
    }


def build_cases(seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    two = [f"s{idx}" for idx in range(1, 9)]
    two_conditions = ["ctrl"] * 4 + ["treated"] * 4
    batches = ["b1", "b2"] * 4
    three = [f"s{idx}" for idx in range(1, 10)]
    three_conditions = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    specs = [
        (
            "condition_refined",
            two,
            two_conditions,
            batches,
            "~condition",
            None,
            True,
            15,
            0.1,
            "TMM",
            0.0,
            "treated",
        ),
        (
            "batch_and_condition",
            two,
            two_conditions,
            batches,
            "~batch + condition",
            None,
            True,
            15,
            0.12,
            "TMM",
            0.0,
            "treated",
        ),
        (
            "three_levels_contrast",
            three,
            three_conditions,
            ["b1", "b2", "b1"] * 3,
            "~0 + condition",
            ["conditionb - conditionc"],
            True,
            12,
            0.1,
            "TMM",
            0.0,
            "b",
        ),
        (
            "unrefined_logms",
            two,
            two_conditions,
            batches,
            "~condition",
            None,
            False,
            15,
            0.08,
            "logMS",
            0.0,
            "treated",
        ),
    ]
    cases = []
    for (
        name,
        samples,
        conditions,
        batch,
        formula,
        contrasts,
        refined,
        k,
        prop,
        norm,
        min_mean,
        expanded,
    ) in specs:
        data = simulate(
            rng,
            samples=samples,
            conditions=conditions,
            batches=batch,
            cells_per_sample=70,
            expanded=expanded,
            dims=8,
        )
        coordinates = np.asarray(data["coordinates"])
        knn = exact_knn(coordinates, k)
        n_cells = coordinates.shape[0]
        sampling_seed = int(rng.integers(0, 2**31 - 1))
        random_vertices = sample_index_cells(n_cells, prop, seed=sampling_seed)
        cases.append(
            {
                "name": name,
                "coordinates": coordinates.tolist(),
                "knn": (knn + 1).tolist(),
                "randomVertices": (random_vertices + 1).tolist(),
                "refined": refined,
                "prop": prop,
                "samplingSeed": sampling_seed,
                "sample": data["sample"],
                "label": data["label"],
                "designDf": data["designDf"],
                "formula": formula,
                "modelContrasts": contrasts,
                "normMethod": norm,
                "minMean": min_mean,
            }
        )
    return cases


def wilk_datastore(directory: Path) -> tuple["DataStore", str]:
    """Mount the Wilk dataset of the pinned CyteBase release under ``directory``.

    The dataset is opened by its URI, read anonymously, so the ``cytebase``
    package is not needed. Returns the mounted store and its CyteBase ID.
    Reopening the same directory reuses every artifact computed before.
    """
    import zarr

    from cytearc import DataStore, mount_datastore

    target = directory / "wilk.zarr"
    options: dict[str, Any] = {
        "storage_options": {"token": False},
        "min_features_per_cell": -1,
        "default_assay": "RNA",
    }
    if not target.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return mount_datastore(WILK_URI, at=str(target), **options), WILK_CYTEBASE_ID
    root = zarr.open_group(str(target), mode="r")
    source = root.attrs.get("matrixSource")
    if not isinstance(source, dict) or source.get("location") != WILK_URI:
        raise ValueError(
            f"{target} is not a mount of {WILK_URI}; choose a new directory"
        )
    return DataStore(str(target), zarr_mode="r+", **options), WILK_CYTEBASE_ID


def wilk_graph(ds: "DataStore") -> dict[str, "ArtifactRef"]:
    """Build the Euclidean kNN graph of every cell of ``I`` with the recipe."""
    active = np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    if not active.all():
        raise RuntimeError("The Wilk recipe expects every cell to be active in I")
    cells = ds.snapshot_cell_selection("I")
    hvg = ds.features.hvgs(
        cells, top_n=WILK["hvgTopN"], blacklist=HUMAN_BLACKLIST, show_plot=False
    )
    normalized = ds.features.normalize(cells, hvg)
    pca = ds.reduction.pca(normalized, dims=WILK["pcaDims"])
    ann = ds.graph.ann_index(
        pca,
        ann_metric="l2",
        ann_efc=WILK["annEfc"],
        ann_ef=WILK["annEf"],
        ann_m=WILK["annM"],
    )
    neighbors = ds.graph.neighbors(ann, k=WILK["k"])
    return {"pca": pca, "ann_index": ann, "neighbors": neighbors}


def wilk_cell_labels(ds: "DataStore", column: str) -> list[str]:
    """Return one column as text over the graph's cells, every cell of ``I``."""
    frame = ds.cells.to_pandas_dataframe([column], key="I")
    values = frame[column]
    if values.isna().any():
        raise RuntimeError(f"The Wilk column {column!r} has missing values")
    return [str(value) for value in values]


def array_sha256(values: np.ndarray) -> str:
    """Fingerprint an array by dtype, shape, and C-ordered bytes."""
    data = np.ascontiguousarray(values)
    digest = hashlib.sha256(f"{data.dtype.str}{data.shape}".encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


def wilk_inputs(ds: "DataStore") -> dict[str, Any]:
    """Run CyteArc on Wilk and collect the arrays that R tests."""
    refs = wilk_graph(ds)
    pcs = np.asarray(ds.artifacts.load(refs["pca"])["data"][:], dtype=np.float32)
    knn = np.asarray(ds.artifacts.load(refs["neighbors"])["indices"][:], np.int64)
    nhoods = ds.abundance.make_neighborhoods(
        refs["neighbors"], prop=WILK["prop"], seed=WILK["samplingSeed"]
    )
    index_cells = ds.abundance.load_neighborhoods(nhoods).index_rows
    random_vertices = sample_index_cells(
        pcs.shape[0], WILK["prop"], seed=WILK["samplingSeed"]
    )
    return {
        "refs": refs,
        "pcs": pcs,
        "knn": knn,
        "randomVertices": random_vertices,
        "indexVertices": np.asarray(index_cells, dtype=np.int64),
        "sample": wilk_cell_labels(ds, WILK["sampleBy"]),
        "condition": wilk_cell_labels(ds, WILK["conditionBy"]),
        "label": wilk_cell_labels(ds, WILK["labelsBy"]),
    }


def wilk_fingerprints(inputs: dict[str, Any]) -> dict[str, str]:
    """Fingerprints that tie a fixture to the graph and draws it was made from."""
    return {
        "pcs": array_sha256(inputs["pcs"]),
        "knn": array_sha256(inputs["knn"]),
        "randomVertices": array_sha256(inputs["randomVertices"]),
        "indexVertices": array_sha256(inputs["indexVertices"]),
        "cells": hashlib.sha256(
            json.dumps(
                [inputs["sample"], inputs["condition"], inputs["label"]],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def export_wilk(args: argparse.Namespace) -> None:
    import cytearc

    cytearc.configure_output(level="WARNING", progress=False)
    ds, cytebase_id = wilk_datastore(args.work_dir)
    inputs = wilk_inputs(ds)
    pcs = inputs["pcs"]
    samples = list(dict.fromkeys(inputs["sample"]))
    condition_of: dict[str, str] = {}
    for sample, condition in zip(inputs["sample"], inputs["condition"], strict=True):
        if condition_of.setdefault(sample, condition) != condition:
            raise RuntimeError(f"Sample {sample!r} belongs to two conditions")
    case = {
        "name": "wilk",
        "coordinatesFile": "pcs.f64",
        "knnFile": "knn.i32",
        "nCells": int(pcs.shape[0]),
        "dims": int(pcs.shape[1]),
        "k": int(inputs["knn"].shape[1]),
        "randomVertices": (inputs["randomVertices"] + 1).tolist(),
        "indexVertices": (inputs["indexVertices"] + 1).tolist(),
        "refined": True,
        "sample": inputs["sample"],
        "label": inputs["label"],
        "designDf": {
            "sample": samples,
            "condition": [condition_of[sample] for sample in samples],
        },
        "conditionLevels": WILK["conditionLevels"],
        "formula": "~condition",
        "modelContrasts": None,
        "normMethod": "TMM",
        "minMean": 0.0,
        "exactKnn": True,
        "propeller": True,
    }
    with tempfile.TemporaryDirectory(prefix="cytearc-milo-wilk-") as raw:
        work = Path(raw)
        pcs.astype("<f8").tofile(work / "pcs.f64")
        (inputs["knn"] + 1).astype("<i4").tofile(work / "knn.i32")
        (work / "cases.json").write_text(json.dumps([case]))
        subprocess.run([str(args.rscript), str(R_SCRIPT), str(work)], check=True)
        output = json.loads((work / "results.json").read_text())
    versions = output["versions"]
    fixture = {
        "provenance": {
            "package": "miloR",
            "packageVersion": versions["miloR"],
            "edgeRVersion": versions["edgeR"],
            "limmaVersion": versions["limma"],
            "speckleVersion": versions["speckle"],
            "biocNeighborsVersion": versions["BiocNeighbors"],
            "igraphVersion": versions["igraph"],
            "rVersion": versions["r"],
            "cytebaseRelease": WILK["release"],
            "cytebaseId": cytebase_id,
            "generator": "scripts/export_milo_golden.py --wilk",
            "published": False,
        },
        "recipe": WILK,
        "inputs": {
            "fingerprints": wilk_fingerprints(inputs),
            "nCells": case["nCells"],
            "designDf": case["designDf"],
        },
        "milor": output["results"][0],
    }
    output_path = args.output if args.output != DEFAULT_FIXTURE else WILK_FIXTURE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    args = parse_args()
    if args.wilk:
        export_wilk(args)
        return
    cases = build_cases(args.seed)
    with tempfile.TemporaryDirectory(prefix="cytearc-milo-golden-") as raw:
        work = Path(raw)
        (work / "cases.json").write_text(json.dumps(cases))
        subprocess.run([str(args.rscript), str(R_SCRIPT), str(work)], check=True)
        output = json.loads((work / "results.json").read_text())
    versions = output["versions"]
    fixture = {
        "provenance": {
            "package": "miloR",
            "packageVersion": versions["miloR"],
            "edgeRVersion": versions["edgeR"],
            "limmaVersion": versions["limma"],
            "biocNeighborsVersion": versions["BiocNeighbors"],
            "igraphVersion": versions["igraph"],
            "rVersion": versions["r"],
            "sourceRepository": "https://bioconductor.org/packages/miloR",
            "referenceFunctions": [
                "miloR:::.refined_sampling",
                "miloR::makeNhoods (body, injected vertices)",
                "miloR::countCells",
                "miloR::testNhoods",
                "miloR::annotateNhoods",
            ],
            "generator": "scripts/export_milo_golden.py",
            "seed": int(args.seed),
        },
        "inputs": {"cases": cases, "sha256": input_fingerprint(cases)},
        "milor": output["results"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
