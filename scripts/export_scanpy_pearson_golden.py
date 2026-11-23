#!/usr/bin/env python
"""Regenerate tests/scanpy_pearson_residuals_1_12_4_golden.json.

Runs scanpy 1.12.4's ``experimental.pp.normalize_pearson_residuals``, with
theta 100 and its default clip, on the RNA counts of the active cells of the
1k PBMC CITE-seq test fixture: once over every gene, as CyteArc normalizes a
gene selection with each cell's total over every gene, and once over the
genes of the selection alone, as it normalizes with ``renormalize_subset``.
It writes the residuals of every sixteenth active cell for the selection, the
200 genes with the most counts. The fixture is downloaded by
``tests/download_fixtures.py``. Run from the repository root:

    uv run --with scanpy==1.12.4 python scripts/export_scanpy_pearson_golden.py
"""

import argparse
import hashlib
import json
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path

import anndata
import numpy as np
import scanpy as sc
import zarr
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests/datasets/1K_pbmc_citeseq.zarr.tar.gz"
DEFAULT_OUTPUT = REPO / "tests/scanpy_pearson_residuals_1_12_4_golden.json"
N_GENES = 200
CELL_STEP = 16
THETA = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def counts_sha256(counts: np.ndarray) -> str:
    """Fingerprint counts by their CSR arrays, as tests/test_hvg_flavors.py does."""
    matrix = csr_matrix(counts.astype(np.int64))
    digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(values, dtype="<i8").tobytes())
    return digest.hexdigest()


def read_counts(fixture: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="cytearc-pearson-golden-") as work:
        with tarfile.open(fixture, "r:gz") as archive:
            archive.extractall(work, filter="data")
        root = zarr.open_group(work, mode="r")
        active = np.asarray(root["cellData/I"][:], dtype=bool)
        return np.asarray(root["RNA/counts"][:])[np.flatnonzero(active)]


def residuals(counts: np.ndarray) -> np.ndarray:
    adata = anndata.AnnData(csr_matrix(counts.astype(np.float64)))
    result = sc.experimental.pp.normalize_pearson_residuals(
        adata, theta=THETA, clip=None, inplace=False
    )
    return np.asarray(result["X"], dtype=np.float64)


def rounded(values: np.ndarray) -> list[list[float]]:
    return [[float(f"{value:.9g}") for value in row] for row in values]


def main() -> None:
    args = parse_args()
    counts = read_counts(args.fixture)
    # Signed sums, so that the negation orders the genes by descending counts.
    totals = counts.sum(axis=0, dtype=np.int64)
    genes = np.sort(np.argsort(-totals, kind="stable")[:N_GENES]).astype(np.int64)
    cells = np.arange(0, len(counts), CELL_STEP, dtype=np.int64)
    every_gene = residuals(counts)[np.ix_(cells, genes)]
    selection = residuals(counts[:, genes])[cells]
    fixture = {
        "provenance": {
            "package": "scanpy",
            "packageVersion": version("scanpy"),
            "anndataVersion": version("anndata"),
            "numpyVersion": version("numpy"),
            "sourceRepository": "https://github.com/scverse/scanpy",
            "license": "BSD-3-Clause",
            "function": "scanpy.experimental.pp.normalize_pearson_residuals",
            "theta": THETA,
            "clip": None,
            "nCells": int(counts.shape[0]),
            "nGenes": int(counts.shape[1]),
            "script": "scripts/export_scanpy_pearson_golden.py",
        },
        "inputs": {"countsSha256": counts_sha256(counts)},
        "cells": cells.tolist(),
        "genes": genes.tolist(),
        "everyGene": rounded(every_gene),
        "selection": rounded(selection),
    }
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")


if __name__ == "__main__":
    main()
