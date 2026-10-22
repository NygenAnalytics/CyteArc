#!/usr/bin/env python
"""Regenerate tests/scanpy_hvg_1_12_4_golden.json.

Runs scanpy 1.12.4 with scikit-misc on the RNA counts of the active cells of
the 1k PBMC CITE-seq test fixture and writes compact reference outputs for
the seurat_v3 and pearson_residuals highly variable gene flavors, with and
without batches. The batches split the active cells into their first and
second halves. The fixture is downloaded by ``tests/download_fixtures.py``.
Run from the repository root:

    uv run --with scanpy==1.12.4 --with scikit-misc==0.5.3 \\
        python scripts/export_scanpy_hvg_golden.py
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
import pandas as pd
import scanpy as sc
import zarr
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests/datasets/1K_pbmc_citeseq.zarr.tar.gz"
DEFAULT_OUTPUT = REPO / "tests/scanpy_hvg_1_12_4_golden.json"
TOP_N = 2000
# Scores are kept for the genes that scanpy ranks first, which hold every
# selection boundary, and for every SAMPLE_STEP-th of the other scored genes.
SCORED_TOP = 2200
SAMPLE_STEP = 50
PEARSON_BATCH_CLIP = 20.0


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


def read_counts(fixture: Path) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory(prefix="cytearc-hvg-golden-") as work:
        with tarfile.open(fixture, "r:gz") as archive:
            archive.extractall(work, filter="data")
        root = zarr.open_group(work, mode="r")
        active = np.asarray(root["cellData/I"][:], dtype=bool)
        counts = np.asarray(root["RNA/counts"][:])[np.flatnonzero(active)]
    return counts, int(active.size)


def scored(scores: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Return the indices and scores of the top-ranked and sampled genes."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    positive = order[scores[order] > 0]
    keep = np.unique(
        np.concatenate([positive[:SCORED_TOP], positive[SCORED_TOP::SAMPLE_STEP]])
    )
    return {
        "indices": keep.tolist(),
        "scores": [float(f"{value:.10g}") for value in scores[keep]],
    }


def selection(var: pd.DataFrame, score: str, batches: bool) -> dict[str, object]:
    selected = np.flatnonzero(var["highly_variable"].to_numpy())
    record: dict[str, object] = {
        "selected": selected.tolist(),
        **scored(var[score].to_numpy(dtype=np.float64)),
    }
    if batches:
        record["batchCount"] = (
            var["highly_variable_nbatches"].to_numpy()[selected].astype(int).tolist()
        )
        record["medianRank"] = [
            float(value)
            for value in var["highly_variable_rank"].to_numpy(dtype=np.float64)[
                selected
            ]
        ]
    return record


def main() -> None:
    args = parse_args()
    counts, n_store_cells = read_counts(args.fixture)
    n_cells, n_genes = counts.shape
    batch = np.where(np.arange(n_cells) < n_cells // 2, "a", "b")
    adata = anndata.AnnData(csr_matrix(counts.astype(np.float64)))
    adata.obs["batch"] = pd.Categorical(batch)

    outputs: dict[str, dict[str, object]] = {}
    single = adata.copy()
    sc.pp.highly_variable_genes(single, flavor="seurat_v3", n_top_genes=TOP_N)
    outputs["seurat_v3"] = selection(single.var, "variances_norm", False)
    batched = adata.copy()
    sc.pp.highly_variable_genes(
        batched, flavor="seurat_v3_paper", n_top_genes=TOP_N, batch_key="batch"
    )
    outputs["seurat_v3_paper_batch"] = selection(batched.var, "variances_norm", True)
    single = adata.copy()
    sc.experimental.pp.highly_variable_genes(
        single, flavor="pearson_residuals", n_top_genes=TOP_N
    )
    outputs["pearson_residuals"] = selection(single.var, "residual_variances", False)
    batched = adata.copy()
    sc.experimental.pp.highly_variable_genes(
        batched,
        flavor="pearson_residuals",
        n_top_genes=TOP_N,
        batch_key="batch",
        clip=PEARSON_BATCH_CLIP,
    )
    outputs["pearson_residuals_batch"] = selection(
        batched.var, "residual_variances", True
    )

    fixture = {
        "provenance": {
            "package": "scanpy",
            "packageVersion": version("scanpy"),
            "scikitMiscVersion": version("scikit-misc"),
            "anndataVersion": version("anndata"),
            "numpyVersion": version("numpy"),
            "sourceRepository": "https://github.com/scverse/scanpy",
            "dataset": "1K_pbmc_citeseq.zarr.tar.gz",
            "datasetSource": "tests/download_fixtures.py",
            "generator": "scripts/export_scanpy_hvg_golden.py",
            "nStoreCells": n_store_cells,
            "nCells": n_cells,
            "nGenes": n_genes,
            "topN": TOP_N,
            "batches": "first and second halves of the active cells, a and b",
            "pearsonBatchClip": PEARSON_BATCH_CLIP,
            "scoredTop": SCORED_TOP,
            "sampleStep": SAMPLE_STEP,
            "calls": {
                "seurat_v3": "sc.pp.highly_variable_genes(flavor='seurat_v3')",
                "seurat_v3_paper_batch": (
                    "sc.pp.highly_variable_genes(flavor='seurat_v3_paper', "
                    "batch_key='batch')"
                ),
                "pearson_residuals": (
                    "sc.experimental.pp.highly_variable_genes("
                    "flavor='pearson_residuals')"
                ),
                "pearson_residuals_batch": (
                    "sc.experimental.pp.highly_variable_genes("
                    "flavor='pearson_residuals', batch_key='batch', clip=20.0)"
                ),
            },
        },
        "inputs": {"countsSha256": counts_sha256(counts)},
        "scanpy": outputs,
    }
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
