#!/usr/bin/env python3
"""Prepare and assemble SCENT's reference outputs for CyteArc's potency score.

SCENT (GPL-3, https://github.com/aet21/SCENT) is run only to produce the
reference values; CyteArc's measures are written from the papers. The steps:

1. ``prepare`` takes 16 cells of each of four stages of the 4K pancreas test
   fixture, the genes of the mouse STRING network that they measure and
   that at least five of the cells detect, and the largest connected part of
   the network among them. It writes the cells' log values
   ``log2(10,000 * counts / total + 1.1)``, with each cell's total over all its
   genes, the network's edges and the inputs that the test reads.
2. ``scripts/export_scent_potency_golden.R`` runs SCENT on those values.
3. ``assemble`` writes ``tests/data/potency/scent_reference.npz``, the inputs
   with SCENT's values, and ``scent_reference.json``, their provenance.

Example:
    uv run python scripts/export_scent_potency_golden.py prepare \\
        --h5ad tests/datasets/bastidas-ponce_4K_pancreas-d15_rnaseq/data.h5ad \\
        --network ~/.cache/cytearc/networks/string_v12.5/mouse_10090.npz --work build/scent
    Rscript scripts/export_scent_potency_golden.R build/scent build/scent
    uv run python scripts/export_scent_potency_golden.py assemble --work build/scent \\
        --out tests/data/potency
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

STAGES = ("Ductal", "Ngn3 high EP", "Pre-endocrine", "Beta")
CELLS_PER_STAGE = 16
MIN_DETECTING_CELLS = 5
SEED = 0


def prepare(h5ad: Path, network_path: Path, work: Path) -> None:
    import anndata
    from scipy import sparse

    from cytearc.trajectory.potency import match_network, read_network

    data = anndata.read_h5ad(h5ad)
    stages = np.asarray(data.obs["clusters"]).astype(str)
    rng = np.random.default_rng(SEED)
    chosen = np.concatenate(
        [
            np.sort(
                rng.choice(
                    np.flatnonzero(stages == stage), CELLS_PER_STAGE, replace=False
                )
            )
            for stage in STAGES
        ]
    )
    counts = sparse.csr_matrix(data.X)[chosen]
    if not np.allclose(counts.data, np.round(counts.data)):
        raise ValueError(f"{h5ad} does not hold raw counts in X")
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    symbols = np.asarray(data.var_names).astype(str)

    network = read_network(network_path)
    detecting = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep = detecting >= MIN_DETECTING_CELLS
    # Match only the genes that enough cells detect; the network's largest
    # connected part among them is the reference network.
    matched = match_network(network, np.where(keep, symbols, ""), "symbol")
    genes = matched.features
    adjacency = sparse.triu(matched.adjacency, k=1).tocoo()
    edges = np.stack([adjacency.row, adjacency.col]).astype(np.int64)
    order = np.lexsort((edges[1], edges[0]))
    edges = edges[:, order]
    gene_counts = counts[:, genes].toarray()
    values = np.log2(10_000.0 * gene_counts / totals[:, None] + 1.1)

    work.mkdir(parents=True, exist_ok=True)
    cell_ids = np.asarray(data.obs_names).astype(str)[chosen]
    with (work / "log_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", *cell_ids])
        for gene, row in zip(symbols[genes], values.T, strict=True):
            writer.writerow([gene, *(repr(float(value)) for value in row)])
    with (work / "edges.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["first", "second"])
        writer.writerows(edges.T.tolist())
    np.savez_compressed(
        work / "inputs.npz",
        symbols=symbols[genes],
        edges=edges.astype(np.uint16),
        cell_ids=cell_ids,
        stages=stages[chosen],
        counts=gene_counts.astype(np.uint16),
        totals=totals,
    )
    (work / "prepared.json").write_text(
        json.dumps(
            {
                "h5ad": h5ad.name,
                "h5adSha256": hashlib.sha256(h5ad.read_bytes()).hexdigest(),
                "network": network.metadata,
                "networkSha256": hashlib.sha256(network_path.read_bytes()).hexdigest(),
                "genes": int(len(genes)),
                "edges": int(edges.shape[1]),
                "cells": int(len(chosen)),
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"{len(chosen)} cells, {len(genes)} genes, {edges.shape[1]} edges")


def assemble(work: Path, out: Path) -> None:
    with np.load(work / "inputs.npz", allow_pickle=False) as stored:
        inputs = {name: stored[name] for name in stored.files}
    with (work / "scent_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if [row["cell"] for row in rows] != inputs["cell_ids"].tolist():
        raise ValueError("SCENT's scores do not follow the prepared cells")
    prepared = json.loads((work / "prepared.json").read_text())
    versions = (work / "versions.txt").read_text().splitlines()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "scent_reference.npz",
        **inputs,
        signaling_entropy=np.array([float(row["signaling_entropy"]) for row in rows]),
        ccat=np.array([float(row["ccat"]) for row in rows]),
    )
    provenance = {
        "tool": versions,
        "calls": [
            "integ.l <- DoIntegPPI(exp.m = values, ppiA.m = adjacency)",
            "sr.o <- CompSRana(integ.l, local = FALSE, mc.cores = 1); sr.o$SR",
            "CompCCAT(exp = values, ppiA = adjacency)",
        ],
        "script": "scripts/export_scent_potency_golden.R",
        "environment": "Rscript in the rocker/r-ver:4.6.1 image, SCENT installed "
        "with remotes::install_github('aet21/SCENT', dependencies = FALSE) after "
        "igraph, mclust, Matrix and qlcMatrix from CRAN and destiny and marray "
        "from Bioconductor",
        "values": "log2(10,000 * counts / total + 1.1), each cell's total over all "
        "its genes; CCAT takes the same values as the signaling entropy rate",
        "cells": f"{CELLS_PER_STAGE} cells of each of {', '.join(STAGES)}, drawn "
        f"with NumPy's default generator seeded {SEED}",
        "data": "4K pancreas test fixture, Bastidas-Ponce et al. (2019), "
        "Development 146:dev173849, doi:10.1242/dev.173849, GEO GSE132188",
        "network": "the largest connected part, among the genes that at least "
        f"{MIN_DETECTING_CELLS} of the cells detect, of the mouse STRING v12.5 "
        "network that CyteArc downloads (STRING, CC BY 4.0)",
        "prepared": prepared,
    }
    (out / "scent_reference.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n"
    )
    print(f"wrote {out / 'scent_reference.npz'} and its provenance")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("prepare")
    first.add_argument("--h5ad", type=Path, required=True)
    first.add_argument("--network", type=Path, required=True)
    first.add_argument("--work", type=Path, required=True)
    last = steps.add_parser("assemble")
    last.add_argument("--work", type=Path, required=True)
    last.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.step == "prepare":
        prepare(args.h5ad, args.network, args.work)
    else:
        assemble(args.work, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
