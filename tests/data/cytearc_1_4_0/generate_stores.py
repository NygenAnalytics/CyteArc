# /// script
# requires-python = ">=3.13"
# dependencies = ["cytearc[extra]==1.4.0"]
#
# [tool.uv]
# exclude-newer = "2026-11-17T00:00:00Z"
# ///
"""Write the stores in this folder with CyteArc 1.4.0, as a user of that release wrote them.

The tests in ``tests/test_stores_written_by_1_4_0.py`` check that later
releases open, read, replay, export and merge these stores as "Stores written
by other releases" in ``docs/source/developers/zarr_internals.md`` states, so
the stores must come from the released package, never from the code in this
checkout. The script refuses to run under any other release.

It reads two downloaded test fixtures, ``1K_pbmc_citeseq.h5`` and
``500_pbmc_atac.zarr.tar.gz`` (``uv run python -m tests.download_fixtures``),
and writes:

- ``rna.zarr``: 200 PBMCs imported from an H5AD file with a categorical
  ``cell_type`` and its colors, a categorical ``batch`` whose categories repeat
  a label, and a categorical ``phase`` inserted with ``ds.cells.insert``. It
  holds the RNA run ``rna``, with doublet scores and markers, the run
  ``rna_text_sampling``, which records a text ``kmeans_sampling`` and a UMAP
  ``min_dist`` above its ``spread``, doublet scores of a ``qc.doublets`` call,
  a pseudotime with a feature aggregation, a label snapshot of ``cell_type``,
  the receptors of ``receptors.tsv``, an AIRR Rearrangement file, with their
  clonotypes, and the clones of ``clones.csv``.
- ``atac.zarr``: 200 cells and 600 peaks with the ATAC run ``atac``.
- ``merged.zarr``: a union merge of two studies of 80 PBMCs that measured
  different genes. The studies hold ``cell_type`` as a categorical with colors
  and as text with a label more, ``note`` as text and as numbers with NaN,
  ``when`` as text and as dates with NaT, and ``collected`` as dates in
  seconds and in nanoseconds. Their sessions set ``sf=None``, and the left
  study records the size factor 2500.5.
- ``manifest.json``: the versions that wrote the stores, and what CyteArc
  1.4.0 read back from them, which the tests compare with.

Rebuild the stores from the repository root after moving the old ones away::

    uv run --script tests/data/cytearc_1_4_0/generate_stores.py

``exclude-newer`` resolves the packages that were released with 1.4.0. The
artifact and run IDs are random, and the results can differ by rounding from
one machine to another, so a rebuild writes other bytes, which the tests do
not depend on.
"""

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CYTEARC_MEM_BUDGET", "2G")
os.environ.setdefault("CYTEARC_WORKERS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

import cytearc  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent / "datasets"
STORES = ("rna.zarr", "atac.zarr", "merged.zarr")
INPUTS = ("receptors.tsv", "clones.csv")
SEED = 0
N_RNA_CELLS = 200
N_RNA_GENES = 300
N_STUDY_CELLS = 80
N_ATAC_CELLS = 200
N_ATAC_PEAKS = 600
# The packages whose versions decide what the stores hold.
RECORDED_PACKAGES = (
    "cytearc",
    "zarr",
    "numcodecs",
    "numpy",
    "pandas",
    "scipy",
    "h5py",
    "anndata",
    "numba",
    "umap-learn",
    "scikit-learn",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE,
        help="The folder to write the stores in (default: the folder of this script)",
    )
    output = parser.parse_args().output.resolve()
    if cytearc.__version__ != "1.4.0":
        sys.exit(
            f"This script writes stores as CyteArc 1.4.0 wrote them, but CyteArc "
            f"{cytearc.__version__} is installed. Run it with uv run --script."
        )
    present = [name for name in (*STORES, *INPUTS) if (output / name).exists()]
    if present:
        sys.exit(
            f"{output} already holds {', '.join(present)}; move them away to "
            "write the stores again"
        )
    cytearc.configure_output(level="WARNING", progress=False)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cytearc_1_4_0_") as scratch:
        work = Path(scratch)
        built = work / "built"
        built.mkdir()
        pbmc = read_pbmc()
        rng = np.random.default_rng(SEED)
        manifest: dict[str, Any] = {
            "packages": {
                name: importlib.metadata.version(name) for name in RECORDED_PACKAGES
            },
            "python": sys.version.split()[0],
        }
        manifest["rna"] = write_rna_store(built, work, pbmc, rng)
        manifest["atac"] = write_atac_store(built, work, rng)
        manifest["merged"] = write_merged_store(built, work, pbmc, rng)
        (built / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        )
        for name in (*STORES, *INPUTS, "manifest.json"):
            shutil.move(built / name, output / name)


def read_pbmc() -> dict[str, Any]:
    """Read the counts of the 1K PBMC CITE-seq fixture and call each cell's type from its antibodies.

    Each cell takes the type of the antibody among CD3, CD14, CD19 and CD56
    with the highest centered log ratio, and keeps it only when that ratio is
    above 1, so cells without a clear marker take ``"Other"``.
    """
    with h5py.File(DATASETS / "1K_pbmc_citeseq.h5", "r") as h5:
        matrix = h5["matrix"]
        n_features, n_cells = (int(value) for value in matrix["shape"][:])
        counts = csr_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=(n_cells, n_features),
        )
        ids = matrix["features/id"][:].astype(str)
        names = matrix["features/name"][:].astype(str)
        kinds = matrix["features/feature_type"][:].astype(str)
        barcodes = matrix["barcodes"][:].astype(str)
    genes = np.flatnonzero(kinds == "Gene Expression")
    proteins = np.flatnonzero(kinds == "Antibody Capture")
    antibodies = np.log1p(counts[:, proteins].toarray().astype(np.float64))
    ratios = antibodies - antibodies.mean(axis=1, keepdims=True)
    markers = {"T": "CD3", "Mono": "CD14", "B": "CD19", "NK": "CD56"}
    columns = [list(names[proteins]).index(marker) for marker in markers.values()]
    best = ratios[:, columns].argmax(axis=1)
    cell_types = np.asarray(list(markers), dtype=object)[best]
    cell_types[ratios[:, columns].max(axis=1) <= 1.0] = "Other"
    return {
        "counts": counts[:, genes].tocsr(),
        "ids": ids[genes],
        "names": names[genes],
        "barcodes": barcodes,
        "cell_types": cell_types,
    }


def write_h5ad(
    path: Path,
    counts: csr_matrix,
    cells: np.ndarray,
    features: np.ndarray,
    names: np.ndarray,
    obs: dict[str, Any],
    colors: dict[str, list[str]],
) -> None:
    """Write an H5AD file as AnnData writes one, with ``obs`` columns and their ``uns`` colors."""
    import anndata

    frame = pd.DataFrame(obs, index=pd.Index(cells, name=None))
    var = pd.DataFrame({"gene_short_name": names}, index=pd.Index(features, name=None))
    data = anndata.AnnData(
        X=counts.astype(np.float32), obs=frame, var=var, uns=dict(colors)
    )
    data.write_h5ad(path)


def repeat_a_category(path: Path, column: str, categories: list[str]) -> None:
    """Rewrite the categories of a categorical ``obs`` column so that a label repeats.

    pandas refuses such categories, so AnnData never writes them, but h5py
    writes them and CyteArc 1.3.0 and 1.4.0 imported each as a level of its own.
    """
    with h5py.File(path, "r+") as h5:
        group = h5[f"obs/{column}"]
        del group["categories"]
        group.create_dataset(
            "categories",
            data=np.asarray(categories, dtype=object),
            dtype=h5py.string_dtype(),
        )


def import_h5ad(source: Path, store: Path, assay: str) -> None:
    """Import an H5AD file with the defaults of CyteArc 1.4.0."""
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    H5adToZarr(H5adReader(str(source)), zarr_loc=str(store), assay_name=assay).dump()


def open_store(store: Path, **options: Any) -> Any:
    return cytearc.DataStore(str(store), nthreads=2, **options)


def chosen_genes(counts: csr_matrix, names: np.ndarray, n_genes: int) -> np.ndarray:
    """Return the genes of the store: the most detected ones, the mitochondrial and ribosomal ones, and the cell-cycle genes."""
    from cytearc.quality_control.cell_cycle_genes import (
        g2m_phase_genes,
        s_phase_genes,
    )

    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    kept = set(np.argsort(-detected, kind="stable")[:n_genes].tolist())
    cycle = set(s_phase_genes) | set(g2m_phase_genes)
    for index, name in enumerate(names):
        if detected[index] and (
            name in cycle or name.startswith(("MT-", "RPS", "RPL"))
        ):
            kept.add(index)
    return np.asarray(sorted(kept))


def write_rna_store(
    built: Path, work: Path, pbmc: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    cells = np.sort(rng.choice(len(pbmc["barcodes"]), N_RNA_CELLS, replace=False))
    counts = pbmc["counts"][cells]
    genes = chosen_genes(counts, pbmc["names"], N_RNA_GENES)
    counts = counts[:, genes]
    barcodes = pbmc["barcodes"][cells]
    cell_types = pbmc["cell_types"][cells]
    type_levels = ["T", "B", "NK", "Mono", "Other"]
    batch = rng.integers(0, 3, size=len(cells))
    source = work / "rna.h5ad"
    write_h5ad(
        source,
        counts,
        barcodes,
        pbmc["ids"][genes],
        pbmc["names"][genes],
        {
            "cell_type": pd.Categorical(cell_types, categories=type_levels),
            "batch": pd.Categorical.from_codes(batch, categories=["b1", "b2", "b3"]),
        },
        {
            "cell_type_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"],
            "batch_colors": ["#000001", "#000002", "#000003"],
        },
    )
    # Two categories of batch read "b1".
    repeat_a_category(source, "batch", ["b1", "b2", "b1"])
    path = built / "rna.zarr"
    import_h5ad(source, path, "RNA")
    ds = open_store(path)
    phases = np.where(np.arange(len(cells)) % 3 == 0, "cycling", "resting")
    ds.cells.insert(
        "phase",
        pd.Categorical(phases, categories=["resting", "cycling"], ordered=True),
        key=None,
    )
    # The default min_cells of 1.4.0 and of later releases differ below 2,000
    # cells, so the runs set it, and a later release selects the same genes
    # under a new revision of the selection.
    hvg = {"top_n": 150, "min_cells": 5}
    run = ds.pipeline.rna(
        label="rna",
        hvg=hvg,
        pca={"dims": 15},
        umap={"n_epochs": 100},
        snapshot_columns=["cell_type"],
    )
    # Settings that 1.4.0 accepted: a kmeans_sampling given as text, which
    # later releases refuse, and a min_dist above the spread.
    text_sampling = ds.pipeline.rna(
        label="rna_text_sampling",
        hvg=hvg,
        pca={"dims": 15},
        embedding_initialization={"kmeans_sampling": "0.2"},
        umap={"n_epochs": 100, "min_dist": 1.5, "spread": 1.0},
    )
    selection = run["analysis_cell_selection"]
    # The run's doublet scores record the run's frozen feature names, which a
    # call of qc.doublets does not take, so a call scores them apart.
    doublets = ds.qc.doublets(run["clusters"], run["connectivity_map"])
    labels = np.asarray(ds.artifacts.load_values(run["clusters"]).values)
    present = sorted(set(labels.tolist()))
    pseudotime = ds.trajectory.pseudotime(
        run["connectivity_map"],
        source_sink=run["clusters"],
        sources=[present[0]],
        sinks=[present[-1]],
    )
    aggregation = ds.trajectory.aggregation(
        pseudotime,
        features=run["highly_variable_features"],
        window_size=30,
        chunk_size=10,
        n_neighbors=5,
        n_clusters=3,
    )
    cell_type_labels = ds.snapshot_cluster_labels("cell_type", cell_selection=selection)
    receptors_file = built / "receptors.tsv"
    write_receptors(receptors_file, barcodes, cell_types, rng)
    receptors = ds.repertoire.import_receptors(
        str(receptors_file), cell_selection=selection
    )
    clonotypes = ds.repertoire.clonotypes(receptors)
    clones_file = built / "clones.csv"
    write_clones(clones_file, barcodes, rng)
    clones = ds.clones.import_clones(
        str(clones_file),
        cell_selection=selection,
        columns={"cell": "cell", "clone": "clone"},
    )
    return {
        "runs": {
            name: {"runId": recorded.run_id, "outputs": refs(recorded)}
            for name, recorded in (("rna", run), ("rna_text_sampling", text_sampling))
        },
        "artifacts": {
            name: ref_record(ref)
            for name, ref in (
                ("doublets", doublets),
                ("pseudotime", pseudotime),
                ("aggregation", aggregation),
                ("cellTypeLabels", cell_type_labels),
                ("receptors", receptors),
                ("clonotypes", clonotypes),
                ("clones", clones),
            )
        },
        "read": {
            "cellType": ds.cells.fetch_all("cell_type").tolist(),
            "batch": ds.cells.fetch_all("batch").tolist(),
            "phase": ds.cells.fetch_all("phase").tolist(),
            "hvgNames": selected_names(ds, run["highly_variable_features"]),
            "sizeFactor": ds.RNA.sf,
        },
    }


def selected_names(ds: Any, selection: Any) -> list[str]:
    selected = np.asarray(ds.artifacts.load(selection)["values"][:], dtype=bool)
    return ds.RNA.feats.fetch_all("names").astype(str)[selected].tolist()


def write_receptors(
    path: Path, barcodes: np.ndarray, cell_types: np.ndarray, rng: np.random.Generator
) -> None:
    """Write an AIRR Rearrangement file of the T and B cells, in clones of one to four cells.

    Each clone's junctions spell its number, so its cells share their chains
    and no two clones share one. T cell clones carry a TRA and a TRB chain, B
    cell clones an IGH and an IGK chain.
    """
    codons = dict(
        zip(
            "ACDEFGHIKLMNPQRSTVWY",
            "GCT TGT GAT GAA TTT GGT CAT ATT AAA CTG ATG AAT CCG CAG CGT AGC ACC "
            "GTG TGG TAT".split(),
            strict=True,
        )
    )
    residues = "ACDEFGHIKLMNPQRSTVWY"
    header = (
        "cell_id\tsequence_id\tlocus\tv_call\td_call\tj_call\tc_call\tjunction"
        "\tjunction_aa\tproductive\tduplicate_count\tconsensus_count"
    )
    rows = []
    clone = 0
    for kind, loci in (("T", ("TRA", "TRB")), ("B", ("IGH", "IGK"))):
        members = np.flatnonzero(cell_types == kind)
        start = 0
        while start < len(members):
            size = int(rng.integers(1, 5))
            spelled = "".join(residues[clone // 20**power % 20] for power in range(3))
            for cell in members[start : start + size]:
                for arm, locus in enumerate(loci):
                    junction_aa = f"C{spelled}{'FW'[arm]}"
                    junction = "".join(codons[residue] for residue in junction_aa)
                    rows.append(
                        f"{barcodes[cell]}\t{barcodes[cell]}_contig_{arm + 1}\t{locus}"
                        f"\t{locus}V1-1*01\t\t{locus}J1*01\t{locus}C\t{junction}"
                        f"\t{junction_aa}\tT\t{int(rng.integers(2, 20))}"
                        f"\t{int(rng.integers(50, 500))}"
                    )
            start += size
            clone += 1
    path.write_text("\n".join([header, *rows]) + "\n")


def write_clones(path: Path, barcodes: np.ndarray, rng: np.random.Generator) -> None:
    """Write a table of a clone for every other cell, in 25 clones."""
    listed = barcodes[::2]
    labels = rng.integers(0, 25, size=len(listed))
    rows = [f"{cell},clone{label}" for cell, label in zip(listed, labels, strict=True)]
    path.write_text("\n".join(["cell,clone", *rows]) + "\n")


def write_atac_store(
    built: Path, work: Path, rng: np.random.Generator
) -> dict[str, Any]:
    archive = DATASETS / "500_pbmc_atac.zarr.tar.gz"
    unpacked = work / "500_pbmc_atac.zarr"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(unpacked, filter="data")
    root = zarr.open_group(str(unpacked), mode="r")
    n_cells = int(root["ATAC/counts"].shape[0])
    cells = np.sort(rng.choice(n_cells, N_ATAC_CELLS, replace=False))
    dense = np.asarray(root["ATAC/counts"][:])[cells]
    detected = (dense > 0).sum(axis=0)
    peaks = np.sort(np.argsort(-detected, kind="stable")[:N_ATAC_PEAKS])
    peak_ids = np.asarray(root["ATAC/featureData/ids"][:]).astype(str)[peaks]
    cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)[cells]
    source = work / "atac.h5ad"
    write_h5ad(
        source,
        csr_matrix(dense[:, peaks]),
        cell_ids,
        peak_ids,
        peak_ids,
        {},
        {},
    )
    path = built / "atac.zarr"
    import_h5ad(source, path, "ATAC")
    ds = open_store(path)
    # Without markers, whose tables of every peak would outweigh the rest of
    # the store.
    run = ds.pipeline.atac(
        label="atac",
        prevalent_peaks={"top_n": 200},
        lsi={"dims": 10},
        umap={"n_epochs": 100},
        markers=False,
    )
    return {"runs": {"atac": {"runId": run.run_id, "outputs": refs(run)}}}


def write_merged_store(
    built: Path, work: Path, pbmc: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Merge two studies of the PBMCs that measured different genes, as CyteArc 1.4.0 merged them."""
    types = pbmc["cell_types"]
    left_cells = np.sort(
        rng.choice(
            np.flatnonzero(np.isin(types, ["T", "B"])), N_STUDY_CELLS, replace=False
        )
    )
    rest = np.setdiff1d(np.flatnonzero(np.isin(types, ["T", "B", "NK"])), left_cells)
    right_cells = np.sort(rng.choice(rest, N_STUDY_CELLS, replace=False))
    counts = pbmc["counts"]
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    ranked = np.argsort(-detected, kind="stable")
    shared, left_only, right_only = ranked[:150], ranked[150:170], ranked[170:190]
    left_genes = np.sort(np.concatenate([shared, left_only]))
    right_genes = np.sort(np.concatenate([shared, right_only]))
    stores = {}
    for name, cells, genes, obs, colors in (
        (
            "left",
            left_cells,
            left_genes,
            {"cell_type": pd.Categorical(types[left_cells], categories=["T", "B"])},
            {"cell_type_colors": ["#1f77b4", "#ff7f0e"]},
        ),
        ("right", right_cells, right_genes, {}, {}),
    ):
        source = work / f"{name}.h5ad"
        write_h5ad(
            source,
            counts[cells][:, genes],
            pbmc["barcodes"][cells],
            pbmc["ids"][genes],
            pbmc["names"][genes],
            obs,
            colors,
        )
        path = work / f"{name}.zarr"
        import_h5ad(source, path, "RNA")
        stores[name] = path
    # Another tool, or an earlier release, can record any size factor, and
    # CyteArc 1.4.0 normalizes the assay with it as stored.
    zarr.open_group(str(stores["left"]), mode="r+")["RNA"].attrs["size_factor"] = 2500.5
    left = open_store(stores["left"])
    right = open_store(stores["right"])
    # AnnData writes text as categoricals, so the text columns are inserted
    # after the import, as plain text that records no categories.
    right.cells.insert("cell_type", types[right_cells].astype(str), key=None)
    left.cells.insert(
        "note",
        np.where(np.arange(N_STUDY_CELLS) % 2 == 0, "fresh", "frozen"),
        key=None,
    )
    left.cells.insert("when", np.full(N_STUDY_CELLS, "day 1"), key=None)
    start = np.datetime64("2026-03-01T08:00:00", "s")
    left.cells.insert(
        "collected", start + np.arange(N_STUDY_CELLS) * np.timedelta64(1, "h"), key=None
    )
    notes = rng.normal(size=N_STUDY_CELLS)
    notes[::4] = np.nan
    right.cells.insert("note", notes, key=None)
    when = (start + np.arange(N_STUDY_CELLS) * np.timedelta64(1, "D")).astype(
        "datetime64[ns]"
    )
    when[::5] = np.datetime64("NaT", "ns")
    right.cells.insert("when", when, key=None)
    right.cells.insert(
        "collected",
        (start + np.arange(N_STUDY_CELLS) * np.timedelta64(90, "m")).astype(
            "datetime64[ns]"
        ),
        key=None,
    )
    left.RNA.sf = None
    right.RNA.sf = None
    from cytearc.merge import DataStoreMerge

    path = built / "merged.zarr"
    DataStoreMerge(
        [left, right],
        str(path),
        ["left", "right"],
        source_column="study",
        nthreads=2,
    ).dump()
    # The first writable open records the size factor, which no source gave.
    merged = open_store(path)
    cells = merged.cells
    return {
        "read": {
            "ids": cells.fetch_all("ids").tolist(),
            "study": cells.fetch_all("study").tolist(),
            "cellType": cells.fetch_all("orig_cell_type").tolist(),
            "cellTypeAttributes": dict(cells._get_array("orig_cell_type").attrs),
            "note": cells.fetch_all("orig_note").tolist(),
            "when": cells.fetch_all("orig_when").tolist(),
            "collected": cells.fetch_all("orig_collected").tolist(),
            "collectedDtype": str(cells.fetch_all("orig_collected").dtype),
            "sizeFactor": merged.RNA.sf,
            "zeroFills": merged_zero_fills(
                merged, path, pbmc["ids"], left_only, right_only
            ),
        },
    }


def merged_zero_fills(
    merged: Any,
    path: Path,
    ids: np.ndarray,
    left_only: np.ndarray,
    right_only: np.ndarray,
) -> dict[str, Any]:
    """Return, for a gene that only one study measured, its stored counts in every merged cell.

    The merge writes zeros in the cells of the study that lacks the gene.
    """
    features = merged.RNA.feats.fetch_all("ids").astype(str).tolist()
    counts = zarr.open_group(str(path), mode="r")["RNA/counts"]
    found = {}
    for name, only in (("left", left_only), ("right", right_only)):
        gene = str(ids[only[0]])
        found[name] = {
            "gene": gene,
            "counts": np.asarray(counts[:, features.index(gene)]).ravel().tolist(),
        }
    return found


def refs(run: Any) -> dict[str, Any]:
    """Return a run's outputs as records of their refs."""
    recorded: dict[str, Any] = {}
    for key, value in run.items():
        if isinstance(value, tuple):
            recorded[key] = [ref_record(ref) for ref in value]
        else:
            recorded[key] = ref_record(value)
    return recorded


def ref_record(ref: Any) -> dict[str, Any]:
    return {
        "scope": ref.scope,
        "kind": ref.kind,
        "artifactId": ref.artifact_id,
        "assay": ref.assay,
    }


if __name__ == "__main__":
    main()
