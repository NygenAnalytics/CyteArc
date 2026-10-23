"""Feature selections, cell-cycle scores, and mapping results on read-only stores.

These operations plan before they check that the store is writable, so a
read-only store returns a result that it already holds and refuses only a
result that must be computed, before it reads the counts.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from cytearc import DataStore
from cytearc.assay import ATACassay, RNAassay
from cytearc.storage.artifacts import ArtifactRef
from tests.storage_helpers import write_count_store
from tests.test_mapping_reference import build_mapping_query, build_mapping_store

N_CELLS = 60
S_GENES = ["RNA1", "RNA2", "RNA3"]
G2M_GENES = ["RNA4", "RNA5", "RNA6"]


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(41)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=24), size=(N_CELLS, 24))
    rna[:, 0] += 2
    return {
        "RNA": rna,
        "ADT": rng.poisson(15.0, size=(N_CELLS, 4)) + 1,
        "ATAC": rng.poisson(0.9, size=(N_CELLS, 30)),
    }


def _open(zarr_loc: Path, **options) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


def _hvgs(store: DataStore, cells: ArtifactRef, **options) -> ArtifactRef:
    settings = {"min_cells": 1, "top_n": 8, "n_bins": 4, "show_plot": False}
    return store.features.hvgs(cells, **(settings | options))


def _flavor_hvgs(store: DataStore, cells: ArtifactRef, **options) -> ArtifactRef:
    settings = {"top_n": 8, "show_plot": False}
    return store.features.hvgs(cells, **(settings | options))


def _cell_cycle(store: DataStore, cells: ArtifactRef, **options) -> ArtifactRef:
    settings = {"s_genes": S_GENES, "g2m_genes": G2M_GENES, "n_bins": 3}
    return store.scores.cell_cycle(cells, **(settings | options))


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    zarr_loc = tmp_path_factory.mktemp("read_only_reuse") / "store.zarr"
    write_count_store(str(zarr_loc), _counts(), "uint16")
    store = _open(zarr_loc)
    store.cells.insert("first_half", np.arange(N_CELLS) < N_CELLS // 2)
    store.cells.insert("batch", np.where(np.arange(N_CELLS) % 2 == 0, "a", "b"))
    cells = store.snapshot_cell_selection()
    refs = {
        "cells": cells,
        "first_half": store.snapshot_cell_selection("first_half"),
        "universe": store.features.universe(from_assay="RNA"),
        "snapshot": store.features.snapshot(
            from_assay="RNA", feature_indexes=[0, 2, 4]
        ),
        "hvgs": _hvgs(store, cells),
        "seurat_v3": _flavor_hvgs(store, cells, flavor="seurat_v3"),
        "batch_hvgs": _flavor_hvgs(
            store, cells, flavor="pearson_residuals", batch="batch"
        ),
        "detected": store.features.detected(cells, min_cells=2),
        "cell_cycle": _cell_cycle(store, cells),
        "peaks": store.features.prevalent_peaks(cells, from_assay="ATAC", top_n=5),
    }
    return zarr_loc, refs


@pytest.fixture
def read_only(template, monkeypatch) -> tuple[DataStore, dict[str, ArtifactRef]]:
    """The template opened read only, with every computing kernel forbidden."""
    zarr_loc, refs = template

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a read-only store must not compute")

    for owner, name in (
        (RNAassay, "_compute_feature_summary"),
        (ATACassay, "_compute_feature_summary"),
        (RNAassay, "_select_hvgs"),
        (RNAassay, "_batch_hvg_scores"),
        (RNAassay, "_score_feature_indices"),
        (ATACassay, "_prevalent_peak_mask"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    return _open(zarr_loc, zarr_mode="r"), refs


def test_read_only_stores_return_saved_selections_and_scores(read_only) -> None:
    store, refs = read_only
    cells = refs["cells"]

    assert store.features.universe(from_assay="RNA") == refs["universe"]
    assert (
        store.features.snapshot(from_assay="RNA", feature_indexes=[0, 2, 4])
        == refs["snapshot"]
    )
    mask = np.zeros(24, dtype=bool)
    mask[[0, 2, 4]] = True
    assert store.features.snapshot(from_assay="RNA", mask=mask) == refs["snapshot"]
    assert _hvgs(store, cells) == refs["hvgs"]
    assert _flavor_hvgs(store, cells, flavor="seurat_v3") == refs["seurat_v3"]
    assert (
        _flavor_hvgs(store, cells, flavor="pearson_residuals", batch="batch")
        == refs["batch_hvgs"]
    )
    assert store.features.detected(cells, min_cells=2) == refs["detected"]
    assert _cell_cycle(store, cells) == refs["cell_cycle"]
    assert (
        store.features.prevalent_peaks(cells, from_assay="ATAC", top_n=5)
        == refs["peaks"]
    )


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            lambda store, refs: store.features.universe(from_assay="ADT"),
            "select_all_features",
        ),
        (
            lambda store, refs: store.features.snapshot(
                from_assay="RNA", feature_indexes=[1, 3]
            ),
            "set_feature_selection",
        ),
        (
            lambda store, refs: _hvgs(store, refs["cells"], top_n=9),
            "select_hvgs",
        ),
        # The count flavors and batch-aware selection refuse before they
        # score any batch.
        (
            lambda store, refs: _flavor_hvgs(
                store, refs["cells"], flavor="seurat_v3", top_n=9
            ),
            "select_hvgs",
        ),
        (
            lambda store, refs: _flavor_hvgs(
                store,
                refs["cells"],
                flavor="pearson_residuals",
                batch="batch",
                top_n=9,
            ),
            "select_hvgs",
        ),
        # A new cell selection has no feature summary either, which is
        # refused before the counts are read.
        (
            lambda store, refs: _hvgs(store, refs["first_half"]),
            "summarize_rna_features",
        ),
        (
            lambda store, refs: store.features.detected(refs["cells"], min_cells=3),
            "select_detected_features",
        ),
        (
            lambda store, refs: _cell_cycle(store, refs["cells"], rand_seed=1),
            "run_cell_cycle_scoring",
        ),
        (
            lambda store, refs: _cell_cycle(store, refs["first_half"]),
            "summarize_rna_features",
        ),
        (
            lambda store, refs: store.features.prevalent_peaks(
                refs["cells"], from_assay="ATAC", top_n=6
            ),
            "select_prevalent_peaks",
        ),
        (
            lambda store, refs: store.features.prevalent_peaks(
                refs["first_half"], from_assay="ATAC", top_n=5
            ),
            "summarize_atac_features",
        ),
    ],
    ids=[
        "universe",
        "snapshot",
        "hvgs",
        "hvgs-seurat-v3",
        "hvgs-batch",
        "hvgs-new-cells",
        "detected",
        "cell-cycle",
        "cell-cycle-new-cells",
        "prevalent-peaks",
        "prevalent-peaks-new-cells",
    ],
)
def test_read_only_stores_refuse_new_results_before_computing(
    read_only, call, operation
) -> None:
    store, refs = read_only
    before = {
        kind: store.artifacts.list(kind=kind, from_assay=assay)
        for kind, assay in (
            ("feature_selection", "RNA"),
            ("feature_summary", "RNA"),
            ("feature_selection", "ATAC"),
        )
    }

    with pytest.raises(
        PermissionError,
        match=f"^{operation} requires a DataStore opened with zarr_mode='r\\+'$",
    ):
        call(store, refs)

    assert before == {
        kind: store.artifacts.list(kind=kind, from_assay=assay)
        for kind, assay in (
            ("feature_selection", "RNA"),
            ("feature_summary", "RNA"),
            ("feature_selection", "ATAC"),
        )
    }


def test_read_only_query_store_returns_a_saved_projection(
    tmp_path_factory, monkeypatch
) -> None:
    import cytearc.datastore._operations.mapping as mapping_operations

    source = build_mapping_store(
        tmp_path_factory.mktemp("read_only_mapping") / "reference.zarr",
        with_references=True,
    )
    reference_store = DataStore(str(source.path), default_assay="RNA")
    assert source.reference is not None
    reference = reference_store.mapping.load_reference(source.reference)
    query_path = build_mapping_query(tmp_path_factory.mktemp("query") / "query.zarr")
    query = DataStore(str(query_path), default_assay="RNA")
    cells = query.snapshot_cell_selection()
    projection = query.mapping.run(reference, cells)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a saved projection must not be recomputed")

    monkeypatch.setattr(mapping_operations, "project_pca", forbidden)
    read_only = DataStore(str(query_path), default_assay="RNA", zarr_mode="r")
    assert read_only.mapping.run(reference, cells) == projection
    with pytest.raises(PermissionError, match="^run_mapping requires a DataStore"):
        read_only.mapping.run(reference, cells, save_k=2)

    # Built on a writable store, a reference is returned by a read-only one.
    shutil.rmtree(query_path)
    reopened = DataStore(str(source.path), default_assay="RNA", zarr_mode="r")
    assert reopened.mapping.build_reference(source.neighbors) == source.reference
