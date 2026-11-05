"""Resident memory of Harmony, WNN, SNN, and the ANN index, admitted before use.

Harmony holds several float64 copies of the coordinates and matrices of
assignments per cell, WNN holds every assay's neighbors and coordinates, the
SNN merge holds every graph with a float64 matrix of shared-neighbor
fractions per graph, and an hnswlib index holds every cell. Pure estimators in
the domain packages bound what each holds, and the datastore operations
compare them with the memory budget before a coordinate or graph is read or
an index is created or loaded, refusing a budget too small before it.
"""

import inspect
import re
import shutil
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc import DataStore
from cytearc.datastore._operations.graph import _ann_index_transfer_bytes
from cytearc.embeddings.harmony import (
    fit_harmony,
    harmony_cluster_count,
    harmony_peak_bytes,
)
from cytearc.neighbors.graph import snn_merge_peak_bytes
from cytearc.neighbors.index import (
    ann_index_file_bytes,
    ann_index_peak_bytes,
    ann_query_block_bytes,
)
from cytearc.neighbors.integration import wnn_peak_bytes
from cytearc.neighbors.stages import (
    AnnIndexStage,
    ChunkedCoordinateStream,
)
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.execution import ExecutionReport, execution_report_scope
from tests.storage_helpers import write_count_store

N_CELLS = 120


# Estimators


def _batches(n_cells: int, levels: int, columns: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"batch{column}": np.asarray(
                [f"b{(cell // (column + 1)) % levels}" for cell in range(n_cells)],
                dtype=object,
            )
            for column in range(columns)
        }
    )


def test_harmony_cluster_count_is_the_default_of_fit_harmony() -> None:
    # The cell count over 30, rounded half to even, kept between 1 and 100.
    assert [harmony_cluster_count(n) for n in (10, 45, 75, 105, 3_000, 9_000)] == [
        1,
        2,
        2,
        4,
        100,
        100,
    ]
    assert harmony_cluster_count(10, 7) == 7
    for nclust in (0, 11):
        with pytest.raises(ValueError, match="between one and the cell count"):
            harmony_cluster_count(10, nclust)
    values = np.random.default_rng(2).normal(size=(3, 75))
    result = fit_harmony(values, _batches(75, 2), max_iter_harmony=1)
    assert result.parameters["nClusters"] == harmony_cluster_count(75) == 2
    # The estimate assumes the update blocks of fit_harmony's default.
    assert (
        inspect.signature(harmony_peak_bytes).parameters["block_size"].default
        == inspect.signature(fit_harmony).parameters["block_size"].default
    )


def test_estimators_refuse_shapes_they_cannot_size() -> None:
    # hnswlib spreads each cell's upper-level links over m - 1 levels.
    with pytest.raises(ValueError, match="m must be at least 2"):
        ann_index_peak_bytes(N_CELLS, 5, 1)
    with pytest.raises(ValueError, match="two or more modalities"):
        wnn_peak_bytes(N_CELLS, [6], [5])
    for block_size, error in (
        (0, ValueError),
        (-0.5, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
    ):
        with pytest.raises(error, match="block_size must be"):
            harmony_peak_bytes(N_CELLS, 5, 4, [2], block_size=block_size)


# Admission


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(23)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=30), size=(N_CELLS, 30))
    rna[: N_CELLS // 2, :5] += 4
    rna[:, 0] += 1
    return {"RNA": rna, "ADT": rng.poisson(20.0, size=(N_CELLS, 4)) + 1}


def _open(zarr_loc: Path, **options: Any) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


def _write_template(zarr_loc: Path) -> dict[str, Any]:
    write_count_store(str(zarr_loc), _counts(), "uint16")
    store = _open(zarr_loc)
    store.cells.insert("batch", np.where(np.arange(N_CELLS) % 2, "a", "b"))
    cells = store.snapshot_cell_selection()
    refs: dict[str, Any] = {"cells": cells}
    for assay, dims in (("RNA", 5), ("ADT", 3)):
        normalized = store.features.normalize(
            cells, store.features.universe(from_assay=assay)
        )
        pca = store.reduction.pca(normalized, dims=dims)
        neighbors = store.graph.neighbors(store.graph.ann_index(pca), k=6)
        refs[assay] = {
            "pca": pca,
            "neighbors": neighbors,
            "graph": store.graph.connectivity(neighbors),
        }
    return refs


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, Any]]:
    zarr_loc = tmp_path_factory.mktemp("resident_memory") / "store.zarr"
    return zarr_loc, _write_template(zarr_loc)


@pytest.fixture
def memory_store(template, tmp_path) -> tuple[DataStore, dict[str, Any]]:
    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open(target), refs


def _record_coordinate_reads(monkeypatch) -> list[str]:
    reads: list[str] = []
    original = ChunkedCoordinateStream.iter_coordinate_blocks

    def recorded(self, message: str):
        reads.append(message)
        yield from original(self, message)

    monkeypatch.setattr(ChunkedCoordinateStream, "iter_coordinate_blocks", recorded)
    return reads


def _split_and_check_blocks(monkeypatch, parts: int = 3) -> list[bool]:
    """Stream each stored block as fresh parts and record held parts.

    Before each part after the first, it records whether the consumer still
    holds the part before it. A read plan reserves only the blocks in flight,
    so a consumer that still holds the previous part exceeds it.
    """
    held: list[bool] = []
    original = ChunkedCoordinateStream.iter_coordinate_blocks

    def checked(self, message: str):
        previous = None
        for block in original(self, message):
            for part in np.array_split(np.asarray(block), parts):
                if previous is not None:
                    held.append(previous() is not None)
                fresh = part.copy()
                previous = weakref.ref(fresh)
                yield fresh
                del fresh

    monkeypatch.setattr(ChunkedCoordinateStream, "iter_coordinate_blocks", checked)
    return held


def _required_bytes(message: str) -> int:
    found = re.search(r"needs about (\d+) bytes", message)
    assert found is not None, message
    return int(found.group(1))


def _reserved_bytes(reports: list[ExecutionReport]) -> list[int]:
    """Return the bytes each coordinate read reserved for what its caller holds."""
    return [
        report.plan.residentBytes
        for report in reports
        if report.unitKind == "countsRowBlock"
    ]


def test_wnn_row_norms_sum_bands_of_every_row(monkeypatch) -> None:
    from cytearc.neighbors import integration

    values = np.random.default_rng(4).normal(size=(50_000, 10)).astype(np.float32)
    expected = 1 / np.linalg.norm(values.astype(np.float64), axis=1)
    # Bands of 1,000 rows, 50 of them.
    monkeypatch.setattr(integration, "_NORM_BAND_VALUES", 10_000)
    np.testing.assert_allclose(
        integration._inverse_row_norms(values), expected, rtol=1e-12
    )


def test_harmony_admission_counts_the_batch_labels(memory_store, monkeypatch) -> None:
    store, refs = memory_store
    reads = _record_coordinate_reads(monkeypatch)
    options = {"harmony_params": {"n_clusters": 10}}
    peak = harmony_peak_bytes(N_CELLS, 5, 10, [2])
    # The fit's own estimate fits, even beside a pointer per label, but not
    # beside the labels the operation reads first and holds through the fit,
    # about 58 bytes per cell here.
    store.resources = ResourceBudget(peak + 16 * N_CELLS, 1)

    with pytest.raises(MemoryError, match="for the batch labels"):
        store.reduction.harmony(refs["RNA"]["pca"], ["batch"], **options)
    assert reads == []
    assert store.artifacts.list(kind="batch_correction", from_assay="RNA") == []

    # A budget that holds them reads the coordinates once and corrects them.
    store.resources = ResourceBudget(1 << 34, 1)
    corrected = store.reduction.harmony(refs["RNA"]["pca"], ["batch"], **options)
    assert store.artifacts.inspect(corrected).complete
    assert reads == ["Loading uncorrected latent dimensions"]


def test_snn_integration_raises_before_loading_graphs_over_budget(
    memory_store, monkeypatch
) -> None:
    store, refs = memory_store
    loads: list[str] = []
    monkeypatch.setattr(
        store, "_store_to_sparse", lambda location, _use_k: loads.append(location)
    )
    peak = snn_merge_peak_bytes(N_CELLS, 6, 2)
    store.resources = ResourceBudget(peak - 1, 1)

    with pytest.raises(MemoryError) as caught:
        store.integration.modalities(
            [refs["RNA"]["graph"], refs["ADT"]["graph"]], method="snn"
        )

    message = str(caught.value)
    assert _required_bytes(message) == peak
    assert f"2 graphs over {N_CELLS} cells with 6 neighbors" in message
    assert f"operation limit is {peak - 1} bytes" in message
    assert loads == []
    assert store.artifacts.list(scope="datastore", kind="integrated_graph") == []


def test_snn_integration_merges_on_the_store_threads(
    template, tmp_path, monkeypatch
) -> None:
    import numba

    from cytearc.neighbors import graph as neighbor_graph

    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    store = DataStore(
        str(target), default_assay="RNA", min_features_per_cell=0, nthreads=2
    )
    merge = neighbor_graph._merge_ranked_rows
    threads: list[int] = []

    def recorded(*args: Any) -> None:
        threads.append(numba.get_num_threads())
        merge(*args)

    monkeypatch.setattr(neighbor_graph, "_merge_ranked_rows", recorded)
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        integrated = store.integration.modalities(
            [refs["RNA"]["graph"], refs["ADT"]["graph"]], method="snn"
        )
        after = numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)

    assert store.artifacts.inspect(integrated).complete
    assert threads == [min(2, numba.config.NUMBA_NUM_THREADS)]
    assert after == 1


def test_wnn_integration_raises_before_loading_coordinates_over_budget(
    memory_store, monkeypatch
) -> None:
    store, refs = memory_store
    reads = _record_coordinate_reads(monkeypatch)
    peak = wnn_peak_bytes(N_CELLS, (6, 6), (5, 3))
    store.resources = ResourceBudget(peak - 1, 1)

    with pytest.raises(MemoryError) as caught:
        store.integration.modalities(
            [refs["RNA"]["neighbors"], refs["ADT"]["neighbors"]]
        )

    message = str(caught.value)
    assert _required_bytes(message) == peak
    assert (
        f"2 assays over {N_CELLS} cells with [6, 6] neighbors and [5, 3] dimensions"
        in message
    )
    assert f"operation limit is {peak - 1} bytes" in message
    assert reads == []
    assert store.artifacts.list(scope="datastore", kind="integrated_graph") == []

    # Within the budget it loads each assay's coordinates once, and holds no
    # block while it reads the next.
    store.resources = ResourceBudget(peak, 1)
    held = _split_and_check_blocks(monkeypatch)
    with execution_report_scope() as reports:
        integrated = store.integration.modalities(
            [refs["RNA"]["neighbors"], refs["ADT"]["neighbors"]]
        )
    assert store.artifacts.inspect(integrated).complete
    assert reads == ["Loading RNA coordinates", "Loading ADT coordinates"]
    assert held and not any(held)
    # Each read reserves the uint32 neighbors and float32 coordinates of its
    # assay beside those of the assays loaded before it.
    rna = N_CELLS * (6 * 4 + 5 * 4)
    assert _reserved_bytes(reports) == [rna, rna + N_CELLS * (6 * 4 + 3 * 4)]


def _record_index_creations(monkeypatch) -> list[int]:
    created: list[int] = []
    create = AnnIndexStage.create

    def recorded(**options):
        created.append(options["n_cells"])
        return create(**options)

    monkeypatch.setattr(AnnIndexStage, "create", staticmethod(recorded))
    return created


def test_build_ann_index_raises_before_creating_the_index_over_budget(
    memory_store, monkeypatch
) -> None:
    store, refs = memory_store
    pca = refs["RNA"]["pca"]
    created = _record_index_creations(monkeypatch)
    reads = _record_coordinate_reads(monkeypatch)
    before = store.artifacts.list(kind="ann_index", from_assay="RNA")
    index_bytes = ann_index_peak_bytes(N_CELLS, 5, 8, nthreads=1)
    store.resources = ResourceBudget(index_bytes, 1)

    with pytest.raises(MemoryError) as caught:
        store.graph.ann_index(pca, ann_m=8)

    message = str(caught.value)
    required = _required_bytes(message)
    # The index with one block of coordinates, or with the copy that saves
    # it through a temporary file.
    saving = _ann_index_transfer_bytes(ann_index_file_bytes(N_CELLS, 5, 8))
    assert required >= index_bytes + saving
    for size in (
        f"ANN index of {N_CELLS} cells with 5 dimensions",
        f"{index_bytes} of them for the hnswlib index with ann_m=8",
        f"operation limit is {index_bytes} bytes",
    ):
        assert size in message
    assert created == [] and reads == []
    assert store.artifacts.list(kind="ann_index", from_assay="RNA") == before

    # The admitted bytes are enough: the reads plan their blocks beside the
    # index, and the index is saved.
    store.resources = ResourceBudget(required, 1)
    with execution_report_scope() as reports:
        built = store.graph.ann_index(pca, ann_m=8)
    assert store.artifacts.inspect(built).complete
    assert created == [N_CELLS]
    assert reads == ["Fitting ANN"]
    assert _reserved_bytes(reports) == [index_bytes]


def test_query_neighbors_raises_before_loading_the_index_over_budget(
    memory_store, monkeypatch
) -> None:
    import cytearc.datastore._operations.graph as graph_operations

    store, refs = memory_store
    ann_index = store.artifacts.inspect(refs["RNA"]["neighbors"]).input_ref("ann_index")
    loads: list[str] = []
    load = graph_operations.load_ann_index

    def recorded(group, *args, **kwargs):
        loads.append(group.path)
        return load(group, *args, **kwargs)

    monkeypatch.setattr(graph_operations, "load_ann_index", recorded)
    reads = _record_coordinate_reads(monkeypatch)
    # build_ann_index's default ann_m.
    index_bytes = ann_index_peak_bytes(N_CELLS, 5, 24, nthreads=1)
    store.resources = ResourceBudget(index_bytes, 1)

    with pytest.raises(MemoryError) as caught:
        store.graph.neighbors(ann_index, k=7)

    message = str(caught.value)
    required = _required_bytes(message)
    # The index and the neighbor results, beside one block with its query
    # results or the copy that loads the index through a temporary file.
    payload = int(artifact_group(store.zw, ann_index)["ann_idx_bytes"].shape[0])
    assert required >= index_bytes + N_CELLS * 7 * 8 + max(
        ann_query_block_bytes(N_CELLS, 7, ef=200, nthreads=1),
        _ann_index_transfer_bytes(payload),
    )
    for size in (
        f"Querying 7 neighbors of {N_CELLS} cells with 5 dimensions",
        f"{index_bytes} of them for the hnswlib index with ann_m=24",
        f"operation limit is {index_bytes} bytes",
    ):
        assert size in message
    assert loads == [] and reads == []
    assert len(store.artifacts.list(kind="neighbors", from_assay="RNA")) == 1

    # Within the budget it holds no block while it reads the next.
    store.resources = ResourceBudget(required, 1)
    held = _split_and_check_blocks(monkeypatch)
    with execution_report_scope() as reports:
        neighbors = store.graph.neighbors(ann_index, k=7)
    assert store.artifacts.inspect(neighbors).complete
    assert len(loads) == 1
    assert reads == ["Identifying neighbors"]
    assert held and not any(held)
    scratch = ann_query_block_bytes(N_CELLS, 7, ef=200, nthreads=1)
    assert _reserved_bytes(reports) == [index_bytes + N_CELLS * 7 * 8 + scratch]


# Identity checks of the cell IDs
