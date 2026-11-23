"""Operations refuse to read an assay over cells that it did not measure.

The fixture merges a source with RNA, ATAC, HTO, and ADT assays and a source
with ADT only. The merged RNA, ATAC, and HTO assays measured the cells of the
first source only: their membership columns ``<assay>_I`` are False for the
cells of the second source, which hold zero counts of those assays. Every
operation that reads an assay's values over cells refuses unmeasured cells
with ``UnmeasuredCellsError`` before it writes or reuses anything, on a
writable and on a read-only store, and runs over the cells that
``select_measured_cells`` keeps. Display reads show unmeasured cells as
missing values instead of zero expression.
"""

import asyncio
import pickle
import shutil
import stat
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import LocalStore, WrapperStore

from cytearc import DataStore
from cytearc.datastore._pipeline.recipes.adt import resolve_adt
from cytearc.datastore._pipeline.recipes.atac import resolve_atac
from cytearc.datastore._pipeline.recipes.rna import resolve_rna
from cytearc.datastore.base_datastore import BaseDataStore
from cytearc.features.values import (
    fetch_normalized_feature_matrix,
    iter_normalized_feature_blocks,
    resolve_feature,
)
from cytearc.mapping.reference import MappingReference
from cytearc.merge import DataStoreMerge
from cytearc.metadata import MetaData
from cytearc.metadata import membership as membership_module
from cytearc.metadata.membership import (
    UnmeasuredCellsError,
    cell_sources,
    cell_sources_column,
    count_unmeasured_cells_with_counts,
    features_measured_by_cells,
    measured_features_by_source,
    measured_rows,
    measured_selection,
    membership_attributes,
    require_measured_cells,
    require_measured_features,
)
from cytearc.metadata.selection import CellField, FeatureRef, NormalizationSpec
from cytearc.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_array,
    list_artifacts,
)
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.arrays import create_metadata_column
from cytearc.storage.categorical import CategoricalColumn
from cytearc.features.variability import fit_lowess
from cytearc.storage.feature_selection import read_feature_selection_indices
from cytearc.storage.selections import read_stored_selection_indices
from cytearc.writers._store import write_membership_column
from tests.fixtures_datastore import build_neighborhood_graph
from tests.storage_helpers import by_cell_id, write_count_store
from tests.test_hto import planted_hto_counts
from tests.test_mapping_reference import mapping_counts

_N_FULL = 64
_N_ADT = 16
_N_CELLS = _N_FULL + _N_ADT
# The merged assays that measured only the cells of the first source.
_PARTIAL = ("RNA", "ATAC", "HTO")


@dataclass(frozen=True)
class _MeasuredStore:
    """A merged store whose RNA, ATAC, and HTO assays measured some cells.

    ``refs`` holds the artifacts that the entry points read: ``all``, every
    cell; ``measured_<assay>``, the cells that an assay measured; label,
    graph, pseudotime, and diffusion artifacts over either; and ``legacy_*``
    artifacts deliberately computed over every cell, including RNA's
    unmeasured cells, to exercise validation and reads.
    """

    path: Path
    reference: MappingReference
    refs: dict[str, ArtifactRef]

    def open(self, zarr_mode: str = "r+") -> DataStore:
        return DataStore(
            str(self.path),
            default_assay="RNA",
            min_features_per_cell=-1,
            nthreads=1,
            zarr_mode=zarr_mode,
        )


def _input(store: DataStore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return store.artifacts.inspect(ref).input_ref(name)


def _write_sources(base: Path) -> tuple[DataStore, DataStore]:
    rng = np.random.default_rng(5)
    programs = np.arange(_N_FULL) % 4
    peaks = rng.binomial(1, 0.15, size=(_N_FULL, 40))
    for program in range(4):
        peaks[programs == program, 5 * program : 5 * program + 5] = 1
    hashtags, _truth = planted_hto_counts(n_singlets=18, n_doublets=5, n_negatives=5)
    write_count_store(
        str(base / "full.zarr"),
        {
            "RNA": mapping_counts(),
            "ATAC": peaks,
            "HTO": hashtags.to_numpy(),
            "ADT": rng.integers(1, 20, size=(_N_FULL, 3)),
        },
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"),
        {"ADT": rng.integers(1, 20, size=(_N_ADT, 3))},
        "uint32",
    )
    full = DataStore(
        str(base / "full.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    adt = DataStore(
        str(base / "adt.zarr"),
        default_assay="ADT",
        min_features_per_cell=0,
        nthreads=1,
    )
    return full, adt


def _mapping_reference(full: DataStore) -> MappingReference:
    """Build a reference from the RNA source, a store apart from the query."""
    cells = full.snapshot_cell_selection("I")
    features = full.features.snapshot(from_assay="RNA", feature_indexes=list(range(26)))
    reduction = full.reduction.pca(
        full.features.normalize(cells, features), dims=4, local_cache=False
    )
    neighbors = full.graph.neighbors(
        full.graph.ann_index(reduction), coordinates=reduction, k=5
    )
    return full.mapping.load_reference(full.mapping.build_reference(neighbors))


def _graph(
    store: DataStore, assay: str, cells: ArtifactRef, features: ArtifactRef
) -> ArtifactRef:
    return build_neighborhood_graph(
        store,
        from_assay=assay,
        cell_selection=cells,
        features=features,
        dims=2 if assay == "ADT" else 3,
        k=5,
        local_cache=False,
    )


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> _MeasuredStore:
    base = tmp_path_factory.mktemp("measured_cells")
    full, adt = _write_sources(base)
    path = base / "merged.zarr"
    DataStoreMerge([full, adt], str(path), ["full", "adt"], nthreads=1).dump()
    reference = _mapping_reference(full)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
    )
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    assert int(membership.sum()) == _N_FULL
    store.cells.insert(
        "program",
        by_cell_id(
            store.cells, np.array([f"p{index % 4}" for index in range(_N_CELLS)])
        ),
    )
    store.cells.insert("adt_cells", by_cell_id(store.cells, ~membership))
    # A live layout: x is the cell's row.
    store.cells.insert(
        "xy1", by_cell_id(store.cells, np.arange(_N_CELLS, dtype=np.float64))
    )
    store.cells.insert(
        "xy2", by_cell_id(store.cells, (np.arange(_N_CELLS) % 7).astype(np.float64))
    )

    refs: dict[str, ArtifactRef] = {"all": store.snapshot_cell_selection("I")}
    for assay in _PARTIAL:
        refs[f"measured_{assay}"] = store.select_measured_cells(
            assay, cell_selection=refs["all"]
        )
    refs["adt_cells"] = store.snapshot_cell_selection("adt_cells")
    refs["rna_features"] = store.features.universe(from_assay="RNA")
    refs["rna_some"] = store.features.snapshot(
        from_assay="RNA", feature_indexes=[0, 1, 2]
    )
    refs["labels_all"] = store.snapshot_cluster_labels(
        "program", cell_selection=refs["all"]
    )
    refs["labels_measured"] = store.snapshot_cluster_labels(
        "program", cell_selection=refs["measured_RNA"]
    )
    adt_features = store.features.universe(from_assay="ADT")
    # ADT measured every cell, so its graph over all of them is valid.
    adt_all = _graph(store, "ADT", refs["all"], adt_features)
    refs["adt_neighbors_all"] = _input(store, adt_all, "neighbors")
    refs["pseudotime_all"] = store.trajectory.pseudotime(
        adt_all, source_sink=refs["labels_all"], sources=["p0"], sinks=["p3"]
    )
    adt_measured = _graph(store, "ADT", refs["measured_RNA"], adt_features)
    refs["adt_neighbors_measured"] = _input(store, adt_measured, "neighbors")
    rna_measured = _graph(store, "RNA", refs["measured_RNA"], refs["rna_features"])
    refs["graph_measured"] = rna_measured
    refs["rna_neighbors_measured"] = _input(store, rna_measured, "neighbors")
    refs["clusters_measured"] = store.clusters.leiden(rna_measured)
    refs["pseudotime_measured"] = store.trajectory.pseudotime(
        rna_measured,
        source_sink=refs["labels_measured"],
        sources=["p0"],
        sinks=["p3"],
    )
    refs["diffusion_measured"] = store.imputation.diffusion(rna_measured)
    with pytest.MonkeyPatch.context() as patch:
        # Bypass validation to build artifacts over cells that RNA did not measure.
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        legacy = _graph(store, "RNA", refs["all"], refs["rna_features"])
    refs["legacy_graph"] = legacy
    refs["legacy_neighbors"] = _input(store, legacy, "neighbors")
    refs["legacy_normalized"] = _input(
        store, _input(store, refs["legacy_neighbors"], "coordinates"), "normalized"
    )
    refs["legacy_clusters"] = store.clusters.leiden(legacy)
    refs["legacy_diffusion"] = store.imputation.diffusion(legacy)
    return _MeasuredStore(path=path, reference=reference, refs=refs)


def _tree(path: Path) -> dict[str, tuple[int, int]]:
    """Return the size and modification time of every file of a local store."""
    tree = {}
    for file in path.rglob("*"):
        info = file.stat()
        if stat.S_ISREG(info.st_mode):
            tree[str(file.relative_to(path))] = (info.st_size, info.st_mtime_ns)
    return tree


_NETWORK = pd.DataFrame(
    {"source": ["s"] * 8, "target": [f"RNA{index}" for index in range(8)]}
)
_CELL_CYCLE: dict[str, Any] = {
    "s_genes": ["RNA0", "RNA1", "RNA2"],
    "g2m_genes": ["RNA5", "RNA6", "RNA7"],
    "n_bins": 5,
}
_HVG: dict[str, Any] = {"top_n": 5, "min_cells": 1, "max_cells": np.inf}
_AGGREGATION: dict[str, Any] = {"n_clusters": 2, "window_size": 10, "chunk_size": 5}
_PIPELINE: dict[str, Any] = {
    "filtering": False,
    "cell_cycle": False,
    "umap": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    # Infinity is no setting, so a cutoff above every cell count keeps the
    # genes that these small stores detect in nearly every cell.
    "hvg": {"top_n": 10, "min_cells": 1, "max_cells": _N_CELLS + 1},
    "pca": {"dims": 3},
    "neighbors": {"k": 5},
    "leiden": {"partitions": [0.5, 1.0]},
}
# The ATAC and ADT recipe settings of these tests, with every keyword of
# their resolvers but the cells.
_ATAC: dict[str, Any] = {
    "assay": None,
    "label": None,
    "filtering": None,
    "prevalent_peaks": {"top_n": 20},
    "normalization": None,
    "lsi": {"dims": 3},
    "harmony": None,
    "ann_index": None,
    "neighbors": {"k": 5},
    "connectivity": None,
    "embedding_initialization": None,
    "umap": False,
    "localmap": False,
    "tsne": False,
    "leiden": False,
    "paris": False,
    "membership_strength": False,
    "doublets": False,
    "markers": False,
    "snapshot_columns": (),
    "species": None,
}
_ADT: dict[str, Any] = {
    "assay": None,
    "label": None,
    "filtering": None,
    "panel": None,
    "normalization": None,
    "pca": None,
    "harmony": None,
    "ann_index": None,
    "neighbors": {"k": 5},
    "connectivity": None,
    "embedding_initialization": None,
    "umap": False,
    "localmap": False,
    "tsne": False,
    "leiden": False,
    "paris": False,
    "membership_strength": False,
    "markers": False,
    "snapshot_columns": (),
    "species": None,
}


def _recipe(store: DataStore, cells: Any, **settings: Any) -> Any:
    """Resolve the RNA recipe over ``cells`` without running it."""
    return resolve_rna(
        store,
        assay=None,
        label=None,
        cells=cells,
        cell_calling=False,
        hashing=False,
        normalization=None,
        harmony=None,
        ann_index=None,
        connectivity=None,
        embedding_initialization=None,
        localmap=False,
        tsne=False,
        membership_strength=False,
        snapshot_columns=(),
        species=None,
        **{**_PIPELINE, **settings},
    )


def _atac_recipe(store: DataStore, cells: Any) -> Any:
    """Resolve the ATAC recipe over ``cells`` without running it."""
    return resolve_atac(store, cells=cells, **_ATAC)


def _narrowed_labels(store: DataStore, labels: ArtifactRef) -> ArtifactRef:
    """Narrow labels to the cells that RNA measured, as the refusal says."""
    cells = _input(store, labels, "cell_selection")
    return store.snapshot_cluster_labels(
        labels,
        cell_selection=store.select_measured_cells("RNA", cell_selection=cells),
    )


type _Call = Callable[[DataStore, _MeasuredStore], Any]


@dataclass(frozen=True)
class _Entry:
    """An entry point, a request over unmeasured cells, and a measured one."""

    operation: str
    assay: str
    refuse: _Call
    accept: _Call
    remedy: str
    # Whether the request reads every cell of the store.
    every_cell: bool = True
    # The test ID, when the operation has several entries.
    name: str | None = None


_SELECTION = "select_measured_cells("
_LABELS = "snapshot_cluster_labels(labels, cell_selection=select_measured_cells("
_SELECTION_KEYWORD = "Pass `cell_selection=select_measured_cells('RNA', "
_GRAPH = "Build the graph over a selection of measured cells"
_CELL_KEY = (
    "Pass as cell_key a boolean cell column that is True only for the cells of "
    "'I' that it measured, such as 'RNA_measured' after "
    "`ds.cells.insert('RNA_measured', pandas.Series(ds.cells.fetch_all('I') & "
    "ds.cells.fetch_all('RNA_I'), index=ds.cells.fetch_all('ids')))`."
)


def _cells_remedy(assay: str) -> str:
    """Return the remedy of a recipe whose ``cells`` hold unmeasured cells of I."""
    return (
        f"Pass as cells the selection from `select_measured_cells('{assay}', "
        "cell_selection=...)`, or a boolean cell column that is True only for the "
        f"cells of 'I' that it measured, such as '{assay}_measured' after "
        f"`ds.cells.insert('{assay}_measured', pandas.Series("
        f"ds.cells.fetch_all('I') & ds.cells.fetch_all('{assay}_I'), "
        "index=ds.cells.fetch_all('ids')))`."
    )


def _selection_cells_remedy(assay: str) -> str:
    """Return the remedy of a recipe given unmeasured cells as a selection or run."""
    return (
        f"Pass as cells the selection from `select_measured_cells('{assay}', "
        "cell_selection=...)`. When the cells are an earlier run's, run that "
        "earlier run on cells that this assay measured too, so that the runs "
        "analyze the same cells."
    )


_ENTRIES = (
    _Entry(
        "run_normalization",
        "RNA",
        lambda s, m: s.features.normalize(m.refs["all"], m.refs["rna_features"]),
        lambda s, m: s.features.normalize(
            m.refs["measured_RNA"], m.refs["rna_features"]
        ),
        _SELECTION,
    ),
    _Entry(
        "integrate_assays",
        "RNA",
        lambda s, m: s.integration.modalities(
            [m.refs["legacy_neighbors"], m.refs["adt_neighbors_all"]], method="wnn"
        ),
        lambda s, m: s.integration.modalities(
            [m.refs["rna_neighbors_measured"], m.refs["adt_neighbors_measured"]],
            method="wnn",
        ),
        _GRAPH,
    ),
    _Entry(
        "select_hvgs",
        "RNA",
        lambda s, m: s.features.hvgs(m.refs["all"], show_plot=False, **_HVG),
        lambda s, m: s.features.hvgs(m.refs["measured_RNA"], show_plot=False, **_HVG),
        _SELECTION,
    ),
    _Entry(
        "select_detected_features",
        "RNA",
        lambda s, m: s.features.detected(m.refs["all"], min_cells=1),
        lambda s, m: s.features.detected(m.refs["measured_RNA"], min_cells=1),
        _SELECTION,
    ),
    _Entry(
        "run_waggr",
        "RNA",
        lambda s, m: s.scores.waggr(
            _NETWORK, m.refs["all"], features=m.refs["rna_features"], tmin=3
        ),
        lambda s, m: s.scores.waggr(
            _NETWORK, m.refs["measured_RNA"], features=m.refs["rna_features"], tmin=3
        ),
        _SELECTION,
    ),
    _Entry(
        "run_aucell",
        "RNA",
        lambda s, m: s.scores.aucell(
            _NETWORK, m.refs["all"], features=m.refs["rna_features"], tmin=3
        ),
        lambda s, m: s.scores.aucell(
            _NETWORK, m.refs["measured_RNA"], features=m.refs["rna_features"], tmin=3
        ),
        _SELECTION,
    ),
    _Entry(
        "run_marker_search",
        "RNA",
        lambda s, m: s.markers.search(
            m.refs["labels_all"], features=m.refs["rna_features"]
        ),
        lambda s, m: s.markers.search(
            _narrowed_labels(s, m.refs["labels_all"]), features=m.refs["rna_features"]
        ),
        _LABELS,
    ),
    # Label consumers accept labels of measured cells, and also every label
    # with the remedy's measured cell_selection.
    _Entry(
        "make_bulk",
        "RNA",
        lambda s, m: s.compare.compute_bulk(m.refs["labels_all"], from_assay="RNA"),
        lambda s, m: (
            s.compare.compute_bulk(m.refs["labels_measured"], from_assay="RNA"),
            s.compare.compute_bulk(
                m.refs["labels_all"],
                from_assay="RNA",
                cell_selection=m.refs["measured_RNA"],
            ),
        ),
        _SELECTION_KEYWORD,
    ),
    _Entry(
        "run_statistical_testing",
        "RNA",
        lambda s, m: s.compare.test(["RNA0"], m.refs["labels_all"]),
        lambda s, m: (
            s.compare.test(["RNA0"], m.refs["labels_measured"]),
            s.compare.test(
                ["RNA0"], m.refs["labels_all"], cell_selection=m.refs["measured_RNA"]
            ),
            # Only the cells that the design keeps are tested and checked.
            s.compare.test(["RNA0"], m.refs["labels_all"], subset_by="RNA_I"),
        ),
        _SELECTION_KEYWORD,
    ),
    _Entry(
        "select_prevalent_peaks",
        "ATAC",
        lambda s, m: s.features.prevalent_peaks(
            m.refs["all"], from_assay="ATAC", top_n=5
        ),
        lambda s, m: s.features.prevalent_peaks(
            m.refs["measured_ATAC"], from_assay="ATAC", top_n=5
        ),
        _SELECTION,
    ),
    _Entry(
        "run_cell_cycle_scoring",
        "RNA",
        lambda s, m: s.scores.cell_cycle(m.refs["all"], **_CELL_CYCLE),
        lambda s, m: s.scores.cell_cycle(m.refs["measured_RNA"], **_CELL_CYCLE),
        _SELECTION,
    ),
    _Entry(
        "run_feature_percentage",
        "RNA",
        lambda s, m: s.qc.feature_percentage(m.refs["all"], m.refs["rna_some"]),
        lambda s, m: s.qc.feature_percentage(
            m.refs["measured_RNA"], m.refs["rna_some"]
        ),
        _SELECTION,
    ),
    _Entry(
        "run_hto_demultiplexing",
        "HTO",
        lambda s, m: s.qc.hto_demultiplexing(m.refs["all"]),
        lambda s, m: s.qc.hto_demultiplexing(m.refs["measured_HTO"]),
        _SELECTION,
    ),
    _Entry(
        "run_doublet_detection",
        "RNA",
        lambda s, m: s.qc.doublets(m.refs["legacy_clusters"], m.refs["legacy_graph"]),
        lambda s, m: s.qc.doublets(
            m.refs["clusters_measured"], m.refs["graph_measured"]
        ),
        _GRAPH,
    ),
    _Entry(
        "run_pseudotime_marker_search",
        "RNA",
        lambda s, m: s.trajectory.markers(
            m.refs["pseudotime_all"], features=m.refs["rna_features"]
        ),
        lambda s, m: s.trajectory.markers(
            m.refs["pseudotime_measured"], features=m.refs["rna_features"]
        ),
        _GRAPH,
        # A pseudotime reads its valid cells, which can be fewer.
        every_cell=False,
    ),
    _Entry(
        "run_pseudotime_aggregation",
        "RNA",
        lambda s, m: s.trajectory.aggregation(
            m.refs["pseudotime_all"], features=m.refs["rna_features"], **_AGGREGATION
        ),
        lambda s, m: s.trajectory.aggregation(
            m.refs["pseudotime_measured"],
            features=m.refs["rna_features"],
            **_AGGREGATION,
        ),
        _GRAPH,
        every_cell=False,
    ),
    _Entry(
        "get_imputed",
        "RNA",
        lambda s, m: s.imputation.compute_imputed("RNA0", m.refs["legacy_diffusion"]),
        lambda s, m: s.imputation.compute_imputed("RNA0", m.refs["diffusion_measured"]),
        _GRAPH,
    ),
    _Entry(
        "run_mapping",
        "RNA",
        lambda s, m: s.mapping.run(m.reference, m.refs["all"]),
        lambda s, m: s.mapping.run(m.reference, m.refs["measured_RNA"]),
        _SELECTION,
    ),
    _Entry(
        "to_anndata",
        "RNA",
        lambda s, m: s.to_anndata(from_assay="RNA", matrix="normed"),
        lambda s, m: s.to_anndata(from_assay="RNA", cell_key="RNA_I", matrix="normed"),
        _CELL_KEY,
    ),
    _Entry(
        "pipeline.rna",
        "RNA",
        lambda s, m: _recipe(s, "I"),
        lambda s, m: _recipe(s, "RNA_I"),
        _cells_remedy("RNA"),
    ),
    # A recipe refuses a selection of cells as it refuses a cell column.
    _Entry(
        "pipeline.rna",
        "RNA",
        lambda s, m: _recipe(s, m.refs["all"]),
        lambda s, m: _recipe(s, m.refs["measured_RNA"]),
        _selection_cells_remedy("RNA"),
        name="pipeline.rna-selection",
    ),
    _Entry(
        "pipeline.atac",
        "ATAC",
        lambda s, m: _atac_recipe(s, "I"),
        lambda s, m: _atac_recipe(s, m.refs["measured_ATAC"]),
        _cells_remedy("ATAC"),
    ),
)
_IDS = [entry.name or entry.operation for entry in _ENTRIES]


def _assert_refusal(error: UnmeasuredCellsError, entry: _Entry) -> None:
    assert isinstance(error, ValueError)
    assert error.operation == entry.operation
    assert error.assay == entry.assay
    assert error.column == f"{entry.assay}_I"
    if entry.every_cell:
        assert (error.unmeasured, error.selected) == (_N_ADT, _N_CELLS)
    else:
        assert 0 < error.unmeasured <= _N_ADT < error.selected <= _N_CELLS
    message = str(error)
    assert message.startswith(
        f"`{entry.operation}` reads assay {entry.assay!r} over {error.unmeasured} "
        f"of {error.selected} selected cells that it did not measure: cell column "
        f"'{entry.assay}_I' is False for them. "
    )
    assert entry.remedy in message


@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_an_unmeasured_cell_is_refused_before_anything_is_written(
    measured, entry
) -> None:
    # A read-only store refuses as a writable one does, not for being read-only.
    stores = [measured.open(), measured.open(zarr_mode="r")]
    before = _tree(measured.path)

    for store in stores:
        with pytest.raises(UnmeasuredCellsError) as raised:
            entry.refuse(store, measured)
        _assert_refusal(raised.value, entry)

    assert _tree(measured.path) == before


# Cell-level tests warn that they are descriptive.
@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_measured_cells_are_accepted(measured, entry, monkeypatch) -> None:
    # test_hto covers the demultiplexing; here only the check matters.
    monkeypatch.setattr(
        "cytearc.datastore._operations.quality_control._hto_calls",
        lambda frame, **_kwargs: (
            frame.idxmax(axis=1),
            dict.fromkeys(frame.columns, 0),
            [],
        ),
    )
    store = measured.open()

    entry.accept(store, measured)


def test_select_measured_cells_snapshots_the_measured_cells(measured) -> None:
    store = measured.open()
    every = measured.refs["all"]
    selection = measured.refs["measured_RNA"]
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)

    np.testing.assert_array_equal(
        np.asarray(store.artifacts.load(selection)["values"][:]), membership
    )
    assert (selection.scope, selection.assay, selection.kind) == (
        "datastore",
        None,
        "cell_selection",
    )
    status = store.artifacts.inspect(selection)
    assert status.operation == "select_measured_cells"
    assert status.parameters == {"assay": "RNA"}
    assert status.input_ref("prior_cell_selection") == every
    assert status.inputs["membership_fingerprint"] == fingerprint_array(membership)
    assert set(status.inputs) == {
        "prior_cell_selection",
        "membership_fingerprint",
        "ordered_row_ids_fingerprint",
        "values_fingerprint",
    }
    # The same request reuses the artifact, also from a read-only store, and
    # an omitted selection snapshots I, which here holds every cell.
    assert store.select_measured_cells("RNA", cell_selection=every) == selection
    assert store.select_measured_cells("RNA") == selection
    read_only = measured.open(zarr_mode="r")
    assert read_only.select_measured_cells("RNA", cell_selection=every) == selection
    # ATAC and HTO measured the same cells; each records its own membership.
    for assay in ("ATAC", "HTO"):
        other = measured.refs[f"measured_{assay}"]
        assert other != selection
        assert store.artifacts.inspect(other).parameters == {"assay": assay}
        np.testing.assert_array_equal(
            np.asarray(store.artifacts.load(other)["values"][:]), membership
        )
    fresh = store.select_measured_cells(
        "RNA", cell_selection=every, invalidate_cache=True
    )
    assert fresh != selection
    np.testing.assert_array_equal(
        np.asarray(store.artifacts.load(fresh)["values"][:]), membership
    )


def test_select_measured_cells_returns_a_measured_selection_unchanged(
    measured,
) -> None:
    store = measured.open()
    every = measured.refs["all"]
    selection = measured.refs["measured_RNA"]
    before = _tree(measured.path)

    # ADT measured every cell, and RNA every cell of its own selection.
    assert store.select_measured_cells("ADT", cell_selection=every) == every
    assert store.select_measured_cells("RNA", cell_selection=selection) == selection
    assert _tree(measured.path) == before


def test_select_measured_cells_raises_when_no_cell_remains(measured) -> None:
    store = measured.open()

    with pytest.raises(
        ValueError,
        match=(
            r"^Assay 'RNA' measured none of the 16 selected cells: cell column "
            r"'RNA_I' is False for all of them\.$"
        ),
    ):
        store.select_measured_cells("RNA", cell_selection=measured.refs["adt_cells"])
    with pytest.raises(ValueError, match="Assay 'GENES' not found"):
        store.select_measured_cells("GENES")
    with pytest.raises(TypeError, match="assay must be the name of an assay"):
        store.select_measured_cells("")
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.select_measured_cells("RNA", cell_selection="RNA_I")  # type: ignore[arg-type]


def _rna_source(path: Path, seed: int, *, membership: bool = False) -> DataStore:
    counts = np.random.default_rng(seed).poisson(4.0, size=(24, 12)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint32")
    if membership:
        # Every cell is a member, as a merge of RNA sources records it.
        cells = zarr.open_group(str(path), mode="r+")["cellData"]
        write_membership_column(cells, "RNA", np.ones(len(counts), dtype=bool))
    return DataStore(str(path), default_assay="RNA", min_features_per_cell=-1)


def test_fully_measured_stores_keep_their_identities(tmp_path) -> None:
    """Membership that marks every cell measured changes no result.

    A store whose ``RNA_I`` is True for every cell, and a store without
    membership columns, pass the check, and ``select_measured_cells`` returns
    the selection that it narrows, so the results of a workflow that narrows
    its cells first are the results of one that does not, with the same
    identities. Results record no membership input, as before.
    """
    member = _rna_source(tmp_path / "member.zarr", 1, membership=True)
    plain = _rna_source(tmp_path / "plain.zarr", 2)
    assert np.asarray(member.cells.fetch_all("RNA_I"), dtype=bool).all()
    assert "RNA_I" not in plain.cells.columns

    for store in (member, plain):
        cells = store.snapshot_cell_selection("I")
        features = store.features.universe(from_assay="RNA")
        normalized = store.features.normalize(cells, features)
        narrowed = store.select_measured_cells("RNA", cell_selection=cells)
        assert narrowed == cells
        assert store.select_measured_cells("RNA") == cells
        assert store.features.normalize(narrowed, features) == normalized
        status = store.artifacts.inspect(normalized)
        assert set(status.inputs) == {
            "cell_selection",
            "dataset_fingerprint",
            "feature_selection",
        }
        values = fetch_normalized_feature_matrix(
            store,
            [resolve_feature(store, "RNA0")],
            np.arange(store.cells.N),
        )
        assert np.isfinite(values).all()
        assert not list_artifacts(
            store.zw, scope="datastore", operation="select_measured_cells"
        )


@pytest.mark.slow
def test_the_pipeline_runs_over_the_cells_that_its_remedy_names(measured) -> None:
    store = measured.open()
    # The column that the remedy writes holds the cells of I that RNA measured.
    # Run the insert that the remedy names, as a user would copy it.
    remedy = _CELL_KEY.rsplit("`", 2)[1]
    assert remedy.startswith("ds.cells.insert('RNA_measured', pandas.Series(")
    eval(remedy, {"ds": store, "pandas": pd})
    run = store.pipeline.rna(label="measured", cells="RNA_measured", **_PIPELINE)

    assert run.status == "completed"
    np.testing.assert_array_equal(
        np.asarray(store.artifacts.load(run["input_cell_selection"])["values"][:]),
        np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
        & np.asarray(store.cells.fetch_all("I"), dtype=bool),
    )
    # The remedy's selection, or a completed run over measured cells, gives
    # the next run its input cells as they are.
    for cells, expected in (
        (measured.refs["measured_RNA"], measured.refs["measured_RNA"]),
        (run, run["analysis_cell_selection"]),
    ):
        again = store.pipeline.rna(cells=cells, **_PIPELINE)
        assert again.status == "completed"
        assert again["input_cell_selection"] == expected


@pytest.mark.slow
def test_results_over_unmeasured_cells_stay_readable_and_their_runs_reopen(
    measured,
) -> None:
    """Results computed over unmeasured cells are neither detected nor repaired.

    They stay listable, loadable, and traceable, a completed run of such a
    configuration reopens, and running its configuration again, or a run over
    its cells, raises before any stage.
    """
    store = measured.open()
    with pytest.MonkeyPatch.context() as patch:
        # Bypass validation to build a completed run over every cell.
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        patch.setattr(
            "cytearc.datastore._pipeline.stages.require_measured_cells",
            lambda *_args, **_kwargs: None,
        )
        store.pipeline.rna(label="earlier", **_PIPELINE)
    legacy = measured.refs["legacy_normalized"]

    assert legacy in store.artifacts.list(from_assay="RNA", kind="normalized")
    assert store.artifacts.inspect(legacy).complete
    assert store.artifacts.load(legacy)["data"].shape[0] == _N_CELLS
    assert measured.refs["all"] in set(store.artifacts.lineage(legacy).graph)
    reopened = store.pipeline.open(label="earlier")
    assert reopened.status == "completed"
    assert reopened.report()["run"]["config"]["cells"] == {"column": "I"}

    runs = [run.run_id for run in store.pipeline.list_runs()]
    before = _tree(measured.path)
    with pytest.raises(UnmeasuredCellsError, match="pipeline.rna"):
        store.pipeline.rna(label="again", **_PIPELINE)
    # A run over the completed run's cells reads RNA over every cell too.
    with pytest.raises(UnmeasuredCellsError) as raised:
        store.pipeline.rna(label="again", cells=reopened, **_PIPELINE)
    assert (raised.value.operation, raised.value.remedy) == ("pipeline.rna", "cells")
    assert (raised.value.unmeasured, raised.value.selected) == (_N_ADT, _N_CELLS)
    assert _selection_cells_remedy("RNA") in str(raised.value)
    assert [run.run_id for run in store.pipeline.list_runs()] == runs
    assert _tree(measured.path) == before


def test_display_reads_show_unmeasured_cells_as_missing(measured) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    every = np.arange(_N_CELLS)
    rna = resolve_feature(store, "RNA0")
    adt = resolve_feature(store, FeatureRef("ADT0", assay="ADT"))

    matrix = fetch_normalized_feature_matrix(store, [rna, adt], every)

    assert np.isnan(matrix[~membership, 0]).all()
    assert np.isfinite(matrix[membership, 0]).all()
    # ADT measured every cell, so each of its values is read.
    assert np.isfinite(matrix[:, 1]).all()
    # Measured cells keep the values that a read of them alone gives.
    np.testing.assert_array_equal(
        matrix[membership, 0],
        fetch_normalized_feature_matrix(store, [rna], np.flatnonzero(membership))[:, 0],
    )
    raw = fetch_normalized_feature_matrix(
        store, [rna], every, NormalizationSpec(source="raw", transform="log1p")
    )
    assert np.isnan(raw[~membership, 0]).all()
    assert (raw[membership, 0] > 0).all()
    blocks = [
        (start, values)
        for _slots, start, values in iter_normalized_feature_blocks(store, [rna], every)
    ]
    streamed = np.concatenate([values for _start, values in blocks])
    np.testing.assert_array_equal(streamed, matrix[:, :1])

    values = store.get_cell_vals("RNA", "I", "RNA0")
    np.testing.assert_array_equal(values, matrix[:, 0])
    np.testing.assert_array_equal(
        store.get_cell_vals("RNA", "adt_cells", "RNA0"), np.full(_N_ADT, np.nan)
    )
    clipped = store.get_cell_vals("RNA", "I", "RNA0", clip_fraction=0.1)
    assert np.isnan(clipped[~membership]).all()
    # A cell metadata column is read as it is stored.
    np.testing.assert_array_equal(
        store.get_cell_vals("RNA", "I", "RNA_nCounts")[~membership], 0
    )


def test_plots_draw_unmeasured_cells_as_missing(measured) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from cytearc import plotting

    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    expression = fetch_normalized_feature_matrix(
        store, [resolve_feature(store, "RNA0")], np.flatnonzero(membership)
    )[:, 0]

    layout = plotting.embedding(store, layout_key="xy", color_by="RNA0", show=False)
    try:
        (axis,) = layout.axes.values()
        points = axis.collections[0]
        rows = np.asarray(points.get_offsets())[:, 0].astype(int)
        gray = np.all(
            np.isclose(
                points.get_facecolors(),
                matplotlib.colors.to_rgba(plotting.ColorScale().missing_color),
            ),
            axis=1,
        )
        # Unmeasured cells take the color of missing values, and the color
        # limits come from measured values only.
        np.testing.assert_array_equal(gray, ~membership[rows])
        assert layout.provenance.extras["color_limits"] == {
            "RNA0": (float(expression.min()), float(expression.max()))
        }
        assert layout.provenance.extras["unmeasured_cells"] == {"RNA": _N_ADT}
    finally:
        layout.close()

    violins = plotting.distribution(
        store, "RNA0", grouping=CellField("program"), show=False
    )
    try:
        table = violins.tables["RNA0"]
        # The unmeasured cells have no value to draw or tabulate.
        assert violins.provenance.extras["unmeasured_cells"] == {"RNA": _N_ADT}
        assert len(table) == _N_FULL
        np.testing.assert_allclose(
            np.sort(table["value"].to_numpy()), np.sort(expression)
        )
    finally:
        violins.close()


def test_require_measured_cells_reads_masks_rows_and_columns(measured) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    stored_mask = zarr.open_group(str(measured.path), mode="r")["cellData/I"]
    cases: list[Any] = [
        "I",
        np.ones(_N_CELLS, dtype=bool),
        stored_mask,
        np.arange(_N_CELLS)[::-1],
        np.concatenate([np.arange(_N_CELLS), [70, 71]]),
        list(range(_N_CELLS)),
    ]
    for cells in cases:
        with pytest.raises(UnmeasuredCellsError) as raised:
            require_measured_cells(store.cells, "RNA", cells, operation="probe")
        assert (raised.value.unmeasured, raised.value.selected) == (
            _N_ADT,
            _N_CELLS,
        )
        assert "Pass a selection of measured cells from " in str(raised.value)
    # Measured cells, an assay that measured every cell, and an empty read pass.
    for cells in ("RNA_I", membership, np.flatnonzero(membership)):
        require_measured_cells(store.cells, "RNA", cells, operation="probe")
    require_measured_cells(store.cells, "ADT", "I", operation="probe")
    require_measured_cells(
        store.cells, "RNA", np.array([], dtype=np.int64), operation="probe"
    )
    with pytest.raises(ValueError, match="one entry per cell"):
        require_measured_cells(
            store.cells, "RNA", np.ones(3, dtype=bool), operation="probe"
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        require_measured_cells(
            store.cells, "RNA", np.zeros((2, 2), dtype=np.int64), operation="probe"
        )
    # A membership column of two-row chunks is read one block at a time.
    group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    for name, values in (
        ("ids", np.array(["c0", "c1", "c2", "c3"])),
        ("names", np.array(["c0", "c1", "c2", "c3"])),
        ("I", np.ones(4, dtype=bool)),
        ("RNA_I", np.array([True, True, False, True])),
    ):
        group.create_array(name, data=values, chunks=(2,))
    group["RNA_I"].attrs.update(membership_attributes("RNA"))
    with pytest.raises(UnmeasuredCellsError) as raised:
        require_measured_cells(
            MetaData(group),
            "RNA",
            np.array([True, True, True, False]),
            operation="probe",
        )
    assert (raised.value.unmeasured, raised.value.selected) == (1, 3)


def test_unmeasured_cells_error_keeps_its_fields_through_pickling() -> None:
    error = UnmeasuredCellsError("run_pca", "ADT", "ADT_I", 2, 9, "graph")

    copy = pickle.loads(pickle.dumps(error))

    assert isinstance(copy, UnmeasuredCellsError)
    assert isinstance(copy, ValueError)
    assert (copy.operation, copy.assay, copy.column) == ("run_pca", "ADT", "ADT_I")
    assert (copy.unmeasured, copy.selected, copy.remedy) == (2, 9, "graph")
    assert copy.cell_key is None
    assert str(copy) == str(error)
    for operation, remedy in (("to_anndata", "cell_key"), ("pipeline.rna", "cells")):
        keyed = UnmeasuredCellsError(operation, "RNA", "RNA_I", 1, 4, remedy, "kept")
        assert "cells of 'kept' that it measured" in str(keyed)
        assert pickle.loads(pickle.dumps(keyed)).cell_key == "kept"
        assert str(pickle.loads(pickle.dumps(keyed))) == str(keyed)


def test_membership_helpers_reject_unprepared_tables_and_misaligned_values() -> None:
    group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(3, dtype=bool))
    group.create_array("ids", data=np.array(["a", "b", "c"]))
    group.create_array("names", data=np.array(["A", "B", "C"]))
    table = MetaData(group)

    with pytest.raises(ValueError, match="no 'RNA_nCounts' column"):
        count_unmeasured_cells_with_counts(table, "RNA", "RNA_I")
    with pytest.raises(ValueError, match="one entry per cell"):
        measured_selection(table, "RNA", np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="one value per cell"):
        write_membership_column(group, "RNA", np.ones((3, 1), dtype=bool))

    assert set(group.array_keys()) == {"I", "ids", "names"}
    np.testing.assert_array_equal(table.fetch_all("I"), [True, True, True])


def test_unmeasured_cells_error_rejects_an_unknown_remedy() -> None:
    with pytest.raises(ValueError, match="Unknown measured-cells remedy"):
        UnmeasuredCellsError("normalize", "RNA", "RNA_I", 1, 3, "unknown")


# ---------------------------------------------------------------------------
# Display values, summary plots, QC, bulk profiles, and exports over a merge
# whose RNA assay measured the cells of one source.

_N_RNA = 40
_N_OTHER = 20


def _merge_partial_rna(base: Path) -> Path:
    """Merge 40 cells with RNA and ADT and 20 cells with ADT only.

    RNA0 is detected in about half of the measured cells. The merge keeps
    each source's cells together, so the first 40 cells are the measured
    ones. ``grp`` alternates two groups over every cell, ``source`` names
    each cell's source, ``adt_only`` marks the unmeasured cells, and ``xy``
    is a live layout.
    """
    rng = np.random.default_rng(3)
    rna = rng.poisson(3.0, size=(_N_RNA, 12)) + 1
    rna[:, 0] = rng.binomial(1, 0.5, size=_N_RNA) * 5
    write_count_store(
        str(base / "full.zarr"),
        {"RNA": rna, "ADT": rng.integers(1, 20, size=(_N_RNA, 3))},
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"),
        {"ADT": rng.integers(1, 20, size=(_N_OTHER, 3))},
        "uint32",
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay=assay,
            min_features_per_cell=0,
            nthreads=1,
        )
        for name, assay in (("full", "RNA"), ("adt", "ADT"))
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["full", "adt"], nthreads=1).dump()
    store = _open(path)
    n_cells = store.cells.N
    store.cells.insert(
        "grp",
        by_cell_id(store.cells, np.where(np.arange(n_cells) % 2 == 0, "g0", "g1")),
    )
    ids = np.asarray(store.cells.fetch_all("ids")).astype(str)
    store.cells.insert(
        "source",
        by_cell_id(store.cells, np.array([value.split("__")[0] for value in ids])),
    )
    store.cells.insert(
        "adt_only",
        by_cell_id(
            store.cells, ~np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
        ),
    )
    store.cells.insert(
        "xy1", by_cell_id(store.cells, np.arange(n_cells, dtype=np.float64))
    )
    store.cells.insert(
        "xy2", by_cell_id(store.cells, (np.arange(n_cells) % 7).astype(np.float64))
    )
    return path


def _open(path: Path, zarr_mode: str = "r+") -> DataStore:
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        zarr_mode=zarr_mode,
    )


@pytest.fixture(scope="module")
def partial_rna(tmp_path_factory) -> Path:
    return _merge_partial_rna(tmp_path_factory.mktemp("partial_rna"))


def _plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from cytearc import plotting

    return plotting


def test_a_grouped_assay_fits_its_source_normalization_on_measured_cells(
    measured, tmp_path
) -> None:
    """Group means of a partly measured assay ignore the cells it did not measure.

    Before, ATAC document frequencies counted the zero rows of the unmeasured
    cells, and the wrong means sat on rows that the membership marks measured.
    The merged store's measured cells now hold the means that the source store,
    which holds only those cells, gives; the others hold zeros.
    """
    import shutil

    stores = {}
    for name in ("merged.zarr", "full.zarr"):
        path = tmp_path / name
        shutil.copytree(measured.path.parent / name, path)
        store = DataStore(
            str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
        )
        n_peaks = store.ATAC.feats.N
        store.ATAC.feats.insert("module", np.arange(n_peaks) % 2, overwrite=True)
        store.add_grouped_assay("module", assay_label="modules", from_assay="ATAC")
        stores[name] = store
    merged, alone = stores["merged.zarr"], stores["full.zarr"]
    means = np.asarray(merged.modules.rawData, dtype=np.float64)
    expected = np.asarray(alone.modules.rawData, dtype=np.float64)
    member = np.asarray(merged.cells.fetch_all("ATAC_I"), dtype=bool)
    # The merge names each cell <source>__<id> and may reorder the cells.
    row_of = {str(cell): row for row, cell in enumerate(alone.cells.fetch_all("ids"))}
    rows = [
        row_of[str(cell).split("__", 1)[1]]
        for cell in np.asarray(merged.cells.fetch_all("ids"))[member]
    ]

    assert member.sum() == _N_FULL and means.shape == (_N_CELLS, 2)
    np.testing.assert_allclose(means[member], expected[rows], rtol=1e-12)
    np.testing.assert_array_equal(means[~member], 0.0)


def test_display_values_of_measured_cells_ignore_unmeasured_cells(measured) -> None:
    """A display read normalizes only the cells that the assay measured.

    TF-IDF fits document frequency over the cells that it normalizes, so
    unmeasured cells in the read raised the values of measured cells 1.4
    times. A measured cell now reads the value that a read of the measured
    cells alone gives, for ATAC as for RNA, in every block order.
    """
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("ATAC_I"), dtype=bool)
    measured_rows_ = np.flatnonzero(membership)
    every = np.arange(_N_CELLS)
    peak = resolve_feature(store, FeatureRef("ATAC30", assay="ATAC"))
    alone = fetch_normalized_feature_matrix(store, [peak], measured_rows_)[:, 0]
    assert np.isfinite(alone).all() and (alone > 0).any()

    whole = fetch_normalized_feature_matrix(store, [peak], every)[:, 0]

    assert np.isnan(whole[~membership]).all()
    np.testing.assert_array_equal(whole[membership], alone)
    # Unmeasured cells before the measured ones, and between them.
    reverse = fetch_normalized_feature_matrix(store, [peak], every[::-1])[:, 0]
    np.testing.assert_array_equal(reverse, whole[::-1])
    mixed = np.array([70, 0, 71, 5, 64, 63, 79])
    read = membership[mixed]
    expected = np.full(len(mixed), np.nan)
    expected[read] = fetch_normalized_feature_matrix(store, [peak], mixed[read])[:, 0]
    np.testing.assert_array_equal(
        fetch_normalized_feature_matrix(store, [peak], mixed)[:, 0], expected
    )
    # A read of unmeasured cells only has no value to show.
    np.testing.assert_array_equal(
        fetch_normalized_feature_matrix(store, [peak], np.flatnonzero(~membership)),
        np.full((_N_ADT, 1), np.nan),
    )
    values = store.get_cell_vals("ATAC", "I", "ATAC30")
    assert np.isnan(values[~membership]).all()
    np.testing.assert_allclose(values[membership], alone, rtol=1e-12)


@pytest.mark.slow
def test_summary_statistics_cover_the_cells_with_a_value(partial_rna) -> None:
    """Every summary statistic has the measured cells of its group as cells.

    The review's merge: each group holds 30 cells, RNA measured 20 of them,
    and RNA0 is detected in 10 and 11 of those. The fraction was the
    detected cells over all 30 cells, 0.333 and 0.367, and ``n_cells``
    counted 30. A group without a measured cell has no fraction, and an
    aggregate over samples skips such a sample as it skips its mean.
    """
    plotting = _plotting()
    store = _open(partial_rna)
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    groups = np.asarray(store.cells.fetch_all("grp")).astype(str)
    detected = np.asarray(store.RNA.rawData[:, [0]].compute()).ravel() > 0
    expected = {
        group: (
            int((membership & (groups == group)).sum()),
            float(detected[membership & (groups == group)].mean()),
        )
        for group in ("g0", "g1")
    }
    assert expected == {"g0": (20, 0.5), "g1": (20, 0.55)}

    result = plotting.dotplot(store, features=["RNA0"], group_by="grp", show=False)
    try:
        table = result.tables["aggregate"].set_index("group")
        for group, (n_cells, fraction) in expected.items():
            assert table.loc[group, "n_cells"] == n_cells
            assert table.loc[group, "fraction"] == pytest.approx(fraction)
        assert result.provenance.extras["unmeasured_cells"] == {"RNA": _N_OTHER}
    finally:
        result.close()

    by_source = plotting.matrixplot(
        store, features=["RNA0"], group_by="source", value="fraction", show=False
    )
    try:
        table = by_source.tables["aggregate"].set_index("group")
        assert table.loc["adt", "n_cells"] == 0
        assert np.isnan(
            table.loc["adt", ["mean", "fraction", "variance"]].to_numpy(dtype=float)
        ).all()
        assert table.loc["full", "n_cells"] == _N_RNA
        assert table.loc["full", "fraction"] == pytest.approx(
            detected[membership].mean()
        )
        assert np.isnan(by_source.tables["matrix"].loc["RNA0", "adt"])
        assert by_source.provenance.extras["unmeasured_cells"] == {"RNA": _N_OTHER}
    finally:
        by_source.close()

    sampled = plotting.dotplot(
        store, features=["RNA0"], group_by="grp", sample_by="source", show=False
    )
    try:
        aggregate = sampled.tables["aggregate"].set_index("group")
        per_sample = sampled.tables["per_sample"].set_index(["sample", "group"])
        for group, (n_cells, fraction) in expected.items():
            assert aggregate.loc[group, "fraction"] == pytest.approx(fraction)
            assert aggregate.loc[group, "n_cells"] == n_cells
            assert aggregate.loc[group, "n_samples"] == 1
            assert per_sample.loc[("adt", group), "n_cells"] == 0
            assert np.isnan(per_sample.loc[("adt", group), "fraction"])
    finally:
        sampled.close()


def test_plots_record_the_cells_that_an_assay_did_not_measure(partial_rna) -> None:
    plotting = _plotting()
    store = _open(partial_rna)

    # Over unmeasured cells only, no cell has a value to set color limits.
    unmeasured = plotting.embedding(
        store, layout_key="xy", color_by="RNA0", cell_key="adt_only", show=False
    )
    try:
        assert unmeasured.provenance.extras["color_limits"] == {"RNA0": None}
        assert unmeasured.provenance.extras["unmeasured_cells"] == {"RNA": _N_OTHER}
    finally:
        unmeasured.close()
    # ADT measured every cell, so nothing is recorded for it.
    adt = plotting.embedding(
        store, layout_key="xy", color_by=FeatureRef("ADT0", assay="ADT"), show=False
    )
    try:
        assert "unmeasured_cells" not in adt.provenance.extras
    finally:
        adt.close()
    with pytest.raises(ValueError, match="No finite values remain") as raised:
        plotting.distribution(store, "RNA0", subset_by="adt_only", show=False)
    message = str(raised.value)
    assert "assay 'RNA' did not measure" in message
    assert "subset_by='RNA_I'" in message
    # The remedy's subset leaves every unmeasured cell out, so none is recorded.
    for result in (
        plotting.distribution(store, "RNA0", subset_by="RNA_I", show=False),
        plotting.embedding(
            store, layout_key="xy", color_by="RNA0", subset_by="RNA_I", show=False
        ),
    ):
        try:
            assert "unmeasured_cells" not in result.provenance.extras
        finally:
            result.close()
    # Unmeasured cells without a group join no summary, so none is recorded.
    store.cells.insert(
        "rna_grp",
        by_cell_id(
            store.cells, np.where(np.arange(_N_RNA) % 2 == 0, "g0", "g1"), key="RNA_I"
        ),
        key="RNA_I",
        overwrite=True,
    )
    grouped = plotting.dotplot(store, features=["RNA0"], group_by="rna_grp", show=False)
    try:
        assert grouped.provenance.extras["dropped_group_cells"] == _N_OTHER
        assert "unmeasured_cells" not in grouped.provenance.extras
    finally:
        grouped.close()


def _merge_low_quality(base: Path) -> Path:
    """Merge 40 RNA cells, five of them low quality, with 50 ADT-only cells."""
    rng = np.random.default_rng(1)
    rna = rng.poisson(3.0, size=(40, 30)) + 1
    rna[:5, 3:] = 0
    write_count_store(
        str(base / "full.zarr"),
        {"RNA": rna, "ADT": rng.integers(1, 9, size=(40, 3))},
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"),
        {"ADT": rng.integers(1, 9, size=(50, 3))},
        "uint32",
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay=assay,
            min_features_per_cell=0,
            nthreads=1,
        )
        for name, assay in (("full", "RNA"), ("adt", "ADT"))
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["full", "adt"], nthreads=1).dump()
    _open(path)
    return path


@pytest.mark.slow
def test_quality_control_refuses_metrics_of_unmeasured_cells(tmp_path) -> None:
    """QC thresholds over an assay's metrics read that assay's cells.

    With 50 of 90 cells unmeasured, the zeros of their RNA metrics set a
    zero MAD, so ``auto_filter_cells`` kept every cell, the five low-quality
    measured cells included.
    """
    from cytearc.metadata.selection import NamedCellArtifact

    path = _merge_low_quality(tmp_path)
    store = _open(path)
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    with pytest.MonkeyPatch.context() as patch:
        # Bypass validation to compute this metric over every cell.
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        legacy_metric = store.qc.feature_percentage(
            store.snapshot_cell_selection("I"),
            store.features.snapshot(from_assay="RNA", feature_indexes=[0, 1]),
        )
    before = _tree(path)

    for call in (
        lambda s: s.qc.auto_filter(attrs=["RNA_nCounts"]),
        lambda s: s.qc.auto_filter(attrs=["RNA_nCounts"], method="gaussian"),
        lambda s: s.qc.auto_filter(
            attrs=[], artifact_metrics=[NamedCellArtifact("pct", legacy_metric)]
        ),
        lambda s: s.qc.filter(["RNA_nFeatures"], [5], [None]),
    ):
        for opened in (store, _open(path, zarr_mode="r")):
            with pytest.raises(UnmeasuredCellsError) as raised:
                call(opened)
            assert (raised.value.assay, raised.value.column) == ("RNA", "RNA_I")
            assert (raised.value.unmeasured, raised.value.selected) == (50, 90)
            assert "Pass `cell_selection=select_measured_cells('RNA'" in str(
                raised.value
            )
    assert _tree(path) == before

    # A metric of an assay that measured every cell reads every cell, and a
    # user column named like a metric is no metric.
    assert store.artifacts.load(store.qc.filter(["ADT_nCounts"], [0], [None]))[
        "values"
    ][:].all()
    store.cells.insert("RNA_score", by_cell_id(store.cells, np.ones(store.cells.N)))
    assert store.artifacts.load(store.qc.filter(["RNA_score"], [0], [None]))["values"][
        :
    ].all()
    # Over the measured cells, the MAD bounds remove the low-quality cells.
    kept = store.qc.auto_filter(
        attrs=["RNA_nCounts"], cell_selection=store.select_measured_cells("RNA")
    )
    values = np.asarray(store.artifacts.load(kept)["values"][:], dtype=bool)
    low_quality = membership & (np.asarray(store.cells.fetch_all("RNA_nFeatures")) < 5)
    assert int(low_quality.sum()) == 5
    assert not (values & low_quality).any()
    assert not (values & ~membership).any()
    assert int(values.sum()) >= 30


def test_membership_reads_return_rows_in_order_and_skip_full_columns(
    measured,
) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    every = np.arange(_N_CELLS)

    for rows in (
        every,
        every[::-1],
        np.array([0, 0, 3, 3, 70, 70, 79]),
        np.array([79, 3, 70, 3, 0]),
        np.array([63, 64]),
    ):
        np.testing.assert_array_equal(
            measured_rows(store.cells, "RNA", rows), membership[rows]
        )
    # Rows that the assay measured, any rows of an assay that measured every
    # cell, and an empty read have nothing to show as missing.
    assert measured_rows(store.cells, "RNA", np.flatnonzero(membership)) is None
    assert measured_rows(store.cells, "ADT", every[::-1]) is None
    assert measured_rows(store.cells, "RNA", np.array([], dtype=np.int64)) is None
    with pytest.raises(TypeError, match="integers or a boolean mask"):
        measured_rows(store.cells, "RNA", np.array([0.5]))
    for assay in ("RNA", "ADT"):
        for rows in (np.array([_N_CELLS]), np.array([-1]), np.array([3, _N_CELLS])):
            with pytest.raises(IndexError, match="out of bounds"):
                measured_rows(store.cells, assay, rows)
            with pytest.raises(IndexError, match="out of bounds"):
                require_measured_cells(store.cells, assay, rows, operation="probe")


def test_a_few_membership_rows_read_only_their_chunks(measured, monkeypatch) -> None:
    """A lookup of a few cells never streams the whole column.

    Before, finding the membership of one cell near the end of the column
    read every block from the first.
    """
    import cytearc.metadata.membership as membership_module

    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)

    def no_scan(*args, **kwargs):
        raise AssertionError("a lookup of a few rows streamed the column")

    monkeypatch.setattr(membership_module, "iter_metadata_column_blocks", no_scan)
    unmeasured = int(np.flatnonzero(~membership)[-1])
    for rows in (
        np.array([unmeasured]),
        np.array([_N_CELLS - 1, 3, _N_CELLS - 1]),
        np.array([unmeasured, 0]),
    ):
        expected = membership[rows]
        observed = measured_rows(store.cells, "RNA", rows)
        if expected.all():
            assert observed is None
        else:
            np.testing.assert_array_equal(observed, expected)
    measured_row = int(np.flatnonzero(membership)[0])
    with pytest.raises(UnmeasuredCellsError) as raised:
        require_measured_cells(
            store.cells,
            "RNA",
            np.array([unmeasured, measured_row, unmeasured]),
            operation="probe",
        )
    # A repeated row is one cell.
    assert (raised.value.unmeasured, raised.value.selected) == (1, 2)


def test_store_check_leaves_cells_of_another_type_to_the_operation(
    partial_rna,
) -> None:
    store = _open(partial_rna)
    rows = list(range(store.cells.N))
    with pytest.raises(UnmeasuredCellsError) as raised:
        require_measured_cells(store.cells, "RNA", rows, operation="probe")
    assert (raised.value.unmeasured, raised.value.selected) == (
        _N_OTHER,
        _N_RNA + _N_OTHER,
    )

    # The store's check reads a selection, a column name, or an array; the
    # operation's own argument checks judge a value of another type.
    assert store._require_measured_cells("RNA", rows, operation="probe") is None


def test_make_bulk_refuses_unmeasured_cells_before_it_writes(partial_rna) -> None:
    """A metadata grouping without a selection is checked over live cells.

    ``make_bulk`` snapshotted the live ``I`` column for such a grouping. It
    wrote that snapshot before it refused, and a read-only store raised
    ``PermissionError`` for the snapshot instead of the refusal. It now reads
    the live cells without a snapshot.
    """
    before = _tree(partial_rna)
    for zarr_mode in ("r+", "r"):
        store = _open(partial_rna, zarr_mode=zarr_mode)
        with pytest.raises(UnmeasuredCellsError) as raised:
            store.compare.compute_bulk("grp", from_assay="RNA")
        assert (raised.value.unmeasured, raised.value.selected) == (
            _N_OTHER,
            _N_RNA + _N_OTHER,
        )
        assert "Pass `cell_selection=select_measured_cells('RNA'" in str(raised.value)
    assert _tree(partial_rna) == before


def test_make_bulk_checks_only_the_cells_that_it_reads(partial_rna, measured) -> None:
    """Cells that join no bulk column are not read, unless a fit reads them.

    RNA sums and RNA library-size means read the cells of the bulk columns
    only, so unmeasured cells that ``null_vals`` excludes pass. A mean off
    the RNA stream fits the assay's normalization, such as ATAC TF-IDF, over
    every selected cell, so it still reads them.
    """
    store = _open(partial_rna)
    for aggr_type in ("sum", "mean"):
        bulk = store.compare.compute_bulk(
            "source", from_assay="RNA", null_vals=["adt"], aggr_type=aggr_type
        )
        assert list(bulk.profiles.columns) == ["full"]

    multiome = measured.open()
    sums = multiome.compare.compute_bulk(
        "adt_cells", from_assay="ATAC", null_vals=[True], aggr_type="sum"
    )
    # A boolean column labels its bulk column by the boolean itself.
    assert list(sums.profiles.columns) == [False]
    with pytest.raises(UnmeasuredCellsError) as raised:
        multiome.compare.compute_bulk("adt_cells", from_assay="ATAC", null_vals=[True])
    assert (raised.value.assay, raised.value.unmeasured) == ("ATAC", _N_ADT)


@pytest.mark.parametrize("scope", [None, "panel"])
def test_a_colorbar_keeps_its_own_scale_beside_a_color_without_values(
    partial_rna, scope
) -> None:
    """Each colorbar keeps its color's scale; a color without values keeps none.

    Composed figures pair colorbars with scales in order. Before, the scale of
    a color without values stayed first, so the next colorbar took it.
    """
    plotting = _plotting()
    store = _open(partial_rna)
    result = plotting.embedding(
        store,
        layout_key="xy",
        color_by=["RNA0", FeatureRef("ADT0", assay="ADT")],
        cell_key="adt_only",
        color_scale=None if scope is None else plotting.ColorScale(scope=scope),
        show=False,
    )
    try:
        colorbars = [legend for legend in result.legends if legend.kind == "colorbar"]
        scales = [
            scale for scale in result.scales if isinstance(scale, plotting.ColorScale)
        ]
        assert len(colorbars) == len(scales) == 1
        assert "RNA0" not in colorbars[0].label
    finally:
        result.close()


def test_a_color_without_values_draws_no_colorbar(partial_rna) -> None:
    """Cells that the assay did not measure get no invented color scale.

    Before, the colorbar showed limits of 0 to 1, and a log scale raised
    "Log color scale requires positive values" although no value existed.
    """
    plotting = _plotting()
    store = _open(partial_rna)
    for scale in (
        plotting.ColorScale(),
        plotting.ColorScale(scale="log"),
        plotting.ColorScale(scope="panel"),
        # A diverging center outside the placeholder limits raised.
        plotting.ColorScale(vcenter=0.0),
        plotting.ColorScale(scope="panel", vcenter=0.0),
        plotting.ColorScale(scope="shared", vcenter=0.0),
        plotting.ColorScale(vcenter=1e20),
    ):
        result = plotting.embedding(
            store,
            layout_key="xy",
            color_by="RNA0",
            cell_key="adt_only",
            color_scale=scale,
            show=False,
        )
        try:
            assert [legend.kind for legend in result.legends] == []
            assert len(result.figure.axes) == 1
        finally:
            result.close()
    # Explicit limits still draw their colorbar.
    explicit = plotting.embedding(
        store,
        layout_key="xy",
        color_by="RNA0",
        cell_key="adt_only",
        color_scale=plotting.ColorScale(vmin=0, vmax=2),
        show=False,
    )
    try:
        assert [legend.kind for legend in explicit.legends] == ["colorbar"]
    finally:
        explicit.close()


def test_raw_exports_refuse_cells_whose_membership_they_cannot_declare(
    partial_rna, tmp_path
) -> None:
    """An MTX directory cannot declare which cells an assay measured.

    Its zero rows for unmeasured cells read back as measured cells through
    ``MtxToZarr``, so the export refuses them before it writes anything.
    """
    from cytearc.writers import to_mtx

    store = _open(partial_rna)
    target = tmp_path / "rna_mtx"

    with pytest.raises(UnmeasuredCellsError) as raised:
        to_mtx(store.RNA, str(target))

    assert (raised.value.operation, raised.value.assay) == ("to_mtx", "RNA")
    assert (raised.value.unmeasured, raised.value.selected) == (
        _N_OTHER,
        _N_RNA + _N_OTHER,
    )
    assert "SubsetZarr(..., cell_key='RNA_I')" in str(raised.value)
    assert "to_h5ad" in str(raised.value)
    assert not target.exists()
    # ADT measured every cell.
    to_mtx(store.ADT, str(tmp_path / "adt_mtx"))
    assert (tmp_path / "adt_mtx" / "matrix.mtx").exists()


def _write_shared_features(
    path: Path, counts: dict[str, np.ndarray], feature_ids: list[str]
) -> None:
    """Write a store whose assays share their feature IDs, as layers need."""
    from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
    from cytearc.storage.stores import load_zarr
    from cytearc.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    root = load_zarr(zarr_loc=str(path), mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    cell_ids = np.array([f"cell{index}" for index in range(n_cells)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    for assay, values in counts.items():
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            n_cells,
            feat_ids=np.array(feature_ids),
            feat_names=np.array(feature_ids),
            dtype="uint32",
        )
        array[:] = values.astype(np.uint32)
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)


def _merge_spliced(base: Path) -> Path:
    """Merge 48 cells of spliced and unspliced counts with 24 of spliced only.

    RNA holds the spliced counts of every cell. URNA holds unspliced counts
    over the same feature IDs and measured the first 48 cells only.
    """
    rng = np.random.default_rng(11)
    genes = [f"g{index}" for index in range(40)]
    programs = np.arange(48) % 4
    spliced = rng.poisson(1.0, (48, 40))
    for program in range(4):
        spliced[programs == program, 10 * program : 10 * program + 10] += 6
    _write_shared_features(
        base / "spliced.zarr",
        {"RNA": spliced, "URNA": rng.integers(1, 9, (48, 40))},
        genes,
    )
    _write_shared_features(
        base / "plain.zarr", {"RNA": rng.poisson(3.0, (24, 40)) + 1}, genes
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )
        for name in ("spliced", "plain")
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["spliced", "plain"], nthreads=1).dump()
    return path


@pytest.fixture(scope="module")
def spliced(tmp_path_factory) -> Path:
    return _merge_spliced(tmp_path_factory.mktemp("spliced"))


@pytest.mark.slow
def test_layers_refuse_cells_that_their_assay_did_not_measure(spliced) -> None:
    """A layer's assay has no membership declaration in the exported object."""
    store = _open(spliced)

    with pytest.raises(UnmeasuredCellsError) as raised:
        store.to_anndata(layers={"unspliced": "URNA"})

    assert (raised.value.operation, raised.value.assay) == ("to_anndata", "URNA")
    assert (raised.value.unmeasured, raised.value.selected) == (24, 72)
    assert raised.value.cell_key == "I"
    exported = store.to_anndata(cell_key="URNA_I", layers={"unspliced": "URNA"})
    assert exported.layers["unspliced"].shape == (48, 40)
    # The exported assay declares its own membership, so its layer is kept.
    assert store.to_anndata(layers={"raw": "RNA"}).layers["raw"].shape == (72, 40)

    # A run export declares no membership at all.
    run = store.pipeline.rna(label="layers", **_PIPELINE)
    with pytest.raises(UnmeasuredCellsError, match="SubsetZarr") as raised:
        store.to_anndata(run=run, layers={"unspliced": "URNA"})
    assert (raised.value.unmeasured, raised.value.selected) == (24, 72)
    assert store.to_anndata(run=run, layers={"raw": "RNA"}).layers["raw"].shape == (
        72,
        40,
    )


@pytest.mark.slow
def test_pipeline_filtering_on_another_assays_metric_reads_that_assay(
    spliced,
) -> None:
    """The pipeline's QC filter reads the assay of each metric, as QC does."""
    store = _open(spliced)
    every = store.snapshot_cell_selection("I")
    before = _tree(spliced)
    filtering = {"method": "mad", "attrs": ["RNA_nCounts", "URNA_nCounts"]}

    # A selection of cells is checked as a cell column is.
    for cells, remedy in (
        ("I", _cells_remedy("URNA")),
        (every, _selection_cells_remedy("URNA")),
    ):
        with pytest.raises(UnmeasuredCellsError) as raised:
            store.pipeline.rna(
                label="urna", cells=cells, **{**_PIPELINE, "filtering": filtering}
            )

        assert (raised.value.operation, raised.value.assay) == ("pipeline.rna", "URNA")
        assert (raised.value.unmeasured, raised.value.selected) == (24, 72)
        assert raised.value.remedy == "cells"
        assert remedy in str(raised.value)
    assert _tree(spliced) == before
    # The pipeline's own assay measured every cell, so its metrics pass.
    _recipe(store, "I", filtering={"attrs": ["RNA_nCounts"]})


def _merge_partial_adt(base: Path) -> Path:
    """Merge 40 cells with RNA and ADT and 24 cells with RNA only.

    RNA measured every cell, and ADT the 40 cells of the CITE-seq source.
    """
    rna = mapping_counts()
    adt = np.random.default_rng(19).integers(1, 20, size=(40, 4))
    write_count_store(str(base / "cite.zarr"), {"RNA": rna[:40], "ADT": adt}, "uint32")
    write_count_store(str(base / "rna.zarr"), {"RNA": rna[40:]}, "uint32")
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )
        for name in ("cite", "rna")
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["cite", "rna"], nthreads=1).dump()
    return path


@pytest.fixture(scope="module")
def partial_adt(tmp_path_factory) -> Path:
    return _merge_partial_adt(tmp_path_factory.mktemp("partial_adt"))


def _assert_adt_refusal(error: UnmeasuredCellsError, remedy: str) -> None:
    assert (error.operation, error.assay, error.column) == (
        "pipeline.adt",
        "ADT",
        "ADT_I",
    )
    assert (error.unmeasured, error.selected, error.remedy) == (24, 64, "cells")
    assert str(error).startswith(
        "`pipeline.adt` reads assay 'ADT' over 24 of 64 selected cells that it did "
        "not measure: cell column 'ADT_I' is False for them. "
    )
    assert remedy in str(error)


def test_the_adt_recipe_refuses_cells_that_its_assay_did_not_measure(
    partial_adt,
) -> None:
    store = _open(partial_adt)
    every = store.snapshot_cell_selection("I")
    runs = [run.run_id for run in store.pipeline.list_runs()]
    before = _tree(partial_adt)

    for cells, remedy in (
        ("I", _cells_remedy("ADT")),
        (every, _selection_cells_remedy("ADT")),
    ):
        with pytest.raises(UnmeasuredCellsError) as raised:
            store.pipeline.adt(cells=cells, **_ADT)
        _assert_adt_refusal(raised.value, remedy)
    assert [run.run_id for run in store.pipeline.list_runs()] == runs
    assert _tree(partial_adt) == before
    # The remedy's selection holds the cells that ADT measured.
    resolve_adt(
        store,
        cells=store.select_measured_cells("ADT", cell_selection=every),
        **_ADT,
    )


@pytest.mark.slow
def test_the_adt_recipe_refuses_an_rna_run_over_cells_without_antibodies(
    partial_adt,
) -> None:
    """An ADT run takes the cells of an RNA run only where ADT measured them."""
    store = _open(partial_adt)
    rna_run = store.pipeline.rna(**{**_PIPELINE, "leiden": False})
    runs = [run.run_id for run in store.pipeline.list_runs()]
    before = _tree(partial_adt)

    with pytest.raises(UnmeasuredCellsError) as raised:
        store.pipeline.adt(cells=rna_run, **_ADT)

    # The cells are the RNA run's, so the remedy restricts that run.
    _assert_adt_refusal(raised.value, _selection_cells_remedy("ADT"))
    assert [run.run_id for run in store.pipeline.list_runs()] == runs
    assert _tree(partial_adt) == before
    measured_cells = store.select_measured_cells(
        "ADT", cell_selection=rna_run["analysis_cell_selection"]
    )
    adt_run = store.pipeline.adt(cells=measured_cells, **_ADT)
    assert adt_run.status == "completed"
    np.testing.assert_array_equal(
        np.asarray(
            store.artifacts.load(adt_run["analysis_cell_selection"])["values"][:]
        ),
        np.asarray(store.cells.fetch_all("ADT_I"), dtype=bool),
    )


def test_cluster_tree_fill_values_show_unmeasured_cells_as_missing(measured) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    graph = measured.refs["legacy_graph"]
    clusters = store.clusters.paris(graph)

    prepared = store._prepare_cluster_tree(
        graph=graph, clusters=clusters, fill_by_value="RNA0"
    )

    values = np.asarray(prepared["color_values"])
    assert values.shape == (_N_CELLS,)
    assert np.isnan(values[~membership]).all()
    np.testing.assert_allclose(
        values[membership],
        fetch_normalized_feature_matrix(
            store, [resolve_feature(store, "RNA0")], np.flatnonzero(membership)
        )[:, 0],
        rtol=1e-12,
    )


def test_marker_heatmaps_leave_unmeasured_cells_out_of_group_means(measured) -> None:
    """A marker table deliberately computed over unmeasured cells."""
    plotting = _plotting()
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    labels = measured.refs["labels_all"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        legacy = store.markers.search(labels, features=measured.refs["rna_features"])

    result = plotting.marker_heatmap(
        store,
        marker=legacy,
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        show=False,
    )
    try:
        matrix = result.tables["matrix"]
        assert result.provenance.extras["unmeasured_cells"] == {"RNA": _N_ADT}
        # Group means average measured cells only; z-scores cannot show it,
        # as the unmeasured cells spread evenly over the programs.
        assert result.provenance.n_cells == _N_FULL
    finally:
        result.close()
    rows = np.flatnonzero(membership)
    groups = np.asarray(store.artifacts.load_values(labels).values).astype(str)[rows]
    for name in matrix.index:
        values = fetch_normalized_feature_matrix(
            store,
            [resolve_feature(store, name)],
            rows,
            NormalizationSpec(transform="log1p"),
        )[:, 0]
        means = pd.Series(values).groupby(groups).mean()
        expected = (means - means.mean()) / means.std()
        np.testing.assert_allclose(
            matrix.loc[name, expected.index.tolist()].to_numpy(dtype=float),
            expected.to_numpy(),
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    ("operation", "call", "error", "match"),
    [
        (
            "select_prevalent_peaks",
            lambda s, m: s.features.prevalent_peaks(m.refs["all"], top_n=5),
            TypeError,
            "ATACassay",
        ),
        (
            "run_cell_cycle_scoring",
            lambda s, m: s.scores.cell_cycle(
                m.refs["all"], from_assay="ATAC", **_CELL_CYCLE
            ),
            TypeError,
            "RNAassay",
        ),
        (
            "run_waggr",
            lambda s, m: s.scores.waggr(
                _NETWORK,
                m.refs["all"],
                from_assay="ATAC",
                features=m.refs["rna_features"],
            ),
            TypeError,
            "RNAassay",
        ),
        (
            "run_aucell",
            lambda s, m: s.scores.aucell(_NETWORK, m.refs["all"], features="RNA0"),
            TypeError,
            "features must be an ArtifactRef",
        ),
        (
            "select_detected_features",
            lambda s, m: s.features.detected(m.refs["all"], min_cells=-1),
            ValueError,
            "min_cells must be non-negative",
        ),
        (
            "run_normalization",
            lambda s, m: s.features.normalize(
                m.refs["all"], m.refs["rna_features"], log_transform="yes"
            ),
            TypeError,
            "log_transform",
        ),
    ],
)
def test_argument_and_assay_type_errors_come_before_the_membership_check(
    measured, operation, call, error, match
) -> None:
    """The membership check runs after an operation's own argument checks.

    ``select_prevalent_peaks`` with the default RNA assay named RNA's
    unmeasured cells instead of saying that it needs an ATAC assay.
    """
    with pytest.raises(error, match=match) as raised:
        call(measured.open(), measured)
    assert not isinstance(raised.value, UnmeasuredCellsError), operation


# Features that a source of a merge did not measure ---------------------------

_SHARED_ANTIBODIES = ["CD3", "CD4", "CD8", "CD19", "CD14", "CD16"]
_PANELS = {
    "left": [*_SHARED_ANTIBODIES, "CD45RA", "CD27"],
    "right": [*_SHARED_ANTIBODIES, "CD20"],
}
_N_STUDY = 150


@dataclass(frozen=True)
class _PanelStores:
    """A merge of two studies' ADT stores, beside the stores that it merged."""

    base: Path

    def open(self, name: str = "merged") -> DataStore:
        return DataStore(
            str(self.base / f"{name}.zarr"),
            default_assay="ADT",
            min_features_per_cell=0,
            nthreads=1,
        )


@pytest.fixture(scope="module")
def panels(tmp_path_factory) -> _PanelStores:
    """Merge two studies of T and B cells, in opposite proportions, with antibodies of their own.

    T cells count 200 of CD3 and B cells 200 of CD19 on average; every other
    count follows Poisson(10), those of CD45RA and CD27, which only the left
    study stained, and CD20, which only the right stained, included.
    """
    base = tmp_path_factory.mktemp("panels")
    rng = np.random.default_rng(0)
    for name, t_share in (("left", 0.8), ("right", 0.2)):
        panel = _PANELS[name]
        t_cell = rng.random(_N_STUDY) < t_share
        rates = np.full((_N_STUDY, len(panel)), 10.0)
        rates[t_cell, panel.index("CD3")] = 200
        rates[~t_cell, panel.index("CD19")] = 200
        write_count_store(
            str(base / f"{name}.zarr"),
            {"ADT": rng.poisson(rates)},
            "uint16",
            feature_ids={"ADT": panel},
        )
    stores = _PanelStores(base)
    # The intersection merge holds the antibodies that both studies stained,
    # for the cells of both, as an oracle of what a merge of different panels
    # gives them.
    for name, features in (("merged", "union"), ("intersection", "intersection")):
        DataStoreMerge(
            [stores.open(study) for study in _PANELS],
            str(base / f"{name}.zarr"),
            list(_PANELS),
            nthreads=1,
            features=features,
        ).dump()
    for name in ("intersection", "left"):
        own = stores.open(name)
        cd3 = own.ADT.rawData[:, [_SHARED_ANTIBODIES.index("CD3")]].compute()
        own.cells.insert(
            "cell_type",
            by_cell_id(own.cells, np.where(np.ravel(cd3) > 100, "T", "B")),
        )
    merged = stores.open()
    ids = merged.cells.fetch_all("ids").astype(str)
    left = np.char.startswith(ids, "left__")
    cd3 = np.asarray(merged.ADT.rawData[:, [_SHARED_ANTIBODIES.index("CD3")]].compute())
    merged.cells.insert("left_cells", by_cell_id(merged.cells, left))
    merged.cells.insert(
        "study", by_cell_id(merged.cells, np.where(left, "left", "right"))
    )
    merged.cells.insert(
        "cell_type", by_cell_id(merged.cells, np.where(cd3.ravel() > 100, "T", "B"))
    )
    # The cell types of the left study's cells, missing in the right study's.
    merged.cells.insert(
        "left_type",
        by_cell_id(
            merged.cells,
            np.where(left, np.where(cd3.ravel() > 100, "T", "B"), None),
        ),
    )
    # A live layout: x is the cell's row.
    merged.cells.insert(
        "xy1", by_cell_id(merged.cells, np.arange(len(ids), dtype=np.float64))
    )
    merged.cells.insert(
        "xy2", by_cell_id(merged.cells, (np.arange(len(ids)) % 7).astype(np.float64))
    )
    return stores


def _values_by_cell(
    store: DataStore, feature: str, merged_ids: np.ndarray
) -> np.ndarray:
    """Return one study's own display values of ``feature`` for merged cells of that study."""
    own = fetch_normalized_feature_matrix(
        store,
        [resolve_feature(store, FeatureRef(feature, assay="ADT"))],
        np.arange(store.cells.N),
    )[:, 0]
    by_id = dict(zip(store.cells.fetch_all("ids").astype(str), own, strict=True))
    return np.array([by_id[str(cell).split("__", 1)[1]] for cell in merged_ids])


def test_display_reads_show_a_feature_that_a_cells_source_did_not_measure_as_missing(
    panels,
) -> None:
    store = panels.open()
    ids = store.cells.fetch_all("ids").astype(str)
    left = np.char.startswith(ids, "left__")
    every = np.arange(len(ids))
    names = ("CD45RA", "CD3", "CD20")
    features = [resolve_feature(store, FeatureRef(name, assay="ADT")) for name in names]
    unmeasured: dict[str, np.ndarray] = {}

    matrix = fetch_normalized_feature_matrix(
        store, features, every, unmeasured=unmeasured
    )

    # Each study's own antibody is missing in the other study's cells, and its
    # cells read what a read of them alone gives: CLR over the study's cells.
    assert np.isnan(matrix[~left, 0]).all() and np.isnan(matrix[left, 2]).all()
    np.testing.assert_allclose(
        matrix[left, 0], _values_by_cell(panels.open("left"), "CD45RA", ids[left])
    )
    np.testing.assert_allclose(
        matrix[~left, 2], _values_by_cell(panels.open("right"), "CD20", ids[~left])
    )
    # Both studies measured CD3, whose CLR spans every cell, as the assay's own
    # normalization gives it.
    cd3 = int(store.ADT.feats.get_index_by(["CD3"], "names")[0])
    normed = store.ADT.normed(cell_idx=every, feat_idx=np.array([cd3]))
    np.testing.assert_allclose(
        matrix[:, 1], np.concatenate(list(normed.stream_blocks(nthreads=1)))[:, 0]
    )
    # The assay measured every cell, so no cell is unmeasured as a whole.
    assert unmeasured == {}
    streamed = np.full_like(matrix, -1.0)
    for slots, start, values in iter_normalized_feature_blocks(store, features, every):
        streamed[start : start + len(values), slots] = values
    np.testing.assert_array_equal(streamed, matrix)
    # Any order of cells reads the same values, to the rounding of the CLR's
    # sums, and the cells of the right study alone have no value of CD45RA.
    np.testing.assert_allclose(
        fetch_normalized_feature_matrix(store, features, every[::-1]),
        matrix[::-1],
        rtol=1e-12,
    )
    assert np.isnan(
        fetch_normalized_feature_matrix(store, features[:1], np.flatnonzero(~left))
    ).all()
    np.testing.assert_array_equal(
        store.get_cell_vals("ADT", "I", "CD45RA"), matrix[:, 0]
    )
    # The left study measured CD45RA and CD3, so a read of its cells reads both
    # in one pass and normalizes CD3 over them, as the study's own store does.
    left_rows = np.flatnonzero(left)
    blocks = list(iter_normalized_feature_blocks(store, features[:2], left_rows))
    assert {tuple(slots) for slots, _, _ in blocks} == {(0, 1)}
    np.testing.assert_allclose(
        fetch_normalized_feature_matrix(store, features[:2], left_rows),
        np.column_stack(
            [
                _values_by_cell(panels.open("left"), name, ids[left])
                for name in names[:2]
            ]
        ),
    )
    # A slot that averages a shared antibody with CD45RA has a value only where
    # both were measured.
    store.ADT.feats.insert(
        "pair",
        np.isin(store.ADT.feats.fetch_all("ids"), ["CD3", "CD45RA"]),
    )
    pair_indices = np.flatnonzero(store.ADT.feats.fetch_all("pair"))
    from cytearc.features.values import measured_feature_means

    means = measured_feature_means(store.ADT, pair_indices, every, nthreads=1)
    assert np.isnan(means[~left]).all()
    np.testing.assert_allclose(
        means[left],
        measured_feature_means(store.ADT, pair_indices, left_rows, nthreads=1),
    )


def test_plots_draw_a_feature_that_a_cells_source_did_not_measure_as_missing(
    panels,
) -> None:
    plotting = _plotting()
    import matplotlib

    store = panels.open()
    ids = store.cells.fetch_all("ids").astype(str)
    left = np.char.startswith(ids, "left__")
    own = _values_by_cell(panels.open("left"), "CD45RA", ids[left])

    layout = plotting.embedding(store, layout_key="xy", color_by="CD45RA", show=False)
    try:
        (axis,) = layout.axes.values()
        points = axis.collections[0]
        rows = np.asarray(points.get_offsets())[:, 0].astype(int)
        gray = np.all(
            np.isclose(
                points.get_facecolors(),
                matplotlib.colors.to_rgba(plotting.ColorScale().missing_color),
            ),
            axis=1,
        )
        # The right study's cells take the color of missing values, and the
        # color limits come from the left study's values alone.
        np.testing.assert_array_equal(gray, ~left[rows])
        assert layout.provenance.extras["color_limits"] == {
            "CD45RA": (float(own.min()), float(own.max()))
        }
    finally:
        layout.close()

    dots = plotting.dotplot(store, features=["CD45RA"], group_by="study", show=False)
    try:
        table = dots.tables["aggregate"].set_index("group")
        # The right study has no cell with a value of CD45RA to summarize.
        assert table.loc["right", "n_cells"] == 0
        assert np.isnan(table.loc["right", "mean"])
        assert table.loc["left", "n_cells"] == _N_STUDY
        np.testing.assert_allclose(table.loc["left", "mean"], own.mean())
    finally:
        dots.close()

    violins = plotting.distribution(
        store, "CD45RA", grouping=CellField("study"), show=False
    )
    try:
        values = violins.tables["CD45RA"]["value"].to_numpy()
        np.testing.assert_allclose(np.sort(values), np.sort(own))
    finally:
        violins.close()


def test_marker_search_ranks_the_features_that_every_selected_cells_source_measured(
    panels,
) -> None:
    store = panels.open()
    every = store.snapshot_cell_selection("I")
    labels = store.snapshot_cluster_labels("cell_type", cell_selection=every)
    universe = store.features.universe(from_assay="ADT")
    names = store.ADT.feats.fetch_all("names").astype(str)
    shared = np.isin(names, _SHARED_ANTIBODIES)

    with pytest.warns(UserWarning) as caught:
        ref = store.markers.search(labels, features=universe)
    assert [str(warning.message) for warning in caught] == [
        "The marker search leaves out 3 features that some source of the "
        "selected cells did not measure, since their zero counts in its cells "
        "are not measurements. Source 'left' lacks 1: 'CD20'. Source 'right' "
        "lacks 2: 'CD45RA', 'CD27'."
    ]
    # The search ranks the shared antibodies, as a search of them alone does.
    ranked = store.artifacts.inspect(ref).input_ref("feature_selection")
    assert read_feature_selection_indices(store.zw, "ADT", ranked).tolist() == (
        np.flatnonzero(shared).tolist()
    )
    alone = store.markers.search(
        labels,
        features=store.features.snapshot(
            from_assay="ADT", feature_indexes=np.flatnonzero(shared).tolist()
        ),
    )
    markers = store.markers.load(ref)
    pd.testing.assert_frame_equal(markers, store.markers.load(alone))
    best = markers.sort_values("score", ascending=False).groupby("group_id")
    assert dict(best["feature_name"].first()) == {"B": "CD19", "T": "CD3"}

    # The cells of the left study measured its own antibodies, which a search
    # of them keeps.
    left_labels = store.snapshot_cluster_labels(
        "cell_type", cell_selection=store.snapshot_cell_selection("left_cells")
    )
    with pytest.warns(
        UserWarning, match=r"measurements\. Source 'left' lacks 1: 'CD20'\.$"
    ):
        left_ref = store.markers.search(left_labels, features=universe)
    left_ranked = store.artifacts.inspect(left_ref).input_ref("feature_selection")
    assert (
        names[read_feature_selection_indices(store.zw, "ADT", left_ranked)].tolist()
        == _PANELS["left"]
    )

    # Labels that the search refuses leave no selection of measured features,
    # such as the CD3 and CD45RA of three antibodies that the left study's
    # cells would keep.
    one_study = store.snapshot_cluster_labels(
        "study", cell_selection=store.snapshot_cell_selection("left_cells")
    )
    trio = store.features.snapshot(
        from_assay="ADT",
        feature_indexes=np.flatnonzero(
            np.isin(names, ["CD3", "CD45RA", "CD20"])
        ).tolist(),
    )
    selections = len(
        list_artifacts(store.zw, scope="assay", assay="ADT", kind="feature_selection")
    )
    with pytest.raises(ValueError, match="at least two populated groups"):
        store.markers.search(one_study, features=trio)
    assert selections == len(
        list_artifacts(store.zw, scope="assay", assay="ADT", kind="feature_selection")
    )

    only_own = store.features.snapshot(
        from_assay="ADT",
        feature_indexes=np.flatnonzero(np.isin(names, ["CD45RA", "CD20"])).tolist(),
    )
    with pytest.raises(ValueError) as refused:
        store.markers.search(labels, features=only_own)
    assert str(refused.value) == (
        "The marker search would leave out every selected feature, since none was "
        "measured by every source of the selected cells. Source 'left' lacks 1: "
        "'CD20'. Source 'right' lacks 1: 'CD45RA'. Search the markers of the cells "
        "of the sources that measured them, or of features that every source "
        "measured."
    )


def test_analyses_refuse_a_feature_that_a_source_of_their_cells_did_not_measure(
    panels,
) -> None:
    store = panels.open()
    every = store.snapshot_cell_selection("I")
    remedy = (
        "Assay 'ADT' records that source 'right' did not measure 'CD45RA', so the "
        "zero counts of its cells are not measurements; "
    )
    with pytest.raises(ValueError) as refused:
        store.compare.test(["CD45RA"], CellField("cell_type"), skip_save=True)
    assert str(refused.value) == remedy + (
        "test the cells of the sources that measured it, with a cell_selection or "
        "subset_by that leaves the others out"
    )
    # Both studies measured CD3, and the left study CD45RA.
    left_cells = store.snapshot_cell_selection("left_cells")
    with pytest.warns(UserWarning, match="^Cell-level statistical testing"):
        # A metadata key reads no feature.
        store.compare.test(["CD3", "xy2"], CellField("cell_type"), skip_save=True)
        store.compare.test(["xy2"], CellField("cell_type"), skip_save=True)
        store.compare.test(
            ["CD45RA"],
            CellField("cell_type"),
            cell_selection=left_cells,
            skip_save=True,
        )
        # So do the cells that subset_by keeps.
        store.compare.test(
            ["CD45RA"], CellField("cell_type"), subset_by="left_cells", skip_save=True
        )

    # A graph needs the normalized values of antibodies that both studies
    # measured, which normalization refuses otherwise.
    shared = store.features.snapshot(
        from_assay="ADT",
        feature_indexes=np.flatnonzero(
            np.isin(store.ADT.feats.fetch_all("names"), _SHARED_ANTIBODIES)
        ).tolist(),
    )
    graph = _graph(store, "ADT", every, shared)
    diffusion = store.imputation.diffusion(graph)
    with pytest.raises(ValueError) as refused:
        store.imputation.compute_imputed(["CD3", "CD45RA"], diffusion)
    assert str(refused.value) == remedy + (
        "impute it over a diffusion operator of the cells of the sources that "
        "measured it"
    )
    assert np.isfinite(store.imputation.compute_imputed("CD3", diffusion)).all()


def test_each_cells_source_is_its_label_in_the_merges_column_of_sources(
    panels, monkeypatch
) -> None:
    store = panels.open()
    ids = store.cells.fetch_all("ids").astype(str)
    left = np.char.startswith(ids, "left__")
    # The merge stores each cell's source in the categorical column 'dataset',
    # which every column of the record names.
    column = store.cells._get_array("dataset")
    assert isinstance(column, CategoricalColumn)
    assert column.categories.tolist() == ["left", "right"]
    assert column.attrs["role"] == "cell_sources"
    np.testing.assert_array_equal(
        store.cells.fetch_all("dataset"), np.where(left, "left", "right")
    )
    record = measured_features_by_source(store.ADT.feats)
    assert record is not None
    assert (record.sources, record.sourceColumn) == (["left", "right"], "dataset")
    assert record.columns == ["measured_in_left", "measured_in_right"]
    expected = np.where(left, 0, 1)
    rows = np.arange(len(ids))
    np.testing.assert_array_equal(
        cell_sources(store.cells, store.ADT.feats, rows, record, assay="ADT"), expected
    )
    # Blocks of a few cells, in any order, give each cell the same source.
    monkeypatch.setattr(membership_module, "_SOURCE_ROWS", 7)
    picked = rows[::-3]
    np.testing.assert_array_equal(
        cell_sources(store.cells, store.ADT.feats, picked, record, assay="ADT"),
        expected[picked],
    )
    # The sources of a block of left cells and a block of right cells are both
    # counted, so only the shared antibodies count as measured.
    both = np.concatenate([np.flatnonzero(left)[:7], np.flatnonzero(~left)[:7]])
    measured = features_measured_by_cells(
        store.cells, store.ADT.feats, both, assay="ADT"
    )
    assert measured is not None
    assert (
        store.ADT.feats.fetch_all("names").astype(str)[measured].tolist()
        == _SHARED_ANTIBODIES
    )
    # The cells of the right study measured the shared antibodies and CD20,
    # and a label that reads CD3 and CD45RA needs both in every cell.
    right_rows = np.flatnonzero(~left)
    measured = features_measured_by_cells(
        store.cells, store.ADT.feats, right_rows, assay="ADT"
    )
    names = store.ADT.feats.fetch_all("names").astype(str)
    assert measured is not None and names[measured].tolist() == _PANELS["right"]
    both = np.flatnonzero(np.isin(names, ["CD3", "CD45RA"])).tolist()
    with pytest.raises(ValueError, match="^Assay 'ADT' records that source 'right'"):
        require_measured_features(
            store.cells,
            store.ADT.feats,
            rows,
            {"CD3 and CD45RA": both},
            assay="ADT",
            remedy="probe",
        )
    # A store without a record measured every feature in every cell.
    own = panels.open("left")
    assert measured_features_by_source(own.ADT.feats) is None
    assert (
        features_measured_by_cells(own.cells, own.ADT.feats, rows[:3], assay="ADT")
        is None
    )
    require_measured_features(
        own.cells,
        own.ADT.feats,
        rows[:3],
        {"CD45RA": [6]},
        assay="ADT",
        remedy="unused",
    )


def _copied_panels(panels: _PanelStores, tmp_path: Path) -> DataStore:
    """Return a copy of the merge of the panels' studies, which a test may change."""
    shutil.copytree(panels.base / "merged.zarr", tmp_path / "merged.zarr")
    return _PanelStores(tmp_path).open()


class _LatencyStore(WrapperStore):
    """A store whose every get waits ``delay`` seconds first, as a remote store's round trip does."""

    def __init__(self, store: Any, delay: float) -> None:
        super().__init__(store)
        self.delay = delay

    def with_read_only(self, read_only: bool = False) -> "_LatencyStore":
        return _LatencyStore(self._store.with_read_only(read_only), self.delay)

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        await asyncio.sleep(self.delay)
        return await self._store.get(key, prototype, byte_range)


def _record_lookup_seconds(location: Path, n_sources: int, delay: float) -> float:
    """Return how long a read of one feature's row of a record of ``n_sources`` sources takes when every get waits ``delay``."""
    names = ["CD3", "CD4", "CD8"]
    write_count_store(
        str(location),
        {"ADT": np.full((4, len(names)), 3)},
        "uint16",
        feature_ids={"ADT": names},
    )
    stored = zarr.open_group(str(location), mode="r+")
    for index in range(n_sources):
        source = f"s{index:02d}"
        create_metadata_column(
            stored["ADT/featureData"],
            f"measured_in_{source}",
            data=np.array([True, index % 2 == 0, True]),
            dtype=bool,
            attributes={
                "role": "measured_features",
                "source": source,
                "sourceColumn": "dataset",
            },
        )
    remote = zarr.open_group(
        store=_LatencyStore(LocalStore(str(location)), delay),
        mode="r",
        path="ADT/featureData",
    )
    table = MetaData(remote)
    start = time.perf_counter()
    record = measured_features_by_source(table, [1])
    elapsed = time.perf_counter() - start
    assert record is not None and len(record.sources) == n_sources
    np.testing.assert_array_equal(record.measured[:, 0], np.arange(n_sources) % 2 == 0)
    return elapsed


def test_a_record_of_many_sources_is_read_in_about_the_round_trips_of_few(
    tmp_path,
) -> None:
    # Each source's column costs a round trip to open and one to read. Read
    # one after another, 24 sources would take 12 times as long as 2.
    delay = 0.03
    few = _record_lookup_seconds(tmp_path / "few.zarr", 2, delay)
    many = _record_lookup_seconds(tmp_path / "many.zarr", 24, delay)
    assert few >= 2 * delay
    assert many < 3 * few, (few, many)


def test_a_record_without_its_cells_sources_is_refused_with_the_calls_that_drop_it(
    panels, tmp_path
) -> None:
    store = _copied_panels(panels, tmp_path)
    stored = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r+")
    for name in ("measured_in_left", "measured_in_right"):
        attributes = stored[f"ADT/featureData/{name}"].attrs
        attributes.put(
            {key: value for key, value in attributes.items() if key != "sourceColumn"}
        )
    store = _PanelStores(tmp_path).open()
    drop = (
        "Merge the sources again, or drop the record with "
        "ds.ADT.feats.drop('measured_in_left'), "
        "ds.ADT.feats.drop('measured_in_right'), after which analyses read the "
        "zero counts of a feature in the cells of a source that did not measure "
        "it as counts again."
    )
    with pytest.raises(ValueError) as refused:
        store.get_cell_vals("ADT", "I", "CD45RA")
    assert str(refused.value) == (
        "Assay 'ADT' records which features each source of a merge measured, in "
        "its feature columns 'measured_in_left', 'measured_in_right', but the "
        "cell column that names each cell's source, which a merge or a composite "
        "writes beside the record, is missing. " + drop
    )
    # A feature that every source measured needs no cell's source.
    np.testing.assert_array_equal(
        store.get_cell_vals("ADT", "I", "CD3"),
        panels.open().get_cell_vals("ADT", "I", "CD3"),
    )
    with pytest.warns(UserWarning, match="^Cell-level statistical testing"):
        store.compare.test(["CD3"], CellField("cell_type"), skip_save=True)
    with pytest.raises(ValueError, match="is missing. Merge the sources again"):
        store.compare.test(["CD45RA"], CellField("cell_type"), skip_save=True)
    # So does a marker search of the shared antibodies alone.
    names = store.ADT.feats.fetch_all("names").astype(str)
    labels = store.snapshot_cluster_labels(
        "cell_type", cell_selection=store.snapshot_cell_selection("I")
    )
    shared = store.features.snapshot(
        from_assay="ADT",
        feature_indexes=np.flatnonzero(np.isin(names, _SHARED_ANTIBODIES)).tolist(),
    )
    store.markers.search(labels, features=shared)
    with pytest.raises(ValueError, match="is missing. Merge the sources again"):
        store.markers.search(labels, features=store.features.universe(from_assay="ADT"))
    # Without the record, the zeros of the right study read as counts again.
    for name in ("measured_in_left", "measured_in_right"):
        store.ADT.feats.drop(name)
    assert np.isfinite(store.get_cell_vals("ADT", "I", "CD45RA")).all()


def test_each_group_of_a_display_read_charges_the_cells_it_holds(
    panels, monkeypatch
) -> None:
    from cytearc.assay import normalization

    store = panels.open()
    every = np.arange(store.cells.N)
    features = [
        resolve_feature(store, FeatureRef(name, assay="ADT"))
        for name in ("CD45RA", "CD3", "CD20")
    ]
    charged: list[int] = []
    read = normalization.feature_value_blocks

    def recorded(*args: Any, resident_bytes: int, **kwargs: Any) -> Any:
        charged.append(resident_bytes)
        return read(*args, resident_bytes=resident_bytes, **kwargs)

    monkeypatch.setattr(normalization, "feature_value_blocks", recorded)
    list(iter_normalized_feature_blocks(store, features, every))

    # Three groups: CD45RA over the 150 left cells, CD3 over all 300, CD20
    # over the 150 right cells. A group's read holds the other groups' masks
    # of one byte per cell, and a group of some cells also its own mask and
    # the 8-byte rows of its cells.
    masks = 2 * len(every)
    own = len(every) + 150 * 8
    assert charged == [masks - len(every) + own, masks, masks - len(every) + own]


def test_a_record_that_names_a_missing_cell_column_is_refused(panels, tmp_path) -> None:
    _copied_panels(panels, tmp_path)
    stored = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r+")
    for name in ("measured_in_left", "measured_in_right"):
        stored[f"ADT/featureData/{name}"].attrs.update({"sourceColumn": "gone"})
    store = _PanelStores(tmp_path).open()
    with pytest.raises(ValueError) as refused:
        store.get_cell_vals("ADT", "I", "CD45RA")
    assert str(refused.value).startswith(
        "Assay 'ADT' records which features each source of a merge measured, in "
        "its feature columns 'measured_in_left', 'measured_in_right', but the "
        "cell column 'gone' that names each cell's source, which a merge or a "
        "composite writes beside the record, is missing. Merge the sources again"
    )


def test_a_column_of_sources_without_its_role_is_refused(panels, tmp_path) -> None:
    store = _copied_panels(panels, tmp_path)
    codes = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r+")[
        "cellData/dataset"
    ]
    codes.attrs.put({key: value for key, value in codes.attrs.items() if key != "role"})
    store = _PanelStores(tmp_path).open()
    with pytest.raises(ValueError) as refused:
        store.get_cell_vals("ADT", "I", "CD45RA")
    assert str(refused.value).startswith(
        "Assay 'ADT' records which features each source of a merge measured, in "
        "its feature columns 'measured_in_left', 'measured_in_right', but the "
        "cell column 'dataset' that names each cell's source, which a merge or a "
        "composite writes beside the record, has no role 'cell_sources', which "
        "marks that column. Merge the sources again"
    )


def test_the_column_of_sources_is_the_cell_column_with_its_role(
    panels, tmp_path
) -> None:
    assert cell_sources_column(panels.open().cells) == "dataset"
    # A store that no merge wrote names no source.
    assert cell_sources_column(panels.open("left").cells) is None
    # A second column with the role leaves each cell's source unclear.
    store = _copied_panels(panels, tmp_path)
    store.cells.insert("batch", by_cell_id(store.cells, np.zeros(store.cells.N)))
    cells = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r+")["cellData"]
    cells["batch"].attrs.update({"role": "cell_sources"})
    with pytest.raises(ValueError) as refused:
        cell_sources_column(_PanelStores(tmp_path).open().cells)
    assert str(refused.value) == (
        "Cell columns 'batch', 'dataset' each hold the role 'cell_sources' of the "
        "column that names each cell's source, which a merge or a composite "
        "writes once; drop the columns that do not name the sources of this "
        "store's cells"
    )


def test_a_cell_without_a_source_is_refused(panels, tmp_path) -> None:
    from cytearc.storage.identity import replace_metadata_column

    store = _copied_panels(panels, tmp_path)
    cells = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r+")["cellData"]
    column = store.cells._get_array("dataset")
    codes = np.asarray(column.codes[:]).astype(np.int64)
    codes[0] = -1
    replace_metadata_column(cells, "dataset", codes, categories=column.categories)
    cells["dataset"].attrs.update({"role": "cell_sources"})
    store = _PanelStores(tmp_path).open()
    first = str(store.cells.fetch_all("ids")[0])
    with pytest.raises(ValueError) as refused:
        store.get_cell_vals("ADT", "I", "CD45RA")
    assert str(refused.value).startswith(
        f"Cell {first!r} has the source '' in cell column 'dataset', which is "
        "not one of the sources whose measured features assay 'ADT' records "
        "('left', 'right')."
    )


def test_a_merge_that_names_its_column_of_sources_says_why_a_name_collides(
    panels, tmp_path
) -> None:
    write_count_store(
        str(tmp_path / "third.zarr"),
        {"ADT": np.full((4, 2), 7)},
        "uint16",
        feature_ids={"ADT": ["CD3", "CD20"]},
    )
    stores = _PanelStores(tmp_path)
    # The merged store holds its own column 'dataset', which a merge without a
    # prefix copies under that name.
    for prefix in (None, ""):
        with pytest.raises(ValueError) as refused:
            DataStoreMerge(
                [panels.open(), stores.open("third")],
                str(tmp_path / "outer.zarr"),
                ["inner", "third"],
                prepend_text=prefix,
                nthreads=1,
            ).dump()
        assert str(refused.value) == (
            "The sources measured different features, so the merge writes each "
            "cell's source in the cell column 'dataset' beside its record of "
            "which features each source measured, but a source's own cell column "
            "takes that name. Pass source_column= another name for the column of "
            "sources, or a prepend_text that keeps the sources' own columns under "
            "other names, such as the default 'orig'."
        )
    DataStoreMerge(
        [panels.open(), stores.open("third")],
        str(tmp_path / "outer.zarr"),
        ["inner", "third"],
        prepend_text=None,
        source_column="study_of_cell",
        nthreads=1,
    ).dump()
    outer = stores.open("outer")
    assert measured_features_by_source(outer.ADT.feats).sourceColumn == "study_of_cell"


def test_a_cell_of_a_source_that_the_record_lacks_is_refused(panels, tmp_path) -> None:
    store = _copied_panels(panels, tmp_path)
    store.ADT.feats.drop("measured_in_right")
    with pytest.raises(ValueError) as refused:
        store.get_cell_vals("ADT", "I", "CD20")
    first_right = next(
        str(cell)
        for cell in store.cells.fetch_all("ids").astype(str)
        if cell.startswith("right__")
    )
    assert str(refused.value) == (
        f"Cell {first_right!r} has the source 'right' in cell column 'dataset', "
        "which is not one of the sources whose measured features assay 'ADT' "
        "records ('left'). Merge the sources again, or drop the record with "
        "ds.ADT.feats.drop('measured_in_left'), after which analyses read the "
        "zero counts of a feature in the cells of a source that did not measure "
        "it as counts again."
    )


def test_the_column_of_cells_sources_is_kept_while_a_record_names_it(
    panels, tmp_path
) -> None:
    store = _copied_panels(panels, tmp_path)
    for change in (
        lambda: store.cells.drop("dataset"),
        lambda: store.cells.insert(
            "dataset", by_cell_id(store.cells, np.zeros(store.cells.N)), overwrite=True
        ),
    ):
        with pytest.raises(
            ValueError,
            match="^Cell column 'dataset' names each cell's source for the records "
            r"of which features each source measured \(ADT/measured_in_left, "
            r"ADT/measured_in_right\)",
        ):
            change()
    # Other columns stay as free as before.
    store.cells.insert("dataset_2", by_cell_id(store.cells, np.zeros(store.cells.N)))
    store.cells.drop("dataset_2")
    for name in ("measured_in_left", "measured_in_right"):
        store.ADT.feats.drop(name)
    store.cells.drop("dataset")
    assert "dataset" not in store.cells.columns


def test_a_merge_names_its_column_of_sources_after_source_column(tmp_path) -> None:
    for name, panel in (
        ("a", ["CD3", "CD4"]),
        ("b", ["CD3", "CD8"]),
        ("c", ["CD3", "CD4"]),
    ):
        write_count_store(
            str(tmp_path / f"{name}.zarr"),
            {"ADT": np.full((3, len(panel)), 5)},
            "uint16",
            feature_ids={"ADT": panel},
        )
    stores = _PanelStores(tmp_path)

    def merged(names: list[str], destination: str, **settings: Any) -> DataStore:
        DataStoreMerge(
            [stores.open(name) for name in names],
            str(tmp_path / f"{destination}.zarr"),
            names,
            nthreads=1,
            **settings,
        ).dump()
        return stores.open(destination)

    # Panels that differ give the record a categorical column of sources,
    # named by source_column.
    named = merged(["a", "b"], "named", source_column="study")
    column = named.cells._get_array("study")
    assert isinstance(column, CategoricalColumn)
    assert column.attrs["role"] == "cell_sources"
    assert "dataset" not in named.cells.columns
    record = measured_features_by_source(named.ADT.feats)
    assert record is not None and record.sourceColumn == "study"
    # Panels that agree record nothing: source_column holds text, as before,
    # and without it the merge writes no column of sources.
    same = merged(["a", "c"], "same", source_column="study")
    assert not isinstance(same.cells._get_array("study"), CategoricalColumn)
    assert "role" not in same.cells._get_array("study").attrs
    assert (
        same.cells.fetch_all("study").tolist() == ["a"] * 3 + ["c"] * 3
        or sorted(same.cells.fetch_all("study").tolist()) == ["a"] * 3 + ["c"] * 3
    )
    assert "dataset" not in merged(["a", "c"], "plain").cells.columns


def test_a_merge_of_a_merged_store_keeps_the_source_of_each_of_its_cells(
    panels, tmp_path
) -> None:
    write_count_store(
        str(tmp_path / "third.zarr"),
        {"ADT": np.full((4, 2), 7)},
        "uint16",
        feature_ids={"ADT": ["CD3", "CD20"]},
    )
    DataStoreMerge(
        [panels.open(), _PanelStores(tmp_path).open("third")],
        str(tmp_path / "outer.zarr"),
        ["inner", "third"],
        nthreads=1,
    ).dump()
    outer = _PanelStores(tmp_path).open("outer")
    inner = panels.open()
    ids = outer.cells.fetch_all("ids").astype(str)
    # The oracle of each cell's source is the inner store's own column of
    # sources, read by cell ID.
    source_of = dict(
        zip(
            inner.cells.fetch_all("ids").astype(str),
            inner.cells.fetch_all("dataset").astype(str),
            strict=True,
        )
    )
    expected = np.array(
        [
            f"inner__{source_of[cell.split('__', 1)[1]]}"
            if cell.startswith("inner__")
            else "third"
            for cell in ids
        ]
    )
    assert {label: int((expected == label).sum()) for label in set(expected)} == {
        "inner__left": _N_STUDY,
        "inner__right": _N_STUDY,
        "third": 4,
    }
    # Each label, as each source's name, starts the IDs of its cells.
    assert all(
        cell.startswith(f"{label}__") for cell, label in zip(ids, expected, strict=True)
    )
    # The outer merge names each cell of the inner store by the inner
    # store's source of it, in its categorical column of sources; the inner
    # store's own column is still copied as an ordinary column of text.
    column = outer.cells._get_array("dataset")
    assert isinstance(column, CategoricalColumn)
    assert column.attrs["role"] == "cell_sources"
    assert column.categories.tolist() == ["inner__left", "inner__right", "third"]
    np.testing.assert_array_equal(outer.cells.fetch_all("dataset"), expected)
    copied = outer.cells._get_array("orig_dataset")
    assert not isinstance(copied, CategoricalColumn)
    assert "role" not in copied.attrs
    # Each of the inner store's sources keeps its own record column, renamed
    # to the outer name, and the third source gets its own.
    record = measured_features_by_source(outer.ADT.feats)
    assert record is not None
    assert record.sources == ["inner__left", "inner__right", "third"]
    assert record.columns == [
        "measured_in_inner__left",
        "measured_in_inner__right",
        "measured_in_third",
    ]
    names = outer.ADT.feats.fetch_all("names").astype(str)
    assert names.tolist() == [*_PANELS["left"], "CD20"]
    for study in ("left", "right"):
        own = dict(
            zip(
                inner.ADT.feats.fetch_all("names").astype(str),
                inner.ADT.feats.fetch_all(f"measured_in_{study}").tolist(),
                strict=True,
            )
        )
        assert [own[name] for name in names] == [
            name in _PANELS[study] for name in names
        ]
        assert outer.ADT.feats.fetch_all(f"measured_in_inner__{study}").tolist() == [
            own[name] for name in names
        ]
    assert outer.ADT.feats.fetch_all("measured_in_third").tolist() == [
        name in {"CD3", "CD20"} for name in names
    ]
    rows = np.arange(len(ids))
    positions = cell_sources(outer.cells, outer.ADT.feats, rows, record, assay="ADT")
    np.testing.assert_array_equal(np.asarray(record.sources)[positions], expected)
    # The cells of each study measured its own panel, as in the inner store.
    for study in ("left", "right"):
        measured = features_measured_by_cells(
            outer.cells,
            outer.ADT.feats,
            np.flatnonzero(expected == f"inner__{study}"),
            assay="ADT",
        )
        assert measured is not None and names[measured].tolist() == _PANELS[study]
    measured = features_measured_by_cells(
        outer.cells, outer.ADT.feats, rows, assay="ADT"
    )
    assert measured is not None and names[measured].tolist() == ["CD3"]
    # The left study's cells read their CD45RA values as the inner store and
    # the left study's own store read them; every other cell has none.
    values = outer.get_cell_vals("ADT", "I", "CD45RA")
    left = expected == "inner__left"
    assert np.isnan(values[~left]).all()
    np.testing.assert_allclose(
        values[left], _values_by_cell(inner, "CD45RA", ids[left])
    )
    inner_ids = np.array([cell.split("__", 1)[1] for cell in ids[left]])
    np.testing.assert_allclose(
        values[left], _values_by_cell(panels.open("left"), "CD45RA", inner_ids)
    )


def test_a_feature_of_an_assay_that_some_sources_lack_is_missing_where_either_says(
    tmp_path,
) -> None:
    rng = np.random.default_rng(3)
    panels = {"left": ["CD3", "CD45RA"], "right": ["CD3", "CD20"]}
    for name, panel in panels.items():
        write_count_store(
            str(tmp_path / f"{name}.zarr"),
            {
                "RNA": rng.integers(1, 9, size=(12, 3)),
                "ADT": rng.integers(1, 30, size=(12, len(panel))),
            },
            "uint32",
            feature_ids={"ADT": panel},
        )
    write_count_store(
        str(tmp_path / "rna.zarr"), {"RNA": rng.integers(1, 9, size=(6, 3))}, "uint32"
    )

    def opened(name: str) -> DataStore:
        return DataStore(
            str(tmp_path / f"{name}.zarr"),
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )

    # The source without ADT comes first, so the record's sources, which hold
    # the assay, are not the first categories of the column of sources.
    DataStoreMerge(
        [opened(name) for name in ("rna", "left", "right")],
        str(tmp_path / "merged.zarr"),
        ["rna", "left", "right"],
        nthreads=1,
    ).dump()
    store = opened("merged")
    ids = store.cells.fetch_all("ids").astype(str)
    left = np.char.startswith(ids, "left__")
    adt_cells = np.asarray(store.cells.fetch_all("ADT_I"), dtype=bool)
    assert int(adt_cells.sum()) == 24
    cd45ra = resolve_feature(store, FeatureRef("CD45RA", assay="ADT"))
    unmeasured: dict[str, np.ndarray] = {}

    values = fetch_normalized_feature_matrix(
        store, [cd45ra], np.arange(len(ids)), unmeasured=unmeasured
    )[:, 0]

    # The cells outside the assay and those of the right study have no value;
    # the left study's cells read what its own store gives.
    np.testing.assert_array_equal(unmeasured["ADT"], ~adt_cells)
    assert np.isnan(values[~left]).all()
    np.testing.assert_allclose(
        values[left], _values_by_cell(opened("left"), "CD45RA", ids[left])
    )
    # get_cell_vals reads the same values, membership and record together.
    np.testing.assert_array_equal(store.get_cell_vals("ADT", "I", "CD45RA"), values)


# Analyses and exports of features that some source of a merge did not
# measure (#552). Each oracle is a store that a different path writes: a
# source's own store, the intersection merge of the sources, which holds only
# the features that both measured, or the formula computed with NumPy from
# the merged counts.

_NORMALIZE_REMEDY = (
    "Normalize the cells of the sources that measured them, or a feature "
    "selection of only the features that every source of the selected cells "
    "measured."
)
_EXPORT_REMEDY = (
    "Export the cells of the sources that measured them, with cell_key or "
    "cell_selection, or only the features that every source of the exported "
    "cells measured, with feature_names or feature_indexes."
)
_PANEL_RECORD = (
    "The feature columns 'measured_in_left', 'measured_in_right' record which "
    "features each source measured."
)
_PANELS_LACK = (
    "Source 'left' lacks 1: 'CD20'. Source 'right' lacks 2: 'CD45RA', 'CD27'."
)


def _left_out(caught: Any) -> list[str]:
    """Return the warnings of analyses that leave out unmeasured features."""
    return [
        str(warning.message)
        for warning in caught
        if "leaves out" in str(warning.message)
    ]


def _feature_indexes(store: DataStore, assay: str, names: list[str]) -> list[int]:
    """Return the positions of ``names`` among the features of ``assay``, in the assay's order."""
    stored = store._get_assay(assay).feats.fetch_all("names").astype(str)
    return np.flatnonzero(np.isin(stored, names)).tolist()


def _normalized_frame(
    store: DataStore, ref: ArtifactRef, prefix: str = ""
) -> pd.DataFrame:
    """Return a normalized artifact by cell ID, ``prefix`` before each, and feature name."""
    status = store.artifacts.inspect(ref)
    rows = read_stored_selection_indices(
        store.zw,
        status.input_ref("cell_selection"),
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    columns = read_feature_selection_indices(
        store.zw, str(ref.assay), status.input_ref("feature_selection")
    )
    names = store._get_assay(str(ref.assay)).feats.fetch_all("names").astype(str)
    ids = store.cells.fetch_all("ids").astype(str)[rows]
    return pd.DataFrame(
        np.asarray(store.artifacts.load(ref)["data"][:]),
        index=[f"{prefix}{cell}" for cell in ids],
        columns=names[columns],
    )


def _exported_frame(adata: Any, prefix: str = "") -> pd.DataFrame:
    """Return an exported AnnData's X by cell ID, ``prefix`` before each, and feature name."""
    return pd.DataFrame(
        np.asarray(adata.X.toarray()),
        index=[f"{prefix}{cell}" for cell in adata.obs_names],
        columns=list(adata.var["names"]),
    )


def test_normalization_refuses_a_feature_that_a_source_of_its_cells_did_not_measure(
    panels,
) -> None:
    store = panels.open()
    every = store.snapshot_cell_selection("I")
    universe = store.features.universe(from_assay="ADT")

    def normalized() -> int:
        return len(
            list_artifacts(store.zw, scope="assay", assay="ADT", kind="normalized")
        )

    before = normalized()
    with pytest.raises(ValueError) as refused:
        store.features.normalize(every, universe)
    assert str(refused.value) == (
        "Assay 'ADT' records that some source of the selected cells did not "
        "measure 3 of the 9 selected features, so their zero counts in its cells "
        f"are not measurements. {_PANELS_LACK} {_PANEL_RECORD} {_NORMALIZE_REMEDY}"
    )
    assert normalized() == before
    # A read-only store refuses the same way, before it would refuse to write.
    read_only = DataStore(
        str(panels.base / "merged.zarr"),
        default_assay="ADT",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )
    with pytest.raises(ValueError, match="^Assay 'ADT' records that some source"):
        read_only.features.normalize(every, universe)

    # The left study did not measure CD20, which every other antibody of the
    # universe lacks in its cells alone.
    left_cells = store.snapshot_cell_selection("left_cells")
    with pytest.raises(ValueError) as refused:
        store.features.normalize(left_cells, universe)
    assert str(refused.value) == (
        "Assay 'ADT' records that some source of the selected cells did not "
        "measure 1 of the 9 selected features, so their zero counts in its cells "
        f"are not measurements. Source 'left' lacks 1: 'CD20'. {_PANEL_RECORD} "
        f"{_NORMALIZE_REMEDY}"
    )

    # The left study's panel over its cells normalizes as the left store does.
    left_panel = store.features.snapshot(
        from_assay="ADT",
        feature_indexes=_feature_indexes(store, "ADT", _PANELS["left"]),
    )
    own = panels.open("left")
    expected = _normalized_frame(
        own,
        own.features.normalize(
            own.snapshot_cell_selection("I"), own.features.universe(from_assay="ADT")
        ),
        prefix="left__",
    )
    found = _normalized_frame(store, store.features.normalize(left_cells, left_panel))
    pd.testing.assert_frame_equal(found.loc[expected.index, expected.columns], expected)
    np.testing.assert_allclose(found["CD45RA"].mean(), expected["CD45RA"].mean())

    # The antibodies that both studies stained, over every cell, normalize as
    # the intersection merge normalizes them.
    shared = store.features.snapshot(
        from_assay="ADT",
        feature_indexes=_feature_indexes(store, "ADT", _SHARED_ANTIBODIES),
    )
    intersection = panels.open("intersection")
    expected = _normalized_frame(
        intersection,
        intersection.features.normalize(
            intersection.snapshot_cell_selection("I"),
            intersection.features.universe(from_assay="ADT"),
        ),
    )
    found = _normalized_frame(store, store.features.normalize(every, shared))
    pd.testing.assert_frame_equal(found.loc[expected.index, expected.columns], expected)


def test_normed_exports_refuse_a_feature_that_a_source_of_their_cells_did_not_measure(
    panels,
) -> None:
    pytest.importorskip("anndata")
    store = panels.open()
    with pytest.raises(ValueError) as refused:
        store.to_anndata(from_assay="ADT", matrix="normed")
    assert str(refused.value) == (
        "Assay 'ADT' records that some source of the exported cells did not "
        "measure 3 of the 9 exported features, so their zero counts in its cells "
        f"are not measurements. {_PANELS_LACK} {_PANEL_RECORD} {_EXPORT_REMEDY}"
    )
    with pytest.raises(ValueError, match="Source 'left' lacks 1: 'CD20'"):
        store.to_anndata(from_assay="ADT", matrix="normed", cell_key="left_cells")

    # The left study's panel of its cells exports as the left store's own.
    found = _exported_frame(
        store.to_anndata(
            from_assay="ADT",
            matrix="normed",
            cell_key="left_cells",
            feature_names=_PANELS["left"],
        )
    )
    expected = _exported_frame(
        panels.open("left").to_anndata(from_assay="ADT", matrix="normed"),
        prefix="left__",
    )
    pd.testing.assert_frame_equal(found.loc[expected.index, expected.columns], expected)
    # The shared antibodies of every cell export as the intersection merge's.
    found = _exported_frame(
        store.to_anndata(
            from_assay="ADT", matrix="normed", feature_names=_SHARED_ANTIBODIES
        )
    )
    expected = _exported_frame(
        panels.open("intersection").to_anndata(
            from_assay="ADT", matrix="normed", feature_names=_SHARED_ANTIBODIES
        )
    )
    pd.testing.assert_frame_equal(found.loc[expected.index], expected)
    # Raw counts export every antibody and declare what each study measured.
    raw = store.to_anndata(from_assay="ADT")
    assert raw.shape == (2 * _N_STUDY, 9)
    assert raw.uns["cytearc"]["measuredFeatures"] == {
        "ADT": {"measured_in_left": "left", "measured_in_right": "right"}
    }


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_profiles_leave_out_features_that_a_source_of_their_cells_did_not_measure(
    panels, aggregation
) -> None:
    store = panels.open()
    options: dict[str, Any] = {
        "from_assay": "ADT",
        "aggr_type": aggregation,
        "count_detected": True,
        "feature_label": "name",
    }
    with pytest.warns(UserWarning) as caught:
        bulk = store.compare.compute_bulk("cell_type", **options)
    assert _left_out(caught) == [
        "compare.compute_bulk leaves out 3 features that some source of the cells "
        "that it reads did not measure, since their zero counts in its cells are "
        f"not measurements. {_PANELS_LACK}"
    ]
    # The shared antibodies' profiles are those of the intersection merge.
    expected = panels.open("intersection").compare.compute_bulk("cell_type", **options)
    pd.testing.assert_frame_equal(bulk.profiles, expected.profiles)
    assert expected.detected is not None
    pd.testing.assert_frame_equal(bulk.detected, expected.detected)
    pd.testing.assert_frame_equal(bulk.columns, expected.columns)
    # The profiles leave the unmeasured antibodies out with empty rows kept too.
    kept = store.compare.compute_bulk(
        "cell_type", remove_empty_features=False, **options
    )
    assert kept.profiles.index.tolist() == _SHARED_ANTIBODIES

    # The left study's cells keep its own antibodies, as its own store's
    # profiles do, and leave out CD20, which the left study did not stain.
    with pytest.warns(UserWarning) as caught:
        left = store.compare.compute_bulk(
            "cell_type",
            cell_selection=store.snapshot_cell_selection("left_cells"),
            **options,
        )
    assert _left_out(caught) == [
        "compare.compute_bulk leaves out 1 feature that some source of the cells "
        "that it reads did not measure, since their zero counts in its cells are "
        "not measurements. Source 'left' lacks 1: 'CD20'."
    ]
    own = panels.open("left").compare.compute_bulk("cell_type", **options)
    pd.testing.assert_frame_equal(left.profiles, own.profiles)
    assert own.detected is not None
    pd.testing.assert_frame_equal(left.detected, own.detected)

    # Groups of the left study's cells alone: sums read only those cells, and
    # keep the left study's antibodies, while a mean fits the normalization
    # over every selected cell, the right study's too.
    with pytest.warns(UserWarning) as caught:
        grouped = store.compare.compute_bulk("left_type", **options)
    lacking = "Source 'left' lacks 1: 'CD20'." if aggregation == "sum" else _PANELS_LACK
    assert [message.split("measurements. ", 1)[1] for message in _left_out(caught)] == [
        lacking
    ]
    assert grouped.profiles.index.tolist() == (
        _PANELS["left"] if aggregation == "sum" else _SHARED_ANTIBODIES
    )


_RNA_SHARED = [f"G{index}" for index in range(120)]
_RNA_LEFT_ONLY = [f"L{index}" for index in range(40)]
_PEAKS_SHARED = [f"chr1:{1000 * index}-{1000 * index + 500}" for index in range(60)]
_PEAKS_LEFT_ONLY = [f"chr2:{1000 * index}-{1000 * index + 500}" for index in range(20)]


@dataclass(frozen=True)
class _MergedSources:
    """A merge of a left source that measured features of its own and a right source, beside their own stores and their intersection merge."""

    base: Path
    assay: str

    def open(self, name: str = "merged") -> DataStore:
        return DataStore(
            str(self.base / f"{name}.zarr"),
            default_assay=self.assay,
            min_features_per_cell=0,
            nthreads=1,
        )

    def copy(self, tmp_path: Path) -> DataStore:
        """Return a copy of the merge, which a test may change."""
        shutil.copytree(self.base / "merged.zarr", tmp_path / "merged.zarr")
        return _MergedSources(tmp_path, self.assay).open()


def _merge_sources(
    base: Path,
    assay: str,
    counts: dict[str, np.ndarray],
    features: dict[str, list[str]],
) -> _MergedSources:
    """Write a store of ``counts`` for each source, then their union and intersection merges."""
    for name, values in counts.items():
        write_count_store(
            str(base / f"{name}.zarr"),
            {assay: values},
            "uint32",
            feature_ids={assay: features[name]},
        )
    stores = _MergedSources(base, assay)
    for name, kind in (("merged", "union"), ("intersection", "intersection")):
        DataStoreMerge(
            [stores.open(source) for source in counts],
            str(base / f"{name}.zarr"),
            list(counts),
            nthreads=1,
            features=kind,
        ).dump()
    for name in ("merged", "intersection"):
        store = stores.open(name)
        ids = store.cells.fetch_all("ids").astype(str)
        store.cells.insert(
            "study", by_cell_id(store.cells, np.char.partition(ids, "__")[:, 0])
        )
        for source in counts:
            store.cells.insert(
                f"{source}_cells",
                by_cell_id(store.cells, np.char.startswith(ids, f"{source}__")),
            )
    return stores


@pytest.fixture(scope="module")
def rna_sources(tmp_path_factory) -> _MergedSources:
    """Merge 200 cells of an RNA source with 40 genes of its own and 200 of a source of only the 120 shared genes.

    Ten shared genes mark each of two cell types, at eight times their base
    rate. The left source's own genes follow Poisson(2) in each of its cells,
    so in a merge of both sources only the right source's zeros spread them.
    """
    rng = np.random.default_rng(11)
    base_rates = rng.gamma(1.0, 2.0, size=len(_RNA_SHARED))
    counts: dict[str, np.ndarray] = {}
    for name in ("left", "right"):
        cell_type = rng.random(200) < 0.5
        rates = np.tile(base_rates, (200, 1))
        rates[cell_type, :10] *= 8
        rates[~cell_type, 10:20] *= 8
        counts[name] = rng.poisson(rates)
    counts["left"] = np.hstack(
        [counts["left"], rng.poisson(2.0, size=(200, len(_RNA_LEFT_ONLY)))]
    )
    return _merge_sources(
        tmp_path_factory.mktemp("rna_sources"),
        "RNA",
        counts,
        {"left": _RNA_SHARED + _RNA_LEFT_ONLY, "right": _RNA_SHARED},
    )


@pytest.fixture(scope="module")
def atac_sources(tmp_path_factory) -> _MergedSources:
    """Merge 150 cells of an ATAC source that called 20 peaks of its own, open in 60% of its cells, and 150 of a source of the 60 shared peaks."""
    rng = np.random.default_rng(12)
    counts = {
        name: rng.binomial(1, 0.15 + 0.1 * (np.arange(60) % 3 == 0), size=(150, 60))
        for name in ("left", "right")
    }
    counts["left"] = np.hstack(
        [counts["left"], rng.binomial(1, 0.6, size=(150, len(_PEAKS_LEFT_ONLY)))]
    )
    return _merge_sources(
        tmp_path_factory.mktemp("atac_sources"),
        "ATAC",
        counts,
        {"left": _PEAKS_SHARED + _PEAKS_LEFT_ONLY, "right": _PEAKS_SHARED},
    )


def _dense_counts(store: DataStore, assay: str) -> np.ndarray:
    return np.asarray(store._get_assay(assay).rawData[:, :].compute(), dtype=np.float64)


def _measured_gene_scores(
    flavor: str, counts: np.ndarray, genes: np.ndarray
) -> np.ndarray:
    """Score the columns ``genes`` of ``counts`` over every row, each cell's total taken over all of its genes.

    cytearc divides the variance of counts scaled to 1,000 by the trend of
    genes with a similar mean; Pearson residuals follow Lause, Berens and
    Kobak (2021): ``mu = n_c p_g``, each gene's share of every count, with
    theta 100, clipped at the square root of the cells.
    """
    totals = counts.sum(axis=1)
    kept = counts[:, genes]
    if flavor == "cytearc":
        values = 1000 * kept / np.where(totals == 0, 1, totals)[:, None]
        average = values.mean(axis=0)
        positive = average > 0
        scores = np.zeros(kept.shape[1])
        scores[positive] = fit_lowess(
            average[positive], values.var(axis=0)[positive], 200, 0.1
        )
        return scores
    expected = totals[:, None] * (kept.sum(axis=0) / totals.sum())[None, :]
    residuals = np.zeros_like(kept)
    positive = expected > 0
    residuals[positive] = (kept - expected)[positive] / np.sqrt(
        expected + expected**2 / 100.0
    )[positive]
    clip = np.sqrt(len(counts))
    return np.clip(residuals, -clip, clip).var(axis=0)


def _top(scores: np.ndarray, candidates: np.ndarray, top_n: int) -> np.ndarray:
    """Return the mask of the ``top_n`` candidates by score, ties to the lower index."""
    index = np.flatnonzero(candidates)
    selected = np.zeros(len(scores), dtype=bool)
    selected[index[np.lexsort((index, -scores[index]))][:top_n]] = True
    return selected


def _selected_names(store: DataStore, ref: ArtifactRef) -> list[str]:
    names = store._get_assay(str(ref.assay)).feats.fetch_all("names").astype(str)
    return names[read_feature_selection_indices(store.zw, str(ref.assay), ref)].tolist()


_HVG_OPTIONS: dict[str, Any] = {
    "top_n": 30,
    "blacklist": "",
    "min_cells": 0,
    "max_cells": np.inf,
    "show_plot": False,
}
_RNA_LACK = "Source 'right' lacks 40: 'L0', 'L1', 'L2', 'L3', 'L4' and 35 more."


@pytest.mark.parametrize("flavor", ["cytearc", "seurat_v3", "pearson_residuals"])
def test_hvgs_leave_out_genes_that_a_source_of_their_cells_did_not_measure(
    rna_sources, flavor
) -> None:
    store = rna_sources.open()
    names = store.RNA.feats.fetch_all("names").astype(str)
    shared = np.isin(names, _RNA_SHARED)
    options = {**_HVG_OPTIONS, "flavor": flavor}
    if flavor == "seurat_v3":
        # seurat_v3 takes no variance bounds and no max_cells of np.inf.
        options.pop("max_cells")

    with pytest.warns(UserWarning) as caught:
        ref = store.features.hvgs(store.snapshot_cell_selection("I"), **options)
    assert _left_out(caught) == [
        "features.hvgs leaves out 40 genes that some source of the selected cells "
        "did not measure, since their zero counts in its cells are not "
        f"measurements. {_RNA_LACK}"
    ]
    group = store.artifacts.load(ref)
    selected = np.asarray(group["values"][:], dtype=bool)
    scores = np.asarray(group["corrected_variance"][:])
    # The left source's own genes take no part and score 0, as genes without
    # counts do; the selection records which genes both sources measured.
    assert not selected[~shared].any() and (scores[~shared] == 0).all()
    assert store.artifacts.inspect(ref).parameters[
        "measured_features_fingerprint"
    ] == fingerprint_array(shared)
    if flavor == "seurat_v3":
        # Raw counts alone set seurat_v3's scores, so the shared genes score
        # as in the intersection merge, whose trend holds them alone.
        intersection = rna_sources.open("intersection")
        expected = intersection.features.hvgs(
            intersection.snapshot_cell_selection("I"), **options
        )
        np.testing.assert_allclose(
            scores[shared],
            intersection.artifacts.load(expected)["corrected_variance"][:],
            rtol=1e-10,
        )
        assert _selected_names(store, ref) == _selected_names(intersection, expected)
    else:
        expected_scores = _measured_gene_scores(
            flavor, _dense_counts(store, "RNA"), np.flatnonzero(shared)
        )
        np.testing.assert_allclose(scores[shared], expected_scores, rtol=1e-8)
        np.testing.assert_array_equal(
            selected[shared], _top(expected_scores, np.ones(shared.sum(), bool), 30)
        )
    # The detection ratio compares the selected genes with the genes that the
    # selection could take, which leave out the left source's own.
    detected = (_dense_counts(store, "RNA") > 0).sum(axis=0)
    eligible = shared & (detected > 0)
    assert store.artifacts.inspect(ref).diagnostics[
        "selectedDetectionRatio"
    ] == pytest.approx(np.median(detected[selected]) / np.median(detected[eligible]))
    # A top_n above the shared genes selects every shared gene and no other.
    with pytest.warns(UserWarning, match="features.hvgs leaves out 40 genes"):
        every_gene = store.features.hvgs(
            store.snapshot_cell_selection("I"), **{**options, "top_n": 150}
        )
    assert _selected_names(store, every_gene) == _RNA_SHARED

    # Each source's cells select what the source's own store selects: the
    # right source lacks the left genes, which the left source measured.
    for source, warned in (("right", [_RNA_LACK]), ("left", [])):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = store.features.hvgs(
                store.snapshot_cell_selection(f"{source}_cells"), **options
            )
        assert [message.rsplit(". ", 1)[-1] for message in _left_out(caught)] == [
            line.rsplit(". ", 1)[-1] for line in warned
        ]
        own = rna_sources.open(source)
        expected = own.features.hvgs(own.snapshot_cell_selection("I"), **options)
        assert _selected_names(store, found) == _selected_names(own, expected)
        own_scores = np.asarray(own.artifacts.load(expected)["corrected_variance"][:])
        positions = _feature_indexes(
            store, "RNA", own.RNA.feats.fetch_all("names").astype(str).tolist()
        )
        np.testing.assert_allclose(
            np.asarray(store.artifacts.load(found)["corrected_variance"][:])[positions],
            own_scores,
            rtol=1e-6,
        )


def test_batch_hvgs_leave_out_genes_that_a_source_of_their_cells_did_not_measure(
    rna_sources,
) -> None:
    store = rna_sources.open()
    intersection = rna_sources.open("intersection")
    options = {**_HVG_OPTIONS, "flavor": "seurat_v3", "batch": "study"}
    options.pop("max_cells")
    with pytest.warns(UserWarning, match="features.hvgs leaves out 40 genes"):
        ref = store.features.hvgs(store.snapshot_cell_selection("I"), **options)
    expected = intersection.features.hvgs(
        intersection.snapshot_cell_selection("I"), **options
    )
    # Each batch is a source, and each scores the shared genes as the
    # intersection merge's batch does; the left batch's own genes are no
    # candidate, though that batch measured them.
    assert _selected_names(store, ref) == _selected_names(intersection, expected)
    shared = np.isin(store.RNA.feats.fetch_all("names").astype(str), _RNA_SHARED)
    for name in ("corrected_variance", "batch_count"):
        np.testing.assert_allclose(
            np.asarray(store.artifacts.load(ref)[name][:])[shared],
            intersection.artifacts.load(expected)[name][:],
            rtol=1e-10,
        )


def test_prevalent_peaks_leave_out_peaks_that_a_source_of_their_cells_did_not_measure(
    atac_sources,
) -> None:
    store = atac_sources.open()
    names = store.ATAC.feats.fetch_all("names").astype(str)
    shared = np.isin(names, _PEAKS_SHARED)
    with pytest.warns(UserWarning) as caught:
        ref = store.features.prevalent_peaks(
            store.snapshot_cell_selection("I"), from_assay="ATAC", top_n=30
        )
    assert _left_out(caught) == [
        "features.prevalent_peaks leaves out 20 peaks that some source of the "
        "selected cells did not measure, since their zero counts in its cells are "
        "not measurements. Source 'right' lacks 20: 'chr2:0-500', 'chr2:1000-1500', "
        "'chr2:2000-2500', 'chr2:3000-3500', 'chr2:4000-4500' and 15 more."
    ]
    assert store.artifacts.inspect(ref).parameters == {
        "top_n": 30,
        "measured_features_fingerprint": fingerprint_array(shared),
    }
    # TF-IDF prevalence: each cell's counts over its total, summed over the
    # cells and weighted by log2(1 + n / (df + 1)).
    counts = _dense_counts(store, "ATAC")
    totals = counts.sum(axis=1)
    frequency = (counts > 0).sum(axis=0)
    prevalence = (counts / np.where(totals == 0, 1, totals)[:, None]).sum(
        axis=0
    ) * np.log2(1 + len(counts) / (frequency + 1))
    ranked = np.flatnonzero(shared)[np.argsort(-prevalence[shared], kind="stable")]
    assert _selected_names(store, ref) == names[np.sort(ranked[:30])].tolist()
    # A top_n above the shared peaks selects every shared peak and no other.
    with pytest.warns(UserWarning, match="features.prevalent_peaks leaves out 20"):
        every_peak = store.features.prevalent_peaks(
            store.snapshot_cell_selection("I"), from_assay="ATAC", top_n=70
        )
    assert _selected_names(store, every_peak) == _PEAKS_SHARED

    # Each source's cells select what its own store selects.
    for source in ("right", "left"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = store.features.prevalent_peaks(
                store.snapshot_cell_selection(f"{source}_cells"),
                from_assay="ATAC",
                top_n=30,
            )
        assert len(_left_out(caught)) == (source == "right")
        own = atac_sources.open(source)
        expected = own.features.prevalent_peaks(
            own.snapshot_cell_selection("I"), from_assay="ATAC", top_n=30
        )
        assert _selected_names(store, found) == _selected_names(own, expected)


def _network() -> pd.DataFrame:
    """Return a network of a set of the left source's own genes and a set of marker genes."""
    return pd.DataFrame(
        {
            "source": ["left_genes"] * 10 + ["type_a"] * 10,
            "target": _RNA_LEFT_ONLY[:10] + _RNA_SHARED[:10],
            "weight": 1.0,
        }
    )


def _scores_frame(store: DataStore, ref: ArtifactRef, prefix: str = "") -> pd.DataFrame:
    result = store.scores.load(ref)
    ids = store.cells.fetch_all("ids").astype(str)[np.asarray(result.cell_index)]
    return pd.DataFrame(
        np.asarray(result.data.compute()),
        index=[f"{prefix}{cell}" for cell in ids],
        columns=[str(name) for name in result.source_names],
    )


@pytest.mark.parametrize("method", ["waggr", "aucell"])
def test_scores_leave_out_genes_that_a_source_of_their_cells_did_not_measure(
    rna_sources, method
) -> None:
    store = rna_sources.open()
    score = getattr(store.scores, method)
    every = store.snapshot_cell_selection("I")
    universe = store.features.universe(from_assay="RNA")
    label = {"waggr": "WAGGR", "aucell": "AUCell"}[method]
    with pytest.warns(UserWarning) as caught:
        ref = score(_network(), every, features=universe, from_assay="RNA")
    assert _left_out(caught) == [
        f"{label} leaves out 40 features that some source of the selected cells "
        "did not measure, since their zero counts in its cells are not "
        f"measurements. {_RNA_LACK}"
    ]
    # The left genes' set keeps none of its targets, and the scores read the
    # selection of the shared genes, which their lineage records.
    found = _scores_frame(store, ref)
    assert found.columns.tolist() == ["type_a"]
    assert _selected_names(store, store.scores.load(ref).feature_selection) == (
        _RNA_SHARED
    )
    if method == "aucell":
        # Ranks of raw counts over the same genes, in the same order, are
        # those of the intersection merge.
        intersection = rna_sources.open("intersection")
        expected = _scores_frame(
            intersection,
            intersection.scores.aucell(
                _network(),
                intersection.snapshot_cell_selection("I"),
                features=intersection.features.universe(from_assay="RNA"),
                from_assay="RNA",
            ),
        )
        pd.testing.assert_frame_equal(found.loc[expected.index], expected)
    else:
        # The weighted mean of counts scaled to 1,000 over each cell's total.
        counts = _dense_counts(store, "RNA")
        targets = _feature_indexes(store, "RNA", _RNA_SHARED[:10])
        totals = counts.sum(axis=1)
        values = 1000 * counts[:, targets] / np.where(totals == 0, 1, totals)[:, None]
        np.testing.assert_allclose(
            found["type_a"].to_numpy(), values.mean(axis=1), rtol=1e-6
        )

    # Each source's cells score as its own store scores them.
    for source in ("right", "left"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = _scores_frame(
                store,
                score(
                    _network(),
                    store.snapshot_cell_selection(f"{source}_cells"),
                    features=universe,
                    from_assay="RNA",
                ),
            )
        assert len(_left_out(caught)) == (source == "right")
        own = rna_sources.open(source)
        expected = _scores_frame(
            own,
            getattr(own.scores, method)(
                _network(),
                own.snapshot_cell_selection("I"),
                features=own.features.universe(from_assay="RNA"),
                from_assay="RNA",
            ),
            prefix=f"{source}__",
        )
        pd.testing.assert_frame_equal(found.loc[expected.index], expected)

    # A selection of only the left source's own genes leaves no gene to score.
    left_genes = store.features.snapshot(
        from_assay="RNA", feature_indexes=_feature_indexes(store, "RNA", _RNA_LEFT_ONLY)
    )
    with pytest.raises(ValueError) as refused:
        score(_network(), every, features=left_genes, from_assay="RNA")
    assert str(refused.value) == (
        f"{label} would leave out every selected feature, since none was measured "
        f"by every source of the selected cells. {_RNA_LACK} Score the cells of "
        "the sources that measured them, or features that every source measured."
    )


def test_selections_record_the_measured_features_only_when_the_record_changes_them(
    rna_sources, atac_sources, tmp_path
) -> None:
    def hvgs(store: DataStore, cells: ArtifactRef) -> ArtifactRef:
        return store.features.hvgs(cells, top_n=30, show_plot=False)

    def peaks(store: DataStore, cells: ArtifactRef) -> ArtifactRef:
        return store.features.prevalent_peaks(cells, from_assay="ATAC", top_n=30)

    for sources, select, folder in (
        (rna_sources, hvgs, "rna"),
        (atac_sources, peaks, "atac"),
    ):
        store = sources.copy(tmp_path / folder)
        assay = store._get_assay(sources.assay)
        every = store.snapshot_cell_selection("I")
        left = store.snapshot_cell_selection("left_cells")
        # The left source measured every feature, so the record leaves none
        # of them out of a selection of its cells, which records nothing.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            measured_by_left = select(store, left)
        assert _left_out(caught) == []
        parameters = store.artifacts.inspect(measured_by_left).parameters
        assert "measured_features_fingerprint" not in parameters
        with pytest.warns(UserWarning, match="leaves out"):
            measured_by_both = select(store, every)
        assert (
            "measured_features_fingerprint"
            in store.artifacts.inspect(measured_by_both).parameters
        )
        assay.feats.drop("measured_in_left")
        assay.feats.drop("measured_in_right")
        # Without the record, the left cells' selection is the same artifact,
        # and the selection of every cell is computed again over every
        # feature, which it records no measured features of.
        assert select(store, left) == measured_by_left
        again = select(store, every)
        assert again != measured_by_both
        assert (
            "measured_features_fingerprint"
            not in store.artifacts.inspect(again).parameters
        )

    # A recorded fingerprint must be one.
    store = rna_sources.copy(tmp_path / "corrupt")
    ref = store.features.hvgs(store.snapshot_cell_selection("I"), **_HVG_OPTIONS)
    group = zarr.open_group(store.zw.store, mode="r+", path=artifact_path(ref))
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = {
        **provenance["parameters"],
        "measured_features_fingerprint": "not a fingerprint",
    }
    group.attrs["provenance"] = provenance
    with pytest.raises(
        ArtifactResolutionError, match="Fingerprint of the measured features"
    ):
        store.features.resolve("RNA", ref)

    # A score of cells whose sources measured every feature records the
    # selection that it was given.
    store = rna_sources.open()
    universe = store.features.universe(from_assay="RNA")
    ref = store.scores.waggr(
        _network(),
        store.snapshot_cell_selection("left_cells"),
        features=universe,
        from_assay="RNA",
    )
    assert store.scores.load(ref).feature_selection == universe


@pytest.mark.slow
def test_rna_and_atac_runs_select_only_features_that_every_source_measured(
    rna_sources, atac_sources, tmp_path
) -> None:
    store = rna_sources.copy(tmp_path / "rna")
    hvg = {"top_n": 30, "min_cells": 1, "max_cells": 401}
    with pytest.warns(UserWarning, match="features.hvgs leaves out 40 genes"):
        run = store.pipeline.rna(**{**_PIPELINE, "hvg": hvg, "leiden": False})
    genes = _selected_names(store, run["highly_variable_features"])
    assert genes and set(genes) <= set(_RNA_SHARED)
    assert genes == _selected_names(
        store,
        store.features.hvgs(store.snapshot_cell_selection("I"), show_plot=False, **hvg),
    )
    assert _normalized_frame(store, run["normalized"]).columns.tolist() == genes

    store = atac_sources.copy(tmp_path / "atac")
    with pytest.warns(UserWarning, match="features.prevalent_peaks leaves out 20"):
        run = store.pipeline.atac(**{**_ATAC, "prevalent_peaks": {"top_n": 30}})
    peaks = _selected_names(store, run["prevalent_peaks"])
    assert len(peaks) == 30 and set(peaks) <= set(_PEAKS_SHARED)
    assert _normalized_frame(store, run["normalized"]).columns.tolist() == peaks
