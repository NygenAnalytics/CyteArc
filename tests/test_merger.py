import re

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.errors import GroupNotFoundError
from zarr.storage import MemoryStore

import cytearc.merge.datasets as merge_datasets
from cytearc.matrix import ChunkedArray
from cytearc.metadata import MetaData
from cytearc.merge import DataStoreMerge
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPolicy,
    load_count_matrix_plan,
    policy_from_payload,
)
from cytearc.storage.layout import count_array_spec
from cytearc.utils.memory_budget import MemoryBudgetError


class _MergeMeta:
    def __init__(self, *, block_rows=None, **columns):
        self._columns = {key: np.asarray(value) for key, value in columns.items()}
        self.columns = list(self._columns)
        self.N = len(next(iter(self._columns.values())))
        self._blockRows = self.N if block_rows is None else int(block_rows)
        # The table's group, whose attributes record no cell order.
        self._group = zarr.open_group(store=MemoryStore(), mode="w")

    def to_pandas_dataframe(self, columns):
        return pd.DataFrame({key: self._columns[key] for key in columns})

    def fetch_all(self, key):
        return self._columns[key]

    def get_dtype(self, key):
        return self._columns[key].dtype

    def _get_array(self, key):
        return self._columns[key]

    def default_block_rows(self, column="I"):
        _ = column
        return max(1, min(self.N, self._blockRows))


class _MergeAssay:
    def __init__(
        self,
        name,
        counts,
        cell_ids,
        feature_ids,
        feature_names,
        block_size,
        *,
        assay_type=None,
    ):
        from cytearc.assay import resolve_persisted_assay_type

        self.name = name
        # An open source carries the type its DataStore resolved for the assay.
        self.assayType = resolve_persisted_assay_type(name, assay_type)
        self.rawData = ChunkedArray.from_numpy(
            np.asarray(counts),
            block_size=block_size,
        )
        self.cells = _MergeMeta(
            block_rows=block_size,
            ids=cell_ids,
            names=cell_ids,
            I=np.ones(len(cell_ids), dtype=bool),
        )
        self.feats = _MergeMeta(ids=feature_ids, names=feature_names)
        # An open source of counts carries no normalizedValues record, and
        # its group no attribute of melded values.
        self.normalizedValues = None
        self.attrs: dict[str, object] = {}
        self._matrixGroup = None
        self._matrixGroupCounts = None

    @property
    def matrixGroup(self):
        # A prepared assay's saved count summaries match its current counts,
        # so they are rebuilt only when a test replaces the counts.
        if self._matrixGroupCounts is not self.rawData:
            from tests.storage_helpers import finalize_test_counts

            group = zarr.open_group(store=MemoryStore(), mode="w")
            finalize_test_counts(
                group.create_array("counts", data=np.asarray(self.rawData.compute()))
            )
            self._matrixGroup = group
            self._matrixGroupCounts = self.rawData
        return self._matrixGroup

    def _percent_features(self):
        return {}


class _MergeDataStore:
    def __init__(self, assays, *, zarr_loc="memory://test"):
        self._assays = {assay.name: assay for assay in assays}
        self.assay_names = list(self._assays)
        self.cells = assays[0].cells
        self.nthreads = 2
        self.memoryBytes = 1024**3
        self.resources = ResourceBudget(self.memoryBytes, self.nthreads)
        self.zarr_loc = zarr_loc
        # A merge reads a source's root group for pending derived assays.
        self.z = zarr.open_group(store=MemoryStore(), mode="w")
        self.workspace = None

    def get_assay(self, name):
        return self._assays[name]


class _NoFullReadMeta(_MergeMeta):
    def fetch_all(self, key):
        raise AssertionError(f"full metadata read attempted for {key}")


# Merged rows of _merge_two_rna, by merged cell ID.
_TWO_RNA_ROWS = {
    "left__c0": [1, 10],
    "left__c1": [2, 20],
    "right__c0": [3, 30],
    "right__c1": [4, 40],
}


def _rows_by_id(
    root, path: str = "RNA/counts", ids: str = "cellData/ids"
) -> dict[str, list]:
    """Return the stored rows of ``path`` keyed by merged cell ID."""
    cell_ids = np.asarray(root[ids][:]).astype(str)
    rows = np.asarray(root[path][:])
    return {cell: row.tolist() for cell, row in zip(cell_ids, rows, strict=True)}


def _merge_two_rna(**kwargs):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    defaults = {
        "datasets": [left, right],
        "names": ["left", "right"],
        "prepend_text": "",
        "overwrite": True,
        "seed": 0,
    }
    defaults.update(kwargs)
    return DataStoreMerge(**defaults)


def _with_partial_quality(merger):
    """Give the left source a quality column that the right source lacks."""
    merger.datasets[0].cells._columns["quality"] = np.array([1, 2])
    merger.datasets[0].cells.columns.append("quality")
    return merger


_SHARDED_POLICY = CountMatrixPolicy(unitBytes=32, chunkBytes=16)


@pytest.fixture(scope="module")
def merged_two_rna(tmp_path_factory):
    """Completed merges of _merge_two_rna that tests copy before changing them.

    ``default`` uses the default options, ``sharded`` a policy of several
    shards, and ``quality`` a left source with a partial quality column.
    """
    root = tmp_path_factory.mktemp("merged_two_rna")
    templates = {}
    for variant, options in (("default", {}), ("sharded", {"policy": _SHARDED_POLICY})):
        templates[variant] = root / f"{variant}.zarr"
        _merge_two_rna(
            zarr_path=str(templates[variant]), overwrite=False, **options
        ).dump()
    templates["quality"] = root / "quality.zarr"
    _with_partial_quality(
        _merge_two_rna(zarr_path=str(templates["quality"]), overwrite=False)
    ).dump()
    return templates


def _copy_merge(templates, variant: str, destination) -> str:
    """Copy a completed merge to ``destination`` and return its location."""
    import shutil

    shutil.copytree(templates[variant], destination)
    return str(destination)


def test_merged_case_distinct_ids_resolve_and_fetch_exactly():
    from cytearc.datastore.datastore import DataStore
    from cytearc.features.values import fetch_normalized_feature_matrix, resolve_feature
    from cytearc.metadata.selection import FeatureRef, NormalizationSpec

    sources = []
    for label, symbol, counts in [
        ("left", "GeneA", [[5, 2]]),
        ("right", "genea", [[7, 3]]),
    ]:
        sources.append(
            _MergeDataStore(
                [
                    _MergeAssay(
                        "RNA",
                        np.array(counts, dtype=np.uint32),
                        [label],
                        [symbol, "shared"],
                        [symbol, "shared"],
                        block_size=1,
                    )
                ],
                zarr_loc=f"memory://{label}",
            )
        )
    output = MemoryStore()
    DataStoreMerge(
        sources, output, ["left", "right"], seed=None, nthreads=1, profile="fast_local"
    ).dump()
    merged = DataStore(
        output,
        default_assay="RNA",
        min_features_per_cell=0,
        mito_pattern="",
        ribo_pattern="",
        nthreads=1,
        zarrProfile="fast_local",
    )

    resolved = [
        resolve_feature(merged, FeatureRef(value, by="id", reduction="sum"))
        for value in ("GeneA", "genea")
    ]
    assert [feature.ids for feature in resolved] == [("GeneA",), ("genea",)]
    values = fetch_normalized_feature_matrix(
        merged, resolved, np.arange(2), NormalizationSpec(source="raw")
    )
    # Each source's cell has no value of the other source's gene, which it did
    # not measure, and a merge of no seed lists the two cells in either order.
    expected = {"left__left": [5, np.nan], "right__right": [np.nan, 7]}
    np.testing.assert_array_equal(
        values, [expected[cell] for cell in merged.cells.fetch_all("ids")]
    )
    with pytest.raises(KeyError, match="not found"):
        resolve_feature(merged, FeatureRef("GENEA", by="id"))
    with pytest.raises(ValueError, match="matches 2 entries"):
        resolve_feature(merged, FeatureRef("GENEA", by="name"))
    assert resolve_feature(merged, "SHARED").ids == ("shared",)


def _two_assay_sources():
    cell_ids = ["c0", "c1"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=2,
            ),
            _MergeAssay(
                "ADT",
                [[101, 110], [102, 120]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=2,
            ),
            _MergeAssay(
                "ADT",
                [[103, 130], [104, 140]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ],
        zarr_loc="memory://right",
    )
    return left, right


def test_dataset_merge_rejects_summary_before_truncating_destination():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_group("sentinel")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "summary",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "summary",
                [[3], [4]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    with pytest.raises(ValueError, match=r"reserved for DataStore\.summary"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=store,
            names=["left", "right"],
        ).dump()
    preserved = zarr.open_group(store=store, mode="r")
    assert set(preserved.group_keys()) == {"sentinel"}


@pytest.fixture(scope="module")
def merged_pbmc(datastore_zarr_root, tmp_path_factory):
    """Merge a copy of the bundled 1K PBMC store with itself.

    The source holds a pipeline run, and the merged store has been opened once
    as a DataStore, which adds its cell QC columns.
    """
    import shutil

    from cytearc.datastore.datastore import DataStore

    root = tmp_path_factory.mktemp("merged_pbmc")
    source_path = root / "source.zarr"
    shutil.copytree(datastore_zarr_root, source_path)
    source = DataStore(str(source_path), default_assay="RNA")
    source.zw.create_group(f"pipeline/runs/{'d' * 64}/stages")
    destination = str(root / "merged.zarr")
    DataStoreMerge(
        datasets=[source, source],
        zarr_path=destination,
        names=["self1", "self2"],
        prepend_text="orig",
        source_column="sample_id",
        overwrite=True,
    ).dump()
    merged = DataStore(destination, default_assay="RNA")
    return source, merged


def _pbmc_counts():
    """Return the RNA and ADT counts of the 1K PBMC file the store was built from."""
    from tests import full_path
    from tests.test_writers import _read_cellranger_h5

    counts, barcodes, features = _read_cellranger_h5(full_path("1K_pbmc_citeseq.h5"))
    types = features["feature_type"]
    return (
        {
            "RNA": counts[:, types == "Gene Expression"],
            "assay2": counts[:, types == "Antibody Capture"],
        },
        barcodes,
    )


def _merged_sources(merged_ids, barcodes):
    """Return the source name and source row of every merged cell ID."""
    position = {barcode: index for index, barcode in enumerate(barcodes)}
    names, rows = zip(
        *(
            (name, position[barcode])
            for name, _separator, barcode in (
                str(value).partition("__") for value in merged_ids
            )
        ),
        strict=True,
    )
    return np.asarray(names), np.asarray(rows)


def _assert_merged_counts(array, expected, rows) -> None:
    """Compare merged counts, one stored chunk at a time, with source ``rows``."""
    assert tuple(array.shape) == (rows.size, expected.shape[1])
    step_rows, step_columns = (int(value) for value in array.chunks)
    for start in range(0, array.shape[0], step_rows):
        band = expected[rows[start : start + step_rows]]
        for column in range(0, array.shape[1], step_columns):
            stored = array[start : start + step_rows, column : column + step_columns]
            wanted = band[:, column : column + step_columns].toarray()
            if not np.array_equal(stored, wanted):
                np.testing.assert_array_equal(stored, wanted)


def test_dataset_merge(merged_pbmc):
    from tests.test_writers import _assert_counts_equal

    source, merged = merged_pbmc
    expected, barcodes = _pbmc_counts()
    root = merged.zw
    names, rows = _merged_sources(root["cellData/ids"][:], barcodes)
    # Each source contributes every one of its cells exactly once.
    for name in ("self1", "self2"):
        np.testing.assert_array_equal(
            np.sort(rows[names == name]), np.arange(barcodes.size)
        )
    assert root.attrs["assayTypes"] == {"RNA": "RNA", "assay2": "Assay"}
    for assay_name, counts in expected.items():
        # Merged counts take the common type of the source count dtypes, and
        # both sources are this one store.
        assert (
            root[f"{assay_name}/counts"].dtype
            == source.zw[f"{assay_name}/counts"].dtype
        )
        _assert_merged_counts(root[f"{assay_name}/counts"], counts, rows)
    # Only the RNA assay holds the feature-major copy, in merged cell order.
    assert "countsT" not in root["assay2"]
    assert root["RNA/countsT"].attrs["complete"] is True
    _assert_counts_equal(root["RNA/countsT"], expected["RNA"][rows].T)


def test_dataset_merge_does_not_copy_source_pipeline_runs(merged_pbmc):
    source, merged = merged_pbmc

    assert "pipeline" in source.zw
    assert "pipeline" not in merged.zw


def test_dataset_merge_maps_features_and_preserves_row_order(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30]],
                ["c0", "c1", "c2"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[4, 40], [5, 50], [6, 60]],
                ["c0", "c1", "c2"],
                ["id_b", "id_c"],
                ["B", "C"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "feature_map.zarr")
    writer = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=1,
        overwrite=True,
    )
    plan = writer.plan()
    assert plan.assays[0].nFeatures == 3
    assert plan.assays[0].featureOverlapFraction == pytest.approx(1 / 3)
    writer.dump()
    root = zarr.open_group(fn, mode="r")
    feat_ids = np.asarray(root["RNA/featureData/ids"][:]).astype(str)
    assert list(feat_ids) == ["id_a", "id_b", "id_c"]
    counts = np.asarray(root["RNA/counts"][:])
    cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    expected = {
        "left__c0": [1, 10, 0],
        "left__c1": [2, 20, 0],
        "left__c2": [3, 30, 0],
        "right__c0": [0, 4, 40],
        "right__c1": [0, 5, 50],
        "right__c2": [0, 6, 60],
    }
    assert sorted(cell_ids.tolist()) == sorted(expected)
    for cell_id, row in zip(cell_ids, counts, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])
    spec = count_array_spec(
        counts.shape[0],
        counts.shape[1],
        dtype=counts.dtype,
        profile="fast_local",
    )
    assert tuple(root["RNA/counts"].chunks) == spec.chunks
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        root["RNA/countsT"][:],
        np.asarray(root["RNA/counts"][:]).T,
    )


def test_dataset_merge_preserves_suffixed_feature_ids(tmp_path):
    feature_ids = ["gene_0", "gene_1"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.array([[200, 100]], dtype=np.uint8),
                ["left"],
                feature_ids,
                feature_ids,
                block_size=1,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.array([[150, 150]], dtype=np.uint8),
                ["right"],
                feature_ids,
                feature_ids,
                block_size=1,
            )
        ],
        zarr_loc="memory://right",
    )
    path = str(tmp_path / "suffixed_features.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
    ).dump()
    root = zarr.open_group(path, mode="r")
    counts = root["RNA/counts"]
    assert counts.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], feature_ids)
    expected = {"left__left": [200, 100], "right__right": [150, 150]}
    assert sorted(root["cellData/ids"][:].tolist()) == sorted(expected)
    for cell_id, row in zip(root["cellData/ids"][:], counts[:], strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])
    np.testing.assert_array_equal(root["RNA/countsT"][:], counts[:].T)


def test_dataset_merge_keeps_source_metadata_aligned_after_permutation(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2], [3], [4]],
                ["c0", "c1", "c2", "c3"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    left.cells = _MergeMeta(
        ids=["c0", "c1", "c2", "c3"],
        names=["c0", "c1", "c2", "c3"],
        I=np.array([True, False, True, True]),
        cluster_labels=np.array(["a", "b", "c", "d"]),
    )
    left._assays["RNA"].cells = left.cells
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[5], [6], [7], [8]],
                ["c0", "c1", "c2", "c3"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    right.cells = _MergeMeta(
        ids=["c0", "c1", "c2", "c3"],
        names=["c0", "c1", "c2", "c3"],
        I=np.array([True, True, False, True]),
        cluster_labels=np.array(["e", "f", "g", "h"]),
    )
    right._assays["RNA"].cells = right.cells
    fn = str(tmp_path / "meta.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="orig",
        reset_cell_filter=False,
        source_column="sample_id",
        seed=3,
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    sample = np.asarray(root["cellData/sample_id"][:]).astype(str)
    labels = np.asarray(root["cellData/orig_cluster_labels"][:]).astype(str)
    included = np.asarray(root["cellData/I"][:])
    expected = {
        "left__c0": ("left", "a", True),
        "left__c1": ("left", "b", False),
        "left__c2": ("left", "c", True),
        "left__c3": ("left", "d", True),
        "right__c0": ("right", "e", True),
        "right__c1": ("right", "f", True),
        "right__c2": ("right", "g", False),
        "right__c3": ("right", "h", True),
    }
    assert sorted(ids.tolist()) == sorted(expected)
    for cell_id, sample_id, label, keep in zip(
        ids, sample, labels, included, strict=True
    ):
        assert (sample_id, label, bool(keep)) == expected[cell_id]


def test_dataset_merge_rejects_source_column_conflict(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    with pytest.raises(ValueError, match="source_column"):
        DataStoreMerge(
            datasets=[left, left],
            zarr_path=str(tmp_path / "conflict.zarr"),
            names=["left", "right"],
            source_column="ids",
        ).plan()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"source_column": "src/ds"}, "use 'src_ds' instead"),
        ({"source_column": "src\\ds"}, "use 'src_ds' instead"),
        ({"prepend_text": "x/y"}, "prepend_text 'x/y' must not contain"),
    ],
    ids=["source-slash", "source-backslash", "prepend-slash"],
)
def test_dataset_merge_rejects_path_separator_column_names(tmp_path, options, message):
    left = _MergeDataStore(
        [_MergeAssay("RNA", [[1], [2]], ["c0", "c1"], ["id_a"], ["A"], block_size=2)],
        zarr_loc="memory://left",
    )
    destination = tmp_path / "separators.zarr"

    with pytest.raises(ValueError, match=message):
        DataStoreMerge(
            datasets=[left, left],
            zarr_path=str(destination),
            names=["left", "right"],
            **options,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"prepend_text": 5}, TypeError, "prepend_text must be a string or None"),
        ({"source_column": "__cytearc_missing__x"}, ValueError, "reserves"),
        ({"prepend_text": "__cytearc_missing_"}, ValueError, "mask prefix"),
    ],
    ids=["prepend-int", "source-mask-prefix", "prepend-mask-prefix"],
)
def test_dataset_merge_rejects_reserved_or_untyped_names(
    tmp_path, options, error, message
):
    left = _MergeDataStore(
        [_MergeAssay("RNA", [[1], [2]], ["c0", "c1"], ["id_a"], ["A"], block_size=2)],
        zarr_loc="memory://left",
    )

    with pytest.raises(error, match=message):
        DataStoreMerge(
            datasets=[left, left],
            zarr_path=str(tmp_path / "reserved.zarr"),
            names=["left", "right"],
            **options,
        )


def test_dataset_merge_plan_checks_names_changed_after_construction(tmp_path):
    left = _MergeDataStore(
        [_MergeAssay("RNA", [[1], [2]], ["c0", "c1"], ["id_a"], ["A"], block_size=2)],
        zarr_loc="memory://left",
    )
    destination = tmp_path / "late.zarr"
    merge = DataStoreMerge(
        datasets=[left, left], zarr_path=str(destination), names=["left", "right"]
    )
    merge.prependText = "x/y"

    with pytest.raises(ValueError, match="prepend_text 'x/y' must not contain"):
        merge.plan()
    assert not destination.exists()


def test_dataset_merge_preserves_order_across_source_block_sizes(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30], [4, 40]],
                ["c0", "c1", "c2", "c3"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=3,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[5, 50], [6, 60], [7, 70], [8, 80]],
                ["c0", "c1", "c2", "c3"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "blocks.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=2,
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["RNA/counts"][:])
    expected = {
        "left__c0": [1, 10],
        "left__c1": [2, 20],
        "left__c2": [3, 30],
        "left__c3": [4, 40],
        "right__c0": [5, 50],
        "right__c1": [6, 60],
        "right__c2": [7, 70],
        "right__c3": [8, 80],
    }
    assert sorted(ids.tolist()) == sorted(expected)
    for cell_id, row in zip(ids, counts, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])


def test_dataset_merge_shares_row_order_across_assay_chunk_sizes(tmp_path):
    cell_ids = ["c0", "c1", "c2", "c3", "c4"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=3,
            ),
            _MergeAssay(
                "ADT",
                [[101, 110], [102, 120], [103, 130], [104, 140], [105, 150]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[6, 60], [7, 70], [8, 80], [9, 90], [10, 100]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=3,
            ),
            _MergeAssay(
                "ADT",
                [[106, 160], [107, 170], [108, 180], [109, 190], [110, 200]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            ),
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "mixed_assay_chunks.zarr")
    writer = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=3,
        overwrite=True,
    )
    writer.dump()
    root = zarr.open_group(fn, mode="r")
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    rna = np.asarray(root["RNA/counts"][:])
    adt = np.asarray(root["ADT/counts"][:])
    expected = {
        "left__c0": ([1, 10], [101, 110]),
        "left__c1": ([2, 20], [102, 120]),
        "left__c2": ([3, 30], [103, 130]),
        "left__c3": ([4, 40], [104, 140]),
        "left__c4": ([5, 50], [105, 150]),
        "right__c0": ([6, 60], [106, 160]),
        "right__c1": ([7, 70], [107, 170]),
        "right__c2": ([8, 80], [108, 180]),
        "right__c3": ([9, 90], [109, 190]),
        "right__c4": ([10, 100], [110, 200]),
    }
    assert sorted(merged_cell_ids.tolist()) == sorted(expected)
    for cell_id, rna_row, adt_row in zip(merged_cell_ids, rna, adt, strict=True):
        expected_rna, expected_adt = expected[cell_id]
        np.testing.assert_array_equal(rna_row, expected_rna)
        np.testing.assert_array_equal(adt_row, expected_adt)


def test_dataset_merge_shared_row_plan_handles_missing_assays(tmp_path):
    cell_ids = ["c0", "c1", "c2"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20], [3, 30]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=3,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[101, 110], [102, 120], [103, 130]],
                cell_ids,
                ["adt_a", "adt_b"],
                ["ADT A", "ADT B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "missing_assays.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=3,
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    merged_cell_ids = np.asarray(root["cellData/ids"][:]).astype(str)
    rna = np.asarray(root["RNA/counts"][:])
    adt = np.asarray(root["ADT/counts"][:])
    rna_i = np.asarray(root["cellData/RNA_I"][:])
    adt_i = np.asarray(root["cellData/ADT_I"][:])
    expected = {
        "left__c0": ([1, 10], [0, 0], True, False),
        "left__c1": ([2, 20], [0, 0], True, False),
        "left__c2": ([3, 30], [0, 0], True, False),
        "right__c0": ([0, 0], [101, 110], False, True),
        "right__c1": ([0, 0], [102, 120], False, True),
        "right__c2": ([0, 0], [103, 130], False, True),
    }
    assert sorted(merged_cell_ids.tolist()) == sorted(expected)
    for cell_id, rna_row, adt_row, rna_flag, adt_flag in zip(
        merged_cell_ids,
        rna,
        adt,
        rna_i,
        adt_i,
        strict=True,
    ):
        expected_rna, expected_adt, expected_rna_i, expected_adt_i = expected[cell_id]
        np.testing.assert_array_equal(rna_row, expected_rna)
        np.testing.assert_array_equal(adt_row, expected_adt)
        assert bool(rna_flag) is expected_rna_i
        assert bool(adt_flag) is expected_adt_i
    assert root["cellData/RNA_I"].attrs["role"] == "assay_membership"
    assert root["cellData/ADT_I"].attrs["assay"] == "ADT"


def test_dataset_merge_missing_assay_policy_error(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[3], [4]],
                ["c0", "c1"],
                ["id_b"],
                ["B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    with pytest.raises(ValueError, match="missing assay"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(tmp_path / "missing_error.zarr"),
            names=["left", "right"],
            missing_assay_policy="error",
        ).plan()


def test_dataset_merge_three_sources_keep_each_source_row(tmp_path):
    sources = []
    for label, block_size in (("a", 2), ("b", 3), ("c", 1)):
        offset = {"a": 0, "b": 100, "c": 200}[label]
        sources.append(
            _MergeDataStore(
                [
                    _MergeAssay(
                        "RNA",
                        [
                            [offset + 1, offset + 10],
                            [offset + 2, offset + 20],
                            [offset + 3, offset + 30],
                        ],
                        ["c0", "c1", "c2"],
                        ["id_a", "id_b"],
                        ["A", "B"],
                        block_size=block_size,
                    )
                ],
                zarr_loc=f"memory://{label}",
            )
        )
    path = str(tmp_path / "three.zarr")
    result = DataStoreMerge(
        datasets=sources,
        zarr_path=path,
        names=["a", "b", "c"],
        prepend_text="",
        source_column="sample_id",
        seed=5,
    ).dump()

    assert result.nCells == 9
    root = zarr.open_group(path, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    assert sorted(ids) == sorted(
        f"{label}__c{cell}" for label in "abc" for cell in range(3)
    )
    expected = {
        f"{label}__c{cell}": [offset + cell + 1, offset + 10 * (cell + 1)]
        for label, offset in (("a", 0), ("b", 100), ("c", 200))
        for cell in range(3)
    }
    counts = np.asarray(root["RNA/counts"][:])
    samples = np.asarray(root["cellData/sample_id"][:]).astype(str)
    for cell_id, row, sample in zip(ids, counts, samples, strict=True):
        np.testing.assert_array_equal(row, expected[cell_id])
        assert sample == cell_id.split("__")[0]
    np.testing.assert_array_equal(root["RNA/countsT"][:], counts.T)


def test_dataset_merge_cells(merged_pbmc):
    _source, merged = merged_pbmc
    expected, barcodes = _pbmc_counts()
    columns = set(merged.cells.columns)
    ids = merged.cells.fetch_all("ids")
    names, rows = _merged_sources(ids, barcodes)
    # Cell QC columns are recomputed for the merged counts rather than copied
    # with the prefix of the source columns.
    assert not any(column.startswith("orig_") for column in columns)
    np.testing.assert_array_equal(merged.cells.fetch_all("sample_id"), names)
    for assay_name, counts in expected.items():
        totals = np.asarray(counts.sum(axis=1)).ravel()[rows]
        np.testing.assert_array_equal(
            merged.cells.fetch_all(f"{assay_name}_nCounts"), totals
        )
        np.testing.assert_array_equal(
            merged.cells.fetch_all(f"{assay_name}_nFeatures"),
            np.diff(counts.indptr)[rows],
        )
        np.testing.assert_array_equal(
            merged.cells.fetch_all(f"{assay_name}_I"), np.ones(rows.size, dtype=bool)
        )
    assert merged.cells.to_pandas_dataframe(["sample_id"])[
        "sample_id"
    ].value_counts().to_dict() == {"self1": barcodes.size, "self2": barcodes.size}


def test_dataset_merge_rejects_duplicate_sample_names(datastore, tmp_path):
    fn = str(tmp_path / "merged_dup_names.zarr")
    with pytest.raises(ValueError, match="unique name"):
        DataStoreMerge(
            zarr_path=fn,
            datasets=[datastore, datastore],
            names=["dup", "dup"],
            overwrite=True,
        )


def test_dataset_merge_requires_two_sources(tmp_path):
    source = _merge_two_rna(
        zarr_path=str(tmp_path / "unused.zarr"),
        overwrite=False,
    ).datasets[0]
    with pytest.raises(ValueError, match="at least two"):
        DataStoreMerge(
            datasets=[source],
            zarr_path=str(tmp_path / "single_source.zarr"),
            names=["only"],
        )


def test_dataset_merge_row_order_ignores_unselected_assay_chunks(tmp_path):
    cell_ids = [f"c{i}" for i in range(6)]

    def sources(include_adt):
        stores = []
        for source_index, source_name in enumerate(("left", "right")):
            offset = 100 * source_index
            assays = [
                _MergeAssay(
                    "RNA",
                    [[offset + i] for i in range(6)],
                    cell_ids,
                    ["rna"],
                    ["RNA"],
                    block_size=2,
                )
            ]
            if include_adt:
                assays.append(
                    _MergeAssay(
                        "ADT",
                        [[offset + i] for i in range(6)],
                        cell_ids,
                        ["adt"],
                        ["ADT"],
                        block_size=1,
                    )
                )
            stores.append(
                _MergeDataStore(
                    assays,
                    zarr_loc=f"memory://{source_name}-{include_adt}",
                )
            )
        return stores

    paths = [
        str(tmp_path / "rna_only.zarr"),
        str(tmp_path / "rna_with_unselected_adt.zarr"),
    ]
    for path, include_adt in zip(paths, (False, True), strict=True):
        DataStoreMerge(
            datasets=sources(include_adt),
            zarr_path=path,
            names=["left", "right"],
            assays=["RNA"],
            prepend_text="",
            seed=7,
        ).dump()

    ids_without_adt = np.asarray(
        zarr.open_group(paths[0], mode="r")["cellData/ids"][:],
    ).astype(str)
    ids_with_adt = np.asarray(
        zarr.open_group(paths[1], mode="r")["cellData/ids"][:],
    ).astype(str)
    np.testing.assert_array_equal(ids_with_adt, ids_without_adt)


def test_dataset_merge_workspace_and_counts_t(tmp_path):
    path = str(tmp_path / "workspace_merged.zarr")
    result = _merge_two_rna(
        zarr_path=path, overwrite=False, out_workspace="merged"
    ).dump()

    assert result.assayNames == ("RNA",)
    root = zarr.open_group(path, mode="r")
    # Workspace metadata lives in the workspace and the counts under matrices.
    assert set(root.group_keys()) == {"matrices", "merged"}
    workspace = root["merged"]
    assert workspace.attrs["assayTypes"] == {"RNA": "RNA"}
    assert workspace.attrs["complete"] is True
    ids = np.asarray(workspace["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["matrices/RNA/counts"][:])
    expected = {
        "left__c0": [1, 10],
        "left__c1": [2, 20],
        "right__c0": [3, 30],
        "right__c1": [4, 40],
    }
    assert {
        cell_id: row.tolist() for cell_id, row in zip(ids, counts, strict=True)
    } == expected
    assert root["matrices/RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(root["matrices/RNA/countsT"][:], counts.T)


def test_dataset_merge_plan_is_side_effect_free():
    store = MemoryStore()
    merger = _merge_two_rna(zarr_path=store)
    plan = merger.plan()
    assert plan.nCells == 4
    assert plan.assays[0].assayName == "RNA"
    assert plan.assays[0].featureOverlapFraction == 1.0
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(store, mode="r")


def test_dataset_merge_idempotent_resume(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "resume.zarr")
    first = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
        seed=0,
    )
    first.dump()
    second = DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    )
    result = second.dump()
    assert [(component.name, component.action) for component in result.components] == [
        ("cellData", "skip"),
        ("counts:RNA", "skip"),
        ("countsT:RNA", "skip"),
    ]
    assert _rows_by_id(zarr.open_group(fn, mode="r")) == _TWO_RNA_ROWS


def test_dataset_merge_metadata_layout_is_budget_independent_and_resumable(
    tmp_path,
    monkeypatch,
):
    import cytearc.merge.metadata as merge_metadata
    import cytearc.merge.row_plan as merge_row_plan

    def sources():
        cell_ids = [f"c{i}" for i in range(20)]
        left = _MergeDataStore(
            [
                _MergeAssay(
                    "RNA",
                    np.arange(20, dtype=np.uint16).reshape(20, 1),
                    cell_ids,
                    ["id_a"],
                    ["A"],
                    block_size=10,
                )
            ],
            zarr_loc="memory://left",
        )
        right = _MergeDataStore(
            [
                _MergeAssay(
                    "RNA",
                    np.arange(20, 40, dtype=np.uint16).reshape(20, 1),
                    cell_ids,
                    ["id_a"],
                    ["A"],
                    block_size=10,
                )
            ],
            zarr_loc="memory://right",
        )
        left.cells._columns["quality"] = np.arange(20, dtype=np.int16)
        left.cells.columns.append("quality")
        return left, right

    def merger(path, *, memory):
        left, right = sources()
        return DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(path),
            names=["left", "right"],
            prepend_text="",
            overwrite=False,
            seed=0,
            mem_budget=memory,
            nthreads=1,
        )

    original_peak = merge_metadata.CellMetadataPlan.peak_write_bytes_at

    def constrained_peak(self, rows, *, chunk_rows):
        _ = self, chunk_rows
        return 50_000 + max(1, int(rows)) * 5_000

    monkeypatch.setattr(
        merge_metadata.CellMetadataPlan,
        "peak_write_bytes_at",
        constrained_peak,
    )
    low_path = tmp_path / "low_budget_layout.zarr"
    low = merger(low_path, memory=100_000)
    low.plan()
    assert low._metadataPlan is not None
    low_width = low._metadataPlan.blockRows
    low.dump()

    monkeypatch.setattr(
        merge_metadata.CellMetadataPlan,
        "peak_write_bytes_at",
        original_peak,
    )
    high_path = tmp_path / "high_budget_layout.zarr"
    high = merger(high_path, memory=1024**3)
    high.plan()
    assert high._metadataPlan is not None
    high_width = high._metadataPlan.blockRows
    high.dump()
    assert low_width < high_width

    def cell_chunks(path):
        group = zarr.open_group(path, mode="r")["cellData"]
        return {
            name: tuple(int(value) for value in group[name].chunks)
            for name in group.array_keys()
        }

    low_chunks = cell_chunks(low_path)
    assert low_chunks == cell_chunks(high_path)
    # Metadata chunks hold PROFILE_METADATA_CHUNK rows, capped by the 40 cells.
    assert set(low_chunks.values()) == {(40,)}

    identity_widths: list[int] = []
    original_identity_read = merge_row_plan.read_metadata_rows_chunkwise

    def tracking_identity_read(table, column, rows):
        identity_widths.append(int(np.asarray(rows).size))
        return original_identity_read(table, column, rows)

    monkeypatch.setattr(
        merge_metadata,
        "resolve_identity_validation_rows",
        lambda *args, **kwargs: 3,
    )
    monkeypatch.setattr(
        merge_row_plan,
        "read_metadata_rows_chunkwise",
        tracking_identity_read,
    )
    resumed = merger(low_path, memory=1024**3).plan()
    assert resumed.cellDataAction == "skip"
    assert resumed.canDump is True
    assert identity_widths
    assert max(identity_widths) <= 3


def test_dataset_merge_incomplete_store_is_rejected(tmp_path):
    from cytearc.datastore.datastore import DataStore

    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3], [4]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "incomplete.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r+")
    root.attrs["cytearc:import_complete"] = False
    with pytest.raises(RuntimeError, match="DataStoreMerge import is incomplete"):
        DataStore(fn, nthreads=1)


def test_dataset_merge_partial_metadata_uses_missing_mask(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1], [2]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    left.cells = _MergeMeta(
        ids=["c0", "c1"],
        names=["c0", "c1"],
        I=np.ones(2, dtype=bool),
        batch=np.array(["x", "y"]),
    )
    left._assays["RNA"].cells = left.cells
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3], [4]],
                ["c0", "c1"],
                ["id_a"],
                ["A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    fn = str(tmp_path / "partial_meta.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=fn,
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
    ).dump()
    root = zarr.open_group(fn, mode="r")
    cell_data = root["cellData"]
    assert cell_data["batch"].attrs["missing_mask"] == "__cytearc_missing__batch"
    # The right source has no batch column, so its cells are missing values.
    assert _rows_by_id(root, "cellData/batch") == {
        "left__c0": "x",
        "left__c1": "y",
        "right__c0": "",
        "right__c1": "",
    }
    assert _rows_by_id(root, "cellData/__cytearc_missing__batch") == {
        "left__c0": False,
        "left__c1": False,
        "right__c0": True,
        "right__c1": True,
    }


def test_dataset_merge_reads_cell_metadata_without_fetch_all(tmp_path):
    left, right = _merge_two_rna(
        zarr_path=str(tmp_path / "unused.zarr"),
        overwrite=False,
    ).datasets
    for store, label in ((left, "left"), (right, "right")):
        metadata = _NoFullReadMeta(
            ids=["c0", "c1"],
            names=["c0", "c1"],
            I=np.ones(2, dtype=bool),
            label=np.array([label, label]),
        )
        store.cells = metadata
        store.get_assay("RNA").cells = metadata

    path = str(tmp_path / "bounded_metadata.zarr")
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    ).dump()

    root = zarr.open_group(path, mode="r")
    assert _rows_by_id(root, "cellData/label") == {
        "left__c0": "left",
        "left__c1": "left",
        "right__c0": "right",
        "right__c1": "right",
    }


def test_dataset_merge_preserves_existing_missing_masks(tmp_path):
    def metadata_table(prefix, missing):
        root = zarr.open_group(store=MemoryStore(), mode="w")
        values = {
            "ids": np.array(["c0", "c1"]),
            "names": np.array(["c0", "c1"]),
            "I": np.ones(2, dtype=bool),
            "batch": np.array([f"{prefix}0", f"{prefix}1"]),
        }
        for name, data in values.items():
            root.create_array(name, data=data, chunks=(2,))
        root.create_array(
            "__cytearc_missing__batch",
            data=np.asarray(missing, dtype=bool),
            chunks=(2,),
        )
        root["batch"].attrs["missing_mask"] = "__cytearc_missing__batch"
        return MetaData(root)

    merger = _merge_two_rna(
        zarr_path=str(tmp_path / "preserved_missing.zarr"),
        overwrite=False,
    )
    for store, metadata in zip(
        merger.datasets,
        (
            metadata_table("l", [False, True]),
            metadata_table("r", [True, False]),
        ),
        strict=True,
    ):
        store.cells = metadata
        store.get_assay("RNA").cells = metadata

    merger.dump()
    root = zarr.open_group(merger.zarr_path, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    missing = np.asarray(root["cellData/__cytearc_missing__batch"][:], dtype=bool)
    expected = {
        "left__c0": False,
        "left__c1": True,
        "right__c0": True,
        "right__c1": False,
    }
    assert {
        cell_id: bool(is_missing)
        for cell_id, is_missing in zip(ids, missing, strict=True)
    } == expected
    # The merged column still names its mask, so readers see the cells as missing.
    assert root["cellData/batch"].attrs["missing_mask"] == "__cytearc_missing__batch"


def test_dataset_merge_finalizes_complete_components_after_interruption(
    tmp_path, merged_two_rna
):
    path = _copy_merge(merged_two_rna, "default", tmp_path / "finalize_only.zarr")
    root = zarr.open_group(path, mode="r+")
    root.attrs["cytearc:import_complete"] = False
    root.attrs["complete"] = False

    merger = _merge_two_rna(zarr_path=path, overwrite=False)
    plan = merger.plan()
    assert plan.canDump is True
    assert plan.willResume is True
    result = merger.dump()

    assert result.resumed is True
    assert [(component.name, component.action) for component in result.components] == [
        ("cellData", "skip"),
        ("counts:RNA", "skip"),
        ("countsT:RNA", "skip"),
    ]
    completed = zarr.open_group(path, mode="r")
    assert completed.attrs["cytearc:import_complete"] is True
    assert completed.attrs["complete"] is True
    assert _rows_by_id(completed) == _TWO_RNA_ROWS


def test_dataset_merge_overwrite_forces_scoped_rebuild(tmp_path, merged_two_rna):
    path = _copy_merge(merged_two_rna, "default", tmp_path / "forced_rebuild.zarr")
    first = _merge_two_rna(zarr_path=path, overwrite=False)
    root = zarr.open_group(path, mode="r+")
    root.create_group("unrelated")

    first.datasets[0].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.full((2, 2), 100, dtype=np.int64),
        block_size=2,
    )
    second = DataStoreMerge(
        datasets=first.datasets,
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
        seed=0,
    )
    result = second.dump()
    rebuilt = zarr.open_group(path, mode="r")

    assert [(component.name, component.action) for component in result.components] == [
        ("cellData", "write"),
        ("counts:RNA", "write"),
        ("countsT:RNA", "write"),
    ]
    # The left counts were replaced, so the rebuild holds the new values.
    assert _rows_by_id(rebuilt) == {
        **_TWO_RNA_ROWS,
        "left__c0": [100, 100],
        "left__c1": [100, 100],
    }
    assert "unrelated" in rebuilt


def test_dataset_merge_instance_can_be_reused(tmp_path):
    path = str(tmp_path / "reuse_instance.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False)

    merger.dump()
    result = merger.dump()

    assert [(component.name, component.action) for component in result.components] == [
        ("cellData", "skip"),
        ("counts:RNA", "skip"),
        ("countsT:RNA", "skip"),
    ]
    assert _rows_by_id(zarr.open_group(path, mode="r")) == _TWO_RNA_ROWS


def test_dataset_merge_plan_reports_destination_conflict(tmp_path, merged_two_rna):
    # The completed merge used seed 0.
    path = _copy_merge(merged_two_rna, "default", tmp_path / "blocked_plan.zarr")
    blocked = _merge_two_rna(zarr_path=path, overwrite=False, seed=1)

    plan = blocked.plan()

    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "different configuration" in plan.blockedReason
    assert plan.cellDataAction == "blocked"
    assert plan.assays[0].countsAction == "blocked"
    with pytest.raises(ValueError, match="different configuration"):
        blocked.dump()


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("cell_identity", "cell IDs or their order differ"),
        ("counts_shape", "not finalized"),
        ("counts_dtype", "not finalized"),
        ("counts_chunks", "not finalized"),
        ("counts_shards", "not finalized"),
        ("feature_ids", "featureData/ids"),
        ("feature_names", "featureData/names"),
        ("feature_selection", "featureData/I"),
        ("metadata_missing_chunks", "missing mask"),
        ("counts_t_shape", "countsT shape"),
        ("counts_t_dtype", "countsT dtype"),
        # A completed countsT in another layout is not rewritten in place.
        ("counts_t_chunks", "Completed countsT for 'RNA' cannot be reused"),
        ("counts_t_shards", "Completed countsT for 'RNA' cannot be reused"),
        ("component_marker", "marked complete"),
        ("import_source", "foreign import source"),
        ("root_import_complete", "marked complete"),
        ("root_complete", "marked complete"),
    ],
)
def test_dataset_merge_plan_blocks_tampered_components(
    tmp_path,
    merged_two_rna,
    case,
    reason,
):
    sharded = case in {"counts_shards", "counts_t_shards"}
    merge_kwargs = {"policy": _SHARDED_POLICY} if sharded else {}
    variant = (
        "sharded"
        if sharded
        else "quality"
        if case == "metadata_missing_chunks"
        else "default"
    )
    path = _copy_merge(merged_two_rna, variant, tmp_path / f"tampered_{case}.zarr")
    root = zarr.open_group(path, mode="r+")
    if case == "cell_identity":
        ids = np.asarray(root["cellData/ids"][:]).astype(str)
        root["cellData/ids"][:] = ids[::-1]
    elif case == "counts_shape":
        del root["RNA/counts"]
        root["RNA"].create_array(
            "counts",
            shape=(4, 1),
            chunks=(4, 1),
            dtype=np.uint16,
        )
    elif case == "counts_dtype":
        values = np.asarray(root["RNA/counts"][:], dtype=np.float32)
        del root["RNA/counts"]
        root["RNA"].create_array("counts", data=values, chunks=(4, 2))
    elif case == "counts_chunks":
        values = np.asarray(root["RNA/counts"][:])
        del root["RNA/counts"]
        root["RNA"].create_array("counts", data=values, chunks=(2, 1))
    elif case == "counts_shards":
        counts = root["RNA/counts"]
        values = np.asarray(counts[:])
        chunks = tuple(int(value) for value in counts.chunks)
        del root["RNA/counts"]
        root["RNA"].create_array(
            "counts",
            data=values,
            chunks=chunks,
            shards=(chunks[0] * 2, chunks[1] * 2),
        )
        root["RNA"].attrs["complete"] = True
    elif case == "feature_ids":
        root["RNA/featureData/ids"][0] = "changed"
    elif case == "feature_names":
        root["RNA/featureData/names"][0] = "changed"
    elif case == "feature_selection":
        root["RNA/featureData/I"][0] = False
    elif case == "metadata_missing_chunks":
        missing_name = "__cytearc_missing__quality"
        values = np.asarray(root[f"cellData/{missing_name}"][:], dtype=bool)
        del root[f"cellData/{missing_name}"]
        root["cellData"].create_array(
            missing_name,
            data=values,
            chunks=(1,),
        )
    elif case == "counts_t_shape":
        del root["RNA/countsT"]
        root["RNA"].create_array(
            "countsT",
            shape=(1, 4),
            chunks=(1, 4),
            dtype=np.uint16,
        )
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "counts_t_dtype":
        values = np.asarray(root["RNA/countsT"][:], dtype=np.float32)
        del root["RNA/countsT"]
        root["RNA"].create_array("countsT", data=values, chunks=(2, 4))
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "counts_t_chunks":
        values = np.asarray(root["RNA/countsT"][:])
        del root["RNA/countsT"]
        root["RNA"].create_array("countsT", data=values, chunks=(1, 2))
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "counts_t_shards":
        counts_t_array = root["RNA/countsT"]
        values = np.asarray(counts_t_array[:])
        chunks = tuple(int(value) for value in counts_t_array.chunks)
        del root["RNA/countsT"]
        root["RNA"].create_array(
            "countsT",
            data=values,
            chunks=chunks,
            shards=(chunks[0] * 2, chunks[1]),
        )
        root["RNA/countsT"].attrs["complete"] = True
    elif case == "component_marker":
        root["RNA"].attrs["complete"] = False
    elif case == "import_source":
        root.attrs["cytearc:import_source"] = "ForeignImporter"
    elif case == "root_import_complete":
        root["RNA"].attrs["complete"] = False
        root.attrs["cytearc:import_complete"] = True
        root.attrs["complete"] = False
    else:
        root["RNA"].attrs["complete"] = False
        root.attrs["cytearc:import_complete"] = False
        root.attrs["complete"] = True

    blocked_merger = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        **merge_kwargs,
    )
    if case == "metadata_missing_chunks":
        _with_partial_quality(blocked_merger)
    blocked = blocked_merger.plan()
    assert blocked.canDump is False
    assert blocked.blockedReason is not None
    assert reason in blocked.blockedReason

    if case == "counts_shape":
        restarted = _merge_two_rna(zarr_path=path, overwrite=True)
        assert restarted.plan().canDump is True
        restarted.dump()
        assert zarr.open_group(path, mode="r")["RNA/counts"].shape == (4, 2)


def test_dataset_merge_manifest_does_not_persist_source_locations(tmp_path):
    path = str(tmp_path / "safe_manifest.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False)

    plan = merger.plan()
    assert "sourceLocations" not in plan.manifest
    merger.dump()
    stored = zarr.open_group(path, mode="r").attrs["cytearc:merge_manifest"]
    assert "sourceLocations" not in stored
    assert stored["sourceFeatureCounts"] == {"RNA": [2, 2]}


def test_dataset_merge_plan_rejects_a_zarr_v2_destination(tmp_path):
    path = str(tmp_path / "zarr_v2.zarr")
    zarr.open_group(path, mode="w", zarr_format=2)
    merger = _merge_two_rna(
        zarr_path=path,
        overwrite=True,
    )

    # Count matrices are paired and sharded, so a Zarr v2 destination is refused.
    with pytest.raises(ValueError, match="Zarr format 3"):
        merger.plan()


@pytest.mark.parametrize(
    ("case", "error", "expected", "completion"),
    [
        (
            "cell_metadata",
            "cellData interruption",
            {
                "cellData": "resume",
                "counts:RNA": "resume",
                "countsT:RNA": "resume",
            },
            {"cellData": False, "RNA": None},
        ),
        (
            "counts",
            "counts interruption",
            {
                "cellData": "skip",
                "counts:RNA": "resume",
                "countsT:RNA": "resume",
            },
            {"cellData": True, "RNA": False},
        ),
        (
            "counts_t",
            "countsT interruption",
            {
                "cellData": "skip",
                "counts:RNA": "skip",
                "countsT:RNA": "resume",
            },
            {"cellData": True, "RNA": True, "RNA/countsT": False},
        ),
        (
            "later_assay",
            "later-assay interruption",
            {
                "counts:RNA": "skip",
                "counts:ADT": "resume",
            },
            {"cellData": True, "RNA": True, "ADT": False},
        ),
    ],
)
def test_dataset_merge_resumes_after_component_interruption(
    tmp_path,
    monkeypatch,
    case,
    error,
    expected,
    completion,
):
    path = str(tmp_path / f"resume_{case}.zarr")

    def assert_interrupted_completion_boundaries():
        interrupted = zarr.open_group(path, mode="r")
        assert interrupted.attrs["cytearc:import_complete"] is False
        assert interrupted.attrs["complete"] is False
        for component_path, expected_complete in completion.items():
            if expected_complete is None:
                assert component_path not in interrupted
            else:
                assert component_path in interrupted
                assert (
                    interrupted[component_path].attrs.get("complete")
                    is expected_complete
                )

    if case == "later_assay":
        left, right = _two_assay_sources()
        original = merge_datasets.write_assay_counts

        def fail_adt(root, assay_name, *args, **kwargs):
            if assay_name == "ADT":
                raise RuntimeError(f"simulated {error}")
            return original(root, assay_name, *args, **kwargs)

        monkeypatch.setattr(merge_datasets, "write_assay_counts", fail_adt)
        dump_kwargs = {
            "datasets": [left, right],
            "zarr_path": path,
            "names": ["left", "right"],
            "prepend_text": "",
        }
        with pytest.raises(RuntimeError, match=error):
            DataStoreMerge(**dump_kwargs).dump()
        assert_interrupted_completion_boundaries()
        monkeypatch.setattr(merge_datasets, "write_assay_counts", original)
        result = DataStoreMerge(**dump_kwargs).dump()
        actions = {component.name: component.action for component in result.components}
        for name, action in expected.items():
            assert actions[name] == action
        assert result.resumed is True
        completed = zarr.open_group(path, mode="r")
        assert _rows_by_id(completed) == _TWO_RNA_ROWS
        assert _rows_by_id(completed, "ADT/counts") == {
            "left__c0": [101, 110],
            "left__c1": [102, 120],
            "right__c0": [103, 130],
            "right__c1": [104, 140],
        }
        return

    if case == "cell_metadata":
        original = merge_datasets.write_cell_metadata

        def fail_after_write(*args, **kwargs):
            group = original(*args, **kwargs)
            group.attrs["complete"] = False
            raise RuntimeError(f"simulated {error}")

        monkeypatch.setattr(merge_datasets, "write_cell_metadata", fail_after_write)
        restore = ("write_cell_metadata", original)
    elif case == "counts":
        original = merge_datasets.write_assay_counts

        def fail_counts(*args, **kwargs):
            raise RuntimeError(f"simulated {error}")

        monkeypatch.setattr(merge_datasets, "write_assay_counts", fail_counts)
        restore = ("write_assay_counts", original)
    else:
        original = merge_datasets.write_assay_counts_t

        def fail_after_counts_t(*args, **kwargs):
            counts_t_array = original(*args, **kwargs)
            assert counts_t_array is not None
            counts_t_array.attrs["complete"] = False
            raise RuntimeError(f"simulated {error}")

        monkeypatch.setattr(merge_datasets, "write_assay_counts_t", fail_after_counts_t)
        restore = ("write_assay_counts_t", original)

    with pytest.raises(RuntimeError, match=error):
        _merge_two_rna(
            zarr_path=path,
            overwrite=False,
        ).dump()
    assert_interrupted_completion_boundaries()
    monkeypatch.setattr(merge_datasets, restore[0], restore[1])

    if case == "cell_metadata":
        root = zarr.open_group(path, mode="r")
        assert root.attrs["cytearc:import_complete"] is False

    result = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
    ).dump()
    actions = {component.name: component.action for component in result.components}
    assert actions == expected
    assert result.resumed is True
    completed = zarr.open_group(path, mode="r")
    assert _rows_by_id(completed) == _TWO_RNA_ROWS
    np.testing.assert_array_equal(
        completed["RNA/countsT"][:], np.asarray(completed["RNA/counts"][:]).T
    )


def test_dataset_merge_rewrites_counts_t_when_counts_resume(tmp_path, merged_two_rna):
    path = _copy_merge(
        merged_two_rna, "default", tmp_path / "resume_counts_dependency.zarr"
    )
    root = zarr.open_group(path, mode="r+")
    root.attrs["cytearc:import_complete"] = False
    root.attrs["complete"] = False
    root["RNA"].attrs["complete"] = False
    root["RNA/counts"][:] = 0

    result = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
    ).dump()

    actions = {component.name: component.action for component in result.components}
    assert actions["counts:RNA"] == "resume"
    assert actions["countsT:RNA"] == "resume"
    completed = zarr.open_group(path, mode="r")
    # The zeroed counts are rewritten from the sources, and countsT with them.
    assert _rows_by_id(completed) == _TWO_RNA_ROWS
    np.testing.assert_array_equal(
        completed["RNA/countsT"][:],
        np.asarray(completed["RNA/counts"][:]).T,
    )


def test_dataset_merge_blocks_inconsistent_complete_marker(tmp_path, merged_two_rna):
    path = _copy_merge(
        merged_two_rna, "default", tmp_path / "inconsistent_complete.zarr"
    )
    root = zarr.open_group(path, mode="r+")
    root["RNA"].attrs["complete"] = False

    plan = _merge_two_rna(zarr_path=path, overwrite=False).plan()

    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "marked complete" in plan.blockedReason


def test_dataset_merge_workspace_overwrite_preserves_root_siblings(tmp_path):
    path = str(tmp_path / "workspace_preservation.zarr")
    first = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        out_workspace="merged",
    )
    first.dump()
    root = zarr.open_group(path, mode="r+")
    root.create_group("sentinel")
    root.create_group(f"merged/pipeline/runs/{'a' * 64}/stages")
    root.create_group("merged/pipeline/keep")
    root.create_group(f"merged/artifacts/cell_selection/{'b' * 64}")
    root.create_group(f"other_workspace/pipeline/runs/{'c' * 64}/stages")

    _merge_two_rna(
        zarr_path=path,
        overwrite=True,
        out_workspace="merged",
    ).dump()

    completed = zarr.open_group(path, mode="r")
    assert "sentinel" in completed
    assert "merged/cellData" in completed
    assert "matrices/RNA/counts" in completed
    assert "merged/pipeline/runs" not in completed
    assert "merged/pipeline/keep" in completed
    assert "merged/artifacts" not in completed
    assert f"other_workspace/pipeline/runs/{'c' * 64}" in completed


def test_dataset_merge_overwrite_removes_old_assay_components(tmp_path):
    path = str(tmp_path / "removed_assay.zarr")
    left, right = _two_assay_sources()
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    ).dump()

    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        assays=["RNA"],
        prepend_text="",
        overwrite=True,
        seed=0,
    ).dump()

    completed = zarr.open_group(path, mode="r")
    assert "RNA" in completed
    assert "ADT" not in completed


@pytest.mark.parametrize(
    ("workspace", "reason"),
    [
        ("", "Workspace names must be non-empty"),
        ("matrices", "'matrices' is reserved for workspace matrix storage"),
        ("cellData", "'cellData' is reserved for datastore cell metadata"),
        ("a/b", "'a/b' must not contain path separators"),
        ("artifacts", "'artifacts' is reserved for datastore artifact storage"),
    ],
)
def test_dataset_merge_rejects_invalid_workspace_names(tmp_path, workspace, reason):
    path = tmp_path / "invalid_workspace.zarr"
    with pytest.raises(ValueError, match=reason):
        _merge_two_rna(zarr_path=str(path), out_workspace=workspace)
    assert not path.exists()


def test_dataset_merge_rejects_duplicate_assay_filter(tmp_path):
    path = str(tmp_path / "dup_assays.zarr")
    with pytest.raises(ValueError, match="duplicate assay"):
        _merge_two_rna(zarr_path=path, assays=["RNA", "RNA"])
    assert not (tmp_path / "dup_assays.zarr").exists()


def test_dataset_merge_rejects_source_shape_mismatch(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    path = str(tmp_path / "shape_mismatch.zarr")
    with pytest.raises(ValueError, match="rawData has 1 rows"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=path,
            names=["left", "right"],
        ).plan()
    assert not (tmp_path / "shape_mismatch.zarr").exists()


def test_dataset_merge_rejects_membership_source_column_collision(tmp_path):
    path = str(tmp_path / "membership_collision.zarr")
    with pytest.raises(ValueError, match="source_column"):
        _merge_two_rna(zarr_path=path, source_column="RNA_I").plan()
    assert not (tmp_path / "membership_collision.zarr").exists()


def test_dataset_merge_rejects_source_destination_alias(tmp_path):
    path = str(tmp_path / "alias.zarr")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc=path,
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
    )
    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "aliases source" in plan.blockedReason
    with pytest.raises(ValueError, match="aliases source"):
        merger.dump()


def test_dataset_merge_blocks_cross_workspace_matrix_overwrite(tmp_path):
    path = str(tmp_path / "cross_workspace.zarr")
    _merge_two_rna(
        zarr_path=path,
        overwrite=False,
        out_workspace="ws_a",
    ).dump()
    root = zarr.open_group(path, mode="r")
    before_counts = np.asarray(root["matrices/RNA/counts"][:]).copy()
    before_attrs = dict(root["ws_a"].attrs)

    blocked = _merge_two_rna(
        zarr_path=path,
        overwrite=True,
        out_workspace="ws_b",
    )
    plan = blocked.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "claimed by workspace" in plan.blockedReason
    with pytest.raises(ValueError, match="claimed by workspace"):
        blocked.dump()

    after = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(after["matrices/RNA/counts"][:], before_counts)
    assert dict(after["ws_a"].attrs) == before_attrs
    assert "ws_b" not in after


def test_dataset_merge_blocks_orphaned_matrix_slot(tmp_path):
    path = str(tmp_path / "orphan_matrix.zarr")
    root = zarr.open_group(path, mode="w")
    matrices = root.create_group("matrices")
    rna = matrices.create_group("RNA")
    rna.create_array("counts", data=np.ones((2, 2), dtype=np.uint16))
    before = np.asarray(rna["counts"][:]).copy()

    blocked = _merge_two_rna(zarr_path=path, out_workspace="merged", overwrite=True)
    plan = blocked.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "orphaned" in plan.blockedReason
    with pytest.raises(ValueError, match="orphaned"):
        blocked.dump()

    after = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(after["matrices/RNA/counts"][:], before)
    assert "merged" not in after


def test_dataset_merge_missing_modality_excluded_from_overlap(tmp_path):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                ["c0", "c1"],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[3, 30], [4, 40]],
                ["c0", "c1"],
                ["id_c", "id_d"],
                ["C", "D"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    missing = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[9], [8]],
                ["c0", "c1"],
                ["adt"],
                ["ADT"],
                block_size=2,
            )
        ],
        zarr_loc="memory://missing",
    )
    with pytest.raises(ValueError, match="No overlapping features"):
        DataStoreMerge(
            datasets=[left, right, missing],
            zarr_path=str(tmp_path / "disjoint_missing.zarr"),
            names=["left", "right", "missing"],
            assays=["RNA"],
        ).plan()


def test_dataset_merge_missing_assay_plan_fields(tmp_path):
    cell_ids = ["c0", "c1"]
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10], [2, 20]],
                cell_ids,
                ["rna_a", "rna_b"],
                ["RNA A", "RNA B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[101], [102]],
                cell_ids,
                ["adt_a"],
                ["ADT A"],
                block_size=2,
            )
        ],
        zarr_loc="memory://right",
    )
    path = str(tmp_path / "missing_plan.zarr")
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    )
    plan = merger.plan()
    rna = next(item for item in plan.assays if item.assayName == "RNA")
    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert rna.sourcePresent == (True, False)
    assert rna.missingSources == ("right",)
    assert rna.nFeatures == 2
    assert rna.featureOverlapFraction == 1.0
    assert adt.sourcePresent == (False, True)
    assert adt.missingSources == ("left",)

    result = merger.dump()
    root = zarr.open_group(path, mode="r")
    assert result.nCells == 4
    # The right source lacks RNA, so its cells hold zeros outside the assay.
    assert _rows_by_id(root) == {
        "left__c0": [1, 10],
        "left__c1": [2, 20],
        "right__c0": [0, 0],
        "right__c1": [0, 0],
    }
    assert _rows_by_id(root, "cellData/RNA_I") == {
        "left__c0": True,
        "left__c1": True,
        "right__c0": False,
        "right__c1": False,
    }


def test_dataset_merge_default_counts_t_follows_real_assay_type(
    datastore,
    tmp_path,
):
    def counts_t_plan(plan):
        # Each merge plans one assay, of the type its sources declare.
        (assay,) = plan.assays
        return assay.assayName, assay.assayType, assay.writeCountsT, assay.countsTAction

    rna_path = str(tmp_path / "default_rna.zarr")
    rna_plan = DataStoreMerge(
        datasets=[datastore, datastore],
        zarr_path=rna_path,
        names=["a", "b"],
        assays=["RNA"],
        prepend_text="",
        overwrite=True,
    ).plan()
    assert counts_t_plan(rna_plan) == ("RNA", "RNA", True, "write")

    adt_only = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[1, 2], [3, 4]],
                ["c0", "c1"],
                ["a", "b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://adt_left",
    )
    adt_right = _MergeDataStore(
        [
            _MergeAssay(
                "ADT",
                [[5, 6], [7, 8]],
                ["c0", "c1"],
                ["a", "b"],
                ["A", "B"],
                block_size=2,
            )
        ],
        zarr_loc="memory://adt_right",
    )
    adt_plan = DataStoreMerge(
        datasets=[adt_only, adt_right],
        zarr_path=str(tmp_path / "default_adt.zarr"),
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
    ).plan()
    assert counts_t_plan(adt_plan) == ("ADT", "ADT", False, "skip")

    mock_plan = _merge_two_rna(zarr_path=str(tmp_path / "mock_rna.zarr")).plan()
    assert counts_t_plan(mock_plan) == ("RNA", "RNA", True, "write")


def test_dataset_merge_skips_counts_t_for_generic_assay_named_rna(tmp_path):
    """A generic Assay whose group is named RNA must not get countsT."""
    from cytearc import DataStore
    from cytearc.storage.schema import create_cell_data
    from cytearc.writers import create_zarr_count_assay

    def _generic_named_rna(path: str, values: np.ndarray) -> DataStore:
        root = zarr.open_group(path, mode="w")
        n_cells, n_feats = values.shape
        create_cell_data(
            root,
            None,
            ids=np.array([f"c{i}" for i in range(n_cells)]),
            names=np.array([f"c{i}" for i in range(n_cells)]),
        )
        create_zarr_count_assay(
            root,
            "RNA",
            None,
            n_cells,
            feat_ids=np.array([f"f{i}" for i in range(n_feats)]),
            feat_names=np.array([f"g{i}" for i in range(n_feats)]),
            dtype="uint32",
        )
        root["RNA/counts"][:] = values
        root.attrs["assayTypes"] = {"RNA": "Assay"}
        from tests.storage_helpers import finalize_test_counts

        finalize_test_counts(root["RNA/counts"])
        return DataStore(
            path,
            default_assay="RNA",
            assay_types={"RNA": "Assay"},
            min_features_per_cell=0,
        )

    left = _generic_named_rna(
        str(tmp_path / "left.zarr"),
        np.array([[1, 10], [2, 20]], dtype=np.uint32),
    )
    right = _generic_named_rna(
        str(tmp_path / "right.zarr"),
        np.array([[3, 30], [4, 40]], dtype=np.uint32),
    )
    out = str(tmp_path / "merged.zarr")
    plan = DataStoreMerge(
        datasets=[left, right],
        zarr_path=out,
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
        seed=0,
    ).plan()
    assert plan.assays[0].writeCountsT is False
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=out,
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
        seed=0,
    ).dump()
    root = zarr.open_group(out, mode="r")
    assert "countsT" not in root["RNA"]
    assert root.attrs["assayTypes"]["RNA"] == "Assay"


def test_dataset_merge_resumes_after_partial_counts_band(tmp_path, monkeypatch):
    import cytearc.merge.writer as merge_writer

    path = str(tmp_path / "partial_band.zarr")
    # Counts are written densely when the feature mapping is injective and
    # through sparse bands otherwise; whichever writer runs is interrupted.
    writers = ("accumulate_sparse_to_shards", "write_dense_from_row_batches")
    originals = {name: getattr(merge_writer, name) for name in writers}

    def fail_after_persisting(write):
        def interrupted(dst, data_stream, **kwargs):
            rows = write(dst, data_stream, **kwargs)
            assert rows > 0
            assert int(np.asarray(dst[:]).sum()) > 0
            raise RuntimeError("simulated partial counts band")

        return interrupted

    for name in writers:
        monkeypatch.setattr(merge_writer, name, fail_after_persisting(originals[name]))
    with pytest.raises(RuntimeError, match="partial counts band"):
        _merge_two_rna(
            zarr_path=path,
            overwrite=False,
        ).dump()
    for name in writers:
        monkeypatch.setattr(merge_writer, name, originals[name])

    interrupted = zarr.open_group(path, mode="r+")
    interrupted.create_group("sentinel")
    assert interrupted.attrs.get("cytearc:import_complete") is not True
    assert interrupted["RNA"].attrs.get("complete") is not True

    result = _merge_two_rna(
        zarr_path=path,
        overwrite=False,
    ).dump()
    actions = {component.name: component.action for component in result.components}
    assert actions["counts:RNA"] == "resume"
    assert actions["countsT:RNA"] == "resume"
    completed = zarr.open_group(path, mode="r")
    assert completed.attrs["cytearc:import_complete"] is True
    assert completed.attrs["complete"] is True
    assert completed["RNA"].attrs["complete"] is True
    assert completed["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        completed["RNA/countsT"][:],
        np.asarray(completed["RNA/counts"][:]).T,
    )
    assert "sentinel" in completed
    assert _rows_by_id(completed) == _TWO_RNA_ROWS


def test_dataset_merge_metadata_admission_bounds_selection(tmp_path, monkeypatch):
    import cytearc.merge.metadata as merge_metadata
    import cytearc.merge.row_plan as merge_row_plan

    widths: list[int] = []
    original_meta = merge_metadata.read_metadata_rows_chunkwise

    def tracking_read(table, column, rows):
        widths.append(int(np.asarray(rows).size))
        return original_meta(table, column, rows)

    monkeypatch.setattr(
        merge_metadata,
        "read_metadata_rows_chunkwise",
        tracking_read,
    )
    monkeypatch.setattr(
        merge_row_plan,
        "read_metadata_rows_chunkwise",
        tracking_read,
    )
    path = str(tmp_path / "meta_admit.zarr")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.arange(20, dtype=np.int64).reshape(10, 2),
                [f"c{i}" for i in range(10)],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=3,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.arange(20, 40, dtype=np.int64).reshape(10, 2),
                [f"c{i}" for i in range(10)],
                ["id_a", "id_b"],
                ["A", "B"],
                block_size=4,
            )
        ],
        zarr_loc="memory://right",
    )
    # The budget admits windows of four merged rows, which bound every read
    # of a source's rows.
    monkeypatch.setattr(
        merge_metadata, "resolve_metadata_segment_rows", lambda *args, **kwargs: 4
    )
    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        overwrite=True,
    ).dump()
    assert widths
    assert max(widths) <= 4


def test_dataset_merge_schema_scan_uses_admitted_width_without_changing_schema(
    tmp_path,
    monkeypatch,
):
    import cytearc.merge.metadata as merge_metadata

    cell_ids = [f"c{i}" for i in range(7)] + ["identifier_is_longest"]
    cell_names = [f"cell {i}" for i in range(7)] + ["longest cell name"]

    def source(name, offset):
        assay = _MergeAssay(
            "RNA",
            np.arange(offset, offset + 8, dtype=np.uint16).reshape(8, 1),
            cell_ids,
            ["id_a"],
            ["A"],
            block_size=4,
        )
        assay.cells._columns["names"] = np.asarray(cell_names)
        return _MergeDataStore([assay], zarr_loc=f"memory://{name}")

    widths: list[int] = []
    original_blocks = merge_metadata.iter_metadata_column_blocks

    def tracking_blocks(*args, **kwargs):
        for block in original_blocks(*args, **kwargs):
            widths.append(int(block.size))
            yield block

    def admitted_scan_rows(
        source_cell_tables,
        resources,
        *,
        resident_bytes,
        preferred_rows,
    ):
        _ = source_cell_tables, resources, resident_bytes
        # One whole merged metadata chunk covers all 16 cells.
        assert preferred_rows == 16
        return 2

    monkeypatch.setattr(
        merge_metadata,
        "iter_metadata_column_blocks",
        tracking_blocks,
    )
    monkeypatch.setattr(
        merge_datasets,
        "resolve_metadata_schema_scan_rows",
        admitted_scan_rows,
    )
    path = tmp_path / "schema_scan.zarr"
    merger = DataStoreMerge(
        datasets=[source("left", 0), source("right", 8)],
        zarr_path=str(path),
        names=["left", "right"],
        prepend_text="",
        overwrite=True,
        seed=0,
    )
    merger.plan()

    assert widths
    assert max(widths) == 2
    assert merger._metadataPlan is not None
    assert merger._metadataPlan.blockRows == 16
    specs = {spec.name: spec for spec in merger._metadataPlan.columns}
    assert specs["ids"].dtype == np.dtype(f"U{len('right') + 2 + len(cell_ids[-1])}")
    assert specs["names"].dtype == np.dtype(f"U{len(cell_names[-1])}")
    assert not path.exists()


def test_dataset_merge_metadata_admission_shrinks_under_budget(monkeypatch):
    import cytearc.merge.metadata as merge_metadata
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.budget import ResourceBudget

    left = _MergeMeta(
        block_rows=10,
        ids=[f"c{i}" for i in range(20)],
        names=[f"c{i}" for i in range(20)],
        I=np.ones(20, dtype=bool),
    )
    right = _MergeMeta(
        block_rows=10,
        ids=[f"c{i}" for i in range(20)],
        names=[f"c{i}" for i in range(20)],
        I=np.ones(20, dtype=bool),
    )
    metadata_plan = merge_metadata.plan_cell_metadata(
        [left, right],
        ["left", "right"],
        prepend_text="",
        reset_cell_filter=True,
        source_column=None,
        membership={"RNA": ["all", "all"]},
        block_rows=10,
    )
    row_plan = build_row_plan([20, 20], ["left", "right"], seed=0)
    assert metadata_plan.blockRows == 10

    def inflated(self, rows, *, chunk_rows):
        _ = chunk_rows
        return max(1, int(rows)) * 1_000

    monkeypatch.setattr(
        merge_metadata.CellMetadataPlan,
        "peak_write_bytes_at",
        inflated,
    )
    resources = ResourceBudget(4_500, 1)
    admitted = merge_metadata.resolve_metadata_segment_rows(
        metadata_plan,
        row_plan,
        resources,
        resident_bytes=2_000,
    )
    # The 2,500 bytes left beside the resident data hold two 1,000-byte rows.
    assert admitted == 2

    updated = merge_metadata.admit_cell_metadata_plan(
        metadata_plan,
        row_plan,
        resources,
        resident_bytes=2_000,
    )
    assert updated.blockRows == admitted
    assert updated.columns == metadata_plan.columns


@pytest.mark.parametrize("write", ["dense", "sparse"])
def test_dataset_merge_sparse_write_respects_admitted_batch_geometry(
    tmp_path,
    monkeypatch,
    write,
):
    import cytearc.merge.writer as merge_writer

    path = str(tmp_path / "sparse_geometry.zarr")
    n_left = 12
    n_right = 12
    n_feats = 8
    left_counts = np.arange(n_left * n_feats, dtype=np.uint16).reshape(n_left, n_feats)
    right_counts = (
        np.arange(n_right * n_feats, dtype=np.uint16).reshape(n_right, n_feats) + 100
    )
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                left_counts,
                [f"c{i}" for i in range(n_left)],
                [f"id_{i}" for i in range(n_feats)],
                [f"G{i}" for i in range(n_feats)],
                block_size=4,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                right_counts,
                [f"c{i}" for i in range(n_right)],
                [f"id_{i}" for i in range(n_feats)],
                [f"G{i}" for i in range(n_feats)],
                block_size=3,
            )
        ],
        zarr_loc="memory://right",
    )

    batches: dict[str, list[int]] = {"dense": [], "sparse": []}
    batch_rows: list[int] = []
    original_remap = merge_writer.remap_block_to_coo
    original_scatter = merge_writer.scatter_block_dense
    original_batch = merge_writer.resolve_sparse_import_batch

    def tracking_remap(block, order_map, n_feats, nthreads, destination_dtype=None):
        result = original_remap(
            block,
            order_map,
            n_feats,
            nthreads,
            destination_dtype,
        )
        batches["sparse"].append(int(result.shape[0]))
        return result

    def tracking_scatter(block, order_map, n_feats, destination_dtype, **options):
        batches["dense"].append(int(np.asarray(block).shape[0]))
        return original_scatter(block, order_map, n_feats, destination_dtype, **options)

    def tracking_batch(*args, **kwargs):
        plan = original_batch(*args, **kwargs)
        batch_rows.append(int(plan.batchRows))
        return plan

    def refused_dense_write(*args, **kwargs):
        raise MemoryBudgetError(
            "dense write refused", needed_bytes=2, budget_bytes=1, item="the write"
        )

    monkeypatch.setattr(merge_writer, "remap_block_to_coo", tracking_remap)
    monkeypatch.setattr(merge_writer, "scatter_block_dense", tracking_scatter)
    monkeypatch.setattr(merge_writer, "resolve_sparse_import_batch", tracking_batch)
    if write == "sparse":
        # Unique feature IDs write densely unless the dense write is refused.
        monkeypatch.setattr(merge_writer, "plan_dense_write", refused_dense_write)

    DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        overwrite=True,
        mem_budget=256 * 1024,
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=64, chunkBytes=32),
    ).dump()

    assert batch_rows
    admitted = max(batch_rows)
    assert batches[write]
    assert batches["sparse" if write == "dense" else "dense"] == []
    assert all(rows <= admitted for rows in batches[write])
    assert sum(batches[write]) == n_left + n_right
    completed = zarr.open_group(path, mode="r")
    assert _rows_by_id(completed) == {
        **{f"left__c{i}": row.tolist() for i, row in enumerate(left_counts)},
        **{f"right__c{i}": row.tolist() for i, row in enumerate(right_counts)},
    }


def _sparse_rna_sources(n_cells=12, n_feats=8):
    rng = np.random.default_rng(3)
    sources = []
    for label in ("left", "right"):
        counts = rng.integers(1, 9, size=(n_cells, n_feats), dtype=np.uint16)
        counts[rng.random(counts.shape) < 0.6] = 0
        sources.append(
            _MergeDataStore(
                [
                    _MergeAssay(
                        "RNA",
                        counts,
                        [f"c{i}" for i in range(n_cells)],
                        [f"id_{i}" for i in range(n_feats)],
                        [f"G{i}" for i in range(n_feats)],
                        block_size=4,
                    )
                ],
                zarr_loc=f"memory://{label}",
            )
        )
    return sources


def test_dataset_merge_counts_preflight_matches_execution_accounting(
    tmp_path, monkeypatch
):
    import cytearc.merge.writer as merge_writer
    from cytearc.storage import sharding

    preflight: list[int] = []
    execution: list[int] = []
    window_nnz: list[int] = []
    original_spec = sharding.resolve_sparse_import_spec
    original_batch = merge_writer.resolve_sparse_import_batch

    def tracking_spec(*args, **kwargs):
        preflight.append(int(kwargs["residentBytes"]))
        return original_spec(*args, **kwargs)

    def tracking_batch(*args, **kwargs):
        execution.append(int(kwargs["residentBytes"]))
        window_nnz.append(int(kwargs["maxWindowNnz"](3)))
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(sharding, "resolve_sparse_import_spec", tracking_spec)
    monkeypatch.setattr(merge_writer, "resolve_sparse_import_batch", tracking_batch)
    path = str(tmp_path / "accounting.zarr")
    DataStoreMerge(
        datasets=_sparse_rna_sources(),
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        nthreads=1,
    ).dump()

    # Preflight admits exactly the resident bytes execution holds, including
    # the destination count summary.
    assert preflight
    assert preflight == execution
    counts = zarr.open_group(path, mode="r")["RNA/counts"][:]
    cumulative = np.concatenate([[0], np.cumsum(np.count_nonzero(counts, axis=1))])
    assert window_nnz == [int(np.max(cumulative[3:] - cumulative[:-3]))]


def test_dataset_merge_counts_t_preflight_uses_merge_policy(tmp_path, monkeypatch):
    from cytearc.storage import sharding
    from cytearc.storage.types import array_metadata_shards

    admitted = []
    original = sharding.preflight_counts_t_spec

    def tracking(*args, **kwargs):
        spec = original(*args, **kwargs)
        admitted.append(spec)
        return spec

    # The layout fit admits the countsT transpose of the counts it plans.
    monkeypatch.setattr(sharding, "preflight_counts_t_spec", tracking)
    path = str(tmp_path / "policy.zarr")
    DataStoreMerge(
        datasets=_sparse_rna_sources(),
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=64, chunkBytes=32),
    ).dump()

    counts_t = zarr.open_group(path, mode="r")["RNA/countsT"]
    assert admitted
    assert tuple(admitted[-1].chunks) == tuple(counts_t.chunks)
    assert tuple(admitted[-1].shards) == tuple(array_metadata_shards(counts_t))


def test_dataset_merge_writes_cell_metadata_in_whole_chunk_bands(tmp_path, monkeypatch):
    import cytearc.merge.metadata as merge_metadata

    def merger(path):
        left, right = _sparse_rna_sources()
        left.cells._columns["quality"] = np.arange(12, dtype=np.int16)
        left.cells.columns.append("quality")
        return DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(path),
            names=["left", "right"],
            prepend_text="",
            seed=0,
            nthreads=1,
        )

    writes: list[tuple[int, int]] = []
    original_create = merge_metadata.create_streamed_metadata_column

    def tracking_create(group, name, *, blocks, **kwargs):
        def tracked():
            for block in blocks:
                writes.append((block.start, len(block.values)))
                yield block

        return original_create(group, name, blocks=tracked(), **kwargs)

    # Five-row chunks make bands straddle the four-row source blocks.
    monkeypatch.setattr(merge_metadata, "PROFILE_METADATA_CHUNK", 5)
    monkeypatch.setattr(
        merge_metadata, "create_streamed_metadata_column", tracking_create
    )
    banded_path = tmp_path / "banded.zarr"
    merger(banded_path).dump()

    assert writes
    assert all(start % 5 == 0 and size == min(5, 24 - start) for start, size in writes)
    root = zarr.open_group(banded_path, mode="r")
    banded = root["cellData"]
    assert set(banded.array_keys()) == {
        "ids",
        "names",
        "I",
        "RNA_I",
        "sourceRow",
        "quality",
        "__cytearc_missing__quality",
    }
    for name in banded.array_keys():
        assert banded[name].chunks == (5,)
    # Bands that straddle the source blocks keep every value with its cell.
    cells = [f"{side}__c{i}" for side in ("left", "right") for i in range(12)]
    assert _rows_by_id(root, "cellData/names") == {
        cell: cell.split("__")[1] for cell in cells
    }
    assert _rows_by_id(root, "cellData/I") == dict.fromkeys(cells, True)
    assert _rows_by_id(root, "cellData/RNA_I") == dict.fromkeys(cells, True)
    # Each cell's row in the concatenation of the sources, left then right.
    assert _rows_by_id(root, "cellData/sourceRow") == {
        cell: int(cell.split("__c")[1]) + (0 if cell.startswith("left") else 12)
        for cell in cells
    }
    assert _rows_by_id(root, "cellData/quality") == {
        cell: int(cell.split("__c")[1]) if cell.startswith("left") else 0
        for cell in cells
    }
    assert _rows_by_id(root, "cellData/__cytearc_missing__quality") == {
        cell: cell.startswith("right") for cell in cells
    }
    resumed = merger(banded_path).plan()
    assert resumed.cellDataAction == "skip"

    ids = zarr.open_group(banded_path, mode="r+")["cellData/ids"]
    swapped = np.asarray(ids[:])
    swapped[[4, 5]] = swapped[[5, 4]]
    ids[:] = swapped
    blocked = merger(banded_path).plan()
    assert blocked.canDump is False
    assert "cell IDs or their order differ" in blocked.blockedReason


def test_dataset_merge_producer_reserve_uses_source_feature_width():
    import cytearc.merge.writer as merge_writer
    from cytearc.merge.features import align_features
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.budget import ResourceBudget

    def assay(counts, feature_ids, feature_names, cell_ids):
        return _MergeAssay(
            "RNA",
            counts,
            cell_ids,
            feature_ids,
            feature_names,
            block_size=len(cell_ids),
        )

    cell_ids = ["c0", "c1", "c2", "c3"]
    # Partial overlap: union is wider than either source matrix.
    partial = [
        assay(
            np.ones((4, 3), dtype=np.uint16),
            ["a", "b", "c"],
            ["A", "B", "C"],
            cell_ids,
        ),
        assay(
            np.ones((4, 4), dtype=np.uint16) * 2,
            ["a", "d", "e", "f"],
            ["A", "D", "E", "F"],
            cell_ids,
        ),
    ]
    dense = [
        assay(
            np.ones((4, 5), dtype=np.uint16),
            ["a", "b", "c", "d", "e"],
            ["A", "B", "C", "D", "E"],
            cell_ids,
        ),
        assay(
            np.ones((4, 5), dtype=np.uint16) * 3,
            ["a", "b", "c", "d", "e"],
            ["A", "B", "C", "D", "E"],
            cell_ids,
        ),
    ]
    suffixed = [
        assay(
            np.ones((4, 2), dtype=np.uint16),
            ["gene_0", "gene_1"],
            ["gene_0", "gene_1"],
            cell_ids,
        ),
        assay(
            np.ones((4, 2), dtype=np.uint16) * 4,
            ["gene_0", "gene_1"],
            ["gene_0", "gene_1"],
            cell_ids,
        ),
    ]
    resources = ResourceBudget(1024**3, 1)

    def requirements_for(assays):
        alignment = align_features(assays, ["left", "right"])
        row_plan = build_row_plan([4, 4], ["left", "right"], seed=0)
        return alignment, merge_writer._merge_import_requirements(
            assays,
            row_plan,
            alignment,
            np.dtype(np.uint16),
            resources=resources,
        )

    alignment, requirements = requirements_for(partial)
    source_width = 4
    assert alignment.nFeats == 6
    assert alignment.nFeats > source_width
    value_bytes = np.dtype(np.uint16).itemsize
    index_bytes = 2 * np.dtype(np.int32).itemsize
    expected = 2 * source_width * value_bytes + 2 * source_width * (
        value_bytes + index_bytes
    )
    # Ignore backing decode, which is source-dependent and additive.
    assert requirements.extraProducerBytes(2) >= expected
    assert requirements.extraProducerBytes(2) < (
        2 * alignment.nFeats * value_bytes
        + 2 * alignment.nFeats * (value_bytes + index_bytes)
        + 1024**2
    )

    dense_alignment, dense_requirements = requirements_for(dense)
    assert dense_alignment.nFeats == 5
    dense_expected = 2 * 5 * value_bytes + 2 * 5 * (value_bytes + index_bytes)
    assert dense_requirements.extraProducerBytes(2) >= dense_expected
    assert dense_requirements.extraProducerBytes(
        4
    ) > dense_requirements.extraProducerBytes(2)

    suffixed_alignment, suffixed_requirements = requirements_for(suffixed)
    assert suffixed_alignment.nFeats == 2
    suffixed_expected = 2 * 2 * value_bytes + 2 * 2 * (value_bytes + index_bytes)
    destination_only = (
        2 * suffixed_alignment.nFeats * value_bytes
        + 2 * suffixed_alignment.nFeats * (value_bytes + index_bytes)
    )
    assert suffixed_requirements.extraProducerBytes(2) >= suffixed_expected
    assert suffixed_requirements.extraProducerBytes(2) >= destination_only

    with pytest.raises(ValueError, match="No overlapping features"):
        align_features(
            [
                assay(
                    np.ones((4, 2), dtype=np.uint16),
                    ["a", "b"],
                    ["A", "B"],
                    cell_ids,
                ),
                assay(
                    np.ones((4, 2), dtype=np.uint16) * 2,
                    ["c", "d"],
                    ["C", "D"],
                    cell_ids,
                ),
            ],
            ["left", "right"],
        )


def test_dataset_merge_charges_a_source_it_reads_its_counts_and_smallest_read():
    import cytearc.merge.writer as merge_writer
    from cytearc.storage.budget import ResourceBudget

    cell_ids = ["c0", "c1", "c2", "c3"]
    features = [f"g{index}" for index in range(3_000)]

    def assay(counts):
        return _MergeAssay("RNA", counts, cell_ids, features, features, block_size=4)

    # Four cells of 3,000 int16 features, which the merge reads to count.
    signed = assay(np.full((4, 3_000), -1, dtype=np.int16))
    reader = signed.rawData._with_resources(ResourceBudget(10**9, 1))
    # 8 bytes for each cell's count, the matrix that the stream keeps in
    # memory (24,000 bytes), one block of its 4 rows (24,000 bytes), nothing
    # decoded, and the boolean copy and the count that count_nonzero makes of
    # each row (3,000 + 8 bytes).
    expected = 4 * 8 + 24_000 + 24_000 + 4 * (3_000 + 8)
    assert merge_writer._row_nnz_load_bytes(signed, reader) == expected
    # Counts whose summaries count their values read that summary whole, in
    # one chunk as finalize_counts stores it: each cell's count, 8 bytes, a
    # decoded copy of the chunk and an encoded one at its bound, 1/128 more
    # and 1 KiB, and the objects of Zarr's read, 32 KiB, 4 KiB for the chunk
    # and 16 KiB for its decode, whatever the number of features.
    summarized = assay(np.ones((4, 3_000), dtype=np.uint16))
    summary = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "rowNonzero", data=np.full(4, 3_000, dtype=np.int64), chunks=(4,)
    )
    expected = 4 * 8 + 4 * 8 + (4 * 8 + 1_024) + 32_768 + 4_096 + 16_384
    assert merge_writer._row_nnz_load_bytes(summarized, summary) == expected
    # Counts without features or without cells hold no value to read: each
    # cell's count of 0 is all that they hold, within a budget of just it.
    for n_cells, n_features in ((5, 0), (0, 4)):
        empty = _MergeAssay(
            "RNA",
            np.zeros((n_cells, n_features), dtype=np.int16),
            [f"c{index}" for index in range(n_cells)],
            features[:n_features],
            features[:n_features],
            block_size=max(1, n_cells),
        )
        resources = ResourceBudget(max(1, 8 * n_cells), 1)
        reader = empty.rawData._with_resources(resources)
        assert merge_writer._row_nnz_load_bytes(empty, reader) == 8 * n_cells
        counted = merge_writer._counted_row_values(
            empty, "left", reader, resources, held_bytes=0
        )
        np.testing.assert_array_equal(counted, np.zeros(n_cells, dtype=np.int64))


def test_dataset_merge_rejects_insufficient_metadata_budget(tmp_path):
    path = str(tmp_path / "meta_budget.zarr")
    # Ordering the 4 merged cells holds 24 bytes for each of them, the
    # shuffle's hashes and order and each cell's source; beyond that, the
    # budget leaves no room for the schema of the cell metadata.
    for budget, message in (
        (95, "Ordering 4 merged cells holds about 96 bytes"),
        (96, "schema discovery"),
    ):
        merger = _merge_two_rna(zarr_path=path, mem_budget=budget, nthreads=1)
        with pytest.raises(MemoryError, match=message):
            merger.plan()
    assert not (tmp_path / "meta_budget.zarr").exists()


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        # The cell metadata fits, but not counting each merged cell's values
        # beside a whole read of a source's count summary.
        (
            170_000,
            "Counting the nonzero values of each merged cell of assay 'RNA' holds "
            r"about \d+ bytes for each merged cell's count and the count "
            "summaries of source 'left'",
        ),
        # Counting fits, but not the counts of one row.
        (
            250_000,
            "The merge does not fit mem_budget even with count shards of one row. "
            "Increase mem_budget. Resident data needs about",
        ),
    ],
)
def test_dataset_merge_rejects_insufficient_counts_budget(tmp_path, budget, reason):
    path = str(tmp_path / "counts_budget.zarr")
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.ones((8, 1_000), dtype=np.uint16),
                [f"c{i}" for i in range(8)],
                [f"id_{i}" for i in range(1_000)],
                [f"G{i}" for i in range(1_000)],
                block_size=4,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.ones((8, 1_000), dtype=np.uint16) * 2,
                [f"c{i}" for i in range(8)],
                [f"id_{i}" for i in range(1_000)],
                [f"G{i}" for i in range(1_000)],
                block_size=4,
            )
        ],
        zarr_loc="memory://right",
    )
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        mem_budget=budget,
        nthreads=1,
        overwrite=True,
    )
    # These budgets hold the cell metadata, which 1,000 features of 8 cells
    # need about 136,000 bytes for, but not the counts write that follows.
    with pytest.raises(MemoryError, match=f"^{reason}"):
        merger.plan()
    assert not (tmp_path / "counts_budget.zarr").exists()


@pytest.mark.parametrize(
    ("left_values", "right_values", "expected_dtype", "expected"),
    [
        (
            np.array([True, False]),
            np.array([2, 3], dtype=np.int16),
            np.dtype(np.int16),
            {
                "left__c0": 1,
                "left__c1": 0,
                "right__c0": 2,
                "right__c1": 3,
            },
        ),
        (
            np.array([-2, 3], dtype=np.int16),
            np.array([1.5, 2.5], dtype=np.float32),
            np.dtype(np.float64),
            {
                "left__c0": -2.0,
                "left__c1": 3.0,
                "right__c0": 1.5,
                "right__c1": 2.5,
            },
        ),
        (
            np.array([1, 2], dtype=np.int16),
            np.array(["three", "four"]),
            np.dtype("U5"),
            {
                "left__c0": "1",
                "left__c1": "2",
                "right__c0": "three",
                "right__c1": "four",
            },
        ),
    ],
)
def test_dataset_merge_promotes_metadata_types(
    left_values,
    right_values,
    expected_dtype,
    expected,
):
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    for source, values in zip(
        merger.datasets,
        (left_values, right_values),
        strict=True,
    ):
        metadata = _MergeMeta(
            ids=["c0", "c1"],
            names=["c0", "c1"],
            I=np.ones(2, dtype=bool),
            score=values,
        )
        source.cells = metadata
        source.get_assay("RNA").cells = metadata

    merger.dump()

    root = zarr.open_group(destination, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    values = np.asarray(root["cellData/score"][:])
    assert values.dtype == expected_dtype
    assert dict(zip(ids, values.tolist(), strict=True)) == expected


def test_dataset_merge_preserves_typed_fill_values_and_missing_masks():
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    left, right = merger.datasets
    left.cells = _MergeMeta(
        ids=["c0", "c1"],
        names=["c0", "c1"],
        I=np.ones(2, dtype=bool),
        passed=np.array([True, False]),
        score=np.array([1.25, 2.5], dtype=np.float32),
    )
    right.cells = _MergeMeta(
        ids=["c0", "c1"],
        names=["c0", "c1"],
        I=np.ones(2, dtype=bool),
    )
    left.get_assay("RNA").cells = left.cells
    right.get_assay("RNA").cells = right.cells

    merger.dump()

    root = zarr.open_group(destination, mode="r")
    cell_data = root["cellData"]
    ids = np.asarray(cell_data["ids"][:]).astype(str)
    positions = {cell_id: index for index, cell_id in enumerate(ids)}
    passed = np.asarray(cell_data["passed"][:])
    passed_missing = np.asarray(cell_data["__cytearc_missing__passed"][:])
    score = np.asarray(cell_data["score"][:])
    score_missing = np.asarray(cell_data["__cytearc_missing__score"][:])
    assert passed.dtype == np.dtype(bool)
    assert score.dtype == np.dtype(np.float32)
    for cell_id in ("left__c0", "left__c1"):
        assert bool(passed_missing[positions[cell_id]]) is False
        assert bool(score_missing[positions[cell_id]]) is False
    for cell_id in ("right__c0", "right__c1"):
        position = positions[cell_id]
        assert bool(passed_missing[position]) is True
        assert bool(score_missing[position]) is True
        assert bool(passed[position]) is False
        assert np.isnan(score[position])


def test_dataset_merge_preserves_signed_counts_and_feature_identity():
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.array([[100, 100]], dtype=np.int8),
                ["c0"],
                ["gene_0", "gene_1"],
                ["gene_0", "gene_1"],
                block_size=1,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                np.array([[-100, -100]], dtype=np.int8),
                ["c0"],
                ["gene_0", "gene_1"],
                ["gene_0", "gene_1"],
                block_size=1,
            )
        ],
        zarr_loc="memory://right",
    )
    destination = MemoryStore()
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=destination,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    )

    plan = merger.plan()
    assert plan.assays[0].dtype == "int8"
    assert plan.assays[0].nFeatures == 2
    merger.dump()

    root = zarr.open_group(destination, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["RNA/counts"][:])
    assert counts.dtype == np.dtype(np.int8)
    assert {
        cell_id: row.tolist() for cell_id, row in zip(ids, counts, strict=True)
    } == {
        "left__c0": [100, 100],
        "right__c0": [-100, -100],
    }


@pytest.mark.parametrize(
    ("left_dtype", "right_dtype", "expected"),
    [
        (np.uint16, np.uint32, np.uint32),
        (np.int16, np.uint16, np.int32),
        (np.uint8, np.float32, np.float32),
    ],
)
def test_dataset_merge_uses_the_common_type_of_the_source_count_dtypes(
    left_dtype, right_dtype, expected
):
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    merger.datasets[0].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.array([[1, 10], [2, 20]], dtype=left_dtype),
        block_size=2,
    )
    merger.datasets[1].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.array([[3, 30], [4, 40]], dtype=right_dtype),
        block_size=2,
    )

    plan = merger.plan()
    assert plan.assays[0].dtype == np.dtype(expected).name
    merger.dump()

    root = zarr.open_group(destination, mode="r")
    assert root["RNA/counts"].dtype == np.dtype(expected)
    assert _rows_by_id(root) == _TWO_RNA_ROWS


def test_dataset_merge_rejects_integer_sources_without_a_common_integer_dtype():
    merger = _merge_two_rna(zarr_path=MemoryStore(), overwrite=False)
    merger.datasets[0].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.array([[1, 10], [2, 20]], dtype=np.uint64),
        block_size=2,
    )
    merger.datasets[1].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.array([[-3, 30], [4, 40]], dtype=np.int64),
        block_size=2,
    )
    # Their common type is float64, which would round counts past 2**53.
    with pytest.raises(ValueError, match="uint64 have no common integer dtype"):
        merger.plan()


@pytest.mark.parametrize(
    ("left_ids", "left_names", "right_ids", "right_names"),
    [
        (
            ["gene_0", "gene_1"],
            ["gene_0", "gene_1"],
            ["right_a", "right_b"],
            ["gene", "gene"],
        ),
        (
            ["gene_1", "gene_2"],
            ["gene_1", "gene_2"],
            ["gene_0", "gene_1"],
            ["gene_0", "gene_1"],
        ),
    ],
)
def test_dataset_merge_matches_suffixed_features_by_exact_id(
    left_ids,
    left_names,
    right_ids,
    right_names,
):
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10]],
                ["c0"],
                left_ids,
                left_names,
                block_size=1,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[2, 20]],
                ["c0"],
                right_ids,
                right_names,
                block_size=1,
            )
        ],
        zarr_loc="memory://right",
    )
    destination = MemoryStore()
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=destination,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    )

    if not set(left_ids).intersection(right_ids):
        with pytest.raises(ValueError, match="No overlapping features"):
            merger.plan()
        return
    plan = merger.plan()
    assert plan.assays[0].nFeatures == 3
    assert plan.assays[0].featureOverlapFraction == pytest.approx(1 / 3)
    merger.dump()

    root = zarr.open_group(destination, mode="r")
    assert np.asarray(root["RNA/featureData/ids"][:]).astype(str).tolist() == [
        "gene_1",
        "gene_2",
        "gene_0",
    ]
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["RNA/counts"][:])
    assert {
        cell_id: row.tolist() for cell_id, row in zip(ids, counts, strict=True)
    } == {
        "left__c0": [1, 10, 0],
        "right__c0": [20, 0, 2],
    }


def test_dataset_merge_rejects_disjoint_suffixed_feature_ids():
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1]],
                ["c0"],
                ["gene_2"],
                ["gene_2"],
                block_size=1,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[2]],
                ["c0"],
                ["gene_0"],
                ["gene_0"],
                block_size=1,
            )
        ],
        zarr_loc="memory://right",
    )
    with pytest.raises(ValueError, match="No overlapping features"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=MemoryStore(),
            names=["left", "right"],
        ).plan()


def _id_scheme_sources():
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 10, 100]],
                ["c0"],
                ["ENSG01", "ENSG02", "ENSG03"],
                ["A", "B", "B"],
                block_size=1,
            )
        ],
        zarr_loc="memory://left",
    )
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[2, 20]],
                ["c0"],
                ["A", "C"],
                ["A", "C"],
                block_size=1,
            )
        ],
        zarr_loc="memory://right",
    )
    return left, right


def test_dataset_merge_matches_features_by_name_across_id_schemes():
    left, right = _id_scheme_sources()
    with pytest.raises(ValueError, match="feature_key='names'"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=MemoryStore(),
            names=["left", "right"],
        ).plan()

    destination = MemoryStore()
    merger = DataStoreMerge(
        datasets=[left, right],
        zarr_path=destination,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        feature_key="names",
    )
    plan = merger.plan()
    assert plan.manifest["featureKey"] == "names"
    assert plan.assays[0].nFeatures == 3
    merger.dump()

    root = zarr.open_group(destination, mode="r")
    ids = np.asarray(root["RNA/featureData/ids"][:]).astype(str).tolist()
    assert ids == ["A", "B", "C"]
    assert np.asarray(root["RNA/featureData/names"][:]).astype(str).tolist() == ids
    cells = np.asarray(root["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["RNA/counts"][:])
    # The two left features named B are summed into one merged feature.
    assert {cell: row.tolist() for cell, row in zip(cells, counts, strict=True)} == {
        "left__c0": [1, 110, 0],
        "right__c0": [2, 0, 20],
    }


def test_dataset_merge_widens_narrow_counts_that_names_sum():
    feature_ids = ["a1", "a2", "b"]
    left = _MergeAssay(
        "RNA",
        np.array([[200, 100, 255], [255, 255, 1]], dtype=np.uint8),
        ["c0", "c1"],
        feature_ids,
        ["A", "A", "B"],
        block_size=2,
    )
    right = _MergeAssay(
        "RNA",
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ["c0", "c1"],
        ["a", "b"],
        ["A", "B"],
        block_size=2,
    )
    destination = MemoryStore()
    merger = DataStoreMerge(
        datasets=[
            _MergeDataStore([left], zarr_loc="memory://left"),
            _MergeDataStore([right], zarr_loc="memory://right"),
        ],
        zarr_path=destination,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        feature_key="names",
    )
    # Two uint8 features named A sum to at most 510, which uint16 holds.
    assert merger.plan().assays[0].dtype == "uint16"
    merger.dump()

    root = zarr.open_group(destination, mode="r")
    cells = np.asarray(root["cellData/ids"][:]).astype(str)
    counts = np.asarray(root["RNA/counts"][:])
    assert counts.dtype == np.uint16
    assert {cell: row.tolist() for cell, row in zip(cells, counts, strict=True)} == {
        "left__c0": [300, 255],
        "left__c1": [510, 1],
        "right__c0": [1, 2],
        "right__c1": [3, 4],
    }


def test_dataset_merge_rejects_unknown_feature_key():
    left, right = _id_scheme_sources()
    with pytest.raises(ValueError, match="feature_key must be one of"):
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=MemoryStore(),
            names=["left", "right"],
            feature_key="symbols",
        )


def test_dataset_merge_namespaces_duplicate_cell_ids_across_sources(merged_two_rna):
    ids = (
        np.asarray(
            zarr.open_group(str(merged_two_rna["default"]), mode="r")["cellData/ids"][:]
        )
        .astype(str)
        .tolist()
    )
    assert len(ids) == len(set(ids))
    assert {cell_id.split("__", maxsplit=1)[0] for cell_id in ids} == {
        "left",
        "right",
    }
    assert sorted(cell_id.split("__", maxsplit=1)[1] for cell_id in ids) == [
        "c0",
        "c0",
        "c1",
        "c1",
    ]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"names": ["left"]}, "same length"),
        ({"assays": ["ADT"]}, "Requested assays were not found"),
        ({"missing_assay_policy": "ignore"}, "missing_assay_policy must be one of"),
    ],
)
def test_dataset_merge_rejects_invalid_source_and_assay_matching(overrides, error):
    sources = _merge_two_rna(zarr_path=MemoryStore(), overwrite=False).datasets
    destination = MemoryStore()
    kwargs = {
        "datasets": sources,
        "zarr_path": destination,
        "names": ["left", "right"],
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=error):
        DataStoreMerge(**kwargs)
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(destination, mode="r")


def test_dataset_merge_rejects_feature_metadata_width_mismatch():
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    merger.datasets[1].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.array([[3], [4]], dtype=np.uint16),
        block_size=2,
    )

    with pytest.raises(ValueError, match="rawData has 1 columns"):
        merger.plan()
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(destination, mode="r")


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("cell_id", "cell IDs or their order differ"),
        ("feature_id", "featureData/ids"),
        ("feature_name", "featureData/names"),
    ],
)
def test_dataset_merge_resume_validates_current_source_identity(
    tmp_path, merged_two_rna, case, reason
):
    destination = _copy_merge(merged_two_rna, "default", tmp_path / "resume.zarr")
    candidate = _merge_two_rna(zarr_path=destination, overwrite=False)

    if case == "cell_id":
        candidate.datasets[0].cells._columns["ids"] = np.array(["x0", "c1"])
    elif case == "feature_id":
        for source in candidate.datasets:
            source.get_assay("RNA").feats._columns["ids"] = np.array(["id_x", "id_b"])
    else:
        candidate.datasets[0].get_assay("RNA").feats._columns["names"] = np.array(
            ["X", "B"]
        )

    plan = candidate.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert reason in plan.blockedReason


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing_column", "column 'names' is missing"),
        ("wrong_shape", "column 'names' has the wrong shape"),
        ("wrong_dtype", "column 'I' has dtype"),
        ("wrong_chunks", "column 'ids' has the wrong chunks"),
        ("wrong_role", "column 'RNA_I' has the wrong role"),
        ("wrong_assay", "column 'RNA_I' has the wrong assay"),
    ],
)
def test_dataset_merge_blocks_tampered_completed_cell_metadata(
    tmp_path, merged_two_rna, case, reason
):
    destination = _copy_merge(merged_two_rna, "default", tmp_path / "tampered.zarr")
    root = zarr.open_group(destination, mode="r+")
    cell_data = root["cellData"]

    if case == "missing_column":
        del cell_data["names"]
    elif case == "wrong_shape":
        values = np.asarray(cell_data["names"][:])[:3]
        del cell_data["names"]
        cell_data.create_array("names", data=values, chunks=(2,))
    elif case == "wrong_dtype":
        values = np.asarray(cell_data["I"][:], dtype=np.uint8)
        del cell_data["I"]
        cell_data.create_array("I", data=values, chunks=(2,))
    elif case == "wrong_chunks":
        values = np.asarray(cell_data["ids"][:])
        del cell_data["ids"]
        cell_data.create_array("ids", data=values, chunks=(1,))
    elif case == "wrong_role":
        cell_data["RNA_I"].attrs["role"] = "other"
    else:
        cell_data["RNA_I"].attrs["assay"] = "ADT"

    plan = _merge_two_rna(zarr_path=destination, overwrite=False).plan()
    assert plan.canDump is False
    assert plan.cellDataAction == "blocked"
    assert plan.blockedReason is not None
    assert reason in plan.blockedReason


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("mask_link", "has no missing mask"),
        ("mask_missing", "is missing"),
        ("mask_shape", "has the wrong shape"),
        ("mask_dtype", "has the wrong dtype"),
    ],
)
def test_dataset_merge_blocks_tampered_completed_missing_masks(
    tmp_path, merged_two_rna, case, reason
):
    destination = _copy_merge(merged_two_rna, "quality", tmp_path / "tampered.zarr")
    root = zarr.open_group(destination, mode="r+")
    cell_data = root["cellData"]
    missing_name = "__cytearc_missing__quality"

    if case == "mask_link":
        cell_data["quality"].attrs["missing_mask"] = "wrong"
    elif case == "mask_missing":
        del cell_data[missing_name]
    elif case == "mask_shape":
        values = np.asarray(cell_data[missing_name][:])[:3]
        del cell_data[missing_name]
        cell_data.create_array(missing_name, data=values, chunks=(2,))
    else:
        values = np.asarray(cell_data[missing_name][:], dtype=np.uint8)
        del cell_data[missing_name]
        cell_data.create_array(missing_name, data=values, chunks=(2,))

    plan = _with_partial_quality(
        _merge_two_rna(zarr_path=destination, overwrite=False)
    ).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert reason in plan.blockedReason


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("counts_missing", "counts array is missing"),
        ("feature_data_missing", "featureData is missing"),
        ("feature_ids_missing", "featureData/ids is missing"),
        ("selection_shape", "featureData/I has the wrong shape"),
        ("selection_dtype", "featureData/I has the wrong dtype"),
    ],
)
def test_dataset_merge_blocks_tampered_completed_assay_components(
    tmp_path, merged_two_rna, case, reason
):
    destination = _copy_merge(merged_two_rna, "default", tmp_path / "tampered.zarr")
    root = zarr.open_group(destination, mode="r+")

    if case == "counts_missing":
        del root["RNA/counts"]
    elif case == "feature_data_missing":
        del root["RNA/featureData"]
    elif case == "feature_ids_missing":
        del root["RNA/featureData/ids"]
    elif case == "selection_shape":
        del root["RNA/featureData/I"]
        root["RNA/featureData"].create_array(
            "I",
            data=np.ones(1, dtype=bool),
            chunks=(1,),
        )
    else:
        values = np.asarray(root["RNA/featureData/I"][:], dtype=np.uint8)
        del root["RNA/featureData/I"]
        root["RNA/featureData"].create_array("I", data=values, chunks=(2,))

    plan = _merge_two_rna(zarr_path=destination, overwrite=False).plan()
    assert plan.canDump is False
    assert plan.assays[0].countsAction == "blocked"
    assert plan.blockedReason is not None
    assert reason in plan.blockedReason


def _typed_source(label, values=((1, 2), (3, 4)), **assay_types):
    """Return a source whose assays, named by keyword, carry these types."""
    return _MergeDataStore(
        [
            _MergeAssay(
                name,
                [list(row) for row in values],
                ["c0", "c1"],
                [f"{name}0", f"{name}1"],
                [f"{name.upper()}0", f"{name.upper()}1"],
                block_size=2,
                assay_type=assay_type,
            )
            for name, assay_type in assay_types.items()
        ],
        zarr_loc=f"memory://{label}",
    )


def test_merge_records_the_declared_type_of_a_custom_named_assay() -> None:
    # GeneActivity and HTO keep their declarations instead of the preset of
    # their assay class, RNA and ADT.
    types = {"tags": "RNA", "genes": "GeneActivity", "hashtags": "HTO"}
    output = MemoryStore()
    merger = DataStoreMerge(
        [
            _typed_source("left", **types),
            _typed_source("right", ((5, 6), (7, 8)), **types),
        ],
        output,
        ["left", "right"],
        seed=0,
        nthreads=1,
    )

    plan = merger.plan()
    assert {assay.assayName: assay.assayType for assay in plan.assays} == types
    assert {assay.assayName: assay.writeCountsT for assay in plan.assays} == {
        "tags": True,
        "genes": True,
        "hashtags": False,
    }
    assert plan.manifest["assayTypes"] == types
    merger.dump()
    root = zarr.open_group(store=output, mode="r")
    assert root.attrs["assayTypes"] == types
    assert root["tags/countsT"].attrs["complete"] is True


@pytest.mark.parametrize(("first", "second"), [("RNA", "Assay"), ("ADT", "HTO")])
def test_merge_rejects_sources_that_declare_different_assay_types(first, second):
    destination = MemoryStore()
    merger = DataStoreMerge(
        [_typed_source("left", tags=first), _typed_source("right", tags=second)],
        destination,
        ["left", "right"],
        seed=0,
        nthreads=1,
    )
    message = (
        rf"Sources declare different types for assay 'tags': 'left' declares "
        rf"'{first}' and 'right' declares '{second}'.*zarr_mode='r\+' and "
        rf"assay_types=\{{'tags': "
    )

    with pytest.raises(ValueError, match=message):
        merger.plan()
    with pytest.raises(ValueError, match=message):
        merger.dump()
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(destination, mode="r")


@pytest.mark.parametrize("feature_ids", [["a", "a"], ["different_a", "different_b"]])
def test_merge_rejects_ambiguous_feature_identity(feature_ids):
    from cytearc.merge.features import align_features

    left = _MergeAssay("RNA", [[1, 2]], ["c"], ["a", "b"], ["A", "B"], 1)
    right = _MergeAssay("RNA", [[3, 4]], ["c"], feature_ids, ["A", "B"], 1)
    with pytest.raises(
        ValueError, match="Duplicate feature IDs|No overlapping features"
    ):
        align_features([left, right], ["left", "right"])


def test_dataset_merge_refuses_destination_holding_other_content():
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    store = MemoryStore()
    payload = default_buffer_prototype().buffer.from_bytes(b"keep")
    sync(store.set("notes/keep.txt", payload))
    merger = _merge_two_rna(zarr_path=store, overwrite=False)

    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "already holds content" in plan.blockedReason
    with pytest.raises(ValueError, match="already holds content"):
        merger.dump()
    assert set(store._store_dict) == {"notes/keep.txt"}


@pytest.mark.parametrize("destination", ["parent", "file_parent", "inside"])
def test_dataset_merge_refuses_destination_overlapping_a_source(tmp_path, destination):
    project = tmp_path / "project"
    source = project / "left"
    source.mkdir(parents=True)
    keep = source / "keep.txt"
    keep.write_text("keep")
    zarr_path = {
        "parent": str(project),
        "file_parent": f"file://{project}",
        "inside": str(source / "merged.zarr"),
    }[destination]
    merger = _merge_two_rna(zarr_path=zarr_path, overwrite=False)
    merger.datasets[0].zarr_loc = str(source)

    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "aliases source DataStore 'left'" in plan.blockedReason
    with pytest.raises(ValueError, match="aliases source"):
        merger.dump()
    assert keep.read_text() == "keep"
    assert sorted(path.name for path in source.iterdir()) == ["keep.txt"]


class _AttributedMeta(_MergeMeta):
    """Merge metadata whose chosen columns are stored arrays with attributes."""

    def __init__(self, attrs_by_column, **columns):
        super().__init__(block_rows=2, **columns)
        root = zarr.open_group(store=MemoryStore(), mode="w")
        self._arrays = {}
        for name, attrs in attrs_by_column.items():
            array = root.create_array(name, data=self._columns[name])
            array.attrs.update(attrs)
            self._arrays[name] = array

    def _get_array(self, key):
        return self._arrays.get(key, self._columns[key])


def test_dataset_merge_reconciles_cell_column_attributes(tmp_path):
    from cytearc.utils.logging import logger

    path = str(tmp_path / "attributes.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False)
    for dataset, sample, stage, colormap in zip(
        merger.datasets,
        ("A", "B"),
        ("early", "late"),
        ("viridis", "magma"),
        strict=True,
    ):
        dataset.cells = _AttributedMeta(
            {
                "sample": {"levels": [sample], "ordered": False},
                "stage": {"levels": [stage], "ordered": True},
                "score": {"display": {"colormap": colormap}, "unit": "count"},
            },
            ids=["c0", "c1"],
            names=["c0", "c1"],
            I=np.ones(2, dtype=bool),
            sample=[sample, sample],
            stage=[stage, stage],
            score=np.array([1.0, 2.0]),
        )
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        assert merger.plan().canDump is True
        merger.dump()
    finally:
        logger.remove(sink)

    cell_data = zarr.open_group(path, mode="r")["cellData"]
    assert dict(cell_data["sample"].attrs) == {
        "levels": ["A", "B"],
        "ordered": False,
    }
    # Ordered levels that differ keep the first source's order with the other
    # source's levels after it, an order that the sources do not agree on.
    assert dict(cell_data["stage"].attrs) == {
        "levels": ["early", "late"],
        "ordered": False,
    }
    assert "display" not in cell_data["score"].attrs
    assert cell_data["score"].attrs["unit"] == "count"
    dropped = [message for message in messages if "attributes that differ" in message]
    assert len(dropped) == 1
    assert "stage (ordered)" in dropped[0]
    assert "score (display)" in dropped[0]


# A display of T and B in the colors of a file, as an H5AD import stores it
# with the levels and order of the file's categorical.
_IMPORTED_CELL_TYPES = {
    "levels": ["T", "B"],
    "ordered": False,
    "display": {
        "kind": "categorical",
        "categories": [
            {"value": "T", "label": "T cells", "color": "#ff0000"},
            {"value": "B", "label": "B cells", "color": "#00aa00"},
        ],
        "missing_label": "unassigned",
        "missing_color": "#cccccc",
    },
}
# The first four colors that plots generate, scanpy's default palette.
_GENERATED = ["#1f77b4", "#ff7f0e", "#279e68", "#d62728"]


def _merged_cell_columns(tmp_path, sources, *, names=("first", "second")):
    """Merge stores of six cells, each given its cell columns, and return the merged cellData group and the merge's warnings.

    ``sources`` holds, for each store, its columns as values or as
    ``(values, attributes)``, whose attributes the column then records.
    """
    from cytearc.utils.logging import logger
    from tests.storage_helpers import by_cell_id, open_store, write_count_store

    stores = []
    for name, columns in zip(names, sources, strict=True):
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((6, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(6)],
        )
        store = open_store(location)
        for column, value in columns.items():
            values, attributes = value if isinstance(value, tuple) else (value, {})
            store.cells.insert(column, by_cell_id(store.cells, values))
            store.zw["cellData"][column].attrs.update(attributes)
        stores.append(store)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        DataStoreMerge(
            stores, str(tmp_path / "merged.zarr"), list(names), nthreads=1
        ).dump()
    finally:
        logger.remove(sink)
    return zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")["cellData"], [
        message for message in messages if "attributes that differ" in message
    ]


@pytest.mark.parametrize(
    "second",
    [
        np.asarray(["NK", "T", "", "DC", "NK", "B"], dtype=object),
        # The same labels in another order of rows.
        np.asarray(["DC", "", "B", "NK", "T", "DC"], dtype=object),
    ],
)
def test_a_merge_adds_the_labels_that_a_stored_display_leaves_out(tmp_path, second):
    first = np.asarray(["T", "B", "T", "B", "T", "B"], dtype=object)
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [{"cell_type": (first, _IMPORTED_CELL_TYPES)}, {"cell_type": second}],
    )
    attributes = dict(cell_data["orig_cell_type"].attrs)
    # The file's order and colors come first, and the labels that the second
    # store adds follow in natural order, whatever the order of its rows, in
    # the colors that plots generate for the third and fourth of four
    # categories; blank text holds no label.
    assert attributes["levels"] == ["T", "B", "DC", "NK"]
    assert attributes["ordered"] is False
    assert attributes["display"] == {
        "kind": "categorical",
        "categories": [
            {"value": "T", "label": "T cells", "color": "#ff0000"},
            {"value": "B", "label": "B cells", "color": "#00aa00"},
            {"value": "DC", "label": "DC", "color": _GENERATED[2]},
            {"value": "NK", "label": "NK", "color": _GENERATED[3]},
        ],
        "missing_label": "unassigned",
        "missing_color": "#cccccc",
    }
    assert dropped == []


def test_a_merge_adds_the_labels_of_a_column_of_variable_width_text(tmp_path):
    from tests.storage_helpers import by_cell_id, open_store, write_count_store

    # The second store's labels are the variable-width text that another tool
    # can store, one of them behind a NUL, which the merged column's
    # fixed-width text drops; the pass that measures their width collects
    # them as the merged column holds them.
    labels = {
        "first": np.asarray(["T", "B"] * 3, dtype=object),
        "second": np.asarray(
            ["NK", "T", "", "DC\x00", "NK", "B"], dtype=np.dtypes.StringDType()
        ),
    }
    stores = []
    for name, values in labels.items():
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((6, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(6)],
        )
        store = open_store(location)
        if name == "first":
            store.cells.insert("cell_type", by_cell_id(store.cells, values))
            store.zw["cellData"]["cell_type"].attrs.update(_IMPORTED_CELL_TYPES)
        else:
            store.zw["cellData"].create_array("cell_type", data=values, chunks=(6,))
            store = open_store(location)
        stores.append(store)

    DataStoreMerge(
        stores, str(tmp_path / "merged.zarr"), ["first", "second"], nthreads=1
    ).dump()

    merged = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")["cellData"]
    assert merged["orig_cell_type"].attrs["levels"] == ["T", "B", "DC", "NK"]
    by_cell = dict(
        zip(
            np.asarray(merged["ids"][:]).tolist(),
            np.asarray(merged["orig_cell_type"][:]).tolist(),
            strict=True,
        )
    )
    assert [by_cell[f"second__second{index}"] for index in range(6)] == [
        "NK",
        "T",
        "",
        "DC",
        "NK",
        "B",
    ]


def test_a_merge_charges_its_scan_of_a_column_of_variable_width_labels(tmp_path):
    import gc
    import tracemalloc

    from cytearc.merge import metadata as merge_metadata
    from cytearc.metadata.rows import array_row_selection_parts
    from tests.storage_helpers import open_store, write_count_store

    # 40,000 labels of 10 characters in one chunk, which another tool stored
    # as variable-width text, beside fixed-width cell IDs of 6 characters.
    n_cells = 40_000
    location = str(tmp_path / "labels.zarr")
    write_count_store(
        location,
        {"RNA": np.ones((n_cells, 2), dtype=np.uint8)},
        "uint8",
        cell_ids=[f"c{index:05d}" for index in range(n_cells)],
    )
    labels = np.asarray([f"type-{index % 40:05d}" for index in range(n_cells)])
    zarr.open_group(f"{location}/cellData", mode="r+").create_array(
        "cell_type", data=labels.astype(np.dtypes.StringDType()), chunks=(n_cells,)
    )
    table = open_store(location).cells
    fixed, per_row = array_row_selection_parts(table._get_array("cell_type"))

    def scan():
        merge_metadata._scan_text_column(
            table, "cell_type", block_rows=n_cells, labels=merge_metadata._Labels()
        )

    scan()
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    scan()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # The pass that measures the labels' width and collects them holds no
    # more than the scan charges the column.
    assert peak - before <= fixed + n_cells * per_row
    # A budget that one row of the labels' read cannot fit, which the IDs
    # alone fit, refuses the scan.
    ids_fixed, ids_per_row = array_row_selection_parts(table._get_array("ids"))
    assert ids_fixed + ids_per_row < fixed
    with pytest.raises(MemoryBudgetError, match="schema discovery cannot fit"):
        merge_metadata.resolve_metadata_schema_scan_rows(
            [table],
            ResourceBudget(fixed, 1),
            resident_bytes=0,
            preferred_rows=n_cells,
        )


def test_a_merge_adds_the_labels_beyond_the_categories_of_a_categorical_column(
    tmp_path,
):
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": pd.Categorical(
                    ["T", "B"] * 3, categories=["T", "B", "Plasma"]
                )
            },
            {"cell_type": np.asarray(["NK", "T"] * 3, dtype=object)},
        ],
    )
    # The categorical column's categories come first, unused Plasma included,
    # and NK, which the second store's rows hold, follows them.
    assert dict(cell_data["orig_cell_type"].attrs) == {
        "levels": ["T", "B", "Plasma", "NK"],
        "ordered": False,
    }
    assert dropped == []


def test_a_merge_takes_the_order_of_the_first_store_that_records_one(tmp_path):
    # The first store records no order; its labels follow the second's.
    first = np.asarray(["X", "B", "X", "B", "X", "B"], dtype=object)
    second = np.asarray(["T", "B", "T", "B", "T", "B"], dtype=object)
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [{"cell_type": first}, {"cell_type": (second, _IMPORTED_CELL_TYPES)}],
    )
    display = cell_data["orig_cell_type"].attrs["display"]
    assert [category["value"] for category in display["categories"]] == [
        "T",
        "B",
        "X",
    ]
    assert display["categories"][2]["color"] == _GENERATED[2]
    assert cell_data["orig_cell_type"].attrs["levels"] == ["T", "B", "X"]


def test_a_merge_keeps_the_color_of_the_first_display_that_lists_a_label(tmp_path):
    other = {
        "display": {
            "kind": "categorical",
            "categories": [
                {"value": "B", "label": "B", "color": "#0000ff"},
                {"value": "NK", "label": "Natural killer", "color": "#aa00aa"},
            ],
        }
    }
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
            {"cell_type": (np.asarray(["B", "NK"] * 3, dtype=object), other)},
        ],
    )
    # B keeps the first store's color, and NK its own, which a display of the
    # second store records; no color is generated.
    assert [
        (category["value"], category["label"], category["color"])
        for category in cell_data["orig_cell_type"].attrs["display"]["categories"]
    ] == [
        ("T", "T cells", "#ff0000"),
        ("B", "B cells", "#00aa00"),
        ("NK", "Natural killer", "#aa00aa"),
    ]
    # Only the first store records levels, which cover every label.
    assert cell_data["orig_cell_type"].attrs["levels"] == ["T", "B", "NK"]
    assert dropped == []


def _scanpy_colored(labels):
    """The levels and display that an import of a file colored by scanpy's default palette keeps."""
    return {
        "levels": list(labels),
        "ordered": False,
        "display": {
            "kind": "categorical",
            "categories": [
                {"value": value, "label": value, "color": _GENERATED[index]}
                for index, value in enumerate(labels)
            ],
        },
    }


def test_a_merge_gives_a_later_display_color_that_an_earlier_one_has_another(
    tmp_path,
):
    # Both files colored their own labels from the start of scanpy's palette,
    # the second spelling the colors in capitals.
    second = _scanpy_colored(["NK", "DC"])
    for category in second["display"]["categories"]:
        category["color"] = category["color"].upper()
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _scanpy_colored(["T", "B"]),
                )
            },
            {"cell_type": (np.asarray(["NK", "DC"] * 3, dtype=object), second)},
        ],
    )
    # DC and NK, whose colors T and B already have, take the colors of their
    # places, the third and fourth.
    assert [
        (category["value"], category["color"])
        for category in cell_data["orig_cell_type"].attrs["display"]["categories"]
    ] == [
        ("T", _GENERATED[0]),
        ("B", _GENERATED[1]),
        ("DC", _GENERATED[2]),
        ("NK", _GENERATED[3]),
    ]
    assert dropped == []


def test_a_merge_keeps_the_missing_label_of_the_first_display_that_records_one(
    tmp_path,
):
    other = {
        "display": {
            **_IMPORTED_CELL_TYPES["display"],
            "missing_label": "no call",
            "missing_color": "#000000",
        }
    }
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
            {"cell_type": (np.asarray(["T", "B"] * 3, dtype=object), other)},
        ],
    )
    display = cell_data["orig_cell_type"].attrs["display"]
    assert (display["missing_label"], display["missing_color"]) == (
        "unassigned",
        "#cccccc",
    )


def test_a_merge_reports_a_continuous_display_that_categories_replace(tmp_path):
    continuous = {
        "display": {
            "kind": "continuous",
            "colormap": "viridis",
            "minimum": None,
            "maximum": None,
            "scale": "linear",
        }
    }
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
            {"cell_type": (np.asarray(["T", "NK"] * 3, dtype=object), continuous)},
        ],
    )
    # The categorical display covers the column, and the continuous one is
    # named among the attributes that the merge dropped.
    assert cell_data["orig_cell_type"].attrs["display"]["kind"] == "categorical"
    assert len(dropped) == 1 and "orig_cell_type (display)" in dropped[0]


def test_a_merge_needs_no_seaborn_beyond_the_palettes(tmp_path, monkeypatch):
    import sys

    # A module set to None in sys.modules raises ImportError when imported,
    # as in an install without seaborn.
    monkeypatch.setitem(sys.modules, "seaborn", None)
    types = [f"type_{index:03d}" for index in range(110)]
    imported = {
        "levels": types,
        "ordered": False,
        "display": {
            "kind": "categorical",
            "categories": [
                {"value": value, "label": value, "color": "#123456"} for value in types
            ],
        },
    }
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {"cell_type": (np.asarray(types[:6], dtype=object), imported)},
            {"cell_type": np.asarray(["type_new"] * 6, dtype=object)},
        ],
    )
    categories = cell_data["orig_cell_type"].attrs["display"]["categories"]
    # The file's categories keep the colors that it gives them, and the 111th
    # of 111 categories takes the hue spaced by the golden ratio that an
    # import gives a categorical of that many, worked out once.
    assert len(categories) == 111
    assert {category["color"] for category in categories[:110]} == {"#123456"}
    assert categories[-1] == {
        "value": "type_new",
        "label": "type_new",
        "color": "#d9626d",
    }


def test_a_merge_keeps_a_display_without_levels_as_a_display(tmp_path):
    # A display that a store records without levels, as a column's display
    # can be set alone.
    display_only = {"display": _IMPORTED_CELL_TYPES["display"]}
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {"cell_type": (np.asarray(["T", "B"] * 3, dtype=object), display_only)},
            {"cell_type": np.asarray(["NK", "T"] * 3, dtype=object)},
        ],
    )
    attributes = dict(cell_data["orig_cell_type"].attrs)
    assert set(attributes) == {"display"}
    assert [category["value"] for category in attributes["display"]["categories"]] == [
        "T",
        "B",
        "NK",
    ]
    assert dropped == []


def test_a_merge_keeps_an_order_that_every_store_records(tmp_path):
    stages = ["late", "early", "mid", "late", "early", "mid"]
    ordered = pd.Categorical(stages, categories=["early", "mid", "late"], ordered=True)
    # Categorical columns record their categories as such, not as attributes.
    cell_data, dropped = _merged_cell_columns(
        tmp_path, [{"stage": ordered}, {"stage": ordered}]
    )
    assert dict(cell_data["orig_stage"].attrs) == {
        "levels": ["early", "mid", "late"],
        "ordered": True,
    }
    assert dropped == []


@pytest.mark.parametrize(
    "second",
    [
        # A store without the order, whose labels the order holds.
        np.asarray(["mid", "early"] * 3, dtype=object),
        # A store with the same order whose rows hold a label beyond it, as a
        # store merged by an earlier release may.
        (
            np.asarray(["mid", "post"] * 3, dtype=object),
            {"levels": ["early", "mid", "late"], "ordered": True},
        ),
    ],
)
def test_a_merge_drops_an_order_that_a_store_does_not_agree_on(tmp_path, second):
    stages = pd.Categorical(
        ["late", "early", "mid"] * 2, categories=["early", "mid", "late"], ordered=True
    )
    cell_data, dropped = _merged_cell_columns(
        tmp_path, [{"stage": stages}, {"stage": second}]
    )
    levels = cell_data["orig_stage"].attrs["levels"]
    assert levels[:3] == ["early", "mid", "late"]
    assert cell_data["orig_stage"].attrs["ordered"] is False
    assert len(dropped) == 1 and "orig_stage (ordered)" in dropped[0]


@pytest.mark.parametrize("kept", [1, 3, 4])
def test_a_merge_keeps_the_labels_alone_beyond_the_categories_a_column_keeps(
    tmp_path, monkeypatch, kept
):
    import cytearc.merge.metadata as merge_metadata

    monkeypatch.setattr(merge_metadata, "MAX_STORED_CATEGORIES", kept)
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
            # NK and DC make four categories; with one kept, the first store
            # alone holds too many labels.
            {"cell_type": np.asarray(["NK", "DC"] * 3, dtype=object)},
        ],
    )
    attributes = cell_data["orig_cell_type"].attrs
    if kept == 4:
        # A column keeps as many categories as it may.
        assert attributes["levels"] == ["T", "B", "DC", "NK"]
        assert dropped == []
        return
    for key in ("levels", "ordered", "display"):
        assert key not in attributes, key
    assert len(dropped) == 1
    assert "orig_cell_type (levels, ordered, display)" in dropped[0]


def test_merges_read_the_labels_of_the_rows_they_take(tmp_path, monkeypatch):
    import cytearc.merge.metadata as merge_metadata
    from tests.storage_helpers import (
        insert_nullable_cell_column,
        open_store,
        write_count_store,
    )

    write_count_store(
        str(tmp_path / "store.zarr"), {"RNA": np.ones((7, 3), dtype=int)}, "uint16"
    )
    store = open_store(tmp_path / "store.zarr")
    insert_nullable_cell_column(
        store,
        "cell_type",
        np.asarray(["NK", "T", "  ", "DC", "B", "NK", "T"]),
        np.asarray([False, False, False, False, True, False, False]),
    )
    insert_nullable_cell_column(
        store,
        "cluster",
        np.asarray([3, 1, 3, 2, 0, 2, 3]),
        np.asarray([False, False, False, False, True, False, False]),
    )

    def text_labels(column, taken=None):
        labels = merge_metadata._Labels()
        scan = merge_metadata._scan_text_column(
            store.cells, column, block_rows=2, labels=labels, taken=taken
        )
        return scan.width, None if labels.overflow else labels.found

    def cast_labels(column, dtype, taken=None):
        labels = merge_metadata._Labels()
        merge_metadata._add_column_labels(
            labels, store.cells, column, np.dtype(dtype), block_rows=2, taken=taken
        )
        return None if labels.overflow else labels.found

    # Read two rows at a time; blank text and the masked rows hold no label.
    assert text_labels("cell_type") == (2, {"NK", "T", "DC"})
    # Rows 1, 2 and 6 alone, as a composite of them takes them.
    assert text_labels("cell_type", np.asarray([1, 2, 6])) == (2, {"T"})
    # Numbers merged as text are compared as the text that the merge writes,
    # the masked placeholder 0 left out.
    assert text_labels("cluster") == (1, {"1", "2", "3"})
    # Numbers merged as numbers come as Python numbers of the merged type.
    floats = cast_labels("cluster", np.float64)
    assert floats == {1.0, 2.0, 3.0}
    assert {type(label) for label in floats} == {float}
    assert cast_labels("cluster", np.int64, np.asarray([1, 3, 4])) == {1, 2}
    # Values stored as Python objects are compared as the text that the merge
    # writes for them.
    objects = _MergeMeta(label=np.asarray(["T", " ", "Plasma", "T", 3], dtype=object))
    labels = merge_metadata._Labels()
    assert (
        merge_metadata._scan_text_column(
            objects, "label", block_rows=2, labels=labels
        ).width
        == 6
    )
    assert labels.found == {"T", "Plasma", "3"}
    # As many labels as a column keeps are listed, and one more is too many.
    monkeypatch.setattr(merge_metadata, "MAX_STORED_CATEGORIES", 3)
    assert cast_labels("cluster", np.int64) == {1, 2, 3}
    monkeypatch.setattr(merge_metadata, "MAX_STORED_CATEGORIES", 2)
    assert cast_labels("cluster", np.int64) is None
    assert text_labels("cell_type")[1] is None


def test_a_merge_reads_each_text_column_with_categories_once_to_plan_it(
    tmp_path, monkeypatch
):
    import cytearc.merge.metadata as merge_metadata

    scans: list[str] = []
    iterate = merge_metadata.iter_metadata_column_blocks

    def counted(table, column, **options):
        scans.append(column)
        return iterate(table, column, **options)

    monkeypatch.setattr(merge_metadata, "iter_metadata_column_blocks", counted)
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
            {"cell_type": np.asarray(["NK", "T"] * 3, dtype=object)},
        ],
    )
    # The pass that measures each store's widest text also collects its
    # labels, so the plan reads the column of each store once.
    assert scans.count("cell_type") == 2
    assert cell_data["orig_cell_type"].attrs["levels"] == ["T", "B", "NK"]


def test_a_plan_reads_the_labels_of_the_rows_it_takes_in_any_order(tmp_path):
    import cytearc.merge.metadata as merge_metadata
    from cytearc.merge.row_plan import build_row_plan
    from tests.storage_helpers import by_cell_id, open_store, write_count_store

    labels = {
        "first": np.asarray(["T", "B"] * 3, dtype=object),
        "second": np.asarray(
            ["T", "NK", "B", "Plasma", "T", "B", "DC", "T"], dtype=object
        ),
    }
    stores = []
    for name, values in labels.items():
        write_count_store(
            str(tmp_path / f"{name}.zarr"),
            {"RNA": np.ones((values.size, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(values.size)],
        )
        store = open_store(tmp_path / f"{name}.zarr")
        store.cells.insert("cell_type", by_cell_id(store.cells, values))
        stores.append(store)
    stores[0].zw["cellData"]["cell_type"].attrs.update(_IMPORTED_CELL_TYPES)
    # A composite of the second store's rows that hold T and B, given in a
    # shuffled order, which the plan takes in the store's order.
    row_plan = build_row_plan(
        [6, 8],
        ["first", "second"],
        selected_rows=[None, np.asarray([7, 0, 5, 2, 4])],
    )
    assert row_plan.source_rows(1).tolist() == [0, 2, 4, 5, 7]
    plan = merge_metadata.plan_cell_metadata(
        [store.cells for store in stores],
        ["first", "second"],
        prepend_text="orig",
        reset_cell_filter=False,
        source_column=None,
        scan_rows=2,
        n_cells=row_plan.nCells,
        row_plan=row_plan,
    )
    # Read two rows at a time, the labels of the rows taken are T and B alone.
    (column,) = [spec for spec in plan.columns if spec.name == "orig_cell_type"]
    assert column.attributes["levels"] == ["T", "B"]


def test_merges_read_the_missing_masks_at_the_rows_they_take_alone(
    tmp_path, monkeypatch
):
    from zarr.storage import LocalStore

    import cytearc.merge.metadata as merge_metadata
    from cytearc.metadata.table import MetaData
    from cytearc.storage.schema import create_cell_data

    root = zarr.open_group(str(tmp_path / "cells.zarr"), mode="w")
    ids = np.asarray([f"cell{index}" for index in range(8)])
    create_cell_data(root, None, ids=ids, names=ids)
    MetaData(root["cellData"]).insert(
        "cell_type", ["T", None, "B", "NK", "T", "B", None, "DC"]
    )
    MetaData(root["cellData"]).insert(
        "cluster", [1.0, None, 2.0, 3.0, 1.0, 2.0, None, 4.0]
    )
    # Every array of the table is stored in chunks of two rows.
    group = root["cellData"]
    for name, array in list(group.arrays()):
        data, attributes = array[:], dict(array.attrs)
        del group[name]
        group.create_array(name, data=data, chunks=(2,)).attrs.update(attributes)
    table = MetaData(root["cellData"])
    masks = {"__cytearc_missing__cell_type", "__cytearc_missing__cluster"}
    assert masks <= set(group.array_keys())
    read: list[str] = []
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if key.split("/")[1] in masks and "/c/" in key:
            read.append(key)
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    # A composite prefetches the objects of a missing mask that hold the rows
    # that it copies, here rows 2, 3 and 7 in the second and fourth chunks,
    # and its scans of labels read no other object of the mask.
    taken = np.asarray([2, 3, 7])
    labels = merge_metadata._Labels()
    merge_metadata._scan_text_column(
        table, "cell_type", block_rows=2, labels=labels, taken=taken
    )
    assert labels.found == {"B", "NK", "DC"}
    assert sorted(read) == [
        "cellData/__cytearc_missing__cell_type/c/1",
        "cellData/__cytearc_missing__cell_type/c/3",
    ]
    read.clear()
    labels = merge_metadata._Labels()
    merge_metadata._add_column_labels(
        labels, table, "cluster", np.dtype(np.float64), block_rows=2, taken=taken
    )
    assert labels.found == {2.0, 3.0, 4.0}
    assert sorted(read) == [
        "cellData/__cytearc_missing__cluster/c/1",
        "cellData/__cytearc_missing__cluster/c/3",
    ]


def _exported_merge(tmp_path):
    """Export the store that :func:`_merged_cell_columns` merged, read the file back with AnnData, and return it with the merged store."""
    import anndata

    import cytearc
    from tests.storage_helpers import open_store

    merged = open_store(tmp_path / "merged.zarr")
    cytearc.to_h5ad(merged.RNA, str(tmp_path / "export.h5ad"))
    return anndata.read_h5ad(tmp_path / "export.h5ad"), merged


def test_a_merge_of_text_and_integer_labels_exports_one_category_per_label(
    tmp_path,
):
    imported = {
        "levels": ["0", "1"],
        "ordered": False,
        "display": {
            "kind": "categorical",
            "categories": [
                {"value": "0", "label": "0", "color": "#ff0000"},
                {"value": "1", "label": "1", "color": "#00aa00"},
            ],
        },
    }
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {"leiden": (np.asarray(["0", "1"] * 3, dtype=object), imported)},
            {"leiden": np.asarray([0, 1, 2] * 2, dtype=np.int64)},
        ],
    )
    # The merged column stores the integers as text, so the integer 1 and the
    # text "1" are one label, and 2 follows the file's labels.
    assert cell_data["orig_leiden"].attrs["levels"] == ["0", "1", "2"]
    exported, merged = _exported_merge(tmp_path)
    column = exported.obs["orig_leiden"]
    assert column.cat.categories.tolist() == ["0", "1", "2"]
    assert exported.uns["orig_leiden_colors"].tolist() == [
        "#ff0000",
        "#00aa00",
        _GENERATED[2],
    ]
    # Each cell exports its stored label.
    stored = dict(
        zip(
            merged.cells.fetch_all("ids").tolist(),
            merged.cells.fetch_all("orig_leiden").tolist(),
            strict=True,
        )
    )
    assert column.astype(str).to_dict() == stored
    assert sorted(stored.values()) == ["0"] * 5 + ["1"] * 5 + ["2"] * 2
    assert dropped == []


def test_a_merge_of_text_and_date_labels_exports_the_labels_that_it_stores(
    tmp_path,
):
    days = np.asarray(["2026-01-01", "2026-01-02"] * 3, dtype="datetime64[ns]")
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "day": (
                    np.asarray(["2026-01-01"] * 6, dtype=object),
                    {"levels": ["2026-01-01"], "ordered": False},
                )
            },
            {"day": days},
        ],
    )
    # The merged column stores the dates as numpy writes them as text, which
    # differs from the file's text of the first day, and each text is a label.
    written = [str(day) for day in days[:2]]
    assert written == ["2026-01-01T00:00:00.000000000", "2026-01-02T00:00:00.000000000"]
    assert cell_data["orig_day"].attrs["levels"] == ["2026-01-01", *written]
    exported, merged = _exported_merge(tmp_path)
    column = exported.obs["orig_day"]
    assert column.cat.categories.tolist() == ["2026-01-01", *written]
    stored = dict(
        zip(
            merged.cells.fetch_all("ids").tolist(),
            merged.cells.fetch_all("orig_day").tolist(),
            strict=True,
        )
    )
    assert column.astype(str).to_dict() == stored
    assert (
        sorted(stored.values())
        == ["2026-01-01"] * 6 + written[:1] * 3 + written[1:] * 3
    )


@pytest.mark.parametrize(
    ("dtype", "labels"),
    [
        # A merged column of dates holds no levels.
        pytest.param(np.dtype("M8[ns]"), [], id="date"),
        pytest.param(np.dtype(np.float64), [float("inf")], id="infinite"),
        # A text category that a merged column of numbers cannot hold.
        pytest.param(np.dtype(np.int64), [], id="text in numbers"),
    ],
)
def test_merges_keep_the_labels_alone_of_categories_that_levels_cannot_hold(
    dtype, labels
):
    from cytearc.merge.metadata import _reconciled_cell_attributes

    root = zarr.open_group(store=MemoryStore(), mode="w")
    imported = root.create_array("imported", data=np.zeros(2))
    imported.attrs.update(
        _IMPORTED_CELL_TYPES
        if dtype.kind != "f"
        else {"levels": [1.5], "ordered": False}
    )
    other = root.create_array("other", data=np.zeros(2))
    found = _reconciled_cell_attributes([imported, other], labels, dtype)
    assert found[0] == {}
    assert set(found[1]) == set(imported.attrs) & {"levels", "ordered", "display"}


def test_merges_keep_the_labels_alone_of_a_categorical_of_too_many_categories(
    tmp_path, monkeypatch
):
    import cytearc.merge.metadata as merge_metadata

    monkeypatch.setattr(merge_metadata, "MAX_STORED_CATEGORIES", 2)
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
            # Three categories, of which the rows hold two the display lists.
            {"cell_type": pd.Categorical(["T", "B"] * 3, categories=["T", "B", "NK"])},
        ],
    )
    for key in ("levels", "ordered", "display"):
        assert key not in cell_data["orig_cell_type"].attrs, key
    assert "orig_cell_type (levels, ordered, display)" in dropped[0]


# A float column holds NaN, and a date column NaT, where a value is missing,
# as inserts and H5AD imports store them, without a missing mask.
_MISSING_NUMBERS = np.asarray([np.nan, 1.5, np.nan, 22.5, 1.5, np.nan])
_MISSING_DAYS = np.asarray(
    ["2026-01-02", "NaT", "2026-01-01", "NaT", "2026-01-02", "2026-01-02"],
    dtype="datetime64[ns]",
)


def _rows_of(cell_data, name):
    """Each merged cell's stored value of a column and whether its missing mask flags it, by cell ID."""
    ids = cell_data["ids"][:].tolist()
    mask = f"__cytearc_missing__{name}"
    missing = cell_data[mask][:].tolist() if mask in cell_data else [False] * len(ids)
    return dict(
        zip(ids, zip(cell_data[name][:].tolist(), missing, strict=True), strict=True)
    )


@pytest.mark.parametrize(
    "second", [_MISSING_NUMBERS, _MISSING_DAYS], ids=["float NaN", "NaT"]
)
def test_a_merge_stores_the_missing_numbers_and_times_of_a_source_as_missing_text(
    tmp_path, second
):
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                ),
                "note": np.asarray(["a", "b"] * 3, dtype=object),
            },
            {"cell_type": second, "note": second},
        ],
    )
    # pandas names the missing values, and NumPy writes the others as text.
    held = [None if pd.isna(value) else str(value) for value in second]
    assert held.count(None) in (2, 3)
    for name, first in (
        ("orig_cell_type", ["T", "B"] * 3),
        ("orig_note", ["a", "b"] * 3),
    ):
        expected = {
            f"first__first{row}": (label, False) for row, label in enumerate(first)
        }
        expected |= {
            f"second__second{row}": ("", True) if label is None else (label, False)
            for row, label in enumerate(held)
        }
        assert _rows_of(cell_data, name) == expected, name
    # The labels of the second store follow the file's in natural order, and
    # no missing value becomes one.
    added = sorted({label for label in held if label is not None})
    assert cell_data["orig_cell_type"].attrs["levels"] == ["T", "B", *added]
    assert [
        category["value"]
        for category in cell_data["orig_cell_type"].attrs["display"]["categories"]
    ] == ["T", "B", *added]
    assert dropped == []
    exported, _merged = _exported_merge(tmp_path)
    for name, first in (
        ("orig_cell_type", ["T", "B"] * 3),
        ("orig_note", ["a", "b"] * 3),
    ):
        column = exported.obs[name]
        written = {
            cell: None if pd.isna(value) else str(value)
            for cell, value in column.astype(object).items()
        }
        assert written == {
            **{f"first__first{row}": label for row, label in enumerate(first)},
            **{f"second__second{row}": label for row, label in enumerate(held)},
        }, name
    assert exported.obs["orig_cell_type"].cat.categories.tolist() == ["T", "B", *added]


def test_a_merge_of_variable_width_labels_with_missing_rows_and_floats_with_nan(
    tmp_path,
):
    from tests.storage_helpers import (
        by_cell_id,
        open_store,
        store_as_variable_width_text,
        write_count_store,
    )

    # The first store's labels are variable-width text, as another tool can
    # store them, whose missing mask flags rows 1 and 4; the second store's
    # are floats with NaN.
    columns = {
        "first": np.asarray(["T", None, "B", "NK", None, "B"], dtype=object),
        "second": np.asarray([np.nan, 2.5, np.nan, 2.5, 2.5, np.nan]),
    }
    stores = []
    for name, values in columns.items():
        location = tmp_path / f"{name}.zarr"
        write_count_store(
            str(location),
            {"RNA": np.ones((6, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(6)],
        )
        store = open_store(location)
        store.cells.insert("cell_type", by_cell_id(store.cells, values))
        if name == "first":
            store.zw["cellData"]["cell_type"].attrs.update(
                {"levels": ["T", "B"], "ordered": False}
            )
            store_as_variable_width_text(
                zarr.open_group(str(location / "cellData"), mode="r+"), "cell_type"
            )
        stores.append(open_store(location))
    first = zarr.open_group(str(tmp_path / "first.zarr"), mode="r")["cellData"]
    assert first["cell_type"].dtype.kind == "T"
    assert first["__cytearc_missing__cell_type"][:].tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    DataStoreMerge(
        stores, str(tmp_path / "merged.zarr"), list(columns), nthreads=1
    ).dump()
    cell_data = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")["cellData"]
    # Rows that either store's missing values or mask leave out are missing,
    # and only the present labels become categories.
    expected = {
        f"first__first{row}": ("", True) if label is None else (label, False)
        for row, label in enumerate(columns["first"])
    }
    expected |= {
        f"second__second{row}": ("", True) if np.isnan(value) else (str(value), False)
        for row, value in enumerate(columns["second"])
    }
    assert _rows_of(cell_data, "orig_cell_type") == expected
    assert cell_data["orig_cell_type"].attrs["levels"] == ["T", "B", "2.5", "NK"]


def test_a_merge_keeps_the_text_nan_of_a_store_as_a_label(tmp_path):
    # A store merged before this release holds the text "nan" where a float
    # source held NaN; text is never read as a missing number.
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {"cell_type": np.asarray(["T", "nan"] * 3)},
            {
                "cell_type": (
                    np.asarray(["T", "B"] * 3, dtype=object),
                    _IMPORTED_CELL_TYPES,
                )
            },
        ],
    )
    rows = _rows_of(cell_data, "orig_cell_type")
    assert {value for value, missing in rows.values() if not missing} == {
        "T",
        "B",
        "nan",
    }
    assert not any(missing for _value, missing in rows.values())
    assert cell_data["orig_cell_type"].attrs["levels"] == ["T", "B", "nan"]


def test_a_merge_plans_no_missing_mask_for_numbers_without_missing_values(tmp_path):
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {"note": np.asarray(["a", "b"] * 3, dtype=object)},
            {"note": np.asarray([1.5, 2.5] * 3)},
        ],
    )
    assert "__cytearc_missing__orig_note" not in cell_data
    assert sorted(set(cell_data["orig_note"][:].tolist())) == ["1.5", "2.5", "a", "b"]


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1.5, np.nan, -0.0, np.inf]),
        np.asarray([complex(np.nan, 1), 1 + 2j, complex(0, np.nan), 0j]),
        np.asarray(["2026-01-01", "NaT", "NaT", "2026-01-02"], dtype="datetime64[s]"),
        np.asarray([5, "NaT", 0, -3], dtype="timedelta64[s]"),
        np.asarray(["nan", "NaT", "", "T"]),
        # Variable-width text holds no NaN; its missing mask alone flags rows.
        np.asarray(["nan", "NaT", "", "T"], dtype=np.dtypes.StringDType()),
        np.asarray([3, 1, 0, 2]),
        np.asarray([True, False, True, True]),
    ],
    ids=[
        "floats",
        "complex",
        "dates",
        "times",
        "text",
        "variable-width text",
        "integers",
        "booleans",
    ],
)
@pytest.mark.parametrize("flagged", [None, [False, False, False, True]])
def test_a_merge_casts_missing_numbers_and_times_to_missing_text(values, flagged):
    from cytearc.merge.metadata import _as_merged_text

    mask = None if flagged is None else np.asarray(flagged)
    text, missing = _as_merged_text(values, np.dtype("U40"), mask)
    # pandas names the missing numbers and times; text is never missing.
    absent = [
        bool(pd.isna(value)) and values.dtype.kind not in "UT" for value in values
    ]
    expected_missing = [
        held or (flagged is not None and flagged[row])
        for row, held in enumerate(absent)
    ]
    if any(expected_missing):
        assert missing is not None and missing.tolist() == expected_missing
    else:
        assert missing is None
    # NumPy writes every other value as its own text.
    assert text.tolist() == [
        "" if held else str(value) for value, held in zip(values, absent, strict=True)
    ]
    assert text.dtype == np.dtype("U40")


def test_a_scan_of_numbers_as_text_leaves_their_missing_values_out():
    from cytearc.merge.metadata import _Labels, _scan_text_column, _TextScan

    table = _MergeMeta(score=np.asarray([np.nan, 1.5, np.nan, 22.25, 1.5]))

    def scan(taken=None):
        labels = _Labels()
        found = _scan_text_column(
            table,
            "score",
            block_rows=2,
            labels=labels,
            taken=None if taken is None else np.asarray(taken),
        )
        return found.width, found.hasMissing, labels.found

    # The widest value is 22.25; "nan" counts toward neither width nor labels.
    assert scan() == (5, True, {"1.5", "22.25"})
    # Rows 1, 3 and 4 hold no missing value, row 2 one, read two at a time.
    assert scan([1, 3, 4]) == (5, False, {"1.5", "22.25"})
    assert scan([1, 2]) == (5, True, {"1.5"})
    # Without labels, the scan still finds the missing values of the rows.
    assert _scan_text_column(table, "score", block_rows=2).hasMissing
    assert not _scan_text_column(
        table, "score", block_rows=2, taken=np.asarray([3])
    ).hasMissing
    # Complex numbers write NaN as "(nan+0j)", wider than "0j".
    numbers = _MergeMeta(z=np.asarray([complex(np.nan, 0), 0j]))
    assert _scan_text_column(numbers, "z", block_rows=2) == _TextScan(2, True)
    # Text holds no missing number, whatever it reads.
    text = _MergeMeta(label=np.asarray(["nan", "NaT", "T"]))
    assert _scan_text_column(text, "label", block_rows=2) == _TextScan(3, False)


def test_a_merge_refuses_missing_values_that_its_plan_gave_no_mask():
    from cytearc.merge.metadata import MetadataColumnSpec, _iter_column_blocks
    from cytearc.merge.row_plan import build_row_plan

    table = _MergeMeta(note=np.asarray([1.5, np.nan]))
    spec = MetadataColumnSpec(name="note", dtype=np.dtype("U3"), hasMissing=False)
    blocks = _iter_column_blocks(
        spec,
        build_row_plan([2], ["only"]),
        [table],
        source_columns=[{"note": "note"}],
        reset_cell_filter=False,
        source_column=None,
        membership={},
        segment_rows=2,
        chunk_rows=2,
    )
    # A source that changed after its merge was planned names the remedy.
    with pytest.raises(
        ValueError,
        match="Source 'only' holds missing values in column 'note' that the "
        "merge's plan gave no missing mask: the source changed after the merge "
        "was planned. Plan the merge again",
    ):
        list(blocks)


@pytest.mark.parametrize(
    "dtype", [np.dtype("U1"), np.dtype(str)], ids=["U1", "flexible"]
)
def test_a_merge_casts_a_block_of_nat_alone_to_text_of_any_width(dtype):
    from cytearc.merge.metadata import _as_merged_text

    # NumPy refuses to write a date in one character, so only the present
    # rows are cast, and a block of NaT casts to empty text at any width.
    with pytest.raises(RuntimeError, match="too short"):
        np.full(3, np.datetime64("NaT", "ns")).astype("U1")
    text, missing = _as_merged_text(
        np.full(3, np.datetime64("NaT", "ns")), dtype, np.asarray([False, True, False])
    )
    assert text.tolist() == ["", "", ""]
    assert missing.tolist() == [True, True, True]
    if dtype.itemsize:
        assert text.dtype == dtype
    days = np.asarray(["2026-01-01", "NaT", "2026-01-02"], dtype="datetime64[D]")
    text, missing = _as_merged_text(days, np.dtype("U10"), None)
    assert text.tolist() == ["2026-01-01", "", "2026-01-02"]
    assert missing.tolist() == [False, True, False]


def test_a_merge_of_narrow_labels_and_dates_that_are_all_nat(tmp_path):
    # One-letter labels plan a column of one character, which the text of NaT
    # would not fit.
    nat = np.full(6, np.datetime64("NaT", "ns"))
    cell_data, dropped = _merged_cell_columns(
        tmp_path,
        [{"cell_type": np.asarray(["T", "B"] * 3, dtype=object)}, {"cell_type": nat}],
    )
    assert cell_data["orig_cell_type"].dtype == np.dtype("U1")
    expected = {
        f"first__first{row}": (label, False) for row, label in enumerate(["T", "B"] * 3)
    }
    expected |= {f"second__second{row}": ("", True) for row in range(6)}
    assert _rows_of(cell_data, "orig_cell_type") == expected
    assert dropped == []
    from tests.storage_helpers import open_store, write_count_store

    # Features annotated with one letter, and dates that are all NaT.
    for name, values in (
        ("first", np.asarray(["a", "b", "c"], dtype=object)),
        ("second", np.full(3, np.datetime64("NaT", "ns"))),
    ):
        location = str(tmp_path / "features" / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((2, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(2)],
        )
        open_store(location).RNA.feats.insert("biotype", values)
    DataStoreMerge(
        [
            open_store(tmp_path / "features" / f"{name}.zarr")
            for name in ("first", "second")
        ],
        str(tmp_path / "features" / "merged.zarr"),
        ["first", "second"],
        nthreads=1,
    ).dump()
    features = zarr.open_group(str(tmp_path / "features" / "merged.zarr"), mode="r")[
        "RNA/featureData"
    ]
    assert features["biotype"].dtype == np.dtype("U1")
    assert dict(
        zip(features["ids"][:].tolist(), features["biotype"][:].tolist(), strict=True)
    ) == {"RNA0": "a", "RNA1": "b", "RNA2": "c"}
    assert not any(features["__cytearc_missing__biotype"][:].tolist())


@pytest.mark.parametrize(
    ("first", "second", "merged"),
    [
        pytest.param(
            np.asarray(["protein_coding", None, "lncRNA"], dtype=object),
            np.asarray(["protein_coding", "lncRNA", "lncRNA"], dtype=object),
            ["protein_coding", "lncRNA", "lncRNA"],
            id="text",
        ),
        # A masked integer holds 0, which a merged float column would read as
        # a value beside the second store's 2.5.
        pytest.param(
            pd.array([1, None, 3], dtype="Int64"),
            np.asarray([1.0, 2.5, 3.0]),
            [1.0, 2.5, 3.0],
            id="integers beside floats",
        ),
        # Integers have no missing value of their own; the mask alone flags it.
        pytest.param(
            pd.array([1, None, 3], dtype="Int64"),
            np.asarray([1, 2, 3]),
            [1, 2, 3],
            id="integers",
        ),
    ],
)
def test_a_merge_keeps_feature_annotations_that_a_missing_mask_leaves_out(
    tmp_path, first, second, merged
):
    from tests.storage_helpers import open_store, write_count_store

    # The first store leaves RNA1 unannotated under its missing mask, and the
    # second annotates it.
    annotations = {"first": first, "second": second}
    for name, values in annotations.items():
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((2, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(2)],
        )
        open_store(location).RNA.feats.insert("annotation", values)
    stored = zarr.open_group(str(tmp_path / "first.zarr"), mode="r")["RNA/featureData"]
    assert stored["__cytearc_missing__annotation"][:].tolist() == [False, True, False]
    DataStoreMerge(
        [open_store(tmp_path / f"{name}.zarr") for name in annotations],
        str(tmp_path / "merged.zarr"),
        list(annotations),
        nthreads=1,
    ).dump()
    features = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")[
        "RNA/featureData"
    ]
    assert dict(
        zip(
            features["ids"][:].tolist(), features["annotation"][:].tolist(), strict=True
        )
    ) == dict(zip(["RNA0", "RNA1", "RNA2"], merged, strict=True))
    assert not any(features["__cytearc_missing__annotation"][:].tolist())


def test_a_scan_budget_charges_dates_that_another_source_holds_as_text(monkeypatch):
    import tracemalloc

    import cytearc.merge.metadata as merge_metadata
    from cytearc.storage.budget import ResourceBudget

    width = 4096
    days = np.arange(width).astype("datetime64[D]").astype("datetime64[ns]")
    days[::2] = np.datetime64("NaT", "ns")
    charged: list[int] = []

    def recorded(resources, *, nTasks, taskBytes, residentBytes=0, requested=None):
        charged.append(int(taskBytes(1)))
        return 1, 1

    monkeypatch.setattr(merge_metadata, "admitted_worker_split", recorded)

    def charge(**columns):
        cells = [f"c{index}" for index in range(width)]
        tables = [
            _MergeMeta(ids=cells, names=cells, note=np.asarray(["a"] * width)),
            _MergeMeta(ids=cells, names=cells, **columns),
        ]
        charged.clear()
        merge_metadata.resolve_metadata_schema_scan_rows(
            tables, ResourceBudget(1 << 34, 1), resident_bytes=0, preferred_rows=width
        )
        return charged[-1]

    # Dates that another source holds as text are scanned as text and charged,
    # unlike dates of a column of their own.
    shared, own = charge(note=days), charge(day=days)
    assert shared > own
    # The charge covers what the scan holds for a block of dates with NaT: the
    # block, and the text that the cast makes of it, measured as it runs.
    tracemalloc.start()
    merge_metadata._as_merged_text(days, np.dtype(str), None)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert shared >= days.nbytes + peak


def _feature_columns_merge(tmp_path, second_features, second_columns):
    """Merge a store whose features RNA0 to RNA2 carry text annotations with one of ``second_features`` and their ``second_columns``, and return the merged feature group, the merge's warnings and the export read back with AnnData."""
    import anndata

    import cytearc
    from cytearc.utils.logging import logger
    from tests.storage_helpers import open_store, write_count_store

    first_columns = {
        "biotype": np.asarray(
            ["protein_coding", "lncRNA", "protein_coding"], dtype=object
        ),
        "source": np.asarray(["ensembl", "ensembl", "refseq"], dtype=object),
    }
    stores = []
    for name, features, columns in (
        ("first", ["RNA0", "RNA1", "RNA2"], first_columns),
        ("second", second_features, second_columns),
    ):
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((4, len(features)), dtype=int)},
            "uint16",
            feature_ids={"RNA": features},
            cell_ids=[f"{name}{index}" for index in range(4)],
        )
        store = open_store(location)
        for column, values in columns.items():
            store.RNA.feats.insert(column, values)
        stores.append(store)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        DataStoreMerge(
            stores, str(tmp_path / "merged.zarr"), ["first", "second"], nthreads=1
        ).dump()
    finally:
        logger.remove(sink)
    merged = open_store(tmp_path / "merged.zarr")
    cytearc.to_h5ad(merged.RNA, str(tmp_path / "export.h5ad"))
    features = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")[
        "RNA/featureData"
    ]
    return features, messages, anndata.read_h5ad(tmp_path / "export.h5ad")


@pytest.mark.parametrize(
    "missing",
    [np.nan, np.datetime64("NaT", "ns")],
    ids=["float NaN", "NaT"],
)
def test_a_merge_keeps_feature_annotations_where_another_source_holds_missing_values(
    tmp_path, missing
):
    # The second store measures RNA1 to RNA3 and annotates none of RNA1 and
    # RNA2, holding the missing value of floats or dates there.
    present = 4.5 if isinstance(missing, float) else np.datetime64("2026-01-01", "ns")
    column = np.asarray([missing, missing, present])
    features, messages, exported = _feature_columns_merge(
        tmp_path, ["RNA1", "RNA2", "RNA3"], {"biotype": column, "source": column}
    )
    assert not [message for message in messages if "Feature columns" in message]
    ids = features["ids"][:].tolist()
    expected = {
        "biotype": ["protein_coding", "lncRNA", "protein_coding", str(present)],
        "source": ["ensembl", "ensembl", "refseq", str(present)],
    }
    for name, labels in expected.items():
        values = dict(zip(ids, features[name][:].tolist(), strict=True))
        flagged = dict(
            zip(ids, features[f"__cytearc_missing__{name}"][:].tolist(), strict=True)
        )
        assert values == dict(zip(["RNA0", "RNA1", "RNA2", "RNA3"], labels)), name
        assert not any(flagged.values()), name
        assert exported.var[name].astype(str).to_dict() == dict(
            zip(["RNA0", "RNA1", "RNA2", "RNA3"], labels)
        ), name


def test_a_merge_keeps_date_feature_annotations_where_a_source_holds_nat(tmp_path):
    from tests.storage_helpers import open_store, write_count_store

    # Both stores annotate RNA0 alike, and NaT stands for a missing date.
    added = {
        "first": np.asarray(
            ["2026-01-01", "NaT", "2026-01-03"], dtype="datetime64[ns]"
        ),
        "second": np.asarray(["2026-01-01", "NaT", "NaT"], dtype="datetime64[ns]"),
    }
    for name, values in added.items():
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((2, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(2)],
        )
        open_store(location).RNA.feats.insert("added", values)
    DataStoreMerge(
        [open_store(tmp_path / f"{name}.zarr") for name in added],
        str(tmp_path / "merged.zarr"),
        list(added),
        nthreads=1,
    ).dump()
    features = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")[
        "RNA/featureData"
    ]
    ids = features["ids"][:].tolist()
    days = np.datetime_as_string(np.asarray(features["added"][:]), unit="D")
    values = dict(zip(ids, days.tolist(), strict=True))
    flagged = dict(
        zip(ids, features["__cytearc_missing__added"][:].tolist(), strict=True)
    )
    # RNA1, which neither store dates, is missing, and RNA2 keeps the first
    # store's date beside the second's NaT.
    assert {feature: flagged[feature] for feature in ids} == {
        "RNA0": False,
        "RNA1": True,
        "RNA2": False,
    }
    assert {feature: values[feature] for feature in ("RNA0", "RNA2")} == {
        "RNA0": "2026-01-01",
        "RNA2": "2026-01-03",
    }


# Dates at seconds and at nanoseconds, and durations at seconds and at
# milliseconds, as inserts store them, with NaT where a value is missing.
_TIMES = {
    "first": {
        "day": np.asarray(
            [
                "2026-01-01T00:00:01",
                "NaT",
                "2026-03-01",
                "NaT",
                "2026-01-01T00:00:01",
                "1970-01-01",
            ],
            dtype="datetime64[s]",
        ),
        "wait": np.asarray([5, "NaT", 0, 86400, "NaT", -3], dtype="timedelta64[s]"),
    },
    "second": {
        "day": np.asarray(
            [
                "2026-01-02T00:00:00.000000001",
                "2026-01-03",
                "NaT",
                "2026-01-02T00:00:00.000000001",
                "1969-12-31T23:59:59.999999999",
                "2026-01-04",
            ],
            dtype="datetime64[ns]",
        ),
        "wait": np.asarray([7, 1500, "NaT", 0, 1, -2], dtype="timedelta64[ms]"),
    },
}


def _merged_stored_columns(tmp_path, sources, *, names=("first", "second")):
    """Merge stores of six cells whose cell columns another tool wrote as they are, at any unit, and return the merged cellData group.

    An insert stores dates and durations coarser than seconds in seconds, so
    each column is written to the store directly.
    """
    from tests.storage_helpers import open_store, write_count_store

    for name, columns in zip(names, sources, strict=True):
        location = tmp_path / f"{name}.zarr"
        write_count_store(
            str(location),
            {"RNA": np.ones((6, 3), dtype=int)},
            "uint16",
            cell_ids=[f"{name}{index}" for index in range(6)],
        )
        group = zarr.open_group(str(location / "cellData"), mode="r+")
        for column, values in columns.items():
            group.create_array(column, data=values, chunks=(6,))
    DataStoreMerge(
        [open_store(tmp_path / f"{name}.zarr") for name in names],
        str(tmp_path / "merged.zarr"),
        list(names),
        nthreads=1,
    ).dump()
    return zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")["cellData"]


def _expected_times(sources, column, dtype, ids_of):
    """Each cell's value of ``column`` as NumPy casts it to ``dtype`` and writes it as text, NaT included, by cell ID."""
    return {
        ids_of(name, row): str(value)
        for name, columns in sources.items()
        for row, value in enumerate(columns[column].astype(dtype))
    }


def test_a_merge_keeps_dates_and_durations_of_different_units_at_the_finer_unit(
    tmp_path,
):
    cell_data, dropped = _merged_cell_columns(
        tmp_path, [_TIMES["first"], _TIMES["second"]]
    )
    ids = cell_data["ids"][:].tolist()
    exported, _merged = _exported_merge(tmp_path)
    for name, column, dtype in (
        ("orig_day", "day", np.dtype("datetime64[ns]")),
        ("orig_wait", "wait", np.dtype("timedelta64[ms]")),
    ):
        # The finer unit of each column holds every value, and NaT stays
        # NaT, which readers take as missing without a mask.
        stored = np.asarray(cell_data[name][:])
        assert stored.dtype == dtype, name
        assert f"__cytearc_missing__{name}" not in cell_data, name
        expected = _expected_times(
            _TIMES, column, dtype, lambda source, row: f"{source}__{source}{row}"
        )
        assert {
            cell: str(value) for cell, value in zip(ids, stored, strict=True)
        } == expected, name
        # An export writes each value as the text that NumPy writes, and NaT
        # as missing, as it writes the dates of any store.
        assert {
            cell: None if pd.isna(value) else str(value)
            for cell, value in exported.obs[name].astype(object).items()
        } == {
            cell: None if value == "NaT" else value for cell, value in expected.items()
        }, name
    assert dropped == []
    merged = _merged.cells.to_pandas_dataframe(["orig_day"])["orig_day"]
    assert int(merged.isna().sum()) == 3


def test_a_merge_of_dates_of_one_unit_reads_no_column_of_them(tmp_path, monkeypatch):
    import cytearc.merge.metadata as merge_metadata

    scanned: list[str] = []
    scan = merge_metadata._scan_time_column

    def recorded(table, column, *args, **kwargs):
        scanned.append(column)
        return scan(table, column, *args, **kwargs)

    monkeypatch.setattr(merge_metadata, "_scan_time_column", recorded)
    cell_data, _dropped = _merged_cell_columns(
        tmp_path, [{"day": _TIMES["first"]["day"]}, {"day": _TIMES["first"]["day"]}]
    )
    # Dates of one unit keep it, and their NaT, without a mask or a scan.
    stored = np.asarray(cell_data["orig_day"][:])
    assert stored.dtype == np.dtype("datetime64[s]")
    assert np.isnat(stored).sum() == 4
    assert "__cytearc_missing__orig_day" not in cell_data
    assert scanned == []


def test_a_merge_plans_no_missing_mask_for_dates_without_nat(tmp_path):
    days = _TIMES["second"]["day"][[0, 1, 3, 4, 5, 0]]
    cell_data, _dropped = _merged_cell_columns(
        tmp_path, [{"day": _TIMES["first"]["day"][[0, 2, 4, 5, 0, 2]]}, {"day": days}]
    )
    assert np.asarray(cell_data["orig_day"][:]).dtype == np.dtype("datetime64[ns]")
    assert "__cytearc_missing__orig_day" not in cell_data


@pytest.mark.parametrize(
    ("first", "second", "kind", "holding"),
    [
        # 1500 and 2300 lie beyond the dates that nanoseconds hold, on either
        # side; NaT, which every unit holds, sits beside them.
        (
            np.asarray(
                ["NaT", "1500-01-01", "2026-01-01", "NaT", "2026-01-01", "2026-01-01"],
                dtype="datetime64[s]",
            ),
            np.asarray(["2026-01-01"] * 6, dtype="datetime64[ns]"),
            "dates, such as 1500-01-01T00:00:00",
            "datetime64[s]",
        ),
        (
            np.asarray(
                ["2026-01-01", "NaT", "2300-01-01", "2026-01-01", "NaT", "2026-01-01"],
                dtype="datetime64[s]",
            ),
            np.asarray(["2026-01-01"] * 6, dtype="datetime64[ns]"),
            "dates, such as 2300-01-01T00:00:00",
            "datetime64[s]",
        ),
        (
            np.asarray([1, "NaT", 10**10, 1, 1, 1], dtype="timedelta64[s]"),
            np.asarray([1] * 6, dtype="timedelta64[ns]"),
            "durations, such as 10000000000 seconds",
            "timedelta64[s]",
        ),
        # Months beside weeks promote to weeks, which start on Thursdays, so
        # July 2000, a Saturday, lies between them; days hold both.
        (
            np.asarray(["2000-06", "2000-07"] * 3, dtype="datetime64[M]"),
            np.asarray(["2000-06-01"] * 6, dtype="datetime64[W]"),
            "dates, such as 2000-07",
            "datetime64[D]",
        ),
    ],
    ids=["dates before", "dates after", "durations", "months beside weeks"],
)
def test_a_merge_refuses_values_that_the_finer_unit_cannot_hold(
    tmp_path, first, second, kind, holding
):
    import datetime

    assert np.result_type(first.dtype, second.dtype) == second.dtype
    if first.dtype == np.dtype("datetime64[M]"):
        # Python's calendar: weeks count from Thursday 1 January 1970.
        assert datetime.date(1970, 1, 1).weekday() == 3
        assert datetime.date(2000, 6, 1).weekday() == 3
        assert datetime.date(2000, 7, 1).weekday() == 5
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Column 'value' of source 'first' holds {kind}, that "
            f"{second.dtype}, the finest unit of the merged sources, cannot hold. "
            "Store the column in every source at one unit that holds each value, "
            f"such as {holding}, and merge again."
        ),
    ):
        _merged_stored_columns(tmp_path, [{"value": first}, {"value": second}])
    assert not (tmp_path / "merged.zarr").exists()


def test_a_merge_keeps_dates_within_one_unit_of_the_finer_units_limits(tmp_path):
    # NumPy documents nanoseconds from 1677-09-21T00:12:43.145224193 to
    # 2262-04-11T23:47:16.854775807, so these whole seconds lie within them.
    first = np.asarray(
        ["1677-09-21T00:12:44", "2262-04-11T23:47:16"] * 3, dtype="datetime64[s]"
    )
    cell_data, _dropped = _merged_cell_columns(
        tmp_path,
        [
            {"day": first},
            {"day": np.asarray(["2026-01-01"] * 6, dtype="datetime64[ns]")},
        ],
    )
    stored = np.asarray(cell_data["orig_day"][:])
    assert sorted({str(value) for value in stored}) == [
        "1677-09-21T00:12:44.000000000",
        "2026-01-01T00:00:00.000000000",
        "2262-04-11T23:47:16.000000000",
    ]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (np.dtype("timedelta64[M]"), np.dtype("timedelta64[s]")),
        (np.dtype("timedelta64[Y]"), np.dtype("timedelta64[D]")),
    ],
)
def test_a_merge_refuses_durations_in_months_or_years_beside_fixed_units(
    tmp_path, first, second
):
    # NumPy finds no unit for both, since a month or a year has no fixed
    # length.
    with pytest.raises(TypeError, match="incompatible nonlinear base time units"):
        np.result_type(first, second)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Column 'wait' of source 'first' holds durations in {first} and column "
            f"'wait' of source 'second' in {second}, which no unit holds both of"
        ),
    ) as refusal:
        _merged_stored_columns(
            tmp_path,
            [
                {"wait": np.arange(6).astype(first)},
                {"wait": np.arange(6).astype(second)},
            ],
        )
    assert f"such as {second} after converting" in str(refusal.value)
    assert not (tmp_path / "merged.zarr").exists()


def test_a_merge_of_durations_in_years_and_months_counts_months(tmp_path):
    cell_data = _merged_stored_columns(
        tmp_path,
        [
            {"wait": np.arange(6).astype("timedelta64[Y]")},
            {"wait": np.arange(6).astype("timedelta64[M]")},
        ],
    )
    stored = np.asarray(cell_data["orig_wait"][:])
    assert stored.dtype == np.dtype("timedelta64[M]")
    assert sorted(stored.astype(np.int64).tolist()) == sorted(
        [12 * years for years in range(6)] + list(range(6))
    )


def test_a_scan_of_dates_reads_every_block_of_the_rows_it_takes():
    from cytearc.merge.metadata import _scan_time_column

    # Read two rows at a time, the only date that nanoseconds cannot hold
    # sits in the third block.
    days = np.asarray(
        ["2026-01-01", "NaT", "2026-01-02", "2026-01-03", "1500-01-01", "2026-01-04"],
        dtype="datetime64[s]",
    )
    table = _MergeMeta(day=days)

    def scan(taken=None):
        _scan_time_column(
            table,
            "day",
            np.dtype("datetime64[ns]"),
            block_rows=2,
            taken=None if taken is None else np.asarray(taken),
            subject="Column 'day' of source 'first'",
        )

    with pytest.raises(ValueError, match="such as 1500-01-01T00:00:00"):
        scan()
    with pytest.raises(ValueError, match="such as 1500-01-01T00:00:00"):
        scan([1, 2, 4])
    # The rows that the merge takes hold none.
    scan([0, 1, 2, 3, 5])


def test_a_scan_budget_charges_dates_and_durations_that_a_merge_reads(monkeypatch):
    import tracemalloc

    import cytearc.merge.metadata as merge_metadata
    from cytearc.storage.budget import ResourceBudget

    width = 4096
    days = np.datetime64("2026-01-01", "s") + np.arange(width).astype("timedelta64[s]")
    days[::3] = np.datetime64("NaT", "s")
    charged: list[int] = []

    def recorded(resources, *, nTasks, taskBytes, residentBytes=0, requested=None):
        charged.append(int(taskBytes(1)))
        return 1, 1

    monkeypatch.setattr(merge_metadata, "admitted_worker_split", recorded)

    def charge(second_days):
        # One-letter IDs, whose scan holds less than the dates' scan.
        cells = [chr(ord("a") + index % 26) for index in range(width)]
        charged.clear()
        merge_metadata.resolve_metadata_schema_scan_rows(
            [
                _MergeMeta(ids=cells, names=cells, day=days),
                _MergeMeta(ids=cells, names=cells, day=second_days),
            ],
            ResourceBudget(1 << 34, 1),
            resident_bytes=0,
            preferred_rows=width,
        )
        return charged[-1]

    # Dates at two units are scanned and charged; dates at one are not.
    other_unit, one_unit = charge(days.astype("datetime64[ns]")), charge(days)
    assert other_unit > one_unit
    # The charge covers what the scan of a block of dates holds beside it,
    # measured as it runs, the block included.
    tracemalloc.start()
    merge_metadata._scan_time_column(
        _MergeMeta(day=days),
        "day",
        np.dtype("datetime64[ns]"),
        block_rows=width,
        taken=None,
        subject="Column 'day'",
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert other_unit >= days.nbytes + peak


def test_an_intersection_merge_checks_the_feature_dates_that_it_keeps(tmp_path):
    from tests.storage_helpers import open_store, write_count_store

    # RNA0, which only the first store measures, holds a date of 1500, which
    # nanoseconds cannot hold; a merge of the shared features leaves it out.
    columns = {
        "first": (
            ["RNA0", "RNA1", "RNA2"],
            np.asarray(["1500-01-01", "2026-01-01", "NaT"], dtype="datetime64[s]"),
        ),
        "second": (
            ["RNA1", "RNA2", "RNA3"],
            np.asarray(
                ["NaT", "2026-01-02T00:00:00.000000001", "2026-01-03"],
                dtype="datetime64[ns]",
            ),
        ),
    }
    for name, (features, values) in columns.items():
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((2, 3), dtype=int)},
            "uint16",
            feature_ids={"RNA": features},
            cell_ids=[f"{name}{index}" for index in range(2)],
        )
        open_store(location).RNA.feats.insert("added", values)
    DataStoreMerge(
        [open_store(tmp_path / f"{name}.zarr") for name in columns],
        str(tmp_path / "merged.zarr"),
        list(columns),
        nthreads=1,
        features="intersection",
    ).dump()
    features = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")[
        "RNA/featureData"
    ]
    stored = np.asarray(features["added"][:])
    assert stored.dtype == np.dtype("datetime64[ns]")
    assert dict(
        zip(features["ids"][:].tolist(), [str(value) for value in stored], strict=True)
    ) == {
        "RNA1": str(np.datetime64("2026-01-01", "s").astype("datetime64[ns]")),
        "RNA2": "2026-01-02T00:00:00.000000001",
    }


def _merged_feature_times(tmp_path, first, second):
    """Merge a store whose features RNA0 to RNA2 hold ``first`` with one whose features RNA1 to RNA3 hold ``second``, column by column, and return the merged feature group, the export read back with AnnData, and the features of each store."""
    import anndata

    import cytearc
    from tests.storage_helpers import open_store, write_count_store

    features = {"first": ["RNA0", "RNA1", "RNA2"], "second": ["RNA1", "RNA2", "RNA3"]}
    for name, columns in (("first", first), ("second", second)):
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"RNA": np.ones((2, 3), dtype=int)},
            "uint16",
            feature_ids={"RNA": features[name]},
            cell_ids=[f"{name}{index}" for index in range(2)],
        )
        store = open_store(location)
        for column, values in columns.items():
            store.RNA.feats.insert(column, values)
    DataStoreMerge(
        [open_store(tmp_path / f"{name}.zarr") for name in features],
        str(tmp_path / "merged.zarr"),
        list(features),
        nthreads=1,
    ).dump()
    merged = open_store(tmp_path / "merged.zarr")
    cytearc.to_h5ad(merged.RNA, str(tmp_path / "export.h5ad"))
    return (
        zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")["RNA/featureData"],
        anndata.read_h5ad(tmp_path / "export.h5ad"),
        features,
    )


def test_a_merge_keeps_feature_dates_and_durations_of_different_units(tmp_path):
    first = {
        "added": np.asarray(
            ["2026-01-01T00:00:01", "NaT", "NaT"], dtype="datetime64[s]"
        ),
        "lag": np.asarray([5, "NaT", 7], dtype="timedelta64[s]"),
    }
    second = {
        "added": np.asarray(
            ["NaT", "NaT", "2026-01-02T00:00:00.000000001"], dtype="datetime64[ns]"
        ),
        "lag": np.asarray(["NaT", 7000, 3], dtype="timedelta64[ms]"),
    }
    features, exported, ids = _merged_feature_times(tmp_path, first, second)
    merged_ids = features["ids"][:].tolist()
    for column, dtype in (
        ("added", np.dtype("datetime64[ns]")),
        ("lag", np.dtype("timedelta64[ms]")),
    ):
        stored = np.asarray(features[column][:])
        assert stored.dtype == dtype, column
        # Each feature takes the value of a store that holds one, cast by
        # NumPy; RNA1, which neither holds, is missing.
        expected: dict[str, str | None] = {feature: None for feature in merged_ids}
        for name, columns in (("first", first), ("second", second)):
            for feature, value in zip(
                ids[name], columns[column].astype(dtype), strict=True
            ):
                if not np.isnat(value):
                    expected[feature] = str(value)
        # Feature columns flag the features that no store annotates.
        missing = features[f"__cytearc_missing__{column}"][:]
        assert {
            feature: None if flagged else str(value)
            for feature, value, flagged in zip(merged_ids, stored, missing, strict=True)
        } == expected, column
        written = {
            feature: None if pd.isna(value) else str(value)
            for feature, value in exported.var[column].astype(object).items()
        }
        assert written == expected, column
    assert expected == {
        "RNA0": "5000 milliseconds",
        "RNA1": None,
        "RNA2": "7000 milliseconds",
        "RNA3": "3 milliseconds",
    }


@pytest.mark.parametrize(
    ("first", "second", "kind"),
    [
        (
            np.asarray(["2026-01-01", "1500-01-01", "NaT"], dtype="datetime64[s]"),
            np.asarray(["2026-01-01"] * 3, dtype="datetime64[ns]"),
            "dates, such as 1500-01-01T00:00:00",
        ),
        (
            np.asarray(["NaT", "2026-01-01", "2300-01-01"], dtype="datetime64[s]"),
            np.asarray(["2026-01-01"] * 3, dtype="datetime64[ns]"),
            "dates, such as 2300-01-01T00:00:00",
        ),
        (
            np.asarray([1, 10**10, "NaT"], dtype="timedelta64[s]"),
            np.asarray([1] * 3, dtype="timedelta64[ns]"),
            "durations, such as 10000000000 seconds",
        ),
    ],
    ids=["dates before", "dates after", "durations"],
)
def test_a_merge_refuses_feature_dates_that_the_finer_unit_cannot_hold(
    tmp_path, first, second, kind
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Feature column 'added' of source 'first' holds {kind}, that "
            f"{second.dtype}, the finest unit of the merged sources, cannot hold. "
            f"Store the column in every source at one unit that holds each value, "
            f"such as {first.dtype}, and merge again."
        ),
    ):
        _merged_feature_times(tmp_path, {"added": first}, {"added": second})
    assert not (tmp_path / "merged.zarr").exists()


def test_dataset_merge_overwrite_refuses_any_prepared_assay(tmp_path):
    path = str(tmp_path / "prepared.zarr")
    left, right = _two_assay_sources()

    def merger(assays, overwrite):
        return DataStoreMerge(
            datasets=[left, right],
            zarr_path=path,
            names=["left", "right"],
            prepend_text="",
            seed=0,
            assays=assays,
            overwrite=overwrite,
        )

    merger(["ADT"], False).dump()
    zarr.open_group(path, mode="r+")["ADT"].attrs["prepared"] = True

    plan = merger(["RNA"], True).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "['ADT'] are prepared" in plan.blockedReason
    assert "ADT" in zarr.open_group(path, mode="r")


def test_dataset_merge_overwrite_resets_default_assay_and_types(tmp_path):
    path = str(tmp_path / "reset_defaults.zarr")
    left, right = _two_assay_sources()

    def merger(assays, overwrite):
        return DataStoreMerge(
            datasets=[left, right],
            zarr_path=path,
            names=["left", "right"],
            prepend_text="",
            seed=0,
            assays=assays,
            overwrite=overwrite,
        )

    merger(None, False).dump()
    root = zarr.open_group(path, mode="r+")
    assert set(root.attrs["assayTypes"]) == {"RNA", "ADT"}
    root.attrs["defaultAssay"] = "ADT"

    merger(["RNA"], True).dump()
    root = zarr.open_group(path, mode="r")
    assert "ADT" not in root
    assert "defaultAssay" not in root.attrs
    assert set(root.attrs["assayTypes"]) == {"RNA"}


def test_dataset_merge_refuses_workspace_slot_of_a_legacy_assay(tmp_path):
    path = str(tmp_path / "legacy_slot.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_group("RNA").attrs["is_assay"] = True
    merger = _merge_two_rna(zarr_path=path, out_workspace="merged", overwrite=False)

    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "claimed by the legacy assay layout" in plan.blockedReason
    with pytest.raises(ValueError, match="legacy assay layout"):
        merger.dump()
    after = zarr.open_group(path, mode="r")
    assert "matrices" not in after
    assert "merged" not in after


def test_dataset_merge_resume_requires_the_same_source_counts(tmp_path):
    path = str(tmp_path / "source_counts.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False)
    fingerprints = merger.plan().manifest["sourceCountFingerprints"]["RNA"]
    assert len(fingerprints) == 2
    assert all(isinstance(value, str) and value for value in fingerprints)
    merger.dump()

    changed = _merge_two_rna(zarr_path=path, overwrite=False)
    changed.datasets[1].get_assay("RNA").rawData = ChunkedArray.from_numpy(
        np.array([[3, 31], [4, 40]]),
        block_size=2,
    )
    plan = changed.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "different configuration" in plan.blockedReason


def test_dataset_merge_rejects_names_containing_the_id_separator():
    with pytest.raises(ValueError, match="cannot contain '__'"):
        _merge_two_rna(zarr_path=MemoryStore(), names=["left", "right__b"])


def test_dataset_merge_drops_feature_annotations_that_disagree_within_a_source(
    tmp_path,
):
    from cytearc.utils.logging import logger

    path = str(tmp_path / "duplicate_names.zarr")
    left = _MergeAssay(
        "RNA",
        [[1, 2, 3], [4, 5, 6]],
        ["c0", "c1"],
        ["a1", "a2", "b"],
        ["A", "A", "B"],
        block_size=2,
    )
    left.feats = _MergeMeta(
        ids=["a1", "a2", "b"],
        names=["A", "A", "B"],
        chrom=["1", "2", "3"],
        biotype=["coding", "coding", "coding"],
    )
    right = _MergeAssay(
        "RNA",
        [[7, 8], [9, 10]],
        ["c0", "c1"],
        ["a", "b"],
        ["A", "B"],
        block_size=2,
    )
    right.feats = _MergeMeta(
        ids=["a", "b"],
        names=["A", "B"],
        chrom=["1", "3"],
        biotype=["coding", "coding"],
    )
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        DataStoreMerge(
            datasets=[
                _MergeDataStore([left], zarr_loc="memory://left"),
                _MergeDataStore([right], zarr_loc="memory://right"),
            ],
            zarr_path=path,
            names=["left", "right"],
            prepend_text="",
            seed=0,
            feature_key="names",
        ).dump()
    finally:
        logger.remove(sink)

    features = zarr.open_group(path, mode="r")["RNA/featureData"]
    assert "chrom" not in features
    np.testing.assert_array_equal(features["biotype"][:], ["coding", "coding"])
    assert any("were not merged: chrom" in message for message in messages)


# Between the need of one-row count shards and that of the default layout for
# the merge of _large_sources, and below the need of its cell metadata.
_FITTED_BUDGET = 16 * 1024**2
_METADATA_SHORTFALL = 128 * 1024


def _large_counts() -> dict[str, np.ndarray]:
    counts = {}
    for index, label in enumerate(("left", "right")):
        values = np.random.default_rng(index).poisson(0.3, size=(1_000, 400))
        # One count past uint16 keeps uint32 storage and wide count rows.
        values[0, 0] = 70_000
        counts[label] = values.astype(np.uint32)
    return counts


def _large_merged_counts(root) -> np.ndarray:
    """Return the source rows of _large_sources in the merged cell order."""
    sources = _large_counts()
    return np.vstack(
        [
            sources[label][int(cell[1:])]
            for label, _separator, cell in (
                str(value).partition("__") for value in root["cellData/ids"][:]
            )
        ]
    )


def _large_sources() -> list[_MergeDataStore]:
    sources = []
    for label, counts in _large_counts().items():
        sources.append(
            _MergeDataStore(
                [
                    _MergeAssay(
                        "RNA",
                        counts,
                        [f"c{i}" for i in range(1_000)],
                        [f"g{i}" for i in range(400)],
                        [f"G{i}" for i in range(400)],
                        block_size=1_000,
                    )
                ],
                zarr_loc=f"memory://{label}",
            )
        )
    return sources


def _large_merge(path, **options) -> DataStoreMerge:
    return DataStoreMerge(
        datasets=_large_sources(),
        zarr_path=str(path),
        names=["left", "right"],
        prepend_text="",
        seed=0,
        nthreads=1,
        **options,
    )


def _merged_policy(path) -> CountMatrixPolicy:
    counts = zarr.open_group(str(path), mode="r")["RNA/counts"]
    return policy_from_payload(load_count_matrix_plan(counts))


def test_dataset_merge_fits_the_count_layout_to_its_budget(tmp_path):
    from tests.storage_helpers import finalize_test_counts

    # The default layout does not fit, and it is refused while planning.
    with pytest.raises(MemoryError, match="default count-matrix policy"):
        _large_merge(
            tmp_path / "default.zarr",
            mem_budget=_FITTED_BUDGET,
            policy=DEFAULT_COUNT_MATRIX_POLICY,
        ).dump()
    assert not (tmp_path / "default.zarr").exists()

    plan = _large_merge(tmp_path / "fitted.zarr", mem_budget=_FITTED_BUDGET).plan()
    _large_merge(tmp_path / "fitted.zarr", mem_budget=_FITTED_BUDGET).dump()

    policy = _merged_policy(tmp_path / "fitted.zarr")
    assert policy.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes
    assert policy.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    fitted = zarr.open_group(str(tmp_path / "fitted.zarr"), mode="r")
    # The plan reports the geometry that the fitted layout writes.
    assert plan.assays[0].chunks == tuple(fitted["RNA/counts"].chunks)
    assert fitted["RNA/counts"].dtype == np.uint32
    assert fitted["RNA/countsT"].attrs["complete"] is True
    expected = _large_merged_counts(fitted)
    np.testing.assert_array_equal(fitted["RNA/counts"][:], expected)
    np.testing.assert_array_equal(fitted["RNA/countsT"][:], expected.T)
    # Identity does not depend on the layout that holds the counts.
    reference = zarr.open_group(store=MemoryStore(), mode="w")
    assert fitted["RNA/counts"].attrs["content_fingerprint"] == finalize_test_counts(
        reference.create_array("counts", data=expected)
    )


def test_dataset_merge_resume_keeps_the_layout_of_completed_counts(
    tmp_path, monkeypatch
):
    path = tmp_path / "resume.zarr"
    original = merge_datasets.write_assay_counts_t

    def interrupt(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(merge_datasets, "write_assay_counts_t", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _large_merge(path, mem_budget=_FITTED_BUDGET).dump()
    monkeypatch.setattr(merge_datasets, "write_assay_counts_t", original)
    fitted = _merged_policy(path)
    assert fitted != DEFAULT_COUNT_MATRIX_POLICY

    # The larger budget fits the default layout, but a resume keeps the
    # layout of the completed counts, so their validation does not block it.
    merger = _large_merge(path, mem_budget="1G")
    plan = merger.plan()
    assert plan.canDump
    assert plan.assays[0].countsAction == "skip"
    assert plan.assays[0].countsTAction == "resume"
    counts = zarr.open_group(str(path), mode="r")["RNA/counts"]
    assert plan.assays[0].chunks == tuple(counts.chunks)
    result = merger.dump()

    assert result.resumed is True
    assert _merged_policy(path) == fitted
    root = zarr.open_group(str(path), mode="r")
    assert root["RNA/countsT"].attrs["complete"] is True
    expected = _large_merged_counts(root)
    np.testing.assert_array_equal(root["RNA/counts"][:], expected)
    np.testing.assert_array_equal(root["RNA/countsT"][:], expected.T)


def test_dataset_merge_refits_the_layout_of_counts_it_rewrites(tmp_path, monkeypatch):
    path = tmp_path / "rewrite.zarr"
    original = merge_datasets.write_assay_counts

    def interrupt(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(merge_datasets, "write_assay_counts", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _large_merge(path, mem_budget=_FITTED_BUDGET).dump()
    monkeypatch.setattr(merge_datasets, "write_assay_counts", original)
    assert _merged_policy(path) != DEFAULT_COUNT_MATRIX_POLICY

    # Incomplete counts are rewritten, in the layout fitted to the new budget.
    result = _large_merge(path, mem_budget="1G").dump()
    actions = {component.name: component.action for component in result.components}
    assert actions["counts:RNA"] == "resume"
    assert _merged_policy(path) == DEFAULT_COUNT_MATRIX_POLICY


def test_dataset_merge_below_the_cell_metadata_need_fails_before_the_destination_exists(
    tmp_path,
):
    path = tmp_path / "tiny.zarr"
    # Planning admits the cell metadata and the count layout before it opens
    # the destination; here the cell metadata is the first that cannot fit.
    with pytest.raises(MemoryError, match="Merged cell metadata cannot fit one row"):
        _large_merge(path, mem_budget=_METADATA_SHORTFALL).dump()
    assert not path.exists()


def test_dataset_merge_below_one_row_shards_fails_before_the_destination_exists(
    tmp_path,
):
    sources = []
    for index, label in enumerate(("left", "right")):
        counts = np.random.default_rng(index).poisson(0.05, size=(100, 20_000))
        # One count past uint16 keeps uint32 storage and wide count rows.
        counts[0, 0] = 70_000
        sources.append(
            _MergeDataStore(
                [
                    _MergeAssay(
                        "RNA",
                        counts.astype(np.uint32),
                        [f"c{i}" for i in range(100)],
                        [f"g{i}" for i in range(20_000)],
                        [f"G{i}" for i in range(20_000)],
                        block_size=100,
                    )
                ],
                zarr_loc=f"memory://{label}",
            )
        )
    path = tmp_path / "wide.zarr"
    # Rows of 80 KB leave room for the cell metadata but not for one-row
    # count shards.
    with pytest.raises(MemoryError, match="count shards of one row"):
        DataStoreMerge(
            datasets=sources,
            zarr_path=str(path),
            names=["left", "right"],
            prepend_text="",
            seed=0,
            nthreads=1,
            mem_budget=int(3.4 * 1024**2),
        ).dump()
    assert not path.exists()


def test_dataset_merge_stores_an_assay_without_features_as_uint8():
    def source(label: str) -> _MergeDataStore:
        cells = [f"{label}0", f"{label}1"]
        return _MergeDataStore(
            [
                _MergeAssay(
                    "RNA",
                    np.array([[1, 2], [3, 4]], dtype=np.uint16),
                    cells,
                    ["g0", "g1"],
                    ["g0", "g1"],
                    block_size=2,
                ),
                _MergeAssay("ADT", np.zeros((2, 0), dtype=np.uint16), cells, [], [], 2),
            ],
            zarr_loc=f"memory://{label}",
        )

    destination = MemoryStore()
    merger = DataStoreMerge(
        datasets=[source("a"), source("b")],
        zarr_path=destination,
        names=["left", "right"],
        prepend_text="",
        seed=0,
        nthreads=1,
    )
    assert {plan.assayName: plan.dtype for plan in merger.plan().assays} == {
        "RNA": "uint16",
        "ADT": "uint8",
    }
    merger.dump()
    counts = zarr.open_group(store=destination, mode="r")["ADT/counts"]
    assert (counts.shape, counts.dtype) == ((4, 0), np.uint8)


@pytest.mark.parametrize(
    ("cells", "names", "options", "error", "message"),
    [
        (
            [2, 2],
            ["a"],
            {},
            ValueError,
            "Source names must match the number of sources",
        ),
        ([2, 2], ["a", "a"], {}, ValueError, "A unique name must be provided"),
        (
            [2, 2],
            ["a", "b"],
            {"seed": -1},
            ValueError,
            re.escape("seed must be from 0 to 2**64 - 1, not -1"),
        ),
        ([2, 2], ["a", "b"], {"seed": 2**64}, ValueError, "seed must be from 0 to 2"),
        ([2, 2], ["a", "b"], {"seed": True}, TypeError, "seed must be an integer"),
        ([2, 2], ["a", "b"], {"seed": 1.5}, TypeError, "seed must be an integer"),
    ],
)
def test_build_row_plan_rejects_inconsistent_sources(
    cells, names, options, error, message
):
    from cytearc.merge.row_plan import build_row_plan

    with pytest.raises(error, match=message):
        build_row_plan(cells, names, **options)


def _plan_coordinates(plan, window_rows: int) -> list[tuple[int, int]]:
    """Return the source and source row of each merged row, read through windows of ``window_rows`` rows."""
    from cytearc.merge.row_plan import iter_row_plan_windows

    coordinates = [(-1, -1)] * plan.nCells
    for window in iter_row_plan_windows(plan, window_rows):
        for part in window.parts:
            for position, row in zip(
                part.positions.tolist(), part.localRows.tolist(), strict=True
            ):
                coordinates[window.destStart + position] = (part.sourceIdx, row)
    return coordinates


@pytest.mark.parametrize("window_rows", [1, 3, 5, 100])
def test_a_row_plan_interleaves_the_sources_and_keeps_each_ones_order(window_rows):
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.cell_order import shuffled_order

    plan = build_row_plan([13, 11], ["a", "b"], seed=7)
    coordinates = _plan_coordinates(plan, window_rows)
    # The source of each merged row is the source of the concatenated cell
    # that the import shuffle puts there, and each source gives its cells in
    # its stored order from its first cell, 4 of a's and 10 of b's for this
    # seed, then from its first stored cell up to that one.
    owners = (shuffled_order(24, 7) >= 13).astype(int)
    assert [source for source, _ in coordinates] == owners.tolist()
    assert plan.firstCells == (4, 10)
    assert [row for source, row in coordinates if source == 0] == [
        *range(4, 13),
        *range(4),
    ]
    assert [row for source, row in coordinates if source == 1] == [10, *range(10)]
    assert plan.owners.dtype == np.uint8
    assert (plan.nCells, plan.sourceCells, plan.seed) == (24, (13, 11), 7)
    # Each cell's merged row, in the source's stored order.
    for source, n_cells in ((0, 13), (1, 11)):
        expected = [coordinates.index((source, row)) for row in range(n_cells)]
        assert plan.destination_rows(source).tolist() == expected


def test_a_row_plan_takes_selected_rows_in_store_order_whatever_their_order():
    from cytearc.merge.row_plan import build_row_plan

    sizes, names = [13, 11], ["a", "b"]
    ordinary = build_row_plan(sizes, names, seed=7)
    full = build_row_plan(
        sizes, names, seed=7, selected_rows=[np.arange(13)[::-1], None]
    )
    assert _plan_coordinates(ordinary, 4) == _plan_coordinates(full, 4)
    selected = [np.array([12, 7, 0, 4]), np.array([10, 0])]
    plan = build_row_plan(sizes, names, seed=7, selected_rows=selected)
    coordinates = _plan_coordinates(plan, 4)
    # Each source gives its selected rows in stored order, from the fourth
    # of a's and the first of b's for this seed.
    assert plan.firstCells == (3, 0)
    assert [row for source, row in coordinates if source == 0] == [12, 0, 4, 7]
    assert [row for source, row in coordinates if source == 1] == [0, 10]
    assert plan.source_rows(0, 1, 3).tolist() == [4, 7]
    assert plan.resident_bytes() == plan.owners.nbytes + 6 * 8
    # A source that gives no cells takes no turn, and a plan of none is empty.
    lone = build_row_plan(sizes, names, selected_rows=[np.array([], dtype=int), None])
    assert lone.owners.tolist() == [1] * 11
    empty = build_row_plan(sizes, names, selected_rows=[np.array([], dtype=int)] * 2)
    assert empty.nCells == 0
    assert _plan_coordinates(empty, 4) == []


@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ([np.array([0])], ValueError, "Selected rows must match"),
        ([np.array([-1]), None], IndexError, "out-of-range"),
        ([None, np.array([3])], IndexError, "out-of-range"),
        ([np.array([1, 1]), None], ValueError, "duplicate"),
        ([np.array([1.0]), None], TypeError, "integers"),
        ([np.array([[1]]), None], ValueError, "one-dimensional"),
    ],
)
def test_selected_row_plan_rejects_invalid_selections(selection, error, message):
    from cytearc.merge.row_plan import build_row_plan

    with pytest.raises(error, match=message):
        build_row_plan([5, 3], ["a", "b"], selected_rows=selection)


def test_a_row_plan_starts_each_source_at_a_uniformly_random_cell():
    from scipy import stats

    from cytearc.merge.row_plan import build_row_plan

    # Over 7,000 seeds, the first cells of sources of 7 and 10 cells take
    # each cell alike, and neither depends on the other or on the source of
    # the first merged row.
    plans = [build_row_plan([7, 10], ["a", "b"], seed=seed) for seed in range(7000)]
    firsts = np.asarray([plan.firstCells for plan in plans])
    turns = np.asarray([int(plan.owners[0]) for plan in plans])
    for source, cells in ((0, 7), (1, 10)):
        assert (
            stats.chisquare(np.bincount(firsts[:, source], minlength=cells)).pvalue
            > 1e-4
        )
        joint = np.zeros((cells, 2))
        np.add.at(joint, (firsts[:, source], turns), 1)
        assert stats.chi2_contingency(joint).pvalue > 1e-4
    both = np.zeros((7, 10))
    np.add.at(both, (firsts[:, 0], firsts[:, 1]), 1)
    assert stats.chi2_contingency(both).pvalue > 1e-4


def test_a_row_plan_gives_the_most_cells_of_one_source_in_a_window():
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.cell_order import shuffled_order

    plan = build_row_plan([13, 11], ["a", "b"], seed=7)
    owners = (shuffled_order(24, 7) >= 13).astype(int)
    for window in (1, 3, 5, 24, 100):
        expected = max(
            int(np.bincount(owners[start : start + window], minlength=2).max())
            for start in range(0, 24, window)
        )
        assert plan.most_source_cells(window) == expected
    assert build_row_plan([0, 0], ["a", "b"]).most_source_cells(4) == 0


def test_building_a_row_plan_holds_no_more_than_its_charge():
    import tracemalloc

    from cytearc.merge.row_plan import ROW_PLAN_BUILD_BYTES_PER_CELL, build_row_plan

    # Ordering 500,000 cells holds 24 bytes for each beside the plan: the
    # shuffle's hashes and order and each cell's source, at most; a source's
    # selected rows count their sorted copies too.
    sizes = [300_000, 200_000]
    needed = 500_000 * ROW_PLAN_BUILD_BYTES_PER_CELL
    build_row_plan(sizes, ["a", "b"], seed=1)
    tracemalloc.start()
    try:
        build_row_plan(sizes, ["a", "b"], seed=1, memory_bytes=needed)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert needed - 500_000 < peak <= needed + 4_096
    with pytest.raises(
        MemoryBudgetError,
        match="^Ordering 500000 merged cells holds about 12000000 bytes",
    ):
        build_row_plan(sizes, ["a", "b"], seed=1, memory_bytes=needed - 1)
    # A third of a's cells and every cell of b: 300,000 cells at 24 bytes
    # and a's 100,000 selected rows at 8.
    selected = [np.arange(0, 300_000, 3), None]
    with pytest.raises(
        MemoryBudgetError,
        match="^Ordering 300000 merged cells holds about 8000000 bytes",
    ):
        build_row_plan(
            sizes, ["a", "b"], seed=1, selected_rows=selected, memory_bytes=7_999_999
        )
    build_row_plan(
        sizes, ["a", "b"], seed=1, selected_rows=selected, memory_bytes=8_000_000
    )


def test_a_row_plan_of_one_seed_is_the_same_on_every_run_and_none_draws_another():
    from cytearc.merge.row_plan import build_row_plan

    plan = build_row_plan([13, 11], ["a", "b"])
    # The order of the default seed, as imports shuffle on every platform.
    assert plan.owners.tolist() == [
        *(1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        *(0, 0, 0, 1, 1, 1, 0, 0),
    ]
    assert build_row_plan([13, 11], ["a", "b"]).owners.tolist() == plan.owners.tolist()
    assert plan.firstCells == (7, 2)
    drawn = [build_row_plan([20, 20], ["a", "b"], seed=None) for _ in range(2)]
    # Two orders of 40 cells agree with odds of one in 138 billion.
    assert drawn[0].owners.tolist() != drawn[1].owners.tolist()
    assert drawn[0].seed != drawn[1].seed


@pytest.mark.parametrize(
    ("window_rows", "boundary_rows"), [(1, None), (4, None), (4, 6), (10, 3), (50, 7)]
)
def test_row_plan_windows_cover_every_cell_once_within_their_bounds(
    window_rows, boundary_rows
):
    from cytearc.merge.row_plan import build_row_plan, iter_row_plan_windows

    plan = build_row_plan([9, 0, 14], ["a", "b", "c"], seed=3)
    windows = list(
        iter_row_plan_windows(plan, window_rows, boundary_rows=boundary_rows)
    )
    assert [window.destStart for window in windows] == np.cumsum(
        [0] + [window.nRows for window in windows[:-1]]
    ).tolist()
    assert sum(window.nRows for window in windows) == 23
    for window in windows:
        assert 1 <= window.nRows <= window_rows
        if boundary_rows is not None:
            last = window.destStart + window.nRows - 1
            assert window.destStart // boundary_rows == last // boundary_rows
        # Each part names its rows of the window, which together are all of
        # them, and the source rows that fill them, in increasing order.
        positions = np.sort(np.concatenate([part.positions for part in window.parts]))
        assert positions.tolist() == list(range(window.nRows))
        for part in window.parts:
            assert part.sourceIdx != 1
            assert np.all(np.diff(part.positions) > 0)
            assert np.all(np.diff(part.localRows) == 1)
    with pytest.raises(ValueError, match="window_rows must be positive"):
        next(iter_row_plan_windows(plan, 0))
    with pytest.raises(ValueError, match="boundary_rows must be positive"):
        next(iter_row_plan_windows(plan, 2, boundary_rows=0))


def _two_source_arrangements(arrange, sizes=(60, 40), seeds=range(2000)) -> np.ndarray:
    """Return, for each seed, the source (0 or 1) of each merged row that ``arrange(seed)`` gives."""
    return np.asarray([arrange(seed) for seed in seeds], dtype=np.int64)


def _assert_random_arrangement(patterns: np.ndarray, sizes=(60, 40)) -> None:
    """Check rows of sources against the laws of a uniformly random arrangement of two sources' cells.

    ``patterns`` holds, for each seed, the source of each merged row. Under
    a uniform arrangement, the cells of the second source in a window of
    rows follow the hypergeometric law, each row comes from it with
    probability ``sizes[1] / sum(sizes)``, and the runs of one source follow
    the Wald-Wolfowitz mean and variance.
    """
    from scipy import stats

    n_first, n_second = sizes
    n_cells = n_first + n_second
    n_seeds = patterns.shape[0]
    for start in (10, n_cells - 20):
        counts = patterns[:, start : start + 20].sum(axis=1)
        law = stats.hypergeom(n_cells, n_second, 20)
        expected = law.pmf(np.arange(21)) * n_seeds
        observed = np.bincount(counts, minlength=21).astype(float)
        # Tails pool until each bin expects at least five windows.
        kept = np.flatnonzero(expected >= 5)
        low, high = int(kept[0]), int(kept[-1])
        pooled_observed = np.r_[
            observed[: low + 1].sum(), observed[low + 1 : high], observed[high:].sum()
        ]
        pooled_expected = np.r_[
            expected[: low + 1].sum(), expected[low + 1 : high], expected[high:].sum()
        ]
        pooled_expected *= pooled_observed.sum() / pooled_expected.sum()
        p_value = stats.chisquare(pooled_observed, pooled_expected).pvalue
        assert p_value > 1e-4, f"window at row {start}: p = {p_value}"
    share = n_second / n_cells
    z_rows = (patterns.mean(axis=0) - share) / np.sqrt(share * (1 - share) / n_seeds)
    assert np.abs(z_rows).max() < 4.5, f"row shares: z = {np.abs(z_rows).max()}"
    runs = 1 + (np.diff(patterns, axis=1) != 0).sum(axis=1)
    mean = 1 + 2 * n_first * n_second / n_cells
    variance = (
        2
        * n_first
        * n_second
        * (2 * n_first * n_second - n_cells)
        / (n_cells**2 * (n_cells - 1))
    )
    z_runs = (runs.mean() - mean) / np.sqrt(variance / n_seeds)
    assert abs(z_runs) < 4.5, f"runs: z = {z_runs}"


def test_a_merge_arranges_its_sources_cells_uniformly_at_random():
    from cytearc.merge.row_plan import build_row_plan

    def interleaved(seed):
        return build_row_plan([60, 40], ["a", "b"], seed=seed).owners.astype(np.int64)

    _assert_random_arrangement(_two_source_arrangements(interleaved))


@pytest.mark.parametrize(
    "arrangement",
    ["blocks of 16 cells", "one source after the other", "round-robin"],
)
def test_the_random_arrangement_check_rejects_orders_that_are_not(arrangement):
    def arrange(seed):
        if arrangement == "blocks of 16 cells":
            # Blocks of a shard of each source in a random order, with each
            # source's last block at the end, as merges were ordered until
            # this order replaced it.
            rng = np.random.default_rng(seed)
            blocks = [(0, 16)] * 3 + [(1, 16)] * 2
            order = [blocks[index] for index in rng.permutation(len(blocks))]
            order += [(0, 12), (1, 8)]
            return np.concatenate([np.full(rows, source) for source, rows in order])
        if arrangement == "one source after the other":
            return np.repeat([0, 1], [60, 40])
        owners = np.zeros(100, dtype=np.int64)
        owners[np.round(np.linspace(0, 99, 40)).astype(int)] = 1
        return owners

    with pytest.raises(AssertionError):
        _assert_random_arrangement(_two_source_arrangements(arrange))


# Sources shuffled with seeds of their own, with one seed for both, as
# default imports share shuffle_seed=0, or kept in file order.
@pytest.mark.parametrize("import_seeds", ["own", "shared", "file order"])
def test_a_merge_of_shuffled_sources_holds_each_group_as_a_random_sample_does(
    import_seeds,
):
    from scipy import stats

    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.cell_order import shuffled_order

    # Each file lists group x, its first fifth, before group y: rows 0 to 11
    # of source a's 60 cells and rows 0 to 7 of b's 40. Each trial imports
    # both and merges them with a seed of its own.
    trials = 3000
    positions, x_counts = [], []
    for trial in range(trials):
        seeds = {"own": (1000 + trial, 5000 + trial), "shared": (1000 + trial,) * 2}
        stored = [
            np.arange(n_cells)
            if import_seeds == "file order"
            else shuffled_order(n_cells, seeds[import_seeds][source])
            for source, n_cells in enumerate((60, 40))
        ]
        plan = build_row_plan([60, 40], ["a", "b"], seed=trial)
        merged = np.empty(100, dtype=np.int64)
        in_x = np.empty(100, dtype=bool)
        for source, (offset, x_rows) in enumerate(((0, 12), (60, 8))):
            rows = plan.destination_rows(source)
            merged[rows] = offset + stored[source]
            in_x[rows] = stored[source] < x_rows
        positions.append(int(np.flatnonzero(merged == 0)[0]))
        x_counts.append(int(np.count_nonzero(in_x[:20])))
    # A fixed cell, a's file row 0, lies anywhere in the merge alike, and the
    # first 20 merged cells hold group x as a random sample does: the
    # hypergeometric law of 20 of 100 cells, of mean 4 and variance 2.59.
    position_p = stats.chisquare(
        np.bincount(np.asarray(positions) // 10, minlength=10)
    ).pvalue
    law = stats.hypergeom(100, 20, 20)
    expected = law.pmf(np.arange(21)) * trials
    observed = np.bincount(x_counts, minlength=21).astype(float)
    kept = np.flatnonzero(expected >= 5)
    low, high = int(kept[0]), int(kept[-1])
    pooled_observed = np.r_[
        observed[: low + 1].sum(), observed[low + 1 : high], observed[high:].sum()
    ]
    pooled_expected = np.r_[
        expected[: low + 1].sum(), expected[low + 1 : high], expected[high:].sum()
    ]
    pooled_expected *= pooled_observed.sum() / pooled_expected.sum()
    window_p = stats.chisquare(pooled_observed, pooled_expected).pvalue
    variance = float(np.var(x_counts))
    if import_seeds == "file order":
        # Each source keeps its file order from a random first cell, so a
        # fixed cell lies anywhere, but a window holds runs of file rows.
        assert position_p > 1e-4 and window_p < 1e-10
    else:
        # A shared seed lists each file row at about the same place in both
        # sources; without their random first cells, the first rows of both
        # files would come together, at a variance of about 3.95.
        assert position_p > 1e-4 and window_p > 1e-4, (variance, law.var())
        assert abs(variance - law.var()) < 0.3


def test_row_plan_identity_checks_cover_every_cell_once():
    from cytearc.merge.row_plan import (
        build_row_plan,
        iter_merged_cell_ids,
        verify_merged_cell_ids,
    )

    plan = build_row_plan([5, 3], ["a", "b"], seed=0)
    tables = [
        _MergeMeta(ids=[f"x{row}" for row in range(5)]),
        _MergeMeta(ids=[f"y{row}" for row in range(3)]),
    ]
    with pytest.raises(ValueError, match="Source cell tables must match the row plan"):
        list(iter_merged_cell_ids(plan, tables[:1], dtype="U8"))
    stored = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "ids", shape=(8,), chunks=(3,), dtype="U8"
    )
    for start, ids in iter_merged_cell_ids(plan, tables, dtype="U8", block_rows=3):
        stored[start : start + ids.size] = ids
    # Each merged row holds the ID of the cell that the plan puts there.
    expected = [
        f"{'ab'[source]}__{'xy'[source]}{row}"
        for source, row in _plan_coordinates(plan, 8)
    ]
    assert stored[:].tolist() == expected
    verify_merged_cell_ids(stored, plan, tables, block_rows=2)
    shorter = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "ids", data=stored[:7]
    )
    with pytest.raises(
        ValueError, match="cell IDs or their order differ from this merge's"
    ):
        verify_merged_cell_ids(shorter, plan, tables)
    swapped = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "ids", data=np.asarray(expected[::-1]), chunks=(3,)
    )
    with pytest.raises(
        ValueError, match="cell IDs or their order differ from this merge's"
    ):
        verify_merged_cell_ids(swapped, plan, tables, block_rows=2)


def test_a_one_row_refusal_names_the_budget_that_admits_the_row():
    from cytearc.merge.metadata import _refuse_one_row

    def refuse(rows):
        raise MemoryBudgetError(
            f"{rows} rows need 900 bytes",
            needed_bytes=900,
            budget_bytes=500,
            item=f"{rows} rows",
        )

    with pytest.raises(MemoryBudgetError) as refused:
        _refuse_one_row(refuse, "Merged rows cannot fit", "one merged row")
    assert str(refused.value) == "Merged rows cannot fit. It needs at least 900 bytes."
    assert (refused.value.needed_bytes, refused.value.budget_bytes) == (900, 500)
    assert refused.value.item == "one merged row"
    assert str(refused.value.__cause__) == "1 rows need 900 bytes"
    # A budget that admits the row it refused while the widths were tried is
    # inconsistent, and the refusal says so rather than returning.
    with pytest.raises(
        RuntimeError, match="^one merged row was admitted after it was refused$"
    ):
        _refuse_one_row(lambda _rows: None, "Merged rows cannot fit", "one merged row")


def test_a_destination_overlaps_a_source_known_by_its_location_alone(tmp_path):
    from types import SimpleNamespace

    from cytearc.merge.datasets import _overlaps_source

    # A source without an open root group is compared by its location.
    source = SimpleNamespace(zarr_loc=str(tmp_path / "source.zarr"))
    assert _overlaps_source(str(tmp_path / "source.zarr" / "merged.zarr"), source)
    assert _overlaps_source(str(tmp_path), source)
    assert not _overlaps_source(str(tmp_path / "merged.zarr"), source)
    assert not _overlaps_source(MemoryStore(), source)


@pytest.mark.parametrize(
    "source_ids",
    [
        np.asarray(["AAAC-1", "G", "", "ünï-2"]),
        np.asarray(["AAAC-1", "G", "", "ünï-2"], dtype=np.dtypes.StringDType()),
        np.asarray([], dtype="U5"),
        # Other values format themselves one at a time.
        np.asarray(["AAAC-1", 7, b"raw", None], dtype=object),
    ],
    ids=["fixed", "variable", "empty", "objects"],
)
@pytest.mark.parametrize("dtype", ["U16", "U6", np.dtypes.StringDType()])
def test_prefixed_cell_ids_format_each_source_id_after_its_name(source_ids, dtype):
    from cytearc.merge.row_plan import prefixed_cell_ids

    expected = np.empty(source_ids.size, dtype=np.dtype(dtype))
    for index, value in enumerate(source_ids):
        expected[index] = f"src__{value}"

    merged = prefixed_cell_ids("src", source_ids, dtype)

    # A narrower dtype truncates every ID alike.
    assert merged.dtype == expected.dtype
    assert merged.tolist() == expected.tolist()


def test_a_merge_names_the_cells_of_a_source_whose_ids_are_variable_width_text(
    tmp_path,
):
    from cytearc import DataStore
    from tests.storage_helpers import store_as_variable_width_text, write_count_store

    # The right source's IDs differ in length and reach beyond Latin-1, as
    # another tool stored them; the merge writes every ID after its source's
    # name as fixed-width text as wide as the widest.
    ids = {
        "left": [f"cell{index}" for index in range(30)],
        "right": [
            f"c-{index}" + "x" * (index % 9) + ("\u00fc" if index % 4 == 0 else "")
            for index in range(20)
        ],
    }
    rng = np.random.default_rng(3)
    counts = {
        name: rng.integers(0, 5, size=(len(cells), 6)).astype(np.uint16)
        for name, cells in ids.items()
    }
    sources = []
    for name, cells in ids.items():
        location = tmp_path / f"{name}.zarr"
        write_count_store(
            str(location), {"RNA": counts[name]}, "uint16", cell_ids=cells
        )
        if name == "right":
            store_as_variable_width_text(
                zarr.open_group(str(location / "cellData"), mode="r+"), "ids"
            )
        sources.append(
            DataStore(
                str(location),
                default_assay="RNA",
                nthreads=1,
                min_features_per_cell=-1,
            )
        )

    DataStoreMerge(
        datasets=sources,
        zarr_path=str(tmp_path / "merged.zarr"),
        names=list(ids),
        prepend_text="",
        nthreads=1,
        overwrite=True,
    ).dump()

    merged = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    expected = np.asarray([f"{name}__{cell}" for name in ids for cell in ids[name]])
    assert merged["cellData/ids"].dtype == expected.dtype
    assert _rows_by_id(merged) == {
        f"{name}__{cell}": counts[name][row].tolist()
        for name in ids
        for row, cell in enumerate(ids[name])
    }


def test_validate_cell_metadata_reports_a_missing_or_incomplete_group():
    import cytearc.merge.metadata as merge_metadata
    from cytearc.merge.row_plan import build_row_plan

    tables = [
        _MergeMeta(ids=["c0", "c1"], names=["c0", "c1"], I=np.ones(2, dtype=bool))
        for _ in range(2)
    ]
    row_plan = build_row_plan([2, 2], ["left", "right"], seed=0)
    metadata_plan = merge_metadata.plan_cell_metadata(
        tables,
        ["left", "right"],
        prepend_text="",
        reset_cell_filter=True,
        source_column=None,
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")

    def reason(workspace):
        return merge_metadata.validate_cell_metadata(
            root,
            workspace,
            row_plan,
            tables,
            metadata_plan,
            resources=ResourceBudget(1024**2, 1),
            resident_bytes=0,
        )

    assert reason("ws") == "cell metadata group 'ws/cellData' is missing"
    root.create_group("ws/cellData")
    assert reason("ws") == "cell metadata group 'ws/cellData' is not complete"


def test_dataset_merge_without_prepend_text_keeps_source_column_names(tmp_path):
    path = str(tmp_path / "unprefixed.zarr")
    merger = _merge_two_rna(zarr_path=path, overwrite=False, prepend_text=None)
    for source, batch in zip(merger.datasets, ("x", "y"), strict=True):
        metadata = _MergeMeta(
            ids=["c0", "c1"],
            names=["c0", "c1"],
            I=np.ones(2, dtype=bool),
            batch=np.array([batch, batch]),
        )
        source.cells = metadata
        source.get_assay("RNA").cells = metadata

    merger.dump()
    root = zarr.open_group(path, mode="r")
    assert {name for name in root["cellData"].array_keys() if "batch" in name} == {
        "batch"
    }
    assert _rows_by_id(root, "cellData/batch") == {
        "left__c0": "x",
        "left__c1": "x",
        "right__c0": "y",
        "right__c1": "y",
    }


def test_dataset_merge_sizes_object_text_columns_by_their_longest_value():
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    for source, labels in zip(
        merger.datasets,
        (np.array(["a", "bcd"], dtype=object), np.array(["ef", "g"], dtype=object)),
        strict=True,
    ):
        metadata = _MergeMeta(
            ids=["c0", "c1"], names=["c0", "c1"], I=np.ones(2, dtype=bool), label=labels
        )
        source.cells = metadata
        source.get_assay("RNA").cells = metadata

    merger.dump()
    root = zarr.open_group(destination, mode="r")
    # Python strings of any length hold the labels; the longest has 3 letters.
    assert root["cellData/label"].dtype == np.dtype("U3")
    assert _rows_by_id(root, "cellData/label") == {
        "left__c0": "a",
        "left__c1": "bcd",
        "right__c0": "ef",
        "right__c1": "g",
    }


@pytest.mark.parametrize(
    ("dtype", "values", "exact"),
    [
        (np.dtype("S3"), [b"abc"], True),
        (np.dtype(bool), [True, False], True),
        (np.dtype(np.int16), [-(2**15), 2**15 - 1], True),
        (np.dtype(np.uint32), [0, 2**32 - 1], True),
        (
            np.dtype(np.float64),
            [-np.finfo(np.float64).max, np.finfo(np.float64).tiny],
            False,
        ),
        (np.dtype(np.complex128), [complex(-1e308, -1e-308)], False),
        (np.dtype("datetime64[ns]"), ["2262-04-11T23:47:16.854775807"], False),
        (np.dtype("timedelta64[ns]"), [-(2**63) + 1], False),
    ],
)
def test_schema_scan_text_bound_holds_the_text_of_every_value(dtype, values, exact):
    from cytearc.merge.metadata import _string_itemsize_bound

    # The schema scan converts values to text; the bound must hold the UTF-32
    # text NumPy makes of the extreme values of each kind.
    text_itemsize = np.asarray(values, dtype=dtype).astype(str).dtype.itemsize
    bound = _string_itemsize_bound(dtype)
    if exact:
        assert bound == text_itemsize
    else:
        assert bound >= text_itemsize
    # Object columns are sized by their values, not by their pointers.
    assert _string_itemsize_bound(np.dtype(object)) == np.dtype(object).itemsize


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        ([], np.dtype(bool)),
        ([np.dtype("timedelta64[s]"), np.dtype(np.int32)], np.dtype(np.float64)),
        # Variable-width text beside numbers is text, which the plan widens.
        (
            [np.dtypes.StringDType(), np.dtype(np.float64)],
            np.dtype("U1"),
        ),
        # Dates and durations of different units take the finest unit.
        (
            [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")],
            np.dtype("datetime64[ns]"),
        ),
        (
            [np.dtype("datetime64[D]"), np.dtype("datetime64[s]")],
            np.dtype("datetime64[s]"),
        ),
        (
            [np.dtype("timedelta64[s]"), np.dtype("timedelta64[ms]")],
            np.dtype("timedelta64[ms]"),
        ),
        # A date beside a duration, and the float column of a store that an
        # earlier release merged beside dates, stay floats.
        (
            [np.dtype("datetime64[s]"), np.dtype("timedelta64[s]")],
            np.dtype(np.float64),
        ),
        (
            [np.dtype(np.float64), np.dtype("datetime64[ns]")],
            np.dtype(np.float64),
        ),
    ],
    ids=[
        "no_sources",
        "time_and_number",
        "variable-width text and number",
        "dates at s and ns",
        "dates at D and s",
        "durations at s and ms",
        "date and duration",
        "float and date",
    ],
)
def test_metadata_dtype_promotion_fallbacks(dtypes, expected):
    from cytearc.merge.metadata import _promote_dtypes

    # A column no source holds is boolean, and kinds that are neither all
    # numeric nor text widen to float64.
    assert _promote_dtypes(dtypes) == expected


def test_dataset_merge_resume_needs_one_identity_row_within_its_budget(tmp_path):
    def merger(**options):
        sources = []
        for offset, label in enumerate(("left", "right")):
            sources.append(
                _MergeDataStore(
                    [
                        _MergeAssay(
                            "RNA",
                            (np.arange(20, dtype=np.uint16) + offset).reshape(20, 1),
                            [f"c{index}" for index in range(20)],
                            ["g"],
                            ["G"],
                            block_size=20,
                        )
                    ],
                    zarr_loc=f"memory://{label}",
                )
            )
        return DataStoreMerge(
            datasets=sources,
            zarr_path=str(tmp_path / "identity.zarr"),
            names=["left", "right"],
            prepend_text="",
            seed=0,
            nthreads=1,
            **options,
        )

    merger().dump()
    assert merger().plan().cellDataAction == "skip"
    # Validating the completed cell IDs holds one stored chunk of all 40 IDs,
    # which this budget cannot, though it holds the schema scan of the sources.
    with pytest.raises(
        MemoryError,
        match="Merged cell identity validation cannot fit one row within the "
        "operation memory budget",
    ):
        merger(mem_budget=4_000).plan()


def test_dataset_merge_keeps_complex_cell_metadata_values():
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    for source, values in zip(
        merger.datasets,
        (
            np.array([1.5, 2.5], dtype=np.float32),
            np.array([1 + 2j, 3 - 1j], dtype=np.complex64),
        ),
        strict=True,
    ):
        metadata = _MergeMeta(
            ids=["c0", "c1"], names=["c0", "c1"], I=np.ones(2, dtype=bool), score=values
        )
        source.cells = metadata
        source.get_assay("RNA").cells = metadata

    merger.dump()
    root = zarr.open_group(destination, mode="r")
    # Real and complex values widen to complex128, as real mixes widen to
    # float64, so every source value is kept exactly.
    assert root["cellData/score"].dtype == np.complex128
    assert _rows_by_id(root, "cellData/score") == {
        "left__c0": 1.5,
        "left__c1": 2.5,
        "right__c0": 1 + 2j,
        "right__c1": 3 - 1j,
    }


def test_dataset_merge_marks_cells_missing_from_a_complex_column():
    destination = MemoryStore()
    merger = _merge_two_rna(zarr_path=destination, overwrite=False)
    for source, columns in zip(
        merger.datasets,
        ({}, {"phase": np.array([1j, 2 + 0j], dtype=np.complex64)}),
        strict=True,
    ):
        metadata = _MergeMeta(
            ids=["c0", "c1"], names=["c0", "c1"], I=np.ones(2, dtype=bool), **columns
        )
        source.cells = metadata
        source.get_assay("RNA").cells = metadata

    merger.dump()
    root = zarr.open_group(destination, mode="r")
    phase = root["cellData/phase"]
    # Cells of the source without the column are filled and marked missing.
    assert phase.dtype == np.complex64
    values = _rows_by_id(root, "cellData/phase")
    assert (values["right__c0"], values["right__c1"]) == (1j, 2 + 0j)
    assert np.isnan(values["left__c0"]) and np.isnan(values["left__c1"])
    assert _rows_by_id(root, f"cellData/{phase.attrs['missing_mask']}") == {
        "left__c0": True,
        "left__c1": True,
        "right__c0": False,
        "right__c1": False,
    }


def _feature_annotated_sources(left_columns, right_columns, attrs=None):
    """Return two sources whose features carry extra annotation columns."""
    datasets = []
    for label, columns, feature_ids in (
        ("left", left_columns, ["a", "b"]),
        ("right", right_columns, ["b", "c"]),
    ):
        assay = _MergeAssay(
            "RNA",
            [[1, 2], [3, 4]],
            ["c0", "c1"],
            feature_ids,
            [name.upper() for name in feature_ids],
            block_size=2,
        )
        assay.feats = _AttributedMeta(
            {} if attrs is None else attrs[label],
            ids=feature_ids,
            names=[name.upper() for name in feature_ids],
            **columns,
        )
        datasets.append(_MergeDataStore([assay], zarr_loc=f"memory://{label}"))
    return datasets


def test_dataset_merge_keeps_float_feature_annotations_where_present(tmp_path):
    path = str(tmp_path / "float_features.zarr")
    DataStoreMerge(
        datasets=_feature_annotated_sources(
            {"gc": np.array([0.5, np.nan])}, {"gc": np.array([0.25, np.nan])}
        ),
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    ).dump()

    features = zarr.open_group(path, mode="r")["RNA/featureData"]
    np.testing.assert_array_equal(features["ids"][:], ["a", "b", "c"])
    # NaN marks a missing value: b takes the right value, and c has none.
    np.testing.assert_array_equal(features["gc"][:], [0.5, 0.25, np.nan])
    np.testing.assert_array_equal(
        features[features["gc"].attrs["missing_mask"]][:], [False, False, True]
    )


def test_dataset_merge_keeps_complex_feature_annotations_where_present(tmp_path):
    path = str(tmp_path / "complex_features.zarr")
    DataStoreMerge(
        datasets=_feature_annotated_sources(
            {"phase": np.array([0.5, np.nan])},
            {"phase": np.array([np.nan, 1 + 1j], dtype=np.complex128)},
        ),
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    ).dump()

    features = zarr.open_group(path, mode="r")["RNA/featureData"]
    # Real and complex values widen to complex128, and NaN marks a missing
    # value in either kind: b has no value, and c keeps its imaginary part.
    assert features["phase"].dtype == np.complex128
    phase = features["phase"][:]
    assert (phase[0], phase[2]) == (0.5 + 0j, 1 + 1j)
    assert np.isnan(phase[1])
    np.testing.assert_array_equal(
        features[features["phase"].attrs["missing_mask"]][:], [False, True, False]
    )


def test_dataset_merge_skips_feature_values_that_differ_only_in_imaginary_part(
    tmp_path,
):
    path = str(tmp_path / "complex_conflict.zarr")
    DataStoreMerge(
        datasets=_feature_annotated_sources(
            {"phase": np.array([0.5, 1.0])},
            {"phase": np.array([1 + 1j, 2 + 0j], dtype=np.complex128)},
        ),
        zarr_path=path,
        names=["left", "right"],
        prepend_text="",
        seed=0,
    ).dump()

    # Feature b is 1 in one source and 1+1j in the other, a conflict that
    # casting to float64 used to hide.
    assert "phase" not in zarr.open_group(path, mode="r")["RNA/featureData"]


def test_dataset_merge_drops_feature_annotations_whose_attributes_differ(tmp_path):
    from cytearc.utils.logging import logger

    path = str(tmp_path / "feature_attributes.zarr")
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        DataStoreMerge(
            datasets=_feature_annotated_sources(
                {"biotype": ["coding", "coding"], "chrom": ["1", "2"]},
                {"biotype": ["coding", "coding"], "chrom": ["2", "3"]},
                attrs={
                    "left": {"biotype": {"unit": "class"}, "chrom": {"unit": "name"}},
                    "right": {"biotype": {"unit": "label"}, "chrom": {"unit": "name"}},
                },
            ),
            zarr_path=path,
            names=["left", "right"],
            prepend_text="",
            seed=0,
        ).dump()
    finally:
        logger.remove(sink)

    features = zarr.open_group(path, mode="r")["RNA/featureData"]
    # The values agree for both columns, but biotype metadata differs.
    assert "biotype" not in features
    np.testing.assert_array_equal(features["chrom"][:], ["1", "2", "3"])
    assert features["chrom"].attrs["unit"] == "name"
    assert any("were not merged: biotype" in message for message in messages)


def test_dataset_merge_below_one_feature_metadata_row_fails_before_the_destination_exists(
    tmp_path,
):
    path = tmp_path / "wide_features.zarr"
    long_text = {"description": np.array(["x" * 5_000, "y" * 5_000])}
    merger = DataStoreMerge(
        datasets=_feature_annotated_sources(long_text, long_text),
        zarr_path=str(path),
        names=["left", "right"],
        prepend_text="",
        seed=0,
        nthreads=1,
        mem_budget=150_000,
    )
    # One 5,000-character value needs about 240 KB to merge, more than the
    # budget that admits the rest of the merge. Planning refuses it, so the
    # destination is never created.
    message = "^Merged feature column 'description' cannot fit the memory budget$"
    with pytest.raises(MemoryError, match=message):
        merger.plan()
    assert not path.exists()
    with pytest.raises(MemoryError, match=message):
        merger.dump()
    assert not path.exists()


def test_dataset_merge_plans_no_source_column_that_its_record_replaces(tmp_path):
    path = tmp_path / "merged.zarr"
    long_text = {"measured_in_right": np.array(["x" * 5_000, "y" * 5_000])}
    # The left source's column of 5,000-character values would need about
    # 240 KB to merge, more than the budget that admits the rest of the merge,
    # but the merge writes its record of the right source under that name, so
    # it plans and writes no such column.
    DataStoreMerge(
        datasets=_feature_annotated_sources(long_text, {}),
        zarr_path=str(path),
        names=["left", "right"],
        prepend_text="",
        seed=0,
        nthreads=1,
        mem_budget=150_000,
    ).dump()

    features = zarr.open_group(str(path), mode="r")["RNA/featureData"]
    np.testing.assert_array_equal(features["ids"][:], ["a", "b", "c"])
    assert features["measured_in_left"][:].tolist() == [True, True, False]
    assert features["measured_in_right"][:].tolist() == [False, True, True]
    assert dict(features["measured_in_right"].attrs) == {
        "role": "measured_features",
        "source": "right",
        "sourceColumn": "dataset",
    }


def test_a_merged_feature_column_names_the_budget_beside_the_feature_mappings(
    tmp_path,
):
    from cytearc.utils.memory_budget import MemoryBudgetError

    long_text = {"description": np.array(["x" * 5_000, "y" * 5_000])}
    sources = _feature_annotated_sources(long_text, long_text)
    # The column's one row of 5,000 characters is planned beside the mappings
    # of the features, which 150 kB cannot hold.
    with pytest.raises(MemoryBudgetError) as refused:
        DataStoreMerge(
            datasets=sources,
            zarr_path=str(tmp_path / "merge.zarr"),
            names=["left", "right"],
            prepend_text="",
            seed=0,
            nthreads=1,
            mem_budget=150_000,
        ).plan()
    assert refused.value.item == "one row of merged feature column 'description'"
    assert refused.value.needed_bytes > refused.value.budget_bytes == 150_000


_MEMBERSHIP_ATTRIBUTES = {"role": "assay_membership", "assay": "ADT"}
_PARTIAL_IDS = ["c0", "c1", "c2", "c3"]
_PARTIAL_ADT = [[0, 0], [5, 1], [0, 0], [2, 7]]


def _partial_adt_sources(
    membership=(False, True, False, True),
    *,
    attributes=_MEMBERSHIP_ATTRIBUTES,
    missing=None,
    adt=_PARTIAL_ADT,
    chunk_rows=None,
):
    """Return a left source whose ADT measures some cells and a right source.

    The left source's cell table is a Zarr-backed table, as a DataStore holds
    it, whose ``ADT_I`` column carries ``membership`` with ``attributes`` and,
    when ``missing`` is given, a linked missing mask. Its columns have chunks
    of ``chunk_rows`` rows when given. The right source measures every cell
    with both assays.
    """
    group = zarr.open_group(store=MemoryStore(), mode="w")
    columns = {
        "ids": np.asarray(_PARTIAL_IDS),
        "names": np.asarray(_PARTIAL_IDS),
        "I": np.ones(len(_PARTIAL_IDS), dtype=bool),
        "ADT_I": np.asarray(membership),
        "ADT_nCounts": np.asarray(adt).sum(axis=1).astype(np.float64),
        "ADT_nFeatures": (np.asarray(adt) > 0).sum(axis=1).astype(np.float64),
    }
    for name, values in columns.items():
        group.create_array(
            name, data=values, chunks="auto" if chunk_rows is None else (chunk_rows,)
        )
    group["ADT_I"].attrs.update(attributes)
    if missing is not None:
        group.create_array("__cytearc_missing__ADT_I", data=np.asarray(missing))
        group["ADT_I"].attrs["missing_mask"] = "__cytearc_missing__ADT_I"
    left = _MergeDataStore(
        [
            _MergeAssay(
                "RNA",
                [[1, 2], [3, 4], [5, 6], [7, 8]],
                _PARTIAL_IDS,
                ["g0", "g1"],
                ["G0", "G1"],
                block_size=2,
            ),
            _MergeAssay(
                "ADT", adt, _PARTIAL_IDS, ["a0", "a1"], ["A0", "A1"], block_size=2
            ),
        ],
        zarr_loc="memory://left",
    )
    left.cells = MetaData(group)
    right = _MergeDataStore(
        [
            _MergeAssay(
                "RNA", [[9, 9], [8, 8]], ["c0", "c1"], ["g0", "g1"], ["G0", "G1"], 2
            ),
            _MergeAssay(
                "ADT", [[3, 3], [4, 4]], ["c0", "c1"], ["a0", "a1"], ["A0", "A1"], 2
            ),
        ],
        zarr_loc="memory://right",
    )
    return left, right


def test_merge_keeps_each_source_cells_assay_membership():
    left, right = _partial_adt_sources()
    output = MemoryStore()
    merger = DataStoreMerge([left, right], output, ["left", "right"], seed=0)

    plan = merger.plan()
    assert plan.manifest["sourceMembership"]["RNA"] == ["all", "all"]
    left_state, right_state = plan.manifest["sourceMembership"]["ADT"]
    assert len(left_state) == 64 and right_state == "all"
    merger.dump()

    root = zarr.open_group(store=output, mode="r")
    assert _rows_by_id(root, "cellData/ADT_I") == {
        "left__c0": False,
        "left__c1": True,
        "left__c2": False,
        "left__c3": True,
        "right__c0": True,
        "right__c1": True,
    }
    assert set(_rows_by_id(root, "cellData/RNA_I").values()) == {True}
    assert root["cellData/ADT_I"].attrs.asdict() == _MEMBERSHIP_ATTRIBUTES
    # The source membership is not also merged as an ordinary column, which
    # the default prefix would have named orig_ADT_I.
    assert not {"orig_ADT_I", "orig_RNA_I"}.intersection(root["cellData"].array_keys())


def test_merge_strips_membership_attributes_from_ordinary_columns():
    # The left source lacks ADT, so its ADT_I is an ordinary column.
    left, right = _partial_adt_sources()
    left._assays.pop("ADT")
    left.assay_names.remove("ADT")
    output = MemoryStore()

    DataStoreMerge([left, right], output, ["left", "right"], seed=0).dump()

    root = zarr.open_group(store=output, mode="r")
    assert "role" not in root["cellData/orig_ADT_I"].attrs
    assert "assay" not in root["cellData/orig_ADT_I"].attrs
    assert _rows_by_id(root, "cellData/ADT_I") == {
        "left__c0": False,
        "left__c1": False,
        "left__c2": False,
        "left__c3": False,
        "right__c0": True,
        "right__c1": True,
    }


def test_merge_rejects_an_ordinary_column_named_like_a_merged_membership():
    left, right = _partial_adt_sources()
    left._assays.pop("ADT")
    left.assay_names.remove("ADT")
    destination = MemoryStore()
    merger = DataStoreMerge(
        [left, right], destination, ["left", "right"], prepend_text=None, seed=0
    )

    message = (
        "Cell column 'ADT_I' of source 'left' would be merged as 'ADT_I', which "
        "is the merged membership column of assay 'ADT'"
    )
    with pytest.raises(ValueError, match=message):
        merger.plan()
    with pytest.raises(ValueError, match=message):
        merger.dump()
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(destination, mode="r")


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        (
            {"attributes": {}},
            r"has attributes \{'role': None, 'assay': None\}.*cells\.drop\('ADT_I'\)",
        ),
        (
            {"missing": [False, False, True, False]},
            "has a missing-value mask.*Import the data again",
        ),
        (
            {"membership": np.asarray([0, 1, 0, 1], dtype=np.uint8)},
            "has dtype uint8.*Import the data again",
        ),
    ],
    ids=["plain", "masked", "not_bool"],
)
def test_merge_rejects_a_malformed_membership_column_before_writing(options, reason):
    left, right = _partial_adt_sources(**options)
    destination = MemoryStore()
    merger = DataStoreMerge([left, right], destination, ["left", "right"], seed=0)

    message = (
        f"Cell column 'ADT_I' is reserved for the membership of assay 'ADT'.*{reason}"
    )
    with pytest.raises(ValueError, match=message):
        merger.plan()
    with pytest.raises(ValueError, match=message):
        merger.dump()
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(destination, mode="r")


def test_merge_rejects_counts_in_cells_outside_the_assay():
    # c2 has ADT counts, but the left source marks it as not measured; counts
    # that cancel to a zero total are counts too. Two-row chunks put c2 in the
    # second block of the membership column.
    left, right = _partial_adt_sources(
        membership=(True, True, False, True),
        adt=[[0, 0], [5, 1], [1, -1], [2, 7]],
        chunk_rows=2,
    )
    destination = MemoryStore()
    merger = DataStoreMerge([left, right], destination, ["left", "right"], seed=0)

    message = (
        "Source 'left' has ADT counts in 1 of the cells that its column 'ADT_I' "
        "marks as not measured by assay 'ADT'"
    )
    with pytest.raises(ValueError, match=message):
        merger.plan()
    with pytest.raises(ValueError, match=message):
        merger.dump()
    with pytest.raises(GroupNotFoundError):
        zarr.open_group(destination, mode="r")


def _sparse_band_write_assay_counts(
    root,
    assay_name,
    workspace,
    assays,
    row_plan,
    alignment,
    *,
    resources,
    additionalResidentBytes=0,
    io=None,
):
    """Write counts by the simplest path: each batch reads its rows of each source alone, then COO bands."""
    from scipy.sparse import coo_matrix

    import cytearc.merge.writer as merge_writer
    from cytearc.merge.row_plan import iter_row_plan_windows
    from cytearc.storage.identity import CountSummary, finalize_counts
    from cytearc.storage.schema import load_count_array
    from cytearc.storage.sharding import (
        accumulate_sparse_to_shards,
        resolve_sparse_import_batch,
    )

    destination = load_count_array(root, assay_name, workspace)
    summary = CountSummary(destination)
    requirements = merge_writer._merge_import_requirements(
        assays,
        row_plan,
        alignment,
        destination.dtype,
        resources=resources,
        additionalResidentBytes=additionalResidentBytes,
    )
    plan = resolve_sparse_import_batch(
        (destination,),
        nRows=row_plan.nCells,
        resources=resources,
        maxWindowNnz=requirements.maxWindowNnz,
        sourceDtype=requirements.sourceDtype,
        residentBytes=requirements.residentBytes,
        extraProducerBytes=requirements.extraProducerBytes,
    )

    def batches():
        for window in iter_row_plan_windows(row_plan, plan.batchRows):
            rows, columns, data = [], [], []
            for part in window.parts:
                assay = assays[part.sourceIdx]
                if assay is None or int(assay.feats.N) == 0:
                    continue
                coordinates = merge_writer.remap_block_to_coo(
                    assay.rawData[np.asarray(part.localRows, dtype=np.int64), :],
                    alignment.featOrderMap[part.sourceIdx],
                    alignment.nFeats,
                    resources.workers,
                    np.dtype(destination.dtype),
                )
                rows.append(part.positions[coordinates.row])
                columns.append(coordinates.col)
                data.append(coordinates.data)
            yield (
                coo_matrix(
                    (
                        np.concatenate(data),
                        (np.concatenate(rows), np.concatenate(columns)),
                    ),
                    shape=(window.nRows, alignment.nFeats),
                )
                if data
                else coo_matrix((window.nRows, alignment.nFeats))
            )

    counter = accumulate_sparse_to_shards(
        destination,
        batches(),
        resources=resources,
        residentBytes=requirements.residentBytes,
        producerReserveBytes=plan.producerReserveBytes,
        io=io,
        countSummary=summary,
    )
    finalize_counts(destination, summary=summary)
    root[merge_writer._matrix_group_path(assay_name, workspace)].attrs["complete"] = (
        True
    )
    return counter


def _store_files(path) -> dict[str, bytes]:
    from pathlib import Path

    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


# Rows of each source's 16-row count chunks, which the row plan keeps together.
_SOURCE_BLOCK_ROWS = 16


def _count_sources(scenario: str) -> tuple[list, dict]:
    """Return two sources with chunked counts, and the merge options of a scenario."""
    rng = np.random.default_rng(len(scenario))
    n_left, n_right = 37, 29
    dtypes = {
        "float": (np.float32, np.float64),
        "names": (np.uint8, np.uint16),
    }.get(scenario, (np.uint16, np.uint16))
    left_counts = rng.integers(0, 9, size=(n_left, 7)).astype(dtypes[0])
    right_counts = rng.integers(0, 300, size=(n_right, 9)).astype(dtypes[1])
    left_counts[rng.random(left_counts.shape) < 0.5] = 0
    right_counts[rng.random(right_counts.shape) < 0.5] = 0
    if scenario == "float":
        # A negative zero merges as zero; fractions keep their bits.
        left_counts[0, :3] = [-0.0, 2.5, 0.5]
        right_counts[1, :2] = [-0.0, -1.25]
    left_ids = [f"id_{i}" for i in range(7)]
    right_ids = [f"id_{i}" for i in range(4, 13)]
    left_names = [f"G{i}" for i in range(7)]
    right_names = [f"G{i}" for i in range(4, 13)]
    options: dict = {"feature_key": "ids"}
    if scenario == "names":
        # Two right features share a name, so the merge sums them.
        right_names[2] = right_names[1]
        options = {"feature_key": "names"}
    assays = [
        _MergeAssay(
            "RNA",
            counts,
            [f"c{i}" for i in range(len(counts))],
            feature_ids,
            feature_names,
            block_size=_SOURCE_BLOCK_ROWS,
        )
        for counts, feature_ids, feature_names in (
            (left_counts, left_ids, left_names),
            (right_counts, right_ids, right_names),
        )
    ]
    sources = [
        _MergeDataStore([assay], zarr_loc=f"memory://{label}")
        for assay, label in zip(assays, ("left", "right"), strict=True)
    ]
    if scenario == "missing":
        # The right source lacks ADT, which the merge fills with zeros.
        adt = _MergeAssay(
            "ADT",
            rng.integers(0, 50, size=(n_left, 3)).astype(np.uint16),
            [f"c{i}" for i in range(n_left)],
            ["p0", "p1", "p2"],
            ["P0", "P1", "P2"],
            block_size=_SOURCE_BLOCK_ROWS,
        )
        sources[0] = _MergeDataStore([assays[0], adt], zarr_loc="memory://left")
    return sources, options


@pytest.fixture(scope="module")
def sparse_band_references():
    """Files of the stores that the earlier sparse band write merges, by case.

    A reference depends only on the scenario and the shards, so the cases that
    refuse other writes share it.
    """
    return {}


@pytest.mark.parametrize(
    ("scenario", "refused", "shard_rows", "dense", "held"),
    [
        ("ids", (), "narrow", True, "block"),
        ("ids", (), "wide", True, "block"),
        # A smaller budget holds windows of 8 rows, which divide the shards of
        # 16, or of 13 rows, which stop at each shard's end, or reads batch
        # by batch.
        ("ids", (), "narrow", True, "8 rows"),
        ("ids", (), "narrow", True, "13 rows"),
        ("ids", (), "narrow", True, "batch"),
        # Or holds a shard of each source as CSR.
        ("ids", (), "narrow", True, "CSR"),
        ("ids", (), "wide", True, "CSR"),
        ("ids", ("dense",), "narrow", False, "block"),
        ("ids", ("dense",), "narrow", False, "8 rows"),
        ("ids", ("dense",), "narrow", False, "batch"),
        ("ids", ("dense",), "narrow", False, "CSR"),
        ("names", (), "narrow", False, "block"),
        ("names", (), "wide", False, "block"),
        ("names", (), "narrow", False, "8 rows"),
        ("names", (), "narrow", False, "13 rows"),
        ("names", (), "narrow", False, "batch"),
        ("names", (), "narrow", False, "CSR"),
        ("float", (), "narrow", True, "block"),
        ("float", (), "wide", True, "block"),
        ("float", (), "narrow", True, "8 rows"),
        ("float", (), "narrow", True, "CSR"),
        ("missing", (), "narrow", True, "block"),
        ("missing", (), "narrow", True, "CSR"),
        # Sparse batches of cells of the source without antibodies alone
        # hold no coordinates.
        ("missing", ("dense",), "narrow", False, "block"),
        ("missing", ("dense",), "narrow", False, "CSR"),
    ],
)
def test_merged_counts_match_the_sparse_band_write_byte_for_byte(
    tmp_path,
    monkeypatch,
    sparse_band_references,
    scenario,
    refused,
    shard_rows,
    dense,
    held,
):
    import cytearc.merge.writer as merge_writer
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.cell_order import shuffled_order

    # Narrow destination shards cut each source shard into several batches;
    # wide ones take whole shards.
    policy = (
        CountMatrixPolicy(unitBytes=128, chunkBytes=64)
        if shard_rows == "narrow"
        else CountMatrixPolicy(unitBytes=1 << 20, chunkBytes=1 << 18)
    )

    def merge(path):
        sources, options = _count_sources(scenario)
        DataStoreMerge(
            datasets=sources,
            zarr_path=str(path),
            names=["left", "right"],
            prepend_text="",
            seed=3,
            overwrite=True,
            mem_budget=64 * 1024 * 1024,
            nthreads=1,
            policy=policy,
            **options,
        ).dump()
        return _store_files(path)

    reads: list[int] = []
    held_rows: list[tuple[int, int]] = []
    batches: list[int] = []
    calls = {"scatter_block_dense": 0, "remap_block_to_coo": 0}
    real = {
        name: getattr(merge_writer, name)
        for name in (
            "controlled_compute",
            "plan_dense_write",
            "scatter_block_dense",
            "remap_block_to_coo",
            "_held_windows",
            "iter_row_plan_windows",
        )
    }

    def read(array, nthreads):
        values = real["controlled_compute"](array, nthreads)
        if hasattr(array, "compute"):
            reads.append(int(values.shape[0]))
        return values

    def dense_write(*args, **kwargs):
        if "dense" in refused:
            raise MemoryBudgetError(
                "refused dense write", needed_bytes=2, budget_bytes=1, item="the write"
            )
        return real["plan_dense_write"](*args, **kwargs)

    def held_windows(shard_rows, part_rows, fits):
        # The budget admits no more held rows of each source than the case
        # names: a shard, 8 or 13 rows, or none, which reads batch by batch,
        # held dense, or a shard held as CSR and no dense rows.
        most = {"block": shard_rows, "8 rows": 8, "13 rows": 13, "CSR": shard_rows}
        rows, sparse = real["_held_windows"](
            shard_rows,
            part_rows,
            lambda rows, sparse: (
                sparse == (held == "CSR")
                and rows <= most.get(held, 0)
                and fits(rows, sparse)
            ),
        )
        held_rows.append((part_rows, rows, sparse))
        return rows, sparse

    def windows_of(row_plan, window_rows, **options):
        # The write's batches are its windows of merged rows.
        batches.append(window_rows)
        return real["iter_row_plan_windows"](row_plan, window_rows, **options)

    def counted(name):
        def call(*args, **kwargs):
            calls[name] += 1
            return real[name](*args, **kwargs)

        return call

    monkeypatch.setattr(merge_writer, "controlled_compute", read)
    monkeypatch.setattr(merge_writer, "plan_dense_write", dense_write)
    monkeypatch.setattr(merge_writer, "_held_windows", held_windows)
    monkeypatch.setattr(merge_writer, "iter_row_plan_windows", windows_of)
    for name in ("scatter_block_dense", "remap_block_to_coo"):
        monkeypatch.setattr(merge_writer, name, counted(name))
    actual = merge(tmp_path / "merged.zarr")
    monkeypatch.undo()
    if (scenario, shard_rows) not in sparse_band_references:
        monkeypatch.setattr(
            merge_datasets, "write_assay_counts", _sparse_band_write_assay_counts
        )
        sparse_band_references[scenario, shard_rows] = merge(
            tmp_path / "reference.zarr"
        )
    expected = sparse_band_references[scenario, shard_rows]

    # Every file of the stores is equal, counts chunks and summaries included.
    assert actual.keys() == expected.keys()
    assert [name for name in actual if actual[name] != expected[name]] == []
    assert any(name.startswith("RNA/counts/") for name in actual)
    assert (calls["scatter_block_dense"] > 0) == dense
    assert (calls["remap_block_to_coo"] > 0) == (not dense)
    # The RNA counts are written first; antibody bands hold every row. A
    # refused dense write searches its windows before the sparse write does.
    # A batch takes about 37 / 66 of its cells from the larger source.
    if scenario != "missing":
        part_rows, rows, sparse = held_rows[-1]
    else:
        part_rows, rows, sparse = held_rows[1 if "dense" in refused else 0]
    batch_rows = batches[0 if scenario == "missing" else -1]
    assert batch_rows < 8 if shard_rows == "narrow" else batch_rows >= 16
    assert part_rows == -(-batch_rows * 37 // 66)
    assert rows == {"block": 16, "8 rows": 8, "13 rows": 13, "CSR": 16}.get(held, 0)
    assert sparse == (held == "CSR")

    # Each source gives its rows from its first cell on, then from its first
    # row up to that one.
    first_cells = build_row_plan([37, 29], ["left", "right"], seed=3).firstCells

    def windows(n_rows, window, first):
        """The reads of a source's rows held ``window`` at a time within its shards of 16."""
        if window >= 16:
            # Whole shards, the first one's rows before the first cell kept
            # apart until the source gives them.
            return [min(16, n_rows - start) for start in range(0, n_rows, 16)]

        def reads(row, stop):
            sizes = []
            while row < stop:
                end = min(n_rows, row + window, (row // 16 + 1) * 16)
                sizes.append(end - row)
                row = end
            return sizes

        return reads(first, n_rows) + reads(0, first)

    # Each source's shards of 16 rows, 16, 16 and 5 rows of the left source
    # and of its antibodies and 16 and 13 of the right, are held whole or in
    # windows of 8 or 13 rows from the first cell, which stop at each
    # shard's end. Batch by batch, each batch of merged rows reads the rows
    # of each source that the merge seed puts in it, in two reads where they
    # wrap round from the source's last row to its first.
    if held == "batch":
        owners = (shuffled_order(66, 3) >= 37).astype(np.int64)
        expected_reads = []
        for source, n_rows in enumerate((37, 29)):
            given = 0
            for start in range(0, 66, batch_rows):
                count = int(
                    np.count_nonzero(owners[start : start + batch_rows] == source)
                )
                first = (given + first_cells[source]) % n_rows
                expected_reads += [
                    size
                    for size in (
                        min(count, n_rows - first),
                        max(0, first + count - n_rows),
                    )
                    if size
                ]
                given += count
    else:
        expected_reads = windows(37, rows, first_cells[0]) + windows(
            29, rows, first_cells[1]
        )
    if scenario == "missing":
        expected_reads += windows(37, 16, first_cells[0])
    assert sorted(reads) == sorted(expected_reads)


def _scattered(block, order_map, n_feats, dtype):
    """Scatter a block into zeroed rows and turn their negative zeros to zeros, as a dense merge batch does."""
    from cytearc.merge.writer import scatter_block_dense

    out = np.zeros((block.shape[0], n_feats), dtype=dtype)
    scatter_block_dense(
        block, order_map, n_feats, dtype, out=out, rows=np.arange(block.shape[0])
    )
    if out.dtype.kind in "fc":
        np.copyto(out, 0, where=out == 0)
    return out


def test_scattered_dense_rows_check_casts_like_sparse_coordinates():
    from cytearc.merge.writer import remap_block_to_coo, scatter_block_dense

    order_map = np.asarray([3, 0, 2])
    integers = np.asarray([[1, 0, 7], [0, 0, 255]], dtype=np.uint16)
    np.testing.assert_array_equal(
        _scattered(integers, order_map, 4, np.dtype(np.uint8)),
        remap_block_to_coo(integers, order_map, 4, 1, np.dtype(np.uint8)).toarray(),
    )
    with pytest.raises(OverflowError, match="exceed the destination dtype"):
        _scattered(integers + 1, order_map, 4, np.dtype(np.uint8))
    with pytest.raises(OverflowError, match="exceed the destination dtype"):
        remap_block_to_coo(integers + 1, order_map, 4, 1, np.dtype(np.uint8))
    floats = np.asarray([[-0.0, np.nan, 1.5]], dtype=np.float32)
    scattered = _scattered(floats, order_map, 4, np.dtype(np.float64))
    expected = remap_block_to_coo(floats, order_map, 4, 1, np.dtype(np.float64))
    assert scattered.dtype == expected.dtype == np.float64
    # Sparse coordinates drop a negative zero, so the batch writes it as zero,
    # and NaN keeps its bits.
    assert scattered.tobytes() == expected.toarray().tobytes()
    assert not np.signbit(scattered[0, 3])
    # Runs of consecutive merged columns copy as slices, and a shuffled map
    # of many runs scatters column by column, to the same rows.
    rng = np.random.default_rng(5)
    block = rng.integers(0, 4, size=(6, 90)).astype(np.uint16)
    for columns in (np.r_[100:130, 0:50, 140:150], rng.permutation(150)[:90]):
        expected = remap_block_to_coo(
            block, columns, 150, 1, np.dtype(np.uint32)
        ).toarray()
        np.testing.assert_array_equal(
            _scattered(block, columns, 150, np.dtype(np.uint32)), expected
        )
        # Into rows of a batch, the rows land there and the others stay.
        batch = np.full((9, 150), 7, dtype=np.uint32)
        rows = np.array([0, 2, 3, 5, 6, 8])
        scatter_block_dense(
            block, columns, 150, np.dtype(np.uint32), out=batch, rows=rows
        )
        placed = np.full((rows.size, 150), 7, dtype=np.uint32)
        placed[:, columns] = block
        np.testing.assert_array_equal(batch[rows], placed)
        np.testing.assert_array_equal(batch[[1, 4, 7]], 7)


@pytest.fixture(scope="module")
def narrow_shard_sources(tmp_path_factory) -> list[str]:
    """Two prepared stores of 150 and 130 cells, which merges only read."""
    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    root = tmp_path_factory.mktemp("narrow_shard_sources")
    rng = np.random.default_rng(7)
    locations = []
    for name, n_cells in (("left", 150), ("right", 130)):
        location = str(root / f"{name}.zarr")
        counts = rng.integers(0, 5, size=(n_cells, 40)).astype(np.uint16)
        write_count_store(location, {"RNA": counts}, "uint16")
        # The first opening prepares the store; tests open it again.
        DataStore(location, default_assay="RNA", nthreads=1)
        locations.append(location)
    return locations


@pytest.mark.parametrize("feature_key", ["ids", "names"])
def test_merge_reads_each_source_chunk_once_under_narrow_shards(
    tmp_path, monkeypatch, narrow_shard_sources, feature_key
):
    from collections import Counter

    from zarr.storage import LocalStore

    from cytearc import DataStore

    sources = [
        DataStore(location, default_assay="RNA", nthreads=1)
        for location in narrow_shard_sources
    ]
    source_chunks = [source.RNA.rawData.chunksize[0] for source in sources]
    fetched: Counter = Counter()
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if key.startswith("RNA/counts/c") and store.root.name in {
            "left.zarr",
            "right.zarr",
        }:
            fetched[(store.root.name, key, repr(args), repr(kwargs))] += 1
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    merger = DataStoreMerge(
        datasets=sources,
        zarr_path=str(tmp_path / "merged.zarr"),
        names=["left", "right"],
        prepend_text="",
        nthreads=1,
        overwrite=True,
        feature_key=feature_key,
        # Shards of a few rows split every source chunk into many batches.
        policy=CountMatrixPolicy(unitBytes=1024, chunkBytes=512),
    )
    shard_rows = merger.plan().assays[0].shards[0]
    assert shard_rows < min(source_chunks)
    merger.dump()

    # Each block of a source chunk's rows is read once, so no part of a chunk
    # is fetched twice.
    assert fetched
    assert max(fetched.values()) == 1
    merged = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    expected = {
        f"{name}__cell{index}": row.tolist()
        for name, source in zip(("left", "right"), sources, strict=True)
        for index, row in enumerate(np.asarray(source.RNA.rawData.compute()))
    }
    assert _rows_by_id(merged) == expected


# Source counts whose chunks span every cell of their shard, ten or twelve
# features wide, so any batch of a source's cells decodes four chunks.
_CHUNK_COLUMN_POLICY = CountMatrixPolicy(unitBytes=48_000, chunkBytes=12_000)


@pytest.fixture(scope="module")
def chunk_column_sources(tmp_path_factory) -> list[str]:
    """Two prepared stores of 600 and 500 cells by 40 features in chunk columns, which merges only read.

    The right store's last feature shares its name with its first, so a merge
    by name sums them.
    """
    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    root = tmp_path_factory.mktemp("chunk_column_sources")
    rng = np.random.default_rng(11)
    locations = []
    for name, n_cells in (("left", 600), ("right", 500)):
        location = str(root / f"{name}.zarr")
        counts = rng.integers(1, 6, size=(n_cells, 40)).astype(np.uint16)
        counts[rng.random(counts.shape) < 0.7] = 0
        names = [f"Counts{index}" for index in range(40)]
        if name == "right":
            names[-1] = names[0]
        write_count_store(
            location,
            {"Counts": counts},
            "uint16",
            feature_names={"Counts": names},
            policy=_CHUNK_COLUMN_POLICY,
        )
        # The first opening prepares the store; tests open it again.
        DataStore(location, default_assay="Counts", nthreads=1)
        locations.append(location)
    return locations


def _merge_chunk_columns(
    locations,
    path,
    monkeypatch,
    *,
    mem_budget=None,
    most_rows=None,
    feature_key="ids",
    sparse=False,
    policy=None,
    nthreads=1,
):
    """Merge the chunk-column sources into narrow bands, holding no more than ``most_rows`` rows of each source.

    ``most_rows`` 0 reads batch by batch. ``sparse`` True holds a shard of
    each source as CSR and False holds dense rows; None leaves the choice to
    the write, and records the budget that a refusal of CSR windows names.
    The bands are of 16 cells of 40 features unless ``policy`` sets them,
    and the merge runs ``nthreads`` workers. Returns the batch rows and held
    rows of the write, whether it holds them as CSR, the bands that the
    write would write at once beside dense or CSR windows of a shard, the
    decodes of each source chunk shape and the columns of each source read
    while the counts are written, the most held windows alive when a read
    starts, what the write keeps and reserves for its producer, and the
    store's files.
    """
    import weakref
    from collections import Counter

    from zarr.codecs import BloscCodec

    import cytearc.merge.writer as merge_writer
    from cytearc import DataStore

    decodes: Counter = Counter()
    widths: Counter = Counter()
    writing = [False]
    held: list[tuple[int, int]] = []
    batches: list[int] = []
    kept: dict[str, int] = {}
    # The held windows that batches were cut from, the windows that refills
    # held, and the most of these alive when a read starts.
    held_cells: list[weakref.ref] = []
    windows: list[weakref.ref] = []
    alive = [0]
    # The shard rows of the source of each refill, and the rows it held.
    refills: list[tuple[int, int]] = []
    # The budget that the plan of a write holding CSR windows named when it
    # refused them, while the write chose its windows.
    choosing_csr = [False]
    csr_needed: list[int] = []
    # The bands written at once beside windows of a shard, dense and CSR.
    shard_writes: dict[bool, int] = {}
    decode = BloscCodec._decode_single
    write = merge_datasets.write_assay_counts
    real = {
        name: getattr(merge_writer, name)
        for name in (
            "_held_windows",
            "iter_row_plan_windows",
            "controlled_compute",
            "write_dense_from_row_batches",
            "accumulate_sparse_to_shards",
            "scatter_block_dense",
            "remap_block_to_coo",
            "plan_dense_write",
            "resolve_sparse_import_batch",
        )
    }
    refill = merge_writer._HeldSourceRows._refill
    init = merge_writer._HeldSourceRows.__init__
    # Whether each held source keeps the head of its first cell's shard.
    heads: list[bool] = []

    async def counted(self, chunk_bytes, chunk_spec):
        if writing[0]:
            decodes[tuple(chunk_spec.shape)] += 1
        return await decode(self, chunk_bytes, chunk_spec)

    def watched(*args, **kwargs):
        writing[0] = True
        try:
            return write(*args, **kwargs)
        finally:
            writing[0] = False

    def held_windows(shard_rows, part_rows, fits):
        most = shard_rows if most_rows is None else most_rows

        def admitted(rows, csr):
            if sparse is not None and csr != sparse or rows > most:
                return 0
            choosing_csr[0] = csr
            try:
                writes = fits(rows, csr)
            finally:
                choosing_csr[0] = False
            if rows == shard_rows:
                shard_writes[csr] = writes
            return writes

        rows, csr = real["_held_windows"](shard_rows, part_rows, admitted)
        held.append((part_rows, rows, csr))
        return rows, csr

    def planned(name):
        def call(*args, **kwargs):
            try:
                return real[name](*args, **kwargs)
            except MemoryBudgetError as refusal:
                if choosing_csr[0]:
                    csr_needed.append(refusal.needed_bytes)
                raise

        return call

    def held_init(self, *args, **options):
        init(self, *args, **options)
        heads.append(bool(options.get("keep_head")))

    def held_refill(self, row):
        refill(self, row)
        # A CSR window holds a piece of each chunk column, which go together.
        held = self._values[0] if isinstance(self._values, list) else self._values
        windows.append(weakref.ref(held))
        refills.append((self._shardRows, self._stop - self._start))

    def windows_of(row_plan, window_rows, **options):
        # The write's batches are its windows of merged rows.
        batches.append(window_rows)
        return real["iter_row_plan_windows"](row_plan, window_rows, **options)

    def read(array, nthreads):
        # Sparse batches pass their cells through the same call.
        reads_source = hasattr(array, "compute")
        if reads_source:
            # Many batches are cut from one window, which counts once.
            live = {
                id(window())
                for window in [*held_cells, *windows]
                if window() is not None
            }
            alive[0] = max(alive[0], len(live))
        values = real["controlled_compute"](array, nthreads)
        if reads_source:
            widths[int(values.shape[1])] += 1
        return values

    def cut(name):
        def call(values, *args, **kwargs):
            # A piece, read alone or cut from a held window, spans no more
            # than a batch.
            assert values.shape[0] <= batches[-1]
            if isinstance(values, np.ndarray) and values.base is not None:
                held_cells.append(weakref.ref(values.base))
            return real[name](values, *args, **kwargs)

        return call

    def recorded(name):
        def call(*args, **kwargs):
            kept.update(
                resident=kwargs["residentBytes"],
                reserve=kwargs["producerReserveBytes"],
            )
            return real[name](*args, **kwargs)

        return call

    with monkeypatch.context() as patched:
        patched.setattr(BloscCodec, "_decode_single", counted)
        patched.setattr(merge_datasets, "write_assay_counts", watched)
        patched.setattr(merge_writer, "_held_windows", held_windows)
        patched.setattr(merge_writer, "iter_row_plan_windows", windows_of)
        patched.setattr(merge_writer, "controlled_compute", read)
        patched.setattr(merge_writer._HeldSourceRows, "_refill", held_refill)
        patched.setattr(merge_writer._HeldSourceRows, "__init__", held_init)
        for name in ("plan_dense_write", "resolve_sparse_import_batch"):
            patched.setattr(merge_writer, name, planned(name))
        for name in ("write_dense_from_row_batches", "accumulate_sparse_to_shards"):
            patched.setattr(merge_writer, name, recorded(name))
        for name in ("scatter_block_dense", "remap_block_to_coo"):
            patched.setattr(merge_writer, name, cut(name))
        DataStoreMerge(
            datasets=[
                DataStore(location, default_assay="Counts", nthreads=1)
                for location in locations
            ],
            zarr_path=str(path),
            names=["left", "right"],
            prepend_text="",
            nthreads=nthreads,
            overwrite=True,
            mem_budget=mem_budget,
            # Bands of 16 cells.
            policy=policy or CountMatrixPolicy(unitBytes=1280, chunkBytes=640),
            feature_key=feature_key,
        ).dump()
    part_rows, rows, csr = held[-1]
    return {
        "batch_rows": batches[-1],
        "part_rows": part_rows,
        "held_rows": rows,
        "sparse": csr,
        "csr_needed": csr_needed[-1] if csr_needed else None,
        "shard_writes": dict(shard_writes),
        "refills": refills,
        "heads": set(heads),
        "decodes": dict(decodes),
        "widths": dict(widths),
        "alive": alive[0],
        "cut": len(held_cells),
        **kept,
        "files": _store_files(path),
    }


def _source_parts(sizes, batch_rows: int, seed: int = 42) -> list[list[int]]:
    """Return the sizes of the reads of each source's cells when each batch of merged rows reads them alone.

    The merge seed puts each merged row's cell, as the import shuffle orders
    the sources' concatenated cells, and starts each source at the first cell
    that the plan draws from it: a batch reads its cells of a source in one
    read, or in two where they wrap round from the source's last cell to
    its first.
    """
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.cell_order import shuffled_order

    owners = np.searchsorted(
        np.cumsum(sizes), shuffled_order(sum(sizes), seed), side="right"
    )
    first_cells = build_row_plan(
        list(sizes), [f"s{index}" for index in range(len(sizes))], seed=seed
    ).firstCells
    parts: list[list[int]] = [[] for _ in sizes]
    given = [0] * len(sizes)
    for start in range(0, sum(sizes), batch_rows):
        counts = np.bincount(owners[start : start + batch_rows], minlength=len(sizes))
        for source, count in enumerate(counts.tolist()):
            if not count:
                continue
            first = (given[source] + first_cells[source]) % sizes[source]
            before = min(count, sizes[source] - first)
            parts[source] += [before, *([count - before] if count > before else [])]
            given[source] += count
    return parts


def _chunk_decodes(
    locations, rows_at_once: int, seed: int = 42
) -> dict[tuple[int, int], int]:
    """Return the decodes of each source's chunk shape when each run of ``rows_at_once`` of its cells decodes each of its chunks once.

    Each source, stored in one shard, gives its cells from the first cell
    that the plan of ``seed`` draws to its last, then from its first, so
    runs start at the first cell and again at the source's first.
    """
    from cytearc.merge.row_plan import build_row_plan

    arrays = [
        zarr.open_group(location, mode="r")["Counts/counts"] for location in locations
    ]
    sizes = [int(counts.shape[0]) for counts in arrays]
    first_cells = build_row_plan(
        sizes, [f"s{index}" for index in range(len(sizes))], seed=seed
    ).firstCells
    decodes = {}
    for counts, first in zip(arrays, first_cells, strict=True):
        n_cells, n_features = counts.shape
        chunk_rows, chunk_columns = counts.chunks
        assert chunk_rows == n_cells
        runs = (
            1
            if rows_at_once >= n_cells
            else -(-(n_cells - first) // rows_at_once) + -(-first // rows_at_once)
        )
        decodes[tuple(counts.chunks)] = runs * -(-n_features // chunk_columns)
    return decodes


def _batch_decodes(
    locations, batch_rows: int, seed: int = 42
) -> dict[tuple[int, int], int]:
    """Return the decodes of each source's chunk shape when each batch of merged rows reads its rows of each source alone, each read decoding each of the source's chunks once."""
    arrays = [
        zarr.open_group(location, mode="r")["Counts/counts"] for location in locations
    ]
    parts = _source_parts([int(counts.shape[0]) for counts in arrays], batch_rows, seed)
    decodes: dict[tuple[int, int], int] = {}
    for counts, reads in zip(arrays, parts, strict=True):
        chunks = -(-int(counts.shape[1]) // int(counts.chunks[1]))
        decodes[tuple(counts.chunks)] = (
            decodes.get(tuple(counts.chunks), 0) + len(reads) * chunks
        )
    return decodes


def _same_files(merged: dict, reference: dict) -> None:
    """Check that two merges stored the same files with the same bytes."""
    files, expected = merged["files"], reference["files"]
    assert files.keys() == expected.keys()
    assert [name for name in files if files[name] != expected[name]] == []


def _traced_band_read(location, rows: int) -> int:
    """Return the traced peak of reading ``rows`` shuffled cells of every column of a source, with its plan."""
    import tracemalloc

    from cytearc import DataStore
    from cytearc.utils.compute import controlled_compute

    counts = DataStore(location, default_assay="Counts", nthreads=1).Counts.rawData
    cells = np.random.default_rng(3).permutation(int(counts.shape[0]))[:rows]
    # The first read makes the objects that every later read reuses.
    controlled_compute(counts[cells, :], 1)
    tracemalloc.start()
    try:
        controlled_compute(counts[cells, :], 1)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("most_rows", [None, 80, 0])
def test_a_merge_decodes_each_source_chunk_once_for_each_window_of_cells_it_holds(
    tmp_path, monkeypatch, chunk_column_sources, most_rows
):
    merged = _merge_chunk_columns(
        chunk_column_sources,
        tmp_path / "merged.zarr",
        monkeypatch,
        most_rows=most_rows,
    )
    reference = _merge_chunk_columns(
        chunk_column_sources, tmp_path / "reference.zarr", monkeypatch, most_rows=0
    )

    # Each source of 600 or 500 cells, one shard, is held whole, or 80 cells
    # at a time from its first cell and again from its first stored cell,
    # and each window decodes each of the source's four chunks once: 4 + 4
    # decodes, or 8 * 4 + 8 * 4. Read batch by batch, each band of 16 merged
    # cells reads its cells of each source, and each read decodes the
    # source's four chunks.
    rows = merged["held_rows"]
    assert merged["batch_rows"] == 16
    assert rows == {None: 600, 80: 80}.get(most_rows, 0)
    if most_rows == 0:
        assert merged["decodes"] == _batch_decodes(chunk_column_sources, 16)
    else:
        assert merged["decodes"] == _chunk_decodes(chunk_column_sources, rows)
        assert {None: 8, 80: 64}[most_rows] == sum(merged["decodes"].values())
        assert sum(merged["decodes"].values()) < sum(reference["decodes"].values())
    # Held windows are read a chunk column at a time, four of 10 features of
    # the left source and three of 12 and one of 4 of the right; a batch
    # reads its cells of each source whole.
    left, right = (
        (
            _chunk_decodes(chunk_column_sources, rows)[shape] // 4
            for shape in ((600, 10), (500, 12))
        )
        if rows
        else (0, 0)
    )
    assert merged["widths"] == (
        {40: sum(_batch_reads(chunk_column_sources, 16))}
        if most_rows == 0
        else {10: 4 * left, 12: 3 * right, 4: right}
    )
    # Batches are cut from held windows. While one source's window refills,
    # the other source's stays, and the window it replaces is let go.
    assert (merged["cut"] > 0) == (most_rows != 0)
    assert merged["alive"] <= 1
    if most_rows == 0:
        # Reading batch by batch, the write reserves what a read of a band's
        # cells of every column holds, traced on a shuffled band of each
        # source.
        assert merged["reserve"] >= max(
            _traced_band_read(location, merged["batch_rows"])
            for location in chunk_column_sources
        )
    # The store's files are those of a merge that reads batch by batch.
    _same_files(merged, reference)
    stored = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    expected = {
        f"{name}__cell{index}": row.tolist()
        for name, location in zip(("left", "right"), chunk_column_sources, strict=True)
        for index, row in enumerate(
            zarr.open_group(location, mode="r")["Counts/counts"][:]
        )
    }
    assert _rows_by_id(stored, "Counts/counts") == expected


# Merged bands of 16 cells of 200 uint16 features.
_SPARSE_BAND_POLICY = CountMatrixPolicy(unitBytes=16 * 400, chunkBytes=16 * 200)


@pytest.fixture(scope="module")
def sparse_source_stores(tmp_path_factory) -> dict[str, str]:
    """Prepared stores of 600 and 500 cells by 200 features, 5% of them 1 to 5, which merges only read.

    The left store holds shards of 256 cells; the right store's cells are
    held in shards of 128 cells, and again in shards of 256 with the name
    of the first feature given to the last too, so a merge by name sums
    them.
    """
    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    root = tmp_path_factory.mktemp("sparse_column_sources")
    locations = {}
    for name, n_cells, shard_rows, seed in (
        ("left", 600, 256, 13),
        ("right", 500, 128, 14),
        ("right in shards of 256", 500, 256, 14),
    ):
        rng = np.random.default_rng(seed)
        location = str(root / f"{name.replace(' ', '_')}.zarr")
        counts = np.zeros((n_cells, 200), dtype=np.uint16)
        nonzero = rng.random(counts.shape) < 0.05
        counts[nonzero] = rng.integers(1, 6, size=int(nonzero.sum()))
        names = [f"Counts{index}" for index in range(200)]
        if name == "right in shards of 256":
            names[-1] = names[0]
        write_count_store(
            location,
            {"Counts": counts},
            "uint16",
            feature_names={"Counts": names},
            policy=CountMatrixPolicy(
                unitBytes=shard_rows * 400, chunkBytes=shard_rows * 100
            ),
        )
        # The first opening prepares the store; tests open it again.
        DataStore(location, default_assay="Counts", nthreads=1)
        locations[name] = location
    return locations


@pytest.fixture(scope="module")
def sparse_column_sources(sparse_source_stores) -> list[str]:
    """The sparse stores in shards of 256 and of 128 cells."""
    return [sparse_source_stores["left"], sparse_source_stores["right"]]


@pytest.fixture(scope="module")
def sparse_shard_sources(sparse_source_stores) -> list[str]:
    """The sparse stores, both in shards of 256 cells."""
    return [
        sparse_source_stores["left"],
        sparse_source_stores["right in shards of 256"],
    ]


def _shard_chunk_reads(
    locations, *, heads_kept: bool = True, seed: int = 42
) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    """Return the decodes of each chunk shape, and the reads of each width, when each stored shard of each source is read once, a chunk column at a time.

    A source of several shards gives the cells of its first cell's shard
    before that cell last; unless the merge keeps them apart, ``heads_kept``,
    their shard is read again.
    """
    from collections import Counter

    from cytearc.merge.row_plan import build_row_plan

    arrays = [
        zarr.open_group(location, mode="r")["Counts/counts"] for location in locations
    ]
    first_cells = build_row_plan(
        [int(counts.shape[0]) for counts in arrays],
        [f"s{index}" for index in range(len(arrays))],
        seed=seed,
    ).firstCells
    decodes: Counter = Counter()
    widths: Counter = Counter()
    for counts, first in zip(arrays, first_cells, strict=True):
        (n_cells, n_features), (chunk_rows, chunk_columns) = counts.shape, counts.chunks
        assert counts.shards[0] == chunk_rows
        shards = -(-n_cells // chunk_rows)
        if not heads_kept and shards > 1 and first % chunk_rows:
            shards += 1
        decodes[tuple(counts.chunks)] += shards * -(-n_features // chunk_columns)
        for start in range(0, n_features, chunk_columns):
            widths[min(chunk_columns, n_features - start)] += shards
    return dict(decodes), dict(widths)


def _source_rows(locations) -> dict[str, list]:
    """Return the rows of the merge sources keyed by merged cell ID, as NumPy reads them."""
    return {
        f"{name}__cell{index}": row.tolist()
        for name, location in zip(("left", "right"), locations, strict=True)
        for index, row in enumerate(
            np.asarray(zarr.open_group(location, mode="r")["Counts/counts"][:])
        )
    }


@pytest.mark.parametrize(
    ("sources", "feature_key"),
    [
        ("chunk_column_sources", "ids"),
        # A merge by name sums two features of the right source, so it writes
        # sparse coordinates.
        ("chunk_column_sources", "names"),
        ("sparse_column_sources", "ids"),
    ],
)
def test_a_merge_holding_csr_windows_decodes_each_source_chunk_once(
    tmp_path, monkeypatch, request, sources, feature_key
):
    locations = request.getfixturevalue(sources)
    policy = None if sources == "chunk_column_sources" else _SPARSE_BAND_POLICY
    merged, reference = (
        _merge_chunk_columns(
            locations,
            tmp_path / f"{name}.zarr",
            monkeypatch,
            policy=policy,
            feature_key=feature_key,
            **options,
        )
        for name, options in (
            ("merged", {"sparse": True}),
            ("reference", {"most_rows": 0}),
        )
    )

    # The write holds one stored shard of each source at a time as CSR: of
    # 600 or 500 cells, or of 256 cells of the left source, which has three,
    # and 128 cells of the right, which has four. Each refill reads a chunk
    # column at a time, so each chunk is decoded once, where batches that
    # read their own cells decode each chunk for each batch that holds the
    # source's cells.
    decodes, widths = _shard_chunk_reads(locations)
    assert merged["sparse"] and not reference["sparse"]
    assert merged["held_rows"] == {"chunk_column_sources": 600}.get(sources, 256)
    assert {shard for shard, _ in merged["refills"]} == (
        {600, 500} if sources == "chunk_column_sources" else {256, 128}
    )
    assert all(rows <= shard for shard, rows in merged["refills"])
    assert merged["decodes"] == decodes
    assert merged["widths"] == widths
    assert sum(merged["decodes"].values()) < sum(reference["decodes"].values())
    # Pieces are dense copies of a window's CSR rows, so none is a view of a
    # window, and a refill lets go of the window that it replaces.
    assert merged["cut"] == 0
    assert merged["alive"] <= 1
    _same_files(merged, reference)
    if feature_key == "ids":
        stored = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
        assert _rows_by_id(stored, "Counts/counts") == _source_rows(locations)
    # Beyond what a write that reads batch by batch keeps, the write keeps a
    # CSR window of each source's shard of the most values, counted with
    # NumPy, in a piece for each chunk column: 2 bytes for each uint16 value
    # and 4 for its column, a spare value in each piece, and 4 bytes for the
    # offset of each row and one more in each piece. A source of several
    # shards also keeps the cells of its first cell's shard before that cell,
    # the head, in pieces of the same kind. The write reserves what a refill
    # and a piece taken from a window hold, a piece as many cells as the
    # merge seed gives a batch of one source at most.
    from cytearc import DataStore
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.merge.writer import _chunk_column_runs, _csr_window_bytes
    from cytearc.storage.cell_order import shuffled_order

    sizes = [
        int(zarr.open_group(location, mode="r")["Counts/counts"].shape[0])
        for location in locations
    ]
    first_cells = build_row_plan(sizes, ["left", "right"], seed=42).firstCells
    owners = np.searchsorted(
        np.cumsum(sizes), shuffled_order(sum(sizes), 42), side="right"
    )
    batch_rows = merged["batch_rows"]
    piece_rows = max(
        int(np.bincount(owners[start : start + batch_rows]).max())
        for start in range(0, sum(sizes), batch_rows)
    )
    assert merged["heads"] == {True}

    def csr_bytes(values, rows, pieces):
        return (values + pieces) * (2 + 4) + pieces * (rows + 1) * 4

    windows, heads, charges = [], [], []
    for location, first in zip(locations, first_cells, strict=True):
        counts = zarr.open_group(location, mode="r")["Counts/counts"]
        shard_rows = counts.chunks[0]
        stored = counts[:]
        values = max(
            int(np.count_nonzero(stored[start : start + shard_rows]))
            for start in range(0, counts.shape[0], shard_rows)
        )
        pieces = -(-counts.shape[1] // counts.chunks[1])
        windows.append(csr_bytes(values, shard_rows, pieces))
        head = first % shard_rows if counts.shape[0] > shard_rows else 0
        if head:
            heads.append(
                csr_bytes(
                    int(np.count_nonzero(stored[first - head : first])), head, pieces
                )
            )
        source = DataStore(location, default_assay="Counts", nthreads=1).Counts
        charges.append(
            _csr_window_bytes(
                source.rawData, _chunk_column_runs(source.rawData), values, piece_rows
            )
        )
    assert merged["resident"] - reference["resident"] == sum(windows) + sum(heads)
    assert bool(heads) == (sources == "sparse_column_sources")
    assert [window for window, _, _ in charges] == windows
    reads = max(refill for _, refill, _ in charges) + max(
        take for _, _, take in charges
    )
    assert merged["reserve"] >= reads
    if (sources, feature_key) == ("chunk_column_sources", "ids"):
        from cytearc.merge.row_plan import WINDOW_BYTES_PER_ROW

        # Beside them, a batch of 16 cells reserves the 40 uint16 values of
        # each cell that it assembles, the band of 16 cells that it is copied
        # into and its window's index arrays, and no cast, as the sources'
        # counts take the merged type.
        assert merged["reserve"] == (
            reads + 2 * 16 * 40 * 2 + 16 * WINDOW_BYTES_PER_ROW
        )


@pytest.mark.slow
def test_a_merge_holds_csr_windows_from_the_least_budget_that_fits_them(
    tmp_path, monkeypatch, sparse_shard_sources
):
    def merge(path, budget, **options):
        return _merge_chunk_columns(
            sparse_shard_sources,
            path,
            monkeypatch,
            mem_budget=budget,
            sparse=None,
            policy=_SPARSE_BAND_POLICY,
            **options,
        )

    # Each refusal names a budget that passes its step, and each refusal of
    # CSR windows the budget that fits them, up to the least budget at which
    # the write holds them.
    budget = 1 << 16
    for _ in range(20):
        try:
            merged = merge(tmp_path / "merged.zarr", budget)
        except MemoryBudgetError as refusal:
            assert refusal.needed_bytes > budget
            budget = refusal.needed_bytes
            continue
        if merged["sparse"]:
            break
        assert merged["csr_needed"] > budget
        budget = merged["csr_needed"]
    else:
        raise AssertionError(f"No budget up to {budget} bytes held CSR windows")
    below = merge(tmp_path / "below.zarr", budget - 1)
    ample = merge(tmp_path / "ample.zarr", None)

    # At that budget the CSR windows have no room for the cells of each
    # source's first shard before its first cell, so that shard is read
    # again at the end. A byte below that budget, the refusal of CSR windows
    # names it, and the write holds dense windows shorter than a shard or
    # reads batch by batch, which decode chunks more often. With room for
    # dense windows of a shard, which leave the write of one band at a time
    # as many bands as CSR windows do, the write holds them dense, keeps the
    # heads, decodes each chunk once and converts nothing.
    decodes, _ = _shard_chunk_reads(sparse_shard_sources)
    assert merged["held_rows"] == 256
    assert merged["heads"] == {False}
    assert (
        merged["decodes"]
        == _shard_chunk_reads(sparse_shard_sources, heads_kept=False)[0]
    )
    assert sum(merged["decodes"].values()) > sum(decodes.values())
    assert not below["sparse"]
    assert below["csr_needed"] == budget
    assert 0 < below["held_rows"] < 256
    assert sum(below["decodes"].values()) > sum(merged["decodes"].values())
    # Dense windows of part of a shard keep no head: beside what a write that
    # reads batch by batch keeps, the write keeps their rows of the sources'
    # 400 uint16 features alone, at that budget and at an ample one.
    batchwise = merge(tmp_path / "batchwise.zarr", None, most_rows=0)
    partial = merge(tmp_path / "partial.zarr", None, most_rows=120)
    assert partial["held_rows"] == 120
    for windows in (below, partial):
        held = windows["held_rows"] * 400 * 2
        assert windows["resident"] - batchwise["resident"] == held
    assert not ample["sparse"]
    assert ample["held_rows"] == 256
    assert ample["heads"] == {True}
    assert ample["decodes"] == decodes
    _same_files(merged, below)
    _same_files(merged, ample)
    _same_files(merged, partial)
    stored = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    assert _rows_by_id(stored, "Counts/counts") == _source_rows(sparse_shard_sources)


@pytest.mark.slow
# A merge by name sums two features of the right source, so it writes sparse
# coordinates.
@pytest.mark.parametrize("feature_key", ["ids", "names"])
def test_a_merge_holds_csr_windows_while_dense_ones_would_write_fewer_bands_at_once(
    tmp_path, monkeypatch, sparse_shard_sources, feature_key
):
    def merge(budget):
        return _merge_chunk_columns(
            sparse_shard_sources,
            tmp_path / f"merged-{budget}.zarr",
            monkeypatch,
            mem_budget=budget,
            sparse=None,
            policy=_SPARSE_BAND_POLICY,
            nthreads=2,
            feature_key=feature_key,
        )

    # Between a budget at which the write of two workers holds CSR windows
    # and the default, at which it holds dense ones, the least budget that
    # holds dense windows of a shard is found by halving.
    low, high = 1 << 19, 1 << 24
    assert merge(low)["sparse"]
    assert not merge(high)["sparse"]
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if merge(middle)["sparse"] else (low, middle)
    holding_csr, holding_dense = merge(low), merge(high)

    # A byte below it, dense windows of a shard of every source fit beside
    # the write of one band at a time, and CSR ones beside two, so the write
    # holds CSR windows; from it on, dense ones leave two bands at once as
    # well. Each decodes each chunk once, but for the first shards that it
    # reads again when it has no room to keep their heads, and both store
    # the same files.
    assert holding_csr["shard_writes"] == {False: 1, True: 2}
    assert holding_dense["shard_writes"] == {False: 2, True: 2}
    assert holding_csr["held_rows"] == holding_dense["held_rows"] == 256
    for merged in (holding_csr, holding_dense):
        (kept,) = merged["heads"]
        assert (
            merged["decodes"]
            == _shard_chunk_reads(sparse_shard_sources, heads_kept=kept)[0]
        )
    _same_files(holding_csr, holding_dense)


def test_a_merge_bounds_no_shard_of_a_source_without_cells(tmp_path, monkeypatch):
    import cytearc.merge.writer as merge_writer
    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    counts = np.arange(20, dtype=np.uint16).reshape(5, 4)
    for name, values in (("full", counts), ("empty", counts[:0])):
        write_count_store(str(tmp_path / f"{name}.zarr"), {"Counts": values}, "uint16")
    bounds: list[list[int]] = []
    real = merge_writer._cumulative_row_nnz

    def recorded(*args, **kwargs):
        cumulative, shard_values, head_values = real(*args, **kwargs)
        bounds.append(shard_values)
        return cumulative, shard_values, head_values

    monkeypatch.setattr(merge_writer, "_cumulative_row_nnz", recorded)
    DataStoreMerge(
        datasets=[
            DataStore(
                str(tmp_path / f"{name}.zarr"), default_assay="Counts", nthreads=1
            )
            for name in ("full", "empty")
        ],
        zarr_path=str(tmp_path / "merged.zarr"),
        names=["full", "empty"],
        prepend_text="",
        overwrite=True,
        nthreads=1,
    ).dump()

    # The full source's one shard holds its 19 nonzero values, and the source
    # without cells holds none; the merge holds the full source's cells alone.
    assert bounds[-1] == [19, 0]
    stored = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    assert _rows_by_id(stored, "Counts/counts") == {
        f"full__cell{index}": row.tolist() for index, row in enumerate(counts)
    }


def _csr_window_source(location, dtype: str, density: float) -> np.ndarray:
    """Write 1,000 cells by 2,000 features in shards of 400 cells, ``density`` of them 1 to 9, and return the counts.

    The layout aligns the chunk columns, of up to 500 features, to 400.
    """
    from tests.storage_helpers import write_count_store

    rng = np.random.default_rng(5)
    values = np.zeros((1000, 2000), dtype=dtype)
    nonzero = rng.random(values.shape) < density
    values[nonzero] = rng.integers(1, 10, size=int(nonzero.sum()))
    itemsize = np.dtype(dtype).itemsize
    write_count_store(
        location,
        {"Counts": values},
        dtype,
        policy=CountMatrixPolicy(
            unitBytes=400 * 2000 * itemsize, chunkBytes=400 * 500 * itemsize
        ),
    )
    return values


@pytest.mark.parametrize(("dtype", "density"), [("uint16", 0.05), ("float32", 0.4)])
def test_a_csr_window_holds_no_more_than_its_charge(tmp_path, dtype, density):
    import tracemalloc

    from cytearc import DataStore
    from cytearc.merge.writer import (
        _chunk_column_runs,
        _csr_window_bytes,
        _HeldSourceRows,
    )
    from cytearc.utils.arrays import sparse_matrix_bytes

    location = str(tmp_path / "source.zarr")
    values = _csr_window_source(location, dtype, density)
    counts = DataStore(location, default_assay="Counts", nthreads=1).Counts.rawData
    runs = _chunk_column_runs(counts)
    assert counts.chunksize[0] == 400 and len(runs) == 5
    # The shard of the most values, as NumPy counts them, bounds the window.
    shard_values = [
        int(np.count_nonzero(values[start : start + 400]))
        for start in range(0, 1000, 400)
    ]
    first = int(np.argmax(shard_values)) * 400
    window, refill, take = _csr_window_bytes(counts, runs, max(shard_values), 64)
    held = _HeldSourceRows(counts, runs, 400, 1, sparse=True)
    # A refill of another shard first makes the objects that later reads reuse.
    for _ in held.take((first + 400) % 1200, 64):
        pass
    tracemalloc.start()
    try:
        offset, rows = next(held.take(first, 64))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert offset == 0
    np.testing.assert_array_equal(rows, values[first : first + 64])
    # The window holds what its charge counts: the values and their columns
    # and the row offsets of a piece for each chunk column, beside a spare
    # value and column in each piece.
    itemsize = np.dtype(dtype).itemsize
    assert len(held._values) == 5
    assert sparse_matrix_bytes(*held._values) + 5 * (itemsize + 4) == window
    # The window, its refill and the first piece taken from it hold no more
    # than their charge, and more than the window and the piece alone, as
    # the read of a chunk column holds the chunk's rows.
    assert peak <= window + refill + take
    assert peak > window + take + 400 * 400 * itemsize
    # A piece taken from the window without a refill holds its dense rows,
    # which its charge counts, and a few objects.
    tracemalloc.start()
    try:
        offset, rows = next(held.take(first + 64, 64))
        piece = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(rows, values[first + 64 : first + 128])
    assert rows.nbytes == take
    assert take <= piece <= take + 4_096


def test_a_dense_merge_write_makes_each_batch_once_the_previous_one_is_gone(
    tmp_path, monkeypatch, chunk_column_sources
):
    import weakref

    import cytearc.merge.writer as merge_writer
    from cytearc import DataStore

    # Each batch that the write takes is watched. Each window of merged rows
    # that the next batch takes, and each piece that a batch scatters, looks
    # whether the batch before it is still alive.
    yielded: list[weakref.ref] = []
    alive: list[bool] = []
    write, scatter, windows = (
        merge_writer.write_dense_from_row_batches,
        merge_writer.scatter_block_dense,
        merge_writer.iter_row_plan_windows,
    )

    def watched_write(destination, batches, **options):
        def watched():
            for batch in batches:
                yielded.append(weakref.ref(batch))
                yield batch
                del batch

        return write(destination, watched(), **options)

    def watched_scatter(*args, **options):
        if yielded:
            alive.append(yielded[-1]() is not None)
        return scatter(*args, **options)

    def watched_windows(*args, **options):
        for window in windows(*args, **options):
            if yielded:
                alive.append(yielded[-1]() is not None)
            yield window

    monkeypatch.setattr(merge_writer, "write_dense_from_row_batches", watched_write)
    monkeypatch.setattr(merge_writer, "scatter_block_dense", watched_scatter)
    monkeypatch.setattr(merge_writer, "iter_row_plan_windows", watched_windows)
    DataStoreMerge(
        datasets=[
            DataStore(location, default_assay="Counts", nthreads=1)
            for location in chunk_column_sources
        ],
        zarr_path=str(tmp_path / "merged.zarr"),
        names=["left", "right"],
        prepend_text="",
        nthreads=1,
        # Bands of 16 cells.
        policy=CountMatrixPolicy(unitBytes=1280, chunkBytes=640),
    ).dump()

    # The write takes 69 batches of 16 cells, and starts each after the first
    # once neither it nor the band buffer holds the batch before it, which
    # the write's reserve does not count.
    assert len(yielded) == -(-1100 // 16)
    assert len(alive) > 2 * len(yielded)
    assert not any(alive)


@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_held_csr_windows_give_each_source_row_once_in_order(tmp_path, dtype):
    from cytearc import DataStore
    from cytearc.merge.writer import _chunk_column_runs, _HeldSourceRows

    location = str(tmp_path / "source.zarr")
    values = _csr_window_source(location, dtype, 0.05)
    counts = DataStore(location, default_assay="Counts", nthreads=1).Counts.rawData
    held = _HeldSourceRows(counts, _chunk_column_runs(counts), 400, 1, sparse=True)
    # Takes of 1 to 300 rows, which cross the shards' ends at rows 400 and
    # 800, come in pieces that each end within one shard.
    rng = np.random.default_rng(2)
    first, pieces = 0, []
    while first < 1000:
        count = min(int(rng.integers(1, 301)), 1000 - first)
        for offset, rows in held.take(first, count):
            start = first + offset
            assert start // 400 == (start + rows.shape[0] - 1) // 400
            assert rows.dtype == values.dtype
            pieces.append((start, rows))
        first += count
    starts = [start for start, _ in pieces]
    assert starts == sorted(starts)
    np.testing.assert_array_equal(np.vstack([rows for _, rows in pieces]), values)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_held_rows_give_the_head_of_the_first_cells_shard_last_without_its_window(
    tmp_path, sparse
):
    import weakref

    from cytearc import DataStore
    from cytearc.merge.writer import _chunk_column_runs, _HeldSourceRows

    location = str(tmp_path / "source.zarr")
    values = _csr_window_source(location, "uint16", 0.05)
    counts = DataStore(location, default_assay="Counts", nthreads=1).Counts.rawData
    held = _HeldSourceRows(
        counts,
        _chunk_column_runs(counts),
        400,
        1,
        sparse=sparse,
        first_row=530,
        keep_head=True,
    )
    refills, first_window = [], []
    refill = held._refill

    def recorded(row):
        refill(row)
        refills.append(row)
        window = held._values[0] if sparse else held._values
        first_window.append(weakref.ref(window))

    held._refill = recorded
    # The source gives its rows from row 530 on and then from row 0, in takes
    # of 50 rows, so rows 400 to 529 of the first window's shard come last.
    # Copies of the rows taken leave the windows to the held rows alone.
    pieces = []
    for first, stop in ((530, 1000), (0, 530)):
        for start in range(first, stop, 50):
            taken = held.take(start, min(50, stop - start))
            pieces.extend(rows.copy() for _, rows in taken)
    # Each shard is read once: the head comes from what the first window
    # kept, which lets the window itself go when the next one is read.
    assert refills == [530, 800, 0]
    assert first_window[0]() is None
    np.testing.assert_array_equal(np.vstack(pieces), np.r_[values[530:], values[:530]])


def _batch_reads(locations, batch_rows: int, seed: int = 42) -> list[int]:
    """Return the reads of each source when each batch of merged rows reads its rows of each source alone."""
    sizes = [
        int(zarr.open_group(location, mode="r")["Counts/counts"].shape[0])
        for location in locations
    ]
    return [len(reads) for reads in _source_parts(sizes, batch_rows, seed)]


@pytest.mark.parametrize("feature_key", ["ids", "names"])
def test_a_merge_at_its_least_budget_holds_windows_of_its_source_cells(
    tmp_path, monkeypatch, chunk_column_sources, feature_key
):
    from cytearc import DataStore
    from cytearc.merge.writer import _chunk_column_runs, _held_read_bytes

    # Each refusal names a budget that passes its step, up to the least
    # budget at which the merge runs. A merge by name sums two features of
    # the right source, so it writes sparse coordinates.
    budget = 1 << 16
    for _ in range(10):
        try:
            merged = _merge_chunk_columns(
                chunk_column_sources,
                tmp_path / "merged.zarr",
                monkeypatch,
                mem_budget=budget,
                feature_key=feature_key,
            )
            break
        except MemoryBudgetError as refusal:
            assert refusal.needed_bytes > budget
            budget = refusal.needed_bytes
    else:
        raise AssertionError(f"No budget up to {budget} bytes ran the merge")
    reference = _merge_chunk_columns(
        chunk_column_sources,
        tmp_path / "reference.zarr",
        monkeypatch,
        mem_budget=budget,
        most_rows=0,
        feature_key=feature_key,
    )

    # At the least budget the write holds a window of each source's cells,
    # several batches long, so each source chunk is decoded once for each
    # window instead of once for each batch that holds the source's cells.
    batch_rows, rows = merged["batch_rows"], merged["held_rows"]
    assert batch_rows == reference["batch_rows"]
    assert reference["held_rows"] == 0
    assert rows >= 2 * batch_rows
    assert merged["decodes"] == _chunk_decodes(chunk_column_sources, rows)
    assert reference["decodes"] == _batch_decodes(chunk_column_sources, batch_rows)
    assert sum(merged["decodes"].values()) < sum(reference["decodes"].values())
    assert merged["cut"] > 0
    assert merged["alive"] <= 1
    _same_files(merged, reference)
    # The write keeps a window of each source, 40 values a cell, beyond what
    # a write that reads batch by batch keeps, and reserves the read of a
    # chunk column of a window. A cell takes the wider of the sources'
    # uint16 and the merged type: uint16 by ID, and uint32 by name, which
    # sums features.
    merged_type = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")[
        "Counts/counts"
    ].dtype
    assert merged_type == {"ids": np.uint16, "names": np.uint32}[feature_key]
    assert merged["resident"] - reference["resident"] == (
        2 * rows * 40 * merged_type.itemsize
    )
    assert merged["reserve"] >= max(
        _held_read_bytes(
            _chunk_column_runs(
                DataStore(location, default_assay="Counts", nthreads=1).Counts.rawData
            )[0],
            rows,
        )
        for location in chunk_column_sources
    )


@pytest.mark.slow
def test_a_merge_holds_windows_of_a_composite_source(
    tmp_path, monkeypatch, chunk_column_sources
):
    from cytearc import DataStore
    from cytearc.composite import create_composite

    left, right = chunk_column_sources
    joint = str(tmp_path / "joint.zarr")
    create_composite(
        {
            name: DataStore(location, default_assay="Counts", nthreads=1)
            for name, location in (("a", left), ("b", right))
        },
        at=joint,
        features="union",
        assay="Counts",
        nthreads=1,
    )
    merged = _merge_chunk_columns(
        [joint, left], tmp_path / "merged.zarr", monkeypatch, most_rows=80
    )
    reference = _merge_chunk_columns(
        [joint, left], tmp_path / "reference.zarr", monkeypatch, most_rows=0
    )

    # The composite's 1,100 cells, which it reads from its sources in one run
    # of every column, are held 80 cells at a time within each of its
    # shards, beside windows of the left source's 600 cells read a chunk
    # column at a time.
    shard = DataStore(
        joint, default_assay="Counts", nthreads=1
    ).Counts.rawData.chunksize[0]
    joint_reads = sum(
        -(-min(shard, 1100 - start) // 80) for start in range(0, 1100, shard)
    )
    assert merged["batch_rows"] == 16
    assert merged["held_rows"] == 80
    assert merged["widths"] == {40: joint_reads, 10: 4 * -(-600 // 80)}
    assert sum(merged["decodes"].values()) < sum(reference["decodes"].values())
    assert merged["cut"] > 0
    assert merged["alive"] <= 1
    _same_files(merged, reference)


def test_a_merge_holds_what_fits_of_one_chunk_column_whose_whole_read_does_not(
    tmp_path, monkeypatch
):
    import cytearc.merge.writer as merge_writer
    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    # Sources of 1,200 and 1,000 cells by 200 float32 features, 10% of them 1
    # to 9, each stored in one chunk of every cell and feature; the right
    # source's last feature shares its name with its first, so a merge by
    # name sums them. At 4,324,240 bytes a merge by name planned to hold each
    # whole block, whose read then refused the budget halfway through the
    # dump (#541).
    ids = [f"g{index}" for index in range(200)]
    right_names = [*ids[:-1], ids[0]]
    sources = {}
    for name, n_cells, seed, names in (
        ("left", 1200, 1, ids),
        ("right", 1000, 2, right_names),
    ):
        rng = np.random.default_rng(seed)
        values = np.zeros((n_cells, 200), dtype=np.float32)
        nonzero = rng.random(values.shape) < 0.1
        values[nonzero] = rng.integers(1, 10, size=int(nonzero.sum()))
        location = str(tmp_path / f"{name}.zarr")
        write_count_store(
            location,
            {"Counts": values},
            "float32",
            feature_ids={"Counts": ids},
            feature_names={"Counts": names},
            cell_ids=[f"{name[0]}{index}" for index in range(n_cells)],
            policy=CountMatrixPolicy(
                unitBytes=1200 * 200 * 4, chunkBytes=1200 * 200 * 4
            ),
        )
        sources[name] = location, values, names
    held: list[tuple[int, int, int, bool]] = []
    real = merge_writer._held_windows

    def recorded(shard_rows, part_rows, fits):
        rows, sparse = real(shard_rows, part_rows, fits)
        held.append((shard_rows, part_rows, rows, sparse))
        return rows, sparse

    monkeypatch.setattr(merge_writer, "_held_windows", recorded)
    path = tmp_path / "merged.zarr"
    DataStoreMerge(
        datasets=[
            DataStore(location, default_assay="Counts", nthreads=1)
            for location, _, _ in sources.values()
        ],
        zarr_path=str(path),
        names=list(sources),
        prepend_text="",
        nthreads=1,
        mem_budget=4_324_240,
        overwrite=True,
        feature_key="names",
    ).dump()

    # The write holds a window of each source, of at least a batch's share
    # of the larger source and at most its 1,200-cell shard, whose reads fit
    # as planned.
    shard_rows, part_rows, rows, sparse = held[-1]
    assert shard_rows == 1200
    assert part_rows <= rows <= shard_rows
    # Each merged cell holds its source's values summed by feature name.
    stored = zarr.open_group(str(path), mode="r")
    merged_names = np.asarray(stored["Counts/featureData/names"][:]).astype(str)
    expected = {}
    for name, (_, values, names) in sources.items():
        columns = np.asarray(names)
        for index, row in enumerate(values):
            expected[f"{name}__{name[0]}{index}"] = [
                float(row[columns == feature].sum()) for feature in merged_names
            ]
    assert _rows_by_id(stored, "Counts/counts") == expected


def test_a_merge_lets_an_allocation_failure_in_its_held_row_search_through(
    tmp_path, monkeypatch, chunk_column_sources
):
    import cytearc.merge.writer as merge_writer
    from cytearc import DataStore

    # Shards of 600 cells span many bands of 16, so the write plans the cells
    # it could hold before it reads any; an allocation that fails while it
    # plans is no refusal of those cells and reaches the caller.
    failure = MemoryError("the allocation failed")
    plans: list[int] = []
    searches: list[tuple[int, int]] = []
    plan = merge_writer.plan_dense_write
    search = merge_writer._held_windows

    def fails_first(*args, **kwargs):
        plans.append(kwargs["residentBytes"])
        if len(plans) == 1:
            raise failure
        return plan(*args, **kwargs)

    def recorded(shard_rows, part_rows, fits):
        searches.append((shard_rows, part_rows))
        return search(shard_rows, part_rows, fits)

    monkeypatch.setattr(merge_writer, "plan_dense_write", fails_first)
    monkeypatch.setattr(merge_writer, "_held_windows", recorded)
    with pytest.raises(MemoryError) as raised:
        DataStoreMerge(
            datasets=[
                DataStore(location, default_assay="Counts", nthreads=1)
                for location in chunk_column_sources
            ],
            zarr_path=str(tmp_path / "merged.zarr"),
            names=["left", "right"],
            prepend_text="",
            nthreads=1,
            overwrite=True,
            policy=CountMatrixPolicy(unitBytes=1280, chunkBytes=640),
        ).dump()
    assert raised.value is failure
    # Batches of 16 cells take 9 of them from the larger source, on average.
    assert searches == [(600, -(-16 * 600 // 1100))]
    assert len(plans) == 1


@pytest.mark.parametrize("column", ["wide", "narrow"])
def test_a_held_read_holds_no_more_than_its_charge(tmp_path, column):
    import tracemalloc

    from cytearc import DataStore
    from cytearc.merge.writer import _chunk_column_runs, _held_read_bytes
    from cytearc.storage.geometry import array_geometry
    from cytearc.utils.compute import controlled_compute
    from tests.storage_helpers import write_count_store

    rng = np.random.default_rng(5)
    if column == "wide":
        # 1,000 cells by 2,000 float32 features in chunks of every cell and
        # 500 features, half zeros, which compress to a fraction of 2 MB.
        n_cells, n_features, width, dtype = 1000, 2000, 500, np.float32
    else:
        # 100,000 cells by 4 uint16 features in chunks of every cell and one
        # feature, so a read holds far more integers indexing its cells
        # than values.
        n_cells, n_features, width, dtype = 100_000, 4, 1, np.uint16
    values = rng.integers(0, 10, size=(n_cells, n_features)).astype(dtype)
    values[rng.random(values.shape) < 0.5] = 0
    location = str(tmp_path / "source.zarr")
    itemsize = np.dtype(dtype).itemsize
    write_count_store(
        location,
        {"Counts": values},
        np.dtype(dtype).name,
        policy=CountMatrixPolicy(
            unitBytes=n_cells * n_features * itemsize,
            chunkBytes=n_cells * width * itemsize,
        ),
    )
    # No floor of features per cell, which four features would trip.
    counts = DataStore(
        location, default_assay="Counts", nthreads=1, min_features_per_cell=-1
    ).Counts.rawData
    run = _chunk_column_runs(counts)[0]
    assert run.shape == (n_cells, width)
    cells = rng.permutation(n_cells)
    # The first read makes the objects that every later read reuses.
    controlled_compute(run[cells, :], 1)
    tracemalloc.start()
    try:
        held = controlled_compute(run[cells, :], 1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    np.testing.assert_array_equal(held, values[cells, :width])
    # A read of every cell of the chunk column by an index array holds the
    # rows that Zarr copies from the chunk and Zarr's indexers of the cells,
    # beside the copies that a read of a run of cells holds. The charge
    # holds the read only with the copy for the wide column, and only with
    # the indexers, 64 bytes a cell of a sharded read, for the narrow one.
    charge = _held_read_bytes(run, n_cells)
    assert peak <= charge
    if column == "wide":
        assert peak > charge - n_cells * width * itemsize
    else:
        geometry = array_geometry(counts._backing)
        assert geometry is not None and geometry.shards is not None
        assert geometry.indexBytes(n_cells) == 64 * n_cells
        assert peak > charge - 64 * n_cells
    # A window reads a run of cells as a slice, whose positions the chunked
    # array keeps in an index array, and the same charge holds it: the
    # narrow read holds three arrays of every cell's position, 2.4 MB, where
    # a charge of a read of slices would give 0.9 MB.
    controlled_compute(run[0:n_cells, :], 1)
    tracemalloc.start()
    try:
        held = controlled_compute(run[0:n_cells, :], 1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(held, values[:, :width])
    assert peak <= charge
    if column == "narrow":
        assert peak > run._read_bytes(n_cells * width * itemsize)


@pytest.mark.parametrize(
    ("shard_rows", "part_rows", "most", "dense_writes", "csr_writes", "expected"),
    [
        # A whole shard of each source held dense, when that leaves the write
        # two bands at once or as many as CSR windows do, or else as CSR.
        (16, 4, 100, 4, 4, (16, False)),
        (16, 4, 16, 4, 2, (16, False)),
        (16, 4, 16, 2, 4, (16, False)),
        (16, 4, 16, 1, 2, (16, True)),
        (16, 4, 16, 1, 1, (16, False)),
        (16, 4, 16, 1, 0, (16, False)),
        # A whole shard as CSR where dense windows of a shard do not fit, or
        # else the most dense rows that fit.
        (16, 4, 15, 4, 1, (16, True)),
        (16, 4, 10, 4, 0, (10, False)),
        # At least a batch's share of the largest source, or else batch by
        # batch.
        (16, 4, 4, 4, 0, (4, False)),
        (16, 4, 3, 4, 0, (0, False)),
        (16, 4, 3, 4, 1, (16, True)),
        (100, 7, 0, 4, 0, (0, False)),
        (100, 7, 0, 4, 2, (100, True)),
        # A shard shorter than that share is held whole or not at all.
        (3, 4, 100, 1, 1, (3, False)),
        (3, 4, 2, 1, 0, (0, False)),
        (3, 4, 2, 1, 1, (3, True)),
        # Sources without rows hold none.
        (0, 7, 100, 1, 1, (0, False)),
    ],
)
def test_a_merge_holds_the_most_rows_of_each_source_that_fit(
    shard_rows, part_rows, most, dense_writes, csr_writes, expected
):
    from cytearc.merge.writer import _held_windows

    asked: list[tuple[int, bool]] = []

    def writes(rows: int, sparse: bool) -> int:
        asked.append((rows, sparse))
        if sparse:
            return csr_writes
        return dense_writes if rows <= most else 0

    assert _held_windows(shard_rows, part_rows, writes) == expected
    # No window longer than a shard is asked about, and CSR windows only of
    # a whole shard, once.
    assert all(0 <= rows <= shard_rows for rows, _ in asked)
    assert [rows for rows, sparse in asked if sparse] == (
        [shard_rows] if shard_rows else []
    )


@pytest.mark.parametrize(
    "shuffled",
    [{"a": True, "b": True}, {"a": True, "b": False}, {"a": False, "b": False}],
    ids=["both_shuffled", "b_in_file_order", "both_in_file_order"],
)
def test_contiguous_slices_of_a_merge_of_imports_of_distinct_seeds_are_random_samples(
    tmp_path, shuffled
):
    from scipy import stats

    from tests.storage_helpers import (
        assert_random_sample_order,
        grouped_file_rows,
        import_grouped_cells,
        open_store,
    )

    sources = import_grouped_cells(tmp_path, shuffled=shuffled)
    path = tmp_path / "merged.zarr"
    DataStoreMerge(
        [sources["a"], sources["b"]], str(path), ["a", "b"], nthreads=1, seed=7
    ).dump()
    merged = open_store(path)
    ids = merged.cells.fetch_all("ids").tolist()

    # Each merged cell's counts are those of its cell: its row in its file,
    # plus one, and 1.
    counts = np.asarray(merged.RNA.matrixGroup["counts"][:])
    np.testing.assert_array_equal(counts[:, 0], grouped_file_rows(ids) + 1)
    np.testing.assert_array_equal(counts[:, 1], 1)
    if all(shuffled.values()):
        assert_random_sample_order(ids)
        assert merged.cells._group.attrs["cellOrder"] == {"shuffled": True, "seed": 7}
        return
    # A source kept in its file's order keeps that order in the merge from a
    # random first cell, wrapping round, so slices hold runs of its file
    # rows, and the merge records no shuffle; the sources still take turns
    # at random.
    assert "cellOrder" not in merged.cells._group.attrs
    rows = grouped_file_rows(ids)
    for name, is_shuffled in shuffled.items():
        positions = np.flatnonzero(
            [identity.startswith(f"{name}__") for identity in ids]
        )
        if is_shuffled:
            rho = stats.spearmanr(positions, rows[positions]).statistic
            assert abs(rho) < 4.5 / np.sqrt(positions.size - 1)
        else:
            given = rows[positions]
            first = int(np.argmin(given))
            assert first > 0
            np.testing.assert_array_equal(np.roll(given, -first), np.arange(given.size))


def test_a_merge_of_imports_that_share_a_seed_takes_random_slices_and_records_no_shuffle(
    tmp_path,
):
    from tests.storage_helpers import (
        assert_random_sample_order,
        import_grouped_cells,
        open_store,
    )

    # Both imports shuffle with the default seed, so each lists a file row at
    # about the same place; the merge starts each at a random cell, so its
    # slices pass the laws of a random order, but it does not claim one.
    sources = import_grouped_cells(
        tmp_path, shuffled={"a": True, "b": True}, seeds={"a": 0, "b": 0}
    )
    path = tmp_path / "merged.zarr"
    DataStoreMerge(
        [sources["a"], sources["b"]], str(path), ["a", "b"], nthreads=1, seed=7
    ).dump()
    merged = open_store(path)
    assert_random_sample_order(merged.cells.fetch_all("ids").tolist())
    assert "cellOrder" not in merged.cells._group.attrs


def test_a_merge_records_a_shuffle_beside_a_source_without_cells(tmp_path):
    from tests.storage_helpers import (
        import_grouped_cells,
        open_store,
        write_count_store,
    )

    # A source that gives no cells adds none to the order, though it records
    # no shuffle.
    sources = import_grouped_cells(tmp_path, shuffled={"a": True, "b": True})
    empty = tmp_path / "empty.zarr"
    write_count_store(
        str(empty),
        {"RNA": np.zeros((0, 2), dtype=np.uint32)},
        "uint32",
        feature_ids={"RNA": ["f0", "f1"]},
    )
    path = tmp_path / "merged.zarr"
    DataStoreMerge(
        [sources["a"], open_store(empty)], str(path), ["a", "e"], nthreads=1, seed=9
    ).dump()
    assert open_store(path).cells._group.attrs["cellOrder"] == {
        "shuffled": True,
        "seed": 9,
    }


def test_a_merge_records_the_shuffle_of_its_sources_only_when_each_records_one(
    tmp_path,
):
    from tests.storage_helpers import import_grouped_cells, open_store

    sources = import_grouped_cells(tmp_path, shuffled={"a": True, "b": True})
    for name, seed in (("first", None), ("second", 2**64 - 1)):
        path = tmp_path / f"{name}.zarr"
        DataStoreMerge(
            [sources["a"], sources["b"]], str(path), ["a", "b"], nthreads=1, seed=seed
        ).dump()
        record = open_store(path).cells._group.attrs["cellOrder"]
        assert set(record) == {"shuffled", "seed"} and record["shuffled"] is True
        if seed is None:
            # A merge of no seed records the seed that it drew.
            assert 0 <= record["seed"] < 2**64
        else:
            assert record["seed"] == seed
    # A merge of a merge that records a shuffle records one too.
    remerged = tmp_path / "remerged.zarr"
    DataStoreMerge(
        [open_store(tmp_path / "first.zarr"), sources["b"]],
        str(remerged),
        ["m", "b"],
        nthreads=1,
        seed=3,
    ).dump()
    assert open_store(remerged).cells._group.attrs["cellOrder"] == {
        "shuffled": True,
        "seed": 3,
    }
    # A source whose record is malformed is refused before anything is
    # written, as subsets refuse it.
    sources["b"].cells._group.attrs["cellOrder"] = {"shuffled": True}
    with pytest.raises(ValueError, match="cellOrder attribute is malformed"):
        DataStoreMerge(
            [sources["a"], sources["b"]],
            str(tmp_path / "refused.zarr"),
            ["a", "b"],
            nthreads=1,
        ).plan()
    assert not (tmp_path / "refused.zarr").exists()


@pytest.mark.parametrize(
    ("seed", "error", "message"),
    [
        (-1, ValueError, re.escape("seed must be from 0 to 2**64 - 1, not -1")),
        (
            2**64,
            ValueError,
            re.escape(f"seed must be from 0 to 2**64 - 1, not {2**64}"),
        ),
        (True, TypeError, "^seed must be an integer$"),
        ("1", TypeError, "^seed must be an integer$"),
    ],
)
def test_a_merge_refuses_a_seed_outside_its_range(tmp_path, seed, error, message):
    sources = [
        _MergeDataStore(
            [_MergeAssay("RNA", np.ones((2, 1)), ["c0", "c1"], ["g"], ["G"], 2)],
            zarr_loc=f"memory://{name}",
        )
        for name in ("left", "right")
    ]
    with pytest.raises(error, match=message):
        DataStoreMerge(sources, str(tmp_path / "merged.zarr"), ["l", "r"], seed=seed)
    assert not (tmp_path / "merged.zarr").exists()


def test_merges_of_one_seed_write_the_same_store_and_refuse_a_store_of_another_order(
    tmp_path, monkeypatch
):
    import cytearc.merge.datasets as datasets
    from cytearc.merge.row_plan import RowPlan

    stores = [
        _open_fixture_store(tmp_path, name, rows)
        for name, rows in (("a", 13), ("b", 11))
    ]
    first, second = tmp_path / "first.zarr", tmp_path / "second.zarr"
    for path in (first, second):
        DataStoreMerge(stores, str(path), ["a", "b"], nthreads=1, seed=5).dump()
    assert _store_files(first) == _store_files(second)

    # A merge interrupted after it wrote its cells in another order, as an
    # earlier release ordered them, cannot resume: its cell table no longer
    # matches the plan's IDs.
    build = datasets.build_row_plan

    def reversed_plan(*args, **kwargs) -> RowPlan:
        plan = build(*args, **kwargs)
        return RowPlan(
            owners=plan.owners[::-1].copy(),
            selectedRows=plan.selectedRows,
            sourceCells=plan.sourceCells,
            firstCells=plan.firstCells,
            nCells=plan.nCells,
            seed=plan.seed,
            sourceNames=plan.sourceNames,
        )

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted")

    interrupted = tmp_path / "interrupted.zarr"
    with monkeypatch.context() as patched:
        patched.setattr(datasets, "build_row_plan", reversed_plan)
        patched.setattr(datasets, "write_assay_counts", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            DataStoreMerge(
                stores, str(interrupted), ["a", "b"], nthreads=1, seed=5
            ).dump()
    resumed = DataStoreMerge(stores, str(interrupted), ["a", "b"], nthreads=1, seed=5)
    plan = resumed.plan()
    assert not plan.canDump
    assert plan.blockedReason == (
        "Completed cellData cannot be reused: its cell IDs or their order differ "
        "from this merge's, as when an earlier CyteArc release wrote it in blocks "
        "of one source or a source has changed. Merge into a fresh destination; "
        "overwrite=True rebuilds this one only until a DataStore opens it."
    )
    DataStoreMerge(
        stores, str(interrupted), ["a", "b"], nthreads=1, seed=5, overwrite=True
    ).dump()
    assert _store_files(interrupted) == _store_files(first)
    # A completed merge of the earlier order is refused alike, and once a
    # DataStore has opened it, overwrite=True is refused too, so the merge
    # goes to a fresh destination.
    completed = tmp_path / "completed.zarr"
    with monkeypatch.context() as patched:
        patched.setattr(datasets, "build_row_plan", reversed_plan)
        DataStoreMerge(stores, str(completed), ["a", "b"], nthreads=1, seed=5).dump()
    from tests.storage_helpers import open_store

    open_store(completed)
    assert (
        DataStoreMerge(stores, str(completed), ["a", "b"], nthreads=1, seed=5)
        .plan()
        .blockedReason
        == plan.blockedReason
    )
    refused = DataStoreMerge(
        stores, str(completed), ["a", "b"], nthreads=1, seed=5, overwrite=True
    ).plan()
    assert not refused.canDump
    assert "prepared datasets require a fresh destination" in refused.blockedReason
    fresh = tmp_path / "fresh.zarr"
    DataStoreMerge(stores, str(fresh), ["a", "b"], nthreads=1, seed=5).dump()
    assert _store_files(fresh) == _store_files(first)


def _open_fixture_store(tmp_path, name: str, n_cells: int):
    """Write and open a store of ``n_cells`` cells by 4 genes whose counts identify each cell."""
    from tests.storage_helpers import open_store, write_count_store

    counts = np.zeros((n_cells, 4), dtype=np.uint16)
    counts[:, 0] = np.arange(1, n_cells + 1)
    counts[:, 1] = 7
    location = tmp_path / f"{name}.zarr"
    write_count_store(str(location), {"RNA": counts}, "uint16")
    return open_store(location)


def test_a_merge_write_refuses_a_plan_that_leaves_cells_of_a_source_out(
    tmp_path, monkeypatch
):
    import cytearc.merge.datasets as datasets

    # A merge writes every cell of each source; a plan of selected rows, as a
    # composite makes, would read other rows than its windows name.
    build = datasets.build_row_plan
    monkeypatch.setattr(
        datasets,
        "build_row_plan",
        lambda cells, names, **options: build(
            cells, names, selected_rows=[np.arange(cells[0]), None], **options
        ),
    )
    stores = [
        _open_fixture_store(tmp_path, name, rows) for name, rows in (("a", 5), ("b", 4))
    ]
    with pytest.raises(ValueError, match="^A merge writes every cell of each source$"):
        DataStoreMerge(
            stores, str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
        ).dump()
