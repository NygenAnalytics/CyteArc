"""Behavior of the 10x HDF5, Matrix Market, and CSV readers at their edges."""

import csv
import gzip
import re
import shutil
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.storage import MemoryStore

from cytearc import DataStore
from cytearc.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    MtxReader,
    inspect_mtx,
)
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.utils.count_values import CountValueRange, compressed_count_ranges
from cytearc.writers import CrToZarr, CSVtoZarr, MtxToZarr

_POLICY = CountMatrixPolicy(unitBytes=16, chunkBytes=8)
_BANNER = "%%MatrixMarket matrix coordinate integer general\n"


def _write_h5(
    path: Path,
    values: np.ndarray,
    *,
    legacy: bool = False,
    data_dtype: str | None = None,
) -> None:
    matrix = csr_matrix(values)
    n_cells, n_features = values.shape
    data = matrix.data if data_dtype is None else matrix.data.astype(data_dtype)
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("genome" if legacy else "matrix")
        group.create_dataset("data", data=data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset(
            "barcodes", data=np.array([f"cell-{i}".encode() for i in range(n_cells)])
        )
        ids = np.array([f"feature-{i}".encode() for i in range(n_features)])
        names = np.array([f"gene-{i}".encode() for i in range(n_features)])
        if legacy:
            group.create_dataset("genes", data=ids)
            group.create_dataset("gene_names", data=names)
            return
        features = group.create_group("features")
        features.create_dataset("id", data=ids)
        features.create_dataset("name", data=names)
        features.create_dataset(
            "feature_type", data=np.array([b"Gene Expression"] * n_features)
        )


def _write_mex(
    directory: Path,
    entries: str,
    *,
    n_features: int,
    n_cells: int,
    n_entries: int | None = None,
    matrix_name: str = "matrix.mtx",
) -> None:
    """Write a 10x MEX triplet whose matrix holds ``entries``, one per line."""
    lines = [line for line in entries.splitlines() if line]
    declared = len(lines) if n_entries is None else n_entries
    (directory / matrix_name).write_text(
        f"{_BANNER}{n_features} {n_cells} {declared}\n{entries}"
    )
    (directory / "features.tsv").write_text(
        "".join(f"feature-{i}\tgene-{i}\tGene Expression\n" for i in range(n_features))
    )
    (directory / "barcodes.tsv").write_text(
        "".join(f"cell-{i}\n" for i in range(n_cells))
    )


def _write_parse(directory: Path, matrix: str) -> None:
    (directory / "DGE.mtx").write_text(matrix)
    (directory / "genes.csv").write_text(
        "gene_id,gene_name\nfeature-0,Gene 0\nfeature-1,Gene 1\nfeature-2,Gene 2\n"
    )
    (directory / "cell_metadata.csv").write_text(
        "bc_wells,sample\ncell-0,A\ncell-1,B\n"
    )


def _counts(store: MemoryStore, assay: str = "RNA") -> np.ndarray:
    return np.asarray(zarr.open_group(store=store, mode="r")[f"{assay}/counts"][:])


class _MemoryReader(CrReader):
    """A reader that implements only the abstract members of ``CrReader``."""

    def __init__(self, values: np.ndarray) -> None:
        self._matrix = csr_matrix(values)
        super().__init__(self._handle_version())

    def _handle_version(self) -> dict[str, str | None]:
        return {
            "feature_ids": "ids",
            "feature_names": "names",
            "feature_types": None,
            "cell_names": "cells",
        }

    def _read_dataset(self, key: str) -> list[str]:
        n_cells, n_features = self._matrix.shape
        return {
            "feature_ids": [f"feature-{i}" for i in range(n_features)],
            "feature_names": [f"gene-{i}" for i in range(n_features)],
            "cell_names": [f"cell-{i}" for i in range(n_cells)],
        }[key]

    @property
    def matrix_dtype(self) -> np.dtype:
        return self._matrix.dtype

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        return compressed_count_ranges(
            self._matrix.indptr,
            self._matrix.indices,
            self._matrix.data,
            minorSize=self._matrix.shape[1],
            maxBytes=maxBytes,
            groups=featureGroups,
        )

    def consume(self, batch_size: int, lines_in_mem: int):
        for start in range(0, self._matrix.shape[0], batch_size):
            yield self._matrix[start : start + batch_size].tocoo()


def test_crreader_subclass_imports_with_the_base_planning_defaults() -> None:
    values = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int64)
    reader = _MemoryReader(values)

    # The base bounds a window by its dense size and reserves no staging.
    assert reader.max_window_nnz(2) == 2 * 3
    assert reader.max_window_nnz(10) == 3 * 3
    with pytest.raises(ValueError, match="window_rows must be positive"):
        reader.max_window_nnz(0)
    assert reader.producer_staging_bytes(2, 1) == 0
    assert list(reader.assayFeats.columns) == ["RNA"]

    store = MemoryStore()
    CrToZarr(
        reader, store, mem_budget="64M", nthreads=1, policy=_POLICY, shuffle_cells=False
    ).dump()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.uint8
    np.testing.assert_array_equal(counts[:], values)


def _write_mixed_mex(directory: Path) -> None:
    (directory / "features.tsv").write_text(
        "f1\tg1\tGene Expression\nf2\th1\tAntibody Capture\nf3\tg2\tGene Expression\n"
    )
    (directory / "barcodes.tsv").write_text("c1\n")
    (directory / "matrix.mtx").write_text(f"{_BANNER}3 1 2\n1 1 2\n2 1 4\n")


@pytest.mark.parametrize(
    ("indexes", "feature_type", "error", "message"),
    [
        ([1], "", ValueError, "non-empty string"),
        ([1], 7, ValueError, "non-empty string"),
        ("1", "HTO", TypeError, "sequence of integer"),
        ([[1]], "HTO", ValueError, "one-dimensional"),
        ([], "HTO", ValueError, "at least one"),
        ([1.0], "HTO", TypeError, "only integers"),
    ],
)
def test_reclassify_features_rejects_malformed_requests(
    tmp_path: Path,
    indexes: object,
    feature_type: object,
    error: type[Exception],
    message: str,
) -> None:
    _write_mixed_mex(tmp_path)
    reader = CrDirReader(str(tmp_path))
    try:
        before = reader.assayFeats.copy()
        with pytest.raises(error, match=message):
            reader.reclassify_features(indexes, feature_type)  # type: ignore[arg-type]
        assert reader.assayFeats.equals(before)
    finally:
        reader.close()


def test_reclassify_features_without_a_required_previous_type(tmp_path: Path) -> None:
    _write_mixed_mex(tmp_path)
    reader = CrDirReader(str(tmp_path))
    try:
        reader.reclassify_features([0], "Multiplexing Capture", require_previous=None)

        assert reader.feature_types() == [
            "Multiplexing Capture",
            "Antibody Capture",
            "Gene Expression",
        ]
        assert list(reader.assayFeats.columns) == ["HTO", "ADT", "RNA"]
    finally:
        reader.close()


def test_unfiltered_cellranger_h5_rejects_pointers_past_its_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "counts.h5"
    _write_h5(path, np.array([[1, 2], [3, 0]], dtype=np.uint16))
    with h5py.File(path, mode="a") as handle:
        indptr = handle["matrix/indptr"]
        indptr[-1] = int(indptr[-1]) + 1

    with pytest.raises(ValueError, match="pointers do not match its data"):
        CrH5Reader(str(path), is_filtered=False, filtering_cutoff=0)
    # The failed reader closed the file, so it opens for writing.
    h5py.File(path, mode="r+").close()


def test_cellranger_h5_rejects_a_negative_filtering_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5"
    _write_h5(path, np.array([[1, 2], [0, 0]], dtype=np.uint16))

    # MtxReader rejected it, while this reader kept every barcode.
    with pytest.raises(ValueError, match="filtering_cutoff cannot be negative"):
        CrH5Reader(str(path), is_filtered=False, filtering_cutoff=-1)


def test_cellranger_h5_feature_tags_skip_reserved_absent_and_group_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tagged.h5"
    _write_h5(path, np.array([[1, 0], [0, 2]], dtype=np.uint8))
    with h5py.File(path, mode="a") as handle:
        features = handle["matrix/features"]
        features.create_dataset(
            "_all_tag_keys", data=np.array([b"id", b"absent", b"nested", b"genome"])
        )
        features.create_group("nested")
        features.create_dataset("genome", data=np.array([b"GRCh38", b"GRCh38"]))
    reader = CrH5Reader(str(path))
    try:
        columns = dict(reader.get_feature_columns())
    finally:
        reader.close()

    assert list(columns) == ["feature_type", "genome"]
    np.testing.assert_array_equal(columns["genome"].astype(str), ["GRCh38"] * 2)


def test_cellranger_h5_rejects_a_feature_tag_of_another_length(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tagged.h5"
    _write_h5(path, np.array([[1, 0], [0, 2]], dtype=np.uint8))
    with h5py.File(path, mode="a") as handle:
        features = handle["matrix/features"]
        features.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
        features.create_dataset("genome", data=np.array([b"GRCh38"] * 3))
    reader = CrH5Reader(str(path))
    try:
        with pytest.raises(ValueError, match="'genome' has shape \\(3,\\)"):
            list(reader.get_feature_columns())
    finally:
        reader.close()


def test_cellranger2_h5_reports_only_feature_types(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    _write_h5(path, np.array([[1, 0], [0, 2]], dtype=np.uint8), legacy=True)
    reader = CrH5Reader(str(path))
    try:
        columns = list(reader.get_feature_columns())
        assert reader.feature_names() == ["gene-0", "gene-1"]
    finally:
        reader.close()

    assert [name for name, _values in columns] == ["feature_type"]
    np.testing.assert_array_equal(columns[0][1], ["Gene Expression"] * 2)


def test_cellranger_h5_yields_empty_batches_for_barcodes_without_entries(
    tmp_path: Path,
) -> None:
    values = np.array([[1, 0], [0, 0], [0, 0], [0, 3]], dtype=np.uint16)
    path = tmp_path / "counts.h5"
    _write_h5(path, values)
    reader = CrH5Reader(str(path))
    try:
        single = list(reader.consume(1))
        paired = list(reader.consume(2))
    finally:
        reader.close()

    assert [batch.nnz for batch in single] == [1, 0, 0, 1]
    assert all(isinstance(batch, coo_matrix) for batch in single)
    assert all(batch.dtype == np.uint16 for batch in single + paired)
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in single]), values
    )
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in paired]), values
    )


@pytest.mark.parametrize(
    ("data_dtype", "values", "read_dtype", "stored_dtype"),
    [
        ("<f2", [[1, 0], [0, 2]], np.float32, np.uint8),
        ("<f2", [[1.5, 0], [0, 2]], np.float32, np.float32),
        (">i4", [[1, 0], [0, 300]], np.int32, np.uint16),
        (">f8", [[1.5, 0], [0, 2]], np.float64, np.float64),
    ],
    ids=["float16-integral", "float16-fraction", "big-endian-int", "big-endian-float"],
)
def test_cellranger_h5_reads_float16_and_big_endian_counts_natively(
    tmp_path: Path,
    data_dtype: str,
    values: list[list[float]],
    read_dtype: type,
    stored_dtype: type,
) -> None:
    path = tmp_path / "counts.h5"
    _write_h5(path, np.array(values, dtype=np.float64), data_dtype=data_dtype)
    reader = CrH5Reader(str(path))
    store = MemoryStore()
    try:
        # SciPy sparse matrices hold neither dtype, so these imports used to
        # fail while writing, after the destination was created.
        assert reader.matrix_dtype == read_dtype
        # The fractional values are counts too, so they are declared so.
        CrToZarr(
            reader,
            store,
            mem_budget="64M",
            nthreads=1,
            policy=_POLICY,
            values="counts",
        ).dump()
    finally:
        reader.close()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == stored_dtype
    np.testing.assert_array_equal(counts[:], values)


@pytest.mark.parametrize(
    ("text", "kwargs"),
    [
        ("id\nc1\nc2\n", {"id_column": 0}),
        ("id,batch\nc1,x\nc2,y\n", {"id_column": 0, "cell_data_cols": ["batch"]}),
        ("a,b\n1,2\n3,4\n", {"skip_cols": ["a", "b"]}),
    ],
    ids=["id-only", "metadata-only", "skipped-only"],
)
def test_csv_without_count_columns_is_rejected(
    tmp_path: Path, text: str, kwargs: dict[str, object]
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(text)

    # Such a file used to import an assay without features, which no
    # DataStore could open.
    with pytest.raises(ValueError, match="no count columns"):
        CSVReader(str(path), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [{"skip_cols": ["a"]}, {"cell_data_cols": ["a"]}],
    ids=["skip_cols", "cell_data_cols"],
)
def test_csv_named_columns_require_a_header(
    tmp_path: Path, kwargs: dict[str, list[str]]
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("1,2\n3,4\n")

    with pytest.raises(ValueError, match="Named columns require a CSV header"):
        CSVReader(str(path), has_header=False, **kwargs)  # type: ignore[arg-type]


def test_csv_without_a_header_names_cells_and_features_by_position(
    tmp_path: Path,
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("1,0,2\n0,3,0\n")
    reader = CSVReader(str(path), has_header=False)

    assert reader.feature_ids().tolist() == ["feature_0", "feature_1", "feature_2"]
    assert reader.cell_ids().tolist() == ["cell_0", "cell_1"]
    store = MemoryStore()
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1, policy=_POLICY).dump()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], [[1, 0, 2], [0, 3, 0]])
    np.testing.assert_array_equal(
        root["RNA/featureData/ids"][:], ["feature_0", "feature_1", "feature_2"]
    )


def test_csv_fraction_in_an_early_chunk_keeps_float_storage(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("a,b\n0.5,1\n2,3\n4,5\n")
    reader = CSVReader(str(path), batch_size=1)

    assert reader.countRange.integral is False
    assert reader.countDtype == np.float64
    store = MemoryStore()
    # The fraction is a count here, so it is declared so.
    CSVtoZarr(
        reader,
        store,
        assay_name="RNA",
        nthreads=1,
        policy=_POLICY,
        values="counts",
        shuffle_cells=False,
    ).dump()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.float64
    np.testing.assert_array_equal(counts[:], [[0.5, 1], [2, 3], [4, 5]])


def test_csv_reader_leaves_the_callers_pandas_kwargs_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("a, b\n1, 2\n")
    pandas_kwargs = {"skipinitialspace": True}

    reader = CSVReader(str(path), batch_size=5, pandas_kwargs=pandas_kwargs)

    # The reader used to add its read_csv settings, such as chunksize, here.
    assert pandas_kwargs == {"skipinitialspace": True}
    assert reader.pandas_kwargs["chunksize"] == 5
    assert reader.feature_ids().tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1,2,x\n4,5,6,7\n", "line 3 has 4 fields, but line 1 has 3"),
        ("1,2,x\n4,5,6,\n", "line 3 has 4 fields, but line 1 has 3"),
        ("1,2,x\n4,5\n", "line 3 has 2 fields, but line 1 has 3"),
        ("c1,1,2,x\nc2,3,4,y\n", "line 2 has 4 fields, but line 1 has 3"),
    ],
    ids=["extra-field", "extra-empty-field", "missing-field", "unnamed-row-names"],
)
def test_csv_rows_without_the_header_field_count_are_rejected(
    tmp_path: Path, rows: str, message: str
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(f"a,b,note\n{rows}")

    # pandas dropped the extra fields of a row that starts a chunk, padded a
    # row without its skipped field, and took row names that the header does
    # not name as its index, so each of these files imported without an error.
    with pytest.raises(ValueError, match=message):
        CSVReader(str(path), skip_cols=["note"], batch_size=1)


@pytest.mark.parametrize(
    ("name", "text", "options"),
    [
        ("counts.csv", 'a,note,b\n1,"x,y",2\n3,"two\nlines",4\n', {}),
        (
            "counts.csv",
            "a,note,b\n1,'x,y',2\n3,z,4\n",
            {"pandas_kwargs": {"quotechar": "'"}},
        ),
        (
            "counts.csv",
            'a,note,b\n1,"x,2\n3,y,4\n',
            {"pandas_kwargs": {"quoting": csv.QUOTE_NONE}},
        ),
        (
            "counts.csv",
            "a,note,b\n1,x\\,y,2\n3,z,4\n",
            {"pandas_kwargs": {"escapechar": "\\"}},
        ),
        (
            "counts.csv",
            'a,note,b\n1, "x,y",2\n3,z,4\n',
            {"pandas_kwargs": {"skipinitialspace": True}},
        ),
        ("counts.csv", '\ufeff"a,1",note,b\n1,x,2\n3,y,4\n', {}),
        ("counts.csv", "a,note,b\n1,x,2\n\n \t \n3,y,4\n", {}),
        ("counts.csv", '"skipped\nrow"\na,note,b\n1,x,2\n3,y,4\n', {"skip_rows": 1}),
        ("counts.csv.gz", "a,note,b\n1,x,2\n3,y,4\n", {}),
    ],
    ids=[
        "quoted",
        "quotechar",
        "quote-none",
        "escapechar",
        "skipinitialspace",
        "byte-order-mark",
        "blank-lines",
        "skip-rows",
        "gzip",
    ],
)
def test_csv_field_check_splits_rows_as_pandas_does(
    tmp_path: Path, name: str, text: str, options: dict[str, object]
) -> None:
    path = tmp_path / name
    data = text.encode()
    path.write_bytes(gzip.compress(data) if name.endswith(".gz") else data)

    # A field check that split rows otherwise than pandas, or that read other
    # text than pandas, would reject each of these files.
    reader = CSVReader(str(path), skip_cols=["note"], batch_size=1, **options)  # type: ignore[arg-type]

    counts = np.vstack([batch for batch, _metadata in reader.consume()])
    np.testing.assert_array_equal(counts, [[1, 2], [3, 4]])


@pytest.mark.parametrize(
    ("text", "options", "message"),
    [
        ("a\tb\n1\t2\n", {"sep": r"\s+"}, "sep must be one character"),
        (
            "a\tb\n1\t2\n",
            {"pandas_kwargs": {"dialect": "excel-tab"}},
            "pandas_kwargs cannot set dialect:",
        ),
        (
            "a,b\n1,2\n",
            {"pandas_kwargs": {"lineterminator": "\n", "comment": "#"}},
            "pandas_kwargs cannot set comment, lineterminator:",
        ),
    ],
    ids=["regex-separator", "dialect", "comment-and-lineterminator"],
)
def test_csv_reader_rejects_settings_that_its_field_check_cannot_follow(
    tmp_path: Path, text: str, options: dict[str, object], message: str
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(text)

    # pandas reads each file under its setting, but the field check cannot
    # split rows as the setting does.
    with pytest.raises(ValueError, match=message):
        CSVReader(str(path), **options)  # type: ignore[arg-type]


_TWO_ROW_CSV = "a,b\n1,2\n3,4\n"
_GENERATED_CELL_IDS = ["cell_0", "cell_1"]


@pytest.mark.parametrize(
    ("text", "options", "features", "cells"),
    [
        pytest.param(_TWO_ROW_CSV, {}, ["a", "b"], _GENERATED_CELL_IDS, id="header"),
        pytest.param(
            "cell,10,20\n001,1,2\n002,3,4\n",
            {"id_column": 0},
            ["10", "20"],
            ["001", "002"],
            id="header-of-numbers-and-id-column",
        ),
        pytest.param(
            "cell,g,g,\nc1,1,2,3\nc2,3,4,5\n",
            {"id_column": 0},
            ["g", "g.1", "Unnamed: 3"],
            ["c1", "c2"],
            id="header-repeated-and-blank",
        ),
        pytest.param(
            "cell,géne,基因\ncafé,1,2\n中文,3,4\n",
            {"id_column": 0},
            ["géne", "基因"],
            ["café", "中文"],
            id="non-ascii",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\nc2,3,4\n",
            {"id_column": 0, "pandas_kwargs": {"names": ["cell", 10, 20]}},
            ["10", "20"],
            ["c1", "c2"],
            id="names-integers-and-id-column",
        ),
        pytest.param(
            _TWO_ROW_CSV,
            {"pandas_kwargs": {"names": np.array([2**63 - 1, -5])}},
            ["9223372036854775807", "-5"],
            _GENERATED_CELL_IDS,
            id="names-numpy-integers",
        ),
        pytest.param(
            "a,b,c\n1,2,3\n4,5,6\n",
            {"pandas_kwargs": {"names": [1.5, 2.0, 1e16]}},
            ["1.5", "2.0", "1e+16"],
            _GENERATED_CELL_IDS,
            id="names-floats",
        ),
        pytest.param(
            _TWO_ROW_CSV,
            {"pandas_kwargs": {"names": np.array([1.5, 2.1], dtype=np.float32)}},
            ["1.5", "2.1"],
            _GENERATED_CELL_IDS,
            id="names-float32",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\nc2,3,4\n",
            {"id_column": 0, "pandas_kwargs": {"names": ["cell", "g1", 20]}},
            ["g1", "20"],
            ["c1", "c2"],
            id="names-text-and-number",
        ),
        pytest.param(
            _TWO_ROW_CSV,
            {"pandas_kwargs": {"names": [b"x", "é".encode()]}},
            ["x", "é"],
            _GENERATED_CELL_IDS,
            id="names-utf8-bytes",
        ),
        # The skipped column is named as pandas holds it, the number 20.
        pytest.param(
            "cell,a,b,c\nc1,1,2,3\nc2,4,5,6\n",
            {
                "id_column": 0,
                "pandas_kwargs": {"names": ["cell", 10, 20, 30]},
                "skip_cols": [20],
            },
            ["10", "30"],
            ["c1", "c2"],
            id="names-integers-and-a-skipped-column",
        ),
    ],
)
def test_csv_reader_ids_are_the_text_that_an_import_stores(
    tmp_path: Path,
    text: str,
    options: dict[str, object],
    features: list[str],
    cells: list[str],
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(text, encoding="utf-8")
    reader = CSVReader(str(path), batch_size=1, **options)  # type: ignore[arg-type]
    read = {"RNA/featureData": reader.feature_ids(), "cellData": reader.cell_ids()}

    store = MemoryStore()
    CSVtoZarr(
        reader, store, assay_name="RNA", nthreads=1, policy=_POLICY, shuffle_cells=False
    ).dump()

    root = zarr.open_group(store=store, mode="r")
    for table, expected in (("RNA/featureData", features), ("cellData", cells)):
        # The reader used to give the names in pandas_kwargs as pandas holds
        # them, such as 10 or b"x", where the store holds "10" or "x".
        assert read[table].dtype.kind == "U", table
        assert read[table].tolist() == expected, table
        # The import stores what it stored before: the text of each name, as
        # wide as the widest.
        width = max(len(value) for value in expected)
        for column in ("ids", "names"):
            stored = root[f"{table}/{column}"]
            assert stored.dtype == np.dtype(f"U{width}"), (table, column)
            assert stored[:].tolist() == expected, (table, column)


def test_values_keyed_by_the_csv_reader_ids_insert_into_the_store(
    tmp_path: Path,
) -> None:
    # Features named by numbers in pandas_kwargs came back as those numbers,
    # 10, 20 and 30, where the store holds "10", "20" and "30", so values
    # keyed by the reader's feature IDs were refused.
    path = tmp_path / "counts.csv"
    path.write_text(
        "cell,a,b,c\n001,1,0,2\n002,0,3,0\n003,4,5,6\n004,7,0,8\n005,9,1,0\n"
    )
    reader = CSVReader(
        str(path),
        id_column=0,
        batch_size=2,
        pandas_kwargs={"names": ["cell", 10, 20, 30]},
    )
    feature_ids = reader.feature_ids()
    cell_ids = reader.cell_ids()
    store = MemoryStore()
    # The import shuffles the cells, as every import does by default.
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1, policy=_POLICY).dump()
    ds = DataStore(store, min_features_per_cell=0, nthreads=1)
    rows = ds.cells.fetch_all("sourceRow")
    assert not np.array_equal(rows, np.arange(5))

    # The values are keyed in another order than the features'.
    ds.RNA.feats.insert(
        "weight", pd.Series([2.5, 0.5, 1.5], index=feature_ids[[2, 0, 1]])
    )
    ds.cells.insert("score", pd.Series(np.arange(5.0), index=cell_ids))

    assert feature_ids.tolist() == ["10", "20", "30"]
    assert cell_ids.tolist() == ["001", "002", "003", "004", "005"]
    np.testing.assert_array_equal(ds.RNA.feats.fetch_all("weight"), [0.5, 1.5, 2.5])
    # Each cell gets the value of its row in the file, wherever it is stored.
    np.testing.assert_array_equal(ds.cells.fetch_all("score"), rows.astype(float))
    # Matching stays exact: the numbers that pandas holds name no feature.
    with pytest.raises(ValueError, match="3 labels are not such ids"):
        ds.RNA.feats.insert("raw", pd.Series([0.5, 1.5, 2.5], index=[10, 20, 30]))
    assert "raw" not in ds.RNA.feats.columns


def _csv_reader(tmp_path: Path, text: str, **options: object) -> CSVReader:
    """Write ``text`` as a CSV file and return a reader of it, one row a chunk."""
    path = tmp_path / "counts.csv"
    path.write_text(text, encoding="utf-8")
    return CSVReader(str(path), batch_size=1, **options)  # type: ignore[arg-type]


def _imported(reader: CSVReader) -> zarr.Group:
    """Import ``reader`` in the file's order and return the store's root."""
    store = MemoryStore()
    CSVtoZarr(
        reader, store, assay_name="RNA", nthreads=1, policy=_POLICY, shuffle_cells=False
    ).dump()
    return zarr.open_group(store=store, mode="r")


_NUMBERED_CELLS = "cell,a,b\n001,1,2\n002,3,4\n"
_FOUR_COLUMNS = "cell,a,b,c\nc1,1,2,3\nc2,4,5,6\n"


@pytest.mark.parametrize("engine", ["c", "python"])
@pytest.mark.parametrize(
    ("names", "features"),
    [
        pytest.param(["cell", 0, 1], ["0", "1"], id="a-count-column-named-0"),
        pytest.param([5, 0, 1], ["0", "1"], id="id-column-named-by-a-number"),
        pytest.param(["cell", True, False], ["True", "False"], id="booleans"),
        # This file imported before too, and must keep importing.
        pytest.param([1, "x", "y"], ["x", "y"], id="id-column-named-by-a-position"),
    ],
)
def test_csv_id_column_is_the_column_at_its_position_whatever_the_names(
    tmp_path: Path, engine: str, names: list[object], features: list[str]
) -> None:
    # pandas looked the ID column's converter up as a column name first, so
    # it also reached a count column named 0, or False, which then held text.
    reader = _csv_reader(
        tmp_path,
        _NUMBERED_CELLS,
        id_column=0,
        pandas_kwargs={"names": names, "engine": engine},
    )

    assert reader.cell_ids().tolist() == ["001", "002"]
    assert reader.feature_ids().tolist() == features
    root = _imported(reader)
    assert root["cellData/ids"][:].tolist() == ["001", "002"]
    np.testing.assert_array_equal(root["RNA/counts"][:], [[1, 2], [3, 4]])


@pytest.mark.parametrize("engine", ["c", "python"])
def test_csv_cell_ids_are_the_text_of_their_fields(tmp_path: Path, engine: str) -> None:
    # pandas read these IDs as missing values, which the reader refused as
    # blank.
    reader = _csv_reader(
        tmp_path,
        "id,batch,a\nNA,NA,1\nnull,x,2\nnan,,3\nNone,y,4\nN/A,z,5\n",
        id_column=0,
        cell_data_cols=["batch"],
        pandas_kwargs={"engine": engine},
    )

    ids = ["NA", "null", "nan", "None", "N/A"]
    assert reader.cell_ids().tolist() == ids
    root = _imported(reader)
    assert root["cellData/ids"][:].tolist() == ids
    # Other columns read missing values as they did.
    assert root["cellData/batch"][:].tolist() == ["", "x", "", "y", "z"]
    assert root["cellData/__cytearc_missing__batch"][:].tolist() == [
        True,
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("engine", ["c", "python"])
@pytest.mark.parametrize("cell_id", ["", "  "], ids=["empty", "spaces"])
def test_csv_blank_cell_ids_are_refused(
    tmp_path: Path, engine: str, cell_id: str
) -> None:
    with pytest.raises(ValueError, match="CSV cell IDs must contain non-empty values"):
        _csv_reader(
            tmp_path,
            f"id,a\nc1,1\n{cell_id},2\n",
            id_column=0,
            pandas_kwargs={"engine": engine},
        )


@pytest.mark.parametrize(
    ("text", "options", "features", "cells"),
    [
        pytest.param(
            "c1,1,2\nc2,3,4\n",
            {
                "has_header": False,
                "id_column": 0,
                "pandas_kwargs": {"names": ["cell", "x", "y"]},
            },
            ["x", "y"],
            ["c1", "c2"],
            id="every-column",
        ),
        pytest.param(
            "c1,1,2\nc2,3,4\n",
            {
                "has_header": False,
                "id_column": 0,
                "pandas_kwargs": {"names": ["x", "y"]},
            },
            ["x", "y"],
            ["c1", "c2"],
            id="every-column-after-the-id-column",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\nc2,3,4\n",
            {"id_column": 0, "pandas_kwargs": {"names": ["x", "y"]}},
            ["x", "y"],
            ["c1", "c2"],
            id="every-column-after-the-id-column-over-a-header",
        ),
        pytest.param(
            "1,2,3\n4,5,6\n",
            {
                "has_header": False,
                "skip_cols": ["b"],
                "pandas_kwargs": {"names": ["a", "b", "c"]},
            },
            ["a", "c"],
            _GENERATED_CELL_IDS,
            id="skipped-column-without-a-header",
        ),
    ],
)
def test_csv_names_name_the_columns_with_or_without_a_header(
    tmp_path: Path,
    text: str,
    options: dict[str, object],
    features: list[str],
    cells: list[str],
) -> None:
    # Without a header, the reader dropped names and numbered the features.
    reader = _csv_reader(tmp_path, text, **options)

    assert reader.feature_ids().tolist() == features
    assert reader.cell_ids().tolist() == cells
    root = _imported(reader)
    assert root["RNA/featureData/ids"][:].tolist() == features
    assert root["cellData/ids"][:].tolist() == cells


@pytest.mark.parametrize(
    ("text", "options", "error", "message"),
    [
        # pandas took the first column as an index, which the reader dropped.
        pytest.param(
            "cell,a,b\n9,1,2\n8,3,4\n",
            {"pandas_kwargs": {"names": ["x", "y"]}},
            ValueError,
            "names 2 columns, but the rows of the CSV file have 3 fields; name "
            "every column in the file's order",
            id="short-without-an-id-column",
        ),
        pytest.param(
            "9,1,2\n8,3,4\n",
            {
                "has_header": False,
                "id_column": 1,
                "pandas_kwargs": {"names": ["x", "y"]},
            },
            ValueError,
            "names 2 columns, but the rows of the CSV file have 3 fields",
            id="short-with-a-later-id-column",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\n",
            {"id_column": 0, "pandas_kwargs": {"names": ["c", "x", "y", "z"]}},
            ValueError,
            "names 4 columns, but the rows of the CSV file have 3 fields; name "
            "every column, or every column after the ID column, in",
            id="long",
        ),
        pytest.param(
            "1,2\n",
            {"has_header": False, "pandas_kwargs": {"names": "ab"}},
            TypeError,
            "must list the column names in order, not 'ab'",
            id="text",
        ),
    ],
)
def test_csv_reader_refuses_names_of_another_number_of_columns(
    tmp_path: Path,
    text: str,
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        _csv_reader(tmp_path, text, **options)


@pytest.mark.parametrize("skipped", [[20], ["20"]], ids=["label", "text"])
def test_csv_skip_cols_name_a_column_by_its_label_or_its_text(
    tmp_path: Path, skipped: list[object]
) -> None:
    reader = _csv_reader(
        tmp_path,
        _FOUR_COLUMNS,
        id_column=0,
        skip_cols=skipped,
        pandas_kwargs={"names": ["cell", 10, 20, 30]},
    )

    assert reader.feature_ids().tolist() == ["10", "30"]
    counts = np.vstack([batch for batch, _metadata in reader.consume()])
    np.testing.assert_array_equal(counts, [[1, 3], [4, 6]])


def test_csv_skip_cols_take_a_label_before_a_text(tmp_path: Path) -> None:
    # "10" is the label of one column and the text of another; it skipped the
    # column of that label before, and still does.
    reader = _csv_reader(
        tmp_path,
        _FOUR_COLUMNS,
        id_column=0,
        skip_cols=["10"],
        pandas_kwargs={"names": ["cell", 10, "10", 30]},
    )

    assert reader.feature_ids().tolist() == ["10", "30"]
    counts = np.vstack([batch for batch, _metadata in reader.consume()])
    np.testing.assert_array_equal(counts, [[1, 3], [4, 6]])


def test_csv_cell_data_cols_name_a_column_by_its_text(tmp_path: Path) -> None:
    # The text "30" named no column before.
    reader = _csv_reader(
        tmp_path,
        _FOUR_COLUMNS,
        id_column=0,
        cell_data_cols=["30"],
        pandas_kwargs={"names": ["cell", 10, 20, 30]},
    )

    assert reader.feature_ids().tolist() == ["10", "20"]
    root = _imported(reader)
    assert root["cellData/30"][:].tolist() == [3, 6]
    np.testing.assert_array_equal(root["RNA/counts"][:], [[1, 2], [4, 5]])


@pytest.mark.parametrize(
    ("names", "options", "error", "message"),
    [
        # The import failed on this name after the reader had read the file.
        pytest.param(
            ["cell", 10, 20, 30],
            {"cell_data_cols": [30]},
            ValueError,
            r"cell_data_cols names columns by \[30\], .* by their text, \['30'\]",
            id="cell-data-name-not-text",
        ),
        pytest.param(
            ["cell", 10, 20, 30],
            {"skip_cols": ["40"]},
            KeyError,
            r"skip_cols are not CSV columns: \['40'\]",
            id="name-of-no-column",
        ),
        pytest.param(
            ["cell", 10, b"10", 30],
            {"skip_cols": ["10"]},
            ValueError,
            r"skip_cols names '10', which is the text of the columns at "
            r"positions \[1, 2\]",
            id="text-of-two-columns",
        ),
    ],
)
def test_csv_reader_refuses_columns_named_by_no_column_or_two(
    tmp_path: Path,
    names: list[object],
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _csv_reader(
            tmp_path,
            _FOUR_COLUMNS,
            id_column=0,
            pandas_kwargs={"names": names},
            **options,
        )


@pytest.mark.parametrize(
    ("text", "options", "positions"),
    [
        pytest.param(
            _FOUR_COLUMNS,
            {"id_column": 0, "pandas_kwargs": {"names": ["cell", "", "b", "c"]}},
            "[1]",
            id="empty-name",
        ),
        pytest.param(
            _FOUR_COLUMNS,
            {"id_column": 0, "pandas_kwargs": {"names": ["cell", None, " ", "c"]}},
            "[1, 2]",
            id="missing-name-and-spaces",
        ),
        pytest.param(
            "1,2\n3,4\n",
            {"pandas_kwargs": {"names": [np.nan, "b"]}},
            "[0]",
            id="nan-name",
        ),
        pytest.param(
            "cell,  ,b\nc1,1,2\nc2,3,4\n",
            {"id_column": 0},
            "[1]",
            id="header-of-spaces",
        ),
        pytest.param(
            "a,cell,  \n1,c1,2\n3,c2,4\n",
            {"id_column": 1},
            "[2]",
            id="after-the-id-column",
        ),
    ],
)
def test_csv_reader_refuses_blank_or_missing_feature_names(
    tmp_path: Path, text: str, options: dict[str, object], positions: str
) -> None:
    # Such names imported as the feature IDs "", " " and "nan".
    message = (
        f"CSV feature IDs must contain non-empty values; the count columns at "
        f"positions {positions}, counted from 0, have blank or missing names"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _csv_reader(tmp_path, text, **options)


def test_csv_column_of_a_blank_name_imports_when_skipped(tmp_path: Path) -> None:
    reader = _csv_reader(
        tmp_path,
        _FOUR_COLUMNS,
        id_column=0,
        skip_cols=[""],
        pandas_kwargs={"names": ["cell", "", "b", "c"]},
    )

    assert reader.feature_ids().tolist() == ["b", "c"]
    np.testing.assert_array_equal(_imported(reader)["RNA/counts"][:], [[2, 3], [5, 6]])


def test_csv_reader_ids_are_copies(tmp_path: Path) -> None:
    reader = _csv_reader(tmp_path, "cell,a,b\nc1,1,2\nc2,3,4\n", id_column=0)

    # Changing the arrays that the methods return changes neither the
    # reader nor what an import stores.
    reader.cell_ids()[0] = "x"
    reader.feature_ids()[0] = "y"

    assert reader.cell_ids().tolist() == ["c1", "c2"]
    assert reader.feature_ids().tolist() == ["a", "b"]
    root = _imported(reader)
    assert root["cellData/ids"][:].tolist() == ["c1", "c2"]
    assert root["RNA/featureData/ids"][:].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("text", "options", "error", "message"),
    [
        # pandas counted the ID column among the columns it selected, so
        # usecols=[1, 2] took the cell IDs from a count column.
        pytest.param(
            "cell,a,b\nc1,1,2\n",
            {"id_column": 0, "pandas_kwargs": {"usecols": [1, 2]}},
            ValueError,
            "pandas_kwargs cannot set usecols",
            id="usecols",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\n",
            {"id_column": 3},
            ValueError,
            "id_column is 3, but the rows of the CSV file have 3 fields, at "
            "positions 0 to 2",
            id="id-column-past-the-last",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\n",
            {"id_column": -1},
            ValueError,
            "id_column counts columns from 0, so it cannot be -1",
            id="negative-id-column",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\n",
            {"id_column": "cell"},
            TypeError,
            "id_column must be None or the position of the ID column, an "
            "integer, not 'cell'",
            id="id-column-of-text",
        ),
        pytest.param(
            "cell,a,b\nc1,1,2\n",
            {"id_column": True},
            TypeError,
            "an integer, not True",
            id="boolean-id-column",
        ),
        pytest.param("", {}, ValueError, "CSV file contains no data rows", id="empty"),
    ],
)
def test_csv_reader_refuses_columns_it_cannot_number(
    tmp_path: Path,
    text: str,
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        _csv_reader(tmp_path, text, **options)


def test_matrix_names_without_the_matrix_suffix_use_plain_sidecars(
    tmp_path: Path,
) -> None:
    _write_mex(
        tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2, matrix_name="counts.mtx"
    )

    candidates = inspect_mtx(tmp_path)

    assert [Path(candidate.matrixPath).name for candidate in candidates] == [
        "counts.mtx"
    ]
    assert Path(candidates[0].featurePath).name == "features.tsv"
    assert Path(candidates[0].cellPath).name == "barcodes.tsv"


def test_parse_matrix_without_its_sidecars_is_not_a_candidate(tmp_path: Path) -> None:
    _write_mex(tmp_path, "1 1 3\n", n_features=1, n_cells=1)
    # A Parse matrix needs Parse gene and cell tables, not 10x sidecars.
    (tmp_path / "count_matrix.mtx").write_text(f"{_BANNER}1 1 1\n1 1 3\n")

    candidates = inspect_mtx(tmp_path)

    assert [Path(candidate.matrixPath).name for candidate in candidates] == [
        "matrix.mtx"
    ]


def test_parse_matrix_stored_features_by_cells_is_read_transposed(
    tmp_path: Path,
) -> None:
    _write_parse(tmp_path, f"{_BANNER}3 2 3\n1 1 4\n3 1 1\n2 2 7\n")
    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.matrixOrientation == "featuresByCells"
    assert (candidate.nCells, candidate.nFeatures) == (2, 3)

    with pytest.raises(KeyError, match="'barcode' was not found"):
        MtxReader(candidate, cell_id_key="barcode")
    reader = MtxReader(candidate)
    store = MemoryStore()
    try:
        MtxToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY).dump()
        assert reader.cell_names() == ["cell-0", "cell-1"]
    finally:
        reader.close()

    np.testing.assert_array_equal(_counts(store), [[4, 0, 1], [0, 7, 0]])


def test_missing_matrix_market_sources_are_rejected(tmp_path: Path) -> None:
    _write_mex(tmp_path, "1 1 3\n", n_features=1, n_cells=1)

    with pytest.raises(FileNotFoundError):
        inspect_mtx(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="missing.tsv"):
        MtxReader(
            str(tmp_path / "matrix.mtx"),
            str(tmp_path / "features.tsv"),
            str(tmp_path / "missing.tsv"),
        )


def test_mtx_reader_rejects_invalid_construction_arguments(tmp_path: Path) -> None:
    _write_mex(tmp_path, "1 1 3\n", n_features=1, n_cells=1)
    candidate = inspect_mtx(tmp_path)[0]

    with pytest.raises(ValueError, match="Do not pass explicit sidecars"):
        MtxReader(candidate, feature_path=str(tmp_path / "features.tsv"))
    # The reader extracted this file from a ZIP archive but never read it.
    with pytest.raises(TypeError, match="cell_metadata_path"):
        MtxReader(candidate, cell_metadata_path=str(tmp_path / "barcodes.tsv"))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="filtering_cutoff cannot be negative"):
        MtxReader(candidate, is_filtered=False, filtering_cutoff=-1)
    with pytest.raises(FileNotFoundError, match="missing"):
        MtxReader(candidate, temp_dir=str(tmp_path / "missing"))


def test_mtx_reader_planning_and_streaming_arguments_must_be_positive(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(reader.consume(0))
        with pytest.raises(ValueError, match="lines_in_mem must be positive"):
            list(reader.consume(1, lines_in_mem=0))
        with pytest.raises(ValueError, match="batch_size must be positive"):
            reader.producer_staging_bytes(0, 1)
        with pytest.raises(ValueError, match="lines_in_mem must be positive"):
            reader.producer_staging_bytes(1, 0)
        with pytest.raises(ValueError, match="window_rows must be positive"):
            reader.max_window_nnz(0)
        assert reader.max_window_nnz(2) == 2
    finally:
        reader.close()


def _flag_first_member_encrypted(path: Path) -> None:
    content = bytearray(path.read_bytes())
    # The general purpose flags follow the central directory signature,
    # the creator version, and the extractor version.
    position = content.index(b"PK\x01\x02") + 8
    content[position] |= 0x1
    path.write_bytes(bytes(content))


@pytest.mark.parametrize("member", ["encrypted", "fifo", "nested"])
def test_zip_members_that_cannot_be_read_safely_are_rejected(
    tmp_path: Path, member: str
) -> None:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, mode="w") as archive:
        if member == "fifo":
            info = zipfile.ZipInfo("matrix.mtx")
            info.create_system = 3
            info.external_attr = (stat.S_IFIFO | 0o644) << 16
            archive.writestr(info, b"")
        elif member == "nested":
            archive.writestr("inner.zip", b"")
        else:
            archive.writestr("matrix.mtx", f"{_BANNER}1 1 0\n")
    if member == "encrypted":
        _flag_first_member_encrypted(path)
    message = {
        "encrypted": "Encrypted ZIP archive member: matrix.mtx",
        "fifo": "Unsupported ZIP archive member: matrix.mtx",
        "nested": "cannot contain another ZIP",
    }[member]

    with pytest.raises(ValueError, match=message):
        inspect_mtx(path)


@pytest.mark.parametrize(
    ("sidecar", "content", "message"),
    [
        (
            "features.tsv",
            "feature-0\tgene-0\n",
            "Feature sidecar has 1 rows, expected 2",
        ),
        ("barcodes.tsv", "", "Cell sidecar has 0 rows, expected 2"),
        ("barcodes.tsv", "cell-0\n", "Cell sidecar has 1 rows, expected 2"),
        ("peaks.bed", "chr1\t10\t20\n", "Feature sidecar has 1 rows, expected 2"),
    ],
)
def test_sidecars_that_changed_after_inspection_are_rejected(
    tmp_path: Path, sidecar: str, content: str, message: str
) -> None:
    _write_mex(tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2)
    if sidecar == "peaks.bed":
        (tmp_path / "features.tsv").unlink()
        (tmp_path / "peaks.bed").write_text("chr1\t10\t20\nchr1\t30\t40\n")
    candidate = inspect_mtx(tmp_path)[0]
    feature_sidecar = "peaks.bed" if sidecar == "peaks.bed" else "features.tsv"
    assert Path(candidate.featurePath).name == feature_sidecar
    (tmp_path / sidecar).write_text(content)

    with pytest.raises(ValueError, match=message):
        MtxReader(candidate)


def test_matrix_declaring_entries_that_it_lacks_is_rejected(tmp_path: Path) -> None:
    _write_mex(tmp_path, "", n_features=2, n_cells=2, n_entries=2)

    with pytest.raises(ValueError, match="declares 2 entries, but 0 were read"):
        MtxReader(inspect_mtx(tmp_path)[0])


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ("", "declares 2 entries, but 0 were read"),
        ("2 2 4\n1 1 3\n", "cellMajor order is invalid at entry 2"),
    ],
    ids=["entries-removed", "entries-reordered"],
)
def test_matrix_changed_after_the_scan_is_rejected_while_streaming(
    tmp_path: Path, entries: str, message: str
) -> None:
    _write_mex(tmp_path, "1 1 3\n2 2 4\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        assert reader.coordinateOrder == "cellMajor"
        (tmp_path / "matrix.mtx").write_text(f"{_BANNER}2 2 2\n{entries}")
        with pytest.raises(ValueError, match=message):
            list(reader.consume(1))
    finally:
        reader.close()


def test_feature_major_matrix_whose_cell_entries_changed_is_rejected(
    tmp_path: Path,
) -> None:
    # Feature-major entries: cell 0 holds one entry and cell 1 holds two.
    _write_mex(tmp_path, "1 2 1\n2 1 1\n2 2 1\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0], temp_dir=str(tmp_path))
    try:
        assert reader.coordinateOrder == "featureMajor"
        # Still feature-major with as many entries, but cell 0 now holds two.
        (tmp_path / "matrix.mtx").write_text(f"{_BANNER}2 2 3\n1 1 1\n2 1 1\n2 2 1\n")
        with pytest.raises(RuntimeError, match="inconsistent row pointers"):
            list(reader.consume(1))
    finally:
        reader.close()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))


def test_feature_major_matrix_without_kept_cells_streams_nothing(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, "1 2 1\n2 1 1\n", n_features=2, n_cells=2)
    reader = MtxReader(
        inspect_mtx(tmp_path)[0],
        is_filtered=False,
        filtering_cutoff=5,
        temp_dir=str(tmp_path),
    )
    try:
        assert reader.coordinateOrder == "featureMajor"
        assert reader.nCells == 0
        assert list(reader.consume(1)) == []
    finally:
        reader.close()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))


def test_feature_major_import_without_free_temporary_disk_fails_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_mex(tmp_path, "1 2 1\n2 1 1\n", n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0], temp_dir=str(tmp_path))
    store = MemoryStore()
    try:
        writer = MtxToZarr(
            reader,
            store,
            mem_budget="64M",
            nthreads=1,
            policy=_POLICY,
            shuffle_cells=False,
        )
        monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
        with pytest.raises(OSError, match="temporary bytes, but 0 bytes are free"):
            writer.dump()
        monkeypatch.undo()
        # The failed preparation left nothing behind, so the import can retry.
        writer.dump()
    finally:
        reader.close()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))
    np.testing.assert_array_equal(_counts(store), [[0, 1], [1, 0]])


# Unordered entries: cell 0 holds feature 1 twice, apart, and feature 0, and
# cell 1 holds feature 0.
_UNORDERED = "2 1 2\n1 2 5\n1 1 3\n2 1 4\n"


@pytest.mark.parametrize(
    ("entries", "error", "message"),
    [
        ("2 1 2\n1 2 5\n1 1 3\n", ValueError, "declares 4 entries, but 3 were read"),
        ("2 1 2\n1 2 5\n2 2 1\n2 1 4\n", RuntimeError, "inconsistent row pointers"),
        ("2 1 2\n1 2 5\n1 1 3\n3 1 4\n", RuntimeError, "inconsistent row pointers"),
        ("2 1 2\n1 2 5\n1 1 3\n2 1 40\n", OverflowError, "the file changed"),
        ("2 1 5\n1 2 5\n1 1 3\n2 1 5\n", OverflowError, "the file changed"),
    ],
    ids=[
        "entries-removed",
        "cell-entries-moved",
        "duplicate-replaced",
        "count-raised",
        "sum-raised",
    ],
)
def test_unordered_matrix_changed_after_construction_is_rejected_when_sorted_again(
    tmp_path: Path, entries: str, error: type[Exception], message: str
) -> None:
    _write_mex(tmp_path, _UNORDERED, n_features=3, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0], temp_dir=str(tmp_path))
    try:
        assert reader.coordinateOrder == "unordered"
        # The first read streams the rows that construction sorted, and
        # removes them.
        np.testing.assert_array_equal(
            np.vstack([batch.toarray() for batch in reader.consume(1)]),
            [[3, 6, 0], [5, 0, 0]],
        )
        (tmp_path / "matrix.mtx").write_text(f"{_BANNER}3 2 4\n{entries}")
        with pytest.raises(error, match=message):
            list(reader.consume(1))
    finally:
        reader.close()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))


def test_unordered_matrix_whose_kept_cell_lost_an_entry_is_rejected(
    tmp_path: Path,
) -> None:
    # Cell 0 holds feature 1 twice, apart, and feature 0; the cutoff drops
    # cell 1, which holds one count.
    _write_mex(tmp_path, "2 1 2\n2 2 1\n1 1 3\n2 1 4\n", n_features=3, n_cells=2)
    reader = MtxReader(
        inspect_mtx(tmp_path)[0],
        is_filtered=False,
        filtering_cutoff=1,
        temp_dir=str(tmp_path),
    )
    try:
        assert reader.coordinateOrder == "unordered"
        np.testing.assert_array_equal(
            np.vstack([batch.toarray() for batch in reader.consume(1)]), [[3, 6, 0]]
        )
        # An entry of the kept cell moves to the dropped cell, so the header
        # still declares every entry and no kept cell gains one.
        (tmp_path / "matrix.mtx").write_text(
            f"{_BANNER}3 2 4\n2 1 2\n2 2 1\n1 2 3\n2 1 4\n"
        )
        with pytest.raises(RuntimeError, match="inconsistent row pointers"):
            list(reader.consume(1))
    finally:
        reader.close()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))


def test_unordered_matrix_without_free_temporary_disk_fails_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_mex(tmp_path, _UNORDERED, n_features=3, n_cells=2)
    candidate = inspect_mtx(tmp_path)[0]
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    # Four uint8 counts with int32 feature indexes.
    with pytest.raises(OSError, match="needs 20 temporary bytes, but 0 bytes are free"):
        MtxReader(candidate, temp_dir=str(tmp_path))
    monkeypatch.undo()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))

    reader = MtxReader(candidate, temp_dir=str(tmp_path))
    stores = [MemoryStore(), MemoryStore()]
    try:
        MtxToZarr(
            reader,
            stores[0],
            mem_budget="64M",
            nthreads=1,
            policy=_POLICY,
            shuffle_cells=False,
        ).dump()
        writer = MtxToZarr(
            reader,
            stores[1],
            mem_budget="64M",
            nthreads=1,
            policy=_POLICY,
            shuffle_cells=False,
        )
        # The second import sorts the rows again and fails for want of disk.
        monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
        with pytest.raises(OSError, match="temporary bytes, but 0 bytes are free"):
            writer.dump()
        monkeypatch.undo()
        assert not list(tmp_path.glob("cytearc-mtx-csr-*"))
        writer.dump()
    finally:
        reader.close()
    assert not list(tmp_path.glob("cytearc-mtx-csr-*"))
    for store in stores:
        np.testing.assert_array_equal(_counts(store), [[3, 6, 0], [5, 0, 0]])


def test_zip_matrix_imports_repeatedly_and_removes_each_extraction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("matrix.mtx", f"{_BANNER}2 2 2\n1 1 3\n2 2 4\n")
        archive.writestr("features.tsv", "feature-0\tgene-0\nfeature-1\tgene-1\n")
        archive.writestr("barcodes.tsv", "cell-0\ncell-1\n")
    reader = MtxReader(inspect_mtx(path)[0], temp_dir=str(tmp_path))
    stores = [MemoryStore(), MemoryStore()]
    try:
        for store in stores:
            MtxToZarr(
                reader, store, mem_budget="64M", nthreads=1, policy=_POLICY
            ).dump()
            assert not list(tmp_path.glob("cytearc-mtx-archive-*"))
    finally:
        reader.close()

    for store in stores:
        np.testing.assert_array_equal(_counts(store), [[3, 0], [0, 4]])
