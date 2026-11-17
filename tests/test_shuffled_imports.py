"""10x HDF5, Matrix Market, Tapestri, CSV and in-memory imports that shuffle cells.

Each writes row ``r`` of every per-cell array from the source's cell
``order[r]``, where the cells are those the import keeps. Counts are checked
against the source's own values; every other per-cell array against an
import of the same source in its own order, whose layout the other import
tests pin. These stores hold no artifacts, so two imports compare file for
file.
"""

import gzip
import hashlib
import os
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from loguru import logger
from scipy.sparse import coo_matrix, csr_matrix

import cytearc.readers._row_spill as row_spill
from cytearc.readers import (
    CrReader,
    CrH5Reader,
    CSVReader,
    MtxReader,
    TapestriReader,
    inspect_mtx,
)
from cytearc.readers._sparse import SparseRowStore
from cytearc.readers.cellranger import _selected_barcodes
from cytearc.storage.cell_order import (
    SOURCE_ROW,
    SOURCE_ROW_ROLE,
    recorded_shuffle_seed,
    shuffled_order,
)
from cytearc.utils.count_values import compressed_count_ranges
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import (
    CrToZarr,
    CSVtoZarr,
    SparseToZarr,
    TapestriToZarr,
)
from tests.variant_truth import tapestri_truth, write_tapestri

_N_CELLS = 157
_N_FEATURES = 12
# Genes, then antibodies, which an import stores as RNA and ADT.
_FEATURE_TYPES = ["Gene Expression"] * 9 + ["Antibody Capture"] * 3
_ASSAY_COLUMNS = {"RNA": np.arange(9), "ADT": np.arange(9, 12)}
# A cell whose counts total no more than this is dropped by readers that
# filter barcodes.
_CUTOFF = 12


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _counts(n_cells: int = _N_CELLS) -> np.ndarray:
    """Counts with empty cells, an empty feature, low totals and a count past uint16."""
    counts = np.random.default_rng(1).poisson(1.2, (n_cells, _N_FEATURES))
    counts[:, 7] = 0
    counts[[row for row in (3, 17) if row < n_cells]] = 0
    if n_cells > 40:
        counts[40, 2] = 70_000
    return counts.astype(np.int64)


def _kept(counts: np.ndarray, *, cutoff: bool) -> np.ndarray:
    """Return the cells that a reader keeps, in the file's order."""
    if not cutoff:
        return np.arange(counts.shape[0])
    return np.flatnonzero(counts.sum(axis=1) > _CUTOFF)


def _write_10x(
    path: Path,
    counts: np.ndarray,
    *,
    data: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    indptr: np.ndarray | None = None,
) -> Path:
    """Write a Cell Ranger HDF5 matrix of ``counts``, or of the given CSR arrays."""
    rows = csr_matrix(counts)
    n_cells, n_features = counts.shape
    with h5py.File(path, "w") as h5:
        group = h5.create_group("matrix")
        group.create_dataset(
            "data", data=rows.data.astype(np.uint32) if data is None else data
        )
        group.create_dataset(
            "indices",
            data=rows.indices.astype(np.int64) if indices is None else indices,
        )
        group.create_dataset(
            "indptr", data=rows.indptr.astype(np.int64) if indptr is None else indptr
        )
        group.create_dataset("shape", data=np.array([n_features, n_cells]))
        group.create_dataset(
            "barcodes", data=np.array([f"BC-{row}".encode() for row in range(n_cells)])
        )
        features = group.create_group("features")
        features.create_dataset(
            "id", data=np.array([f"feature-{f}".encode() for f in range(n_features)])
        )
        features.create_dataset(
            "name", data=np.array([f"gene-{f}".encode() for f in range(n_features)])
        )
        features.create_dataset(
            "feature_type",
            data=np.array([value.encode() for value in _FEATURE_TYPES[:n_features]]),
        )
    return path


def _mtx_entries(counts: np.ndarray, layout: str) -> list[tuple[int, int, int]]:
    """Return the one-based (feature, cell, count) entries of ``counts`` in ``layout``.

    ``unordered`` lists them in random order and splits each count above one
    into two entries, mostly far apart.
    """
    cells, features = np.nonzero(counts)
    entries = [
        (int(feature) + 1, int(cell) + 1, int(counts[cell, feature]))
        for cell, feature in zip(cells, features, strict=True)
    ]
    if layout == "cellMajor":
        return entries
    if layout == "featureMajor":
        return sorted(entries, key=lambda entry: (entry[0], entry[1]))
    entries = [
        part
        for feature, cell, count in entries
        for part in (
            [(feature, cell, count - count // 2), (feature, cell, count // 2)]
            if count > 1
            else [(feature, cell, count)]
        )
    ]
    order = np.random.default_rng(11).permutation(len(entries))
    return [entries[index] for index in order]


def _write_mtx(directory: Path, counts: np.ndarray, layout: str) -> Path:
    """Write a gzip Matrix Market triplet of ``counts`` in ``layout``."""
    directory.mkdir()
    entries = _mtx_entries(counts, layout)
    n_cells, n_features = counts.shape
    with gzip.open(directory / "matrix.mtx.gz", "wt") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write(f"{n_features} {n_cells} {len(entries)}\n")
        stream.writelines(f"{f} {c} {v}\n" for f, c, v in entries)
    (directory / "features.tsv").write_text(
        "".join(
            f"feature-{f}\tgene-{f}\t{_FEATURE_TYPES[f]}\n" for f in range(n_features)
        )
    )
    (directory / "barcodes.tsv").write_text(
        "".join(f"BC-{row}\n" for row in range(n_cells))
    )
    return directory


def _write_parse(directory: Path, counts: np.ndarray) -> Path:
    """Write a Parse Biosciences matrix whose cell metadata holds a ``sourceRow`` column."""
    directory.mkdir()
    n_cells, n_features = counts.shape
    cells, features = np.nonzero(counts)
    with gzip.open(directory / "DGE.mtx.gz", "wt") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write(f"{n_cells} {n_features} {cells.size}\n")
        stream.writelines(
            f"{cell + 1} {feature + 1} {counts[cell, feature]}\n"
            for cell, feature in zip(cells, features, strict=True)
        )
    pd.DataFrame(
        {
            "gene_id": [f"feature-{f}" for f in range(n_features)],
            "gene_name": [f"gene-{f}" for f in range(n_features)],
        }
    ).to_csv(directory / "all_genes.csv", index=False)
    rng = np.random.default_rng(4)
    pd.DataFrame(
        {
            "bc_wells": [f"well-{row}" for row in range(n_cells)],
            "sample": rng.choice(["A", "B", "C"], n_cells),
            "quality": rng.normal(size=n_cells),
            "genes": rng.integers(0, 100, n_cells),
            SOURCE_ROW: np.arange(n_cells)[::-1],
        }
    ).to_csv(directory / "cell_metadata.csv", index=False)
    return directory


def _write_csv(path: Path, counts: np.ndarray) -> Path:
    """Write a CSV of cells by features with cell columns of every kind."""
    rng = np.random.default_rng(6)
    n_cells = counts.shape[0]
    frame = pd.DataFrame(
        counts[:, :9], columns=[f"feature-{f}" for f in range(9)]
    ).astype(np.float64)
    frame.insert(0, "cell", [f"cell-{row}" for row in range(n_cells)])
    label = rng.choice(["a", "b", "c"], n_cells).astype(object)
    label[[row for row in (2, 30) if row < n_cells]] = ""
    score = rng.normal(size=n_cells)
    score[[row for row in (5, 50) if row < n_cells]] = np.nan
    frame["label"] = label
    frame["score"] = score
    frame["count"] = rng.integers(0, 1000, n_cells)
    frame["flag"] = rng.random(n_cells) < 0.5
    frame[SOURCE_ROW] = np.arange(n_cells)[::-1]
    frame.to_csv(path, index=False)
    return path


def _csv_reader(path: Path, **options: Any) -> CSVReader:
    return CSVReader(
        str(path),
        id_column=0,
        cell_data_cols=["label", "score", "count", "flag", SOURCE_ROW],
        batch_size=40,
        **options,
    )


def _open(path: Path) -> zarr.Group:
    return zarr.open_group(str(path), mode="r")


def _cell_arrays(root: zarr.Group) -> dict[str, np.ndarray]:
    return {name: np.asarray(array[...]) for name, array in root["cellData"].arrays()}


def _assert_rows_in_order(
    shuffled: np.ndarray, plain: np.ndarray, order: np.ndarray, name: str
) -> None:
    expected = plain[order]
    assert shuffled.dtype == expected.dtype, name
    # Byte views compare NaN, negative zero and text exactly.
    np.testing.assert_array_equal(
        np.ascontiguousarray(shuffled).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
        err_msg=name,
    )


def _assert_shuffled_store(
    plain: zarr.Group,
    shuffled: zarr.Group,
    order: np.ndarray,
    seed: int,
    expected_counts: dict[str, np.ndarray],
) -> None:
    """Check every per-cell array of the shuffled store against the plain store's rows.

    ``expected_counts`` holds each assay's counts in the source's order, read
    from the source itself.
    """
    n_cells = order.size
    assays = [name for name in plain.group_keys() if "counts" in plain[name]]
    assert sorted(assays) == sorted(expected_counts)
    for assay in assays:
        stored = np.asarray(shuffled[f"{assay}/counts"][...])
        np.testing.assert_array_equal(stored, expected_counts[assay][order])
        assert shuffled[f"{assay}/counts"].dtype == plain[f"{assay}/counts"].dtype
        if f"{assay}/countsT" in plain:
            np.testing.assert_array_equal(
                np.asarray(shuffled[f"{assay}/countsT"][...]), stored.T
            )
        else:
            assert f"{assay}/countsT" not in shuffled
        for name, array in plain[f"{assay}/countSummaries"].arrays():
            values = np.asarray(array[...])
            found = np.asarray(shuffled[f"{assay}/countSummaries/{name}"][...])
            if values.shape[:1] == (n_cells,):
                _assert_rows_in_order(found, values, order, name)
            else:
                np.testing.assert_array_equal(found, values, err_msg=name)
    plain_cells = _cell_arrays(plain)
    shuffled_cells = _cell_arrays(shuffled)
    assert set(shuffled_cells) == set(plain_cells) | {SOURCE_ROW}
    for name, values in plain_cells.items():
        if name == SOURCE_ROW:
            continue
        _assert_rows_in_order(shuffled_cells[name], values, order, name)
        assert dict(shuffled[f"cellData/{name}"].attrs) == dict(
            plain[f"cellData/{name}"].attrs
        ), name
    np.testing.assert_array_equal(shuffled_cells[SOURCE_ROW], order)
    assert shuffled_cells[SOURCE_ROW].dtype == np.int64
    assert dict(shuffled[f"cellData/{SOURCE_ROW}"].attrs) == {"role": SOURCE_ROW_ROLE}
    assert recorded_shuffle_seed(shuffled["cellData"]) == seed
    assert recorded_shuffle_seed(plain["cellData"]) is None


def _store_files(path: Path) -> dict[str, str]:
    """Digest every file of a store."""
    digests = {}
    for directory, _folders, files in os.walk(path):
        for name in files:
            full = os.path.join(directory, name)
            with open(full, "rb") as stream:
                digests[os.path.relpath(full, path)] = hashlib.sha256(
                    stream.read()
                ).hexdigest()
    return digests


def _no_temporary_files(directory: Path) -> bool:
    return not [
        path for path in directory.iterdir() if path.name.startswith("cytearc-")
    ]


# Each source: how to write it, read it, import it, and which counts and
# kept cells it holds.


def _cr_import(
    make_reader: Callable[[], Any], writer: type = CrToZarr, **writer_options: Any
) -> Callable[[Path, dict[str, Any]], Any]:
    def run(destination: Path, options: dict[str, Any]) -> Any:
        reader = make_reader()
        try:
            imported = writer(
                reader,
                str(destination),
                **{"mem_budget": "256M", "nthreads": 1, **writer_options, **options},
            )
            imported.dump()
        finally:
            reader.close()
        return imported

    return run


def _source(
    tmp_path: Path, kind: str
) -> tuple[Callable[[Path, dict[str, Any]], Any], dict[str, np.ndarray]]:
    """Return how to import a source of ``kind`` and the counts it holds, by assay."""
    counts = _counts()
    by_assay = {name: counts[:, columns] for name, columns in _ASSAY_COLUMNS.items()}
    scratch = str(tmp_path)
    if kind.startswith("10x"):
        cutoff = kind == "10x-cutoff"
        path = _write_10x(tmp_path / "matrix.h5", counts)
        kept = _kept(counts, cutoff=cutoff)

        def h5_reader() -> CrH5Reader:
            return CrH5Reader(
                str(path),
                is_filtered=not cutoff,
                filtering_cutoff=_CUTOFF,
                temp_dir=scratch,
            )

        return _cr_import(h5_reader), {
            name: values[kept] for name, values in by_assay.items()
        }
    if kind.startswith("mtx"):
        _prefix, layout, filtering = kind.split("-")
        cutoff = filtering == "cutoff"
        directory = _write_mtx(tmp_path / "mtx", counts, layout)
        kept = _kept(counts, cutoff=cutoff)

        def mtx_reader() -> MtxReader:
            return MtxReader(
                inspect_mtx(directory)[0],
                is_filtered=not cutoff,
                filtering_cutoff=_CUTOFF,
                temp_dir=scratch,
            )

        # Small parses split cells across chunks.
        return _cr_import(mtx_reader, lines_in_mem=50), {
            name: values[kept] for name, values in by_assay.items()
        }
    if kind == "parse":
        directory = _write_parse(tmp_path / "parse", counts[:, :9])
        return _cr_import(
            lambda: MtxReader(inspect_mtx(directory)[0], temp_dir=scratch),
            lines_in_mem=50,
        ), {"RNA": counts[:, :9]}
    if kind == "tapestri":
        truth = tapestri_truth()
        path = write_tapestri(truth, tmp_path / "run.dna+protein.h5")
        return _cr_import(
            lambda: TapestriReader(str(path), temp_dir=scratch), TapestriToZarr
        ), {"CNV": truth.amplicon_reads, "ADT": truth.protein_reads}
    if kind == "csv":
        path = _write_csv(tmp_path / "counts.csv", counts)

        def csv_import(destination: Path, options: dict[str, Any]) -> Any:
            writer = CSVtoZarr(
                _csv_reader(path, temp_dir=scratch),
                str(destination),
                "RNA",
                **{"mem_budget": "256M", "nthreads": 1, **options},
            )
            writer.dump()
            return writer

        return csv_import, {"RNA": counts[:, :9].astype(np.float64)}
    assert kind in ("sparse", "sparse-assays")
    matrices: Any = (
        csr_matrix(counts[:, :9])
        if kind == "sparse"
        else {name: csr_matrix(values) for name, values in by_assay.items()}
    )
    feature_ids: Any = (
        [f"feature-{f}" for f in range(9)]
        if kind == "sparse"
        else {
            name: [f"feature-{f}" for f in columns]
            for name, columns in _ASSAY_COLUMNS.items()
        }
    )

    def sparse_import(destination: Path, options: dict[str, Any]) -> Any:
        writer = SparseToZarr(
            matrices,
            str(destination),
            [f"cell-{row}" for row in range(_N_CELLS)],
            feature_ids,
            **{"mem_budget": "256M", "nthreads": 1, **options},
        )
        # Batches of 17 cells cross the shards and the buckets.
        writer.dump(batch_size=17)
        return writer

    return sparse_import, (
        {"RNA": counts[:, :9]} if kind == "sparse" else dict(by_assay)
    )


_KINDS = [
    "10x-filtered",
    "10x-cutoff",
    "mtx-cellMajor-filtered",
    "mtx-cellMajor-cutoff",
    "mtx-featureMajor-filtered",
    "mtx-featureMajor-cutoff",
    "mtx-unordered-filtered",
    "mtx-unordered-cutoff",
    "parse",
    "tapestri",
    "csv",
    "sparse",
    "sparse-assays",
]


@pytest.mark.parametrize("kind", _KINDS)
def test_every_per_cell_array_follows_the_shuffled_order(tmp_path, kind) -> None:
    run, expected = _source(tmp_path, kind)
    with _warnings() as plain_warnings:
        run(tmp_path / "plain.zarr", {})
    with _warnings() as shuffled_warnings:
        writer = run(
            tmp_path / "shuffled.zarr", {"shuffle_cells": True, "shuffle_seed": 11}
        )
    assert writer.shuffleCells is True and writer.shuffleSeed == 11
    plain = _open(tmp_path / "plain.zarr")
    n_cells = int(plain["cellData/ids"].shape[0])
    order = shuffled_order(n_cells, 11)
    _assert_shuffled_store(
        plain, _open(tmp_path / "shuffled.zarr"), order, 11, expected
    )
    # A source cell column named sourceRow imports only in the file's order.
    skipped = "Skipped source cell metadata column 'sourceRow' because CyteArc stores"
    if kind in ("parse", "csv"):
        assert "role" not in plain[f"cellData/{SOURCE_ROW}"].attrs
        assert any(message.startswith(skipped) for message in shuffled_warnings)
    assert not any(SOURCE_ROW in message for message in plain_warnings)
    # The temporary rows are gone once the import ends.
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("kind", ["10x-cutoff", "mtx-unordered-cutoff", "csv"])
def test_one_seed_gives_the_same_store_and_another_seed_another_order(
    tmp_path, kind
) -> None:
    run, _expected = _source(tmp_path, kind)
    for name, seed in (("first", 6), ("second", 6), ("other", 7)):
        run(tmp_path / f"{name}.zarr", {"shuffle_cells": True, "shuffle_seed": seed})
    first = _store_files(tmp_path / "first.zarr")
    assert first == _store_files(tmp_path / "second.zarr")
    assert "cellData/sourceRow/zarr.json" in first
    other = _store_files(tmp_path / "other.zarr")
    assert set(other) == set(first)
    assert other["cellData/sourceRow/c/0"] != first["cellData/sourceRow/c/0"]


@pytest.mark.parametrize("kind", _KINDS)
def test_an_import_without_the_shuffle_writes_no_cell_order(tmp_path, kind) -> None:
    run, _expected = _source(tmp_path, kind)
    writer = run(tmp_path / "default.zarr", {})
    assert writer.shuffleCells is False
    run(tmp_path / "explicit.zarr", {"shuffle_cells": False, "shuffle_seed": 9})
    default = _store_files(tmp_path / "default.zarr")
    # The seed changes nothing without the shuffle.
    assert default == _store_files(tmp_path / "explicit.zarr")
    cells = _open(tmp_path / "default.zarr")["cellData"]
    assert "cellOrder" not in cells.attrs
    if SOURCE_ROW in cells:
        assert "role" not in cells[SOURCE_ROW].attrs


def test_selected_barcodes_pass_on_only_the_cells_a_reader_keeps() -> None:
    received: list[tuple[int, np.ndarray]] = []
    # Barcodes 4 to 12, whose values are contiguous; 6, 7 and 10 are not
    # selected and hold no values. The selected ones start at position 20.
    selected = np.array([4, 5, 8, 9, 11, 12])
    consume = _selected_barcodes(
        lambda first, block: received.append((first, block.toarray())),
        selected,
        20,
    )
    values = np.arange(1, 10)[:, None] * np.array([[1, 0]])
    values[[2, 3, 6]] = 0
    block = csr_matrix(values)
    consume(0, block[:2])
    consume(2, block[2:4])
    consume(2, block[2:9])
    assert [first for first, _rows in received] == [20, 22]
    np.testing.assert_array_equal(received[0][1], block[:2].toarray())
    np.testing.assert_array_equal(received[1][1], block[[4, 5, 7, 8]].toarray())
    # A block of selected barcodes alone passes on unchanged.
    whole = block[:2]
    seen: list[Any] = []
    _selected_barcodes(lambda first, rows: seen.append(rows), selected, 0)(0, whole)
    assert seen[0] is whole


def test_10x_repeated_coordinates_sum_as_the_file_order_sums_them(tmp_path) -> None:
    rng = np.random.default_rng(5)
    n_cells, n_features = 60, 9
    base = coo_matrix(rng.poisson(0.6, (n_cells, n_features)) * 0.1)
    repeats = rng.random(base.nnz) < 0.4
    rows = np.concatenate([base.row, base.row[repeats], base.row[repeats]])
    columns = np.concatenate([base.col, base.col[repeats], base.col[repeats]])
    values = np.concatenate(
        [base.data, base.data[repeats] * 1e-9, base.data[repeats] * 3.3]
    )
    # Each barcode holds its entries in a random order.
    order = np.lexsort((rng.random(rows.size), rows))
    pointers = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n_cells))))
    path = _write_10x(
        tmp_path / "unsorted.h5",
        np.zeros((n_cells, n_features)),
        data=values[order],
        indices=columns[order].astype(np.int64),
        indptr=pointers.astype(np.int64),
    )
    stores = {}
    for shuffle in (False, True):
        reader = CrH5Reader(str(path))
        try:
            CrToZarr(
                reader,
                str(tmp_path / f"{shuffle}.zarr"),
                nthreads=1,
                values="normalized",
                shuffle_cells=shuffle,
                shuffle_seed=4,
            ).dump()
        finally:
            reader.close()
        stores[shuffle] = np.asarray(
            _open(tmp_path / f"{shuffle}.zarr")["RNA/counts"][...]
        )
    summed = coo_matrix((values, (rows, columns)), shape=(n_cells, n_features))
    np.testing.assert_allclose(stores[False], summed.toarray(), rtol=1e-12)
    # Bit for bit the plain import's sums, in shuffled order.
    _assert_rows_in_order(
        stores[True], stores[False], shuffled_order(n_cells, 4), "counts"
    )


def _whole_then_fractional(path: Path) -> tuple[Path, np.ndarray]:
    """Write a 10x HDF5 file whose first 100 cells hold whole numbers and the rest fractions."""
    values = np.random.default_rng(8).poisson(2.0, (300, 6)).astype(np.float32) + 1
    values[100:] += 0.25
    rows = csr_matrix(values)
    path = _write_10x(
        path,
        np.zeros((300, 6)),
        data=rows.data,
        indices=rows.indices.astype(np.int64),
        indptr=rows.indptr.astype(np.int64),
    )
    return path, values


@pytest.mark.parametrize("shuffle", [False, True])
def test_the_values_are_decided_from_the_files_first_cells(tmp_path, shuffle) -> None:
    path, values = _whole_then_fractional(tmp_path / "values.h5")
    reader = CrH5Reader(str(path))
    try:
        # Shuffled cells mix fractions into the first 100, which would show
        # what the values are; the file's first cells are whole numbers.
        with pytest.raises(ValueError, match="first 100 cells hold only whole numbers"):
            CrToZarr(
                reader,
                str(tmp_path / "store.zarr"),
                nthreads=1,
                shuffle_cells=shuffle,
            )
        np.testing.assert_array_equal(
            reader._file_first_rows(100, 0).toarray(), values[:100]
        )
    finally:
        reader.close()
    assert not (tmp_path / "store.zarr").exists()


def _10x_reader(tmp_path: Path, n_cells: int = _N_CELLS, **options: Any) -> CrH5Reader:
    path = tmp_path / f"matrix-{n_cells}.h5"
    if not path.exists():
        _write_10x(path, _counts(n_cells))
    return CrH5Reader(str(path), temp_dir=str(tmp_path), **options)


def test_counts_are_written_once_from_the_cells_of_their_own_import(tmp_path) -> None:
    reader = _10x_reader(tmp_path)
    try:
        first = CrToZarr(
            reader, str(tmp_path / "first.zarr"), nthreads=1, shuffle_cells=True
        )
        # A second import of the reader shuffles anew and replaces the cells.
        second = CrToZarr(
            reader,
            str(tmp_path / "second.zarr"),
            nthreads=1,
            shuffle_cells=True,
            shuffle_seed=1,
        )
        with pytest.raises(RuntimeError, match="another import of the reader"):
            first.dump()
        second.dump()
        with pytest.raises(RuntimeError, match="written once and removed"):
            second.dump()
        # Without the shuffle, the reader reads the file again.
        CrToZarr(reader, str(tmp_path / "plain.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    order = shuffled_order(_N_CELLS, 1)
    plain = np.asarray(_open(tmp_path / "plain.zarr")["RNA/counts"][...])
    np.testing.assert_array_equal(
        _open(tmp_path / "second.zarr")["RNA/counts"][...], plain[order]
    )
    assert _no_temporary_files(tmp_path)


def test_a_matrix_market_reader_imports_shuffled_and_plain_from_one_archive(
    tmp_path,
) -> None:
    counts = _counts()
    directory = _write_mtx(tmp_path / "mtx", counts, "unordered")
    archive = tmp_path / "matrix.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in sorted(directory.iterdir()):
            stream.write(path, f"sample/{path.name}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    reader = MtxReader(inspect_mtx(archive)[0], temp_dir=str(scratch))
    try:
        for name, seed in (("first", 2), ("plain", None), ("second", 3)):
            options = (
                {} if seed is None else {"shuffle_cells": True, "shuffle_seed": seed}
            )
            CrToZarr(
                reader, str(tmp_path / f"{name}.zarr"), nthreads=1, **options
            ).dump()
            # The archive's members and every row are removed after each import.
            assert list(scratch.iterdir()) == []
    finally:
        reader.close()
    plain = _open(tmp_path / "plain.zarr")
    np.testing.assert_array_equal(plain["RNA/counts"][...], counts[:, :9])
    for name, seed in (("first", 2), ("second", 3)):
        order = shuffled_order(_N_CELLS, seed)
        np.testing.assert_array_equal(
            _open(tmp_path / f"{name}.zarr")["RNA/counts"][...], counts[order][:, :9]
        )


@pytest.mark.parametrize("kind", ["10x-filtered", "mtx-featureMajor-filtered", "csv"])
def test_cells_of_many_buckets_are_read_ahead_and_each_deleted_once_read(
    tmp_path, monkeypatch, kind
) -> None:
    # Buckets of a few cells each.
    monkeypatch.setattr(row_spill, "_SPILL_READ_BYTES", 3_000)
    deleted: list[tuple[int, int]] = []
    loaded: list[int] = []
    original_delete = SparseRowStore._delete
    original_load = SparseRowStore._load_bucket

    def delete(self, bucket, start, stop):
        deleted.append(
            tuple(int(row) for row in self.bucket_starts[bucket : bucket + 2])
        )
        original_delete(self, bucket, start, stop)

    def load(self, bucket):
        loaded.append(bucket)
        return original_load(self, bucket)

    monkeypatch.setattr(SparseRowStore, "_delete", delete)
    monkeypatch.setattr(SparseRowStore, "_load_bucket", load)
    run, expected = _source(tmp_path, kind)
    run(tmp_path / "plain.zarr", {})
    run(tmp_path / "shuffled.zarr", {"shuffle_cells": True})
    # Every bucket was read once and deleted once read, in order.
    assert len(deleted) > 5
    assert sorted(loaded) == loaded == list(range(len(deleted)))
    assert [start for start, _stop in deleted] == sorted(
        start for start, _stop in deleted
    )
    assert deleted[0][0] == 0 and deleted[-1][1] == _N_CELLS
    assert all(
        left[1] == right[0] for left, right in zip(deleted, deleted[1:], strict=False)
    )
    _assert_shuffled_store(
        _open(tmp_path / "plain.zarr"),
        _open(tmp_path / "shuffled.zarr"),
        shuffled_order(_N_CELLS, 0),
        0,
        expected,
    )
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"shuffle_cells": "yes"}, TypeError),
        ({"shuffle_cells": True, "shuffle_seed": -1}, ValueError),
        ({"shuffle_cells": True, "shuffle_seed": 2**64}, ValueError),
        ({"shuffle_cells": True, "shuffle_seed": 1.5}, TypeError),
    ],
)
@pytest.mark.parametrize("kind", ["10x-filtered", "tapestri", "csv", "sparse"])
def test_bad_shuffle_arguments_are_refused_before_the_store_exists(
    tmp_path, kind, options, error
) -> None:
    run, _expected = _source(tmp_path, kind)
    with pytest.raises(error, match="shuffle_"):
        run(tmp_path / "refused.zarr", options)
    assert not (tmp_path / "refused.zarr").exists()
    assert _no_temporary_files(tmp_path)


def test_a_10x_budget_without_room_for_the_spill_is_refused_with_the_bytes_it_needs(
    tmp_path,
) -> None:
    counts = _counts()
    nonzeros = np.count_nonzero(counts, axis=1)
    rows, longest = _N_CELLS, int(nonzeros.max())
    # The order and its inverse are held beside the scan.
    held = 2 * 8 * rows
    budget = 8_000
    # Buckets read in a thirty-second of what the scan may hold, at 38 bytes
    # for each uint16 entry read and 80 for each row.
    row_bytes = nonzeros.mean() * 38 + 80
    buckets = -(-rows // max(1, int(((budget - held) // 32) // row_bytes)))
    # Row counts of the store; the writer's counts and marks, a few numbers
    # per bucket, and the destinations of its rows; a scan window of the
    # longest row and what appending it holds.
    least = (
        (rows + 1) * 32 + rows * 9 + (buckets + 1) * 25 + 8 * rows + longest * (72 + 96)
    )
    reader = _10x_reader(tmp_path)
    try:
        with pytest.raises(MemoryBudgetError) as caught:
            CrToZarr(
                reader,
                str(tmp_path / "refused.zarr"),
                mem_budget=budget,
                nthreads=1,
                shuffle_cells=True,
            )
        assert caught.value.needed_bytes == least
        assert caught.value.budget_bytes == budget - held
        assert caught.value.item == "the row spill of a cell shuffle"
        assert reader._shuffledRows is None
        # With that much, the spill fits; a later step of so small a budget
        # refuses for itself.
        with pytest.raises(MemoryError) as caught:
            CrToZarr(
                reader,
                str(tmp_path / "admitted.zarr"),
                mem_budget=least + held,
                nthreads=1,
                shuffle_cells=True,
            )
        assert "Shuffling cells" not in str(caught.value)
    finally:
        reader.close()
    assert not (tmp_path / "refused.zarr").exists()
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("kind", ["mtx-cellMajor-filtered", "csv"])
def test_a_budget_without_room_for_a_batch_and_its_spill_names_the_bytes_it_needs(
    tmp_path, kind
) -> None:
    run, _expected = _source(tmp_path, kind)
    attempts = iter(range(1_000))

    def at_budget(budget: int) -> None:
        destination = tmp_path / f"store-{next(attempts)}.zarr"
        try:
            run(destination, {"shuffle_cells": True, "mem_budget": budget})
        except MemoryBudgetError as error:
            if error.item == "the row spill of a cell shuffle":
                assert not destination.exists()
                raise
        except MemoryError:
            # A later step of so small a budget refuses for itself.
            pass

    with pytest.raises(MemoryBudgetError) as caught:
        at_budget(4_000)
    assert caught.value.item == "the row spill of a cell shuffle"
    assert caught.value.needed_bytes > caught.value.budget_bytes
    # The order and its inverse, and the reader's own arrays, are held
    # beside the spill.
    held = 4_000 - caught.value.budget_bytes
    assert held >= 16 * _N_CELLS
    # With the bytes it names beside them, the spill fits; buckets grow with
    # the budget, so it needs no more there.
    at_budget(caught.value.needed_bytes + held)
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        ("10x-filtered", "Shuffled 10x HDF5 import"),
        ("mtx-unordered-filtered", "Shuffled Matrix Market import"),
        ("tapestri", "Shuffled Tapestri import"),
        ("csv", "Shuffled CSV import"),
    ],
)
def test_a_disk_without_room_for_the_spill_is_refused(
    tmp_path, monkeypatch, kind, operation
) -> None:
    import cytearc.readers._sparse as sparse

    run, _expected = _source(tmp_path, kind)
    real_usage = sparse.shutil.disk_usage

    def disk_usage(path):
        usage = real_usage(path)
        if "cytearc-sparse-" in str(path):
            return usage._replace(free=10)
        return usage

    monkeypatch.setattr(sparse.shutil, "disk_usage", disk_usage)
    with pytest.raises(
        OSError, match=f"^{operation} needs [0-9]+ temporary bytes .* which has 10 free"
    ):
        run(tmp_path / "refused.zarr", {"shuffle_cells": True})
    assert not (tmp_path / "refused.zarr").exists()
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("n_cells", [1, 2])
@pytest.mark.parametrize("kind", ["10x", "mtx", "csv", "sparse"])
def test_sources_of_one_or_two_cells_shuffle(tmp_path, kind, n_cells) -> None:
    counts = _counts(n_cells)
    counts[:, 0] += 1
    stores = {}
    for shuffle in (False, True):
        destination = tmp_path / f"{shuffle}.zarr"
        options = {"nthreads": 1, "shuffle_cells": shuffle, "shuffle_seed": 5}
        if kind == "10x":
            path = tmp_path / "matrix.h5"
            if not path.exists():
                _write_10x(path, counts)
            _cr_import(lambda: CrH5Reader(str(path), temp_dir=str(tmp_path)))(
                destination, {"shuffle_cells": shuffle, "shuffle_seed": 5}
            )
        elif kind == "mtx":
            directory = tmp_path / "mtx"
            if not directory.exists():
                _write_mtx(directory, counts, "featureMajor")
            _cr_import(
                lambda: MtxReader(inspect_mtx(directory)[0], temp_dir=str(tmp_path))
            )(destination, {"shuffle_cells": shuffle, "shuffle_seed": 5})
        elif kind == "csv":
            path = tmp_path / "counts.csv"
            if not path.exists():
                _write_csv(path, counts)
            CSVtoZarr(
                _csv_reader(path, temp_dir=str(tmp_path)),
                str(destination),
                "RNA",
                **options,
            ).dump()
        else:
            SparseToZarr(
                csr_matrix(counts),
                str(destination),
                [f"cell-{row}" for row in range(n_cells)],
                [f"feature-{f}" for f in range(_N_FEATURES)],
                **options,
            ).dump()
        stores[shuffle] = _open(destination)
    order = shuffled_order(n_cells, 5)
    np.testing.assert_array_equal(
        stores[True]["RNA/counts"][...], stores[False]["RNA/counts"][...][order]
    )
    np.testing.assert_array_equal(stores[True]["cellData/sourceRow"][...], order)
    np.testing.assert_array_equal(
        stores[True]["cellData/ids"][...], stores[False]["cellData/ids"][...][order]
    )
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("kind", ["10x", "mtx", "sparse"])
def test_sources_without_cells_shuffle(tmp_path, kind) -> None:
    counts = _counts()
    destination = tmp_path / "store.zarr"
    if kind == "10x":
        # No barcode holds more than the cutoff.
        path = _write_10x(tmp_path / "matrix.h5", counts)
        _cr_import(
            lambda: CrH5Reader(
                str(path),
                is_filtered=False,
                filtering_cutoff=10**9,
                temp_dir=str(tmp_path),
            )
        )(destination, {"shuffle_cells": True})
    elif kind == "mtx":
        directory = _write_mtx(tmp_path / "mtx", counts, "unordered")
        _cr_import(
            lambda: MtxReader(
                inspect_mtx(directory)[0],
                is_filtered=False,
                filtering_cutoff=10**9,
                temp_dir=str(tmp_path),
            )
        )(destination, {"shuffle_cells": True})
    else:
        SparseToZarr(
            csr_matrix((0, 4)),
            str(destination),
            [],
            [f"feature-{f}" for f in range(4)],
            nthreads=1,
            shuffle_cells=True,
        ).dump()
    root = _open(destination)
    assert root["RNA/counts"].shape[0] == 0
    assert root["cellData/sourceRow"].shape == (0,)
    assert recorded_shuffle_seed(root["cellData"]) == 0
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("rows", [-3, 5])
def test_a_csv_file_that_changes_after_the_reader_read_it_is_refused(
    tmp_path, rows
) -> None:
    counts = _counts()
    path = _write_csv(tmp_path / "counts.csv", counts)
    reader = _csv_reader(path, temp_dir=str(tmp_path))
    changed = (
        counts[: _N_CELLS + rows] if rows < 0 else np.vstack([counts, counts[:rows]])
    )
    _write_csv(path, changed)
    with pytest.raises(ValueError, match="no longer holds the 157 rows"):
        CSVtoZarr(
            reader, str(tmp_path / "store.zarr"), "RNA", nthreads=1, shuffle_cells=True
        )
    assert not (tmp_path / "store.zarr").exists()
    assert _no_temporary_files(tmp_path)


class _MemoryReader(CrReader):
    """A reader that implements only the abstract members of ``CrReader``, as before shuffles."""

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
            "feature_ids": [f"feature-{f}" for f in range(n_features)],
            "feature_names": [f"gene-{f}" for f in range(n_features)],
            "cell_names": [f"cell-{row}" for row in range(n_cells)],
        }[key]

    @property
    def matrix_dtype(self) -> np.dtype:
        return self._matrix.dtype

    def count_value_ranges(self, maxBytes, featureGroups=None):  # type: ignore[override]
        return compressed_count_ranges(
            self._matrix.indptr,
            self._matrix.indices,
            self._matrix.data,
            minorSize=self._matrix.shape[1],
            maxBytes=maxBytes,
            groups=featureGroups,
        )

    def consume(self, batch_size, lines_in_mem):
        for start in range(0, self._matrix.shape[0], batch_size):
            yield self._matrix[start : start + batch_size].tocoo()


def test_a_reader_whose_scan_cannot_shuffle_is_refused_and_still_imports_in_order(
    tmp_path,
) -> None:
    values = np.random.default_rng(3).random((150, 4)) * 4
    values[values < 2] = 0
    reader = _MemoryReader(values)
    with pytest.raises(TypeError, match="takes no destinationRows"):
        CrToZarr(reader, str(tmp_path / "refused.zarr"), nthreads=1, shuffle_cells=True)
    assert not (tmp_path / "refused.zarr").exists()
    # Its first cells, which decide what the values are, come from consume.
    first = reader._file_first_rows(100, 1)
    np.testing.assert_array_equal(first.toarray(), values[:100])
    CrToZarr(
        reader, str(tmp_path / "plain.zarr"), nthreads=1, values="normalized"
    ).dump()
    np.testing.assert_allclose(
        _open(tmp_path / "plain.zarr")["RNA/counts"][...], values, rtol=1e-6
    )
    assert _MemoryReader(np.zeros((0, 4)))._file_first_rows(100, 1).shape == (0, 4)


def test_each_reader_reads_its_first_cells_from_the_file_once_shuffled(
    tmp_path,
) -> None:
    counts = _counts()
    truth = tapestri_truth()
    readers = [
        (
            CrH5Reader(
                str(_write_10x(tmp_path / "matrix.h5", counts)), temp_dir=str(tmp_path)
            ),
            counts,
        ),
        (
            MtxReader(
                inspect_mtx(_write_mtx(tmp_path / "mtx", counts, "unordered"))[0],
                temp_dir=str(tmp_path),
            ),
            counts,
        ),
        (
            TapestriReader(
                str(write_tapestri(truth, tmp_path / "run.h5")), temp_dir=str(tmp_path)
            ),
            np.hstack([truth.amplicon_reads, truth.protein_reads]),
        ),
    ]
    try:
        for reader, values in readers:
            reader.count_value_ranges(
                1 << 28,
                destinationRows=np.argsort(shuffled_order(reader.nCells, 0)),
            )
            assert reader._shuffledRows is not None
            np.testing.assert_array_equal(
                reader._file_first_rows(30, 20).toarray(), values[:30]
            )
            with pytest.raises(ValueError, match="batch_size must be positive"):
                next(reader.consume(0, 20))
            # The shuffled cells are still all there.
            stacked = np.vstack([batch.toarray() for batch in reader.consume(64, 20)])
            np.testing.assert_array_equal(
                stacked, values[shuffled_order(reader.nCells, 0)]
            )
    finally:
        for reader, _values in readers:
            reader.close()
    assert _no_temporary_files(tmp_path)


def test_readers_without_cells_read_no_first_cells(tmp_path) -> None:
    counts = _counts()
    readers = [
        CrH5Reader(
            str(_write_10x(tmp_path / "matrix.h5", counts)),
            is_filtered=False,
            filtering_cutoff=10**9,
        ),
        MtxReader(
            inspect_mtx(_write_mtx(tmp_path / "mtx", counts, "cellMajor"))[0],
            is_filtered=False,
            filtering_cutoff=10**9,
        ),
    ]
    try:
        for reader in readers:
            assert reader.nCells == 0
            assert reader._file_first_rows(100, 50).shape == (0, _N_FEATURES)
    finally:
        for reader in readers:
            reader.close()


def test_a_csv_import_that_fails_after_its_spill_removes_the_rows(tmp_path) -> None:
    path = _write_csv(tmp_path / "counts.csv", _counts())
    destination = tmp_path / "taken.zarr"
    destination.mkdir()
    (destination / "file.txt").write_text("not a store")
    with pytest.raises(FileExistsError):
        CSVtoZarr(
            _csv_reader(path, temp_dir=str(tmp_path)),
            str(destination),
            "RNA",
            nthreads=1,
            shuffle_cells=True,
        )
    assert _no_temporary_files(tmp_path)
    writer = CSVtoZarr(
        _csv_reader(path, temp_dir=str(tmp_path)),
        str(tmp_path / "store.zarr"),
        "RNA",
        nthreads=1,
        shuffle_cells=True,
    )
    writer.dump()
    with pytest.raises(RuntimeError, match="written once and removed"):
        writer.dump()


def test_a_matrix_market_shuffle_parses_with_the_imports_parse_buffer(
    tmp_path, monkeypatch
) -> None:
    import cytearc.readers.mtx as mtx

    parses: list[int] = []
    original = mtx._MtxEngine._raw_chunks

    def raw_chunks(self, lines_in_mem):
        parses.append(lines_in_mem)
        yield from original(self, lines_in_mem)

    monkeypatch.setattr(mtx._MtxEngine, "_raw_chunks", raw_chunks)
    directory = _write_mtx(tmp_path / "mtx", _counts(), "cellMajor")
    reader = MtxReader(inspect_mtx(directory)[0], temp_dir=str(tmp_path))
    try:
        # Construction scans with the reader's own buffer.
        assert parses == [100_000]
        parses.clear()
        CrToZarr(
            reader,
            str(tmp_path / "store.zarr"),
            nthreads=1,
            lines_in_mem=50,
            shuffle_cells=True,
        ).dump()
    finally:
        reader.close()
    # The shuffle's pass over the cells is the import's only parse.
    assert parses == [50]


def test_a_10x_import_removes_its_shuffled_cells_before_counts_t(
    tmp_path, monkeypatch
) -> None:
    import cytearc.writers.counts_t as counts_t

    reader = _10x_reader(tmp_path)
    seen: list[tuple[Any, bool]] = []
    original = counts_t.finalize_writer_counts_t_many

    def finalize(*args, **kwargs):
        seen.append((reader._shuffledRows, _no_temporary_files(tmp_path)))
        return original(*args, **kwargs)

    monkeypatch.setattr(counts_t, "finalize_writer_counts_t_many", finalize)
    try:
        CrToZarr(
            reader, str(tmp_path / "store.zarr"), nthreads=1, shuffle_cells=True
        ).dump()
    finally:
        reader.close()
    # countsT reads counts; the shuffled cells and their files are gone.
    assert seen == [(None, True)]
