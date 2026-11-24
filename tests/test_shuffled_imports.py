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
    source_order_rows,
)
from cytearc.utils.count_values import compressed_count_ranges
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import (
    CrToZarr,
    CSVtoZarr,
    SparseToZarr,
    TapestriToZarr,
)
from tests.storage_helpers import drawn_shuffles, shuffle_refusal
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
        **{
            "id_column": 0,
            "cell_data_cols": ["label", "score", "count", "flag", SOURCE_ROW],
            "batch_size": 40,
            **options,
        },
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
    file_rows: np.ndarray,
) -> None:
    """Check every per-cell array of the shuffled store against the plain store's rows.

    ``expected_counts`` holds each assay's counts in the source's order, read
    from the source itself, and ``file_rows`` the row in the file of each
    cell that the import keeps.
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
    # Each cell's row in the file: with every row kept, the order itself.
    np.testing.assert_array_equal(shuffled_cells[SOURCE_ROW], file_rows[order])
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
    if kind.startswith("parse"):
        cutoff = kind == "parse-cutoff"
        directory = _write_parse(tmp_path / "parse", counts[:, :9])
        return _cr_import(
            lambda: MtxReader(
                inspect_mtx(directory)[0],
                is_filtered=not cutoff,
                filtering_cutoff=_CUTOFF,
                temp_dir=scratch,
            ),
            lines_in_mem=50,
        ), {"RNA": counts[_source_rows(kind)][:, :9]}
    if kind.startswith("tapestri"):
        truth = tapestri_truth()
        # Every row of the truth, or its first eight.
        rows = _tapestri_rows(kind)
        path = write_tapestri(truth, tmp_path / "run.dna+protein.h5", rows=rows)
        return _cr_import(
            lambda: TapestriReader(str(path), temp_dir=scratch), TapestriToZarr
        ), {"CNV": truth.amplicon_reads[rows], "ADT": truth.protein_reads[rows]}
    if kind.startswith("csv"):
        path = _write_csv(tmp_path / "counts.csv", counts)
        # Without an ID column, the reader names the cells cell_0, cell_1 and
        # so on, and the file's cell column is left out.
        reader_options: dict[str, Any] = (
            {"id_column": None, "skip_cols": ["cell"]}
            if kind == "csv-without-ids"
            else {}
        )

        def csv_import(destination: Path, options: dict[str, Any]) -> Any:
            writer = CSVtoZarr(
                _csv_reader(path, temp_dir=scratch, **reader_options),
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


def _tapestri_rows(kind: str) -> np.ndarray:
    """Return the truth's rows that a Tapestri source of ``kind`` holds: all, or the first eight."""
    return np.arange(8 if kind == "tapestri-8" else tapestri_truth().n_rows)


def _source_rows(kind: str) -> np.ndarray:
    """Return the row in the file of each cell that an import of ``kind`` keeps.

    Readers that filter keep the cells whose counts total more than the
    cutoff; the Parse matrix holds the first nine features only.
    """
    counts = _counts()
    if kind == "tapestri":
        return np.arange(len(tapestri_truth().row_ids))
    if kind == "parse-cutoff":
        return np.flatnonzero(counts[:, :9].sum(axis=1) > _CUTOFF)
    return _kept(counts, cutoff=kind.endswith("cutoff"))


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
    "parse-cutoff",
    "tapestri",
    "csv",
    "sparse",
    "sparse-assays",
]


@pytest.mark.parametrize("kind", _KINDS)
def test_every_per_cell_array_follows_the_shuffled_order(tmp_path, kind) -> None:
    run, expected = _source(tmp_path, kind)
    with _warnings() as plain_warnings:
        run(tmp_path / "plain.zarr", {"shuffle_cells": False})
    with _warnings() as shuffled_warnings:
        writer = run(
            tmp_path / "shuffled.zarr", {"shuffle_cells": True, "shuffle_seed": 11}
        )
    assert writer.shuffleCells is True and writer.shuffleSeed == 11
    plain = _open(tmp_path / "plain.zarr")
    n_cells = int(plain["cellData/ids"].shape[0])
    order = shuffled_order(n_cells, 11)
    file_rows = _source_rows(kind)
    assert file_rows.size == n_cells
    _assert_shuffled_store(
        plain, _open(tmp_path / "shuffled.zarr"), order, 11, expected, file_rows
    )
    # A source cell column named sourceRow imports only in the file's order.
    skipped = "Skipped source cell metadata column 'sourceRow' because CyteArc stores"
    if kind in ("parse", "parse-cutoff", "csv"):
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
def test_an_import_shuffles_with_seed_0_unless_told_not_to(tmp_path, kind) -> None:
    run, _expected = _source(tmp_path, kind)
    writer = run(tmp_path / "default.zarr", {})
    assert writer.shuffleCells is True and writer.shuffleSeed == 0
    run(tmp_path / "seed0.zarr", {"shuffle_cells": True, "shuffle_seed": 0})
    default = _store_files(tmp_path / "default.zarr")
    assert default == _store_files(tmp_path / "seed0.zarr")
    cells = _open(tmp_path / "default.zarr")["cellData"]
    assert recorded_shuffle_seed(cells) == 0
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    # Each cell's row in the file, in the order that seed 0 gives.
    np.testing.assert_array_equal(
        cells[SOURCE_ROW][...],
        _source_rows(kind)[shuffled_order(cells["ids"].shape[0], 0)],
    )

    # shuffle_cells=False writes the file's order, whatever the seed.
    plain = run(tmp_path / "plain.zarr", {"shuffle_cells": False})
    assert plain.shuffleCells is False
    run(tmp_path / "explicit.zarr", {"shuffle_cells": False, "shuffle_seed": 9})
    plain_files = _store_files(tmp_path / "plain.zarr")
    assert plain_files == _store_files(tmp_path / "explicit.zarr")
    assert plain_files != default
    cells = _open(tmp_path / "plain.zarr")["cellData"]
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
        CrToZarr(
            reader, str(tmp_path / "plain.zarr"), nthreads=1, shuffle_cells=False
        ).dump()
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
                {"shuffle_cells": False}
                if seed is None
                else {"shuffle_cells": True, "shuffle_seed": seed}
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
    run(tmp_path / "plain.zarr", {"shuffle_cells": False})
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
        np.arange(_N_CELLS),
    )
    assert _no_temporary_files(tmp_path)


# Shuffle arguments that an import refuses, the error and its message.
BAD_SHUFFLE_ARGUMENTS = [
    ({"shuffle_cells": "yes"}, TypeError, "shuffle_cells must be a boolean"),
    (
        {"shuffle_cells": True, "shuffle_seed": -1},
        ValueError,
        "shuffle_seed must be from 0 to 2**64 - 1, not -1",
    ),
    (
        {"shuffle_cells": True, "shuffle_seed": 2**64},
        ValueError,
        "shuffle_seed must be from 0 to 2**64 - 1, not 18446744073709551616",
    ),
    (
        {"shuffle_cells": True, "shuffle_seed": 1.5},
        TypeError,
        "shuffle_seed must be an integer",
    ),
]


@pytest.mark.parametrize(("options", "error", "message"), BAD_SHUFFLE_ARGUMENTS)
@pytest.mark.parametrize("kind", ["10x-filtered", "tapestri", "csv", "sparse"])
def test_bad_shuffle_arguments_are_refused_before_the_store_exists(
    tmp_path, kind, options, error, message
) -> None:
    run, _expected = _source(tmp_path, kind)
    with pytest.raises(error) as refused:
        run(tmp_path / "refused.zarr", options)
    assert str(refused.value) == message
    assert not (tmp_path / "refused.zarr").exists()
    assert _no_temporary_files(tmp_path)


def test_a_10x_budget_without_room_for_the_spill_is_refused_with_the_bytes_it_needs(
    tmp_path,
) -> None:
    counts = _counts()
    nonzeros = np.count_nonzero(counts, axis=1)
    rows, longest = _N_CELLS, int(nonzeros.max())
    # The order and its inverse are held beside the scan, and the reader's
    # arrays: the indexes of its barcodes, the file's row pointers and the
    # barcodes' cumulative values, 8 bytes each.
    held = 2 * 8 * rows
    arrays = 8 * rows + 8 * (rows + 1) + 8 * (rows + 1)
    budget = 8_000
    # Buckets read in a thirty-second of what the spill may hold, at 38 bytes
    # for each uint16 entry read and 80 for each row: one cell each at this
    # budget, and at the least budget that passes the spill.
    row_bytes = nonzeros.mean() * 38 + 80
    buckets = -(-rows // max(1, int(((budget - held - arrays) // 32) // row_bytes)))
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
        # The refusal states the import's budget: what the spill needs beside
        # the order, its inverse and the reader's arrays.
        assert buckets == rows and int((least // 32) // row_bytes) == 1
        needed = held + arrays + least
        item = (
            f"the row spill of a cell shuffle beside the {held} bytes of its cell order"
        )
        assert caught.value.needed_bytes == needed
        assert caught.value.budget_bytes == budget
        assert caught.value.item == item
        assert str(caught.value) == (
            f"Importing the counts of {rows} cells holds about {needed} bytes for "
            f"{item}, more than the memory budget of {budget} bytes. Raise "
            f"mem_budget to at least {needed} bytes."
        )
        assert reader._shuffledRows is None
        # With that much, the spill fits; a later step of so small a budget
        # refuses for itself.
        with pytest.raises(MemoryError) as caught:
            CrToZarr(
                reader,
                str(tmp_path / "admitted.zarr"),
                mem_budget=needed,
                nthreads=1,
                shuffle_cells=True,
            )
        assert "row spill" not in str(caught.value)
    finally:
        reader.close()
    assert not (tmp_path / "refused.zarr").exists()
    assert _no_temporary_files(tmp_path)


def _spill_attempts(
    tmp_path: Path, run: Callable[[Path, dict[str, Any]], Any]
) -> Callable[[int], None]:
    """Return a shuffled import at a budget that raises only the refusal of its spill.

    A later step of so small a budget refuses for itself, which passes.
    """
    attempts = iter(range(1_000))

    def at_budget(budget: int) -> None:
        destination = tmp_path / f"store-{next(attempts)}.zarr"
        try:
            run(destination, {"shuffle_cells": True, "mem_budget": budget})
        except MemoryBudgetError as error:
            if error.item.startswith("the row spill of a cell shuffle"):
                assert not destination.exists()
                raise
        except MemoryError:
            pass

    return at_budget


def _held_before_the_shuffle(kind: str) -> int:
    """Return the bytes of the arrays that an import of ``kind`` holds when it shuffles.

    A 10x HDF5 reader holds the indexes of the barcodes it keeps, the
    file's int64 row pointers and the kept cells' cumulative values, and a
    Matrix Market reader the indexes and cumulative values and, for an
    unordered file, each kept cell's next slot, all of 8 bytes, beside the
    file's barcodes. A Tapestri
    reader holds each cell's sample, barcode and ID as Python strings of
    ASCII, 41 bytes and one a character, each beside a reference of 8, and
    counts the cells' cumulative reads, 8 bytes a cell and one, before its
    scan reads them. A CSV import holds the reader's IDs and its own copy,
    NumPy text of 4 bytes a character of the longest ID, or without an ID
    column only the names that it makes, and the in-memory import its
    matrices and its copy of the cell IDs, as NumPy text too.
    """
    counts = _counts()
    kept = _kept(counts, cutoff=kind.endswith("cutoff")).size
    if kind.startswith("10x"):
        return 8 * kept + 8 * (counts.shape[0] + 1) + 8 * (kept + 1)
    if kind.startswith("parse"):
        # A Parse reader keeps the file's wells as ASCII strings, 41 bytes and
        # one a character each beside a reference of 8, its samples A, B or
        # C alike, and its qualities, gene counts and source rows of 8 bytes;
        # the kept cells' copies refer to the strings and copy the numbers.
        kept = _source_rows(kind).size
        n_file = counts.shape[0]
        wells = sum(8 + 41 + len(f"well-{row}") for row in range(n_file))
        cell_columns = n_file * (8 + 41 + 1) + 3 * 8 * n_file
        return 8 * kept + 8 * (kept + 1) + wells + cell_columns + 5 * 8 * kept
    if kind.startswith("mtx"):
        slots = 8 * (kept + 1) if "unordered" in kind else 0
        # The file's barcodes as ASCII strings of 41 bytes and one a
        # character, each beside a reference of 8, and a reference to each
        # kept one.
        barcodes = sum(8 + 41 + len(f"BC-{row}") for row in range(counts.shape[0]))
        return 8 * kept + 8 * (kept + 1) + slots + barcodes + 8 * kept
    if kind.startswith("tapestri"):
        truth = tapestri_truth()
        rows = _tapestri_rows(kind)
        texts = [
            *truth.samples[rows].tolist(),
            *truth.barcodes[rows].tolist(),
            *(truth.row_ids[row] for row in rows),
        ]
        return sum(8 + 41 + len(text) for text in texts) + 8 * (rows.size + 1)
    if kind == "csv":
        return 2 * 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
    if kind == "csv-without-ids":
        return 4 * len(f"cell_{_N_CELLS - 1}") * _N_CELLS
    matrices = (
        [csr_matrix(counts[:, :9])]
        if kind == "sparse"
        else [csr_matrix(counts[:, columns]) for columns in _ASSAY_COLUMNS.values()]
    )
    matrix_bytes = sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        for matrix in matrices
    )
    return matrix_bytes + 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS


@pytest.mark.parametrize(
    "kind",
    [
        "10x-filtered",
        "10x-cutoff",
        "mtx-cellMajor-filtered",
        "mtx-unordered-cutoff",
        "parse-cutoff",
        "tapestri",
        "csv",
        "csv-without-ids",
        "sparse",
        "sparse-assays",
    ],
)
def test_a_budget_without_room_for_the_shuffle_is_refused_before_it_runs(
    tmp_path, monkeypatch, kind
) -> None:
    run, expected = _source(tmp_path, kind)
    n_cells = next(iter(expected.values())).shape[0]
    held = _held_before_the_shuffle(kind)
    # Drawing the order and its inverse holds 24 bytes a cell beside them.
    needed = held + 24 * n_cells
    drawn = drawn_shuffles(monkeypatch)
    refused = tmp_path / "refused.zarr"
    with pytest.raises(MemoryBudgetError) as caught:
        run(refused, {"shuffle_cells": True, "mem_budget": needed - 1})
    refusal = caught.value
    assert (refusal.needed_bytes, refusal.budget_bytes) == (needed, needed - 1)
    assert str(refusal) == shuffle_refusal(n_cells, held, needed - 1)
    # The import is refused before it draws the order or writes anything.
    assert drawn == []
    assert not refused.exists()
    # With the budget it names, the import draws the order, and a later step
    # of so small a budget refuses for itself, if at all.
    try:
        run(tmp_path / "admitted.zarr", {"shuffle_cells": True, "mem_budget": needed})
    except MemoryError as later:
        assert "random order" not in str(later)
    assert drawn == [n_cells]
    assert _no_temporary_files(tmp_path)


def _planned_spill_budgets(monkeypatch) -> list[int]:
    """Record the budget of each spill plan that a 10x, MTX or Tapestri reader makes."""
    from cytearc.readers import cellranger, mtx

    budgets: list[int] = []

    def recorded(*args: Any, maxBytes: int, **options: Any) -> Any:
        budgets.append(int(maxBytes))
        return row_spill.row_spill_plan(*args, maxBytes=maxBytes, **options)

    for module in (cellranger, mtx):
        monkeypatch.setattr(module, "row_spill_plan", recorded)
    return budgets


@pytest.mark.parametrize("kind", ["10x-cutoff", "mtx-cellMajor-filtered", "tapestri-8"])
def test_a_shuffled_import_of_a_reader_names_the_budget_that_passes_its_spill(
    tmp_path, kind, monkeypatch
) -> None:
    run, expected = _source(tmp_path, kind)
    cells = next(iter(expected.values())).shape[0]
    at_budget = _spill_attempts(tmp_path, run)
    # The budget that the shuffle names, 24 bytes a cell beside the reader's
    # arrays, passes the shuffle and not the spill.
    start = _held_before_the_shuffle(kind) + 24 * cells
    with pytest.raises(MemoryBudgetError) as caught:
        at_budget(start)
    # The spill runs beside the order of the cells and its inverse, 8 bytes a
    # cell each, and the reader's arrays, so the refusal names the import's
    # budget beside them. Eight Tapestri cells hold rows wide enough that
    # the spill's buckets, a quarter of its budget, must read the longest.
    assert caught.value.item == (
        f"the row spill of a cell shuffle beside the {16 * cells} bytes of its "
        "cell order"
    )
    assert caught.value.budget_bytes == start
    needed = caught.value.needed_bytes
    assert needed > start
    # With the budget it names, the spill fits, planned within what the
    # budget leaves beside the order, its inverse and the reader's arrays;
    # one byte less is refused, naming the same budget.
    planned = _planned_spill_budgets(monkeypatch)
    at_budget(needed)
    assert planned[0] == needed - 16 * cells - _held_before_the_shuffle(kind)
    with pytest.raises(MemoryBudgetError) as below:
        at_budget(needed - 1)
    assert (below.value.needed_bytes, below.value.budget_bytes) == (needed, needed - 1)
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("longest", [50, 0])
def test_a_spill_plan_refuses_a_budget_whose_buckets_cannot_read_the_longest_row(
    longest,
) -> None:
    from cytearc.readers._row_spill import row_spill_plan

    # A row arrives whole in one bucket, which reads in a quarter of the
    # budget: each float64 value takes 3 * 8 + 32 bytes as it is read, and
    # the row 112 more for its piece and pointers; a row of no values reads
    # as one value. Three rows need less beside the buckets than this.
    read = max(1, longest) * (3 * 8 + 32) + 112
    destinations = np.array([2, 0, 1])

    def planned(budget: int) -> Any:
        return row_spill_plan(
            destinations,
            (3, 50),
            np.float64,
            entries=None,
            longest=longest,
            maxBytes=budget,
        )

    assert planned(4 * read).bucket_bytes() >= read
    with pytest.raises(MemoryBudgetError) as caught:
        planned(4 * read - 1)
    assert (caught.value.needed_bytes, caught.value.item) == (
        4 * read,
        "the row spill of a cell shuffle",
    )


def test_a_tapestri_scan_in_the_files_order_names_the_budget_beside_its_strings(
    tmp_path,
) -> None:
    run, expected = _source(tmp_path, "tapestri")
    n_cells = next(iter(expected.values())).shape[0]
    # The scan runs beside the reader's strings and the cells' cumulative
    # reads, and holds one dense row of every amplicon and antibody at 72
    # bytes a value.
    held = _held_before_the_shuffle("tapestri")
    features = sum(values.shape[1] for values in expected.values())
    needed = held + 72 * features
    attempts = iter(range(10))

    def at_budget(budget: int) -> None:
        destination = tmp_path / f"store-{next(attempts)}.zarr"
        try:
            run(destination, {"shuffle_cells": False, "mem_budget": budget})
        except MemoryBudgetError as error:
            if "dense row" in error.item:
                assert not destination.exists()
                raise
        except MemoryError:
            pass

    with pytest.raises(MemoryBudgetError) as caught:
        at_budget(1)
    item = (
        f"the {features} values of one dense row beside the {held} bytes of the "
        "reader's arrays"
    )
    assert (caught.value.item, caught.value.needed_bytes) == (item, needed)
    assert str(caught.value) == (
        f"Scanning the counts of {n_cells} cells holds about {needed} bytes for "
        f"{item}, more than the memory budget of 1 bytes. Raise mem_budget to at "
        f"least {needed} bytes."
    )
    # With the budget it names, the scan runs; one byte less is refused.
    at_budget(needed)
    with pytest.raises(MemoryBudgetError) as below:
        at_budget(needed - 1)
    assert below.value.needed_bytes == needed


def test_a_shuffled_csv_import_names_the_least_budget_that_passes_its_spill(
    tmp_path,
) -> None:
    run, _expected = _source(tmp_path, "csv")
    at_budget = _spill_attempts(tmp_path, run)
    # A budget sizes the spill's buckets, and larger buckets need less beside
    # them, so what the spill needs depends on the budget. Whatever budget it
    # refuses, the refusal names the least import budget that passes it. The
    # least budget tried passes the shuffle and the gather of the IDs in the
    # order: the reader's IDs, NumPy text of 4 bytes a character of cell-156,
    # and the import's copy twice, beside the order, 8 bytes a cell.
    ids = 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
    gathered = 3 * ids + 8 * _N_CELLS
    # Beside the spill, the import holds the order and its inverse, 8 bytes a
    # cell each, the reader's IDs and its copy, and the cell columns, each a
    # value and a missing mark a cell: the labels a, b and c as text of 4
    # bytes, the scores, counts and source rows of 8, and the flags of 1.
    columns = _N_CELLS * ((4 + 1) + 3 * (8 + 1) + (1 + 1))
    held = 16 * _N_CELLS + 2 * ids + columns
    named = set()
    for budget in (gathered, 20_000, 40_000):
        with pytest.raises(MemoryBudgetError) as caught:
            at_budget(budget)
        assert caught.value.item == (
            f"the row spill of a cell shuffle beside the {held} bytes of its cell "
            "order, cell IDs and cell columns"
        )
        assert caught.value.budget_bytes == budget
        named.add(caught.value.needed_bytes)
    (needed,) = named
    assert needed > 40_000
    # With the budget it names, the spill fits; one byte less is refused,
    # naming the same budget.
    at_budget(needed)
    with pytest.raises(MemoryBudgetError) as below:
        at_budget(needed - 1)
    assert (below.value.needed_bytes, below.value.budget_bytes) == (needed, needed - 1)
    assert str(below.value) == (
        f"Importing the counts of {_N_CELLS} cells holds about {needed} bytes for "
        f"{below.value.item}, more than the memory budget of {needed - 1} bytes. "
        f"Raise mem_budget to at least {needed} bytes."
    )
    assert _no_temporary_files(tmp_path)


def test_a_shuffled_csv_import_names_the_budget_that_gathers_its_ids_in_the_order(
    tmp_path, monkeypatch
) -> None:
    run, _expected = _source(tmp_path, "csv")
    # The reader's IDs and the import's copy, NumPy text of 4 bytes a
    # character of cell-156, and the order, 8 bytes a cell, beside a second
    # copy of the IDs in the order.
    ids = 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
    held = 2 * ids + 8 * _N_CELLS
    needed = held + ids
    drawn = drawn_shuffles(monkeypatch)
    refused = tmp_path / "refused.zarr"
    with pytest.raises(MemoryBudgetError) as caught:
        run(refused, {"shuffle_cells": True, "mem_budget": needed - 1})
    item = (
        f"the cell IDs in shuffled order beside the {held} bytes that the import holds"
    )
    assert (caught.value.item, caught.value.needed_bytes) == (item, needed)
    # The order was drawn, and the import was refused before its spill and
    # before its destination exists.
    assert drawn == [_N_CELLS]
    assert not refused.exists()
    assert _no_temporary_files(tmp_path)
    # With the budget it names, the import gathers the IDs, and a later step
    # of so small a budget refuses for itself, if at all.
    try:
        run(tmp_path / "admitted.zarr", {"shuffle_cells": True, "mem_budget": needed})
    except MemoryError as later:
        assert "cell IDs in shuffled order" not in str(later)
    assert _no_temporary_files(tmp_path)


def test_a_shuffled_in_memory_import_names_the_budget_that_gathers_its_ids_in_the_order(
    tmp_path,
) -> None:
    matrix = csr_matrix(
        np.random.default_rng(5).poisson(2.0, size=(30, 20)).astype(np.uint32) + 1
    )
    ids = [f"cell-{row:03d}" + "x" * 190 for row in range(30)]
    # The matrix holds 600 uint32 values and their int32 columns, and 31
    # int32 row pointers, the order 8 bytes a cell, and the import's copy of
    # the IDs NumPy text of 4 bytes a character of the longest, which the
    # import copies once more in the order for the cell table.
    id_bytes = 30 * 4 * len(ids[0])
    held = 600 * 4 + 600 * 4 + 31 * 4 + 8 * 30 + id_bytes
    needed = held + id_bytes

    def construct(destination: Path, budget: int, shuffle_cells: bool = True) -> None:
        SparseToZarr(
            matrix,
            str(destination),
            ids,
            [f"f{i}" for i in range(20)],
            mem_budget=budget,
            nthreads=1,
            shuffle_cells=shuffle_cells,
        )

    refused = tmp_path / "refused.zarr"
    with pytest.raises(MemoryBudgetError) as caught:
        construct(refused, needed - 1)
    item = (
        f"the cell IDs in shuffled order beside the {held} bytes of the matrix, the "
        "cell order and the cell IDs"
    )
    assert (caught.value.item, caught.value.needed_bytes) == (item, needed)
    assert not refused.exists()
    # With the budget it names, the import gathers the IDs, and a later step
    # refuses for itself, if at all. In the file's order, the import gathers
    # nothing, even at a budget below two copies of the IDs beside the matrix.
    for destination, budget, shuffle_cells in (
        (tmp_path / "admitted.zarr", needed, True),
        (tmp_path / "in-order.zarr", needed - 8 * 30 - 1, False),
    ):
        try:
            construct(destination, budget, shuffle_cells)
        except MemoryError as later:
            assert "cell IDs in shuffled order" not in str(later)


def test_a_shuffled_csv_import_refuses_an_occupied_destination_before_it_reads_the_rows(
    tmp_path, monkeypatch
) -> None:
    reader = _csv_reader(
        _write_csv(tmp_path / "counts.csv", _counts()), temp_dir=str(tmp_path)
    )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "notes.txt").write_text("not a store")
    drawn = drawn_shuffles(monkeypatch)

    def unread(*_args: Any) -> Any:
        raise AssertionError("the file's rows were read before the destination check")

    monkeypatch.setattr(CSVReader, "consume", unread)
    with pytest.raises(FileExistsError, match="is not empty"):
        CSVtoZarr(reader, str(occupied), "RNA", nthreads=1, shuffle_cells=True)
    # Neither the order nor the spill of the rows was started.
    assert drawn == []
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


@pytest.mark.parametrize("kind", ["10x", "mtx-unordered", "tapestri"])
def test_a_reader_refuses_destination_rows_that_are_no_permutation(
    tmp_path, kind
) -> None:
    make_reader, _writer, _options, _expected = _reusable_source(tmp_path, kind)
    reader = make_reader()
    n_cells = reader.nCells
    shape = (
        "destinationRows must hold one integer row for each of the "
        f"{n_cells} cells; found dtype {{}} and shape {{}}"
    )
    permutation = (
        "destinationRows must be a permutation of the rows, each row from 0 once"
    )
    try:
        for rows, message in (
            (np.arange(n_cells - 1), shape.format("int64", (n_cells - 1,))),
            (np.full(n_cells, 0.5), shape.format("float64", (n_cells,))),
            (np.zeros(n_cells, dtype=np.int64), permutation),
            (np.arange(n_cells) - 1, permutation),
        ):
            with pytest.raises(ValueError) as refused:
                reader.count_value_ranges(1 << 28, destinationRows=rows)
            assert str(refused.value) == message
            assert reader._shuffledRows is None
    finally:
        reader.close()
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
    # A default import shuffles too, and the refusal names the way out.
    with pytest.raises(TypeError, match="pass shuffle_cells=False"):
        CrToZarr(reader, str(tmp_path / "default.zarr"), nthreads=1)
    assert not (tmp_path / "default.zarr").exists()
    # Its first cells, which decide what the values are, come from consume.
    first = reader._file_first_rows(100, 1)
    np.testing.assert_array_equal(first.toarray(), values[:100])
    CrToZarr(
        reader,
        str(tmp_path / "plain.zarr"),
        nthreads=1,
        values="normalized",
        shuffle_cells=False,
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


def _file_barcodes(tmp_path: Path, kind: str) -> np.ndarray:
    """Read the barcodes of a source in the file's order, without CyteArc."""
    if kind == "10x-cutoff":
        with h5py.File(tmp_path / "matrix.h5", "r") as h5:
            return np.array([value.decode() for value in h5["matrix/barcodes"][:]])
    if kind == "parse-cutoff":
        return pd.read_csv(tmp_path / "parse" / "cell_metadata.csv")[
            "bc_wells"
        ].to_numpy(dtype=str)
    lines = (tmp_path / "mtx" / "barcodes.tsv").read_text().splitlines()
    return np.array(lines)


@pytest.mark.parametrize(
    "kind",
    ["10x-cutoff", "mtx-cellMajor-cutoff", "mtx-unordered-cutoff", "parse-cutoff"],
)
def test_a_reader_that_drops_cells_keeps_each_cells_row_in_the_file(
    tmp_path, kind
) -> None:
    run, _expected = _source(tmp_path, kind)
    run(tmp_path / "plain.zarr", {"shuffle_cells": False})
    run(tmp_path / "shuffled.zarr", {"shuffle_cells": True, "shuffle_seed": 3})
    barcodes = _file_barcodes(tmp_path, kind)
    shuffled = _open(tmp_path / "shuffled.zarr")
    rows = np.asarray(shuffled["cellData/sourceRow"][...])
    ids = np.asarray(shuffled["cellData/ids"][...])
    # Each cell is the barcode at its row of the file.
    np.testing.assert_array_equal(barcodes[rows], ids)
    # The rows of the dropped cells are missing, so the rows have gaps.
    assert rows.dtype == np.int64
    assert np.unique(rows).size == rows.size < barcodes.size
    assert int(rows.max()) > rows.size - 1
    # Sorting by sourceRow restores the order of an import in the file's order.
    plain_ids = np.asarray(_open(tmp_path / "plain.zarr")["cellData/ids"][...])
    restored = source_order_rows(shuffled["cellData"], np.ones(rows.size, dtype=bool))
    np.testing.assert_array_equal(ids[restored], plain_ids)
    np.testing.assert_array_equal(rows[restored], np.sort(rows))


def test_a_reader_that_names_too_few_file_rows_is_refused_before_it_is_read(
    tmp_path, monkeypatch
) -> None:
    reader = _10x_reader(tmp_path)
    scans: list[Any] = []
    monkeypatch.setattr(reader, "_file_rows", lambda: np.arange(3))
    monkeypatch.setattr(
        reader,
        "count_value_ranges",
        lambda *args, destinationRows=None: scans.append(1),
    )
    try:
        with pytest.raises(
            ValueError, match="one integer file row for each of the 157"
        ):
            CrToZarr(
                reader, str(tmp_path / "refused.zarr"), nthreads=1, shuffle_cells=True
            )
    finally:
        reader.close()
    assert scans == []
    assert not (tmp_path / "refused.zarr").exists()


def _reusable_source(
    tmp_path: Path, kind: str
) -> tuple[Callable[[], Any], type, dict[str, Any], dict[str, np.ndarray]]:
    """Return a reader factory of ``kind``, its writer, the writer's options, and its counts by assay."""
    counts = _counts()
    scratch = str(tmp_path)
    by_assay = {name: counts[:, columns] for name, columns in _ASSAY_COLUMNS.items()}
    if kind == "10x":
        path = _write_10x(tmp_path / "matrix.h5", counts)
        return (
            lambda: CrH5Reader(str(path), temp_dir=scratch),
            CrToZarr,
            {},
            by_assay,
        )
    if kind.startswith("mtx-"):
        directory = _write_mtx(tmp_path / "mtx", counts, kind.split("-")[1])
        return (
            lambda: MtxReader(inspect_mtx(directory)[0], temp_dir=scratch),
            CrToZarr,
            {"lines_in_mem": 50},
            by_assay,
        )
    assert kind == "tapestri"
    truth = tapestri_truth()
    path = write_tapestri(truth, tmp_path / "run.dna+protein.h5")
    return (
        lambda: TapestriReader(str(path), temp_dir=scratch),
        TapestriToZarr,
        {},
        {"CNV": truth.amplicon_reads, "ADT": truth.protein_reads},
    )


@pytest.mark.parametrize("shuffled_first", [True, False])
@pytest.mark.parametrize(
    "kind", ["10x", "mtx-cellMajor", "mtx-unordered", "mtx-featureMajor", "tapestri"]
)
def test_an_import_in_the_readers_order_reads_it_whatever_another_import_left(
    tmp_path, kind, shuffled_first
) -> None:
    make_reader, writer, options, expected = _reusable_source(tmp_path, kind)
    reader = make_reader()
    try:

        def shuffled() -> Any:
            return writer(
                reader,
                str(tmp_path / "shuffled.zarr"),
                nthreads=1,
                shuffle_cells=True,
                **options,
            )

        def plain() -> Any:
            return writer(
                reader,
                str(tmp_path / "plain.zarr"),
                nthreads=1,
                shuffle_cells=False,
                **options,
            )

        if shuffled_first:
            shuffled_writer = shuffled()
            plain_writer = plain()
        else:
            # Built before the shuffled import sends its cells, dumped after.
            plain_writer = plain()
            shuffled_writer = shuffled()
        plain_writer.dump()
        # The plain import removed the shuffled cells, which the shuffled
        # import cannot replace with the reader's order.
        with pytest.raises(RuntimeError, match="written once and removed"):
            shuffled_writer.dump()
    finally:
        reader.close()
    root = _open(tmp_path / "plain.zarr")
    for assay, values in expected.items():
        np.testing.assert_array_equal(root[f"{assay}/counts"][...], values)
    assert SOURCE_ROW not in root["cellData"]
    assert _no_temporary_files(tmp_path)


def test_a_shuffled_10x_import_refused_after_its_spill_leaves_no_cells_for_a_retry(
    tmp_path, monkeypatch
) -> None:
    def refuse(self, *_args: Any) -> None:
        raise OSError("the destination refused a write")

    counts = _counts()
    reader = _10x_reader(tmp_path)
    try:
        monkeypatch.setattr(CrToZarr, "_write_reader_metadata", refuse)
        with pytest.raises(OSError, match="refused a write"):
            CrToZarr(
                reader, str(tmp_path / "refused.zarr"), nthreads=1, shuffle_cells=True
            )
        assert reader._shuffledRows is None
        assert _no_temporary_files(tmp_path)
        monkeypatch.undo()
        CrToZarr(
            reader, str(tmp_path / "retry.zarr"), nthreads=1, shuffle_cells=False
        ).dump()
    finally:
        reader.close()
    root = _open(tmp_path / "retry.zarr")
    np.testing.assert_array_equal(
        root["cellData/ids"][...], [f"BC-{row}" for row in range(_N_CELLS)]
    )
    np.testing.assert_array_equal(root["RNA/counts"][...], counts[:, :9])


def test_interleaved_shuffled_10x_imports_never_write_each_others_cells(
    tmp_path,
) -> None:
    counts = _counts()
    reader = _10x_reader(tmp_path)
    try:
        first = CrToZarr(
            reader, str(tmp_path / "first.zarr"), nthreads=1, shuffle_cells=True
        )
        second = CrToZarr(
            reader,
            str(tmp_path / "second.zarr"),
            nthreads=1,
            shuffle_cells=True,
            shuffle_seed=1,
        )
        # The second import's cells replaced the first's, in another order.
        with pytest.raises(RuntimeError, match="another import of the reader"):
            first.dump()
        second.dump()
        with pytest.raises(RuntimeError, match="another import of the reader"):
            first.dump()
    finally:
        reader.close()
    order = shuffled_order(_N_CELLS, 1)
    assert not np.array_equal(order, shuffled_order(_N_CELLS, 0))
    root = _open(tmp_path / "second.zarr")
    np.testing.assert_array_equal(
        root["cellData/ids"][...], [f"BC-{row}" for row in order]
    )
    np.testing.assert_array_equal(root["RNA/counts"][...], counts[order][:, :9])
    assert _no_temporary_files(tmp_path)


def test_csv_imports_of_one_reader_each_hold_their_own_shuffled_rows(
    tmp_path,
) -> None:
    counts = _counts()
    path = _write_csv(tmp_path / "counts.csv", counts)
    reader = _csv_reader(path, temp_dir=str(tmp_path))
    shuffled = CSVtoZarr(
        reader, str(tmp_path / "shuffled.zarr"), "RNA", nthreads=1, shuffle_cells=True
    )
    # The shuffled rows are the writer's, so an import of the same reader in
    # the file's order reads the file, and the shuffled import still writes.
    CSVtoZarr(
        reader, str(tmp_path / "plain.zarr"), "RNA", nthreads=1, shuffle_cells=False
    ).dump()
    shuffled.dump()
    order = shuffled_order(_N_CELLS, 0)
    np.testing.assert_array_equal(
        _open(tmp_path / "plain.zarr")["RNA/counts"][...], counts[:, :9]
    )
    np.testing.assert_array_equal(
        _open(tmp_path / "shuffled.zarr")["RNA/counts"][...], counts[order][:, :9]
    )
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize(
    ("kind", "operation"),
    [("tapestri", "Shuffled Tapestri import"), ("csv", "Shuffled CSV import")],
)
def test_a_disk_without_room_for_dense_rows_is_refused_before_any_is_written(
    tmp_path, monkeypatch, kind, operation
) -> None:
    import cytearc.readers._sparse as sparse

    run, _expected = _source(tmp_path, kind)
    spilled: list[int] = []
    appended: list[int] = []
    original_finalize = SparseRowStore.finalize
    original_append = SparseRowStore.append_rows

    def finalize(self):
        original_finalize(self)
        spilled.append(int(self._fileBytes.sum()))

    def append_rows(self, rows, matrix):
        appended.append(int(matrix.nnz))
        return original_append(self, rows, matrix)

    monkeypatch.setattr(SparseRowStore, "finalize", finalize)
    monkeypatch.setattr(SparseRowStore, "append_rows", append_rows)
    run(tmp_path / "measured.zarr", {"shuffle_cells": True})
    written = sum(spilled)
    real_usage = sparse.shutil.disk_usage

    def usage(free: int):
        def disk_usage(path):
            found = real_usage(path)
            if "cytearc-sparse-" in str(path):
                return found._replace(free=free)
            return found

        return disk_usage

    appended.clear()
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(0))
    with pytest.raises(OSError, match=rf"^{operation} needs (\d+) temporary") as caught:
        run(tmp_path / "refused.zarr", {"shuffle_cells": True})
    # Refused from the first rows' estimate, before any row was sent.
    assert not appended
    required = int(str(caught.value).split(" needs ")[1].split()[0])
    # The estimate, from the values beside zeros of the first rows and their
    # encoding, covers what the spill wrote.
    assert 0 < written <= required
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(required))
    run(tmp_path / "admitted.zarr", {"shuffle_cells": True})
    assert not (tmp_path / "refused.zarr").exists()
    assert _no_temporary_files(tmp_path)
