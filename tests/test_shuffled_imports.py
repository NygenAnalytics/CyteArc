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
import re
import sys
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
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from cytearc.writers import (
    CrToZarr,
    CSVtoZarr,
    SparseToZarr,
    TapestriToZarr,
)
from tests.storage_helpers import drawn_shuffles, shuffle_refusal
from tests.variant_truth import tapestri_truth, write_tapestri
from tests.memory_helpers import NEEDS_NAMING_ONE_MIB, held_within_charge

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
            f"the row spill of a cell shuffle beside the {held} bytes of its cell "
            f"order and the {arrays} bytes of the reader's arrays"
        )
        assert caught.value.needed_bytes == needed
        assert caught.value.budget_bytes == budget
        assert caught.value.item == item
        assert needed <= NEEDS_NAMING_ONE_MIB
        assert str(caught.value) == (
            f"Importing the counts of {rows} cells holds about {needed} bytes for "
            f"{item}, more than the memory budget of {budget} bytes. Raise "
            "mem_budget to at least 1 MiB."
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
    reader holds each cell's sample, barcode and ID as Python strings, as
    Python sizes them, each beside a reference of 8, and counts the cells'
    cumulative reads, 8 bytes a cell and one, before its scan reads them. A CSV import holds the reader's IDs, NumPy text of 4
    bytes a character of the longest ID, or without an ID column the names
    that it makes, and the in-memory import its matrices and its copy of
    the cell IDs, as NumPy text too.
    """
    counts = _counts()
    kept = _kept(counts, cutoff=kind.endswith("cutoff")).size
    if kind.startswith("10x"):
        return 8 * kept + 8 * (counts.shape[0] + 1) + 8 * (kept + 1)
    if kind.startswith("parse"):
        # A Parse reader keeps the file's wells as strings, as Python sizes
        # them, each beside a reference of 8; its samples as references to
        # the three strings A, B and C, which pandas reads once each; and its
        # qualities, gene counts and source rows of 8 bytes. The kept cells'
        # copies refer to the strings and copy the numbers.
        kept = _source_rows(kind).size
        n_file = counts.shape[0]
        wells = sum(8 + sys.getsizeof(f"well-{row}") for row in range(n_file))
        samples = 8 * n_file + 3 * sys.getsizeof("A")
        cell_columns = samples + 3 * 8 * n_file
        return 8 * kept + 8 * (kept + 1) + wells + cell_columns + 5 * 8 * kept
    if kind.startswith("mtx"):
        slots = 8 * (kept + 1) if "unordered" in kind else 0
        # The file's barcodes as strings, as Python sizes them, each beside a
        # reference of 8, and a reference to each kept one.
        barcodes = sum(8 + sys.getsizeof(f"BC-{row}") for row in range(counts.shape[0]))
        return 8 * kept + 8 * (kept + 1) + slots + barcodes + 8 * kept
    if kind.startswith("tapestri"):
        truth = tapestri_truth()
        rows = _tapestri_rows(kind)
        texts = [
            *truth.samples[rows].tolist(),
            *truth.barcodes[rows].tolist(),
            *(truth.row_ids[row] for row in rows),
        ]
        return sum(8 + sys.getsizeof(str(text)) for text in texts) + 8 * (rows.size + 1)
    if kind == "csv":
        return 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
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
    # Drawing the order and its inverse holds 24 bytes a cell beside them; a
    # budget with room for half of the order is refused.
    below = held + 12 * n_cells
    if kind.startswith("sparse"):
        # The in-memory import first checks its cell IDs, which needs more
        # than the shuffle: a copy of their NumPy text, 4 bytes a character
        # of the longest, cell-156, and a mark each. At the budget that the
        # check names the shuffle passes.
        check = 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS + _N_CELLS
        assert check > 24 * n_cells
        below = held + check // 2
    drawn = drawn_shuffles(monkeypatch)
    refused = tmp_path / "refused.zarr"
    with pytest.raises(MemoryBudgetError) as caught:
        run(refused, {"shuffle_cells": True, "mem_budget": below})
    refusal = caught.value
    assert refusal.budget_bytes == below
    if kind.startswith("sparse"):
        matrices = "matrix" if kind == "sparse" else "matrices"
        ids = 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
        assert refusal.needed_bytes == held + check
        assert refusal.item == (
            f"checking the cell IDs, an array of {ids} bytes, beside the "
            f"{held - ids} bytes of the {matrices}"
        )
    else:
        # The refusal names what the import holds by its own estimate of the
        # readers' text, which the derivation above approximates, beside the
        # order.
        found = re.search(r"beside the (\d+) bytes", str(refusal))
        named = held if found is None else int(found[1])
        assert refusal.needed_bytes == named + 24 * n_cells > below
        assert str(refusal) == shuffle_refusal(n_cells, named, below)
    # The import is refused before it draws the order or writes anything.
    assert drawn == []
    assert not refused.exists()
    # With the budget it names, the import draws the order, and a later step
    # of so small a budget refuses for itself, if at all.
    try:
        run(
            tmp_path / "admitted.zarr",
            {"shuffle_cells": True, "mem_budget": refusal.needed_bytes},
        )
    except MemoryError as later:
        assert "random order" not in str(later)
        assert "checking the cell IDs" not in str(later)
    assert drawn == [n_cells]
    assert _no_temporary_files(tmp_path)


def test_a_10x_import_beside_a_reader_that_reports_no_arrays_names_only_its_order(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(CrH5Reader, "_sparse_import_resident_bytes", lambda _: 0)
    run, expected = _source(tmp_path, "10x-cutoff")
    cells = next(iter(expected.values())).shape[0]
    # In the file's order the import holds nothing beside the scan, so its
    # refusal is the reader's own.
    with pytest.raises(MemoryBudgetError) as plain:
        run(tmp_path / "plain.zarr", {"shuffle_cells": False, "mem_budget": 1})
    assert plain.value.item.endswith("values of the longest compressed vector")
    # Shuffled, the spill runs beside the order and its inverse alone, 8
    # bytes a cell each, at the budget that the shuffle names, 24 bytes a cell.
    with pytest.raises(MemoryBudgetError) as shuffled:
        _spill_attempts(tmp_path, run)(24 * cells)
    assert shuffled.value.item == (
        f"the row spill of a cell shuffle beside the {16 * cells} bytes of its "
        "cell order"
    )
    assert _no_temporary_files(tmp_path)


def test_a_shuffled_10x_import_of_many_cells_names_a_budget_that_passes_its_spill(
    tmp_path,
) -> None:
    n_cells = 3000
    path = _write_10x(tmp_path / "matrix.h5", _counts(n_cells))
    run = _cr_import(lambda: CrH5Reader(str(path), temp_dir=str(tmp_path)))
    at_budget = _spill_attempts(tmp_path, run)
    # The reader keeps every barcode: the indexes, the file's row pointers
    # and the cumulative values, 8 bytes each. Beside them the shuffle needs
    # 24 bytes a cell.
    arrays = 8 * n_cells + 8 * (n_cells + 1) + 8 * (n_cells + 1)
    with pytest.raises(MemoryBudgetError) as caught:
        at_budget(arrays + 24 * n_cells)
    needed = caught.value.needed_bytes
    # That budget sizes buckets of a cell each, which need more beside them
    # than the larger buckets of the budget that the refusal names, which
    # passes the spill.
    assert needed > arrays + 24 * n_cells
    at_budget(needed)
    assert _no_temporary_files(tmp_path)


def _many_cell_import(
    tmp_path: Path, kind: str
) -> Callable[[Path, dict[str, Any]], Any]:
    """Return an import of 3,000 cells of a 10x HDF5 file or a Matrix Market directory, or of the Tapestri truth."""
    if kind == "10x":
        path = _write_10x(tmp_path / "matrix.h5", _counts(3000))
        return _cr_import(lambda: CrH5Reader(str(path), temp_dir=str(tmp_path)))
    if kind == "mtx":
        directory = _write_mtx(tmp_path / "mtx", _counts(3000), "cellMajor")
        return _cr_import(
            lambda: MtxReader(inspect_mtx(directory)[0], temp_dir=str(tmp_path)),
            lines_in_mem=50,
        )
    path = write_tapestri(tapestri_truth(), tmp_path / "run.dna+protein.h5")
    return _cr_import(
        lambda: TapestriReader(str(path), temp_dir=str(tmp_path)), TapestriToZarr
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["10x", "mtx", "tapestri"])
def test_a_shuffled_import_names_a_budget_that_writes_its_count_layout(
    tmp_path, kind
) -> None:
    from tests.test_memory_refusals import named_budget_runs

    run = _many_cell_import(tmp_path, kind)
    attempts = iter(range(100))
    layout = "the default count-matrix policy"

    def at_budget(budget: int) -> None:
        destination = tmp_path / f"store-{next(attempts)}.zarr"
        try:
            run(destination, {"shuffle_cells": True, "mem_budget": budget})
        except MemoryBudgetError as error:
            if error.item == layout:
                assert not destination.exists()
                raise

    # The bytes that the refusals before the layout need lead to the budget
    # of the spill.
    budget = 1
    while True:
        try:
            run(
                tmp_path / f"climb-{budget}.zarr",
                {"shuffle_cells": True, "mem_budget": budget},
            )
        except MemoryBudgetError as error:
            if error.item == layout:
                break
            budget = error.needed_bytes
    # The spill sizes its buckets from the budget, and the counts write reads
    # a bucket at a time, so the layout needs more at a larger budget. Its
    # refusal names a budget whose own spill's buckets the layout fits: the
    # import passes the layout there.
    refusal = named_budget_runs(at_budget, budget, item=layout)
    assert refusal.needed_bytes > budget
    assert _no_temporary_files(tmp_path)


def _uneven_counts() -> np.ndarray:
    """3,000 cells of one value of 2,000 genes each, but every 500th, which holds all 2,000."""
    counts = np.zeros((3000, 2000), dtype=np.int64)
    counts[np.arange(3000), np.arange(3000) % 2000] = 1
    counts[::500] = 2
    return counts


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["10x", "10x-uneven", "mtx", "tapestri"])
def test_a_spills_reads_at_another_budget_are_those_of_a_spill_planned_there(
    tmp_path, kind
) -> None:
    from cytearc.storage.cell_order import destination_rows

    # A store of a spill planned at each budget, as the counts write reads
    # it, against what the reader of a spill at the first budget names for
    # that budget from its own row pointers alone. Uneven cells leave some
    # buckets of a budget too large to read, which the store splits.
    make, n_cells = {
        "10x": (
            lambda: CrH5Reader(
                str(_write_10x(tmp_path / "m.h5", _counts(3000))),
                temp_dir=str(tmp_path),
            ),
            3000,
        ),
        "10x-uneven": (
            lambda: CrH5Reader(
                str(_write_10x(tmp_path / "u.h5", _uneven_counts())),
                temp_dir=str(tmp_path),
            ),
            3000,
        ),
        "mtx": (
            lambda: MtxReader(
                inspect_mtx(_write_mtx(tmp_path / "mtx", _counts(3000), "cellMajor"))[
                    0
                ],
                temp_dir=str(tmp_path),
            ),
            3000,
        ),
        "tapestri": (
            lambda: TapestriReader(
                str(write_tapestri(tapestri_truth(), tmp_path / "run.dna+protein.h5")),
                temp_dir=str(tmp_path),
            ),
            None,
        ),
    }[kind]
    reader = make()
    try:
        n_cells = reader.nCells
        destinations = destination_rows(shuffled_order(n_cells, 0))
        groups = np.zeros(reader.nFeatures, dtype=np.int64)

        def spill(budget: int) -> None:
            if kind == "mtx":
                reader._set_sparse_import_lines_in_mem(50)
            reader.count_value_ranges(budget, groups, destinationRows=destinations)

        # From the least budget of the spill, which its refusal names, up.
        with pytest.raises(MemoryBudgetError) as least:
            spill(1)
        budgets = tuple(least.value.needed_bytes * scale for scale in (1, 3, 9, 81))
        spill(budgets[0])
        named = [reader._shuffled_staging_within(budget)(17) for budget in budgets]
        made = []
        split = []
        for budget in budgets:
            spill(budget)
            made.append(reader._shuffled_staging_bytes(17))
            planned_buckets = -(-n_cells // reader._shuffledPlanned(budget).bucketRows)
            split.append(reader._shuffledRows.bucket_starts.size - 1 > planned_buckets)
        # The budgets size buckets of more and more cells.
        assert len(set(made)) > 1
        assert named == made
        # The uneven cells' spills split buckets; the others' need not.
        assert any(split) == (kind == "10x-uneven")
    finally:
        reader.close()
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
    # budget beside them, the arrays by the reader's own estimate of its
    # text. Eight Tapestri cells hold rows wide enough that the spill's
    # buckets, a quarter of its budget, must read the longest.
    found = re.fullmatch(
        rf"the row spill of a cell shuffle beside the {16 * cells} bytes of its "
        r"cell order and the (\d+) bytes of the reader's arrays",
        caught.value.item,
    )
    assert found is not None
    arrays = int(found[1])
    assert caught.value.budget_bytes == start
    needed = caught.value.needed_bytes
    assert needed > start
    # With the budget it names, the spill fits, planned within what the
    # budget leaves beside the order, its inverse and the reader's arrays.
    planned = _planned_spill_budgets(monkeypatch)
    at_budget(needed)
    assert planned[0] == needed - 16 * cells - arrays
    assert _no_temporary_files(tmp_path)


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
@pytest.mark.parametrize(
    "kind", ["10x-cutoff", "mtx-cellMajor-filtered", "csv", "sparse"]
)
def test_an_import_charges_its_counts_t_write_with_what_it_still_holds(
    tmp_path, monkeypatch, kind, shuffle_cells
) -> None:
    from cytearc.storage import sharding

    charged: list[int] = []
    preflight = sharding.preflight_counts_t_spec

    def recorded(*args: Any, residentBytes: int = 0, **options: Any) -> Any:
        charged.append(int(residentBytes))
        return preflight(*args, residentBytes=residentBytes, **options)

    monkeypatch.setattr(sharding, "preflight_counts_t_spec", recorded)
    run, expected = _source(tmp_path, kind)
    run(tmp_path / "store.zarr", {"shuffle_cells": shuffle_cells})
    cells = next(iter(expected.values())).shape[0]
    # countsT runs beside the reader's arrays, a CSV reader's IDs, NumPy text
    # of 4 bytes a character of cell-156, or the in-memory matrix, as SciPy
    # holds it, and a shuffled import's order, 8 bytes a cell: the counts
    # write lets go of the shuffled cells and the count summaries first.
    if kind == "csv":
        reader = 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
    elif kind == "sparse":
        matrix = csr_matrix(_counts()[:, :9])
        reader = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    else:
        reader = _held_before_the_shuffle(kind)
    held = reader + (8 * cells if shuffle_cells else 0)
    # The layout fit plans the write and the write admits itself, both with
    # that charge.
    assert len(charged) >= 2
    assert set(charged) == {held}


def test_a_10x_import_lets_go_of_its_count_summaries_before_counts_t(
    tmp_path, monkeypatch
) -> None:
    import gc

    from cytearc.storage import sharding
    from cytearc.storage.identity import CountSummary

    run, _expected = _source(tmp_path, "10x-cutoff")
    summaries: list[int] = []
    preflight = sharding.preflight_counts_t_spec

    def recorded(*args: Any, **options: Any) -> Any:
        gc.collect()
        alive = sum(isinstance(item, CountSummary) for item in gc.get_objects())
        summaries.append(alive)
        return preflight(*args, **options)

    monkeypatch.setattr(sharding, "preflight_counts_t_spec", recorded)
    run(tmp_path / "store.zarr", {"shuffle_cells": True})
    # The layout fit plans countsT before the counts write makes a summary,
    # and the write admits it once the summaries are let go.
    assert summaries and summaries[-1] == 0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("longest", [50, 0])
def test_a_spill_plan_refuses_a_budget_whose_buckets_cannot_read_the_longest_row(
    longest, dtype
) -> None:
    from cytearc.readers._row_spill import row_spill_plan

    # A row arrives whole in one bucket, which reads in a quarter of the
    # budget: each value takes three times its bytes and 32 more as it is
    # read, 8 or 4 bytes for float64 or float32, and the row 112 more for
    # its piece and pointers; a row of no values reads as one value. Three
    # rows need less beside the buckets than this.
    itemsize = 8 if dtype is np.float64 else 4
    read = max(1, longest) * (3 * itemsize + 32) + 112
    destinations = np.array([2, 0, 1])

    def planned(budget: int) -> Any:
        return row_spill_plan(
            destinations,
            (3, 50),
            dtype,
            entries=None,
            longest=longest,
            maxBytes=budget,
        )

    assert planned(4 * read).bucket_bytes() >= read
    with pytest.raises(MemoryBudgetError) as caught:
        planned(2 * read)
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
    features = sum(values.shape[1] for values in expected.values())
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
    # The reader's strings are named by its own estimate of their text.
    item = caught.value.item
    found = re.fullmatch(
        rf"the {features} values of one dense row beside the (\d+) bytes of the "
        r"reader's arrays",
        item,
    )
    assert found is not None
    needed = int(found[1]) + 72 * features
    assert caught.value.needed_bytes == needed
    assert needed <= NEEDS_NAMING_ONE_MIB
    assert str(caught.value) == (
        f"Importing the counts of {n_cells} cells holds about {needed} bytes for "
        f"{item}, more than the memory budget of 1 bytes. Raise mem_budget to at "
        "least 1 MiB."
    )
    # With the bytes it needs, the scan runs.
    at_budget(needed)


def test_a_shuffled_csv_import_names_one_budget_that_passes_its_spill(
    tmp_path,
) -> None:
    run, _expected = _source(tmp_path, "csv")
    at_budget = _spill_attempts(tmp_path, run)
    # A budget sizes the spill's buckets, and larger buckets need less beside
    # them, so what the spill needs depends on the budget. Whatever budget it
    # refuses, the refusal names the same import budget, which passes it. The
    # least budget tried passes the shuffle: the reader's IDs, NumPy text of
    # 4 bytes a character of cell-156, beside 24 bytes a cell.
    ids = 4 * len(f"cell-{_N_CELLS - 1}") * _N_CELLS
    shuffled = ids + 24 * _N_CELLS
    # Beside the spill, the import holds the order and its inverse, 8 bytes a
    # cell each, the reader's IDs, and the cell columns, each a value and a
    # missing mark a cell: the labels a, b and c as text of 4 bytes, the
    # scores, counts and source rows of 8, and the flags of 1.
    columns = _N_CELLS * ((4 + 1) + 3 * (8 + 1) + (1 + 1))
    held = 16 * _N_CELLS + ids + columns
    named = set()
    for budget in (shuffled, 20_000, 40_000):
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
    assert needed <= NEEDS_NAMING_ONE_MIB
    assert str(caught.value) == (
        f"Importing the counts of {_N_CELLS} cells holds about {needed} bytes for "
        f"{caught.value.item}, more than the memory budget of 40000 bytes. "
        "Raise mem_budget to at least 1 MiB."
    )
    # With the bytes it needs, and so with the budget it names, the spill fits.
    at_budget(needed)
    assert _no_temporary_files(tmp_path)


def _replace_barcodes(path: Path, barcodes: Any, **options: Any) -> None:
    """Replace the barcodes of a 10x HDF5 file."""
    with h5py.File(path, "r+") as h5:
        del h5["matrix/barcodes"]
        h5["matrix"].create_dataset("barcodes", data=barcodes, **options)


@pytest.mark.parametrize("kind", ["10x-cutoff", "mtx-cellMajor-filtered"])
def test_a_shuffled_import_lets_go_of_its_cells_destinations_once_it_spills_them(
    tmp_path, monkeypatch, kind
) -> None:
    import gc
    import weakref

    import cytearc.readers.cellranger as cellranger
    import cytearc.readers.mtx as mtx
    import cytearc.storage.cell_order as cell_order
    import cytearc.storage.schema as schema

    # The destination row of each cell, 8 bytes a cell, as the import makes
    # it and as each reader checks it before its spill.
    made: list[weakref.ref[np.ndarray]] = []

    def recorded(make: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        def made_rows(*args: Any) -> np.ndarray:
            rows = make(*args)
            made.append(weakref.ref(rows))
            return rows

        return made_rows

    monkeypatch.setattr(
        cell_order, "destination_rows", recorded(cell_order.destination_rows)
    )
    for module in (cellranger, mtx):
        monkeypatch.setattr(
            module, "checked_destinations", recorded(module.checked_destinations)
        )
    held: list[bool] = []
    write = schema.create_cell_data

    def checked(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        held.extend(ref() is not None for ref in made)
        return write(*args, **kwargs)

    monkeypatch.setattr(schema, "create_cell_data", checked)
    run, _expected = _source(tmp_path, kind)
    run(tmp_path / "store.zarr", {"shuffle_cells": True})
    # By the cell table's write, after the counts layout's checks have
    # planned the buckets of other budgets, nothing holds them.
    assert len(made) >= 2
    assert held == [False] * len(made)


@pytest.mark.parametrize("cutoff", [False, True], ids=["filtered", "cutoff"])
def test_a_10x_reader_gives_its_kept_barcodes_as_the_file_holds_them(
    tmp_path, cutoff
) -> None:
    from tests.memory_helpers import traced_charge

    counts = _counts()
    path = _write_10x(tmp_path / "matrix.h5", counts)
    reader = CrH5Reader(str(path), is_filtered=not cutoff, filtering_cutoff=_CUTOFF)
    try:
        read = traced_charge(lambda _ledger: reader._cell_names_source()).held
        names, held = reader._cell_names_source()
    finally:
        reader.close()
    kept = _kept(counts, cutoff=cutoff)
    # Bytes of the file's width, six for BC-156, of the kept barcodes.
    np.testing.assert_array_equal(
        names, np.array([f"BC-{row}".encode() for row in kept], dtype="S6")
    )
    # Every barcode of the file is read, and a cutoff's kept ones copied
    # beside them.
    assert held == 6 * _N_CELLS + (6 * kept.size if cutoff else 0)
    # Within the few kilobytes of the objects of h5py's read.
    assert 6 * _N_CELLS <= read
    assert held_within_charge(read, held)


def test_a_10x_import_stores_barcodes_that_are_not_ascii_as_text(tmp_path) -> None:
    counts = _counts()
    path = _write_10x(tmp_path / "matrix.h5", counts)
    texts = [f"BC-é{row}" for row in range(_N_CELLS)]
    _replace_barcodes(path, np.array([text.encode() for text in texts]))
    reader = CrH5Reader(str(path))
    try:
        names, held = reader._cell_names_source()
    finally:
        reader.close()
    # The file's bytes, 8 for BC-é156, which the cell table decodes.
    assert (names.dtype, held) == (np.dtype("S8"), 8 * _N_CELLS)
    run = _cr_import(lambda: CrH5Reader(str(path), temp_dir=str(tmp_path)))
    run(tmp_path / "store.zarr", {"shuffle_cells": True})
    table = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")["cellData"]
    order = table["sourceRow"][:]
    assert table["ids"].dtype == np.dtype("U7")
    assert table["ids"][:].tolist() == [texts[row] for row in order]
    assert table["names"][:].tolist() == [texts[row] for row in order]


def test_a_10x_reader_of_barcodes_of_variable_length_decodes_them_into_a_new_array(
    tmp_path,
) -> None:
    counts = _counts()
    path = _write_10x(tmp_path / "matrix.h5", counts)
    texts = [f"BC-{row}" for row in range(_N_CELLS)]
    _replace_barcodes(path, texts, dtype=h5py.string_dtype())
    reader = CrH5Reader(str(path))
    try:
        names, held = reader._cell_names_source()
    finally:
        reader.close()
    np.testing.assert_array_equal(names, np.array(texts))
    # Text of 4 bytes a character of BC-156, which the import holds.
    assert held == names.nbytes == 4 * 6 * _N_CELLS


def _recorded_checks(monkeypatch) -> list[tuple[str, int]]:
    """Record the step and the need of each check that an import makes of its ledger's budget."""
    checks: list[tuple[str, int]] = []
    refuse_beyond = MemoryLedger.refuse_beyond

    def recorded(
        ledger: MemoryLedger, item: str, needed: int, remedy: str | None = None
    ) -> None:
        checks.append((item, int(needed)))
        refuse_beyond(ledger, item, needed, remedy)

    monkeypatch.setattr(MemoryLedger, "refuse_beyond", recorded)
    return checks


def _encoded(chunk: int) -> int:
    """A codec's buffer of a chunk of ``chunk`` bytes: at most a 128th more and 1,024 bytes."""
    return chunk + chunk // 128 + 1_024


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
@pytest.mark.parametrize(
    "kind",
    [
        "10x-filtered",
        "10x-cutoff",
        "mtx-cellMajor-filtered",
        "csv",
        "csv-without-ids",
        "sparse",
    ],
)
def test_an_import_charges_its_cell_table_write_beside_what_it_holds(
    tmp_path, monkeypatch, kind, shuffle_cells
) -> None:
    run, expected = _source(tmp_path, kind)
    checks = _recorded_checks(monkeypatch)
    run(tmp_path / "store.zarr", {"shuffle_cells": shuffle_cells})
    n_cells = next(iter(expected.values())).shape[0]
    counts = _counts()
    # The table's text: the barcodes BC-0 to BC-156 that a 10x reader keeps,
    # bytes of six characters in the file, or the IDs cell-0 to cell-156,
    # or cell_0 to cell_156 that a CSV import makes, NumPy text of 4 bytes
    # a character of the longest, in one chunk of every cell's text.
    kept = _kept(counts, cutoff=kind.endswith("cutoff"))
    texts = (
        [f"BC-{row}" for row in kept]
        if kind.startswith(("10x", "mtx"))
        else [f"cell-{row}" for row in range(_N_CELLS)]
    )
    stored = 4 * max(map(len, texts))
    write = _encoded(n_cells * stored)
    order = 8 * n_cells if shuffle_cells else 0
    # The shuffled cells' row pointers, 8 bytes a cell and one.
    pointers = 8 * (n_cells + 1) if shuffle_cells else 0
    if kind.startswith("10x"):
        # The import reads every barcode of the file, bytes of six characters
        # at the longest, beside a copy of a cutoff's kept ones. A block of
        # them, gathered in a shuffled import's order, becomes text twice, as
        # made and as stored, and the stored text is kept as Zarr writes it.
        gather = 6 if shuffle_cells else 0
        table = max(n_cells * (gather + 2 * stored), write + n_cells * stored)
        names = 6 * _N_CELLS + (6 * n_cells if kind.endswith("cutoff") else 0)
        arrays = _held_before_the_shuffle(kind) + pointers
        item = "the cell table's IDs and names beside " + (
            f"the {order} bytes of its cell order, " if shuffle_cells else ""
        )
        item += (
            f"the {arrays} bytes of the reader's arrays and the {names} bytes of "
            "the cells' names read for it"
        )
        held = order + arrays + names
    elif kind.startswith("mtx"):
        # The reader's barcodes are Python strings: a gathered block holds a
        # reference to each and its text twice, as made and as stored.
        gather = 8 if shuffle_cells else 0
        table = max(n_cells * (gather + 2 * stored), write + n_cells * stored)
        arrays = _held_before_the_shuffle(kind) + pointers
        item = "the cell table's IDs and names beside " + (
            f"the {order} bytes of its cell order and " if shuffle_cells else ""
        )
        item += f"the {arrays} bytes of the reader's arrays"
        held = order + arrays
    else:
        # NumPy text, which an import in the file's order lends to Zarr and
        # a shuffled one gathers in the order.
        table = write + (n_cells * stored if shuffle_cells else 0)
        ids = n_cells * stored
        if kind == "sparse":
            matrix = _held_before_the_shuffle(kind) - ids
            held = matrix + order + ids
            what = (
                "the matrix, the cell order and the cell IDs"
                if shuffle_cells
                else "the matrix and the cell IDs"
            )
            item = f"the cell table's IDs and names beside the {held} bytes of {what}"
        else:
            # Beside the reader's IDs, or the names that the import makes, a
            # shuffled import holds its order, its rows' pointers and the
            # cell columns, a value and a missing mark a cell: labels as text
            # of 4 bytes, scores, counts and source rows of 8, and flags of 1.
            columns = n_cells * ((4 + 1) + 3 * (8 + 1) + (1 + 1))
            held = ids + (order + pointers + columns if shuffle_cells else 0)
            item = (
                f"the cell table's IDs and names beside the {held} bytes that the "
                "import holds"
            )
    assert (item, held + table) in checks


# IDs of 1,999 characters at the longest, cell-156- and 1,990 more, are 7,996
# bytes each in a cell table, whose 157 rows are one chunk of 1,255,372 bytes
# that a codec encodes into a buffer of at most 1,266,203.
_WIDE_IDS = [f"cell-{row}-" + "x" * 1_990 for row in range(_N_CELLS)]
_WIDE_STORED = 4 * 1_999


def _wide_id_import(tmp_path: Path, kind: str) -> Callable[[Path, dict[str, Any]], Any]:
    """Return an import of the counts whose cells' IDs or barcodes are :data:`_WIDE_IDS`."""
    counts = _counts()
    if kind == "10x":
        path = _write_10x(tmp_path / "matrix.h5", counts)
        with h5py.File(path, "r+") as h5:
            del h5["matrix/barcodes"]
            h5["matrix"].create_dataset(
                "barcodes", data=np.array([text.encode() for text in _WIDE_IDS])
            )
        return _cr_import(lambda: CrH5Reader(str(path), temp_dir=str(tmp_path)))
    lines = ["cell," + ",".join(f"g{column}" for column in range(_N_FEATURES))]
    lines += [
        f"{_WIDE_IDS[row]}," + ",".join(str(value) for value in counts[row])
        for row in range(_N_CELLS)
    ]
    path = tmp_path / "wide.csv"
    path.write_text("\n".join(lines) + "\n")

    def csv_import(destination: Path, options: dict[str, Any]) -> Any:
        writer = CSVtoZarr(
            CSVReader(str(path), id_column=0, temp_dir=str(tmp_path)),
            str(destination),
            "RNA",
            **{"mem_budget": "256M", "nthreads": 1, **options},
        )
        writer.dump()
        return writer

    return csv_import


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
@pytest.mark.parametrize("kind", ["10x", "csv"])
def test_an_import_of_wide_ids_names_the_budget_that_writes_its_cell_table(
    tmp_path, kind, shuffle_cells
) -> None:
    from tests.test_memory_refusals import named_budget_runs

    run = _wide_id_import(tmp_path, kind)
    attempts = iter(range(100))
    chunk = _N_CELLS * _WIDE_STORED
    if kind == "10x":
        # The reader's arrays: the kept barcodes' indexes, the file's row
        # pointers and the cumulative values, 8 bytes each, and a shuffle's
        # row pointers of the shuffled cells beside its order, 8 bytes a cell.
        arrays = 8 * _N_CELLS + 2 * 8 * (_N_CELLS + 1)
        # The import reads the file's barcodes, bytes of 1,999 characters at
        # the longest.
        names = _N_CELLS * 1_999
        if shuffle_cells:
            arrays += 8 * (_N_CELLS + 1)
            item = (
                f"the cell table's IDs and names beside the {8 * _N_CELLS} bytes "
                f"of its cell order, the {arrays} bytes of the reader's arrays "
                f"and the {names} bytes of the cells' names read for it"
            )
            held = 8 * _N_CELLS + arrays + names
        else:
            item = (
                "the cell table's IDs and names beside the "
                f"{arrays} bytes of the reader's arrays and the {names} bytes of "
                "the cells' names read for it"
            )
            held = arrays + names
        # The barcodes, gathered in a shuffled import's order, become text
        # twice, as made and as stored, which outweighs the stored text that
        # Zarr writes beside the chunk's buffer; in the file's order Zarr's
        # write outweighs the two copies.
        gather = 1_999 if shuffle_cells else 0
        made = _N_CELLS * (gather + 2 * _WIDE_STORED)
        assert (made > _encoded(chunk) + chunk) == shuffle_cells
        table = max(made, _encoded(chunk) + chunk)
    else:
        # The reader's IDs and a shuffle's order, the shuffled rows' pointers
        # and its gathered IDs, which the file's order lends to Zarr instead.
        held = chunk + (8 * _N_CELLS + 8 * (_N_CELLS + 1) if shuffle_cells else 0)
        item = (
            f"the cell table's IDs and names beside the {held} bytes that the "
            "import holds"
        )
        table = _encoded(chunk) + (chunk if shuffle_cells else 0)

    def at_budget(budget: int) -> None:
        destination = tmp_path / f"store-{next(attempts)}.zarr"
        try:
            run(destination, {"shuffle_cells": shuffle_cells, "mem_budget": budget})
        except MemoryBudgetError as error:
            if error.item.startswith("the cell table's"):
                assert not destination.exists()
                raise

    # A budget with room for half of the table beside what the import holds.
    refusal = named_budget_runs(at_budget, held + table // 2, item=item)
    assert refusal.needed_bytes == held + table
    assert _no_temporary_files(tmp_path)


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
            # The plain import let go of the shuffled cells before it scanned
            # the file beside the reader's arrays alone.
            assert reader._shuffledRows is None
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
