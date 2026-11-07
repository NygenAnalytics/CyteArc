"""Scan an H5AD matrix's values in processes of their own, as one process does."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.readers import H5adReader
from cytearc.storage.io_policy import StorageIoPolicy
from cytearc.utils import count_values
from cytearc.utils.count_values import CountValueRange, merge_count_scans
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import H5adToZarr
from tests.test_import_batching import _write_h5ad

# The import batching files type even features as genes and odd ones as
# antibodies, which a split stores as RNA and ADT.
_GROUPS = np.arange(10) % 2


def _values(kind: str) -> np.ndarray:
    """Return 23 cells by 10 features of counts, with two empty cells and an empty feature.

    One antibody count needs uint32. ``"fractional"`` puts 2.5 among the genes
    and ``"negative"`` puts -3 among them, so the two groups differ.
    """
    values = np.random.default_rng(4).poisson(0.8, size=(23, 10)).astype(np.float32)
    values[[2, 11], :] = 0
    values[:, 6] = 0
    values[20, 3] = 70_000
    if kind == "fractional":
        values[15, 4] = 2.5
    elif kind == "negative":
        values[9, 0] = -3.0
    return values


def _expected_ranges(values: np.ndarray, groups: np.ndarray | None) -> list:
    """Return each group's range from all its values in one block, by NumPy."""
    selections = (
        [np.ones(values.shape[1], dtype=bool)]
        if groups is None
        else [groups == code for code in range(int(groups.max()) + 1)]
    )
    expected = []
    for selected in selections:
        value_range = CountValueRange()
        value_range.update(values[:, selected])
        expected.append(value_range)
    return expected


def _vectors(encoding: str, values: np.ndarray) -> int:
    return values.shape[1] if encoding == "csc" else values.shape[0]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("kind", ["counts", "fractional", "negative"])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_windows_starting_and_ending_at_every_vector_merge_into_the_whole_scan(
    tmp_path: Path, encoding: str, kind: str, grouped: bool
) -> None:
    values = _values(kind)
    path = tmp_path / "values.h5ad"
    _write_h5ad(path, values, encoding)
    groups = _GROUPS if grouped else None
    expected = _expected_ranges(values, groups)
    n_vectors = _vectors(encoding, values)
    reader = H5adReader(str(path))
    try:
        whole = reader._scan_values(0, n_vectors, 1 << 20, groups)
        assert whole.valueRanges == expected
        assert [r.largest for r in whole.valueRanges] == [r.largest for r in expected]
        middle = n_vectors // 2
        splits = [[(0, cut), (cut, n_vectors)] for cut in range(1, n_vectors)] + [
            [(0, cut), (cut, middle), (middle, n_vectors)] for cut in range(1, middle)
        ]
        for windows in splits:
            scans = [reader._scan_values(a, b, 1 << 20, groups) for a, b in windows]
            merged = merge_count_scans(scans[::-1])
            assert merged.valueRanges == expected, windows
            assert [r.largest for r in merged.valueRanges] == [
                r.largest for r in expected
            ]
            assert merged.canonical is True
            if encoding == "dense":
                np.testing.assert_array_equal(
                    merged.rowNonzeros, np.count_nonzero(values, axis=1)
                )
            else:
                assert merged.rowNonzeros is None
    finally:
        reader.close()


def _spy_on_processes(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[int, int]]]:
    """Scan every matrix in processes when asked, and record their windows."""
    monkeypatch.setattr(count_values, "_SCAN_PROCESS_VALUES", 0)
    seen: list[list[tuple[int, int]]] = []
    original = H5adReader._scan_in_processes

    def spy(self, windows, maxBytes, featureGroups):
        seen.append(list(windows))
        return original(self, windows, maxBytes, featureGroups)

    monkeypatch.setattr(H5adReader, "_scan_in_processes", spy)
    return seen


def _assert_balanced_windows(
    windows: list[tuple[int, int]], encoding: str, values: np.ndarray
) -> None:
    """Check three windows that cover every vector, each with a third of the values.

    A dense matrix, stored without HDF5 chunks, splits its 23 rows at rows 7
    and 15, the first rows at or past a third and two thirds of them. A
    sparse window holds a third of the stored values to within its longest
    vector.
    """
    if encoding == "dense":
        assert windows == [(0, 7), (7, 15), (15, 23)]
        return
    stored = np.count_nonzero(values, axis=0 if encoding == "csc" else 1)
    assert len(windows) == 3
    assert windows[0][0] == 0 and windows[-1][1] == len(stored)
    assert all(before[1] == after[0] for before, after in zip(windows, windows[1:]))
    for start, stop in windows:
        assert abs(stored[start:stop].sum() - stored.sum() / 3) <= stored.max()


@pytest.mark.parametrize("kind", ["fractional", "negative"])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_scans_in_processes_equal_the_scan_in_this_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    kind: str,
) -> None:
    values = _values(kind)
    path = tmp_path / "values.h5ad"
    _write_h5ad(path, values, encoding)
    if encoding == "csr":
        # One row of the last window holds its indices in reverse order.
        with h5py.File(path, "r+") as h5:
            indptr = h5["X/indptr"][:]
            start, stop = int(indptr[21]), int(indptr[22])
            assert stop - start > 1
            for name in ("indices", "data"):
                h5[f"X/{name}"][start:stop] = h5[f"X/{name}"][start:stop][::-1]
    seen = _spy_on_processes(monkeypatch)
    results = []
    for processes in (1, 3):
        reader = H5adReader(str(path))
        try:
            ranges = reader.count_value_ranges(1 << 30, _GROUPS, processes=processes)
            results.append(
                (
                    ranges,
                    [value_range.largest for value_range in ranges],
                    reader._cumulativeRowNnz,
                    reader._canonicalRows,
                )
            )
        finally:
            reader.close()

    assert len(seen) == 1
    _assert_balanced_windows(seen[0], encoding, values)
    (ranges, largest, row_counts, canonical), in_processes = results
    assert ranges == _expected_ranges(values, _GROUPS)
    assert in_processes[0] == ranges
    assert in_processes[1] == largest
    assert in_processes[3] is canonical
    if encoding == "dense":
        np.testing.assert_array_equal(in_processes[2], row_counts)
        np.testing.assert_array_equal(
            np.diff(row_counts), np.count_nonzero(values, axis=1)
        )
    else:
        assert in_processes[2] is row_counts is None
    assert canonical is (False if encoding == "csr" else None)


@pytest.mark.parametrize("encoding", ["csr", "dense"])
def test_a_nan_in_any_window_raises_as_the_scan_in_this_process_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    values = _values("counts")
    values[21, 5] = np.nan
    path = tmp_path / "values.h5ad"
    _write_h5ad(path, values, encoding)
    seen = _spy_on_processes(monkeypatch)
    for processes in (1, 3):
        reader = H5adReader(str(path))
        try:
            with pytest.raises(
                ValueError,
                match="^Count matrices must hold finite values; found NaN or infinity$",
            ):
                reader.count_value_ranges(1 << 30, processes=processes)
        finally:
            reader.close()
    assert len(seen) == 1 and len(seen[0]) == 3


def test_a_budget_without_room_for_two_processes_scans_in_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _values("counts")
    path = tmp_path / "values.h5ad"
    _write_h5ad(path, values, "csr")
    seen = _spy_on_processes(monkeypatch)
    window_bytes = count_values._SCAN_BYTES_PER_VALUE * count_values._SCAN_WINDOW_VALUES
    reader = H5adReader(str(path))
    try:
        reader.count_value_ranges(2 * window_bytes - 1, processes=4)
        assert seen == []
        # A row that does not fit the budget is refused as one process
        # refuses it.
        with pytest.raises(MemoryBudgetError, match="One compressed vector holds"):
            reader.count_value_ranges(count_values._SCAN_BYTES_PER_VALUE, processes=4)
        with pytest.raises(ValueError, match="processes must be at least 1"):
            reader.count_value_ranges(1 << 30, processes=0)
        with pytest.raises(TypeError, match="processes must be an integer"):
            reader.count_value_ranges(1 << 30, processes=True)
    finally:
        reader.close()
    assert seen == []


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_a_matrix_of_one_row_or_column_scans_in_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    # One cell, or for CSC one feature, cannot be split into windows.
    values = _values("counts")
    single = values[:1] if encoding != "csc" else values[:, 3:4]
    path = tmp_path / "values.h5ad"
    _write_h5ad(path, single, encoding)
    seen = _spy_on_processes(monkeypatch)
    reader = H5adReader(str(path))
    try:
        ranges = reader.count_value_ranges(1 << 30, processes=3)
    finally:
        reader.close()
    assert seen == []
    assert ranges == _expected_ranges(single, None)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_h5ad_imports_store_the_same_counts_whether_one_process_or_three_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str, split: bool
) -> None:
    values = _values("counts")
    path = tmp_path / "values.h5ad"
    _write_h5ad(path, values, encoding)
    seen = _spy_on_processes(monkeypatch)
    imported = []
    for io in (None, StorageIoPolicy(readWorkers=1)):
        store = MemoryStore()
        reader = H5adReader(str(path), feature_name_key="feature_name")
        try:
            writer = H5adToZarr(
                reader,
                store,
                assay_split_key="feature_types" if split else None,
                mem_budget="1G",
                nthreads=3,
                io=io,
            )
            dtypes = dict(writer.storageDtypes)
            writer.dump()
        finally:
            reader.close()
        root = zarr.open_group(store=store, mode="r")
        imported.append(
            (
                dtypes,
                {
                    name: (
                        np.asarray(root[f"{name}/counts"][:]),
                        root[f"{name}/counts"].attrs["content_fingerprint"],
                    )
                    for name in dtypes
                },
            )
        )

    # Three workers scan in three processes; a single read worker scans here.
    assert [len(windows) for windows in seen] == [3]
    (in_processes, stored), (in_this_process, expected) = imported
    assert in_processes == in_this_process
    assert in_processes == (
        {"RNA": np.dtype(np.uint8), "ADT": np.dtype(np.uint32)}
        if split
        else {"RNA": np.dtype(np.uint32)}
    )
    for name, (counts, fingerprint) in stored.items():
        np.testing.assert_array_equal(counts, expected[name][0])
        assert fingerprint == expected[name][1]
    columns = {"RNA": _GROUPS == 0, "ADT": _GROUPS == 1} if split else None
    for name, (counts, _fingerprint) in stored.items():
        source = values if columns is None else values[:, columns[name]]
        np.testing.assert_array_equal(counts, source)
