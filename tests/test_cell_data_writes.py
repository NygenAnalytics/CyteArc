"""Text arrays of cell-data artifacts, written a stored chunk at a time.

``write_cell_data_artifact`` stores text as master's whole-column write did,
``values.astype(str)`` at the width of the widest value, but converts and
writes one stored chunk at a time, so a write holds what
``cell_data_write_bytes`` states, rather than three copies of the column.
"""

import contextlib
import hashlib
import io
import itertools
from pathlib import Path

import numpy as np
import pytest
import zarr

from cytearc.metadata.artifacts import (
    cell_data_write_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from cytearc.storage.arrays import create_metadata_column, zarr_write_buffer_bytes
from cytearc.storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from cytearc.storage.refs import artifact_path
from cytearc.storage.selections import resolve_generated_selection_artifact
from tests.memory_helpers import cost_bound, hold_previous_write, traced_peak

# The cells of the small store; arrays of other lengths are written beside a
# per-cell array of this many rows.
_N_CELLS = 3
# The rows of each stored chunk of an artifact's arrays.
_CHUNK_ROWS = 100_000
_planned = itertools.count()


def _root(path: Path, n_cells: int = _N_CELLS, zarr_format: int = 3):
    """A store of ``n_cells`` cells, with a selection of all of them."""
    root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    cell_data = root.create_group("cellData")
    ids = np.asarray([f"cell-{index}" for index in range(n_cells)])
    rows = max(1, min(n_cells, _CHUNK_ROWS))
    cell_data.create_array("ids", data=ids, chunks=(rows,))
    selection = np.ones(n_cells, dtype=bool)
    cell_data.create_array("I", data=selection, chunks=(rows,))
    ref = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]
    return root, ref


def _plan(root, selection, n_cells: int):
    """Plan a new clones artifact of ``n_cells`` cells, with codes per cell."""
    return plan_cell_data_artifact(
        root,
        scope="datastore",
        kind="clones",
        operation="probe_writes",
        parameters={"write": next(_planned)},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={"codes": ((n_cells,), "i")},
    )


def _write(root, selection, n_cells: int, arrays: dict[str, np.ndarray]):
    """Write ``arrays`` beside per-cell codes, and return the artifact."""
    planned = _plan(root, selection, n_cells)
    write_cell_data_artifact(
        root, planned, {"codes": np.zeros(n_cells, dtype=np.int32), **arrays}
    )
    return planned.ref


def _stored(store: Path, ref: ArtifactRef, name: str) -> dict[str, str]:
    """The files of array ``name`` of an artifact of the store at ``store``."""
    return _files(store / artifact_path(ref) / name)


def _files(folder: Path) -> dict[str, str]:
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _whole_column(path: Path, values: np.ndarray, zarr_format: int) -> dict[str, str]:
    """The files of ``values`` written as master's writer wrote text, whole."""
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    create_metadata_column(
        group,
        "labels",
        data=values.astype(str),
        overwrite=True,
        chunkSize=min(max(len(values), 1), _CHUNK_ROWS),
    )
    return _files(path / "labels")


def _labels(rows: int, kind: str) -> np.ndarray:
    """Clone-like labels of 40 characters, which compress well."""
    lead = {"ascii": "c", "two-byte": "é", "four-byte": "\U0001f600"}[kind]
    distinct = np.asarray(
        [f"{lead * 34}{index:06d}" for index in range(min(rows, 5_000))], dtype=object
    )
    return distinct[np.arange(rows) % len(distinct)]


def _random_labels(rows: int, width: int, low: int, high: int, seed: int = 0):
    """Labels of random characters in [low, high), which hardly compress."""
    rng = np.random.default_rng(seed)
    distinct = min(rows, 20_000)
    codes = rng.integers(low, high, size=(distinct, width), dtype=np.uint32)
    codes[(codes >= 0xD800) & (codes < 0xE000)] = 0x41
    text = codes.view(f"U{width}").reshape(distinct).astype(object)
    return text[rng.integers(0, distinct, rows)]


_TEXT = {
    "ASCII": lambda rows: _labels(rows, "ascii"),
    "two-byte": lambda rows: _labels(rows, "two-byte"),
    "four-byte": lambda rows: _labels(rows, "four-byte"),
    "with missing labels": lambda rows: np.where(
        np.arange(rows) % 3 == 0, "", _labels(rows, "ascii")
    ).astype(object),
    "empty labels": lambda rows: np.full(rows, "", dtype=object),
    "None and numbers": lambda rows: np.asarray(
        [None if i % 3 == 0 else (i if i % 3 == 1 else i / 3) for i in range(rows)],
        dtype=object,
    ),
    "ASCII bytes and NaN": lambda rows: np.asarray(
        [
            b"ab" if i % 3 == 0 else (np.nan if i % 3 == 1 else "cde")
            for i in range(rows)
        ],
        dtype=object,
    ),
    "a trailing NUL": lambda rows: np.asarray(
        [f"c{i % 9}\x00" for i in range(rows)], dtype=object
    ),
    "fixed-width text wider than its values": lambda rows: np.asarray(
        [f"c{i % 9}" for i in range(rows)], dtype="U30"
    ),
    "byte strings": lambda rows: np.asarray(
        [f"c{i % 13}".encode() for i in range(rows)], dtype="S8"
    ),
    "one long label last": lambda rows: np.asarray(
        ["a"] * (rows - 1) + ["w" * 300], dtype=object
    ),
}


@pytest.mark.parametrize("rows", [1, 7, 100_000, 100_001, 250_000])
@pytest.mark.parametrize("text", list(_TEXT))
def test_text_is_stored_byte_for_byte_as_the_whole_column_write_stored_it(
    tmp_path, text, rows
) -> None:
    values = _TEXT[text](rows)
    root, selection = _root(tmp_path / "store.zarr")
    ref = _write(root, selection, _N_CELLS, {"labels": values})

    whole = _whole_column(tmp_path / "whole.zarr", values, 3)
    assert _stored(tmp_path / "store.zarr", ref, "labels") == whole
    stored = artifact_group(root, ref)["labels"][:]
    np.testing.assert_array_equal(stored, values.astype(str))


def test_a_zarr_format_2_store_keeps_its_bytes_too(tmp_path) -> None:
    values = _TEXT["four-byte"](100_001)
    root, selection = _root(tmp_path / "store.zarr", zarr_format=2)
    ref = _write(root, selection, _N_CELLS, {"labels": values})

    whole = _whole_column(tmp_path / "whole.zarr", values, 2)
    assert _stored(tmp_path / "store.zarr", ref, "labels") == whole


def test_an_empty_text_array_is_stored_as_one_character_wide(tmp_path) -> None:
    root, selection = _root(tmp_path / "store.zarr")
    empty = np.asarray([], dtype=object)
    ref = _write(root, selection, _N_CELLS, {"labels": empty})

    stored = artifact_group(root, ref)["labels"]
    assert stored.shape == (0,)
    assert stored.dtype == np.dtype("U1")
    whole = _whole_column(tmp_path / "whole.zarr", empty, 3)
    assert _stored(tmp_path / "store.zarr", ref, "labels") == whole


@pytest.mark.parametrize(
    "values",
    [
        np.asarray(["a", b"\xc3\xa9"], dtype=object),
        np.asarray([b"a", b"\xc3\xa9"], dtype="S2"),
    ],
    ids=["object", "fixed-width bytes"],
)
def test_bytes_that_are_not_ascii_are_refused_and_nothing_is_published(
    tmp_path, values
) -> None:
    # The whole-column write decoded byte strings as ASCII and raised; the
    # streamed write raises the same error and publishes nothing.
    root, selection = _root(tmp_path / "store.zarr")
    with pytest.raises(UnicodeDecodeError):
        values.astype(str)
    planned = _plan(root, selection, _N_CELLS)
    with pytest.raises(UnicodeDecodeError):
        write_cell_data_artifact(
            root,
            planned,
            {"codes": np.zeros(_N_CELLS, dtype=np.int32), "labels": values},
        )
    assert not inspect_artifact(root, planned.ref).exists


def _text_write(root, selection, values: np.ndarray):
    """A function that writes ``values`` as the text of a new artifact."""

    def write() -> None:
        _write(root, selection, _N_CELLS, {"labels": values})

    return write


def _stated(values: np.ndarray) -> int:
    width = int(np.char.str_len(values.astype(str)).max())
    return cell_data_write_bytes(
        {
            "codes": ((_N_CELLS,), np.int32),
            "labels": ((len(values),), np.dtype(f"U{width}")),
        }
    )


@pytest.mark.parametrize("kind", ["ascii", "two-byte", "four-byte"])
def test_writing_text_of_one_chunk_holds_what_its_cost_states(tmp_path, kind) -> None:
    # 60,000 labels of 40 characters are 9.6 MB as stored; the write holds
    # them at that width, a chunk at a time, and Zarr's encoded chunk.
    values = _labels(60_000, kind)
    root, selection = _root(tmp_path / "store.zarr")
    write = _text_write(root, selection, values)
    write()
    _result, peak = cost_bound(_stated(values), write)

    block = 4 * 40 * 60_000
    assert 2 * block <= peak < 2.1 * block, peak


@pytest.mark.parametrize(
    ("kind", "low", "high"),
    [
        ("ascii", 0x21, 0x7F),
        ("two-byte", 0x100, 0x800),
        ("four-byte", 0x10000, 0x10FFFF),
    ],
)
def test_writing_text_of_many_chunks_holds_the_chunk_before_beside_the_next(
    tmp_path, kind, low, high
) -> None:
    # 250,000 labels are two full chunks and a last one of 50,000 rows, which
    # Zarr copies into a full chunk. With the stored bytes of each chunk held
    # until the next write ends, as a busy machine holds them, the write
    # holds them beside the next chunk; labels of random characters hardly
    # compress, four-byte ones not at all.
    values = _random_labels(250_000, 40, low, high)
    root, selection = _root(tmp_path / "store.zarr")
    write = _text_write(root, selection, values)
    write()
    stated = _stated(values)
    _result, free = traced_peak(write, stated)
    assert free <= stated

    def held() -> None:
        with hold_previous_write():
            write()

    _result, peak = cost_bound(stated, held)

    block = 4 * 40 * _CHUNK_ROWS
    # The last chunk's rows, its full chunk and Zarr's encoded copy of it.
    assert free >= 2.5 * block, free
    if kind == "four-byte":
        # Without the chunk before, the cost would leave out a block.
        assert peak > stated - block + block // 2, (peak, stated)


def test_a_last_chunk_of_fewer_rows_holds_a_full_chunk_more(tmp_path) -> None:
    root, selection = _root(tmp_path / "store.zarr")
    peaks = {}
    for rows in (200_000, 199_999):
        values = _labels(rows, "ascii")
        write = _text_write(root, selection, values)
        write()
        _result, peaks[rows] = traced_peak(write, _stated(values))
        assert peaks[rows] <= _stated(values)
    block = 4 * 40 * _CHUNK_ROWS
    assert peaks[199_999] - peaks[200_000] > 0.9 * block, peaks
    assert _stated(_labels(199_999, "ascii")) - _stated(_labels(200_000, "ascii")) == (
        4 * 40 * 99_999
    )


def test_the_whole_column_write_held_three_copies_of_the_column(tmp_path) -> None:
    # Master's write converted the column twice and Zarr encoded its chunks,
    # up to all three at once; the streamed write of 300,000 labels of 10
    # characters holds two chunks, two thirds of one copy of the column.
    values = np.asarray([f"c{index:09d}" for index in range(300_000)], dtype=object)
    group = zarr.open_group(tmp_path / "whole.zarr", mode="w")

    def whole() -> None:
        create_metadata_column(
            group,
            "labels",
            data=values.astype(str),
            overwrite=True,
            chunkSize=_CHUNK_ROWS,
        )

    whole()
    _result, whole_peak = traced_peak(whole, 1 << 40)
    root, selection = _root(tmp_path / "store.zarr")
    write = _text_write(root, selection, values)
    write()
    _result, streamed = traced_peak(write, _stated(values))

    column = 4 * 10 * 300_000
    assert whole_peak >= 2 * column + column // 3, whole_peak
    assert streamed < 0.7 * column, streamed


def test_writing_numbers_holds_what_zarr_writes_at_once(tmp_path) -> None:
    # 1,050,000 int64 values that do not compress are ten full chunks and a
    # short one; Zarr writes up to ten chunks at once.
    n_cells = 1_050_000
    root, selection = _root(tmp_path / "store.zarr", n_cells=n_cells)
    values = np.random.default_rng(1).integers(0, 2**63 - 1, n_cells)

    def write() -> None:
        planned = _plan(root, selection, n_cells)
        write_cell_data_artifact(root, planned, {"codes": values})

    stated = cell_data_write_bytes({"codes": ((n_cells,), np.int64)})
    chunk = 8 * _CHUNK_ROWS
    assert stated - zarr_write_buffer_bytes() >= 12 * chunk
    write()
    for hold in (False, True):
        with hold_previous_write() if hold else contextlib.nullcontext():
            _result, peak = traced_peak(write, stated)
        assert 9 * chunk <= peak <= stated, (hold, peak, stated)


def test_the_cost_counts_a_buffer_for_each_write_zarr_runs_at_once() -> None:
    arrays = {"labels": ((50,), "U12"), "codes": ((50,), np.int32)}
    with zarr.config.set({"async.concurrency": 10}):
        ten = cell_data_write_bytes(arrays)
    with zarr.config.set({"async.concurrency": 20}):
        twenty = cell_data_write_bytes(arrays)
    assert twenty - ten == 10 * io.DEFAULT_BUFFER_SIZE


def test_the_costliest_array_sets_the_cost_whatever_their_order() -> None:
    text = {"labels": ((250_000,), "U40")}
    numbers = {"codes": ((250_000,), np.int32), "sizes": ((250_000,), np.int64)}
    alone = cell_data_write_bytes(text)
    assert cell_data_write_bytes({**numbers, **text}) == alone
    assert cell_data_write_bytes({**text, **numbers}) == alone
    # A chunk length shorter than the rows lets more chunks of numbers be
    # written at once.
    long = cell_data_write_bytes({"codes": ((250_000,), np.int64)})
    short = cell_data_write_bytes(
        {"codes": ((250_000,), np.int64)}, chunk_lengths={"codes": 10_000}
    )
    assert short < long


@pytest.mark.parametrize(
    ("arrays", "error", "message"),
    [
        ([("labels", ((3,), "U2"))], TypeError, "must map each array"),
        ({"labels": ((3,), "U2", 1)}, TypeError, "must be given as a"),
        ({"labels": [(3,), "U2"]}, TypeError, "must be given as a"),
        ({"labels": ((), "U2")}, ValueError, "needs a shape of one or more"),
        ({"labels": ((-1,), "U2")}, ValueError, "needs a shape of one or more"),
        ({"labels": ((3.0,), "U2")}, ValueError, "needs a shape of one or more"),
        ({"labels": ([3], "U2")}, ValueError, "needs a shape of one or more"),
        ({"labels": ((3,), object)}, ValueError, "at the fixed width of its widest"),
        ({"labels": ((3,), "S4")}, ValueError, "at the fixed width of its widest"),
        ({"labels": ((3, 2), "U4")}, ValueError, "must be one-dimensional"),
    ],
)
def test_the_cost_refuses_arrays_it_cannot_size(arrays, error, message) -> None:
    with pytest.raises(error, match=message):
        cell_data_write_bytes(arrays)
