"""Text arrays of cell-data artifacts, written a stored chunk at a time.

``write_cell_data_artifact`` stores text as master's whole-column write did,
``values.astype(str)`` at the width of the widest value, but converts and
writes one stored chunk at a time.
"""

import hashlib
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
from cytearc.storage.arrays import create_metadata_column
from cytearc.storage.layout import column_chunk_rows
from cytearc.storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from cytearc.storage.refs import artifact_path
from cytearc.storage.selections import resolve_generated_selection_artifact

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
        # Text takes the chunks of cell columns: at most 100,000 rows and
        # 16 MiB of values at its stored width.
        chunkSize=column_chunk_rows(len(values), values.astype(str).dtype.itemsize),
    )
    return _files(path / "labels")


def _labels(rows: int, kind: str) -> np.ndarray:
    """Clone-like labels of 40 characters, which compress well."""
    lead = {"ascii": "c", "two-byte": "é", "four-byte": "\U0001f600"}[kind]
    distinct = np.asarray(
        [f"{lead * 34}{index:06d}" for index in range(min(rows, 5_000))], dtype=object
    )
    return distinct[np.arange(rows) % len(distinct)]


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
        ({"labels": ((True,), "U2")}, ValueError, "needs a shape of one or more"),
        ({"labels": ((3,), object)}, ValueError, "at the fixed width of its widest"),
        ({"labels": ((3,), "S4")}, ValueError, "at the fixed width of its widest"),
        ({"labels": ((3,), str)}, ValueError, "at the fixed width of its widest"),
        ({"labels": ((3,), "U0")}, ValueError, "at the fixed width of its widest"),
        ({"labels": ((3, 2), "U4")}, ValueError, "must be one-dimensional"),
    ],
)
def test_the_cost_refuses_arrays_it_cannot_size(arrays, error, message) -> None:
    with pytest.raises(error, match=message):
        cell_data_write_bytes(arrays)


def test_wide_text_is_chunked_by_its_bytes_and_charged_by_its_chunks(tmp_path) -> None:
    rows = 50_000
    # 200 characters, 800 bytes a value as stored: 16 MiB holds 20,971 rows.
    values = np.asarray([f"{index:06d}" + "x" * 194 for index in range(rows)])
    root, selection = _root(tmp_path / "store.zarr", n_cells=rows)
    ref = _write(root, selection, rows, {"labels": values})

    labels = artifact_group(root, ref)["labels"]
    assert labels.chunks == (20_971,)
    np.testing.assert_array_equal(labels[:], values)
    # A chunk and its encoded copy, and the last, shorter chunk copied into a
    # full one: 2 * 800 * 20,971 + 800 * 8,058 bytes, far below the 80 MB of
    # two chunks of 50,000 rows.
    assert cell_data_write_bytes({"labels": ((rows,), "U200")}) == (
        2 * 800 * 20_971 + 800 * (rows % 20_971)
    )
    # Labels of up to 41 characters keep chunks of 100,000 rows.
    assert cell_data_write_bytes({"labels": ((250_000,), "U41")}) == (
        2 * 164 * 100_000 + 164 * 50_000
    )
