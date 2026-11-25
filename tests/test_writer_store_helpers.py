"""Tests for the metadata-column and payload helpers shared by the writers."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.storage.arrays import linked_missing_mask
from cytearc.writers._store import (
    DEFAULT_IMPORT_BLOCK_ROWS,
    bounded_block_rows,
    decode_text,
    fingerprint_row_blocks,
    floating_payload_dtype,
    metadata_column_write_bytes,
    write_metadata_column,
)
from tests.memory_helpers import traced_charge


def _group() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _astral_text(n_rows: int, width: int) -> np.ndarray:
    """Return text of ``width`` random characters beyond the Basic Multilingual Plane a value.

    Its four bytes a character barely compress, so each chunk that Zarr
    encodes holds about the chunk's bytes once encoded.
    """
    codes = np.random.default_rng(4).integers(
        0x10000, 0x10FFFF, size=(n_rows, width), dtype=np.uint32
    )
    return codes.view(f"<U{width}").ravel()


def _traced_write(
    values: np.ndarray,
    missing: np.ndarray,
    threads: int,
    encodes_at_once: int = 1,
) -> int:
    """Trace a write of a column to a store that encodes on ``threads`` threads and keeps no chunks.

    With more than one encode at once, each encode waits until that many
    hold their chunks, so the block of a write holds them together on any
    machine.
    """
    from contextlib import nullcontext

    from zarr.codecs import BloscCodec

    from cytearc.storage.async_execution import register_store_threads
    from tests.memory_helpers import chunkless_store, encodes_held_together

    def write(_ledger: object) -> None:
        store = chunkless_store()
        register_store_threads(store, threads)
        group = zarr.open_group(store=store, mode="w")
        write_metadata_column(group, "column", values, missing)

    together = (
        encodes_held_together(BloscCodec, encodes_at_once)
        if encodes_at_once > 1
        else nullcontext()
    )
    with together:
        return traced_charge(write).held


# Text of 200 characters takes 16,777,216 // 800 = 20,971 rows to a chunk of
# 16,776,800 bytes, and a codec's buffer of a chunk at most a 128th more and
# 1,024 bytes beside: 16,908,892 bytes. 50,000 rows are two whole chunks and
# 8,058 rows of a third, which Zarr fills to a whole chunk before encoding it.
_WIDE_CHUNK = 20_971 * 800
_WIDE_ENCODED = _WIDE_CHUNK + _WIDE_CHUNK // 128 + 1_024


@pytest.mark.parametrize(
    ("threads", "expected", "least"),
    [
        # One chunk at a time: the last block holds the filled chunk and its
        # encoded buffer.
        (1, _WIDE_CHUNK + _WIDE_ENCODED, 2 * _WIDE_CHUNK),
        # Every chunk at once: three encoded buffers and the filled chunk.
        (4, 3 * _WIDE_ENCODED + _WIDE_CHUNK, 4 * _WIDE_CHUNK),
    ],
)
def test_a_wide_text_column_write_holds_one_encoded_chunk_a_write_thread(
    threads: int, expected: int, least: int
) -> None:
    values = _astral_text(50_000, 200)
    missing = np.zeros(50_000, dtype=bool)
    charge = metadata_column_write_bytes(50_000, 800, missing=False, threads=threads)
    assert charge == expected
    held = _traced_write(values, missing, threads, encodes_at_once=min(threads, 3))
    # Beside the column, the write holds its chunks' buffers, each at least
    # the chunk's bytes, and at most the charge with Zarr's objects of the
    # array and its writes, tens of kilobytes, and what a coverage tracer
    # records of the lines that the write runs.
    assert least <= held <= charge + 128 * 1024


@pytest.mark.parametrize(
    ("values", "missing", "threads", "expected", "least"),
    [
        # 20,000 values of 200 characters, one blank, are one chunk of
        # 16,000,000 bytes, whose codec's buffer takes at most 16,126,024:
        # the text is copied with a placeholder in the blank row, from a copy
        # of its present values, beside which the write first finds them
        # with a mark a row; then Zarr encodes the copy.
        (
            _astral_text(20_000, 200),
            np.arange(20_000) == 0,
            1,
            16_000_000 + max(16_000_000 + 20_000, 16_126_024),
            2 * 16_000_000 - 800,
        ),
        # 250,000 float64 values are chunks of 100,000 rows, 800,000 bytes,
        # and 50,000 rows of a third, which Zarr fills, beside a buffer of
        # at most 807,274 bytes.
        (
            np.random.default_rng(5).random(250_000),
            np.zeros(250_000, dtype=bool),
            1,
            800_000 + 807_274,
            2 * 800_000,
        ),
        # Two encoded buffers at a time of 400,000 values of two characters
        # in four whole chunks of 100,000 rows, 800,000 bytes each.
        (
            _astral_text(400_000, 2),
            np.zeros(400_000, dtype=bool),
            2,
            2 * 807_274,
            2 * 800_000,
        ),
        # 200,000 values of two characters, one blank, are two chunks of
        # 100,000 rows: the copy of the column is made beside a copy of the
        # present values and a mark a row, more than Zarr's buffer of a
        # chunk, 807,274 bytes at most, beside the copy.
        (
            _astral_text(200_000, 2),
            np.arange(200_000) == 0,
            1,
            1_600_000 + 1_600_000 + 200_000,
            1_600_000 + 1_599_992 + 200_000,
        ),
        # Text of one character, one blank: the present values' width is
        # measured with their int64 lengths, 8 bytes each, which outweigh
        # the text itself, 4 bytes a value, beside the present values.
        (
            _astral_text(200_000, 1),
            np.arange(200_000) == 0,
            1,
            800_000 + 8 * 200_000,
            799_996 + 8 * 199_999,
        ),
        # 250,000 flags in chunks of 100,000 rows, of 100,000 bytes, whose
        # buffer takes at most 101,805, beside the filled last chunk.
        (
            np.arange(250_000) % 2 == 0,
            np.zeros(250_000, dtype=bool),
            1,
            100_000 + 101_805,
            2 * 100_000,
        ),
    ],
    ids=[
        "text-one-blank",
        "float",
        "narrow-two-threads",
        "narrow-text-one-blank",
        "letters-one-blank",
        "flags",
    ],
)
def test_a_column_write_holds_what_its_size_helper_charges(
    values: np.ndarray, missing: np.ndarray, threads: int, expected: int, least: int
) -> None:
    charge = metadata_column_write_bytes(
        values.size,
        values.dtype.itemsize,
        missing=bool(missing.any()),
        threads=threads,
    )
    assert charge == expected
    held = _traced_write(values, missing, threads, encodes_at_once=threads)
    assert least <= held <= charge + 128 * 1024


def test_a_column_with_a_third_missing_holds_less_than_its_charge() -> None:
    # The charge counts a copy of the present values as large as the
    # column, since a blank in one row leaves nearly all of them; with a
    # third of the rows blank, their copy is two thirds of the column.
    values = _astral_text(30_000, 200)
    missing = np.arange(30_000) % 3 == 0
    charge = metadata_column_write_bytes(30_000, 800, missing=True, threads=1)
    held = _traced_write(values, missing, 1)
    assert held <= charge
    # It holds the column's copy and the present values' at least.
    assert held >= 30_000 * 800 + 20_000 * 800


def test_a_column_of_blank_text_holds_less_than_the_charge_of_missing_rows() -> None:
    # Text that is missing in every row has no present values to copy, so
    # its write holds less than the charge, which counts them.
    values, missing = np.full(100_000, ""), np.ones(100_000, dtype=bool)
    charge = metadata_column_write_bytes(100_000, 4, missing=True, threads=1)
    held = _traced_write(values, missing, 1)
    assert held <= charge


def test_a_column_write_hands_zarr_one_chunk_for_each_write_thread_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.storage.async_execution import register_store_threads

    # 700,000 one-byte flags are seven chunks of 100,000 rows; three write
    # threads take three chunks at a time, the last block one, and the
    # missing marks go with their block.
    writes: list[tuple[str, int, int]] = []
    setitem = zarr.Array.__setitem__

    def recorded(array: zarr.Array, selection: object, value: object) -> None:
        assert isinstance(selection, slice)
        writes.append((array.basename, int(selection.start), int(selection.stop)))
        setitem(array, selection, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", recorded)
    store = MemoryStore()
    register_store_threads(store, 3)
    group = zarr.open_group(store=store, mode="w")
    values = (np.arange(700_000) % 3).astype(np.int8)
    missing = np.arange(700_000) % 7 == 0
    write_metadata_column(group, "column", values, missing)
    blocks = [(0, 300_000), (300_000, 600_000), (600_000, 700_000)]
    assert writes == [
        (name, start, stop)
        for start, stop in blocks
        for name in ("column", "__cytearc_missing__column")
    ]
    np.testing.assert_array_equal(group["column"][:], np.where(missing, 0, values))
    np.testing.assert_array_equal(linked_missing_mask(group, "column")[:], missing)


def test_metadata_column_write_bytes_counts_a_block_of_write_threads() -> None:
    from cytearc.storage.arrays import column_write_bytes

    # Ten chunks of 100,000 float64 rows, each 800,000 bytes and at most
    # 807,274 encoded. Four threads encode four at a time; the last block
    # holds two.
    assert (
        metadata_column_write_bytes(1_000_000, 8, missing=False, threads=4)
        == 4 * 807_274
    )
    # Ten threads, or more, take every chunk at once.
    for threads in (10, 64):
        assert column_write_bytes(1_000_000, 8, threads) == 10 * 807_274
    # A last chunk of 1 row past 900,000 is filled whole, and with three
    # threads it is the last block's one chunk beside the filled one, which
    # the whole blocks' three buffers outweigh.
    assert column_write_bytes(900_001, 8, 3) == 3 * 807_274
    assert column_write_bytes(900_001, 8, 10) == 10 * 807_274 + 800_000
    # A column of fewer rows than a chunk is one chunk of its rows, whole.
    assert column_write_bytes(10, 8, 4) == 80 + 80 // 128 + 1_024
    # A column of no rows writes nothing.
    assert column_write_bytes(0, 8, 4) == 0
    # A writer's block beside them: a gathered block of 4 bytes a row in
    # blocks of 400,000 rows, and the last of 200,000.
    assert column_write_bytes(1_000_000, 8, 4, block_row_bytes=4) == 4 * 807_274 + (
        4 * 400_000
    )


def test_decode_text_accepts_text_and_rejects_other_values() -> None:
    assert decode_text(b"caf\xc3\xa9") == "café"
    assert decode_text(np.bytes_(b"abc")) == "abc"
    assert decode_text(np.str_("abc")) == "abc"
    with pytest.raises(ValueError, match="valid UTF-8"):
        decode_text(b"\xff")
    with pytest.raises(TypeError, match="Expected text, found int"):
        decode_text(3)


@pytest.mark.parametrize(
    ("values", "kind", "expected"),
    [
        ([True, None, False], "b", [True, False, False]),
        ([1, None, np.int32(3)], "i", [1, 0, 3]),
        # Masked float rows hold NaN, as in every float column with a mask.
        ([1, np.nan, 2.5], "f", [1.0, np.nan, 2.5]),
        ([b"a", None, "bc", 4], "U", ["a", "", "bc", "4"]),
    ],
)
def test_object_metadata_columns_store_typed_values_and_mask_missing_rows(
    values, kind, expected
) -> None:
    group = _group()
    write_metadata_column(group, "column", np.asarray(values, dtype=object))

    stored = group["column"]
    assert np.dtype(stored.dtype).kind == kind
    np.testing.assert_array_equal(stored[:], expected)
    mask = linked_missing_mask(group, "column")
    assert mask is not None
    np.testing.assert_array_equal(
        mask[:], [value is None or value is np.nan for value in values]
    )


def test_complete_object_metadata_columns_are_typed_without_a_mask() -> None:
    group = _group()
    write_metadata_column(group, "counts", np.asarray([1, 2, 3], dtype=object))

    assert np.dtype(group["counts"].dtype) == np.dtype(np.int64)
    np.testing.assert_array_equal(group["counts"][:], [1, 2, 3])
    assert linked_missing_mask(group, "counts") is None


@pytest.mark.parametrize(
    ("values", "missing", "expected"),
    [
        (np.asarray([b"x", b"yy"]), [False, True], ["x", ""]),
        (np.asarray(["x", "yy"]), [True, False], ["", "yy"]),
        (np.asarray([1.5, 2.5]), [True, False], [np.nan, 2.5]),
        (np.asarray([5, 6]), [False, True], [5, 0]),
    ],
)
def test_explicit_missing_rows_hold_a_placeholder(values, missing, expected) -> None:
    group = _group()
    write_metadata_column(group, "column", values, missing)

    np.testing.assert_array_equal(group["column"][:], expected)
    mask = linked_missing_mask(group, "column")
    assert mask is not None
    np.testing.assert_array_equal(mask[:], missing)


def test_metadata_columns_reject_bad_shapes() -> None:
    group = _group()
    with pytest.raises(ValueError, match="one value per row"):
        write_metadata_column(group, "grid", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="misaligned missing mask"):
        write_metadata_column(group, "column", np.zeros(3), [True, False])
    assert "grid" not in group
    assert "column" not in group


def test_floating_payload_dtype_widens_integers_and_rejects_text() -> None:
    assert floating_payload_dtype(np.float32, "Payload") == np.dtype(np.float32)
    assert floating_payload_dtype(np.int16, "Payload") == np.dtype(np.float64)
    assert floating_payload_dtype(bool, "Payload") == np.dtype(np.float64)
    with pytest.raises(TypeError, match="Payload uses unsupported dtype"):
        floating_payload_dtype("U3", "Payload")


def test_bounded_block_rows_respects_request_and_memory() -> None:
    assert bounded_block_rows(None, row_bytes=1, memory_bytes=1 << 40) == (
        DEFAULT_IMPORT_BLOCK_ROWS
    )
    assert bounded_block_rows(10, row_bytes=1, memory_bytes=1 << 40) == 10
    assert bounded_block_rows(10, row_bytes=100, memory_bytes=1600) == 2
    assert bounded_block_rows(10, row_bytes=100, memory_bytes=1) == 1
    assert bounded_block_rows(0, row_bytes=0, memory_bytes=0) == 1


def test_row_block_fingerprints_ignore_blocking_and_reject_non_finite() -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    whole = fingerprint_row_blocks(
        [values], values.shape, values.dtype, label="Payload"
    )
    split = fingerprint_row_blocks(
        [values[:1], values[1:]], values.shape, values.dtype, label="Payload"
    )
    changed = values.copy()
    changed[3, 2] += 1
    assert whole == split
    assert whole != fingerprint_row_blocks(
        [changed], values.shape, values.dtype, label="Payload"
    )

    values[2, 1] = np.inf
    with pytest.raises(ValueError, match="Payload contains non-finite values"):
        fingerprint_row_blocks([values], values.shape, values.dtype, label="Payload")


def test_create_zarr_obj_array_writes_metadata_columns_from_the_writers_facade():
    from zarr.errors import ContainsArrayError

    from cytearc.writers import create_zarr_obj_array

    group = _group()
    labels = create_zarr_obj_array(
        group, "labels", np.asarray([b"a", b"bbb", b"cc"]), chunk_size=2
    )
    # Byte strings are decoded and stored at the width of the longest value.
    assert labels.dtype == np.dtype("U3")
    assert labels.chunks == (2,)
    np.testing.assert_array_equal(group["labels"][:], ["a", "bbb", "cc"])

    counts = create_zarr_obj_array(group, "counts", np.asarray([1, 2]), np.int64)
    assert counts.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(group["counts"][:], [1, 2])

    empty = create_zarr_obj_array(
        group, "scores", None, np.float32, chunk_size=2, shape=3
    )
    assert (empty.shape, empty.chunks, empty.dtype) == ((3,), (2,), np.float32)
    with pytest.raises(ValueError, match="shape is required when data is None"):
        create_zarr_obj_array(group, "unsized", None, np.float32)

    with pytest.raises(ContainsArrayError):
        create_zarr_obj_array(group, "labels", np.asarray(["x"]), overwrite=False)
    np.testing.assert_array_equal(group["labels"][:], ["a", "bbb", "cc"])
    create_zarr_obj_array(group, "labels", np.asarray(["x"]))
    np.testing.assert_array_equal(group["labels"][:], ["x"])


@pytest.mark.parametrize(
    ("data", "shape"),
    [
        pytest.param(np.asarray(5), "()", id="zero-dimensional-array"),
        pytest.param(5, "()", id="integer"),
        pytest.param("abc", "()", id="text"),
        pytest.param(b"abc", "()", id="bytes"),
        pytest.param(np.ones((2, 2)), "(2, 2)", id="grid"),
        pytest.param([[1, 2], [3, 4]], "(2, 2)", id="nested-lists"),
        pytest.param(np.zeros((3, 1, 2)), "(3, 1, 2)", id="three-dimensions"),
    ],
)
def test_metadata_column_writers_refuse_values_that_are_not_one_per_row(
    data, shape
) -> None:
    import cytearc
    from cytearc.storage.arrays import create_metadata_column

    group = _group()
    cytearc.create_zarr_obj_array(group, "kept", np.asarray(["a", "b", "c"]))
    for write in (
        lambda name: cytearc.create_zarr_obj_array(group, name, data),
        lambda name: create_metadata_column(group, name, data=data),
    ):
        # A new column, and one that the write would replace, which is left
        # as it was. Zarr used to refuse them, saying only that the chunks
        # have 1 dimension and the shape has another number.
        for name in ("column", "kept"):
            with pytest.raises(ValueError) as refused:
                write(name)
            assert str(refused.value) == (
                f"Column {name!r} needs one value per row; found values of shape "
                f"{shape}"
            )
            assert sorted(group.keys()) == ["kept"]
            np.testing.assert_array_equal(group["kept"][:], ["a", "b", "c"])
