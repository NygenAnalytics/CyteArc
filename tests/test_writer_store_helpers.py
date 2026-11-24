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


def _third_missing(n_rows: int) -> np.ndarray:
    return np.arange(n_rows) % 3 == 0


@pytest.mark.parametrize(
    ("values", "missing"),
    [
        # Text of 200 characters, with and without missing rows, whose
        # copies dwarf what else the write holds.
        (np.full(20_000, "n" * 200), _third_missing(20_000)),
        (np.full(20_000, "n" * 200), np.zeros(20_000, dtype=bool)),
        # Narrow text, numbers and flags.
        (np.full(200_000, "ab"), _third_missing(200_000)),
        (np.arange(200_000, dtype=np.float64), _third_missing(200_000)),
        (np.arange(200_000, dtype=np.float64), np.zeros(200_000, dtype=bool)),
        (np.arange(200_000) % 2 == 0, np.zeros(200_000, dtype=bool)),
    ],
    ids=["text", "text-none-missing", "narrow", "float", "float-none-missing", "flags"],
)
def test_a_column_write_holds_what_its_size_helper_charges(values, missing) -> None:
    n_rows = values.size
    charge = metadata_column_write_bytes(
        n_rows, values.dtype.itemsize, missing=bool(missing.any())
    )
    held = traced_charge(
        lambda _ledger: write_metadata_column(_group(), "column", values, missing)
    ).held
    # Beside the column and its mask, which the caller holds, the write holds
    # the charge, within Zarr's objects of the new array and of its writes,
    # which the charge leaves out.
    zarr_objects = 64 * 1024
    assert charge - zarr_objects <= held <= charge + zarr_objects


def test_a_column_of_blank_text_holds_less_than_the_charge_of_missing_rows() -> None:
    # Text that is missing in every row has no present values to copy, so
    # its write holds less than the charge, which counts them.
    values, missing = np.full(200_000, ""), np.ones(200_000, dtype=bool)
    charge = metadata_column_write_bytes(200_000, 4, missing=True)
    held = traced_charge(
        lambda _ledger: write_metadata_column(_group(), "column", values, missing)
    ).held
    assert held <= charge


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
