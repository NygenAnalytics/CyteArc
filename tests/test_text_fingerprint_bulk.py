"""Bulk text hashing gives exactly the digests of the per-value code it replaced.

Stored identities, such as the dataset fingerprint that preparation records and
the sample-assignment fingerprint of an auto filter, hash text. The reference
functions below are the previous per-value implementations, kept here so every
input is compared byte for byte: a digest that changed would give existing
artifacts new identities.
"""

from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc.storage.artifacts as artifacts_module
from cytearc.storage.arrays import (
    _decode_metadata_values,
    _measured_text_dtype,
    text_dtype,
    text_value,
)
from cytearc.storage.artifacts import (
    ValueFingerprintBuilder,
    canonical_bytes,
    fingerprint_array,
    fingerprint_strings,
)
from cytearc.storage.identity import _hash_column, calculate_dataset_fingerprint


def _reference_decode(data: Any) -> np.ndarray:
    values = np.asarray(data)
    if values.dtype.kind == "S" or (
        values.dtype.hasobject
        and any(
            value is None or isinstance(value, bytes | bytearray | np.bytes_)
            for value in values.flat
        )
    ):
        return np.asarray(
            [text_value(value) for value in values.flat], dtype=str
        ).reshape(values.shape)
    return values


def _reference_measured_dtype(blocks: Any) -> np.dtype[Any]:
    width = 1
    for block in blocks:
        for value in np.asarray(block).flat:
            width = max(width, len(text_value(value)))
    return np.dtype(f"U{width}")


def _reference_fingerprint_strings(values: Any) -> str:
    array = np.asarray(values)
    decoded = _reference_decode(array)
    resolved = np.dtype(array.dtype)
    dtype = (
        np.empty(0, dtype=resolved).astype(str).dtype
        if not resolved.hasobject
        else _reference_measured_dtype((decoded,))
    )
    return fingerprint_array(decoded.astype(dtype))


def _reference_hash_column(
    builder: ValueFingerprintBuilder, name: str, column: Any, rows: int
) -> None:
    builder.update_bytes(name, canonical_bytes({"shape": list(column.shape)}))
    for start in range(0, column.shape[0], rows):
        for value in _reference_decode(column[start : start + rows]):
            builder.update_bytes("value", str(value).encode("utf-8"))


def _per_value_digest(values: np.ndarray) -> str:
    builder = ValueFingerprintBuilder()
    builder.update_bytes("before", b"prefix")
    for value in values:
        builder.update_bytes("value", str(value).encode("utf-8"))
    return builder.hexdigest()


def _bulk_digest(values: np.ndarray) -> str:
    builder = ValueFingerprintBuilder()
    builder.update_bytes("before", b"prefix")
    builder.update_text_values("value", values)
    return builder.hexdigest()


def _code_points(*points: int) -> np.ndarray:
    """Return one fixed-width value holding exactly these code points."""
    return np.asarray(points, dtype="<u4").view(f"<U{len(points)}")


_LONG = "x" * 50_000 + "é" * 30_000 + "中"
_TEXT_CASES = {
    "fixed_length_barcodes": np.asarray(
        ["AAACCTGAGAAACCAT-1", "AAACCTGAGAAACCGC-1", "TTTGTCATCTTGTTTG-1"]
    ),
    "variable_lengths": np.asarray(["a", "", "abc", "ab", "cell_10"]),
    "all_empty": np.asarray(["", "", ""]),
    "single": np.asarray(["only"]),
    "no_values": np.asarray([], dtype="U5"),
    "inner_nul": np.asarray(["a\x00b", "\x00x", "ab"]),
    "only_nul": np.asarray(["\x00", "\x00\x00", "a"]),
    "two_byte": np.asarray(["café", "naïve", "ß"]),
    "three_byte": np.asarray(["中文", "€", "ࠀ", "￿"]),
    "four_byte": np.asarray(["😀", "a😀b", "\U00010000", "\U0010ffff"]),
    "combining": np.asarray(["é", "ǟ"]),
    "boundaries": np.asarray(["\x7f", "\x80", "߿", "ࠀ", "\U00010000"]),
    "mixed_widths": np.asarray(["plain", "Zellé-中-😀", "", "x\x00中"]),
    "long_value": np.asarray(["short", _LONG, "é"]),
    "big_endian": np.asarray(["big", "endian", "中😀"], dtype=">U8"),
    "beyond_unicode": np.concatenate(
        [np.asarray(["ok"], dtype="U2"), _code_points(0x110000, 0x41)]
    ),
    "beyond_unicode_large": _code_points(0xFFFFFFFF, 0x41),
}


@pytest.mark.parametrize("name", sorted(_TEXT_CASES))
@pytest.mark.parametrize("pass_characters", [1, 7, 1 << 20])
def test_bulk_text_values_hash_the_bytes_of_one_record_per_value(
    monkeypatch: pytest.MonkeyPatch, name: str, pass_characters: int
) -> None:
    values = _TEXT_CASES[name]
    # Small passes put pass boundaries between and inside the cases.
    monkeypatch.setattr(artifacts_module, "_TEXT_PASS_CHARACTERS", pass_characters)

    assert _bulk_digest(values) == _per_value_digest(values)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray(["a", 2, 3.5, True, None, b"raw", "x\x00"], dtype=object),
        np.asarray([np.str_("np"), np.float32(0.1), np.int8(-3)], dtype=object),
        np.asarray(["a", "b\x00"], dtype=np.dtypes.StringDType()),
        np.asarray([b"bytes", b"x"], dtype="S5"),
        np.asarray([1, 22, 333]),
        np.asarray([[b"a", b"b"], [b"c", b"d"]], dtype="S1"),
        np.asarray([["row", "one"], ["row", "two"]]),
    ],
    ids=[
        "objects",
        "numpy_scalars",
        "variable_width",
        "bytes",
        "integers",
        "2d_bytes",
        "2d_text",
    ],
)
def test_other_values_hash_one_record_each(values: np.ndarray) -> None:
    assert _bulk_digest(values) == _per_value_digest(values)


def test_text_that_utf8_cannot_encode_raises_the_same_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.asarray(["fine", "bad\ud800", "later"])
    monkeypatch.setattr(artifacts_module, "_TEXT_PASS_CHARACTERS", 6)

    with pytest.raises(UnicodeEncodeError) as expected:
        _per_value_digest(values)
    with pytest.raises(UnicodeEncodeError) as observed:
        _bulk_digest(values)
    assert str(observed.value) == str(expected.value)


def test_text_values_need_a_finished_array() -> None:
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (1,), np.int64)

    with pytest.raises(RuntimeError, match="Finish the active array"):
        builder.update_text_values("value", np.asarray(["a"]))


def _stored_text_columns() -> dict[str, zarr.Array]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    legacy = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=2)
    rng = np.random.default_rng(7)
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=(97, 16))]
    barcodes = np.char.add(letters.view("S16").reshape(-1).astype(str), "-1")
    labels = np.asarray(["Zellé", "中", "a\x00b", "", "😀x"])[
        rng.integers(0, 5, size=97)
    ]
    return {
        "barcodes": root.create_array("barcodes", data=barcodes, chunks=(10,)),
        "unicode": root.create_array("unicode", data=labels, chunks=(13,)),
        "one_chunk": root.create_array("one_chunk", data=barcodes, chunks=(97,)),
        "variable_width": root.create_array(
            "variable_width",
            data=labels.astype(np.dtypes.StringDType()),
            chunks=(16,),
        ),
        "legacy_bytes": legacy.create_array(
            "legacy_bytes", data=np.char.encode(labels, "utf-8"), chunks=(11,)
        ),
        "legacy_unicode": legacy.create_array(
            "legacy_unicode", data=barcodes, chunks=(9,)
        ),
        "empty": root.create_array(
            "empty", shape=(0,), dtype="<U4", chunks=(5,), fill_value=""
        ),
        "numbers": root.create_array(
            "numbers", data=np.arange(97, dtype=np.float32), chunks=(10,)
        ),
    }


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
def test_dataset_fingerprint_columns_hash_as_before() -> None:
    for name, column in _stored_text_columns().items():
        expected = ValueFingerprintBuilder()
        if column.dtype.kind in "OUS" or column.dtype.hasobject:
            _reference_hash_column(expected, name, column, rows=column.chunks[0] * 2)
        else:
            expected.update_array(name, np.asarray(column[:]))
        observed = ValueFingerprintBuilder()
        _hash_column(observed, name, column)
        assert observed.hexdigest() == expected.hexdigest(), name


def test_dataset_fingerprint_hashes_text_columns_in_bulk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    column = _stored_text_columns()["barcodes"]
    calls: list[str] = []
    original = ValueFingerprintBuilder.update_bytes

    def counting(self: ValueFingerprintBuilder, name: str, payload: bytes) -> None:
        calls.append(name)
        original(self, name, payload)

    monkeypatch.setattr(ValueFingerprintBuilder, "update_bytes", counting)
    _hash_column(ValueFingerprintBuilder(), "cell_ids", column)

    # The shape record alone; the 97 values were hashed in bulk, not one by one.
    assert calls == ["cell_ids"]


def test_object_label_fingerprints_measure_text_without_a_call_per_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cytearc.storage.arrays as arrays_module

    labels = np.asarray(["sample_a", "sample_bb"] * 500).astype(object)
    expected = _reference_fingerprint_strings(labels)
    calls = 0
    original = arrays_module.text_value

    def counting(value: Any) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(arrays_module, "text_value", counting)
    assert fingerprint_strings(labels) == expected
    assert calls == 0


def test_dataset_fingerprint_matches_the_one_the_fixture_recorded(
    datastore_zarr_root: str,
) -> None:
    # The fixture store was prepared by an earlier release, which hashed each
    # cell and feature ID one at a time.
    root = zarr.open_group(datastore_zarr_root, mode="r")
    for name in ("RNA", "assay2"):
        assay = root[name]
        assert (
            calculate_dataset_fingerprint(assay, root["cellData"], assay["counts"])
            == assay.attrs["dataset_fingerprint"]
        )


_STRING_CASES = {
    "object_text": np.asarray(["sample_a", "sample_bb", "", "x\x00"], dtype=object),
    "typed_text_as_objects": np.asarray(["s1", "s22", "s1"]).astype(object),
    "object_unicode": np.asarray(["中", "😀😀", "é"], dtype=object),
    "object_with_none": np.asarray(["a", None, "bb"], dtype=object),
    "object_with_bytes": np.asarray([b"caf\xc3\xa9", "tea"], dtype=object),
    "object_integers": np.asarray([1, 22, -333], dtype=object),
    "object_floats": np.asarray([0.1, 1e16, -0.0], dtype=object),
    "object_numpy_floats": np.asarray([np.float32(0.1), np.float64(2.5)], dtype=object),
    "object_booleans": np.asarray([True, False], dtype=object),
    "object_mixed": np.asarray(["a", 1, np.str_("b")], dtype=object),
    "object_empty": np.asarray([], dtype=object),
    "object_scalar": np.asarray("lone", dtype=object),
    "object_2d": np.asarray([["a", "bb"], ["ccc", ""]], dtype=object),
    "variable_width": np.asarray(["a", "longer", "中"], dtype=np.dtypes.StringDType()),
    "fixed_text": np.asarray(["x", "yy"], dtype="U6"),
    "fixed_bytes": np.asarray([b"x", b"caf\xc3\xa9"], dtype="S8"),
    "integers": np.asarray([3, 44]),
    "booleans": np.asarray([True, False]),
}


@pytest.mark.parametrize("name", sorted(set(_STRING_CASES) - {"object_scalar"}))
def test_string_fingerprints_match_the_previous_code(name: str) -> None:
    values = _STRING_CASES[name]
    assert fingerprint_strings(values) == _reference_fingerprint_strings(values)


def test_string_fingerprints_still_refuse_a_scalar() -> None:
    values = _STRING_CASES["object_scalar"]
    with pytest.raises(ValueError) as expected:
        _reference_fingerprint_strings(values)
    with pytest.raises(ValueError) as observed:
        fingerprint_strings(values)
    assert str(observed.value) == str(expected.value)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([b"\xff", "fine"], dtype=object),
        np.asarray([b"fine", b"\xc3"], dtype="S2"),
    ],
    ids=["object", "fixed"],
)
def test_invalid_utf8_raises_the_same_error(values: np.ndarray) -> None:
    with pytest.raises(UnicodeDecodeError) as expected:
        _reference_fingerprint_strings(values)
    with pytest.raises(UnicodeDecodeError) as observed:
        fingerprint_strings(values)
    assert str(observed.value) == str(expected.value)


@pytest.mark.parametrize("name", sorted(_STRING_CASES))
def test_text_helpers_match_the_previous_code(name: str) -> None:
    values = _STRING_CASES[name]
    decoded = _decode_metadata_values(values)
    expected = _reference_decode(values)
    assert decoded.dtype == expected.dtype
    assert decoded.shape == expected.shape
    assert decoded.tolist() == expected.tolist()
    blocks = (values, values[:0], expected) if values.ndim else (values,)
    assert _measured_text_dtype(blocks) == _reference_measured_dtype(blocks)
    assert text_dtype(values.dtype, lambda: blocks) == (
        np.empty(0, dtype=values.dtype).astype(str).dtype
        if not values.dtype.hasobject
        else _reference_measured_dtype(blocks)
    )


def test_text_width_of_no_blocks_is_one_character() -> None:
    assert _measured_text_dtype(()) == np.dtype("U1")
    assert _measured_text_dtype((np.asarray([], dtype=object),)) == np.dtype("U1")
