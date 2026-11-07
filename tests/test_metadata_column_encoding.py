"""Metadata columns store exactly the arrays of the per-value code that bulk conversion replaced.

Imports and inserts convert text with whole-array NumPy operations, and a
categorical column from its categories, once each, taken by code. The
reference functions below are the previous per-value implementations, kept
here so that every stored array, its dtype and width, its missing mask and its
fingerprint are compared byte for byte.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

import cytearc.storage.arrays as arrays_module
from cytearc.metadata.encoding import (
    _shared_dtype,
    _typed_numbers,
    missing_placeholder,
    missing_rows,
    stored_values,
)
from cytearc.readers import CrH5Reader, CSVReader, H5adReader, MtxReader, inspect_mtx
from cytearc.readers._h5ad_columns import decode_categories
from cytearc.storage.arrays import _decode_metadata_values
from cytearc.storage.artifacts import fingerprint_array
from cytearc.storage.layout import column_chunk_rows
from cytearc.writers import CrToZarr, CSVtoZarr, H5adToZarr, MtxToZarr, SparseToZarr
from cytearc.writers._store import write_metadata_column


def _reference_text_value(value: Any) -> str:
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _reference_text(
    values: np.ndarray, missing: np.ndarray, *, name: str
) -> np.ndarray:
    try:
        text = [
            "" if absent else _reference_text_value(value)
            for value, absent in zip(values.tolist(), missing.tolist(), strict=True)
        ]
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Column {name!r} holds text that is not valid UTF-8; pass str values "
            "or UTF-8 encoded bytes"
        ) from exc
    width = max((len(value) for value in text), default=1)
    return np.asarray(text, dtype=f"U{max(width, 1)}")


def _reference_stored_values(
    values: np.ndarray, missing: np.ndarray, *, name: str
) -> np.ndarray:
    kind = values.dtype.kind
    if kind == "O":
        present = values[~missing]
        dtype = _shared_dtype(present)
        if dtype is None:
            return _reference_text(values, missing, name=name)
        stored = np.full(values.shape, missing_placeholder(dtype), dtype=dtype)
        stored[~missing] = _typed_numbers(present, dtype, name=name)
        return stored
    if kind in "SU":
        return _reference_text(values, missing, name=name)
    stored = values.copy()
    if missing.any():
        stored[missing] = missing_placeholder(values.dtype)
    return stored


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
            [_reference_text_value(value) for value in values.flat], dtype=str
        ).reshape(values.shape)
    return values


def _reference_encode(data: Any, dtype: Any) -> np.ndarray:
    """What ``encode_metadata_values`` stored for ``data``."""
    raw = np.asarray(data)
    values = _reference_decode(raw)
    if (
        dtype is None
        or np.dtype(dtype).kind == "O"
        or raw.dtype.kind in {"S", "O"}
        and values.dtype.kind == "U"
    ):
        width = max([1, *(len(_reference_text_value(value)) for value in values.flat)])
        return values.astype(f"U{width}")
    return np.asarray(values, dtype=dtype)


def _reference_column(
    values: Any, missing: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """The values and mask that ``write_metadata_column`` stored for a column."""
    array = np.asarray(values)
    mask = missing_rows(array) if missing is None else np.asarray(missing, dtype=bool)
    stored = (
        _reference_stored_values(array, mask, name="column")
        if mask.any() or array.dtype.kind == "O"
        else array
    )
    return _reference_encode(stored, stored.dtype), mask


def _assert_same_array(observed: np.ndarray, expected: np.ndarray) -> None:
    assert observed.dtype == expected.dtype
    assert observed.shape == expected.shape
    assert observed.tobytes() == expected.tobytes()


_TEXT_COLUMNS = {
    "ascii_bytes": np.array([b"T cell", b"B", b"", b"NK cell", b"B"], dtype=object),
    "utf8_bytes": np.array(
        ["Zellé".encode(), "中文".encode(), "😀 x".encode(), b"plain", b""],
        dtype=object,
    ),
    # NumPy sizes byte strings by their longest value, trailing NULs included,
    # as the length of the decoded text counts them.
    "bytes_with_trailing_nul": np.array(
        [b"ab\x00\x00", b"abc", b"\x00", b"a"], dtype=object
    ),
    "utf8_bytes_with_trailing_nul": np.array(
        ["é\x00\x00".encode(), b"xy", "中".encode()], dtype=object
    ),
    "numpy_bytes": np.array(
        [np.bytes_(b"a\x00"), np.bytes_(b"bc"), b"d"], dtype=object
    ),
    "str": np.array(["T cell", "", "Zellé", "中文😀", "B"], dtype=object),
    "str_with_trailing_nul": np.array(["a\x00\x00", "bc", "d"], dtype=object),
    # str() of an np.str_ drops a trailing NUL that its length counts.
    "numpy_str": np.array([np.str_("a\x00"), np.str_("bcd"), "e"], dtype=object),
    "str_and_bytes": np.array(["a", b"bb", "ccc"], dtype=object),
    "mixed_objects": np.array(["a", b"b", 3, 2.5, True, None], dtype=object),
    "fixed_ascii": np.array([b"short", b"a", b"", b"mid"], dtype="S10"),
    "fixed_utf8": np.array(["é".encode(), b"zz", "中".encode(), b""], dtype="S6"),
    "fixed_inner_nul": np.array([b"a\x00b", b"c", b"\x00d"], dtype="S5"),
    "fixed_text": np.array(["x", "longer", "", "中"], dtype="U12"),
    "big_endian_text": np.array(["big", "中😀", ""], dtype=">U8"),
    # NumPy sizes values that are all empty differently through np.asarray.
    "all_empty_bytes": np.array([b"", b"", b""], dtype=object),
    "all_empty_str": np.array(["", "", ""], dtype=object),
    "one_empty_bytes": np.array([b""], dtype=object),
    "no_rows_object": np.array([], dtype=object),
    "no_rows_fixed": np.array([], dtype="S3"),
    "no_rows_text": np.array([], dtype="U4"),
}


def _missing(values: np.ndarray, pattern: str) -> np.ndarray:
    rows = np.arange(len(values))
    if pattern == "none":
        return np.zeros(len(values), dtype=bool)
    if pattern == "some":
        return rows % 3 == 1
    return np.ones(len(values), dtype=bool)


@pytest.mark.parametrize("pattern", ["none", "some", "all"])
@pytest.mark.parametrize("name", sorted(_TEXT_COLUMNS))
def test_text_columns_store_what_the_per_value_conversion_stored(
    name: str, pattern: str
) -> None:
    values = _TEXT_COLUMNS[name]
    missing = _missing(values, pattern)

    _assert_same_array(
        stored_values(values, missing, name="label"),
        _reference_stored_values(values, missing, name="label"),
    )


_DECODED_CASES = {
    **_TEXT_COLUMNS,
    "fixed_2d": np.array([[b"a", b"bb"], [b"ccc", b""]], dtype="S3"),
    "fixed_utf8_2d": np.array(
        [["é".encode(), b"b"], [b"c", "中".encode()]], dtype="S3"
    ),
    "object_bytes_2d": np.array([[b"a", b"bb"], [b"c", "é".encode()]], dtype=object),
    "object_str_2d": np.array([["a", "bb"], ["ccc", ""]], dtype=object),
    "lone_bytes": np.asarray(b"lone", dtype=object),
    "lone_fixed": np.asarray(b"lone", dtype="S6"),
    "bytes_and_none": np.array([b"a", None, b"ccc"], dtype=object),
}


@pytest.mark.parametrize("name", sorted(_DECODED_CASES))
def test_metadata_values_decode_as_the_per_value_decoding_did(name: str) -> None:
    values = _DECODED_CASES[name]

    _assert_same_array(_decode_metadata_values(values), _reference_decode(values))


@pytest.mark.parametrize("name", sorted(_DECODED_CASES))
def test_text_arrays_hold_the_text_of_each_value(name: str) -> None:
    from cytearc.storage.arrays import text_array

    values = _DECODED_CASES[name]

    _assert_same_array(
        text_array(values),
        np.asarray(
            [_reference_text_value(value) for value in values.flat], dtype=str
        ).reshape(values.shape),
    )


_CATEGORICALS = {
    # The unused category is wider than any used one, and code 5 names none.
    "text_bytes": (
        np.array([0, 1, -1, 0, 5, 1], dtype=np.int8),
        np.array(
            [b"T cell", b"B", b"an unused category, wider than any"], dtype=object
        ),
    ),
    "utf8_bytes": (
        np.array([2, 0, 1, 2], dtype=np.int16),
        np.array(["Zellé".encode(), "中文".encode(), "😀".encode()], dtype=object),
    ),
    "str": (
        np.array([1, 1, 0, -1], dtype=np.int32),
        np.array(["low", "middle"], dtype=object),
    ),
    "fixed_bytes": (
        np.array([2, 0, -1, 2], dtype=np.int8),
        np.array([b"low", b"mid", b"high"], dtype="S4"),
    ),
    "fixed_text": (
        np.array([0, 1, 1], dtype=np.int64),
        np.array(["x", "yy"], dtype="U9"),
    ),
    "integers": (
        np.array([0, 2, -1, 1, 2], dtype=np.int8),
        np.array([10, 20, 30], dtype=np.int64),
    ),
    "floats": (
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([0.5, 1.25], dtype=np.float32),
    ),
    "booleans": (np.array([1, 0, -1], dtype=np.int8), np.array([False, True])),
    "unsigned": (
        np.array([0, 1], dtype=np.int8),
        np.array([7, 2**40], dtype=np.uint64),
    ),
    # Present rows use only the integer, so the column holds int64.
    "mixed_integers_used": (
        np.array([0, 0, -1], dtype=np.int8),
        np.array([4, "text", 2.5], dtype=object),
    ),
    "mixed_text_used": (
        np.array([0, 1, 2], dtype=np.int8),
        np.array([4, "text", 2.5], dtype=object),
    ),
    "none_category": (
        np.array([0, 1], dtype=np.int8),
        np.array(["a", None], dtype=object),
    ),
    "all_missing": (
        np.array([-1, -1, 3], dtype=np.int8),
        np.array([b"x", b"yy"], dtype=object),
    ),
    "no_rows": (np.array([], dtype=np.int8), np.array([b"x"], dtype=object)),
    "no_categories": (np.array([0, -1], dtype=np.int8), np.array([], dtype=object)),
}


@pytest.mark.parametrize("name", sorted(_CATEGORICALS))
def test_categorical_columns_store_what_their_decoded_labels_stored(name: str) -> None:
    from cytearc.metadata.encoding import stored_categorical_values

    codes, categories = _CATEGORICALS[name]
    labels, expected_missing = decode_categories(codes, categories)

    stored, missing = stored_categorical_values(codes, categories, name="label")

    _assert_same_array(
        stored, _reference_stored_values(labels, expected_missing, name="label")
    )
    np.testing.assert_array_equal(missing, expected_missing)


@pytest.mark.parametrize(
    "name", ["text_bytes", "utf8_bytes", "integers", "all_missing"]
)
def test_a_categorical_column_writes_the_arrays_of_its_decoded_labels(
    name: str,
) -> None:
    codes, categories = _CATEGORICALS[name]
    group = zarr.open_group(store=MemoryStore(), mode="w")

    write_metadata_column(group, "label", codes, categories=categories)

    expected, mask = _reference_column(*decode_categories(codes, categories))
    column = group["label"]
    _assert_same_array(column[:], expected)
    assert fingerprint_array(column[:]) == fingerprint_array(expected)
    assert column.chunks == (column_chunk_rows(len(codes), expected.dtype.itemsize),)
    if mask.any():
        np.testing.assert_array_equal(group[column.attrs["missing_mask"]][:], mask)
    else:
        assert "missing_mask" not in column.attrs


def test_category_codes_and_categories_are_checked() -> None:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    categories = np.array([b"a", b"b"], dtype=object)

    with pytest.raises(ValueError, match="takes its missing rows from its category"):
        write_metadata_column(
            group,
            "label",
            np.array([0, 1]),
            np.array([False, True]),
            categories=categories,
        )
    with pytest.raises(TypeError, match="needs integer category codes"):
        write_metadata_column(
            group, "label", np.array([0.0, 1.0]), categories=categories
        )
    with pytest.raises(ValueError, match="one category per position"):
        write_metadata_column(
            group, "label", np.array([0, 1]), categories=np.array([[b"a"], [b"b"]])
        )
    assert list(group.keys()) == []


@pytest.mark.parametrize(
    "values",
    [
        np.array([b"ok", b"\xff", b"later"], dtype=object),
        np.array([b"ok", "é".encode()[:1], b"x"], dtype="S2"),
    ],
    ids=["object", "fixed"],
)
def test_invalid_utf8_raises_the_error_that_names_the_column(
    values: np.ndarray,
) -> None:
    present = np.zeros(len(values), dtype=bool)
    with pytest.raises(ValueError) as expected:
        _reference_stored_values(values, present, name="label")
    with pytest.raises(ValueError) as observed:
        stored_values(values, present, name="label")
    assert str(observed.value) == str(expected.value)
    assert "Column 'label' holds text that is not valid UTF-8" in str(observed.value)
    assert str(observed.value.__cause__) == str(expected.value.__cause__)

    # A missing row is never decoded.
    invalid_missing = np.array([False, True, False])
    _assert_same_array(
        stored_values(values, invalid_missing, name="label"),
        _reference_stored_values(values, invalid_missing, name="label"),
    )
    # The decoding of stored values raises what it raised.
    with pytest.raises(UnicodeDecodeError) as expected_decode:
        _reference_decode(values)
    with pytest.raises(UnicodeDecodeError) as observed_decode:
        _decode_metadata_values(values)
    assert str(observed_decode.value) == str(expected_decode.value)


def _text_value_calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Count the calls to ``text_value`` from every module that converts columns."""
    import cytearc.metadata.encoding as encoding_module

    calls: list[Any] = []
    original = arrays_module.text_value

    def counting(value: Any) -> str:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(arrays_module, "text_value", counting)
    # The per-value conversion of earlier releases called it from here too.
    monkeypatch.setattr(encoding_module, "text_value", counting, raising=False)
    return calls


def test_text_columns_convert_without_a_call_per_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 100_000
    texts = np.array([f"cell-{row}" for row in range(rows)], dtype=object)
    utf8 = np.array([f"Zellé-{row}".encode() for row in range(rows)], dtype=object)
    missing = np.arange(rows) % 5 == 0
    calls = _text_value_calls(monkeypatch)
    group = zarr.open_group(store=MemoryStore(), mode="w")

    write_metadata_column(group, "text", texts)
    write_metadata_column(group, "ascii_bytes", texts.astype(bytes).astype(object))
    write_metadata_column(group, "utf8_bytes", utf8, missing)
    write_metadata_column(group, "fixed", texts.astype("S"))
    stored_values(texts.astype(str), missing, name="unicode")

    assert calls == []
    assert group["utf8_bytes"][:2].tolist() == ["", "Zellé-1"]
    # Objects of several types still convert one at a time.
    mixed = np.array([b"a", 1] * 500, dtype=object)
    stored_values(mixed, np.zeros(len(mixed), dtype=bool), name="mixed")
    assert len(calls) == len(mixed)


def test_an_h5ad_import_converts_each_category_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = 100_000
    text = h5py.string_dtype()
    path = tmp_path / "cells.h5ad"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X", data=np.ones((rows, 2), dtype=np.float32))
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset(
            "_index", data=np.array([f"cell-{row}" for row in range(rows)], dtype=text)
        )
        column = obs.create_group("cell_type")
        column.create_dataset("codes", data=(np.arange(rows) % 4 - 1).astype(np.int8))
        column.create_dataset(
            "categories", data=np.array(["T cell", "B cell", "Zellé"], dtype=text)
        )
        obs.attrs["column-order"] = np.array(["cell_type"], dtype=text)
        var = h5.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(["g0", "g1"], dtype=text))
    calls = _text_value_calls(monkeypatch)
    reader = H5adReader(str(path))
    try:
        H5adToZarr(reader, str(tmp_path / "cells.zarr"), nthreads=1)
    finally:
        reader.close()

    assert calls == []
    cells = zarr.open_group(str(tmp_path / "cells.zarr"), mode="r")["cellData"]
    assert cells["cell_type"][:5].tolist() == ["", "T cell", "B cell", "Zellé", ""]
    assert cells["ids"][:2].tolist() == ["cell-0", "cell-1"]


def _write_h5ad_columns(path: Path) -> tuple[list[str], dict[str, slice]]:
    """Write an h5ad whose obs and var hold every column kind an import encodes.

    Returns:
        The obs columns, and the features of each assay of the split by
        ``feature_types``.
    """
    rng = np.random.default_rng(5)
    n_cells, n_genes, n_proteins = 60, 8, 4
    text = h5py.string_dtype()

    def categorical(group: h5py.Group, name: str, codes: Any, categories: Any) -> None:
        column = group.create_group(name)
        column.attrs["encoding-type"] = "categorical"
        column.create_dataset("codes", data=np.asarray(codes, dtype=np.int8))
        column.create_dataset("categories", data=categories)

    def nullable(group: h5py.Group, name: str, values: Any, mask: Any) -> None:
        column = group.create_group(name)
        column.create_dataset("values", data=values)
        column.create_dataset("mask", data=np.asarray(mask, dtype=bool))

    labels = [
        "T cell",
        "B",
        "Zellé",
        "中文",
        "😀 mixed",
        "unused, and wider than any used",
    ]
    with h5py.File(path, "w") as h5:
        matrix = csr_matrix(rng.poisson(0.5, (n_cells, n_genes + n_proteins)))
        x = h5.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = matrix.shape
        x.create_dataset("data", data=matrix.data.astype(np.float32))
        x.create_dataset("indices", data=matrix.indices)
        x.create_dataset("indptr", data=matrix.indptr)
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset(
            "_index", data=np.array([f"cell-{i}-é" for i in range(n_cells)], dtype=text)
        )
        categorical(
            obs, "cell_type", rng.integers(0, 5, n_cells), np.array(labels, dtype=text)
        )
        categorical(
            obs,
            "with_missing",
            np.where(rng.random(n_cells) < 0.25, -1, rng.integers(0, 3, n_cells)),
            np.array(["a", "bb", "ccc"], dtype=text),
        )
        categorical(
            obs, "all_missing", np.full(n_cells, -1), np.array(["x", "yy"], dtype=text)
        )
        categorical(
            obs, "int_cat", rng.integers(-1, 4, n_cells), np.array([3, 1, 4, 1000])
        )
        categorical(
            obs, "float_cat", rng.integers(0, 2, n_cells), np.array([0.5, -1.25])
        )
        categorical(
            obs, "bool_cat", rng.integers(0, 2, n_cells), np.array([False, True])
        )
        nullable(
            obs,
            "note",
            np.array(
                [f"note {i}" if i % 3 else "" for i in range(n_cells)], dtype=text
            ),
            np.arange(n_cells) % 3 == 0,
        )
        obs.create_dataset(
            "barcode",
            data=np.array([f"AAAC-{i}-ü" for i in range(n_cells)], dtype=text),
        )
        obs.create_dataset(
            "fixed", data=np.array([f"fx{i % 7}".encode() for i in range(n_cells)])
        )
        obs.create_dataset(
            "fixed_utf8",
            data=np.array([("é" * (i % 3)).encode() + b"z" for i in range(n_cells)]),
        )
        obs.create_dataset(
            "score", data=np.where(rng.random(n_cells) < 0.2, np.nan, 1.5)
        )
        obs.create_dataset("count", data=rng.integers(0, 100, n_cells))
        obs.create_dataset("flag", data=rng.random(n_cells) < 0.5)
        nullable(obs, "nullable_int", np.arange(n_cells), np.arange(n_cells) % 4 == 0)
        # AnnData 0.7 kept a categorical's categories apart from its codes.
        obs.create_dataset("legacy", data=rng.integers(-1, 3, n_cells).astype(np.int8))
        obs.create_group("__categories").create_dataset(
            "legacy", data=np.array([b"low", b"mid", b"high"])
        )
        obs_names = [name for name in obs if name not in {"_index", "__categories"}]
        obs.attrs["column-order"] = np.array(obs_names, dtype=text)
        var = h5.create_group("var")
        var.attrs["_index"] = "_index"
        genes = [f"g{j}" for j in range(n_genes)]
        proteins = [f"p{j}" for j in range(n_proteins)]
        var.create_dataset("_index", data=np.array(genes + proteins, dtype=text))
        categorical(
            var,
            "feature_types",
            [0] * n_genes + [1] * n_proteins,
            np.array(["Gene Expression", "Antibody Capture"], dtype=text),
        )
        # Genes use only short symbols and proteins a longer one, so each
        # assay stores its own width.
        categorical(
            var,
            "symbol",
            [j % 2 for j in range(n_genes)] + [2, 2, -1, 3],
            np.array(["A", "B", "a protein symbol", "中"], dtype=text),
        )
        var.create_dataset(
            "chrom",
            data=np.array(
                [f"chr{j % 3}" for j in range(n_genes + n_proteins)], dtype=text
            ),
        )
        var.attrs["column-order"] = np.array(
            ["feature_types", "symbol", "chrom"], dtype=text
        )
        h5.create_group("obsm")
    return obs_names, {
        "RNA": slice(0, n_genes),
        "ADT": slice(n_genes, n_genes + n_proteins),
    }


def _assert_stored_column(
    group: zarr.Group, key: str, values: Any, missing: Any
) -> None:
    expected, mask = _reference_column(values, missing)
    column = group[key]
    _assert_same_array(column[:], expected)
    assert fingerprint_array(column[:]) == fingerprint_array(expected)
    assert column.chunks == (column_chunk_rows(len(expected), expected.dtype.itemsize),)
    if mask.any():
        _assert_same_array(group[column.attrs["missing_mask"]][:], mask)
    else:
        assert "missing_mask" not in column.attrs


def test_an_h5ad_import_stores_the_columns_of_the_per_value_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "columns.h5ad"
    obs_names, assays = _write_h5ad_columns(path)
    reader = H5adReader(str(path))
    try:
        H5adToZarr(
            reader,
            str(tmp_path / "store.zarr"),
            nthreads=1,
            assay_split_key="feature_types",
        ).dump()
        root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
        cells = root["cellData"]
        assert set(obs_names) <= set(cells.keys())
        # Every column holds what master's writer stored from the decoded
        # labels and values that the reader returns.
        for name in obs_names:
            _assert_stored_column(cells, name, *reader._read_column("obs", name))
        ids = reader.cell_ids()
        _assert_same_array(cells["ids"][:], _reference_encode(ids, ids.dtype))
        _assert_same_array(cells["names"][:], _reference_encode(ids, ids.dtype))
        widths = {}
        for assay, rows in assays.items():
            group = root[f"{assay}/featureData"]
            for name in ("symbol", "chrom"):
                values, missing = reader._read_column("var", name)
                _assert_stored_column(group, name, values[rows], missing[rows])
            widths[assay] = group["symbol"].dtype
        assert widths == {"RNA": np.dtype("U1"), "ADT": np.dtype("U16")}
        # The stored labels are those of the codes in the file.
        for name, categories in (
            ("with_missing", ["a", "bb", "ccc"]),
            ("legacy", ["low", "mid", "high"]),
        ):
            codes = reader.h5[
                f"obs/{name}/codes" if name != "legacy" else "obs/legacy"
            ][:]
            assert cells[name][:].tolist() == [
                categories[code] if 0 <= code < len(categories) else ""
                for code in codes.tolist()
            ]
        # The categorical attributes keep every category and the source order.
        assert cells["cell_type"].attrs["levels"] == [
            "T cell",
            "B",
            "Zellé",
            "中文",
            "😀 mixed",
            "unused, and wider than any used",
        ]
        assert cells["with_missing"].attrs["display"]["missing_label"] == "NA"
        assert "missing_label" not in cells["cell_type"].attrs["display"]
        # The reader's public columns still hold decoded labels as objects.
        public = dict(reader.get_cell_columns())
        assert public["cell_type"].dtype == object
        np.testing.assert_array_equal(
            public["with_missing"],
            np.asarray(reader._read_column("obs", "with_missing")[0]),
        )
    finally:
        reader.close()


def _stored_columns(root: zarr.Group, path: str) -> dict[str, np.ndarray]:
    group = root[path]
    return {key: group[key][:] for key in group.array_keys()}


def test_csv_10x_matrix_market_and_sparse_imports_store_their_columns_as_before(
    tmp_path: Path,
) -> None:
    # CSV: a text column with a missing value and non-ASCII text.
    csv = tmp_path / "counts.csv"
    csv.write_text("cell,batch,score,g0,g1\nc0,é,1.5,1,0\nc1,,2.5,0,2\nc2,中文,,3,1\n")
    CSVtoZarr(
        CSVReader(str(csv), id_column=0, cell_data_cols=["batch", "score"]),
        str(tmp_path / "csv.zarr"),
        assay_name="RNA",
        nthreads=1,
    ).dump()
    root = zarr.open_group(str(tmp_path / "csv.zarr"), mode="r")
    # The CSV writer passes blank text cells as missing rows of fixed-width text.
    _assert_stored_column(
        root["cellData"], "batch", np.array(["é", "", "中文"]), [False, True, False]
    )
    _assert_stored_column(root["cellData"], "score", np.array([1.5, 2.5, np.nan]), None)
    for key in ("ids", "names"):
        _assert_same_array(
            root[f"cellData/{key}"][:],
            _reference_encode(np.array(["c0", "c1", "c2"]), "U2"),
        )

    # 10x HDF5 and Matrix Market: the reader's own metadata columns and IDs.
    h5 = tmp_path / "counts.h5"
    with h5py.File(h5, "w") as handle:
        matrix = handle.create_group("matrix")
        counts = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]]))
        matrix.create_dataset("data", data=counts.data.astype(np.int32))
        matrix.create_dataset("indices", data=counts.indices.astype(np.int64))
        matrix.create_dataset("indptr", data=counts.indptr.astype(np.int64))
        matrix.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
        matrix.create_dataset("barcodes", data=np.array([b"AAAC-1", b"TTTG-1"]))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.array([b"ENSG1", b"ENSG2", b"ENSG3"]))
        features.create_dataset(
            "name", data=np.array(["Gène".encode(), "中".encode(), b"CD3"])
        )
        features.create_dataset("feature_type", data=np.array([b"Gene Expression"] * 3))
        features.create_dataset("genome", data=np.array([b"GRCh38"] * 3))
    mtx = tmp_path / "mtx"
    mtx.mkdir()
    (mtx / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 1\n3 1 2\n2 2 3\n"
    )
    (mtx / "features.tsv").write_text(
        "ENSG1\tGène\tGene Expression\nENSG2\t中\tGene Expression\nENSG3\tCD3\tGene Expression\n"
    )
    (mtx / "barcodes.tsv").write_text("AAAC-1\nTTTG-1\n")
    for name, reader, writer in (
        ("h5", CrH5Reader(str(h5)), CrToZarr),
        ("mtx", MtxReader(inspect_mtx(str(mtx))[0]), MtxToZarr),
    ):
        try:
            writer(reader, str(tmp_path / f"{name}.zarr"), nthreads=1).dump()
            root = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="r")
            for column, values in reader.get_cell_columns():
                _assert_stored_column(root["cellData"], column, values, None)
            for column, values in reader.get_feature_columns():
                if np.asarray(values).dtype.kind == "O" and all(
                    value is None for value in values
                ):
                    continue
                _assert_stored_column(root["RNA/featureData"], column, values, None)
            barcodes = np.array(reader.cell_names())
            _assert_same_array(
                root["cellData/ids"][:], _reference_encode(barcodes, barcodes.dtype)
            )
            for key, values in (
                ("ids", reader.feature_ids("RNA")),
                ("names", reader.feature_names("RNA")),
            ):
                _assert_same_array(
                    root[f"RNA/featureData/{key}"][:], _reference_encode(values, None)
                )
            assert _stored_columns(root, "RNA/featureData")["names"].tolist() == [
                "Gène",
                "中",
                "CD3",
            ]
        finally:
            reader.close()

    # Sparse: byte-string cell IDs and display names.
    store = MemoryStore()
    SparseToZarr(
        csr_matrix(np.array([[1, 0], [0, 2], [3, 1]], dtype=np.float32)),
        store,
        cell_ids=np.array([b"c-\xc3\xa9", b"c-2", b"c-3"], dtype=object),
        feature_ids=["g1", "g2"],
        feature_names=np.array(["Gène", "中"], dtype=object),
        nthreads=1,
    ).dump()
    root = zarr.open_group(store=store, mode="r")
    assert root["cellData/ids"][:].tolist() == ["c-é", "c-2", "c-3"]
    _assert_same_array(
        root["RNA/featureData/names"][:],
        _reference_encode(np.array(["Gène", "中"]), None),
    )
