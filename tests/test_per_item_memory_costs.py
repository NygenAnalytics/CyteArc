"""Trace each per-item memory cost against the pandas or NumPy operation that it stands for, at sizes where the slack of a traced call is negligible beside the charge.

A per-item cost multiplies the data, so it must bound what the operation
holds at every size rather than average it. The sizes run from 300,000 values
to 3 million, among them those just above 77% of a power of two, at which
pandas' hash tables hold the most buckets a value, and one just above the
size hint of about a million values that pandas sizes a table for at first.
"""

import functools
import gc
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.storage.geometry import variable_text_extent
from cytearc.utils.memory_budget import (
    factorize_bytes,
    index_lookup_bytes,
    object_array_bytes,
    python_text_bytes,
    sort_bytes,
)
from tests.memory_helpers import held_within_charge

pytestmark = pytest.mark.slow

SIZES = (300_000, 403_703, 807_405, 1_048_584, 1_614_810, 3_000_000)


@functools.lru_cache(maxsize=1)
def _ids(n_values: int) -> np.ndarray:
    """Distinct strings of 11 ASCII characters, shuffled, as an object array."""
    order = np.random.default_rng(n_values).permutation(n_values)
    return np.array([f"ID{index:09d}" for index in order.tolist()], dtype=object)


def _traced(call: Callable[[], Any]) -> tuple[Any, int, int]:
    """Return what ``call`` returns, the bytes that it keeps and the most that it held, as traced."""
    gc.collect()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        result = call()
        kept, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, kept - before, peak - before


@pytest.mark.parametrize("n_values", SIZES)
@pytest.mark.parametrize("kind", ["strings", "integers"])
@pytest.mark.parametrize("n_distinct", [None, 10], ids=["distinct", "ten labels"])
@pytest.mark.parametrize("unique", [False, True], ids=["factorize", "unique"])
def test_factorizing_holds_no_more_than_its_charge(
    n_values, kind, n_distinct, unique
) -> None:
    values = (
        _ids(n_values)
        if kind == "strings"
        else np.random.default_rng(1).permutation(n_values).astype(np.int64)
    )
    if n_distinct is not None:
        picks = np.random.default_rng(2).integers(0, n_distinct, size=n_values)
        values = values[:n_distinct][picks]
    if unique:
        returned, _, peak = _traced(lambda: pd.unique(values))
        outputs = returned.nbytes
    else:
        returned, _, peak = _traced(lambda: pd.factorize(values))
        outputs = returned[0].nbytes + returned[1].nbytes
    # What pandas holds beside the codes and distinct values that it returns:
    # its table, sized in powers of two, its working arrays and the buffer in
    # which it gathers the distinct values.
    held = peak - outputs
    charged = factorize_bytes(n_values, n_distinct=n_distinct)
    assert held_within_charge(held, charged)
    if kind == "strings":
        # A table of numbers takes 16 bytes a bucket rather than 24, beside
        # one working array, so the charge may be three times what it holds.
        assert charged < 2.5 * held


@pytest.mark.parametrize("n_values", SIZES)
@pytest.mark.parametrize("shape", ["as many lookups", "1,000 values", "1,000 lookups"])
@pytest.mark.parametrize("kind", ["inferred", "object", "integers"])
def test_index_lookups_hold_no_more_than_their_charge(n_values, shape, kind) -> None:
    n_index = 1_000 if shape == "1,000 values" else n_values
    n_lookups = 1_000 if shape == "1,000 lookups" else n_values
    rng = np.random.default_rng(3)
    if kind == "integers":
        index_values = rng.permutation(n_index).astype(np.int64)
        lookups: Any = rng.integers(0, n_index, size=n_lookups)
    else:
        index_values = _ids(n_values)[:n_index]
        lookups = index_values[rng.integers(0, n_index, size=n_lookups)]
        if kind == "object":
            # A lookup of names as a list, as a lookup of the names a caller
            # gives is.
            lookups = lookups.tolist()
    dtype = object if kind == "object" else None
    # pandas copies the index's values, infers the type of an object array,
    # hashes the values, and copies and infers the lookups before it returns
    # their positions.
    _, _, held = _traced(
        lambda: pd.Index(index_values, dtype=dtype).get_indexer(lookups)
    )
    charged = index_lookup_bytes(n_index, n_lookups, objects=kind != "integers")
    assert held_within_charge(held, charged)
    assert charged < 2.5 * held


@pytest.mark.parametrize("n_values", [300_000, 1_000_000, 3_000_000])
@pytest.mark.parametrize("utf8_bytes", [16, 18, 40])
def test_numpy_keeps_variable_width_text_within_its_extent(
    n_values, utf8_bytes
) -> None:
    # Values of one length in UTF-8, made from Python strings as Zarr's
    # decode makes them. NumPy keeps a 16-byte element for each and, beyond
    # 15 bytes, its bytes behind a one-byte size in an arena that it grows by
    # a quarter at a time (NEP 55).
    tail = "x" * (utf8_bytes - 11)
    strings = np.array([value + tail for value in _ids(n_values).tolist()])
    strings = strings.astype(object)
    values, kept, _ = _traced(lambda: strings.astype(np.dtypes.StringDType()))
    extent = variable_text_extent(values)
    assert kept >= n_values * (16 + utf8_bytes + 1)
    assert held_within_charge(kept, extent.keptBytes())
    assert extent.keptBytes() <= n_values * extent.valueBytes()


def test_numpy_keeps_long_text_within_its_extent() -> None:
    # A value of more than 255 bytes takes an 8-byte size in the arena.
    n_values, tail = 300_000, "x" * 289
    strings = np.array([value + tail for value in _ids(n_values).tolist()])
    strings = strings.astype(object)
    values, kept, _ = _traced(lambda: strings.astype(np.dtypes.StringDType()))
    extent = variable_text_extent(values)
    assert kept >= n_values * (16 + 300 + 8)
    assert held_within_charge(kept, extent.keptBytes())


@pytest.mark.parametrize(
    "character",
    ["a", "é", "中", "\U0001f600"],
    ids=["ASCII", "Latin-1", "UCS-2", "UCS-4"],
)
@pytest.mark.parametrize("length", [1, 8, 40])
def test_python_strings_hold_no_more_than_their_charge(character, length) -> None:
    n_values = 300_000
    # Distinct strings, as a column's rows are, referenced from an object
    # array.
    texts, kept, _ = _traced(
        lambda: np.array(
            [character * length + str(index) for index in range(n_values)],
            dtype=object,
        )
    )
    characters = sum(map(len, texts.tolist()))
    charged = object_array_bytes(n_values) + python_text_bytes(
        n_values, characters, widest=ord(character)
    )
    assert kept >= object_array_bytes(n_values)
    assert held_within_charge(kept, charged)
    assert charged < 2 * kept


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64", "U12"]
)
@pytest.mark.parametrize("stable", [False, True], ids=["default", "stable"])
@pytest.mark.parametrize("order", [False, True], ids=["sort", "argsort"])
def test_sorts_hold_no_more_than_their_charge(dtype, stable, order) -> None:
    n_values = 1_000_000
    rng = np.random.default_rng(4)
    values = rng.integers(0, 100, size=n_values).astype(dtype)
    kind = "stable" if stable else None
    sort = np.argsort if order else np.sort
    _, _, held = _traced(lambda: sort(values, kind=kind))
    charged = sort_bytes(n_values, values.itemsize, stable=stable, order=order)
    assert held >= (8 * n_values if order else values.nbytes)
    assert held_within_charge(held, charged)
