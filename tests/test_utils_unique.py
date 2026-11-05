"""The sort-based replacements of ``np.unique`` equal it on integers.

NumPy 2.3 and later answer an ``np.unique`` call that asks for no indices,
inverse or counts by hashing the values, which on millions of distinct
integers takes tens of times as long as sorting them, so the clone stack's
modules find distinct values with ``sorted_unique`` and ``has_duplicates``.
"""

import ast

import numpy as np
import pytest

from cytearc.utils import arrays
from cytearc.utils.arrays import (
    sorted_unique,
    union_sorted,
    unique_inverse,
    unique_rows,
)
from tests.stack_modules import CYTEARC_ROOT, stack_module_paths

# The guard checks the clone stack's modules. Master's other modules keep their
# plain calls, on feature, chunk, group and float arrays where sorting saves
# nothing measurable.
# np.unique sorts when a call asks for the indices, the inverse or the counts
# of the values, and hashes them otherwise, whatever axis it names, as
# np.unique_values always does.
_SORTING_FLAGS = ("return_index", "return_inverse", "return_counts")


def _numpy_names(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """The names that a module binds to NumPy, and to functions imported from it."""
    modules = {"np", "numpy"}
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "numpy"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            functions.update(
                (alias.asname or alias.name, alias.name) for alias in node.names
            )
    return modules, functions


def _hashed_unique_lines(source: str) -> list[int]:
    """The lines of the calls of ``source`` whose distinct values NumPy hashes.

    A call of ``np.unique`` sorts only when it passes ``return_index``,
    ``return_inverse`` or ``return_counts`` as the literal True, by keyword
    or by position, so a flag held in a name or unpacked counts as hashing.
    """
    tree = ast.parse(source)
    modules, functions = _numpy_names(tree)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name):
            name = called.attr if called.value.id in modules else ""
        elif isinstance(called, ast.Name):
            name = functions.get(called.id, "")
        else:
            continue
        if name == "unique":
            flags = [
                *node.args[1:4],
                *(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in _SORTING_FLAGS
                ),
            ]
            if any(
                isinstance(flag, ast.Constant) and flag.value is True for flag in flags
            ):
                continue
        elif name != "unique_values":
            continue
        lines.append(node.lineno)
    return sorted(lines)


def test_stack_modules_find_distinct_values_by_sorting() -> None:
    paths = stack_module_paths()
    hashed = [
        f"{path.relative_to(CYTEARC_ROOT).as_posix()}:{line}"
        for path in paths
        for line in _hashed_unique_lines(path.read_text())
    ]
    assert not hashed, (
        "NumPy hashes the values of these calls, which takes many times as long "
        "as sorting them: use sorted_unique for integers, has_duplicates to test "
        "whether values repeat, and for other values a call that asks for the "
        f"inverse or the counts, which NumPy finds by sorting: {hashed}"
    )


def test_the_guard_finds_each_call_that_hashes_and_passes_sorting_calls() -> None:
    source = "\n".join(
        [
            "import numpy",
            "import numpy as np",
            "import numpy as onp",
            "from numpy import unique as distinct, unique_values",
            "a = np.unique(values)",
            "b = len(numpy.unique(values[keep]))",
            "c = np.unique(values, equal_nan=False, sorted=True)",
            "d = np.unique(values, return_counts=False)",
            "e = np.unique(values, return_inverse=wanted)",
            "f = np.unique(values, axis=0)",
            "g = onp.unique(values)",
            "h = distinct(values)",
            "i = unique_values(values) + np.unique_values(values)",
            "j = np.unique(values, **options)",
            "k = np.unique(*arguments)",
            "m = np.unique(values, return_counts=True)",
            "n = np.unique(rows, return_inverse=True, axis=0)",
            "o = np.unique(values, return_index=True)",
            "p = np.unique(values, True)",
            "q = np.unique(values, False, True)",
            "r = distinct(values, return_counts=True)",
            "s = np.unique_counts(values)",
            "t = pd.unique(values) + sorted_unique(values) + unique(values)",
            "def inner():",
            "    return [value for value in np.unique(found).tolist()]",
        ]
    )
    expected = [5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 25]
    assert _hashed_unique_lines(source) == expected
    assert _hashed_unique_lines("x = 1\n") == []


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("high", [1, 3, 1_000, 2**40])
def test_sorted_unique_and_unique_inverse_equal_numpy(seed: int, high: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.integers(-high, high + 1, size=int(rng.integers(0, 400)))
    assert np.array_equal(sorted_unique(values), np.unique(values))
    distinct, inverse = unique_inverse(values)
    expected, expected_inverse = np.unique(values, return_inverse=True)
    assert np.array_equal(distinct, expected)
    assert np.array_equal(inverse, expected_inverse)
    assert inverse.dtype == np.int64
    assert distinct.dtype == values.dtype


def test_sorted_unique_keeps_the_dtype_of_small_and_unsigned_integers() -> None:
    values = np.asarray([3, 1, 3, 255, 0], dtype=np.uint8)
    assert sorted_unique(values).tolist() == [0, 1, 3, 255]
    assert sorted_unique(values).dtype == np.uint8
    hashes = np.asarray([2**64 - 1, 5, 2**64 - 1], dtype=np.uint64)
    assert sorted_unique(hashes).tolist() == [5, 2**64 - 1]
    assert sorted_unique(np.asarray([True, False, True])).tolist() == [False, True]
    assert sorted_unique(np.asarray([7])).tolist() == [7]
    assert sorted_unique(np.asarray([], dtype=np.int64)).tolist() == []
    values = np.asarray([5, 3, 5, 9, 3, 3], dtype=np.int64)
    assert sorted_unique(values).tolist() == [3, 5, 9]
    assert sorted_unique(values[:1]).tolist() == [5]


@pytest.mark.parametrize("values", [np.zeros((2, 2), dtype=int), np.asarray([1.5])])
def test_sorted_unique_refuses_other_arrays(values) -> None:
    with pytest.raises(TypeError, match="one-dimensional array of integers"):
        sorted_unique(values)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("high", [1, 50, 2**40])
def test_has_duplicates_tells_whether_np_unique_drops_a_value(
    seed: int, high: int
) -> None:
    rng = np.random.default_rng(seed)
    drawn = rng.integers(-high, high + 1, size=int(rng.integers(0, 60)))
    # Drawn values, the same made distinct, ascending and descending, and as
    # finite floats, which is what the stack's callers pass.
    distinct = np.unique(drawn)
    for values in (drawn, distinct, distinct[::-1], drawn / 4, distinct / 4):
        expected = len(np.unique(values)) != len(values)
        assert arrays.has_duplicates(values) is expected


def test_has_duplicates_at_the_ends_and_of_signed_zeros() -> None:
    assert arrays.has_duplicates(np.asarray([], dtype=np.int64)) is False
    assert arrays.has_duplicates(np.asarray([3])) is False
    assert arrays.has_duplicates(np.asarray([3, 3])) is True
    assert arrays.has_duplicates(np.asarray([3, 4])) is False
    assert arrays.has_duplicates(np.asarray([4, 3])) is False
    # A repeat among ascending values ends the shortcut for ascending arrays.
    assert arrays.has_duplicates(np.asarray([1, 2, 3, 3])) is True
    assert arrays.has_duplicates(np.asarray([4, 1, 2, 4])) is True
    assert arrays.has_duplicates(np.asarray([2**64 - 1, 0, 2**64 - 1], np.uint64))
    # np.unique counts -0.0 and 0.0 as one value.
    assert arrays.has_duplicates(np.asarray([0.0, 1.5, -0.0])) is True


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("columns", [1, 2, 3])
def test_unique_rows_equals_numpy(seed: int, columns: int) -> None:
    rng = np.random.default_rng(seed)
    n_rows = int(rng.integers(1, 500))
    highs = rng.choice([1, 4, 50, 2**31], size=columns)
    rows = np.column_stack(
        [rng.integers(-1, high, size=n_rows) for high in highs]
    ).astype(np.int64)
    distinct, inverse = unique_rows(rows)
    expected, expected_inverse = np.unique(rows, axis=0, return_inverse=True)
    assert np.array_equal(distinct, expected)
    assert np.array_equal(inverse, expected_inverse.reshape(-1))
    assert np.array_equal(distinct[inverse], rows)


def test_unique_rows_never_overflows_wide_columns() -> None:
    big = np.iinfo(np.int64)
    rows = np.asarray(
        [[big.max, big.min], [big.min, big.max], [big.max, big.min], [0, 0]],
        dtype=np.int64,
    )
    distinct, inverse = unique_rows(rows)
    expected, expected_inverse = np.unique(rows, axis=0, return_inverse=True)
    assert np.array_equal(distinct, expected)
    assert np.array_equal(inverse, expected_inverse.reshape(-1))
    unsigned = np.asarray([[2**64 - 1, 1], [2**64 - 2, 1], [2**64 - 1, 1]], np.uint64)
    distinct, inverse = unique_rows(unsigned)
    assert distinct.tolist() == [[2**64 - 2, 1], [2**64 - 1, 1]]
    assert inverse.tolist() == [1, 0, 1]


def test_unique_rows_and_inverse_of_no_values() -> None:
    distinct, inverse = unique_rows(np.zeros((0, 2), dtype=np.int64))
    assert distinct.shape == (0, 2)
    assert inverse.shape == (0,)
    distinct, inverse = unique_inverse(np.asarray([], dtype=np.int32))
    assert distinct.dtype == np.int32 and distinct.shape == (0,)
    assert inverse.dtype == np.int64 and inverse.shape == (0,)


def test_unique_inverse_and_rows_refuse_other_arrays() -> None:
    with pytest.raises(TypeError, match="unique_inverse needs a one-dimensional"):
        unique_inverse(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(TypeError, match="unique_inverse needs a one-dimensional"):
        unique_inverse(np.asarray([0.5, 1.5]))
    with pytest.raises(TypeError, match="unique_rows needs a two-dimensional"):
        unique_rows(np.asarray([1, 2]))
    with pytest.raises(TypeError, match="unique_rows needs a two-dimensional"):
        unique_rows(np.asarray([[True], [False]]))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("high", [2, 50, 2**63 - 1])
def test_union_sorted_equals_numpy(seed: int, high: int) -> None:
    rng = np.random.default_rng(seed)
    held = np.unique(rng.integers(0, high, size=int(rng.integers(0, 300)))).astype(
        np.uint64
    )
    # Values repeat among themselves and with the held ones, in any order.
    values = rng.integers(0, high, size=int(rng.integers(0, 300))).astype(np.uint64)
    if len(held) and len(values):
        values[: len(values) // 3] = rng.choice(held, size=len(values) // 3)
    merged = union_sorted(held, values)
    assert np.array_equal(merged, np.union1d(held, values))
    assert merged.dtype == np.uint64


def test_union_sorted_of_empty_and_extreme_values() -> None:
    hashes = np.asarray([0, 2**64 - 1], dtype=np.uint64)
    assert union_sorted(
        hashes, np.asarray([2**64 - 1, 5, 5], dtype=np.uint64)
    ).tolist() == [0, 5, 2**64 - 1]
    empty = np.empty(0, dtype=np.uint64)
    assert union_sorted(empty, np.asarray([3, 1, 3], dtype=np.uint64)).tolist() == [
        1,
        3,
    ]
    assert union_sorted(hashes, empty) is hashes
    assert union_sorted(empty, empty).tolist() == []
    signed = np.asarray([-5, 7], dtype=np.int64)
    assert union_sorted(signed, np.asarray([-9, 7], dtype=np.int64)).tolist() == [
        -9,
        -5,
        7,
    ]


def test_union_sorted_of_a_million_held_values_keeps_every_new_one() -> None:
    held = np.arange(0, 2_000_000, 2, dtype=np.uint64)
    values = np.arange(1, 20_001, 2, dtype=np.uint64)
    merged = union_sorted(held, values)
    assert len(merged) == len(held) + len(values)


def test_union_sorted_kernel_runs_as_python_too() -> None:
    held = np.asarray([1, 4, 9], dtype=np.int64)
    new = np.asarray([0, 4, 4, 10], dtype=np.int64)
    kernel = arrays._union_kernel
    assert kernel(held, new, held[:0]) == 5
    out = np.empty(5, dtype=np.int64)
    assert kernel(held, new, out) == 5
    assert out.tolist() == [0, 1, 4, 9, 10]
    assert kernel(held[:0], new[:0], out[:0]) == 0


@pytest.mark.parametrize(
    ("held", "values"),
    [
        (np.zeros((2, 2), dtype=np.uint64), np.asarray([1], dtype=np.uint64)),
        (np.asarray([1], dtype=np.uint64), np.asarray([1.5])),
        (np.asarray([1], dtype=np.uint64), np.asarray([1], dtype=np.int64)),
    ],
)
def test_union_sorted_refuses_other_arrays(held, values) -> None:
    with pytest.raises(TypeError, match="union_sorted needs"):
        union_sorted(held, values)
