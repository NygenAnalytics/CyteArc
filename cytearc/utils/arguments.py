import math
from collections.abc import Sequence
from numbers import Real

import numpy as np


def float_argument(value: object, name: str) -> float:
    """Return ``value`` as a finite Python float.

    Python and NumPy integers and floats are accepted, so ``1`` and ``1.0``
    give the same value, and booleans are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def bool_argument(value: object, name: str) -> bool:
    """Return ``value`` as a Python boolean.

    Python and NumPy booleans are accepted, so ``np.True_`` gives ``True``,
    and integers and text are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is not a boolean.
    """
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def optional_bool_argument(value: object, name: str) -> bool | None:
    """Return ``value`` as a Python boolean, or None.

    Accepts what :func:`bool_argument` accepts, and None.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is neither None nor a boolean.
    """
    if value is None:
        return None
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean or None")
    return bool(value)


#: The most characters of a value that an error message quotes.
QUOTED_LIMIT = 80


def quoted_value(value: object) -> str:
    """Return ``repr(value)`` for an error message, cut to ``QUOTED_LIMIT``.

    A value longer than that, such as a contig ID of 20,000 characters, keeps
    its start and its end around ``...``, so the message stays readable and a
    long path still shows its file's name.
    """
    text = repr(value)
    if len(text) <= QUOTED_LIMIT:
        return text
    head = (QUOTED_LIMIT - 3) // 2
    return text[:head] + "..." + text[head + 3 - QUOTED_LIMIT :]


#: What pandas infers for object labels of one type that print as they compare.
_ONE_TEXT_PER_VALUE = frozenset({"string", "bytes", "integer", "boolean", "empty"})


def label_array(labels: object) -> np.ndarray:
    """Return labels as a NumPy array, the labels of a list each of its own type.

    An array, or a pandas Series, Index or Categorical, gives its values as
    they are. NumPy would cast a list to one type: ``[b"a", "a"]`` to text
    that reads ``a`` twice, ``[b"1", 1]`` to bytes, and integers above 2**53
    beside floats to floats that merge them.
    """
    import pandas as pd

    if isinstance(
        labels, np.ndarray | pd.Series | pd.Index | pd.api.extensions.ExtensionArray
    ):
        return np.asarray(labels)
    return np.asarray(labels, dtype=object)


def _listed(values: Sequence[object]) -> str:
    """Quote two values or more as a sentence lists them: ``a and b``, ``a, b and c``."""
    quoted = [quoted_value(value) for value in values]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _first_shared_group(
    values: np.ndarray, first_rows: np.ndarray, groups: np.ndarray
) -> list[object]:
    """Return the first value of each pair in the first group that several pairs share."""
    shared = np.bincount(groups)[groups] > 1
    if not shared.any():
        return []
    group = groups[int(np.argmax(shared))]
    return [values[row] for row in first_rows[groups == group]]


def _label_conflicts(values: np.ndarray) -> tuple[list[object], list[object]]:
    """Return one group of values that would form one label but print differently, and one of values that print alike but would form separate labels.

    pandas groups the values by Python's equality and hashing, and their texts
    by ``str``. Each distinct pair of a value's group and its text joins one
    group of each, so the groupings agree when the pairs number as many as
    the groups of either.
    """
    import pandas as pd

    value_codes, distinct_values = pd.factorize(values)
    # pandas codes missing values -1; they are no labels.
    present = value_codes >= 0
    if not present.all():
        values = values[present]
        value_codes = value_codes[present]
    texts = np.fromiter(map(str, values.tolist()), dtype=object, count=len(values))
    # 0.0 and -0.0 compare equal and form one label, which reads 0.0.
    zeros = [
        code
        for code, value in enumerate(distinct_values.tolist())
        if isinstance(value, float | np.floating) and value == 0
    ]
    if zeros:
        rows = np.flatnonzero(np.isin(value_codes, zeros))
        texts[rows] = [
            str(abs(value)) if isinstance(value, float | np.floating) else str(value)
            for value in values[rows].tolist()
        ]
    text_codes, distinct_texts = pd.factorize(texts)
    n_texts = len(distinct_texts)
    pairs = value_codes.astype(np.int64) * n_texts + text_codes
    if len(pd.unique(pairs)) == len(distinct_values) == n_texts:
        return [], []
    # Each distinct pair, in the order in which the values first show it.
    keys, first_rows = np.unique(pairs, return_index=True)
    order = np.argsort(first_rows, kind="stable")
    keys = keys[order]
    first_rows = first_rows[order]
    return (
        _first_shared_group(values, first_rows, keys // n_texts),
        _first_shared_group(values, first_rows, keys % n_texts),
    )


def require_labels_match_their_text(labels: object, *, name: str) -> None:
    """Refuse labels that would group otherwise than their text reads.

    Python and pandas group labels by value, and numbers that compare equal
    are one value whatever their type, so ``1``, ``1.0`` and ``True`` would
    form one label, while ``1`` and ``"1"``, or ``np.float32(0.1)`` and
    ``0.1``, would form separate labels that print alike. Labels pass whose
    grouping by value is their grouping by text, ``str(label)``, whatever
    their types, such as ``"a"`` beside ``1``; ``0.0`` and ``-0.0`` form one
    label, which reads ``0.0``, and missing labels are left out.

    Args:
        labels: One-dimensional labels, such as a list, an array, or a pandas
            Series, Index or Categorical.
        name: What holds them, as the error names it.

    Raises:
        ValueError: If the labels are not one-dimensional, or hold values that
            would form one label but print differently, or values that print
            alike but would form separate labels; the message quotes one
            group of each.
    """
    import pandas as pd

    values = label_array(labels)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # Typed arrays of booleans, numbers, text, bytes or times print as they
    # compare, but for 0.0 beside -0.0, and are not read.
    if values.dtype.kind in "biufSUTMm":
        return
    if values.dtype.kind == "O":
        inferred = pd.api.types.infer_dtype(values, skipna=True)
        if inferred in _ONE_TEXT_PER_VALUE or (
            inferred == "floating"
            and set(map(type, values.tolist())) <= {float, np.float64}
        ):
            return
    equal, alike = _label_conflicts(values.astype(object))
    found = [
        f"{what}, such as {_listed(group)}"
        for what, group in (
            ("values that would form one label but print differently", equal),
            ("values that print alike but would form separate labels", alike),
        )
        if group
    ]
    if found:
        raise ValueError(
            f"{name} holds {', and '.join(found)}; convert them to text, or to one "
            "type, first"
        )


def optional_name_argument(
    value: object, name: str, *, noun: str = "a non-empty string"
) -> str | None:
    """Return ``value``, a name such as a column's, or None.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.
        noun: What the name must be, for the error message, such as
            ``"a feature column name"``.

    Raises:
        TypeError: If ``value`` is neither None nor a non-empty string.
    """
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{name} must be {noun} or None")
    return value


def clip_fraction_argument(value: object, name: str = "clip_fraction") -> float:
    """Return a two-sided clipping fraction as a Python float."""
    resolved = float_argument(value, name)
    if not 0.0 <= resolved < 0.5:
        raise ValueError(f"{name} must be at least 0 and less than 0.5")
    return resolved


def integer_argument(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return ``value`` as a Python integer within inclusive bounds.

    Python and NumPy integers are accepted and booleans are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.
        minimum: Smallest accepted value, if any.
        maximum: Largest accepted value, if any.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` lies outside the bounds.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return resolved
