"""The categorical layout of a metadata column: integer codes beside their categories.

A categorical column ``<name>`` stores one code per row, the position of the
row's category or -1 for a missing row, in the smallest signed integer type
that holds the position of every category. Its categories are the
one-dimensional array ``__cytearc_categories__<name>`` beside it, which the
codes name in their ``categories`` attribute, as a nullable column names its
missing mask, and whose ``ordered`` attribute records whether they are
ordered. Codes that hold -1 also link the missing mask that flags exactly
those rows, as every nullable column does, so readers of missing values need
not know the layout.

A release before categorical columns opens a table only when each of its
arrays holds one value per row, so the categories array never holds as many
entries as its table has rows: such a release then refuses the table rather
than read the codes as values. When the categories number exactly the rows,
the array holds one reserved entry after the last category, the placeholder
of their dtype, which no code names, and records it with its ``padded``
attribute. :func:`stored_categories` builds that array, and readers leave the
entry out.

:class:`CategoricalColumn` reads such a column as its labels. A read selects
codes and returns the categories at them, with the placeholder of the
categories' dtype in missing rows, so it returns what a column that stores
the labels themselves returns.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr

from .arrays import missing_placeholder
from .geometry import array_geometry
from .partition import row_band
from .types import as_zarr_array

CATEGORIES_PREFIX = "__cytearc_categories__"
"""Name prefix of the array that holds the categories of a categorical column."""

CATEGORIES_LINK = "categories"
"""Attribute of a categorical column's codes that names its categories array."""

ORDERED_ATTRIBUTE = "ordered"
"""Attribute of a categories array that records whether its categories are ordered."""

PADDED_ATTRIBUTE = "padded"
"""Attribute of a categories array that ends in a reserved entry, which no code names."""

_CODE_TYPES = (np.int8, np.int16, np.int32, np.int64)
# Text, booleans and numbers, as metadata columns type labels.
_CATEGORY_KINDS = frozenset("biufcU")


def categories_name(name: str) -> str:
    """Return the name of the array that holds the categories of column ``name``."""
    return f"{CATEGORIES_PREFIX}{name}"


def category_codes_dtype(n_categories: int) -> np.dtype[Any]:
    """Return the smallest signed integer dtype that holds a code for each of ``n_categories`` categories.

    Codes run from 0 to ``n_categories - 1``, and -1 marks a missing row, so
    int8 holds the codes of up to 128 categories.

    Raises:
        ValueError: If ``n_categories`` is negative or beyond the int64 range.
    """
    count = int(n_categories)
    if count < 0:
        raise ValueError(f"A categorical column cannot have {count} categories")
    largest = max(count - 1, 0)
    for code_type in _CODE_TYPES:
        if largest <= np.iinfo(code_type).max:
            return np.dtype(code_type)
    raise ValueError(f"No signed integer type holds codes for {count} categories")


def stored_categories(
    categories: np.ndarray, n_rows: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the array that stores ``categories`` beside ``n_rows`` codes, and the attributes that record its reserved entry.

    When the categories number exactly ``n_rows``, the array holds one more
    entry after the last category, the placeholder of their dtype, and the
    attributes are ``{"padded": True}``, so the array's length differs from
    the table's row count. Otherwise the array holds the categories alone and
    the attributes are empty.
    """
    labels = np.asarray(categories)
    if len(labels) != int(n_rows):
        return labels, {}
    padded = np.empty(len(labels) + 1, dtype=labels.dtype)
    padded[:-1] = labels
    padded[-1] = missing_placeholder(labels.dtype)
    return padded, {PADDED_ATTRIBUTE: True}


def category_count(categories: zarr.Array) -> int:
    """Return how many categories a stored categories array holds, its reserved entry left out."""
    padded = categories.attrs.get(PADDED_ATTRIBUTE) is True
    return int(categories.shape[0]) - int(padded)


def linked_categories(
    group: zarr.Group,
    name: str,
    *,
    label: str | None = None,
    values: zarr.Array | None = None,
    opened: Mapping[str, zarr.Array | zarr.Group] | None = None,
) -> zarr.Array | None:
    """Return the categories array linked to ``group[name]``, or None for a column of values.

    A categorical column names its categories in ``attrs["categories"]``.
    The link must name the canonical ``__cytearc_categories__<name>``
    sibling, the column must store signed integer codes wide enough for
    every category, and the sibling must be a one-dimensional array of text,
    booleans or numbers with a boolean ``ordered`` attribute and no link of
    its own. A ``padded`` attribute, when present, must be True, and the
    sibling then holds one entry more than the column has rows, the last of
    which is reserved. ``label`` names the column in errors and defaults to
    ``Column '<name>'``. Pass the already opened ``values`` array to skip
    reopening it, and members of ``group`` that are already open, keyed by
    name, as ``opened``.

    Raises:
        ValueError: If the column links categories in any other way.
    """
    if values is None:
        values = as_zarr_array(group[name], name=name)
    if CATEGORIES_LINK not in values.attrs:
        return None
    subject = f"Column {name!r}" if label is None else label
    linked = categories_name(name)
    if values.attrs[CATEGORIES_LINK] != linked:
        raise ValueError(f"{subject} has a malformed categories link")
    codes_dtype = np.dtype(values.dtype)
    if values.ndim != 1 or codes_dtype.kind != "i":
        raise ValueError(
            f"{subject} links categories but does not store one signed integer "
            f"code per row; it stores {codes_dtype} values of shape {values.shape}"
        )
    try:
        node = (
            opened[linked] if opened is not None and linked in opened else group[linked]
        )
    except KeyError:
        raise ValueError(
            f"{subject} links a categories array that is missing"
        ) from None
    if (
        not isinstance(node, zarr.Array)
        or node.ndim != 1
        or np.dtype(node.dtype).kind not in _CATEGORY_KINDS
        or CATEGORIES_LINK in node.attrs
        or "missing_mask" in node.attrs
        or not isinstance(node.attrs.get(ORDERED_ATTRIBUTE), bool)
    ):
        raise ValueError(f"{subject} has a malformed categories array")
    if PADDED_ATTRIBUTE in node.attrs and (
        node.attrs[PADDED_ATTRIBUTE] is not True
        or int(node.shape[0]) != int(values.shape[0]) + 1
    ):
        raise ValueError(
            f"{subject} has a categories array whose reserved entry is malformed: "
            f"its {PADDED_ATTRIBUTE!r} attribute must be True, and it must then hold "
            "one entry more than the column has rows"
        )
    n_categories = category_count(node)
    if n_categories > int(np.iinfo(codes_dtype.type).max) + 1:
        raise ValueError(
            f"{subject} has {n_categories} categories, more than its {codes_dtype} "
            "codes can name"
        )
    return node


class CategoricalColumn:
    """A categorical metadata column, read as its labels.

    A read selects codes as Zarr selects them from the stored codes and
    returns the categories at them: fixed-width text at the width of the
    widest category, or the categories' booleans or numbers, with the
    placeholder of that dtype in each missing row, as a column that stores
    the labels holds there. The view has the shape, chunks and attributes of
    its codes and the dtype of its categories, so code that plans or reads a
    column without knowing its layout sees its labels.

    Attributes:
        codes: The stored codes, one per row, -1 for a missing row.
        ordered: Whether the categories are ordered.
    """

    __slots__ = (
        "codes",
        "ordered",
        "_categories_array",
        "_n_categories",
        "_categories",
        "_lookup",
    )

    def __init__(self, codes: zarr.Array, categories: zarr.Array, *, ordered: bool):
        self.codes = codes
        self.ordered = bool(ordered)
        self._categories_array = categories
        # A reserved entry after the last category is no category.
        self._n_categories = category_count(categories)
        self._categories: np.ndarray | None = None
        self._lookup: np.ndarray | None = None

    @classmethod
    def open(
        cls,
        group: zarr.Group,
        name: str,
        *,
        values: zarr.Array | None = None,
        opened: Mapping[str, zarr.Array | zarr.Group] | None = None,
    ) -> "CategoricalColumn | None":
        """Return column ``name`` of ``group`` as its labels, or None for a column of values.

        Raises:
            ValueError: As :func:`linked_categories` raises for a malformed link.
        """
        if values is None:
            values = as_zarr_array(group[name], name=name)
        categories = linked_categories(group, name, values=values, opened=opened)
        if categories is None:
            return None
        return cls(
            values, categories, ordered=categories.attrs[ORDERED_ATTRIBUTE] is True
        )

    def with_codes(self, codes: zarr.Array) -> "CategoricalColumn":
        """Return this column reading its codes through ``codes``, such as a cached view of them."""
        if codes.shape != self.codes.shape or codes.dtype != self.codes.dtype:
            raise ValueError(
                "The codes of a categorical column must keep their shape and dtype"
            )
        column = CategoricalColumn(codes, self._categories_array, ordered=self.ordered)
        column._categories = self._categories
        column._lookup = self._lookup
        return column

    @property
    def categories_array(self) -> zarr.Array:
        """The stored categories, with a reserved entry after the last one when it records one."""
        return self._categories_array

    @property
    def categories(self) -> np.ndarray:
        """The categories, read once, without the stored array's reserved entry."""
        if self._categories is None:
            self._categories = np.asarray(self._categories_array[: self._n_categories])
        return self._categories

    @property
    def n_categories(self) -> int:
        """The number of categories, from the stored array's shape and its reserved entry."""
        return self._n_categories

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._categories_array.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(extent) for extent in self.codes.shape)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return int(self.codes.shape[0])

    @property
    def nbytes(self) -> int:
        """The bytes of every label as this view returns them."""
        return self.size * int(self.dtype.itemsize)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(self.codes.chunks)

    @property
    def metadata(self) -> Any:
        """The metadata of the stored codes, whose chunks and shards a read follows."""
        return self.codes.metadata

    @property
    def attrs(self) -> Any:
        """The attributes of the stored codes, which are the column's."""
        return self.codes.attrs

    @property
    def basename(self) -> str:
        return str(self.codes.basename)

    @property
    def name(self) -> str:
        return str(self.codes.name)

    def _decoding(self) -> np.ndarray:
        """The categories followed by the placeholder that code -1 selects."""
        if self._lookup is None:
            categories = self.categories
            lookup = np.empty(len(categories) + 1, dtype=categories.dtype)
            lookup[:-1] = categories
            lookup[-1] = missing_placeholder(categories.dtype)
            self._lookup = lookup
        return self._lookup

    def check_codes(self, codes: np.ndarray) -> np.ndarray:
        """Return ``codes`` after checking that each names a category or is -1.

        Raises:
            ValueError: If a code lies outside -1 to the last category.
        """
        array = np.asarray(codes)
        if array.size:
            low = int(array.min())
            high = int(array.max())
            if low < -1 or high >= self.n_categories:
                bad = low if low < -1 else high
                raise ValueError(
                    f"Categorical column {self.basename!r} holds the code {bad}, but "
                    f"its codes name {self.n_categories} categories and -1 marks a "
                    "missing row; the column is corrupt, so write it again"
                )
        return array

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Return the labels of ``codes``, with the placeholder of their dtype for -1."""
        array = self.check_codes(codes)
        # Code -1 selects the placeholder at the end of the lookup.
        return np.asarray(self._decoding()[array])

    def decode_objects(self, codes: np.ndarray) -> np.ndarray:
        """Return the labels of ``codes`` as an object array that shares one Python value per category.

        A missing row holds the placeholder of the categories' dtype.
        """
        array = self.check_codes(codes)
        lookup = np.empty(self.n_categories + 1, dtype=object)
        lookup[:] = self._decoding().tolist()
        return np.asarray(lookup[array])

    def __getitem__(self, selection: Any) -> np.ndarray:
        return self.decode(np.asarray(self.codes[selection]))

    def get_orthogonal_selection(self, selection: Any) -> np.ndarray:
        return self.decode(np.asarray(self.codes.get_orthogonal_selection(selection)))

    def get_coordinate_selection(self, selection: Any) -> np.ndarray:
        return self.decode(np.asarray(self.codes.get_coordinate_selection(selection)))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        values = self[:]
        return values if dtype is None else values.astype(dtype)

    def used_categories(self) -> np.ndarray:
        """Return which categories some row holds, reading the codes one band of chunks at a time."""
        used = np.zeros(self.n_categories, dtype=bool)
        n_rows = self.size
        band = row_band(
            array_geometry(self.codes), unit="chunk", fallback=max(1, n_rows)
        )
        for start in range(0, n_rows, band):
            codes = self.check_codes(np.asarray(self.codes[start : start + band]))
            used[codes[codes >= 0]] = True
        return used

    def stored_labels_dtype(self) -> np.dtype[Any]:
        """Return the dtype that a column storing these labels as values has.

        Text is stored at the width of the widest label that a row holds,
        one character at least, so text categories that no row holds do not
        widen it; booleans and numbers keep the categories' dtype.
        """
        if self.dtype.kind != "U":
            return self.dtype
        held = self.categories[self.used_categories()]
        width = int(np.strings.str_len(held).max(initial=0)) if held.size else 0
        return np.dtype(f"U{max(1, width)}")

    def __repr__(self) -> str:
        return (
            f"<CategoricalColumn {self.basename!r} of {self.size} rows, "
            f"{self.n_categories} {self.dtype} categories>"
        )
