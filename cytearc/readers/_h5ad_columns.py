from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from ..utils.arguments import quoted_value
from ._text import as_text, utf8_refusal

# AnnData writes a dataframe column as a group when it needs more than one
# array: a categorical needs codes with categories, and a pandas nullable
# column (integer, boolean, or string) needs values with a missingness mask.
CATEGORICAL_KEYS = frozenset({"codes", "categories"})
NULLABLE_KEYS = frozenset({"values", "mask"})
SPARSE_KEYS = frozenset({"data", "indices", "indptr"})
# The rows of a column that the check of its text reads at a time.
_TEXT_CHECK_ROWS = 65_536

type H5adNode = h5py.Group | h5py.Dataset


def column_encoding(node: H5adNode) -> str:
    """Return the AnnData ``encoding-type`` of a node, or ``unknown``."""
    encoding = node.attrs.get("encoding-type")
    return "unknown" if encoding is None else as_text(encoding)


def is_categorical(node: Any) -> bool:
    return isinstance(node, h5py.Group) and CATEGORICAL_KEYS.issubset(node.keys())


def is_nullable(node: Any) -> bool:
    return isinstance(node, h5py.Group) and NULLABLE_KEYS.issubset(node.keys())


def is_column(node: Any) -> bool:
    """Return whether a dataframe child holds values that can be decoded."""
    return isinstance(node, h5py.Dataset) or is_categorical(node) or is_nullable(node)


def column_place(path: str, table: str, key: str, *, index: bool) -> str:
    """Name a column of an H5AD table as refusals name it.

    For example ``the obs index '_index' of 'pbmc.h5ad'``.
    """
    kind = "index" if index else "column"
    return f"the {table} {kind} {key!r} of {quoted_value(path)}"


# How the refusal of a name that is not UTF-8 text starts.
_NAME_TEXT = "H5AD column names must be UTF-8 text"


def _table_place(table: h5py.Group) -> str:
    """Name a table of an H5AD file as refusals name it, such as ``the obs table of 'pbmc.h5ad'``."""
    name = str(table.name).strip("/")
    return f"the {name} table of {quoted_value(table.file.filename)}"


def _name_text(value: Any, refusal: Callable[[bytes], ValueError]) -> str:
    """Return a name that h5py read from an attribute or a link, as text.

    h5py gives a name whose bytes are not UTF-8 text as bytes, or, from a
    text attribute, as text that holds each such byte as a lone surrogate.

    Raises:
        ValueError: As ``refusal`` refuses the bytes of such a name.
    """
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            raise refusal(bytes(value)) from None
    text = str(value)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # The escape gives back the bytes that the surrogates stand for.
        raise refusal(text.encode("utf-8", "surrogateescape")) from None
    return text


def _member_name(table: h5py.Group, name: Any, group: str = "") -> str:
    """Return the name of a member of a table, refusing one that is not UTF-8 text.

    Args:
        table: The table.
        name: The member's name as h5py gives it.
        group: The path, within the table, of the group that holds the
            member, or an empty string for the table itself.

    Raises:
        ValueError: If the name is not UTF-8 text, naming the table and the
            member's path within it.
    """

    def refusal(raw: bytes) -> ValueError:
        member = b"/".join([group.encode(), raw]) if group else raw
        return ValueError(
            f"{_NAME_TEXT}; {_table_place(table)} holds a member named "
            f"{quoted_value(member)}"
        )

    return _name_text(name, refusal)


@dataclass(frozen=True, slots=True)
class H5adTableMembers:
    """Children of an H5AD dataframe group, found without reading values.

    Attributes:
        members: Source name and node of each child. Names listed in
            ``column-order`` come first, in that order, followed by direct
            children the attribute does not list, such as the index.
        unresolved: Names listed in ``column-order`` that name no child.
    """

    members: tuple[tuple[str, H5adNode], ...]
    unresolved: tuple[str, ...]


def column_order(table: h5py.Group) -> tuple[str, ...] | None:
    """Return the column names an AnnData dataframe lists in ``column-order``.

    Reading the attribute touches no dataset values. AnnData stores an empty
    table's attribute as an empty array without a string dtype.

    Returns:
        The listed names without repeats, or None when the attribute is absent.

    Raises:
        ValueError: If the attribute holds values that are not text, or a
            name that is not UTF-8 text, naming the table and the name's
            bytes.
    """

    def refusal(name: bytes) -> ValueError:
        return ValueError(
            f"{_NAME_TEXT}; the column-order attribute of {_table_place(table)} "
            f"holds {quoted_value(name)}"
        )

    raw = table.attrs.get("column-order")
    if raw is None:
        return None
    values = np.asarray(raw).reshape(-1)
    names = []
    for value in values.tolist():
        if not isinstance(value, str | bytes):
            raise ValueError(
                f"H5AD table {table.name!r} has a column-order attribute that "
                "does not list column names"
            )
        names.append(_name_text(value, refusal))
    return tuple(dict.fromkeys(names))


def _is_member_path(name: str) -> bool:
    # h5py resolves a leading "/" from the file root and "." as the group itself.
    return not name.startswith("/") and all(
        part not in {"", ".", ".."} for part in name.split("/")
    )


def table_members(table: h5py.Group) -> H5adTableMembers:
    """Resolve the children of an AnnData dataframe group by source name.

    Old AnnData versions wrote a column whose name contains ``/`` as nested
    HDF5 groups, while ``column-order`` kept the full name. Resolving each
    listed name as an HDF5 path finds such a column. The index, which
    ``column-order`` leaves out, is resolved the same way through the
    ``_index`` attribute, with or without ``column-order``. Other direct
    children follow; a group that only holds the nested levels of a resolved
    name is not reported. This reads one attribute and each member's object
    header, never column values.

    Args:
        table: The dataframe group, such as ``obs`` or ``var``.

    Returns:
        The resolved members and the listed names that resolve to nothing.

    Raises:
        ValueError: If ``column-order`` does not list text, or it, the
            ``_index`` attribute or the name of a member is not UTF-8 text,
            naming the table and the attribute or the member.
    """
    order = column_order(table)
    listed: tuple[str, ...] = () if order is None else order
    members: list[tuple[str, H5adNode]] = []
    unresolved: list[str] = []
    for name in listed:
        node = table.get(name) if _is_member_path(name) else None
        if isinstance(node, h5py.Group | h5py.Dataset):
            members.append((name, node))
        else:
            unresolved.append(name)
    index = index_key(table)
    if index is not None and "/" in index and index not in listed:
        node = table.get(index) if _is_member_path(index) else None
        if isinstance(node, h5py.Group | h5py.Dataset):
            members.append((index, node))
            listed = (*listed, index)
    resolved = set(listed)
    levels = {
        name.rsplit("/", depth)[0]
        for name in resolved
        for depth in range(1, name.count("/") + 1)
    }
    for key in table.keys():
        name = _member_name(table, key)
        if name in resolved:
            continue
        node = table[name]
        # A decodable column keeps its place even when a listed path runs
        # through it. A group of nested levels is dropped only when resolved
        # names cover everything in it; otherwise the reader reports it.
        if name in levels and not is_column(node):
            if _holds_only(table, node, name, resolved, levels):
                continue
        members.append((name, node))
    return H5adTableMembers(tuple(members), tuple(unresolved))


def _holds_only(
    table: h5py.Group,
    group: h5py.Group,
    path: str,
    resolved: set[str],
    levels: set[str],
) -> bool:
    """Return whether resolved names cover every member under ``group`` of ``table``."""
    for key in group.keys():
        child = _member_name(table, key, path)
        child_path = f"{path}/{child}"
        if child_path in resolved:
            continue
        node = group[child]
        if child_path in levels and isinstance(node, h5py.Group):
            if not is_column(node) and _holds_only(
                table, node, child_path, resolved, levels
            ):
                continue
        return False
    return True


def is_table_column(node: Any) -> bool:
    """Return whether a dataframe member imports as one metadata column.

    A dataset holding more than one dimension decodes but is not imported.
    """
    if isinstance(node, h5py.Dataset):
        return int(node.ndim) == 1
    return is_column(node)


def table_column_names(table: Any) -> list[str]:
    """Return the source names of a dataframe's decodable columns.

    A compound dataset, as AnnData 0.6 wrote, lists its fields. A group lists
    the members that :func:`is_table_column` accepts, in :func:`table_members`
    order.
    """
    if isinstance(table, h5py.Dataset):
        return list(table.dtype.names or ())
    if not isinstance(table, h5py.Group):
        return []
    return [
        name for name, node in table_members(table).members if is_table_column(node)
    ]


def column_length(node: Any) -> int | None:
    """Return the number of rows of a decodable dataframe column."""
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.shape else None
    if is_categorical(node):
        return int(node["codes"].shape[0])
    if is_nullable(node):
        return int(node["values"].shape[0])
    return None


def index_key(table: Any) -> str | None:
    """Return the dataframe index name that AnnData records in ``_index``.

    Raises:
        ValueError: If the name is not UTF-8 text, naming the table and the
            name's bytes.
    """
    if not isinstance(table, h5py.Group):
        return None

    def refusal(name: bytes) -> ValueError:
        return ValueError(
            f"{_NAME_TEXT}; the _index attribute of {_table_place(table)} holds "
            f"{quoted_value(name)}"
        )

    value = table.attrs.get("_index")
    return None if value is None else _name_text(value, refusal)


def decode_categories(
    codes: np.ndarray,
    categories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map category codes to labels; negative or unknown codes are missing."""
    codes = np.asarray(codes)
    categories = np.asarray(categories)
    valid = (codes >= 0) & (codes < len(categories))
    values = np.empty(codes.shape, dtype=object)
    values[valid] = categories[codes[valid]]
    values[~valid] = None
    return values, ~valid


@dataclass(frozen=True, slots=True)
class CategoryCodes:
    """The codes of a categorical column and the categories that they index.

    A negative code, or one past the last category, marks a missing row.

    Attributes:
        codes: One integer code per row, as stored.
        categories: The categories, as stored.
    """

    codes: np.ndarray
    categories: np.ndarray

    @property
    def missing(self) -> np.ndarray:
        """The rows whose code names no category."""
        return (self.codes < 0) | (self.codes >= len(self.categories))


type TableColumn = tuple[np.ndarray, np.ndarray] | CategoryCodes


def decoded_column(column: TableColumn) -> tuple[np.ndarray, np.ndarray]:
    """Return a column's values and missing rows, decoding a categorical's codes."""
    if isinstance(column, CategoryCodes):
        return decode_categories(column.codes, column.categories)
    return column


def _node_column(node: H5adNode, selection: slice) -> TableColumn:
    """Return one dataframe column slice, a categorical as its codes."""
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[selection])
        return values, np.zeros(values.shape, dtype=bool)
    if is_categorical(node):
        return CategoryCodes(
            np.asarray(node["codes"][selection]), np.asarray(node["categories"][:])
        )
    if is_nullable(node):
        values = np.asarray(node["values"][selection])
        missing = np.asarray(node["mask"][selection], dtype=bool)
        if missing.shape != values.shape:
            raise ValueError(
                f"Column {node.name!r} has a missing-value mask that does not "
                "align with its values"
            )
        return values, missing
    raise TypeError(
        f"Column {node.name!r} uses the unsupported H5AD encoding "
        f"{column_encoding(node)!r}"
    )


def read_column(
    node: H5adNode,
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one dataframe column slice and its missing-value mask.

    Categorical labels come back as an object array holding ``None`` in
    missing rows. Pandas nullable columns keep their stored values and dtype;
    the mask flags the rows that are missing.

    Raises:
        TypeError: If the node uses an encoding that cannot be decoded.
        ValueError: If a nullable column's mask does not align with its values.
    """
    return decoded_column(_node_column(node, slice(start, stop)))


def legacy_categories(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
) -> np.ndarray | None:
    """Return categories that old AnnData files store apart from their codes.

    AnnData 0.7 kept them in a ``__categories`` group beside the columns and
    AnnData 0.6 in ``uns/<key>_categories``.
    """
    if isinstance(table, h5py.Group):
        for name in category_groups:
            group = table.get(name)
            if isinstance(group, h5py.Group) and key in group:
                return np.asarray(group[key][:])
    uns = h5.get("uns")
    if isinstance(uns, h5py.Group) and f"{key}_categories" in uns:
        return np.asarray(uns[f"{key}_categories"][:])
    return None


def table_column(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
    start: int = 0,
    stop: int | None = None,
) -> TableColumn:
    """Read one column of a dataframe group or of an old compound dataset.

    A categorical column comes back as its codes and categories, whether it
    keeps its categories beside its codes or, as old AnnData versions wrote
    it, apart from them; any other column as its values and missing rows.
    """
    if isinstance(table, h5py.Dataset):
        fields = table.dtype.fields
        if fields is None or key not in fields:
            raise KeyError(f"Column {key!r} was not found in {table.name!r}")
        values = np.asarray(table.fields(key)[start:stop])
        missing = np.zeros(values.shape, dtype=bool)
    else:
        node = table[key]
        column = _node_column(node, slice(start, stop))
        if isinstance(column, CategoryCodes) or not isinstance(node, h5py.Dataset):
            return column
        values, missing = column
    categories = legacy_categories(h5, table, key, category_groups)
    if categories is not None and values.dtype.kind in "iu":
        return CategoryCodes(values, categories)
    return values, missing


def read_table_column(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one column of a dataframe group or of an old compound dataset.

    Categorical labels are decoded as :func:`read_column` decodes them.
    """
    return decoded_column(table_column(h5, table, key, category_groups, start, stop))


@dataclass(frozen=True, slots=True)
class H5adCategories:
    """The declared categories of an AnnData categorical column.

    Attributes:
        values: Every category in its stored order, unused ones included,
            decoded as the column's values decode: byte strings as text and
            NumPy scalars as Python scalars.
        ordered: Whether the categories are ordered.
    """

    values: tuple[Any, ...]
    ordered: bool


def category_refusal(
    value: bytes, position: int, expectation: str, place: str
) -> ValueError:
    """Return the refusal of a category that is bytes that are not UTF-8 text.

    Args:
        value: The bytes.
        position: The category's position, counting from 0, which the
            refusal counts from 1.
        expectation: What the column must hold, which the refusal starts
            with, such as ``"H5AD cell metadata must be UTF-8 text"``.
        place: Where the column is, as :func:`column_place` names it.
    """
    return ValueError(
        f"{expectation}; category {position + 1} of {place} holds "
        f"{quoted_value(bytes(value))}"
    )


def _category_values(
    categories: np.ndarray, expectation: str, place: str
) -> tuple[Any, ...]:
    """Return categories with byte strings as text and NumPy scalars as Python scalars.

    Raises:
        ValueError: If a category is bytes that are not UTF-8 text, as
            :func:`category_refusal` refuses it.
    """
    values = []
    for position, value in enumerate(categories):
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                raise category_refusal(value, position, expectation, place) from None
        elif isinstance(value, np.generic):
            value = value.item()
        values.append(value)
    return tuple(values)


def column_categories(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
    *,
    expectation: str,
    place: str,
) -> H5adCategories | None:
    """Return the categories of a categorical column, or None for another column.

    A group-encoded categorical keeps its categories beside its codes and
    records ``ordered`` as an attribute. AnnData 0.7 and 0.6 kept the
    categories of integer codes apart from them, as
    :func:`legacy_categories` finds them, and such categories are unordered
    unless their dataset records ``ordered``.

    Args:
        h5: The file.
        table: The dataframe group, or an old compound dataset.
        key: The column.
        category_groups: Groups beside the columns that may hold the
            categories of integer codes, as :func:`legacy_categories` reads.
        expectation: What the column must hold, which a refusal starts
            with, such as ``"H5AD cell metadata must be UTF-8 text"``.
        place: Where the column is, as :func:`column_place` names it.

    Raises:
        ValueError: If a category is bytes that are not UTF-8 text, as
            :func:`category_refusal` refuses it.
    """
    if isinstance(table, h5py.Group):
        node = table[key]
        if is_categorical(node):
            return H5adCategories(
                _category_values(node["categories"][:], expectation, place),
                bool(node.attrs.get("ordered", False)),
            )
        if not isinstance(node, h5py.Dataset):
            return None
        stored = node.dtype
    else:
        stored = np.dtype(table.dtype.fields[key][0])
    if stored.kind not in "iu":
        return None
    categories = legacy_categories(h5, table, key, category_groups)
    if categories is None:
        return None
    ordered = False
    if isinstance(table, h5py.Group):
        for name in category_groups:
            group = table.get(name)
            if isinstance(group, h5py.Group) and key in group:
                ordered = bool(group[key].attrs.get("ordered", False))
                break
    return H5adCategories(_category_values(categories, expectation, place), ordered)


def _holds_text(dtype: np.dtype[Any]) -> bool:
    """Return whether values of an HDF5 dtype read as byte strings."""
    if dtype.kind == "S":
        return True
    return dtype.kind == "O" and h5py.check_string_dtype(dtype) is not None


def _first_undecodable(values: list[bytes]) -> int | None:
    """Return the position of the first value that is not UTF-8 text, or None."""
    # Joined by a byte of one character, which no character of several bytes
    # holds, the values decode together exactly when each decodes alone, and
    # one decode takes a fraction of the time of one decode per value.
    try:
        b"\n".join(values).decode("utf-8")
    except UnicodeDecodeError:
        for position, value in enumerate(values):
            try:
                value.decode("utf-8")
            except UnicodeDecodeError:
                return position
    return None


def _check_text_rows(
    values: Any,
    refusal: Callable[[bytes, int], ValueError],
    *,
    mask: h5py.Dataset | None = None,
) -> None:
    """Refuse the first byte string of ``values`` that is not UTF-8 text.

    ``values`` is one-dimensional, such as a dataset, which is read
    ``_TEXT_CHECK_ROWS`` rows at a time, and ``refusal(value, row)`` returns
    the refusal of a value. The rows that ``mask`` flags are missing and are
    not checked.
    """
    rows = len(values)
    for start in range(0, rows, _TEXT_CHECK_ROWS):
        stop = min(start + _TEXT_CHECK_ROWS, rows)
        block = np.asarray(values[start:stop])
        positions = None
        if mask is not None:
            present = ~np.asarray(mask[start:stop], dtype=bool)
            positions = np.flatnonzero(present)
            block = block[present]
        if block.dtype.kind == "S":
            # Bytes below 0x80 are ASCII, which is UTF-8 text.
            if int(np.ascontiguousarray(block).view(np.uint8).max(initial=0)) < 0x80:
                continue
        texts = block.tolist()
        position = _first_undecodable(texts)
        if position is not None:
            row = position if positions is None else int(positions[position])
            raise refusal(texts[position], start + row)


def check_column_text(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
    *,
    expectation: str,
    place: str,
) -> None:
    """Refuse a column of an H5AD table that holds bytes that are not UTF-8 text.

    The column's text is read in blocks of rows: the values of a column of
    text, except those of its missing rows, or every category of a
    categorical, used or not, which an import stores with the column. A
    column without text passes, and so does one that an import skips, such
    as one of more than one dimension or whose missing-value mask does not
    align with its values.

    Args:
        h5: The file.
        table: The dataframe group, or an old compound dataset.
        key: The column, which ``table`` holds.
        category_groups: Groups beside the columns that may hold the
            categories of integer codes, as :func:`legacy_categories` reads.
        expectation: What the column must hold, which a refusal starts
            with, such as ``"H5AD cell metadata must be UTF-8 text"``.
        place: Where the column is, as :func:`column_place` names it.

    Raises:
        ValueError: If a value is bytes that are not UTF-8 text, naming its
            row as ``utf8_refusal`` does, or if a category is, as
            :func:`category_refusal` refuses it.
    """

    def refuse_row(value: bytes, row: int) -> ValueError:
        return utf8_refusal(value, row, expectation, place)

    def refuse_category(value: bytes, position: int) -> ValueError:
        return category_refusal(value, position, expectation, place)

    values: Any
    mask: h5py.Dataset | None = None
    if isinstance(table, h5py.Dataset):
        # A field of a compound dataset, as older h5py reads it, has no dtype
        # or dimensions of its own, so they come from the dataset.
        values = table.fields(key)
        dtype = table.dtype.fields[key][0]
        ndim = table.ndim
    else:
        node = table[key]
        if is_categorical(node):
            codes, categories = node["codes"], node["categories"]
            if (
                isinstance(codes, h5py.Dataset)
                and isinstance(categories, h5py.Dataset)
                and codes.ndim == 1
                and categories.ndim == 1
                and _holds_text(categories.dtype)
            ):
                _check_text_rows(categories, refuse_category)
            return
        if is_nullable(node):
            values, mask = node["values"], node["mask"]
            if not (
                isinstance(values, h5py.Dataset)
                and isinstance(mask, h5py.Dataset)
                and mask.shape == values.shape
            ):
                return
        elif isinstance(node, h5py.Dataset):
            values = node
        else:
            return
        dtype, ndim = values.dtype, values.ndim
    if ndim != 1 or dtype.subdtype is not None:
        return
    if mask is None and dtype.kind in "iu":
        categories = legacy_categories(h5, table, key, category_groups)
        if categories is not None:
            if categories.ndim == 1 and _holds_text(categories.dtype):
                _check_text_rows(categories, refuse_category)
            return
    if _holds_text(dtype):
        _check_text_rows(values, refuse_row, mask=mask)


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def hex_color(value: str) -> str | None:
    """Return a color as ``#rrggbb``, or None when it is not a color.

    ``#rgb``, ``#rrggbb``, and ``#rrggbbaa`` are read directly, dropping any
    alpha. Other names, such as ``red`` or ``tab:blue``, are resolved with
    Matplotlib when it is installed.
    """
    text = value.strip()
    if text.startswith("#") and set(text[1:]) <= _HEX_DIGITS:
        digits = text[1:]
        if len(digits) in {3, 4}:
            digits = "".join(digit * 2 for digit in digits[:3])
        if len(digits) in {6, 8}:
            return f"#{digits[:6].lower()}"
        return None
    try:
        from matplotlib.colors import to_hex
    except ImportError:
        return None
    try:
        return str(to_hex(text, keep_alpha=False))
    except ValueError:
        return None


def category_colors(h5: h5py.File, key: str, count: int) -> tuple[str, ...] | None:
    """Return the ``uns["<key>_colors"]`` of a categorical with ``count`` categories.

    AnnData and Scanpy keep one color per category, in category order. Like
    Scanpy, a list with more colors than categories gives each category the
    color at its position. A list with fewer colors, or with a value that is
    not a color, such as bytes that are not UTF-8 text, is not used, and None
    is returned.
    """
    node = h5.get(f"uns/{key}_colors")
    if not isinstance(node, h5py.Dataset) or node.ndim != 1 or count < 1:
        return None
    if int(node.shape[0]) < count:
        return None
    colors = []
    for value in node[:count]:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(value, str):
            return None
        color = hex_color(value)
        if color is None:
            return None
        colors.append(color)
    return tuple(colors)


def table_column_dtype(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
) -> np.dtype[Any]:
    """Return the dtype of the labels or values a table column decodes to."""
    source: np.dtype[Any]
    if isinstance(table, h5py.Dataset):
        fields = table.dtype.fields
        if fields is None or key not in fields:
            raise KeyError(f"Column {key!r} was not found in {table.name!r}")
        source = np.dtype(fields[key][0])
    else:
        node = table[key]
        if is_categorical(node):
            source = np.dtype(node["categories"].dtype)
            return source
        if is_nullable(node):
            source = np.dtype(node["values"].dtype)
            return source
        source = np.dtype(node.dtype)
    categories = legacy_categories(h5, table, key, category_groups)
    if categories is not None and source.kind in "iu":
        source = np.dtype(categories.dtype)
    return source


def present_column(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Show missing rows as NaN in numeric columns and as None otherwise."""
    if not bool(missing.any()):
        return values
    if values.dtype.kind in "iuf":
        decoded = values.astype(np.float64)
        decoded[missing] = np.nan
        return decoded
    decoded = values.astype(object)
    decoded[missing] = None
    return decoded


def sparse_encoding(group: h5py.Group) -> str | None:
    """Return ``csr`` or ``csc`` for a sparse matrix group, or None.

    AnnData records the layout in ``encoding-type``; files written by the
    older h5sparse layout record it in ``h5sparse_format``.
    """
    encoding = group.attrs.get("encoding-type")
    if encoding is None:
        encoding = group.attrs.get("h5sparse_format")
    if encoding is None:
        return None
    normalized = as_text(encoding).lower()
    if normalized in {"csr", "csr_matrix"}:
        return "csr"
    if normalized in {"csc", "csc_matrix"}:
        return "csc"
    return None


def sparse_shape(group: h5py.Group) -> tuple[int, int] | None:
    """Return the stored ``shape`` (or h5sparse ``h5sparse_shape``) attribute."""
    shape: Any = group.attrs.get("shape")
    if shape is None:
        shape = group.attrs.get("h5sparse_shape")
    if shape is None:
        return None
    values = np.asarray(shape).reshape(-1)
    if values.size != 2:
        return None
    return int(values[0]), int(values[1])


def _is_array(node: Any) -> bool:
    """Whether a node of an HDF5 file or a Zarr store holds an array."""
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _index_node(table: Any, path: str) -> tuple[Any, str]:
    """Return the node of a dataframe's index and how it is stored.

    AnnData writes a text index as an array, or, in current releases, as a
    nullable string array: a group of values and a mask; a categorical index
    reads too. Nodes of HDF5 files and of Zarr stores read alike.

    Raises:
        ValueError: If the dataframe names no index, or stores it in a form
            CyteArc cannot read.
    """
    where = str(getattr(table, "name", "")).strip("/") or "dataframe"
    raw = None if _is_array(table) else table.attrs.get("_index")
    key = None if raw is None else as_text(raw)
    node = None if key is None or key not in table else table[key]
    if node is None:
        raise ValueError(f"The {where} of {path} has no index of names")
    unreadable = ValueError(
        f"The {where} of {path} stores its index of names in a form CyteArc cannot read"
    )
    if _is_array(node):
        if len(node.shape) != 1:
            raise unreadable
        return node, "array"
    members = set(node.keys())
    if CATEGORICAL_KEYS <= members:
        return node, "categorical"
    if NULLABLE_KEYS <= members:
        return node, "nullable"
    raise unreadable


def index_length(table: Any, path: str) -> int:
    """Return the rows of a dataframe of an AnnData file or a TreeData store.

    Raises:
        ValueError: As :func:`index_names` raises for its index.
    """
    node, form = _index_node(table, path)
    if form == "array":
        return int(node.shape[0])
    return int(node["codes" if form == "categorical" else "values"].shape[0])


def index_names(
    table: Any,
    start: int = 0,
    stop: int | None = None,
    *,
    path: str,
    what: str,
) -> np.ndarray:
    """Return the names of rows ``[start, stop)`` of a dataframe's index, as text.

    Args:
        table: The dataframe's group, such as ``obs``, ``var`` or a TreeData
            store's ``obsm['characters']``, of an HDF5 file or a Zarr store.
        start: The first row.
        stop: The row after the last, or None for the last row.
        path: The file, as errors name it.
        what: What a row is, such as ``"cell"``, for errors.

    Returns:
        An object array of each row's name.

    Raises:
        ValueError: If the index is stored in a form CyteArc cannot read, or
            a name is missing.
    """
    node, form = _index_node(table, path)
    rows = slice(start, stop)
    if form == "array":
        values = np.asarray(node[rows]).reshape(-1)
        missing = np.zeros(len(values), dtype=bool)
    elif form == "nullable":
        values = np.asarray(node["values"][rows]).reshape(-1)
        missing = np.asarray(node["mask"][rows], dtype=bool).reshape(-1)
        if missing.shape != values.shape:
            where = str(getattr(table, "name", "")).strip("/") or "dataframe"
            raise ValueError(
                f"The index of the {where} of {path} has a missing-value mask that "
                "does not align with its names"
            )
    else:
        codes = np.asarray(node["codes"][rows], dtype=np.int64).reshape(-1)
        categories = np.asarray(node["categories"][:]).reshape(-1)
        missing = (codes < 0) | (codes >= len(categories))
        values = np.empty(len(codes), dtype=object)
        values[~missing] = categories[codes[~missing]]
    if missing.any():
        where = str(getattr(table, "name", "")).strip("/") or "dataframe"
        raise ValueError(
            f"{path} has {what}s without a name in the index of its {where}, the "
            f"first in row {start + int(np.flatnonzero(missing)[0]) + 1}"
        )
    return np.asarray([as_text(value) for value in values.tolist()], dtype=object)


def obs_names(
    handle: Any, start: int = 0, stop: int | None = None, *, path: str
) -> np.ndarray:
    """Return the names of cells ``[start, stop)`` of an AnnData file, as text.

    Raises:
        ValueError: If the file has no ``obs`` table, stores its index in a
            form CyteArc cannot read, or a cell has no name.
    """
    obs = handle.get("obs")
    if obs is None or _is_array(obs):
        raise ValueError(f"{path} has no AnnData obs table with cell names")
    return index_names(obs, start, stop, path=path, what="cell")
