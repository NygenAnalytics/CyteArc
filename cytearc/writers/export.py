"""Export assays and completed pipeline runs to H5AD and Matrix Market files.

Every H5AD export is an :class:`H5adExportPlan`: the matrices, columns,
embeddings, graphs, and unstructured values that the file holds, each read
only while it is written. :func:`live_h5ad_plan` plans the export of an assay
with its live metadata, :func:`with_artifacts` adds the ``obsm``, ``obs``,
``obsp``, ``varm``, and ``uns`` entries of artifact references, and a
``DataStore`` plans the export of a completed pipeline run.
:func:`write_h5ad_plan` writes any of them with h5py, row block by row block,
so it never holds a complete count matrix and needs no AnnData.
:func:`materialize_h5ad_matrix` and :func:`h5ad_table` build the in-memory
form of the same plan, which ``DataStore.to_anndata`` returns.
"""

import gzip
import os
import re
import secrets
import stat
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack

from ..metadata.rows import apply_missing_mask, metadata_missing_mask
from ..storage.refs import ArtifactRef
from ..utils.compute import compute_with_progress
from ..utils.logging import logger

if TYPE_CHECKING:
    import pandas as pd
    import zarr

type H5adText = Literal["string", "categorical"]
type H5adSource = ArtifactRef | tuple[ArtifactRef, str]
"""An artifact to export, or an artifact with the name of one of its per-cell arrays."""

# Element storage of a streamed CSR matrix.
_CSR_CHUNK = 65_536
# Value kinds that hold text: strings, bytes, objects, and NumPy strings.
_TEXT_KINDS = frozenset("USOT")
# AnnData stores an unnamed dataframe index under this name, which no
# dataframe column may take; such a column exports under the next name.
_RESERVED_COLUMN = "_index"
_SOURCE_INDEX = "source_index"


@dataclass(frozen=True, slots=True)
class H5adMatrix:
    """A cells-by-features matrix that an H5AD export reads in row blocks.

    ``export_dtype``, when set, is the dtype that the export stores instead
    of ``dtype``: each converted piece is cast to it, and a value that it
    cannot hold raises ``OverflowError``.
    """

    shape: tuple[int, int]
    dtype: np.dtype[Any]
    blocks: Callable[[], Iterator[csr_matrix | np.ndarray]]
    reserve: Callable[[], int] | None = None
    export_dtype: np.dtype[Any] | None = None


@dataclass(frozen=True, slots=True)
class H5adCategorical:
    """How an exported column becomes a pandas categorical.

    Attributes:
        levels: The categories in their order, unused ones included. A value
            that they do not list is appended, in natural order, with a
            warning. None takes the distinct present values, as text, in
            natural order.
        ordered: Whether the categories are ordered.
        colors: The color of each category. When every category has one,
            the export writes them as ``uns["<column>_colors"]`` in category
            order, as Scanpy reads them.
    """

    levels: tuple[Any, ...] | None = None
    ordered: bool = False
    colors: Mapping[Any, str] | None = None


@dataclass(frozen=True, slots=True)
class H5adColumn:
    """One ``obs`` or ``var`` column of an H5AD export, or the index of either."""

    name: str
    read: Callable[[], tuple[np.ndarray, np.ndarray | None]]
    text: H5adText = "string"
    categorical: H5adCategorical | None = None


@dataclass(frozen=True, slots=True)
class H5adExportPlan:
    """Everything that one H5AD file holds, read while it is written.

    ``obsp`` holds cells-by-cells sparse matrices, ``varm`` features-by-k
    arrays, and ``uns`` unstructured values: mappings with text keys, text,
    numbers, booleans, and arrays or lists of them.
    """

    x: H5adMatrix
    obs_index: H5adColumn
    obs: tuple[H5adColumn, ...]
    var_index: H5adColumn
    var: tuple[H5adColumn, ...]
    obsm: Mapping[str, Callable[[], np.ndarray]] = field(default_factory=dict)
    layers: Mapping[str, H5adMatrix] = field(default_factory=dict)
    membership: Mapping[str, str] = field(default_factory=dict)
    obsp: Mapping[str, Callable[[], csr_matrix]] = field(default_factory=dict)
    varm: Mapping[str, Callable[[], np.ndarray]] = field(default_factory=dict)
    uns: Mapping[str, Any] = field(default_factory=dict)


# Dense values that one conversion step turns into CSR. A dense row block is
# converted a step of rows at a time, and each step is written before the
# next is converted, so the conversion holds a bounded number of bytes
# whatever the size of the block.
_CSR_STEP_VALUES = 1 << 20
# Bytes of one value of the widest export dtype, which a converted piece may
# be cast to.
_CAST_ITEMSIZE = 8


def _csr_step_rows(n_cols: int) -> int:
    """Return the rows of one conversion step: a step of values, or one row."""
    return max(1, _CSR_STEP_VALUES // max(1, int(n_cols)))


def _csr_index_dtype(n_cols: int, step_values: int) -> np.dtype[Any]:
    """Return the dtype of a piece's column indices and row offsets.

    SciPy keeps int32 indexes that it can hold as they are, and would copy
    int64 indexes that fit into int32.
    """
    limit = int(np.iinfo(np.int32).max)
    return np.dtype(
        np.int32 if max(int(n_cols), int(step_values)) <= limit else np.int64
    )


def h5ad_conversion_bytes(dtype: Any, n_cols: int, block_rows: int) -> int:
    """Return the bytes held while converting dense row blocks to CSR.

    They include a copy of a step's values in the export dtype, at most
    eight bytes each, which a cast to another dtype holds.
    """
    rows = min(_csr_step_rows(n_cols), int(block_rows))
    if rows <= 0:
        return 0
    values = rows * max(1, int(n_cols))
    itemsize = int(np.dtype(dtype).itemsize)
    index_bytes = _csr_index_dtype(n_cols, values).itemsize
    int64 = np.dtype(np.int64).itemsize
    per_value = 2 * itemsize + int64 + index_bytes + _CAST_ITEMSIZE
    per_row = 2 * int64 + index_bytes
    return values * per_value + (rows + 1) * per_row


def largest_block_rows(matrix: Any) -> int:
    """Return the rows of the largest block that ``matrix`` streams."""
    rows = int(matrix.shape[0])
    chunks = getattr(matrix, "chunksize", None)
    return min(rows, int(chunks[0])) if rows and chunks is not None else rows


def _dense_csr_pieces(block: np.ndarray) -> Iterator[csr_matrix]:
    """Yield a dense row block as CSR pieces of one conversion step each.

    Only nonzero values are stored, in row-major order, as
    ``csr_matrix(block)`` stores them, and each piece holds what
    :func:`h5ad_conversion_bytes` bounds. A block without rows yields
    nothing.
    """
    n_rows, n_cols = block.shape
    step = _csr_step_rows(n_cols)
    index_dtype = _csr_index_dtype(n_cols, step * max(1, n_cols))
    for start in range(0, n_rows, step):
        rows = block[start : start + step]
        n_step = int(rows.shape[0])
        # A view of C-contiguous rows; a copy of the step otherwise.
        flat = np.ravel(rows)
        positions = np.flatnonzero(flat)
        data = flat[positions]
        del flat
        row_starts = np.arange(1, n_step + 1, dtype=np.int64)
        row_starts *= n_cols
        indptr = np.empty(n_step + 1, dtype=index_dtype)
        indptr[0] = 0
        indptr[1:] = np.searchsorted(positions, row_starts)
        del row_starts
        np.remainder(positions, max(1, n_cols), out=positions)
        indices = positions.astype(index_dtype, copy=False)
        del positions
        yield csr_matrix(
            (data, indices, indptr), shape=(n_step, n_cols), dtype=block.dtype
        )
        del data, indices, indptr


def export_count_dtype(stored: Any, requested: Any | None = None) -> np.dtype[Any]:
    """Return the dtype in which an export stores raw counts.

    An explicit ``requested`` dtype, which must be boolean, integer, or
    floating-point, is used as given. Otherwise counts keep their stored
    dtype, except that ``uint8`` and ``uint16`` counts become ``float32``:
    arithmetic on them wraps silently or narrows, as ``X * 300`` does in
    ``uint16`` and ``numpy.log1p`` does to ``float16`` on ``uint8``. Every
    value of either type is a whole number that ``float32`` holds exactly,
    and ``float32`` is the dtype in which Scanpy's readers return counts.
    """
    if requested is not None:
        resolved: np.dtype[Any] = np.dtype(requested)
        if resolved.kind not in "biuf":
            raise TypeError(
                f"dtype must be a boolean, integer, or floating-point dtype, "
                f"not {resolved}"
            )
        return resolved
    source: np.dtype[Any] = np.dtype(stored)
    if source in (np.dtype(np.uint8), np.dtype(np.uint16)):
        return np.dtype(np.float32)
    return source


def _cast_values(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """Cast count values, raising ``OverflowError`` for a value ``dtype`` cannot hold."""
    from ..utils.arrays import checked_sparse_cast

    # A count beyond a float dtype becomes infinite, which is refused below.
    with np.errstate(over="ignore"):
        cast_values = checked_sparse_cast(np.asarray(values), dtype)
    if dtype.kind == "f" and not bool(np.isfinite(cast_values).all()):
        raise OverflowError(f"Counts exceed the range of the export dtype {dtype}")
    return cast_values


def cast_counts(matrix: csr_matrix, dtype: np.dtype[Any]) -> csr_matrix:
    """Return a CSR count matrix in ``dtype``, checked like an export's pieces."""
    if matrix.dtype == dtype:
        return matrix
    return csr_matrix(
        (_cast_values(matrix.data, dtype), matrix.indices, matrix.indptr),
        shape=matrix.shape,
    )


def iter_h5ad_blocks(
    matrix: H5adMatrix,
    label: str = "The X matrix",
) -> Iterator[csr_matrix]:
    """Yield the rows of ``matrix`` as CSR pieces, checked against it.

    Pieces are cast to ``matrix.export_dtype`` when it is set.
    """
    n_rows, n_cols = matrix.shape
    rows = 0
    for values in matrix.blocks():
        dimensions = int(np.ndim(values))
        if dimensions != 2:
            plural = "" if dimensions == 1 else "s"
            raise ValueError(
                f"{label} has a row block with {dimensions} dimension{plural}; "
                "blocks are two-dimensional"
            )
        block = values if issparse(values) else np.asarray(values)
        if block.shape[1] != n_cols:
            raise ValueError(
                f"{label} has a row block of {block.shape[1]} columns; "
                f"the export declares {n_cols}"
            )
        if block.dtype != matrix.dtype:
            raise ValueError(
                f"{label} has a row block of dtype {block.dtype}; "
                f"the export declares {matrix.dtype}"
            )
        rows += block.shape[0]
        if rows > n_rows:
            raise ValueError(
                f"{label} has more than the {n_rows} rows that the export declares"
            )
        pieces = (
            iter((csr_matrix(block),)) if issparse(block) else _dense_csr_pieces(block)
        )
        for piece in pieces:
            if matrix.export_dtype is not None:
                piece = cast_counts(piece, matrix.export_dtype)
            yield piece
        # The stream reads the next block while this one is held no longer.
        del values, block
    if rows != n_rows:
        raise ValueError(f"{label} has {rows} rows; the export declares {n_rows}")


def _stored_dtype(matrix: H5adMatrix) -> np.dtype[Any]:
    """Return the dtype in which an export stores ``matrix``."""
    return matrix.dtype if matrix.export_dtype is None else matrix.export_dtype


def materialize_h5ad_matrix(
    matrix: H5adMatrix,
    label: str = "The X matrix",
) -> csr_matrix:
    """Return ``matrix`` as one CSR matrix, built from its checked row blocks."""
    blocks = list(iter_h5ad_blocks(matrix, label))
    if not blocks:
        return csr_matrix(matrix.shape, dtype=_stored_dtype(matrix))
    return cast(csr_matrix, vstack(blocks, format="csr"))


_DIGIT_RUNS = re.compile(r"(\d+)")


def _natural_key(text: str) -> tuple[str | int, ...]:
    """Return the key of the natural order of a run export's categories.

    The canonical decomposition (NFD) of ``text`` splits into runs of decimal
    digits, which compare as integers, and the text between them, which
    compares as strings, so ``d2`` sorts before ``d10``. A key that would
    start with a number starts with an empty string, so a number never meets
    a string at one position. A digit that is not a decimal digit, such as a
    superscript or a circled digit, is text. This order is CyteArc's own: the
    run plan applies it, and AnnData keeps it when it writes the categorical
    that ``to_anndata`` returns, although natsort, which AnnData sorts the
    categories of text columns with, reads such digits as numbers.
    """
    parts: list[str | int] = [
        int(part) if index % 2 else part
        for index, part in enumerate(
            _DIGIT_RUNS.split(unicodedata.normalize("NFD", text))
        )
        if part
    ]
    if parts and isinstance(parts[0], int):
        parts.insert(0, "")
    return tuple(parts)


def _categorical_column(
    values: np.ndarray,
    missing: np.ndarray | None,
) -> "pd.Categorical | None":
    """Return a ``"categorical"`` column as the categorical it is, or None.

    This is the one rule of run exports, which both the file and the object
    that ``DataStore.to_anndata(run=...)`` returns follow: text whose values
    repeat or are missing, so that it holds fewer distinct values than rows,
    is a categorical whose categories are its distinct values as text in
    natural order (:func:`_natural_key`), with no category for a missing row
    and none at all for text that is missing in every row. A column that is
    neither numeric nor boolean and has missing rows is such text too.
    Numbers, booleans, and other text are not categoricals, and return None.
    Columns with an :class:`H5adCategorical` follow it instead.
    """
    import pandas as pd

    if values.dtype.kind in "biuf":
        return None
    masked = missing if missing is not None and bool(missing.any()) else None
    if masked is None and values.dtype.kind not in _TEXT_KINDS:
        return None
    text = values.astype(str).astype(object)
    if masked is not None:
        text[masked] = None
    categorical = pd.Categorical(text)
    if len(categorical.categories) >= len(categorical):
        return None
    natural = sorted(categorical.categories, key=_natural_key)
    return categorical.reorder_categories(natural)


def _explicit_categorical(
    name: str,
    values: np.ndarray,
    missing: np.ndarray | None,
    spec: H5adCategorical,
) -> tuple["pd.Categorical", tuple[str, ...] | None]:
    """Return a column as the categorical that ``spec`` describes, with its colors.

    Missing rows have no category. The colors are returned in category
    order when ``spec`` gives every category one.
    """
    import pandas as pd

    present = (
        np.ones(len(values), dtype=bool) if missing is None else ~np.asarray(missing)
    )
    if spec.levels is None:
        objects = values.astype(str).astype(object)
        objects[~present] = None
        levels = sorted({str(value) for value in objects[present]}, key=_natural_key)
    else:
        objects = values.astype(object)
        objects[~present] = None
        levels = list(spec.levels)
        positions = pd.Index(levels, dtype=object).get_indexer(objects)
        unlisted = (positions < 0) & present
        if unlisted.any():
            extra = sorted(
                set(objects[unlisted].tolist()),
                key=lambda value: _natural_key(str(value)),
            )
            logger.warning(
                f"Column {name!r} holds {len(extra)} values that are not among its "
                f"stored categories, such as {extra[0]!r}; the export appends them "
                "to its categories"
            )
            levels = [*levels, *extra]
    categorical = pd.Categorical(objects, categories=levels, ordered=spec.ordered)
    colors: tuple[str, ...] | None = None
    if spec.colors is not None and levels:
        palette = [spec.colors.get(level) for level in categorical.categories.tolist()]
        if all(isinstance(color, str) for color in palette):
            colors = tuple(cast(list[str], palette))
    return categorical, colors


def unique_column_names(
    index: str,
    names: Sequence[str],
    *,
    table: str,
) -> list[str]:
    """Return the export name of each column, so that no two names clash.

    AnnData reserves ``_index``, so a column of that name exports as
    ``source_index``. A column whose name an earlier column or the index
    already takes exports under the first free name among ``<name>_2``,
    ``<name>_3``, and so on. Each rename is logged as a warning.
    """
    taken = {index}
    resolved: list[str] = []
    for name in names:
        base = _SOURCE_INDEX if name == _RESERVED_COLUMN else name
        candidate, suffix = base, 2
        while candidate in taken:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        if candidate != name:
            logger.warning(
                f"Exporting {table} column {name!r} as {candidate!r} because "
                + (
                    "AnnData reserves its name"
                    if name == _RESERVED_COLUMN
                    else "the export already uses its name"
                )
            )
        taken.add(candidate)
        resolved.append(candidate)
    return resolved


def _encoded(node: Any, encoding: str, version: str) -> Any:
    """Record the AnnData element encoding on an HDF5 dataset or group."""
    node.attrs["encoding-type"] = encoding
    node.attrs["encoding-version"] = version
    return node


def _write_array(group: Any, name: str, values: np.ndarray) -> None:
    """Write a numeric, boolean, or text array as an AnnData element."""
    import h5py

    if values.dtype.kind in "biufc":
        _encoded(group.create_dataset(name, data=values), "array", "0.2.0")
        return
    text = values.astype(str).astype(object)
    _encoded(
        group.create_dataset(name, data=text, dtype=h5py.string_dtype()),
        "string-array",
        "0.2.0",
    )


def _write_categorical(
    group: Any,
    name: str,
    categories: np.ndarray,
    codes: np.ndarray,
    ordered: bool = False,
) -> None:
    """Write one categorical whose code -1 marks a missing row."""
    column = group.create_group(name)
    _encoded(column, "categorical", "0.2.0")
    column.attrs["ordered"] = bool(ordered)
    _write_array(column, "categories", categories)
    _write_array(column, "codes", codes)


def _write_masked_column(
    group: Any,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
    text: H5adText,
) -> None:
    """Write one H5AD column whose masked rows are AnnData missing values.

    Numeric columns become float64 with NaN, boolean columns use the nullable
    boolean encoding, and other columns become categoricals whose masked rows
    have code -1. With ``text`` ``"categorical"``, a masked boolean row stores
    False, as AnnData stores it, and other columns are the categoricals of
    :func:`_categorical_column`, which :func:`_write_column` writes. With
    ``"string"``, categories are sorted, codes are int32, and a masked
    boolean row keeps its stored placeholder.
    """
    if values.dtype.kind in {"f", "i", "u"}:
        _write_array(group, name, apply_missing_mask(values, missing))
        return
    if values.dtype.kind == "b":
        column = group.create_group(name)
        _encoded(column, "nullable-boolean", "0.1.0")
        if text == "categorical":
            values = values & ~missing
        _write_array(column, "values", values)
        _write_array(column, "mask", missing)
        return
    categories, observed_codes = np.unique(
        values[~missing].astype(str),
        return_inverse=True,
    )
    codes = np.full(len(values), -1, dtype=np.int32)
    codes[~missing] = observed_codes
    _write_categorical(group, name, categories, codes)


def _read_column(
    column: H5adColumn,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one column and check that it and its missing mask have ``n_rows``."""
    raw_values, raw_missing = column.read()
    values = np.asarray(raw_values)
    if values.ndim != 1:
        raise ValueError(
            f"Column {column.name!r} has {values.ndim} dimensions; "
            "columns are one-dimensional"
        )
    if values.shape[0] != n_rows:
        raise ValueError(
            f"Column {column.name!r} has {values.shape[0]} rows; "
            f"the export declares {n_rows}"
        )
    if raw_missing is None:
        return values, None
    missing = np.asarray(raw_missing, dtype=bool)
    if missing.shape != values.shape:
        raise ValueError(
            f"Column {column.name!r} has a missing mask of shape {missing.shape}; "
            f"its values have {n_rows} rows"
        )
    return values, missing


def _write_column(
    group: Any,
    column: H5adColumn,
    n_rows: int,
) -> tuple[bool, tuple[str, ...] | None]:
    """Write one column; skip one of unsupported dtype with a warning.

    Returns whether the column was written and the colors of its categories,
    if it has them.
    """
    values, missing = _read_column(column, n_rows)
    if column.categorical is not None:
        explicit, colors = _explicit_categorical(
            column.name, values, missing, column.categorical
        )
        _write_categorical(
            group,
            column.name,
            np.asarray(explicit.categories),
            np.asarray(explicit.codes),
            explicit.ordered,
        )
        return True, colors
    if column.text == "categorical":
        categorical = _categorical_column(values, missing)
        if categorical is not None:
            _write_categorical(
                group,
                column.name,
                np.asarray(categorical.categories, dtype=object),
                np.asarray(categorical.codes),
            )
            return True, None
    if missing is not None and missing.any():
        _write_masked_column(group, column.name, values, missing, column.text)
        return True, None
    try:
        _write_array(group, column.name, values)
    except TypeError:
        logger.warning(
            f"Skipping metadata column {column.name!r} with unsupported dtype "
            f"{values.dtype}"
        )
        return False, None
    return True, None


def h5ad_table(
    index: H5adColumn,
    columns: tuple[H5adColumn, ...],
    n_rows: int,
) -> tuple["pd.DataFrame", dict[str, tuple[str, ...]]]:
    """Return an ``obs`` or ``var`` table as an in-memory export holds it.

    Also returns the colors of the categories of each column that has them,
    which the export writes as ``uns["<column>_colors"]``.
    """
    import pandas as pd

    from ..metadata.queries import missing_frame_values

    index_values, index_missing = _read_column(index, n_rows)
    if index_missing is not None and index_missing.any():
        raise ValueError(f"Index {index.name!r} holds missing values")
    data: dict[str, Any] = {}
    colors: dict[str, tuple[str, ...]] = {}
    for column in columns:
        values, missing = _read_column(column, n_rows)
        if column.categorical is not None:
            data[column.name], palette = _explicit_categorical(
                column.name, values, missing, column.categorical
            )
            if palette is not None:
                colors[column.name] = palette
            continue
        categorical = (
            _categorical_column(values, missing)
            if column.text == "categorical"
            else None
        )
        if categorical is not None:
            data[column.name] = categorical
        elif column.text == "categorical" and values.dtype.kind in _TEXT_KINDS:
            # Text without missing rows; the file stores it as a string array.
            data[column.name] = values.astype(str)
        else:
            data[column.name] = missing_frame_values(values, missing)
    frame = pd.DataFrame(data, index=pd.Index(index_values, name=index.name))
    return frame, colors


def h5ad_frame(
    index: H5adColumn,
    columns: tuple[H5adColumn, ...],
    n_rows: int,
) -> "pd.DataFrame":
    """Return an ``obs`` or ``var`` table as an in-memory export holds it."""
    return h5ad_table(index, columns, n_rows)[0]


def _write_frame(
    group: Any,
    index: H5adColumn,
    columns: tuple[H5adColumn, ...],
    n_rows: int,
) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    """Write an ``obs`` or ``var`` dataframe.

    Returns the columns that it holds and the category colors of each that
    has them.
    """
    values, missing = _read_column(index, n_rows)
    if missing is not None and missing.any():
        raise ValueError(f"Index {index.name!r} holds missing values")
    _write_array(group, index.name, values)
    # column-order names only written columns: AnnData reads every column it
    # names, so a skipped one would make the file unreadable.
    written: list[str] = []
    colors: dict[str, tuple[str, ...]] = {}
    for column in columns:
        stored, palette = _write_column(group, column, n_rows)
        if stored:
            written.append(column.name)
        if palette is not None:
            colors[column.name] = palette
    group.attrs["_index"] = index.name
    # HDF5 cannot store an empty object array; AnnData writes an empty list.
    group.attrs["column-order"] = np.array(written, dtype=object) if written else []
    _encoded(group, "dataframe", "0.2.0")
    return set(written), colors


def _write_csr(group: Any, matrix: H5adMatrix, label: str) -> None:
    """Append ``matrix`` to a CSR group row block by row block."""
    n_rows, n_cols = matrix.shape
    capacity = 0 if matrix.reserve is None else int(matrix.reserve())
    # The blocks define the row boundaries, even when stored QC is stale.
    indptr = group.create_dataset(
        "indptr",
        (n_rows + 1,),
        chunks=True,
        compression="gzip",
        dtype="int64",
    )
    data = group.create_dataset(
        "data",
        (capacity,),
        maxshape=(None,),
        chunks=(_CSR_CHUNK,),
        compression="gzip",
        dtype=_stored_dtype(matrix),
    )
    indices = group.create_dataset(
        "indices",
        (capacity,),
        maxshape=(None,),
        chunks=(_CSR_CHUNK,),
        compression="gzip",
        dtype="int64",
    )
    row = offset = 0
    indptr[0] = 0
    for block in iter_h5ad_blocks(matrix, label):
        end_row = row + block.shape[0]
        end = offset + block.nnz
        if end > data.shape[0]:
            data.resize((end,))
            indices.resize((end,))
        data[offset:end] = block.data
        indices[offset:end] = block.indices
        indptr[row + 1 : end_row + 1] = block.indptr[1:].astype(np.int64) + offset
        row, offset = end_row, end
        # Each piece is released before the next one is converted.
        del block
    data.resize((offset,))
    indices.resize((offset,))
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.array([n_rows, n_cols])


def _write_sparse(group: Any, name: str, matrix: csr_matrix) -> None:
    """Write one in-memory CSR matrix as an AnnData element."""
    node = group.create_group(name)
    _encoded(node, "csr_matrix", "0.1.0")
    node.attrs["shape"] = np.array(matrix.shape)
    node.create_dataset("data", data=matrix.data)
    node.create_dataset("indices", data=matrix.indices)
    node.create_dataset("indptr", data=matrix.indptr)


def _write_uns_value(group: Any, name: str, value: Any) -> None:
    """Write one ``uns`` value with the encoding AnnData reads it with."""
    import h5py

    if isinstance(value, Mapping):
        child = _encoded(group.create_group(name), "dict", "0.1.0")
        for key, item in value.items():
            _write_uns_value(child, str(key), item)
    elif isinstance(value, str):
        _encoded(
            group.create_dataset(name, data=value, dtype=h5py.string_dtype()),
            "string",
            "0.2.0",
        )
    elif isinstance(value, bool | int | float | np.generic):
        _encoded(
            group.create_dataset(name, data=np.asarray(value)),
            "numeric-scalar",
            "0.2.0",
        )
    else:
        _write_array(group, name, np.asarray(value))


def _merged(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``extra`` merged in, mapping by mapping."""
    merged = dict(base)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merged(current, value)
        else:
            merged[key] = value
    return merged


def plan_uns(
    plan: H5adExportPlan,
    obs_columns: Iterable[str],
    colors: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Return the ``uns`` of an export: the plan's values, colors, and membership.

    Each written ``obs`` column whose categories have colors adds
    ``<column>_colors``, and the membership columns that the export holds are
    declared under ``uns["cytearc"]["assayMembership"]``.
    """
    from ..metadata.membership import membership_declaration

    written = set(obs_columns)
    uns = dict(plan.uns)
    for name, palette in colors.items():
        uns[f"{name}_colors"] = np.asarray(palette, dtype=object)
    declared = {
        assay_name: column
        for assay_name, column in plan.membership.items()
        if column in written
    }
    return _merged(uns, membership_declaration(declared))


def _check_plan(plan: H5adExportPlan) -> None:
    """Reject a plan that no file can hold, before anything is written."""
    for table, index, columns in (
        ("obs", plan.obs_index, plan.obs),
        ("var", plan.var_index, plan.var),
    ):
        names = [index.name, *(column.name for column in columns)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Column {repeated[0]!r} is written twice in {table}")
    for name, layer in plan.layers.items():
        if layer.shape != plan.x.shape:
            raise ValueError(
                f"Layer {name!r} has shape {layer.shape}; X has shape {plan.x.shape}"
            )


def read_obsp(plan: H5adExportPlan, name: str) -> csr_matrix:
    """Read one ``obsp`` matrix and check that it pairs the exported cells."""
    n_obs = plan.x.shape[0]
    matrix = csr_matrix(plan.obsp[name]())
    if matrix.shape != (n_obs, n_obs):
        raise ValueError(
            f"obsp {name!r} has shape {matrix.shape}; the export declares "
            f"{(n_obs, n_obs)}"
        )
    return matrix


def read_varm(plan: H5adExportPlan, name: str) -> np.ndarray:
    """Read one ``varm`` array and check that it has a row per exported feature."""
    n_vars = plan.x.shape[1]
    values = np.asarray(plan.varm[name]())
    if values.ndim != 2 or values.shape[0] != n_vars:
        raise ValueError(
            f"varm {name!r} has shape {values.shape}; the export declares "
            f"{n_vars} features"
        )
    return values


def read_obsm(plan: H5adExportPlan, name: str) -> np.ndarray:
    """Read one ``obsm`` array and check that it has a row per exported cell."""
    n_obs = plan.x.shape[0]
    values = np.asarray(plan.obsm[name]())
    if values.ndim != 2:
        raise ValueError(
            f"obsm {name!r} has {values.ndim} dimensions; "
            "obsm arrays are two-dimensional"
        )
    if values.shape[0] != n_obs:
        raise ValueError(
            f"obsm {name!r} has {values.shape[0]} rows; the export declares {n_obs}"
        )
    return values


def _write_plan(h5: Any, plan: H5adExportPlan) -> None:
    n_obs, n_vars = plan.x.shape
    _encoded(h5, "anndata", "0.1.0")
    for name in ("X", "obs", "var"):
        h5.create_group(name)
    obsm = _encoded(h5.create_group("obsm"), "dict", "0.1.0")
    _write_csr(h5["X"], plan.x, "The X matrix")
    obs_columns, colors = _write_frame(h5["obs"], plan.obs_index, plan.obs, n_obs)
    _write_frame(h5["var"], plan.var_index, plan.var, n_vars)
    for name in plan.obsm:
        _write_array(obsm, name, read_obsm(plan, name))
    if plan.layers:
        layers = _encoded(h5.create_group("layers"), "dict", "0.1.0")
        for name, layer in plan.layers.items():
            _write_csr(layers.create_group(name), layer, f"Layer {name!r}")
    if plan.obsp:
        obsp = _encoded(h5.create_group("obsp"), "dict", "0.1.0")
        for name in plan.obsp:
            _write_sparse(obsp, name, read_obsp(plan, name))
    if plan.varm:
        varm = _encoded(h5.create_group("varm"), "dict", "0.1.0")
        for name in plan.varm:
            _write_array(varm, name, read_varm(plan, name))
    uns = plan_uns(plan, obs_columns, colors)
    if uns:
        group = _encoded(h5.create_group("uns"), "dict", "0.1.0")
        for name, value in uns.items():
            _write_uns_value(group, name, value)


def write_h5ad_plan(plan: H5adExportPlan, path: str | os.PathLike[str]) -> None:
    """Stream ``plan`` to an H5AD file at ``path``, replacing it only once complete."""
    import h5py

    _check_plan(plan)
    directory = os.path.dirname(os.path.abspath(path))
    temporary = os.path.join(directory, f".{secrets.token_hex(8)}.h5ad.tmp")
    try:
        mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None
    # A replaced file keeps its permissions, also while its data is written.
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    os.close(os.open(temporary, flags, 0o666 if mode is None else 0o600))
    try:
        with h5py.File(temporary, "w") as h5:
            _write_plan(h5, plan)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.remove(temporary)
        raise


def with_count_dtype(
    plan: H5adExportPlan,
    dtype: Any | None,
    *,
    raw_x: bool,
) -> H5adExportPlan:
    """Return ``plan`` storing its raw counts in :func:`export_count_dtype`.

    The raw counts are ``X`` when ``raw_x`` is True, and every layer.
    """

    def converted(matrix: H5adMatrix) -> H5adMatrix:
        target = export_count_dtype(matrix.dtype, dtype)
        if target == matrix.dtype:
            return matrix
        return replace(matrix, export_dtype=target)

    return replace(
        plan,
        x=converted(plan.x) if raw_x else plan.x,
        layers={name: converted(layer) for name, layer in plan.layers.items()},
    )


# ---------------------------------------------------------------------------
# Artifacts


# Kinds whose rows follow the cell selection of their lineage.
_LINEAGE_KINDS = frozenset({"reduction", "batch_correction"})
_GRAPH_KINDS = frozenset({"connectivity_map", "integrated_graph"})
_SCANPY_METRICS = {"l2": "euclidean", "cosine": "cosine"}
# Bytes that exporting a graph holds per stored edge: its edges and weights,
# the directed and transposed CSR matrices, their product and fuzzy union,
# and the distances.
_GRAPH_EDGE_BYTES = 96


@dataclass(frozen=True, slots=True)
class H5adArtifactEntries:
    """The ``obsm``, ``obs``, ``obsp``, ``varm``, and ``uns`` entries of artifacts."""

    obsm: Mapping[str, Callable[[], np.ndarray]]
    obs: tuple[H5adColumn, ...]
    obsp: Mapping[str, Callable[[], csr_matrix]]
    varm: Mapping[str, Callable[[], np.ndarray]]
    uns: Mapping[str, Any]


def _source(value: Any, label: str) -> tuple[ArtifactRef, str | None]:
    """Return an export source as its artifact and the per-cell array it names."""
    if isinstance(value, ArtifactRef):
        return value, None
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], ArtifactRef)
        and isinstance(value[1], str)
        and value[1]
    ):
        return value[0], value[1]
    raise TypeError(
        f"{label} must be an ArtifactRef or an (ArtifactRef, array name) pair"
    )


def _sources(
    values: Mapping[str, H5adSource] | None,
    label: str,
) -> dict[str, tuple[ArtifactRef, str | None]]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must map names to artifact references")
    resolved: dict[str, tuple[ArtifactRef, str | None]] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} names must be non-empty strings")
        if "/" in name:
            raise ValueError(
                f"{label} name {name!r} contains '/', which AnnData does not allow "
                "in keys"
            )
        resolved[name] = _source(value, f"{label}[{name!r}]")
    return resolved


class _ArtifactRows:
    """Resolve and read the cell-aligned rows of artifacts, once per selection."""

    def __init__(
        self, root: "zarr.Group", cell_rows: np.ndarray, max_bytes: int | None
    ):
        self._root = root
        self._cellRows = np.asarray(cell_rows, dtype=np.int64)
        self._maxBytes = max_bytes
        self._selections: dict[ArtifactRef, np.ndarray] = {}

    def selection_rows(self, selection: ArtifactRef) -> np.ndarray:
        from ..storage.selections import read_stored_selection_indices

        if selection not in self._selections:
            self._selections[selection] = read_stored_selection_indices(
                self._root,
                selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)
        return self._selections[selection]

    def positions(
        self, label: str, ref: ArtifactRef, selection: ArtifactRef
    ) -> np.ndarray:
        """Return where each exported cell lies among the artifact's rows."""
        rows = self.selection_rows(selection)
        positions = np.searchsorted(rows, self._cellRows)
        covered = positions < rows.size
        covered[covered] = rows[positions[covered]] == self._cellRows[covered]
        if not covered.all():
            raise ValueError(
                f"{label}: the {ref.kind} artifact holds values for "
                f"{int(covered.sum())} of the {self._cellRows.size} exported cells. "
                "Export the cells of its cell selection, with "
                "cell_selection=<that selection>"
            )
        return positions

    def same_cells(self, label: str, ref: ArtifactRef, selection: ArtifactRef) -> None:
        """Require that an artifact holds exactly the exported cells."""
        rows = self.selection_rows(selection)
        if not np.array_equal(rows, self._cellRows):
            raise ValueError(
                f"{label}: the {ref.kind} artifact covers {rows.size} cells, and a "
                f"graph is exported only over exactly its cells; the export holds "
                f"{self._cellRows.size} other or more cells. Export the cells of "
                "its cell selection, with cell_selection=<that selection>"
            )


def _lineage_selection(root: "zarr.Group", ref: ArtifactRef) -> ArtifactRef:
    from ..graph.feature_projection import resolve_coordinate_inputs

    return resolve_coordinate_inputs(root, ref).cell_selection


def _recorded_selection(root: "zarr.Group", ref: ArtifactRef) -> ArtifactRef:
    from ..storage.artifacts import inspect_artifact

    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise ValueError(f"The {ref.kind} artifact is unavailable or incomplete")
    return status.input_ref("cell_selection")


def _cell_value_name(
    ref: ArtifactRef, value: str | None, label: str
) -> tuple[str, bool]:
    """Return the per-cell array of an artifact to export and whether it holds labels."""
    from ..metadata.selection import cell_value_array

    if ref.kind in _LINEAGE_KINDS:
        if value not in (None, "data"):
            raise ValueError(
                f"{label}: {ref.kind} artifacts export only their 'data' coordinates"
            )
        return "data", False
    try:
        return cell_value_array(ref.kind, value)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from None


def _array_ndim(root: "zarr.Group", ref: ArtifactRef, name: str, label: str) -> int:
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    group = artifact_group(root, ref)
    if name not in group:
        raise ValueError(f"{label}: the {ref.kind} artifact has no {name!r} array")
    return int(as_zarr_array(group[name], name=name).ndim)


def _read_cell_array(
    root: "zarr.Group",
    ref: ArtifactRef,
    value: str,
    positions: np.ndarray,
    max_bytes: int | None,
) -> np.ndarray:
    """Read an artifact's per-cell array at the rows of the exported cells."""
    return _read_cell_values(root, ref, value, positions, max_bytes)[0]


def _read_cell_values(
    root: "zarr.Group",
    ref: ArtifactRef,
    value: str,
    positions: np.ndarray,
    max_bytes: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an artifact's per-cell array at the rows of the exported cells."""
    from ..metadata.rows import read_array_rows_chunkwise
    from ..metadata.selection import resolve_cell_aligned_artifact
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    if ref.kind in _LINEAGE_KINDS:
        data = as_zarr_array(artifact_group(root, ref)["data"], name="data")
        return np.asarray(read_array_rows_chunkwise(data, positions)), None
    resolved = resolve_cell_aligned_artifact(
        root, ref, value_name=value, ndim=None, max_bytes=max_bytes
    )
    missing = (
        None if resolved.missing_mask is None else resolved.missing_mask[positions]
    )
    return np.asarray(resolved.values)[positions], missing


def _stored_labels(root: "zarr.Group", ref: ArtifactRef, value: str) -> H5adCategorical:
    """Return the categories that a label array records, or their natural order.

    Labels imported from a categorical keep its ``levels``, ``ordered``, and
    the colors of its ``display`` with their array.
    """
    from ..storage.artifacts import artifact_group

    attributes = artifact_group(root, ref)[value].attrs
    return stored_categorical(attributes) or H5adCategorical()


def stored_categorical(attributes: Mapping[str, Any]) -> H5adCategorical | None:
    """Return the categorical that a stored column's attributes record, or None.

    A column keeps the categories of a source categorical as ``levels`` and
    ``ordered`` attributes, and a categorical ``display`` gives their colors.
    """
    levels = attributes.get("levels")
    if not isinstance(levels, list | tuple):
        return None
    colors: dict[Any, str] | None = None
    display = attributes.get("display")
    if isinstance(display, Mapping) and display.get("kind") == "categorical":
        colors = {
            category["value"]: category["color"]
            for category in display.get("categories", ())
            if isinstance(category, Mapping) and "value" in category
        }
    return H5adCategorical(
        levels=tuple(levels),
        ordered=bool(attributes.get("ordered", False)),
        colors=colors,
    )


def _require_graph_memory(
    root: "zarr.Group",
    graph: ArtifactRef,
    max_bytes: int | None,
) -> None:
    """Refuse a graph whose export does not fit the memory budget."""
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    edges = as_zarr_array(artifact_group(root, graph)["edges"], name="edges")
    n_edges = int(edges.shape[0])
    required = n_edges * _GRAPH_EDGE_BYTES
    if max_bytes is not None and required > max_bytes:
        raise MemoryError(
            f"Exporting the graph's {n_edges} edges needs about {required} bytes, "
            f"but the memory budget is {max_bytes} bytes. Raise mem_budget."
        )


def _graph_matrices(
    root: "zarr.Group",
    graph: ArtifactRef,
) -> tuple[csr_matrix, csr_matrix | None]:
    """Return a graph's symmetric connectivities and its KNN distances.

    The connectivities are the fuzzy union ``W + W.T - W * W.T`` of the
    directed edge weights, which CyteArc's UMAP and Scanpy read. Distances
    exist only for a connectivity map, whose neighbors record them.
    """
    from ..graph.feature_projection import resolve_native_graph_inputs
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    group = artifact_group(root, graph)
    n_cells = int(cast(int, group.attrs["n_cells"]))
    edges = np.asarray(as_zarr_array(group["edges"], name="edges")[:]).astype(
        np.int64, copy=False
    )
    weights = np.asarray(as_zarr_array(group["weights"], name="weights")[:])
    distances: csr_matrix | None = None
    if graph.kind == "connectivity_map":
        neighbors = resolve_native_graph_inputs(root, graph).neighbors
        knn = artifact_group(root, neighbors)
        indices = np.asarray(as_zarr_array(knn["indices"], name="indices")[:])
        values = np.asarray(as_zarr_array(knn["distances"], name="distances")[:])
        distances = csr_matrix(
            (
                values.ravel(),
                (
                    np.repeat(np.arange(n_cells), indices.shape[1]),
                    indices.ravel().astype(np.int64),
                ),
            ),
            shape=(n_cells, n_cells),
        )
    directed = csr_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells)
    )
    transposed = directed.T.tocsr()
    union = csr_matrix(directed + transposed - directed.multiply(transposed))
    union.eliminate_zeros()
    return union, distances


def _graph_parameters(
    root: "zarr.Group",
    graph: ArtifactRef,
    use_rep: str | None,
) -> dict[str, Any]:
    """Return ``uns["neighbors"]["params"]`` of a graph in Scanpy's keys.

    ``method`` is ``umap`` for a connectivity map, whose weights follow
    umap-learn's kernel, and the integration method of an integrated graph.
    """
    from ..graph.feature_projection import resolve_native_graph_inputs
    from ..storage.artifacts import artifact_group, inspect_artifact

    k = int(cast(int, artifact_group(root, graph).attrs["n_neighbors"]))
    # Scanpy counts each cell among its own neighbors.
    params: dict[str, Any] = {"n_neighbors": k + 1}
    if graph.kind == "connectivity_map":
        neighbors = resolve_native_graph_inputs(root, graph).neighbors
        metric = str(
            (inspect_artifact(root, neighbors).parameters or {})["distance_metric"]
        )
        params["method"] = "umap"
        params["metric"] = _SCANPY_METRICS.get(metric, metric)
    else:
        params["method"] = str(
            (inspect_artifact(root, graph).parameters or {})["method"]
        )
    if use_rep is not None:
        params["use_rep"] = use_rep
    return params


def _pca_loadings(
    root: "zarr.Group",
    ref: ArtifactRef,
    feature_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the loading rows of a PCA and the feature ID of each, if it has them."""
    from ..assay.normalization import load_normalized_inputs
    from ..graph.feature_projection import resolve_coordinate_inputs
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    group = artifact_group(root, ref)
    if "loadings" not in group:
        return None
    loadings = np.asarray(as_zarr_array(group["loadings"], name="loadings")[:])
    if ref.kind == "imported_coordinates":
        ids = np.asarray(as_zarr_array(group["feature_ids"], name="feature_ids")[:])
        return loadings, ids.astype(str)
    normalized = resolve_coordinate_inputs(root, ref).normalized
    assert normalized is not None
    _group, selections = load_normalized_inputs(root, normalized)
    assay = cast(str, ref.assay)
    all_ids = np.asarray(
        as_zarr_array(root[f"{assay}/featureData/ids"], name="ids")[:]
    ).astype(str)
    return loadings, all_ids[np.asarray(selections.featureMask, dtype=bool)]


def _varm_pcs(
    loadings: np.ndarray, ids: np.ndarray, feature_ids: np.ndarray
) -> np.ndarray:
    """Place loading rows at the exported features that they belong to.

    Exported features without a loading hold zeros, as Scanpy stores the
    features that a PCA did not use.
    """
    pcs = np.zeros((feature_ids.size, loadings.shape[1]), dtype=loadings.dtype)
    position = {feature: index for index, feature in enumerate(ids.tolist())}
    rows = np.asarray([position.get(feature, -1) for feature in feature_ids.tolist()])
    found = rows >= 0
    pcs[found] = loadings[rows[found]]
    return pcs


def artifact_entries(
    root: "zarr.Group",
    *,
    cell_rows: np.ndarray,
    feature_ids: np.ndarray,
    obsm: Mapping[str, H5adSource] | None = None,
    obs: Mapping[str, H5adSource] | None = None,
    graph: ArtifactRef | None = None,
    max_bytes: int | None = None,
    exported: Mapping[str, ArtifactRef] | None = None,
) -> H5adArtifactEntries:
    """Plan the export of artifacts beside the cells at store rows ``cell_rows``.

    ``obsm`` names two-dimensional per-cell arrays, such as a ``reduction``,
    ``batch_correction``, ``imported_coordinates``, or ``embedding``. The
    entry named ``X_pca`` also writes its loadings, when the artifact has
    them, as ``varm["PCs"]``, aligned to ``feature_ids`` with zeros for the
    features it did not use. A PCA ``reduction`` that stores its variance and
    variance ratio writes them as ``uns["pca"]["variance"]`` and
    ``uns["pca"]["variance_ratio"]``, and an ``imported_coordinates``
    artifact with ``stdev`` writes their squares as
    ``uns["pca"]["variance"]``. ``obs``
    names one-dimensional per-cell arrays: labels become categoricals, with
    the categories, order, and colors that imported labels record, and
    numbers stay numbers. A pair ``(ref, name)`` names another per-cell
    array of the artifact. Each artifact must hold every exported cell.

    ``graph``, a ``connectivity_map`` or ``integrated_graph`` over exactly
    the exported cells, becomes ``obsp["connectivities"]``, the fuzzy union
    of its directed weights, and, for a connectivity map,
    ``obsp["distances"]``, its KNN distances, with ``uns["neighbors"]`` in
    Scanpy's layout, whose ``use_rep`` names the ``obsm`` entry, among
    ``obsm`` and the ``exported`` arrays of the export, that holds the
    graph's coordinates. ``uns["cytearc"]["artifacts"]`` records the
    reference of every exported artifact by element.

    Every reference is resolved and aligned here, and a graph's memory need
    is checked, so a mismatch raises before anything is written; values are
    read while the export writes them.
    """
    sources = _sources(obsm, "obsm")
    labels = _sources(obs, "obs")
    if graph is not None and not isinstance(graph, ArtifactRef):
        raise TypeError("graph must be an ArtifactRef")
    if graph is not None and graph.kind not in _GRAPH_KINDS:
        raise ValueError(
            f"graph must be a connectivity_map or integrated_graph artifact, not "
            f"{graph.kind!r}"
        )
    rows = _ArtifactRows(root, cell_rows, max_bytes)
    lineage: dict[str, Any] = {}
    obsm_entries: dict[str, Callable[[], np.ndarray]] = {}
    for name, (ref, value) in sources.items():
        label = f"obsm {name!r}"
        array, _labels = _cell_value_name(ref, value, label)
        selection = (
            _lineage_selection(root, ref)
            if ref.kind in _LINEAGE_KINDS
            else _recorded_selection(root, ref)
        )
        if _array_ndim(root, ref, array, label) != 2:
            raise ValueError(
                f"{label}: the {ref.kind} array {array!r} is not two-dimensional; "
                "export it in obs"
            )
        positions = rows.positions(label, ref, selection)
        obsm_entries[name] = partial(
            _read_cell_array, root, ref, array, positions, max_bytes
        )
        lineage.setdefault("obsm", {})[name] = _lineage_record(ref, value)
    obs_columns: list[H5adColumn] = []
    for name, (ref, value) in labels.items():
        label = f"obs {name!r}"
        array, categorical = _cell_value_name(ref, value, label)
        if ref.kind in _LINEAGE_KINDS:
            raise ValueError(f"{label}: {ref.kind} coordinates belong in obsm")
        selection = _recorded_selection(root, ref)
        if _array_ndim(root, ref, array, label) != 1:
            raise ValueError(
                f"{label}: the {ref.kind} array {array!r} is not one-dimensional; "
                "export it in obsm"
            )
        positions = rows.positions(label, ref, selection)
        obs_columns.append(
            H5adColumn(
                name,
                partial(_read_cell_values, root, ref, array, positions, max_bytes),
                categorical=(_stored_labels(root, ref, array) if categorical else None),
            )
        )
        lineage.setdefault("obs", {})[name] = _lineage_record(ref, value)
    varm: dict[str, Callable[[], np.ndarray]] = {}
    uns: dict[str, Any] = {}
    if "X_pca" in sources:
        ref, _value = sources["X_pca"]
        if ref.kind in {"reduction", "imported_coordinates"}:
            loadings = _pca_loadings(root, ref, feature_ids)
            if loadings is not None:
                varm["PCs"] = partial(_varm_pcs, *loadings, np.asarray(feature_ids))
        if ref.kind == "reduction":
            variance = _reduction_variance(root, ref)
            if variance is not None:
                uns["pca"] = variance
        elif ref.kind == "imported_coordinates":
            stdev = _imported_stdev(root, ref)
            if stdev is not None:
                uns["pca"] = {"variance": stdev**2}
    obsp: dict[str, Callable[[], csr_matrix]] = {}
    if graph is not None:
        from ..graph.feature_projection import graph_cell_selection

        rows.same_cells("graph", graph, graph_cell_selection(root, graph))
        _require_graph_memory(root, graph, max_bytes)
        matrices = _LazyGraph(root, graph)
        obsp["connectivities"] = matrices.connectivities
        neighbors: dict[str, Any] = {"connectivities_key": "connectivities"}
        if graph.kind == "connectivity_map":
            obsp["distances"] = matrices.distances
            neighbors["distances_key"] = "distances"
        coordinates = _graph_coordinates(root, graph)
        represented = {
            **(exported or {}),
            **{name: ref for name, (ref, value) in sources.items() if value is None},
        }
        use_rep = next(
            (name for name, ref in represented.items() if ref == coordinates), None
        )
        neighbors["params"] = _graph_parameters(root, graph, use_rep)
        uns["neighbors"] = neighbors
        lineage["obsp"] = {"connectivities": _lineage_record(graph, None)}
    if lineage:
        uns["cytearc"] = {"artifacts": lineage}
    return H5adArtifactEntries(
        obsm=obsm_entries,
        obs=tuple(obs_columns),
        obsp=obsp,
        varm=varm,
        uns=uns,
    )


def _lineage_record(ref: ArtifactRef, value: str | None) -> dict[str, str]:
    record = ref.to_dict()
    if value is not None:
        record["value"] = value
    return record


def _reduction_variance(
    root: "zarr.Group", ref: ArtifactRef
) -> dict[str, np.ndarray] | None:
    """Return a PCA's stored variance and variance ratio, or None without them.

    PCA artifacts store both from release 1.3.0; earlier ones store neither.
    """
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    group = artifact_group(root, ref)
    if "variance" not in group or "variance_ratio" not in group:
        return None
    return {
        name: np.asarray(as_zarr_array(group[name], name=name)[:])
        for name in ("variance", "variance_ratio")
    }


def _imported_stdev(root: "zarr.Group", ref: ArtifactRef) -> np.ndarray | None:
    from ..storage.artifacts import artifact_group
    from ..storage.types import as_zarr_array

    group = artifact_group(root, ref)
    if "stdev" not in group:
        return None
    return np.asarray(as_zarr_array(group["stdev"], name="stdev")[:])


def _graph_coordinates(root: "zarr.Group", graph: ArtifactRef) -> ArtifactRef | None:
    """Return the coordinates that a connectivity map was built on."""
    from ..graph.feature_projection import resolve_native_graph_inputs

    if graph.kind != "connectivity_map":
        return None
    return resolve_native_graph_inputs(root, graph).coordinates


class _LazyGraph:
    """Read a graph's matrices once, when the export first writes one."""

    def __init__(self, root: "zarr.Group", graph: ArtifactRef):
        self._args = (root, graph)
        self._matrices: tuple[csr_matrix, csr_matrix | None] | None = None

    def _read(self) -> tuple[csr_matrix, csr_matrix | None]:
        if self._matrices is None:
            self._matrices = _graph_matrices(*self._args)
        return self._matrices

    def connectivities(self) -> csr_matrix:
        return self._read()[0]

    def distances(self) -> csr_matrix:
        distances = self._read()[1]
        assert distances is not None
        return distances


def plain_uns(value: Any) -> Any:
    """Return ``uns`` values with every mapping as a plain ``dict``."""
    if isinstance(value, Mapping):
        return {str(key): plain_uns(item) for key, item in value.items()}
    return value


def _exported_obsm(plan: H5adExportPlan) -> dict[str, ArtifactRef]:
    """Return the artifact of each ``obsm`` array that a plan already exports.

    Earlier :func:`with_artifacts` calls record them in
    ``uns["cytearc"]["artifacts"]["obsm"]``.
    """
    records = plan.uns.get("cytearc", {}).get("artifacts", {}).get("obsm", {})
    return {
        name: ArtifactRef.from_dict(record)
        for name, record in records.items()
        if "value" not in record
    }


def with_artifacts(
    plan: H5adExportPlan,
    root: "zarr.Group",
    *,
    cell_rows: np.ndarray,
    obsm: Mapping[str, H5adSource] | None = None,
    obs: Mapping[str, H5adSource] | None = None,
    graph: ArtifactRef | None = None,
    max_bytes: int | None = None,
) -> H5adExportPlan:
    """Return ``plan`` with the entries of :func:`artifact_entries` added.

    ``cell_rows`` are the store rows of the plan's cells, in their order. A
    name that the plan already exports raises ``ValueError``.
    """
    feature_ids = np.asarray(_read_column(plan.var_index, plan.x.shape[1])[0]).astype(
        str
    )
    entries = artifact_entries(
        root,
        cell_rows=cell_rows,
        feature_ids=feature_ids,
        obsm=obsm,
        obs=obs,
        graph=graph,
        max_bytes=max_bytes,
        exported=_exported_obsm(plan),
    )
    planned = {plan.obs_index.name, *(column.name for column in plan.obs)}
    for column in entries.obs:
        if column.name in planned:
            raise ValueError(
                f"obs {column.name!r} names a column that the export already holds; "
                "choose another name"
            )
    for name in entries.obsm:
        if name in plan.obsm:
            raise ValueError(
                f"obsm {name!r} names an array that the export already holds; "
                "choose another name"
            )
    return replace(
        plan,
        obs=(*plan.obs, *entries.obs),
        obsm={**plan.obsm, **entries.obsm},
        obsp={**plan.obsp, **entries.obsp},
        varm={**plan.varm, **entries.varm},
        uns=_merged(plan.uns, entries.uns),
    )


# ---------------------------------------------------------------------------
# Live exports


def _embedding_groups(
    columns: list[str],
    assay_name: str,
    prefixes: list[str],
) -> dict[str, list[str]]:
    """Group live embedding columns by prefix in numeric component order."""
    components: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for column in columns:
        match = re.fullmatch(r"(.+?)(\d+)", column)
        if match is None:
            continue
        prefix, component = match.groups()
        if any(prefix.startswith(f"{assay_name}_{name}") for name in prefixes):
            components[prefix].append((int(component), column))
    return {
        prefix: [column for _component, column in sorted(items)]
        for prefix, items in components.items()
    }


def _live_column(
    table: Any,
    column: str,
    rows: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one live metadata column and its linked missing mask."""
    mask = metadata_missing_mask(table, column)
    missing = None if mask is None else np.asarray(mask[:], dtype=bool)
    values = np.asarray(table.fetch_all(column))
    if rows is None:
        return values, missing
    return values[rows], None if missing is None else missing[rows]


def live_categorical(table: Any, column: str) -> H5adCategorical | None:
    """Return the categorical that a live column's stored attributes record."""
    return stored_categorical(table._get_array(column).attrs)


def _live_embedding(
    cells: Any, columns: list[str], rows: np.ndarray | None
) -> np.ndarray:
    """Stack live embedding columns into a cells-by-components array."""
    stacked = np.array([cells.fetch_all(column) for column in columns]).T
    return stacked if rows is None else stacked[rows]


def _live_count_blocks(
    assay: Any,
    nthreads: int,
    rows: np.ndarray | None,
) -> Iterator[np.ndarray]:
    """Stream the raw counts of an assay's cells, every cell by default.

    The writer's conversion of each block to CSR is charged as resident.
    """
    counts = assay.rawData if rows is None else assay.rawData[rows, :]
    blocks: Iterator[np.ndarray] = counts._stream_blocks(
        nthreads=nthreads,
        msg="Writing raw counts",
        prefetch=None,
        row_mask=None,
        resident_bytes=h5ad_conversion_bytes(
            counts.dtype, counts.shape[1], largest_block_rows(counts)
        ),
    )
    return blocks


def _live_count_nonzero(assay: Any, nthreads: int) -> int:
    """Count the nonzero raw counts of an assay."""
    return int(
        compute_with_progress(
            assay.rawData.count_nonzero(),
            msg="Counting nonzero entries",
            nthreads=nthreads,
        )
    )


def live_h5ad_plan(
    assay: Any,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    nthreads: int = 4,
    *,
    cell_rows: np.ndarray | None = None,
) -> H5adExportPlan:
    """Plan the H5AD export of an assay with its live metadata.

    Every cell is exported, or the cells at store rows ``cell_rows``. A
    column that holds a source categorical's ``levels`` exports as that
    categorical, with the colors of its ``display``. Column names that
    AnnData reserves or that the export already uses are renamed
    (:func:`unique_column_names`).
    """
    from ..metadata.membership import exported_membership

    membership, omitted = exported_membership(assay.cells, assay.name)
    cells, feats = assay.cells, assay.feats
    if embeddings_cols is None:
        embeddings_cols = ["UMAP", "tSNE"]
    embeddings = _embedding_groups(list(cells.columns), assay.name, embeddings_cols)
    embedding_columns = {
        column for columns in embeddings.values() for column in columns
    }
    obsm: dict[str, Callable[[], np.ndarray]] = {}
    for prefix, columns in embeddings.items():
        name = prefix.lower().replace(f"{assay.name.lower()}_", "X_")
        if name in obsm:
            raise ValueError(
                f"Embedding columns of two prefixes both export as obsm[{name!r}]"
            )
        obsm[name] = partial(_live_embedding, cells, columns, cell_rows)

    def planned(
        table: Any,
        keys: list[str],
        names: list[str],
        rows: np.ndarray | None,
    ) -> tuple[H5adColumn, ...]:
        return tuple(
            H5adColumn(
                name,
                partial(_live_column, table, key, rows),
                categorical=live_categorical(table, key),
            )
            for key, name in zip(keys, names, strict=True)
        )

    obs_keys = [
        key
        for key in cells.columns
        if key != "ids" and key not in omitted and key not in embedding_columns
    ]
    var_keys = [key for key in feats.columns if key != "ids"]
    var_names = unique_column_names(
        "_index",
        ["gene_short_name" if key == "names" else key for key in var_keys],
        table="var",
    )
    n_cells = int(cells.N) if cell_rows is None else int(np.asarray(cell_rows).size)
    return H5adExportPlan(
        x=H5adMatrix(
            shape=(n_cells, int(feats.N)),
            dtype=np.dtype(assay.rawData.dtype),
            blocks=partial(_live_count_blocks, assay, nthreads, cell_rows),
            # A preliminary pass counts the stored values to allocate.
            reserve=(
                None
                if skip_recalc_nfeats or cell_rows is not None
                else partial(_live_count_nonzero, assay, nthreads)
            ),
        ),
        obs_index=H5adColumn("_index", partial(_live_column, cells, "ids", cell_rows)),
        obs=planned(
            cells,
            obs_keys,
            unique_column_names("_index", obs_keys, table="obs"),
            cell_rows,
        ),
        var_index=H5adColumn("_index", partial(_live_column, feats, "ids", None)),
        var=planned(feats, var_keys, var_names, None),
        obsm=obsm,
        membership=membership,
    )


def cell_selection_rows(root: Any, selection: ArtifactRef) -> np.ndarray:
    """Return the store rows of the cells that a cell selection selects."""
    from ..storage.selections import read_stored_selection_indices

    if not isinstance(selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    if selection.kind != "cell_selection":
        raise ValueError(
            f"cell_selection must be a cell_selection artifact, not {selection.kind!r}"
        )
    return read_stored_selection_indices(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)


def to_h5ad(
    assay: Any,
    h5ad_filename: str,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    nthreads: int = 4,
    *,
    run: object | None = None,
    matrix: Literal["raw", "normed"] = "raw",
    cell_selection: ArtifactRef | None = None,
    obsm: Mapping[str, H5adSource] | None = None,
    obs: Mapping[str, H5adSource] | None = None,
    graph: ArtifactRef | None = None,
    dtype: Any | None = None,
) -> None:
    """Save an assay or a completed pipeline run as an H5AD file.

    The file is streamed block by block with h5py and needs no AnnData.
    Rows that a nullable metadata column's linked missing mask flags are
    written as missing values: NaN in a float64 column for numeric columns, a
    nullable boolean for boolean columns, and a missing category for other
    columns. A column that keeps a source categorical's categories, as H5AD
    and Seurat imports store them, is written as that categorical, with its
    category order, unused categories, ``ordered`` flag, and colors in
    ``uns["<column>_colors"]``. A column that AnnData reserves, ``_index``,
    exports as ``source_index``, and a column whose name the export already
    uses, such as a ``gene_short_name`` beside the feature names, takes the
    next free ``_2`` name, with a warning.

    Args:
        assay: Assay to save in H5ad format
        h5ad_filename: Name for the H5ad file to be created.
        embeddings_cols: Cell-metadata column prefixes treated as embeddings
                         (for example UMAP, tSNE). When None, uses
                         ``["UMAP", "tSNE"]``. Pass an empty list to skip
                         embeddings.
        skip_recalc_nfeats: Skip a preliminary nonzero-count pass. (Default value: True)
        nthreads: Number of processing threads to use (Default value: 4)
        run: Completed pipeline run opened from the datastore that owns
             ``assay``. The frozen run selections and fields are exported.
             Consecutive frozen UMAP fields are in ``obsm['X_umap']`` and
             the run's PCA in ``obsm['X_pca']``, as in
             ``to_anndata(run=...)``; cluster labels (``clusters``,
             ``leiden_*``, ``paris``) are categoricals in ``obs``. Live
             embedding columns, a cell selection, and feature-count
             recalculation options do not apply to run export.
        matrix: ``"raw"`` counts, or ``"normed"`` values, which require
                ``run``: the run's stored normalized values over its highly
                variable features.
        cell_selection: Cell-selection artifact whose cells are exported, in
                        store order. None exports every cell of the assay.
        obsm: Artifacts to export as ``obsm`` arrays, such as
              ``{"X_pca": pca, "X_umap": umap}``.
        obs: Cell-aligned artifacts to export as ``obs`` columns, such as
             ``{"leiden": leiden}``; labels become categoricals.
        graph: Connectivity map or integrated graph to export as
               ``obsp["connectivities"]`` and ``obsp["distances"]`` with
               ``uns["neighbors"]``, over exactly the exported cells.
        dtype: Dtype of the exported raw counts, in ``X`` and every layer.
               None keeps the stored dtype, except that ``uint8`` and
               ``uint16`` counts become ``float32``. A count that the dtype
               cannot hold raises ``OverflowError``.

    Returns:
        None

    Every artifact is resolved and aligned to the exported cells before
    anything is written, so a misaligned artifact raises ``ValueError`` and
    leaves no file.
    """
    if matrix not in ("raw", "normed"):
        raise ValueError("matrix must be either 'raw' or 'normed'")
    if dtype is not None:
        export_count_dtype(None, dtype)
    if run is None:
        if matrix != "raw":
            raise ValueError(
                "matrix='normed' requires run: only a pipeline run freezes "
                "normalized values"
            )
        cell_rows = (
            None
            if cell_selection is None
            else cell_selection_rows(assay._artifact_root, cell_selection)
        )
        plan = live_h5ad_plan(
            assay,
            embeddings_cols,
            skip_recalc_nfeats,
            nthreads,
            cell_rows=cell_rows,
        )
        if obsm is not None or obs is not None or graph is not None:
            plan = with_artifacts(
                plan,
                assay._artifact_root,
                cell_rows=(
                    np.arange(int(assay.cells.N), dtype=np.int64)
                    if cell_rows is None
                    else cell_rows
                ),
                obsm=obsm,
                obs=obs,
                graph=graph,
                max_bytes=int(assay.resources.memoryBytes),
            )
        write_h5ad_plan(with_count_dtype(plan, dtype, raw_x=True), h5ad_filename)
        logger.info(
            f"Exported {plan.x.shape[0]} cells and {assay.feats.N} features "
            f"to {h5ad_filename}"
        )
        return None

    from ..storage.pipeline_runs import PipelineRunRecord

    if not isinstance(getattr(run, "_record", None), PipelineRunRecord):
        raise TypeError("run must be a PipelineRun")
    pipeline_run = cast(Any, run)
    pipeline_run._require_completed("H5AD export")
    owner = pipeline_run._owner
    get_assay = getattr(owner, "_get_assay", None)
    # Writers may not import the datastore, so the run's owner plans it.
    run_plan = getattr(owner, "_h5ad_run_plan", None)
    if not callable(get_assay) or not callable(run_plan):
        raise TypeError("run must be opened from a DataStore")
    if get_assay(pipeline_run.assay) is not assay:
        raise ValueError("assay must be the exact run assay owned by the run datastore")
    if embeddings_cols is not None:
        raise ValueError(
            "Run-aware export uses frozen embedding fields; "
            "embeddings_cols cannot be provided"
        )
    if skip_recalc_nfeats is not True:
        raise ValueError(
            "Run-aware export uses the frozen selection; "
            "skip_recalc_nfeats cannot be disabled"
        )
    if nthreads != 4:
        raise ValueError(
            "Run-aware export uses the datastore execution settings; "
            "nthreads cannot be overridden"
        )
    if cell_selection is not None:
        raise ValueError(
            "Run-aware export uses the frozen run cells; cell_selection cannot "
            "be provided"
        )

    plan = run_plan(pipeline_run, matrix=matrix)
    if obsm is not None or obs is not None or graph is not None:
        plan = with_artifacts(
            plan,
            owner.zw,
            cell_rows=np.flatnonzero(pipeline_run.cells.fetch_all("I")),
            obsm=obsm,
            obs=obs,
            graph=graph,
            max_bytes=int(assay.resources.memoryBytes),
        )
    write_h5ad_plan(with_count_dtype(plan, dtype, raw_x=matrix == "raw"), h5ad_filename)
    n_cells, n_features = plan.x.shape
    logger.info(
        f"Exported pipeline run {pipeline_run.run_id} with {n_cells} cells and "
        f"{n_features} features to {h5ad_filename}"
    )
    return None


# ---------------------------------------------------------------------------
# Matrix Market


def _digit_counts(values: np.ndarray) -> np.ndarray:
    """Return the number of decimal digits of each non-negative integer."""
    width = np.ones(values.shape, dtype=np.int64)
    alive = np.flatnonzero(values >= 10)
    rest: Any = values[alive] // 10
    while alive.size:
        width[alive] += 1
        rest //= 10
        keep = rest > 0
        alive = alive[keep]
        rest = rest[keep]
    return width


def _integer_lines(columns: Sequence[np.ndarray]) -> bytes:
    """Return rows of non-negative integer columns as space-separated text lines.

    The digits are computed with NumPy, so the text equals what Python's
    ``str`` writes for each value, and each line ends with a newline.
    """
    n_rows = int(columns[0].size)
    if n_rows == 0:
        return b""
    values = [
        np.asarray(column).astype(
            np.uint32 if int(np.max(column)) <= np.iinfo(np.uint32).max else np.uint64,
            copy=False,
        )
        for column in columns
    ]
    widths = [_digit_counts(column) for column in values]
    lengths = np.add.reduce(widths) + len(columns)
    ends = np.cumsum(lengths)
    text = np.empty(int(ends[-1]), dtype=np.uint8)
    position = ends - lengths
    for index, (column, width) in enumerate(zip(values, widths, strict=True)):
        # Digits are written from the last to the first, while any remain.
        last = position + width - 1
        alive = np.arange(n_rows)
        rest: Any = column.copy()
        while alive.size:
            text[last[alive]] = (rest % 10).astype(np.uint8) + ord("0")
            last[alive] -= 1
            rest //= 10
            keep = rest > 0
            alive = alive[keep]
            rest = rest[keep]
        position = position + width
        text[position] = ord(" ") if index < len(columns) - 1 else ord("\n")
        position += 1
    return text.tobytes()


def _matrix_market_lines(
    features: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
) -> bytes:
    """Return Matrix Market coordinate lines, ``feature cell value``, one-based.

    Integer values are formatted with NumPy; other values as ``str`` of the
    value in its dtype, as pandas writes them.
    """
    if values.dtype.kind in "biu":
        return _integer_lines([features, cells, values.astype(np.uint64)])
    text = values.astype(str).tolist()
    return "".join(
        f"{feature} {cell} {value}\n"
        for feature, cell, value in zip(
            features.tolist(), cells.tolist(), text, strict=True
        )
    ).encode("utf-8")


def _assay_feature_type(assay: Any, default: str) -> str:
    """Return the feature type of an exported assay.

    It is the value of the assay's ``feature_type`` column when every
    feature holds it, and ``default`` otherwise.
    """
    if "feature_type" not in assay.feats.columns:
        return default
    stored = np.unique(np.asarray(assay.feats.fetch_all("feature_type")).astype(str))
    if stored.size == 1:
        return str(stored[0])
    logger.warning(
        f"Assay {assay.name!r} holds {stored.size} feature types; the Matrix Market "
        f"export writes {default!r} for every feature"
    )
    return default


def to_mtx(assay: Any, mtx_directory: str, compress: bool = False) -> None:
    """Save an assay as a Matrix Market directory.

    The features file holds a feature ID, name, and type column, as Cell
    Ranger 3 writes them, whether or not it is compressed, so an import
    keeps the assay's type: ``features.tsv.gz`` with ``compress=True``, and
    ``genes.tsv``, the Cell Ranger 2 name, without compression. Every
    feature takes one type: the assay's ``feature_type`` column when every
    feature holds the same value, such as ``Antibody Capture``, else
    ``Gene Expression`` for an RNA assay and the assay name for another, so
    gene biotypes that CELLxGENE keeps in ``feature_type`` do not split the
    assay on import. Compressed files use gzip level 6.

    Args:
        assay: CyteArc assay. For example: `ds.RNA`
        mtx_directory: Out directory where MTX file will be saved along with barcodes and features file
        compress: If True, then the files are compressed and saved with .gz extension, using Cell Ranger 3
                  names. (Default value: False).

    Returns:
        None

    Raises:
        UnmeasuredCellsError: If the assay did not measure a cell.
    """
    from ..assay.classification import is_rna_assay_type
    from ..metadata.membership import require_measured_cells

    require_measured_cells(
        assay.cells,
        assay.name,
        np.ones(assay.cells.N, dtype=bool),
        operation="to_mtx",
        remedy="export",
    )
    if os.path.isdir(mtx_directory) is False:
        os.mkdir(mtx_directory)

    tot_counts = int(
        compute_with_progress(
            assay.rawData.count_nonzero(),
            msg="Counting nonzero entries",
            nthreads=assay.nthreads,
        )
    )
    if compress:
        barcodes_fn = "barcodes.tsv.gz"
        features_fn = "features.tsv.gz"
        matrix_path = os.path.join(mtx_directory, "matrix.mtx.gz")
    else:
        barcodes_fn = "barcodes.tsv"
        features_fn = "genes.tsv"
        matrix_path = os.path.join(mtx_directory, "matrix.mtx")
    numeric_type = (
        "integer" if np.issubdtype(assay.rawData.dtype, np.integer) else "real"
    )
    with (
        gzip.open(matrix_path, "wb", compresslevel=6)
        if compress
        else open(matrix_path, "wb")
    ) as handle:
        handle.write(
            f"%%MatrixMarket matrix coordinate {numeric_type} general\n"
            "% Generated by CyteArc\n"
            f"{assay.feats.N} {assay.cells.N} {tot_counts}\n".encode()
        )
        start = 0
        for values in assay.rawData.stream_blocks(
            nthreads=assay.nthreads,
            msg="Writing Matrix Market counts",
        ):
            block = np.asarray(values)
            rows, columns = np.nonzero(block)
            handle.write(
                _matrix_market_lines(
                    columns.astype(np.int64) + 1,
                    rows.astype(np.int64) + start + 1,
                    block[rows, columns],
                )
            )
            start += block.shape[0]
    assay.cells.to_pandas_dataframe(["ids"]).to_csv(
        os.path.join(mtx_directory, barcodes_fn), sep="\t", header=False, index=False
    )

    features = assay.feats.to_pandas_dataframe(["ids", "names"])
    # Cell Ranger 3 feature files carry a third feature-type column, which
    # holds the one type of the exported assay.
    features["feature_type"] = _assay_feature_type(
        assay, "Gene Expression" if is_rna_assay_type(assay) else assay.name
    )
    features.to_csv(
        os.path.join(mtx_directory, features_fn), sep="\t", header=False, index=False
    )
    logger.info(
        f"Exported {assay.cells.N} cells and {assay.feats.N} features "
        f"to {mtx_directory}"
    )
