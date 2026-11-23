from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import zarr

from ..metadata.artifacts import (
    MAX_STORED_CATEGORIES,
    distinct_level_positions,
    stored_levels,
    validate_display_metadata,
)
from ..metadata.membership import measured_feature_columns
from ..metadata.palettes import stored_category_colors
from ..metadata.rows import (
    array_row_selection_parts,
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_array_rows_chunkwise,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ..storage.arrays import (
    MISSING_MASK_PREFIX,
    MetadataBlock,
    create_metadata_column,
    create_streamed_metadata_column,
)
from ..metadata.selection import valid_category_mask
from ..storage.budget import ResourceBudget
from ..storage.categorical import CategoricalColumn
from ..storage.cell_order import (
    SOURCE_ROW,
    SOURCE_ROW_ROLE,
    is_source_row_column,
    source_order_ranks,
)
from ..storage.copy import COLUMN_METADATA_ATTRIBUTES
from ..storage.execution import admitted_worker_split
from ..storage.identity import GENERATED_FEATURE_COLUMNS, clear_column
from ..storage.layout import (
    PROFILE_METADATA_CHUNK,
    _encoded_chunk_bound,
    column_chunk_rows,
)
from ..storage.metadata_keys import (
    ASSAY_MEMBERSHIP_ROLE,
    MEASURED_FEATURES_ROLE,
    assay_membership_column,
    is_reserved_metadata_name,
    measured_features_column,
    metadata_column_key,
    validate_metadata_column_name,
)
from ..storage.partition import affordable_width
from ..storage.profiles import StorageProfile
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.arrays import sort_categories
from ..utils.logging import logger
from ..utils.memory_budget import MemoryBudgetError
from .row_plan import (
    RowPlan,
    RowPlanSegment,
    iter_row_plan_segments,
    max_row_plan_block_rows,
    prefixed_cell_ids,
    verify_merged_cell_ids,
)


_PROTECTED = frozenset({"ids", "I", "names", SOURCE_ROW})
# Bytes per merged cell that writing the merged sourceRow holds: each cell's
# row in its source and its merged row, which the write keeps, and, while a
# source is read, its cells' rows as gathered and sorted and their source
# rows and ranks.
_SOURCE_ROW_BYTES = 40
# Column attributes carried into merged metadata; a selection fingerprint
# describes one source's features and never survives a merge.
_MERGED_ATTRIBUTES = tuple(
    key for key in COLUMN_METADATA_ATTRIBUTES if key != "feature_selection_fingerprint"
)
# Only the merge marks a merged cell column as an assay's membership, so an
# ordinary column never carries the membership attributes of a source.
_CELL_ATTRIBUTES = tuple(
    key for key in _MERGED_ATTRIBUTES if key not in {"role", "assay"}
)
# The attributes through which a cell column records its categories, which a
# merge reconciles together.
_CATEGORY_ATTRIBUTES = ("levels", "ordered", "display")

SOURCE_LACKS_ASSAY = "missing"
"""Membership state of a source that does not hold the assay."""
SOURCE_MEASURES_ALL = "all"
"""Membership state of a source that measures every cell with the assay."""

type SourceMembership = Mapping[str, Sequence[str]]
"""Per merged assay, each source's membership state in source order.

A state is :data:`SOURCE_LACKS_ASSAY`, :data:`SOURCE_MEASURES_ALL`, or the
name of the source's membership column of the assay.
"""


def _cell_data_path(workspace: str | None) -> str:
    return "cellData" if workspace is None else f"{workspace}/cellData"


class _SourceConflict(ValueError):
    """Merged sources disagree on a column value or its metadata."""


@dataclass(frozen=True, slots=True)
class MetadataColumnSpec:
    name: str
    dtype: np.dtype[Any]
    hasMissing: bool
    role: str | None = None
    assay: str | None = None
    sourceReadFixedBytes: int = 0
    sourceReadBytesPerRow: int = 0
    maskReadFixedBytes: int = 0
    maskReadBytesPerRow: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


def _chunk_resident_bytes(dtype: np.dtype[Any], chunk_rows: int) -> int:
    raw = max(1, int(chunk_rows)) * max(1, int(dtype.itemsize))
    return int(2 * raw + 2 * _encoded_chunk_bound(raw))


def _column_working_bytes(
    spec: MetadataColumnSpec,
    rows: int,
) -> int:
    width = max(1, int(rows))
    itemsize = max(1, int(spec.dtype.itemsize))
    mask_bytes = width * np.dtype(bool).itemsize if spec.hasMissing else 0
    string_bytes = width * itemsize if spec.dtype.kind in {"U", "S", "O"} else 0
    staging = width * 3 * itemsize + mask_bytes + string_bytes
    source_read = max(0, int(spec.sourceReadFixedBytes)) + width * max(
        0, int(spec.sourceReadBytesPerRow)
    )
    mask_read = (
        width * itemsize
        + max(0, int(spec.maskReadFixedBytes))
        + width * max(0, int(spec.maskReadBytesPerRow))
    )
    peaks = [staging, source_read]
    if spec.maskReadFixedBytes or spec.maskReadBytesPerRow:
        peaks.append(mask_read)
    return max(peaks, default=1)


def _band_write_bytes(spec: MetadataColumnSpec, rows: int, chunk_rows: int) -> int:
    """Bound writing one column through whole destination chunk bands.

    A band holds one chunk of values and missing flags while source segments
    of ``rows`` rows fill it. The full chunk is then copied and encoded once.
    """
    chunk = max(1, int(chunk_rows))
    values = chunk * max(1, int(spec.dtype.itemsize))
    missing = chunk * np.dtype(bool).itemsize if spec.hasMissing else 0
    encode = max(
        values + _encoded_chunk_bound(values),
        missing + _encoded_chunk_bound(missing) if missing else 0,
    )
    return values + missing + max(_column_working_bytes(spec, rows), encode)


@dataclass(frozen=True, slots=True)
class CellMetadataPlan:
    columns: tuple[MetadataColumnSpec, ...]
    blockRows: int
    # Public column name to source column name, per source.
    sourceColumns: tuple[dict[str, str], ...]

    def peak_write_bytes_at(self, rows: int, *, chunk_rows: int) -> int:
        """Bound writing each column ``rows`` rows at a time, the costliest of them.

        ``chunk_rows`` is the chunk of a column of narrow values, which a
        wider column clips to the rows whose values fit 16 MiB.
        """
        return max(
            (
                _band_write_bytes(
                    spec,
                    rows,
                    column_chunk_rows(
                        chunk_rows, max(1, spec.dtype.itemsize), max_rows=chunk_rows
                    ),
                )
                for spec in self.columns
            ),
            default=1,
        )


def metadata_chunk_rows(row_plan: RowPlan, dtype: Any = bool) -> int:
    """Return the rows of each stored chunk of a merged cell column of ``dtype``.

    A column of narrow values, the default, has chunks of at most 100,000 of
    the merged cells; a wider one, as many of them as fit 16 MiB.
    """
    return column_chunk_rows(
        int(row_plan.nCells),
        max(1, np.dtype(dtype).itemsize),
        max_rows=PROFILE_METADATA_CHUNK,
    )


def effective_metadata_segment_rows(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
) -> int:
    return max(
        1,
        min(int(metadata_plan.blockRows), metadata_chunk_rows(row_plan)),
    )


def _admits(admit: Callable[[int], None], width: int) -> bool:
    """Whether ``admit`` admits ``width`` rows; it refuses with MemoryBudgetError."""
    try:
        admit(width)
    except MemoryBudgetError:
        return False
    return True


def _refuse_one_row(admit: Callable[[int], None], words: str, item: str) -> NoReturn:
    """Raise the refusal of one row, which names the budget that admits it."""
    try:
        admit(1)
    except MemoryBudgetError as error:
        raise MemoryBudgetError(
            f"{words}. It needs at least {error.needed_bytes} bytes.",
            needed_bytes=error.needed_bytes,
            budget_bytes=error.budget_bytes,
            item=item,
        ) from error
    raise RuntimeError(f"{item} was admitted after it was refused")


def resolve_metadata_segment_rows(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> int:
    """Return the largest metadata write width that fits the budget."""
    preferred = max(
        1,
        min(int(metadata_plan.blockRows), metadata_chunk_rows(row_plan)),
    )
    chunk_rows = metadata_chunk_rows(row_plan)

    def admit(width: int) -> None:
        admitted_worker_split(
            resources,
            nTasks=1,
            residentBytes=max(0, int(resident_bytes)),
            taskBytes=lambda _: metadata_plan.peak_write_bytes_at(
                width,
                chunk_rows=chunk_rows,
            ),
            requested=1,
        )

    rows = affordable_width(partial(_admits, admit), preferred)
    if rows < 1:
        _refuse_one_row(
            admit,
            "Merged cell metadata cannot fit one row within the operation memory budget",
            "one row of merged cell metadata",
        )
    return int(rows)


def admit_cell_metadata_plan(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> CellMetadataPlan:
    """Resolve a budget-admitted metadata width and keep schema specs stable."""
    admitted = resolve_metadata_segment_rows(
        metadata_plan,
        row_plan,
        resources,
        resident_bytes=resident_bytes,
    )
    return replace(metadata_plan, blockRows=admitted)


def resolve_identity_validation_rows(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
    stored_ids: Any,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> int:
    """Return the largest admitted cell-identity validation width."""
    preferred = min(
        metadata_chunk_rows(row_plan),
        max(1, max_row_plan_block_rows(row_plan)),
    )
    ids_spec = next(spec for spec in metadata_plan.columns if spec.name == "ids")
    stored_fixed, _ = array_row_selection_parts(stored_ids)
    band_rows = min(max(1, int(stored_ids.chunks[0])), max(1, row_plan.nCells))
    band_bytes = band_rows * max(1, int(np.dtype(stored_ids.dtype).itemsize))

    def admit(width: int) -> None:
        # One decoded chunk band of stored ids stays resident while the
        # expected ids of its segments are read and compared.
        task_bytes = band_bytes + max(
            stored_fixed, _column_working_bytes(ids_spec, width)
        )
        admitted_worker_split(
            resources,
            nTasks=1,
            residentBytes=max(0, int(resident_bytes)),
            taskBytes=lambda _: task_bytes,
            requested=1,
        )

    rows = affordable_width(partial(_admits, admit), preferred)
    if rows < 1:
        _refuse_one_row(
            admit,
            "Merged cell identity validation cannot fit one row within the "
            "operation memory budget",
            "one row of the merged cell identity validation",
        )
    return int(rows)


def validate_prepend_text(prepend_text: str | None) -> None:
    """Reject a merged-column prefix that would nest or hide columns.

    Raises:
        TypeError: If ``prepend_text`` is neither a string nor None.
        ValueError: If it contains ``/`` or ``\\``, which Zarr reads as path
            separators, or makes prefixed names start with the missing-value
            mask prefix.
    """
    if prepend_text is None:
        return
    if not isinstance(prepend_text, str):
        raise TypeError(
            f"prepend_text must be a string or None, not {type(prepend_text).__name__}"
        )
    if prepend_text and (
        metadata_column_key(prepend_text) != prepend_text
        or is_reserved_metadata_name(f"{prepend_text}_")
    ):
        raise ValueError(
            f"prepend_text {prepend_text!r} must not contain '/' or '\\', which "
            "Zarr reads as path separators, or start with CyteArc's missing-value "
            "mask prefix"
        )


def _public_column_name(
    column: str,
    prepend_text: str | None,
) -> str:
    if column in _PROTECTED:
        return column
    if prepend_text is None or prepend_text == "":
        return column
    return f"{prepend_text}_{column}"


class _Labels:
    """The labels of a merged column, as it stores them, up to the categories that a column keeps.

    Missing and blank values hold no label. More than
    :data:`MAX_STORED_CATEGORIES` labels overflow, and the merged column then
    keeps its labels alone.
    """

    def __init__(self) -> None:
        self.found: set[Any] = set()
        self.overflow = False

    def add(self, values: np.ndarray, missing: np.ndarray | None) -> None:
        if self.overflow:
            return
        valid = valid_category_mask(values, missing_mask=missing)
        self.found.update(pd.unique(np.asarray(values)[valid]).tolist())
        if len(self.found) > MAX_STORED_CATEGORIES:
            self.overflow = True
            self.found = set()


def _max_text_width(
    table: Any,
    column: str,
    *,
    block_rows: int,
    labels: _Labels | None = None,
    taken: np.ndarray | None = None,
) -> int:
    """The width of a source column's widest value as text, read a block at a time.

    With ``labels``, the same pass adds the labels of the rows that the
    merged store takes, ``taken`` in increasing order or every row for None,
    as text, which the merged column stores them as. Its missing mask is
    read at those rows alone, as the merge reads it to write them.
    """
    width = 1
    mask = None if labels is None else metadata_missing_mask(table, column)
    start = 0
    for values in iter_metadata_column_blocks(
        table,
        column,
        block_rows=block_rows,
    ):
        stop = start + len(values)
        strings = None
        if np.asarray(values).dtype.kind == "O":
            width = max(width, max((len(str(value)) for value in values), default=0))
        else:
            strings = np.asarray(values, dtype=str)
            width = max(width, int(np.char.str_len(strings).max(initial=0)))
        if labels is not None:
            if strings is None:
                strings = np.asarray(values, dtype=str)
            if taken is None:
                labels.add(
                    strings,
                    None if mask is None else np.asarray(mask[start:stop], dtype=bool),
                )
            else:
                rows = taken[
                    np.searchsorted(taken, start) : np.searchsorted(taken, stop)
                ]
                labels.add(
                    strings[rows - start],
                    None
                    if mask is None
                    else np.asarray(read_array_rows_chunkwise(mask, rows), dtype=bool),
                )
        start = stop
    return width


def _add_column_labels(
    labels: _Labels,
    table: Any,
    column: str,
    dtype: np.dtype[Any],
    *,
    block_rows: int,
    taken: np.ndarray | None,
) -> None:
    """Add the labels of a source column's rows that the merged store takes, cast to the merged ``dtype`` as the merge writes them.

    Without ``taken`` every row is read a block at a time; with it, the
    rows that it lists, as the merge reads them to write them.
    """
    if taken is None:
        mask = metadata_missing_mask(table, column)
        start = 0
        for values in iter_metadata_column_blocks(table, column, block_rows=block_rows):
            stop = start + len(values)
            labels.add(
                np.asarray(values).astype(dtype, copy=False),
                None if mask is None else np.asarray(mask[start:stop], dtype=bool),
            )
            start = stop
        return
    for start in range(0, len(taken), block_rows):
        rows = taken[start : start + block_rows]
        labels.add(
            np.asarray(read_metadata_rows_chunkwise(table, column, rows)).astype(
                dtype, copy=False
            ),
            read_metadata_missing_rows_chunkwise(table, column, rows),
        )


def _string_itemsize_bound(dtype: np.dtype[Any]) -> int:
    if dtype.kind == "U":
        return max(1, int(dtype.itemsize))
    if dtype.kind == "S":
        return max(1, 4 * int(dtype.itemsize))
    if dtype.kind == "b":
        return 4 * len("False")
    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        return 4 * max(len(str(info.min)), len(str(info.max)))
    if dtype.kind == "f":
        return 4 * 64
    if dtype.kind == "c":
        return 4 * 128
    if dtype.kind in {"M", "m"}:
        return 4 * 64
    return max(1, int(dtype.itemsize))


def resolve_metadata_schema_scan_rows(
    source_cell_tables: list[Any],
    resources: ResourceBudget,
    *,
    resident_bytes: int,
    preferred_rows: int,
    axis: str = "cell",
) -> int:
    """Return the largest admitted metadata schema-scan width.

    ``axis`` names the tables scanned, cell or feature tables, in a refusal.
    """
    arrays: list[Any] = []
    for table in source_cell_tables:
        for column in table.columns:
            dtype = np.dtype(table.get_dtype(column))
            if column in {"ids", "names"} or dtype.kind in {"U", "S", "O"}:
                arrays.append(table._get_array(column))
    preferred = max(1, int(preferred_rows))

    def admit(width: int) -> None:
        task_bytes: list[int] = []
        for array in arrays:
            dtype = np.dtype(array.dtype)
            itemsize = max(1, int(dtype.itemsize))
            text_itemsize = _string_itemsize_bound(dtype)
            selection = array_row_selection_parts(array)
            selection_peak = selection[0] + width * selection[1]
            conversion_peak = width * (
                itemsize + text_itemsize + np.dtype(np.int64).itemsize
            )
            task_bytes.append(max(selection_peak, conversion_peak))
        admitted_worker_split(
            resources,
            nTasks=1,
            residentBytes=max(0, int(resident_bytes)),
            taskBytes=lambda _: max(task_bytes, default=1),
            requested=1,
        )

    rows = affordable_width(partial(_admits, admit), preferred)
    if rows < 1:
        _refuse_one_row(
            admit,
            f"Merged {axis} metadata schema discovery cannot fit one source row "
            "within the operation memory budget",
            f"one source row of merged {axis} metadata schema discovery",
        )
    return int(rows)


def _max_selection_parts(arrays: Iterable[Any]) -> tuple[int, int]:
    fixed = 0
    per_row = 0
    for array in arrays:
        array_fixed, array_per_row = array_row_selection_parts(array)
        fixed = max(fixed, array_fixed)
        per_row = max(per_row, array_per_row)
    return fixed, per_row


def _metadata_column_spec(
    name: str,
    dtype: np.dtype[Any],
    has_missing: bool,
    *,
    value_arrays: Iterable[Any] = (),
    mask_arrays: Iterable[Any] = (),
    role: str | None = None,
    assay: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> MetadataColumnSpec:
    source_fixed, source_per_row = _max_selection_parts(value_arrays)
    mask_fixed, mask_per_row = _max_selection_parts(mask_arrays)
    return MetadataColumnSpec(
        name,
        dtype,
        has_missing,
        role=role,
        assay=assay,
        sourceReadFixedBytes=source_fixed,
        sourceReadBytesPerRow=source_per_row,
        maskReadFixedBytes=mask_fixed,
        maskReadBytesPerRow=mask_per_row,
        attributes={} if attributes is None else attributes,
    )


def _source_attribute(attrs: Mapping[str, Any], key: str) -> Any:
    """Return a source column's attribute as a merge compares and keeps it.

    Imports of CyteArc 1.3.0 and 1.4.0 store levels that read alike for
    categories that read alike once stored. A merge takes such levels as the
    first of them (:func:`~cytearc.metadata.artifacts.distinct_level_positions`),
    so their import of a file agrees with an import by this release.
    """
    value = attrs[key]
    if key != "levels" or not isinstance(value, list | tuple):
        return value
    return [value[position] for position in distinct_level_positions(value)]


def _records_categories(array: Any) -> bool:
    """Whether a source column records categories: ``levels``, a categorical ``display``, or the categories of a categorical column."""
    attrs = getattr(array, "attrs", {})
    display = attrs.get("display")
    return (
        isinstance(array, CategoricalColumn)
        or isinstance(attrs.get("levels"), list | tuple)
        or (isinstance(display, Mapping) and display.get("kind") == "categorical")
    )


def _storable_category(value: Any) -> bool:
    """Whether a category is a value that ``levels`` and a display hold: text, a finite number or a boolean."""
    return isinstance(value, bool | int | str) or (
        isinstance(value, float) and bool(np.isfinite(value))
    )


def _source_categories(
    array: Any, recorded: Mapping[str, Any]
) -> tuple[list[Any] | None, bool]:
    """A source column's categories in the order its plots list them, or None, and whether they are ordered.

    A categorical display lists them in its order. Otherwise the ``levels``
    that an import keeps from a source categorical list them, or the
    categories of a categorical column, which records whether they are
    ordered; a categorical column of more than
    :data:`MAX_STORED_CATEGORIES` categories, which a column keeps none of,
    is not read.
    """
    levels = recorded.get("levels")
    if not isinstance(levels, list | tuple):
        # Levels of another type are no categories, and are compared as stored.
        levels = None
    ordered = recorded.get("ordered") is True
    if isinstance(array, CategoricalColumn):
        ordered = ordered or array.ordered
        if levels is None and array.n_categories <= MAX_STORED_CATEGORIES:
            levels = array.categories.tolist()
    display = recorded.get("display")
    if isinstance(display, Mapping) and display.get("kind") == "categorical":
        return [category["value"] for category in display["categories"]], ordered
    return (None if levels is None else list(levels)), ordered


def _merged_categories(
    sources: Sequence[tuple[list[Any] | None, bool]],
    observed: Collection[Any] | None,
    dtype: np.dtype[Any],
) -> tuple[list[Any], bool] | None:
    """The categories of a merged column, as it stores them, and whether they stay ordered, or None.

    Each source's categories are compared as the merged column of ``dtype``
    stores its values (:func:`~cytearc.metadata.artifacts.stored_levels`),
    as are ``observed``, the labels of the rows that the merged store takes.
    The first source that records categories gives their order, and every
    other category and label follows in natural order, as plots draw the
    labels that a stored order leaves out. The categories stay ordered only
    when every source records them ordered, in that order. None, which
    keeps the merged column's labels alone, as an import keeps a categorical
    that a column cannot hold, stands for ``observed`` of more labels than a
    column keeps, a merged column of a type that holds no levels, such as
    dates, a source's categories that the merged column cannot hold, or
    more than :data:`MAX_STORED_CATEGORIES` categories in all.
    """
    if observed is None:
        return None
    orders: list[list[Any] | None] = []
    for order, _ordered in sources:
        stored = None if order is None else stored_levels(dtype, order)
        if order is not None and stored is None:
            return None
        orders.append(None if stored is None else list(dict.fromkeys(stored)))
    listed = dict.fromkeys(next(order for order in orders if order is not None))
    others = {label for order in orders for label in order or ()}
    others.update(observed)
    categories = [
        *listed,
        *sort_categories([label for label in others if label not in listed]),
    ]
    if len(categories) > MAX_STORED_CATEGORIES or not all(
        _storable_category(value) for value in categories
    ):
        return None
    ordered = all(
        is_ordered and order == categories
        for order, (_order, is_ordered) in zip(orders, sources, strict=True)
    )
    return categories, ordered


def _merged_display(
    displays: Sequence[Mapping[str, Any]],
    categories: Sequence[Any],
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    """The categorical display of a merged column's categories, in their order.

    A category takes its label and color from the first display that lists
    it, its value as the merged column of ``dtype`` stores it. A later
    display's color is kept only where no earlier entry has it, and each
    category without a color takes the color that a stored display gives its
    place in the order (:func:`~cytearc.metadata.palettes.stored_category_colors`),
    so the display colors every category of the column. The missing label
    and color come from the first display that records them.
    """
    entries: dict[Any, dict[str, Any]] = {}
    used: set[str] = set()
    missing: dict[str, Any] = {}
    for number, display in enumerate(displays):
        values = stored_levels(
            dtype, [category["value"] for category in display["categories"]]
        )
        assert values is not None
        for value, category in zip(values, display["categories"], strict=True):
            if value in entries:
                continue
            color = str(category["color"])
            kept = number == 0 or color.lower() not in used
            entries[value] = {
                "value": value,
                "label": category["label"],
                "color": color if kept else None,
            }
            if kept:
                used.add(color.lower())
        for key in ("missing_label", "missing_color"):
            if key in display:
                missing.setdefault(key, display[key])
    generated: list[str] | None = None
    listed = []
    for position, value in enumerate(categories):
        entry = entries.get(value) or {
            "value": value,
            "label": str(value),
            "color": None,
        }
        if entry["color"] is None:
            if generated is None:
                generated = stored_category_colors(len(categories))
            entry = {**entry, "color": generated[position]}
        listed.append(entry)
    return {"kind": "categorical", "categories": listed, **missing}


def _reconciled_cell_attributes(
    arrays: Iterable[Any],
    observed: Collection[Any] | None,
    dtype: np.dtype[Any],
) -> tuple[dict[str, Any], list[str]]:
    """Merge one cell column's source attributes.

    Equal values are kept, and any other disagreement drops the attribute;
    the dropped names are returned. Membership attributes are never
    carried. A column whose sources record categories, as ``levels``, a
    categorical ``display`` or the categories of a categorical column, keeps
    the categories of :func:`_merged_categories`: as ``levels`` when a source
    records levels or categories, with ``ordered`` when every source records
    them ordered in that order, and as a ``display`` that covers each of them
    when a source records one (:func:`_merged_display`); a display of
    another kind is dropped. ``observed`` holds the labels of the rows that
    the merged store takes, as the merged column of ``dtype`` stores them,
    or None for more labels than a column keeps.
    """
    arrays = list(arrays)
    recorded: list[dict[str, Any]] = []
    for array in arrays:
        attrs = getattr(array, "attrs", {})
        recorded.append(
            {
                key: _source_attribute(attrs, key)
                for key in _CELL_ATTRIBUTES
                if key in attrs
            }
        )
    sources = [
        _source_categories(array, attributes)
        for array, attributes in zip(arrays, recorded, strict=True)
    ]
    categorical = any(order is not None for order, _ in sources)
    found: dict[str, list[Any]] = {}
    for attributes in recorded:
        for key, value in attributes.items():
            found.setdefault(key, []).append(value)
    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for key, values in found.items():
        if categorical and key in _CATEGORY_ATTRIBUTES:
            continue
        if all(value == values[0] for value in values[1:]):
            kept[key] = values[0]
        else:
            dropped.append(key)
    if not categorical:
        return kept, dropped
    displays = [
        display
        for attributes in recorded
        if isinstance(display := attributes.get("display"), Mapping)
        and display.get("kind") == "categorical"
    ]
    # A display of another kind, such as a continuous one, holds no categories.
    other_display = len(displays) < len(found.get("display", ()))
    merged = (
        None
        # A categorical column of more categories than a column keeps.
        if any(
            isinstance(array, CategoricalColumn)
            and array.n_categories > MAX_STORED_CATEGORIES
            for array in arrays
        )
        else _merged_categories(sources, observed, dtype)
    )
    if merged is None:
        dropped.extend(key for key in _CATEGORY_ATTRIBUTES if key in found)
        return kept, dropped
    categories, ordered = merged
    if "levels" in found or any(
        isinstance(array, CategoricalColumn) for array in arrays
    ):
        kept["levels"] = categories
        kept["ordered"] = ordered
    if not ordered and any(is_ordered for _, is_ordered in sources):
        dropped.append("ordered")
    if other_display:
        dropped.append("display")
    if displays:
        kept["display"] = validate_display_metadata(
            _merged_display(displays, categories, dtype)
        )
    return kept, dropped


def plan_cell_metadata(
    source_cell_tables: list[Any],
    source_names: list[str],
    *,
    prepend_text: str | None,
    reset_cell_filter: bool,
    source_column: str | None,
    membership: SourceMembership | None = None,
    block_rows: int = 100_000,
    scan_rows: int | None = None,
    excluded_columns: list[frozenset[str]] | None = None,
    n_cells: int | None = None,
    row_plan: RowPlan | None = None,
) -> CellMetadataPlan:
    """Resolve the destination cell-metadata schema without writing.

    ``n_cells``, the cells that the merge takes from its sources, sizes what
    writing ``sourceRow`` holds; None counts every cell of every source.
    ``row_plan``, the rows that a composite takes from each source, limits
    the labels that become a column's categories to those rows; None takes
    every row.
    """
    block_rows = max(1, int(block_rows))
    scan_rows = block_rows if scan_rows is None else max(1, int(scan_rows))
    if source_column is not None and (
        not isinstance(source_column, str)
        or not source_column.strip()
        or source_column in _PROTECTED
    ):
        raise ValueError(
            "source_column must be a non-empty string that is not ids, I, names, "
            f"or {SOURCE_ROW}"
        )
    if source_column is not None:
        validate_metadata_column_name(source_column)
    validate_prepend_text(prepend_text)
    if prepend_text == "":
        prepend_text = None

    # Collect public columns per source.
    per_source: list[dict[str, str]] = []
    for index, table in enumerate(source_cell_tables):
        mapping: dict[str, str] = {}
        for column in table.columns:
            if excluded_columns is not None and column in excluded_columns[index]:
                continue
            public = _public_column_name(column, prepend_text)
            if public not in _PROTECTED:
                # Settings changed after construction reach the plan unchecked.
                validate_metadata_column_name(public)
            mapping[public] = column
        per_source.append(mapping)

    membership = dict(membership or {})
    membership_columns = {
        assay_membership_column(assay_name): assay_name for assay_name in membership
    }
    for name, mapping in zip(source_names, per_source, strict=True):
        for public, column in mapping.items():
            if public in membership_columns:
                raise ValueError(
                    f"Cell column {column!r} of source {name!r} would be merged "
                    f"as {public!r}, which is the merged membership column of "
                    f"assay {membership_columns[public]!r}. Rename the column in "
                    "the source or choose a prepend_text that keeps the names apart."
                )
    all_public: set[str] = set()
    for mapping in per_source:
        all_public.update(mapping)
    all_public.update(_PROTECTED)
    all_public.update(membership_columns)
    if source_column is not None:
        if source_column in all_public:
            raise ValueError(
                f"source_column {source_column!r} conflicts with merged metadata"
            )
        all_public.add(source_column)

    columns: list[MetadataColumnSpec] = []
    dropped_attributes: dict[str, list[str]] = {}

    taken_rows: dict[int, np.ndarray] = {}

    def taken(index: int) -> np.ndarray | None:
        """The rows of source ``index`` that the plan takes, in increasing order, or None for every row."""
        if row_plan is None:
            return None
        if index not in taken_rows:
            parts = [
                segment.localRows
                for segment in iter_row_plan_segments(row_plan)
                if segment.sourceIdx == index
            ]
            taken_rows[index] = (
                np.sort(np.concatenate(parts)) if parts else np.empty(0, np.int64)
            )
        return taken_rows[index]

    def sources_of(public: str) -> list[tuple[int, Any, str]]:
        """Each source that holds the column, by its index, table and column name."""
        return [
            (index, table, source_col)
            for index, (table, mapping) in enumerate(
                zip(source_cell_tables, per_source, strict=True)
            )
            if (source_col := mapping.get(public)) is not None
        ]

    def category_labels(public: str) -> _Labels | None:
        """A collector of the column's labels, when a source records its categories."""
        if any(
            _records_categories(table._get_array(column))
            for _, table, column in sources_of(public)
        ):
            return _Labels()
        return None

    def attributes_for(
        public: str, dtype: np.dtype[Any], labels: _Labels | None = None
    ) -> dict[str, Any]:
        """The column's attributes, from ``labels`` that a scan of its text widths collected, or else read here."""
        sources = sources_of(public)
        arrays = [table._get_array(column) for _, table, column in sources]
        if labels is None and (labels := category_labels(public)) is not None:
            for (index, table, column), array in zip(sources, arrays, strict=True):
                # A categorical column holds no label beyond its categories.
                if not isinstance(array, CategoricalColumn):
                    _add_column_labels(
                        labels,
                        table,
                        column,
                        dtype,
                        block_rows=scan_rows,
                        taken=taken(index),
                    )
        kept, dropped = _reconciled_cell_attributes(
            arrays,
            () if labels is None else None if labels.overflow else labels.found,
            dtype,
        )
        if dropped:
            dropped_attributes[public] = dropped
        return kept

    for name, table in zip(source_names, source_cell_tables, strict=True):
        if SOURCE_ROW in table.columns and not is_source_row_column(
            table._get_array(SOURCE_ROW)
        ):
            logger.warning(
                f"Source {name!r} holds a cell column {SOURCE_ROW!r} that no "
                "import wrote; the merged store's column holds each cell's row "
                "in the concatenation of the sources instead"
            )

    # Stable order: ids, names, I, sourceRow, source, membership, then
    # remaining sorted.
    ordered = ["ids", "names", "I", SOURCE_ROW]
    if source_column is not None:
        ordered.append(source_column)
    ordered.extend(membership_columns)
    remaining = sorted(name for name in all_public if name not in ordered)
    ordered.extend(remaining)

    for public in ordered:
        if public == "ids":
            max_len = 1
            for table, name in zip(source_cell_tables, source_names, strict=True):
                max_len = max(
                    max_len,
                    len(name)
                    + 2
                    + _max_text_width(
                        table,
                        "ids",
                        block_rows=scan_rows,
                    ),
                )
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(f"U{max_len}"),
                    False,
                    value_arrays=(
                        table._get_array("ids") for table in source_cell_tables
                    ),
                    attributes=attributes_for(public, np.dtype(f"U{max_len}")),
                )
            )
            continue
        if public == "names":
            max_len = 1
            for table in source_cell_tables:
                max_len = max(
                    max_len,
                    _max_text_width(
                        table,
                        "names",
                        block_rows=scan_rows,
                    ),
                )
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(f"U{max_len}"),
                    False,
                    value_arrays=(
                        table._get_array("names") for table in source_cell_tables
                    ),
                    attributes=attributes_for(public, np.dtype(f"U{max_len}")),
                )
            )
            continue
        if public == "I":
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(bool),
                    False,
                    value_arrays=(
                        ()
                        if reset_cell_filter
                        else (table._get_array("I") for table in source_cell_tables)
                    ),
                    attributes=attributes_for(public, np.dtype(bool)),
                )
            )
            continue
        if public == SOURCE_ROW:
            columns.append(
                replace(
                    _metadata_column_spec(
                        public,
                        np.dtype(np.int64),
                        False,
                        value_arrays=(
                            table._get_array(SOURCE_ROW)
                            for table in source_cell_tables
                            if SOURCE_ROW in table.columns
                        ),
                        role=SOURCE_ROW_ROLE,
                    ),
                    sourceReadFixedBytes=_SOURCE_ROW_BYTES
                    * (
                        sum(int(table.N) for table in source_cell_tables)
                        if n_cells is None
                        else int(n_cells)
                    ),
                )
            )
            continue
        if public == source_column:
            max_len = max(1, max(len(name) for name in source_names))
            columns.append(
                _metadata_column_spec(public, np.dtype(f"U{max_len}"), False)
            )
            continue
        if public in membership_columns:
            assay_name = membership_columns[public]
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(bool),
                    False,
                    value_arrays=(
                        table._get_array(state)
                        for table, state in zip(
                            source_cell_tables, membership[assay_name], strict=True
                        )
                        if state not in {SOURCE_LACKS_ASSAY, SOURCE_MEASURES_ALL}
                    ),
                    role=ASSAY_MEMBERSHIP_ROLE,
                    assay=assay_name,
                )
            )
            continue

        present_dtypes: list[np.dtype[Any]] = []
        present_count = 0
        for table, mapping in zip(source_cell_tables, per_source, strict=True):
            source_col = mapping.get(public)
            if source_col is None:
                continue
            present_count += 1
            present_dtypes.append(np.dtype(table.get_dtype(source_col)))
        has_missing = present_count < len(source_cell_tables) or any(
            source_col is not None
            and metadata_missing_mask(table, source_col) is not None
            for table, mapping in zip(source_cell_tables, per_source, strict=True)
            if (source_col := mapping.get(public)) is not None
        )
        dtype = _promote_dtypes(present_dtypes)
        labels: _Labels | None = None
        if dtype.kind in {"U", "S", "O"}:
            # The pass that measures the text widths also collects the labels
            # of a column whose sources record categories.
            labels = category_labels(public)
            max_len = 1
            for index, _table, source_col in sources_of(public):
                max_len = max(
                    max_len,
                    _max_text_width(
                        source_cell_tables[index],
                        source_col,
                        block_rows=scan_rows,
                        labels=labels,
                        taken=None if labels is None else taken(index),
                    ),
                )
            dtype = np.dtype(f"U{max_len}")
        value_arrays: list[Any] = []
        mask_arrays: list[Any] = []
        for table, mapping in zip(source_cell_tables, per_source, strict=True):
            source_col = mapping.get(public)
            if source_col is None:
                continue
            value_arrays.append(table._get_array(source_col))
            mask = metadata_missing_mask(table, source_col)
            if mask is not None:
                mask_arrays.append(mask)
        columns.append(
            _metadata_column_spec(
                public,
                dtype,
                has_missing,
                value_arrays=value_arrays,
                mask_arrays=mask_arrays,
                attributes=attributes_for(public, dtype, labels),
            )
        )

    if dropped_attributes:
        logger.warning(
            "Cell metadata attributes that differ between merged sources were "
            "dropped: "
            + "; ".join(
                f"{column} ({', '.join(keys)})"
                for column, keys in dropped_attributes.items()
            )
        )
    return CellMetadataPlan(
        tuple(columns),
        block_rows,
        sourceColumns=tuple(per_source),
    )


def _promote_dtypes(dtypes: list[np.dtype[Any]]) -> np.dtype[Any]:
    if not dtypes:
        return np.dtype(bool)
    if all(dtype == dtypes[0] for dtype in dtypes):
        return dtypes[0]
    if all(dtype.kind in "biu" for dtype in dtypes):
        return np.result_type(*dtypes)
    if all(dtype.kind in "bif" for dtype in dtypes):
        return np.dtype(np.float64)
    # Numbers mixed with complex values widen to complex128 like real mixes
    # widen to float64, so no imaginary part is cast away.
    if all(dtype.kind in "bifc" for dtype in dtypes):
        return np.dtype(np.complex128)
    if any(dtype.kind in {"U", "S", "O"} for dtype in dtypes):
        return np.dtype("U1")
    return np.dtype(np.float64)


def _fill_value(dtype: np.dtype[Any]) -> Any:
    if dtype.kind == "b":
        return False
    if dtype.kind in "iu":
        return 0
    if dtype.kind in "fc":
        return np.nan
    return ""


def _merged_column_attributes(arrays: Iterable[Any], name: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for array in arrays:
        attrs = getattr(array, "attrs", {})
        for key in _MERGED_ATTRIBUTES:
            if key not in attrs:
                continue
            value = _source_attribute(attrs, key)
            if key in attributes and attributes[key] != value:
                raise _SourceConflict(
                    f"Conflicting metadata for column {name!r}: {key}"
                )
            attributes[key] = value
    return attributes


@dataclass(frozen=True)
class _FeatureColumnPlan:
    """One merged feature column: its sources, dtype, and rows per write."""

    name: str
    sources: tuple[tuple[Any, np.ndarray], ...]
    arrays: tuple[Any, ...]
    dtype: np.dtype[Any]
    blockRows: int


def _annotation_columns(tables: list[Any]) -> list[set[str]]:
    """Return each table's columns that a merge may carry as feature annotations.

    A table's own record of which features its sources measured describes
    that table's sources, which the merge records anew, so it is left out
    with the columns that CyteArc writes for every feature table.
    """
    excluded = {"ids", "names", "I", *GENERATED_FEATURE_COLUMNS}
    return [
        set(table.columns) - excluded - set(measured_feature_columns(table))
        for table in tables
    ]


def _plan_feature_columns(
    tables: list[Any],
    mappings: list[np.ndarray],
    n_features: int,
    *,
    resources: ResourceBudget,
    resident_bytes: int,
    reserved: frozenset[str] = frozenset(),
) -> list[_FeatureColumnPlan]:
    """Resolve every merged feature column and the rows it writes at once.

    Columns named in ``reserved``, which the merge writes itself, are left
    out. Raises MemoryError when a column cannot fit one row within the
    budget.
    """
    table_columns = _annotation_columns(tables)
    columns = sorted(set().union(*table_columns) - reserved)
    mapping_bytes = max((mapping.nbytes * 3 for mapping in mappings), default=0)
    scan_rows = resolve_metadata_schema_scan_rows(
        tables,
        resources,
        resident_bytes=resident_bytes + mapping_bytes,
        preferred_rows=PROFILE_METADATA_CHUNK,
        axis="feature",
    )
    plans: list[_FeatureColumnPlan] = []
    for name in columns:
        sources = [
            (table, mapping)
            for table, mapping, present in zip(
                tables, mappings, table_columns, strict=True
            )
            if name in present
        ]
        arrays = [table._get_array(name) for table, _ in sources]
        dtype = _promote_dtypes([np.dtype(array.dtype) for array in arrays])
        if dtype.kind in {"U", "S", "O"}:
            width = max(
                (np.dtype(array.dtype).itemsize // 4)
                if np.dtype(array.dtype).kind == "U"
                else _max_text_width(table, name, block_rows=scan_rows)
                for (table, _), array in zip(sources, arrays, strict=True)
            )
            dtype = np.dtype(f"U{width}")
        masks = [metadata_missing_mask(table, name) for table, _ in sources]
        spec = _metadata_column_spec(
            name,
            dtype,
            True,
            value_arrays=arrays,
            mask_arrays=[mask for mask in masks if mask is not None],
        )
        available = resources.memoryBytes - resident_bytes - mapping_bytes

        def cost(rows: int) -> int:
            return (
                _column_working_bytes(spec, rows)
                + rows * (4 * dtype.itemsize + 32)
                + _chunk_resident_bytes(dtype, rows)
            )

        # The column is written a stored chunk at a time, of at most the rows
        # that the rule gives a feature column of this width, and of fewer
        # where the budget holds fewer.
        block_rows = affordable_width(
            lambda rows: cost(rows) <= available,
            column_chunk_rows(n_features, max(1, dtype.itemsize)),
        )
        if block_rows < 1:
            raise MemoryBudgetError(
                f"Merged feature column {name!r} cannot fit the memory budget",
                needed_bytes=resident_bytes + mapping_bytes + cost(1),
                budget_bytes=int(resources.memoryBytes),
                item=f"one row of merged feature column {name!r}",
            )
        plans.append(
            _FeatureColumnPlan(name, tuple(sources), tuple(arrays), dtype, block_rows)
        )
    return plans


def admit_feature_metadata(
    tables: list[Any],
    mappings: list[np.ndarray],
    n_features: int,
    *,
    resources: ResourceBudget,
    resident_bytes: int,
    sources: Iterable[str] = (),
) -> None:
    """Raise the MemoryError that merging feature annotations would raise.

    Planning calls this so a merge whose feature columns cannot fit the
    budget fails before it creates or changes the destination. ``sources``
    names the sources whose measured features the merge records, as
    :func:`write_feature_metadata` takes them.
    """
    _plan_feature_columns(
        tables,
        mappings,
        n_features,
        resources=resources,
        resident_bytes=resident_bytes,
        reserved=frozenset(measured_features_column(name) for name in sources),
    )


def write_feature_metadata(
    tables: list[Any],
    mappings: list[np.ndarray],
    destination: zarr.Group,
    n_features: int,
    *,
    resources: ResourceBudget,
    resident_bytes: int,
    profile: StorageProfile,
    measured: Mapping[str, np.ndarray],
) -> None:
    """Merge feature annotations that agree across sources, and record each source's measured features.

    A column whose values or metadata differ for a shared feature describes
    the source datasets rather than the features, so it is left out with a
    warning. Per-dataset analysis outputs such as highly variable gene flags
    usually fall in this group. Every column is planned before any is written.

    ``measured`` maps each source that holds the assay to whether it
    measured each merged feature, which a Boolean column of its own records,
    named by :func:`~cytearc.storage.metadata_keys.measured_features_column`
    and marked by the ``role`` attribute
    :data:`~cytearc.storage.metadata_keys.MEASURED_FEATURES_ROLE`. A source
    column of the same name is left out with a warning, and a source's own
    record is left out, as the merged record holds it.
    """
    reserved = frozenset(measured_features_column(name) for name in measured)
    taken = sorted(reserved & set().union(*_annotation_columns(tables)))
    skipped: list[str] = []
    for column_plan in _plan_feature_columns(
        tables,
        mappings,
        n_features,
        resources=resources,
        resident_bytes=resident_bytes,
        reserved=reserved,
    ):
        name = column_plan.name
        sources = column_plan.sources
        dtype = column_plan.dtype
        block_rows = column_plan.blockRows

        def blocks() -> Iterator[MetadataBlock]:
            for start in range(0, n_features, block_rows):
                stop = min(start + block_rows, n_features)
                values = np.full(stop - start, _fill_value(dtype), dtype=dtype)
                missing = np.ones(stop - start, dtype=bool)
                for table, mapping in sources:
                    rows = np.flatnonzero((mapping >= start) & (mapping < stop))
                    incoming = read_metadata_rows_chunkwise(table, name, rows).astype(
                        dtype
                    )
                    absent = read_metadata_missing_rows_chunkwise(table, name, rows)
                    positions = mapping[rows] - start
                    present = (
                        np.ones(len(rows), dtype=bool) if absent is None else ~absent
                    )
                    if dtype.kind in "fc":
                        present &= ~np.isnan(incoming)
                    conflict = (
                        present & ~missing[positions] & (values[positions] != incoming)
                    )
                    if np.any(conflict):
                        raise _SourceConflict(name)
                    kept = positions[present]
                    values[kept] = incoming[present]
                    # Rows of one source summed into one feature must agree;
                    # the assignment keeps only the last of them.
                    if np.any(values[kept] != incoming[present]):
                        raise _SourceConflict(name)
                    missing[kept] = False
                yield MetadataBlock(start=start, values=values, missing=missing)

        try:
            attributes = _merged_column_attributes(column_plan.arrays, name)
            column = create_streamed_metadata_column(
                destination,
                name,
                shape=n_features,
                dtype=dtype,
                blocks=blocks(),
                chunkSize=block_rows,
                hasMissing=True,
                profile=profile,
            )
        except _SourceConflict:
            clear_column(destination, name)
            skipped.append(name)
            continue
        column.attrs.update(attributes)
    if skipped:
        logger.warning(
            "Feature columns that differ between merged sources were not merged: "
            + ", ".join(skipped)
        )
    for name, vector in measured.items():
        values = np.asarray(vector, dtype=bool)
        assert values.shape == (n_features,), (name, values.shape, n_features)
        create_metadata_column(
            destination,
            measured_features_column(name),
            data=values,
            dtype=bool,
            chunkSize=column_chunk_rows(n_features, values.dtype.itemsize),
            profile=profile,
            attributes={"role": MEASURED_FEATURES_ROLE},
        )
    if taken:
        logger.warning(
            "Feature columns of the sources were not merged, since the merge "
            "records in each which features one source measured: " + ", ".join(taken)
        )


def _iter_destination_chunk_segments(
    row_plan: RowPlan,
    *,
    segment_rows: int,
    chunk_rows: int,
) -> Iterator[RowPlanSegment]:
    for segment in iter_row_plan_segments(row_plan):
        offset = 0
        while offset < segment.localRows.size:
            dest_start = segment.destStart + offset
            rows_to_boundary = chunk_rows - (dest_start % chunk_rows)
            width = min(
                segment_rows,
                rows_to_boundary,
                int(segment.localRows.size) - offset,
            )
            yield RowPlanSegment(
                sourceIdx=segment.sourceIdx,
                destStart=dest_start,
                localRows=segment.localRows[offset : offset + width],
            )
            offset += width


def merged_source_rows(
    row_plan: RowPlan, source_cell_tables: list[Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return each merged cell's ``sourceRow``: its row in the concatenation of its sources.

    The sources are concatenated in the merge's order, each with the cells
    that the merge takes from it in the order that an export of it writes
    them: by the source's own ``sourceRow``, or in its store order when it
    holds none. A merged cell's row is then its source's offset, the cells
    taken from the sources before it, plus its rank in that order. An
    export of the merged store in source order thus writes the cells as
    exports of its sources, one after the other, would.

    Returns:
        For each source, its selected rows in increasing order and the
        merged ``sourceRow`` of each.
    """
    lookups: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for index, table in enumerate(source_cell_tables):
        blocks = list(row_plan.permutationsRows[index].values())
        selected = (
            np.sort(np.concatenate(blocks)).astype(np.int64, copy=False)
            if blocks
            else np.zeros(0, dtype=np.int64)
        )
        del blocks
        column = table._get_array(SOURCE_ROW) if SOURCE_ROW in table.columns else None
        if is_source_row_column(column):
            values = read_metadata_rows_chunkwise(table, SOURCE_ROW, selected)
            ranks = source_order_ranks(values)
            del values
        else:
            ranks = np.arange(selected.size, dtype=np.int64)
        ranks += offset
        lookups.append((selected, ranks))
        offset += int(selected.size)
    return lookups


def _iter_column_blocks(
    spec: MetadataColumnSpec,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    source_columns: Sequence[dict[str, str]],
    reset_cell_filter: bool,
    source_column: str | None,
    membership: SourceMembership,
    segment_rows: int,
    chunk_rows: int,
) -> Iterator[MetadataBlock]:
    # Segments fill one destination chunk band at a time, so each band is
    # written with a single whole-chunk write instead of partial updates.
    band_start = 0
    band_values: np.ndarray | None = None
    band_missing: np.ndarray | None = None
    lookups = (
        merged_source_rows(row_plan, source_cell_tables)
        if spec.name == SOURCE_ROW
        else None
    )
    for segment in _iter_destination_chunk_segments(
        row_plan,
        segment_rows=segment_rows,
        chunk_rows=chunk_rows,
    ):
        source_idx = segment.sourceIdx
        local_rows = segment.localRows
        name = row_plan.sourceNames[source_idx]
        table = source_cell_tables[source_idx]
        n = int(local_rows.size)
        missing: np.ndarray | None = None

        if spec.name == "ids":
            source_values = read_metadata_rows_chunkwise(table, "ids", local_rows)
            values = prefixed_cell_ids(name, source_values, spec.dtype)
            del source_values
        elif spec.name == "names":
            source_values = read_metadata_rows_chunkwise(table, "names", local_rows)
            values = np.asarray(
                source_values,
                dtype=spec.dtype,
            )
            del source_values
        elif spec.name == "I":
            if reset_cell_filter:
                values = np.ones(n, dtype=bool)
            else:
                values = np.asarray(
                    read_metadata_rows_chunkwise(table, "I", local_rows),
                    dtype=bool,
                )
        elif lookups is not None:
            selected, merged = lookups[source_idx]
            values = merged[np.searchsorted(selected, local_rows)]
        elif spec.name == source_column:
            values = np.full(n, name, dtype=spec.dtype)
        elif spec.role == ASSAY_MEMBERSHIP_ROLE:
            assert spec.assay is not None
            state = membership[spec.assay][source_idx]
            if state == SOURCE_LACKS_ASSAY:
                values = np.zeros(n, dtype=bool)
            elif state == SOURCE_MEASURES_ALL:
                values = np.ones(n, dtype=bool)
            else:
                values = np.asarray(
                    read_metadata_rows_chunkwise(table, state, local_rows),
                    dtype=bool,
                )
        else:
            source_col = source_columns[source_idx].get(spec.name)
            if source_col is None:
                # The plan gives a column that a source lacks a missing mask,
                # which flags the source's rows.
                values = np.full(n, _fill_value(spec.dtype), dtype=spec.dtype)
                missing = np.ones(n, dtype=bool)
            else:
                raw = read_metadata_rows_chunkwise(table, source_col, local_rows)
                values = raw.astype(spec.dtype, copy=False)
                del raw
                if spec.hasMissing:
                    source_missing = read_metadata_missing_rows_chunkwise(
                        table,
                        source_col,
                        local_rows,
                    )
                    missing = (
                        np.zeros(n, dtype=bool)
                        if source_missing is None
                        else source_missing
                    )

        if band_values is None:
            band_rows = min(chunk_rows, row_plan.nCells - band_start)
            band_values = np.empty(band_rows, dtype=spec.dtype)
            band_missing = np.zeros(band_rows, dtype=bool) if spec.hasMissing else None
        offset = segment.destStart - band_start
        band_values[offset : offset + n] = values
        if band_missing is not None and missing is not None:
            band_missing[offset : offset + n] = missing
        del values, missing
        if offset + n == band_values.size:
            yield MetadataBlock(band_start, band_values, band_missing)
            band_start += band_values.size
            band_values = band_missing = None


def write_cell_metadata(
    root: zarr.Group,
    workspace: str | None,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    metadata_plan: CellMetadataPlan,
    *,
    profile: StorageProfile,
    reset_cell_filter: bool,
    source_column: str | None,
    membership: SourceMembership,
) -> zarr.Group:
    """Stream cell metadata columns in merged row order, replacing any partial slot."""
    cell_slot = _cell_data_path(workspace)
    if cell_slot in root:
        del root[cell_slot]
    group = root.create_group(cell_slot)
    segment_rows = effective_metadata_segment_rows(metadata_plan, row_plan)
    for spec in metadata_plan.columns:
        chunk_size = metadata_chunk_rows(row_plan, spec.dtype)
        blocks = _iter_column_blocks(
            spec,
            row_plan,
            source_cell_tables,
            source_columns=metadata_plan.sourceColumns,
            reset_cell_filter=reset_cell_filter,
            source_column=source_column,
            membership=membership,
            segment_rows=segment_rows,
            chunk_rows=chunk_size,
        )
        # The merge's own membership contract wins over source attributes.
        attributes = dict(spec.attributes)
        if spec.role is not None:
            attributes["role"] = spec.role
        if spec.assay is not None:
            attributes["assay"] = spec.assay
        create_streamed_metadata_column(
            group,
            spec.name,
            shape=row_plan.nCells,
            dtype=spec.dtype,
            blocks=blocks,
            overwrite=True,
            chunkSize=chunk_size,
            hasMissing=spec.hasMissing,
            profile=profile,
            attributes=attributes or None,
        )
    group.attrs["complete"] = True
    return group


def validate_cell_metadata(
    root: zarr.Group,
    workspace: str | None,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    metadata_plan: CellMetadataPlan,
    *,
    resources: ResourceBudget,
    resident_bytes: int,
) -> str | None:
    """Return why a completed cellData component cannot be reused."""
    cell_path = _cell_data_path(workspace)
    if cell_path not in root:
        return f"cell metadata group {cell_path!r} is missing"
    group = as_zarr_group(root[cell_path], name=cell_path)
    if group.attrs.get("complete") is not True:
        return f"cell metadata group {cell_path!r} is not complete"

    for spec in metadata_plan.columns:
        expected_chunks = (metadata_chunk_rows(row_plan, spec.dtype),)
        if spec.name not in group:
            return f"cell metadata column {spec.name!r} is missing"
        array = as_zarr_array(group[spec.name], name=f"{cell_path}/{spec.name}")
        if tuple(int(value) for value in array.shape) != (row_plan.nCells,):
            return f"cell metadata column {spec.name!r} has the wrong shape"
        if np.dtype(array.dtype) != spec.dtype:
            return (
                f"cell metadata column {spec.name!r} has dtype "
                f"{np.dtype(array.dtype)}, expected {spec.dtype}"
            )
        if tuple(int(value) for value in array.chunks) != expected_chunks:
            return f"cell metadata column {spec.name!r} has the wrong chunks"
        if spec.role is not None and array.attrs.get("role") != spec.role:
            return f"cell metadata column {spec.name!r} has the wrong role"
        if spec.assay is not None and array.attrs.get("assay") != spec.assay:
            return f"cell metadata column {spec.name!r} has the wrong assay"
        missing_name = f"{MISSING_MASK_PREFIX}{spec.name}"
        if spec.hasMissing:
            if array.attrs.get("missing_mask") != missing_name:
                return f"cell metadata column {spec.name!r} has no missing mask"
            if missing_name not in group:
                return f"missing mask for cell metadata column {spec.name!r} is missing"
            missing = as_zarr_array(
                group[missing_name],
                name=f"{cell_path}/{missing_name}",
            )
            if tuple(int(value) for value in missing.shape) != (row_plan.nCells,):
                return f"missing mask for cell metadata column {spec.name!r} has the wrong shape"
            if np.dtype(missing.dtype) != np.dtype(bool):
                return f"missing mask for cell metadata column {spec.name!r} has the wrong dtype"
            if tuple(int(value) for value in missing.chunks) != expected_chunks:
                return f"missing mask for cell metadata column {spec.name!r} has the wrong chunks"

    identity_rows = resolve_identity_validation_rows(
        metadata_plan,
        row_plan,
        group["ids"],
        resources,
        resident_bytes=resident_bytes,
    )
    try:
        verify_merged_cell_ids(
            as_zarr_array(group["ids"], name=f"{cell_path}/ids"),
            row_plan,
            source_cell_tables,
            block_rows=identity_rows,
        )
    except ValueError as error:
        return str(error)
    return None
