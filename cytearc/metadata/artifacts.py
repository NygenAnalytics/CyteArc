import colorsys
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ..storage.arrays import (
    MetadataBlock,
    _decode_metadata_values,
    _measured_text_dtype,
    create_streamed_metadata_column,
    create_zarr_dataset,
)
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
    planned_diagnostics,
    reused_artifact_group,
)
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_group,
    fingerprint_array,
    fingerprint_stored_arrays,
)
from ..storage.categorical import CategoricalColumn
from ..storage.diagnostics import DIAGNOSTICS_ATTRIBUTE
from ..storage.finite_values import requires_finite_values, write_finite_array
from ..storage.layout import column_chunk_rows
from ..storage.selections import validate_cell_selection
from ..storage.types import as_zarr_array, as_zarr_group
from .selection import (
    CellField,
    cell_value_spec,
    resolve_grouping,
    valid_category_mask,
)

_CATEGORY_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
)
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_TEXT_LABEL_KINDS = frozenset({"O", "S", "T", "U"})
_LABEL_KINDS = _TEXT_LABEL_KINDS | {"b", "i", "u"}
# int64 holds the whole numbers in [-_INT64_BOUND, _INT64_BOUND). A float64
# bound, unlike a Python float, compares with float16 labels without overflow.
_INT64_BOUND = np.float64(2.0**63)
# The rows of each stored chunk of an artifact's numeric arrays, unless
# chunk_lengths names others; text follows column_chunk_rows.
_CHUNK_ROWS = 100_000
# The pass that finds the widest value of a text array converts this many
# values to text at a time.
_MEASURED_ROWS = 4_096


def validate_display_metadata(
    display: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(display)
    kind = value.get("kind")
    if kind == "continuous":
        required = {"kind", "colormap", "minimum", "maximum", "scale"}
        if not required.issubset(value):
            raise ValueError("Continuous display metadata is incomplete")
        if set(value) != required:
            raise ValueError("Continuous display metadata has unknown fields")
        if not isinstance(value.get("colormap"), str):
            raise TypeError("Continuous display colormap must be a string")
        if value.get("scale") not in {"linear", "log", "symlog"}:
            raise ValueError("Continuous display scale is invalid")
        minimum = value.get("minimum")
        maximum = value.get("maximum")
        if minimum is not None and (
            isinstance(minimum, bool)
            or not isinstance(minimum, int | float)
            or not np.isfinite(float(minimum))
        ):
            raise TypeError("Continuous display minimum must be numeric or null")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int | float)
            or not np.isfinite(float(maximum))
        ):
            raise TypeError("Continuous display maximum must be numeric or null")
        if (
            minimum is not None
            and maximum is not None
            and float(minimum) > float(maximum)
        ):
            raise ValueError("Continuous display minimum exceeds maximum")
        return value
    if kind == "categorical":
        if not {"kind", "categories"}.issubset(value):
            raise ValueError("Categorical display metadata is incomplete")
        allowed = {
            "kind",
            "categories",
            "missing_label",
            "missing_color",
        }
        if not set(value).issubset(allowed):
            raise ValueError("Categorical display metadata has unknown fields")
        categories = value.get("categories")
        if not isinstance(categories, list):
            raise TypeError("Categorical display categories must be a list")
        seen_values: set[tuple[str, Any]] = set()
        mapping_values: list[Any] = []
        for category in categories:
            if not isinstance(category, dict):
                raise TypeError("Each display category must be a mapping")
            if set(category) != {"value", "label", "color"}:
                raise ValueError(
                    "Each display category requires value, label, and color"
                )
            category_value = category["value"]
            if category_value is None or (
                not isinstance(
                    category_value,
                    bool | int | float | str,
                )
                or (
                    isinstance(category_value, float)
                    and not np.isfinite(category_value)
                )
            ):
                raise TypeError("Display category value must be a JSON scalar")
            typed_value = (
                type(category_value).__name__,
                category_value,
            )
            if typed_value in seen_values:
                raise ValueError("Display category values must be unique")
            if any(category_value == existing for existing in mapping_values):
                raise ValueError("Display category values collide as mapping keys")
            seen_values.add(typed_value)
            mapping_values.append(category_value)
            if not isinstance(category.get("label"), str):
                raise TypeError("Display category label must be a string")
            color = category.get("color")
            if not isinstance(color, str) or _HEX_COLOR.fullmatch(color) is None:
                raise ValueError("Display category color must be a hex color")
        missing_label = value.get("missing_label")
        if missing_label is not None and not isinstance(missing_label, str):
            raise TypeError("Categorical missing_label must be a string")
        missing_color = value.get("missing_color")
        if missing_color is not None and (
            not isinstance(missing_color, str)
            or _HEX_COLOR.fullmatch(missing_color) is None
        ):
            raise ValueError("Categorical missing_color must be a hex color")
        return value
    raise ValueError("Display kind must be continuous or categorical")


def plan_cell_data_artifact(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    execution_options: dict[str, Any],
    cell_selection: ArtifactRef,
    arrays: Mapping[str, tuple[tuple[int | None, ...], str | None]],
    assay: str | None = None,
    invalidate_cache: bool = False,
    required_attributes: tuple[str | AttributeRequirement, ...] = (),
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
) -> PlannedArtifact:
    """Plan an artifact of arrays with one row per selected cell.

    A later dimension of an array's shape may be None when its size is known
    only once the values are computed, such as the width of arrays padded to
    the largest number of entries of one cell; reuse then accepts any size.
    """
    if cell_selection.kind != "cell_selection":
        raise ValueError("cell_selection must reference a cell-selection artifact")
    selected_count = int(validate_cell_selection(root, cell_selection).selected_count)
    if any(shape[0] != selected_count for shape, _kind in arrays.values()):
        raise ValueError("Artifact arrays must align with the selected cell count")
    artifact_inputs = {**inputs, "cell_selection": cell_selection}
    requirements = tuple(
        ArrayRequirement(name, shape=shape, dtype_kind=dtype_kind)
        for name, (shape, dtype_kind) in arrays.items()
    )
    return plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters=parameters,
        inputs=artifact_inputs,
        execution_options=execution_options,
        invalidate_cache=invalidate_cache,
        required_arrays=requirements,
        required_attributes=required_attributes,
        reuse_validator=reuse_validator,
    )


def write_cell_data_artifact(
    root: zarr.Group,
    planned: PlannedArtifact,
    arrays: Mapping[str, np.ndarray],
    *,
    fingerprint_payload: bool = False,
    attributes: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    chunk_lengths: Mapping[str, int] | None = None,
) -> zarr.Group:
    """Write and publish a planned cell-data artifact, or return the reused one.

    ``attributes`` are JSON values stored on the artifact group before it is
    published. ``diagnostics`` are those that the operation declares, checked
    before anything is written and saved in the artifact's ``diagnostics``
    attribute; an operation that declares none passes None. Numeric arrays
    are chunked in 100,000 rows; ``chunk_lengths`` sets the rows of each chunk
    of named numeric arrays instead, such as packed bytes, which read and
    write faster in chunks of megabytes. Text is stored as
    ``values.astype(str)`` holds it, at the width of the widest value, in
    chunks of at most 100,000 rows and 16 MiB of values, as
    :func:`~cytearc.storage.layout.column_chunk_rows` chunks cell columns,
    and is written a chunk at a time, so a write holds what
    :func:`cell_data_write_bytes` states rather than copies of whole arrays.
    """
    if planned.reused:
        return reused_artifact_group(root, planned)
    kind = planned.ref.kind
    operation = str(planned.provenance["operation"])
    if attributes is not None and DIAGNOSTICS_ATTRIBUTE in attributes:
        raise ValueError(
            f"{operation} passes its diagnostics as an attribute; pass them as "
            "diagnostics, which checks them against those it declares"
        )
    saved = planned_diagnostics(planned, diagnostics)
    stored = {
        **(attributes or {}),
        **({} if saved is None else {DIAGNOSTICS_ATTRIBUTE: saved}),
    }
    with artifact_transaction(root, planned) as group:
        if stored:
            group.attrs.update(stored)
        for name, raw_values in arrays.items():
            values = np.asarray(raw_values)
            if values.ndim < 1:
                raise ValueError("Artifact arrays must have at least one dimension")
            if values.dtype.kind in {"O", "S", "U"}:
                if values.ndim != 1:
                    raise ValueError("String artifact arrays must be one-dimensional")
                _write_text_array(group, name, values)
            else:
                chunks = (
                    _chunk_rows(
                        int(values.shape[0]), _chunk_length(name, chunk_lengths)
                    ),
                    *values.shape[1:],
                )
                output = create_zarr_dataset(
                    group,
                    name,
                    chunks,
                    values.dtype,
                    values.shape,
                )
                if requires_finite_values(kind, name, output.dtype):
                    write_finite_array(output, values, operation=operation)
                else:
                    output[...] = values
        if fingerprint_payload:
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                tuple(arrays),
            )
    return group


def cell_data_write_bytes(
    arrays: Mapping[str, tuple[tuple[int, ...], Any]],
    *,
    chunk_lengths: Mapping[str, int] | None = None,
) -> int:
    """The most bytes that :func:`write_cell_data_artifact` holds to write these arrays, beside them.

    Each array is given as its shape and the dtype it is stored at; a text
    array as the fixed-width text dtype of its widest value, such as
    ``np.dtype("U12")`` for labels of up to 12 characters, whatever dtype the
    caller holds it in. ``chunk_lengths`` is the writer's. Arrays are written
    one after another, in the order given, so the write holds what its
    costliest array holds:

    - text, a stored chunk of at most 16 MiB of values at a time: the chunk's
      values at the stored width,
      four bytes a character, and the chunk that Zarr encodes from them, a
      last chunk of fewer rows also copied into a full chunk first;
    - numbers, written whole: each chunk that Zarr encodes at once, up to its
      async concurrency of them, and a last chunk of fewer rows twice.

    Attributes that grow with the data, and a payload fingerprint, which
    reads the arrays back after they are written, are left to the caller.

    Raises:
        TypeError: If ``arrays`` is not a mapping of shapes and dtypes.
        ValueError: If a size is not a non-negative integer, the shape is
            empty, or a text array is not one-dimensional or is not given at
            a fixed width of at least one character.
    """
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must map each array's name to its shape and dtype")
    costliest = 0
    for name, spec in arrays.items():
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise TypeError(
                f"Array {name!r} must be given as a (shape, dtype) pair, not {spec!r}"
            )
        shape, raw_dtype = spec
        dtype = np.dtype(raw_dtype)
        if (
            not isinstance(shape, tuple)
            or not shape
            or not all(
                isinstance(size, int | np.integer)
                and not isinstance(size, bool)
                and size >= 0
                for size in shape
            )
        ):
            raise ValueError(
                f"Array {name!r} needs a shape of one or more non-negative sizes, "
                f"not {shape!r}"
            )
        if dtype.kind in {"O", "S"} or (dtype.kind == "U" and not dtype.itemsize):
            raise ValueError(
                f"Text array {name!r} must be given at the fixed width of its "
                f"widest value, such as U12, not {dtype}"
            )
        if dtype.kind == "U" and len(shape) != 1:
            raise ValueError(f"Text array {name!r} must be one-dimensional")
        costliest = max(
            costliest,
            _array_write_bytes(
                tuple(int(size) for size in shape),
                dtype,
                _chunk_length(name, chunk_lengths),
            ),
        )
    return costliest


def _chunk_length(name: str, chunk_lengths: Mapping[str, int] | None) -> int:
    """The rows of a chunk of numeric array ``name``, as the writer chunks it."""
    if chunk_lengths is None:
        return _CHUNK_ROWS
    return int(chunk_lengths.get(name, _CHUNK_ROWS))


def _chunk_rows(n_rows: int, length: int) -> int:
    """The rows of each stored chunk of an array of ``n_rows``: at most ``length``."""
    return min(max(n_rows, 1), max(length, 1))


def _array_write_bytes(
    shape: tuple[int, ...], dtype: np.dtype[Any], length: int
) -> int:
    """What writing one array holds beside it, as :func:`cell_data_write_bytes` counts."""
    n_rows = shape[0]
    if dtype.kind == "U":
        rows = column_chunk_rows(n_rows, dtype.itemsize)
        # A chunk and its encoded copy. Measuring the widest value converts
        # 4,096 values at a time beside their lengths, which holds less.
        partial = n_rows % rows if n_rows > rows else 0
        return 2 * dtype.itemsize * rows + dtype.itemsize * partial
    rows = _chunk_rows(n_rows, length)
    chunk = rows * math.prod(shape[1:]) * dtype.itemsize
    n_chunks = -(-n_rows // rows)
    at_once = min(n_chunks, int(zarr.config.get("async.concurrency")))
    partial = 1 if n_rows > rows and n_rows % rows else 0
    return (at_once + partial) * chunk


def _write_text_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    """Write text values a stored chunk at a time, after one pass that finds the widest."""
    n_rows = len(values)
    dtype = _stored_text_dtype(values)
    rows = column_chunk_rows(n_rows, dtype.itemsize)
    create_streamed_metadata_column(
        group,
        name,
        shape=n_rows,
        dtype=dtype,
        chunkSize=rows,
        blocks=(
            MetadataBlock(start, _as_text(values[start : start + rows], dtype))
            for start in range(0, n_rows, rows)
        ),
    )


def _stored_text_dtype(values: np.ndarray) -> np.dtype[Any]:
    """The fixed-width text dtype at which ``values.astype(str)`` holds the widest value.

    Byte strings and objects are converted as ``astype(str)`` converts them, a
    few thousand at a time, so that measuring them holds little beside them.
    """
    width = max(
        (
            _widest_text_length(values[start : start + _MEASURED_ROWS])
            for start in range(0, len(values), _MEASURED_ROWS)
        ),
        default=0,
    )
    return np.dtype(f"U{max(1, width)}")


def _widest_text_length(values: np.ndarray) -> int:
    """The characters of the widest of ``values`` as text."""
    if values.dtype.kind == "U":
        return int(np.char.str_len(values).max())
    if values.dtype.kind == "S":
        # Byte strings decode as ASCII, one character a byte, which raises for
        # any byte that is not ASCII.
        width = max(1, int(np.char.str_len(values).max()))
        return int(np.char.str_len(_as_text(values, np.dtype(f"U{width}"))).max())
    return int(np.char.str_len(values.astype(str)).max())


def _as_text(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """``values`` as fixed-width text of ``dtype``, at least as wide as each value.

    Byte strings are first cut to that width, which loses no byte of a value,
    since NumPy's cast of byte strings to text holds a buffer that grows with
    their declared width.
    """
    if values.dtype.kind == "S":
        values = values.astype(f"S{dtype.itemsize // 4}")
    return values.astype(dtype)


def snapshot_cluster_labels(
    root: zarr.Group,
    cells: Any,
    labels: str | ArtifactRef,
    *,
    cell_selection: ArtifactRef,
) -> ArtifactRef:
    """Freeze one label per selected cell as a ``cluster_labels`` artifact.

    ``labels`` names a column of ``cells`` or references a cell-label
    artifact, whose own cell selection must contain ``cell_selection``.
    Labels must be text, integer, or boolean values, or floating-point whole
    numbers within the int64 range, such as the float64 ids that pandas
    writes for integer ids with missing values. Other floating-point labels
    are rejected whether or not they hold NaN, and every selected cell needs
    a label that is not missing, masked, or blank. Text is stored at the
    width of the selected labels, so the declared width and encoding of the
    source do not change the identity; floating-point whole numbers are
    stored as int64, and integer and boolean labels keep their dtype. The
    identity holds the source column name or source artifact, the cell
    selection, and a fingerprint of the stored labels, and a match is
    reused only while its payload is exactly those labels.

    Raises:
        TypeError: If ``labels`` or ``cell_selection`` has the wrong type, or
            if a label is not a text, integer, boolean, or whole-number
            value.
        ValueError: If a selected cell has no label.
        PermissionError: If no snapshot matches and ``root`` is read-only.
    """
    source: ArtifactRef | CellField
    parameters: dict[str, Any]
    inputs: dict[str, Any]
    if isinstance(labels, str):
        source = CellField(labels, kind="categorical")
        described = f"column {labels!r}"
        example = "snapshot_cell_selection of a boolean column that marks them"
        parameters, inputs = {"source_column": labels}, {}
    elif isinstance(labels, ArtifactRef):
        source = labels
        described = f"the {labels.kind} artifact"
        example = "select_cells(labels, include=[...])"
        parameters, inputs = {}, {"source_labels": labels}
    else:
        raise TypeError(
            "labels must be a cell metadata column name or a cell-label ArtifactRef"
        )
    if not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    resolved = resolve_grouping(root, cells, source, cell_selection=cell_selection)
    valid = valid_category_mask(resolved.labels, missing_mask=resolved.missing_mask)
    kind = resolved.labels.dtype.kind
    if kind == "f":
        # Checked before completeness but over labeled cells only, so neither
        # NaN nor a masked placeholder decides whether the floats are ids.
        labeled = resolved.labels[valid]
        whole = (labeled >= -_INT64_BOUND) & (labeled < _INT64_BOUND)
        whole &= labeled == np.trunc(labeled)
        if not whole.all():
            raise TypeError(
                "Floating-point cluster labels must be whole numbers within the "
                f"int64 range, but {described} holds {labeled[~whole][0]!s}"
            )
    elif kind not in _LABEL_KINDS:
        raise TypeError(
            "Cluster labels must be text, integer, boolean, or whole-number "
            f"values, but {described} holds {resolved.labels.dtype} values"
        )
    unlabeled = int(np.count_nonzero(~valid))
    if unlabeled:
        raise ValueError(
            f"{unlabeled} of {valid.size} selected cells have no label in "
            f"{described}. Pass a cell_selection of labeled cells only, such "
            f"as {example}"
        )
    values = _decode_metadata_values(resolved.labels)
    if kind == "f":
        values = values.astype(np.int64)
    elif kind in _TEXT_LABEL_KINDS:
        values = values.astype(_measured_text_dtype((values,)))
    fingerprint = fingerprint_array(values)
    inputs["values_fingerprint"] = fingerprint

    def payload_matches(_ref: ArtifactRef, group: zarr.Group) -> bool:
        # Reuse and completion checks call this only after the values array
        # matched the planned shape and dtype kind.
        return (
            not set(group.group_keys())
            and set(group.array_keys()) == {"values"}
            and fingerprint_stored_arrays(group, ("values",)) == fingerprint
        )

    planned = plan_cell_data_artifact(
        root,
        scope="datastore",
        kind="cluster_labels",
        operation="snapshot_cluster_labels",
        parameters=parameters,
        inputs=inputs,
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (values.shape, values.dtype.kind)},
        reuse_validator=payload_matches,
    )
    write_cell_data_artifact(root, planned, {"values": values})
    return planned.ref


def artifact_values(
    group: zarr.Group,
    name: str,
    value_index: int | None = None,
) -> np.ndarray:
    values = np.asarray(as_zarr_array(group[name], name=name)[:])
    return values if value_index is None else values[:, value_index]


@dataclass(frozen=True, slots=True)
class RecordedCategories:
    """The categories that a stored array of labels records in its attributes.

    An import keeps a source categorical with the column or label artifact
    that holds its labels, in the attributes that :func:`recorded_categories`
    reads.

    Attributes:
        levels: Every category in its source order, unused ones included, or
            None when the array records none.
        ordered: Whether the levels are ordered.
        display: The validated display contract, or None.
    """

    levels: tuple[Any, ...] | None
    ordered: bool
    display: dict[str, Any] | None

    @property
    def ordered_levels(self) -> tuple[Any, ...] | None:
        """The levels when they are ordered, or None."""
        return self.levels if self.ordered else None


def recorded_categories(attributes: Mapping[str, Any]) -> RecordedCategories:
    """Read the ``levels``, ``ordered``, and ``display`` attributes of a stored array.

    Raises:
        TypeError: If ``levels`` is not a list or ``display`` not a mapping.
        ValueError: If ``display`` is not a valid display contract.
    """
    levels = attributes.get("levels")
    if levels is not None and not isinstance(levels, list | tuple):
        raise TypeError(
            f"Stored levels must be a list of categories, not {type(levels).__name__}"
        )
    display = attributes.get("display")
    if display is not None and not isinstance(display, Mapping):
        raise TypeError("Existing display metadata must be a mapping")
    return RecordedCategories(
        levels=None if levels is None else tuple(levels),
        ordered=attributes.get("ordered") is True,
        display=None if display is None else validate_display_metadata(display),
    )


def artifact_categories(root: zarr.Group, ref: ArtifactRef) -> RecordedCategories:
    """Return the categories that a cell-aligned artifact records with its values.

    An H5AD import keeps the categories, order, and colors of each clustering
    it imports on the artifact's value array; other artifacts record none.

    Raises:
        ValueError: If ``ref`` is not a cell-aligned artifact kind.
    """
    name = cell_value_spec(ref.kind).name
    group = artifact_group(root, ref)
    return recorded_categories(as_zarr_array(group[name], name=name).attrs)


def column_display(
    root: zarr.Group,
    column: str,
) -> dict[str, Any] | None:
    cell_data = as_zarr_group(root["cellData"], name="cellData")
    if column not in cell_data:
        return None
    return recorded_categories(
        as_zarr_array(cell_data[column], name=column).attrs
    ).display


def column_category_order(
    root: zarr.Group,
    column: str,
) -> tuple[Any, ...] | None:
    """Return the categories of a cell column stored as ordered, in their stored order.

    A categorical column is ordered when its categories array records
    ``ordered=True``. A column of labels is ordered when its ``levels``
    attribute, the categories of the source categorical that an import
    keeps, comes with an ``ordered`` attribute of True. Any other column, or
    a column the table does not hold, returns None.

    Raises:
        TypeError: If a column of labels records ``levels`` that are not a list.
    """
    cell_data = as_zarr_group(root["cellData"], name="cellData")
    if column not in cell_data:
        return None
    array = as_zarr_array(cell_data[column], name=column)
    categorical = CategoricalColumn.open(cell_data, column, values=array)
    if categorical is not None:
        return tuple(categorical.categories.tolist()) if categorical.ordered else None
    return recorded_categories(array.attrs).ordered_levels


def continuous_display(
    values: np.ndarray,
    *,
    colormap: str = "viridis",
    scale: str = "linear",
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    minimum = float(finite.min()) if len(finite) else None
    maximum = float(finite.max()) if len(finite) else None
    return {
        "kind": "continuous",
        "colormap": colormap,
        "minimum": minimum,
        "maximum": maximum,
        "scale": scale,
    }


def _category_color(index: int, count: int) -> str:
    if count <= len(_CATEGORY_COLORS):
        return _CATEGORY_COLORS[index]
    red, green, blue = colorsys.hsv_to_rgb(
        (index * 0.618033988749895) % 1.0,
        0.55,
        0.85,
    )
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def categorical_display(values: np.ndarray) -> dict[str, Any]:
    categories: list[Any] = []
    seen: set[tuple[str, str]] = set()
    has_missing = False
    for raw_value in np.asarray(values).reshape(-1):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, float) and np.isnan(value):
            value = None
        if value is None:
            has_missing = True
            continue
        key = (type(value).__name__, repr(value))
        if key not in seen:
            seen.add(key)
            categories.append(value)
    try:
        categories = sorted(categories)
    except TypeError:
        pass
    count = len(categories)
    display = {
        "kind": "categorical",
        "categories": [
            {
                "value": value,
                "label": "NA" if value is None else str(value),
                "color": _category_color(index, count),
            }
            for index, value in enumerate(categories)
        ],
    }
    if has_missing:
        display["missing_label"] = "NA"
        display["missing_color"] = "#bdbdbd"
    return display


def levels_display(
    levels: list[Any],
    colors: tuple[str, ...] | None,
    has_missing: bool,
) -> dict[str, Any] | None:
    """Return the categorical display contract of levels in their order, or None.

    The categories keep the order of ``levels``, so plots show them in it.
    Colors come from ``colors``, one per level, or else from the palette of
    :func:`categorical_display`, so each category takes the color that the
    natural order of the levels gives it. Levels that a display cannot hold,
    such as values that are not JSON scalars, give None.
    """
    if colors is None:
        generated = {
            category["value"]: category["color"]
            for category in categorical_display(np.asarray(levels, dtype=object))[
                "categories"
            ]
        }
        colors = tuple(generated[level] for level in levels)
    display: dict[str, Any] = {
        "kind": "categorical",
        "categories": [
            {"value": level, "label": str(level), "color": color}
            for level, color in zip(levels, colors, strict=True)
        ],
    }
    if has_missing:
        display.update({"missing_label": "NA", "missing_color": "#bdbdbd"})
    try:
        return validate_display_metadata(display)
    except (TypeError, ValueError):
        return None
