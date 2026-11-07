from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cache, partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zarr

from ..metadata.encoding import missing_rows, stored_values
from ..storage.arrays import (
    MISSING_MASK_PREFIX,
    create_zarr_dataset as _create_zarr_dataset,
    create_zarr_obj_array as _create_zarr_obj_array,
)
from ..storage.schema import (
    create_zarr_count_assay as _create_zarr_count_assay,
)
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.layout import column_chunk_rows
from ..storage.metadata_keys import (
    is_reserved_metadata_name,
    metadata_column_key,
    validate_metadata_column_name,
)
from ..storage.profiles import StorageProfile
from ..storage.refs import ArtifactRef
from ..utils.count_values import CountValueRange
from ..utils.logging import logger
from ..utils.matrix_values import (
    MatrixValues,
    decide_matrix_values,
    import_values_argument,
    linear_values,
)
from ..utils.arguments import bool_argument

if TYPE_CHECKING:
    from ..assay import Assay

DEFAULT_IMPORT_BLOCK_ROWS = 65_536


def keyed_metadata_columns[T](
    columns: Iterable[tuple[str, T]],
    keys: Mapping[str, str],
    axis: str,
    *,
    membership: Mapping[str, str] | None = None,
) -> Iterator[tuple[str, T]]:
    """Yield source metadata columns under the keys planned for them.

    Plan ``keys`` with :func:`~cytearc.storage.metadata_keys.metadata_column_keys`
    over the source column names. Columns stream one at a time, so a caller
    holds a single column payload in memory. A source name that the plan
    leaves out is skipped with a warning: it is reserved by CyteArc, cannot name
    a Zarr array, or is already a column of the destination. A repeated source
    name is skipped after its first column. CyteArc keeps no record of a renamed
    column's source name; the warning is the only report of it.

    Args:
        columns: Pairs of source column name and payload, in source order.
        keys: Destination key of each source name to write.
        axis: Table description used in warnings, such as ``cell``.
        membership: Assay of each reserved membership column, used in warnings.

    Yields:
        Pairs of destination key and payload.
    """
    seen: set[str] = set()
    for name, payload in columns:
        if name in seen:
            logger.warning(
                f"Skipped source {axis} metadata column {name!r} because the "
                "source repeats that column name"
            )
            continue
        seen.add(name)
        key = keys.get(name)
        if key is None:
            logger.warning(_skipped_column_reason(name, axis, membership or {}))
            continue
        if key != name:
            base = metadata_column_key(name)
            clash = f", and {base!r} is already used" if key != base else ""
            logger.warning(
                f"Stored source {axis} metadata column {name!r} as {key!r} "
                f"because Zarr reads '/' and '\\' as path separators{clash}"
            )
        yield key, payload


def _skipped_column_reason(name: str, axis: str, membership: Mapping[str, str]) -> str:
    if name in membership:
        return (
            f"Skipped source {axis} metadata column {name!r} because CyteArc "
            "reserves it for the membership of the imported assay "
            f"{membership[name]!r}"
        )
    if is_reserved_metadata_name(metadata_column_key(name)):
        return (
            f"Skipped source {axis} metadata column {name!r} because CyteArc "
            "reserves the column names 'I', 'ids', and 'names' and the "
            f"prefix {MISSING_MASK_PREFIX!r}"
        )
    if name in {"", ".", ".."}:
        return (
            f"Skipped source {axis} metadata column {name!r} because that name "
            "cannot name a Zarr array"
        )
    return (
        f"Skipped source {axis} metadata column {name!r} because the "
        f"destination already has a column named {name!r}"
    )


def decode_text(value: Any) -> str:
    """Return a string value, decoding byte strings as UTF-8."""
    if isinstance(value, bytes | np.bytes_):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text values must be valid UTF-8") from exc
    if isinstance(value, str | np.str_):
        return str(value)
    raise TypeError(f"Expected text, found {type(value).__name__}")


def write_metadata_column(
    group: zarr.Group,
    name: str,
    values: Any,
    missing: Any = None,
    *,
    profile: StorageProfile | None = None,
) -> None:
    """Write one metadata column and link a missing-value mask when needed.

    The column and its mask are chunked by
    :func:`~cytearc.storage.layout.column_chunk_rows`, in a cell or a feature
    table.

    Args:
        group: Destination metadata group.
        name: Column name.
        values: One value per row.
        missing: Rows to flag as missing. When None, ``None``, NaN, ``pd.NA``,
            and NaT entries of an object array are missing.
        profile: Zarr encoding profile. When None, chosen from the store.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` cannot name one Zarr array, or the values are
            not one-dimensional or the mask does not align with them.
    """
    from ..storage.arrays import (
        MetadataBlock,
        create_streamed_metadata_column,
        encode_metadata_values,
    )

    validate_metadata_column_name(name)
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"Metadata column {name!r} must hold one value per row; "
            f"found shape {array.shape}"
        )
    mask = missing_rows(array) if missing is None else np.asarray(missing, dtype=bool)
    if mask.shape != array.shape:
        raise ValueError(f"Metadata column {name!r} has a misaligned missing mask")
    has_missing = bool(mask.any())
    stored = (
        stored_values(array, mask, name=name)
        if has_missing or array.dtype.kind == "O"
        else array
    )
    # The stored width sets the chunk, so byte strings are decoded first; the
    # write then copies them no further.
    stored = encode_metadata_values(stored, stored.dtype, name=name)
    rows = column_chunk_rows(int(array.size), max(1, stored.dtype.itemsize))
    if not has_missing:
        _create_zarr_obj_array(
            group, name, stored, stored.dtype, chunk_size=rows, profile=profile
        )
        return
    create_streamed_metadata_column(
        group,
        name,
        shape=int(array.size),
        dtype=stored.dtype,
        blocks=(MetadataBlock(0, stored, mask),),
        chunkSize=rows,
        hasMissing=True,
        profile=profile,
    )


def write_membership_column(
    group: zarr.Group,
    assay: str,
    values: np.ndarray,
    *,
    profile: StorageProfile | None = None,
) -> None:
    """Write the cell column that records which cells ``assay`` measured."""
    from ..metadata.membership import membership_attributes
    from ..storage.arrays import create_metadata_column
    from ..storage.metadata_keys import assay_membership_column

    name = assay_membership_column(assay)
    validate_metadata_column_name(name)
    members = np.asarray(values, dtype=bool)
    if members.ndim != 1:
        raise ValueError(
            f"Membership column {name!r} must hold one value per cell; "
            f"found shape {members.shape}"
        )
    create_metadata_column(
        group,
        name,
        data=members,
        dtype=bool,
        chunkSize=column_chunk_rows(len(members), members.dtype.itemsize),
        profile=profile,
        attributes=membership_attributes(assay),
    )


def floating_payload_dtype(dtype: Any, label: str) -> np.dtype[Any]:
    """Return the floating dtype that stores an imported numeric payload."""
    source: np.dtype[Any] = np.dtype(dtype)
    if source.kind == "f":
        return np.dtype(source.str)
    if source.kind in "biu":
        return np.dtype(np.float64)
    raise TypeError(f"{label} uses unsupported dtype {source}")


def bounded_block_rows(
    requested: int | None,
    *,
    row_bytes: int,
    memory_bytes: int,
) -> int:
    """Return rows per import block within an eighth of the memory budget."""
    bytes_per_row = max(1, int(row_bytes))
    memory_rows = max(1, int(memory_bytes) // (8 * bytes_per_row))
    preferred = DEFAULT_IMPORT_BLOCK_ROWS if requested is None else requested
    return int(max(1, min(int(preferred), memory_rows)))


def fingerprint_row_blocks(
    blocks: Iterable[np.ndarray],
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    label: str,
) -> str:
    """Fingerprint row blocks of one payload, rejecting non-finite numbers."""
    from ..storage.artifacts import ValueFingerprintBuilder

    builder = ValueFingerprintBuilder()
    builder.begin_array("values", shape, dtype)
    start = 0
    for block in blocks:
        if block.dtype.kind in "fc" and not bool(np.isfinite(block).all()):
            raise ValueError(f"{label} contains non-finite values")
        builder.update_array_block(
            "values",
            (start, *(0,) * (block.ndim - 1)),
            block,
        )
        start += int(block.shape[0])
    builder.end_array("values")
    return str(builder.hexdigest())


def resolve_import_cell_selection(
    root: zarr.Group,
    *,
    source: str,
    inputs: Mapping[str, Any],
) -> ArtifactRef:
    """Return the artifact that selects every imported cell."""
    from ..storage.selections import resolve_stored_selection_artifact

    return resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="import_cell_selection",
        parameters={"source": source},
        inputs=dict(inputs),
    )


def create_zarr_dataset(
    g: zarr.Group,
    name: str,
    chunks: tuple[int, ...] | int,
    dtype: Any,
    shape: tuple[int, ...],
    overwrite: bool = True,
) -> zarr.Array:
    """Creates and returns a Zarr array.

    Args:
        g: Parent Zarr group.
        name: Array name within the group.
        chunks: Chunk shape, or a single integer applied to every axis.
        dtype: NumPy dtype for the array.
        shape: Array shape.
        overwrite: If True, replace an existing array of the same name.

    Returns:
        A Zarr Array.
    """
    overwrite = bool_argument(overwrite, "overwrite")
    return _create_zarr_dataset(g, name, chunks, dtype, shape, overwrite)


def create_zarr_obj_array(
    g: zarr.Group,
    name: str,
    data: Any,
    dtype: str | Any = None,
    overwrite: bool = True,
    chunk_size: int = 100000,
    shape: int | None = None,
) -> zarr.Array:
    """Creates and returns a metadata column array.

    Args:
        g: Parent Zarr group.
        name: Array name within the group.
        data: Values to write, or None to create an empty array.
        dtype: Optional dtype. Inferred from ``data`` when omitted.
        overwrite: If True, replace an existing array of the same name.
        chunk_size: Chunk length along the first axis.
        shape: Explicit length when ``data`` is None.

    Returns:
        A Zarr Array.
    """
    overwrite = bool_argument(overwrite, "overwrite")
    return _create_zarr_obj_array(
        g,
        name,
        data,
        dtype,
        overwrite,
        chunk_size,
        shape,
    )


def create_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: Any,
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
    normalized_values: Mapping[str, Any] | None = None,
) -> zarr.Array:
    """Creates and returns a Zarr array with name 'counts'.

    Args:
        z: Root Zarr group.
        assay_name: Assay group that will own the counts array.
        workspace: Workspace name. None uses the legacy layout.
        n_cells: Number of cells (rows).
        feat_ids: Feature identifiers written to feature metadata.
        feat_names: Feature display names written to feature metadata.
        dtype: Storage dtype for counts. Import writers store the dtype that
               :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves
               from the canonical values.
        profile: Zarr encoding profile. When None, chosen from the store.
        policy: Count-matrix geometry policy. When None, the default plan
                is used.
        normalized_values: The ``normalizedValues`` record, such as
            ``{"importedAs": "normalized", "cellTotal": None}``, of an assay
            whose values were normalized before import, so that steps that
            need counts refuse it. None declares counts. Log values must be
            written linear, with the log undone.

    Returns:
        The created ``counts`` array.
    """
    return _create_zarr_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        feat_ids,
        feat_names,
        dtype,
        profile=profile,
        policy=policy,
        normalized_values=normalized_values,
    )


def resolve_import_values(
    values: object,
    value_ranges: Mapping[str, CountValueRange],
    sample: Callable[[], Mapping[str, Any]],
    assay_types: Mapping[str, str | None],
    dtypes: Mapping[str, Any],
) -> dict[str, MatrixValues]:
    """Return what each imported assay holds, before the destination exists.

    ``sample`` returns the first cells of each assay, and is called at most
    once. ``assay_types`` holds each assay's declared preset, or None, and
    ``dtypes`` each assay's storage dtype.

    Raises:
        TypeError: If an assay whose class needs counts holds normalized values.
        ValueError: As :func:`~cytearc.utils.matrix_values.decide_matrix_values`,
            or if undoing a log would overflow the storage dtype.
    """
    from ..assay.classification import preset_assay_types, resolve_persisted_assay_type

    declared = import_values_argument(values, list(value_ranges))
    first_cells = cache(sample)

    def assay_sample(name: str) -> Any:
        return first_cells()[name]

    decisions: dict[str, MatrixValues] = {}
    for name, value_range in value_ranges.items():
        decision = decide_matrix_values(
            declared.get(name) if isinstance(declared, dict) else declared,
            value_range,
            partial(assay_sample, name),
            assay=name,
        )
        assay_class = cast(
            "type[Assay]",
            preset_assay_types()[
                resolve_persisted_assay_type(name, assay_types.get(name))
            ],
        )
        if decision.kind != "counts" and assay_class.requiresCounts:
            raise TypeError(assay_class.normalized_values_refusal(name))
        # The largest value fails here, before the destination exists, if
        # undoing its log overflows the storage dtype.
        linear_values(
            np.array([value_range.largest]), decision.kind, dtypes[name], assay=name
        )
        decisions[name] = decision
    return decisions
