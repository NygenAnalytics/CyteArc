import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import zarr
from zarr.errors import ContainsArrayError

from .types import as_zarr_array, as_zarr_group, read_fresh_group
from .arrays import (
    MetadataBlock,
    column_write_rows,
    create_metadata_column,
    create_streamed_metadata_column,
    create_zarr_obj_array,
    encode_metadata_values,
)
from .count_matrix import CountMatrixPolicy, create_product_counts_array
from .layout import _group_zarr_format, column_chunk_rows
from .metadata_keys import (
    ASSAY_MEMBERSHIP_ROLE,
    assay_membership_attributes,
    assay_membership_column,
)
from .profiles import StorageProfile, resolve_storage_profile
from ..utils.logging import logger
from ..utils.matrix_values import MATRIX_VALUES, MatrixValues
from ..utils.shutdown import is_interruption

_ASSAY_NAME_OWNERS = {
    "artifacts": "datastore artifact storage",
    "cellData": "datastore cell metadata",
    "matrices": "workspace matrix storage",
    "pipeline": "DataStore.pipeline",
    "plots": "DataStore.plots",
    "summary": "DataStore.summary",
}
RESERVED_ASSAY_NAMES = frozenset(_ASSAY_NAME_OWNERS)
# Marks a derived assay whose counts are still being written. The group has no
# ``is_assay`` marker until the write is complete, so assay scans skip it.
PENDING_ASSAY_ATTR = "cytearc:pending_assay"
# The assay group attribute of an assay that holds values normalized before
# import. An assay without it holds counts.
NORMALIZED_VALUES_ATTR = "normalizedValues"
# The assay group attribute of an assay of TF-IDF values, such as melded gene
# scores: what each term frequency divides a cell's counts by.
TF_DENOMINATOR_ATTR = "tfDenominator"


def normalized_values_record(values: MatrixValues) -> dict[str, Any] | None:
    """Return the ``normalizedValues`` record of an import, or None for counts."""
    if values.kind == "counts":
        return None
    return {"importedAs": values.kind, "cellTotal": values.cellTotal}


def derived_normalized_values() -> dict[str, Any]:
    """Return the record of an assay derived from normalized values, not imported."""
    return {"importedAs": None, "cellTotal": None}


def read_normalized_values(
    attrs: Mapping[str, Any], *, assay: str
) -> dict[str, Any] | None:
    """Return an assay group's ``normalizedValues`` record, or None for counts.

    Raises:
        ValueError: If the record is not a mapping of ``importedAs``, None or a
            normalized kind, and ``cellTotal``, None or a positive number.
    """
    record = attrs.get(NORMALIZED_VALUES_ATTR)
    if record is None:
        return None
    total = record.get("cellTotal") if isinstance(record, Mapping) else None
    if not (
        isinstance(record, Mapping)
        and set(record) == {"importedAs", "cellTotal"}
        and record["importedAs"] in (None, *MATRIX_VALUES[1:])
        and (
            total is None
            or (
                isinstance(total, int | float)
                and not isinstance(total, bool)
                and math.isfinite(total)
                and total > 0
            )
        )
    ):
        raise ValueError(
            f"Assay {assay!r} has the {NORMALIZED_VALUES_ATTR} attribute {record!r}, "
            "which is not {'importedAs': None or one of "
            f"{list(MATRIX_VALUES[1:])}, 'cellTotal': None or a positive number}}"
        )
    return dict(record)


def merged_normalized_values(
    records: Iterable[Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    """Return the record of an assay combined from sources with ``records``.

    The result holds normalized values when any source does, and keeps the
    import kind only when every source shares it, counts included. It keeps no
    cell total, since a combined assay's features can differ from a source's.
    """
    sources = list(records)
    if all(record is None for record in sources):
        return None
    kinds = {None if record is None else record["importedAs"] for record in sources}
    return {"importedAs": kinds.pop() if len(kinds) == 1 else None, "cellTotal": None}


def _assay_paths(assay_name: str, workspace: str | None) -> tuple[str, str | None]:
    """Return the logical assay path and the separate matrix path, if any."""
    if workspace is None:
        return assay_name, None
    return f"{workspace}/{assay_name}", f"matrices/{assay_name}"


def pending_assay_message(
    assay_name: str, workspace: str | None, operation: object
) -> str:
    """Explain how to remove a derived assay left pending by an interruption."""
    place = "" if workspace is None else f" of workspace {workspace!r}"
    store = (
        "a DataStore"
        if workspace is None
        else f"a DataStore opened with workspace={workspace!r}"
    )
    return (
        f"Assay {assay_name!r}{place} is pending: another process may still be "
        f"running {operation}, or an interrupted {operation} left it. If no "
        "process is writing it, remove it with "
        f"discard_interrupted_assay({assay_name!r}) on {store}, then retry."
    )


def validate_new_assay(z: zarr.Group, assay_name: str, workspace: str | None) -> None:
    """Raise ``ValueError`` unless ``assay_name`` can be created in ``workspace``."""
    validate_assay_name(assay_name)
    validate_workspace_name(workspace)
    logical, matrix = _assay_paths(assay_name, workspace)
    physical = logical if matrix is None else matrix
    manifest = read_fresh_group(z).attrs.get("matrixSource", {})
    mounted = manifest.get("assays", {}) if isinstance(manifest, dict) else {}
    existing = z.get(logical)
    if isinstance(existing, zarr.Group):
        operation = existing.attrs.get(PENDING_ASSAY_ATTR)
        if operation is not None:
            raise ValueError(pending_assay_message(assay_name, workspace, operation))
    if matrix is not None:
        # Workspaces share matrices/<assay>, and discarding a pending assay
        # deletes it, so a name pending in any workspace is taken in all.
        for name, holder, operation in pending_assays(z):
            if name == assay_name:
                raise ValueError(pending_assay_message(name, holder, operation))
    if existing is not None or physical in z or assay_name in mounted:
        raise ValueError(
            f"Assay {assay_name!r} already has metadata or a count matrix; choose a new name"
        )


def pending_assays(root: zarr.Group) -> list[tuple[str, str | None, str]]:
    """List ``(assay, workspace, operation)`` for every pending derived assay."""
    found: list[tuple[str, str | None, str]] = []
    for name, group in root.groups():
        operation = group.attrs.get(PENDING_ASSAY_ATTR)
        if operation is not None:
            found.append((name, None, str(operation)))
        elif name not in RESERVED_ASSAY_NAMES and group.attrs.get("is_assay") is None:
            # Any other group can be a workspace that holds assays.
            found.extend(
                (child_name, name, str(child.attrs[PENDING_ASSAY_ATTR]))
                for child_name, child in group.groups()
                if PENDING_ASSAY_ATTR in child.attrs
            )
    return sorted(found, key=lambda item: (item[1] or "", item[0]))


def discard_pending_assay(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    missing_ok: bool = False,
    keep_matrix: bool = False,
) -> bool:
    """Delete a derived assay that an interrupted write left pending.

    Only a group that still carries the pending marker is removed. The matrix
    group and the recorded assay type go first and the marked group last, so
    an interrupted discard can be repeated.

    Returns:
        True when a pending assay was removed.
    """
    validate_assay_name(assay_name)
    validate_workspace_name(workspace)
    logical, matrix = _assay_paths(assay_name, workspace)
    try:
        existing = read_fresh_group(root, logical, missing_ok=True)
    except ContainsArrayError:
        existing = None
    if existing is None or PENDING_ASSAY_ATTR not in existing.attrs:
        if missing_ok:
            return False
        raise ValueError(
            f"Assay {assay_name!r} is not an interrupted derived assay; "
            "nothing was removed"
        )
    if matrix is not None and not keep_matrix and matrix in root:
        del root[matrix]
    workspace_root = read_fresh_group(root, workspace or "", mode="r+", missing_ok=True)
    raw_types = (
        None if workspace_root is None else workspace_root.attrs.get("assayTypes")
    )
    if (
        workspace_root is not None
        and isinstance(raw_types, dict)
        and assay_name in raw_types
    ):
        workspace_root.attrs["assayTypes"] = {
            key: value for key, value in raw_types.items() if key != assay_name
        }
    _discard_membership_column(root, assay_name, workspace)
    del root[logical]
    return True


def _discard_membership_column(
    root: zarr.Group, assay_name: str, workspace: str | None
) -> None:
    """Delete ``<assay>_I`` when its stored attributes name ``assay_name``.

    A column of that name without them is not the assay's membership, and is
    kept.
    """
    cells = read_fresh_group(
        root, _cell_data_path(workspace), mode="r+", missing_ok=True
    )
    if cells is None:
        return
    name = assay_membership_column(assay_name)
    column = cells.get(name)
    if (
        isinstance(column, zarr.Array)
        and column.attrs.get("role") == ASSAY_MEMBERSHIP_ROLE
        and column.attrs.get("assay") == assay_name
    ):
        del cells[name]


def _require_free_membership_name(
    root: zarr.Group, assay_name: str, workspace: str | None
) -> None:
    """Raise when the cell table holds a column named for a new assay's membership."""
    name = assay_membership_column(assay_name)
    cells = root.get(_cell_data_path(workspace))
    if isinstance(cells, zarr.Group) and name in cells:
        raise ValueError(
            f"Cell column {name!r} already exists, and it is the name of the "
            f"column that records which cells a new assay {assay_name!r} "
            f"measures. Drop the column with cells.drop({name!r}), or choose "
            "another assay name."
        )


class DerivedAssayTransaction:
    """Build one derived assay that scans see only after it is complete.

    Use it through :func:`derived_assay_transaction`. The logical group carries
    a pending marker, and no ``is_assay`` marker, until the context exits
    normally with finalized counts.
    """

    def __init__(
        self,
        root: zarr.Group,
        assay_name: str,
        workspace: str | None,
        operation: str,
        membership: str | np.ndarray | None = None,
    ) -> None:
        self.root = root
        self.assay_name = assay_name
        self.workspace = workspace
        self.operation = operation
        self.membership = membership
        self._logical, self._matrix = _assay_paths(assay_name, workspace)
        # Set right after each create_group, so cleanup deletes only the
        # groups that this call created.
        self._created_group = False
        self._created_matrix = False
        self._counts: zarr.Array | None = None

    @property
    def group(self) -> zarr.Group:
        """The pending logical assay group, for provenance attributes."""
        return as_zarr_group(self.root[self._logical], name=self._logical)

    def create_counts(
        self,
        n_cells: int,
        feat_ids: np.ndarray | list[str],
        feat_names: np.ndarray | list[str],
        dtype: Any,
        *,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
    ) -> zarr.Array:
        """Create the pending assay with incomplete counts in ``dtype``."""
        if self._created_group:
            raise RuntimeError("The derived assay counts were already created")
        _validate_dimensions(n_cells, len(feat_ids))
        if (
            self.membership is not None
            and not isinstance(self.membership, str)
            and np.shape(self.membership) != (n_cells,)
        ):
            raise ValueError(
                f"A membership mask needs one value per cell ({n_cells}); got "
                f"shape {np.shape(self.membership)}"
            )
        # These raise before this call creates anything, so its cleanup never
        # touches an assay that another writer created.
        validate_new_assay(self.root, self.assay_name, self.workspace)
        _require_free_membership_name(self.root, self.assay_name, self.workspace)
        group = self.root.create_group(
            self._logical,
            attributes={
                PENDING_ASSAY_ATTR: self.operation,
                "prepared": False,
                "misc": {},
            },
        )
        self._created_group = True
        matrix_group = group
        if self._matrix is not None:
            matrix_group = self.root.create_group(self._matrix)
            self._created_matrix = True
        counts, feature_group, resolved_profile = _create_assay_counts(
            group,
            matrix_group,
            n_cells,
            len(feat_ids),
            dtype,
            profile=profile,
            policy=policy,
        )
        self._counts = counts
        _write_feature_columns(feature_group, feat_ids, feat_names, resolved_profile)
        if self.membership is not None:
            self._write_membership(self.membership, n_cells, resolved_profile)
        return counts

    def _write_membership(
        self, source: str | np.ndarray, n_cells: int, profile: StorageProfile
    ) -> None:
        """Write the pending assay's membership column.

        ``source`` names a boolean cell column to copy, or holds one boolean
        per cell. The column is chunked as
        :func:`~cytearc.storage.layout.column_chunk_rows` chunks a new cell
        column, whatever the chunks of the column it copies.
        """
        cells = as_zarr_group(
            self.root[_cell_data_path(self.workspace)], name="cellData"
        )
        values: zarr.Array | np.ndarray
        if isinstance(source, str):
            values = as_zarr_array(cells[source], name=source)
        else:
            values = np.asarray(source, dtype=bool)
        rows = column_chunk_rows(n_cells, np.dtype(bool).itemsize)
        # The copy streams one chunk at a time and never replaces a column.
        create_streamed_metadata_column(
            cells,
            assay_membership_column(self.assay_name),
            shape=n_cells,
            dtype=bool,
            blocks=(
                MetadataBlock(start, np.asarray(values[start : start + rows]))
                for start in range(0, n_cells, rows)
            ),
            overwrite=False,
            chunkSize=rows,
            profile=profile,
            attributes=assay_membership_attributes(self.assay_name),
        )

    def _ready_group(self) -> zarr.Group:
        """Return the stored pending group once its counts can be published."""
        if self._counts is None:
            raise RuntimeError("The derived assay has no counts to publish")
        counts = zarr.open_array(
            store=self._counts.store,
            path=self._counts.path,
            mode="r",
            zarr_format=self._counts.metadata.zarr_format,
        )
        if counts.attrs.get("complete") is not True:
            raise RuntimeError(
                f"Derived assay {self.assay_name!r} counts were not finalized"
            )
        return read_fresh_group(self.root, self._logical)

    def _discard(self, error: BaseException) -> None:
        if not self._created_group:
            # This call created nothing, so it deletes nothing.
            return
        if is_interruption(error):
            # An interrupted write may still land, so the pending assay stays.
            logger.warning(
                pending_assay_message(self.assay_name, self.workspace, self.operation)
            )
            return
        try:
            discard_pending_assay(
                self.root,
                self.assay_name,
                self.workspace,
                missing_ok=True,
                keep_matrix=not self._created_matrix,
            )
        except Exception as exc:
            logger.warning(
                f"Could not remove the incomplete assay {self.assay_name!r}: {exc}. "
                + pending_assay_message(self.assay_name, self.workspace, self.operation)
            )


@contextmanager
def derived_assay_transaction(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    operation: str,
    membership: str | np.ndarray | None = None,
) -> Iterator[DerivedAssayTransaction]:
    """Create a derived assay atomically with respect to assay scans.

    The body creates counts with :meth:`DerivedAssayTransaction.create_counts`,
    writes and finalizes them, and records provenance on ``group``. A normal
    exit publishes the ``is_assay`` marker last. An exception before that
    deletes the groups this call created; an interruption, such as
    ``KeyboardInterrupt`` or the cancellation of a Zarr call, keeps the pending
    assay and logs how to remove it.

    Args:
        root: Root Zarr group of the store.
        assay_name: Name of the assay to create.
        workspace: Workspace name. None uses the legacy layout.
        operation: Name of the operation recorded in the pending marker.
        membership: Optional cells that the new assay measured: the name of a
            boolean cell column to copy, or one boolean per cell.
    """
    validate_new_assay(root, assay_name, workspace)
    _require_free_membership_name(root, assay_name, workspace)
    transaction = DerivedAssayTransaction(
        root, assay_name, workspace, operation, membership=membership
    )
    try:
        yield transaction
        group = transaction._ready_group()
    except BaseException as error:
        transaction._discard(error)
        raise
    # The publication write follows the try, so nothing is deleted once it is
    # issued. One attribute write swaps the pending marker for the assay marker.
    attributes = dict(group.attrs)
    attributes.pop(PENDING_ASSAY_ATTR, None)
    group.attrs.put({**attributes, "is_assay": True})


def validate_assay_name(assay_name: str) -> None:
    """Reject invalid assay names and names reserved by the datastore layout."""
    if not assay_name or not assay_name.strip():
        raise ValueError("Assay names must be non-empty")
    if "/" in assay_name or "\\" in assay_name:
        raise ValueError(f"Assay name {assay_name!r} must not contain path separators")
    if assay_name in RESERVED_ASSAY_NAMES:
        owner = _ASSAY_NAME_OWNERS[assay_name]
        raise ValueError(
            f"Assay name {assay_name!r} is reserved for {owner}. "
            "Choose another name, or explicitly migrate an existing assay before "
            "opening the store with CyteArc."
        )


def validate_workspace_name(workspace: str | None) -> None:
    """Reject invalid workspace names and names reserved by the datastore layout."""
    if workspace is None:
        return
    if not workspace or not workspace.strip():
        raise ValueError("Workspace names must be non-empty")
    if "/" in workspace or "\\" in workspace:
        raise ValueError(
            f"Workspace name {workspace!r} must not contain path separators"
        )
    if workspace in RESERVED_ASSAY_NAMES:
        owner = _ASSAY_NAME_OWNERS[workspace]
        raise ValueError(
            f"Workspace name {workspace!r} is reserved for {owner}. "
            "Choose another workspace name."
        )


def _validate_dimensions(n_cells: int, n_features: int) -> None:
    if n_cells < 0 or n_features < 0:
        raise ValueError("Assay dimensions must be non-negative")


def _create_assay_counts(
    group: zarr.Group,
    matrix_group: zarr.Group,
    n_cells: int,
    n_features: int,
    dtype: Any,
    *,
    profile: StorageProfile | None,
    policy: CountMatrixPolicy | None,
) -> tuple[zarr.Array, zarr.Group, StorageProfile]:
    """Create the feature group of a new assay group and incomplete counts.

    ``matrix_group`` holds the counts: the assay group itself in the legacy
    layout, or its ``matrices/<assay>`` group in a workspace layout.
    """
    resolved_profile = profile or resolve_storage_profile(group.store)
    feature_group = group.create_group("featureData")
    counts = create_product_counts_array(
        matrix_group,
        n_cells,
        n_features,
        dtype,
        profile=resolved_profile,
        policy=policy,
        zarrFormat=_group_zarr_format(matrix_group),
    )
    counts.attrs["complete"] = False
    return counts, feature_group, resolved_profile


def _create_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    n_features: int,
    dtype: Any,
    *,
    profile: StorageProfile | None,
    policy: CountMatrixPolicy | None,
    normalized_values: Mapping[str, Any] | None = None,
) -> tuple[zarr.Array, zarr.Group, StorageProfile]:
    """Create a complete-marked assay group with incomplete paired counts."""
    validate_new_assay(z, assay_name, workspace)
    _validate_dimensions(n_cells, n_features)
    logical, matrix = _assay_paths(assay_name, workspace)
    attributes: dict[str, Any] = {"is_assay": True, "prepared": False, "misc": {}}
    if normalized_values is not None:
        read_normalized_values(
            {NORMALIZED_VALUES_ATTR: normalized_values}, assay=assay_name
        )
        attributes[NORMALIZED_VALUES_ATTR] = dict(normalized_values)
    group = z.create_group(logical, attributes=attributes)
    return _create_assay_counts(
        group,
        group if matrix is None else z.create_group(matrix),
        n_cells,
        n_features,
        dtype,
        profile=profile,
        policy=policy,
    )


def _write_feature_columns(
    feature_group: zarr.Group,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    profile: StorageProfile,
) -> None:
    """Write a feature table's ``ids``, ``names`` and an all-true ``I``.

    Each column is chunked by :func:`~cytearc.storage.layout.column_chunk_rows`,
    as a cell table's are.
    """
    for name, values, column_dtype in (
        ("ids", feat_ids, None),
        ("names", feat_names, None),
        ("I", np.ones(len(feat_ids), dtype=bool), "bool"),
    ):
        # Encoded once: the stored width of the text sets its chunk, and
        # writing the encoded values copies them no further.
        stored = encode_metadata_values(values, column_dtype, name=name)
        create_zarr_obj_array(
            feature_group,
            name,
            stored,
            stored.dtype,
            chunk_size=column_chunk_rows(len(stored), max(1, stored.dtype.itemsize)),
            profile=profile,
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
    """Create an assay group and its incomplete ``counts`` array in ``dtype``.

    Import writers pass the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves. The
    counts carry ``complete=False`` until the writer calls
    ``finalize_counts``. Derived assays use :func:`derived_assay_transaction`
    instead. ``normalized_values`` is the ``normalizedValues`` record of an
    assay whose values were normalized before import; None writes counts.
    """
    counts, feature_group, resolved_profile = _create_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        len(feat_ids),
        dtype,
        profile=profile,
        policy=policy,
        normalized_values=normalized_values,
    )
    _write_feature_columns(feature_group, feat_ids, feat_names, resolved_profile)
    return counts


def create_empty_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    n_features: int,
    feature_id_dtype: Any,
    feature_name_dtype: Any,
    dtype: Any,
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
    normalized_values: Mapping[str, Any] | None = None,
) -> tuple[zarr.Array, zarr.Group]:
    """Create an assay with counts in ``dtype`` and blockwise feature metadata."""
    counts, feature_group, resolved_profile = _create_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        n_features,
        dtype,
        profile=profile,
        policy=policy,
        normalized_values=normalized_values,
    )
    _create_empty_columns(
        feature_group,
        n_features,
        ids_dtype=feature_id_dtype,
        names_dtype=feature_name_dtype,
        profile=resolved_profile,
    )
    return counts, feature_group


def load_count_array(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
) -> zarr.Array:
    if workspace is None:
        return as_zarr_array(
            root[f"{assay_name}/counts"],
            name=f"{assay_name}/counts",
        )
    return as_zarr_array(
        root[f"matrices/{assay_name}/counts"],
        name=f"matrices/{assay_name}/counts",
    )


def _cell_data_path(workspace: str | None) -> str:
    return "cellData" if workspace is None else f"{workspace}/cellData"


def _create_empty_columns(
    group: zarr.Group,
    n_rows: int,
    *,
    ids_dtype: Any,
    names_dtype: Any,
    profile: StorageProfile | None,
) -> None:
    """Create blockwise-fillable ``ids`` and ``names`` and an all-true ``I``.

    Each column is chunked by :func:`~cytearc.storage.layout.column_chunk_rows`
    from its rows and the bytes of its values, in a cell or a feature table.
    """
    for name, dtype in (("ids", ids_dtype), ("names", names_dtype), ("I", bool)):
        column = create_metadata_column(
            group,
            name,
            dtype=dtype,
            shape=n_rows,
            chunkSize=column_chunk_rows(n_rows, max(1, np.dtype(dtype).itemsize)),
            profile=profile,
        )
        if name == "I":
            column[:] = True


def _stored_cell_dtype(
    values: np.ndarray, threads: int, *, name: str
) -> tuple[np.dtype[Any], bool]:
    """Return the dtype that a cell table stores ``values`` in, as :func:`encode_metadata_values` gives the whole column, and whether making a block of them text takes Python strings.

    Unicode text keeps its width, and numbers their dtype. Objects that hold
    no byte string or None are as wide as the longest text of one, made in
    turn. Byte strings, and objects that hold one or None, are made text a
    block at a time, as :func:`~cytearc.storage.arrays.text_array_and_python`
    makes them, and are as wide as the longest text without its trailing
    NULs, one character at least.

    Raises:
        ValueError: If byte strings are not valid UTF-8, naming the column
            ``name``.
    """
    from .arrays import (
        _text_width,
        decodes_metadata_text,
        invalid_utf8_refusal,
        text_array_and_python,
        text_value,
    )

    dtype: np.dtype[Any] = np.dtype(values.dtype)
    if dtype.kind not in "SO":
        return dtype, False
    if not decodes_metadata_text(values):
        return np.dtype(f"U{max(1, _text_width(values))}"), False
    n_rows = len(values)
    width, python = 0, False
    try:
        # Decoding adds no characters, so text as wide as the longest value's
        # bytes, characters or text sets blocks no longer than the write's.
        if dtype.kind == "S":
            widest = dtype.itemsize
        elif set(map(type, values.flat)) <= {str, bytes, np.bytes_, type(None)}:
            widest = max(map(len, filter(None, values.flat)), default=0)
        else:
            widest = max((len(text_value(value)) for value in values.flat), default=0)
        rows = column_write_rows(column_chunk_rows(n_rows, 4 * max(1, widest)), threads)
        for start in range(0, n_rows, rows):
            text, made = text_array_and_python(values[start : start + rows])
            python = python or made
            width = max(width, _text_width(text))
            del text
    except UnicodeDecodeError as exc:
        raise invalid_utf8_refusal(name) from exc
    return np.dtype(f"U{max(1, width)}"), python


def _cell_text_block_bytes(
    values: np.ndarray, dtype: np.dtype[Any], *, gathered: bool, python: bool
) -> tuple[int, int]:
    """Return what one row of ``values`` holds in a block of a cell table's write: as it is read and made the stored text, and as it is written.

    An array gathered in an order copies its rows, and one in place lends
    them. Byte strings and objects become text of their block's width,
    measured with the length of each, and then of the column's, two copies
    at once; with ``python`` they are made Python strings first, each with
    a reference and up to 60 bytes and four a character, beside objects'
    bytes as NumPy bytes. Only the stored text is kept while Zarr writes
    it.
    """
    source = np.dtype(values.dtype)
    gather = source.itemsize if gathered else 0
    if source.kind not in "SO":
        return gather, gather
    stored = dtype.itemsize
    making = 68 + 3 * stored if python else max(stored + 8, 2 * stored)
    return gather + making, stored


def _cell_column_phases(
    columns: list[tuple[tuple[str, ...], Any]], n_rows: int, threads: int
) -> list[tuple[tuple[str, ...], Any, np.dtype[Any], bool, int, int]]:
    """Return each text source of a cell table with its stored dtype, whether its text takes Python strings, its chunk rows and its block rows."""
    phases = []
    for names, values in columns:
        # The stored width sets the chunk.
        dtype, python = _stored_cell_dtype(values, threads, name=names[0])
        rows = column_chunk_rows(n_rows, max(1, dtype.itemsize))
        phases.append(
            (names, values, dtype, python, rows, column_write_rows(rows, threads))
        )
    return phases


def _cell_columns(ids: Any, names: Any) -> list[tuple[tuple[str, ...], Any]]:
    """Return the text columns of a cell table by source: names that are the IDs are read once for both."""
    same = names is ids
    ids = np.asarray(ids)
    if same:
        return [(("ids", "names"), ids)]
    names = np.asarray(names)
    if len(names) != len(ids):
        raise ValueError(
            f"A cell table needs one name for each of its {len(ids)} cells; "
            f"found {len(names)} names"
        )
    return [(("ids",), ids), (("names",), names)]


def _checked_cell_order(order: np.ndarray | None, n_rows: int) -> np.ndarray | None:
    if order is None:
        return None
    rows = np.asarray(order)
    if rows.shape != (n_rows,) or rows.dtype.kind not in "iu":
        raise ValueError(
            f"A cell table's order must hold one integer row for each of its "
            f"{n_rows} cells; found dtype {rows.dtype} and shape {rows.shape}"
        )
    return rows


def cell_table_write_bytes(
    ids: Any, names: Any, threads: int, *, order: np.ndarray | None = None
) -> int:
    """Return the most that :func:`create_cell_data` holds beside ``ids``, ``names`` and ``order``.

    Each text column is written a block of
    :func:`~cytearc.storage.arrays.column_write_rows` rows at a time: the
    block is read from its source, in ``order`` when it is given, and made
    the stored text, which Zarr then writes, holding what
    :func:`~cytearc.storage.arrays.column_write_bytes` counts on ``threads``
    threads. ``I`` is written after them the same way, a block of ones at a
    time.

    Args:
        ids: The cell IDs, as :func:`create_cell_data` takes them.
        names: The cell names, which may be the IDs.
        threads: The threads that the store encodes on, such as the
            ``nthreads`` that a writer registers for it.
        order: The order of the cells that the table stores, or None.

    Raises:
        ValueError: As :func:`create_cell_data` refuses its arguments.
    """
    from .arrays import column_write_bytes

    columns = _cell_columns(ids, names)
    n_rows = len(columns[0][1])
    gathered = _checked_cell_order(order, n_rows) is not None
    held = []
    for _names, values, dtype, python, _rows, block in _cell_column_phases(
        columns, n_rows, threads
    ):
        read, kept = _cell_text_block_bytes(
            values, dtype, gathered=gathered, python=python
        )
        held.append(
            max(
                min(n_rows, block) * read,
                column_write_bytes(
                    n_rows, dtype.itemsize, threads, block_row_bytes=kept
                ),
            )
        )
    held.append(column_write_bytes(n_rows, 1, threads, block_row_bytes=1))
    return max(held)


def create_cell_data(
    root: zarr.Group,
    workspace: str | None,
    ids: Any,
    names: Any,
    profile: StorageProfile | None = None,
    *,
    order: np.ndarray | None = None,
) -> zarr.Group:
    """Create a cell table of ``ids``, ``names`` and an all-true ``I``.

    Each column is chunked by :func:`~cytearc.storage.layout.column_chunk_rows`
    and written a block of one chunk for each of the store's write threads
    at a time, so the write holds what :func:`cell_table_write_bytes` counts
    beside ``ids`` and ``names`` rather than a copy of either. Unicode text
    keeps its width and numbers their dtype; byte strings, decoded as UTF-8,
    and objects are stored as text as wide as the longest. Names that are
    the IDs themselves are read once for both columns.

    Args:
        root: The store's root group.
        workspace: The workspace of the table, or None for the root's.
        ids: Each cell's ID: an array, or values that ``np.asarray`` makes
            one.
        names: Each cell's name, of the same kinds.
        profile: Zarr encoding profile. When None, chosen from the store.
        order: Row ``r`` of the table holds ``ids[order[r]]`` and
            ``names[order[r]]``, gathered a block at a time, as a shuffled
            import stores its cells. None keeps the rows in place.

    Returns:
        The table's group.

    Raises:
        ValueError: If ``names`` and ``ids`` differ in length, ``order``
            is not one integer row for each cell, or byte strings are not
            valid UTF-8, naming the column.
    """
    from .async_execution import store_codec_threads

    columns = _cell_columns(ids, names)
    n_rows = len(columns[0][1])
    rows_order = _checked_cell_order(order, n_rows)
    group = root.create_group(_cell_data_path(workspace))
    threads = store_codec_threads(group.store)
    for column_names, values, dtype, _python, rows, block in _cell_column_phases(
        columns, n_rows, threads
    ):
        outputs = [
            create_metadata_column(
                group,
                name,
                dtype=dtype,
                shape=n_rows,
                chunkSize=rows,
                profile=profile,
            )
            for name in column_names
        ]
        for start in range(0, n_rows, block):
            stop = min(start + block, n_rows)
            selected = (
                values[start:stop]
                if rows_order is None
                else values[rows_order[start:stop]]
            )
            stored = encode_metadata_values(
                selected, values.dtype, name=column_names[0]
            )
            del selected
            if stored.dtype != dtype:
                stored = stored.astype(dtype)
            for output in outputs:
                output[start:stop] = stored
            del stored
    rows = column_chunk_rows(n_rows, 1)
    flags = create_metadata_column(
        group, "I", dtype=bool, shape=n_rows, chunkSize=rows, profile=profile
    )
    block = column_write_rows(rows, threads)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        flags[start:stop] = np.ones(stop - start, dtype=bool)
    return group


def create_empty_cell_data(
    root: zarr.Group,
    workspace: str | None,
    n_cells: int,
    id_dtype: Any,
    name_dtype: Any,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    """Create cell metadata columns that can be filled blockwise."""
    group = root.create_group(_cell_data_path(workspace))
    _create_empty_columns(
        group,
        n_cells,
        ids_dtype=id_dtype,
        names_dtype=name_dtype,
        profile=profile,
    )
    return group
