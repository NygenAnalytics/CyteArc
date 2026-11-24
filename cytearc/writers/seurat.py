from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import zarr
from scipy.sparse import coo_matrix, csr_matrix, issparse

from ..metadata.membership import membership_attributes
from ..readers import SeuratReader
from ..readers._row_spill import RowSpillPlan, row_spill_plan
from ..readers._sparse import SparseRowStore
from ..readers.seurat import (
    SeuratAssay,
    SeuratDiagnostic,
    SeuratItemInspection,
    SeuratMetadata,
    SeuratMetadataColumn,
    SeuratMembership,
    SeuratNotice,
    SeuratNumericVector,
    SeuratRMatrix,
    SeuratReduction,
    _value_bound,
)
from ..storage.arrays import MISSING_MASK_PREFIX
from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.artifact_writer import (
    ArrayRequirement,
    artifact_transaction,
    plan_artifact,
)
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import ChunkRows, column_chunk_rows
from ..storage.metadata_keys import (
    RESERVED_METADATA_COLUMNS,
    assay_membership_column,
    is_reserved_metadata_name,
    metadata_column_key,
    metadata_column_keys,
)
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.refs import ArtifactRef
from ..utils.arrays import (
    canonicalize_sparse,
    csr_to_coo,
    cumulative_nnz,
    max_window_nnz,
)
from ..utils.count_values import CountValueRange
from ..utils.memory_budget import MemoryBudgetError, refusal_words
from ._store import (
    resolve_import_values,
    DEFAULT_IMPORT_BLOCK_ROWS,
    bounded_block_rows,
    decode_text,
    fingerprint_row_blocks,
    floating_payload_dtype,
    keyed_metadata_columns,
    resolve_import_cell_selection,
)
from ..utils.arguments import bool_argument
from ..utils.matrix_values import ImportValues

if TYPE_CHECKING:
    from ..utils.matrix_values import MatrixValues
    from ..storage.identity import CountSummary

# Cell and feature columns are chunked by the rows of their table and their
# stored width, in at most the import's blocks of rows.
_COLUMN_CHUNK_ROWS = partial(column_chunk_rows, max_rows=DEFAULT_IMPORT_BLOCK_ROWS)


@dataclass(frozen=True, slots=True)
class SeuratImportResult:
    """Result of writing a Seurat object into a CyteArc Zarr store.

    Attributes:
        assayNames: Assay groups written to the destination store.
        defaultAssay: Default assay of the store: ``default_assay``, or else
            the Seurat object's active assay, except that an active SCTAssay
            gives way to the assay of raw counts that its models were fitted
            to.
        cellSelection: Artifact for the imported cell filter column.
        activeIdentity: Imported Seurat active identity as immutable cluster labels.
        reductionArtifacts: Imported reductions keyed by result name.
        notices: Non-fatal import notices, from the reader and of the assays
            and reductions that the import skipped.
    """

    assayNames: tuple[str, ...]
    defaultAssay: str
    cellSelection: ArtifactRef
    activeIdentity: ArtifactRef
    reductionArtifacts: Mapping[str, ArtifactRef]
    notices: tuple[SeuratNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reductionArtifacts",
            MappingProxyType(dict(self.reductionArtifacts)),
        )


def _string_blocks(
    values: Sequence[str],
    block_rows: int,
) -> Iterator[tuple[str, ...]]:
    read_block = getattr(values, "read_block", None)
    for start in range(0, len(values), block_rows):
        stop = min(start + block_rows, len(values))
        block = read_block(start, stop) if callable(read_block) else values[start:stop]
        yield tuple(_decode_text(value) for value in block)


def _bounded_string_dtype(
    values: Sequence[str],
    block_rows: int,
) -> np.dtype[Any]:
    maximum = 1
    for block in _string_blocks(values, block_rows):
        maximum = max(maximum, max((len(value) for value in block), default=1))
    return np.dtype(f"U{maximum}")


def _decode_text(value: str | bytes | None) -> str:
    return "" if value is None else decode_text(value)


def _consecutive_runs(positions: np.ndarray) -> Iterator[tuple[int, int, int]]:
    """Yield (offset, start, stop) for each run of consecutive positions.

    ``positions`` must not be empty.
    """
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    starts = np.concatenate(([0], breaks)).tolist()
    stops = np.concatenate((breaks, [positions.size])).tolist()
    for offset, stop in zip(starts, stops, strict=True):
        yield int(offset), int(positions[offset]), int(positions[stop - 1]) + 1


# A logged assay's batches are copied into a canonical batch of linear values.
_LOGGED_KINDS = frozenset({"log_normalized", "log2_normalized"})


def _sparse_batch(block: Any) -> coo_matrix:
    """Return a block of cells as COO, marked canonical when its cells are.

    A CSR block is marked when each cell's features are sorted without
    repeats, on every supported SciPy; a dense block converts in canonical
    order; another sparse block keeps its own mark.
    """
    if not issparse(block):
        return coo_matrix(np.asarray(block))
    if block.format == "csr":
        return cast(coo_matrix, csr_to_coo(block))
    return cast(coo_matrix, block.tocoo(copy=False))


def _cell_rows(block: Any) -> csr_matrix:
    """Return a block of cells as CSR rows, each keeping its entries in the block's order.

    Repeated coordinates stay apart, so the write sums them as it sums the
    block's own.
    """
    if not issparse(block):
        return csr_matrix(np.asarray(block))
    if block.format == "csr":
        return cast(csr_matrix, block)
    coo = block.tocoo(copy=False)
    order = np.argsort(coo.row, kind="stable")
    return csr_matrix(
        (
            coo.data[order],
            coo.col[order],
            cumulative_nnz(np.bincount(coo.row, minlength=coo.shape[0])),
        ),
        shape=coo.shape,
    )


def _stored_cell_values(block: Any) -> np.ndarray:
    """Return the values that each cell of a block stores, as its batch holds them."""
    if not issparse(block):
        return np.asarray(np.count_nonzero(np.asarray(block), axis=1))
    if block.format == "csr":
        return np.diff(block.indptr)
    return np.bincount(block.tocoo(copy=False).row, minlength=block.shape[0])


class _CellIdsAt(Sequence[str]):
    """The IDs at given positions of a cell ID sequence, read in runs."""

    def __init__(self, ids: Sequence[str] | np.ndarray, positions: np.ndarray) -> None:
        self._ids = ids
        self._positions = positions

    def __len__(self) -> int:
        return int(self._positions.size)

    @overload
    def __getitem__(self, key: int) -> str: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[str, ...]: ...

    def __getitem__(self, key: int | slice) -> str | tuple[str, ...]:
        if isinstance(key, int):
            return self._ids[int(self._positions[key])]
        positions = self._positions[key]
        if not positions.size:
            return ()
        return tuple(
            value
            for _offset, start, stop in _consecutive_runs(positions)
            for value in self._ids[start:stop]
        )


class SeuratToZarr:
    """Convert a serialized Seurat object into a CyteArc Zarr store.

    Args:
        reader: Open ``SeuratReader`` for the source ``.rds`` file.
        zarr_loc: Destination Zarr path or store.
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system
                    memory (e.g. '0.6'). When None, auto-detected.
        nthreads: Worker count for write-time concurrency. When None,
                  auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix layout of every assay: a ``CountMatrixPolicy``,
                used exactly; None, the default layout; or ``"fit"``, for
                each assay the largest halving of the default ``unitBytes``
                and ``chunkBytes`` whose counts write and ``countsT`` fit
                ``mem_budget``. A policy or the default that does not fit
                raises ``CountLayoutMemoryError``, a ``MemoryError``, before
                the destination is created, with the largest halving that
                fits as its ``fittingPolicy``.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        overwrite: Replace an existing CyteArc store that no ``DataStore`` has opened.
        default_assay: Default assay of the store, one of the imported assays.
            None uses the Seurat object's active assay, except that an active
            ``SCTAssay``, whose counts are SCTransform-corrected, gives way to
            the assay of raw counts that its models were fitted to when that
            assay is imported, with an ``sct_default_assay`` notice. An active
            assay that is not imported, such as the ``integrated`` assay of
            ``IntegrateData``, which holds no raw counts, requires it.
        values: What the matrix holds, ``"counts"``, ``"normalized"``,
            ``"log_normalized"`` or ``"log2_normalized"``, as the import
            reference's normalized values section describes. None decides
            from the values.
        shuffle_cells: If True, the default, write the cells in a uniformly
            random order, so that any contiguous slice of the store, such as
            one shard or a training mini-batch, is a random sample of its
            cells. Every per-cell array follows the one order: counts, cell
            IDs and names, cell metadata, assay membership, the active
            identity and reductions; each cell keeps its position among the
            object's cells, from 0, in the int64 cell column ``sourceRow``.
            The value scan sends each cell's counts to temporary files on
            local disk, in ``temp_dir`` of the reader, as it reads them. The
            import holds every per-cell array other than counts whole while
            it writes it, and refuses before the destination exists if one
            does not fit in ``mem_budget``. False writes the cells in the
            object's order.
        shuffle_seed: The seed of the shuffle, from 0 to 2**64 - 1. One seed
            gives the same order on every platform.

    The import writes the assays and reductions that the reader selected.
    With the reader's default selection (``assays`` or ``reductions`` None),
    an assay or reduction that cannot be imported is skipped with a
    ``skipped_assay`` or ``skipped_reduction`` notice; one that the caller
    named raises instead. A reduction is stored under the assay that it was
    computed on (``assay.used``) when that assay is imported, otherwise under
    the default assay, without feature loadings. A reduction of only some of
    the cells, or of the cells in another order, is aligned to the cells by
    name and stored over a cell selection of its cells. A display embedding
    (UMAP or t-SNE) becomes an ``embedding`` artifact and any other reduction
    ``imported_coordinates``.

    Construction prepares every selected assay's counts and reads them once.
    Each assay stores its counts in the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves from their
    canonical values, and a count that the stored dtype cannot hold raises
    instead of wrapping.

    Raises:
        FileExistsError: If ``zarr_loc`` is not empty and may not be replaced.
        ValueError: If the default assay is not imported, or a selected assay
            or reduction cannot be imported; an assay raises its own
            ``SeuratImportError``, which is a ``ValueError``; or if
            ``shuffle_seed`` lies outside its range.
        TypeError: If ``shuffle_cells`` is not a boolean, or ``shuffle_seed``
            is not an integer.
        OSError: If ``shuffle_cells`` is True and the temporary disk lacks
            room for the shuffled cells.
        MemoryBudgetError: If ``shuffle_cells`` is True and the shuffle of
            the cells does not fit in memory beside the count sources, before
            the shuffle runs, or a per-cell array that the import holds whole
            does not fit beside what it holds throughout.

    Attributes:
        policy: The count-matrix layout that each imported assay was written
            with, by assay name.
        shuffleCells: Whether the import shuffles the cells.
        shuffleSeed: The seed of the shuffle.
    """

    # Row r of every per-cell array holds the object's cell _cellOrder[r];
    # None keeps the object's order.
    _cellOrder: np.ndarray | None = None
    # The destination row of each cell, held while the counts are scanned.
    _destinations: np.ndarray | None = None

    def __init__(
        self,
        reader: SeuratReader,
        zarr_loc: ZarrLocation,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | Literal["fit"] | None = None,
        io: StorageIoPolicy | None = None,
        *,
        overwrite: bool = False,
        default_assay: str | None = None,
        values: ImportValues = None,
        shuffle_cells: bool = True,
        shuffle_seed: int = 0,
    ) -> None:
        overwrite = bool_argument(overwrite, "overwrite")
        self.shuffleCells = bool_argument(shuffle_cells, "shuffle_cells")
        from ..storage.cell_order import (
            SOURCE_ROW,
            check_shuffle_fits,
            destination_rows,
            shuffle_seed_argument,
            shuffled_order,
        )

        self.shuffleSeed = shuffle_seed_argument(shuffle_seed)
        from ..utils.matrix_values import import_values_argument

        import_values_argument(values)
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_empty_cell_data,
            create_empty_zarr_count_assay,
            normalized_values_record,
            validate_assay_name,
        )
        from ..storage.async_execution import register_store_threads
        from ..storage.destinations import check_destination, create_destination
        from ..storage.sharding import count_layout_request

        layout_request = count_layout_request(policy)
        resources = resolve_budget(mem_budget, nthreads)
        inspection = reader.inspection
        selection_notices: list[SeuratNotice] = []
        assay_names = self._importable_names(
            inspection.assays,
            explicit=reader.requestedAssays is not None,
            kind="assay",
            notices=selection_notices,
        )
        for assay_name in assay_names:
            validate_assay_name(assay_name)
        resolved_default = self._default_assay(
            reader, assay_names, default_assay, selection_notices
        )

        assays = tuple(reader.get_assay(name) for name in assay_names)
        blocked_reductions = tuple(
            item for item in inspection.reductions if not item.importable
        )
        if blocked_reductions and reader.requestedReductions is not None:
            details = ", ".join(
                f"{item.name} "
                f"({item.blockingDiagnostic.code if item.blockingDiagnostic is not None else 'unsupported'})"
                for item in blocked_reductions
            )
            raise ValueError(f"Selected reductions cannot be imported: {details}")
        reductions = tuple(
            reader.get_reduction(name)
            for name in self._importable_names(
                inspection.reductions,
                explicit=False,
                kind="reduction",
                notices=selection_notices,
            )
        )
        # A reduction whose assay is not imported has no loadings, and its
        # coordinates are stored under the default assay.
        imported_assays = set(assay_names)
        reduction_assays = {
            reduction.name: (
                reduction.assayUsed
                if reduction.assayUsed in imported_assays
                else resolved_default
            )
            for reduction in reductions
        }
        active_identity = reader.activeIdentity
        self._validate_metadata_names(reader.cellMetadata, "cell")
        # The membership column name of every imported assay is reserved,
        # whether or not the import writes the column.
        membership_names = {assay_membership_column(assay.name) for assay in assays}
        cell_names = set(reader.cellMetadata.columnNames)
        conflicts = sorted(cell_names.intersection(membership_names))
        if conflicts:
            raise ValueError(
                "Assay membership columns conflict with cell metadata: "
                + ", ".join(conflicts)
                + ". CyteArc reserves the cell column '<assay>_I' of every imported "
                "assay for the membership of that assay; rename the metadata "
                "column in the Seurat object, leave it out with "
                "SeuratReader(cell_metadata={'exclude': [...]}), or leave the "
                "assay out of the import."
            )
        for assay in assays:
            self._validate_metadata_names(
                assay.featureMetadata,
                f"{assay.name} feature",
            )
            # Matrix sources admit complex values, which are not counts. The
            # reader sizes every source from its feature and cell IDs.
            dtype = np.dtype(assay.counts.dtype)
            if dtype.kind not in "biuf":
                raise TypeError(
                    f"Assay {assay.name!r} counts use unsupported dtype {dtype}"
                )
        # Zarr nests a name with '/' or '\\', so such a column is stored with
        # '_' in their place; a valid name keeps its exact key. A shuffled
        # import writes its own sourceRow, so a source column of that name is
        # skipped.
        cell_metadata_keys = metadata_column_keys(
            reader.cellMetadata.columnNames,
            taken={
                *RESERVED_METADATA_COLUMNS,
                *membership_names,
                *((SOURCE_ROW,) if self.shuffleCells else ()),
            },
        )
        feature_metadata_keys = {
            assay.name: metadata_column_keys(
                assay.featureMetadata.columnNames,
                taken=RESERVED_METADATA_COLUMNS,
            )
            for assay in assays
        }
        # A destination that would be refused fails before the passes over
        # the source below; create_destination checks it again.
        check_destination(
            zarr_loc, overwrite=overwrite, storage_options=storage_options
        )

        source_digest = bytes.fromhex(reader.document.source.source_sha256)
        string_block_rows = max(
            1,
            min(
                DEFAULT_IMPORT_BLOCK_ROWS,
                int(resources.memoryBytes) // (8 * 64),
            ),
        )
        cell_dtype = _bounded_string_dtype(reader.cellIds, string_block_rows)
        feature_dtypes = {
            assay.name: _bounded_string_dtype(assay.featureIds, string_block_rows)
            for assay in assays
        }

        self.reader = reader
        self.workspace = workspace
        self.storageOptions = storage_options
        self.resources = resources
        from ..storage.profiles import resolve_storage_profile

        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        self.assayNames = assay_names
        self.defaultAssay = resolved_default
        self._assays = assays
        self._reductions = reductions
        self._reductionAssays = reduction_assays
        self._activeIdentity = active_identity
        self._cellMetadataKeys = cell_metadata_keys
        self._featureMetadataKeys = feature_metadata_keys
        self._sourceDigest = source_digest
        self._notices = self._collect_notices(
            (
                *inspection.notices,
                *selection_notices,
                *inspection.cellMetadata.notices,
            ),
            assays,
            reductions,
        )
        self._lastImportPlans: dict[str, Any] = {}
        self._lastDenseBatchRows: dict[str, int] = {}
        # The value scan of each sparse assay counts its cells' stored values,
        # as prefix sums over the cells, and checks that every cell holds its
        # features sorted without repeats. The layout fit and the write plan
        # from both.
        self._cellValues: dict[str, np.ndarray] = {}
        self._canonicalCells: dict[str, bool] = {}
        # Every assay's dtype, values and layout are resolved, and a layout
        # that does not fit fails, before the destination exists.
        self._valuesArgument = values
        self.values: dict[str, "MatrixValues"] = {}
        # Row r of every per-cell array holds the object's cell _cellOrder[r].
        # The value scan of each assay sends the cells' counts to their
        # destination rows, _destinations, in a store of the assay's own,
        # which the counts write reads once and removes.
        n_cells = len(reader.cellIds)
        self._shuffledRows: dict[str, SparseRowStore] = {}
        # Whether a dump has begun, after which the shuffled cells are spent.
        self._shuffleSpent = False
        if self.shuffleCells:
            # Drawing the order and its inverse runs beside the count sources.
            check_shuffle_fits(
                n_cells, self._source_resident_bytes(), self.resources.memoryBytes
            )
            self._cellOrder = shuffled_order(n_cells, self.shuffleSeed)
            self._destinations = destination_rows(self._cellOrder)
        # The dtype of each cell metadata column, which a shuffled import
        # finds before the destination exists.
        self._cellMetadataDtypes: dict[str, np.dtype[Any]] = {}
        try:
            count_dtypes = {assay.name: self._prepare_counts(assay) for assay in assays}
            self._destinations = None
            self._residentSourceBytes = self._source_resident_bytes()
            if self.shuffleCells:
                # Per-cell arrays that do not fit in memory are refused
                # before the destination exists.
                self._check_shuffled_holds(cell_dtype)
            self.policy = {
                assay.name: self._fit_count_layout(
                    assay, count_dtypes[assay.name], layout_request
                )
                for assay in assays
            }

            # The destination must be empty, or with overwrite an unprepared
            # store.
            self.z = create_destination(
                zarr_loc, overwrite=overwrite, storage_options=storage_options
            )
            # The import writes the new store on its threads.
            register_store_threads(self.z.store, self.resources.workers)
        except BaseException:
            self._release_shuffled_rows()
            raise
        self.root = (
            self.z
            if workspace is None
            else self.z.create_group(workspace, overwrite=True)
        )
        self.root.attrs["complete"] = False
        self.root.attrs["cytearc:import_source"] = "seurat"
        self.root.attrs["cytearc:import_complete"] = False
        self.root.attrs["cytearc:import_source_sha256"] = (
            reader.document.source.source_sha256
        )
        self.root.attrs["cytearc:import_payload_sha256"] = (
            reader.document.source.payload_sha256
        )

        self.cellData = create_empty_cell_data(
            self.z,
            workspace,
            len(reader.cellIds),
            cell_dtype,
            cell_dtype,
            profile=self.profile,
        )
        self.counts: dict[str, zarr.Array] = {}
        self.featureData: dict[str, zarr.Group] = {}
        for assay in assays:
            feature_dtype = feature_dtypes[assay.name]
            counts, feature_data = create_empty_zarr_count_assay(
                self.z,
                assay.name,
                workspace,
                len(reader.cellIds),
                len(assay.featureIds),
                feature_dtype,
                feature_dtype,
                dtype=count_dtypes[assay.name],
                profile=self.profile,
                policy=self.policy[assay.name],
                normalized_values=normalized_values_record(self.values[assay.name]),
            )
            self.counts[assay.name] = counts
            self.featureData[assay.name] = feature_data
        self.root.attrs["defaultAssay"] = self.defaultAssay

    @staticmethod
    def _validate_metadata_name(name: str, axis: str) -> None:
        # Cell metadata columns can be left out of the import.
        remedy = (
            f"; leave it out with SeuratReader(cell_metadata={{'exclude': [{name!r}]}})"
            if axis == "cell"
            else ""
        )
        if name in {"", ".", ".."}:
            raise ValueError(
                f"{axis} metadata column {name!r} cannot name a Zarr array{remedy}"
            )
        if not is_reserved_metadata_name(metadata_column_key(name)):
            return
        if name in RESERVED_METADATA_COLUMNS:
            raise ValueError(f"{axis} metadata column {name!r} is reserved{remedy}")
        raise ValueError(
            f"{axis} metadata column {name!r} uses CyteArc's internal prefix{remedy}"
        )

    @classmethod
    def _validate_metadata_names(cls, metadata: SeuratMetadata, axis: str) -> None:
        # The reader rejects repeated names, and a name with the missing-mask
        # prefix fails here, so no column can collide with a generated mask.
        for name in metadata.columnNames:
            cls._validate_metadata_name(name, axis)

    @staticmethod
    def _collect_notices(
        root_notices: tuple[SeuratNotice, ...],
        assays: tuple[SeuratAssay, ...],
        reductions: tuple[SeuratReduction, ...],
    ) -> tuple[SeuratNotice, ...]:
        return (
            *root_notices,
            *(notice for assay in assays for notice in assay.notices),
            *(notice for reduction in reductions for notice in reduction.notices),
        )

    @staticmethod
    def _importable_names(
        items: tuple[SeuratItemInspection, ...],
        *,
        explicit: bool,
        kind: str,
        notices: list[SeuratNotice],
    ) -> tuple[str, ...]:
        """Return the names of the selected items to import.

        A selection that the caller named keeps every name, so that an item
        that cannot be imported raises; the default selection skips such an
        item with a notice.
        """
        names: list[str] = []
        for item in items:
            if item.importable or explicit:
                names.append(item.name)
                continue
            diagnostic = cast(SeuratDiagnostic, item.blockingDiagnostic)
            notices.append(
                SeuratNotice(
                    code=f"skipped_{kind}",
                    message=(
                        f"{kind} {item.name!r} is not imported: {diagnostic.message}"
                    ),
                    objectPath=diagnostic.objectPath,
                    context={
                        "reason": diagnostic.code,
                        "details": dict(diagnostic.context),
                    },
                )
            )
        return tuple(names)

    @staticmethod
    def _default_assay(
        reader: SeuratReader,
        assay_names: tuple[str, ...],
        requested: str | None,
        notices: list[SeuratNotice],
    ) -> str:
        """Resolve the store's default assay among the imported assays."""
        if requested is not None:
            if not isinstance(requested, str):
                raise TypeError("default_assay must be an assay name or None")
            if requested not in assay_names:
                if requested in reader.assayNames:
                    # Selected, but skipped because it cannot be imported.
                    reader.get_assay(requested)
                raise ValueError(
                    f"default_assay {requested!r} is not imported; pass one of "
                    f"the imported assays {list(assay_names)}"
                )
            return requested
        active = reader.inspection.activeAssay
        if active in assay_names:
            umi_assay = reader.get_assay(active).umiAssay
            if umi_assay is None or umi_assay not in assay_names:
                return active
            # SCTransform-corrected counts are not raw counts, which CyteArc
            # analyses expect of the default assay.
            notices.append(
                SeuratNotice(
                    code="sct_default_assay",
                    message=(
                        f"active assay {active!r} holds SCTransform-corrected "
                        f"counts, so the default assay is {umi_assay!r}, whose "
                        "raw counts its models were fitted to; pass "
                        f"default_assay={active!r} to keep the active assay"
                    ),
                    objectPath="active.assay",
                    context={"activeAssay": active, "defaultAssay": umi_assay},
                )
            )
            return umi_assay
        if active in reader.assayNames and assay_names:
            diagnostic = cast(
                SeuratDiagnostic, reader.inspection.assay(active).blockingDiagnostic
            )
            raise ValueError(
                f"Active assay {active!r} cannot be imported ({diagnostic.code}: "
                f"{diagnostic.message}); pass default_assay naming one of the "
                f"imported assays {list(assay_names)}"
            )
        if active in reader.assayNames:
            # Nothing else imports, so the active assay's own error explains.
            reader.get_assay(active)
        raise ValueError(
            f"Active assay {active!r} is not selected for import; select it, or "
            f"pass default_assay naming one of the imported assays "
            f"{list(assay_names)}"
        )

    def dump(self, batch_size: int | None = None) -> SeuratImportResult:
        """Write assays, RNA ``countsT``, and importable reductions.

        A shuffled import reads each assay's shuffled counts once, deleting
        them as it goes, and removes them once they are written or the dump
        fails, so it dumps once.

        Args:
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default.

        Returns:
            Imported assay names, cell selection, and reduction artifacts.

        Raises:
            ValueError: If ``batch_size`` is not positive.
            RuntimeError: If a shuffled import was already dumped, or holds
                no shuffled cells of an assay with cells.
        """
        if batch_size is not None and (
            isinstance(batch_size, bool) or int(batch_size) <= 0
        ):
            raise ValueError("batch_size must be positive")
        requested_rows = None if batch_size is None else int(batch_size)
        if self._shuffleSpent:
            raise RuntimeError(
                "The shuffled cells of this import were written once and "
                "removed; create a new SeuratToZarr to import the object again"
            )
        self._shuffleSpent = self.shuffleCells

        self.reader.inspection
        self.root.attrs["complete"] = False
        self.root.attrs["cytearc:import_complete"] = False
        try:
            metadata_rows = self._bounded_block_rows(
                requested_rows,
                row_bytes=64,
            )
            self._write_cell_data(metadata_rows)
            for assay in self._assays:
                self._write_feature_data(assay, metadata_rows)
            for assay in self._assays:
                self._write_counts(assay, requested_rows)
            from .counts_t import finalize_writer_counts_t

            # Each assay takes the preset of its name, or the generic type.
            for assay in self._assays:
                finalize_writer_counts_t(
                    self.z,
                    assay.name,
                    self.workspace,
                    resources=self.resources,
                    profile=self.profile,
                    io=self.io,
                )
            cell_selection = self._write_cell_selection()
            active_identity = self._write_active_identity(
                cell_selection,
                metadata_rows,
            )
            reduction_artifacts = self._write_reductions(
                cell_selection,
                requested_rows,
            )
        except BaseException:
            self._release_shuffled_rows()
            self.root.attrs["complete"] = False
            self.root.attrs["cytearc:import_complete"] = False
            raise
        self.root.attrs["complete"] = True
        self.root.attrs["cytearc:import_complete"] = True
        return SeuratImportResult(
            assayNames=self.assayNames,
            defaultAssay=self.defaultAssay,
            cellSelection=cell_selection,
            activeIdentity=active_identity,
            reductionArtifacts=reduction_artifacts,
            notices=self._notices,
        )

    def _bounded_block_rows(
        self,
        requested: int | None,
        *,
        row_bytes: int,
    ) -> int:
        return bounded_block_rows(
            requested,
            row_bytes=row_bytes,
            memory_bytes=int(self.resources.memoryBytes),
        )

    def _write_cell_data(self, block_rows: int) -> None:
        order = self._cellOrder
        if order is None:
            self._write_string_axis(
                self.cellData["ids"],
                self.cellData["names"],
                self.reader.cellIds,
                block_rows,
            )
        else:
            self._write_shuffled_cell_ids(order, block_rows)
            from ..storage.cell_order import write_shuffled_cell_order

            write_shuffled_cell_order(
                self.cellData,
                order,
                self.shuffleSeed,
                profile=self.profile,
            )
        for key, column in keyed_metadata_columns(
            ((column.name, column) for column in self.reader.cellMetadata.columns),
            self._cellMetadataKeys,
            "cell",
        ):
            self._write_metadata_column(
                self.cellData,
                column,
                block_rows,
                name=key,
                chunk_rows=_COLUMN_CHUNK_ROWS,
                order=order,
                dtype=self._cellMetadataDtypes.get(column.name),
            )
        for assay in self._assays:
            if assay.cellMembership.allIncluded:
                continue
            # The attributes are part of the column's first metadata write.
            self._create_boolean_column(
                self.cellData,
                assay_membership_column(assay.name),
                assay.cellMembership,
                block_rows,
                attributes=membership_attributes(assay.name),
                order=order,
            )

    def _write_shuffled_cell_ids(self, order: np.ndarray, block_rows: int) -> None:
        """Write the cell IDs and names, held in the object's order, in shuffled order."""
        ids = cast(zarr.Array, self.cellData["ids"])
        names = cast(zarr.Array, self.cellData["names"])
        # Construction checked that the IDs fit in memory.
        held = np.empty(len(self.reader.cellIds), dtype=ids.dtype)
        start = 0
        for values_block in _string_blocks(self.reader.cellIds, block_rows):
            held[start : start + len(values_block)] = values_block
            start += len(values_block)
        start = 0
        for (block,) in self._shuffled_blocks((held,), block_rows):
            stop = start + block.shape[0]
            ids[start:stop] = block
            names[start:stop] = block.astype(names.dtype, copy=False)
            start = stop

    def _check_shuffled_holds(self, cell_dtype: np.dtype[Any]) -> None:
        """Refuse a shuffled import whose per-cell arrays do not fit in memory.

        The import holds its cell IDs, each cell metadata column, each
        partial assay membership, the active identity, and each reduction's
        coordinates and cell names whole, in the object's order, with one
        gathered block of them, to write them in shuffled order. A block
        takes the most rows that any ``batch_size`` of :meth:`dump` gives
        it, and each array is checked beside what the import holds when it
        writes its cell table, before its counts: the count sources' row
        counts, the order of the cells and the row counts of the shuffled
        cells.

        Raises:
            MemoryBudgetError: If one of them does not fit in the budget.
        """
        assert self._cellOrder is not None
        resident = self._source_resident_bytes()
        n_cells = int(self._cellOrder.size)
        largest = max(1, n_cells)
        metadata_rows = self._bounded_block_rows(largest, row_bytes=64)
        self._check_shuffled_rows_memory(
            cell_dtype.itemsize,
            metadata_rows,
            "the cell IDs in shuffled order",
            resident,
        )
        scan_rows = self._bounded_block_rows(None, row_bytes=64)
        for column in self.reader.cellMetadata.columns:
            dtype = self._metadata_dtype(column, scan_rows)
            self._cellMetadataDtypes[column.name] = dtype
            self._check_shuffled_rows_memory(
                dtype.itemsize + 1,
                metadata_rows,
                f"cell metadata column {column.name!r} in shuffled order",
                resident,
            )
        for assay in self._assays:
            if not assay.cellMembership.allIncluded:
                name = assay_membership_column(assay.name)
                self._check_shuffled_rows_memory(
                    1,
                    metadata_rows,
                    f"the membership column {name!r} in shuffled order",
                    resident,
                )
        identity = self._activeIdentity
        self._check_shuffled_rows_memory(
            self._metadata_dtype(identity, scan_rows).itemsize + 1,
            metadata_rows,
            f"cell metadata column {identity.name!r} in shuffled order",
            resident,
        )
        for reduction in self._reductions:
            matrix = reduction.cellEmbeddings
            # Each row's coordinates and cell name, and its place in the
            # rows' order and among the store's rows; beside them, the
            # store row of every cell and the mask of the reduction's cells.
            row_bytes = (
                max(1, matrix.shape[1]) * self._floating_dtype(matrix.dtype).itemsize
                + cell_dtype.itemsize
                + 16
            )
            self._check_shuffled_rows_memory(
                row_bytes,
                self._reduction_block_rows(reduction, largest),
                f"the coordinates of reduction {reduction.name!r} in shuffled order",
                resident + 9 * n_cells,
                n_rows=int(matrix.shape[0]),
            )

    def _check_shuffled_rows_memory(
        self,
        row_bytes: int,
        block_rows: int,
        item: str,
        resident: int,
        *,
        n_rows: int | None = None,
    ) -> None:
        """Check that the budget holds a per-cell array of ``row_bytes`` per row in memory.

        A shuffled import holds the object's rows of a cell array, such as a
        metadata column or a reduction, with one gathered block of them,
        beside the ``resident`` bytes it holds throughout, to write them in
        shuffled order. The array has a row for each cell unless ``n_rows``
        says otherwise, as for a reduction of only some of the cells.

        Raises:
            MemoryBudgetError: If they do not fit in the import's budget.
        """
        assert self._cellOrder is not None
        rows = int(self._cellOrder.size if n_rows is None else n_rows)
        needed = int(resident) + (rows + min(int(block_rows), rows)) * int(row_bytes)
        budget = int(self.resources.memoryBytes)
        if needed > budget:
            raise MemoryBudgetError(
                f"Writing {item} exceeds the memory limit. A shuffled import "
                "holds the object's rows to write them in shuffled order, beside "
                f"the {int(resident)} bytes of its count sources and cell order, "
                f"which needs {needed} bytes; raise mem_budget or import with "
                "shuffle_cells=False.",
                needed_bytes=needed,
                budget_bytes=budget,
                item=item,
            )

    def _shuffled_blocks(
        self, arrays: tuple[np.ndarray, ...], block_rows: int
    ) -> Iterator[tuple[np.ndarray, ...]]:
        """Yield blocks of per-cell arrays, held in the object's order, in shuffled order."""
        assert self._cellOrder is not None
        for start in range(0, int(self._cellOrder.size), int(block_rows)):
            rows = self._cellOrder[start : start + int(block_rows)]
            yield tuple(array[rows] for array in arrays)

    def _write_feature_data(self, assay: SeuratAssay, block_rows: int) -> None:
        group = self.featureData[assay.name]
        self._write_string_axis(
            group["ids"],
            group["names"],
            assay.featureIds,
            block_rows,
        )
        for key, column in keyed_metadata_columns(
            ((column.name, column) for column in assay.featureMetadata.columns),
            self._featureMetadataKeys[assay.name],
            f"{assay.name} feature",
        ):
            self._write_metadata_column(
                group, column, block_rows, name=key, chunk_rows=_COLUMN_CHUNK_ROWS
            )

    @staticmethod
    def _write_string_axis(
        ids: Any,
        names: Any,
        values: Sequence[str],
        block_rows: int,
    ) -> None:
        start = 0
        for values_block in _string_blocks(values, block_rows):
            stop = start + len(values_block)
            block = np.asarray(values_block, dtype=ids.dtype)
            ids[start:stop] = block
            names[start:stop] = block.astype(names.dtype, copy=False)
            start = stop

    def _metadata_dtype(
        self,
        column: SeuratMetadataColumn,
        block_rows: int,
    ) -> np.dtype[Any]:
        if column.kind == "logical":
            return np.dtype(bool)
        if column.kind == "integer":
            return np.dtype(np.int64)
        if column.kind == "real":
            return np.dtype(np.float64)
        if column.kind == "factor":
            return _bounded_string_dtype(column.levels, block_rows)
        # The reader's remaining kind is character, whose blocks hold strings.
        maximum = 1
        for start in range(0, column.length, block_rows):
            stop = min(start + block_rows, column.length)
            block = column.read_block(start, stop)
            maximum = max(
                maximum,
                max((len(_decode_text(value)) for value in block.values), default=1),
            )
        return np.dtype(f"U{maximum}")

    def _metadata_blocks(
        self,
        column: SeuratMetadataColumn,
        dtype: np.dtype[Any],
        block_rows: int,
        order: np.ndarray | None = None,
    ) -> Iterator[Any]:
        """Yield the blocks of a metadata column, in the shuffled ``order`` if given.

        A shuffled column is held whole in the object's order, values and
        missing marks, and gathered block by block.
        """
        from ..storage.arrays import MetadataBlock

        if order is not None:
            # Construction checked that the column fits in memory.
            held_values = np.empty(column.length, dtype=dtype)
            held_missing = np.empty(column.length, dtype=bool)
            for block in self._metadata_blocks(column, dtype, block_rows):
                stop = block.start + block.values.shape[0]
                held_values[block.start : stop] = block.values
                held_missing[block.start : stop] = block.missing
            start = 0
            for values, missing in self._shuffled_blocks(
                (held_values, held_missing), block_rows
            ):
                yield MetadataBlock(start, values, missing)
                start += int(values.shape[0])
            return

        for start in range(0, column.length, block_rows):
            stop = min(start + block_rows, column.length)
            block = column.read_block(start, stop)
            missing = np.asarray(block.missing, dtype=bool)
            if column.kind == "character":
                values = np.asarray(
                    [_decode_text(value) for value in block.values],
                    dtype=dtype,
                )
            elif column.kind == "factor":
                codes = np.asarray(block.values)
                values = np.asarray(
                    [
                        "" if missing[index] else column.levels[int(code) - 1]
                        for index, code in enumerate(codes)
                    ],
                    dtype=dtype,
                )
            else:
                values = np.asarray(block.values, dtype=dtype)
                if column.kind == "logical":
                    values[missing] = False
                elif column.kind == "integer":
                    values[missing] = 0
                else:
                    values[missing] = np.nan
            yield MetadataBlock(start, values, missing)

    def _write_metadata_column(
        self,
        group: zarr.Group,
        column: SeuratMetadataColumn,
        block_rows: int,
        *,
        name: str | None = None,
        chunk_rows: ChunkRows | None = None,
        order: np.ndarray | None = None,
        dtype: np.dtype[Any] | None = None,
    ) -> zarr.Array:
        """Write one metadata column of the source.

        ``chunk_rows`` gives its chunks from its rows and stored width, as the
        writers of the cell and feature tables pass it; without it, as for
        the active identity's artifact, a chunk holds the rows of one of the
        import's blocks. A cell column of a shuffled import passes the
        ``order`` of its cells, and the ``dtype`` that construction found.
        """
        from ..storage.arrays import create_streamed_metadata_column

        if dtype is None:
            dtype = self._metadata_dtype(column, block_rows)
        output = create_streamed_metadata_column(
            group,
            column.name if name is None else name,
            shape=column.length,
            dtype=dtype,
            blocks=self._metadata_blocks(column, dtype, block_rows, order),
            overwrite=True,
            chunkSize=(
                min(DEFAULT_IMPORT_BLOCK_ROWS, max(1, column.length))
                if chunk_rows is None
                else chunk_rows(column.length, max(1, dtype.itemsize))
            ),
            hasMissing=True,
            profile=self.profile,
        )
        if column.kind == "factor":
            output.attrs["levels"] = list(column.levels)
            output.attrs["ordered"] = bool(column.ordered)
        return output

    def _write_active_identity(
        self,
        cell_selection: ArtifactRef,
        block_rows: int,
    ) -> ArtifactRef:
        """Store Seurat's analytical active identity without a live column."""
        column = self._activeIdentity
        dtype = self._metadata_dtype(column, block_rows)
        missing_name = f"{MISSING_MASK_PREFIX}values"
        planned = plan_artifact(
            self.root,
            scope="assay",
            assay=self.defaultAssay,
            kind="cluster_labels",
            operation="import_active_identity",
            parameters={
                "source": "seurat",
                "source_key": "active.ident",
                "levels": list(column.levels),
                "ordered": bool(column.ordered),
            },
            inputs={
                "source_digest": self._sourceDigest,
                "cell_selection": cell_selection,
            },
            execution_options={"block_rows": block_rows},
            required_arrays=(
                ArrayRequirement("values", shape=(column.length,), dtype=dtype),
                ArrayRequirement(
                    missing_name,
                    shape=(column.length,),
                    dtype=bool,
                ),
            ),
        )
        if planned.reused:
            return planned.ref
        with artifact_transaction(self.root, planned) as group:
            self._write_metadata_column(
                group,
                column,
                block_rows,
                name="values",
                order=self._cellOrder,
            )
        return planned.ref

    def _create_boolean_column(
        self,
        group: zarr.Group,
        name: str,
        values: SeuratMembership,
        block_rows: int,
        *,
        attributes: dict[str, Any] | None = None,
        order: np.ndarray | None = None,
    ) -> zarr.Array:
        """Write an assay's membership column, in the shuffled ``order`` if given."""
        from ..storage.arrays import MetadataBlock, create_streamed_metadata_column

        read_block = values.read_block
        blocks: Iterator[MetadataBlock] = (
            MetadataBlock(
                start,
                read_block(start, min(start + block_rows, len(values))),
            )
            for start in range(0, len(values), block_rows)
        )
        if order is not None:
            # Construction checked that the column fits in memory.
            held = read_block(0, len(values))
            blocks = (
                MetadataBlock(start, block)
                for start, (block,) in zip(
                    range(0, len(values), block_rows),
                    self._shuffled_blocks((held,), block_rows),
                    strict=True,
                )
            )
        return create_streamed_metadata_column(
            group,
            name,
            shape=len(values),
            dtype=bool,
            blocks=blocks,
            overwrite=True,
            chunkSize=_COLUMN_CHUNK_ROWS(len(values), np.dtype(bool).itemsize),
            profile=self.profile,
            attributes=attributes,
        )

    def _read_peaks(self, source: Any) -> Callable[[int], tuple[int, int]]:
        """Return what reading a window of cells holds, and stores, for a width.

        A pass of a width reads the windows ``[0, width)``, ``[width,
        2 * width)`` and so on. For each width, one source estimate per window
        gives the most that reading one window holds, its working and output
        bytes, and the most values that one window stores. Each width is
        estimated once.
        """
        n_cells = int(source.shape[1])
        peaks: dict[int, tuple[int, int]] = {}

        def window_peaks(rows: int) -> tuple[int, int]:
            width = max(1, min(int(rows), max(1, n_cells)))
            if width not in peaks:
                held = values = 0
                for start in range(0, n_cells, width):
                    stop = min(start + width, n_cells)
                    estimate = source.estimate_read_memory(start, stop)
                    held = max(
                        held,
                        max(0, int(estimate.workingBytes))
                        + max(0, int(estimate.outputBytes)),
                    )
                    values = max(values, _value_bound(source, estimate, stop - start))
                peaks[width] = (held, values)
            return peaks[width]

        return window_peaks

    def _source_resident_bytes(self) -> int:
        """Return the bytes that the count sources of every assay hold.

        They include each sparse assay's counts of its cells' stored values,
        and what a shuffled import holds: the order of the cells, its
        inverse while the counts are scanned, and each scanned assay's row
        pointers of its shuffled cells.
        """
        sources = sum(
            max(0, int(assay.counts.resident_bytes)) for assay in self._assays
        )
        return (
            sources
            + sum(values.nbytes for values in self._cellValues.values())
            + self._shuffle_resident_bytes()
        )

    def _shuffle_resident_bytes(self) -> int:
        """Return what a shuffled import holds beside its count sources; 0 otherwise."""
        if self._cellOrder is None:
            return 0
        return (
            int(self._cellOrder.nbytes)
            + (0 if self._destinations is None else int(self._destinations.nbytes))
            + sum(store.resident_bytes for store in self._shuffledRows.values())
        )

    def _shuffled_store(self, assay_name: str) -> SparseRowStore | None:
        """Return the store of an assay's shuffled cells, or None in the object's order.

        An object without cells has no shuffled cells, and its order is the
        object's.

        Raises:
            RuntimeError: If a shuffled import with cells holds no store of
                the assay's cells, which its counts would otherwise read in
                the object's order.
        """
        if self._cellOrder is None or not self._cellOrder.size:
            return None
        store = self._shuffledRows.get(assay_name)
        if store is None:
            raise RuntimeError(
                f"The shuffled cells of assay {assay_name!r} were written once and "
                "removed, or never sent to their rows; create a new SeuratToZarr "
                "to import the object again"
            )
        return store

    def _release_shuffled_rows(self) -> None:
        """Remove every assay's shuffled cells that the counts write has not read."""
        if self._cellOrder is None:
            return
        for store in self._shuffledRows.values():
            store.close()
        self._shuffledRows.clear()

    def _sparse_write_plan(
        self, assay_name: str, source: Any
    ) -> tuple[Callable[[int], int], Callable[[int], int], bool]:
        """Return how a sparse counts write plans what its producer holds.

        The first function returns the most values of any window of cells of a
        width, wherever it starts, from the values that the scan counted in
        each cell: the rows that the write buffers between two bands are not
        one of its batches. The second returns the most that reading one batch
        of a width holds beyond the batch. The read holds the working and
        output bytes that the source estimates, and the plan already charges
        the batch it returns, an int64 row and column and a value for each of
        its values, so only the rest is staging.

        The flag says whether every batch arrives canonical, so the write
        sorts and sums none: every cell holds its features sorted without
        repeats, and the write passes each batch on as it is read, as it does
        for every kind of values but logged ones, whose log it undoes in a
        copy.
        """
        from ..storage.sharding import sparse_triplet_bytes

        n_cells = int(source.shape[1])
        canonical = (
            self._canonicalCells[assay_name]
            and self.values[assay_name].kind not in _LOGGED_KINDS
        )
        store = self._shuffled_store(assay_name)
        if store is not None:
            # Shuffled cells are read a bucket at a time, with the next bucket
            # loading ahead, beside each batch's row pointers.
            shuffled = store

            def shuffled_values(rows: int) -> int:
                return max_window_nnz(shuffled.indptr, rows) if rows > 0 else 0

            def shuffled_staging(rows: int) -> int:
                batch = min(max(1, int(rows)), max(1, n_cells))
                return int((batch + 1) * 3 * np.dtype(np.int64).itemsize) + int(
                    shuffled.read_bytes(read_ahead=True)
                )

            return shuffled_values, shuffled_staging, canonical
        cumulative = self._cellValues[assay_name]
        batch_value_bytes = sparse_triplet_bytes(source.dtype.itemsize)
        staging_cache: dict[int, int] = {}

        def window_values(rows: int) -> int:
            return max_window_nnz(cumulative, rows) if rows > 0 else 0

        def read_staging(rows: int) -> int:
            width = max(1, min(int(rows), max(1, n_cells)))
            if width not in staging_cache:
                staging = 0
                for start in range(0, n_cells, width):
                    stop = min(start + width, n_cells)
                    estimate = source.estimate_read_memory(start, stop)
                    held = max(0, int(estimate.workingBytes)) + max(
                        0, int(estimate.outputBytes)
                    )
                    values = int(cumulative[stop] - cumulative[start])
                    staging = max(staging, held - values * batch_value_bytes)
                staging_cache[width] = staging
            return staging_cache[width]

        return window_values, read_staging, canonical

    def _prepare_counts(self, assay: SeuratAssay) -> np.dtype[Any]:
        """Prepare one assay's count source and return its storage dtype.

        Every count is read once, in blocks, and duplicate coordinates of
        sparse blocks are summed, as the write stores them. For a sparse
        source, the same pass counts the values that each cell stores and
        checks that every cell holds its features sorted without repeats. A
        shuffled import's pass also sends each cell to its destination row
        in a row bucket store of the assay's own, which the counts write
        reads.

        Raises:
            MemoryError: If the preparation, or reading one cell, and for a
                shuffled import sending it to its row, does not fit
                ``mem_budget``.
            OSError: If the temporary disk lacks room for the shuffled cells.
            ValueError: If a count is NaN or infinite.
        """
        from ..storage.identity import CountSummary
        from ..storage.partition import affordable_width
        from ..storage.sharding import sparse_producer_peak_bytes

        source = assay.counts
        n_features, n_cells = (int(value) for value in source.shape)
        other_sources = (
            sum(
                max(0, int(item.counts.resident_bytes))
                for item in self._assays
                if item is not assay
            )
            + sum(values.nbytes for values in self._cellValues.values())
            + self._shuffle_resident_bytes()
        )
        # The count summary is allocated before the write and stays resident
        # through it, so preparation must fit beside it.
        reserved = other_sources + CountSummary.nbytes_for(n_cells, n_features)
        try:
            self.reader._prepare_assay(
                assay.name, max_bytes=int(self.resources.memoryBytes) - reserved
            )
        except MemoryBudgetError as error:
            needed = reserved + error.needed_bytes
            raise MemoryBudgetError(
                f"{refusal_words(error)} Beside {reserved} bytes for the other "
                "assays' sources and the count summary, the import needs at least "
                f"{needed} bytes.",
                needed_bytes=needed,
                budget_bytes=int(self.resources.memoryBytes),
                item=error.item,
            ) from error
        value_range = CountValueRange()
        # Prefix sums of the values that each cell stores, filled by the scan.
        cell_values = np.zeros(n_cells + 1 if source.is_sparse else 0, dtype=np.int64)
        if source.is_sparse:
            self._cellValues[assay.name] = cell_values
        canonical = True
        if n_cells:
            read_peaks = self._read_peaks(source)
            itemsize = max(1, source.dtype.itemsize)
            available = int(self.resources.memoryBytes) - self._source_resident_bytes()

            def read_bytes(rows: int) -> int:
                held, values = read_peaks(rows)
                if source.is_sparse:
                    # Summing a block's duplicates costs one unbuffered pull,
                    # and counting its cells' values one count per cell.
                    scan = sparse_producer_peak_bytes(0, values, itemsize)
                    scan += rows * np.dtype(np.int64).itemsize
                else:
                    # The integrality check holds a truncated copy and masks.
                    scan = rows * n_features * (itemsize + 2)
                return held + scan

            def shuffling_refusal(needed: int) -> MemoryBudgetError:
                return MemoryBudgetError(
                    f"Assay {assay.name!r} counts cannot be read and sent to their "
                    "shuffled rows one cell at a time within mem_budget. It needs "
                    f"at least {needed} bytes.",
                    needed_bytes=needed,
                    budget_bytes=int(self.resources.memoryBytes),
                    item=f"shuffling one cell of assay {assay.name!r}",
                )

            # A shuffled import also sends each block to its rows' buckets,
            # whose reads must hold the longest cell beside the sources.
            plan = None
            if self._destinations is not None:
                try:
                    plan = self._row_spill_plan(source, available)
                except MemoryBudgetError as error:
                    raise shuffling_refusal(
                        self._source_resident_bytes() + error.needed_bytes
                    ) from error

            def held_bytes(rows: int) -> int:
                held = read_bytes(rows)
                if plan is not None:
                    held += plan.batch_bytes(read_peaks(rows)[1])
                return held

            def fits(rows: int) -> bool:
                return held_bytes(rows) <= available

            rows = affordable_width(fits, n_cells)
            if rows < 1:
                needed = self._source_resident_bytes() + held_bytes(1)
                if plan is None:
                    raise MemoryBudgetError(
                        f"Assay {assay.name!r} counts cannot be read one cell at "
                        "a time within mem_budget. It needs at least "
                        f"{needed} bytes.",
                        needed_bytes=needed,
                        budget_bytes=int(self.resources.memoryBytes),
                        item=f"reading one cell of assay {assay.name!r}",
                    )
                raise shuffling_refusal(needed)

            def scan(start: int, stop: int, raw: Any) -> None:
                nonlocal canonical
                if source.is_sparse:
                    cell_values[start + 1 : stop + 1] = _stored_cell_values(raw)
                if issparse(raw):
                    batch = _sparse_batch(raw)
                    canonical = canonical and bool(batch.has_canonical_format)
                    value_range.update(canonicalize_sparse(batch).data)
                    del batch
                else:
                    value_range.update(raw)

            if plan is None:
                for start in range(0, n_cells, rows):
                    stop = min(start + rows, n_cells)
                    raw = source.read_cells(start, stop)
                    scan(start, stop, raw)
                    # The next read does not run beside this block.
                    del raw
            else:
                self._shuffledRows[assay.name] = self._spill_cells(
                    source,
                    plan,
                    rows,
                    scan,
                    plan.batch_buffer_bytes(
                        available - plan.ownerBytes - read_bytes(rows),
                        read_peaks(rows)[1],
                    ),
                )
        if source.is_sparse:
            np.cumsum(cell_values, out=cell_values)
            self._canonicalCells[assay.name] = canonical
        from ..utils.matrix_values import DETECTION_CELLS

        storage_dtype = count_storage_dtype(source.dtype, value_range)
        self.values[assay.name] = resolve_import_values(
            self._valuesArgument,
            {assay.name: value_range},
            lambda: {
                assay.name: csr_matrix(
                    source.read_cells(0, min(DETECTION_CELLS, n_cells))
                )
            },
            {assay.name: None},
            {assay.name: storage_dtype},
        )[assay.name]
        if self.shuffleCells:
            # The counts write reads the shuffled cells, never the source, so
            # the bucket of converted cells that the source keeps between
            # reads would stay held through it.
            self.reader._release_cached_rows(assay.name)
        return storage_dtype

    def _row_spill_plan(self, source: Any, available: int) -> RowSpillPlan:
        """Plan the row bucket store of an assay's shuffled cells within ``available``.

        Buckets are sized from the values that the source estimates it
        stores, or for dense cells. The plan refuses only a budget whose
        buckets cannot read the longest cell; whether one cell fits beside
        its read is the scan's check.
        """
        assert self._destinations is not None
        n_features, n_cells = (int(value) for value in source.shape)
        entries = (
            _value_bound(source, source.estimate_read_memory(0, n_cells), n_cells)
            if source.is_sparse
            else None
        )
        return row_spill_plan(
            self._destinations,
            (n_cells, n_features),
            source.dtype,
            entries=entries,
            longest=n_features,
            maxBytes=available,
            least=lambda _plan: 0,
        )

    def _spill_cells(
        self,
        source: Any,
        plan: RowSpillPlan,
        rows: int,
        scan: Callable[[int, int, Any], None],
        buffer_bytes: int,
    ) -> SparseRowStore:
        """Scan the source's cells in blocks of ``rows`` and send each to its destination row.

        ``scan(start, stop, block)`` takes each block's values as the
        unshuffled scan does. The first block's first million values measure
        what the cells take on disk, and for dense cells also how many values
        they hold beside zeros, which check the free disk before the spill.
        """
        from ..readers._row_spill import (
            dense_sample_rows,
            dense_spill_estimate,
            sample_rows,
            spill_rows,
        )
        from ..readers._sparse import encoded_entry_bytes

        n_features, n_cells = (int(value) for value in source.shape)
        first = source.read_cells(0, min(rows, n_cells))
        first_rows = _cell_rows(first)
        entries: int | None = None
        if source.is_sparse:
            sample = first_rows[: sample_rows(np.asarray(first_rows.indptr))]
            entry_bytes = encoded_entry_bytes(sample) if sample.nnz else None
            del sample
        else:
            # The first cells, of about a million values, estimate how many
            # values the cells hold beside zeros and what they take on disk.
            entries, entry_bytes = dense_spill_estimate(
                first_rows[: dense_sample_rows(n_cells, n_features)], n_cells
            )

        def fill(consumer: Callable[[int, Any], None]) -> None:
            nonlocal first, first_rows
            for start in range(0, n_cells, rows):
                stop = min(start + rows, n_cells)
                raw = first if start == 0 else source.read_cells(start, stop)
                cells = first_rows if start == 0 else _cell_rows(raw)
                first = first_rows = None
                scan(start, stop, raw)
                consumer(start, cells)
                # The next read does not run beside this block.
                del raw, cells

        store, _ = spill_rows(
            plan,
            fill,
            operation="shuffled Seurat import",
            entry_bytes=entry_bytes,
            entries=entries,
            buffer_bytes=buffer_bytes,
            dense=False,
            temp_dir=(
                None
                if self.reader._scratchDir is None
                else Path(self.reader._scratchDir)
            ),
        )
        return store

    def _fit_count_layout(
        self,
        assay: SeuratAssay,
        storage_dtype: Any,
        requested: CountMatrixPolicy | None,
    ) -> CountMatrixPolicy:
        """Return the ``requested`` layout if its writes fit the budget.

        None fits the default layout to the budget. Both writes read source batches of one destination row band, so the
        fit estimates the source once per band, not once per cell.
        """
        from ..storage.identity import CountSummary
        from ..storage.sharding import (
            dense_counts_admission,
            fit_count_layout,
            sparse_counts_admission,
        )
        from .counts_t import counts_t_assays

        source = assay.counts
        n_features, n_cells = (int(value) for value in source.shape)
        resident = CountSummary.nbytes_for(n_cells, n_features) + (
            self._residentSourceBytes
        )
        if source.is_sparse:
            window_values, read_staging, canonical = self._sparse_write_plan(
                assay.name, source
            )
            admit_counts = sparse_counts_admission(
                nRows=n_cells,
                maxWindowNnz=window_values,
                sourceDtype=source.dtype,
                residentBytes=resident,
                producerStagingBytes=read_staging,
                canonicalBatches=canonical,
                alignedBatches=True,
            )
        else:
            admit_counts = dense_counts_admission(
                resident, self._dense_batch_bytes(assay.name, source)
            )
        return fit_count_layout(
            {assay.name: (n_features, storage_dtype)},
            nCells=n_cells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((assay.name,)),
            admitCounts=admit_counts,
            requested=requested,
        )

    def _write_counts(
        self,
        assay: SeuratAssay,
        requested_rows: int | None,
    ) -> None:
        from ..storage.identity import CountSummary, finalize_counts

        source = assay.counts
        destination = self.counts[assay.name]
        summary = CountSummary(destination)
        n_cells = len(self.reader.cellIds)
        try:
            if n_cells == 0:
                finalize_counts(destination, summary=summary)
                return
            # Construction prepared every source.
            self._residentSourceBytes = summary.nbytes + self._source_resident_bytes()
            if source.is_sparse:
                self._write_sparse_counts(
                    assay.name,
                    source,
                    destination,
                    requested_rows,
                    summary,
                )
            else:
                self._write_dense_counts(
                    assay.name,
                    source,
                    destination,
                    requested_rows,
                    summary,
                )
            finalize_counts(destination, summary=summary)
        finally:
            # A shuffled assay's cells are read once, written or not.
            store = self._shuffledRows.pop(assay.name, None)
            if store is not None:
                store.close()

    def _write_sparse_counts(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
        summary: "CountSummary",
    ) -> None:
        from ..storage.sharding import (
            accumulate_sparse_to_shards,
            resolve_sparse_import_batch,
        )

        n_cells = int(source.shape[1])
        window_values, read_staging, canonical = self._sparse_write_plan(
            assay_name, source
        )
        plan = resolve_sparse_import_batch(
            (destination,),
            nRows=n_cells,
            resources=self.resources,
            maxWindowNnz=window_values,
            sourceDtype=source.dtype,
            batchRows=requested_rows,
            residentBytes=self._residentSourceBytes,
            producerStagingBytes=read_staging,
            canonicalBatches=canonical,
            alignedBatches=True,
        )
        self._lastImportPlans[assay_name] = plan
        self._lastImportPlan = plan
        kind = self.values[assay_name].kind

        store = self._shuffled_store(assay_name)

        # A source returns one row per requested cell; the shard writer
        # rejects a batch of another width and a stream of another length.
        def batches() -> Iterator[coo_matrix]:
            from ..utils.matrix_values import linear_coo

            for start, block in self._cell_blocks(source, store, plan.batchRows):
                batch = linear_coo(
                    _sparse_batch(block),
                    kind,
                    destination.dtype,
                    assay=assay_name,
                )
                del block
                yield batch
                # The writer has buffered the batch, which would otherwise
                # stay held beside the next read.
                del batch

        accumulate_sparse_to_shards(
            destination,
            batches(),
            resources=self.resources,
            residentBytes=self._residentSourceBytes,
            producerReserveBytes=plan.producerReserveBytes,
            msg=f"Writing {assay_name} counts",
            io=self.io,
            countSummary=summary,
        )

    def _resolve_dense_batch_rows(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
    ) -> tuple[int, int]:
        """Return the rows of each source batch and their read staging.

        The layout fit admitted batches of one destination row band, so a
        batch holds one band, or fewer rows on request; the writer admits the
        batch again when it plans the write.
        """
        from ..storage.layout import array_shard_rows

        rows = min(int(source.shape[1]), array_shard_rows(destination))
        if requested_rows is not None:
            rows = min(requested_rows, rows)
        return rows, self._dense_batch_bytes(assay_name, source)(rows)

    def _dense_batch_bytes(self, assay_name: str, source: Any) -> Callable[[int], int]:
        """Return what reading a dense batch of a width holds.

        A batch of the source holds what the source estimates; a batch of
        shuffled cells holds the batch, dense and as read from its bucket,
        and a bucket of rows with the next one loading ahead.
        """
        store = self._shuffled_store(assay_name)
        if store is None:
            read_peaks = self._read_peaks(source)
            return lambda rows: read_peaks(rows)[0]
        shuffled = store
        n_features = int(source.shape[0])
        itemsize = int(source.dtype.itemsize)

        def shuffled_bytes(rows: int) -> int:
            values = max_window_nnz(shuffled.indptr, rows) if rows > 0 else 0
            return (
                rows * n_features * itemsize
                + values * (itemsize + np.dtype(np.int64).itemsize)
                + (rows + 1) * np.dtype(np.int64).itemsize
                + int(shuffled.read_bytes(read_ahead=True))
            )

        return shuffled_bytes

    def _cell_blocks(
        self, source: Any, store: SparseRowStore | None, rows: int
    ) -> Iterator[tuple[int, Any]]:
        """Yield blocks of ``rows`` cells and their first row, in the store's order.

        Without ``store``, the blocks are read from the source in the
        object's order. A shuffled import's blocks come from its store,
        once, each bucket deleted once read, with the next bucket loading
        ahead.
        """
        n_cells = int(source.shape[1])
        if store is None:
            for start in range(0, n_cells, rows):
                yield start, source.read_cells(start, min(start + rows, n_cells))
            return
        start = 0
        for block in store.iter_rows(0, n_cells, rows, read_ahead=True, delete=True):
            yield start, block
            start += int(block.shape[0])

    def _write_dense_counts(
        self,
        assay_name: str,
        source: Any,
        destination: zarr.Array,
        requested_rows: int | None,
        summary: "CountSummary",
    ) -> None:
        from ..storage.sharding import write_dense_from_row_batches

        rows, producer_reserve = self._resolve_dense_batch_rows(
            assay_name,
            source,
            destination,
            requested_rows,
        )
        self._lastDenseBatchRows[assay_name] = rows

        def batches() -> Iterator[np.ndarray]:
            from ..utils.matrix_values import linear_values

            for _start, raw in self._cell_blocks(
                source, self._shuffled_store(assay_name), rows
            ):
                # The writer casts to the stored dtype and rejects a count
                # that dtype cannot hold, a batch of another width, and a
                # stream of another length.
                yield linear_values(
                    raw.toarray() if issparse(raw) else np.asarray(raw),
                    self.values[assay_name].kind,
                    destination.dtype,
                    assay=assay_name,
                )

        write_dense_from_row_batches(
            destination,
            batches(),
            msg=f"Writing {assay_name} counts",
            resources=self.resources,
            residentBytes=self._residentSourceBytes,
            producerReserveBytes=producer_reserve,
            io=self.io,
            countSummary=summary,
        )

    def _write_cell_selection(self) -> ArtifactRef:
        return resolve_import_cell_selection(
            self.root,
            source="seurat",
            inputs={"source_digest": self._sourceDigest},
        )

    @staticmethod
    def _floating_dtype(dtype: Any) -> np.dtype[Any]:
        return floating_payload_dtype(dtype, "Reduction payload")

    @staticmethod
    def _matrix_blocks(
        matrix: SeuratRMatrix,
        block_rows: int,
        dtype: np.dtype[Any],
        rows: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield the matrix in row blocks, in the order of ``rows`` if given."""
        for start in range(0, matrix.shape[0], block_rows):
            stop = min(start + block_rows, matrix.shape[0])
            if rows is None:
                yield np.asarray(matrix.read_rows(start, stop), dtype=dtype)
                continue
            block = np.empty((stop - start, matrix.shape[1]), dtype=dtype)
            for offset, run_start, run_stop in _consecutive_runs(rows[start:stop]):
                block[offset : offset + run_stop - run_start] = matrix.read_rows(
                    run_start, run_stop
                )
            yield block

    @staticmethod
    def _vector_blocks(
        vector: SeuratNumericVector,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> Iterator[np.ndarray]:
        for start in range(0, vector.length, block_rows):
            stop = min(start + block_rows, vector.length)
            yield np.asarray(vector.read_block(start, stop), dtype=dtype)

    @classmethod
    def _fingerprint_matrix(
        cls,
        matrix: SeuratRMatrix,
        block_rows: int,
        dtype: np.dtype[Any],
        rows: np.ndarray | None = None,
    ) -> str:
        return fingerprint_row_blocks(
            cls._matrix_blocks(matrix, block_rows, dtype, rows),
            tuple(matrix.shape),
            dtype,
            label="Reduction payload",
        )

    @classmethod
    def _fingerprint_vector(
        cls,
        vector: SeuratNumericVector,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> str:
        return fingerprint_row_blocks(
            cls._vector_blocks(vector, block_rows, dtype),
            (vector.length,),
            dtype,
            label="Reduction payload",
        )

    @staticmethod
    def _fingerprint_feature_ids(
        values: Sequence[str],
        block_rows: int,
    ) -> str:
        dtype = _bounded_string_dtype(values, block_rows)
        return fingerprint_row_blocks(
            (
                np.asarray(block, dtype=dtype)
                for block in _string_blocks(values, block_rows)
            ),
            (len(values),),
            dtype,
            label="Reduction feature IDs",
        )

    def _reduction_block_rows(
        self,
        reduction: SeuratReduction,
        requested_rows: int | None,
    ) -> int:
        row_bytes = max(1, reduction.cellEmbeddings.shape[1]) * max(
            1, reduction.cellEmbeddings.dtype.itemsize
        )
        return self._bounded_block_rows(requested_rows, row_bytes=row_bytes)

    def _reduction_cells(
        self,
        reduction: SeuratReduction,
        cell_selection: ArtifactRef,
    ) -> tuple[ArtifactRef, Sequence[str] | np.ndarray | zarr.Array, np.ndarray | None]:
        """Return a reduction's cell selection, cell IDs, and row order.

        Rows are stored in the order of the store's cells. A reduction of
        only some of the cells is stored over a selection of those cells; the
        row order is None when the rows already follow the cells.
        """
        positions = reduction.cellPositions
        if positions is None:
            return cell_selection, self.reader.cellIds, None
        order = np.argsort(positions, kind="stable")
        ordered_positions = positions[order]
        if ordered_positions.size < len(self.reader.cellIds):
            from ..storage.selections import resolve_generated_selection_artifact

            selected = np.zeros(len(self.reader.cellIds), dtype=bool)
            selected[ordered_positions] = True
            cell_selection, _values = resolve_generated_selection_artifact(
                self.root,
                scope="datastore",
                kind="cell_selection",
                values=selected,
                row_ids=cast(zarr.Array, self.cellData["ids"]),
                operation="import_reduction_cells",
                parameters={"source": "seurat", "dimreduc_key": reduction.name},
                inputs={
                    "source_digest": self._sourceDigest,
                    "cell_selection": cell_selection,
                },
                source_column="seurat",
            )
        return (
            cell_selection,
            _CellIdsAt(self.reader.cellIds, ordered_positions),
            order,
        )

    def _shuffled_reduction_cells(
        self,
        reduction: SeuratReduction,
        cell_selection: ArtifactRef,
    ) -> tuple[ArtifactRef, np.ndarray]:
        """Return a shuffled import's reduction cell selection and row order.

        Reduction row ``i`` belongs to the object's cell ``positions[i]``,
        which the shuffle sends to store row ``destination_rows[positions[i]]``;
        the rows are stored in the order of their store rows.
        """
        from ..storage.cell_order import destination_rows

        assert self._cellOrder is not None
        n_cells = int(self._cellOrder.size)
        positions = reduction.cellPositions
        destinations = destination_rows(self._cellOrder)
        store_rows = (
            destinations
            if positions is None
            else destinations[np.asarray(positions, dtype=np.int64)]
        )
        del destinations
        order = np.argsort(store_rows, kind="stable")
        selected_rows = store_rows[order]
        del store_rows
        if selected_rows.size == n_cells:
            return cell_selection, order
        from ..storage.selections import resolve_generated_selection_artifact

        selected = np.zeros(n_cells, dtype=bool)
        selected[selected_rows] = True
        cell_selection, _values = resolve_generated_selection_artifact(
            self.root,
            scope="datastore",
            kind="cell_selection",
            values=selected,
            row_ids=cast(zarr.Array, self.cellData["ids"]),
            operation="import_reduction_cells",
            parameters={"source": "seurat", "dimreduc_key": reduction.name},
            inputs={
                "source_digest": self._sourceDigest,
                "cell_selection": cell_selection,
            },
            source_column="seurat",
        )
        return cell_selection, order

    def _held_rows(
        self,
        matrix: SeuratRMatrix,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a reduction's rows and their cell names, read whole in the object's order.

        A shuffled import gathers both in its order; construction checked
        that they fit in memory.
        """
        held = np.empty(matrix.shape, dtype=dtype)
        start = 0
        for block in self._matrix_blocks(matrix, block_rows, dtype):
            held[start : start + block.shape[0]] = block
            start += int(block.shape[0])
        names = np.empty(
            matrix.shape[0], dtype=cast(zarr.Array, self.cellData["ids"]).dtype
        )
        start = 0
        for values_block in _string_blocks(matrix.rowIds, block_rows):
            names[start : start + len(values_block)] = values_block
            start += len(values_block)
        return held, names

    def _write_reductions(
        self,
        cell_selection: ArtifactRef,
        requested_rows: int | None,
    ) -> dict[str, ArtifactRef]:
        from ..embeddings.imported import (
            write_imported_coordinates,
            write_imported_embedding,
        )

        artifacts: dict[str, ArtifactRef] = {}
        for reduction in self._reductions:
            block_rows = self._reduction_block_rows(reduction, requested_rows)
            coordinate_dtype = self._floating_dtype(reduction.cellEmbeddings.dtype)
            coordinate_blocks: Callable[[], Iterator[np.ndarray]]
            cell_ids: Sequence[str] | np.ndarray | zarr.Array
            if self._cellOrder is None:
                reduction_selection, cell_ids, row_order = self._reduction_cells(
                    reduction, cell_selection
                )

                def coordinate_blocks(
                    matrix: SeuratRMatrix = reduction.cellEmbeddings,
                    rows: int = block_rows,
                    dtype: np.dtype[Any] = coordinate_dtype,
                    order: np.ndarray | None = row_order,
                ) -> Iterator[np.ndarray]:
                    return self._matrix_blocks(matrix, rows, dtype, order)

            else:
                reduction_selection, row_order = self._shuffled_reduction_cells(
                    reduction, cell_selection
                )
                # Rows read one by one in a random order would each decode
                # a block of the object, so they are read whole and gathered.
                held, names = self._held_rows(
                    reduction.cellEmbeddings, block_rows, coordinate_dtype
                )
                # The reduction's own cell names, in the order of its rows
                # written, which the write checks against the store's IDs.
                cell_ids = _CellIdsAt(names, row_order)

                def coordinate_blocks(
                    values: np.ndarray = held,
                    rows: int = block_rows,
                    order: np.ndarray = row_order,
                ) -> Iterator[np.ndarray]:
                    for start in range(0, order.size, rows):
                        yield values[order[start : start + rows]]

            coordinate_fingerprint = fingerprint_row_blocks(
                coordinate_blocks(),
                tuple(reduction.cellEmbeddings.shape),
                coordinate_dtype,
                label="Reduction payload",
            )

            assay = self._reductionAssays[reduction.name]
            if reduction.embeddingRole is not None:
                ref = write_imported_embedding(
                    self.root,
                    assay=assay,
                    dimreduc_key=reduction.name,
                    role=reduction.embeddingRole,
                    coordinates=coordinate_blocks,
                    coordinate_shape=reduction.cellEmbeddings.shape,
                    coordinate_dtype=coordinate_dtype,
                    source_digest=self._sourceDigest,
                    payload_fingerprints={"values": coordinate_fingerprint},
                    source_cell_ids=cell_ids,
                    cell_selection=reduction_selection,
                    block_rows=block_rows,
                )
            else:
                payload_fingerprints = {"data": coordinate_fingerprint}
                loadings = reduction.featureLoadings
                loading_blocks = None
                loading_shape = None
                loading_dtype = None
                feature_ids: Sequence[str] | None = None
                if loadings is not None:
                    loading_dtype = self._floating_dtype(loadings.dtype)
                    payload_fingerprints["loadings"] = self._fingerprint_matrix(
                        loadings,
                        block_rows,
                        loading_dtype,
                    )
                    feature_ids = loadings.rowIds
                    payload_fingerprints["feature_ids"] = self._fingerprint_feature_ids(
                        feature_ids, block_rows
                    )
                    loading_shape = loadings.shape

                    def loading_blocks(
                        matrix: SeuratRMatrix = loadings,
                        rows: int = block_rows,
                        dtype: np.dtype[Any] = loading_dtype,
                    ) -> Iterator[np.ndarray]:
                        return self._matrix_blocks(matrix, rows, dtype)

                stdev = reduction.stdev
                stdev_blocks = None
                stdev_shape = None
                stdev_dtype = None
                if stdev is not None:
                    stdev_dtype = self._floating_dtype(stdev.dtype)
                    payload_fingerprints["stdev"] = self._fingerprint_vector(
                        stdev,
                        block_rows,
                        stdev_dtype,
                    )
                    stdev_shape = (stdev.length,)

                    def stdev_blocks(
                        vector: SeuratNumericVector = stdev,
                        rows: int = block_rows,
                        dtype: np.dtype[Any] = stdev_dtype,
                    ) -> Iterator[np.ndarray]:
                        return self._vector_blocks(vector, rows, dtype)

                ref = write_imported_coordinates(
                    self.root,
                    assay=assay,
                    dimreduc_key=reduction.name,
                    role=reduction.role,
                    coordinates=coordinate_blocks,
                    coordinate_shape=reduction.cellEmbeddings.shape,
                    coordinate_dtype=coordinate_dtype,
                    source_digest=self._sourceDigest,
                    payload_fingerprints=payload_fingerprints,
                    source_cell_ids=cell_ids,
                    cell_selection=reduction_selection,
                    loadings=loading_blocks,
                    loadings_shape=loading_shape,
                    loadings_dtype=loading_dtype,
                    feature_ids=feature_ids,
                    stdev=stdev_blocks,
                    stdev_shape=stdev_shape,
                    stdev_dtype=stdev_dtype,
                    block_rows=block_rows,
                )
            artifacts[reduction.name] = ref
        return artifacts


__all__ = ["SeuratImportResult", "SeuratToZarr"]
