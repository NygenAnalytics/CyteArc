from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import DEFAULT_COUNT_MATRIX_POLICY, CountMatrixPolicy
from ..storage.identity import CountSummary, finalize_counts
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import accumulate_sparse_to_shards
from ..utils.arrays import has_duplicates, max_window_nnz, sparse_matrix_bytes
from ..utils.count_values import compressed_count_ranges
from ..utils.logging import logger
from ..utils.memory_budget import MemoryBudgetError, refusal_words
from ._store import decode_text
from ..utils.arguments import bool_argument
from ..utils.matrix_values import ImportValues


def _identifiers(values: Any, label: str) -> np.ndarray:
    """Return identifiers as text after rejecting missing, blank, and repeated ones.

    Byte strings are decoded as UTF-8, as every reader decodes them.
    """
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    texts = np.asarray(
        [
            ""
            if value is None or (isinstance(value, float) and value != value)
            else (decode_text(value) if isinstance(value, str | bytes) else str(value))
            for value in array
        ],
        dtype=str,
    )
    if texts.size and bool(np.any(np.char.strip(texts) == "")):
        raise ValueError(f"{label} must contain non-empty values")
    if has_duplicates(texts):
        raise ValueError(f"{label} must contain unique values")
    return texts


@dataclass(frozen=True, slots=True)
class _AssayMatrix:
    """One assay that an import writes: its matrix and its features."""

    name: str
    matrix: csr_matrix
    feature_ids: Any
    feature_names: Any
    assay_type: str | None


def _csr(matrix: Any, assay: str | None) -> csr_matrix:
    """Return ``matrix`` as CSR; a CSR matrix is used as given, without a copy."""
    if not issparse(matrix) and np.ndim(matrix) != 2:
        where = "" if assay is None else f" of assay {assay!r}"
        raise ValueError(f"The count matrix{where} must be two-dimensional")
    return matrix if isinstance(matrix, csr_matrix) else csr_matrix(matrix)


def _per_assay(value: Any, assays: list[str], label: str) -> dict[str, Any]:
    """Return a per-assay argument as a mapping, refusing keys that name no assay."""
    if not isinstance(value, Mapping):
        return dict.fromkeys(assays, value)
    unknown = sorted(set(value) - set(assays))
    if unknown:
        raise ValueError(
            f"{label} names {unknown}, which the matrices do not; they name {assays}"
        )
    return {name: value.get(name) for name in assays}


def _assay_matrices(
    csr_mat: Any,
    feature_ids: Any,
    feature_names: Any,
    assay_name: str | None,
    assay_type: Any,
) -> list[_AssayMatrix]:
    """Return the assays that an import writes, one for a single matrix."""
    if not isinstance(csr_mat, Mapping):
        if isinstance(feature_ids, Mapping) or isinstance(feature_names, Mapping):
            raise TypeError(
                "feature_ids and feature_names map assays to their features only "
                "beside a mapping of matrices"
            )
        if isinstance(assay_type, Mapping):
            raise TypeError(
                "assay_type maps assays to types only beside a mapping of matrices"
            )
        name = "RNA" if assay_name is None else assay_name
        return [
            _AssayMatrix(
                name,
                _csr(csr_mat, None),
                feature_ids,
                feature_ids if feature_names is None else feature_names,
                assay_type,
            )
        ]
    if assay_name is not None:
        raise ValueError(
            "assay_name names the assay of a single matrix; the keys of a mapping "
            "of matrices name their assays"
        )
    if not csr_mat:
        raise ValueError("A mapping of matrices needs at least one assay")
    assays = [str(name) for name in csr_mat]
    if not isinstance(feature_ids, Mapping):
        raise TypeError(
            "With a mapping of matrices, feature_ids must map each assay to its "
            "feature IDs"
        )
    missing = sorted(set(assays) - set(feature_ids))
    if missing:
        raise ValueError(f"feature_ids has no IDs for the assays {missing}")
    ids = _per_assay(feature_ids, assays, "feature_ids")
    names = _per_assay(
        {} if feature_names is None else feature_names, assays, "feature_names"
    )
    if feature_names is not None and not isinstance(feature_names, Mapping):
        raise TypeError(
            "With a mapping of matrices, feature_names must map assays to their "
            "feature names, or be None"
        )
    types = _per_assay(assay_type, assays, "assay_type")
    return [
        _AssayMatrix(
            name,
            _csr(csr_mat[name], name),
            ids[name],
            ids[name] if names[name] is None else names[name],
            types[name],
        )
        for name in assays
    ]


class SparseToZarr:
    """Write count matrices held in memory to a new store.

    A single matrix writes one assay. A mapping of assay names to matrices
    writes one assay for each, over the same cells, so tables of several
    modalities, such as RNA, guide, hashtag and protein counts read from
    GEO, become one store without a file between them; ``feature_ids``,
    ``feature_names``, ``assay_type`` and ``values`` then take the same
    keys, and a feature ID need be unique only within its assay.

    Args:
        csr_mat: A cells-by-features matrix: a SciPy sparse matrix or array of
                 any format, or a dense two-dimensional array; or a mapping
                 of assay names to such matrices, all of the same cells.
                 Other formats are converted to CSR; a CSR matrix is used as
                 given.
        zarr_loc: Output Zarr filename with path
        cell_ids: Cell IDs for the cells in the dataset. They must be present
                  and unique, as every reader requires.
        feature_ids: Feature IDs for the features in the dataset, present and
                     unique; with a mapping of matrices, a mapping of each
                     assay to its feature IDs.
        assay_name: Name for the output assay of a single matrix. If not
                    provided then automatically set to RNA. A mapping of
                    matrices names its assays by its keys and takes none.
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        feature_names: Optional display names aligned with ``feature_ids``;
                       with a mapping of matrices, a mapping of the assays
                       that have them.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly for every assay.
                When None, the default policy is used. An import that does
                not fit ``mem_budget`` raises MemoryError before the
                destination is created, naming the largest smaller policy
                that fits.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        assay_type: Preset assay type, such as ``RNA``, for an assay whose name
                    is not a preset, or a mapping of assays to types. When
                    None, the assay name decides the type.
        overwrite: Replace an existing CyteArc store that no ``DataStore`` has opened.
        values: What a matrix holds, ``"counts"``, ``"normalized"``,
            ``"log_normalized"`` or ``"log2_normalized"``, as the import
            reference's normalized values section describes, or a mapping of
            assays to these. None decides from the values.

    Each assay's counts are stored in the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves from the
    canonical (duplicate-summed) values of its matrix, and each takes its own
    count layout. Every matrix stays in memory until the import ends, so each
    assay's plan charges the others' matrices too.

    Raises:
        ValueError: Raised if number of input cell or feature IDs does not match
            a matrix, if an ID is missing, blank, or repeated, if a matrix
            is not two-dimensional, if a count is NaN or infinite, or if a
            per-assay argument names an assay that the matrices do not.
        TypeError: If per-assay arguments are mappings beside a single matrix,
            or ``feature_ids`` is not a mapping beside a mapping of matrices.
        FileExistsError: If ``zarr_loc`` is not empty and may not be replaced.

    Attributes:
        assayNames: The names of the output assays, in the order written.
        mat: The input CSR matrix, or a mapping of assay names to them.
        assayName: The name of the output assay of a single matrix, or None
            for a mapping of matrices.
        z: The Zarr hierarchy (array or group).
        values: What the assay holds, or a mapping of assay names to it,
            decided before the store was created.
    """

    def __init__(
        self,
        csr_mat: csr_matrix | Mapping[str, Any],
        zarr_loc: ZarrLocation,
        cell_ids: np.ndarray | list[str],
        feature_ids: np.ndarray | list[str] | Mapping[str, Any],
        assay_name: str | None = None,
        workspace: str | None = None,
        feature_names: np.ndarray | list[str] | Mapping[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
        assay_type: str | Mapping[str, str] | None = None,
        *,
        overwrite: bool = False,
        values: ImportValues = None,
    ) -> None:
        overwrite = bool_argument(overwrite, "overwrite")
        from ..utils.matrix_values import import_values_argument

        import_values_argument(values)
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_cell_data,
            create_zarr_count_assay,
            normalized_values_record,
            validate_assay_name,
        )
        from ..storage.destinations import create_destination
        from ..assay.classification import validate_assay_type
        from ..utils.matrix_values import DETECTION_CELLS
        from ._store import resolve_import_values

        single = not isinstance(csr_mat, Mapping)
        if single and assay_name is None:
            logger.debug("Using RNA as the default assay name")
        assays = _assay_matrices(
            csr_mat, feature_ids, feature_names, assay_name, assay_type
        )
        for assay in assays:
            validate_assay_name(assay.name)
            validate_assay_type(assay.assay_type, assay=assay.name)
        self._assays = assays
        self.assayNames = tuple(assay.name for assay in assays)
        self.mat: Any = (
            assays[0].matrix
            if single
            else {assay.name: assay.matrix for assay in assays}
        )
        self.assayName = assays[0].name if single else None
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = np.array(cell_ids)
        self.nCells = len(cell_ids)
        for assay in assays:
            where = "" if single else f" of assay {assay.name!r}"
            if assay.matrix.shape[0] != self.nCells:
                raise ValueError(
                    "ERROR: Number of cell ids are not same as number of cells in "
                    f"the matrix{where}"
                )
            if len(assay.feature_ids) != assay.matrix.shape[1]:
                raise ValueError(
                    "ERROR: Number of feature ids are not same as number of "
                    f"features in the matrix{where}"
                )
        _identifiers(cell_ids, "Cell IDs")
        for assay in assays:
            _identifiers(
                assay.feature_ids,
                "Feature IDs" if single else f"Feature IDs of assay {assay.name!r}",
            )
        # The scans read windows beside the matrices, which stay held.
        held = sum(sparse_matrix_bytes(assay.matrix) for assay in assays)
        self._heldMatrixBytes = held
        if held >= self.resources.memoryBytes:
            what = "The sparse matrix holds" if single else "The sparse matrices hold"
            raise MemoryBudgetError(
                f"{what} {held} bytes, but mem_budget is "
                f"{self.resources.memoryBytes} bytes. Increase mem_budget.",
                needed_bytes=held + 1,
                budget_bytes=int(self.resources.memoryBytes),
                item="the sparse matrix" if single else "the sparse matrices",
            )
        value_ranges = {}
        dtypes = {}
        for assay in assays:
            try:
                (value_range,) = compressed_count_ranges(
                    assay.matrix.indptr,
                    assay.matrix.indices,
                    assay.matrix.data,
                    minorSize=assay.matrix.shape[1],
                    maxBytes=self.resources.memoryBytes - held,
                )
            except MemoryBudgetError as error:
                # The scan runs beside the matrices, so the budget it needs holds both.
                needed = held + error.needed_bytes
                what = "the matrix" if single else "the matrices"
                raise MemoryBudgetError(
                    f"{refusal_words(error)} Beside the {held} bytes of {what}, the "
                    f"import needs at least {needed} bytes.",
                    needed_bytes=needed,
                    budget_bytes=int(self.resources.memoryBytes),
                    item=error.item,
                ) from error
            value_ranges[assay.name] = value_range
            dtypes[assay.name] = count_storage_dtype(assay.matrix.dtype, value_range)
        decided = resolve_import_values(
            values,
            value_ranges,
            lambda: {assay.name: assay.matrix[:DETECTION_CELLS] for assay in assays},
            {assay.name: assay.assay_type for assay in assays},
            dtypes,
        )
        self._assayValues = decided
        self.values: Any = decided[assays[0].name] if single else decided
        # A layout that does not fit fails here, before the destination exists.
        layouts = {
            assay.name: self._fit_count_layout(assay, dtypes[assay.name], policy)
            for assay in assays
        }
        # The destination must be empty, or with overwrite an unprepared store.
        self.z = create_destination(
            zarr_loc, overwrite=overwrite, storage_options=storage_options
        )
        _ = create_cell_data(
            root=self.z,
            workspace=self.workspace,
            ids=cell_ids,
            names=cell_ids,
            profile=self.profile,
        )
        for assay in assays:
            create_zarr_count_assay(
                z=self.z,
                assay_name=assay.name,
                workspace=workspace,
                n_cells=self.nCells,
                feat_ids=assay.feature_ids,
                feat_names=assay.feature_names,
                dtype=dtypes[assay.name],
                profile=self.profile,
                policy=layouts[assay.name],
                normalized_values=normalized_values_record(decided[assay.name]),
            )

    def _count_import_requirements(
        self, assay: _AssayMatrix
    ) -> tuple[int, Callable[[int], int]]:
        """Return the resident bytes and the window entries of an assay's counts write.

        Every matrix stays held while each assay is written. The layout fit
        and the write plan the same import from these.
        """
        indptr = np.asarray(assay.matrix.indptr)
        resident = self._heldMatrixBytes + CountSummary.nbytes_for(
            self.nCells, assay.matrix.shape[1]
        )
        return resident, lambda rows: max_window_nnz(indptr, rows)

    def _fit_count_layout(
        self,
        assay: _AssayMatrix,
        storage_dtype: Any,
        requested: CountMatrixPolicy | None,
    ) -> CountMatrixPolicy:
        """Return the requested or default count layout if an assay's writes fit the budget."""
        from ..storage.sharding import fit_count_layout, sparse_counts_admission
        from .counts_t import counts_t_assays

        resident, window_nnz = self._count_import_requirements(assay)
        assay_types = {} if assay.assay_type is None else {assay.name: assay.assay_type}
        return fit_count_layout(
            {assay.name: (assay.matrix.shape[1], storage_dtype)},
            nCells=self.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((assay.name,), assay_types),
            admitCounts=sparse_counts_admission(
                nRows=self.nCells,
                maxWindowNnz=window_nnz,
                sourceDtype=assay.matrix.dtype,
                residentBytes=resident,
            ),
            requested=DEFAULT_COUNT_MATRIX_POLICY if requested is None else requested,
        )

    def dump(self, batch_size: int | None = None) -> None:
        """Write out each assay's matrix into the Zarr hierarchy, one assay at a time.

        Args:
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default.

        Raises:
            ValueError: If ``batch_size`` is not positive.

        Returns:
            None
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for assay in self._assays:
            self._write_assay(assay, batch_size)

    def _write_assay(self, assay: _AssayMatrix, batch_size: int | None) -> None:
        """Write one assay's counts and finish its ``countsT``."""
        from ..storage.schema import load_count_array
        from ..storage.sharding import resolve_sparse_import_batch
        from ..utils.matrix_values import linear_coo
        from .counts_t import finalize_writer_counts_t

        store = load_count_array(self.z, assay.name, self.workspace)
        summary = CountSummary(store)
        resident_bytes, window_nnz = self._count_import_requirements(assay)
        plan = resolve_sparse_import_batch(
            (store,),
            nRows=self.nCells,
            resources=self.resources,
            maxWindowNnz=window_nnz,
            sourceDtype=assay.matrix.dtype,
            batchRows=batch_size,
            residentBytes=resident_bytes,
        )
        self._lastImportPlan = plan
        resolved_batch_rows = plan.batchRows
        logger.debug(
            f"Resolved sparse source batch rows={resolved_batch_rows} "
            f"write_tasks={plan.writeTasks} for assay {assay.name}"
        )
        kind = self._assayValues[assay.name].kind

        def row_batches() -> Iterator[coo_matrix]:
            for start in range(0, self.nCells, resolved_batch_rows):
                batch = assay.matrix[start : start + resolved_batch_rows]
                yield linear_coo(batch, kind, store.dtype, assay=assay.name).tocoo()

        accumulate_sparse_to_shards(
            store,
            row_batches(),
            resources=self.resources,
            residentBytes=resident_bytes,
            producerReserveBytes=plan.producerReserveBytes,
            msg="Writing sparse counts",
            io=self.io,
            countSummary=summary,
        )
        logger.info(
            f"Wrote {self.nCells} cells and {assay.matrix.shape[1]} features "
            f"to assay {assay.name}"
        )
        finalize_counts(store, summary=summary)
        finalize_writer_counts_t(
            self.z,
            assay.name,
            self.workspace,
            assay_type=assay.assay_type,
            resources=self.resources,
            profile=self.profile,
            io=self.io,
        )
