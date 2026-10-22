from collections.abc import Callable, Iterator
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
from ._store import decode_text


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


class SparseToZarr:
    """A class for converting data in a sparse matrix to a Zarr hierarchy.

    Args:
        csr_mat: A cells-by-features matrix: a SciPy sparse matrix or array of
                 any format, or a dense two-dimensional array. Other formats
                 are converted to CSR; a CSR matrix is used as given.
        zarr_loc: Output Zarr filename with path
        cell_ids: Cell IDs for the cells in the dataset. They must be present
                  and unique, as every reader requires.
        feature_ids: Feature IDs for the features in the dataset. They must be
                     present and unique.
        assay_name: Name for the output assay. If not provided then automatically set to RNA.
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        feature_names: Optional display names aligned with ``feature_ids``.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly. When None, the
                default policy is used. An import that does not fit
                ``mem_budget`` raises MemoryError before the destination is
                created, naming the largest smaller policy that fits.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        assay_type: Preset assay type, such as ``RNA``, for an assay whose name
                    is not a preset. When None, the assay name decides the type.
        overwrite: Replace an existing CyteArc store that no ``DataStore`` has opened.

    The counts are stored in the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves from the
    canonical (duplicate-summed) values of the matrix.

    Raises:
        ValueError: Raised if number of input cell or feature IDs does not match
            the matrix, if an ID is missing, blank, or repeated, if the matrix
            is not two-dimensional, or if a count is NaN or infinite.
        FileExistsError: If ``zarr_loc`` is not empty and may not be replaced.

    Attributes:
        mat: Input CSR matrix
        assayName: The name of the output assay.
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        csr_mat: csr_matrix,
        zarr_loc: ZarrLocation,
        cell_ids: np.ndarray | list[str],
        feature_ids: np.ndarray | list[str],
        assay_name: str | None = None,
        workspace: str | None = None,
        feature_names: np.ndarray | list[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
        assay_type: str | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        from ..storage.budget import resolve_budget
        from ..storage.schema import (
            create_cell_data,
            create_zarr_count_assay,
            validate_assay_name,
        )
        from ..storage.destinations import create_destination
        from ..assay.classification import validate_assay_type

        validate_assay_type(
            assay_type, assay="RNA" if assay_name is None else assay_name
        )
        if not issparse(csr_mat) and np.ndim(csr_mat) != 2:
            raise ValueError("The count matrix must be two-dimensional")
        # COO, CSC, and dense inputs become CSR, which the scan and the row
        # batches read; a CSR matrix is used as given, without a copy.
        self.mat = csr_mat if isinstance(csr_mat, csr_matrix) else csr_matrix(csr_mat)
        self.assayType = assay_type
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = np.array(cell_ids)
        if assay_name is None:
            logger.debug("Using RNA as the default assay name")
            self.assayName = "RNA"
        else:
            self.assayName = assay_name
        validate_assay_name(self.assayName)
        self.nCells, self.nFeatures = self.mat.shape
        if len(cell_ids) != self.nCells:
            raise ValueError(
                "ERROR: Number of cell ids are not same as number of cells in the matrix"
            )
        if len(feature_ids) != self.nFeatures:
            raise ValueError(
                "ERROR: Number of feature ids are not same as number of features in the matrix"
            )
        _identifiers(cell_ids, "Cell IDs")
        _identifiers(feature_ids, "Feature IDs")
        # The scan reads windows beside the matrix it holds.
        held = sparse_matrix_bytes(self.mat)
        if held >= self.resources.memoryBytes:
            raise MemoryError(
                f"The sparse matrix holds {held} bytes, but mem_budget is "
                f"{self.resources.memoryBytes} bytes. Increase mem_budget."
            )
        (value_range,) = compressed_count_ranges(
            self.mat.indptr,
            self.mat.indices,
            self.mat.data,
            minorSize=self.nFeatures,
            maxBytes=self.resources.memoryBytes - held,
        )
        storage_dtype = count_storage_dtype(self.mat.dtype, value_range)
        # A layout that does not fit fails here, before the destination exists.
        layout = self._fit_count_layout(storage_dtype, policy)
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
        if feature_names is None:
            feature_names = feature_ids
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            n_cells=self.nCells,
            feat_ids=feature_ids,
            feat_names=feature_names,
            dtype=storage_dtype,
            profile=self.profile,
            policy=layout,
        )

    def _count_import_requirements(self) -> tuple[int, Callable[[int], int]]:
        """Return the resident bytes and the window entries of the counts write.

        The layout fit and the write plan the same import from these.
        """
        indptr = np.asarray(self.mat.indptr)
        resident = sparse_matrix_bytes(self.mat) + CountSummary.nbytes_for(
            self.nCells, self.nFeatures
        )
        return resident, lambda rows: max_window_nnz(indptr, rows)

    def _fit_count_layout(
        self, storage_dtype: Any, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the requested or default count layout if its writes fit the budget."""
        from ..storage.sharding import fit_count_layout, sparse_counts_admission
        from .counts_t import counts_t_assays

        resident, window_nnz = self._count_import_requirements()
        assay_types = {} if self.assayType is None else {self.assayName: self.assayType}
        return fit_count_layout(
            {self.assayName: (self.nFeatures, storage_dtype)},
            nCells=self.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((self.assayName,), assay_types),
            admitCounts=sparse_counts_admission(
                nRows=self.nCells,
                maxWindowNnz=window_nnz,
                sourceDtype=self.mat.dtype,
                residentBytes=resident,
            ),
            requested=DEFAULT_COUNT_MATRIX_POLICY if requested is None else requested,
        )

    def dump(self, batch_size: int | None = None) -> None:
        """Write out the data matrix into the Zarr hierarchy.

        Args:
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default.

        Raises:
            ValueError: If ``batch_size`` is not positive.

        Returns:
            None
        """
        from ..storage.schema import load_count_array
        from ..storage.sharding import resolve_sparse_import_batch

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        store = load_count_array(self.z, self.assayName, self.workspace)
        summary = CountSummary(store)
        resident_bytes, window_nnz = self._count_import_requirements()
        plan = resolve_sparse_import_batch(
            (store,),
            nRows=self.nCells,
            resources=self.resources,
            maxWindowNnz=window_nnz,
            sourceDtype=self.mat.dtype,
            batchRows=batch_size,
            residentBytes=resident_bytes,
        )
        self._lastImportPlan = plan
        resolved_batch_rows = plan.batchRows
        logger.debug(
            f"Resolved sparse source batch rows={resolved_batch_rows} "
            f"write_tasks={plan.writeTasks}"
        )

        def row_batches() -> Iterator[coo_matrix]:
            for start in range(0, self.nCells, resolved_batch_rows):
                yield self.mat[start : start + resolved_batch_rows].tocoo()

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
            f"Wrote {self.nCells} cells and {self.nFeatures} features "
            f"to assay {self.assayName}"
        )
        finalize_counts(store, summary=summary)
        from .counts_t import finalize_writer_counts_t

        finalize_writer_counts_t(
            self.z,
            self.assayName,
            self.workspace,
            assay_type=self.assayType,
            resources=self.resources,
            profile=self.profile,
            io=self.io,
        )
