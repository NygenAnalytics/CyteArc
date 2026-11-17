from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ..storage.types import as_zarr_group
from ..readers import CSVReader
from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.sharding import write_dense_from_row_batches
from ..utils.logging import logger
from ..utils.arguments import bool_argument
from ..utils.matrix_values import ImportValues

if TYPE_CHECKING:
    from ..readers._sparse import SparseRowStore


class CSVtoZarr:
    """A class for converting data from CSV format to a Zarr hierarchy.

    Args:
        cr: A CSVReader object
        zarr_loc: The file name for the Zarr hierarchy.
        assay_name: A label for the assay. Ex. "RNA" or "ATAC"
        workspace: Workspace name in the destination store. None uses the
                   legacy layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix layout: a ``CountMatrixPolicy``, used exactly;
                None, the default layout; or ``"fit"``, the largest halving
                of the default ``unitBytes`` and ``chunkBytes`` whose counts
                write and ``countsT`` fit ``mem_budget``. A policy or the
                default that does not fit raises ``CountLayoutMemoryError``,
                a ``MemoryError``, before the destination is created, with
                the largest halving that fits as its ``fittingPolicy``.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        assay_type: Preset assay type, such as ``RNA``, for an assay whose name
                    is not a preset. When None, the assay name decides the type.
        overwrite: If True, replace a CyteArc store that no ``DataStore`` has opened.
        values: What the matrix holds, ``"counts"``, ``"normalized"``,
            ``"log_normalized"`` or ``"log2_normalized"``, as the import
            reference's normalized values section describes. None decides
            from the values.
        shuffle_cells: If True, write the cells in a uniformly random order,
            so that any contiguous slice of the store, such as one shard or
            a training mini-batch, is a random sample of its cells. The
            counts, the cell IDs and the ``cell_data_cols`` columns follow
            the one order, and each cell keeps its row in the file, from 0,
            in the int64 cell column ``sourceRow``. Before the store is
            created, the writer reads the file's rows, as its counts write
            would, sending each to temporary files on local disk, in
            ``temp_dir`` of the reader, and keeping the cell columns in
            memory; the counts write then reads the rows from those files.
            Off by default in this release.
        shuffle_seed: The seed of the shuffle, from 0 to 2**64 - 1. One seed
            gives the same order on every platform.

    The counts are stored in the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves from the
    range the reader found in its first pass over every row. A count that
    the stored dtype cannot hold raises instead of wrapping.

    Raises:
        FileExistsError: If ``zarr_loc`` is not empty and may not be replaced.
        TypeError: If ``shuffle_cells`` is not a boolean, or ``shuffle_seed``
            is not an integer.
        ValueError: If ``shuffle_seed`` lies outside its range, or a shuffled
            import finds that the file no longer holds the rows that the
            reader counted.
        OSError: If ``shuffle_cells`` is True and the temporary disk lacks
            room for the shuffled rows.

    Attributes:
        csvr: A CSVReader object
        z: The Zarr hierarchy (array or group).
        policy: The count-matrix layout that the assay was written with.
        shuffleCells: Whether the import shuffles the cells.
        shuffleSeed: The seed of the shuffle.
    """

    def __init__(
        self,
        cr: CSVReader,
        zarr_loc: ZarrLocation,
        assay_name: str,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | Literal["fit"] | None = None,
        io: StorageIoPolicy | None = None,
        assay_type: str | None = None,
        *,
        overwrite: bool = False,
        values: ImportValues = None,
        shuffle_cells: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        overwrite = bool_argument(overwrite, "overwrite")
        self.shuffleCells = bool_argument(shuffle_cells, "shuffle_cells")
        from ..storage.cell_order import shuffle_seed_argument, shuffled_order

        self.shuffleSeed = shuffle_seed_argument(shuffle_seed)
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
        from ..storage.sharding import count_layout_request
        from ..assay.classification import validate_assay_type
        from ._store import resolve_import_values

        layout_request = count_layout_request(policy)
        self.csvr = cr
        self.assayName = assay_name
        validate_assay_name(self.assayName)
        validate_assay_type(assay_type, assay=self.assayName)
        self.assayType = assay_type
        self._cellDataColumns = _cell_data_columns(cr)
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        self.workspace = workspace
        self.storage_options = storage_options
        cell_ids = self.csvr.cell_ids()
        storage_dtype = count_storage_dtype(self.csvr.countDtype, self.csvr.countRange)
        self.values = resolve_import_values(
            values,
            {self.assayName: self.csvr.countRange},
            self._first_cells,
            {self.assayName: assay_type},
            {self.assayName: storage_dtype},
        )[self.assayName]
        # Row r of every per-cell array holds the file's row _cellOrder[r].
        # A shuffled import reads the file's rows here, as its counts write
        # would, sending each to its destination and keeping the cell
        # columns, so that the counts write reads the rows in that order.
        self._cellOrder: np.ndarray | None = None
        self._shuffledStore: "SparseRowStore | None" = None
        self._shuffledParts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        if self.shuffleCells:
            self._cellOrder = shuffled_order(self.csvr.nCells, self.shuffleSeed)
            self._spill_rows()
            cell_ids = cell_ids[self._cellOrder]
        try:
            # A layout that does not fit fails here, before the destination
            # exists.
            self.policy = self._fit_count_layout(storage_dtype, layout_request)
            # The destination must be empty, or with overwrite an unprepared
            # store.
            self.z = create_destination(
                zarr_loc, overwrite=overwrite, storage_options=storage_options
            )
        except BaseException:
            self._release_shuffled_rows()
            raise
        cell_group = create_cell_data(
            root=self.z,
            workspace=workspace,
            ids=cell_ids,
            names=cell_ids,
            profile=self.profile,
        )
        if self._cellOrder is not None:
            from ..storage.cell_order import write_shuffled_cell_order

            # Written before the keys of the cell columns are planned, so a
            # source column of the same name is skipped.
            write_shuffled_cell_order(
                cell_group, self._cellOrder, self.shuffleSeed, profile=self.profile
            )
        create_zarr_count_assay(
            z=self.z,
            assay_name=self.assayName,
            workspace=workspace,
            n_cells=self.csvr.nCells,
            feat_ids=self.csvr.feature_ids(),
            feat_names=self.csvr.feature_ids(),
            dtype=storage_dtype,
            profile=self.profile,
            policy=self.policy,
            normalized_values=normalized_values_record(self.values),
        )

    def _first_cells(self) -> dict[str, Any]:
        """Return the values of the first cells, which decide what they are."""
        from scipy.sparse import csr_matrix

        from ..utils.matrix_values import DETECTION_CELLS

        blocks = []
        batches = self.csvr.consume()
        try:
            for counts, _cell_values in batches:
                blocks.append(counts)
                if sum(len(block) for block in blocks) >= DETECTION_CELLS:
                    break
        finally:
            batches.close()
        cells = np.vstack(blocks) if blocks else np.zeros((0, self.csvr.nFeatures))
        return {self.assayName: csr_matrix(cells[:DETECTION_CELLS])}

    def _chunk_rows(self) -> int:
        """Return the rows of one reader chunk, which is also a batch of shuffled rows."""
        return min(int(self.csvr.pandas_kwargs["chunksize"]), self.csvr.nCells)

    def _count_import_requirements(self) -> tuple[int, int]:
        """Return the resident and source batch bytes of the counts write.

        A batch is one reader chunk of every count column, held with its
        DataFrame. A batch of shuffled rows is as wide, read as sparse rows
        from a bucket with the next one loading ahead, beside the order of
        the cells and the rows' pointers. The layout fit and the write plan
        the same import from these.
        """
        from ..storage.identity import CountSummary

        rows = self._chunk_rows()
        itemsize = self.csvr.countDtype.itemsize
        batch_bytes = rows * self.csvr.nFeatures * itemsize
        resident = CountSummary.nbytes_for(self.csvr.nCells, self.csvr.nFeatures)
        store = self._shuffledStore
        if self._cellOrder is None or store is None:
            return resident, 2 * batch_bytes
        sparse_batch = rows * self.csvr.nFeatures * (itemsize + 8) + (rows + 1) * 8
        return (
            resident + int(self._cellOrder.nbytes) + store.resident_bytes,
            2 * batch_bytes + sparse_batch + store.read_bytes(read_ahead=True),
        )

    def _spill_rows(self) -> None:
        """Read the file once, sending each row to its destination and keeping the cell columns.

        The pass holds the order and its inverse, one reader chunk twice as
        the counts write does, and what sending the chunk's rows to their
        buckets holds, beside up to 1 GiB of rows waiting for their files.

        Raises:
            MemoryBudgetError: If the budget cannot hold one chunk and its spill.
            ValueError: If the file no longer holds the rows that the reader
                counted.
            OSError: If the temporary disk lacks room for the rows.
        """
        from ..readers._row_spill import row_spill_plan, spill_rows
        from ..storage.cell_order import destination_rows

        assert self._cellOrder is not None
        n_cells, n_features = self.csvr.nCells, self.csvr.nFeatures
        destinations = destination_rows(self._cellOrder)
        available = int(self.resources.memoryBytes) - 2 * int(destinations.nbytes)
        chunk_values = self._chunk_rows() * n_features
        chunk_bytes = 2 * chunk_values * self.csvr.countDtype.itemsize
        plan = row_spill_plan(
            destinations,
            (n_cells, n_features),
            self.csvr.countDtype,
            entries=None,
            longest=n_features,
            maxBytes=available,
            least=lambda plan: plan.batch_bytes(chunk_values) + chunk_bytes,
        )
        parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            position: [] for _name, (position, _dtype) in self._cellDataColumns
        }

        def fill(consumer: Callable[[int, Any], None]) -> None:
            start = 0
            for counts, cell_values in self.csvr.consume():
                if start + counts.shape[0] > n_cells:
                    raise ValueError(_changed_rows_message(n_cells))
                if cell_values is not None:
                    for _name, (position, dtype) in self._cellDataColumns:
                        parts[position].append(
                            _metadata_part(cell_values[:, position], dtype)
                        )
                consumer(start, counts)
                start += int(counts.shape[0])
            if start != n_cells:
                raise ValueError(_changed_rows_message(n_cells))

        self._shuffledStore, _ = spill_rows(
            plan,
            fill,
            operation="shuffled CSV import",
            entry_bytes=None,
            buffer_bytes=plan.batch_buffer_bytes(
                available - plan.ownerBytes - chunk_bytes, chunk_values
            ),
            dense=True,
            temp_dir=self.csvr.tempDir,
        )
        self._shuffledParts = parts

    def _release_shuffled_rows(self) -> None:
        """Remove the shuffled rows, once written or given up."""
        if self._shuffledStore is not None:
            self._shuffledStore.close()
        self._shuffledStore = None

    def _fit_count_layout(
        self, storage_dtype: Any, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the ``requested`` count layout if its writes fit the budget.

        None fits the default layout to the budget.
        """
        from ..storage.sharding import dense_counts_admission, fit_count_layout
        from .counts_t import counts_t_assays

        resident, producer_reserve = self._count_import_requirements()
        assay_types = {} if self.assayType is None else {self.assayName: self.assayType}
        return fit_count_layout(
            {self.assayName: (self.csvr.nFeatures, storage_dtype)},
            nCells=self.csvr.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((self.assayName,), assay_types),
            admitCounts=dense_counts_admission(resident + producer_reserve),
            requested=requested,
        )

    def dump(self) -> None:
        """Writes the count values into the Zarr matrix.

        A shuffled import reads its shuffled rows once, deleting them as it
        goes, and removes them once the counts are written or the write
        fails, so it writes its counts once.

        Raises:
            OverflowError: If a count no longer fits the stored dtype because
                the file changed after the reader's pass.
            ValueError: If the file no longer holds the rows that the
                reader's pass counted.
            RuntimeError: If the shuffled rows of this import were already
                written.

        Returns:
            None
        """
        from ..metadata.membership import reserved_membership_columns
        from ..storage.identity import CountSummary, finalize_counts
        from ..storage.schema import load_count_array
        from ..storage.metadata_keys import metadata_column_keys
        from ..utils.matrix_values import linear_values
        from ._store import keyed_metadata_columns, write_metadata_column

        if self.shuffleCells and self._shuffledStore is None:
            raise RuntimeError(
                "The shuffled rows of this import were written once and removed; "
                "create a new CSVtoZarr to import the file again"
            )
        store = load_count_array(self.z, self.assayName, self.workspace)
        summary = CountSummary(store)
        resident, producer_reserve = self._count_import_requirements()
        cell_data_path = (
            "cellData" if self.workspace is None else f"{self.workspace}/cellData"
        )
        cell_data_grp = as_zarr_group(
            self.z[cell_data_path],
            name=cell_data_path,
        )
        # The membership column name of the imported assay is reserved.
        reserved = reserved_membership_columns([self.assayName])
        # Each entry pairs a column's position in the reader payload with its
        # dtype across every row, so skipped columns keep the mapping.
        metadata = list(
            keyed_metadata_columns(
                self._cellDataColumns,
                metadata_column_keys(
                    [name for name, _payload in self._cellDataColumns],
                    taken=[*cell_data_grp.keys(), *reserved],
                ),
                "cell",
                membership=reserved,
            )
        )
        parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            position: [] for _name, (position, _dtype) in metadata
        }
        shuffled = self._shuffledStore
        if shuffled is not None:
            # The rows' pass kept every cell column in the file's order, which
            # this write takes over.
            parts, self._shuffledParts = self._shuffledParts, {}

        def count_batches() -> Iterator[np.ndarray]:
            if shuffled is not None:
                for batch in shuffled.iter_rows(
                    0,
                    self.csvr.nCells,
                    max(1, self._chunk_rows()),
                    read_ahead=True,
                    delete=True,
                ):
                    yield linear_values(
                        batch.toarray(),
                        self.values.kind,
                        store.dtype,
                        assay=self.assayName,
                    )
                return
            for counts, cell_values in self.csvr.consume():
                if cell_values is not None:
                    for _name, (position, dtype) in metadata:
                        parts[position].append(
                            _metadata_part(cell_values[:, position], dtype)
                        )
                # The writer casts to the stored dtype and rejects a count
                # that dtype cannot hold.
                yield linear_values(
                    counts, self.values.kind, store.dtype, assay=self.assayName
                )

        try:
            write_dense_from_row_batches(
                store,
                count_batches(),
                msg="Writing CSV counts",
                resources=self.resources,
                io=self.io,
                producerReserveBytes=producer_reserve,
                residentBytes=resident,
                countSummary=summary,
            )
        finally:
            self._release_shuffled_rows()
        for name, (position, _dtype) in metadata:
            values = np.concatenate([values for values, _missing in parts[position]])
            missing = np.concatenate([missing for _values, missing in parts[position]])
            if self._cellOrder is not None:
                values = values[self._cellOrder]
                missing = missing[self._cellOrder]
            write_metadata_column(
                cell_data_grp, name, values, missing, profile=self.profile
            )
        logger.info(
            f"Wrote {self.csvr.nCells} cells and {self.csvr.nFeatures} features "
            f"from CSV to assay {self.assayName}"
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


def _changed_rows_message(n_cells: int) -> str:
    return (
        f"The CSV file no longer holds the {n_cells} rows that the reader "
        "counted; the file changed after the reader read it"
    )


def _cell_data_columns(
    reader: CSVReader,
) -> list[tuple[str, tuple[int, np.dtype[Any]]]]:
    """Pair each cell metadata column with its payload position and dtype.

    The reader records the dtype of every ``cell_data_cols`` column across all
    rows, and no dtypes without such columns, so a count mismatch means that
    the reader no longer describes its payload.
    """
    columns = list(reader.cellDataCols)
    dtypes = list(reader.cellDataDtypes or [])
    if len(dtypes) != len(columns):
        raise ValueError(
            f"CSVReader.cellDataDtypes holds {len(dtypes)} dtypes for "
            f"{len(columns)} cell_data_cols columns; it needs one dtype per column"
        )
    return list(zip(columns, enumerate(dtypes), strict=True))


def _metadata_part(
    values: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one chunk of a CSV metadata column to its dtype for every row.

    Blank text cells are missing; blank numeric cells stay NaN.
    """
    if dtype.kind != "O":
        converted = np.asarray(values, dtype=dtype)
        return converted, np.zeros(converted.shape, dtype=bool)
    missing = np.asarray(pd.isna(values), dtype=bool)
    text = np.asarray(
        ["" if absent else str(value) for value, absent in zip(values, missing)],
        dtype=str,
    )
    return text, missing
