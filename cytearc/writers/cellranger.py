from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal

import numpy as np

from ..readers import CrReader
from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..utils.arrays import assay_feature_ranges
from ..utils.logging import logger
from ..utils.memory_budget import MemoryBudgetError, memory_refusal
from ..utils.progress import iter_progress
from ..utils.arguments import bool_argument
from ..utils.matrix_values import ImportValues


class CrToZarr:
    """A class for converting data in the Cellranger format to a Zarr
    hierarchy.

    Args:
        cr: A CrReader object, containing the Cellranger data.
        zarr_loc: The file name for the Zarr hierarchy or a store
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
        assay_types: Preset assay types, such as ``{"GEX": "RNA"}``, for
                     assays whose names are not presets. Other assays take
                     their type from their name.
        lines_in_mem: Matrix Market lines parsed at a time (Matrix Market
                      readers only). A file sorted by cell is parsed as its
                      counts are written, and a feature-major file before
                      the write, as its entries are sorted into rows of
                      cells; an unordered file's entries are sorted when its
                      reader is built. The import reserves this parse buffer
                      for the step that parses, never for more lines than
                      the file holds.
        overwrite: If True, replace a CyteArc store that no ``DataStore`` has opened.
        values: What the matrix holds, ``"counts"``, ``"normalized"``,
            ``"log_normalized"`` or ``"log2_normalized"``, as the import
            reference's normalized values section describes. None decides
            from the values. Matrix Market readers
            read whole numbers only.
        shuffle_cells: If True, the default, write the cells in a uniformly
            random order, so that any contiguous slice of the store, such as
            one shard or a training mini-batch, is a random sample of its
            cells. Every per-cell array follows the one order, and each cell
            keeps its row in the file, from 0, in the int64 cell column
            ``sourceRow``: for a reader that drops barcodes, such as one
            with ``is_filtered=False``, its barcode's index in the file. The
            reader sends each cell to temporary files on local disk, in its
            ``temp_dir``, as it reads it: a 10x HDF5 or Tapestri reader
            during its value scan, a Matrix Market reader as it reads its
            cells for the counts write, before the store is created. A
            reader holds the shuffled cells of one import at a time, and an
            import in the reader's order of a reader that another import
            shuffled reads the reader's order. False writes the cells in the
            file's order.
        shuffle_seed: The seed of the shuffle, from 0 to 2**64 - 1. One seed
            gives the same order on every platform.

    Each assay stores its counts in the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves from the
    canonical values of its own features over the selected cells.

    Raises:
        FileExistsError: If ``zarr_loc`` is not empty, unless ``overwrite`` replaces it.
        TypeError: If ``shuffle_cells`` is not a boolean, or ``shuffle_seed``
            is not an integer, or ``shuffle_cells`` is True and the reader's
            ``count_value_ranges`` takes no ``destinationRows``.
        ValueError: If ``shuffle_seed`` lies outside its range.
        OSError: If ``shuffle_cells`` is True and the temporary disk lacks
            room for the shuffled cells.
        MemoryBudgetError: If ``mem_budget`` cannot hold the shuffle of the
            cells beside the reader's arrays, when ``shuffle_cells`` is True,
            or the sorting of a feature-major Matrix Market file's entries
            into rows of cells. Each is refused before it starts and before
            the destination exists, naming the budget that holds it; the
            sorting also names a smaller ``lines_in_mem`` when one fits.

    Attributes:
        cr: A CrReader object, containing the Cellranger data.
        z: The Zarr hierarchy (array or group).
        policy: The count-matrix layout that every imported assay was written
            with.
        shuffleCells: Whether the import shuffles the cells.
        shuffleSeed: The seed of the shuffle.
    """

    #: The tool whose files the writer reads, as its log names it.
    source_name = "Cell Ranger"

    def __init__(
        self,
        cr: CrReader,
        zarr_loc: ZarrLocation,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | Literal["fit"] | None = None,
        io: StorageIoPolicy | None = None,
        assay_types: dict[str, str] | None = None,
        lines_in_mem: int = 100_000,
        *,
        overwrite: bool = False,
        values: ImportValues = None,
        shuffle_cells: bool = True,
        shuffle_seed: int = 0,
    ) -> None:
        overwrite = bool_argument(overwrite, "overwrite")
        self.shuffleCells = bool_argument(shuffle_cells, "shuffle_cells")
        from ..storage.cell_order import (
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
            create_cell_data,
            create_zarr_count_assay,
            normalized_values_record,
            validate_assay_name,
        )
        from ..storage.async_execution import register_store_threads
        from ..storage.destinations import check_destination, create_destination
        from ..storage.sharding import count_layout_request
        from ..assay.classification import validate_assay_type
        from ._store import resolve_import_values

        layout_request = count_layout_request(policy)
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        self.cr = cr
        mark_schema_captured = getattr(self.cr, "_mark_schema_captured", None)
        if callable(mark_schema_captured):
            mark_schema_captured()
        self.workspace = workspace
        self.storage_options = storage_options
        assay_names = tuple(dict.fromkeys(self.cr.assayFeats.columns))
        for assay_name in assay_names:
            validate_assay_name(assay_name)
        self.assayTypes = dict(assay_types or {})
        unknown = sorted(set(self.assayTypes).difference(assay_names))
        if unknown:
            raise ValueError(
                f"assay_types names assays that are not imported: {', '.join(unknown)}"
            )
        for name, assay_type in self.assayTypes.items():
            validate_assay_type(assay_type, assay=name)
        if lines_in_mem <= 0:
            raise ValueError("lines_in_mem must be positive")
        self.linesInMem = lines_in_mem
        # A destination that would be refused fails before the pass over
        # the counts; create_destination checks it again.
        check_destination(
            zarr_loc, overwrite=overwrite, storage_options=storage_options
        )
        ranges = assay_feature_ranges(self.cr.assayFeats)
        groups = np.full(self.cr.nFeatures, -1, dtype=np.int64)
        for code, spans in enumerate(ranges.values()):
            for start, end in spans:
                groups[start:end] = code
        # Row r of every per-cell array holds the reader's cell _cellOrder[r];
        # the reader sends each cell to its destination as it reads it.
        self._cellOrder: np.ndarray | None = None
        scan_options: dict[str, Any] = {}
        held = 0
        if self.shuffleCells:
            _require_shuffled_scan(self.cr)
            # A Matrix Market reader reads its cells for the shuffle with the
            # parse buffer that the import reserves.
            configure_lines = getattr(self.cr, "_set_sparse_import_lines_in_mem", None)
            if callable(configure_lines):
                configure_lines(self.linesInMem)
            # Drawing the order and its inverse runs beside the reader's arrays.
            check_shuffle_fits(
                self.cr.nCells,
                self._reader_resident_bytes(),
                self.resources.memoryBytes,
            )
            self._cellOrder = shuffled_order(self.cr.nCells, self.shuffleSeed)
            scan_options["destinationRows"] = destination_rows(self._cellOrder)
            held = 2 * self._cellOrder.nbytes
            # A Matrix Market reader sorts a feature-major file's entries into
            # rows as it reads its cells for the shuffle, beside its arrays and
            # the order of the cells and its inverse.
            self._admit_reader_preparation(self._reader_resident_bytes() + held)
        else:
            # Another import of the reader may hold its shuffled cells, which
            # this import, in the reader's order, never reads.
            self._release_other_shuffled_cells()
        # The cells on local disk before the scan; cells there after a
        # failure are this import's, which no later import may read.
        rows_before = getattr(self.cr, "_shuffledRows", None)
        try:
            try:
                scanned = self.cr.count_value_ranges(
                    self.resources.memoryBytes - held, groups, **scan_options
                )
            except MemoryBudgetError as error:
                if not held:
                    raise
                # The scan and its spill run beside the order of the cells and
                # its inverse, so the budget that passes them holds those too.
                raise memory_refusal(
                    f"Importing the counts of {self.cr.nCells} cells",
                    f"{error.item} beside the {held} bytes of its cell order",
                    held + error.needed_bytes,
                    int(self.resources.memoryBytes),
                ) from error
            value_ranges = dict(zip(ranges, scanned, strict=True))
            del scan_options, scanned
            # The rows that this import's counts write reads, which no other
            # import of the same reader may replace.
            self._shuffledStore = self.cr._shuffledRows if self.shuffleCells else None
            storage_dtypes = {
                name: count_storage_dtype(self.cr.matrix_dtype, value_range)
                for name, value_range in value_ranges.items()
            }
            logger.debug(f"Resolved Cell Ranger count storage dtypes={storage_dtypes}")
            self.values = resolve_import_values(
                values,
                value_ranges,
                lambda: self._first_cells(ranges),
                {name: self.assayTypes.get(name) for name in value_ranges},
                storage_dtypes,
            )
            # A preparation or a layout that does not fit fails here, before
            # the destination exists. A shuffled import's counts write reads
            # the shuffled cells, which need no sorting.
            if not self.shuffleCells:
                self._admit_reader_preparation(self._count_import_resident_bytes())
            self.policy = self._fit_count_layout(storage_dtypes, layout_request)
            # The destination must be empty, or with overwrite an unprepared store.
            self.z = create_destination(
                zarr_loc, overwrite=overwrite, storage_options=storage_options
            )
            # The import writes the new store on its threads.
            register_store_threads(self.z.store, self.resources.workers)
            cell_names = np.array(self.cr.cell_names())
            if self._cellOrder is not None:
                cell_names = cell_names[self._cellOrder]
            cell_group = create_cell_data(
                root=self.z,
                workspace=self.workspace,
                ids=cell_names,
                names=cell_names,
                profile=self.profile,
            )
            del cell_names
            if self._cellOrder is not None:
                from ..storage.cell_order import write_shuffled_cell_order

                # Each cell's row in the file, which a reader that drops cells
                # keeps with gaps. Written before the keys are planned, so a
                # source column of the same name is skipped.
                write_shuffled_cell_order(
                    cell_group,
                    self.cr._file_rows()[self._cellOrder],
                    self.shuffleSeed,
                    profile=self.profile,
                )
            for assay_name in assay_names:
                create_zarr_count_assay(
                    z=self.z,
                    assay_name=assay_name,
                    workspace=workspace,
                    n_cells=self.cr.nCells,
                    feat_ids=self.cr.feature_ids(assay_name),
                    feat_names=self.cr.feature_names(assay_name),
                    dtype=storage_dtypes[assay_name],
                    profile=self.profile,
                    policy=self.policy,
                    normalized_values=normalized_values_record(self.values[assay_name]),
                )
            self._write_reader_metadata(cell_group, assay_names)
        except BaseException:
            held_rows = getattr(self.cr, "_shuffledRows", None)
            if (
                self.shuffleCells
                and held_rows is not None
                and held_rows is not rows_before
            ):
                self.cr._release_shuffled_rows()
            raise

    def _release_other_shuffled_cells(self) -> None:
        """Release the shuffled cells that another import of the reader left.

        An import in the reader's order then reads the cells in that order.
        A reader that cannot shuffle holds none.
        """
        release = getattr(self.cr, "_release_shuffled_rows", None)
        if callable(release):
            release()

    def _first_cells(
        self, ranges: dict[str, tuple[tuple[int, int], ...]]
    ) -> dict[str, Any]:
        """Return the values of each assay's first cells, which decide what they are.

        They are the reader's first cells, whether or not the import shuffles
        them, so the shuffle never changes what the values are taken to be.
        """
        from ..utils.matrix_values import DETECTION_CELLS

        cells = self.cr._file_first_rows(DETECTION_CELLS, self.linesInMem)
        return {
            name: cells[:, np.concatenate([np.arange(a, b) for a, b in spans])]
            for name, spans in ranges.items()
        }

    def _assay_widths(self) -> dict[str, int]:
        """Return the feature count of each imported assay."""
        return {
            name: sum(end - start for start, end in spans)
            for name, spans in assay_feature_ranges(self.cr.assayFeats).items()
        }

    def _reader_resident_bytes(self) -> int:
        """Return the bytes of the arrays that the reader holds for an import."""
        reader_resident = getattr(self.cr, "_sparse_import_resident_bytes", None)
        return max(0, int(reader_resident())) if callable(reader_resident) else 0

    def _count_import_resident_bytes(self) -> int:
        """Return what the import holds from the start of its counts write to its end."""
        from ..storage.identity import CountSummary

        return int(
            self._reader_resident_bytes()
            + sum(
                CountSummary.nbytes_for(self.cr.nCells, width)
                for width in self._assay_widths().values()
            )
            # A shuffled import holds the order of its cells.
            + (0 if self._cellOrder is None else int(self._cellOrder.nbytes))
        )

    def _admit_reader_preparation(self, held: int) -> None:
        """Refuse a budget that cannot hold the reader's sorting of its entries into rows.

        A Matrix Market reader sorts a feature-major file's entries, and an
        unordered file's whose sorted rows were removed, into rows of cells
        before the counts write reads them, or as it reads its cells for a
        shuffle, beside the ``held`` bytes that the import holds then. The
        refusal names ``lines_in_mem`` halved until the sorting fits, when
        such a value exists.
        """
        preparation = getattr(self.cr, "_sparse_import_preparation_bytes", None)
        if not callable(preparation):
            return
        sorting = int(preparation(self.linesInMem))
        if not sorting:
            return
        budget = int(self.resources.memoryBytes)
        needed = held + sorting
        if needed <= budget:
            return
        remedy = None
        lines = self.linesInMem
        while lines > 1:
            lines //= 2
            if held + int(preparation(lines)) <= budget:
                remedy = f"pass lines_in_mem={lines}"
                break
        raise memory_refusal(
            f"Importing the counts of {self.cr.nCells} cells",
            "sorting the Matrix Market entries into rows of cells, "
            f"{self.linesInMem} lines at a time, beside the {held} bytes that "
            "the import holds",
            needed,
            budget,
            remedy=remedy,
        )

    def _count_import_requirements(
        self,
        storage_dtypes: Iterable[Any],
    ) -> tuple[int, Callable[[int], int], Callable[[int], int]]:
        """Return the resident, staging, and projection bytes of the counts write.

        The layout fit and the write plan the same import from these.
        """
        resident = self._count_import_resident_bytes()
        projection_value_bytes = int(
            max(
                [
                    self.cr.matrix_dtype.itemsize,
                    *(np.dtype(dtype).itemsize for dtype in storage_dtypes),
                ]
            )
        )

        def staging_bytes(rows: int) -> int:
            return int(self.cr.producer_staging_bytes(rows, self.linesInMem))

        def projection_bytes(rows: int) -> int:
            source_values = max(0, int(self.cr.max_window_nnz(rows)))
            return source_values * (
                projection_value_bytes
                + 3 * np.dtype(np.int64).itemsize
                + 2 * np.dtype(np.bool_).itemsize
            )

        return resident, staging_bytes, projection_bytes

    def _fit_count_layout(
        self,
        storage_dtypes: dict[str, Any],
        requested: CountMatrixPolicy | None,
    ) -> CountMatrixPolicy:
        """Return the ``requested`` count layout if its writes fit the budget.

        None fits the default layout to the budget.
        """
        from ..storage.sharding import fit_count_layout, sparse_counts_admission
        from .counts_t import counts_t_assays

        resident, staging_bytes, projection_bytes = self._count_import_requirements(
            storage_dtypes.values()
        )
        widths = self._assay_widths()
        return fit_count_layout(
            {name: (widths[name], dtype) for name, dtype in storage_dtypes.items()},
            nCells=self.cr.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays(tuple(storage_dtypes), self.assayTypes),
            admitCounts=sparse_counts_admission(
                nRows=self.cr.nCells,
                maxWindowNnz=self.cr.max_window_nnz,
                sourceDtype=self.cr.matrix_dtype,
                residentBytes=resident,
                producerStagingBytes=staging_bytes,
                extraProducerBytes=projection_bytes,
                alignedBatches=True,
            ),
            requested=requested,
        )

    def _write_reader_metadata(
        self,
        cell_group: Any,
        assay_names: tuple[str, ...],
    ) -> None:
        from ..metadata.membership import reserved_membership_columns
        from ..storage.metadata_keys import metadata_column_keys
        from ..storage.types import as_zarr_group
        from ._store import keyed_metadata_columns, write_metadata_column

        # These readers hold their metadata in memory, so listing the columns
        # to plan their keys reads nothing more from the source.
        cell_columns = getattr(self.cr, "get_cell_columns", None)
        if callable(cell_columns):
            columns = list(cell_columns())
            # The membership column name of every imported assay is reserved.
            reserved = reserved_membership_columns(assay_names)
            keys = metadata_column_keys(
                (name for name, _values in columns),
                taken=[*cell_group.keys(), *reserved],
            )
            for key, (name, raw_values) in keyed_metadata_columns(
                ((name, (name, values)) for name, values in columns),
                keys,
                "cell",
                membership=reserved,
            ):
                values = np.asarray(raw_values)
                if values.ndim != 1 or values.size != self.cr.nCells:
                    raise ValueError(
                        f"Cell metadata column {name!r} has shape {values.shape}; "
                        f"expected ({self.cr.nCells},)"
                    )
                if self._cellOrder is not None:
                    values = values[self._cellOrder]
                write_metadata_column(cell_group, key, values, profile=self.profile)

        feature_columns = getattr(self.cr, "get_feature_columns", None)
        if not callable(feature_columns):
            return
        ranges = assay_feature_ranges(self.cr.assayFeats)
        targets: dict[str, tuple[Any, np.ndarray]] = {}
        for assay_name in assay_names:
            indexes = np.concatenate(
                [
                    np.arange(start, end, dtype=np.int64)
                    for start, end in ranges[assay_name]
                ]
            )
            group_path = (
                f"{assay_name}/featureData"
                if self.workspace is None
                else f"{self.workspace}/{assay_name}/featureData"
            )
            targets[assay_name] = (
                as_zarr_group(self.z[group_path], name=group_path),
                indexes,
            )
        columns = []
        for name, raw_values in feature_columns():
            values = np.asarray(raw_values)
            if values.ndim != 1 or values.size != self.cr.nFeatures:
                raise ValueError(
                    f"Feature metadata column {name!r} has shape {values.shape}; "
                    f"expected ({self.cr.nFeatures},)"
                )
            # A reference column that is empty for every feature is never
            # written, so it must not claim a key or report a rename.
            if values.dtype.kind == "O" and all(value is None for value in values):
                continue
            columns.append((name, values))
        keys = metadata_column_keys(
            (name for name, _values in columns),
            taken={key for group, _indexes in targets.values() for key in group.keys()},
        )
        for key, (name, values) in keyed_metadata_columns(
            ((name, (name, values)) for name, values in columns),
            keys,
            "feature",
        ):
            for group, indexes in targets.values():
                selected = values[indexes]
                if selected.dtype.kind == "O" and all(
                    value is None for value in selected
                ):
                    # A 10x feature reference describes only Feature Barcode
                    # features, so its columns do not reach other assays.
                    continue
                write_metadata_column(group, key, selected, profile=self.profile)

    @staticmethod
    def _prep_feat_index_offset(
        ranges: dict[str, tuple[tuple[int, int], ...]],
    ) -> dict[str, list[int]]:
        feat_offset: dict[str, list[int]] = {}
        for i in ranges:
            feat_offset[i] = []
            lv = 0
            for j in ranges[i]:
                feat_offset[i].append(-j[0] + lv)
                lv += j[1] - j[0]
        return feat_offset

    def dump(self, batch_size: int | None = None) -> None:
        """Writes the count values into the Zarr matrix.

        A shuffled import reads its shuffled cells once, deleting them as
        it goes, and removes them once the counts are written or the write
        fails, so it writes its counts once. An import in the reader's order
        removes the shuffled cells that another import of the reader left.

        Args:
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default.

        Raises:
            ValueError: If ``batch_size`` is not positive.
            RuntimeError: If the shuffled cells of this import were already
                written, or another import of the reader replaced them.
            MemoryBudgetError: If the reader must sort a Matrix Market
                file's entries into rows again, as after another import of
                it removed them, and ``mem_budget`` cannot hold that.

        Returns:
            None
        """
        from scipy.sparse import coo_matrix

        from ..storage.identity import CountSummary, finalize_counts
        from ..storage.schema import load_count_array
        from ..storage.sharding import (
            SparseShardBuffer,
            SparseWriteBand,
            resolve_sparse_import_batch,
            write_sparse_bands,
        )
        from ..utils.arrays import sparse_matrix_bytes
        from ..utils.matrix_values import linear_coo

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.shuffleCells:
            # Another import of the reader, built after this one, may have
            # left its shuffled cells; this import writes the reader's order.
            self._release_other_shuffled_cells()
        elif (
            self._shuffledStore is None
            or self.cr._shuffledRows is not self._shuffledStore
        ):
            raise RuntimeError(
                "The shuffled cells of this import were written once and removed, "
                f"or another import of the reader replaced them; create a new "
                f"{type(self).__name__} to import the cells again"
            )
        input_ranges = assay_feature_ranges(self.cr.assayFeats)
        stores = {
            assay: load_count_array(self.z, assay, self.workspace)
            for assay in input_ranges
        }
        buffers = {assay: SparseShardBuffer(store) for assay, store in stores.items()}
        summaries = {assay: CountSummary(store) for assay, store in stores.items()}
        feat_offset = self._prep_feat_index_offset(input_ranges)
        configure_lines = getattr(
            self.cr,
            "_set_sparse_import_lines_in_mem",
            None,
        )
        prepare = getattr(self.cr, "_prepare_sparse_import", None)
        release = getattr(self.cr, "_release_sparse_import", None)

        try:
            if callable(configure_lines):
                configure_lines(self.linesInMem)
            # The reader prepares its rows before the write is planned, alone
            # beside what the import holds; a shuffled import reads its
            # shuffled cells instead.
            if not self.shuffleCells:
                self._admit_reader_preparation(self._count_import_resident_bytes())
            if callable(prepare):
                prepare()
            (
                resident_reader_bytes,
                staging_bytes,
                projection_bytes,
            ) = self._count_import_requirements(
                store.dtype for store in stores.values()
            )
            plan = resolve_sparse_import_batch(
                tuple(stores.values()),
                nRows=self.cr.nCells,
                resources=self.resources,
                maxWindowNnz=self.cr.max_window_nnz,
                sourceDtype=self.cr.matrix_dtype,
                batchRows=batch_size,
                residentBytes=resident_reader_bytes,
                producerStagingBytes=staging_bytes,
                extraProducerBytes=projection_bytes,
                alignedBatches=True,
            )
            self._lastImportPlan = plan
            resolved_batch_rows = plan.batchRows
            logger.debug(
                f"Resolved Cell Ranger source batch rows={resolved_batch_rows} "
                f"write_tasks={plan.writeTasks}"
            )

            def writes() -> Iterator[SparseWriteBand]:
                n_chunks = (
                    self.cr.nCells + resolved_batch_rows - 1
                ) // resolved_batch_rows
                source = iter_progress(
                    self.cr.consume(resolved_batch_rows, self.linesInMem),
                    total=n_chunks,
                    desc="Writing counts",
                )
                for matrix in source:
                    chunk = matrix.tocoo(copy=False)
                    source_bytes = sparse_matrix_bytes(matrix, chunk)
                    for assay in input_ranges:
                        selected = np.zeros(chunk.col.shape[0], dtype=bool)
                        columns = chunk.col.copy()
                        for bounds, offset in zip(
                            input_ranges[assay],
                            feat_offset[assay],
                            strict=True,
                        ):
                            inside = (chunk.col >= bounds[0]) & (chunk.col < bounds[1])
                            columns[inside] += offset
                            selected |= inside
                        projected = coo_matrix(
                            (
                                chunk.data[selected],
                                (chunk.row[selected], columns[selected]),
                            ),
                            shape=(chunk.shape[0], buffers[assay].nColumns),
                        )
                        projected = linear_coo(
                            projected,
                            self.values[assay].kind,
                            stores[assay].dtype,
                            assay=assay,
                        )
                        for band in buffers[assay].add(projected):
                            producer_bytes = (
                                source_bytes
                                + columns.nbytes
                                + 2 * selected.nbytes
                                + sparse_matrix_bytes(projected)
                                + sum(item.residentBytes for item in buffers.values())
                            )
                            yield SparseWriteBand(
                                stores[assay],
                                band,
                                producer_bytes,
                            )
                            # The write holds the band it was handed; keeping
                            # it here would hold its values beside the rest.
                            del band
                if self.cr.nCells:
                    del matrix, chunk, selected, columns, inside, projected
                for assay, buffer in buffers.items():
                    for band in buffer.finish():
                        producer_bytes = sum(
                            item.residentBytes for item in buffers.values()
                        )
                        yield SparseWriteBand(
                            stores[assay],
                            band,
                            producer_bytes,
                        )

            write_sparse_bands(
                writes(),
                resources=self.resources,
                residentBytes=resident_reader_bytes,
                producerReserveBytes=plan.producerReserveBytes,
                total=plan.writeTasks,
                io=self.io,
                countSummaries={
                    stores[name].path: summary for name, summary in summaries.items()
                },
            )
            # Finishing each buffer checked that the source supplied every row.
            logger.info(
                f"Wrote {self.cr.nCells} cells and "
                f"{sum(buffer.nColumns for buffer in buffers.values())} features "
                f"from {self.source_name} to {len(stores)} assay(s)"
            )
            for assay, summary in summaries.items():
                finalize_counts(stores[assay], summary=summary)
            if self.shuffleCells:
                # The shuffled cells were read once; countsT reads counts.
                self.cr._release_shuffled_rows()
                self._shuffledStore = None
            from .counts_t import finalize_writer_counts_t_many

            finalize_writer_counts_t_many(
                self.z,
                tuple(stores),
                self.workspace,
                assay_types=self.assayTypes,
                resources=self.resources,
                profile=self.profile,
                io=self.io,
            )
        finally:
            if callable(release):
                release()
            if self.shuffleCells:
                self.cr._release_shuffled_rows()
                self._shuffledStore = None


def _require_shuffled_scan(reader: CrReader) -> None:
    """Refuse a reader that cannot shuffle its cells or name their rows in the file.

    Raises:
        TypeError: If the reader's ``count_value_ranges`` takes no
            ``destinationRows``, as readers written before cells could be
            shuffled do.
        ValueError: If the reader's ``_file_rows`` does not give one row for
            each selected cell.
    """
    import inspect

    parameters = inspect.signature(reader.count_value_ranges).parameters
    if "destinationRows" not in parameters:
        raise TypeError(
            f"{type(reader).__name__}.count_value_ranges takes no destinationRows, "
            "so its cells cannot be shuffled; pass shuffle_cells=False, or give "
            "the reader the keyword-only destinationRows that CrReader describes"
        )
    rows = np.asarray(reader._file_rows())
    if rows.shape != (reader.nCells,) or rows.dtype.kind not in "iu":
        raise ValueError(
            f"{type(reader).__name__}._file_rows gave dtype {rows.dtype} and shape "
            f"{rows.shape}; it must give one integer file row for each of the "
            f"{reader.nCells} selected cells"
        )


MtxToZarr = CrToZarr
