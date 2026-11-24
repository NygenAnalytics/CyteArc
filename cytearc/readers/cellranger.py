import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.progress import iter_progress
from ..utils.files import open_binary
from ._contigs import contig_chunks, header_fields
from ..utils.delimited import stripped_text
from ._assay_names import (
    AUTO_ASSAY_NAMES,
    auto_name_feat_table,
    make_feat_table_from_types,
)
from ._text import (
    DuplicateFeatureIds,
    as_text,
    dataset_place,
    duplicate_feature_ids_option,
    require_unique_identifiers,
    unique_feature_identifiers,
    utf8_texts,
)
from ..utils.arrays import (
    assay_feature_ranges,
    csr_to_coo,
    cumulative_nnz,
    has_duplicates,
    max_window_nnz,
)
from ..utils.count_values import (
    CountValueRange,
    compressed_count_ranges,
    longest_vector_refusal,
    new_count_ranges,
)
from ..utils.memory_budget import MemoryBudgetError
from ..utils.arguments import bool_argument
from ._row_spill import (
    RowSpillPlan,
    admitted_spill_plan,
    checked_destinations,
    row_spill_plan,
    sample_rows,
    spill_rows,
)
from ._sparse import SparseRowStore, encoded_entry_bytes

# What refusals call the values of each dataset of a 10x HDF5 file, by its
# ``grpNames`` key.
_H5_VALUES = {
    "cell_names": "Cell Ranger barcodes",
    "feature_ids": "Cell Ranger feature IDs",
    "feature_names": "Cell Ranger feature names",
    "feature_types": "Cell Ranger feature types",
}


class CrReader(ABC):
    """A class to read in CellRanger (Cr) data.

    Args:
        grp_names (Dict): A dictionary that specifies where to find the matrix, features and barcodes.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
    """

    #: The rows of an import that shuffles cells, in the import's order, on
    #: local disk; ``consume`` yields them once, deleting them as it goes.
    _shuffledRows: SparseRowStore | None = None
    #: The plan of the spill that wrote them, of each budget of the spill.
    _shuffledPlanned: Callable[[int], RowSpillPlan] | None = None

    def __init__(self, grp_names: dict[str, Any]) -> None:
        self.autoNames = dict(AUTO_ASSAY_NAMES)
        self._featureTypeOverrides: dict[int, str] = {}
        self._schemaCaptured = False
        self.grpNames: dict[str, Any] = grp_names
        self.nFeatures: int = len(self.feature_names())
        self.nCells: int = len(self.cell_names())
        self.assayFeats = self._make_feat_table()
        self._auto_rename_assay_names()

    @abstractmethod
    def _handle_version(self) -> dict[str, Any]:
        """Return the dataset locations that the subclass passes as ``grp_names``."""

    @abstractmethod
    def _read_dataset(self, key: str) -> list[Any]:
        """Return the values of the ``grpNames`` dataset ``key``."""

    @abstractmethod
    def consume(
        self, batch_size: int, lines_in_mem: int
    ) -> Generator[coo_matrix, None, None]:
        """Yield CSR matrix chunks of cell rows.

        Args:
            batch_size: Number of cells per yielded chunk.
            lines_in_mem: MTX lines buffered in memory (Matrix Market readers
                only).

        Yields:
            scipy.sparse.coo_matrix chunks.
        """

    def max_window_nnz(self, window_rows: int) -> int:
        """Bound nnz in any source row window."""
        if window_rows <= 0:
            raise ValueError("window_rows must be positive")
        return min(window_rows, self.nCells) * self.nFeatures

    @property
    @abstractmethod
    def matrix_dtype(self) -> np.dtype[Any]:
        """Return the dtype in which ``consume`` yields count values."""

    @abstractmethod
    def count_value_ranges(
        self,
        maxBytes: int,
        featureGroups: np.ndarray | None = None,
        *,
        destinationRows: np.ndarray | None = None,
    ) -> list[CountValueRange]:
        """Return the range of the canonical values of each group of features.

        The ranges cover the selected cells, with duplicate coordinates summed
        as the import stores them. Pass a range to
        :func:`~cytearc.storage.count_dtype.count_storage_dtype` to resolve the
        storage dtype of its group's counts.

        With ``destinationRows``, the reader also sends each selected cell to
        its destination row in temporary files on local disk, for an import
        that shuffles cells, and ``consume`` then yields the cells once in
        destination order. ``maxBytes`` also holds what the files' writer
        holds.

        Args:
            maxBytes: Memory available to the scan, beside the arrays that
                ``_sparse_import_resident_bytes`` reports, which an import
                charges itself.
            featureGroups: Group, numbered from zero, of each feature, such as
                the assay that stores it. None puts every feature in one
                group.
            destinationRows: The destination row of each selected cell, a
                permutation, or None to keep the cells in place.
        """

    def _file_rows(self) -> np.ndarray:
        """Return each selected cell's row in the file, from 0, as int64.

        A reader that keeps every cell of the file returns 0 to ``nCells -
        1``; one that keeps only some, such as :class:`CrH5Reader` with
        ``is_filtered=False``, returns the rows of those it keeps, in order.
        An import that shuffles cells stores them as ``sourceRow``.
        """
        return np.arange(self.nCells, dtype=np.int64)

    def _file_first_rows(self, count: int, lines_in_mem: int) -> csr_matrix:
        """Return the first ``count`` selected cells in the file's order.

        They decide what the values are, whether or not an import shuffles
        the cells. A reader that shuffles reads them from the file.
        """
        batch = next(iter(self.consume(max(1, int(count)), lines_in_mem)), None)
        if batch is None:
            return csr_matrix((0, self.nFeatures), dtype=self.matrix_dtype)
        return csr_matrix(batch.tocsr()[: int(count)])

    def _spill_rows(
        self,
        destinationRows: np.ndarray,
        maxBytes: int,
        fill: Callable[[int, Callable[[int, Any], None]], Any],
        *,
        entries: int | None,
        longest: int,
        entry_bytes: float | None,
        operation: str,
        temp_dir: str | Path | None,
        expected_entries: int | None = None,
    ) -> Any:
        """Send the selected cells to their destination rows as a scan reads them.

        ``fill(scanBytes, consumer)`` scans every selected cell once within
        ``scanBytes``, sends each block of consecutive cells to
        ``consumer(first, block)``, and returns what the caller keeps, such
        as the scan's value ranges. ``entries`` are the cells' stored values,
        or None for dense rows, ``longest`` the most values of one cell, and
        ``entry_bytes`` what the file's first cells take on disk per value,
        as :func:`~cytearc.readers._sparse.encoded_entry_bytes` measures it.
        Dense rows pass ``expected_entries``, the values they hold beside
        zeros as :func:`~cytearc.readers._row_spill.dense_spill_estimate`
        estimates them, so that the free disk is checked before the spill.
        The finalized store then holds the rows that ``consume`` yields.

        Raises:
            ValueError: If ``destinationRows`` is not a permutation of the cells.
            MemoryBudgetError: If ``maxBytes`` cannot hold the spill of the
                longest cell, naming the least budget that holds it.
            OSError: If the temporary disk lacks room for the rows.
        """
        self._release_shuffled_rows()
        destinations = checked_destinations(destinationRows, self.nCells)

        def planned(budget: int) -> RowSpillPlan:
            return row_spill_plan(
                destinations,
                (self.nCells, self.nFeatures),
                self.matrix_dtype,
                entries=entries,
                longest=longest,
                maxBytes=budget,
            )

        plan = admitted_spill_plan(planned, maxBytes)
        share = plan.maxBytes - plan.ownerBytes
        store, result = spill_rows(
            plan,
            lambda consumer: fill(plan.scan_bytes(share, 0, self.nCells), consumer),
            operation=operation,
            entry_bytes=entry_bytes,
            buffer_bytes=plan.buffer_bytes(share, 0, self.nCells),
            dense=entries is None,
            temp_dir=temp_dir,
            entries=expected_entries,
        )
        self._shuffledRows = store
        self._shuffledPlanned = planned
        return result

    def _consume_shuffled(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Yield the shuffled cells once, deleting each bucket of them once read.

        The next bucket loads while the current one is yielded.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        store = self._shuffledRows
        assert store is not None
        for batch in store.iter_rows(
            0, self.nCells, batch_size, read_ahead=True, delete=True
        ):
            yield csr_to_coo(batch)

    def _shuffled_staging_bytes(self, batch_size: int) -> int:
        """Return what reading a batch of shuffled cells holds beside the batch.

        The batch's row pointers, and a bucket of rows with the next one
        loading ahead.
        """
        store = self._shuffledRows
        assert store is not None
        rows = min(max(1, int(batch_size)), self.nCells)
        return int((rows + 1) * 3 * np.dtype(np.int64).itemsize) + store.read_bytes(
            read_ahead=True
        )

    def _shuffled_staging_within(self, maxBytes: int) -> Callable[[int], int]:
        """Return what reading a batch of shuffled cells would hold beside the batch, had their spill been planned within ``maxBytes``.

        The spill sizes its buckets from its budget, and the counts write
        reads a bucket at a time, so a budget other than the spill's would
        read other buckets of the same cells, whose row pointers give them
        without reading the cells, as :meth:`_shuffled_staging_bytes` counts
        them for the buckets that the spill made.

        Raises:
            MemoryBudgetError: If ``maxBytes`` cannot hold the spill, as its
                plan refuses it.
        """
        store, planned = self._shuffledRows, self._shuffledPlanned
        assert store is not None and planned is not None
        plan = planned(int(maxBytes))
        reading = SparseRowStore.planned_read_bytes(
            store.indptr,
            bucket_rows=plan.bucketRows,
            split_bytes=2 * plan.readBytes,
            dtype=store.dtype,
            read_ahead=True,
        )

        def staging(batch_size: int) -> int:
            rows = min(max(1, int(batch_size)), self.nCells)
            return (rows + 1) * 3 * np.dtype(np.int64).itemsize + reading

        return staging

    def _release_shuffled_rows(self) -> None:
        """Remove the shuffled cells of an import, once written or given up."""
        if self._shuffledRows is not None:
            self._shuffledRows.close()
        self._shuffledRows = None
        self._shuffledPlanned = None

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Return source-reader bytes retained while a matrix batch is yielded."""
        valid_idx = getattr(self, "validBarcodeIdx", None)
        return int(valid_idx.nbytes) if isinstance(valid_idx, np.ndarray) else 0

    def _prepare_sparse_import(self) -> None:
        """Prepare optional reader-owned state used by import planning."""

    def _release_sparse_import(self) -> None:
        """Release optional reader-owned state created for one import."""

    def _sparse_import_resident_bytes(self) -> int:
        """Return reader arrays retained for the duration of an import."""
        return 0

    def _subset_by_assay(self, v: list[Any], assay: str | None) -> list[Any]:
        if assay is None:
            return v
        ranges = assay_feature_ranges(self.assayFeats)
        if assay not in ranges:
            raise ValueError(f"ERROR: Assay ID {assay} is not valid")
        return [value for start, end in ranges[assay] for value in v[start:end]]

    @staticmethod
    def _make_feat_table_from_types(feature_types: Sequence[str]) -> pd.DataFrame:
        return make_feat_table_from_types(feature_types)

    def _make_feat_table(self) -> pd.DataFrame:
        return self._make_feat_table_from_types(self.feature_types())

    def _auto_named_feat_table(self, assay_feats: pd.DataFrame) -> pd.DataFrame:
        return auto_name_feat_table(assay_feats, self.autoNames)

    def _auto_rename_assay_names(self) -> None:
        self.assayFeats = self._auto_named_feat_table(self.assayFeats)

    def _mark_schema_captured(self) -> None:
        self._schemaCaptured = True

    def reclassify_features(
        self,
        indexes: Sequence[int],
        feature_type: str,
        *,
        require_previous: str | None = "Antibody Capture",
    ) -> None:
        """Reclassify global feature rows before a writer captures the schema."""
        if self._schemaCaptured:
            raise RuntimeError(
                "Features cannot be reclassified after a writer captures the schema"
            )
        if not isinstance(feature_type, str) or feature_type == "":
            raise ValueError("feature_type must be a non-empty string")
        if isinstance(indexes, str):
            raise TypeError("indexes must be a sequence of integer feature indexes")
        index_array = np.asarray(indexes)
        if index_array.ndim != 1:
            raise ValueError("indexes must be one-dimensional")
        if index_array.size == 0:
            raise ValueError("indexes must contain at least one feature index")
        if not np.issubdtype(index_array.dtype, np.integer):
            raise TypeError("indexes must contain only integers")
        index_array = index_array.astype(np.int64, copy=False)
        if has_duplicates(index_array):
            raise ValueError("indexes must contain unique feature indexes")
        if np.any(index_array < 0) or np.any(index_array >= self.nFeatures):
            raise IndexError("indexes contains an out-of-range feature index")

        conflicting = [
            int(index)
            for index in index_array
            if index in self._featureTypeOverrides
            and self._featureTypeOverrides[int(index)] != feature_type
        ]
        if conflicting:
            raise ValueError(
                "Features already have a conflicting reclassification: "
                + ", ".join(map(str, conflicting))
            )

        current_types = self.feature_types()
        pending = np.asarray(
            [
                index
                for index in index_array
                if current_types[int(index)] != feature_type
            ],
            dtype=np.int64,
        )
        if pending.size == 0:
            return None
        if require_previous is not None:
            invalid = [
                int(index)
                for index in pending
                if current_types[int(index)] != require_previous
            ]
            if invalid:
                raise ValueError(
                    f"Features must currently have type {require_previous!r}: "
                    + ", ".join(map(str, invalid))
                )

        updated_types = list(current_types)
        updated_overrides = dict(self._featureTypeOverrides)
        for index in pending:
            updated_types[int(index)] = feature_type
            updated_overrides[int(index)] = feature_type
        updated_table = self._auto_named_feat_table(
            self._make_feat_table_from_types(updated_types)
        )

        self._featureTypeOverrides = updated_overrides
        self.assayFeats = updated_table
        return None

    def rename_assays(self, name_map: dict[str, str]) -> None:
        """Renames specified assays in the Reader.

        Args:
            name_map: A Dictionary containing current name as key and new name as value.
        """
        self.assayFeats.rename(columns=name_map, inplace=True)

    def feature_ids(self, assay: str | None = None) -> list[str]:
        """Returns a list of feature IDs in a specified assay.

        Args:
            assay: Select which assay to retrieve feature IDs from.
        """
        return self._subset_by_assay(self._read_dataset("feature_ids"), assay)

    def feature_names(self, assay: str | None = None) -> list[str]:
        """Returns a list of features in the dataset.

        Args:
            assay: Select which assay to retrieve features from.
        """
        return self._subset_by_assay(self._read_dataset("feature_names"), assay)

    def feature_types(self) -> list[str]:
        """Returns a list of feature types in the dataset."""
        if self.grpNames["feature_types"] is not None:
            feature_types = list(self._read_dataset("feature_types"))
        else:
            feature_types = []
        if not feature_types:
            default_name = list(self.autoNames.keys())[0]
            feature_types = [default_name for _ in range(self.nFeatures)]
        for index, feature_type in self._featureTypeOverrides.items():
            feature_types[index] = feature_type
        return feature_types

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        return self._read_dataset("cell_names")

    def _cell_names_source(self) -> Any:
        """Return the selected cells' names, as a cell table's write reads them a block of rows at a time.

        This is an array that the reader keeps, or a source that reads them
        from the file, so the write holds no copy of every name; a reader
        that keeps neither returns them as a new array.
        """
        return np.asarray(self.cell_names())

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield optional cell metadata columns supplied by the reader."""
        yield from ()

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield source feature types and optional feature metadata."""
        yield "feature_type", np.asarray(self.feature_types(), dtype=object)


class CrH5Reader(CrReader):
    # noinspection PyUnresolvedReferences
    """A class to read in CellRanger (Cr) data, in the form of an H5 file.

    Subclass of CrReader.

    Args:
        h5_fn: File name for the h5 file.
        is_filtered: Whether the barcodes are already filtered. When False,
            only barcodes whose total count exceeds ``filtering_cutoff`` are
            kept.
        filtering_cutoff: Total count that a barcode must exceed to be kept
            when ``is_filtered`` is False.
        duplicate_feature_ids: ``"error"`` rejects repeated feature IDs.
            ``"make_unique"`` keeps the first occurrence of each ID and
            renames each later one, in file order and across assays, by
            appending ``-1``, ``-2``, and so on (skipping names that are
            already IDs), as AnnData's ``var_names_make_unique`` does, and
            logs a warning. Feature names are never changed, and repeated
            barcodes always raise.
        temp_dir: Parent directory of the temporary files of an import that
            shuffles cells. None uses the system temporary directory.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
        h5obj: A File object from the h5py package.
        grp: Current active group in the hierarchy.
        duplicateFeatureIds: How repeated feature IDs are treated.

    Raises:
        ValueError: If a barcode, feature ID, feature name or feature type is
            not UTF-8 text, or a barcode or feature ID is missing, blank or
            repeated. The refusal names the dataset and the rows that hold
            them, except for a repeat.
    """

    def __init__(
        self,
        h5_fn: str,
        is_filtered: bool = True,
        filtering_cutoff: int = 500,
        *,
        duplicate_feature_ids: DuplicateFeatureIds = "error",
        temp_dir: str | Path | None = None,
    ) -> None:
        is_filtered = bool_argument(is_filtered, "is_filtered")
        if filtering_cutoff < 0:
            raise ValueError("filtering_cutoff cannot be negative")
        self.duplicateFeatureIds = duplicate_feature_ids_option(duplicate_feature_ids)
        self._tempDir = temp_dir
        self.h5obj: h5py.File = h5py.File(h5_fn, mode="r")
        self.grp: h5py.Group
        self.validBarcodeIdx: np.ndarray | None = None
        self._indptrCache: np.ndarray | None = None
        self._cumulativeRowNnz: np.ndarray | None = None
        self._featureIds: list[str] | None = None
        try:
            # Decodes every barcode, feature name and feature type, refusing
            # text that is not UTF-8.
            super().__init__(self._handle_version())
            require_unique_identifiers(
                self.cell_names(),
                _H5_VALUES["cell_names"],
                place=self._dataset_place("cell_names"),
            )
            self._featureIds = [
                as_text(value)
                for value in unique_feature_identifiers(
                    np.asarray(self._read_dataset("feature_ids"), dtype=object),
                    _H5_VALUES["feature_ids"],
                    self.duplicateFeatureIds,
                    place=self._dataset_place("feature_ids"),
                )
            ]
            if is_filtered:
                self.validBarcodeIdx = np.arange(self.nCells)
            else:
                self.validBarcodeIdx = self._get_valid_barcodes(filtering_cutoff)
            self.nCells = len(self.validBarcodeIdx)
        except BaseException:
            self.h5obj.close()
            raise

    def _handle_version(self) -> dict[str, str | None]:
        root_keys = list(self.h5obj.keys())
        if "matrix" in root_keys:
            root_key = "matrix"
        else:
            # Cell Ranger 2 keeps one group per genome at the root.
            genomes = [
                key for key in root_keys if isinstance(self.h5obj[key], h5py.Group)
            ]
            if len(genomes) != 1:
                raise ValueError(
                    "Cell Ranger HDF5 input needs a `matrix` group or exactly one "
                    f"genome group; found: {', '.join(genomes) or 'none'}"
                )
            root_key = genomes[0]
        self.grp = self.h5obj[root_key]
        if root_key == "matrix":
            grps: dict[str, str | None] = {
                "feature_ids": "features/id",
                "feature_names": "features/name",
                "feature_types": "features/feature_type",
                "cell_names": "barcodes",
            }
        else:
            grps = {
                "feature_ids": "genes",
                "feature_names": "gene_names",
                "feature_types": None,
                "cell_names": "barcodes",
            }
        return grps

    def _get_valid_barcodes(
        self, filtering_cutoff: int, batch_size: int = 1000
    ) -> np.ndarray:
        indptr = self._source_indptr()
        n_barcodes = len(indptr) - 1
        # One open dataset, whose chunk cache serves consecutive batches.
        data_node = self.grp["data"]
        if int(indptr[-1]) != int(data_node.shape[0]):
            raise ValueError("Cell Ranger matrix pointers do not match its data")
        valid = np.zeros(n_barcodes, dtype=bool)
        for start in iter_progress(
            range(0, n_barcodes, batch_size),
            desc="Filtering out background barcodes",
        ):
            stop = min(start + batch_size, n_barcodes)
            data = np.asarray(data_node[indptr[start] : indptr[stop]])
            rows = np.repeat(
                np.arange(stop - start),
                np.diff(indptr[start : stop + 1]),
            )
            totals = np.bincount(rows, weights=data, minlength=stop - start)
            valid[start:stop] = totals > filtering_cutoff
        return np.flatnonzero(valid)

    def _source_indptr(self) -> np.ndarray:
        if self._indptrCache is None:
            self._indptrCache = np.asarray(self.grp["indptr"][:])
        return self._indptrCache

    def _selected_cumulative_nnz(self) -> np.ndarray:
        if self._cumulativeRowNnz is None:
            valid_idx = self.validBarcodeIdx
            assert valid_idx is not None
            self._cumulativeRowNnz = cumulative_nnz(
                np.diff(self._source_indptr())[valid_idx]
            )
        return self._cumulativeRowNnz

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        """Return the stored dtype in native byte order, with float16 as float32.

        SciPy sparse matrices hold neither float16 nor another byte order, so
        ``consume`` reads the values in this dtype.
        """
        dtype: np.dtype[Any] = np.dtype(self.grp["data"].dtype).newbyteorder("=")
        return np.dtype(np.float32) if dtype == np.float16 else dtype

    def count_value_ranges(
        self,
        maxBytes: int,
        featureGroups: np.ndarray | None = None,
        *,
        destinationRows: np.ndarray | None = None,
    ) -> list[CountValueRange]:
        """Return the range of the canonical values of each group of features.

        Each barcode is one compressed vector of the matrix. Selected barcodes
        whose values are contiguous in the matrix are scanned together, as
        ``consume`` reads them; the barcodes between them hold no values.
        With ``destinationRows``, the scan also sends each selected barcode
        to its destination row as it reads it, so an import that shuffles
        cells reads the file once.

        Raises:
            MemoryError: If one barcode holds more values than fit in
                ``maxBytes``, or with ``destinationRows`` its spill does not.
            OSError: If the temporary disk lacks room for the shuffled cells.
            ValueError: If a value is NaN or infinite, or ``destinationRows``
                is not a permutation of the selected barcodes.
        """
        value_ranges = new_count_ranges(featureGroups)
        if destinationRows is None:
            self._scan_runs(maxBytes, featureGroups, value_ranges, None)
            return value_ranges
        cumulative = self._selected_cumulative_nnz()
        rows = sample_rows(cumulative)
        self._spill_rows(
            destinationRows,
            maxBytes,
            lambda scan_bytes, consumer: self._scan_runs(
                scan_bytes, featureGroups, value_ranges, consumer
            ),
            entries=int(cumulative[-1]),
            longest=int(np.diff(cumulative).max(initial=0)),
            entry_bytes=(
                encoded_entry_bytes(self._file_first_rows(rows, 0)) if rows else None
            ),
            operation="shuffled 10x HDF5 import",
            temp_dir=self._tempDir,
        )
        return value_ranges

    def _scan_runs(
        self,
        maxBytes: int,
        featureGroups: np.ndarray | None,
        value_ranges: list[CountValueRange],
        consumer: Callable[[int, Any], None] | None,
    ) -> None:
        """Scan the runs of selected barcodes whose values are contiguous.

        ``consumer`` receives each block of selected barcodes as the scan
        reads it, by the position of its first barcode among the selected
        ones; the barcodes between them, which hold no values, are left out.

        Raises:
            MemoryBudgetError: If ``maxBytes`` cannot hold one selected
                barcode. It names the budget of the longest selected barcode,
                whichever run holds it, so that budget scans every run.
        """
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        if valid_idx.size == 0:
            return
        indptr = self._source_indptr()
        gaps = indptr[valid_idx[1:]] != indptr[valid_idx[:-1] + 1]
        bounds = np.r_[0, np.flatnonzero(gaps) + 1, valid_idx.size]
        try:
            for left, right in zip(bounds[:-1], bounds[1:]):
                compressed_count_ranges(
                    indptr[int(valid_idx[left]) : int(valid_idx[right - 1]) + 2],
                    self.grp["indices"],
                    self.grp["data"],
                    minorSize=self.nFeatures,
                    maxBytes=maxBytes,
                    groups=featureGroups,
                    valueRanges=value_ranges,
                    consumer=(
                        None
                        if consumer is None
                        else _selected_barcodes(
                            consumer, valid_idx[int(left) : int(right)], int(left)
                        )
                    ),
                )
        except MemoryBudgetError as error:
            # A run's scan names the longest barcode of that run; a later run
            # can hold a longer one.
            longest = int(np.diff(self._selected_cumulative_nnz()).max(initial=0))
            raise longest_vector_refusal(longest, maxBytes) from error

    def _read_dataset(self, key: str) -> list[str]:
        # Construction resolves repeated feature IDs once, so every caller
        # reads the IDs that the import stores.
        if key == "feature_ids" and self._featureIds is not None:
            return list(self._featureIds)
        grp_key = self.grpNames[key]
        return utf8_texts(
            self.grp[grp_key][:],
            f"{_H5_VALUES[key]} must be UTF-8 text",
            self._dataset_place(key),
        )

    def _dataset_place(self, key: str) -> str:
        """Name the dataset of ``grpNames`` key ``key`` as refusals name it."""
        name = self.grp[self.grpNames[key]].name
        return dataset_place(self.h5obj.filename, name.lstrip("/"))

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        vals = np.array(self._read_dataset("cell_names"))
        if self.validBarcodeIdx is not None:
            vals = vals[self.validBarcodeIdx]
        return list(vals)

    def _cell_names_source(self) -> Any:
        """Return the selected barcodes as a source that reads them from the file a block at a time.

        Barcodes of variable length, which the file holds as objects, are
        read whole into a new array instead.
        """
        dataset = self.grp[self.grpNames["cell_names"]]
        if np.dtype(dataset.dtype).kind != "S":
            return super()._cell_names_source()
        assert self.validBarcodeIdx is not None
        return _FileBarcodes(
            dataset,
            self.validBarcodeIdx,
            f"{_H5_VALUES['cell_names']} must be UTF-8 text",
            self._dataset_place("cell_names"),
        )

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from super().get_feature_columns()
        if "features" not in self.grp:
            return
        features = self.grp["features"]
        if not isinstance(features, h5py.Group) or "_all_tag_keys" not in features:
            return
        raw_keys = np.asarray(features["_all_tag_keys"][:]).reshape(-1)
        for raw_key in raw_keys:
            key = as_text(raw_key)
            if key in {"id", "name", "feature_type"} or key not in features:
                continue
            values = features[key]
            if not isinstance(values, h5py.Dataset):
                continue
            if values.ndim != 1 or int(values.shape[0]) != self.nFeatures:
                raise ValueError(
                    f"10x feature tag {key!r} has shape {values.shape}; "
                    f"expected ({self.nFeatures},)"
                )
            yield key, np.asarray(values[:])

    def consume(
        self, batch_size: int, lines_in_mem: int | None = None
    ) -> Generator[coo_matrix, None, None]:
        """Yield CSR chunks from the Cell Ranger H5 matrix.

        After an import's scan shuffled the cells, the chunks hold its
        shuffled cells, once.

        Args:
            batch_size: Number of cells per chunk.
            lines_in_mem: Unused; kept for CrReader API compatibility.
        """
        if self._shuffledRows is not None:
            return self._consume_shuffled(batch_size)
        return self._consume_file(batch_size)

    def _file_rows(self) -> np.ndarray:
        """Return each selected barcode's row in the file, its index among the barcodes."""
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        return np.asarray(valid_idx).astype(np.int64, copy=False)

    def _file_first_rows(self, count: int, lines_in_mem: int) -> csr_matrix:
        """Return the first ``count`` selected barcodes, read from the file."""
        batch = next(self._consume_file(max(1, int(count))), None)
        if batch is None:
            return csr_matrix((0, self.nFeatures), dtype=self.matrix_dtype)
        return csr_matrix(batch.tocsr()[: int(count)])

    # noinspection DuplicatedCode
    def _consume_file(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Yield chunks of the selected barcodes in the file's order."""
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        indptr = self._source_indptr()
        # HDF5 caches decompressed chunks per open dataset. Reopening the
        # datasets for every run of contiguous cells would decompress a chunk
        # again for each run that reads part of it.
        data_node = self.grp["data"]
        indices_node = self.grp["indices"]
        matrix_dtype = self.matrix_dtype
        for s in range(0, len(valid_idx), batch_size):
            v_pos = valid_idx[s : s + batch_size]
            starts = indptr[v_pos]
            ends = indptr[v_pos + 1]
            counts = ends - starts
            cell_idx = np.repeat(
                np.arange(len(v_pos)),
                counts,
            )
            nnz = int(counts.sum())
            data = np.empty(nnz, dtype=matrix_dtype)
            indices = np.empty(nnz, dtype=indices_node.dtype)
            boundaries = np.r_[
                0, np.flatnonzero(starts[1:] != ends[:-1]) + 1, len(v_pos)
            ]
            offset = 0
            for first, stop in zip(boundaries[:-1], boundaries[1:]):
                start, end = int(starts[first]), int(ends[stop - 1])
                size = end - start
                if size:
                    source = np.s_[start:end]
                    destination = np.s_[offset : offset + size]
                    data_node.read_direct(data, source, destination)
                    indices_node.read_direct(indices, source, destination)
                    offset += size
            yield coo_matrix(
                (data, (cell_idx, indices)), shape=(len(v_pos), self.nFeatures)
            )

    def max_window_nnz(self, window_rows: int) -> int:
        """Return the largest selected-cell row-window nnz, in the order ``consume`` yields them."""
        if self._shuffledRows is not None:
            return max_window_nnz(self._shuffledRows.indptr, window_rows)
        return max_window_nnz(self._selected_cumulative_nnz(), window_rows)

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Count row-pointer arrays created while one H5 batch is produced.

        Shuffled cells are read a bucket at a time with the next one loading
        ahead.
        """
        rows = min(max(1, int(batch_size)), self.nCells)
        if rows == 0:
            return 0
        if self._shuffledRows is not None:
            return self._shuffled_staging_bytes(rows)
        itemsize = self._source_indptr().dtype.itemsize
        return int(rows * (3 * itemsize + np.dtype(np.int64).itemsize))

    def _prepare_sparse_import(self) -> None:
        self._source_indptr()
        self._selected_cumulative_nnz()

    def _sparse_import_resident_bytes(self) -> int:
        """Return the bytes of the arrays that an import holds.

        They are the selected barcodes, the matrix pointers, and the cumulative
        entries of the selected barcodes, counted before they are loaded so
        that an import can plan before it prepares, and the row pointers of
        an import's shuffled cells.
        """
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        pointers = self.grp["indptr"]
        return int(
            valid_idx.nbytes
            + int(pointers.shape[0]) * pointers.dtype.itemsize
            + (valid_idx.size + 1) * np.dtype(np.int64).itemsize
            + (0 if self._shuffledRows is None else self._shuffledRows.resident_bytes)
        )

    def close(self) -> None:
        """Closes file connection and removes any shuffled cells."""
        self._release_shuffled_rows()
        self.h5obj.close()


# The barcodes that a read takes from the file at once, at most.
_BARCODE_SLAB_ROWS = 65_536


class _FileBarcodes:
    """The barcodes of a 10x HDF5 file's selected cells, read from the file a block at a time.

    Indexing by positions among the selected cells, an array or a slice,
    reads those barcodes from the file in the file's order, a slab of it at
    a time, and returns them in the positions' order as text as wide as the
    longest selected barcode. A cell table's write reads them a block of
    rows at a time, so no copy of every barcode is held. Construction
    measures the barcodes, a slab at a time.

    Attributes:
        shape: The selected cells, as an array's shape.
        dtype: Text as wide as the longest selected barcode.
        rowReadBytes: The most that reading one barcode of a block holds,
            with the text it returns, at any step of the read: its file row,
            its place in the read and the file row in order, as int64; then
            its bytes as the file holds them, beside the slab of the file
            that they come from, with an index and the bytes of each barcode
            taken from it; then its text as decoded; and last its text put
            in order. Barcodes that are not ASCII are decoded through Python
            bytes and strings, which are counted at the widest kind of
            character.
    """

    def __init__(
        self, dataset: h5py.Dataset, rows: np.ndarray, expectation: str, place: str
    ) -> None:
        self._dataset = dataset
        self._rows = np.asarray(rows).astype(np.int64, copy=False)
        self._expectation = expectation
        self._place = place
        self.shape = (int(self._rows.size),)
        width = 0
        self._ascii = True
        for start in range(0, self.shape[0], _BARCODE_SLAB_ROWS):
            raw = self._read_sorted(self._rows[start : start + _BARCODE_SLAB_ROWS])
            if _is_ascii(raw):
                width = max(width, int(np.strings.str_len(raw).max(initial=0)))
            else:
                self._ascii = False
                texts = utf8_texts(raw, expectation, place)
                width = max(width, max(map(len, texts), default=0))
        self.dtype = np.dtype(f"U{max(1, width)}")
        stored = self.dtype.itemsize
        raw_bytes = int(np.dtype(dataset.dtype).itemsize)
        # The place in the read, the bytes and the decoded text; bytes that
        # are not ASCII become a Python bytes object, 33 bytes and their own,
        # and a string, 60 bytes and four a character at most, each with a
        # reference of 8.
        decoding = 8 + raw_bytes + stored + (0 if self._ascii else 109 + 5 * raw_bytes)
        self.rowReadBytes = max(24 + 3 * raw_bytes, decoding, 8 + 2 * stored)

    def __len__(self) -> int:
        return self.shape[0]

    def _read_sorted(self, ordered: np.ndarray) -> np.ndarray:
        """Return the barcodes of increasing file rows ``ordered``, reading the file a slab of at most as many rows at a time."""
        raw = np.empty(ordered.size, dtype=self._dataset.dtype)
        slab_rows = min(_BARCODE_SLAB_ROWS, int(ordered.size))
        first = 0
        while first < ordered.size:
            start = int(ordered[first])
            stop = start + slab_rows
            last = int(np.searchsorted(ordered, stop, side="left"))
            slab = self._dataset[start : min(stop, self._dataset.shape[0])]
            raw[first:last] = slab[ordered[first:last] - start]
            del slab
            first = last
        return raw

    def __getitem__(self, positions: np.ndarray | slice) -> np.ndarray:
        file_rows = self._rows[positions]
        sorter = np.argsort(file_rows, kind="stable")
        ordered = file_rows[sorter]
        del file_rows
        raw = self._read_sorted(ordered)
        del ordered
        text = (
            raw.astype(self.dtype)
            if self._ascii
            else np.asarray(
                utf8_texts(raw, self._expectation, self._place), dtype=self.dtype
            )
        )
        del raw
        placed = np.empty(text.shape, dtype=self.dtype)
        placed[sorter] = text
        return placed


def _is_ascii(values: np.ndarray) -> bool:
    """Return whether fixed-width bytes hold only ASCII characters."""
    return int(np.ascontiguousarray(values).view(np.uint8).max(initial=0)) < 0x80


def _selected_barcodes(
    consumer: Callable[[int, Any], None], selected: np.ndarray, first: int
) -> Callable[[int, Any], None]:
    """Return a scan consumer that passes on only the selected barcodes of each block.

    The scan reads the barcodes from ``selected[0]`` to ``selected[-1]``,
    sorted barcode indexes whose values are contiguous, so the barcodes
    between them hold no values. ``consumer`` receives the selected ones by
    their position among all the selected barcodes, of which ``selected``
    starts at ``first``.
    """
    origin = int(selected[0])

    def consume(start: int, block: Any) -> None:
        barcodes = origin + start + np.arange(block.shape[0], dtype=np.int64)
        positions = np.searchsorted(selected, barcodes)
        kept = positions < selected.size
        kept[kept] = selected[positions[kept]] == barcodes[kept]
        if not kept.any():
            return
        if kept.all():
            consumer(first + int(positions[0]), block)
            return
        consumer(first + int(positions[kept][0]), block[kept])

    return consume


_PROTOSPACER_COLUMNS = ("cell_barcode", "num_features", "feature_call", "num_umis")
# Rows without a call, which Cell Ranger may write as an empty field or None.
_NO_CALL = frozenset({"", "None"})
# The largest UMI count of a call, the uint32 maximum that guide calls store.
_MAX_UMIS = int(np.iinfo(np.uint32).max)
# The decompressed bytes that the size scan reads at a time, a small share of
# what parsing the file holds.
_SCAN_BYTES = 1 << 15
#: The most bytes that :func:`protospacer_calls_size` holds while it counts a
#: file, whatever the file holds: a chunk as read and the flags of its
#: characters, and the positions of its line ends and its separators, the
#: line of each separator and the separators of each line, 8 bytes each, up
#: to 22 bytes for each byte of a chunk of nothing but line ends and
#: separators.
PROTOSPACER_SCAN_BYTES = 24 * _SCAN_BYTES
# The bytes that read_protospacer_calls holds while it parses, by the size of
# the file: for each row, its four fields as Python text, pandas' copies of
# them, and its lists of calls and UMI counts; for each piece of a list, a
# feature ID or a UMI count, its text, the flattened copies and its number;
# and for each byte, the text that the pieces copy.
_PARSE_BYTES_PER_ROW = 640
_PARSE_BYTES_PER_PIECE = 160
_PARSE_BYTES_PER_BYTE = 3
# The bytes that the parsed calls hold: each barcode as Python text and its
# pointer, and each call's row, UMIs, feature pointer and feature text.
_PARSED_BYTES_PER_ROW = 64
_PARSED_BYTES_PER_CALL = 96
_PARSED_BYTES_PER_BYTE = 1


@dataclass(frozen=True, slots=True)
class ProtospacerCalls:
    """The guide calls in Cell Ranger's ``protospacer_calls_per_cell.csv``.

    Attributes:
        barcodes: The cell barcode of each row, in file order.
        rows: The row in ``barcodes`` of each call.
        features: The feature ID of each call.
        umis: The UMIs of each call.
    """

    barcodes: np.ndarray
    rows: np.ndarray
    features: np.ndarray
    umis: np.ndarray


@dataclass(frozen=True, slots=True)
class ProtospacerCallsSize:
    """The size of Cell Ranger's call file, read before the file is parsed.

    The counts bound what the file holds, so that the memory of reading it
    can be planned: a row of ``k`` calls separates its features and its UMI
    counts with ``k - 1`` ``|`` characters each.

    Attributes:
        n_bytes: The bytes of the file's text, decompressed.
        n_rows: The most rows of calls: the lines after the header.
        n_pipes: The ``|`` characters of the file.
        most_calls: The most calls of one row.
    """

    n_bytes: int
    n_rows: int
    n_pipes: int
    most_calls: int

    @property
    def n_calls(self) -> int:
        """The most calls of the file: one per row and one per two ``|``."""
        return self.n_rows + self.n_pipes // 2

    def parse_bytes(self) -> int:
        """Return the most bytes that :func:`read_protospacer_calls` holds while it parses."""
        pieces = 2 * self.n_rows + self.n_pipes
        return (
            _PARSE_BYTES_PER_ROW * self.n_rows
            + _PARSE_BYTES_PER_PIECE * pieces
            + _PARSE_BYTES_PER_BYTE * self.n_bytes
        )

    def parsed_bytes(self) -> int:
        """Return the most bytes that the :class:`ProtospacerCalls` read from the file hold."""
        return (
            _PARSED_BYTES_PER_ROW * self.n_rows
            + _PARSED_BYTES_PER_CALL * self.n_calls
            + _PARSED_BYTES_PER_BYTE * self.n_bytes
        )


def _open_call_file(path: str | os.PathLike[str]) -> Any:
    """Open the call file for reading bytes, decompressed when it is gzip-compressed.

    A gzip stream is recognized by its first bytes, whatever the file's name.
    """
    return open_binary(path)


def protospacer_calls_size(path: str | os.PathLike[str]) -> ProtospacerCallsSize:
    """Count the rows, the ``|`` separators and the bytes of Cell Ranger's call file.

    The file is read 256 KiB at a time, so the count holds little memory
    whatever the file's size.

    Args:
        path: The call file, plain or gzip-compressed.

    Returns:
        The counts, which bound the rows and the calls of the file.
    """
    n_bytes = n_newlines = n_returns = n_pipes = most = carried = 0
    with _open_call_file(path) as handle:
        while chunk := handle.read(_SCAN_BYTES):
            data = np.frombuffer(chunk, dtype=np.uint8)
            n_bytes += len(data)
            newlines = np.flatnonzero(data == ord("\n"))
            pipes = np.flatnonzero(data == ord("|"))
            n_returns += int(np.count_nonzero(data == ord("\r")))
            n_newlines += len(newlines)
            n_pipes += len(pipes)
            # The separators of each line; the first line of the chunk goes on
            # with the last line of the chunk before.
            per_line = np.bincount(
                np.searchsorted(newlines, pipes), minlength=len(newlines) + 1
            )
            per_line[0] += carried
            most = max(most, int(per_line.max()))
            carried = int(per_line[-1])
    # pandas ends lines at a newline, a carriage return, or both; the last
    # line may have no end.
    n_lines = max(n_newlines, n_returns) + 1
    return ProtospacerCallsSize(
        n_bytes=n_bytes,
        n_rows=max(n_lines - 1, 0),
        n_pipes=n_pipes,
        most_calls=most // 2 + 1,
    )


def _line_numbers(rows: np.ndarray) -> str:
    """Name the file lines of data rows, counting the header as line 1."""
    lines = [str(int(row) + 2) for row in rows[:5]]
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return ", ".join(lines) + more


def _read_call_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read the call file as text, one row per line after the header.

    Blank lines are kept as rows, so that each row's position names its
    line, and the header sets the number of fields of every line.

    Raises:
        ValueError: If a line holds more fields than the header.
    """
    with _open_call_file(path) as handle:
        try:
            # The header is read as the first row, so that pandas compares
            # every line with it instead of taking a longer line's first
            # field as an index.
            table = pd.read_csv(
                handle,
                header=None,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                skip_blank_lines=False,
                compression=None,
            )
        except pd.errors.ParserError as error:
            fields = re.search(
                r"Expected (\d+) fields in line (\d+), saw (\d+)", str(error)
            )
            if fields is None:
                raise ValueError(
                    f"The call file cannot be read as CSV: {error}"
                ) from error
            expected, line, found = fields.groups()
            raise ValueError(
                f"Line {line} of the call file holds {found} fields, but its header "
                f"names {expected}"
            ) from error
    names = table.iloc[0].tolist()
    frame = table.iloc[1:].reset_index(drop=True)
    missing = [name for name in _PROTOSPACER_COLUMNS if name not in names]
    if missing:
        raise ValueError(
            "The call file lacks the columns "
            + ", ".join(repr(name) for name in missing)
            + "; Cell Ranger writes "
            + ", ".join(_PROTOSPACER_COLUMNS)
        )
    # The first column of each name, as pandas would read it.
    return frame[[names.index(name) for name in _PROTOSPACER_COLUMNS]].set_axis(
        list(_PROTOSPACER_COLUMNS), axis=1
    )


def read_protospacer_calls(path: str | os.PathLike[str]) -> ProtospacerCalls:
    """Read the guide calls of each barcode from Cell Ranger's call file.

    Cell Ranger's CRISPR Guide Capture analysis writes
    ``crispr_analysis/protospacer_calls_per_cell.csv``, one row per barcode with
    a confident call: ``cell_barcode``, ``num_features``, and the
    ``|``-separated ``feature_call`` and ``num_umis`` of its called guides.
    Barcodes without a confident call are absent, and a row whose
    ``num_features`` is 0 holds no call. A gzip-compressed file is read too,
    whatever its name. Blank lines at the end of the file are left out.

    Args:
        path: The call file.

    Returns:
        The barcode of each row and the feature ID and UMIs of each call.

    Raises:
        ValueError: If a column is missing, a line holds more fields than the
            header, a blank line comes before a row, a barcode is empty or
            repeated, ``num_features`` is not a whole number or disagrees
            with the number of called features or UMI counts, a UMI count is
            not a whole number or exceeds the uint32 maximum, a feature ID is
            empty, or a row calls one feature twice.
    """
    frame = _read_call_table(path)
    # The fields are stripped and split one at a time, without pandas' .str
    # accessor, which pandas 2 keeps in a reference cycle with its text.
    barcodes = stripped_text(frame["cell_barcode"]).to_numpy(dtype=object)
    declared_text = stripped_text(frame["num_features"])
    feature_text = stripped_text(frame["feature_call"])
    umi_text = stripped_text(frame["num_umis"])
    del frame
    blank = (
        (barcodes == "")
        & (declared_text == "").to_numpy()
        & (feature_text == "").to_numpy()
        & (umi_text == "").to_numpy()
    )
    # Blank lines after the last row hold nothing; a blank line before a row
    # may be a row that lost its fields.
    kept = len(blank) - int(np.argmin(blank[::-1])) if not blank.all() else 0
    if np.any(blank[:kept]):
        raise ValueError(
            "The call file holds blank lines among its rows, on lines "
            f"{_line_numbers(np.flatnonzero(blank[:kept]))}; Cell Ranger writes "
            "one row per barcode"
        )
    if kept < len(blank):
        barcodes = barcodes[:kept]
        declared_text = declared_text.iloc[:kept]
        feature_text = feature_text.iloc[:kept]
        umi_text = umi_text.iloc[:kept]
    empty = np.flatnonzero(barcodes == "")
    if len(empty):
        raise ValueError(f"Empty cell barcodes on lines {_line_numbers(empty)}")
    repeated = np.flatnonzero(pd.Index(barcodes).duplicated(keep="first"))
    if len(repeated):
        raise ValueError(
            f"Cell barcodes repeat on lines {_line_numbers(repeated)}, for example "
            f"{barcodes[repeated[0]]!r}"
        )
    declared = pd.to_numeric(declared_text, errors="coerce")
    declared_values = declared.to_numpy(dtype=np.float64)
    invalid = np.flatnonzero(
        ~np.isfinite(declared_values)
        | (declared_values < 0)
        | (declared_values != np.floor(declared_values))
    )
    if len(invalid):
        raise ValueError(
            "num_features must be a whole number of at least 0 on lines "
            f"{_line_numbers(invalid)}"
        )
    # A row lists far fewer calls than this, so a larger number disagrees
    # with its calls below without overflowing int64.
    counts = np.minimum(declared_values, _MAX_UMIS).astype(np.int64)
    uncalled = counts == 0
    stray = np.flatnonzero(
        uncalled
        & ~(
            feature_text.isin(_NO_CALL).to_numpy()
            & umi_text.isin(_NO_CALL | {"0"}).to_numpy()
        )
    )
    if len(stray):
        raise ValueError(
            f"Rows with num_features 0 list calls on lines {_line_numbers(stray)}"
        )
    called = np.flatnonzero(~uncalled)
    feature_lists = [text.split("|") for text in feature_text.iloc[called].tolist()]
    umi_lists = [text.split("|") for text in umi_text.iloc[called].tolist()]
    feature_lengths = np.fromiter(
        map(len, feature_lists), dtype=np.int64, count=len(feature_lists)
    )
    umi_lengths = np.fromiter(map(len, umi_lists), dtype=np.int64, count=len(umi_lists))
    mismatched = called[
        (feature_lengths != counts[called]) | (umi_lengths != counts[called])
    ]
    if len(mismatched):
        raise ValueError(
            "num_features disagrees with the number of called features or UMI "
            f"counts on lines {_line_numbers(mismatched)}"
        )
    rows = np.repeat(called, counts[called]).astype(np.int64)
    if len(rows):
        features = np.asarray(
            [piece.strip() for pieces in feature_lists for piece in pieces],
            dtype=object,
        )
        umi_values = pd.to_numeric(
            pd.Series(
                [piece.strip() for pieces in umi_lists for piece in pieces],
                dtype=object,
            ),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    else:
        features = np.empty(0, dtype=object)
        umi_values = np.empty(0, dtype=np.float64)
    bad_umis = rows[
        ~np.isfinite(umi_values)
        | (umi_values < 0)
        | (umi_values != np.floor(umi_values))
    ]
    if len(bad_umis):
        raise ValueError(
            "UMI counts must be whole numbers of at least 0 on lines "
            f"{_line_numbers(np.unique(bad_umis))}"
        )
    # A larger count would overflow int64 below, and guide calls store
    # uint32 counts.
    large_umis = rows[umi_values > _MAX_UMIS]
    if len(large_umis):
        raise ValueError(
            f"UMI counts must be at most {_MAX_UMIS} on lines "
            f"{_line_numbers(np.unique(large_umis))}"
        )
    blank = rows[features == ""]
    if len(blank):
        raise ValueError(
            f"Empty feature IDs on lines {_line_numbers(np.unique(blank))}"
        )
    twice = rows[pd.MultiIndex.from_arrays([rows, features]).duplicated(keep="first")]
    if len(twice):
        raise ValueError(
            "A feature is called twice in one row on lines "
            f"{_line_numbers(np.unique(twice))}"
        )
    return ProtospacerCalls(
        barcodes=barcodes,
        rows=rows,
        features=features,
        umis=umi_values.astype(np.int64),
    )


#: The format name of Cell Ranger's contig annotation files.
CELLRANGER_VDJ_FORMAT = "cellranger"
# Cell Ranger's contig annotation columns and their AIRR names. Cell Ranger
# writes the junction, with its conserved cysteine and phenylalanine or
# tryptophan, as cdr3 and cdr3_nt, and the amino-acid and nucleotide sequence
# of each other region as fwr1 and fwr1_nt, and so on, from version 6.
_VDJ_COLUMNS = {
    "barcode": "cell_id",
    "contig_id": "sequence_id",
    "chain": "locus",
    "v_gene": "v_call",
    "d_gene": "d_call",
    "j_gene": "j_call",
    "c_gene": "c_call",
    "cdr3": "junction_aa",
    "cdr3_nt": "junction",
    "productive": "productive",
    "umis": "umi_count",
    "reads": "consensus_count",
    "is_cell": "is_cell",
    "high_confidence": "high_confidence",
    "full_length": "full_length",
    "fwr1": "fwr1_aa",
    "fwr1_nt": "fwr1",
    "cdr1": "cdr1_aa",
    "cdr1_nt": "cdr1",
    "fwr2": "fwr2_aa",
    "fwr2_nt": "fwr2",
    "cdr2": "cdr2_aa",
    "cdr2_nt": "cdr2",
    "fwr3": "fwr3_aa",
    "fwr3_nt": "fwr3",
    "fwr4": "fwr4_aa",
    "fwr4_nt": "fwr4",
    "raw_clonotype_id": "raw_clonotype_id",
    "raw_consensus_id": "raw_consensus_id",
    "exact_subclonotype_id": "exact_subclonotype_id",
}
_VDJ_REQUIRED = ("barcode", "contig_id", "chain", "cdr3", "productive")


def vdj_contig_columns(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Return the header columns of a Cell Ranger contig annotation file, as written.

    Raises:
        ValueError: If the file has no header line.
    """
    return header_fields(path, ",", quoted=True)


def iter_vdj_contigs(
    path: str | os.PathLike[str],
    *,
    chunk_rows: int = 50_000,
) -> Iterator[tuple[np.ndarray, pd.DataFrame]]:
    """Parse Cell Ranger's contig annotations in chunks of rows.

    Cell Ranger's V(D)J pipeline writes ``filtered_contig_annotations.csv``,
    the contigs of the barcodes that it calls cells and that it assembled with
    high confidence, and ``all_contig_annotations.csv``, every contig. Each row
    is one contig. Its columns take their AIRR names: ``barcode`` becomes
    ``cell_id``, ``contig_id`` ``sequence_id``, ``chain`` ``locus``,
    ``v_gene`` ``v_call`` (and so for ``d_gene``, ``j_gene`` and ``c_gene``),
    ``cdr3`` ``junction_aa`` and ``cdr3_nt`` ``junction``, since Cell Ranger's
    CDR3 includes the conserved residues of the junction, ``umis``
    ``umi_count`` and ``reads`` ``consensus_count``. The region sequences of
    Cell Ranger 6 and later, such as ``fwr1`` and ``fwr1_nt``, become
    ``fwr1_aa`` and ``fwr1``. ``is_cell``, ``high_confidence``,
    ``full_length``, ``raw_clonotype_id``, ``raw_consensus_id`` and
    ``exact_subclonotype_id`` keep their names, and other columns are not
    kept. A plain or gzip-compressed file is read, blank lines hold no contig,
    and a row of fewer or more fields than the header is refused.

    Args:
        path: ``filtered_contig_annotations.csv`` or
            ``all_contig_annotations.csv``.
        chunk_rows: The most rows to parse at once.

    Yields:
        The file line of each contig of a chunk and the chunk's parsed fields:
        ``cell_id`` and each AIRR field that the file holds, text as Python
        strings with None where missing, flags as int8 and counts as int64.

    Raises:
        ValueError: If a required column is missing, a column is repeated, a
            row holds another number of fields than the header, a barcode is
            empty, or a value is of the wrong kind.
    """
    columns = vdj_contig_columns(path)
    repeated = sorted({name for name in columns if columns.count(name) > 1})
    if repeated:
        raise ValueError(
            "The contig file repeats the columns " + ", ".join(map(repr, repeated))
        )
    missing = [name for name in _VDJ_REQUIRED if name not in columns]
    if missing:
        raise ValueError(
            "The contig file lacks the columns "
            + ", ".join(repr(name) for name in missing)
            + "; Cell Ranger's contig annotations hold "
            + ", ".join(_VDJ_REQUIRED)
        )
    yield from contig_chunks(
        path,
        what="a contig annotation file",
        columns=_VDJ_COLUMNS,
        quoted=True,
        chunk_rows=chunk_rows,
        infer_locus=False,
    )


# The per-molecule datasets of Cell Ranger's molecule_info.h5 that a lineage
# barcode import reads, and the lookups they index.
_MOLECULE_COLUMNS = ("barcode_idx", "umi", "feature_idx", "count", "gem_group")
_MOLECULE_LOOKUPS = ("barcodes", "features/id", "library_info", "library_idx")
# The library type of Cell Ranger's gene expression libraries, which hold no
# feature barcodes.
_GENE_EXPRESSION = "Gene Expression"
# The bases of a 2-bit packed UMI: A=0, C=1, G=2, T=3.
_UMI_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
#: Rows of the per-molecule datasets read at once, at least.
MIN_MOLECULE_ROWS = 1 << 16


@dataclass(frozen=True, slots=True)
class MoleculeInfoLibrary:
    """One feature barcode library of a Cell Ranger ``molecule_info.h5``.

    Attributes:
        path: The file.
        library_type: The library's type, such as ``"Custom"`` or ``"CRISPR
            Guide Capture"``.
        library_indices: The positions in ``library_info`` of the libraries
            of that type.
        umi_length: The length of the library's UMIs, from ``metrics_json``.
        n_molecules: The molecules of the file, of every library.
        barcode_block_rows: The cell barcodes read at once: the rows of the
            dataset's chunks, or 65,536 if more.
        barcode_block_bytes: The bytes of such a block of cell barcodes.
    """

    path: str
    library_type: str
    library_indices: tuple[int, ...]
    umi_length: int
    n_molecules: int
    barcode_block_rows: int
    barcode_block_bytes: int


@dataclass(frozen=True, slots=True)
class MoleculeRows:
    """A chunk of the molecules of one library of a ``molecule_info.h5``.

    Attributes:
        first_row: The position of the chunk's first row in the file.
        barcodes: Each molecule's cell barcode with ``-`` and its GEM well.
        umis: Each molecule's UMI, decoded.
        features: The ID of each molecule's feature, its barcode.
        reads: Each molecule's reads.
    """

    first_row: int
    barcodes: np.ndarray
    umis: np.ndarray
    features: np.ndarray
    reads: np.ndarray


def is_molecule_info(path: str | os.PathLike[str]) -> bool:
    """Return whether an HDF5 file holds Cell Ranger's per-molecule datasets."""
    with h5py.File(path, "r") as handle:
        return all(
            isinstance(handle.get(name), h5py.Dataset)
            for name in (*_MOLECULE_COLUMNS, "library_idx", "barcodes", "library_info")
        )


def _json_dataset(handle: h5py.File, name: str) -> Any:
    """Decode a dataset that holds one JSON text, as a scalar or one entry."""
    import json

    value = np.asarray(handle[name][()]).reshape(-1)[0]
    return json.loads(as_text(value))


def _umi_length(handle: h5py.File, library_type: str, path: str) -> int:
    """Read the UMI length of a library type from the file's metrics."""
    metrics = _json_dataset(handle, "metrics_json") if "metrics_json" in handle else {}
    candidates: list[Any] = []
    definitions = metrics.get("chemistry_defs") if isinstance(metrics, dict) else None
    if isinstance(definitions, dict) and isinstance(
        definitions.get(library_type), dict
    ):
        candidates.append(definitions[library_type].get("umi"))
    if isinstance(metrics, dict):
        candidates.append(metrics.get("chemistry_umi"))
    for candidate in candidates:
        found = candidate[0] if isinstance(candidate, list) and candidate else candidate
        if isinstance(found, dict) and isinstance(found.get("length"), int):
            return int(found["length"])
    raise ValueError(
        f"{path} records no UMI length in its metrics_json, so its 2-bit packed "
        "UMIs cannot be decoded; Cell Ranger records it from release 6"
    )


def molecule_info_library(
    path: str | os.PathLike[str], library_type: str | None = None
) -> MoleculeInfoLibrary:
    """Find the feature barcode library to read from a ``molecule_info.h5``.

    Cell Ranger's ``molecule_info.h5`` lists every molecule of every library
    of a run. A lineage barcode library is a feature barcode library, of type
    ``"Custom"`` or another, whose features are the barcodes.

    Args:
        path: The file.
        library_type: The type of the library to read, such as ``"Custom"``,
            or None for the file's only library that is not gene expression.

    Raises:
        ValueError: If the file lacks a per-molecule dataset, holds no library
            of the type, or, without a type, holds no library or several
            besides gene expression, or records no UMI length.
    """
    name = os.fspath(path)
    with h5py.File(name, "r") as handle:
        missing = [
            dataset
            for dataset in (*_MOLECULE_COLUMNS, *_MOLECULE_LOOKUPS)
            if not isinstance(handle.get(dataset), h5py.Dataset)
        ]
        if missing:
            raise ValueError(
                f"{name} is not a Cell Ranger molecule_info.h5: it lacks "
                + ", ".join(missing)
            )
        libraries = _json_dataset(handle, "library_info")
        if not isinstance(libraries, list) or not all(
            isinstance(entry, dict) and "library_type" in entry for entry in libraries
        ):
            raise ValueError(f"The library_info of {name} lists no library types")
        types = [str(entry["library_type"]) for entry in libraries]
        features = sorted({kind for kind in types if kind != _GENE_EXPRESSION})
        if library_type is None:
            if len(features) != 1:
                raise ValueError(
                    f"{name} holds "
                    + (
                        "no feature barcode library"
                        if not features
                        else "the feature barcode libraries "
                        + ", ".join(map(repr, features))
                    )
                    + "; name the library type to read with library=..."
                )
            library_type = features[0]
        elif library_type not in types:
            raise ValueError(
                f"{name} holds no library of type {library_type!r}; its libraries "
                "are " + ", ".join(map(repr, types))
            )
        indices = tuple(
            position for position, kind in enumerate(types) if kind == library_type
        )
        umi_length = _umi_length(handle, library_type, name)
        n_molecules = int(handle["library_idx"].shape[0])
        barcodes = handle["barcodes"]
        chunk_rows = barcodes.chunks[0] if barcodes.chunks else 0
        block_rows = max(int(chunk_rows), MIN_MOLECULE_ROWS)
        block_bytes = block_rows * int(barcodes.dtype.itemsize)
        lengths = {int(handle[column].shape[0]) for column in _MOLECULE_COLUMNS}
        if lengths != {n_molecules}:
            raise ValueError(
                f"The per-molecule datasets of {name} differ in length, so its "
                "molecules cannot be read"
            )
    if not 1 <= umi_length <= 16:
        raise ValueError(
            f"{name} records a UMI length of {umi_length}; a 32-bit packed UMI "
            "holds 1 to 16 bases"
        )
    return MoleculeInfoLibrary(
        path=name,
        library_type=library_type,
        library_indices=indices,
        umi_length=umi_length,
        n_molecules=n_molecules,
        barcode_block_rows=block_rows,
        barcode_block_bytes=block_bytes,
    )


class BarcodeBlocks:
    """Cell barcodes of a ``molecule_info.h5``, read one block at a time.

    Molecules are sorted by their barcode, so the chunks of molecules name
    blocks of barcodes in order; the block read last is kept, and each block
    is read once even when its dataset is stored as one compressed chunk.

    Args:
        barcodes: The ``barcodes`` dataset, or anything sliced like it.
        block_rows: The barcodes of each block.
    """

    def __init__(self, barcodes: Any, *, block_rows: int) -> None:
        self._barcodes = barcodes
        self._rows = max(1, int(block_rows))
        self._n = int(barcodes.shape[0])
        self._block = -1
        self._values: np.ndarray = np.empty(0, dtype=object)

    def texts(self, indices: np.ndarray) -> np.ndarray:
        """Return the barcodes at ``indices``, as text."""
        positions = np.asarray(indices, dtype=np.int64)
        texts = np.empty(len(positions), dtype=object)
        blocks = positions // self._rows
        for block in np.unique(blocks).tolist():
            if block != self._block:
                start = block * self._rows
                self._values = np.asarray(
                    self._barcodes[slice(start, min(start + self._rows, self._n))]
                )
                self._block = block
            chosen = blocks == block
            texts[chosen] = [
                as_text(value)
                for value in self._values[positions[chosen] - block * self._rows]
            ]
        return texts


def decode_umis(packed: Any, length: int) -> np.ndarray:
    """Decode 2-bit packed UMIs, whose last base holds the lowest bits.

    Args:
        packed: The packed UMIs, as unsigned integers.
        length: The bases of each UMI.

    Returns:
        Each UMI as text.
    """
    values = np.asarray(packed, dtype=np.uint64).reshape(-1)
    shifts = np.arange(2 * (length - 1), -1, -2, dtype=np.uint64)
    digits = (values[:, None] >> shifts[None, :]) & np.uint64(3)
    letters = np.ascontiguousarray(_UMI_BASES[digits.astype(np.intp)])
    if not len(values):
        return np.empty(0, dtype=object)
    return letters.view(f"S{length}").reshape(-1).astype(str).astype(object)


def iter_molecule_info(
    library: MoleculeInfoLibrary, *, chunk_rows: int = MIN_MOLECULE_ROWS
) -> Iterator[MoleculeRows]:
    """Yield the molecules of one library of a ``molecule_info.h5`` in chunks.

    Each molecule is a cell barcode, a UMI and a feature, with the reads that
    Cell Ranger assigned to it. Cell barcodes are stored without their GEM
    well, which each molecule records, so each gets ``-`` and its GEM well,
    as Cell Ranger's matrices name cells. UMIs are decoded with the length
    that the file's metrics record, and features named by their IDs.
    Barcodes that Cell Ranger did not call cells are included.

    Args:
        library: The library, from :func:`molecule_info_library`.
        chunk_rows: The rows of the file to read at once, at least 65,536.

    Raises:
        ValueError: If a molecule names a barcode or feature that the file
            does not list.
    """
    rows = max(MIN_MOLECULE_ROWS, int(chunk_rows))
    wanted = np.asarray(library.library_indices, dtype=np.int64)
    with h5py.File(library.path, "r") as handle:
        feature_ids = np.asarray(
            [as_text(value) for value in handle["features/id"][:].tolist()],
            dtype=object,
        )
        barcode_table = handle["barcodes"]
        n_barcodes = int(barcode_table.shape[0])
        blocks = BarcodeBlocks(barcode_table, block_rows=library.barcode_block_rows)
        for start in range(0, library.n_molecules, rows):
            stop = min(start + rows, library.n_molecules)
            kept = np.flatnonzero(
                np.isin(np.asarray(handle["library_idx"][start:stop]), wanted)
            )
            if not len(kept):
                continue
            columns = {
                name: np.asarray(handle[name][start:stop])[kept]
                for name in _MOLECULE_COLUMNS
            }
            barcode_index = columns["barcode_idx"].astype(np.int64)
            feature_index = columns["feature_idx"].astype(np.int64)
            if barcode_index.max() >= n_barcodes or feature_index.max() >= len(
                feature_ids
            ):
                raise ValueError(
                    f"{library.path} lists molecules of barcodes or features that "
                    "it does not hold"
                )
            keys, positions = np.unique(
                np.stack([barcode_index, columns["gem_group"].astype(np.int64)]),
                axis=1,
                return_inverse=True,
            )
            texts = np.asarray(
                [
                    f"{barcode}-{gem_group}"
                    for barcode, gem_group in zip(
                        blocks.texts(keys[0]).tolist(), keys[1].tolist(), strict=True
                    )
                ],
                dtype=object,
            )
            yield MoleculeRows(
                first_row=start,
                barcodes=texts[np.asarray(positions).reshape(-1)],
                umis=decode_umis(columns["umi"], library.umi_length),
                features=feature_ids[feature_index],
                reads=columns["count"].astype(np.int64),
            )
