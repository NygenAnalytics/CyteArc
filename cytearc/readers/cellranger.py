import os
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ..utils.progress import iter_progress
from ._contigs import compression, header_fields, parse_contig_columns
from ._assay_names import (
    AUTO_ASSAY_NAMES,
    auto_name_feat_table,
    make_feat_table_from_types,
)
from ._text import (
    DuplicateFeatureIds,
    as_text,
    duplicate_feature_ids_option,
    require_unique_identifiers,
    unique_feature_identifiers,
)
from ..utils.arrays import (
    assay_feature_ranges,
    cumulative_nnz,
    has_duplicates,
    max_window_nnz,
)
from ..utils.count_values import (
    CountValueRange,
    compressed_count_ranges,
    new_count_ranges,
)


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
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        """Return the range of the canonical values of each group of features.

        The ranges cover the selected cells, with duplicate coordinates summed
        as the import stores them. Pass a range to
        :func:`~cytearc.storage.count_dtype.count_storage_dtype` to resolve the
        storage dtype of its group's counts.

        Args:
            maxBytes: Memory available to the scan.
            featureGroups: Group, numbered from zero, of each feature, such as
                the assay that stores it. None puts every feature in one
                group.
        """

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

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
        h5obj: A File object from the h5py package.
        grp: Current active group in the hierarchy.
        duplicateFeatureIds: How repeated feature IDs are treated.
    """

    def __init__(
        self,
        h5_fn: str,
        is_filtered: bool = True,
        filtering_cutoff: int = 500,
        *,
        duplicate_feature_ids: DuplicateFeatureIds = "error",
    ) -> None:
        if filtering_cutoff < 0:
            raise ValueError("filtering_cutoff cannot be negative")
        self.duplicateFeatureIds = duplicate_feature_ids_option(duplicate_feature_ids)
        self.h5obj: h5py.File = h5py.File(h5_fn, mode="r")
        self.grp: h5py.Group
        self.validBarcodeIdx: np.ndarray | None = None
        self._indptrCache: np.ndarray | None = None
        self._cumulativeRowNnz: np.ndarray | None = None
        self._featureIds: list[str] | None = None
        try:
            super().__init__(self._handle_version())
            require_unique_identifiers(self.cell_names(), "Cell Ranger barcodes")
            self._featureIds = [
                as_text(value)
                for value in unique_feature_identifiers(
                    np.asarray(self._read_dataset("feature_ids"), dtype=object),
                    "Cell Ranger feature IDs",
                    self.duplicateFeatureIds,
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
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        """Return the range of the canonical values of each group of features.

        Each barcode is one compressed vector of the matrix. Selected barcodes
        whose values are contiguous in the matrix are scanned together, as
        ``consume`` reads them; the barcodes between them hold no values.

        Raises:
            MemoryError: If one barcode holds more values than fit in
                ``maxBytes``.
            ValueError: If a value is NaN or infinite.
        """
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        value_ranges = new_count_ranges(featureGroups)
        if valid_idx.size == 0:
            return value_ranges
        indptr = self._source_indptr()
        gaps = indptr[valid_idx[1:]] != indptr[valid_idx[:-1] + 1]
        bounds = np.r_[0, np.flatnonzero(gaps) + 1, valid_idx.size]
        for left, right in zip(bounds[:-1], bounds[1:]):
            compressed_count_ranges(
                indptr[int(valid_idx[left]) : int(valid_idx[right - 1]) + 2],
                self.grp["indices"],
                self.grp["data"],
                minorSize=self.nFeatures,
                maxBytes=maxBytes,
                groups=featureGroups,
                valueRanges=value_ranges,
            )
        return value_ranges

    def _read_dataset(self, key: str) -> list[str]:
        # Construction resolves repeated feature IDs once, so every caller
        # reads the IDs that the import stores.
        if key == "feature_ids" and self._featureIds is not None:
            return list(self._featureIds)
        grp_key = self.grpNames[key]
        return [as_text(x) for x in self.grp[grp_key][:]]

    def cell_names(self) -> list[str]:
        """Returns a list of names of the cells in the dataset."""
        vals = np.array(self._read_dataset("cell_names"))
        if self.validBarcodeIdx is not None:
            vals = vals[self.validBarcodeIdx]
        return list(vals)

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

    # noinspection DuplicatedCode
    def consume(
        self, batch_size: int, lines_in_mem: int | None = None
    ) -> Generator[coo_matrix, None, None]:
        """Yield CSR chunks from the Cell Ranger H5 matrix.

        Args:
            batch_size: Number of cells per chunk.
            lines_in_mem: Unused; kept for CrReader API compatibility.
        """
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
        """Return the largest selected-cell row-window nnz."""
        return max_window_nnz(self._selected_cumulative_nnz(), window_rows)

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        """Count row-pointer arrays created while one H5 batch is produced."""
        rows = min(max(1, int(batch_size)), self.nCells)
        if rows == 0:
            return 0
        itemsize = self._source_indptr().dtype.itemsize
        return int(rows * (3 * itemsize + np.dtype(np.int64).itemsize))

    def _prepare_sparse_import(self) -> None:
        self._source_indptr()
        self._selected_cumulative_nnz()

    def _sparse_import_resident_bytes(self) -> int:
        """Return the bytes of the arrays that an import holds.

        They are the selected barcodes, the matrix pointers, and the cumulative
        entries of the selected barcodes, counted before they are loaded so
        that an import can plan before it prepares.
        """
        valid_idx = self.validBarcodeIdx
        assert valid_idx is not None
        pointers = self.grp["indptr"]
        return int(
            valid_idx.nbytes
            + int(pointers.shape[0]) * pointers.dtype.itemsize
            + (valid_idx.size + 1) * np.dtype(np.int64).itemsize
        )

    def close(self) -> None:
        """Closes file connection."""
        self.h5obj.close()


_PROTOSPACER_COLUMNS = ("cell_barcode", "num_features", "feature_call", "num_umis")
# Rows without a call, which Cell Ranger may write as an empty field or None.
_NO_CALL = frozenset({"", "None"})


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


def _line_numbers(rows: np.ndarray) -> str:
    """Name the file lines of data rows, counting the header as line 1."""
    lines = [str(int(row) + 2) for row in rows[:5]]
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return ", ".join(lines) + more


def read_protospacer_calls(path: str | os.PathLike[str]) -> ProtospacerCalls:
    """Read the guide calls of each barcode from Cell Ranger's call file.

    Cell Ranger's CRISPR Guide Capture analysis writes
    ``crispr_analysis/protospacer_calls_per_cell.csv``, one row per barcode with
    a confident call: ``cell_barcode``, ``num_features``, and the
    ``|``-separated ``feature_call`` and ``num_umis`` of its called guides.
    Barcodes without a confident call are absent, and a row whose
    ``num_features`` is 0 holds no call. A gzip-compressed file is read too.

    Args:
        path: The call file.

    Returns:
        The barcode of each row and the feature ID and UMIs of each call.

    Raises:
        ValueError: If a column is missing, a barcode is empty or repeated,
            ``num_features`` is not a whole number or disagrees with the
            number of called features or UMI counts, a UMI count is not a
            whole number, a feature ID is empty, or a row calls one feature
            twice.
    """
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        compression="infer",
    )
    missing = [name for name in _PROTOSPACER_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            "The call file lacks the columns "
            + ", ".join(repr(name) for name in missing)
            + "; Cell Ranger writes "
            + ", ".join(_PROTOSPACER_COLUMNS)
        )
    barcodes = frame["cell_barcode"].str.strip().to_numpy(dtype=object)
    empty = np.flatnonzero(barcodes == "")
    if len(empty):
        raise ValueError(f"Empty cell barcodes on lines {_line_numbers(empty)}")
    repeated = np.flatnonzero(pd.Index(barcodes).duplicated(keep="first"))
    if len(repeated):
        raise ValueError(
            f"Cell barcodes repeat on lines {_line_numbers(repeated)}, for example "
            f"{barcodes[repeated[0]]!r}"
        )
    declared = pd.to_numeric(frame["num_features"].str.strip(), errors="coerce")
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
    counts = declared_values.astype(np.int64)
    feature_text = frame["feature_call"].str.strip()
    umi_text = frame["num_umis"].str.strip()
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
    feature_lists = feature_text.iloc[called].str.split("|")
    umi_lists = umi_text.iloc[called].str.split("|")
    feature_lengths = feature_lists.str.len().to_numpy(dtype=np.int64)
    umi_lengths = umi_lists.str.len().to_numpy(dtype=np.int64)
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
        features = feature_lists.explode().str.strip().to_numpy(dtype=object)
        umi_values = pd.to_numeric(
            umi_lists.explode().str.strip(), errors="coerce"
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
    return header_fields(path, ",")


def iter_vdj_contigs(
    path: str | os.PathLike[str],
    *,
    chunk_rows: int = 50_000,
) -> Iterator[tuple[int, pd.DataFrame]]:
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
    kept. A plain or gzip-compressed file is read.

    Args:
        path: ``filtered_contig_annotations.csv`` or
            ``all_contig_annotations.csv``.
        chunk_rows: The most rows to parse at once.

    Yields:
        The file line of each chunk's first row and the chunk's parsed fields:
        ``cell_id`` and each AIRR field that the file holds, text as Python
        strings with None where missing, flags as int8 and counts as int64.

    Raises:
        ValueError: If a required column is missing, a column is repeated, a
            barcode is empty, or a value is of the wrong kind.
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
    line = 2
    reader = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        compression=compression(path),
        usecols=lambda name: str(name).strip() in _VDJ_COLUMNS,
        chunksize=max(1, int(chunk_rows)),
    )
    with reader:
        for chunk in reader:
            chunk.columns = [_VDJ_COLUMNS[str(name).strip()] for name in chunk.columns]
            yield line, parse_contig_columns(chunk, first_line=line, infer_locus=False)
            line += len(chunk)
