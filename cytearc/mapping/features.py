import copy
from collections.abc import Generator, Mapping
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..assay import RNAassay, _read_block, norm_lib_size
from ..assay.normalization import library_size_divisors, library_size_values
from ..metadata.rows import read_metadata_rows_chunkwise
from ..storage.artifacts import callable_identity
from ..storage.budget import ResourceBudget
from ..storage.execution import admit_stream
from ..storage.geometry import ArrayGeometry, array_geometry
from ..storage.parallel import stream_shards
from ..storage.partition import (
    affordable_width,
    checked_indices,
    contiguous_ranges,
    is_contiguous,
    row_band,
    row_read_parts,
)
from ..utils.arrays import read_only_copy
from ..utils.count_values import is_real_count_dtype
from ..utils.memory_budget import MemoryBudgetError
from ..utils.warnings import warn

if TYPE_CHECKING:
    from ..assay import Assay


@dataclass(frozen=True, slots=True)
class AlignedFeatureBlock:
    """One aligned row block and its offset in the selected query rows.

    ``observed`` marks rows with at least one nonzero raw count in the
    reference features present in the query. Other rows carry no query
    evidence: every aligned value is a normalized zero or a missing-feature
    fill.
    """

    row_offset: int
    values: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True, slots=True)
class AlignedRowGeometry:
    """Source geometry and deterministic selected-row block boundaries."""

    source: ArrayGeometry | None
    block_rows: int
    boundaries: tuple[tuple[int, int], ...]


def _feature_keys(values: Any, *, name: str, unique: bool = True) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if raw.dtype.kind not in {"O", "S", "U"}:
        raise TypeError(f"{name} must contain strings")
    if raw.dtype.kind == "O" and any(
        not isinstance(value, str | bytes | np.str_ | np.bytes_) for value in raw
    ):
        raise TypeError(f"{name} must contain strings")
    identifiers = np.asarray(raw).astype(str)
    if unique:
        distinct, counts = np.unique(identifiers, return_counts=True)
        if np.any(counts > 1):
            duplicates = distinct[counts > 1][:5].tolist()
            raise ValueError(f"{name} must be unique: {duplicates}")
    return read_only_copy(identifiers)


def _reference_means(values: Any, *, n_features: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Reference normalized means must be real numeric values")
    means = np.asarray(raw, dtype=np.float64)
    if means.shape != (n_features,):
        raise ValueError(
            "Reference normalized means must have one value per reference feature"
        )
    if not np.all(np.isfinite(means)):
        raise ValueError("Reference normalized means must be finite")
    return read_only_copy(means, np.float64)


def _normalization_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    values = copy.deepcopy(dict(parameters))
    required = {
        "normalization_method",
        "size_factor",
        "log_transform",
        "renormalize_subset",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(
            "Reference normalization parameters are missing: "
            + ", ".join(sorted(missing))
        )
    unknown = set(values) - required
    if unknown:
        raise ValueError(
            "Unsupported reference normalization parameters: "
            + ", ".join(sorted(unknown))
            + ". Recompute the normalization with features.normalize, then PCA and "
            "its descendants"
        )

    method = values["normalization_method"]
    supported_method = callable_identity(norm_lib_size)
    if not isinstance(method, Mapping):
        raise ValueError(
            "Unsupported reference normalization method. "
            f"Expected {supported_method!r}, received {method!r}"
        )
    method_identity = dict(method)
    external_hook = method_identity.pop("external_hook", True)
    if external_hook is not True or method_identity != supported_method:
        raise ValueError(
            "Unsupported reference normalization method. "
            f"Expected {supported_method!r}, received {method!r}"
        )
    values["normalization_method"] = dict(method)

    size_factor = values["size_factor"]
    if (
        isinstance(size_factor, bool | np.bool_)
        or not isinstance(size_factor, int | float | np.integer | np.floating)
        or not np.isfinite(size_factor)
        or float(size_factor) <= 0
    ):
        raise ValueError(
            "Reference normalization size_factor must be finite and positive"
        )
    values["size_factor"] = float(size_factor)

    for name in ("log_transform", "renormalize_subset"):
        value = values[name]
        if not isinstance(value, bool | np.bool_):
            raise TypeError(f"Reference normalization {name} must be a boolean")
        values[name] = bool(value)
    return values


def normalize_reference_counts(
    raw: np.ndarray,
    *,
    size_factor: float,
    log_transform: bool,
    source: str,
    denominator: np.ndarray | None = None,
) -> np.ndarray:
    """Library-size normalize rows of counts as a mapping reference does."""
    if denominator is None:
        divisors = library_size_divisors(
            raw.sum(axis=1, dtype=np.float64), source=source, copy=False
        )
    elif np.shape(denominator) != (len(raw),):
        raise ValueError("Normalization totals must hold one value per row of counts")
    else:
        divisors = library_size_divisors(denominator, source=source)
    return library_size_values(
        raw,
        divisors,
        size_factor,
        dtype=np.float64,
        log_transform=log_transform,
    )


class AlignedFeatureStream:
    """Replay normalized query rows in immutable reference feature order.

    Query features match reference features by ID, or by name with
    ``feature_key="names"``. Query features that share a reference feature's
    name add their counts before normalization, so the query keeps every
    count of the feature.
    """

    def __init__(
        self,
        query_assay: "Assay",
        query_cell_indices: np.ndarray,
        reference_feature_keys: np.ndarray,
        reference_normalized_means: np.ndarray,
        reference_normalization_parameters: Mapping[str, Any],
        missing_feature_policy: str,
        resources: ResourceBudget,
        *,
        feature_key: str = "ids",
        reserved_resident_bytes: int = 0,
        reserved_per_row_bytes: int = 0,
    ) -> None:
        if not isinstance(query_assay, RNAassay):
            raise TypeError("AlignedFeatureStream supports RNA query assays only")
        if not isinstance(resources, ResourceBudget):
            raise TypeError("resources must be a ResourceBudget")
        if resources.memoryBytes < 1 or resources.workers < 1:
            raise ValueError("ResourceBudget fields must be positive")
        for name, value in (
            ("reserved_resident_bytes", reserved_resident_bytes),
            ("reserved_per_row_bytes", reserved_per_row_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if missing_feature_policy not in {"reference_mean", "zero", "error"}:
            raise ValueError(
                "missing_feature_policy must be one of "
                "'reference_mean', 'zero', or 'error'"
            )
        if feature_key not in {"ids", "names"}:
            raise ValueError("feature_key must be 'ids' or 'names'")

        raw_data = query_assay.rawData
        if len(raw_data.shape) != 2:
            raise ValueError("Query raw counts must be two-dimensional")
        raw_dtype = np.dtype(raw_data.dtype)
        if not is_real_count_dtype(raw_dtype):
            raise TypeError(f"Query raw counts must be real numbers, not {raw_dtype}")

        cell_indices = checked_indices(
            query_cell_indices,
            limit=int(raw_data.shape[0]),
            name="query_cell_indices",
        )
        if cell_indices.size == 0:
            raise ValueError("query_cell_indices cannot be empty")
        self._query_cell_indices = read_only_copy(cell_indices, np.int64)

        label = "identifiers" if feature_key == "ids" else "names"
        self._reference_feature_keys = _feature_keys(
            reference_feature_keys,
            name=f"Reference feature {label}",
        )
        self._reference_normalized_means = _reference_means(
            reference_normalized_means,
            n_features=len(self._reference_feature_keys),
        )
        self._normalization_parameters = _normalization_parameters(
            reference_normalization_parameters
        )
        self._missing_feature_policy = missing_feature_policy
        self._resources = resources
        self._query_assay = query_assay
        self._raw_backing = raw_data._backing
        self._source_geometry = array_geometry(self._raw_backing)

        # Preparation guarantees one identifier per count column; several
        # count columns may share a name.
        query_feature_keys = _feature_keys(
            query_assay.feats.fetch_all(feature_key),
            name=f"Query feature {label}",
            unique=feature_key == "ids",
        )
        reference_lookup = {
            key: index for index, key in enumerate(self._reference_feature_keys)
        }
        column_reference = np.fromiter(
            (reference_lookup.get(key, -1) for key in query_feature_keys),
            dtype=np.int64,
            count=len(query_feature_keys),
        )
        query_indices = np.flatnonzero(column_reference >= 0)
        # Order the matched query columns by reference feature, then by query
        # position, so each reference feature's columns are adjacent.
        query_indices = query_indices[
            np.argsort(column_reference[query_indices], kind="stable")
        ]
        column_reference = column_reference[query_indices]
        column_starts = np.flatnonzero(np.diff(column_reference, prepend=-1))
        reference_indices = column_reference[column_starts]
        if reference_indices.size == 0:
            if feature_key == "ids":
                raise ValueError(
                    "No reference features overlap the query feature IDs. If the "
                    "query names the same genes with another ID scheme, such as "
                    "gene symbols against Ensembl IDs, pass feature_key='names' "
                    "to match features by name."
                )
            raise ValueError("No reference features overlap the query feature names")
        if missing_feature_policy == "error" and reference_indices.size != len(
            self._reference_feature_keys
        ):
            missing = len(self._reference_feature_keys) - reference_indices.size
            raise ValueError(
                f"Query data is missing {missing} required reference features"
            )
        self._reference_index_map = read_only_copy(reference_indices, np.int64)
        self._query_index_map = read_only_copy(query_indices, np.int64)
        # The first of each reference feature's query columns, which
        # np.add.reduceat sums, when a reference feature has several.
        self._column_starts: np.ndarray | None = (
            None
            if column_starts.size == query_indices.size
            else read_only_copy(column_starts, np.int64)
        )
        self._feature_coverage = float(
            reference_indices.size / len(self._reference_feature_keys)
        )
        if self.renormalize_subset and self._feature_coverage < 1:
            warn(
                "The reference divides each cell by its total over the reference "
                "features (renormalize_subset=True), but the query measures only "
                f"{self._feature_coverage:.1%} of them. Query totals cover the "
                "shared features alone, so normalized query values are larger "
                "than the reference would compute for the same cell."
            )

        self._cell_scalars: np.ndarray | None = None
        if self.renormalize_subset:
            self._totals_source = (
                f"The reference-feature subset of query assay {query_assay.name!r}"
            )
        else:
            # Preparation guarantees a numeric total for every cell.
            scalar_name = f"{query_assay.name}_nCounts"
            self._totals_source = f"Query assay {scalar_name!r}"
            scalars = library_size_divisors(
                read_metadata_rows_chunkwise(
                    query_assay.cells,
                    scalar_name,
                    self._query_cell_indices,
                ),
                source=self._totals_source,
            )
            scalars.setflags(write=False)
            self._cell_scalars = scalars

        self._decode_bytes, read_row_bytes = row_read_parts(
            self._source_geometry,
            self._query_index_map,
            rows_contiguous=is_contiguous(self._query_cell_indices),
        )
        self._resident_bytes = self._calculate_resident_bytes() + int(
            reserved_resident_bytes
        )
        normalized_bytes = np.dtype(np.float64).itemsize
        summed_bytes = (
            0
            if self._column_starts is None
            else len(self._reference_index_map) * normalized_bytes
        )
        self._stream_row_bytes = (
            len(self._query_index_map) * (raw_dtype.itemsize + normalized_bytes)
            + summed_bytes
            + len(self._reference_feature_keys) * normalized_bytes
            + (normalized_bytes if self.renormalize_subset else 0)
            + np.dtype(bool).itemsize
            + int(reserved_per_row_bytes)
            + read_row_bytes
        )
        block_rows, boundaries, io_concurrency = self._plan_rows(
            bytes_per_row=self._stream_row_bytes,
        )
        self._row_geometry = AlignedRowGeometry(
            source=self._source_geometry,
            block_rows=block_rows,
            boundaries=boundaries,
        )
        self._io_concurrency = io_concurrency

    @property
    def row_geometry(self) -> AlignedRowGeometry:
        return self._row_geometry

    @property
    def feature_coverage(self) -> float:
        return self._feature_coverage

    @property
    def size_factor(self) -> float:
        return float(self._normalization_parameters["size_factor"])

    @property
    def log_transform(self) -> bool:
        return bool(self._normalization_parameters["log_transform"])

    @property
    def renormalize_subset(self) -> bool:
        return bool(self._normalization_parameters["renormalize_subset"])

    @property
    def reference_index_map(self) -> np.ndarray:
        """Reference feature positions that the query measures."""
        return self._reference_index_map.view()

    @property
    def query_feature_indices(self) -> np.ndarray:
        """Query feature positions that match ``reference_index_map``, in its order.

        Matched by name, several adjacent positions may match one reference
        feature.
        """
        return self._query_index_map.view()

    def _calculate_resident_bytes(self) -> int:
        arrays = [
            self._query_cell_indices,
            self._reference_feature_keys,
            self._reference_normalized_means,
            self._reference_index_map,
            self._query_index_map,
        ]
        if self._column_starts is not None:
            arrays.append(self._column_starts)
        if self._cell_scalars is not None:
            arrays.append(self._cell_scalars)
        return sum(array.nbytes for array in arrays) + np.dtype(np.float64).itemsize

    def _plan_rows(
        self,
        *,
        bytes_per_row: int,
    ) -> tuple[int, tuple[tuple[int, int], ...], int]:
        n_rows = len(self._query_cell_indices)
        resident = self._resident_bytes
        preferred = min(
            n_rows,
            row_band(
                self._source_geometry,
                unit="chunk",
                fallback=n_rows,
            ),
        )

        def fits(rows: int) -> bool:
            return (
                resident + rows * max(1, int(bytes_per_row)) + self._decode_bytes
                <= self._resources.memoryBytes
            )

        block_rows = affordable_width(fits, preferred)
        if block_rows < 1:
            minimum = resident + max(1, int(bytes_per_row)) + self._decode_bytes
            raise MemoryBudgetError(
                f"One aligned row needs about {minimum} bytes, but the operation "
                f"limit is {self._resources.memoryBytes} bytes",
                needed_bytes=minimum,
                budget_bytes=int(self._resources.memoryBytes),
                item="one aligned row beside the resident data",
            )
        boundaries = tuple(contiguous_ranges(n_rows, block_rows))
        admission = admit_stream(
            self._resources,
            nBlocks=1,
            blockBytes=max(1, block_rows * int(bytes_per_row)),
            decodeBytes=self._decode_bytes,
            residentBytes=resident,
            requested=1,
        )
        return block_rows, boundaries, admission.ioConcurrency

    def _read_raw(
        self,
        start: int,
        end: int,
        columns: np.ndarray,
    ) -> np.ndarray:
        rows = self._query_cell_indices[start:end]
        if isinstance(self._raw_backing, np.ndarray):
            return np.asarray(self._raw_backing[np.ix_(rows, columns)])
        return _read_block(self._raw_backing, rows, columns)

    def iter_blocks(self) -> Generator[AlignedFeatureBlock, None, None]:
        """Return a fresh iterator over normalized aligned row blocks."""

        def read(boundary: tuple[int, int]) -> tuple[int, np.ndarray]:
            start, end = boundary
            return start, self._read_raw(
                start,
                end,
                self._query_index_map,
            )

        # Close the block reader when this iterator closes, so a stopped reader
        # releases its last block then rather than at collection.
        with closing(
            stream_shards(
                self._row_geometry.boundaries,
                read,
                workers=1,
                io_concurrency=self._io_concurrency,
                total=len(self._row_geometry.boundaries),
            )
        ) as raw_blocks:
            for start, raw in raw_blocks:
                if self._column_starts is not None:
                    # Query features that share a reference feature's name add
                    # their counts, in float64, which sums integer counts exactly.
                    raw = np.add.reduceat(
                        raw, self._column_starts, axis=1, dtype=np.float64
                    )
                observed = np.count_nonzero(raw, axis=1) > 0
                normalized = normalize_reference_counts(
                    raw,
                    size_factor=self.size_factor,
                    log_transform=self.log_transform,
                    source=self._totals_source,
                    denominator=(
                        None
                        if self._cell_scalars is None
                        else self._cell_scalars[start : start + len(raw)]
                    ),
                )

                values = np.empty(
                    (len(raw), len(self._reference_feature_keys)),
                    dtype=np.float64,
                )
                if self._missing_feature_policy == "reference_mean":
                    values[:] = self._reference_normalized_means
                else:
                    values.fill(0)
                values[:, self._reference_index_map] = normalized
                yield AlignedFeatureBlock(
                    row_offset=start,
                    values=values,
                    observed=observed,
                )
                del raw, normalized, values, observed
