from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from ..utils.arrays import assay_feature_ranges
from ..utils.logging import logger
from ._assay_names import (
    AUTO_ASSAY_NAMES,
    auto_name_feat_table,
    make_feat_table_from_types,
)
from ._h5ad_columns import (
    SPARSE_KEYS,
    column_length,
    index_key as _index_key,
    is_column,
    present_column,
    read_table_column,
    sparse_encoding,
    sparse_shape,
    table_column_names,
)
from ._text import as_text as _as_text

_FEATURE_ID_KEYS = (
    "_index",
    "gene_ids",
    "gene_id",
    "ensembl_id",
    "feature_ids",
    "feature_id",
    "id",
    "index",
)
_FEATURE_NAME_KEYS = (
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "feature_name",
    "gene_short_name",
    "name",
    "index",
)
_CELL_ID_KEYS = (
    "_index",
    "index",
    "cell_id",
    "cell_ids",
    "barcode",
    "barcodes",
)
_LEGACY_CATEGORY_GROUPS = ("__categories", "categories")
_NON_MATRIX_PREFIXES = (
    "obs/",
    "var/",
    "obsm/",
    "varm/",
    "obsp/",
    "varp/",
    "uns/",
    "raw/var/",
    "raw/varm/",
)


@dataclass(frozen=True)
class H5adInspectResult:
    h5adFn: str
    matrixKey: str
    matrixCandidates: tuple[str, ...]
    matrixEncoding: str
    integerLike: bool
    cellAttrsKey: str
    cellIdsKey: str
    featureAttrsKey: str
    featureIdsKey: str
    featureNameKey: str
    categoryNamesKey: str
    assaySplitKey: str | None
    suggestedAssays: dict[str, int]
    layers: tuple[str, ...]
    title: str | None
    description: str | None
    nCells: int
    nFeatures: int

    def to_reader_kwargs(self) -> dict[str, Any]:
        return {
            "h5ad_fn": self.h5adFn,
            "cell_attrs_key": self.cellAttrsKey,
            "cell_ids_key": self.cellIdsKey,
            "feature_attrs_key": self.featureAttrsKey,
            "feature_ids_key": self.featureIdsKey,
            "feature_name_key": self.featureNameKey,
            "matrix_key": self.matrixKey,
            "category_names_key": self.categoryNamesKey,
        }


@dataclass(frozen=True)
class _MatrixCandidate:
    key: str
    encoding: str
    shape: tuple[int, int]
    integerLike: bool
    # Total of the matrix, estimated from the sampled values.
    estimatedTotal: float = 0.0

    @property
    def isSparse(self) -> bool:
        return self.encoding in {"csr", "csc"}


# The sample that decides whether a matrix holds counts: up to this many
# windows, spread evenly over its stored values, of up to this many values
# each. Reading a window decompresses at most the chunks that it overlaps, so
# inspection reads a bounded part of any file.
_SAMPLE_WINDOWS = 32
_SAMPLE_WINDOW_VALUES = 1024
# Matrices whose key names counts rank by these tiers, lowest first: raw
# counts, the canonical matrix, spliced counts, other matrices, and the
# unspliced and ambiguous counts of velocity data.
_COUNT_LAYER_NAMES = frozenset({"counts", "raw_counts"})
_SECONDARY_LAYER_NAMES = frozenset({"unspliced", "ambiguous"})


def _name_tier(key: str) -> int:
    """Return how conventional a name for a count matrix ``key`` is."""
    if key == "raw/X":
        return 0
    layer = key.removeprefix("layers/").lower() if key.startswith("layers/") else None
    if layer in _COUNT_LAYER_NAMES:
        return 0
    if key == "X":
        return 1
    if layer == "spliced":
        return 2
    if layer in _SECONDARY_LAYER_NAMES:
        return 4
    return 3


def _sample_windows(size: int, width: int) -> list[slice]:
    """Return up to ``_SAMPLE_WINDOWS`` windows spread evenly over ``size`` items."""
    if size <= 0:
        return []
    width = max(1, min(width, size))
    starts = np.linspace(0, size - width, num=min(_SAMPLE_WINDOWS, size)).astype(int)
    return [slice(start, start + width) for start in np.unique(starts).tolist()]


def _sampled_counts(sample: np.ndarray) -> bool:
    """Return whether sampled values are finite, non-negative, and whole."""
    if sample.size == 0 or not np.issubdtype(sample.dtype, np.number):
        return False
    if np.issubdtype(sample.dtype, np.integer):
        return bool(np.all(sample >= 0))
    return bool(
        np.all(np.isfinite(sample))
        and np.all(sample >= 0)
        and np.allclose(sample, np.round(sample), rtol=0, atol=1e-8)
    )


def _node_length(node: h5py.Group | h5py.Dataset | None) -> int | None:
    if node is None:
        return None
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.shape else None

    for key in (_index_key(node), "_index", "index"):
        if key is not None and key in node:
            length = column_length(node[key])
            if length is not None:
                return length
    for values in node.values():
        length = column_length(values)
        if length is not None:
            return length
    return None


def _sparse_counts(dataset: h5py.Dataset) -> tuple[bool, float]:
    """Sample the stored values of a sparse matrix.

    Returns whether every sampled value is a count, and the matrix total
    that the sampled mean estimates.
    """
    size = int(dataset.shape[0])
    windows = _sample_windows(size, _SAMPLE_WINDOW_VALUES)
    if not windows:
        return False, 0.0
    sample = np.concatenate([np.asarray(dataset[window]) for window in windows])
    if not _sampled_counts(sample):
        return False, 0.0
    return True, float(np.mean(sample, dtype=np.float64)) * size


def _dense_counts(dataset: h5py.Dataset) -> tuple[bool, float]:
    """Sample whole rows of a dense matrix, as :func:`_sparse_counts` samples."""
    n_rows, n_columns = (int(extent) for extent in dataset.shape)
    rows = max(1, _SAMPLE_WINDOW_VALUES // max(1, n_columns))
    windows = _sample_windows(n_rows, rows)
    if not windows or n_columns == 0:
        return False, 0.0
    sample = np.concatenate([np.asarray(dataset[window]) for window in windows])
    if not _sampled_counts(sample):
        return False, 0.0
    return True, float(np.mean(sample, dtype=np.float64)) * n_rows * n_columns


def _is_matrix_path(key: str) -> bool:
    return not key.startswith(_NON_MATRIX_PREFIXES)


def _matrix_candidates(h5: h5py.File) -> list[_MatrixCandidate]:
    candidates: list[_MatrixCandidate] = []

    def visit(key: str, node: h5py.Group | h5py.Dataset) -> None:
        if not _is_matrix_path(key):
            return
        if isinstance(node, h5py.Group) and SPARSE_KEYS.issubset(node.keys()):
            encoding = sparse_encoding(node)
            if encoding is None:
                logger.warning(
                    f"Ignoring sparse matrix candidate with unknown encoding: {key}"
                )
                return
            shape = sparse_shape(node)
            if shape is None:
                logger.warning(
                    f"Ignoring sparse matrix candidate without a shape attribute: {key}"
                )
                return
            integer_like, total = _sparse_counts(node["data"])
            candidates.append(
                _MatrixCandidate(
                    key=key,
                    encoding=encoding,
                    shape=shape,
                    integerLike=integer_like,
                    estimatedTotal=total,
                )
            )
        elif (
            isinstance(node, h5py.Dataset)
            # A null dataspace, as anndata before 0.13 writes ``X=None``, has
            # no shape.
            and node.shape is not None
            and len(node.shape) == 2
        ):
            if not np.issubdtype(node.dtype, np.number):
                return
            integer_like, total = _dense_counts(node)
            candidates.append(
                _MatrixCandidate(
                    key=key,
                    encoding="dense",
                    shape=(int(node.shape[0]), int(node.shape[1])),
                    integerLike=integer_like,
                    estimatedTotal=total,
                )
            )

    h5.visititems(visit)

    def rank(candidate: _MatrixCandidate) -> tuple[int, int, float, int, str]:
        # Counts take priority over storage layout, so a dense count matrix
        # outranks a sparse transformed layer. Among counts, a conventional
        # name, such as layers/counts or raw/X, outranks X, which outranks
        # spliced, other, and then unspliced or ambiguous counts; matrices of
        # one tier rank by their estimated total, so the matrix that holds
        # the most reads wins, and then by layout and key.
        integer = 0 if candidate.integerLike else 1
        total = -candidate.estimatedTotal if candidate.integerLike else 0.0
        sparse = 0 if candidate.isSparse else 1
        return integer, _name_tier(candidate.key), total, sparse, candidate.key

    return sorted(candidates, key=rank)


def _warn_competing_counts(
    chosen: _MatrixCandidate,
    candidates: list[_MatrixCandidate],
    obs_length: int | None,
) -> None:
    """Warn when other matrices of the same cells also hold counts."""
    if not chosen.integerLike:
        return
    others = [
        candidate
        for candidate in candidates
        if candidate is not chosen
        and candidate.integerLike
        and (obs_length is None or candidate.shape[0] == obs_length)
    ]
    if not others:
        return
    described = ", ".join(
        f"{candidate.key} (about {candidate.estimatedTotal:.3g} counts)"
        for candidate in (chosen, *others)
    )
    logger.warning(
        f"Several matrices hold integer counts: {described}. inspect_h5ad chose "
        f"{chosen.key}; review the choice and pass matrix_key to choose another"
    )


def _read_column(
    node: h5py.Group | h5py.Dataset,
    key: str,
) -> np.ndarray | None:
    if isinstance(node, h5py.Dataset):
        if node.dtype.names is None or key not in node.dtype.names:
            return None
    elif key not in node or not is_column(node[key]):
        return None
    try:
        values, missing = read_table_column(
            node.file, node, key, _LEGACY_CATEGORY_GROUPS
        )
    except (TypeError, ValueError):
        return None
    return present_column(values, missing)


def _matching_key(names: list[str], preferences: tuple[str, ...]) -> str | None:
    normalized = {name.lower(): name for name in names}
    for preferred in preferences:
        if preferred in normalized:
            return normalized[preferred]
    return None


def _is_string_column(values: np.ndarray) -> bool:
    if values.dtype.kind in {"S", "U"}:
        return True
    if values.dtype.kind != "O":
        return False
    return all(
        value is None or isinstance(value, str | bytes | np.str_ | np.bytes_)
        for value in values[:100]
    )


def _distinct_count(values: np.ndarray) -> int:
    return len({None if value is None else _as_text(value) for value in values})


def _is_unique(values: np.ndarray, expected_length: int) -> bool:
    if values.ndim != 1 or len(values) != expected_length:
        return False
    return _distinct_count(values) == expected_length


def _table_index(node: h5py.Group | h5py.Dataset, names: list[str]) -> str | None:
    """Return the column that holds the dataframe index, if it has one."""
    for key in (_index_key(node), "_index", "index"):
        if key is not None and key in names:
            return key
    return None


def _is_identifier_column(values: np.ndarray | None, n_cells: int) -> bool:
    """Return whether a column holds unique text or integers, one per cell.

    Floating-point values, such as scores, are never identifiers, even when
    every value differs.
    """
    if values is None or not _is_unique(values, n_cells):
        return False
    return values.dtype.kind in "iu" or _is_string_column(values)


def _find_cell_ids(
    node: h5py.Group | h5py.Dataset | None,
    n_cells: int,
) -> str:
    if node is None:
        return "_index"
    names = table_column_names(node)
    index = _table_index(node, names)
    if index is not None and _is_identifier_column(_read_column(node, index), n_cells):
        return index
    normalized = {name.lower(): name for name in names}
    for preferred in _CELL_ID_KEYS:
        name = normalized.get(preferred)
        if name is not None and _is_identifier_column(
            _read_column(node, name), n_cells
        ):
            return name

    # Other columns hold IDs only as text: a column of distinct numbers, such
    # as a count per cell, is not one.
    for name in sorted(names):
        values = _read_column(node, name)
        if _is_unique_text(values, n_cells):
            return name

    if index is not None:
        logger.warning(
            f"The cell IDs in the obs index {index!r} repeat values, and no "
            "other column holds unique text IDs. Pass "
            "duplicate_cell_ids='make_unique' to H5adReader, also through "
            "H5adReader.from_inspect, to rename repeated IDs"
        )
        return index
    logger.warning("No cell ID column found; generated IDs will be used")
    return "_index"


def _is_unique_text(values: np.ndarray | None, n_cells: int) -> bool:
    return (
        values is not None and _is_string_column(values) and _is_unique(values, n_cells)
    )


def _mean_text_length(values: np.ndarray) -> float:
    sample = [
        _as_text(value)
        for value in values[: min(100, len(values))]
        if value is not None
    ]
    if not sample:
        return 0
    return float(np.mean([len(value) for value in sample]))


class _FeatureColumns:
    """Read each candidate feature column once and classify it."""

    def __init__(self, node: h5py.Group | h5py.Dataset, n_features: int) -> None:
        self._node = node
        self._n = n_features
        self._values: dict[str, np.ndarray | None] = {}

    def values(self, name: str) -> np.ndarray | None:
        if name not in self._values:
            self._values[name] = _read_column(self._node, name)
        return self._values[name]

    def text(self, name: str) -> np.ndarray | None:
        values = self.values(name)
        if (
            values is None
            or values.ndim != 1
            or len(values) != self._n
            or not _is_string_column(values)
        ):
            return None
        return values

    def unique_text(self, name: str) -> bool:
        values = self.text(name)
        return values is not None and _is_unique(values, self._n)

    def mean_length(self, name: str) -> float:
        values = self.text(name)
        return 0.0 if values is None else _mean_text_length(values)

    def varied(self, name: str) -> bool:
        # Display names are nearly unique; feature types and genome names are
        # repeated labels.
        values = self.text(name)
        return values is not None and 2 * _distinct_count(values) > self._n


def _find_features(
    node: h5py.Group | h5py.Dataset,
    n_features: int,
) -> tuple[str, str]:
    names = table_column_names(node)
    columns = _FeatureColumns(node, n_features)
    index = _table_index(node, names)
    explicit_ids = [
        name
        for preferred in _FEATURE_ID_KEYS
        for name in names
        if name.lower() == preferred and name != index and columns.unique_text(name)
    ]
    id_key: str | None = None
    if index is not None and columns.unique_text(index):
        # AnnData files often hold display symbols in the index next to an
        # identifier column such as ``gene_ids``; the identifiers win then.
        id_key = explicit_ids[0] if explicit_ids else index
    elif explicit_ids:
        id_key = explicit_ids[0]
    else:
        unique_columns = [name for name in names if columns.unique_text(name)]
        if unique_columns:
            id_key = max(
                unique_columns,
                key=lambda name: (columns.mean_length(name), name),
            )

    name_key = _matching_key(names, _FEATURE_NAME_KEYS)
    if name_key is not None and columns.text(name_key) is None:
        name_key = None
    if (
        name_key is None
        and index is not None
        and id_key not in {None, index}
        and columns.text(index) is not None
    ):
        name_key = index
    if name_key is None:
        alternatives = [
            name for name in names if name != id_key and columns.varied(name)
        ]
        if alternatives:
            name_key = min(
                alternatives,
                key=lambda name: (columns.mean_length(name), name),
            )

    if id_key is None and name_key is None:
        logger.warning("No feature ID or name column found; generated IDs will be used")
        return "_index", "_index"
    if id_key is None:
        id_key = name_key
    if name_key is None:
        name_key = id_key
    assert id_key is not None
    assert name_key is not None
    return id_key, name_key


def _category_names_key(
    *nodes: h5py.Group | h5py.Dataset | None,
) -> str:
    for node in nodes:
        if isinstance(node, h5py.Group) and "categories" in node:
            if isinstance(node["categories"], h5py.Group):
                return "categories"
    return "__categories"


def _read_text_scalar(
    h5: h5py.File,
    key: str,
    max_length: int,
) -> str | None:
    """Return a scalar or one-element text dataset, or None for any other node."""
    node = h5.get(key)
    if not isinstance(node, h5py.Dataset) or node.shape not in {(), (1,)}:
        return None
    value = node[()] if node.shape == () else node[0]
    return _as_text(value)[:max_length]


def _feature_group_for(key: str) -> str:
    return "raw/var" if key.startswith("raw/") else "var"


def _select_matrix(
    h5: h5py.File,
    candidates: list[_MatrixCandidate],
) -> tuple[_MatrixCandidate, str]:
    obs_length = _node_length(h5.get("obs"))
    lengths = {
        "raw/var": _node_length(h5.get("raw/var")),
        "var": _node_length(h5.get("var")),
    }

    for candidate in candidates:
        n_cells, n_features = candidate.shape
        if obs_length is not None and obs_length != n_cells:
            logger.warning(
                f"Ignoring matrix candidate {candidate.key}: "
                f"{n_cells} rows do not match obs length {obs_length}"
            )
            continue

        own_key = _feature_group_for(candidate.key)
        own_length = lengths[own_key]
        if own_length == n_features:
            return candidate, own_key
        if own_length is not None:
            logger.warning(
                f"Ignoring matrix candidate {candidate.key}: feature metadata "
                f"group `{own_key}` length {own_length} does not match feature "
                f"count {n_features}"
            )
            continue

        # The conventional feature group for this matrix is absent. Fall back to
        # a dimension-matched group, otherwise keep the conventional key so the
        # reader generates feature IDs rather than borrowing an unrelated table.
        other_key = "var" if own_key == "raw/var" else "raw/var"
        if lengths[other_key] == n_features:
            return candidate, other_key
        if lengths[other_key] is None:
            return candidate, own_key
        logger.warning(
            f"Ignoring matrix candidate {candidate.key}: no feature metadata "
            f"group has {n_features} rows"
        )

    raise ValueError("No matrix candidate matches the obs and var dimensions")


def _suggested_assays(feature_types: list[str]) -> dict[str, int]:
    ranges = assay_feature_ranges(
        auto_name_feat_table(make_feat_table_from_types(feature_types))
    )
    return {
        name: sum(end - start for start, end in spans) for name, spans in ranges.items()
    }


def inspect_h5ad(
    h5ad_fn: str,
    *,
    matrix_key: str | None = None,
) -> H5adInspectResult:
    """Report the matrix and metadata layout of an H5AD file.

    Candidate matrices are ranked so that the counts come first. A matrix
    holds counts when a sample of its stored values, up to 32 windows of 1,024
    values spread over the whole matrix, is finite, non-negative, and whole.
    Among counts, a conventional name wins: ``raw/X``, ``layers/counts``, and
    ``layers/raw_counts`` first, then ``X``, ``layers/spliced``, other
    matrices, and the ``unspliced`` and ``ambiguous`` layers of velocity data
    last. Matrices of one tier rank by their total, estimated from the
    sample, so the one that holds the most reads wins. When other matrices of
    the cells also hold counts, a warning lists them with their estimated
    totals; review ``matrixKey`` and ``matrixCandidates`` and pass
    ``matrix_key`` to choose another.

    Args:
        h5ad_fn: Path to the H5AD file.
        matrix_key: Optional matrix path to force (for example ``X`` or
            ``raw/X``). When set, that candidate must exist and match obs/var
            dimensions.

    Returns:
        Keys, shape, and column names needed to configure
        :class:`~cytearc.readers.H5adReader`.
    """
    with h5py.File(h5ad_fn, mode="r") as h5:
        candidates = _matrix_candidates(h5)
        if not candidates:
            raise ValueError("No sparse or numeric 2D matrix found in the H5AD file")

        if matrix_key is not None:
            forced = [
                candidate for candidate in candidates if candidate.key == matrix_key
            ]
            if not forced:
                available = ", ".join(candidate.key for candidate in candidates)
                raise ValueError(
                    f"matrix_key {matrix_key!r} not found. Available: {available}"
                )
            matrix, feature_attrs_key = _select_matrix(h5, forced)
        else:
            matrix, feature_attrs_key = _select_matrix(h5, candidates)
            _warn_competing_counts(matrix, candidates, _node_length(h5.get("obs")))
        n_cells, n_features = matrix.shape
        cell_node = h5.get("obs")
        feature_node = h5.get(feature_attrs_key)
        if feature_node is None or not isinstance(
            feature_node, h5py.Group | h5py.Dataset
        ):
            feature_ids_key = "_index"
            feature_name_key = "_index"
        else:
            feature_ids_key, feature_name_key = _find_features(feature_node, n_features)

        cell_ids_key = _find_cell_ids(
            cell_node if isinstance(cell_node, h5py.Group | h5py.Dataset) else None,
            n_cells,
        )
        category_names_key = _category_names_key(
            cell_node if isinstance(cell_node, h5py.Group | h5py.Dataset) else None,
            feature_node
            if isinstance(feature_node, h5py.Group | h5py.Dataset)
            else None,
        )

        assay_split_key = None
        suggested_assays: dict[str, int] = {}
        if isinstance(feature_node, h5py.Group | h5py.Dataset):
            feature_columns = table_column_names(feature_node)
            assay_split_key = _matching_key(
                feature_columns, ("feature_types", "feature_type")
            )
            if assay_split_key is not None:
                values = _read_column(feature_node, assay_split_key)
                if values is None or len(values) != n_features:
                    assay_split_key = None
                else:
                    feature_types = [_as_text(value) for value in values]
                    # CELLxGENE stores Ensembl biotypes in feature_type. Those are
                    # not assay modalities; splitting on them invents thousands of
                    # ASSAY* spans. Require at least one known modality label.
                    if not any(
                        feature_type in AUTO_ASSAY_NAMES
                        for feature_type in feature_types
                    ):
                        assay_split_key = None
                    else:
                        suggested_assays = _suggested_assays(feature_types)

        layers_node = h5.get("layers")
        layers = (
            tuple(sorted(layers_node.keys()))
            if isinstance(layers_node, h5py.Group)
            else ()
        )
        title = _read_text_scalar(h5, "uns/title", 500)
        description = _read_text_scalar(h5, "uns/description", 10_000)
        if description is None:
            description = _read_text_scalar(h5, "uns/citation", 10_000)

    return H5adInspectResult(
        h5adFn=h5ad_fn,
        matrixKey=matrix.key,
        matrixCandidates=tuple(candidate.key for candidate in candidates),
        matrixEncoding=matrix.encoding,
        integerLike=matrix.integerLike,
        cellAttrsKey="obs",
        cellIdsKey=cell_ids_key,
        featureAttrsKey=feature_attrs_key,
        featureIdsKey=feature_ids_key,
        featureNameKey=feature_name_key,
        categoryNamesKey=category_names_key,
        assaySplitKey=assay_split_key,
        suggestedAssays=suggested_assays,
        layers=layers,
        title=title,
        description=description,
        nCells=n_cells,
        nFeatures=n_features,
    )
