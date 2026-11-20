import sys
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack

from ..utils.arrays import (
    assay_feature_ranges,
    canonicalize_sparse,
    csr_to_coo,
    cumulative_nnz,
    max_window_nnz,
)
from ..utils.count_values import (
    CountScan,
    CountValueRange,
    dense_row_nonzeros,
    merge_count_scans,
    scan_processes,
    value_windows,
)
from ..utils.arguments import integer_argument, quoted_value
from ..utils.logging import logger
from ..utils.matrix_values import DETECTION_CELLS
from ..utils.memory_budget import MemoryBudgetError, refusal_words
from ..utils.progress import iter_progress
from ._assay_names import auto_name_feat_table, make_feat_table_from_types
from ._h5ad_columns import (
    SPARSE_KEYS,
    CategoryCodes,
    H5adCategories,
    TableColumn,
    categories_node,
    categories_problem,
    category_colors,
    check_column_text,
    column_categories,
    column_encoding,
    column_length,
    column_place,
    decoded_column,
    index_key,
    is_column,
    is_nullable,
    present_column,
    read_table_column,
    sparse_encoding,
    sparse_shape,
    table_column,
    table_column_dtype,
    table_column_names,
    table_members,
    values_problem,
)
from ._h5ad_inspect import H5adInspectResult, inspect_h5ad as inspect_h5ad
from ._h5ad_values import (
    RowSpill,
    matrix_values,
    scan_file_window,
    scan_window,
)
from ._row_spill import (
    RowSpillPlan,
    checked_destinations,
    dense_sample_rows,
    dense_spill_estimate,
    row_spill_plan,
    sample_rows,
    spill_rows,
)
from ._sparse import SparseRowStore, encoded_entry_bytes
from ._text import (
    DuplicateCellIds,
    DuplicateFeatureIds,
    as_text,
    check_identifiers,
    check_utf8,
    duplicate_cell_ids_option,
    duplicate_feature_ids_option,
    unique_cell_identifiers,
    unique_feature_identifiers,
)

type H5adEmbeddingRole = Literal["umap", "tsne", "coordinates"]

_EMBEDDING_ROLES = frozenset({"umap", "tsne", "coordinates"})
# AnnData stores an unnamed dataframe index under this name, which no
# dataframe column may take.
_UNNAMED_INDEX = "_index"
# The name under which a table's unnamed index imports when it is neither the
# ID nor the name column.
_SOURCE_INDEX = "source_index"
# The bytes that the CSC row conversion holds for each entry of a chunk of
# columns as it reads the chunk and sends its entries to their rows' buckets.
_CSC_ENTRY_BYTES = 384
# The most entries of one chunk of columns that the conversion reads. Each
# chunk gives a row one piece, so larger chunks give fewer, longer pieces.
_CSC_CHUNK_ENTRIES = 8 * 1024 * 1024
# What the row bucket stores of the reader do, as their errors name it.
_CSC_OPERATION = "CSC row conversion"
_SHUFFLE_OPERATION = "shuffled H5AD import"


def _texts(values: np.ndarray) -> list[str | None]:
    return [None if value is None else as_text(value) for value in values]


def _stored_texts(values: np.ndarray) -> np.ndarray:
    """Return IDs or names as the text that an import stores.

    Byte strings are decoded as UTF-8, and other values, such as integers,
    become their ``str`` text.
    """
    return np.asarray([as_text(value) for value in values.astype(object)], dtype=str)


def _decoded_texts(values: np.ndarray) -> np.ndarray:
    """Return a column's values with byte strings as text, in an object array.

    The caller checked that the byte strings are UTF-8 text. Other values,
    such as numbers or None in missing rows, stay as they are, and so does
    a column of a dtype that holds no byte strings.
    """
    if values.dtype.kind not in "OS":
        return values
    texts = np.empty(values.shape, dtype=object)
    texts[:] = [
        value.decode("utf-8") if isinstance(value, bytes) else value
        for value in values.tolist()
    ]
    return texts


def _is_missing_name(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float | np.floating):
        return not bool(np.isfinite(value))
    return not as_text(value).strip()


# What refusals call the IDs, names and other text that an import stores.
_CELL_IDS = "H5AD cell IDs"
_FEATURE_IDS = "H5AD feature IDs"
_FEATURE_NAMES = "H5AD feature names"
_CELL_METADATA = "H5AD cell metadata"
_FEATURE_METADATA = "H5AD feature metadata"
_CLUSTER_LABELS = "H5AD cluster labels"
_FEATURE_TYPES = "H5AD feature types"
_MEMBERSHIP_COLUMNS = "H5AD membership column names"


def _array_bytes(values: np.ndarray) -> int:
    """Return the bytes of an array, with those of the objects an object array holds."""
    held = int(values.nbytes)
    if values.dtype.kind == "O":
        held += sum(map(sys.getsizeof, values))
    return held


def _matrix_group_contents(group: h5py.Group) -> str:
    """Describe what a matrix group holds: its declared encoding and members."""
    declared = group.attrs.get("encoding-type", group.attrs.get("h5sparse_format"))
    encoded = "" if declared is None else f" encoded as `{as_text(declared)}`"
    members = sorted(group.keys())
    if not members:
        return f"an empty group{encoded}"
    shown = ", ".join(f"`{name}`" for name in members[:5])
    more = f" and {len(members) - 5} more members" if len(members) > 5 else ""
    return f"a group{encoded} holding {shown}{more}"


@dataclass(frozen=True)
class _H5adAssayFeatures:
    featureIndexes: np.ndarray
    featureIds: np.ndarray
    featureNames: np.ndarray


class H5adReader:
    """A class to read in data from a H5ad file (h5 file with AnnData
    information).

    Args:
        h5ad_fn: Path to H5AD file
        cell_attrs_key: H5 group under which cell attributes are saved.(Default value: 'obs')
        feature_attrs_key: H5 group under which feature attributes are saved.(Default value: 'var')
        cell_ids_key: Key in `obs` group that contains unique cell IDs. By default the index will be used.
        feature_ids_key: Key in `var` group that contains unique feature IDs. By default the index will be used.
        feature_name_key: Key in `var` group that contains feature names. (Default: gene_short_name)
        matrix_key: Group where in the sparse matrix resides (default: 'X')
        category_names_key: Looks up this group and replaces the values in `var` and 'obs' child datasets with the
                            corresponding index value within this group.
        temp_dir: Parent directory for temporary CSC row storage. None uses the system temporary directory.
        duplicate_feature_ids: ``"error"`` rejects repeated feature IDs. ``"make_unique"`` keeps the
                               first occurrence of each ID and renames each later one, in file order
                               and across assays, by appending ``-1``, ``-2``, and so on (skipping
                               names that are already IDs), as AnnData's ``var_names_make_unique``
                               does, and logs a warning. Feature names are never changed: without a
                               feature name column they are the IDs before any rename.
        duplicate_cell_ids: ``"error"`` rejects repeated cell IDs. ``"make_unique"`` renames each
                            later occurrence of a repeated cell ID as ``duplicate_feature_ids``
                            renames feature IDs, as AnnData's ``obs_names_make_unique`` does,
                            which suits files that ``anndata.concat`` wrote without
                            ``index_unique``.

    A table's dataframe index that is neither the ID nor the name column
    imports as a metadata column only when it holds other values: an index
    that repeats the IDs or the names is left out, and AnnData's unnamed
    index, which no dataframe column may be named, imports as
    ``source_index`` (``source_index_2`` and so on when that name is taken).
    Categorical columns keep their category order, unused categories, and
    ``ordered`` flag, which ``H5adToZarr`` stores with the column; ``obs``
    categories also keep the colors that ``uns["<column>_colors"]`` holds.
    A missing or blank feature name takes the feature's ID before any rename.

    Attributes:
        h5: A File object from the h5py package.
        matrixKey: Group where in the sparse matrix resides (default: 'X')
        cellAttrsKey: Group wherein the cell attributes are present
        featureAttrsKey: Group wherein the feature attributes are present
        groupCodes: Used to ensure compatibility with different AnnData versions.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        cellIdsKey: Key in `obs` group that contains unique cell IDs. By default the index will be used.
        featIdsKey: Key in `var` group that contains unique feature IDs. By default the index will be used.
        featNamesKey: Key in `var` group that contains feature names. (Default: gene_short_name)
        catNamesKey: Looks up this group and replaces the values in `var` and 'obs' child datasets with the
                     corresponding index value within this group.
        sourceMatrixDtype: dtype in which the source holds the matrix values, in native byte order.
                           float16 is not a count storage dtype, so float16 source values are read
                           as float32.
        embedding_roles: Exact ``obsm`` keys to import as immutable artifacts:
                         ``"umap"`` and ``"tsne"`` layouts become ``embedding``
                         artifacts, and ``"coordinates"``, such as ``X_pca``,
                         ``X_scVI``, or ``X_pca_harmony``, become
                         ``imported_coordinates`` artifacts that graph
                         construction, embeddings, and mapping accept.
        cluster_keys: Exact ``obs`` keys to import as immutable cluster-label
                      artifacts instead of raw cell metadata.
        duplicateFeatureIds: How repeated feature IDs are treated.
        duplicateCellIds: How repeated cell IDs are treated.
    """

    def __init__(
        self,
        h5ad_fn: str,
        cell_attrs_key: str = "obs",
        cell_ids_key: str = "_index",
        feature_attrs_key: str = "var",
        feature_ids_key: str = "_index",
        feature_name_key: str = "gene_short_name",
        matrix_key: str = "X",
        obsm_attrs_key: str = "obsm",
        category_names_key: str = "__categories",
        embedding_roles: Mapping[str, H5adEmbeddingRole] | None = None,
        cluster_keys: Sequence[str] = (),
        *,
        temp_dir: str | Path | None = None,
        duplicate_feature_ids: DuplicateFeatureIds = "error",
        duplicate_cell_ids: DuplicateCellIds = "error",
    ) -> None:
        self.h5adFn = h5ad_fn
        self._tempDir = temp_dir
        self.duplicateFeatureIds = duplicate_feature_ids_option(duplicate_feature_ids)
        self.duplicateCellIds = duplicate_cell_ids_option(duplicate_cell_ids)
        # The feature and cell IDs as the file holds them and with any
        # repeated ID renamed, read on first use.
        self._featureIdentifiers: tuple[np.ndarray, np.ndarray] | None = None
        self._cellIdentifiers: tuple[np.ndarray, np.ndarray] | None = None
        self._featureNames: np.ndarray | None = None
        self.h5: h5py.File = h5py.File(h5ad_fn, mode="r")
        try:
            self.matrixKey = matrix_key
            self.cellAttrsKey, self.featureAttrsKey, self.obsmAttrsKey = (
                cell_attrs_key,
                feature_attrs_key,
                obsm_attrs_key,
            )
            self.groupCodes: dict[str, int] = {
                self.cellAttrsKey: self._validate_group(self.cellAttrsKey),
                self.featureAttrsKey: self._validate_group(self.featureAttrsKey),
                self.obsmAttrsKey: self._validate_group(self.obsmAttrsKey),
                self.matrixKey: self._matrix_slot_code(),
            }
            # A malformed column-order, and an attribute or a member name of a
            # table that is not UTF-8 text, must fail here, before a writer
            # opens its destination.
            for group in (self.cellAttrsKey, self.featureAttrsKey):
                if self.groupCodes[group] == 2:
                    table_members(self.h5[group])
            self.matrixOrientation = self._validate_sparse_matrix()
            self.sourceMatrixDtype: Any = self._get_matrix_dtype()
            # The rows that consume yields from local disk: a CSC matrix's
            # converted rows, or an import's shuffled rows.
            self._convertedCsr: SparseRowStore | None = None
            # Whether the rows on local disk are in an import's shuffled
            # order, which its counts write reads once, deleting them as it
            # goes, and releases once written.
            self._shuffledRows = False
            # The file's first rows, which decide what the values are, kept
            # by a shuffled conversion of CSC columns.
            self._sourceFirstRows: csr_matrix | None = None
            self._indptrCache: np.ndarray | None = None
            # The cumulative nonzero values of a dense matrix's rows, which
            # plan its sparse batches as CSR row pointers do.
            self._cumulativeRowNnz: np.ndarray | None = None
            # Whether every row of a CSR matrix holds its indices sorted
            # without repeats, which the value scan finds; None before it.
            self._canonicalRows: bool | None = None
            matrix_shape = self._matrix_shape()
            self.nCells, self.nFeatures = (
                self._get_n(self.cellAttrsKey, matrix_shape[0]),
                self._get_n(self.featureAttrsKey, matrix_shape[1]),
            )
            if (self.nCells, self.nFeatures) != matrix_shape:
                raise ValueError(
                    f"ERROR: Matrix `{self.matrixKey}` has shape {matrix_shape}, "
                    f"but `{self.cellAttrsKey}` has {self.nCells} rows and "
                    f"`{self.featureAttrsKey}` has {self.nFeatures} rows"
                )
            self.cellIdsKey = self._fix_name_key(self.cellAttrsKey, cell_ids_key)
            self.featIdsKey = self._fix_name_key(self.featureAttrsKey, feature_ids_key)
            self.featNamesKey = feature_name_key
            self.catNamesKey = category_names_key
            self.embeddingRoles = self._validate_embedding_roles(embedding_roles)
            self.clusterKeys = self._validate_cluster_keys(cluster_keys)
        except BaseException:
            self.h5.close()
            raise

    def _clone_kwargs(self) -> dict[str, Any]:
        return {
            "h5ad_fn": self.h5adFn,
            "cell_attrs_key": self.cellAttrsKey,
            "cell_ids_key": self.cellIdsKey,
            "feature_attrs_key": self.featureAttrsKey,
            "feature_ids_key": self.featIdsKey,
            "feature_name_key": self.featNamesKey,
            "matrix_key": self.matrixKey,
            "obsm_attrs_key": self.obsmAttrsKey,
            "category_names_key": self.catNamesKey,
            "embedding_roles": dict(self.embeddingRoles),
            "cluster_keys": self.clusterKeys,
            "temp_dir": self._tempDir,
            "duplicate_feature_ids": self.duplicateFeatureIds,
            "duplicate_cell_ids": self.duplicateCellIds,
        }

    def _count_matrix_keys(self) -> list[str]:
        """Return the keys of the file's matrices whose sampled values are counts."""
        from ._h5ad_inspect import _matrix_candidates

        return [
            candidate.key
            for candidate in _matrix_candidates(self.h5)
            if candidate.integerLike
        ]

    def close(self) -> None:
        self.h5.close()
        self._release_rows()
        self._indptrCache = None
        self._cumulativeRowNnz = None
        self._canonicalRows = None

    def _release_rows(self) -> None:
        """Close the rows on local disk, removing them if this reader made them."""
        if self._convertedCsr is not None:
            self._convertedCsr.close()
        self._convertedCsr = None
        self._shuffledRows = False
        self._sourceFirstRows = None

    def _release_shuffled_rows(self) -> None:
        """Release an import's shuffled rows once its counts are written.

        Closing the store removes the buckets that no producer deleted, and
        ``consume`` then yields the rows in the file's order again.
        """
        if self._shuffledRows:
            self._release_rows()

    def _open_shuffled_rows(self, directory: str) -> None:
        """Yield the shuffled rows of an import from its finalized store in ``directory``.

        A producer process of the counts write opens the store that the
        import's parent process made; reads delete each bucket once read,
        and closing the reader leaves the rest to the parent.
        """
        self._release_rows()
        self._convertedCsr = SparseRowStore.open(directory)
        self._shuffledRows = True

    @classmethod
    def from_inspect(
        cls,
        inspection: H5adInspectResult,
        **overrides: Any,
    ) -> "H5adReader":
        """Open a reader with the keys that :func:`~cytearc.inspect_h5ad` found.

        Args:
            inspection: Result of :func:`~cytearc.inspect_h5ad`.
            **overrides: Constructor arguments that replace or add to the
                inspected ones, such as ``embedding_roles``, ``temp_dir``,
                ``duplicate_feature_ids``, or ``duplicate_cell_ids``.
        """
        reader_kwargs = inspection.to_reader_kwargs()
        reader_kwargs.update(overrides)
        return cls(**reader_kwargs)

    def _validate_sparse_matrix(self) -> str:
        if self.groupCodes[self.matrixKey] != 2:
            return "dense"

        group = self.h5[self.matrixKey]
        missing = SPARSE_KEYS.difference(group.keys())
        if missing:
            contents = _matrix_group_contents(group)
            if len(group):
                lacking = ", ".join(f"`{key}`" for key in sorted(missing))
                contents += f" but not {lacking}"
            raise ValueError(
                f"ERROR: Matrix `{self.matrixKey}` is {contents}. H5adReader reads "
                "a matrix from a two-dimensional dense dataset, or from a CSR or "
                "CSC group holding `data`, `indices` and `indptr`."
            )

        encoding = sparse_encoding(group)
        if encoding is None:
            declared = group.attrs.get(
                "encoding-type", group.attrs.get("h5sparse_format")
            )
            if declared is None:
                # A square matrix fits either layout, so the indptr's length
                # cannot tell them apart.
                raise ValueError(
                    f"ERROR: Matrix `{self.matrixKey}` is "
                    f"{_matrix_group_contents(group)} without an `encoding-type` "
                    "or `h5sparse_format` attribute, so H5adReader cannot tell CSR "
                    "from CSC. Set its `encoding-type` attribute to `csr_matrix` or "
                    "`csc_matrix`, as AnnData writes it."
                )
            raise ValueError(
                f"ERROR: Sparse matrix encoding `{declared}` of `{self.matrixKey}` "
                "is not supported. H5adReader supports CSR and CSC encoding."
            )
        return encoding

    def _matrix_slot_code(self) -> int:
        """Return the group code of the matrix slot.

        An empty table group stands for an absent table, but an empty matrix
        group holds no matrix, so any group keeps code 2 for
        :meth:`_validate_sparse_matrix` to check. A dataset or another kind
        of node takes the code of :meth:`_validate_group`.

        Raises:
            ValueError: If the file has nothing at ``matrix_key``.
        """
        node = self.h5.get(self.matrixKey)
        if node is None:
            raise ValueError(
                f"ERROR: The file has no matrix `{self.matrixKey}`. Pass "
                "`matrix_key` to read another of its matrices, which "
                "`inspect_h5ad` lists."
            )
        if isinstance(node, h5py.Group):
            return 2
        return self._validate_group(self.matrixKey)

    def _validate_group(self, group: str) -> int:
        if group not in self.h5:
            logger.warning(f"`{group}` group not found in the H5ad file")
            ret_val = 0
        elif isinstance(self.h5[group], h5py.Dataset):
            ret_val = 1
        elif isinstance(self.h5[group], h5py.Group):
            ret_val = 2
        else:
            logger.warning(
                f"`{group}` slot in H5ad file is not of Dataset or Group type. "
                f"Due to this, no information in `{group}` can be used"
            )
            ret_val = 0
        if ret_val == 2:
            if len(self.h5[group].keys()) == 0:
                # AnnData writes an empty obsm whenever a file holds no
                # embeddings, so an empty group is ordinary; a requested
                # embedding that it lacks raises where it is read.
                logger.debug(f"`{group}` slot in H5ad file is empty.")
                ret_val = 0
            elif (
                len(
                    set(
                        [
                            self.h5[group][x].shape[0]
                            for x in self.h5[group].keys()
                            if isinstance(self.h5[group][x], h5py.Dataset)
                        ]
                    )
                )
                > 1
            ):
                logger.warning(
                    f"`{group}` slot in H5ad file has unequal sized child groups"
                )
        return ret_val

    def _get_matrix_dtype(self) -> Any:
        """Return the stored dtype in native byte order, with float16 as float32.

        SciPy sparse matrices hold neither float16 nor another byte order, and
        float16 is not a count storage dtype, so the reader yields matrix
        values in this dtype.
        """
        if self.groupCodes[self.matrixKey] == 1:
            dtype = self.h5[self.matrixKey].dtype
        elif self.groupCodes[self.matrixKey] == 2:
            dtype = self.h5[self.matrixKey]["data"].dtype
        else:
            raise ValueError(
                f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
            )
        dtype = dtype.newbyteorder("=")
        return np.dtype(np.float32) if dtype == np.float16 else dtype

    def _matrix_values(self, node: h5py.Dataset, selection: slice) -> np.ndarray:
        """Read matrix values in ``sourceMatrixDtype``."""
        return matrix_values(node, selection, self.sourceMatrixDtype)

    def _matrix_shape(self) -> tuple[int, int]:
        """Return the stored matrix shape after checking that it is consistent.

        A sparse group must record its shape in an attribute, and its indptr
        must hold one pointer per row (CSR) or column (CSC) of that shape.
        """
        matrix = self.h5[self.matrixKey]
        if self.matrixOrientation == "dense":
            if matrix.ndim != 2:
                raise ValueError(
                    f"ERROR: Dense matrix `{self.matrixKey}` must be two-dimensional"
                )
            return int(matrix.shape[0]), int(matrix.shape[1])
        shape = sparse_shape(matrix)
        if shape is None:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` has no shape attribute"
            )
        expected = ((shape[0] if self.matrixOrientation == "csr" else shape[1]) + 1,)
        if matrix["indptr"].shape != expected:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` of shape {shape} "
                f"needs an indptr of shape {expected}; found {matrix['indptr'].shape}"
            )
        return shape

    def _check_exists(self, group: str, key: str) -> bool:
        if group in self.groupCodes:
            group_code = self.groupCodes[group]
        else:
            group_code = self._validate_group(group)
            self.groupCodes[group] = group_code
        if group_code == 1:
            # A plain dataset has no fields, so it holds no named column.
            if key in (self.h5[group].dtype.names or ()):
                return True
        if group_code == 2:
            if key in self.h5[group].keys():
                return True
        return False

    def _fix_name_key(self, group: str, key: str) -> str:
        if self._check_exists(group, key):
            return key
        if key == "_index" and self.groupCodes.get(group) == 2:
            # AnnData stores a named dataframe index under its name and records
            # that name in the ``_index`` attribute.
            recorded = index_key(self.h5[group])
            if recorded is not None and self._check_exists(group, recorded):
                return recorded
        if key.startswith("_"):
            temp_key = key[1:]
            if self._check_exists(group, temp_key):
                return temp_key
        return key

    @property
    def _categoryGroups(self) -> tuple[str, ...]:
        return (self.catNamesKey,)

    def _read_column(
        self,
        group: str,
        key: str,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return read_table_column(
            self.h5,
            self.h5[group],
            key,
            self._categoryGroups,
            start,
            stop,
        )

    def _table_column(self, group: str, key: str) -> TableColumn:
        """Return a whole column, a categorical as its codes and categories."""
        return table_column(self.h5, self.h5[group], key, self._categoryGroups)

    def _validate_embedding_roles(
        self,
        roles: Mapping[str, H5adEmbeddingRole] | None,
    ) -> dict[str, H5adEmbeddingRole]:
        if roles is None:
            return {}
        if not isinstance(roles, Mapping):
            raise TypeError("embedding_roles must be a mapping")
        resolved: dict[str, H5adEmbeddingRole] = {}
        for raw_key, raw_role in roles.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError("embedding_roles keys must be non-empty strings")
            if raw_role not in _EMBEDDING_ROLES:
                raise ValueError(
                    "embedding_roles values must be 'umap', 'tsne', or 'coordinates'"
                )
            if not self._check_exists(self.obsmAttrsKey, raw_key):
                raise KeyError(
                    f"Embedding key {raw_key!r} was not found in {self.obsmAttrsKey}"
                )
            node = self.h5[self.obsmAttrsKey][raw_key]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f"Embedding key {raw_key!r} must be a dense H5AD array")
            if node.ndim != 2 or node.shape[0] != self.nCells or node.shape[1] < 1:
                raise ValueError(
                    f"Embedding key {raw_key!r} has incompatible shape {node.shape}"
                )
            if np.dtype(node.dtype).kind not in "biuf":
                raise TypeError(
                    f"Embedding key {raw_key!r} must contain numeric values"
                )
            resolved[raw_key] = raw_role
        return resolved

    def _validate_cluster_keys(self, keys: Sequence[str]) -> tuple[str, ...]:
        if isinstance(keys, str | bytes) or not isinstance(keys, Sequence):
            raise TypeError("cluster_keys must be a sequence of column names")
        resolved = tuple(keys)
        if len(set(resolved)) != len(resolved):
            raise ValueError("cluster_keys must be unique")
        for key in resolved:
            if not isinstance(key, str) or not key:
                raise ValueError("cluster_keys must contain non-empty strings")
            if key in {self.cellIdsKey, self.catNamesKey}:
                raise ValueError(f"Cluster key {key!r} is reserved H5AD metadata")
            if not self._check_exists(self.cellAttrsKey, key):
                raise KeyError(
                    f"Cluster key {key!r} was not found in {self.cellAttrsKey}"
                )
            cell_attrs = self.h5[self.cellAttrsKey]
            if isinstance(cell_attrs, h5py.Dataset):
                if cell_attrs.ndim != 1:
                    raise TypeError(
                        f"Cluster key {key!r} must contain one scalar value per cell"
                    )
                field_dtype = np.dtype(cell_attrs.dtype.fields[key][0])
                if field_dtype.subdtype is not None:
                    raise TypeError(
                        f"Cluster key {key!r} must contain one scalar value per cell; "
                        f"found vector dtype {field_dtype}"
                    )
                values_dtype = field_dtype
                length = int(cell_attrs.shape[0])
            else:
                node = cell_attrs[key]
                if not is_column(node):
                    raise TypeError(
                        f"Cluster key {key!r} uses unsupported H5AD encoding "
                        f"{column_encoding(node)!r}"
                    )
                value_node = (
                    node
                    if isinstance(node, h5py.Dataset)
                    else node["codes" if "codes" in node else "values"]
                )
                if value_node.ndim != 1:
                    raise TypeError(
                        f"Cluster key {key!r} must contain one scalar value per cell"
                    )
                if is_nullable(node) and node["mask"].shape != node["values"].shape:
                    raise ValueError(
                        f"Cluster key {key!r} has a misaligned missingness mask"
                    )
                values_dtype = value_node.dtype
                length = int(value_node.shape[0])
            if length != self.nCells:
                raise ValueError(
                    f"Cluster key {key!r} has {length} rows; expected {self.nCells}"
                )
            # A categorical's codes are integers, and its categories are checked.
            problem = values_problem(values_dtype)
            if problem is None:
                categories = categories_node(
                    self.h5, cell_attrs, key, self._categoryGroups
                )
                problem = None if categories is None else categories_problem(categories)
            if problem is not None:
                raise TypeError(f"Cluster key {key!r} {problem}")
            dtype = self._cell_column_value_dtype(key)
            if dtype.kind not in "biufOSU":
                raise TypeError(f"Cluster key {key!r} uses unsupported dtype {dtype}")
        return resolved

    def _get_n(self, group: str, matrix_length: int) -> int:
        """Return the row count of a table; an absent table takes the matrix's."""
        if self.groupCodes[group] == 0:
            return matrix_length
        if self.groupCodes[group] == 1:
            return int(self.h5[group].shape[0])
        table = self.h5[group]
        for key in (index_key(table), *table.keys()):
            if key is None or key not in table:
                continue
            length = column_length(table[key])
            if length is not None:
                return length
        raise KeyError(
            f"ERROR: `{group}` key doesn't contain any child node of Dataset type."
            f"Aborting because unexpected H5ad format."
        )

    def _column_place(self, group: str, key: str) -> str:
        """Name a column of a table of the file as refusals name it."""
        return column_place(
            str(self.h5adFn), group, key, index=key == self._index_member(group)
        )

    def _identifiers(
        self, group: str, key: str, generated: str, label: str
    ) -> np.ndarray:
        """Return a table's IDs as the file holds them, or IDs that the reader generates.

        Raises:
            ValueError: If the file holds IDs that an import cannot store:
                bytes that are not UTF-8 text, or missing or blank IDs.
        """
        if self._check_exists(group, key):
            values = present_column(*self._read_column(group, key)).astype(object)
            check_identifiers(values, label, self._column_place(group, key))
        else:
            n_rows = self.nCells if group == self.cellAttrsKey else self.nFeatures
            logger.warning(
                f"ID key {key!r} was not found in H5AD {group}; generated IDs "
                "will be used"
            )
            values = np.array([f"{generated}_{x}" for x in range(n_rows)])
        return values

    def _cell_identifiers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cell IDs as the file holds them and with any repeated ID renamed.

        They differ only when ``duplicate_cell_ids="make_unique"`` renamed
        repeated IDs, which happens once, on first use.
        """
        if self._cellIdentifiers is None:
            source = self._identifiers(
                self.cellAttrsKey, self.cellIdsKey, "cell", _CELL_IDS
            )
            resolved = unique_cell_identifiers(source, _CELL_IDS, self.duplicateCellIds)
            self._cellIdentifiers = (source, resolved)
        return self._cellIdentifiers

    def cell_ids(self) -> np.ndarray:
        """Return the cell IDs in the file's order, as the text that an import stores.

        The IDs are a NumPy array of ``str``, whatever the file holds: byte
        strings are decoded as UTF-8, and numbers become text, so ``101``
        becomes ``"101"``. Per-cell values keyed by these IDs, such as
        ``pd.Series(values, index=reader.cell_ids())`` for values in the
        file's order, match the imported store's cells in ``ds.cells.insert``.

        Repeated IDs raise ValueError, or are renamed with
        ``duplicate_cell_ids="make_unique"``. IDs that an import cannot
        store raise ValueError that names the file, the table, the column
        and the rows: missing IDs, such as NaN in an index of numbers, blank
        IDs, and bytes that are not UTF-8 text.
        """
        return _stored_texts(self._cell_identifiers()[1])

    def _cell_ids_at(self, rows: np.ndarray) -> np.ndarray:
        """Return the cell IDs of file rows ``rows``, in that order, as the text an import stores."""
        return _stored_texts(self._cell_identifiers()[1][rows])

    def _check_ids_and_names(self) -> None:
        """Read the cell IDs, feature IDs and feature names that an import stores.

        The reader keeps them, so an import that calls this before it reads
        the matrix reads them once, and refuses IDs and names that it cannot
        store before it writes anything.

        Raises:
            ValueError: If the reader refuses an ID or a name, as
                :meth:`cell_ids`, :meth:`feat_ids` and :meth:`feat_names`
                refuse them.
        """
        self._cell_identifiers()
        self._feature_identifiers()
        self._feature_names()

    def _check_metadata_text(self) -> None:
        """Refuse text that is not UTF-8 in the metadata and cluster labels that an import stores.

        An import calls this before it writes anything. It checks the
        metadata columns of ``obs`` and ``var``, as :meth:`_check_table_text`
        does, and the column of each cluster key.

        Raises:
            ValueError: If a column holds bytes that are not UTF-8 text,
                naming the file, the table, the column, and the row, or the
                category of a categorical, that holds them.
        """
        self._check_table_text(self.cellAttrsKey)
        self._check_table_text(self.featureAttrsKey)
        for key in self.clusterKeys:
            check_column_text(
                self.h5,
                self.h5[self.cellAttrsKey],
                key,
                self._categoryGroups,
                expectation=f"{_CLUSTER_LABELS} must be UTF-8 text",
                place=self._column_place(self.cellAttrsKey, key),
            )

    def _check_table_text(self, group: str) -> None:
        """Refuse text that is not UTF-8 in the metadata columns of a table.

        The columns are those that :meth:`_table_columns` reads, the index
        among them, and the check reads their text a block of rows at a
        time: the values of a column of text, except those of its missing
        rows, and every category of a categorical, used or not, which an
        import stores with the column.

        Raises:
            ValueError: If a column holds bytes that are not UTF-8 text, as
                :meth:`_check_metadata_text` refuses them.
        """
        code = self.groupCodes.get(group)
        if code not in {1, 2}:
            return
        table = self.h5[group]
        names: Sequence[str]
        if code == 1:
            names = table.dtype.names or ()
        else:
            names = [name for name, _node in table_members(table).members]
        left_out = self._left_out_columns(group)
        for name in names:
            if name in left_out:
                continue
            check_column_text(
                self.h5,
                table,
                name,
                self._categoryGroups,
                expectation=self._metadata_expectation(group),
                place=self._column_place(group, name),
            )

    def _left_out_columns(self, group: str) -> tuple[str, ...]:
        """Return the columns of a table that metadata leaves out.

        They are the IDs and names, which an import stores as such, the
        group of categories that old AnnData files keep beside the columns,
        and the cluster keys.
        """
        if group == self.cellAttrsKey:
            return (self.cellIdsKey, self.catNamesKey, *self.clusterKeys)
        return (self.featIdsKey, self.featNamesKey, self.catNamesKey)

    def _metadata_expectation(self, group: str) -> str:
        """Return what a refusal of a table's metadata says it must be."""
        held = _CELL_METADATA if group == self.cellAttrsKey else _FEATURE_METADATA
        return f"{held} must be UTF-8 text"

    def _id_bytes(self) -> int:
        """Return the bytes of the IDs and names that the reader keeps once it read them.

        It keeps the cell and feature IDs as the file holds them and, when
        it renamed repeated IDs, with those IDs renamed, and the feature
        names as the text that an import stores.
        """
        held = 0
        for source, resolved in (self._cell_identifiers(), self._feature_identifiers()):
            for values in (source,) if resolved is source else (source, resolved):
                held += _array_bytes(values)
        return held + _array_bytes(self._feature_names())

    def _feature_identifiers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the feature IDs as the file holds them and with any repeated ID renamed.

        They differ only when ``duplicate_feature_ids="make_unique"`` renamed
        repeated IDs, which happens once, on first use.
        """
        if self._featureIdentifiers is None:
            source = self._identifiers(
                self.featureAttrsKey, self.featIdsKey, "feature", _FEATURE_IDS
            )
            resolved = unique_feature_identifiers(
                source,
                _FEATURE_IDS,
                self.duplicateFeatureIds,
            )
            self._featureIdentifiers = (source, resolved)
        return self._featureIdentifiers

    def feat_ids(self) -> np.ndarray:
        """Return the feature IDs in the file's order, as the text that an import stores.

        The IDs are a NumPy array of ``str``, converted as :meth:`cell_ids`
        converts cell IDs.

        Repeated IDs raise ValueError, or are renamed with
        ``duplicate_feature_ids="make_unique"``. IDs that an import cannot
        store raise ValueError, as :meth:`cell_ids` describes.
        """
        return _stored_texts(self._feature_identifiers()[1])

    def feat_names(self) -> np.ndarray:
        """Return the feature names in the file's order, as the text that an import stores.

        The names are a NumPy array of ``str``, converted as :meth:`cell_ids`
        converts cell IDs. Without the feature name key, the names are the
        feature IDs before any repeated ID is renamed. A missing or blank
        name, such as a missing category, takes the ID of its feature before
        any rename. Bytes that are not UTF-8 text raise ValueError that names
        the file, the table, the column and the row.
        """
        return self._feature_names().copy()

    def _feature_names(self) -> np.ndarray:
        """Return the feature names that the reader keeps, read on first use."""
        if self._featureNames is None:
            self._featureNames = _stored_texts(self._read_feature_names())
        return self._featureNames

    def _read_feature_names(self) -> np.ndarray:
        if not self._check_exists(self.featureAttrsKey, self.featNamesKey):
            logger.warning(
                f"Feature name key {self.featNamesKey!r} was not found in "
                f"{self.featureAttrsKey}; feature IDs will be used"
            )
            return self._feature_identifiers()[0].copy()
        names = present_column(
            *self._read_column(self.featureAttrsKey, self.featNamesKey)
        ).astype(object)
        check_utf8(
            names,
            f"{_FEATURE_NAMES} must be UTF-8 text",
            self._column_place(self.featureAttrsKey, self.featNamesKey),
        )
        missing = np.asarray([_is_missing_name(name) for name in names], dtype=bool)
        if missing.any():
            names[missing] = self._feature_identifiers()[0][missing]
            logger.warning(
                f"{int(missing.sum())} feature names in {self.featNamesKey!r} are "
                "missing or blank; those features take their IDs as names"
            )
        return names

    def _index_member(self, group: str) -> str | None:
        """Return the source name of a dataframe group's index, if it has one."""
        if self.groupCodes.get(group) != 2:
            return None
        table = self.h5[group]
        recorded = index_key(table)
        for key in (recorded, _UNNAMED_INDEX):
            if key is not None and key in table:
                return key
        return None

    def _import_name(self, group: str, name: str) -> str:
        """Return the name under which a source column imports.

        Only AnnData's unnamed index changes its name, to the first of
        ``source_index``, ``source_index_2``, and so on that the table does
        not use.
        """
        if name != _UNNAMED_INDEX:
            return name
        taken = set(table_column_names(self.h5[group]))
        candidate, suffix = _SOURCE_INDEX, 2
        while candidate in taken:
            candidate, suffix = f"{_SOURCE_INDEX}_{suffix}", suffix + 1
        return candidate

    def _repeats_identifiers(
        self,
        group: str,
        values: np.ndarray,
        missing: np.ndarray,
    ) -> bool:
        """Return whether an index holds the IDs or the names of its table."""
        texts = _texts(present_column(values, missing))
        known: tuple[np.ndarray, ...]
        if group == self.cellAttrsKey:
            known = (self._cell_identifiers()[0],)
        else:
            known = (self._feature_identifiers()[0], self.feat_names())
        return any(texts == _texts(column) for column in known)

    def _table_columns(
        self, group: str, ignore_keys: Sequence[str]
    ) -> Iterator[tuple[str, TableColumn, H5adCategories | None]]:
        """Yield each decodable column with its values and categories.

        A column comes as its values and missing mask, or a categorical as
        its codes and categories, which :func:`decoded_column` decodes to
        labels. Columns come by the name they import under, which is their
        source name except for an unnamed index. A dataframe group lists them
        in ``column-order``, which resolves a name that old AnnData versions
        nested into groups because it contains ``/``. An index that repeats
        the table's IDs or names is left out. Callers check the table's text
        with :meth:`_check_table_text` first, so that its byte strings decode.
        """
        code = self.groupCodes[group]
        if code not in {1, 2}:
            return
        table = self.h5[group]
        nodes: dict[str, Any] = {}
        if code == 1:
            names: tuple[str, ...] = tuple(table.dtype.names or ())
        else:
            members = table_members(table)
            for name in members.unresolved:
                logger.warning(
                    f"Skipping {group} column {name!r} because column-order lists "
                    "it but the file does not contain it"
                )
            nodes = dict(members.members)
            names = tuple(nodes)
        index = self._index_member(group)
        for name in iter_progress(names, desc=f"Reading attributes from group {group}"):
            if name in ignore_keys:
                continue
            if code == 2 and not is_column(nodes[name]):
                if isinstance(nodes[name], h5py.Group):
                    logger.warning(
                        f"Skipping {group} column {name!r} because its H5AD encoding "
                        f"{column_encoding(nodes[name])!r} is not supported"
                    )
                continue
            try:
                column = self._table_column(group, name)
            except ValueError as error:
                logger.warning(f"Skipping {group} column {name!r}: {error}")
                continue
            ndim = (
                column.codes if isinstance(column, CategoryCodes) else column[0]
            ).ndim
            if ndim != 1:
                logger.warning(
                    f"Skipping {group} column {name!r} because it holds "
                    f"{ndim}-dimensional values"
                )
                continue
            if name == index and self._repeats_identifiers(
                group, *decoded_column(column)
            ):
                logger.info(
                    f"Leaving out the {group} index {name!r} because it holds the "
                    "IDs or the names that the import stores"
                )
                continue
            categories = column_categories(
                self.h5,
                table,
                name,
                self._categoryGroups,
                expectation=self._metadata_expectation(group),
                place=self._column_place(group, name),
            )
            yield self._import_name(group, name), column, categories

    def _source_column_names(self, group: str) -> list[str]:
        """Return the import names of the decodable columns of a table.

        The names include the ID, name, and cluster columns that metadata
        import leaves out, so a caller can plan storage keys over the whole
        table.
        """
        if self.groupCodes.get(group) not in {1, 2}:
            return []
        return [
            self._import_name(group, name)
            for name in table_column_names(self.h5[group])
        ]

    def _category_colors(
        self,
        name: str,
        categories: H5adCategories,
    ) -> tuple[str, ...] | None:
        """Return the ``uns`` colors of an ``obs`` categorical, in category order."""
        return category_colors(self.h5, name, len(categories.values))

    def _cell_column_categories(self, key: str) -> H5adCategories | None:
        """Return the categories of a cluster key's ``obs`` column."""
        return column_categories(
            self.h5,
            self.h5[self.cellAttrsKey],
            key,
            self._categoryGroups,
            expectation=f"{_CLUSTER_LABELS} must be UTF-8 text",
            place=self._column_place(self.cellAttrsKey, key),
        )

    def _cell_column_value_dtype(self, key: str) -> np.dtype[Any]:
        return table_column_dtype(
            self.h5,
            self.h5[self.cellAttrsKey],
            key,
            self._categoryGroups,
        )

    def _cell_column_block(
        self,
        key: str,
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return rows of a cluster column with non-finite values marked missing.

        The constructor validated each cluster column as one scalar per cell.
        """
        values, missing = self._read_column(self.cellAttrsKey, key, start, stop)
        if values.dtype.kind in "fc":
            missing = missing | ~np.isfinite(values)
        return values, missing

    def _cell_ids_block(self, start: int, stop: int) -> np.ndarray:
        """Return cell IDs ``[start, stop)`` as the text an import stores.

        An import checks every ID with :meth:`_check_ids_and_names` before
        it reads blocks.
        """
        if not self._check_exists(self.cellAttrsKey, self.cellIdsKey):
            return np.asarray([f"cell_{index}" for index in range(start, stop)])
        if self.duplicateCellIds == "make_unique":
            # Renamed IDs exist only once every ID was read.
            values = self._cell_identifiers()[1][start:stop]
        else:
            values, _ = self._read_column(
                self.cellAttrsKey, self.cellIdsKey, start, stop
            )
        return _stored_texts(values)

    def _obsm_array(self, key: str) -> h5py.Dataset:
        # The constructor validated each embedding key as a dense array.
        return self.h5[self.obsmAttrsKey][key]

    def _iter_obsm_blocks(
        self,
        key: str,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> Iterator[np.ndarray]:
        node = self._obsm_array(key)
        for start in range(0, self.nCells, block_rows):
            stop = min(start + block_rows, self.nCells)
            yield np.asarray(node[start:stop], dtype=dtype)

    def _cell_columns(
        self,
    ) -> Iterator[tuple[str, TableColumn, H5adCategories | None]]:
        return self._table_columns(
            self.cellAttrsKey, self._left_out_columns(self.cellAttrsKey)
        )

    def _feature_columns(
        self,
    ) -> Iterator[tuple[str, TableColumn, H5adCategories | None]]:
        return self._table_columns(
            self.featureAttrsKey, self._left_out_columns(self.featureAttrsKey)
        )

    def get_cell_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yield the ``obs`` metadata columns, excluding selected cluster keys.

        Text comes back as ``str`` in an object array, the text that an
        import stores: byte strings are decoded as UTF-8. Missing
        categorical and text values are ``None``; missing numeric values
        are NaN.

        Raises:
            ValueError: If a column holds bytes that are not UTF-8 text,
                naming the file, the table, the column, and the row, or the
                category of a categorical, that holds them, before any
                column is yielded.
        """
        self._check_table_text(self.cellAttrsKey)
        for name, column, _categories in self._cell_columns():
            yield name, _decoded_texts(present_column(*decoded_column(column)))

    def get_feat_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yield the feature metadata columns, as :meth:`get_cell_columns` yields those of cells.

        Raises:
            ValueError: If a column holds bytes that are not UTF-8 text, as
                :meth:`get_cell_columns` refuses them.
        """
        self._check_table_text(self.featureAttrsKey)
        for name, column, _categories in self._feature_columns():
            yield name, _decoded_texts(present_column(*decoded_column(column)))

    def assay_membership(self) -> dict[str, str]:
        """Return the membership column of each assay that the file declares.

        Raises:
            ValueError: If the declaration is not a mapping from assay name
                to one text value, a column name is bytes that are not UTF-8
                text, naming the file and where the bytes are, or it names
                a column other than ``<assay>_I``.
        """
        from ..metadata.membership import (
            MEMBERSHIP_UNS_KEY,
            CYTEARC_UNS_KEY,
            parse_membership_declaration,
        )

        location = f"uns/{CYTEARC_UNS_KEY}/{MEMBERSHIP_UNS_KEY}"
        node = self.h5.get(location)
        if node is None:
            return {}
        if not isinstance(node, h5py.Group):
            raise ValueError(
                f"{location} must be a mapping from assay name to membership column"
            )
        declared: dict[str, str] = {}
        for assay, entry in node.items():
            if not isinstance(entry, h5py.Dataset) or entry.shape != ():
                raise ValueError(
                    f"{location}/{assay} must hold one text value, the membership "
                    "column of the assay"
                )
            value = entry[()]
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(
                        f"{_MEMBERSHIP_COLUMNS} must be UTF-8 text; "
                        f"{location}/{assay} of {quoted_value(str(self.h5adFn))} "
                        f"holds {quoted_value(bytes(value))}"
                    ) from None
            declared[assay] = as_text(value)
        return parse_membership_declaration(declared)

    def feature_types(self, key: str) -> list[str]:
        """Return decoded feature types from a var column.

        Raises:
            KeyError: If the column is absent.
            ValueError: If it does not hold one value per feature, it is a
                categorical whose categories are not one text, number or
                boolean each, or a value is bytes that are not UTF-8 text,
                naming the file, the table, the column and the row.
        """
        if not self._check_exists(self.featureAttrsKey, key):
            raise KeyError(
                f"Feature type key `{key}` was not found in {self.featureAttrsKey}"
            )
        values = present_column(*self._read_column(self.featureAttrsKey, key))
        if values.ndim != 1 or len(values) != self.nFeatures:
            raise ValueError(
                f"Feature type key `{key}` has {len(values)} values; "
                f"expected {self.nFeatures}"
            )
        check_utf8(
            values,
            f"{_FEATURE_TYPES} must be UTF-8 text",
            self._column_place(self.featureAttrsKey, key),
        )
        return [as_text(value) for value in values]

    def assay_feature_slices(
        self,
        key: str,
        name_map: Mapping[str, str] | None = None,
    ) -> dict[str, _H5adAssayFeatures]:
        """Resolve feature ranges and metadata for each assay."""
        ranges = assay_feature_ranges(
            auto_name_feat_table(
                make_feat_table_from_types(self.feature_types(key)),
                name_map,
            )
        )
        feature_ids = self.feat_ids()
        feature_names = self.feat_names()
        assays: dict[str, _H5adAssayFeatures] = {}
        for assay_name, spans in ranges.items():
            indexes = np.concatenate(
                [np.arange(start, end, dtype=np.int64) for start, end in spans]
            )
            assays[assay_name] = _H5adAssayFeatures(
                featureIndexes=indexes,
                featureIds=feature_ids[indexes],
                featureNames=feature_names[indexes],
            )
        return assays

    # noinspection DuplicatedCode
    def consume_dataset(
        self,
        batch_size: int = 1000,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        for values in self.consume_dense_rows(batch_size, row_start, row_end):
            yield coo_matrix(values)

    def consume_dense_rows(
        self,
        batch_size: int,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Yield rows ``[row_start, row_end)`` of a dense matrix, ``batch_size`` at a time.

        Each batch is a NumPy array of the values in ``sourceMatrixDtype``.

        Raises:
            TypeError: If the matrix is sparse, whose rows ``consume`` yields.
            ValueError: If ``batch_size`` is not positive or the rows lie
                outside the matrix.
        """
        if self.matrixOrientation != "dense":
            raise TypeError(
                f"{self.matrixKey} holds a {self.matrixOrientation} matrix; "
                "consume yields its rows"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dset = self.h5[self.matrixKey]
        start = max(0, int(row_start))
        stop = int(dset.shape[0] if row_end is None else row_end)
        if stop < start or stop > int(dset.shape[0]):
            raise ValueError("consume row range is outside the matrix")
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            yield self._matrix_values(dset, slice(offset, end))

    def count_value_ranges(
        self,
        maxBytes: int,
        featureGroups: np.ndarray | None = None,
        *,
        processes: int = 1,
        destinationRows: np.ndarray | None = None,
    ) -> list[CountValueRange]:
        """Return the range of the canonical values of each group of features.

        Duplicate coordinates of a sparse matrix are summed, as the import
        stores them, so the ranges depend neither on the order of distinct
        coordinates, nor on orientation or encoding. Pass a range to
        :func:`~cytearc.storage.count_dtype.count_storage_dtype` to resolve the
        storage dtype of its group's counts. The scan of a dense matrix also
        counts the nonzero values of each row, which :meth:`max_batch_nnz`
        plans from, and the scan of a CSR matrix finds whether every row holds
        its indices sorted without repeats, so that an import plans no sort
        or sum of its batches.

        A matrix large enough that starting processes saves time is scanned
        in up to ``processes`` new processes at once, each over its own rows,
        or columns of a CSC matrix, with an equal share of ``maxBytes``: as
        many as that share holds a whole scan window and the longest row or
        column, which ``scan_processes`` of ``cytearc.utils.count_values``
        counts. The ranges and records are the same for any number of
        processes.

        With ``destinationRows``, the scan of a dense or CSR matrix also
        sends each row it reads to its destination row in a row bucket store
        on local disk, from which ``consume`` then yields rows in destination
        order, so an import that shuffles cells reads the file once. Each
        scan process writes its own part of the store, and ``maxBytes`` also
        holds the parts' pending rows and the store's row counts. A CSC
        matrix's rows take their destination in :meth:`materialize_csc`.

        Args:
            maxBytes: Memory available to the scan.
            featureGroups: Group, numbered from zero, of each feature, such as
                the assay that stores it. None puts every feature in one
                group.
            processes: Most processes that scan at once. 1 scans in this
                process.
            destinationRows: The destination row of each row of the matrix,
                a permutation, or None to keep the rows in place.

        Raises:
            MemoryError: If one compressed vector or dense row does not fit in
                ``maxBytes``, or with ``destinationRows`` its spill does not.
            OSError: If the temporary disk lacks room for the spill.
            ValueError: If the matrix holds NaN or an infinite value,
                ``processes`` is below 1, or ``destinationRows`` is not a
                permutation of the rows.
            TypeError: If ``processes`` is not an integer.
        """
        processes = integer_argument(processes, "processes", minimum=1)
        node = self.h5[self.matrixKey]
        dense = self.matrixOrientation == "dense"
        n_vectors = int(node.shape[0] if dense else int(node["indptr"].shape[0]) - 1)
        spill = None
        if destinationRows is not None and self.matrixOrientation != "csc":
            # Plans read the file's rows, never earlier shuffled ones.
            self._release_rows()
            spill = self._row_spill_plan(destinationRows, int(maxBytes))
        count = 1
        pointers: np.ndarray | None = None
        if processes > 1:
            if dense:
                values, longest = self.nCells * self.nFeatures, self.nFeatures
            else:
                pointers = (
                    self._csr_indptr()
                    if self.matrixOrientation == "csr"
                    else np.asarray(node["indptr"][:])
                )
                assert pointers is not None
                values = int(pointers[-1] - pointers[0])
                longest = int(np.diff(pointers).max(initial=0))
            count = scan_processes(
                values,
                longest,
                int(maxBytes) - (0 if spill is None else spill[0].ownerBytes),
                processes,
                extraBytes=0 if spill is None else spill[0].process_extra_bytes(),
            )
        windows = [(0, n_vectors)]
        if count > 1 and pointers is not None:
            windows = value_windows(pointers, count)
        elif count > 1:
            # Dense windows hold whole rows of HDF5 chunks, which no two
            # processes then decompress: the chunks' rows split as vectors
            # whose values are their rows.
            bounds = np.append(
                np.arange(0, n_vectors, self._dense_chunk_rows() or 1), n_vectors
            )
            windows = [
                (int(bounds[first]), int(bounds[last]))
                for first, last in value_windows(bounds, count)
            ]
        if spill is None:
            scan = self._scan_windows(windows, int(maxBytes), featureGroups)
        else:
            scan = self._scan_and_spill(windows, *spill, featureGroups)
        if dense:
            assert scan.rowNonzeros is not None
            self._cumulativeRowNnz = cumulative_nnz(scan.rowNonzeros)
        elif self.matrixOrientation == "csr":
            self._canonicalRows = scan.canonical
        return scan.valueRanges

    def _scan_windows(
        self,
        windows: list[tuple[int, int]],
        maxBytes: int,
        featureGroups: np.ndarray | None,
    ) -> CountScan:
        """Scan the windows, in processes of their own when there are several."""
        if len(windows) > 1:
            logger.debug(f"Scanning H5AD values in {len(windows)} processes")
            return merge_count_scans(
                self._scan_in_processes(
                    windows, maxBytes // len(windows), featureGroups
                )
            )
        return self._scan_values(*windows[0], maxBytes, featureGroups)

    def _row_spill_plan(
        self, destinationRows: np.ndarray, maxBytes: int
    ) -> tuple[RowSpillPlan, float | None, int]:
        """Plan the row bucket store of a shuffle and check that ``maxBytes`` holds it.

        Returns:
            The plan; the bytes per entry that the file's first rows take as
            the store encodes them, or None without rows; and the values that
            the rows hold beside zeros, counted for a sparse matrix and
            estimated from its first rows for a dense one, from which the
            free disk is checked before the spill.

        Raises:
            ValueError: If ``destinationRows`` is not a permutation of the rows.
            MemoryBudgetError: If the budget cannot hold the store's row
                counts and one window of the longest row.
        """
        destinations = checked_destinations(destinationRows, self.nCells)
        entry_bytes: float | None = None
        if self.matrixOrientation == "dense":
            entries = None
            longest = self.nFeatures
            # The first rows, of about a million values, estimate how many
            # values the rows hold beside zeros and what they take on disk.
            rows = dense_sample_rows(self.nCells, self.nFeatures)
            expected, entry_bytes = dense_spill_estimate(
                self._matrix_values(self.h5[self.matrixKey], slice(0, rows)),
                self.nCells,
            )
        else:
            pointers = self._csr_indptr()
            assert pointers is not None
            entries = int(pointers[-1] - pointers[0])
            expected = entries
            longest = int(np.diff(pointers).max(initial=0))
            # The first rows, of about a million entries, measure what the
            # spill takes on disk.
            rows = sample_rows(pointers)
            if rows:
                entry_bytes = encoded_entry_bytes(
                    next(self._file_csr_batches(pointers, rows, 0, rows))
                )
        plan = row_spill_plan(
            destinations,
            (self.nCells, self.nFeatures),
            self.sourceMatrixDtype,
            entries=entries,
            longest=longest,
            maxBytes=maxBytes,
        )
        return plan, entry_bytes, expected

    def _scan_and_spill(
        self,
        windows: list[tuple[int, int]],
        plan: RowSpillPlan,
        entry_bytes: float | None,
        entries: int,
        featureGroups: np.ndarray | None,
    ) -> CountScan:
        """Scan the windows and spill their rows into a new row bucket store.

        The store, finalized, then holds the rows that ``consume`` yields.
        """
        self._release_rows()
        self._indptrCache = None
        share = (plan.maxBytes - plan.ownerBytes) // len(windows)
        if len(windows) == 1:
            start, stop = windows[0]
            store, scan = spill_rows(
                plan,
                lambda consumer: self._scan_values(
                    start,
                    stop,
                    plan.scan_bytes(share, start, stop),
                    featureGroups,
                    consumer=consumer,
                ),
                operation=_SHUFFLE_OPERATION,
                entry_bytes=entry_bytes,
                entries=entries,
                buffer_bytes=plan.buffer_bytes(share, start, stop),
                dense=self.matrixOrientation == "dense",
                temp_dir=self._tempDir,
            )
        else:
            # The store holds its own empty part and one for each window.
            store = plan.new_store(
                operation=_SHUFFLE_OPERATION,
                entry_bytes=entry_bytes,
                entries=entries,
                temp_dir=self._tempDir,
                parts=1 + len(windows),
            )
            try:
                logger.debug(
                    f"Scanning H5AD values and spilling rows in {len(windows)} processes"
                )
                scan = merge_count_scans(
                    self._scan_in_processes(
                        windows,
                        0,
                        featureGroups,
                        spills=[
                            RowSpill(
                                str(store.directory),
                                part,
                                plan.buffer_bytes(share, start, stop),
                                plan.destinations[start:stop],
                            )
                            for part, (start, stop) in enumerate(windows, start=1)
                        ],
                        scanBytes=[
                            plan.scan_bytes(share, start, stop)
                            for start, stop in windows
                        ],
                    )
                )
                store.finalize()
            except BaseException:
                store.close()
                raise
        self._convertedCsr = store
        self._shuffledRows = True
        return scan

    def _scan_values(
        self,
        start: int,
        stop: int,
        maxBytes: int,
        featureGroups: np.ndarray | None,
        consumer: Any | None = None,
    ) -> CountScan:
        """Scan the values of compressed vectors ``[start, stop)`` in this process."""
        return scan_window(
            self.h5[self.matrixKey],
            self.matrixOrientation,
            self.sourceMatrixDtype,
            self.nFeatures if self.matrixOrientation == "csr" else self.nCells,
            start,
            stop,
            maxBytes,
            featureGroups,
            consumer,
        )

    def _scan_in_processes(
        self,
        windows: list[tuple[int, int]],
        maxBytes: int,
        featureGroups: np.ndarray | None,
        *,
        spills: list[RowSpill] | None = None,
        scanBytes: list[int] | None = None,
    ) -> list[CountScan]:
        """Scan each window in a new process of its own, with ``maxBytes`` each.

        With ``spills``, each window's process also sends the rows it reads
        to its part of a row bucket store, and scans within its entry of
        ``scanBytes``. A window's error, such as a NaN or a vector that does
        not fit, is raised here as the scan in this process raises it.
        """
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        # A forked child of a process that runs threads can deadlock, so scan
        # processes start as new interpreters, which import only the scan.
        with ProcessPoolExecutor(
            max_workers=len(windows), mp_context=get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    scan_file_window,
                    self.h5adFn,
                    self.matrixKey,
                    self.matrixOrientation,
                    self.sourceMatrixDtype,
                    self.nFeatures if self.matrixOrientation == "csr" else self.nCells,
                    start,
                    stop,
                    maxBytes if scanBytes is None else scanBytes[index],
                    featureGroups,
                    None if spills is None else spills[index],
                )
                for index, (start, stop) in enumerate(windows)
            ]
            return [future.result() for future in futures]

    @property
    def consumeDtype(self) -> np.dtype[Any]:
        """Return the dtype in which ``consume`` yields the matrix values.

        It is ``sourceMatrixDtype``, except that the converted rows of a CSC
        integer source yield their duplicate-summed values in int64, or uint64
        for unsigned and boolean sources.
        """
        source = (
            self.sourceMatrixDtype
            if self._convertedCsr is None
            else self._convertedCsr.dtype
        )
        dtype: np.dtype[Any] = np.dtype(source)
        return dtype

    def materialized_csr_bytes(self) -> int:
        """Return bytes retained by the materialized CSC-to-CSR conversion.

        They are its row pointers; reading its rows holds more, which
        :meth:`producer_batch_staging_bytes` counts.
        """
        if self._convertedCsr is None:
            return 0
        return int(self._convertedCsr.resident_bytes)

    def _csr_indptr(self) -> np.ndarray | None:
        if self._convertedCsr is not None:
            return np.asarray(self._convertedCsr.indptr)
        if self.matrixOrientation != "csr":
            return None
        if self._indptrCache is None:
            self._indptrCache = np.asarray(self.h5[self.matrixKey]["indptr"][:])
        return self._indptrCache

    def _dense_cumulative_nnz(self) -> np.ndarray:
        """Return the cumulative nonzero values of the dense matrix's rows.

        :meth:`count_value_ranges` records them as it scans the values;
        before it runs, the rows are read here to count them.
        """
        if self._cumulativeRowNnz is None:
            node = self.h5[self.matrixKey]
            self._cumulativeRowNnz = cumulative_nnz(
                dense_row_nonzeros(
                    lambda start, stop: self._matrix_values(node, slice(start, stop)),
                    self.nCells,
                    self.nFeatures,
                )
            )
        return self._cumulativeRowNnz

    def _dense_chunk_rows(self) -> int | None:
        """Return the rows of a dense matrix's HDF5 chunks, or None if unchunked."""
        chunks = self.h5[self.matrixKey].chunks
        return None if chunks is None else int(chunks[0])

    def _prepare_sparse_import(self) -> None:
        if self.matrixOrientation == "dense":
            self._dense_cumulative_nnz()
        else:
            self._csr_indptr()

    def _canonical_batches(self) -> bool:
        """Return whether every batch that ``consume`` yields is marked canonical.

        A canonical batch holds sorted coordinates without repeats, which an
        import neither sorts nor sums. The rows of a dense matrix and the
        converted rows of a CSC matrix always are; the rows of a CSR matrix
        are when :meth:`count_value_ranges` found every row's indices sorted
        without repeats, and before that scan they count as not canonical.
        """
        if self.matrixOrientation == "csr":
            return bool(self._canonicalRows)
        return True

    def _sparse_import_resident_bytes(self) -> int:
        """Return the bytes of the row counts that plan an import's batches.

        They are the cached CSR row pointers, or the cumulative nonzero values
        of a dense matrix's rows. Converted CSC rows count under
        :meth:`materialized_csr_bytes`.
        """
        if self._cumulativeRowNnz is not None:
            return int(self._cumulativeRowNnz.nbytes)
        if self._convertedCsr is None and self._indptrCache is not None:
            return int(self._indptrCache.nbytes)
        return 0

    def max_batch_nnz(self, batch_size: int) -> int:
        """Return the most nonzero values of any ``batch_size`` consecutive rows.

        CSR row pointers, converted CSC rows and shuffled rows give the count
        without reading values, the latter two in the order that ``consume``
        yields the rows. A dense matrix plans from the nonzero values of each
        row, which its value scan counts, or which are counted here before
        the scan. CSC columns not yet converted bound a window by its rows
        times the features.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.matrixOrientation == "dense" and self._convertedCsr is None:
            return max_window_nnz(self._dense_cumulative_nnz(), batch_size)
        batch_rows = min(batch_size, self.nCells)
        indptr = self._csr_indptr()
        if indptr is None:
            return int(batch_rows * self.nFeatures)
        return max_window_nnz(indptr, batch_size)

    def producer_batch_staging_bytes(self, batch_size: int) -> int:
        """Bound what a producer holds beside one batch.

        It holds the batch's sparse row pointers, and converted CSC rows and
        shuffled rows are read a bucket at a time with the next bucket
        loading ahead, which the producer holds as it reads.
        """
        rows = min(max(1, int(batch_size)), self.nCells)
        if rows == 0 or (
            self.matrixOrientation == "dense" and self._convertedCsr is None
        ):
            return 0
        reading = 0
        if self._convertedCsr is not None:
            itemsize = np.asarray(self._convertedCsr.indptr).dtype.itemsize
            reading = self._convertedCsr.read_bytes(read_ahead=True)
        else:
            itemsize = self.h5[self.matrixKey]["indptr"].dtype.itemsize
        normalized_itemsize = np.dtype(np.int32).itemsize
        return int((rows + 1) * (2 * itemsize + normalized_itemsize) + reading)

    def _csc_widest_row(self) -> int:
        """Return the most entries that one row of a CSC matrix holds.

        Repeated coordinates count each time, as the conversion appends them.
        The row indices are read a chunk at a time, without any value.

        Raises:
            ValueError: If a row index lies outside the matrix.
        """
        indices = self.h5[self.matrixKey]["indices"]
        counts = np.zeros(self.nCells, dtype=np.int64)
        for start in range(0, int(indices.shape[0]), _CSC_CHUNK_ENTRIES):
            rows = np.asarray(indices[start : start + _CSC_CHUNK_ENTRIES])
            if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.nCells):
                raise ValueError(
                    f"{self.matrixKey} holds a row index outside its {self.nCells} "
                    "rows; the file is corrupted"
                )
            counts += np.bincount(rows, minlength=self.nCells)
        return int(counts.max(initial=0))

    def materialize_csc(
        self,
        maxBytes: int = 64 * 1024 * 1024,
        *,
        destinationRows: np.ndarray | None = None,
    ) -> None:
        """Convert CSC into temporary row storage in bounded blocks.

        One pass over the columns sends each entry to the bucket of its row
        on local disk, and reads join each bucket's rows. The rows hold the
        source values with duplicate coordinates summed. Floating-point
        values keep ``sourceMatrixDtype``; integers are held in the 64-bit
        integer dtype in which their duplicates are summed, so a sum past the
        range of a narrow source dtype is kept, as a CSR import keeps it.

        With ``destinationRows``, the destination row of each row of the
        matrix, a permutation, each entry goes to its row's destination, so
        ``consume`` yields the rows of an import that shuffles cells in
        their shuffled order at no extra cost. The conversion then also keeps
        the file's first rows, which decide what the values are.

        Raises:
            MemoryBudgetError: If ``maxBytes`` cannot hold the conversion. A
                budget too small for the column pointers and one entry is
                refused before any entry is read, naming the budget that
                converts every row, as the row indices give it.
        """
        if self.matrixOrientation != "csc":
            return
        destinations = (
            None
            if destinationRows is None
            else checked_destinations(destinationRows, self.nCells)
        )
        if self._convertedCsr is not None:
            if destinations is None and not self._shuffledRows:
                return
            self._release_rows()
        group = self.h5[self.matrixKey]
        data_node = group["data"]
        source = np.dtype(self.sourceMatrixDtype)
        dtype = (
            np.dtype(np.uint64 if source.kind in "bu" else np.int64)
            if source.kind in "biu"
            else source
        )
        operation = _CSC_OPERATION if destinations is None else _SHUFFLE_OPERATION
        itemsize = max(source.itemsize, dtype.itemsize)
        pointers = (self.nFeatures + 1) * np.dtype(np.int64).itemsize
        least = pointers + SparseRowStore.conversion_bytes(
            self.nCells, entry_bytes=_CSC_ENTRY_BYTES, itemsize=itemsize
        )
        if maxBytes < least:
            # Converting also reads the row with the most entries whole, so
            # the refusal names the budget that holds it, which the row
            # indices give without converting anything.
            needed = pointers + SparseRowStore.conversion_bytes(
                self.nCells,
                entry_bytes=_CSC_ENTRY_BYTES,
                row_entries=max(1, self._csc_widest_row()),
                itemsize=itemsize,
            )
            if needed == least:
                raise MemoryBudgetError(
                    "CSC row conversion exceeds the memory limit. It needs at "
                    f"least {least} bytes.",
                    needed_bytes=least,
                    budget_bytes=int(maxBytes),
                    item="the pointers and one entry of the CSC row conversion",
                )
            # Worded and named as the conversion refuses its widest row.
            raise MemoryBudgetError(
                f"One row of the {operation} exceeds its memory limit. Beside the "
                f"{pointers} bytes of the matrix's column pointers, the CSC row "
                f"conversion needs at least {needed} bytes.",
                needed_bytes=needed,
                budget_bytes=int(maxBytes),
                item=f"one row of the {operation}",
            )
        indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        shape = (self.nCells, self.nFeatures)
        # The row store converts beside the source's pointers.
        store_bytes = int(maxBytes) - indptr.nbytes
        chunk_nnz = min(
            _CSC_CHUNK_ENTRIES,
            SparseRowStore.chunk_entries(store_bytes, self.nCells, _CSC_ENTRY_BYTES),
        )
        # The entries of the file's first rows, in the order they are read.
        first_rows = min(DETECTION_CELLS, self.nCells)
        kept: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def chunks() -> Iterator[coo_matrix]:
            for start in range(0, int(data_node.size), chunk_nnz):
                stop = min(int(data_node.size), start + chunk_nnz)
                columns = (
                    np.searchsorted(
                        indptr, np.arange(start, stop, dtype=np.int64), side="right"
                    )
                    - 1
                )
                rows = np.asarray(group["indices"][start:stop])
                values = self._matrix_values(data_node, slice(start, stop))
                if destinations is not None:
                    first = rows < first_rows
                    kept.append((rows[first], columns[first], values[first]))
                    rows = destinations[rows]
                yield coo_matrix((values, (rows, columns)), shape=shape)

        try:
            self._convertedCsr = SparseRowStore.from_entries(
                chunks(),
                shape,
                dtype,
                operation=operation,
                max_bytes=store_bytes,
                entry_bytes=_CSC_ENTRY_BYTES,
                entries=int(data_node.size),
                source_dtype=source,
                temp_dir=self._tempDir,
            )
        except MemoryBudgetError as error:
            needed = indptr.nbytes + error.needed_bytes
            raise MemoryBudgetError(
                f"{refusal_words(error)} Beside the {indptr.nbytes} bytes of the "
                "matrix's column pointers, the CSC row conversion needs at least "
                f"{needed} bytes.",
                needed_bytes=needed,
                budget_bytes=int(maxBytes),
                item=error.item,
            ) from error
        if destinations is not None:
            self._shuffledRows = True
            # Summed as the store sums each row's entries, in the order read.
            first = coo_matrix(
                (
                    np.concatenate([values for _rows, _columns, values in kept])
                    if kept
                    else np.zeros(0, dtype=source),
                    (
                        np.concatenate([rows for rows, _columns, _values in kept])
                        if kept
                        else np.zeros(0, dtype=np.int64),
                        np.concatenate([columns for _rows, columns, _values in kept])
                        if kept
                        else np.zeros(0, dtype=np.int64),
                    ),
                ),
                shape=(first_rows, self.nFeatures),
            )
            self._sourceFirstRows = csr_matrix(canonicalize_sparse(first, dtype))
        logger.debug("Prepared H5AD row storage for the CSC conversion")

    def consume_group(
        self,
        batch_size: int,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        start = max(0, int(row_start))
        stop = int(self.nCells if row_end is None else row_end)
        if stop < start or stop > self.nCells:
            raise ValueError("consume row range is outside the matrix")

        if self._convertedCsr is not None or self.matrixOrientation == "csc":
            yield from self._consume_converted_csr(batch_size, start, stop)
            return

        source_indptr = self._csr_indptr()
        assert source_indptr is not None
        for batch in self._file_csr_batches(source_indptr, batch_size, start, stop):
            yield csr_to_coo(batch)

    def _file_csr_batches(
        self,
        pointers: np.ndarray,
        batch_size: int,
        start: int,
        stop: int,
    ) -> Iterator[csr_matrix]:
        """Yield the file's CSR rows ``[start, stop)`` in batches of ``batch_size``.

        ``pointers`` are the file's row pointers from row 0 to at least row
        ``stop``.
        """
        grp = self.h5[self.matrixKey]
        # HDF5 caches decompressed chunks per open dataset. Reopening the
        # datasets for every batch would decompress a chunk again for each
        # batch that reads part of it.
        data = grp["data"]
        indices = grp["indices"]
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            indptr = pointers[offset : end + 1]
            data_start = int(indptr[0])
            data_end = int(indptr[-1])
            local_indptr = indptr - data_start
            n_rows = end - offset
            yield csr_matrix(
                (
                    self._matrix_values(data, slice(data_start, data_end)),
                    np.asarray(indices[data_start:data_end]),
                    local_indptr,
                ),
                shape=(n_rows, self.nFeatures),
            )

    def _consume_converted_csr(
        self,
        batch_size: int,
        start: int,
        stop: int,
    ) -> Generator[coo_matrix, None, None]:
        """Yield row batches from the temporary CSC conversion or shuffled rows.

        Shuffled rows are read once: each bucket that lies wholly inside the
        rows is deleted once its batches are taken.
        """
        if self._convertedCsr is None:
            self.materialize_csc()
        assert self._convertedCsr is not None
        # The next bucket of rows decodes while this one's batches are written.
        for batch in self._convertedCsr.iter_rows(
            start, stop, batch_size, read_ahead=True, delete=self._shuffledRows
        ):
            yield csr_to_coo(batch)

    def consume_row_range(
        self,
        batch_size: int,
        row_start: int,
        row_end: int,
    ) -> Generator[coo_matrix, None, None]:
        """Yield source batches covering ``[row_start, row_end)``.

        After an import's scan shuffled the rows, the batches hold its
        shuffled rows, as sparse rows also for a dense matrix.
        """
        # The constructor rejects a matrix slot that is neither a dataset nor
        # a group.
        if self.groupCodes[self.matrixKey] == 1 and self._convertedCsr is None:
            return self.consume_dataset(batch_size, row_start, row_end)
        return self.consume_group(batch_size, row_start, row_end)

    def _file_first_rows(self, count: int) -> coo_matrix:
        """Return the file's first ``count`` rows, which decide what the values are.

        They are the rows that ``consume`` yields first unless an import
        shuffled them: then they are read from the file, or for a CSC
        matrix kept by its shuffled conversion.

        Raises:
            ValueError: If ``count`` is negative or past the matrix.
        """
        if not 0 <= count <= self.nCells:
            raise ValueError("consume row range is outside the matrix")
        blocks: list[coo_matrix]
        if not self._shuffledRows or count == 0:
            blocks = list(self.consume_row_range(max(1, count), 0, count))
        elif self._sourceFirstRows is not None:
            blocks = [csr_to_coo(self._sourceFirstRows[:count])]
        elif self.matrixOrientation == "dense":
            blocks = list(self.consume_dataset(count, 0, count))
        else:
            pointers = np.asarray(self.h5[self.matrixKey]["indptr"][: count + 1])
            blocks = [
                csr_to_coo(batch)
                for batch in self._file_csr_batches(pointers, count, 0, count)
            ]
        return (vstack(blocks) if blocks else coo_matrix((0, 0))).tocoo()

    def consume(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        return self.consume_row_range(batch_size, 0, self.nCells)
