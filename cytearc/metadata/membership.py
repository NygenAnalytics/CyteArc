"""Cell columns that record which cells an assay measures.

A store whose assay measures only some of its cells marks the measured cells
in the boolean cell column ``<assay>_I`` (see
:func:`~cytearc.storage.metadata_keys.assay_membership_column`). The column has
no missing-value mask, and its attributes include those
:func:`membership_attributes` returns; other attributes are allowed. A store
without that column measures every cell with the assay, and a cell outside an
assay holds none of its counts.

Only imports, merges, and derived assays, which take the membership of the
assay that they derive from, write the column, directly in storage.
:class:`~cytearc.metadata.MetaData` refuses to write the membership column of
any assay of its store or workspace, whether or not it exists, and to drop one
that carries the membership role (see
:func:`~cytearc.storage.identity.protect_metadata_column`). A subset or a merge
that leaves an assay out leaves its membership column out too.

An AnnData that CyteArc exports holds the counts of one assay. It keeps that
assay's membership column as an ordinary ``obs`` column and declares it in
``uns["cytearc"]["assayMembership"]``, which maps the assay to its membership
column, so an import restores the column as the membership of that assay
instead of guessing it from the name. The membership columns of other assays
describe assays that the file does not hold, so the export leaves them out
(see :func:`exported_membership`).

An operation that reads an assay's values over a cell selection refuses cells
that the assay did not measure, whose zero counts are no measurement, with
:class:`UnmeasuredCellsError` (see :func:`require_measured_cells`), and
:func:`measured_selection` keeps the measured cells of a selection.
Display reads show those cells as missing (see :func:`measured_rows`).

Membership is read in bounded blocks. A selection of a few cells reads only
the chunks that hold them; a selection of at least one cell in
``_DENSE_SELECTION`` streams the column once.

A merge whose sources measured different features records, the same way,
which features each of them measured: one Boolean feature column for each
source that holds the assay, named by
:func:`~cytearc.storage.metadata_keys.measured_features_column`, whose
``role``, ``source`` and ``sourceColumn`` attributes are those that
:func:`~cytearc.storage.metadata_keys.measured_features_attributes` returns
(see :func:`measured_feature_columns`). ``sourceColumn`` names the cell
column of each cell's source, a categorical column of the source names that
the merge writes beside the record, with the ``role`` that
:func:`~cytearc.storage.metadata_keys.cell_sources_attributes` returns;
readers pair the record with each cell through it (see :func:`cell_sources`).
A feature table without such columns, as an import or an earlier release
writes it, counts every feature as measured by every source (see
:func:`features_measured_by_every_source`). An export declares the record of
the exported assay in ``uns["cytearc"]["measuredFeatures"]``, which maps the
assay to each column and its source, and the cell column of sources in
``uns["cytearc"]["cellSources"]``, so an import restores both.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
import zarr

from ..storage.artifacts import ValueFingerprintBuilder
from ..storage.types import member_names
from ..storage.stores import metadata_workers, run_concurrently
from ..storage.categorical import CategoricalColumn, category_codes_dtype
from ..storage.metadata_keys import (
    ASSAY_MEMBERSHIP_ROLE,
    CELL_SOURCES_ROLE,
    MEASURED_FEATURES_PREFIX,
    MEASURED_FEATURES_ROLE,
    SOURCE_COLUMN_ATTRIBUTE,
    assay_membership_attributes,
    assay_membership_column,
    measured_features_column,
)
from .rows import (
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_array_rows_chunkwise,
    read_metadata_rows_chunkwise,
)

# A selection of at least one cell in this many streams the membership column
# once, since it touches nearly every chunk; a smaller one reads only the
# chunks that hold its rows.
_DENSE_SELECTION = 16
# The cells whose sources a read of the cell column of sources holds at once.
_SOURCE_ROWS = 1 << 16

type MeasuredCellsRemedy = Literal[
    "labels", "graph", "cell_key", "cells", "cell_selection", "export"
]
"""How a refused caller selects measured cells, by the input it takes.

``"labels"``: the cells come from a label artifact, which
``snapshot_cluster_labels`` narrows. ``"graph"``: the cells come from the
lineage of a graph, which must be built again over measured cells.
``"cell_key"``: the cells are those of a boolean cell column, so the caller
takes another column, True only for the cells of that column that the assay
measured. ``"cells"``: a pipeline recipe's ``cells`` keyword names the cells,
as a boolean column or a selection, so the caller passes measured cells
there. ``"cell_selection"``: the caller's ``cell_selection`` keyword
narrows the cells that it reads, such as those of its labels or of the live
``I`` column. ``"export"``: a raw export that cannot declare which cells the
assay measured exports only measured cells, or an export that declares them
is used. Without one, the caller takes a cell selection, which
``select_measured_cells`` narrows.
"""

CYTEARC_UNS_KEY = "cytearc"
"""Key of CyteArc's namespace in the ``uns`` mapping of an exported AnnData."""

MEMBERSHIP_UNS_KEY = "assayMembership"
"""Key, in CyteArc's ``uns`` namespace, of each assay's membership column."""

MEASURED_FEATURES_UNS_KEY = "measuredFeatures"
"""Key, in CyteArc's ``uns`` namespace, of each assay's record of which features each source measured."""

CELL_SOURCES_UNS_KEY = "cellSources"
"""Key of ``uns["cytearc"]`` that names the ``obs`` column of each cell's source, which a record of measured features names."""


def membership_attributes(assay: str) -> dict[str, str]:
    """Return the attributes that mark a cell column as ``assay``'s membership."""
    return assay_membership_attributes(assay)


def _has_column(table: Any, name: str) -> bool:
    """Return whether ``table`` has a column ``name``.

    A ``MetaData`` table looks the one column up, so an object store does not
    list every column; another table answers from its column list.
    """
    has_column = getattr(table, "_has_column", None)
    if callable(has_column):
        return bool(has_column(name))
    return name in table.columns


def resolve_assay_membership(table: Any, assay: str) -> str | None:
    """Return ``assay``'s membership column, or None if the cell table has none."""
    name = assay_membership_column(assay)
    if not _has_column(table, name):
        return None
    dtype = np.dtype(table.get_dtype(name))
    expected = membership_attributes(assay)
    attributes = getattr(table._get_array(name), "attrs", {})
    problem: str | None = None
    if dtype.kind != "b":
        problem = f"has dtype {dtype}, not bool"
    elif metadata_missing_mask(table, name) is not None:
        problem = "has a missing-value mask"
    else:
        found = {key: attributes.get(key) for key in expected}
        if found != expected:
            problem = f"has attributes {found}, not {expected}"
    if problem is None:
        return name
    if attributes.get("role") == ASSAY_MEMBERSHIP_ROLE:
        remedy = "Import the data again to write a valid column."
    else:
        remedy = (
            "Without the membership role it is a plain column: drop it from the "
            f"store's cells with cells.drop({name!r}), after which the assay "
            "counts every cell as measured, or import the data again to "
            "restore the membership."
        )
    raise ValueError(
        f"Cell column {name!r} is reserved for the membership of assay "
        f"{assay!r}, but it {problem}. A membership column is a boolean "
        f"column without missing values whose role and assay attributes are "
        f"these. {remedy}"
    )


def _column_arrays(table: Any) -> tuple[list[str], Callable[[str], Any]]:
    """Return a table's column names and a way to open each column's array.

    A ``MetaData`` table lists its columns once and reuses the arrays that the
    listing opened; another table lists its columns and opens each one.
    """
    list_members = getattr(table, "_members", None)
    if not callable(list_members):
        return list(table.columns), table._get_array
    members = list_members()
    opened = dict(members)
    return table._public_column_names(members), partial(
        table._opened_array, opened=opened
    )


def membership_columns(table: Any) -> dict[str, str]:
    """Return the membership column of each assay that a cell table records."""
    found: dict[str, str] = {}
    names, column_array = _column_arrays(table)
    for name in names:
        attributes = getattr(column_array(name), "attrs", {})
        if attributes.get("role") != ASSAY_MEMBERSHIP_ROLE:
            continue
        assay = attributes.get("assay")
        if isinstance(assay, str) and assay_membership_column(assay) == name:
            column = resolve_assay_membership(table, assay)
            assert column is not None
            found[assay] = column
    return found


def _record_candidates(table: Any) -> list[str]:
    """Return the columns of a table whose names a record of measured features takes, from a listing of its keys.

    A ``MetaData`` table's keys are listed from its store, which opens no
    column, so a table without such names costs one listing.
    """
    group = getattr(table, "_group", None)
    names = member_names(group) if isinstance(group, zarr.Group) else table.columns
    return sorted(name for name in names if name.startswith(MEASURED_FEATURES_PREFIX))


def _table_workers(table: Any) -> int:
    """Return how many small reads to overlap on a table's store, one for a local store."""
    group = getattr(table, "_group", None)
    return metadata_workers(group) if isinstance(group, zarr.Group) else 1


def _record_arrays(table: Any) -> list[tuple[str, str, Any]]:
    """Return the name, source and opened array of each record column of a feature table, opened concurrently.

    Raises:
        ValueError: As :func:`measured_feature_columns` raises.
    """
    names = _record_candidates(table)
    arrays = run_concurrently(
        [partial(table._get_array, name) for name in names],
        workers=_table_workers(table),
    )
    found: list[tuple[str, str, Any]] = []
    for name, array in zip(names, arrays, strict=True):
        attributes = getattr(array, "attrs", {})
        if attributes.get("role") != MEASURED_FEATURES_ROLE:
            continue
        source = attributes.get("source")
        problem: str | None = None
        if not isinstance(source, str) or measured_features_column(source) != name:
            problem = f"its source attribute, {source!r}, does not name it"
        elif np.dtype(array.dtype).kind != "b":
            problem = f"it has dtype {np.dtype(array.dtype)}, not bool"
        elif "missing_mask" in attributes:
            problem = "it has missing values"
        if problem is not None:
            store = _record_store(table)
            raise ValueError(
                f"Feature column {name!r} has the role of a record of which "
                f"features one source of a merge measured, but {problem}. Earlier "
                "releases of CyteArc copy such a column without its source and "
                "with missing values when they merge stores. Drop it with "
                f"{store}.feats.drop({name!r}), or merge the sources again to "
                "record what each one measured."
            )
        found.append((name, str(source), array))
    return found


def measured_feature_columns(table: Any) -> dict[str, str]:
    """Return the feature columns that record which features each source of a merge measured, mapped to their sources.

    A merge or a composite whose sources measured different features writes
    one Boolean column for each source that holds the assay, named by
    :func:`~cytearc.storage.metadata_keys.measured_features_column` of the
    source that its ``source`` attribute keeps, with the ``role`` attribute
    :data:`~cytearc.storage.metadata_keys.MEASURED_FEATURES_ROLE`; other
    feature tables hold none. Only columns named ``measured_in_<source>``
    are opened.

    Raises:
        ValueError: If a column with that role is not such a column, as an
            earlier release writes one when it merges a store that holds the
            record, naming the call that drops it.
    """
    return {name: source for name, source, _array in _record_arrays(table)}


def _linked_column(arrays: Iterable[Any]) -> str | None:
    """Return the cell column that every record column names, or None when they name none or different ones."""
    linked = {array.attrs.get(SOURCE_COLUMN_ATTRIBUTE) for array in arrays}
    column = next(iter(linked), None)
    return column if len(linked) == 1 and isinstance(column, str) else None


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """Which features each source of a merge measured, as a feature table records it.

    Attributes:
        sources: Each source, in the order of the record's columns.
        columns: The feature column that records each source's features.
        measured: Whether each source measured each feature of ``features``,
            sources by features.
        features: The features, in increasing order, that ``measured``
            covers.
        sourceColumn: The cell column that names each cell's source, or None
            when the record names none.
    """

    sources: list[str]
    columns: list[str]
    measured: np.ndarray
    features: np.ndarray
    sourceColumn: str | None

    def measured_of(self, features: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return whether each source measured each of ``features``, which ``self.features`` holds, sources by features."""
        indices = np.asarray(features, dtype=np.int64)
        return self.measured[:, np.searchsorted(self.features, indices)]


def measured_features_by_source(
    table: Any, features: Sequence[int] | np.ndarray | None = None
) -> FeatureRecord | None:
    """Return the record of which features each source measured that a feature table holds, or None.

    The record's columns are those of :func:`measured_feature_columns`; a
    table without them holds no record. With ``features``, only their rows
    of each column are read.
    """
    record = _record_arrays(table)
    if not record:
        return None
    rows = (
        np.arange(int(table.N))
        if features is None
        else np.unique(np.asarray(features, dtype=np.int64))
    )
    arrays = [array for _name, _source, array in record]
    read = (
        [partial(np.asarray, array) for array in arrays]
        if features is None
        else [partial(read_array_rows_chunkwise, array, rows) for array in arrays]
    )
    return FeatureRecord(
        sources=[source for _name, source, _array in record],
        columns=[name for name, _source, _array in record],
        measured=np.stack(
            [
                np.asarray(values, dtype=bool)
                for values in run_concurrently(read, workers=_table_workers(table))
            ]
        ),
        features=rows,
        sourceColumn=_linked_column(arrays),
    )


def features_measured_by_every_source(table: Any) -> np.ndarray:
    """Return whether every source of a merge measured each feature of ``table``.

    Every feature of a table without a record
    (:func:`measured_features_by_source`) counts as measured by every source.
    """
    record = measured_features_by_source(table)
    if record is None:
        return np.ones(int(table.N), dtype=bool)
    return np.asarray(record.measured.all(axis=0))


def present_sources(codes: np.ndarray, n_sources: int) -> np.ndarray:
    """Return the positions of the sources that ``codes``, the sources of some cells, hold."""
    return np.flatnonzero(np.bincount(codes, minlength=n_sources))


def _record_store(table: Any) -> str:
    """Return how a message names the assay of a feature table, as ``ds.ADT`` or ``ds.get_assay('...')``."""
    # A feature table is the featureData group of its assay.
    assay = str(table._group.path).strip("/").split("/")[-2]
    return f"ds.{assay}" if assay.isidentifier() else f"ds.get_assay({assay!r})"


def _dropped_record(features: Any, record: FeatureRecord) -> str:
    """Return the remedy that drops a record, and what dropping it does."""
    store = _record_store(features)
    calls = ", ".join(f"{store}.feats.drop({column!r})" for column in record.columns)
    return (
        f"Merge the sources again, or drop the record with {calls}, after which "
        "analyses read the zero counts of a feature in the cells of a source "
        "that did not measure it as counts again."
    )


def _source_blocks(
    cells: Any, features: Any, rows: np.ndarray, record: FeatureRecord, *, assay: str
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield the offset of each block of ``rows`` and the position in ``record.sources`` of each of its cells' sources.

    The codes take the smallest integer type of the sources' positions.

    Raises:
        ValueError: As :func:`cell_sources` raises.
    """
    column = record.sourceColumn
    stored = (
        cells._get_array(column)
        if column is not None and _has_column(cells, column)
        else None
    )
    if stored is None or stored.attrs.get("role") != CELL_SOURCES_ROLE:
        named = "" if column is None else f" {column!r}"
        problem = (
            "is missing"
            if stored is None
            else f"has no role {CELL_SOURCES_ROLE!r}, which marks that column"
        )
        raise ValueError(
            f"Assay {assay!r} records which features each source of a merge "
            f"measured, in its feature columns {', '.join(map(repr, record.columns))}, "
            f"but the cell column{named} that names each cell's source, which a "
            f"merge or a composite writes beside the record, {problem}. "
            + _dropped_record(features, record)
        )
    assert column is not None
    dtype = category_codes_dtype(len(record.sources))
    lookup: np.ndarray | None = None
    if isinstance(stored, CategoricalColumn):
        # Each category's position among the sources, and -1 after the last
        # category, which a missing row's code -1 takes.
        lookup = np.full(stored.n_categories + 1, -1, dtype=dtype)
        for position, source in enumerate(record.sources):
            lookup[np.flatnonzero(stored.categories == source)] = position
    indices = np.asarray(rows, dtype=np.int64)
    for start in range(0, indices.size, _SOURCE_ROWS):
        block = indices[start : start + _SOURCE_ROWS]
        if lookup is not None:
            codes = lookup[read_array_rows_chunkwise(stored.codes, block)]
        else:
            labels = read_metadata_rows_chunkwise(cells, column, block)
            codes = np.full(block.size, -1, dtype=dtype)
            for position, source in enumerate(record.sources):
                codes[labels == source] = position
        unknown = np.flatnonzero(codes < 0)
        if unknown.size:
            row = block[unknown[:1]]
            cell = read_metadata_rows_chunkwise(cells, "ids", row)[0]
            label = read_metadata_rows_chunkwise(cells, column, row)[0]
            raise ValueError(
                f"Cell {str(cell)!r} has the source {str(label)!r} in cell column "
                f"{column!r}, which is not one of the sources whose measured "
                f"features assay {assay!r} records "
                f"({', '.join(map(repr, record.sources))}). "
                + _dropped_record(features, record)
            )
        yield start, codes


def cell_sources(
    cells: Any, features: Any, rows: np.ndarray, record: FeatureRecord, *, assay: str
) -> np.ndarray:
    """Return the position in ``record.sources`` of the source of each cell at ``rows``.

    The cells must be cells that ``assay`` measured. Their sources are the
    labels of the cell column that the record names, which holds the role
    :data:`~cytearc.storage.metadata_keys.CELL_SOURCES_ROLE`, read a block
    of rows at a time: the codes of the categorical column that a merge or a
    composite writes, or the labels of a column of text. The positions take
    the smallest integer type that holds them.

    Raises:
        ValueError: If the record names no cell column, the cells lack it,
            it lacks the role, or a cell's source is not a source of the
            record, giving the calls that drop the record.
    """
    positions = np.empty(
        np.asarray(rows).size, dtype=category_codes_dtype(len(record.sources))
    )
    for start, codes in _source_blocks(cells, features, rows, record, assay=assay):
        positions[start : start + codes.size] = codes
    return positions


def sources_of_cells(
    cells: Any, features: Any, rows: np.ndarray, record: FeatureRecord, *, assay: str
) -> np.ndarray:
    """Return the positions in ``record.sources`` of the sources that some cell at ``rows`` belongs to.

    Each block's sources are counted as it is read, so no code is held for
    every cell.

    Raises:
        ValueError: As :func:`cell_sources` raises.
    """
    present = np.zeros(len(record.sources), dtype=bool)
    for _start, codes in _source_blocks(cells, features, rows, record, assay=assay):
        present[present_sources(codes, len(record.sources))] = True
    return np.flatnonzero(present)


def features_measured_by_cells(
    cells: Any,
    features: Any,
    rows: np.ndarray,
    *,
    assay: str,
    selected: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return whether the source of every cell at ``rows`` measured each feature of ``features``.

    The cells must be cells that ``assay`` measured. None when the feature
    table holds no record, or when every source of the record measured every
    feature of the Boolean mask ``selected``, or every feature without one;
    the cells' sources are read only otherwise.

    Raises:
        ValueError: As :func:`cell_sources` raises.
    """
    record = measured_features_by_source(features)
    if record is None:
        return None
    if (record.measured if selected is None else record.measured[:, selected]).all():
        return None
    present = sources_of_cells(cells, features, rows, record, assay=assay)
    return np.asarray(np.logical_and.reduce(record.measured[present], axis=0))


def require_measured_features(
    cells: Any,
    features: Any,
    rows: np.ndarray,
    requested: Mapping[str, Sequence[int]],
    *,
    assay: str,
    remedy: str,
) -> None:
    """Refuse a requested feature that the source of a cell at ``rows`` did not measure.

    ``requested`` maps each requested label to the features of ``assay``
    that it reads. An analysis that reads values over cells refuses zero
    counts that are not measurements, as it refuses cells that the assay did
    not measure (:func:`require_measured_cells`). The cells' sources are read
    only when some source did not measure a requested feature.

    Raises:
        ValueError: Naming the first label that reads a feature that the
            source of some of the cells did not measure, those sources, and
            ``remedy``; or as :func:`cell_sources` raises.
    """
    record = measured_features_by_source(
        features,
        np.concatenate(
            [np.asarray(indices, dtype=np.int64) for indices in requested.values()]
            or [np.empty(0, dtype=np.int64)]
        ),
    )
    if record is None:
        return
    lacked = {
        label: measured
        for label, measured in (
            (label, record.measured_of(indices)) for label, indices in requested.items()
        )
        if not measured.all()
    }
    if not lacked:
        return
    present = sources_of_cells(cells, features, rows, record, assay=assay)
    for label, measured in lacked.items():
        lacking = [record.sources[code] for code in present if not measured[code].all()]
        if lacking:
            noun, pronoun = (
                ("source", "its") if len(lacking) == 1 else ("sources", "their")
            )
            raise ValueError(
                f"Assay {assay!r} records that {noun} "
                f"{', '.join(map(repr, lacking))} did not measure {label!r}, so "
                f"the zero counts of {pronoun} cells are not measurements; {remedy}"
            )


def exported_measured_features(
    table: Any, assay: str
) -> tuple[dict[str, dict[str, str]], str | None]:
    """Return the record of which features each source measured that an export of ``assay`` declares, and the cell column of sources that it names."""
    record = _record_arrays(table)
    if not record:
        return {}, None
    return (
        {assay: {name: source for name, source, _array in record}},
        _linked_column(array for _name, _source, array in record),
    )


def cell_sources_declaration(column: str | None) -> dict[str, dict[str, str]]:
    """Return the AnnData ``uns`` entries that declare the ``obs`` column of each cell's source."""
    if column is None:
        return {}
    return {CYTEARC_UNS_KEY: {CELL_SOURCES_UNS_KEY: column}}


def measured_features_declaration(
    declared: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Return the AnnData ``uns`` entries that declare each assay's record of what its sources measured."""
    if not declared:
        return {}
    return {
        CYTEARC_UNS_KEY: {
            MEASURED_FEATURES_UNS_KEY: {
                assay: dict(columns) for assay, columns in declared.items()
            }
        }
    }


def parse_measured_features_declaration(
    declared: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Check and return each assay's record of what its sources measured that an AnnData ``uns`` declares."""
    for assay, columns in declared.items():
        for column, source in columns.items():
            if measured_features_column(source) != column:
                raise ValueError(
                    f"uns['{CYTEARC_UNS_KEY}']['{MEASURED_FEATURES_UNS_KEY}'] maps "
                    f"column {column!r} of assay {assay!r} to the source {source!r}; "
                    "CyteArc declares the column of each source as "
                    "'measured_in_<source>'. Remove the entry from the file or "
                    "correct it."
                )
    return {assay: dict(columns) for assay, columns in declared.items()}


def checked_measured_features_values(
    values: np.ndarray, missing: np.ndarray, *, column: str, assay: str
) -> None:
    """Check the values of a column that a file declares as a record of what a source of ``assay`` measured."""
    declared = (
        f"Column {column!r} is declared as the record of which features a source "
        f"of assay {assay!r} measured, but it"
    )
    array = np.asarray(values)
    if array.dtype.kind != "b":
        raise ValueError(f"{declared} has dtype {array.dtype}, not bool")
    if np.asarray(missing, dtype=bool).any():
        raise ValueError(f"{declared} has missing values")


def exported_membership(
    table: Any, assay: str
) -> tuple[dict[str, str], frozenset[str]]:
    """Return the membership that an export of ``assay`` writes and leaves out."""
    columns = membership_columns(table)
    declared = {assay: columns[assay]} if assay in columns else {}
    omitted = frozenset(column for name, column in columns.items() if name != assay)
    return declared, omitted


def membership_declaration(
    declared: Mapping[str, str],
) -> dict[str, dict[str, dict[str, str]]]:
    """Return the AnnData ``uns`` entries that declare assay membership."""
    if not declared:
        return {}
    return {CYTEARC_UNS_KEY: {MEMBERSHIP_UNS_KEY: dict(declared)}}


def parse_membership_declaration(declared: Mapping[Any, Any]) -> dict[str, str]:
    """Check and return the assay membership that an AnnData ``uns`` declares."""
    parsed: dict[str, str] = {}
    for assay, column in declared.items():
        if not isinstance(assay, str) or column != assay_membership_column(assay):
            raise ValueError(
                f"uns['{CYTEARC_UNS_KEY}']['{MEMBERSHIP_UNS_KEY}'] maps {assay!r} to "
                f"{column!r}; CyteArc declares each assay's membership column as "
                "'<assay>_I'. Remove the entry from the file or correct it."
            )
        parsed[assay] = column
    return parsed


def checked_membership_values(
    values: np.ndarray,
    missing: np.ndarray,
    *,
    assay: str,
) -> np.ndarray:
    """Return declared membership values as booleans after checking them."""
    column = assay_membership_column(assay)
    array = np.asarray(values)
    if array.dtype.kind != "b":
        raise ValueError(
            f"Column {column!r} is declared as the membership of assay {assay!r}, "
            f"but it has dtype {array.dtype}, not bool"
        )
    if np.asarray(missing, dtype=bool).any():
        raise ValueError(
            f"Column {column!r} is declared as the membership of assay {assay!r}, "
            "but it has missing values"
        )
    return array.astype(bool, copy=False)


def reserved_membership_columns(assays: Iterable[str]) -> dict[str, str]:
    """Return the membership column name of each assay, mapped to its assay."""
    return {assay_membership_column(assay): assay for assay in assays}


def count_unmeasured_cells_with_counts(table: Any, assay: str, column: str) -> int:
    """Return how many cells outside ``assay``'s membership have its counts."""
    totals_name, positives_name = f"{assay}_nCounts", f"{assay}_nFeatures"
    for name in (totals_name, positives_name):
        if name not in table.columns:
            raise ValueError(
                f"The cell table has no {name!r} column to check membership "
                f"column {column!r} against. Open the store as a DataStore with "
                "zarr_mode='r+' once so that its assays are prepared."
            )
    totals = table._get_array(totals_name)
    # A row whose counts cancel to a zero total still has a positive entry.
    positives = table._get_array(positives_name)
    count = 0
    start = 0
    for members in iter_metadata_column_blocks(table, column):
        stop = start + len(members)
        outside = ~np.asarray(members, dtype=bool)
        if outside.any():
            counted = (np.asarray(totals[start:stop]) != 0) | (
                np.asarray(positives[start:stop]) != 0
            )
            count += int(np.count_nonzero(counted[outside]))
        start = stop
    return count


def _measured_insert(name: str, key: str, column: str) -> str:
    """Return the insert of a column of the cells of ``key`` that ``column`` measured."""
    return (
        f"`ds.cells.insert({name!r}, pandas.Series(ds.cells.fetch_all({key!r}) & "
        f"ds.cells.fetch_all({column!r}), index=ds.cells.fetch_all('ids')))`"
    )


def _measured_cells_remedy(
    remedy: MeasuredCellsRemedy | None,
    assay: str,
    column: str,
    cell_key: str | None,
) -> str:
    """Return the sentence that tells a refused caller how to select cells."""
    measured = f"select_measured_cells({assay!r}, cell_selection=...)"
    if remedy is None:
        return f"Pass a selection of measured cells from `{measured}`."
    if remedy == "labels":
        return (
            "Narrow the labels to measured cells with "
            f"`snapshot_cluster_labels(labels, cell_selection={measured})`."
        )
    if remedy == "graph":
        return f"Build the graph over a selection of measured cells from `{measured}`."
    if remedy == "cell_selection":
        return (
            f"Pass `cell_selection={measured}` to read only the cells that it measured."
        )
    if remedy == "cell_key":
        # The membership column alone would also select cells outside the
        # caller's column, such as cells that a filter removed from I.
        key = "I" if cell_key is None else cell_key
        name = f"{assay}_measured"
        return (
            "Pass as cell_key a boolean cell column that is True only for the "
            f"cells of {key!r} that it measured, such as {name!r} after "
            f"{_measured_insert(name, key, column)}."
        )
    if remedy == "cells":
        if cell_key is None:
            # A selection or an earlier run gave the cells, and a modality run
            # that takes a run's cells must keep them to be combined with it.
            return (
                f"Pass as cells the selection from `{measured}`. When the cells "
                "are an earlier run's, run that earlier run on cells that this "
                "assay measured too, so that the runs analyze the same cells."
            )
        return (
            f"Pass as cells the selection from `{measured}`, or a boolean cell "
            f"column that is True only for the cells of {cell_key!r} that it "
            f"measured, such as {f'{assay}_measured'!r} after "
            f"{_measured_insert(f'{assay}_measured', cell_key, column)}."
        )
    if remedy == "export":
        return (
            "Export only the cells that it measured, such as those of a subset "
            f"that `SubsetZarr(..., cell_key={column!r})` writes, or export "
            f"assay {assay!r} with `to_h5ad` or with `to_anndata(from_assay="
            f"{assay!r})` without layers, which declare which cells it measured."
        )
    raise ValueError(f"Unknown measured-cells remedy {remedy!r}")


class UnmeasuredCellsError(ValueError):
    """An operation would read an assay over cells that the assay did not measure.

    Attributes:
        operation: The refusing operation.
        assay: The assay that the operation reads.
        column: The assay's membership column, ``<assay>_I``.
        unmeasured: How many of the selected cells the assay did not measure.
        selected: How many cells the operation would read.
        remedy: The input that the message tells the caller to narrow.
        cell_key: The boolean cell column whose cells the operation reads, or None.
    """

    def __init__(
        self,
        operation: str,
        assay: str,
        column: str,
        unmeasured: int,
        selected: int,
        remedy: MeasuredCellsRemedy | None = None,
        cell_key: str | None = None,
    ) -> None:
        self.operation = operation
        self.assay = assay
        self.column = column
        self.unmeasured = unmeasured
        self.selected = selected
        self.remedy = remedy
        self.cell_key = cell_key
        super().__init__(
            f"`{operation}` reads assay {assay!r} over {unmeasured} of {selected} "
            f"selected cells that it did not measure: cell column {column!r} is "
            f"False for them. "
            f"{_measured_cells_remedy(remedy, assay, column, cell_key)}"
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            type(self),
            (
                self.operation,
                self.assay,
                self.column,
                self.unmeasured,
                self.selected,
                self.remedy,
                self.cell_key,
            ),
            self.__dict__,
        )


def _strictly_increasing(rows: np.ndarray) -> bool:
    return rows.size < 2 or bool(np.all(rows[1:] > rows[:-1]))


def _cell_rows(rows: Any, n_cells: int) -> np.ndarray:
    """Return integer cell rows as int64 after checking them.

    Raises:
        ValueError: If ``rows`` is not one-dimensional.
        TypeError: If ``rows`` does not hold integers.
        IndexError: If a row is negative or not below ``n_cells``.
    """
    indices = np.asarray(rows)
    if indices.ndim != 1:
        raise ValueError("Cell rows must be one-dimensional")
    if indices.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Cell rows must be integers or a boolean mask")
    indices = indices.astype(np.int64, copy=False)
    if int(indices.min()) < 0 or int(indices.max()) >= n_cells:
        raise IndexError(f"Cell rows are out of bounds for a table of {n_cells} cells")
    return indices


def _checked_cells(table: Any, cells: Any) -> Any:
    """Return ``cells`` as a boolean mask over every cell, or as checked rows.

    The name of a boolean cell column gives the column's array. A boolean
    mask, such as the stored values of a cell-selection artifact, must have
    one entry per cell; it is read later in bounded blocks.
    """
    if isinstance(cells, str):
        return table._bool_array(cells)
    if not hasattr(cells, "dtype"):
        cells = np.asarray(cells)
    if np.dtype(cells.dtype).kind != "b":
        return _cell_rows(cells, table.N)
    if tuple(cells.shape) != (table.N,):
        raise ValueError(
            f"A cell mask needs one entry per cell ({table.N}); got shape "
            f"{tuple(cells.shape)}"
        )
    return cells


def _dense(rows: np.ndarray, n_cells: int) -> bool:
    return rows.size * _DENSE_SELECTION >= n_cells


def _membership_at(table: Any, column: str, rows: np.ndarray) -> np.ndarray:
    """Read the membership of each of the checked ``rows``, in their order.

    A dense selection streams the column once into a copy of one byte per
    cell. Otherwise only the chunks that hold the rows are read, and rows
    that are not strictly increasing are read once each.
    """
    if rows.size == 0:
        return np.zeros(0, dtype=bool)
    if _dense(rows, table.N):
        members = np.empty(table.N, dtype=bool)
        start = 0
        for block in iter_metadata_column_blocks(table, column):
            members[start : start + len(block)] = block
            start += len(block)
        gathered: np.ndarray = members[rows]
        return gathered
    if _strictly_increasing(rows):
        return np.asarray(read_metadata_rows_chunkwise(table, column, rows), dtype=bool)
    distinct, positions = np.unique(rows, return_inverse=True)
    values = np.asarray(
        read_metadata_rows_chunkwise(table, column, distinct), dtype=bool
    )
    return values[positions]


def measured_rows(table: Any, assay: str, rows: Any) -> np.ndarray | None:
    """Return whether ``assay`` measured each cell of ``rows``.

    Returns None when it measured all of them.
    """
    column = resolve_assay_membership(table, assay)
    if column is None:
        return None
    indices = _cell_rows(rows, table.N)
    if indices.size == 0:
        return None
    members = _membership_at(table, column, indices)
    return None if bool(members.all()) else members


def _count_unmeasured(table: Any, column: str, cells: Any) -> tuple[int, int]:
    """Return how many distinct cells of ``cells`` ``column`` marks False.

    ``cells`` is checked by :func:`_checked_cells`. Returns the number of
    unmeasured cells and the number of distinct cells.
    """
    if np.dtype(cells.dtype).kind != "b":
        rows = cells
        if not _strictly_increasing(rows):
            if _dense(rows, table.N):
                # A mask of the rows counts each cell once without sorting.
                chosen = np.zeros(table.N, dtype=bool)
                chosen[rows] = True
                return _count_unmeasured(table, column, chosen)
            rows = np.unique(rows)
        members = _membership_at(table, column, rows)
        return int(np.count_nonzero(~members)), int(members.size)
    unmeasured = 0
    selected = 0
    start = 0
    # The membership column bounds each read; the mask is read at its rows.
    for members in iter_metadata_column_blocks(table, column):
        stop = start + len(members)
        chosen = np.asarray(cells[start:stop], dtype=bool)
        selected += int(np.count_nonzero(chosen))
        unmeasured += int(np.count_nonzero(chosen & ~np.asarray(members, dtype=bool)))
        start = stop
    return unmeasured, selected


def require_measured_cells(
    table: Any,
    assay: str,
    cells: Any,
    *,
    operation: str,
    remedy: MeasuredCellsRemedy | None = None,
) -> None:
    """Raise ``UnmeasuredCellsError`` if ``assay`` did not measure one of ``cells``.

    ``cells`` is a boolean cell column name, a mask with one entry per cell, or
    integer rows.
    """
    column = resolve_assay_membership(table, assay)
    if column is None:
        return
    checked = _checked_cells(table, cells)
    unmeasured, selected = _count_unmeasured(table, column, checked)
    if unmeasured:
        raise UnmeasuredCellsError(
            operation,
            assay,
            column,
            unmeasured,
            selected,
            remedy,
            cells if isinstance(cells, str) else None,
        )


def measured_selection(
    table: Any, assay: str, selected: np.ndarray
) -> tuple[str, np.ndarray, str] | None:
    """Return the cells of ``selected`` that ``assay`` measured, as a mask.

    Returns the membership column, the mask, and the column's fingerprint, or
    None when ``assay`` has no membership column.
    """
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (table.N,):
        raise ValueError(
            f"A cell mask needs one entry per cell ({table.N}); got shape {mask.shape}"
        )
    column = resolve_assay_membership(table, assay)
    if column is None:
        return None
    measured = np.zeros(table.N, dtype=bool)
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (table.N,), np.dtype(bool))
    start = 0
    for block in iter_metadata_column_blocks(table, column):
        members = np.asarray(block, dtype=bool)
        stop = start + len(members)
        builder.update_array_block("values", (start,), members)
        measured[start:stop] = mask[start:stop] & members
        start = stop
    builder.end_array("values")
    return column, measured, builder.hexdigest()
