"""Names of Zarr-backed metadata columns.

Zarr reads ``/`` and ``\\`` in an array name as path separators, so a column
named with either one would nest into groups instead of forming one array.
Imports store such source columns under a key with ``_`` in place of each
separator. CyteArc's own ``I``, ``ids``, and ``names`` columns, the
missing-value masks linked through ``__cytearc_missing__<name>``, and the
categories of categorical columns linked through
``__cytearc_categories__<name>`` are reserved.
The cell column named by :func:`assay_membership_column` records which cells
an assay measures, and the feature columns named by
:func:`measured_features_column` which features each source of a merge
measured, under names that :func:`measured_features_columns` checks.
"""

from collections.abc import Iterable

from .arrays import MISSING_MASK_PREFIX
from .categorical import CATEGORIES_PREFIX

_LINKED_PREFIXES = (MISSING_MASK_PREFIX, CATEGORIES_PREFIX)

RESERVED_METADATA_COLUMNS = frozenset({"I", "ids", "names"})
"""Columns that CyteArc writes into every cell and feature table."""

ASSAY_MEMBERSHIP_ROLE = "assay_membership"
"""``role`` attribute of the cell column that records which cells an assay measures."""


def assay_membership_column(assay: str) -> str:
    """Return the name of the cell column that records which cells ``assay`` measures.

    An assay without this column measures every cell.
    """
    return f"{assay}_I"


def assay_membership_attributes(assay: str) -> dict[str, str]:
    """Return the attributes that mark ``<assay>_I`` as the membership of ``assay``."""
    return {"role": ASSAY_MEMBERSHIP_ROLE, "assay": assay}


MEASURED_FEATURES_PREFIX = "measured_in_"
"""Start of the name of each feature column that records which features one source of a merge measured."""

MEASURED_FEATURES_ROLE = "measured_features"
"""``role`` attribute of a feature column that records which features one source of a merge measured."""

CELL_SOURCES_ROLE = "cell_sources"
"""``role`` attribute of the cell column that names each cell's source in a merge or a composite that records which features each source measured."""

SOURCE_COLUMN_ATTRIBUTE = "sourceColumn"
"""Attribute of a feature column of :data:`MEASURED_FEATURES_ROLE` that names the cell column of each cell's source."""


def measured_features_column(source: str) -> str:
    """Return the name of the feature column that records which features ``source`` measured.

    It is ``measured_in_<source>``, with ``_`` in place of each ``/`` and
    ``\\`` (:func:`metadata_column_key`). A merge or a composite writes it as
    a Boolean column with the attributes that
    :func:`measured_features_attributes` returns, which keep the source's own
    name.
    """
    return metadata_column_key(f"{MEASURED_FEATURES_PREFIX}{source}")


def measured_features_attributes(source: str, source_column: str) -> dict[str, str]:
    """Return the attributes that mark a feature column as the record of what ``source`` measured.

    ``source_column`` is the cell column of :data:`CELL_SOURCES_ROLE` that
    names each cell's source.
    """
    return {
        "role": MEASURED_FEATURES_ROLE,
        "source": source,
        SOURCE_COLUMN_ATTRIBUTE: source_column,
    }


def cell_sources_attributes() -> dict[str, str]:
    """Return the attributes that mark a cell column as the name of each cell's source."""
    return {"role": CELL_SOURCES_ROLE}


def measured_features_columns(sources: Iterable[str]) -> dict[str, str]:
    """Return the column that records which features each source measured, by source.

    Raises:
        ValueError: If two sources share a column, as names that differ only
            in ``/``, ``\\`` and ``_`` do, or their columns are named alike
            once case is ignored, as a file system that ignores case stores
            them as one column.
    """
    columns: dict[str, str] = {}
    first: dict[str, str] = {}
    for source in sources:
        column = measured_features_column(source)
        other = first.setdefault(column.casefold(), source)
        if other != source:
            if columns[other] == column:
                raise ValueError(
                    f"Sources {other!r} and {source!r} would both record which "
                    f"features they measured in the feature column {column!r}, "
                    "since a column name counts '/' and '\\' as '_'. Name the "
                    "sources so that the names differ once '/' and '\\' are "
                    "counted as '_'."
                )
            raise ValueError(
                f"Sources {other!r} and {source!r} would record which features "
                f"they measured in the feature columns {columns[other]!r} and "
                f"{column!r}, which a file system that ignores case stores as one "
                "column. Name the sources so that the names differ in more than "
                "case, with '/' and '\\' counted as '_'."
            )
        columns[source] = column
    return columns


_UNSTORABLE_NAMES = frozenset({"", ".", ".."})


def is_reserved_metadata_name(name: str) -> bool:
    """Return whether a column name collides with CyteArc's own columns."""
    return name in RESERVED_METADATA_COLUMNS or name.startswith(_LINKED_PREFIXES)


def metadata_column_key(name: str) -> str:
    """Return ``name`` with each ``/`` and ``\\`` replaced by ``_``."""
    return name.replace("/", "_").replace("\\", "_")


def is_metadata_column_key(name: object) -> bool:
    """Return whether ``name`` names one metadata array as it stands."""
    return (
        isinstance(name, str)
        and name not in _UNSTORABLE_NAMES
        and metadata_column_key(name) == name
    )


def nested_group_error(column: str) -> TypeError:
    """Return the error for a table member that is a group, not a column.

    Stores written before CyteArc renamed imported columns nested a source
    column whose name contains ``/`` or ``\\`` into groups.
    """
    return TypeError(
        f"The metadata table holds a nested group named {column!r} instead of "
        "a column. A source column name containing '/' or '\\' created it; "
        "import the source again to store that column under a name with '_' "
        "in their place."
    )


def validate_metadata_column_name(name: str) -> None:
    """Reject a metadata column name that Zarr cannot store as one array.

    Names that start with the prefix of missing-value masks or of the
    categories of categorical columns are rejected too, so a written column
    can never replace or orphan a linked array.

    Args:
        name: Column name to check.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty, ``.``, or ``..``, contains ``/`` or
            ``\\``, or starts with ``__cytearc_missing__`` or
            ``__cytearc_categories__``.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"Metadata column names must be strings, not {type(name).__name__}"
        )
    if name in _UNSTORABLE_NAMES:
        raise ValueError(f"Metadata column name {name!r} cannot name a Zarr array")
    key = metadata_column_key(name)
    if key.startswith(MISSING_MASK_PREFIX):
        raise ValueError(
            f"Metadata column name {name!r} uses the prefix "
            f"{MISSING_MASK_PREFIX!r}, which CyteArc reserves for missing-value masks"
        )
    if key.startswith(CATEGORIES_PREFIX):
        raise ValueError(
            f"Metadata column name {name!r} uses the prefix "
            f"{CATEGORIES_PREFIX!r}, which CyteArc reserves for the categories of "
            "categorical columns"
        )
    if key != name:
        raise ValueError(
            f"Metadata column name {name!r} must not contain '/' or '\\' because "
            f"Zarr reads them as path separators; use {key!r} instead"
        )


def metadata_column_keys(
    names: Iterable[str],
    *,
    taken: Iterable[str] = (),
) -> dict[str, str]:
    """Plan the key that stores each source metadata column.

    Source names that are already valid keys keep their name. Each name that
    contains ``/`` or ``\\`` takes :func:`metadata_column_key` of itself, or,
    when that key is in use, the first free key among ``<key>_2``,
    ``<key>_3``, and so on, in source order. A name is left out when it or its
    key is reserved, when it cannot name a Zarr array, or when it is valid but
    already in ``taken``.

    Args:
        names: Source column names in source order. Repeated names are planned
            once.
        taken: Names already present in the destination table.

    Returns:
        The key of each planned source name, in source order.

    Raises:
        TypeError: If a name is not a string.
    """
    sources = list(dict.fromkeys(names))
    for name in sources:
        if not isinstance(name, str):
            raise TypeError(
                f"Metadata column names must be strings, not {type(name).__name__}"
            )
    used = set(taken)
    keys: dict[str, str] = {}
    for name in sources:
        if metadata_column_key(name) != name:
            continue
        if name in _UNSTORABLE_NAMES or is_reserved_metadata_name(name):
            continue
        if name in used:
            continue
        keys[name] = name
        used.add(name)
    for name in sources:
        base = metadata_column_key(name)
        if base == name or is_reserved_metadata_name(base):
            continue
        key, suffix = base, 2
        while key in used or is_reserved_metadata_name(key):
            key, suffix = f"{base}_{suffix}", suffix + 1
        keys[name] = key
        used.add(key)
    return {name: keys[name] for name in sources if name in keys}
