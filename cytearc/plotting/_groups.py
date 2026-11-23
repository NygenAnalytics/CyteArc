"""The groups of the figures of clones, receptors and variants, and their colors.

Figures group cells as the clone summaries do, through the store's
``_figure_groups``, and figures of a summary take the groups of its record. The
groups follow the labels of their first key column, then of the second: each
column's labels in the order that the column or label artifact stores, as a
stored display or ordered categories, then in natural order, and the receptor
types in their vocabulary's order. A group keeps its color in every figure: a
grouping by one cell metadata column takes the column's stored display palette,
or colors assigned over every label of the column; a label artifact's groups
take colors over every label of the artifact; receptor types take their
vocabulary's palette; and groups of several columns take colors over the groups
drawn. The labels that colors range over are read only when a figure colors its
groups: a column's a stored chunk at a time, and a label artifact's within the
store's memory budget.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from ..clones.groups import group_names
from ..metadata.rows import (
    apply_missing_mask,
    iter_metadata_column_blocks,
    metadata_missing_mask,
)
from ..metadata.selection import (
    NamedCellArtifact,
    cell_value_spec,
    resolve_cell_aligned_artifact,
    valid_category_mask,
)
from ..storage.refs import ArtifactRef
from ..utils.arguments import quoted_value
from ..utils.memory_budget import MemoryLedger
from ._contracts import CategoricalScale
from ._data import store_method
from ._display import (
    groups_in_stored_order,
    stored_artifact_scale,
    stored_categorical_scale,
    text_keyed_scale,
)
from ._style import RECEPTOR_TYPE_SCALE, palette_for_n, resolve_category_scale
from ._titles import artifact_titles, label_artifact

__all__ = [
    "PlotGroups",
    "labels_by_group",
    "names_columns",
    "plot_groups",
    "table_groups",
]

# The key columns whose labels are a vocabulary of fixed colors, and their
# titles.
_VOCABULARY_SCALES = {"receptor_type": RECEPTOR_TYPE_SCALE}
_VOCABULARY_TITLES = {"receptor_type": "Receptor type"}


@dataclass(frozen=True, slots=True)
class PlotGroups:
    """The groups of one figure: their names, each cell's group and their colors.

    Attributes:
        names: The groups' names, the labels of each joined by commas, in
            the order of the first key column's labels, then the second's:
            each column's labels in the order that the column or label
            artifact stores, then in natural order, and the receptor types
            in their vocabulary's order; ``("all cells",)`` without a
            grouping.
        columns: The grouping's key columns, empty without one.
        codes: Each cell's group, a position in ``names``, -1 for none; empty
            for the groups of a summary's record, whose tables name them.
        colors: Returns every group's color when the grouping keeps colors
            across figures, or None when groups take colors over those
            drawn; :meth:`scale` calls it, so the labels that colors range
            over are read only for a figure that colors its groups.
        titles: The title of each key column in the figure: a column's own
            name, and for the label artifact's kind, the artifact's name or
            title. Empty takes the key columns.
    """

    names: tuple[str, ...]
    columns: tuple[str, ...]
    codes: np.ndarray
    colors: Callable[[], CategoricalScale | None] = field(repr=False, compare=False)
    titles: tuple[str, ...] = ()

    @property
    def title(self) -> str | None:
        """The grouping's title, its key columns' titles joined by commas, or None without one."""
        return ", ".join(self.titles or self.columns) or None

    def scale(self, drawn: Sequence[str]) -> CategoricalScale:
        """Return the colors of the groups that a figure draws.

        Raises:
            ValueError: If a drawn group is not a group of the figure.
            TypeError: If the store does not give the labels that the
                groups' colors range over.
            MemoryError: If reading a label artifact's labels does not fit
                the memory budget.
        """
        drawn = [str(name) for name in drawn]
        unknown = sorted(set(drawn) - set(self.names))
        if unknown:
            raise ValueError(
                f"The figure has no group named {quoted_value(unknown[0])}"
            )
        colors = self.colors()
        if colors is None:
            return CategoricalScale(
                order=tuple(drawn),
                palette=dict(zip(drawn, palette_for_n(len(drawn)), strict=True)),
            )
        assert colors.order is not None and colors.palette is not None
        wanted = set(drawn)
        order = tuple(name for name in colors.order if name in wanted)
        return CategoricalScale(
            order=order,
            palette={name: colors.palette[name] for name in order},
            labels=(
                None
                if colors.labels is None
                else {name: colors.labels[name] for name in order}
            ),
        )


def names_columns(group_by: Any) -> bool:
    """Whether ``group_by`` names a column: text, or a sequence that is not empty.

    A value of another type, bytes among them, counts as naming one, so that
    the grouping refuses it.
    """
    if group_by is None:
        return False
    if isinstance(group_by, str | bytes) or not isinstance(group_by, Sequence):
        return True
    return len(group_by) > 0


def _texts(values: Any) -> list[str]:
    """The text of each distinct label that names a group, as the grouping names them."""
    distinct = np.asarray(pd.unique(np.asarray(values).ravel()), dtype=object)
    labels = distinct[valid_category_mask(distinct)].tolist()
    return list(dict.fromkeys(str(value) for value in labels))


def _store_labels(store: Any) -> None:
    """Refuse a store that cannot give the labels which group colors range over."""
    if not hasattr(store, "zw") or not hasattr(store, "cells"):
        raise TypeError(
            "store does not provide the labels of its cells, over which group "
            "colors are assigned"
        )


def _column_labels(store: Any, column: str) -> list[str]:
    """Every label of a cell metadata column, as groups name them, in bounded reads.

    The column is read a stored chunk at a time, so the read holds one chunk
    and the distinct labels, whatever the store's size. A row that the
    column's missing mask flags holds no label.
    """
    cells = store.cells
    mask = metadata_missing_mask(cells, column)
    found: dict[str, None] = {}
    start = 0
    for block in iter_metadata_column_blocks(cells, column):
        stop = start + len(block)
        missing = None if mask is None else np.asarray(mask[start:stop], dtype=bool)
        found.update(
            dict.fromkeys(_texts(apply_missing_mask(block, missing, labels=True)))
        )
        start = stop
    return list(found)


def _artifact_labels(store: Any, groups: ArtifactRef) -> list[str]:
    """Every label of a cell-label artifact, as groups name them, within the budget.

    Raises:
        MemoryError: If reading the artifact's labels does not fit the
            store's memory budget.
    """
    resolved = resolve_cell_aligned_artifact(
        store.zw,
        groups,
        value_name=cell_value_spec(groups.kind).name,
        expected_kind=groups.kind,
        ledger=MemoryLedger(
            int(store.memoryBytes),
            f"Reading the labels of the {groups.kind} artifact",
            codec_threads=store.nthreads,
        ),
    )
    return _texts(
        apply_missing_mask(resolved.values, resolved.missing_mask, labels=True)
    )


def _colors(
    store: Any,
    names: Sequence[str],
    columns: Sequence[str],
    scales: Sequence[CategoricalScale | None],
    *,
    group_by: Sequence[str],
    groups: ArtifactRef | None,
) -> CategoricalScale | None:
    """Every group's color, when the groups are those of one column or vocabulary.

    ``scales`` holds the scale of each key column that :func:`_key_scales`
    read, so the stored scales are read once for a figure.
    """
    if len(columns) != 1:
        return None
    (column,) = columns
    (stored,) = scales
    if tuple(group_by) == (column,):
        _store_labels(store)
        return resolve_category_scale(
            list(names),
            None,
            stored=stored,
            universe=partial(_column_labels, store, column),
        )
    if not group_by and groups is not None and column == groups.kind:
        _store_labels(store)
        return resolve_category_scale(
            list(names),
            None,
            stored=stored,
            universe=partial(_artifact_labels, store, groups),
        )
    if column in _VOCABULARY_SCALES:
        return resolve_category_scale(list(names), _VOCABULARY_SCALES[column])
    return None


def _no_colors() -> None:
    """The colors of a figure's one group, which take the first color drawn."""
    return None


def _key_titles(
    store: Any,
    columns: Sequence[str],
    *,
    n_group_by: int,
    groups: ArtifactRef | None,
    name: str | None,
) -> tuple[str, ...]:
    """The title of each key column: its name, the vocabulary's for the labels that clones keep apart, and the label artifact's name or title for its kind."""
    titles = [str(column) for column in columns]
    for position in range(n_group_by + (groups is not None), len(columns)):
        titles[position] = _VOCABULARY_TITLES.get(titles[position], titles[position])
    if groups is not None:
        (titles[n_group_by],) = artifact_titles(store, [(groups, name)])
    return tuple(titles)


def _key_scales(
    store: Any,
    columns: Sequence[str],
    *,
    n_group_by: int,
    groups: ArtifactRef | None,
) -> list[CategoricalScale | None]:
    """The scale of each key column, keyed by the text of its labels.

    A ``group_by`` column takes the scale it stores, the label artifact's
    kind the artifact's, and the receptor types their vocabulary's; other
    columns store none. The groups follow these scales' orders, and the
    groups of one column or vocabulary take their colors.
    """
    scales = []
    for position, column in enumerate(columns):
        if position < n_group_by:
            scale = text_keyed_scale(stored_categorical_scale(store, column))
        elif groups is not None and position == n_group_by:
            scale = text_keyed_scale(stored_artifact_scale(store, groups))
        else:
            scale = _VOCABULARY_SCALES.get(column)
        scales.append(scale)
    return scales


def plot_groups(
    store: Any,
    artifact: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None,
    groups: ArtifactRef | NamedCellArtifact | None,
    kept_apart: Sequence[tuple[str, np.ndarray]] = (),
    n_cells: int,
) -> PlotGroups:
    """Group the cells of a cell-aligned artifact for one figure, as the summaries group cells.

    The groups are the combinations of the labels of the ``group_by`` cell
    metadata columns, one or several, of the ``groups`` cell-label artifact,
    such as a clustering, and of ``kept_apart``, each with every artifact
    cell's label, such as the receptor types that clonotypes keep apart. The
    store groups the cells as ``ds.clones.diversity`` groups them, within the
    memory budget, and names each group by its labels joined by commas. A
    label artifact may hold fewer cells than ``artifact``: the cells it lacks
    have no group, nor do cells with a missing or blank label. Without a
    grouping, the artifact's ``n_cells`` cells are one group, ``"all
    cells"``, and the store is not read. A ``NamedCellArtifact`` groups by
    its artifact and titles the artifact's key column with its name in the
    figure; the grouping itself names the column by the artifact's kind.

    Raises:
        TypeError: If the store does not group cells for figures, or an
            argument has the wrong type.
        ValueError: If a ``group_by`` column is missing, named twice or kept
            apart already, or two groups get one name.
        MemoryError: If grouping the cells does not fit the memory budget.
    """
    groups, name = label_artifact(groups, "groups")
    if not names_columns(group_by) and groups is None and not kept_apart:
        return PlotGroups(
            names=("all cells",),
            columns=(),
            codes=np.zeros(int(n_cells), dtype=np.int64),
            colors=_no_colors,
        )
    grouped = store_method(store, "_figure_groups", "the groups of figures")(
        artifact, group_by=group_by, groups=groups, kept_apart=tuple(kept_apart)
    )
    columns = tuple(grouped.key_columns)
    by = columns[: len(columns) - (groups is not None) - len(kept_apart)]
    scales = _key_scales(store, columns, n_group_by=len(by), groups=groups)
    order = groups_in_stored_order(
        [tuple(key[column] for column in columns) for key in grouped.keys], scales
    )
    names = tuple(group_names([grouped.keys[at] for at in order]))
    codes = np.asarray(grouped.codes, dtype=np.int64)
    if order != list(range(len(order))):
        # Each cell's code moves to its group's place in the figure's order,
        # and -1, a cell without a group, to the last place, which keeps -1.
        place = np.full(len(order) + 1, -1, dtype=np.int64)
        place[order] = np.arange(len(order))
        codes = place[codes]
    return PlotGroups(
        names=names,
        columns=columns,
        codes=codes,
        colors=partial(
            _colors, store, names, columns, scales, group_by=by, groups=groups
        ),
        titles=_key_titles(
            store, columns, n_group_by=len(by), groups=groups, name=name
        ),
    )


def table_groups(
    store: Any, record: Any, groups: NamedCellArtifact | None = None
) -> PlotGroups:
    """Return the groups of a summary's record, which its tables name.

    ``record`` is a summary's record, such as a ``Rarefaction``: its
    ``group_keys`` holds one row per group, indexed by the group's name,
    and its ``group_by`` and ``groups`` name the grouping, after which the
    key columns hold the labels the clones keep apart. The record's label
    artifact is titled by ``groups``, a ``NamedCellArtifact`` of that
    artifact, or else by its own record. The groups follow the order that
    each key column stores, as :func:`plot_groups` orders them; the record's
    tables keep their own order.

    The store's labels are read only when a figure colors the groups.

    Raises:
        TypeError: If ``groups`` is not a ``NamedCellArtifact`` or None.
        ValueError: If ``groups`` names another artifact than the record's.
    """
    if groups is not None and not isinstance(groups, NamedCellArtifact):
        raise TypeError(
            "groups must be a NamedCellArtifact that names the record's label "
            f"artifact, not {type(groups).__name__}"
        )
    if groups is not None and groups.artifact != record.groups:
        raise ValueError(
            f"groups names the {groups.artifact.kind} artifact "
            f"{groups.artifact.artifact_id[:12]}, but the record groups its cells by "
            + (
                "no label artifact"
                if record.groups is None
                else f"the {record.groups.kind} artifact {record.groups.artifact_id[:12]}"
            )
        )
    keys: pd.DataFrame = record.group_keys
    columns = tuple(str(column) for column in keys.columns)
    scales = _key_scales(
        store, columns, n_group_by=len(record.group_by), groups=record.groups
    )
    # Each group's labels as text, also for a record without key columns,
    # whose one group is "all cells".
    labels = [tuple(map(str, row)) for row in keys.to_numpy(dtype=object)]
    found = [str(name) for name in keys.index.tolist()]
    names = tuple(found[at] for at in groups_in_stored_order(labels, scales))
    return PlotGroups(
        names=names,
        columns=columns,
        codes=np.empty(0, dtype=np.int64),
        colors=partial(
            _colors,
            store,
            names,
            columns,
            scales,
            group_by=tuple(record.group_by),
            groups=record.groups,
        ),
        titles=_key_titles(
            store,
            columns,
            n_group_by=len(record.group_by),
            groups=record.groups,
            name=None if groups is None else groups.name,
        ),
    )


def labels_by_group(
    cell_groups: Any, group_keys: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[str, np.ndarray]]:
    """Each cell's label in each column, from its group's row of ``group_keys``.

    ``cell_groups`` holds each cell's group name, missing for none, as a
    categorical or as text, and ``group_keys`` one row per group, indexed by
    its name, as a summary's record holds it. A cell without a group, or of a
    group that ``group_keys`` does not hold, has the label None. A
    categorical is matched through its codes, so the names are looked up
    once per category.
    """
    names = pd.Index(group_keys.index)
    values = pd.Series(cell_groups)
    if isinstance(values.dtype, pd.CategoricalDtype):
        # Each category's row, and -1 at the end for the code of no category.
        at = np.append(names.get_indexer(values.cat.categories), -1)
        rows = at[np.asarray(values.cat.codes)]
    else:
        rows = names.get_indexer(values.to_numpy(dtype=object))
    # The row -1 takes the None after the groups' labels.
    found = []
    for column in columns:
        labels = np.empty(len(names) + 1, dtype=object)
        labels[:-1] = group_keys[column].to_numpy(dtype=object)
        found.append((column, labels[rows]))
    return found
