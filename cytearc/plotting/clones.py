"""Figures of clones: the cells that descend from one ancestor cell."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..clones.diversity import INDEX_COLUMNS, ClonalDiversity, Rarefaction
from ..clones.expansion import DEFAULT_RANK_SPLITS, rank_window_labels, rank_windows
from ..clones.fate import CloneCoupling, dominant_fates
from ..clones.overlap import OVERLAP_INDICES, CloneOverlap, CloneTracking
from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument, quoted_value
from ..utils.arrays import sort_categories, sorted_unique
from ._charts import draw_bars, draw_lines, draw_matrix, pair_matrix
from ._clone_names import CloneNames, clone_names
from ._contracts import CategoricalScale, ColorScale
from ._data import store_method
from ._figure import (
    PlotResult,
    close_figures_on_error,
    finish_plot,
    normalize_axes_target,
)
from ._groups import labels_by_group, names_columns, plot_groups, table_groups
from ._heatmap_utils import draw_dendrogram
from ._style import (
    MAX_FIGURE_WIDTH_INCHES,
    colormap_palette,
    figure_size,
    foreground_color,
    item_inches,
    palette_for_n,
    theme_context,
)

__all__ = [
    "clonal_diversity",
    "clonal_expansion",
    "clonal_rarefaction",
    "clone_abundance",
    "clone_coupling",
    "clone_fate_bias",
    "clone_overlap",
    "clone_tracking",
]

# A panel of a fixed size, such as one order of a rarefaction, is about this
# many inches, and small multiples fill rows of at most three.
_PANEL_INCHES = (4.2, 3.6)
_SMALL_MULTIPLES = 3
# Room in inches beside a heatmap for its colour bar.
_COLORBAR_INCHES = 1.2
# The indices that clonal_diversity draws unless told otherwise: the clones,
# Shannon entropy and the inverse Simpson index, the observed Hill numbers of
# orders 0, 1, as its logarithm, and 2.
_DIVERSITY_INDICES = ("n_clones", "shannon", "inverse_simpson")
_COUPLING_VALUES = ("z_score", "coupling")


def _order_title(order: int, names: CloneNames) -> str:
    return {
        0: f"{names.plural.capitalize()} (order 0)",
        1: "Shannon diversity (order 1)",
        2: "Simpson diversity (order 2)",
    }[order]


def _overlap_titles(names: CloneNames) -> dict[str, str]:
    return {
        "shared": f"Shared {names.plural}",
        "jaccard": "Jaccard index",
        "sorensen": "Sørensen index",
        "overlap": "Overlap coefficient",
        "cosine": "Cosine of presence",
        "morisita_horn": "Morisita-Horn index",
        "abundance_cosine": f"Cosine of {names.singular} sizes",
    }


def _coupling_titles(names: CloneNames) -> dict[str, str]:
    return {
        "share_cosine": f"Cosine of {names.singular} shares",
        "abundance_cosine": f"Cosine of {names.singular} sizes",
        "normalized_covariance": "Normalized covariance",
        "jaccard": "Jaccard index",
    }


def _index_titles(names: CloneNames) -> dict[str, str]:
    """The title of each column of the diversity indices, as the figure names it."""
    plural = names.plural.capitalize()
    return {
        "n_cells": "Cells",
        "n_clones": plural,
        "n_singletons": f"{plural} of one cell",
        "n_doubletons": f"{plural} of two cells",
        "coverage": "Sample coverage",
        "shannon": "Shannon entropy",
        "pielou": "Pielou's evenness",
        "simpson": "Simpson index",
        "inverse_simpson": "Inverse Simpson index",
        "gini_simpson": "Gini-Simpson index",
        "gini": "Gini coefficient",
        "d50": "D50",
        "chao1": "Chao1",
        "chao1_bias_corrected": "Bias-corrected Chao1",
        "ace": "ACE",
    }


def _color_scale(color_scale: Any) -> ColorScale | None:
    if color_scale is not None and not isinstance(color_scale, ColorScale):
        raise TypeError("color_scale must be a ColorScale or None")
    return color_scale


def _small_multiples(n_panels: int, height: float | None = None) -> tuple[float, float]:
    """The size of each of small multiples, which share the page three to a row.

    A panel is at most ``_PANEL_INCHES`` wide, and as tall as ``height``, or
    of the shape of ``_PANEL_INCHES``.
    """
    width = min(
        _PANEL_INCHES[0], MAX_FIGURE_WIDTH_INCHES / min(_SMALL_MULTIPLES, n_panels)
    )
    return width, (
        width * _PANEL_INCHES[1] / _PANEL_INCHES[0] if height is None else height
    )


def _kept_apart_labels(
    cells: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[str, np.ndarray]]:
    """Each cell's labels in the columns of ``cells`` that the clones keep apart."""
    return [(column, np.asarray(cells[column], dtype=object)) for column in columns]


@close_figures_on_error
def clone_abundance(
    store: Any,
    clones: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    max_clones: int = 20,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how many clones have each size, and the largest clones.

    ``clones`` is a clone artifact, such as one from
    ``ds.clones.import_clones`` or ``ds.repertoire.clonotypes``. The first
    panel counts clones by their number of cells on log scales: in most
    experiments many clones hold one or a few cells, and expanded clones form
    the tail. The second panel draws the ``max_clones`` clones with the most
    cells, labelled with their IDs. Clonotypes are labelled instead by the
    junction amino acids of their representative cell's ``VDJ_1`` chain, or
    of its ``VJ_1`` chain without a VDJ chain, such as ``CASSLGQAYEQYF``, and
    those defined by ``receptor_arms="VJ"`` by the ``VJ_1`` chain first,
    which their cells share. Where two of the clonotypes drawn share a
    junction, each label also shows the clonotype's number, the rank in its
    ID, such as ``CAVA 1`` and ``CAVA 3``. The figure names the clones as
    their artifact does, so for clonotypes the second panel's title reads
    ``Largest 20 clonotypes``, and for clonotype clusters, which a distance
    metric defines, ``Largest 20 clonotype clusters``.

    ``group_by`` names cell metadata columns, one or several, such as the
    sample or time point, and ``groups`` a cell-label artifact, such as a
    clustering; the groups are the combinations of their labels, as the clone
    summaries group cells, named by their labels joined by commas. With a
    grouping, the first panel draws one line per group, counting the cells of
    each clone within the group, and the second stacks each clone's cells by
    group, with one legend of the groups. Cells without a group are left
    out. Clonotypes are counted within each receptor type, as every clone
    summary counts them, so their groups are the receptor types, or each
    group's receptor types, named like ``"s1, TCR"``.

    Tables: ``sizes`` (``group``, ``clone_size``, ``n_clones``) and ``top``,
    one row per clone drawn and group, the largest clone first, zeros
    included (``clone``, ``label``, ``group``, ``n_cells``); without a
    grouping, every cell is in the group ``"all cells"``. Groups are values
    of the tables, so a group may take any name.

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load, group or name clones for figures.
        ValueError: If ``clones`` is not a clone artifact, a ``group_by``
            column is missing, named twice or kept apart by the clones
            already, two groups get one name, or no cell, or no cell with a
            group, has a clone.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact or grouping the cells does not
            fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_clones = integer_argument(max_clones, "max_clones", minimum=1)
    loaded = store_method(store, "clones.load_clones")(clones)
    plotted = plot_groups(
        store,
        clones,
        group_by=group_by,
        groups=groups,
        kept_apart=_kept_apart_labels(loaded.cells, loaded.kept_apart),
        n_cells=len(loaded.cells),
    )
    codes = np.asarray(loaded.cells["clone"].cat.codes, dtype=np.int64)
    cell_group = plotted.codes
    keep = (codes >= 0) & (cell_group >= 0)
    if not keep.any():
        raise ValueError(
            "No cell with a group has a clone, so there is nothing to plot"
            if (codes >= 0).any()
            else "No cell has a clone, so there is nothing to plot"
        )
    # Each clone's cells in each group, from the codes alone.
    n_clones = len(loaded.clones)
    pairs, pair_cells = np.unique(
        cell_group[keep] * n_clones + codes[keep], return_counts=True
    )
    held = sorted_unique(pairs // n_clones)
    group_order = [plotted.names[code] for code in held.tolist()]
    pair_groups = np.searchsorted(held, pairs // n_clones)
    pair_codes = pairs % n_clones
    width = int(pair_cells.max()) + 1
    size_keys, size_counts = np.unique(
        pair_groups * width + pair_cells, return_counts=True
    )
    sizes = pd.DataFrame(
        {
            "group": np.asarray(group_order, dtype=object)[size_keys // width],
            "clone_size": (size_keys % width).astype(np.int64),
            "n_clones": size_counts.astype(np.int64),
        }
    )
    totals = np.bincount(pair_codes, weights=pair_cells, minlength=n_clones)
    present = np.flatnonzero(totals > 0)
    # The largest clones, ties in the clones' rank order.
    chosen = present[np.lexsort((present, -totals[present]))[:max_clones]]
    rows = np.full(n_clones, -1, dtype=np.int64)
    rows[chosen] = np.arange(len(chosen))
    taken = rows[pair_codes] >= 0
    matrix = np.zeros((len(chosen), len(group_order)), dtype=np.int64)
    matrix[rows[pair_codes[taken]], pair_groups[taken]] = pair_cells[taken]
    ids = loaded.clones["clone"].to_numpy(dtype=object)[chosen]
    names = clone_names(store, clones)
    clone_labels = np.asarray(names.labels(ids), dtype=object)
    # One row per clone drawn and group, zeros included, so the groups are
    # values of the table, whatever their names.
    top_table = pd.DataFrame(
        {
            "clone": np.repeat(ids, len(group_order)),
            "label": np.repeat(clone_labels, len(group_order)),
            "group": np.tile(np.asarray(group_order, dtype=object), len(chosen)),
            "n_cells": matrix.reshape(-1),
        }
    )
    colors = plotted.scale(group_order)
    if figsize is None and target is None:
        figsize, _ = figure_size(
            2,
            panel=(
                MAX_FIGURE_WIDTH_INCHES / 2,
                item_inches(len(chosen), per_item=0.28, base=1.2, minimum=3.6),
            ),
            columns=2,
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["sizes", "top"], figsize=figsize, n_columns=2
        )
        draw_lines(
            axes["sizes"],
            sizes,
            x="clone_size",
            y="n_clones",
            series="group",
            colors=colors,
            markers=True,
            log_x=True,
            log_y=True,
            x_label=f"Cells of the {names.singular}",
            y_label=names.plural.capitalize(),
            title=f"{names.singular.capitalize()} sizes",
        )
        # One legend of the groups, beside the second panel, serves both.
        drawn = draw_bars(
            axes["top"],
            top_table,
            item="label",
            value="n_cells",
            segment="group",
            colors=colors,
            value_label="Cells",
            title=names.largest(len(chosen)),
            legend_title=plotted.title,
            show_legend=show_legend,
        )
    return finish_plot(
        figure,
        axes,
        owns,
        name="clone_abundance",
        theme=theme,
        tables={"sizes": sizes, "top": top_table},
        drawn=[drawn],
        inputs={"clones": clones},
        settings={"grouping": plotted.title, "max_clones": max_clones},
        n_cells=int(keep.sum()),
        show=show,
    )


def _long_table(wide: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Turn a table of groups by columns into rows of group, column, value, fraction."""
    totals = wide.sum(axis=1).to_numpy(dtype=np.float64)
    shares = np.divide(
        wide.to_numpy(dtype=np.float64),
        totals[:, None],
        out=np.full(wide.shape, np.nan),
        where=totals[:, None] > 0,
    )
    return pd.DataFrame(
        {
            "group": np.repeat(wide.index.to_numpy(dtype=object), wide.shape[1]),
            column: np.tile(wide.columns.to_numpy(dtype=object), wide.shape[0]),
            value: wide.to_numpy(dtype=np.int64).reshape(-1),
            "fraction": shares.reshape(-1),
        }
    )


@close_figures_on_error
def clonal_expansion(
    store: Any,
    expansion: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    count: str = "cells",
    normalize: bool = True,
    splits: Sequence[int] = DEFAULT_RANK_SPLITS,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the clonal expansion and clonal space of groups of cells.

    ``expansion`` is a ``clonal_expansion`` artifact, such as one from
    ``ds.clones.expansion``. The first panel draws one stacked bar per group:
    the fraction of its cells (``count="cells"``) or of its clones
    (``count="clones"``) in each expansion category, as scirpy's
    ``summarize_clonal_expansion`` and scRepertoire's ``clonalHomeostasis``
    do. The second panel ranks each group's clones by their cells in the
    group and draws the fraction of its cells that the clones of each window
    of ranks hold, as scRepertoire's ``clonalProportion`` does: a sample
    whose ten largest clones hold most of its cells is dominated by a few
    clones. Windows that hold no cell are left out.

    The groups are the expansion's own, such as its samples, unless
    ``group_by`` names cell metadata columns, one or several, such as the
    cell type, or ``groups`` a cell-label artifact, such as a clustering; the
    plotted groups are then the combinations of their labels, as the clone
    summaries group cells. Each cell keeps the category measured within its
    expansion group, so grouping the cells by cell type shows which cell
    types hold the expanded cells of their samples; the ranks are counted
    within the plotted groups. Labels that the clones keep apart stay apart:
    the plotted groups of clonotypes are each group's receptor types, named
    like ``"blood, TCR"``. Cells without a category or a group are left out.

    Tables: ``categories`` (``group``, ``category``, then ``n_cells`` or
    ``n_clones``, and ``fraction``) and ``rank_windows`` (``group``,
    ``window``, ``n_cells`` and ``fraction``).

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load, group or name clones for figures.
        ValueError: If ``expansion`` is not a ``clonal_expansion``
            artifact, ``count`` is neither ``"cells"`` nor
            ``"clones"``, a ``group_by`` column is missing, named twice or
            kept apart by the clones already, two groups get one name, or no
            cell with a group has a category.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact or grouping the cells does not
            fit the memory budget.
    """
    normalize = bool_argument(normalize, "normalize")
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if count not in ("cells", "clones"):
        raise ValueError("count must be 'cells' or 'clones'")
    loaded = store_method(store, "clones.load_expansion")(expansion, splits=splits)
    # The expansion's key columns after its group_by columns and its label
    # artifact's kind hold the labels that its clones keep apart, such as the
    # receptor type, which each cell takes from its group.
    key_columns = [str(column) for column in loaded.group_keys.columns]
    kept_apart = key_columns[len(loaded.group_by) + (loaded.groups is not None) :]
    category_names = list(loaded.labels)
    window_names = list(rank_window_labels(splits))
    if groups is None and not names_columns(group_by):
        # The expansion's own groups, which the loaded tables count.
        group_name = ", ".join(key_columns) or None
        order = sort_categories(pd.unique(loaded.categories["group"].to_numpy(object)))
        wide = (
            loaded.categories.pivot(
                index="group", columns="category", values=f"n_{count}"
            )
            .reindex(index=order, columns=category_names, fill_value=0)
            .astype(np.int64)
        )
        windows = (
            loaded.rank_windows.pivot(index="group", columns="window", values="n_cells")
            .reindex(index=order, columns=window_names, fill_value=0)
            .astype(np.int64)
        )
        n_cells = int(loaded.categories["n_cells"].sum())
    else:
        plotted = plot_groups(
            store,
            expansion,
            group_by=group_by,
            groups=groups,
            kept_apart=labels_by_group(
                loaded.cells["group"], loaded.group_keys, kept_apart
            ),
            n_cells=len(loaded.cells),
        )
        labels, group_name = plotted.codes, plotted.title
        category = np.asarray(loaded.cells["category"].cat.codes, dtype=np.int64)
        clone = np.asarray(loaded.cells["clone"].cat.codes, dtype=np.int64)
        keep = (category >= 0) & (labels >= 0)
        held = sorted_unique(labels[keep])
        order = [plotted.names[code] for code in held.tolist()]
        group_codes = np.searchsorted(held, labels[keep])
        width = len(category_names)
        if count == "cells":
            keys = group_codes * width + category[keep]
        else:
            # Each clone counts once per group and category it holds there.
            n_clones = max(1, len(loaded.cells["clone"].cat.categories))
            triples = sorted_unique(
                (group_codes * n_clones + clone[keep]) * width + category[keep]
            )
            keys = triples // (n_clones * width) * width + triples % width
        wide = pd.DataFrame(
            np.bincount(keys, minlength=len(order) * width).reshape(len(order), width),
            index=pd.Index(order, name="group"),
            columns=pd.Index(category_names, name="category"),
        )
        windows = pd.DataFrame(
            rank_windows(clone[keep], group_codes, len(order), splits),
            index=pd.Index(order),
            columns=pd.Index(window_names),
        )
        n_cells = int(keep.sum())
    if not len(order):
        raise ValueError(
            "No cell with a group has an expansion category, so there is nothing to plot"
        )
    names = clone_names(store, loaded.clones)
    noun = names.singular.capitalize()
    counted = "cells" if count == "cells" else names.plural
    shown_windows = windows.loc[:, windows.sum(axis=0) > 0]
    # Categories take colours in size order, as do the windows of ranks.
    window_palette = colormap_palette(window_names, "viridis")
    category_colors = CategoricalScale(
        order=tuple(category_names),
        palette=colormap_palette(category_names, "viridis_r"),
    )
    window_colors = CategoricalScale(
        order=tuple(shown_windows.columns),
        palette={window: window_palette[window] for window in shown_windows.columns},
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            2,
            panel=(
                MAX_FIGURE_WIDTH_INCHES,
                item_inches(len(order), per_item=0.22, base=1.2, minimum=2.4),
            ),
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["categories", "rank_windows"],
            figsize=figsize,
            n_columns=1,
        )
        drawn = [
            draw_bars(
                axes["categories"],
                _long_table(wide, "category", f"n_{count}"),
                item="group",
                value=f"n_{count}",
                segment="category",
                colors=category_colors,
                normalize=normalize,
                value_label=(
                    f"Fraction of {counted}" if normalize else counted.capitalize()
                ),
                title="Clonal expansion",
                legend_title=(
                    f"{noun} size"
                    if loaded.breakpoints is not None
                    else f"{noun} frequency"
                ),
                show_legend=show_legend,
            ),
            draw_bars(
                axes["rank_windows"],
                _long_table(shown_windows, "window", "n_cells"),
                item="group",
                value="n_cells",
                segment="window",
                colors=window_colors,
                normalize=normalize,
                value_label="Fraction of cells" if normalize else "Cells",
                title=f"Clonal space by {names.singular} rank",
                legend_title=f"{noun} rank",
                show_legend=show_legend,
            ),
        ]
    return finish_plot(
        figure,
        axes,
        owns,
        name="clonal_expansion",
        theme=theme,
        tables={
            "categories": _long_table(wide, "category", f"n_{count}"),
            "rank_windows": _long_table(windows, "window", "n_cells"),
        },
        drawn=drawn,
        inputs={"expansion": expansion, "clones": loaded.clones},
        settings={
            "grouping": group_name,
            "count": count,
            "normalize": normalize,
            "splits": list(loaded.splits),
        },
        n_cells=n_cells,
        show=show,
    )


@close_figures_on_error
def clonal_rarefaction(
    store: Any,
    rarefaction: Rarefaction,
    *,
    base: str = "size",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the rarefaction and extrapolation curves of groups of cells.

    ``rarefaction`` is the result of ``ds.clones.rarefaction``, which sets
    the groups, the orders and the sizes of each curve. Each panel draws one
    order of Hill numbers, the effective number of clones, against the cells
    sampled (``base="size"``) or the sample coverage (``base="coverage"``),
    as iNEXT's ``ggiNEXT`` draws them: one line per group, solid where the
    group is rarefied, dashed where it is extrapolated, and a point at its
    observed size. With ``base="size"`` a last panel draws each group's
    sample coverage against its size, which shows how complete each sample
    is. Curves that keep rising at the observed point have not reached their
    plateau: more cells would find more clones. The panels fill rows of
    three, and one legend of the groups stands beside the first row.

    Tables: ``curves``, the record's curves (``group``, ``order``, ``size``,
    ``method``, ``diversity``, ``coverage``); the record's ``group_keys``
    holds each group's labels.

    Raises:
        TypeError: If ``rarefaction`` is not a ``Rarefaction`` record,
            another argument has the wrong type, or the store does not name
            clones for figures or give the labels that the groups' colours
            range over.
        ValueError: If ``base`` is neither ``"size"`` nor ``"coverage"``, or
            the record holds no curve.
        KeyError: If ``theme`` names no theme, or the store does not hold
            the record's clones.
        RuntimeError: If the record's clones are incomplete, as a call that
            stopped before it finished leaves them.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if not isinstance(rarefaction, Rarefaction):
        raise TypeError("rarefaction must be the Rarefaction of ds.clones.rarefaction")
    if base not in ("size", "coverage"):
        raise ValueError("base must be 'size' or 'coverage'")
    curves = rarefaction.curves
    if curves.empty:
        raise ValueError("The rarefaction holds no curve, so there is nothing to plot")
    names = clone_names(store, rarefaction.clones)
    plotted = table_groups(store, rarefaction)
    held = set(curves["group"].tolist())
    colors = plotted.scale([name for name in plotted.names if name in held])
    panels = [f"order_{order}" for order in rarefaction.orders]
    if base == "size":
        panels.append("coverage")
    # One legend of the groups, beside the last panel of the first row.
    legend_panel = panels[min(_SMALL_MULTIPLES, len(panels)) - 1]
    marked = curves.assign(
        extrapolated=curves["method"].eq("extrapolation").to_numpy(),
        observed=curves["method"].eq("observed").to_numpy(),
    )
    by_order = dict(tuple(marked.groupby("order", sort=False)))
    if figsize is None and target is None:
        figsize, _ = figure_size(
            len(panels), panel=_small_multiples(len(panels)), columns=_SMALL_MULTIPLES
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=panels, figsize=figsize, n_columns=_SMALL_MULTIPLES
        )
        drawn = None
        for panel in panels:
            order = (
                int(panel.split("_")[1])
                if panel != "coverage"
                else rarefaction.orders[0]
            )
            found = draw_lines(
                axes[panel],
                by_order[order],
                x="size" if base == "size" else "coverage",
                y="coverage" if panel == "coverage" else "diversity",
                series="group",
                colors=colors,
                dashed="extrapolated",
                marked="observed",
                x_label="Cells" if base == "size" else "Sample coverage",
                y_label="Sample coverage" if panel == "coverage" else "Hill number",
                title=(
                    "Sample completeness"
                    if panel == "coverage"
                    else _order_title(order, names)
                ),
                legend_title=plotted.title,
                show_legend=show_legend and panel == legend_panel,
            )
            if panel == legend_panel:
                drawn = found
    return finish_plot(
        figure,
        axes,
        owns,
        name="clonal_rarefaction",
        theme=theme,
        tables={"curves": curves},
        drawn=[drawn],
        inputs={"clones": rarefaction.clones},
        settings={
            "grouping": plotted.title,
            "orders": list(rarefaction.orders),
            "base": base,
        },
        n_cells=int(
            curves.loc[curves["method"] == "observed"]
            .drop_duplicates("group")["size"]
            .sum()
        ),
        show=show,
    )


def _diversity_indices(indices: Any) -> tuple[str, ...]:
    """Return the diversity indices to draw, checked against the indices measured."""
    if isinstance(indices, str):
        chosen: list[Any] = [indices]
    elif isinstance(indices, Sequence) and not isinstance(indices, bytes):
        chosen = list(indices)
    else:
        raise TypeError("indices must be the name of an index or a sequence of names")
    if not chosen:
        raise ValueError("indices must name at least one index")
    for name in chosen:
        if not isinstance(name, str):
            raise TypeError(
                "indices must be the name of an index or a sequence of names"
            )
        if name not in INDEX_COLUMNS:
            raise ValueError(
                f"indices names {quoted_value(name)}, which ds.clones.diversity does not "
                f"measure; choose from {', '.join(INDEX_COLUMNS)}"
            )
    if len(set(chosen)) != len(chosen):
        raise ValueError("indices names an index twice")
    return tuple(chosen)


@close_figures_on_error
def clonal_diversity(
    store: Any,
    diversity: ClonalDiversity,
    *,
    indices: str | Sequence[str] = _DIVERSITY_INDICES,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot diversity indices of groups of cells, one panel of bars per index.

    ``diversity`` is the result of ``ds.clones.diversity``, which sets the
    groups. ``indices`` names one or several columns of its ``indices``
    table, by default ``("n_clones", "shannon", "inverse_simpson")``: the
    clones, Shannon entropy and the inverse Simpson index, which are the
    observed Hill numbers of orders 0, 1, as its logarithm, and 2. Each
    panel draws one bar per group, in the groups' colours, and the panels
    fill rows of three, with one legend of the groups beside the first row.
    A group whose index is not defined, such as Pielou's evenness of a group
    of one clone, draws no bar, and a panel of none says so; a figure of
    none is refused. An observed
    index grows with the cells of a group, so compare groups of different
    sizes by the Hill numbers at a common coverage or size, which the
    record's ``hill`` table holds.

    Tables: ``indices``, the record's indices; the record's ``group_keys``
    holds each group's labels.

    Raises:
        TypeError: If ``diversity`` is not a ``ClonalDiversity`` record,
            another argument has the wrong type, or the store does not name
            clones for figures or give the labels that the groups' colours
            range over.
        ValueError: If ``indices`` is empty, names an index twice or one that
            ``ds.clones.diversity`` does not measure, or the record holds no
            group, or no group of it has a value of any index named.
        KeyError: If ``theme`` names no theme, or the store does not hold
            the record's clones.
        RuntimeError: If the record's clones are incomplete, as a call that
            stopped before it finished leaves them.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    chosen = _diversity_indices(indices)
    if not isinstance(diversity, ClonalDiversity):
        raise TypeError("diversity must be the ClonalDiversity of ds.clones.diversity")
    table = diversity.indices
    if table.empty:
        raise ValueError("The diversity measures no group, so there is nothing to plot")
    if not np.isfinite(table[list(chosen)].to_numpy(dtype=np.float64)).any():
        raise ValueError(
            f"No group has a value of {', '.join(chosen)}, so there is nothing to plot"
        )
    names = clone_names(store, diversity.clones)
    plotted = table_groups(store, diversity)
    colors = plotted.scale(table["group"].tolist())
    titles = _index_titles(names)
    legend_panel = chosen[min(_SMALL_MULTIPLES, len(chosen)) - 1]
    if figsize is None and target is None:
        width, height = _small_multiples(len(chosen))
        figsize, _ = figure_size(
            len(chosen),
            panel=(
                width,
                item_inches(len(table), per_item=0.25, base=1.0, minimum=height),
            ),
            columns=_SMALL_MULTIPLES,
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(chosen), figsize=figsize, n_columns=_SMALL_MULTIPLES
        )
        drawn = None
        for index in chosen:
            ax = axes[index]
            found = draw_bars(
                ax,
                table,
                item="group",
                value=index,
                colors=colors,
                value_label="",
                title=titles[index],
                legend_title=plotted.title if index == legend_panel else None,
                show_legend=show_legend,
            )
            if index == legend_panel:
                drawn = found
            if not np.isfinite(table[index].to_numpy(dtype=np.float64)).any():
                ax.text(
                    0.5,
                    0.5,
                    "Not defined for any group",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize="small",
                    color=foreground_color(theme),
                )
    return finish_plot(
        figure,
        axes,
        owns,
        name="clonal_diversity",
        theme=theme,
        tables={"indices": table},
        drawn=[drawn],
        inputs={"clones": diversity.clones},
        settings={"grouping": plotted.title, "indices": list(chosen)},
        n_cells=int(table["n_cells"].sum()),
        show=show,
    )


def _matrix_figure_size(
    n_groups: int, *, extra_height: float = 0.0
) -> tuple[float, float]:
    """The default size of a heatmap of groups by groups, beside its colour bar.

    Its side grows by 0.55 inches a group from at least 4 and stops where
    the colour bar still fits on the page.
    """
    side = min(
        item_inches(n_groups, per_item=0.55, base=2.5, minimum=4.0),
        MAX_FIGURE_WIDTH_INCHES - _COLORBAR_INCHES,
    )
    size, _ = figure_size(1, panel=(side + _COLORBAR_INCHES, side + extra_height))
    return size


@close_figures_on_error
def clone_overlap(
    store: Any,
    overlap: CloneOverlap,
    *,
    index: str = "morisita_horn",
    color_scale: ColorScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot an overlap index between every pair of groups as a heatmap.

    ``overlap`` is the result of ``ds.clones.overlap``, which sets the
    groups. ``index`` is one of its indices: ``"shared"``, ``"jaccard"``,
    ``"sorensen"``, ``"overlap"``, ``"cosine"``, ``"morisita_horn"`` (the
    default) and ``"abundance_cosine"``. The diagonal is left blank so that
    the colours span the pairs, and pairs that were not compared take the
    scale's missing colour; the ``matrix`` table holds each group's clones
    on its diagonal for ``"shared"`` and 1 otherwise. ``color_scale`` sets
    the colour map and limits, viridis over the pairs' values by default,
    so two figures can share them.

    Tables: ``matrix`` (groups by groups) and ``pairs``, the record's pairs.

    Raises:
        TypeError: If ``overlap`` is not a ``CloneOverlap`` record,
            ``color_scale`` is not a ``ColorScale``, another argument has the
            wrong type, or the store does not name clones for figures.
        ValueError: If ``index`` is not an overlap index, or the record
            compares no pair.
        KeyError: If ``theme`` names no theme, ``color_scale`` names no
            colour map, or the store does not hold the record's clones.
        RuntimeError: If the record's clones are incomplete, as a call that
            stopped before it finished leaves them.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if not isinstance(overlap, CloneOverlap):
        raise TypeError("overlap must be the CloneOverlap of ds.clones.overlap")
    if index not in OVERLAP_INDICES:
        raise ValueError(f"index must be one of {', '.join(OVERLAP_INDICES)}")
    color_scale = _color_scale(color_scale)
    if overlap.pairs.empty:
        raise ValueError(
            "The overlap compares no pair of groups, so there is nothing to plot"
        )
    names = list(overlap.group_sizes["group"])
    matrix = pair_matrix(
        overlap.pairs,
        names,
        index,
        diagonal=(
            overlap.group_sizes["n_clones"].to_numpy(dtype=np.float64)
            if index == "shared"
            else 1.0
        ),
    )
    title = _overlap_titles(clone_names(store, overlap.clones))[index]
    if figsize is None and target is None:
        figsize = _matrix_figure_size(len(names))
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["matrix"], figsize=figsize, n_columns=1
        )
        drawn = draw_matrix(
            axes["matrix"],
            matrix,
            color_scale=color_scale or ColorScale(cmap="viridis"),
            value_label=title,
            blank_diagonal=True,
            value_format=".0f" if index == "shared" else ".3g",
            show_legend=show_legend,
        )
        axes["matrix"].set_title(title)
    return finish_plot(
        figure,
        axes,
        owns,
        name="clone_overlap",
        theme=theme,
        tables={"matrix": matrix, "pairs": overlap.pairs},
        drawn=[drawn],
        inputs={"clones": overlap.clones},
        settings={
            "grouping": table_groups(store, overlap).title,
            "index": index,
            "groups": names,
        },
        n_cells=int(overlap.group_sizes["n_cells"].sum()),
        show=show,
    )


@close_figures_on_error
def clone_tracking(
    store: Any,
    tracking: CloneTracking,
    *,
    order: Sequence[str] | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the fraction of cells of chosen clones across groups of cells.

    ``tracking`` is the result of ``ds.clones.track``, which chooses the
    clones and sets the groups: the ``top_n`` clones with the most cells in
    each group, or the clones it was given. Each clone draws one line across
    the groups, such as time points or cell types, of its fraction of each
    group's cells with a clone. A line is labelled with its clone's ID, or
    for a clonotype with its representative's junction, such as ``CASSA``,
    and also its number where two of the clonotypes followed share a
    junction, such as ``CASSA 1`` and ``CASSA 3``, as ``clone_abundance``
    labels the largest clones. Groups are placed by their labels in the
    ``group_by`` columns and the ``groups`` artifact, joined by commas;
    ``order`` lists those names from left to right, or values whose text is
    the name, such as time points as integers, and by default the names
    follow their natural order. The legend lists the clones, the largest
    first.

    Tables: ``tracking``, the record's ``table`` with each clone's ``label``
    after its ``clone`` (``clone``, ``label``, ``group``, ``n_cells`` and
    ``fraction``); the record's ``group_keys`` holds each group's labels.

    Raises:
        TypeError: If ``tracking`` is not a ``CloneTracking`` record,
            ``order`` is not a sequence of group names, another argument has
            the wrong type, or the store does not name or label clones for
            figures.
        ValueError: If ``order`` does not list the groups, or the record
            follows no clone.
        KeyError: If the store does not hold the record's clones.
        RuntimeError: If the record's clones are incomplete, as a call that
            stopped before it finished leaves them.
        MemoryError: If reading the labels of the clones followed does not
            fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if order is not None:
        if isinstance(order, np.ndarray) and order.ndim == 1:
            order = order.tolist()
        if isinstance(order, str | bytes) or not isinstance(order, Sequence):
            raise TypeError("order must be a sequence of group names")
    if not isinstance(tracking, CloneTracking):
        raise TypeError("tracking must be the CloneTracking of ds.clones.track")
    if not tracking.tracked:
        raise ValueError("The tracking follows no clone, so there is nothing to plot")
    names = clone_names(store, tracking.clones)
    # Every followed clone is labelled at once, so that the junctions which
    # two followed clonotypes share are found among the lines of this figure.
    line_labels = dict(
        zip(tracking.tracked, names.labels(tracking.tracked), strict=True)
    )
    cells = tracking.table.copy(deep=False)
    cells.insert(
        1,
        "label",
        np.asarray(
            [line_labels[clone] for clone in cells["clone"].tolist()], dtype=object
        ),
    )
    # Groups that differ only in labels that the clones keep apart share a
    # position, named by their labels in the grouping columns alone: the
    # group_by columns and the label artifact's kind, the first key columns.
    group_keys = tracking.group_keys
    n_grouping = len(tracking.group_by) + (tracking.groups is not None)
    group_names = [str(group) for group in group_keys.index.tolist()]
    position_of = dict(zip(group_names, group_names, strict=True))
    if n_grouping:
        grouping = group_keys.iloc[:, :n_grouping].astype(str)
        position_of = dict(
            zip(group_names, grouping.agg(", ".join, axis=1).tolist(), strict=True)
        )
    placed = cells.assign(
        position=np.asarray(
            [position_of[group] for group in cells["group"].tolist()], dtype=object
        )
    )
    present = sort_categories(pd.unique(placed["position"]))
    if order is None:
        positions = list(present)
    else:
        # Groups are placed by name, so an order of numbers names them by
        # their text.
        positions = [str(name) for name in order]
        if sorted(positions) != sorted(map(str, present)):
            raise ValueError(f"order must list the groups {present}")
    # The clones' colours, in the order they were chosen, the largest first.
    labels = [line_labels[clone] for clone in tracking.tracked]
    colors = CategoricalScale(
        order=tuple(labels),
        palette=dict(zip(labels, palette_for_n(len(labels)), strict=True)),
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            1,
            panel=(
                item_inches(len(positions), per_item=0.7, base=4.0, minimum=6.0),
                4.2,
            ),
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["tracking"], figsize=figsize, n_columns=1
        )
        drawn = draw_lines(
            axes["tracking"],
            placed,
            x="position",
            y="fraction",
            series="label",
            colors=colors,
            positions=positions,
            markers=True,
            y_label="Fraction of the group's cells",
            title=f"{names.plural.capitalize()} across groups",
            legend_title=names.singular.capitalize(),
            show_legend=show_legend,
        )
    return finish_plot(
        figure,
        axes,
        owns,
        name="clone_tracking",
        theme=theme,
        tables={"tracking": cells},
        drawn=[drawn],
        inputs={"clones": tracking.clones},
        settings={
            "grouping": table_groups(store, tracking).title,
            "tracked": list(tracking.tracked),
            "order": positions,
        },
        n_cells=int(cells["n_cells"].sum()),
        show=show,
    )


@close_figures_on_error
def clone_coupling(
    store: Any,
    coupling: CloneCoupling,
    *,
    value: str = "z_score",
    color_scale: ColorScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the fate coupling of every pair of groups as a heatmap with its dendrogram.

    ``coupling`` is the result of ``ds.clones.coupling``. ``value`` is
    ``"z_score"`` (the default), the coupling's distance from its null in
    null standard deviations, or ``"coupling"``, the method's value. The
    groups follow the result's average-linkage order, whose dendrogram is
    drawn above the matrix. The diagonal is left blank, and pairs that were
    not compared and z-scores that are not finite take the scale's missing
    colour; a coupling measured without permutations has no z-scores and is
    refused. By default z-scores, and couplings below 0, take a scale centred
    on 0, from minus to plus their largest size, and other couplings
    viridis; ``color_scale`` sets the map and limits instead, so two figures
    can share them.

    Tables: ``matrix`` (groups by groups, in the drawn order) and ``pairs``.

    Raises:
        TypeError: If ``coupling`` is not a ``CloneCoupling`` record,
            ``color_scale`` is not a ``ColorScale``, another argument has the
            wrong type, or the store does not name clones for figures.
        ValueError: If ``value`` is unknown, or names z-scores of a
            coupling measured without permutations, the record compares no
            pair, or the coupling's order does not follow its linkage.
        KeyError: If ``theme`` names no theme, ``color_scale`` names no
            colour map, or the store does not hold the record's clones.
        RuntimeError: If the record's clones are incomplete, as a call that
            stopped before it finished leaves them.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if not isinstance(coupling, CloneCoupling):
        raise TypeError("coupling must be the CloneCoupling of ds.clones.coupling")
    if value not in _COUPLING_VALUES:
        raise ValueError(f"value must be one of {', '.join(_COUPLING_VALUES)}")
    color_scale = _color_scale(color_scale)
    if value == "z_score" and not coupling.n_permutations:
        raise ValueError(
            "The coupling was measured without permutations, so it has no "
            "z-scores; plot value='coupling', or measure it with n_permutations"
        )
    if coupling.pairs.empty:
        raise ValueError(
            "The coupling compares no pair of groups, so there is nothing to plot"
        )
    names = list(coupling.order)
    matrix = pair_matrix(coupling.pairs, names, value)
    values = matrix.to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if color_scale is None:
        if value == "z_score" or bool((finite < 0).any()):
            limit = float(np.abs(finite).max()) if finite.size else 0.0
            limit = limit if limit > 0 else 1.0
            color_scale = ColorScale(
                cmap="RdBu_r", vmin=-limit, vmax=limit, vcenter=0.0
            )
        else:
            color_scale = ColorScale(cmap="viridis")
    title = _coupling_titles(clone_names(store, coupling.clones)).get(
        coupling.method, str(coupling.method)
    )
    tree = _ordered_linkage(coupling)
    if figsize is None and target is None:
        figsize = _matrix_figure_size(len(names), extra_height=0.8)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["matrix"], figsize=figsize, n_columns=1
        )
        ax = axes["matrix"]
        drawn = draw_matrix(
            ax,
            matrix,
            color_scale=color_scale,
            value_label="Z-score of the coupling" if value == "z_score" else title,
            blank_diagonal=True,
            value_format=".2g" if value == "z_score" else ".3g",
            show_legend=show_legend,
        )
        ax.set_title(f"{title}, z-score" if value == "z_score" else title, y=1.18)
        axes["dendrogram"] = draw_dendrogram(
            ax, tree, side="top", color=foreground_color(theme)
        )
    return finish_plot(
        figure,
        axes,
        owns,
        name="clone_coupling",
        theme=theme,
        tables={"matrix": matrix, "pairs": coupling.pairs},
        drawn=[drawn],
        inputs={"clones": coupling.clones},
        settings={
            "grouping": table_groups(store, coupling).title,
            "method": coupling.method,
            "value": value,
            "groups": names,
        },
        n_cells=int(coupling.group_sizes["n_cells"].sum()),
        show=show,
    )


def _ordered_linkage(coupling: Any) -> np.ndarray:
    """The coupling's tree, checked to place its leaves in the drawn order."""
    from scipy.cluster.hierarchy import leaves_list

    tree = np.asarray(coupling.linkage, dtype=np.float64)
    names = coupling.group_sizes["group"].to_numpy(dtype=object)
    if [str(name) for name in names[leaves_list(tree)].tolist()] != list(
        coupling.order
    ):
        raise ValueError("The coupling's order does not follow its linkage")
    return tree


@close_figures_on_error
def clone_fate_bias(
    store: Any,
    bias: ArtifactRef,
    *,
    max_clones: int = 40,
    normalize: bool = False,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the fates of the clones of early cells, each clone's late cells by fate.

    ``bias`` is a ``clone_fate_bias`` artifact, such as one from
    ``ds.clones.fate_bias``. The plot draws the ``max_clones`` clones with
    the most late cells, ties to the larger clone overall, grouped by their
    dominant fate, the fate that holds the most of their late cells, in the
    order of the fates, and within a fate from the most late cells to the
    fewest; clones whose late cells tie between fates come last. The lower
    panel stacks each clone's late cells by fate, the first fate on top, or
    their fractions with ``normalize=True``, and the upper panel draws its
    early cells. The fates take the colours of their groups, as every
    figure grouped by the same columns colours them, and a title that drops
    clones says how many it draws of how many. Clones are labelled as
    ``clone_abundance`` labels them: by their IDs, and clonotypes by their
    representative's junction, with their number where two drawn share one.

    Tables: ``clones``, one row per clone drawn, in the drawn order
    (``clone``, ``label``, ``fate``, the dominant fate, missing for a tie,
    ``n_early_cells``, ``n_late_cells``), and ``fates``, one row per clone
    drawn and fate, zeros included (``clone``, ``label``, ``fate``,
    ``n_cells``, the clone's late cells in the fate). Fates are values of the
    tables, so a fate may take any name.

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load or name clones for figures, or give the labels that the
            fates' colours range over.
        ValueError: If ``bias`` is not a ``clone_fate_bias`` artifact, or no
            clone of the early cells has late cells.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact, or the labels of the clones
            drawn, does not fit the memory budget.
    """
    normalize = bool_argument(normalize, "normalize")
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_clones = integer_argument(max_clones, "max_clones", minimum=1)
    loaded = store_method(store, "clones.load_fate_bias")(bias)
    fate_names = [str(name) for name in loaded.fates["fate"].tolist()]
    rows = loaded.clone_fates
    if rows.empty:
        raise ValueError(
            "No clone of the early cells has late cells, so there is nothing to plot"
        )
    clone_codes = np.asarray(rows["clone"].cat.codes, dtype=np.int64)
    fate_codes = np.asarray(rows["fate"].cat.codes, dtype=np.int64)
    # One row per clone, in the order of the clone codes, with its late cells
    # in each fate.
    codes, at = np.unique(clone_codes, return_inverse=True)
    cells = np.zeros((len(codes), len(fate_names)), dtype=np.int64)
    np.add.at(cells, (at.reshape(-1), fate_codes), rows["n_cells"].to_numpy(np.int64))
    early_cells = np.zeros(len(codes), dtype=np.int64)
    early_cells[at.reshape(-1)] = rows["n_early_cells"].to_numpy(np.int64)
    late_cells = cells.sum(axis=1)
    dominant = dominant_fates(cells)
    # The clones with the most late cells, ties to the lower code, the larger
    # clone overall, drawn by dominant fate and then by late cells.
    chosen = np.lexsort((codes, -late_cells))[:max_clones]
    rank = np.where(dominant[chosen] >= 0, dominant[chosen], len(fate_names))
    chosen = chosen[np.lexsort((codes[chosen], -late_cells[chosen], rank))]
    ids = rows["clone"].cat.categories.to_numpy(dtype=object)[codes[chosen]]
    shown_ids = [str(clone) for clone in ids.tolist()]
    names = clone_names(store, loaded.clones)
    shown_labels = names.labels(ids)
    fate_index = pd.Index(fate_names)
    table = pd.DataFrame(
        {
            "clone": shown_ids,
            "label": shown_labels,
            "fate": pd.Categorical.from_codes(dominant[chosen], categories=fate_index),
            "n_early_cells": early_cells[chosen],
            "n_late_cells": late_cells[chosen],
        }
    )
    # One row per clone drawn and fate, zeros included, so the fates are
    # values of the table, whatever their names.
    n_fates = len(fate_names)
    fates_table = pd.DataFrame(
        {
            "clone": np.repeat(np.asarray(shown_ids, dtype=object), n_fates),
            "label": np.repeat(np.asarray(shown_labels, dtype=object), n_fates),
            "fate": pd.Categorical.from_codes(
                np.tile(np.arange(n_fates), len(chosen)), categories=fate_index
            ),
            "n_cells": cells[chosen].reshape(-1),
        }
    )
    fates = table_groups(store, loaded)
    title = (
        f"The {len(chosen)} of {len(codes)} {names.plural} with the most late cells"
        if len(chosen) < len(codes)
        else f"Fates of the {names.plural} of early cells"
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            2,
            panel=(
                item_inches(len(chosen), per_item=0.22, base=3.5, minimum=6.0),
                2.5,
            ),
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["early", "fates"], figsize=figsize, n_columns=1
        )
        if owns:
            # The early cells take a quarter of the height.
            axes["early"].get_gridspec().set_height_ratios([1, 3])
        draw_bars(
            axes["early"],
            table,
            item="label",
            value="n_early_cells",
            colors=foreground_color(theme),
            horizontal=False,
            value_label="Early cells",
            title=title,
        )
        axes["early"].set_xticks([])
        ax = axes["fates"]
        drawn = draw_bars(
            ax,
            fates_table,
            item="label",
            value="n_cells",
            segment="fate",
            colors=fates.scale(fate_names),
            horizontal=False,
            normalize=normalize,
            value_label="Fraction of late cells" if normalize else "Late cells",
            legend_title=fates.title,
            show_legend=show_legend,
        )
        ax.set_xlabel(names.singular.capitalize())
        axes["early"].set_xlim(ax.get_xlim())
    return finish_plot(
        figure,
        axes,
        owns,
        name="clone_fate_bias",
        theme=theme,
        tables={"clones": table, "fates": fates_table},
        drawn=[drawn],
        inputs={"bias": bias, "clones": loaded.clones},
        settings={
            "grouping": fates.title,
            "max_clones": max_clones,
            "normalize": normalize,
        },
        n_cells=int(late_cells[chosen].sum()),
        show=show,
    )
