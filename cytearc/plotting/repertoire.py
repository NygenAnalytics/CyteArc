"""Figures of single-cell immune receptor repertoires."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..repertoire.chains import CHAIN_PAIRINGS, RECEPTOR_TYPES
from ..repertoire.usage import GeneUsage, Spectratype
from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument, optional_bool_argument
from ..utils.arrays import sort_categories
from ._charts import capped_counts, draw_bars, draw_lines, draw_matrix
from ._clone_names import clone_names
from ._contracts import CategoricalScale, ColorScale
from ._data import store_method
from ._figure import (
    Drawn,
    PlotResult,
    close_figures_on_error,
    finish_plot,
    normalize_axes_target,
)
from ._groups import plot_groups, table_groups
from ._style import (
    CHAIN_ARM_SCALE,
    CHAIN_PAIRING_SCALE,
    LOCUS_SCALE,
    MAX_FIGURE_WIDTH_INCHES,
    RECEPTOR_TYPE_SCALE,
    figure_size,
    item_inches,
    resolve_category_scale,
    resolve_color_limits,
    theme_context,
)

__all__ = ["gene_pairing", "gene_usage", "receptor_qc", "spectratype"]

_QC_PANELS = ("receptor_type", "chain_pairing", "chains_per_cell")
# The panels of receptor_qc stand in one column at the page width, each this
# many inches tall.
_QC_PANEL_HEIGHT = 2.4
# Cells with this many chains of an arm or more share the last bar.
_MANY_CHAINS = 3
# Locus panels of bars and lines fill rows of three at the page width, and
# heatmaps rows of two.
_LOCUS_PANELS_PER_ROW = 3
_LOCUS_HEATMAPS_PER_ROW = 2
_LOCUS_PANEL_HEIGHT = 2.4
# What the gene fields and levels of gene usage are called on an axis.
_SEGMENTS = {"v_call": "V", "d_call": "D", "j_call": "J", "c_call": "C"}


def _category_counts(
    values: np.ndarray, order: Sequence[str], groups: np.ndarray, n_groups: int
) -> np.ndarray:
    """Count cells per group and category: one row per group, one column per category."""
    categories = pd.Index(list(order)).get_indexer(np.asarray(values, dtype=object))
    keep = (categories >= 0) & (groups >= 0)
    return np.bincount(
        groups[keep] * len(order) + categories[keep],
        minlength=n_groups * len(order),
    ).reshape(n_groups, len(order))


def _category_table(
    counts: np.ndarray, names: Sequence[str], order: Sequence[str], column: str
) -> pd.DataFrame:
    """One row per group and category, zeros included: ``group``, the category, ``n_cells``."""
    return pd.DataFrame(
        {
            "group": np.repeat(np.asarray(list(names), dtype=object), len(order)),
            column: np.tile(np.asarray(list(order), dtype=object), len(names)),
            "n_cells": counts.reshape(-1).astype(np.int64),
        }
    )


def _draw_categories(
    ax: Any,
    table: pd.DataFrame,
    column: str,
    vocabulary: CategoricalScale,
    *,
    grouped: bool,
    normalize: bool,
    title: str,
    show_legend: bool,
) -> Drawn | None:
    """Draw the cells of each category of a vocabulary that holds cells.

    Without a grouping, each category draws one bar, named on its axis, of
    its cells or their fraction of every cell; with one, each group draws
    one vertical bar stacked by category, under a legend of the categories.
    """
    held = table.groupby(column, sort=False)["n_cells"].sum()
    present = [name for name, cells in held.items() if cells > 0]
    colors = resolve_category_scale(present, vocabulary)
    rows = table.loc[table[column].isin(present)]
    value_label = "Fraction of cells" if normalize else "Cells"
    if not grouped:
        cells = rows["n_cells"].to_numpy(dtype=np.float64)
        bars = pd.DataFrame(
            {
                column: rows[column].to_numpy(dtype=object),
                "value": cells / cells.sum() if normalize else cells,
            }
        )
        return draw_bars(
            ax,
            bars,
            item=column,
            value="value",
            colors=colors,
            value_label=value_label,
            title=title,
        )
    return draw_bars(
        ax,
        rows,
        item="group",
        value="n_cells",
        segment=column,
        colors=colors,
        horizontal=False,
        normalize=normalize,
        value_label=value_label,
        title=title,
        legend_title=title,
        show_legend=show_legend,
    )


@close_figures_on_error
def receptor_qc(
    store: Any,
    receptors: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    normalize: bool | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the receptor types, chain pairings and chains per cell of imported receptors.

    ``receptors`` is an ``immune_receptors`` artifact, such as one from
    ``ds.repertoire.import_receptors``. The first panel counts cells by
    receptor type (TCR, BCR, ambiguous, multichain, no IR), the second by chain
    pairing, and the third counts cells by their number of chains of each arm,
    VJ (TRA, TRG, IGK, IGL) and VDJ (TRB, TRD, IGH), side by side, with three
    or more in the last bar. Many ``"orphan VDJ"`` or ``"orphan VJ"`` cells
    point to chains lost in capture, and many ``"multichain"`` or
    ``"ambiguous"`` cells to doublets. The panels stand in one column, the
    categories that hold cells in the colours of their vocabulary.

    ``group_by`` names cell metadata columns, one or several, such as the
    sample, and ``groups`` a cell-label artifact, such as a clustering; the
    groups are the combinations of their labels, as the clone summaries group
    cells, named by their labels joined by commas. With a grouping, the first
    two panels draw one bar per group, stacked by category, the first on
    top as its legend lists them, of the fraction of the group's cells by
    default (``normalize``), and cells without a group are left out. A
    label artifact may hold fewer cells than the receptors.

    Tables: ``receptor_type`` and ``chain_pairing``, one row per group and
    category, zeros included (``group``, the category in a column named like
    the table, ``n_cells``), with every cell in the group ``"all cells"``
    without a grouping, so a group may take any name; and
    ``chains_per_cell`` (cells per number of chains of each arm, indexed by
    ``n_chains``).

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load immune receptors or group cells for figures.
        ValueError: If ``receptors`` is not an ``immune_receptors``
            artifact, a ``group_by`` column is missing or named twice, two
            groups get one name, or no cell, or no cell with a group, is
            left to draw.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact or grouping the cells does not
            fit the memory budget.
    """
    normalize = optional_bool_argument(normalize, "normalize")
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    loaded = store_method(store, "repertoire.load_receptors", "immune receptors")(
        receptors, fields=()
    )
    cells = loaded.cells
    plotted = plot_groups(
        store, receptors, group_by=group_by, groups=groups, n_cells=len(cells)
    )
    labels, names = plotted.codes, list(plotted.names)
    grouped = bool(plotted.columns)
    fractions = grouped if normalize is None else normalize
    keep = labels >= 0
    if not keep.any():
        raise ValueError(
            "No cell has a group, so there is nothing to plot"
            if len(cells)
            else "The receptors hold no cell, so there is nothing to plot"
        )
    tables = {
        "receptor_type": _category_table(
            _category_counts(
                cells["receptor_type"].to_numpy(dtype=object),
                RECEPTOR_TYPES,
                labels,
                len(names),
            ),
            names,
            RECEPTOR_TYPES,
            "receptor_type",
        ),
        "chain_pairing": _category_table(
            _category_counts(
                cells["chain_pairing"].to_numpy(dtype=object),
                CHAIN_PAIRINGS,
                labels,
                len(names),
            ),
            names,
            CHAIN_PAIRINGS,
            "chain_pairing",
        ),
    }
    chains_per_cell = pd.DataFrame(
        {
            arm: capped_counts(
                cells[column].to_numpy(dtype=np.int64)[keep], _MANY_CHAINS
            )
            for arm, column in (("VJ", "n_vj"), ("VDJ", "n_vdj"))
        }
    )
    chains_per_cell.index.name = "n_chains"
    tables["chains_per_cell"] = chains_per_cell
    arms = pd.DataFrame(
        {
            "n_chains": np.tile(chains_per_cell.index.to_numpy(dtype=object), 2),
            "arm": np.repeat(
                np.asarray(["VJ", "VDJ"], dtype=object), len(chains_per_cell)
            ),
            "n_cells": chains_per_cell[["VJ", "VDJ"]]
            .to_numpy(dtype=np.int64)
            .T.reshape(-1),
        }
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            len(_QC_PANELS), panel=(MAX_FIGURE_WIDTH_INCHES, _QC_PANEL_HEIGHT)
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_QC_PANELS), figsize=figsize, n_columns=1
        )
        drawn = [
            _draw_categories(
                axes[column],
                tables[column],
                column,
                vocabulary,
                grouped=grouped,
                normalize=fractions,
                title=title,
                show_legend=show_legend,
            )
            for column, vocabulary, title in (
                ("receptor_type", RECEPTOR_TYPE_SCALE, "Receptor type"),
                ("chain_pairing", CHAIN_PAIRING_SCALE, "Chain pairing"),
            )
        ]
        drawn.append(
            draw_bars(
                axes["chains_per_cell"],
                arms,
                item="n_chains",
                value="n_cells",
                segment="arm",
                colors=CHAIN_ARM_SCALE,
                horizontal=False,
                side_by_side=True,
                value_label="Cells",
                title="Chains per cell",
                legend_title="Arm",
                show_legend=show_legend,
            )
        )
        axes["chains_per_cell"].set_xlabel("Chains of the arm")
    return finish_plot(
        figure,
        axes,
        owns,
        name="receptor_qc",
        theme=theme,
        tables=tables,
        drawn=drawn,
        inputs={"receptors": receptors},
        settings={"grouping": plotted.title, "normalize": fractions},
        n_cells=int(keep.sum()),
        show=show,
    )


def _locus_panels(table: pd.DataFrame, what: str) -> list[str]:
    """The loci of a table, in their natural order, or raise when it has none."""
    if table.empty:
        raise ValueError(
            f"No chain of a cell with a group in the slots has {what}, so there "
            "is nothing to plot"
        )
    return [str(locus) for locus in sort_categories(pd.unique(table["locus"]))]


def _counted(store: Any, source: ArtifactRef, count: str) -> str:
    """What the counts count, as figures name them: cells, clonotypes or clusters."""
    if count != "clonotypes":
        return "cells"
    return clone_names(store, source).plural


def _ranked(rows: pd.DataFrame, field: str, count: str) -> pd.Index:
    """The values of ``field`` by their chains, most first, ties in name order."""
    return (
        rows.groupby(field)[f"n_{count}"]
        .sum()
        .sort_values(ascending=False, kind="stable")
        .index
    )


def _gene_axis(field: str, level: str, shown: int, held: int) -> str:
    """Name an axis of genes, saying how many of how many it draws when capped."""
    name = f"{_SEGMENTS[field]} {level}"
    return name if shown == held else f"{name} ({shown} of {held})"


def _shared_fractions(matrices: Sequence[pd.DataFrame]) -> ColorScale:
    """One colour scale of fractions from 0 for every locus panel of a figure."""
    values = np.concatenate(
        [matrix.to_numpy(dtype=np.float64).ravel() for matrix in matrices]
    )
    low, high = resolve_color_limits(values, ColorScale(cmap="viridis", vmin=0.0))
    return ColorScale(cmap="viridis", vmin=low, vmax=high)


def _locus_heatmaps(
    axes: dict[Any, Any],
    matrices: dict[str, pd.DataFrame],
    *,
    value_label: str,
    show_legend: bool,
) -> Drawn:
    """Draw a heatmap per locus under one colour scale, beside one colour bar."""
    scale = _shared_fractions(list(matrices.values()))
    loci = list(matrices)
    drawn: Drawn | None = None
    for locus in loci:
        drawn = draw_matrix(
            axes[locus],
            matrices[locus],
            color_scale=scale,
            value_label=value_label,
            value_format=".2f",
            show_legend=show_legend and locus == loci[-1],
            colorbar_beside=[axes[name] for name in loci],
        )
        axes[locus].set_title(locus)
    assert drawn is not None
    return drawn


@close_figures_on_error
def gene_usage(
    store: Any,
    usage: GeneUsage,
    *,
    max_genes: int = 30,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the gene usage of groups of cells, one panel per locus.

    ``usage`` is the result of ``ds.repertoire.gene_usage`` for one gene
    field, such as ``genes="j_call"``, which sets the groups, the level, the
    slots and what is counted. Each panel holds one locus, such as TRA or
    TRB, and its ``max_genes`` genes with the most chains over all groups,
    its axis saying how many of how many it draws when it leaves some out:
    for groups of ``group_by`` columns or a ``groups`` artifact, a heatmap of
    each gene's fraction of each group's chains of that locus, which names
    the group even where one group holds the locus, the panels of every
    locus sharing one colour bar from 0 to the largest fraction drawn, two
    panels to a row; without, bars in the locus's colour, three panels to a
    row. With ``count="clonotypes"`` the fractions count clonotypes, or
    clonotype clusters for clonotypes that a distance metric linked. The
    plot raises ``ValueError`` when ``usage`` counts several gene fields or
    no chain has a call of its field.

    Tables: ``usage``, the record's rows.

    Raises:
        TypeError: If ``usage`` is not a ``GeneUsage`` record, or another
            argument has the wrong type.
        ValueError: If ``usage`` counts several gene fields, or no chain has
            a call of its field.
        KeyError: If the store does not hold the clonotypes that ``usage``
            counts.
        RuntimeError: If those clonotypes are incomplete, as a call that
            stopped before it finished leaves them.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_genes = integer_argument(max_genes, "max_genes", minimum=1)
    if not isinstance(usage, GeneUsage):
        raise TypeError("usage must be the GeneUsage of ds.repertoire.gene_usage")
    if len(usage.genes) != 1:
        raise ValueError(
            f"usage counts the combinations of {', '.join(usage.genes)}; "
            "gene_usage draws one gene field, so pass the GeneUsage of one, "
            "and draw V and J genes together with gene_pairing"
        )
    gene = usage.genes[0]
    source, count = usage.source, usage.count
    table = usage.table
    loci = _locus_panels(table, f"a {gene}")
    counted = _counted(store, source, count)
    grouping = table_groups(store, usage).title
    grouped = bool(usage.group_by) or usage.groups is not None
    by_locus = dict(tuple(table.groupby("locus", sort=False)))
    ranked = {locus: _ranked(by_locus[locus], gene, count) for locus in loci}
    shown = {locus: list(ranked[locus][:max_genes]) for locus in loci}
    tallest = max(len(genes) for genes in shown.values())
    columns = _LOCUS_HEATMAPS_PER_ROW if grouped else _LOCUS_PANELS_PER_ROW
    if figsize is None and target is None:
        figsize, _ = figure_size(
            len(loci),
            panel=(
                MAX_FIGURE_WIDTH_INCHES / columns,
                item_inches(
                    tallest, per_item=0.22, base=1.2, minimum=_LOCUS_PANEL_HEIGHT
                ),
            ),
            columns=columns,
        )
    value_label = f"Fraction of {counted}"
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=loci, figsize=figsize, n_columns=columns
        )
        drawn: Drawn | None = None
        if grouped:
            matrices = {}
            for locus in loci:
                wide = by_locus[locus].pivot_table(
                    index=gene, columns="group", values="fraction", fill_value=0
                )
                matrices[locus] = wide.reindex(
                    index=shown[locus], columns=sort_categories(wide.columns)
                )
            drawn = _locus_heatmaps(
                axes, matrices, value_label=value_label, show_legend=show_legend
            )
        for locus in loci:
            ax = axes[locus]
            if not grouped:
                rows = by_locus[locus].set_index(gene).loc[shown[locus]]
                draw_bars(
                    ax,
                    rows.reset_index(),
                    item=gene,
                    value="fraction",
                    colors=(LOCUS_SCALE.palette or {})[locus],
                    value_label=value_label,
                    title=locus,
                )
            else:
                ax.set_xlabel(grouping or "")
            ax.set_ylabel(
                _gene_axis(gene, usage.level, len(shown[locus]), len(ranked[locus]))
            )
    return finish_plot(
        figure,
        axes,
        owns,
        name="gene_usage",
        theme=theme,
        tables={"usage": table},
        drawn=[drawn],
        inputs={"source": source},
        settings={
            "grouping": grouping,
            "gene": gene,
            "level": usage.level,
            "slots": list(usage.slots),
            "count": count,
            "max_genes": max_genes,
        },
        n_cells=int(usage.n_cells),
        show=show,
    )


@close_figures_on_error
def gene_pairing(
    store: Any,
    usage: GeneUsage,
    *,
    max_genes: int = 20,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how the V and J genes of each locus pair, one panel per locus.

    ``usage`` is the result of ``ds.repertoire.gene_usage`` for the pairs of
    ``genes=("v_call", "j_call")`` without groups, which sets the level, the
    slots, the cells and what is counted. Each panel draws the fraction of
    the locus's chains with each pair of its ``max_genes`` V genes and
    ``max_genes`` J genes with the most chains, its axes saying how many of
    how many they draw when they leave some out. The panels stand two to a
    row and share one colour bar, from 0 to the largest fraction drawn. The
    plot raises ``ValueError`` when ``usage`` counts other gene fields or
    groups of cells, or no chain has both calls.

    Tables: ``pairs``, the record's rows.

    Raises:
        TypeError: If ``usage`` is not a ``GeneUsage`` record, or another
            argument has the wrong type.
        ValueError: If ``usage`` counts other gene fields or groups of
            cells, or no chain has both calls.
        KeyError: If ``theme`` names no theme.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_genes = integer_argument(max_genes, "max_genes", minimum=1)
    if not isinstance(usage, GeneUsage):
        raise TypeError("usage must be the GeneUsage of ds.repertoire.gene_usage")
    if sorted(usage.genes) != ["j_call", "v_call"]:
        raise ValueError(
            f"usage counts {', '.join(usage.genes)}; gene_pairing draws the "
            "pairs of v_call and j_call, so pass the GeneUsage of "
            'genes=("v_call", "j_call")'
        )
    if usage.group_by or usage.groups is not None:
        raise ValueError(
            "usage counts the pairs within groups of cells; gene_pairing draws "
            "the pairs of all its cells, so pass a GeneUsage without group_by "
            "or groups, and choose the cells with cell_selection"
        )
    count = usage.count
    table = usage.table
    loci = _locus_panels(table, "a v_call and a j_call")
    counted = _counted(store, usage.source, count)
    by_locus = dict(tuple(table.groupby("locus", sort=False)))
    matrices: dict[str, pd.DataFrame] = {}
    held: dict[str, tuple[int, int]] = {}
    for locus in loci:
        rows = by_locus[locus]
        v_ranked, j_ranked = (
            _ranked(rows, field, count) for field in ("v_call", "j_call")
        )
        held[locus] = (len(v_ranked), len(j_ranked))
        wide = rows.pivot_table(
            index="v_call", columns="j_call", values="fraction", fill_value=0
        )
        matrices[locus] = wide.reindex(
            index=list(v_ranked[:max_genes]),
            columns=sort_categories(j_ranked[:max_genes]),
        )
    tallest = max(len(matrix) for matrix in matrices.values())
    if figsize is None and target is None:
        figsize, _ = figure_size(
            len(loci),
            panel=(
                MAX_FIGURE_WIDTH_INCHES / _LOCUS_HEATMAPS_PER_ROW,
                item_inches(
                    tallest, per_item=0.22, base=1.6, minimum=_LOCUS_PANEL_HEIGHT
                ),
            ),
            columns=_LOCUS_HEATMAPS_PER_ROW,
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=loci,
            figsize=figsize,
            n_columns=_LOCUS_HEATMAPS_PER_ROW,
        )
        drawn = _locus_heatmaps(
            axes,
            matrices,
            value_label=f"Fraction of {counted}",
            show_legend=show_legend,
        )
        for locus in loci:
            n_v, n_j = matrices[locus].shape
            axes[locus].set_ylabel(
                _gene_axis("v_call", usage.level, n_v, held[locus][0])
            )
            axes[locus].set_xlabel(
                _gene_axis("j_call", usage.level, n_j, held[locus][1])
            )
    return finish_plot(
        figure,
        axes,
        owns,
        name="gene_pairing",
        theme=theme,
        tables={"pairs": table},
        drawn=[drawn],
        inputs={"source": usage.source},
        settings={
            "level": usage.level,
            "slots": list(usage.slots),
            "count": count,
            "max_genes": max_genes,
        },
        n_cells=int(usage.n_cells),
        show=show,
    )


@close_figures_on_error
def spectratype(
    store: Any,
    spectratype: Spectratype,
    *,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the junction lengths of groups of cells, one panel per locus.

    ``spectratype`` is the result of ``ds.repertoire.spectratype``, which
    sets the groups, the sequence, the slots and what is counted. Each panel
    draws, for one locus, each group's fraction of chains at each junction
    length, one line per group in its colour, or bars in the locus's colour
    for one group. The panels stand three to a row, with one legend of the
    groups, titled by the grouping, beside the first row. With
    ``count="clonotypes"`` the fractions count clonotypes, or clonotype
    clusters for clonotypes that a distance metric linked. The plot raises
    ``ValueError`` when no chain has a junction.

    Tables: ``spectratype``, the record's rows.

    Raises:
        TypeError: If ``spectratype`` is not a ``Spectratype`` record, or
            another argument has the wrong type, or the store does not give
            the labels that the groups' colours range over.
        ValueError: If no chain has a junction.
        KeyError: If the store does not hold the clonotypes that
            ``spectratype`` counts.
        RuntimeError: If those clonotypes are incomplete, as a call that
            stopped before it finished leaves them.
        MemoryError: If reading the labels that the groups' colours range
            over does not fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if not isinstance(spectratype, Spectratype):
        raise TypeError(
            "spectratype must be the Spectratype of ds.repertoire.spectratype"
        )
    source, sequence, count = (
        spectratype.source,
        spectratype.sequence,
        spectratype.count,
    )
    table = spectratype.table
    loci = _locus_panels(table, "a junction")
    counted = _counted(store, source, count)
    plotted = table_groups(store, spectratype)
    held = set(table["group"].astype(str))
    group_order = [name for name in plotted.names if name in held]
    by_locus = dict(tuple(table.groupby("locus", sort=False)))
    legend_panel = loci[min(_LOCUS_PANELS_PER_ROW, len(loci)) - 1]
    if figsize is None and target is None:
        figsize, _ = figure_size(
            len(loci),
            panel=(
                MAX_FIGURE_WIDTH_INCHES / _LOCUS_PANELS_PER_ROW,
                _LOCUS_PANEL_HEIGHT,
            ),
            columns=_LOCUS_PANELS_PER_ROW,
        )
    x_label = (
        "Junction length (residues)"
        if sequence == "aa"
        else "Junction length (nucleotides)"
    )
    y_label = f"Fraction of {counted}"
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=loci,
            figsize=figsize,
            n_columns=_LOCUS_PANELS_PER_ROW,
        )
        drawn: Drawn | None = None
        colors = plotted.scale(group_order) if len(group_order) > 1 else None
        for locus in loci:
            ax = axes[locus]
            rows = by_locus[locus]
            if colors is None:
                draw_bars(
                    ax,
                    rows,
                    item="length",
                    value="fraction",
                    colors=(LOCUS_SCALE.palette or {})[locus],
                    horizontal=False,
                    numeric_items=True,
                    value_label=y_label,
                    title=locus,
                )
                ax.set_xlabel(x_label)
            else:
                found = draw_lines(
                    ax,
                    rows,
                    x="length",
                    y="fraction",
                    series="group",
                    colors=colors,
                    markers=True,
                    x_label=x_label,
                    y_label=y_label,
                    title=locus,
                    legend_title=plotted.title if locus == legend_panel else None,
                    show_legend=show_legend,
                )
                if locus == legend_panel:
                    drawn = found
            # Lengths are whole numbers, and a locus of one length ticks it.
            ax.xaxis.get_major_locator().set_params(integer=True, min_n_ticks=1)
    return finish_plot(
        figure,
        axes,
        owns,
        name="spectratype",
        theme=theme,
        tables={"spectratype": table},
        drawn=[drawn],
        inputs={"source": source},
        settings={
            "grouping": plotted.title,
            "sequence": sequence,
            "slots": list(spectratype.slots),
            "count": count,
        },
        n_cells=int(spectratype.n_cells),
        show=show,
    )
