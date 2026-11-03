"""Figures of single-cell immune receptor repertoires."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..repertoire.chains import CHAIN_PAIRINGS, RECEPTOR_TYPES
from ..repertoire.usage import GeneUsage, Spectratype
from ..storage.refs import ArtifactRef
from ..utils.arrays import sort_categories
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._groups import cell_groups
from ._style import (
    LEGEND_SIDE_MAX_ENTRIES,
    apply_figure_chrome,
    capped_figsize,
    categorical_color_map,
    legend_side_columns,
    legend_side_rows,
    theme_context,
)
from ..utils.arguments import bool_argument, optional_bool_argument

__all__ = ["gene_pairing", "gene_usage", "receptor_qc", "spectratype"]

_QC_PANELS = ("receptor_type", "chain_pairing", "chains_per_cell")
_RECEPTOR_TYPE_COLORS = {
    "TCR": "#2b6cb0",
    "BCR": "#dd6b20",
    "ambiguous": "#805ad5",
    "multichain": "#c53030",
    "no IR": "#bdbdbd",
}
_CHAIN_PAIRING_COLORS = {
    "single pair": "#2f855a",
    "orphan VJ": "#9ae6b4",
    "orphan VDJ": "#68d391",
    "extra VJ": "#4299e1",
    "extra VDJ": "#2b6cb0",
    "two full chains": "#1a365d",
    "ambiguous": "#805ad5",
    "multichain": "#c53030",
    "no IR": "#bdbdbd",
}
_ARM_COLORS = {"VJ": "#2b6cb0", "VDJ": "#dd6b20"}
# Cells with this many chains of an arm or more share the last bar.
_MANY_CHAINS = 3
# The figures of gene usage and junction lengths draw one panel per locus, at
# most this many side by side.
_LOCUS_PANELS_PER_ROW = 2


def _store_method(store: Any, name: str, description: str | None = None) -> Any:
    """Return a method of the store's repertoire namespace, or raise."""
    method = getattr(getattr(store, "repertoire", None), name, None)
    if not callable(method):
        raise TypeError(f"store does not provide {description or name}")
    return method


def _category_table(
    values: np.ndarray,
    order: Sequence[str],
    groups: np.ndarray | None,
    names: Sequence[str],
) -> pd.DataFrame:
    """Count cells per category, one column per group, or ``n_cells`` without groups."""
    categories = pd.Index(list(order)).get_indexer(np.asarray(values, dtype=object))
    if groups is None:
        counts = np.bincount(categories[categories >= 0], minlength=len(order))
        return pd.DataFrame({"n_cells": counts}, index=pd.Index(order))
    keep = (categories >= 0) & (groups >= 0)
    matrix = np.bincount(
        categories[keep] * len(names) + groups[keep],
        minlength=len(order) * len(names),
    ).reshape(len(order), len(names))
    return pd.DataFrame(
        matrix, index=pd.Index(order), columns=pd.Index([str(name) for name in names])
    )


def _draw_categories(
    ax: Any,
    table: pd.DataFrame,
    colors: dict[str, str],
    *,
    grouped: bool,
    normalize: bool,
    title: str,
    show_legend: bool,
) -> None:
    """Draw one bar per category, or with a grouping one stacked bar per group.

    Without a grouping the table holds one column, ``n_cells``; with one, a
    column per group, which a group's name sets, ``n_cells`` among them.
    """
    if not grouped:
        shown = table[table["n_cells"] > 0]
        positions = np.arange(len(shown))[::-1]
        total = max(1, int(table["n_cells"].sum()))
        values = shown["n_cells"].to_numpy(dtype=np.float64)
        if normalize:
            values = values / total
        ax.barh(
            positions,
            values,
            color=[colors[name] for name in shown.index],
        )
        ax.set_yticks(positions, shown.index.tolist())
        ax.set_xlabel("Fraction of cells" if normalize else "Cells")
    else:
        totals = table.sum(axis=0).to_numpy(dtype=np.float64)
        positions = np.arange(table.shape[1])
        bottom = np.zeros(table.shape[1])
        for name, row in table.iterrows():
            heights = row.to_numpy(dtype=np.float64)
            if normalize:
                heights = np.divide(
                    heights, totals, out=np.zeros_like(heights), where=totals > 0
                )
            if not heights.any():
                continue
            ax.bar(
                positions, heights, bottom=bottom, color=colors[str(name)], label=name
            )
            bottom += heights
        ax.set_xticks(positions, table.columns.tolist(), rotation=90)
        ax.set_ylabel("Fraction of cells" if normalize else "Cells")
        if show_legend:
            ax.legend(
                frameon=False, fontsize="small", loc="upper left", bbox_to_anchor=(1, 1)
            )
    ax.set_title(title)


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
    VJ (TRA, TRG, IGK, IGL) and VDJ (TRB, TRD, IGH), with three or more in the
    last bar. Many ``"orphan VDJ"`` or ``"orphan VJ"`` cells point to chains
    lost in capture, and many ``"multichain"`` or ``"ambiguous"`` cells to
    doublets.

    ``group_by`` names cell metadata columns, one or several, such as the
    sample, and ``groups`` a cell-label artifact, such as a clustering; the
    groups are the combinations of their labels, as the clone summaries group
    cells, named by their labels joined by commas. With a grouping, the first
    two panels draw one stacked bar per group, of the fraction of its cells
    by default (``normalize``), and cells without a group are left out. A
    label artifact may hold fewer cells than the receptors.

    Tables: ``receptor_type`` and ``chain_pairing`` (cells per category, one
    column per group or ``n_cells``), and ``chains_per_cell`` (cells per
    number of chains of each arm, indexed by ``n_chains``).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``receptors`` is not an ``immune_receptors``
            artifact, a ``group_by`` column is missing or named twice, or
            two groups get one name.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact or grouping the cells does not
            fit the memory budget.
    """
    normalize = optional_bool_argument(normalize, "normalize")
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    loaded = _store_method(store, "load_receptors", "immune receptors")(
        receptors, fields=()
    )
    cells = loaded.cells
    labels, names, group_name = cell_groups(
        store, receptors, group_by=group_by, groups=groups
    )
    fractions = labels is not None if normalize is None else normalize
    keep = np.ones(len(cells), dtype=bool) if labels is None else labels >= 0
    receptor_type = _category_table(
        cells["receptor_type"].to_numpy(dtype=object), RECEPTOR_TYPES, labels, names
    )
    chain_pairing = _category_table(
        cells["chain_pairing"].to_numpy(dtype=object), CHAIN_PAIRINGS, labels, names
    )
    bins = [str(count) for count in range(_MANY_CHAINS)] + [f"{_MANY_CHAINS}+"]
    chains_per_cell = pd.DataFrame(
        {
            arm: np.bincount(
                np.minimum(cells[column].to_numpy(dtype=np.int64)[keep], _MANY_CHAINS),
                minlength=_MANY_CHAINS + 1,
            )
            for arm, column in (("VJ", "n_vj"), ("VDJ", "n_vdj"))
        },
        index=pd.Index(bins, name="n_chains"),
    )
    if figsize is None and target is None:
        width = (
            13.0 if labels is None else max(13.0, 4.0 + 0.5 * receptor_type.shape[1])
        )
        figsize = (width, 3.8 if labels is None else 4.4)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_QC_PANELS), figsize=figsize, n_columns=3
        )
        _draw_categories(
            axes["receptor_type"],
            receptor_type,
            _RECEPTOR_TYPE_COLORS,
            grouped=labels is not None,
            normalize=fractions,
            title="Receptor type",
            show_legend=show_legend,
        )
        _draw_categories(
            axes["chain_pairing"],
            chain_pairing,
            _CHAIN_PAIRING_COLORS,
            grouped=labels is not None,
            normalize=fractions,
            title="Chain pairing",
            show_legend=show_legend,
        )
        ax = axes["chains_per_cell"]
        positions = np.arange(len(bins))
        width = 0.4
        for offset, arm in ((-width / 2, "VJ"), (width / 2, "VDJ")):
            ax.bar(
                positions + offset,
                chains_per_cell[arm],
                width=width,
                color=_ARM_COLORS[arm],
                label=f"{arm} chains",
            )
        ax.set_xticks(positions, bins)
        ax.set_xlabel("Chains of the arm")
        ax.set_ylabel("Cells")
        ax.set_title("Chains per cell")
        if show_legend:
            ax.legend(frameon=False)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "receptor_type": receptor_type,
            "chain_pairing": chain_pairing,
            "chains_per_cell": chains_per_cell,
        },
        legends=(
            LegendSpec(kind="categorical", label="Receptor type"),
            LegendSpec(kind="categorical", label="Chain pairing"),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(keep.sum()),
            renderer="matplotlib",
            notes=("immune_receptors",),
            extras={
                "immune_receptors": receptors.to_dict(),
                "grouping": group_name,
                "normalize": fractions,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _max_genes(value: Any) -> int:
    """Return a positive ``max_genes``, refusing booleans and other types."""
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or value < 1
    ):
        raise ValueError("max_genes must be a positive integer")
    return int(value)


def _locus_panels(table: pd.DataFrame, what: str) -> list[str]:
    """The loci of a table, in their natural order, or raise when it has none."""
    if table.empty:
        raise ValueError(
            f"No chain of a cell with a group in the slots has {what}, so there "
            "is nothing to plot"
        )
    return [str(locus) for locus in sort_categories(pd.unique(table["locus"]))]


def _panel_grid(
    loci: list[str], panel: tuple[float, float]
) -> tuple[int, tuple[float, float]]:
    """The columns of the panels of the loci and the figure's size, page-sized."""
    n_columns = min(len(loci), _LOCUS_PANELS_PER_ROW)
    n_rows = -(-len(loci) // n_columns)
    return n_columns, capped_figsize(panel[0] * n_columns, panel[1] * n_rows)


def _counted(store: Any, source: ArtifactRef, count: str) -> str:
    """What the counts count, as figures name them: cells, clonotypes or clusters."""
    if count != "clonotypes":
        return "cells"
    method = getattr(store, "_clone_figure_noun", None)
    if not callable(method):
        raise TypeError("store does not provide what figures call clonotypes")
    return str(method(source))


def _heatmap(
    ax: Any, matrix: pd.DataFrame, *, title: str, show_legend: bool, figure: Any
) -> None:
    """Draw a matrix of fractions with labelled rows and columns."""
    image = ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy(dtype=np.float64)),
        cmap="viridis",
        aspect="auto",
        vmin=0,
    )
    ax.set_xticks(
        np.arange(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=90
    )
    ax.set_yticks(
        np.arange(matrix.shape[0]), [str(r) for r in matrix.index], fontsize="small"
    )
    ax.set_title(title)
    if show_legend:
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction")


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
    TRB, and its ``max_genes`` genes with the most chains over all groups:
    for groups of ``group_by`` columns or a ``groups`` artifact, a heatmap of
    each gene's fraction of each group's chains of that locus, which names
    the group even where one group holds the locus; without, bars. Panels
    stand two to a row. With ``count="clonotypes"`` the bars count
    clonotypes, or clonotype clusters for clonotypes that a distance metric
    linked. The plot raises ``ValueError`` when ``usage`` counts several gene
    fields or no chain has a call of its field.

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
    max_genes = _max_genes(max_genes)
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
    grouped = bool(usage.group_by) or usage.groups is not None
    by_locus = dict(tuple(table.groupby("locus", sort=False)))
    n_columns, size = _panel_grid(
        loci,
        (
            max(5.0, 3.0 + 0.4 * table["group"].nunique()),
            max(4.0, 0.22 * max_genes + 1.5),
        ),
    )
    if figsize is None and target is None:
        figsize = size
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=loci, figsize=figsize, n_columns=n_columns
        )
        for locus in loci:
            ax = axes[locus]
            rows = by_locus[locus]
            totals = (
                rows.groupby(gene)[f"n_{count}"]
                .sum()
                .sort_values(ascending=False, kind="stable")
            )
            shown = list(totals.index[:max_genes])
            wide = rows.pivot_table(
                index=gene, columns="group", values="fraction", fill_value=0
            ).reindex(index=shown)
            wide = wide.reindex(columns=sort_categories(wide.columns))
            if not grouped:
                positions = np.arange(len(shown))[::-1]
                ax.barh(positions, wide.iloc[:, 0].to_numpy(), color="#2b6cb0")
                ax.set_yticks(positions, shown, fontsize="small")
                ax.set_xlabel(f"Fraction of {counted}")
                ax.set_title(locus)
            else:
                _heatmap(ax, wide, title=locus, show_legend=show_legend, figure=figure)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"usage": table},
        legends=(LegendSpec(kind="continuous", label="Fraction"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(usage.n_cells),
            renderer="matplotlib",
            notes=(source.kind,),
            extras={
                "source": source.to_dict(),
                "gene": gene,
                "level": usage.level,
                "slots": list(usage.slots),
                "count": count,
                "max_genes": max_genes,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


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
    ``max_genes`` J genes with the most chains. Panels stand two to a row.
    The plot raises ``ValueError`` when ``usage`` counts other gene fields or
    groups of cells, or no chain has both calls.

    Tables: ``pairs``, the record's rows.

    Raises:
        TypeError: If ``usage`` is not a ``GeneUsage`` record, or another
            argument has the wrong type.
        ValueError: If ``usage`` counts other gene fields or groups of
            cells, or no chain has both calls.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_genes = _max_genes(max_genes)
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
    n_columns, size = _panel_grid(loci, (6.5, max(4.0, 0.25 * max_genes + 2.0)))
    if figsize is None and target is None:
        figsize = size
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=loci, figsize=figsize, n_columns=n_columns
        )
        by_locus = dict(tuple(table.groupby("locus", sort=False)))
        for locus in loci:
            rows = by_locus[locus]
            v_genes, j_genes = (
                rows.groupby(field)[f"n_{count}"]
                .sum()
                .sort_values(ascending=False, kind="stable")
                .index[:max_genes]
                for field in ("v_call", "j_call")
            )
            wide = rows.pivot_table(
                index="v_call", columns="j_call", values="fraction", fill_value=0
            ).reindex(index=list(v_genes), columns=sort_categories(j_genes))
            _heatmap(
                axes[locus], wide, title=locus, show_legend=show_legend, figure=figure
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"pairs": table},
        legends=(LegendSpec(kind="continuous", label="Fraction"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(usage.n_cells),
            renderer="matplotlib",
            notes=(usage.source.kind,),
            extras={
                "source": usage.source.to_dict(),
                "level": usage.level,
                "slots": list(usage.slots),
                "count": count,
                "max_genes": max_genes,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


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
    length, one line per group, or bars for one group; the legend names the
    first 80 groups. Panels stand two to a row. With ``count="clonotypes"``
    the fractions count clonotypes, or clonotype clusters for clonotypes that
    a distance metric linked. The plot raises ``ValueError`` when no chain
    has a junction.

    Tables: ``spectratype``, the record's rows.

    Raises:
        TypeError: If ``spectratype`` is not a ``Spectratype`` record, or
            another argument has the wrong type.
        ValueError: If no chain has a junction.
        KeyError: If the store does not hold the clonotypes that
            ``spectratype`` counts.
        RuntimeError: If those clonotypes are incomplete, as a call that
            stopped before it finished leaves them.
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
    group_order = sort_categories(pd.unique(table["group"]))
    colors = categorical_color_map(list(group_order))
    by_line = dict(tuple(table.groupby(["locus", "group"], sort=False)))
    n_columns, size = _panel_grid(loci, (4.5, 3.8))
    if figsize is None and target is None:
        figsize = size
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=loci, figsize=figsize, n_columns=n_columns
        )
        for locus in loci:
            ax = axes[locus]
            for name in group_order:
                rows = by_line.get((locus, name), table.iloc[:0])
                if len(group_order) == 1:
                    ax.bar(rows["length"], rows["fraction"], color="#2b6cb0")
                else:
                    ax.plot(
                        rows["length"],
                        rows["fraction"],
                        marker="o",
                        markersize=3,
                        linewidth=1.2,
                        color=colors[name],
                        label=name,
                    )
            ax.set_xlabel(
                "Junction length (residues)"
                if sequence == "aa"
                else "Junction length (nucleotides)"
            )
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.set_ylabel(f"Fraction of {counted}")
            ax.set_title(locus)
        if show_legend and len(group_order) > 1:
            # The first groups, in columns that fit the figure's height.
            ax = axes[loci[-1]]
            lines = ax.get_lines()
            shown = min(len(lines), LEGEND_SIDE_MAX_ENTRIES)
            rows = legend_side_rows(float(figure.get_size_inches()[1]), title=True)
            ax.legend(
                handles=lines[:shown],
                frameon=False,
                fontsize="small",
                title=(
                    "Group"
                    if shown == len(lines)
                    else f"Group ({shown} of {len(lines)})"
                ),
                title_fontsize="small",
                loc="upper left",
                bbox_to_anchor=(1, 1),
                ncols=legend_side_columns(shown, rows=rows),
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"spectratype": table},
        legends=(LegendSpec(kind="categorical", label="Group"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(spectratype.n_cells),
            renderer="matplotlib",
            notes=(source.kind,),
            extras={
                "source": source.to_dict(),
                "sequence": sequence,
                "slots": list(spectratype.slots),
                "count": count,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
