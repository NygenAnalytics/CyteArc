"""Figures of single-cell immune receptor repertoires."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..metadata.selection import CellField, resolve_grouping
from ..repertoire.chains import CHAIN_PAIRINGS, RECEPTOR_TYPES
from ..repertoire.diversity import ORDERS
from ..repertoire.expansion import (
    DEFAULT_RANK_SPLITS,
    rank_window_labels,
    rank_windows,
)
from ..storage.artifacts import inspect_artifact
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
from ._style import (
    apply_figure_chrome,
    categorical_color_map,
    colormap_palette,
    theme_context,
)

__all__ = [
    "cell_groups",
    "clonal_expansion",
    "clonal_rarefaction",
    "clonotype_abundance",
    "receptor_qc",
]

_ORDER_TITLES = {
    0: "Clonotypes (order 0)",
    1: "Shannon diversity (order 1)",
    2: "Simpson diversity (order 2)",
}

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


def _load_receptors(store: Any, receptors: ArtifactRef) -> Any:
    """Load saved receptors through the store's narrow loader."""
    loader = getattr(getattr(store, "repertoire", None), "load_receptors", None)
    if not callable(loader):
        raise TypeError("store does not provide immune receptors")
    return loader(receptors, fields=())


def cell_groups(
    store: Any,
    artifact: ArtifactRef,
    cell_ids: pd.Index,
    *,
    group_by: str | None,
    groups: ArtifactRef | None,
) -> tuple[np.ndarray | None, str | None]:
    """Return the group label of each cell of a cell-aligned artifact.

    ``group_by`` names a cell metadata column and ``groups`` a cell-label
    artifact, such as a clustering; at most one may be given. A label artifact
    may hold fewer cells than ``artifact``: the cells it lacks have no group,
    and their label is None.

    Returns:
        The labels aligned to ``cell_ids``, or None without a grouping, and
        the grouping's name.
    """
    if group_by is not None and groups is not None:
        raise ValueError("Pass group_by or groups, not both")
    if group_by is None and groups is None:
        return None, None
    if group_by is not None:
        if not isinstance(group_by, str):
            raise TypeError("group_by must be the name of a cell metadata column")
        selection = inspect_artifact(store.zw, artifact).input_ref("cell_selection")
        resolved = resolve_grouping(
            store.zw,
            store.cells,
            CellField(group_by, kind="categorical"),
            cell_selection=selection,
        )
        name = group_by
    else:
        if not isinstance(groups, ArtifactRef):
            raise TypeError("groups must be an ArtifactRef")
        resolved = resolve_grouping(store.zw, store.cells, groups)
        name = groups.kind
    ids = np.asarray(store.cells.fetch_all("ids")).astype(str)[resolved.cell_idx]
    labels = np.asarray(resolved.labels, dtype=object).copy()
    if resolved.missing_mask is not None:
        labels[np.asarray(resolved.missing_mask, dtype=bool)] = None
    aligned = pd.Series(labels, index=pd.Index(ids)).reindex(cell_ids)
    values = aligned.to_numpy(dtype=object)
    values[pd.isna(aligned).to_numpy()] = None
    return values, name


def _category_table(
    values: np.ndarray,
    order: Sequence[str],
    labels: np.ndarray | None,
) -> pd.DataFrame:
    """Count cells per category, one column per group, or ``cells`` without groups."""
    if labels is None:
        counts = pd.Series(values).value_counts().reindex(list(order), fill_value=0)
        return pd.DataFrame({"cells": counts.to_numpy()}, index=pd.Index(order))
    keep = np.asarray([label is not None for label in labels], dtype=bool)
    present = sort_categories(pd.unique(labels[keep]))
    table = pd.crosstab(
        pd.Categorical(values[keep], categories=list(order)),
        pd.Categorical(labels[keep], categories=present),
        dropna=False,
    )
    table.index = pd.Index(order)
    table.columns = pd.Index([str(value) for value in present])
    return table


def _draw_categories(
    ax: Any,
    table: pd.DataFrame,
    colors: dict[str, str],
    *,
    normalize: bool,
    title: str,
    show_legend: bool,
) -> None:
    """Draw one bar per category, or one stacked bar per group."""
    if list(table.columns) == ["cells"]:
        shown = table[table["cells"] > 0]
        positions = np.arange(len(shown))[::-1]
        total = max(1, int(table["cells"].sum()))
        values = shown["cells"].to_numpy(dtype=np.float64)
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
    group_by: str | None = None,
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

    ``group_by`` names a cell metadata column, such as the sample, and
    ``groups`` a cell-label artifact, such as a clustering; with either, the
    first two panels draw one stacked bar per group, of the fraction of its
    cells by default (``normalize``), and cells without a group are left out.
    A label artifact may hold fewer cells than the receptors.

    Tables: ``receptor_type`` and ``chain_pairing`` (cells per category, one
    column per group or ``cells``), and ``chains_per_cell`` (cells per number
    of chains of each arm).
    """
    if normalize is not None and not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean or None")
    loaded = _load_receptors(store, receptors)
    cells = loaded.cells
    labels, group_name = cell_groups(
        store, receptors, pd.Index(cells.index), group_by=group_by, groups=groups
    )
    fractions = labels is not None if normalize is None else normalize
    keep = (
        np.ones(len(cells), dtype=bool)
        if labels is None
        else np.asarray([label is not None for label in labels], dtype=bool)
    )
    receptor_type = _category_table(
        cells["receptor_type"].to_numpy(dtype=object), RECEPTOR_TYPES, labels
    )
    chain_pairing = _category_table(
        cells["chain_pairing"].to_numpy(dtype=object), CHAIN_PAIRINGS, labels
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
        index=pd.Index(bins, name="chains"),
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
            normalize=fractions,
            title="Receptor type",
            show_legend=show_legend,
        )
        _draw_categories(
            axes["chain_pairing"],
            chain_pairing,
            _CHAIN_PAIRING_COLORS,
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


def _load_clonotypes(store: Any, clonotypes: ArtifactRef) -> Any:
    """Load saved clonotypes through the store's narrow loader."""
    loader = getattr(getattr(store, "repertoire", None), "load_clonotypes", None)
    if not callable(loader):
        raise TypeError("store does not provide clonotypes")
    return loader(clonotypes)


def _clonotype_label(name: str, row: pd.Series) -> str:
    """Name a clonotype by its label and its representative's main junction.

    The representative of a clonotype has a chain in ``VDJ_1`` or ``VJ_1``, so
    one of their junctions names it.
    """
    junction = row.get("VDJ_1_junction_aa")
    if not isinstance(junction, str):
        junction = row.get("VJ_1_junction_aa")
    return f"{name} ({junction})"


@close_figures_on_error
def clonotype_abundance(
    store: Any,
    clonotypes: ArtifactRef,
    *,
    group_by: str | None = None,
    groups: ArtifactRef | None = None,
    top: int = 20,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how many clonotypes have each size, and the largest clonotypes.

    ``clonotypes`` is a ``clonotypes`` artifact, such as one from
    ``ds.repertoire.clonotypes``. The first panel counts clonotypes by their
    number of cells on log scales: most clonotypes hold one cell, and expanded
    clonotypes form the tail. The second panel draws the ``top`` clonotypes
    with the most cells, each labelled with its representative's ``VDJ_1``
    junction amino acids, or its ``VJ_1`` junction without a VDJ chain.

    ``group_by`` names a cell metadata column, such as the sample, and
    ``groups`` a cell-label artifact, such as a clustering; with either, the
    first panel draws one line per group, counting the cells of each clonotype
    within the group, and the second stacks each clonotype's cells by group.
    Cells without a group are left out.

    Tables: ``sizes`` (``group``, ``clone_size``, ``n_clonotypes``) and
    ``top`` (``clonotype``, ``label``, then the cells of each group, or
    ``cells``).
    """
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ValueError("top must be a positive integer")
    loaded = _load_clonotypes(store, clonotypes)
    cells = loaded.cells
    group_labels, group_name = cell_groups(
        store, clonotypes, pd.Index(cells.index), group_by=group_by, groups=groups
    )
    clonotype = cells["clonotype"].to_numpy(dtype=object)
    keep = ~pd.isna(cells["clonotype"]).to_numpy()
    if group_labels is None:
        group_values = np.full(len(cells), "all cells", dtype=object)
    else:
        group_values = np.asarray(group_labels, dtype=object)
        keep &= np.asarray([label is not None for label in group_values], dtype=bool)
    frame = pd.DataFrame({"group": group_values[keep], "clonotype": clonotype[keep]})
    counts = frame.groupby(["group", "clonotype"], sort=False).size()
    group_order = sort_categories(pd.unique(frame["group"]))
    sizes = (
        counts.rename("clone_size")
        .reset_index()
        .groupby(["group", "clone_size"])
        .size()
        .rename("n_clonotypes")
        .reset_index()
    )
    sizes["group"] = pd.Categorical(sizes["group"], categories=group_order)
    sizes = sizes.sort_values(["group", "clone_size"]).reset_index(drop=True)
    sizes["group"] = sizes["group"].astype(object)
    per_group = (
        counts.unstack("group", fill_value=0)
        .reindex(columns=group_order, fill_value=0)
        .astype(np.int64)
    )
    totals = per_group.sum(axis=1)
    rank = loaded.clonotypes.set_index("clonotype")
    # The largest clonotypes, ties in the clonotypes' rank order.
    positions = rank.index.get_indexer(totals.index)
    order = np.lexsort((positions, -totals.to_numpy()))[:top]
    chosen = totals.index[order]
    top_table = per_group.loc[chosen].reset_index()
    top_table.insert(
        1, "label", [_clonotype_label(name, rank.loc[name]) for name in chosen]
    )
    if group_labels is None:
        top_table = top_table.rename(columns={"all cells": "cells"})
    if figsize is None and target is None:
        figsize = (12.0, max(3.6, 0.28 * len(top_table) + 1.2))
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["sizes", "top"], figsize=figsize, n_columns=2
        )
        ax = axes["sizes"]
        for group in group_order:
            rows = sizes[sizes["group"] == group]
            ax.plot(
                rows["clone_size"],
                rows["n_clonotypes"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=str(group),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cells of the clonotype")
        ax.set_ylabel("Clonotypes")
        ax.set_title("Clone sizes")
        if show_legend and group_labels is not None:
            ax.legend(frameon=False, fontsize="small")
        ax = axes["top"]
        positions = np.arange(len(top_table))[::-1]
        left = np.zeros(len(top_table))
        value_columns = [
            column
            for column in top_table.columns
            if column not in ("clonotype", "label")
        ]
        for column in value_columns:
            widths = top_table[column].to_numpy(dtype=np.float64)
            ax.barh(positions, widths, left=left, label=str(column))
            left += widths
        ax.set_yticks(positions, top_table["label"].tolist(), fontsize="small")
        ax.set_xlabel("Cells")
        ax.set_title(f"Largest {len(top_table)} clonotypes")
        if show_legend and group_labels is not None:
            ax.legend(frameon=False, fontsize="small", loc="lower right")
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"sizes": sizes, "top": top_table},
        legends=(LegendSpec(kind="categorical", label=group_name),)
        if group_labels is not None
        else (),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(keep.sum()),
            renderer="matplotlib",
            notes=("clonotypes",),
            extras={
                "clonotypes": clonotypes.to_dict(),
                "grouping": group_name,
                "top": top,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _load_expansion(store: Any, expansion: ArtifactRef, splits: Sequence[int]) -> Any:
    """Load saved clonal expansion through the store's narrow loader."""
    loader = getattr(getattr(store, "repertoire", None), "load_clonal_expansion", None)
    if not callable(loader):
        raise TypeError("store does not provide clonal expansion")
    return loader(expansion, splits=splits)


def _expansion_groups(loaded: Any) -> np.ndarray:
    """Name each cell's group of the expansion, None without one.

    A group is named by its labels in the key columns, joined by commas, or
    ``"all cells"`` without key columns.
    """
    cells = loaded.cells
    keys = [
        *loaded.within,
        *(("receptor_type",) if loaded.within_receptor_type else ()),
    ]
    names = np.full(len(cells), None, dtype=object)
    counted = ~pd.isna(cells["category"]).to_numpy()
    if not keys:
        names[counted] = "all cells"
        return names
    columns = [cells[key].to_numpy(dtype=object)[counted] for key in keys]
    names[counted] = [", ".join(map(str, row)) for row in zip(*columns, strict=True)]
    return names


def _stacked_shares(
    ax: Any,
    table: pd.DataFrame,
    colors: dict[str, str],
    *,
    normalize: bool,
    xlabel: str,
    title: str,
    show_legend: bool,
    legend_title: str,
) -> None:
    """Draw one horizontal stacked bar per group, the first group on top."""
    totals = table.sum(axis=1).to_numpy(dtype=np.float64)
    positions = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for name in table.columns:
        widths = table[name].to_numpy(dtype=np.float64)
        if normalize:
            widths = np.divide(
                widths, totals, out=np.zeros_like(widths), where=totals > 0
            )
        ax.barh(positions, widths, left=left, color=colors[str(name)], label=str(name))
        left += widths
    ax.set_yticks(positions, [str(name) for name in table.index])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if show_legend and table.size:
        ax.legend(
            title=legend_title,
            frameon=False,
            fontsize="small",
            loc="upper left",
            bbox_to_anchor=(1, 1),
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
    group_by: str | None = None,
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
    ``ds.repertoire.clonal_expansion``. The first panel draws one stacked bar
    per group: the share of its cells (``count="cells"``) or of its
    clonotypes (``count="clonotypes"``) in each expansion category, as
    scirpy's ``summarize_clonal_expansion`` and scRepertoire's
    ``clonalHomeostasis`` do. The second panel ranks each group's clonotypes
    by their cells in the group and draws the share of its cells that the
    clonotypes of each window of ranks hold, as scRepertoire's
    ``clonalProportion`` does: a sample whose ten largest clonotypes hold
    most of its cells is dominated by a few clones.

    The groups are the expansion's own, such as its samples, unless
    ``group_by`` names a cell metadata column, such as a cell type, or
    ``groups`` a cell-label artifact, such as a clustering. Each cell keeps
    the category measured within its expansion group, so grouping the cells
    by cluster shows which clusters hold the expanded cells of their samples;
    the ranks are counted within the plotted groups. Cells without a
    clonotype or a group are left out.

    Tables: ``categories`` (``group``, ``category``, then ``cells`` or
    ``clonotypes``, and ``fraction``) and ``rank_windows`` (``group``,
    ``window``, ``cells`` and ``fraction``).
    """
    if count not in ("cells", "clonotypes"):
        raise ValueError("count must be 'cells' or 'clonotypes'")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    loaded = _load_expansion(store, expansion, splits)
    cells = loaded.cells
    labels, group_name = cell_groups(
        store, expansion, pd.Index(cells.index), group_by=group_by, groups=groups
    )
    if labels is None:
        names = _expansion_groups(loaded)
        group_name = (
            ", ".join(
                [
                    *loaded.within,
                    *(("receptor_type",) if loaded.within_receptor_type else ()),
                ]
            )
            or None
        )
    else:
        names = labels
    category = cells["category"].to_numpy(dtype=object)
    keep = ~pd.isna(cells["category"]).to_numpy() & np.asarray(
        [name is not None for name in names], dtype=bool
    )
    order = sort_categories(pd.unique(names[keep]))
    frame = pd.DataFrame(
        {
            "group": pd.Categorical(names[keep], categories=order),
            "category": pd.Categorical(category[keep], categories=list(loaded.labels)),
            "clonotype": cells["clonotype"].to_numpy(dtype=object)[keep],
        }
    )
    if count == "clonotypes":
        frame = frame.drop_duplicates()
    wide = (
        pd.crosstab(frame["group"], frame["category"], dropna=False)
        .reindex(index=order, columns=list(loaded.labels), fill_value=0)
        .astype(np.int64)
    )
    group_codes = pd.Categorical(names[keep], categories=order).codes.astype(np.int64)
    clonotype_codes = pd.factorize(cells["clonotype"].to_numpy(dtype=object)[keep])[0]
    window_names = list(rank_window_labels(splits))
    windows = pd.DataFrame(
        rank_windows(clonotype_codes, group_codes, len(order), splits),
        index=pd.Index(order),
        columns=pd.Index(window_names),
    )
    # Windows that hold no cell of any group are left out of the figure.
    shown_windows = windows.loc[:, windows.sum(axis=0) > 0]
    if figsize is None and target is None:
        figsize = (13.0, max(3.2, 0.32 * len(order) + 1.6))
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["categories", "rank_windows"],
            figsize=figsize,
            n_columns=2,
        )
        noun = "cells" if count == "cells" else "clonotypes"
        _stacked_shares(
            axes["categories"],
            wide,
            colormap_palette(list(loaded.labels), "viridis_r"),
            normalize=normalize,
            xlabel=f"Fraction of {noun}" if normalize else noun.capitalize(),
            title="Clonal expansion",
            show_legend=show_legend,
            legend_title=(
                "Clone size" if loaded.breakpoints is not None else "Clone frequency"
            ),
        )
        _stacked_shares(
            axes["rank_windows"],
            shown_windows,
            colormap_palette(window_names, "viridis"),
            normalize=normalize,
            xlabel="Fraction of cells" if normalize else "Cells",
            title="Clonal space by clonotype rank",
            show_legend=show_legend,
            legend_title="Clonotype rank",
        )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "categories": _long_table(wide, "category", count),
            "rank_windows": _long_table(windows, "window", "cells"),
        },
        legends=(
            LegendSpec(kind="categorical", label="Expansion category"),
            LegendSpec(kind="categorical", label="Clonotype rank"),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(keep.sum()),
            renderer="matplotlib",
            notes=("clonal_expansion",),
            extras={
                "clonal_expansion": expansion.to_dict(),
                "grouping": group_name,
                "count": count,
                "normalize": normalize,
                "splits": list(loaded.splits),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _load_rarefaction(store: Any, clonotypes: ArtifactRef, **options: Any) -> Any:
    """Compute the curves through the store's narrow repertoire method."""
    method = getattr(getattr(store, "repertoire", None), "rarefaction", None)
    if not callable(method):
        raise TypeError("store does not provide rarefaction")
    return method(clonotypes, **options)


def _key_names(table: pd.DataFrame, key_columns: Sequence[str]) -> np.ndarray:
    """Name each row's group by its labels, joined by commas, or ``"all cells"``."""
    if not key_columns:
        return np.full(len(table), "all cells", dtype=object)
    columns = [table[column].astype(str).to_numpy() for column in key_columns]
    return np.asarray(
        [", ".join(row) for row in zip(*columns, strict=True)], dtype=object
    )


@close_figures_on_error
def clonal_rarefaction(
    store: Any,
    clonotypes: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    within_receptor_type: bool = True,
    cell_selection: ArtifactRef | None = None,
    orders: Sequence[int] = ORDERS,
    base: str = "size",
    endpoint: int | None = None,
    knots: int = 40,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the rarefaction and extrapolation curves of groups of cells.

    ``clonotypes`` is a ``clonotypes`` artifact. Each panel draws one order of
    Hill numbers, the effective number of clonotypes, against the cells
    sampled (``base="size"``) or the sample coverage (``base="coverage"``),
    as iNEXT's ``ggiNEXT`` draws them: one line per group, solid where the
    group is rarefied, dashed where it is extrapolated, and a point at its
    observed size. With ``base="size"`` a last panel draws each group's
    sample coverage against its size, which shows how complete each sample
    is. Curves that keep rising at the observed point have not reached their
    plateau: more cells would find more clonotypes.

    The groups are those of ``ds.repertoire.diversity``: ``group_by`` cell
    metadata columns, a ``groups`` label artifact and, with
    ``within_receptor_type``, the receptor type, among the cells of
    ``cell_selection``, such as the T cells. ``endpoint`` and ``knots`` set
    the sizes of each curve, as ``ds.repertoire.rarefaction`` does.

    Tables: ``curves`` (``group``, the groups' key columns, ``order``,
    ``size``, ``method``, ``diversity``, ``coverage``).
    """
    if base not in ("size", "coverage"):
        raise ValueError("base must be 'size' or 'coverage'")
    result = _load_rarefaction(
        store,
        clonotypes,
        group_by=group_by,
        groups=groups,
        within_receptor_type=within_receptor_type,
        cell_selection=cell_selection,
        orders=orders,
        endpoint=endpoint,
        knots=knots,
    )
    curves = result.curves.copy()
    key_columns = [
        column
        for column in curves.columns
        if column not in ("order", "size", "method", "diversity", "coverage")
    ]
    curves.insert(0, "group", _key_names(curves, key_columns))
    names = sort_categories(pd.unique(curves["group"]))
    colors = categorical_color_map(list(names))
    panels = [f"order_{order}" for order in result.orders]
    if base == "size":
        panels.append("coverage")
    if figsize is None and target is None:
        figsize = (4.2 * len(panels) + 1.5, 3.8)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=panels, figsize=figsize, n_columns=len(panels)
        )
        for panel in panels:
            ax = axes[panel]
            for name in names:
                rows = curves[curves["group"] == name]
                if panel != "coverage":
                    rows = rows[rows["order"] == int(panel.split("_")[1])]
                else:
                    rows = rows[rows["order"] == result.orders[0]]
                x = rows["size" if base == "size" else "coverage"].to_numpy()
                y = rows["coverage" if panel == "coverage" else "diversity"].to_numpy()
                method = rows["method"].to_numpy()
                inner = method != "extrapolation"
                outer = method != "rarefaction"
                ax.plot(
                    x[inner], y[inner], color=colors[name], linewidth=1.4, label=name
                )
                ax.plot(x[outer], y[outer], color=colors[name], linewidth=1.4, ls="--")
                ax.plot(
                    x[method == "observed"],
                    y[method == "observed"],
                    "o",
                    color=colors[name],
                    markersize=4,
                )
            ax.set_xlabel("Cells" if base == "size" else "Sample coverage")
            if panel == "coverage":
                ax.set_ylabel("Sample coverage")
                ax.set_title("Sample completeness")
            else:
                ax.set_ylabel("Hill number")
                ax.set_title(_ORDER_TITLES[int(panel.split("_")[1])])
        if show_legend and len(names) > 1:
            axes[panels[-1]].legend(
                frameon=False, fontsize="small", loc="upper left", bbox_to_anchor=(1, 1)
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"curves": curves},
        legends=(LegendSpec(kind="categorical", label="Group"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(
                curves.loc[curves["method"] == "observed"]
                .drop_duplicates("group")["size"]
                .sum()
            ),
            renderer="matplotlib",
            notes=("clonotypes",),
            extras={
                "clonotypes": clonotypes.to_dict(),
                "grouping": ", ".join(key_columns) or None,
                "orders": list(result.orders),
                "base": base,
                "endpoint": endpoint,
                "knots": knots,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
