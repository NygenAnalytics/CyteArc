"""Figures of clones: the cells that descend from one ancestor cell."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import integer_argument
from ..utils.arrays import sorted_unique
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._groups import cell_groups
from ._style import apply_figure_chrome, theme_context

__all__ = ["clone_abundance"]


def _load_clones(store: Any, clones: ArtifactRef) -> Any:
    loader = getattr(getattr(store, "clones", None), "load_clones", None)
    if not callable(loader):
        raise TypeError("store does not provide clones.load_clones")
    return loader(clones)


@close_figures_on_error
def clone_abundance(
    store: Any,
    clones: ArtifactRef,
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
    """Plot how many clones have each size, and the largest clones.

    ``clones`` is a ``clones`` artifact, such as one from
    ``ds.clones.import_clones``. The first panel counts clones by their
    number of cells on log scales: in most experiments many clones hold one
    or a few cells, and expanded clones form the tail. The second panel draws
    the ``top`` clones with the most cells, labelled with their IDs.

    ``group_by`` names a cell metadata column, such as the sample or time
    point, and ``groups`` a cell-label artifact, such as a clustering; with
    either, the first panel draws one line per group, counting the cells of
    each clone within the group, and the second stacks each clone's cells by
    group. Cells without a group are left out. The plot raises ``ValueError``
    when no cell with a group has a clone.

    Tables: ``sizes`` (``group``, ``clone_size``, ``n_clones``) and ``top``
    (``clone``, then the cells of each group, or ``cells``).
    """
    top = integer_argument(top, "top", minimum=1)
    loaded = _load_clones(store, clones)
    group_labels, names, group_name = cell_groups(
        store, clones, group_by=group_by, groups=groups
    )
    codes = np.asarray(loaded.cells["clone"].cat.codes, dtype=np.int64)
    if group_labels is None:
        cell_group = np.zeros(len(codes), dtype=np.int64)
        names = ["all cells"]
    else:
        cell_group = group_labels
    keep = (codes >= 0) & (cell_group >= 0)
    if not keep.any():
        raise ValueError(
            "No cell with a group has a clone, so there is nothing to plot"
            if group_labels is not None
            else "No cell has a clone, so there is nothing to plot"
        )
    # Each clone's cells in each group, from the codes alone.
    n_clones = len(loaded.clones)
    pairs, pair_cells = np.unique(
        cell_group[keep] * n_clones + codes[keep], return_counts=True
    )
    held = sorted_unique(pairs // n_clones)
    group_order = [names[code] for code in held.tolist()]
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
    chosen = present[np.lexsort((present, -totals[present]))[:top]]
    rows = np.full(n_clones, -1, dtype=np.int64)
    rows[chosen] = np.arange(len(chosen))
    taken = rows[pair_codes] >= 0
    matrix = np.zeros((len(chosen), len(group_order)), dtype=np.int64)
    matrix[rows[pair_codes[taken]], pair_groups[taken]] = pair_cells[taken]
    ids = loaded.clones["clone"].to_numpy(dtype=object)
    top_table = pd.DataFrame(
        matrix, columns=pd.Index([str(name) for name in group_order])
    )
    top_table.insert(0, "clone", ids[chosen])
    if group_labels is None:
        top_table = top_table.rename(columns={"all cells": "cells"})
    if figsize is None and target is None:
        figsize = (12.0, max(3.6, 0.28 * len(top_table) + 1.2))
    require_matplotlib()
    from matplotlib.ticker import LogFormatter

    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["sizes", "top"], figsize=figsize, n_columns=2
        )
        ax = axes["sizes"]
        by_group = dict(tuple(sizes.groupby("group", sort=False)))
        for group in group_order:
            group_sizes = by_group[group]
            ax.plot(
                group_sizes["clone_size"],
                group_sizes["n_clones"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=str(group),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(LogFormatter())
            axis.set_minor_formatter(LogFormatter())
        ax.set_xlabel("Cells of the clone")
        ax.set_ylabel("Clones")
        ax.set_title("Clone sizes")
        if show_legend and group_labels is not None:
            ax.legend(frameon=False, fontsize="small")
        ax = axes["top"]
        positions = np.arange(len(top_table))[::-1]
        left = np.zeros(len(top_table))
        for column in top_table.columns[1:]:
            widths = top_table[column].to_numpy(dtype=np.float64)
            ax.barh(positions, widths, left=left, label=str(column))
            left += widths
        ax.set_yticks(
            positions, [str(clone) for clone in top_table["clone"]], fontsize="small"
        )
        ax.set_xlabel("Cells")
        ax.set_title(f"Largest {len(top_table)} clones")
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
            notes=("clones",),
            extras={
                "clones": clones.to_dict(),
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
