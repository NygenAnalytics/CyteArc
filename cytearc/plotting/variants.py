"""Figures of per-cell allele evidence at variant sites."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
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

__all__ = ["variant_depth"]

# Bins of the histogram of the cells' mean depth, on a log scale.
_DEPTH_BINS = 40


def _load_counts(store: Any, counts: ArtifactRef, **options: Any) -> Any:
    loader = getattr(getattr(store, "variants", None), "load_counts", None)
    if not callable(loader):
        raise TypeError("store does not provide variants.load_counts")
    return loader(counts, **options)


@close_figures_on_error
def variant_depth(
    store: Any,
    counts: ArtifactRef,
    *,
    group_by: str | None = None,
    groups: ArtifactRef | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the depth of variant counts at each site and in each cell.

    ``counts`` is a ``variant_counts`` artifact, such as one from
    ``ds.variants.import_counts``. The first panel draws each site's mean
    depth over the cells that the files name, along the contig for sites with
    coordinates, such as the mitochondrial genome, and as bars for named
    sites, such as GoT targets or GIFT probes; a site that no cell covers is
    left out of the log scale. The second panel draws how many cells have
    each mean depth over the sites, on log scales: cells with little depth
    give little evidence about their variants.

    ``group_by`` names a cell metadata column, such as the sample, and
    ``groups`` a cell-label artifact; with either, the second panel draws one
    histogram per group. Cells without a group, and cells that no file
    names, are left out of it.

    Tables: ``sites`` (``site``, ``contig``, ``position``, ``mean_depth``,
    ``covered_fraction``) and ``cells`` (``cell``, ``group``,
    ``mean_depth``, ``covered_sites``).

    Raises:
        ValueError: If no file names a cell, or no named cell has a group.
    """
    by_site = _load_counts(store, counts, cells=[])
    by_cell = _load_counts(store, counts, sites=[])
    cells = by_cell.cells
    listed = cells["listed"].to_numpy(dtype=bool)
    n_listed = int(listed.sum())
    if not n_listed:
        raise ValueError(
            "No file names a cell of the counts, so there is nothing to plot"
        )
    sites = by_site.sites
    mean_site = sites["depth"].to_numpy(dtype=np.float64) / n_listed
    sites_table = pd.DataFrame(
        {
            "site": sites.index.to_numpy(dtype=object),
            "contig": sites["contig"].to_numpy(dtype=object),
            "position": sites["position"].to_numpy(dtype=np.int64),
            "mean_depth": mean_site,
            "covered_fraction": sites["cells"].to_numpy(dtype=np.float64) / n_listed,
        }
    )
    group_labels, names, group_name = cell_groups(
        store, counts, group_by=group_by, groups=groups
    )
    if group_labels is None:
        cell_group = np.zeros(len(cells), dtype=np.int64)
        names = ["all cells"]
    else:
        cell_group = group_labels
    keep = listed & (cell_group >= 0)
    if not keep.any():
        raise ValueError(
            "No cell that a file names has a group, so there is nothing to plot"
        )
    cells_table = pd.DataFrame(
        {
            "cell": cells.index.to_numpy(dtype=object)[keep],
            "group": np.asarray(names, dtype=object)[cell_group[keep]],
            "mean_depth": cells["mean_depth"].to_numpy(dtype=np.float64)[keep],
            "covered_sites": cells["covered_sites"].to_numpy(dtype=np.int64)[keep],
        }
    )
    if figsize is None and target is None:
        figsize = (12.0, 4.2)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["sites", "cells"], figsize=figsize, n_columns=2
        )
        ax = axes["sites"]
        coordinates = sites_table["position"].to_numpy() >= 0
        shown = np.where(mean_site > 0, mean_site, np.nan)
        if coordinates.any():
            contigs = list(dict.fromkeys(sites_table["contig"][coordinates].tolist()))
            for contig in contigs:
                chosen = coordinates & (sites_table["contig"] == contig).to_numpy()
                ax.plot(
                    sites_table["position"].to_numpy()[chosen],
                    shown[chosen],
                    linewidth=0.8,
                    label=str(contig) if contig else "positions",
                )
            ax.set_xlabel("Position")
            if len(contigs) > 1 and show_legend:
                ax.legend(frameon=False, fontsize="small")
        if (~coordinates).any():
            named = np.flatnonzero(~coordinates)
            ax.bar(np.arange(len(named)), shown[named])
            ax.set_xticks(
                np.arange(len(named)),
                [str(label) for label in sites_table["site"].to_numpy()[named]],
                rotation=60,
                ha="right",
                fontsize="small",
            )
        if np.any(mean_site > 0):
            ax.set_yscale("log")
        ax.set_ylabel(f"Mean depth ({by_site.count_unit})")
        ax.set_title("Depth per site")
        ax = axes["cells"]
        positive = cells_table["mean_depth"].to_numpy() > 0
        low = (
            float(cells_table["mean_depth"][positive].min()) if positive.any() else 1.0
        )
        high = float(cells_table["mean_depth"].max()) if positive.any() else 10.0
        bins = np.geomspace(low, max(high, low * 10), _DEPTH_BINS)
        for group in dict.fromkeys(cells_table["group"].tolist()):
            values = cells_table["mean_depth"][cells_table["group"] == group].to_numpy()
            ax.hist(
                values[values > 0],
                bins=bins,
                histtype="step",
                linewidth=1.2,
                label=str(group),
            )
        ax.set_xscale("log")
        ax.set_xlabel(f"Mean depth per site ({by_site.count_unit})")
        ax.set_ylabel("Cells")
        ax.set_title("Depth per cell")
        if show_legend and group_labels is not None:
            ax.legend(frameon=False, fontsize="small")
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"sites": sites_table, "cells": cells_table},
        legends=(LegendSpec(kind="categorical", label=group_name),)
        if group_labels is not None
        else (),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(keep.sum()),
            renderer="matplotlib",
            notes=("variant_counts",),
            extras={
                "counts": counts.to_dict(),
                "grouping": group_name,
                "count_unit": by_site.count_unit,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
