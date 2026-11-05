"""Figures of per-cell allele evidence at variant sites."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument, quoted_value
from ..utils.arrays import sorted_unique
from ._charts import (
    draw_bars,
    draw_histograms,
    draw_lines,
    draw_matrix,
    legend_of,
    log_values,
)
from ._clone_names import clone_names
from ._contracts import CategoricalScale, ColorScale
from ._data import store_method
from ._deps import require_matplotlib
from ._figure import (
    Drawn,
    PlotResult,
    close_figures_on_error,
    finish_plot,
    normalize_axes_target,
    side_legend,
)
from ._groups import plot_groups
from ._style import (
    DEFAULT_RASTERIZE_THRESHOLD,
    GENOTYPE_CALL_SCALE,
    LEGEND_SIDE_MAX_COLUMNS,
    LEGEND_SIDE_MAX_ENTRIES,
    MAX_FIGURE_WIDTH_INCHES,
    MAX_NAMED_ITEMS,
    MUTED_COLOR,
    NO_VALUE_COLOR,
    category_label,
    default_point_edgewidth,
    default_point_size,
    figure_size,
    finish_embedding_axes,
    foreground_color,
    item_inches,
    legend_side_height,
    padded_square_limits,
    palette_for_n,
    panel_area_inches,
    refresh_layout_point_sizes,
    register_layout_point_size,
    scatter_edgecolor,
    theme_context,
)

__all__ = ["allele_frequencies", "genotypes", "variant_depth", "variant_statistics"]

# Bins of the histogram of the cells' mean depth, on a log scale.
_DEPTH_BINS = 40
# The two panels of variant_depth stand in one column at the page width, each
# this many inches tall.
_DEPTH_PANEL_HEIGHT = 2.6
# The room a panel's title takes above a side legend, which starts at the
# panel's top.
_PANEL_TITLE_INCHES = 0.35
# The smallest VMR drawn on the log scale; variants that no covered cell
# varies in sit there.
_VMR_FLOOR = 1e-6
# The colour of the selection's thresholds.
_THRESHOLD_COLOR = "#c0392b"
# What the legend of variant_statistics calls each kind of variant, and its
# marker and area.
_VARIANT_KINDS = {
    "informative": ("Informative", "o", 14.0),
    "masked": ("Masked", "x", 18.0),
    "other": ("Not informative", "o", 6.0),
}
# The height of the bar of clone colours above the heatmap.
_CLONE_BAR_INCHES = 0.25
# The calls panel of genotypes is this wide alone, and grows by this many
# inches for each bar it names; beside a layout, the two panels share the
# page, at least this tall.
_CALLS_PANEL_WIDTH = 6.0
_CALL_BAR_INCHES = 0.22
_LAYOUT_PANEL_HEIGHT = 3.6
# Genotype calls that carry no call of the site: drawn first on a layout, so
# the cells called sit on top of them.
_NO_CALL = ("missing", "filtered")
# The seed of the shuffle that layout points draw in, as embedding draws them.
_LAYOUT_SEED = 0


def _load_selection(store: Any, variants: ArtifactRef) -> Any:
    loader = store_method(store, "variants.load_informative")
    # The loader names its own argument, so the plot checks its argument first.
    if not isinstance(variants, ArtifactRef):
        raise TypeError("variants must be an ArtifactRef")
    if variants.kind != "variant_selection":
        raise ValueError(
            "variants must be a variant_selection artifact from "
            f"ds.variants.informative, not {quoted_value(variants.kind)}"
        )
    return loader(variants)


def _draw_site_depth(
    ax: Any, sites: pd.DataFrame, unit: str, theme: str, show_legend: bool
) -> Drawn | None:
    """Draw each site's mean depth: along its contig, or as a bar of a named site."""
    coordinates = sites["position"].to_numpy() >= 0
    if coordinates.any() and not coordinates.all():
        raise ValueError(
            "The counts hold sites with coordinates and named sites, which one "
            "panel cannot draw together"
        )
    # A site that no cell covers is left out of the log scale.
    shown = sites.assign(mean_depth=sites["mean_depth"].where(sites["mean_depth"] > 0))
    drawn = None
    if coordinates.all():
        contigs = list(dict.fromkeys(sites["contig"].tolist()))
        several = len(contigs) > 1
        drawn = draw_lines(
            ax,
            shown,
            x="position",
            y="mean_depth",
            series="contig",
            colors=CategoricalScale(
                order=tuple(contigs),
                palette=dict(
                    zip(
                        contigs,
                        palette_for_n(len(contigs))
                        if several
                        else [foreground_color(theme)],
                        strict=True,
                    )
                ),
            ),
            x_label=f"Position on {contigs[0]}"
            if not several and contigs[0]
            else "Position",
            y_label=f"Mean depth ({unit})",
            title="Depth per site",
            legend_title="Contig" if several else None,
            show_legend=show_legend,
        )
    else:
        draw_bars(
            ax,
            shown,
            item="site",
            value="mean_depth",
            colors=foreground_color(theme),
            horizontal=False,
            value_label=f"Mean depth ({unit})",
            title="Depth per site",
        )
    ax.set_yscale("log")
    log_values(ax.yaxis)
    return drawn


@close_figures_on_error
def variant_depth(
    store: Any,
    counts: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
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
    coordinates, such as the mitochondrial genome, one line per contig, and
    as bars for named sites, such as GoT targets or GIFT probes; a site that
    no cell covers is left out of the log scale. The second panel draws how
    many cells have each mean depth over the sites, on log bins, leaving out
    cells of no depth: cells with little depth give little evidence about
    their variants. The panels stand in one column at the page width.

    ``group_by`` names cell metadata columns, one or several, such as the
    sample, and ``groups`` a cell-label artifact; the groups are the
    combinations of their labels, as the clone summaries group cells, named
    by their labels joined by commas. With a grouping, the second panel draws
    one histogram per group in the group's colour, under a legend titled by
    the grouping, and says so when no cell of a group has depth. Cells
    without a group, and cells that no file names, are left out of it.

    Tables: ``sites`` (``site``, ``contig``, ``position``, ``mean_depth``,
    ``covered_fraction``) and ``cells`` (``ids``, ``group``, ``mean_depth``,
    ``n_covered_sites``).

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load variant counts or group cells for figures.
        ValueError: If no file names a cell, no site has depth, no named cell
            has a group, a ``group_by`` column is missing or named twice, two
            groups get one name, or the counts hold both sites with
            coordinates and named sites.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact or grouping the cells does not
            fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    load = store_method(store, "variants.load_counts")
    by_site = load(counts, cell_selection=[])
    by_cell = load(counts, sites=[])
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
            "covered_fraction": sites["n_cells"].to_numpy(dtype=np.float64) / n_listed,
        }
    )
    if not np.any(mean_site > 0):
        raise ValueError("No site of the counts has depth, so there is nothing to plot")
    plotted = plot_groups(
        store, counts, group_by=group_by, groups=groups, n_cells=len(cells)
    )
    cell_group, names = plotted.codes, list(plotted.names)
    keep = listed & (cell_group >= 0)
    if not keep.any():
        raise ValueError(
            "No cell that a file names has a group, so there is nothing to plot"
        )
    cells_table = pd.DataFrame(
        {
            "ids": cells.index.to_numpy(dtype=object)[keep],
            "group": np.asarray(names, dtype=object)[cell_group[keep]],
            "mean_depth": cells["mean_depth"].to_numpy(dtype=np.float64)[keep],
            "n_covered_sites": cells["n_covered_sites"].to_numpy(dtype=np.int64)[keep],
        }
    )
    unit = str(by_site.count_unit)
    if figsize is None and target is None:
        # Each panel grows so that a legend of many groups or contigs fits
        # beside it in four columns.
        entries = max(
            len(set(cells_table["group"].tolist())) if plotted.columns else 0,
            len(set(sites_table["contig"].tolist())),
        )
        rows = -(-min(entries, LEGEND_SIDE_MAX_ENTRIES) // LEGEND_SIDE_MAX_COLUMNS)
        figsize, _ = figure_size(
            2,
            panel=(
                MAX_FIGURE_WIDTH_INCHES,
                max(
                    _DEPTH_PANEL_HEIGHT,
                    legend_side_height(rows, title=True) + _PANEL_TITLE_INCHES,
                ),
            ),
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["sites", "cells"], figsize=figsize, n_columns=1
        )
        drawn = [_draw_site_depth(axes["sites"], sites_table, unit, theme, show_legend)]
        ax = axes["cells"]
        x_label = f"Mean depth per site ({unit})"
        if (cells_table["mean_depth"] > 0).any():
            # The groups with a cell, in their colours' order.
            held = set(cells_table["group"].tolist())
            drawn.append(
                draw_histograms(
                    ax,
                    cells_table,
                    value="mean_depth",
                    series="group",
                    colors=plotted.scale([name for name in names if name in held]),
                    bins=_DEPTH_BINS,
                    log_x=True,
                    x_label=x_label,
                    y_label="Cells",
                    title="Depth per cell",
                    legend_title=plotted.title,
                    show_legend=show_legend,
                )
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No cell of a group has depth",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize="small",
                color=foreground_color(theme),
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Cells")
            ax.set_title("Depth per cell")
    return finish_plot(
        figure,
        axes,
        owns,
        name="variant_depth",
        theme=theme,
        tables={"sites": sites_table, "cells": cells_table},
        drawn=drawn,
        inputs={"counts": counts},
        settings={"grouping": plotted.title, "count_unit": unit},
        n_cells=int(keep.sum()),
        show=show,
    )


@close_figures_on_error
def variant_statistics(
    store: Any,
    variants: ArtifactRef,
    *,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot each variant's VMR against its strand correlation.

    ``variants`` is a ``variant_selection`` artifact from
    ``ds.variants.informative``. Each point is a variant that some cell
    shows: the Pearson correlation of its forward and reverse counts across
    cells, against the ratio of the variance of its allele frequency to its
    mean, on a log scale, as mgatk draws it. Sequencing errors sit on one
    strand and spread evenly across cells, so informative variants sit at
    the top right. Informative variants take a colour, the others the muted
    grey, masked ones drawn as crosses, under a legend of the kinds drawn,
    and the selection's VMR and strand correlation thresholds are drawn as
    dotted lines. A variant without a strand correlation, such as one that a
    single cell shows, is left out; for counts without strands, the x axis
    is the mean depth at the site instead.

    Tables: ``variants`` (``variant``, ``vmr``, ``strand_correlation``,
    ``mean_coverage``, ``masked``, ``informative``).

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load variant selections.
        ValueError: If ``variants`` is not a ``variant_selection`` artifact,
            or no variant has a value on the x axis.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    selection = _load_selection(store, variants)
    frame = selection.variants
    strand = frame["strand_correlation"].to_numpy(dtype=np.float64)
    stranded = bool(selection.stranded)
    x_values = strand if stranded else frame["mean_coverage"].to_numpy(dtype=np.float64)
    shown = np.isfinite(x_values)
    if not shown.any():
        raise ValueError(
            "No variant has a strand correlation, so there is nothing to plot"
            if stranded
            else "No variant has a mean depth, so there is nothing to plot"
        )
    vmr = frame["vmr"].to_numpy(dtype=np.float64)
    table = pd.DataFrame(
        {
            "variant": frame.index.to_numpy(dtype=object),
            "vmr": vmr,
            "strand_correlation": strand,
            "mean_coverage": frame["mean_coverage"].to_numpy(dtype=np.float64),
            "masked": frame["masked"].to_numpy(dtype=bool),
            "informative": frame["informative"].to_numpy(dtype=bool),
        }
    )
    y_values = np.where(np.isfinite(vmr) & (vmr > _VMR_FLOOR), vmr, _VMR_FLOOR)
    informative = table["informative"].to_numpy() & shown
    masked = table["masked"].to_numpy() & shown
    kinds = {
        "informative": informative,
        "masked": masked,
        "other": shown & ~informative & ~masked,
    }
    held = [kind for kind in kinds if kinds[kind].any()]
    scale = CategoricalScale(
        order=tuple(held),
        palette={
            kind: palette_for_n(1)[0] if kind == "informative" else MUTED_COLOR
            for kind in held
        },
        labels={kind: _VARIANT_KINDS[kind][0] for kind in held},
    )
    assert scale.palette is not None
    if figsize is None and target is None:
        figsize = (6.0, 4.8)
    _, mpl = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["variants"], figsize=figsize
        )
        ax = axes["variants"]
        # The variants that are not informative first, so the others sit on top.
        for kind in sorted(held, key=("other", "informative", "masked").index):
            _label, marker, area = _VARIANT_KINDS[kind]
            ax.scatter(
                x_values[kinds[kind]],
                y_values[kinds[kind]],
                s=area,
                marker=marker,
                color=scale.palette[kind],
                label=kind,
            )
        thresholds = selection.thresholds
        if thresholds.get("min_vmr") is not None:
            ax.axhline(
                float(thresholds["min_vmr"]), color=_THRESHOLD_COLOR, linestyle=":"
            )
        if stranded and thresholds.get("min_strand_correlation") is not None:
            ax.axvline(
                float(thresholds["min_strand_correlation"]),
                color=_THRESHOLD_COLOR,
                linestyle=":",
            )
        if not stranded and thresholds.get("min_depth") is not None:
            ax.axvline(
                float(thresholds["min_depth"]), color=_THRESHOLD_COLOR, linestyle=":"
            )
        ax.set_yscale("log")
        log_values(ax.yaxis)
        if not stranded:
            ax.set_xscale("log")
            log_values(ax.xaxis)
        ax.set_xlabel("Strand correlation" if stranded else "Mean depth at the site")
        ax.set_ylabel("Variance-mean ratio")
        ax.set_title(
            f"{int(informative.sum())} of {int(shown.sum())} variants informative"
        )
        if show_legend:
            side_legend(
                ax,
                [
                    mpl.lines.Line2D(
                        [],
                        [],
                        marker=_VARIANT_KINDS[kind][1],
                        linestyle="",
                        color=scale.palette[kind],
                        markersize=5,
                    )
                    for kind in held
                ],
                [category_label(scale, kind) for kind in held],
                title="Variant",
            )
    return finish_plot(
        figure,
        axes,
        owns,
        name="variant_statistics",
        theme=theme,
        tables={"variants": table},
        drawn=[legend_of(scale, held, "Variant")],
        inputs={"variants": variants},
        settings={
            "x": "strand_correlation" if stranded else "mean_coverage",
            "thresholds": dict(selection.thresholds),
        },
        n_cells=int(selection.n_cells),
        show=show,
    )


@close_figures_on_error
def allele_frequencies(
    store: Any,
    variants: ArtifactRef,
    *,
    clones: ArtifactRef | None = None,
    max_cells: int = 2000,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot each cell's allele frequency of each informative variant.

    ``variants`` is a ``variant_selection`` artifact from
    ``ds.variants.informative``; its informative variants are the rows, in
    their order along the genome, and cells are the columns, coloured from
    0 to 1. With ``clones``, such as one from ``ds.variants.frequency_clones``,
    the cells with a clone are drawn ordered by clone, largest first, under
    a bar of clone colours whose legend names the clones as their artifact
    does; without it, the cells that the statistics covered are drawn in
    their order. A cell that was not covered at a variant's site has no
    allele frequency and is drawn in the grey of no value, apart from a
    frequency of 0. At most ``max_cells`` cells are drawn, chosen evenly
    along the order so that each clone keeps its share, and the axis says
    how many of how many it draws.

    Tables: ``frequencies`` (``ids``, ``variant``, ``frequency``, NaN where
    the cell was not covered) and ``cells`` (``ids``, ``clone``, NA without
    clones).

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load variant selections, counts or clones.
        ValueError: If ``variants`` is not a ``variant_selection`` artifact,
            the selection has no informative variant, or no cell has a clone.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_cells = integer_argument(max_cells, "max_cells", minimum=1)
    selection = _load_selection(store, variants)
    chosen = selection.variants[selection.variants["informative"]]
    if chosen.empty:
        raise ValueError(
            "The selection has no informative variant, so there is nothing to plot"
        )
    load_counts = store_method(store, "variants.load_counts")
    names = None
    if clones is not None:
        clone_cells = store_method(store, "clones.load_clones")(clones).cells
        codes = np.asarray(clone_cells["clone"].cat.codes, dtype=np.int64)
        held = codes >= 0
        if not held.any():
            raise ValueError("No cell has a clone, so there is nothing to plot")
        names = clone_names(store, clones)
        order = np.flatnonzero(held)[np.argsort(codes[held], kind="stable")]
        cell_ids = clone_cells.index.to_numpy(dtype=object)[order]
        cell_clones = clone_cells["clone"].astype(object).to_numpy(dtype=object)[order]
    else:
        listed = load_counts(
            selection.counts, sites=[], cell_selection=selection.cell_selection
        ).cells
        listed = listed[listed["listed"].to_numpy(dtype=bool)]
        cell_ids = listed.index.to_numpy(dtype=object)
        cell_clones = np.full(len(cell_ids), None, dtype=object)
    n_ordered = len(cell_ids)
    if n_ordered > max_cells:
        picked = sorted_unique(
            np.linspace(0, n_ordered - 1, max_cells).round().astype(np.int64)
        )
        cell_ids = cell_ids[picked]
        cell_clones = cell_clones[picked]
    site_labels = list(dict.fromkeys(chosen["site"].tolist()))
    loaded = load_counts(
        selection.counts, sites=site_labels, cell_selection=list(cell_ids)
    )
    column = {cell: position for position, cell in enumerate(cell_ids.tolist())}
    depth = np.zeros((len(site_labels), len(cell_ids)), dtype=np.float64)
    site_row = {site: position for position, site in enumerate(site_labels)}
    depth_frame = loaded.depth
    depth[
        [site_row[site] for site in depth_frame["site"].astype(str)],
        [column[cell] for cell in depth_frame["ids"].astype(str)],
    ] = depth_frame["depth"].to_numpy(dtype=np.float64)
    count_frame = loaded.counts
    keys = {
        (site, allele): position
        for position, (site, allele) in enumerate(
            zip(chosen["site"].tolist(), chosen["alternative"].tolist(), strict=True)
        )
    }
    counts = np.zeros((len(chosen), len(cell_ids)), dtype=np.float64)
    for site, allele, cell, count in zip(
        count_frame["site"].astype(str).tolist(),
        count_frame["allele"].astype(str).tolist(),
        count_frame["ids"].astype(str).tolist(),
        count_frame["count"].tolist(),
        strict=True,
    ):
        row = keys.get((site, allele))
        if row is not None:
            counts[row, column[cell]] = float(count)
    variant_depth = depth[[site_row[site] for site in chosen["site"].tolist()]]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(variant_depth > 0, counts / variant_depth, np.nan)
    frequencies = pd.DataFrame(
        {
            "ids": np.repeat(cell_ids, len(chosen)),
            "variant": np.tile(chosen.index.to_numpy(dtype=object), len(cell_ids)),
            "frequency": matrix.T.reshape(-1),
        }
    )
    cells_table = pd.DataFrame(
        {"ids": cell_ids, "clone": pd.array(cell_clones, dtype="string")}
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            1,
            panel=(
                MAX_FIGURE_WIDTH_INCHES,
                item_inches(len(chosen), per_item=0.18, base=2.0, minimum=3.0),
            ),
        )
    _, mpl = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["frequencies"], figsize=figsize
        )
        ax = axes["frequencies"]
        drawn: list[Drawn | None] = [
            draw_matrix(
                ax,
                pd.DataFrame(matrix, index=chosen.index, columns=cell_ids),
                color_scale=ColorScale(
                    cmap="viridis", vmin=0.0, vmax=1.0, missing_color=NO_VALUE_COLOR
                ),
                value_label="Allele frequency (grey: not covered)",
                show_legend=show_legend,
            )
        ]
        # Many cells share the panel's width, and only the variants are named.
        ax.set_aspect("auto")
        ax.set_xticks([])
        if len(chosen) <= MAX_NAMED_ITEMS:
            ax.set_yticks(
                np.arange(len(chosen)), [str(label) for label in chosen.index]
            )
        else:
            ax.set_yticks([])
        ax.set_xlabel(
            "Cells"
            if len(cell_ids) == n_ordered
            else f"Cells ({len(cell_ids):,} of {n_ordered:,})"
        )
        if names is not None:
            # A bar of clone colours, a quarter inch high, above the cells.
            height = _CLONE_BAR_INCHES / max(
                figure.get_figheight() * ax.get_position().height, 1e-6
            )
            strip = ax.inset_axes((0.0, 1.0 + height / 4, 1.0, height))
            axes["clones"] = strip
            ids = list(dict.fromkeys(cell_clones.tolist()))
            labels = names.labels(ids)
            colors = palette_for_n(len(labels))
            scale = CategoricalScale(
                order=tuple(labels), palette=dict(zip(labels, colors, strict=True))
            )
            lookup = {clone: position for position, clone in enumerate(ids)}
            strip.imshow(
                np.asarray([[lookup[clone] for clone in cell_clones.tolist()]]),
                aspect="auto",
                interpolation="nearest",
                cmap=mpl.colors.ListedColormap(colors),
                vmin=-0.5,
                vmax=len(labels) - 0.5,
            )
            strip.set_xticks([])
            strip.set_yticks([0], [names.singular.capitalize()])
            title = names.singular.capitalize()
            if show_legend:
                side_legend(
                    strip,
                    [
                        mpl.patches.Patch(facecolor=color, edgecolor="none")
                        for color in colors
                    ],
                    labels,
                    title=title,
                )
            drawn.append(legend_of(scale, labels, title))
    return finish_plot(
        figure,
        axes,
        owns,
        name="allele_frequencies",
        theme=theme,
        tables={"frequencies": frequencies, "cells": cells_table},
        drawn=drawn,
        inputs={"variants": variants},
        settings={
            "clones": None if clones is None else clones.to_dict(),
            "n_variants": len(chosen),
            "max_cells": max_cells,
        },
        n_cells=len(cell_ids),
        show=show,
    )


@close_figures_on_error
def genotypes(
    store: Any,
    genotypes: ArtifactRef,
    *,
    sites: Any | None = None,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    layout: ArtifactRef | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the cells' genotype calls by group and on an embedding.

    ``genotypes`` is a ``variant_genotypes`` artifact, such as one from
    ``ds.variants.genotypes``, and ``sites`` the sites to draw, by label or
    position as ``ds.variants.load_genotypes`` takes them, every site by
    default. The first panel draws, for each site, one bar per group of the
    fractions of the group's cells with each call, the first call of the
    legend at the left; a cell that the calls do not list at a site is
    ``missing`` there. ``group_by`` names cell metadata columns, one or
    several, such as the cell type, and ``groups`` a cell-label artifact;
    the groups are the combinations of their labels, as the clone summaries
    group cells, named by their labels joined by commas. Without either,
    each site has one bar of all cells. Cells without a group are left out.

    With ``layout``, an embedding artifact of the genotypes' cells, such as a
    UMAP whose cell selection is the genotypes' own, the second panel draws
    the cells at the embedding's coordinates, as ``embedding(layout=...)``
    draws them, coloured by their call at the one site that ``sites`` names
    or the genotypes hold; cells without a call there are drawn under the
    others. The calls take one palette, named in words under one legend
    beside the last panel, missing in the grey of no value and filtered in
    the muted grey.

    Tables: ``calls`` (``site``, ``group``, ``call``, ``n_cells``,
    ``share``) and, with a layout, ``cells`` (``ids``, ``x``, ``y``,
    ``call``).

    Raises:
        TypeError: If ``layout`` is not an ArtifactRef, another argument
            has the wrong type, or the store does not load genotypes or
            group cells for figures.
        ValueError: If no cell has a group, a ``group_by`` column is missing
            or named twice, two groups get one name, ``sites`` names a site
            the genotypes do not hold, a layout is drawn for several sites,
            or ``layout`` is not an embedding of the genotypes' cells.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact or grouping the cells does not
            fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    if layout is not None and not isinstance(layout, ArtifactRef):
        raise TypeError("layout must be an ArtifactRef or None")
    loaded = store_method(store, "variants.load_genotypes")(genotypes, sites=sites)
    site_names = [str(name) for name in loaded.sites.index]
    if layout is not None and len(site_names) != 1:
        raise ValueError(
            f"A layout draws the calls of one site, and the plot draws "
            f"{len(site_names)}; name one with sites"
        )
    # The calls in the vocabulary's order, which their codes follow.
    calls = [str(call) for call in loaded.calls["call"].cat.categories]
    n_cells = len(loaded.cells)
    plotted = plot_groups(
        store, genotypes, group_by=group_by, groups=groups, n_cells=n_cells
    )
    cell_group, names, group_name = plotted.codes, list(plotted.names), plotted.title
    if not np.any(cell_group >= 0):
        raise ValueError(
            "No cell of the genotypes has a group, so there is nothing to plot"
        )
    table = loaded.calls
    row_cells = table["ids"].cat.codes.to_numpy(dtype=np.int64)
    row_sites = table["site"].cat.codes.to_numpy(dtype=np.int64)
    row_calls = table["call"].cat.codes.to_numpy(dtype=np.int64)
    missing = calls.index("missing")
    # The calls of each site and group; a group's cells that the calls do not
    # list at a site are missing there.
    row_groups = cell_group[row_cells]
    grouped = row_groups >= 0
    counted = np.zeros((len(site_names), len(names), len(calls)), dtype=np.int64)
    np.add.at(counted, (row_sites[grouped], row_groups[grouped], row_calls[grouped]), 1)
    totals = np.bincount(cell_group[cell_group >= 0], minlength=len(names))
    counted[:, :, missing] += totals[None, :] - counted.sum(axis=2)
    # Every group holds a cell of the genotypes, as plot_groups names only
    # the groups that some cell holds.
    records = []
    for site_rank, name in enumerate(site_names):
        for group_rank, group in enumerate(names):
            total = int(totals[group_rank])
            found = counted[site_rank, group_rank]
            for code, call in enumerate(calls):
                records.append(
                    {
                        "site": name,
                        "group": str(group),
                        "call": call,
                        "n_cells": int(found[code]),
                        "share": float(found[code]) / total,
                    }
                )
    calls_table = pd.DataFrame.from_records(
        records, columns=["site", "group", "call", "n_cells", "share"]
    )
    cells_table = None
    coordinates = np.empty((0, 2), dtype=np.float64)
    if layout is not None:
        from ..storage.artifacts import inspect_artifact
        from ._data import _resolve_layout

        coordinates, _layout_cells, layout_selection = _resolve_layout(store, layout)
        selection = inspect_artifact(store.zw, genotypes).input_ref("cell_selection")
        if layout_selection != selection:
            raise ValueError(
                "layout must be an embedding of the genotypes' cells: its cell "
                "selection must be the genotypes' own"
            )
        chosen = np.full(n_cells, missing, dtype=np.int64)
        chosen[row_cells] = row_calls
        cells_table = pd.DataFrame(
            {
                "ids": loaded.cells.index.to_numpy(dtype=object),
                "x": coordinates[:, 0],
                "y": coordinates[:, 1],
                "call": np.asarray(calls, dtype=object)[chosen],
            }
        )
    # One bar for each site and group, labelled by both when grouped.
    bars = calls_table.assign(
        bar=calls_table["site"]
        if not plotted.columns
        else calls_table["site"] + ", " + calls_table["group"]
    )
    present = set(calls_table.loc[calls_table["n_cells"] > 0, "call"])
    if cells_table is not None:
        present |= set(cells_table["call"])
    scale = _call_scale(present)
    assert scale.order is not None and scale.palette is not None
    bars = bars.loc[bars["call"].isin(present)]
    n_bars = len(site_names) * len(names)
    panel_keys = ["calls"] + (["layout"] if cells_table is not None else [])
    if figsize is None and target is None:
        height = item_inches(n_bars, per_item=_CALL_BAR_INCHES, base=1.2, minimum=2.4)
        figsize, _ = (
            figure_size(1, panel=(_CALLS_PANEL_WIDTH, height))
            if cells_table is None
            else figure_size(
                2,
                panel=(
                    MAX_FIGURE_WIDTH_INCHES / 2,
                    max(height, _LAYOUT_PANEL_HEIGHT),
                ),
                columns=2,
            )
        )
    _, mpl = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=panel_keys, figsize=figsize, n_columns=len(panel_keys)
        )
        draw_bars(
            axes["calls"],
            bars,
            item="bar",
            value="n_cells",
            segment="call",
            colors=scale,
            normalize=True,
            value_label="Fraction of cells",
            title="Calls by group" if plotted.columns else "Calls",
        )
        if cells_table is not None:
            _draw_layout_calls(
                axes["layout"],
                coordinates,
                cells_table["call"].to_numpy(dtype=object),
                scale=scale,
                theme=theme,
                title=f"Calls at {site_names[0]}",
            )
        if show_legend:
            side_legend(
                axes[panel_keys[-1]],
                [
                    mpl.patches.Patch(facecolor=scale.palette[call], edgecolor="none")
                    for call in scale.order
                ],
                [category_label(scale, call) for call in scale.order],
                title="Call",
            )
        if cells_table is not None:
            # Point areas follow the layout panel's final size.
            refresh_layout_point_sizes(figure)
    tables = {"calls": calls_table}
    if cells_table is not None:
        tables["cells"] = cells_table
    return finish_plot(
        figure,
        axes,
        owns,
        name="genotypes",
        theme=theme,
        tables=tables,
        drawn=[legend_of(scale, list(scale.order), "Call")],
        inputs={"genotypes": genotypes},
        settings={
            "grouping": group_name,
            "method": loaded.method,
            "layout": None if layout is None else layout.to_dict(),
            "site": None if layout is None else site_names[0],
        },
        n_cells=int(np.count_nonzero(cell_group >= 0)),
        show=show,
    )


def _call_scale(present: set[str]) -> CategoricalScale:
    """The calls' palette and words, of the calls that a figure draws."""
    assert GENOTYPE_CALL_SCALE.order is not None
    assert GENOTYPE_CALL_SCALE.palette is not None
    assert GENOTYPE_CALL_SCALE.labels is not None
    order = tuple(call for call in GENOTYPE_CALL_SCALE.order if call in present)
    return CategoricalScale(
        order=order,
        palette={call: GENOTYPE_CALL_SCALE.palette[call] for call in order},
        labels={call: GENOTYPE_CALL_SCALE.labels[call] for call in order},
    )


def _draw_layout_calls(
    ax: Any,
    coordinates: np.ndarray,
    cell_calls: np.ndarray,
    *,
    scale: CategoricalScale,
    theme: str,
    title: str,
) -> None:
    """Draw each cell at its embedding coordinates in its call's colour, as embeddings draw cells.

    The cells draw in the seeded shuffle of ``embedding``, those without a
    call first, at the point sizes, square limits and rasterization of
    embeddings.
    """
    from .embedding import _draw_categorical, _panel_draw_order

    _, mpl = require_matplotlib()
    n_cells = len(coordinates)
    x, y = coordinates[:, 0], coordinates[:, 1]
    rank = np.random.default_rng(_LAYOUT_SEED).permutation(n_cells)
    order = _panel_draw_order(rank, (~np.isin(cell_calls, _NO_CALL)).astype(np.int64))
    assert order is not None
    size = default_point_size(n_cells, panel_area=panel_area_inches(ax))
    edgecolor = scatter_edgecolor(theme)
    artist = _draw_categorical(
        ax,
        x[order],
        y[order],
        cell_calls[order],
        np.full(n_cells, size, dtype=np.float64),
        scale=scale,
        edgecolor=edgecolor,
        edgewidth=default_point_edgewidth(n_cells, point_size=size),
        alpha=1.0,
        rasterized=n_cells >= DEFAULT_RASTERIZE_THRESHOLD,
        mpl=mpl,
    )
    register_layout_point_size(
        artist, n_points=n_cells, size_min=1.0, size_max=28.0, edgecolor=edgecolor
    )
    xlim, ylim = padded_square_limits(x, y)
    finish_embedding_axes(ax, xlim=xlim, ylim=ylim, title=title)
