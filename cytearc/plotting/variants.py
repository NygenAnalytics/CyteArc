"""Figures of per-cell allele evidence at variant sites."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import integer_argument
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._groups import cell_groups
from ._style import apply_figure_chrome, palette_for_n, theme_context

__all__ = ["allele_frequencies", "genotypes", "variant_depth", "variant_statistics"]

# Bins of the histogram of the cells' mean depth, on a log scale.
_DEPTH_BINS = 40
# The smallest VMR drawn on the log scale; variants that no covered cell
# varies in sit there.
_VMR_FLOOR = 1e-6
# The colour of uncovered entries of the allele frequency heatmap.
_UNCOVERED_COLOR = "#d9d9d9"
# The height of the bar of clone colours above the heatmap.
_CLONE_BAR_INCHES = 0.25
# The colour of each genotype call, in the vocabulary's order.
_CALL_COLORS = {
    "wild_type": "#4c78a8",
    "heterozygous": "#f58518",
    "homozygous_alternative": "#e45756",
    "alternative_present": "#b279a2",
    "missing": "#e5e5e5",
    "filtered": "#9d9d9d",
}


def _load_counts(store: Any, counts: ArtifactRef, **options: Any) -> Any:
    loader = getattr(getattr(store, "variants", None), "load_counts", None)
    if not callable(loader):
        raise TypeError("store does not provide variants.load_counts")
    return loader(counts, **options)


def _load_genotypes(store: Any, genotypes: ArtifactRef, **options: Any) -> Any:
    loader = getattr(getattr(store, "variants", None), "load_genotypes", None)
    if not callable(loader):
        raise TypeError("store does not provide variants.load_genotypes")
    return loader(genotypes, **options)


def _load_selection(store: Any, variants: ArtifactRef) -> Any:
    loader = getattr(getattr(store, "variants", None), "load_informative", None)
    if not callable(loader):
        raise TypeError("store does not provide variants.load_informative")
    return loader(variants)


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
    the top right. Informative variants are marked, masked ones drawn as
    crosses, and the selection's VMR and strand correlation thresholds drawn
    as lines. A variant without a strand correlation, such as one that a
    single cell shows, is left out; for counts without strands, the x axis is
    the mean depth at the site instead.

    Tables: ``variants`` (``variant``, ``vmr``, ``strand_correlation``,
    ``mean_coverage``, ``masked``, ``informative``).

    Raises:
        ValueError: If no variant has a value on the x axis.
    """
    selection = _load_selection(store, variants)
    frame = selection.variants
    strand = frame["strand_correlation"].to_numpy(dtype=np.float64)
    stranded = bool(selection.stranded)
    x_values = strand if stranded else frame["mean_coverage"].to_numpy(dtype=np.float64)
    shown = np.isfinite(x_values)
    if not shown.any():
        raise ValueError(
            "No variant has a strand correlation to plot"
            if stranded
            else "No variant has a mean depth to plot"
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
    if figsize is None and target is None:
        figsize = (6.0, 4.8)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["variants"], figsize=figsize
        )
        ax = axes["variants"]
        y_values = np.where(np.isfinite(vmr) & (vmr > _VMR_FLOOR), vmr, _VMR_FLOOR)
        informative = table["informative"].to_numpy() & shown
        masked = table["masked"].to_numpy() & shown
        other = shown & ~informative & ~masked
        colors = palette_for_n(2)
        ax.scatter(
            x_values[other], y_values[other], s=6, color="#9e9e9e", label="other"
        )
        ax.scatter(
            x_values[informative],
            y_values[informative],
            s=14,
            color=colors[0],
            label="informative",
        )
        if masked.any():
            ax.scatter(
                x_values[masked],
                y_values[masked],
                s=18,
                marker="x",
                color=colors[1],
                label="masked",
            )
        thresholds = selection.thresholds
        if thresholds.get("min_vmr") is not None:
            ax.axhline(float(thresholds["min_vmr"]), color="#c0392b", linestyle=":")
        if stranded and thresholds.get("min_strand_correlation") is not None:
            ax.axvline(
                float(thresholds["min_strand_correlation"]),
                color="#c0392b",
                linestyle=":",
            )
        if not stranded and thresholds.get("min_depth") is not None:
            ax.axvline(float(thresholds["min_depth"]), color="#c0392b", linestyle=":")
        ax.set_yscale("log")
        if not stranded:
            ax.set_xscale("log")
        ax.set_xlabel("Strand correlation" if stranded else "Mean depth at the site")
        ax.set_ylabel("Variance-mean ratio")
        ax.set_title(f"{int(informative.sum())} informative of {int(shown.sum())}")
        if show_legend:
            ax.legend(frameon=False, fontsize="small")
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"variants": table},
        legends=(LegendSpec(kind="categorical", label="variants"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(selection.n_cells),
            renderer="matplotlib",
            notes=("variant_selection",),
            extras={
                "variants": variants.to_dict(),
                "x": "strand_correlation" if stranded else "mean_coverage",
                "thresholds": dict(selection.thresholds),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


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
    their order along the genome, and cells are the columns. With ``clones``,
    such as one from ``ds.variants.clones``, the cells with a clone are drawn
    ordered by clone, largest first, under a bar of clone colours; without
    it, the cells that the statistics covered are drawn in their order. A
    cell that was not covered at a variant's site has no allele frequency
    and is drawn grey, apart from a frequency of 0. At most ``max_cells``
    cells are drawn, chosen evenly along the order so that each clone keeps
    its share.

    Tables: ``frequencies`` (``cell``, ``variant``, ``frequency``, NaN where
    the cell was not covered) and ``cells`` (``cell``, ``clone``, NA without
    clones).

    Raises:
        ValueError: If the selection has no informative variant, or no cell
            has a clone.
    """
    max_cells = integer_argument(max_cells, "max_cells", minimum=1)
    selection = _load_selection(store, variants)
    chosen = selection.variants[selection.variants["informative"]]
    if chosen.empty:
        raise ValueError("The selection has no informative variant to plot")
    if clones is not None:
        loader = getattr(getattr(store, "clones", None), "load_clones", None)
        if not callable(loader):
            raise TypeError("store does not provide clones.load_clones")
        clone_cells = loader(clones).cells
        codes = np.asarray(clone_cells["clone"].cat.codes, dtype=np.int64)
        held = codes >= 0
        if not held.any():
            raise ValueError("No cell has a clone, so there is nothing to plot")
        order = np.flatnonzero(held)[np.argsort(codes[held], kind="stable")]
        cell_ids = clone_cells.index.to_numpy(dtype=object)[order]
        cell_clones = clone_cells["clone"].astype(object).to_numpy(dtype=object)[order]
    else:
        listed = _load_counts(
            store, selection.counts, sites=[], cells=selection.cells
        ).cells
        listed = listed[listed["listed"].to_numpy(dtype=bool)]
        cell_ids = listed.index.to_numpy(dtype=object)
        cell_clones = np.full(len(cell_ids), None, dtype=object)
    if len(cell_ids) > max_cells:
        picked = np.unique(np.linspace(0, len(cell_ids) - 1, max_cells).round())
        cell_ids = cell_ids[picked.astype(np.int64)]
        cell_clones = cell_clones[picked.astype(np.int64)]
    site_labels = list(dict.fromkeys(chosen["site"].tolist()))
    loaded = _load_counts(
        store, selection.counts, sites=site_labels, cells=list(cell_ids)
    )
    column = {cell: position for position, cell in enumerate(cell_ids.tolist())}
    depth = np.zeros((len(site_labels), len(cell_ids)), dtype=np.float64)
    site_row = {site: position for position, site in enumerate(site_labels)}
    depth_frame = loaded.depth
    depth[
        [site_row[site] for site in depth_frame["site"].astype(str)],
        [column[cell] for cell in depth_frame["cell"].astype(str)],
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
        count_frame["cell"].astype(str).tolist(),
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
            "cell": np.repeat(cell_ids, len(chosen)),
            "variant": np.tile(chosen.index.to_numpy(dtype=object), len(cell_ids)),
            "frequency": matrix.T.reshape(-1),
        }
    )
    cells_table = pd.DataFrame(
        {"cell": cell_ids, "clone": pd.array(cell_clones, dtype="string")}
    )
    if figsize is None and target is None:
        figsize = (10.0, max(3.0, 0.25 * len(chosen) + 1.5))
    plt, _ = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["frequencies"], figsize=figsize
        )
        colormap = plt.get_cmap("viridis").with_extremes(bad=_UNCOVERED_COLOR)
        ax = axes["frequencies"]
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_yticks(
            np.arange(len(chosen)),
            [str(label) for label in chosen.index.tolist()],
            fontsize="small",
        )
        ax.set_xticks([])
        ax.set_xlabel(f"Cells ({len(cell_ids)}; grey: not covered)")
        if show_legend:
            figure.colorbar(image, ax=ax, label="Allele frequency", fraction=0.03)
        if clones is not None:
            # A bar of clone colours, a quarter inch high, above the cells.
            height = _CLONE_BAR_INCHES / max(
                figure.get_figheight() * ax.get_position().height, 1e-6
            )
            strip = ax.inset_axes((0.0, 1.0 + height / 4, 1.0, height))
            axes["clones"] = strip
            names = list(dict.fromkeys(cell_clones.tolist()))
            colors = palette_for_n(len(names))
            lookup = {name: position for position, name in enumerate(names)}
            from matplotlib.colors import ListedColormap

            strip.imshow(
                np.asarray([[lookup[name] for name in cell_clones.tolist()]]),
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap(colors),
                vmin=-0.5,
                vmax=len(names) - 0.5,
            )
            strip.set_xticks([])
            strip.set_yticks([0], ["clone"])
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"frequencies": frequencies, "cells": cells_table},
        legends=(LegendSpec(kind="continuous", label="Allele frequency"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=len(cell_ids),
            renderer="matplotlib",
            notes=("variant_selection",),
            extras={
                "variants": variants.to_dict(),
                "clones": None if clones is None else clones.to_dict(),
                "n_variants": len(chosen),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


@close_figures_on_error
def genotypes(
    store: Any,
    genotypes: ArtifactRef,
    *,
    sites: Any | None = None,
    group_by: str | None = None,
    groups: ArtifactRef | None = None,
    layout_key: str | None = None,
    site: str | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the cells' genotype calls by group and on a two-dimensional layout.

    ``genotypes`` is a ``variant_genotypes`` artifact, such as one from
    ``ds.variants.genotypes``. The first panel draws, for each site, one bar
    per group whose segments are the shares of the group's cells with each
    call; a cell that the calls do not list at a site is ``missing`` there.
    ``group_by`` names a cell metadata column, such as the cell type, and
    ``groups`` a cell-label artifact; without either, each site has one bar
    of all cells. Cells without a group are left out.

    With ``layout_key``, such as ``"UMAP"``, the second panel draws the cells
    on the layout stored in the cell metadata columns ``<layout_key>1`` and
    ``<layout_key>2``, coloured by their call at ``site``, which may be left
    out when one site is drawn.

    Tables: ``calls`` (``site``, ``group``, ``call``, ``cells``, ``share``)
    and, with a layout, ``cells`` (``cell``, ``x``, ``y``, ``call``).

    Raises:
        ValueError: If no cell has a group, ``site`` names a site that is not
            drawn, or several sites are drawn on a layout without ``site``.
    """
    from ._data import _fetch_cell_column

    loaded = _load_genotypes(store, genotypes, sites=sites)
    site_names = [str(name) for name in loaded.sites.index]
    calls = list(_CALL_COLORS)
    n_cells = len(loaded.cells)
    group_labels, names, group_name = cell_groups(
        store, genotypes, group_by=group_by, groups=groups
    )
    if group_labels is None:
        cell_group = np.zeros(n_cells, dtype=np.int64)
        names = ["all cells"]
    else:
        cell_group = np.asarray(group_labels, dtype=np.int64)
    if not np.any(cell_group >= 0):
        raise ValueError(
            "No cell of the genotypes has a group, so there is nothing to plot"
        )
    table = loaded.calls
    row_cells = table["cell"].cat.codes.to_numpy(dtype=np.int64)
    row_sites = table["site"].cat.codes.to_numpy(dtype=np.int64)
    row_calls = table["call"].cat.codes.to_numpy(dtype=np.int64)
    # Each cell's call at each drawn site, missing where the calls list none.
    matrix = np.full((n_cells, len(site_names)), calls.index("missing"), dtype=np.int64)
    matrix[row_cells, row_sites] = row_calls
    records = []
    for site_rank, name in enumerate(site_names):
        for group_rank, group in enumerate(names):
            members = cell_group == group_rank
            total = int(members.sum())
            if not total:
                continue
            found = np.bincount(matrix[members, site_rank], minlength=len(calls))
            for code, call in enumerate(calls):
                records.append(
                    {
                        "site": name,
                        "group": str(group),
                        "call": call,
                        "cells": int(found[code]),
                        "share": float(found[code]) / total,
                    }
                )
    calls_table = pd.DataFrame.from_records(
        records, columns=["site", "group", "call", "cells", "share"]
    )
    cells_table = None
    if layout_key is not None:
        if site is None:
            if len(site_names) != 1:
                raise ValueError(
                    f"The plot draws {len(site_names)} sites; name the one to colour "
                    "the layout by with site"
                )
            site = site_names[0]
        if site not in site_names:
            raise ValueError(f"site {site!r} is not among the drawn sites")
        from ..metadata.selection import cell_selection_indices
        from ..storage.artifacts import inspect_artifact

        selection = inspect_artifact(store.zw, genotypes).input_ref("cell_selection")
        rows = cell_selection_indices(store.zw, selection)
        x = np.asarray(
            _fetch_cell_column(store, f"{layout_key}1", cell_idx=rows), dtype=np.float64
        )
        y = np.asarray(
            _fetch_cell_column(store, f"{layout_key}2", cell_idx=rows), dtype=np.float64
        )
        chosen = matrix[:, site_names.index(site)]
        cells_table = pd.DataFrame(
            {
                "cell": loaded.cells.index.to_numpy(dtype=object),
                "x": x,
                "y": y,
                "call": np.asarray(calls, dtype=object)[chosen],
            }
        )
    panel_keys = ["calls"] + (["layout"] if cells_table is not None else [])
    if figsize is None and target is None:
        figsize = (
            6.0 * len(panel_keys),
            max(3.0, 0.45 * len(site_names) * len(names) + 1.5),
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=panel_keys, figsize=figsize, n_columns=len(panel_keys)
        )
        ax = axes["calls"]
        bars = [
            (name, str(group))
            for name in site_names
            for group in names
            if not calls_table[
                (calls_table["site"] == name) & (calls_table["group"] == str(group))
            ].empty
        ]
        positions = np.arange(len(bars))
        left = np.zeros(len(bars))
        for call in calls:
            shares = np.asarray(
                [
                    float(
                        calls_table[
                            (calls_table["site"] == name)
                            & (calls_table["group"] == group)
                            & (calls_table["call"] == call)
                        ]["share"].sum()
                    )
                    for name, group in bars
                ]
            )
            ax.barh(positions, shares, left=left, color=_CALL_COLORS[call], label=call)
            left += shares
        ax.set_yticks(
            positions,
            [
                name if group_labels is None else f"{name}, {group}"
                for name, group in bars
            ],
            fontsize="small",
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of cells")
        ax.set_title("Calls by group" if group_labels is not None else "Calls")
        if show_legend:
            ax.legend(frameon=False, fontsize="small", loc="lower right")
        if cells_table is not None:
            ax = axes["layout"]
            for call in calls:
                chosen_rows = (cells_table["call"] == call).to_numpy()
                if chosen_rows.any():
                    ax.scatter(
                        cells_table["x"].to_numpy()[chosen_rows],
                        cells_table["y"].to_numpy()[chosen_rows],
                        s=6,
                        color=_CALL_COLORS[call],
                        label=call,
                        rasterized=True,
                    )
            ax.set_xlabel(f"{layout_key}1")
            ax.set_ylabel(f"{layout_key}2")
            ax.set_title(str(site))
            if show_legend:
                ax.legend(frameon=False, fontsize="small", markerscale=2)
        apply_figure_chrome(figure, theme)
    tables = {"calls": calls_table}
    if cells_table is not None:
        tables["cells"] = cells_table
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=(LegendSpec(kind="categorical", label="call"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(np.count_nonzero(cell_group >= 0)),
            renderer="matplotlib",
            notes=("variant_genotypes",),
            extras={
                "genotypes": genotypes.to_dict(),
                "grouping": group_name,
                "method": loaded.method,
                "layout_key": layout_key,
                "site": site,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
