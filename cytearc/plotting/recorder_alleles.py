"""Figures of lineage recorder alleles: what each site of the recorder reads."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._style import COLORBLIND_PALETTE, apply_figure_chrome, theme_context

__all__ = ["recorder_states"]

# The states drawn, in stacking order, with their colors and labels; not
# applicable sites leave a cell out of its site's share.
_DRAWN = ("mark", "unknown_mark", "unedited", "missing")
_COLORS = {
    "mark": COLORBLIND_PALETTE[0],
    "unknown_mark": COLORBLIND_PALETTE[1],
    "unedited": "#BBBBBB",
    "missing": "#F2F2F2",
}
_LABELS = {
    "mark": "Mark",
    "unknown_mark": "Unknown mark",
    "unedited": "Unedited",
    "missing": "Missing",
}


def _load_alleles(store: Any, alleles: ArtifactRef) -> Any:
    method = getattr(getattr(store, "clones", None), "load_alleles", None)
    if not callable(method):
        raise TypeError("store does not provide clones.load_alleles")
    return method(alleles)


def _site_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Each site's cells in each state and their share of the site's cells.

    A site's cells are those the recorder reached there, so not applicable
    cells are left out of the shares.
    """
    table = counts.copy()
    keys = ["integration", "site"]
    applicable = table["state"] != "not_applicable"
    totals = table.loc[applicable].groupby(keys, sort=False)["n_cells"].sum()
    denominator = table.set_index(keys).index.map(totals).to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(
            applicable & (denominator > 0),
            table["n_cells"].to_numpy(dtype=np.float64) / denominator,
            np.nan,
        )
    table["share"] = share
    return table.reset_index(drop=True)


@close_figures_on_error
def recorder_states(
    store: Any,
    alleles: ArtifactRef,
    *,
    max_sites: int = 60,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the share of cells that each recorder site reads as each state.

    ``alleles`` is a ``lineage_alleles`` artifact, such as one from
    ``ds.clones.import_alleles``. Each bar is a site of an integration,
    named ``{integration}-{site}``, stacked by the share of the site's cells
    that hold a mark, an unknown mark, an unedited site, or no read. A
    site's cells are the cells with an allele record, the cells the recorder
    reached, of which those whose integration lacks the site are left out;
    a cell that holds several alleles at a site counts once, in its first
    state in the order above. Sites in natural order; with more than
    ``max_sites``, the ``max_sites`` sites read in the most cells are drawn
    and the table keeps every site. Sites that are mostly missing were
    rarely captured, and sites that are mostly unedited carry little
    lineage.

    Tables: ``site_states`` (``integration``, ``site``, ``state``,
    ``n_cells``, ``share``), one row per site and state, the record's
    :meth:`~cytearc.clones.LineageAlleles.site_states` with each share.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``alleles`` is not a ``lineage_alleles`` artifact.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact does not fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_sites = integer_argument(max_sites, "max_sites", minimum=1)
    loaded = _load_alleles(store, alleles)
    table = _site_table(loaded.site_states())
    keys = list(dict.fromkeys(zip(table["integration"], table["site"], strict=True)))
    shares = {
        state: table.loc[table["state"] == state]
        .set_index(["integration", "site"])["share"]
        .reindex(keys)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        for state in _DRAWN
    }
    read = table.loc[table["state"].isin(("mark", "unknown_mark", "unedited"))]
    read_cells = (
        read.groupby(["integration", "site"], sort=False)["n_cells"]
        .sum()
        .reindex(keys)
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    shown = np.arange(len(keys))
    if len(keys) > max_sites:
        shown = np.sort(np.argsort(-read_cells, kind="stable")[:max_sites])
    labels = [
        integration if site == "" else f"{integration}-{site}"
        for integration, site in (keys[position] for position in shown)
    ]
    if figsize is None and target is None:
        figsize = (max(4.0, min(14.0, 2.0 + 0.22 * len(shown))), 4.0)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["site_states"], figsize=figsize, n_columns=1
        )
        ax = axes["site_states"]
        bottom = np.zeros(len(shown), dtype=np.float64)
        positions = np.arange(len(shown))
        for state in _DRAWN:
            heights = shares[state][shown]
            ax.bar(
                positions,
                heights,
                bottom=bottom,
                color=_COLORS[state],
                edgecolor="#888888" if state == "missing" else "none",
                linewidth=0.4,
                label=_LABELS[state],
                width=0.85,
            )
            bottom += heights
        ax.set_xticks(positions, labels, rotation=90, fontsize="x-small")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Share of the site's cells")
        ax.set_title("Recorder states per site")
        if not len(shown):
            ax.text(0.5, 0.5, "No allele records", ha="center", va="center")
        if show_legend and len(shown):
            ax.legend(
                frameon=False,
                fontsize="small",
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"site_states": table},
        legends=(LegendSpec(kind="categorical", label="state"),)
        if show_legend and len(shown)
        else (),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(len(loaded.cells)),
            renderer="matplotlib",
            notes=("lineage_alleles",),
            extras={
                "alleles": alleles.to_dict(),
                "max_sites": max_sites,
                "sites": len(keys),
                "drawn_sites": int(len(shown)),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
