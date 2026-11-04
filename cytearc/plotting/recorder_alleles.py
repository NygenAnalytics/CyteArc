"""Figures of lineage recorder alleles: what each site of the recorder reads."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument
from ._charts import draw_bars
from ._data import store_method
from ._figure import (
    PlotResult,
    close_figures_on_error,
    finish_plot,
    normalize_axes_target,
)
from ._style import RECORDER_STATE_SCALE, figure_size, item_inches, theme_context

__all__ = ["recorder_states"]

# The states that hold a read of the site.
_READ = ("mark", "unknown_mark", "unedited")


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
    """Plot the fraction of cells that each recorder site reads as each state.

    ``alleles`` is a ``lineage_alleles`` artifact, such as one from
    ``ds.clones.import_alleles``. Each bar is a site of an integration,
    named ``{integration}-{site}``, stacked by the fraction of the site's
    cells that hold a mark, an unknown mark, an unedited site, or no read,
    the first on top as the legend lists them. A site's cells are the cells
    with an allele record, the cells the recorder reached, of which those
    whose integration lacks the site are left out; a cell that holds several
    alleles at a site counts once, in its first state in the order above.
    Sites in natural order; with more than ``max_sites``, the ``max_sites``
    sites read in the most cells are drawn, the title says how many of how
    many, and the table keeps every site. Sites that are mostly missing were
    rarely captured, and sites that are mostly unedited carry little
    lineage.

    Tables: ``site_states`` (``integration``, ``site``, ``state``,
    ``n_cells``, ``share``), one row per site and state, the record's
    :meth:`~cytearc.clones.LineageAlleles.site_states` with each share.

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load lineage alleles.
        ValueError: If ``alleles`` is not a ``lineage_alleles`` artifact, or
            holds no allele record.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact does not fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_sites = integer_argument(max_sites, "max_sites", minimum=1)
    loaded = store_method(store, "clones.load_alleles")(alleles)
    table = _site_table(loaded.site_states())
    keys = list(dict.fromkeys(zip(table["integration"], table["site"], strict=True)))
    if not keys:
        raise ValueError("The alleles hold no record, so there is nothing to plot")
    states = list(RECORDER_STATE_SCALE.order or ())
    shares = {
        state: table.loc[table["state"] == state]
        .set_index(["integration", "site"])["share"]
        .reindex(keys)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        for state in states
    }
    read = table.loc[table["state"].isin(_READ)]
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
    bars = pd.DataFrame(
        {
            "site": np.repeat(np.asarray(labels, dtype=object), len(states)),
            "state": np.tile(np.asarray(states, dtype=object), len(shown)),
            "share": np.column_stack(
                [shares[state][shown] for state in states]
            ).reshape(-1),
        }
    )
    title = (
        f"The {len(shown)} of {len(keys)} sites read in the most cells"
        if len(shown) < len(keys)
        else "Recorder states per site"
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            1,
            panel=(item_inches(len(shown), per_item=0.18, base=2.4, minimum=4.0), 3.6),
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["site_states"], figsize=figsize, n_columns=1
        )
        ax = axes["site_states"]
        drawn = draw_bars(
            ax,
            bars,
            item="site",
            value="share",
            segment="state",
            colors=RECORDER_STATE_SCALE,
            horizontal=False,
            value_label="Fraction of the site's cells",
            title=title,
            legend_title="State",
            show_legend=show_legend,
        )
        ax.set_ylim(0.0, 1.0)
    return finish_plot(
        figure,
        axes,
        owns,
        name="recorder_states",
        theme=theme,
        tables={"site_states": table},
        drawn=[drawn],
        inputs={"alleles": alleles},
        settings={
            "max_sites": max_sites,
            "sites": len(keys),
            "drawn_sites": int(len(shown)),
        },
        n_cells=int(len(loaded.cells)),
        show=show,
    )
