"""Figures of pooled CRISPR screens."""

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
from ._style import apply_figure_chrome, foreground_color, theme_context

_PANELS = ("guides_per_cell", "cells_per_guide", "top_fraction")
# A library of at most this many guides gets one labelled bar per guide; a
# larger one is drawn as guides ranked by their cells.
_LABELLED_GUIDES = 40
_NEGATIVE_COLOR = "#bdbdbd"
_ONE_GUIDE_COLOR = "#2b6cb0"
_MULTIPLE_COLOR = "#d1495b"
_CALL_GROUPS = (
    ("No guide called", _NEGATIVE_COLOR),
    ("One guide called", _ONE_GUIDE_COLOR),
    ("Several guides called", _MULTIPLE_COLOR),
)


def _guide_calls(store: Any, calls: ArtifactRef) -> Any:
    """Load saved guide calls through the store's narrow loader."""
    loader = getattr(getattr(store, "perturbation", None), "load_guide_calls", None)
    if not callable(loader):
        raise TypeError("store does not provide guide calls")
    return loader(calls)


def _call_group(n_guides: np.ndarray) -> np.ndarray:
    groups = np.full(len(n_guides), _CALL_GROUPS[2][0], dtype=object)
    groups[n_guides == 0] = _CALL_GROUPS[0][0]
    groups[n_guides == 1] = _CALL_GROUPS[1][0]
    return groups


@close_figures_on_error
def guide_calls(
    store: Any,
    calls: ArtifactRef,
    *,
    max_guides_per_cell: int = 5,
    bins: int = 20,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how many guides each cell carries and how many cells carry each guide.

    ``calls`` is a ``guide_calls`` artifact, such as one from
    ``ds.perturbation.import_guide_calls``. The first panel counts cells by
    their number of called guides; its last bar gathers the cells with
    ``max_guides_per_cell`` or more. The second counts, for each guide, the
    cells in which it is the one called guide and the cells in which it is one
    of several: one labelled bar per guide for a library of at most 40 guides,
    and guides ranked by their cells for a larger one. The third shows the top
    guide's share of each cell's guide UMIs, for cells with guide UMIs, by
    call. A cell with several called guides whose top guide holds nearly all
    of its guide UMIs may carry one guide and ambient molecules of the others.
    """
    if (
        isinstance(max_guides_per_cell, bool)
        or not isinstance(max_guides_per_cell, int)
        or max_guides_per_cell < 2
    ):
        raise ValueError("max_guides_per_cell must be an integer of at least 2")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    loaded = _guide_calls(store, calls)
    cells = loaded.cells
    n_guides = cells["n_guides"].to_numpy(dtype=np.int64)
    bucket_counts = np.bincount(
        np.minimum(n_guides, max_guides_per_cell), minlength=max_guides_per_cell + 1
    )
    guides_per_cell = pd.DataFrame(
        {
            "guides": [str(count) for count in range(max_guides_per_cell)]
            + [f"{max_guides_per_cell}+"],
            "cells": bucket_counts,
        }
    )
    single_calls = cells.loc[n_guides == 1, "call"].value_counts()
    all_calls = loaded.calls["guide"].value_counts()
    guide_ids = pd.Index(loaded.guides, name="guide")
    cells_per_guide = pd.DataFrame(
        {
            "one_guide_cells": single_calls.reindex(guide_ids, fill_value=0),
            "several_guide_cells": (
                all_calls.reindex(guide_ids, fill_value=0)
                - single_calls.reindex(guide_ids, fill_value=0)
            ),
        },
        index=guide_ids,
    )
    cells_per_guide["cells"] = (
        cells_per_guide["one_guide_cells"] + cells_per_guide["several_guide_cells"]
    )
    cells_per_guide = cells_per_guide.sort_values(
        ["cells", "one_guide_cells"], ascending=False, kind="stable"
    ).reset_index()
    groups = _call_group(n_guides)
    fractions = cells["top_fraction"].to_numpy(dtype=np.float64)
    counted = np.isfinite(fractions)
    edges = np.linspace(0.0, 1.0, bins + 1)
    top_fraction = pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_stop": edges[1:],
            **{
                group: np.histogram(fractions[counted & (groups == group)], edges)[0]
                for group, _color in _CALL_GROUPS
            },
        }
    )
    labelled = len(guide_ids) <= _LABELLED_GUIDES
    if figsize is None and target is None:
        figsize = (13.0, max(3.6, 0.22 * len(guide_ids) + 1.2) if labelled else 3.6)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_PANELS), figsize=figsize, n_columns=3
        )
        ax = axes["guides_per_cell"]
        bar_colors = [_NEGATIVE_COLOR, _ONE_GUIDE_COLOR] + [_MULTIPLE_COLOR] * (
            max_guides_per_cell - 1
        )
        ax.bar(guides_per_cell["guides"], guides_per_cell["cells"], color=bar_colors)
        ax.set_xlabel("Called guides")
        ax.set_ylabel("Cells")
        ax.set_title("Guides per cell")

        ax = axes["cells_per_guide"]
        if labelled:
            positions = np.arange(len(cells_per_guide))[::-1]
            ax.barh(
                positions,
                cells_per_guide["one_guide_cells"],
                color=_ONE_GUIDE_COLOR,
                label="Only guide called",
            )
            ax.barh(
                positions,
                cells_per_guide["several_guide_cells"],
                left=cells_per_guide["one_guide_cells"],
                color=_MULTIPLE_COLOR,
                label="One of several called",
            )
            ax.set_yticks(positions, cells_per_guide["guide"].tolist())
            ax.set_xlabel("Cells")
        else:
            ranks = np.arange(1, len(cells_per_guide) + 1)
            ax.plot(
                ranks,
                cells_per_guide["cells"],
                color=foreground_color(theme),
                label="All cells with the guide",
            )
            ax.plot(
                ranks,
                cells_per_guide["one_guide_cells"],
                color=_ONE_GUIDE_COLOR,
                label="Only guide called",
            )
            ax.set_xlabel("Guides, ranked by cells")
            ax.set_ylabel("Cells")
        ax.set_title("Cells per guide")
        if show_legend:
            ax.legend(frameon=False)

        ax = axes["top_fraction"]
        for group, color in _CALL_GROUPS:
            values = fractions[counted & (groups == group)]
            if len(values):
                ax.hist(
                    values,
                    bins=edges,
                    histtype="step",
                    linewidth=1.4,
                    color=color,
                    label=group,
                )
        ax.set_xlabel("Top guide's share of guide UMIs")
        ax.set_ylabel("Cells")
        ax.set_title("Top guide share")
        if show_legend:
            ax.legend(frameon=False)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "guides_per_cell": guides_per_cell,
            "cells_per_guide": cells_per_guide,
            "top_fraction": top_fraction,
        },
        legends=(LegendSpec(kind="categorical", label="Call"),),
        scales=(),
        provenance=PlotProvenance(
            assay=calls.assay,
            cell_key=None,
            n_cells=len(cells),
            renderer="matplotlib",
            notes=("guide_calls",),
            extras={
                "guide_calls": calls.to_dict(),
                "source": loaded.source,
                "max_guides_per_cell": max_guides_per_cell,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
