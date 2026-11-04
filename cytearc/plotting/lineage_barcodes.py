"""Figures of expressed lineage barcodes: what each cell and molecule carries."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument
from ._charts import capped_counts, draw_bars, draw_lines
from ._data import store_method
from ._figure import (
    Drawn,
    PlotResult,
    close_figures_on_error,
    finish_plot,
    normalize_axes_target,
)
from ._style import (
    MAX_FIGURE_WIDTH_INCHES,
    figure_size,
    foreground_color,
    resolve_category_scale,
    theme_context,
)

__all__ = ["barcode_calls"]

_PANELS = ("barcodes_per_cell", "umis_per_barcode", "reads_per_molecule")
# The three panels fill one row at the page width, each this many inches tall.
_PANEL_HEIGHT = 2.6


def _distribution(
    libraries: pd.Series, values: pd.Series, value: str, count: str
) -> pd.DataFrame:
    """Count rows by library and value, leaving out unknown values."""
    known = values.notna()
    frame = pd.DataFrame(
        {
            "library": libraries[known].astype(str).to_numpy(dtype=object),
            value: values[known].to_numpy(dtype=np.int64),
        }
    )
    if frame.empty:
        return pd.DataFrame(
            {
                "library": pd.Series(dtype=object),
                value: pd.Series(dtype=np.int64),
                count: pd.Series(dtype=np.int64),
            }
        )
    counted = frame.groupby(["library", value], sort=True).size()
    return counted.rename(count).reset_index().astype({count: np.int64})


def _capped_ticks(cap: int) -> tuple[list[int], list[str]]:
    """Ticks of an axis of counts from 0 to a last bin of ``cap`` or more, ``"{cap}+"``.

    Whole numbers at Matplotlib's steps, every count up to 10, and the last
    bin, without a step closer to it than half a step.
    """
    from matplotlib.ticker import MaxNLocator

    steps = [
        int(tick)
        for tick in MaxNLocator(integer=True).tick_values(0, cap)
        if 0 <= tick < cap
    ]
    step = steps[1] - steps[0] if len(steps) > 1 else 1
    ticks = [tick for tick in steps if cap - tick >= step / 2] + [cap]
    return ticks, [str(tick) for tick in ticks[:-1]] + [f"{cap}+"]


@close_figures_on_error
def barcode_calls(
    store: Any,
    barcodes: ArtifactRef,
    *,
    max_barcodes_per_cell: int = 10,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the barcodes per cell, the UMIs per barcode and the reads per molecule.

    ``barcodes`` is a ``lineage_barcodes`` artifact, such as one from
    ``ds.clones.import_barcodes``. The three panels show what clone calling
    filters: the cells by their number of distinct barcodes, cells without a
    barcode included, with ``max_barcodes_per_cell`` or more in the last bar;
    each library's pairs of a cell and a barcode by their UMIs; and each
    library's molecules by their reads, on log scales, so molecules of 0
    reads appear in the table only. Barcodes of one or two UMIs and
    molecules of one read are mostly errors and ambient molecules. A library
    that counts no UMIs or no reads is left out of its panel, and a panel
    that no library fills says so. The libraries take colours over every
    library of the artifact, under one legend beside the last panel that
    draws them.

    Tables: ``barcodes_per_cell`` (``n_barcodes``, ``n_cells``),
    ``umis_per_barcode`` (``library``, ``umis``, ``n_pairs``) and
    ``reads_per_molecule`` (``library``, ``reads``, ``n_molecules``).

    Raises:
        TypeError: If an argument has the wrong type, or the store does not
            load lineage barcodes.
        ValueError: If ``barcodes`` is not a ``lineage_barcodes``
            artifact, or holds no cell.
        KeyError: If the store holds no such artifact.
        RuntimeError: If an artifact is incomplete, as a call that stopped
            before it finished leaves it.
        MemoryError: If reading the artifact does not fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_barcodes_per_cell = integer_argument(
        max_barcodes_per_cell, "max_barcodes_per_cell", minimum=1
    )
    loaded = store_method(store, "clones.load_barcodes")(barcodes)
    per_cell = loaded.cells["n_barcodes"].to_numpy(dtype=np.int64)
    if not len(per_cell):
        raise ValueError("The barcodes hold no cell, so there is nothing to plot")
    values, counts = np.unique(per_cell, return_counts=True)
    per_cell_table = pd.DataFrame(
        {"n_barcodes": values.astype(np.int64), "n_cells": counts.astype(np.int64)}
    )
    pairs = loaded.pairs()
    umis_table = _distribution(pairs["library"], pairs["umis"], "umis", "n_pairs")
    records = loaded.records
    is_molecule = records["umi"].notna()
    reads_table = _distribution(
        records["library"][is_molecule],
        records["reads"][is_molecule],
        "reads",
        "n_molecules",
    )
    libraries = loaded.libraries["library"].astype(str).tolist()
    drawn_libraries = set(umis_table["library"]) | set(reads_table["library"])
    colors = resolve_category_scale(
        [library for library in libraries if library in drawn_libraries],
        None,
        universe=libraries,
    )
    capped = capped_counts(per_cell, max_barcodes_per_cell)
    lines = [
        (
            "umis_per_barcode",
            umis_table,
            "umis",
            "n_pairs",
            "UMIs of the barcode in the cell",
            "Barcodes of cells",
            "UMIs per barcode",
            "No library counts UMIs",
        ),
        (
            "reads_per_molecule",
            reads_table,
            "reads",
            "n_molecules",
            "Reads of the molecule",
            "Molecules",
            "Reads per molecule",
            "No library counts reads per molecule",
        ),
    ]
    legend_panel = next(
        (line[0] for line in reversed(lines) if not line[1].empty), None
    )
    if figsize is None and target is None:
        figsize, _ = figure_size(
            len(_PANELS),
            panel=(MAX_FIGURE_WIDTH_INCHES / len(_PANELS), _PANEL_HEIGHT),
            columns=len(_PANELS),
        )
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_PANELS), figsize=figsize, n_columns=len(_PANELS)
        )
        ax = axes["barcodes_per_cell"]
        # The bins stand at their counts, the last at the cap, so an axis of
        # many bins keeps its ticks.
        draw_bars(
            ax,
            pd.DataFrame(
                {
                    "n_barcodes": np.arange(max_barcodes_per_cell + 1),
                    "n_cells": capped.to_numpy(dtype=np.int64),
                }
            ),
            item="n_barcodes",
            value="n_cells",
            colors=foreground_color(theme),
            horizontal=False,
            numeric_items=True,
            value_label="Cells",
            title="Barcodes per cell",
        )
        ax.set_xticks(*_capped_ticks(max_barcodes_per_cell))
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("Barcodes of the cell")
        drawn: Drawn | None = None
        for key, table, value, count, x_label, y_label, title, empty in lines:
            ax = axes[key]
            if table.empty:
                ax.text(
                    0.5,
                    0.5,
                    empty,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize="small",
                    color=foreground_color(theme),
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                continue
            found = draw_lines(
                ax,
                table,
                x=value,
                y=count,
                series="library",
                colors=colors,
                markers=True,
                log_x=True,
                log_y=True,
                x_label=x_label,
                y_label=y_label,
                title=title,
                legend_title="Library" if key == legend_panel else None,
                show_legend=show_legend,
            )
            if key == legend_panel:
                drawn = found
    return finish_plot(
        figure,
        axes,
        owns,
        name="barcode_calls",
        theme=theme,
        tables={
            "barcodes_per_cell": per_cell_table,
            "umis_per_barcode": umis_table,
            "reads_per_molecule": reads_table,
        },
        drawn=[drawn],
        inputs={"barcodes": barcodes},
        settings={
            "libraries": libraries,
            "max_barcodes_per_cell": max_barcodes_per_cell,
        },
        n_cells=int(len(per_cell)),
        show=show,
    )
