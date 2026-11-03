"""Figures of expressed lineage barcodes: what each cell and molecule carries."""

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
from ._style import apply_figure_chrome, theme_context

__all__ = ["barcode_calls"]

_PANELS = ("barcodes_per_cell", "umis_per_barcode", "reads_per_molecule")


def _load_barcodes(store: Any, barcodes: ArtifactRef) -> Any:
    method = getattr(getattr(store, "clones", None), "load_barcodes", None)
    if not callable(method):
        raise TypeError("store does not provide clones.load_barcodes")
    return method(barcodes)


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


def _draw_lines(
    ax: Any, table: pd.DataFrame, value: str, count: str, *, show_legend: bool
) -> None:
    from matplotlib.ticker import LogFormatter

    for library, rows in table.groupby("library", sort=True):
        ax.plot(
            rows[value],
            rows[count],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=str(library),
        )
    if not table.empty:
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(LogFormatter())
            axis.set_minor_formatter(LogFormatter())
        if show_legend:
            ax.legend(frameon=False, fontsize="small")


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
    that counts no UMIs or no reads is left out of its panel.

    Tables: ``barcodes_per_cell`` (``n_barcodes``, ``n_cells``),
    ``umis_per_barcode`` (``library``, ``umis``, ``n_pairs``) and
    ``reads_per_molecule`` (``library``, ``reads``, ``n_molecules``).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``barcodes`` is not a ``lineage_barcodes``
            artifact, or the grouping refuses its labels.
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
    loaded = _load_barcodes(store, barcodes)
    per_cell = loaded.cells["n_barcodes"].to_numpy(dtype=np.int64)
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
    if figsize is None and target is None:
        figsize = (14.0, 4.0)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_PANELS), figsize=figsize, n_columns=3
        )
        ax = axes["barcodes_per_cell"]
        capped = np.minimum(per_cell, max_barcodes_per_cell)
        bars = np.bincount(capped, minlength=max_barcodes_per_cell + 1)
        labels = [str(position) for position in range(max_barcodes_per_cell)] + [
            f"{max_barcodes_per_cell}+"
        ]
        ax.bar(np.arange(max_barcodes_per_cell + 1), bars)
        ax.set_xticks(np.arange(max_barcodes_per_cell + 1), labels)
        ax.set_xlabel("Barcodes of the cell")
        ax.set_ylabel("Cells")
        ax.set_title("Barcodes per cell")
        ax = axes["umis_per_barcode"]
        _draw_lines(ax, umis_table, "umis", "n_pairs", show_legend=show_legend)
        ax.set_xlabel("UMIs of the barcode in the cell")
        ax.set_ylabel("Barcodes of cells")
        ax.set_title("UMIs per barcode")
        if umis_table.empty:
            ax.text(0.5, 0.5, "No library counts UMIs", ha="center", va="center")
        ax = axes["reads_per_molecule"]
        _draw_lines(ax, reads_table, "reads", "n_molecules", show_legend=show_legend)
        ax.set_xlabel("Reads of the molecule")
        ax.set_ylabel("Molecules")
        ax.set_title("Reads per molecule")
        if reads_table.empty:
            ax.text(
                0.5,
                0.5,
                "No library counts reads per molecule",
                ha="center",
                va="center",
            )
        apply_figure_chrome(figure, theme)
    libraries = loaded.libraries["library"].astype(str).tolist()
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "barcodes_per_cell": per_cell_table,
            "umis_per_barcode": umis_table,
            "reads_per_molecule": reads_table,
        },
        legends=(LegendSpec(kind="categorical", label="library"),)
        if len(libraries) and show_legend
        else (),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(len(per_cell)),
            renderer="matplotlib",
            notes=("lineage_barcodes",),
            extras={
                "barcodes": barcodes.to_dict(),
                "libraries": libraries,
                "max_barcodes_per_cell": max_barcodes_per_cell,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
