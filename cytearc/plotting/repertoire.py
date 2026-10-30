"""Figures of single-cell immune receptor repertoires."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..repertoire.chains import CHAIN_PAIRINGS, RECEPTOR_TYPES
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

__all__ = ["receptor_qc"]

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


def _store_method(store: Any, name: str, description: str | None = None) -> Any:
    """Return a method of the store's repertoire namespace, or raise."""
    method = getattr(getattr(store, "repertoire", None), name, None)
    if not callable(method):
        raise TypeError(f"store does not provide {description or name}")
    return method


def _category_table(
    values: np.ndarray,
    order: Sequence[str],
    groups: np.ndarray | None,
    names: Sequence[str],
) -> pd.DataFrame:
    """Count cells per category, one column per group, or ``cells`` without groups."""
    categories = pd.Index(list(order)).get_indexer(np.asarray(values, dtype=object))
    if groups is None:
        counts = np.bincount(categories[categories >= 0], minlength=len(order))
        return pd.DataFrame({"cells": counts}, index=pd.Index(order))
    keep = (categories >= 0) & (groups >= 0)
    matrix = np.bincount(
        categories[keep] * len(names) + groups[keep],
        minlength=len(order) * len(names),
    ).reshape(len(order), len(names))
    return pd.DataFrame(
        matrix, index=pd.Index(order), columns=pd.Index([str(name) for name in names])
    )


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
    loaded = _store_method(store, "load_receptors", "immune receptors")(
        receptors, fields=()
    )
    cells = loaded.cells
    labels, names, group_name = cell_groups(
        store, receptors, group_by=group_by, groups=groups
    )
    fractions = labels is not None if normalize is None else normalize
    keep = np.ones(len(cells), dtype=bool) if labels is None else labels >= 0
    receptor_type = _category_table(
        cells["receptor_type"].to_numpy(dtype=object), RECEPTOR_TYPES, labels, names
    )
    chain_pairing = _category_table(
        cells["chain_pairing"].to_numpy(dtype=object), CHAIN_PAIRINGS, labels, names
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
