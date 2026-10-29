"""Figures of single-cell immune receptor repertoires."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..metadata.selection import CellField, resolve_grouping
from ..repertoire.chains import CHAIN_PAIRINGS, RECEPTOR_TYPES
from ..storage.artifacts import inspect_artifact
from ..storage.refs import ArtifactRef
from ..utils.arrays import sort_categories
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
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


def _load_receptors(store: Any, receptors: ArtifactRef) -> Any:
    """Load saved receptors through the store's narrow loader."""
    loader = getattr(getattr(store, "repertoire", None), "load_receptors", None)
    if not callable(loader):
        raise TypeError("store does not provide immune receptors")
    return loader(receptors, fields=())


def receptor_groups(
    store: Any,
    receptors: ArtifactRef,
    cell_ids: pd.Index,
    *,
    group_by: str | None,
    groups: ArtifactRef | None,
) -> tuple[np.ndarray | None, str | None]:
    """Return each receptor cell's group label, None where it has none.

    ``group_by`` names a cell metadata column and ``groups`` a cell-label
    artifact, such as a clustering; at most one may be given. A label artifact
    may hold fewer cells than the receptors: the cells it lacks have no group.

    Returns:
        The labels aligned to ``cell_ids``, or None without a grouping, and
        the grouping's name.
    """
    if group_by is not None and groups is not None:
        raise ValueError("Pass group_by or groups, not both")
    if group_by is None and groups is None:
        return None, None
    if group_by is not None:
        if not isinstance(group_by, str):
            raise TypeError("group_by must be the name of a cell metadata column")
        selection = inspect_artifact(store.zw, receptors).input_ref("cell_selection")
        resolved = resolve_grouping(
            store.zw,
            store.cells,
            CellField(group_by, kind="categorical"),
            cell_selection=selection,
        )
        name = group_by
    else:
        if not isinstance(groups, ArtifactRef):
            raise TypeError("groups must be an ArtifactRef")
        resolved = resolve_grouping(store.zw, store.cells, groups)
        name = groups.kind
    ids = np.asarray(store.cells.fetch_all("ids")).astype(str)[resolved.cell_idx]
    labels = np.asarray(resolved.labels, dtype=object).copy()
    if resolved.missing_mask is not None:
        labels[np.asarray(resolved.missing_mask, dtype=bool)] = None
    aligned = pd.Series(labels, index=pd.Index(ids)).reindex(cell_ids)
    # An object Series gives a read-only view, so copy before filling in None.
    values = aligned.to_numpy(dtype=object, copy=True)
    values[pd.isna(aligned).to_numpy()] = None
    return values, name


def _category_table(
    values: np.ndarray,
    order: Sequence[str],
    labels: np.ndarray | None,
) -> pd.DataFrame:
    """Count cells per category, one column per group, or ``cells`` without groups."""
    if labels is None:
        counts = pd.Series(values).value_counts().reindex(list(order), fill_value=0)
        return pd.DataFrame({"cells": counts.to_numpy()}, index=pd.Index(order))
    keep = np.asarray([label is not None for label in labels], dtype=bool)
    present = sort_categories(pd.unique(labels[keep]))
    table = pd.crosstab(
        pd.Categorical(values[keep], categories=list(order)),
        pd.Categorical(labels[keep], categories=present),
        dropna=False,
    )
    table.index = pd.Index(order)
    table.columns = pd.Index([str(value) for value in present])
    return table


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
    loaded = _load_receptors(store, receptors)
    cells = loaded.cells
    labels, group_name = receptor_groups(
        store, receptors, pd.Index(cells.index), group_by=group_by, groups=groups
    )
    fractions = labels is not None if normalize is None else normalize
    keep = (
        np.ones(len(cells), dtype=bool)
        if labels is None
        else np.asarray([label is not None for label in labels], dtype=bool)
    )
    receptor_type = _category_table(
        cells["receptor_type"].to_numpy(dtype=object), RECEPTOR_TYPES, labels
    )
    chain_pairing = _category_table(
        cells["chain_pairing"].to_numpy(dtype=object), CHAIN_PAIRINGS, labels
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
