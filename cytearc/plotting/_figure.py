"""PlotResult and figure ownership helpers."""

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Hashable, cast

import numpy as np
import pandas as pd

from ..storage.validation_scope import validation_scope
from ._contracts import CategoricalScale, ColorScale, PlotProvenance, SizeScale
from ._deps import require_matplotlib
from ._labels import place_layout_labels
from ._style import (
    LEGEND_SIDE_ENTRIES_PER_COLUMN,
    LegendMarkerBoxes,
    continuous_norm,
    legend_side_columns,
    refresh_layout_point_sizes,
    size_legend_layout,
    theme_context,
)


def _json_ready(value: Any) -> Any:
    if not isinstance(value, type) and is_dataclass(value):
        return _json_ready(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


@dataclass(slots=True)
class LegendSpec:
    """Description of a legend or colorbar attached to a plot."""

    kind: str
    label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def close_figures_on_error[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Close pyplot figures a plotting call created when that call raises."""

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        plt, _ = require_matplotlib()
        existing = set(plt.get_fignums())
        with validation_scope():
            try:
                return function(*args, **kwargs)
            except BaseException:
                for number in set(plt.get_fignums()) - existing:
                    plt.close(number)
                raise

    return wrapper


def is_colorbar_axes(ax: Any) -> bool:
    """Whether an axes holds a Matplotlib colorbar."""
    return getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>"


@dataclass(slots=True)
class PlotResult:
    """Return value from cytearc.plotting functions.

    ``figure`` is the matplotlib figure. ``axes`` maps panel keys to axes.
    ``tables`` holds any summary data used to build the plot. ``owns_figure``
    is True when CyteArc created the figure. ``show()`` and ``close()`` release
    owned figures without discarding the result metadata.
    """

    figure: Any
    axes: dict[Hashable, Any]
    tables: dict[str, pd.DataFrame]
    legends: tuple[LegendSpec, ...]
    scales: tuple[Any, ...]
    provenance: PlotProvenance
    owns_figure: bool
    theme: str = "notebook"
    _rendered: bool = field(default=False, init=False, repr=False, compare=False)

    def show(self) -> None:
        plt, _ = require_matplotlib()
        in_ipython = False
        try:
            from IPython import get_ipython

            in_ipython = get_ipython() is not None  # type: ignore[no-untyped-call]
        except ImportError:
            pass
        try:
            if in_ipython:
                from IPython.display import display

                display(self.figure)  # type: ignore[no-untyped-call]
                self._rendered = True
                return
            if (
                getattr(
                    self.figure.canvas,
                    "required_interactive_framework",
                    None,
                )
                is not None
            ):
                plt.show()
                self._rendered = True
        finally:
            if self.owns_figure:
                plt.close(self.figure)

    def __repr__(self) -> str:
        return (
            "PlotResult("
            f"axes={len(self.axes)}, "
            f"tables={len(self.tables)}, "
            f"legends={len(self.legends)}, "
            f"scales={len(self.scales)}, "
            f"owns_figure={self.owns_figure}, "
            f"rendered={self._rendered}"
            ")"
        )

    def _ipython_display_(self) -> None:
        # show() already displayed the figure, so echoing the summary repr
        # underneath it only adds noise. Results that were never displayed keep
        # their plain-text repr.
        if self._rendered:
            return
        from IPython.display import display

        display({"text/plain": repr(self)}, raw=True)  # type: ignore[no-untyped-call]

    def close(self) -> None:
        if not self.owns_figure:
            return
        plt, _ = require_matplotlib()
        plt.close(self.figure)

    def save_provenance(
        self,
        path: str | Path,
        *,
        figure_path: str | Path | None = None,
        dpi: float | None = None,
    ) -> Path:
        """Write a JSON file describing scales, legends, and cell counts."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.figure.get_size_inches()
        payload = {
            "provenance": self.provenance,
            "figure": {
                "width_inches": float(width),
                "height_inches": float(height),
                "dpi": float(self.figure.dpi),
            },
            "legends": self.legends,
            "scales": [
                {
                    "type": type(scale).__name__,
                    "values": scale,
                }
                for scale in self.scales
            ],
            "tables": {
                name: {
                    "rows": len(table),
                    "columns": [str(column) for column in table.columns],
                }
                for name, table in self.tables.items()
            },
        }
        if figure_path is not None:
            exported = Path(figure_path)
            payload["export"] = {
                "filename": exported.name,
                "format": exported.suffix.lower().lstrip("."),
                "dpi": float(dpi if dpi is not None else self.figure.dpi),
            }
        out.write_text(
            json.dumps(
                _json_ready(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return out

    def save(
        self,
        path: str | Path,
        *,
        dpi: int | None = None,
        transparent: bool = False,
        exact_size: bool = True,
        tiff_compression: str = "tiff_lzw",
        provenance_sidecar: bool | str | Path = False,
    ) -> Path:
        """Write the figure to disk (PNG, PDF, SVG, TIFF, and other backends).

        The default background is opaque white for light themes and charcoal
        for the dark theme. Pass ``transparent=True`` when the destination
        supplies its own background. ``exact_size=True`` keeps the inch size you set
        when the figure was created; set it to False only when you want a tight
        crop for display. ``provenance_sidecar=True`` writes a sibling JSON
        file with the plot metadata.
        """
        out = Path(path)
        if not out.suffix:
            raise ValueError("Export path must include a file extension")
        if dpi is not None and dpi <= 0:
            raise ValueError("dpi must be positive")
        file_type = out.suffix.lower().lstrip(".")
        supported = self.figure.canvas.get_supported_filetypes()
        if file_type not in supported:
            raise ValueError(
                f"Unsupported export format {out.suffix!r}; choose from "
                f"{sorted(supported)}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"transparent": transparent}
        if dpi is not None:
            kwargs["dpi"] = dpi
        elif file_type in ("tif", "tiff"):
            kwargs["dpi"] = 300
        if file_type in ("tif", "tiff"):
            kwargs["pil_kwargs"] = {"compression": tiff_compression}
        # Exact-size export must not crop; tight bbox changes physical size.
        kwargs["bbox_inches"] = None if exact_size else "tight"
        _, mpl = require_matplotlib()
        # Honour explicit transparent=False even when the active theme prefers none.
        export_rc: dict[str, Any] = {"savefig.bbox": None} if exact_size else {}
        figure_face = self.figure.patch.get_facecolor()
        figure_alpha = self.figure.patch.get_alpha()
        axes_faces = [
            (ax, ax.patch.get_facecolor(), ax.patch.get_alpha())
            for ax in self.figure.axes
        ]
        if transparent:
            export_rc["savefig.transparent"] = True
            export_rc["savefig.facecolor"] = "none"
        else:
            background = "#111111" if self.theme == "dark" else "white"
            export_rc["savefig.transparent"] = False
            export_rc["savefig.facecolor"] = background
            self.figure.patch.set_facecolor(background)
            self.figure.patch.set_alpha(1.0)
            for ax in self.figure.axes:
                ax.patch.set_facecolor(background)
                ax.patch.set_alpha(1.0)
        try:
            with theme_context(self.theme), mpl.rc_context(export_rc):
                self.figure.savefig(out, **kwargs)
        finally:
            self.figure.patch.set_facecolor(figure_face)
            self.figure.patch.set_alpha(figure_alpha)
            for ax, face, alpha in axes_faces:
                ax.patch.set_facecolor(face)
                ax.patch.set_alpha(alpha)
        if provenance_sidecar:
            sidecar = (
                out.with_suffix(out.suffix + ".json")
                if provenance_sidecar is True
                else Path(provenance_sidecar)
            )
            if sidecar.resolve() == out.resolve():
                raise ValueError("Provenance sidecar path must differ from figure path")
            self.save_provenance(
                sidecar,
                figure_path=out,
                dpi=float(kwargs.get("dpi", self.figure.dpi)),
            )
        return out


def normalize_axes_target(
    target: Any | None,
    *,
    panel_keys: Sequence[Hashable],
    figsize: tuple[float, float] | None,
    n_columns: int | None = None,
    slots: Sequence[int] | None = None,
) -> tuple[Any, dict[Hashable, Any], bool]:
    """Return (figure, axes_map, owns_figure).

    A new figure fills a grid of ``n_columns`` row by row, or places each
    panel in its ``slots`` entry, a distinct grid position counted row by row.
    Grid positions without a panel are removed.
    """
    plt, _ = require_matplotlib()
    keys = list(panel_keys)
    if not keys:
        raise ValueError("panel_keys must be non-empty")

    if target is None:
        if figsize is None:
            from ._style import DEFAULT_PANEL_INCHES

            panel = DEFAULT_PANEL_INCHES
            figsize = (
                (panel * len(keys), panel) if len(keys) > 1 else (panel + 0.4, panel)
            )
        if len(keys) == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
            return fig, {keys[0]: ax}, True
        ncols = n_columns if n_columns is not None else min(len(keys), 4)
        ncols = max(1, min(ncols, len(keys)))
        positions = list(range(len(keys))) if slots is None else list(slots)
        nrows = max(positions) // ncols + 1
        fig, axs = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False, layout="constrained"
        )
        mapping: dict[Hashable, Any] = {
            key: axs[position // ncols, position % ncols]
            for key, position in zip(keys, positions, strict=True)
        }
        occupied = set(positions)
        for j in range(nrows * ncols):
            if j not in occupied:
                fig.delaxes(axs[j // ncols, j % ncols])
        return fig, mapping, True

    if figsize is not None:
        raise ValueError("figsize is invalid when a caller-owned target is provided")

    if isinstance(target, Mapping):
        missing = [k for k in keys if k not in target]
        if missing:
            raise KeyError(f"target mapping missing panel keys: {missing}")
        mapping = {k: target[k] for k in keys}
        figures = {id(ax.figure): ax.figure for ax in mapping.values()}
        if len(figures) != 1:
            raise ValueError("All target axes must belong to the same figure")
        return next(iter(figures.values())), mapping, False

    if isinstance(target, (list, tuple)) or (
        isinstance(target, np.ndarray) and target.dtype == object
    ):
        arr = np.asarray(target, dtype=object).ravel()
        if len(arr) != len(keys):
            raise ValueError(f"Expected {len(keys)} axes for panels, got {len(arr)}")
        figures = {id(ax.figure): ax.figure for ax in arr}
        if len(figures) != 1:
            raise ValueError("All target axes must belong to the same figure")
        fig = next(iter(figures.values()))
        return fig, {k: ax for k, ax in zip(keys, arr)}, False

    if len(keys) != 1:
        raise TypeError(
            "A single Axes target is only valid for one-panel plots; "
            "pass a sequence or mapping of axes for faceted plots"
        )
    return target.figure, {keys[0]: target}, False


def label_panels(
    axes: Mapping[Hashable, Any] | Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    fontsize: float | str | None = None,
    fontweight: str = "bold",
    x: float = -0.06,
    y: float = 1.04,
) -> None:
    """Add A/B/C panel labels to axes in order."""
    if fontsize is None:
        _, mpl = require_matplotlib()
        fontsize = mpl.rcParams["axes.titlesize"]
    if isinstance(axes, Mapping):
        ax_list = list(axes.values())
    else:
        ax_list = list(axes)
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(ax_list))]
    if len(labels) != len(ax_list):
        raise ValueError("labels length must match number of axes")
    for ax, lab in zip(ax_list, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
        )


def _deduplicate_legend_specs(
    legends: Sequence[LegendSpec],
) -> tuple[LegendSpec, ...]:
    seen: set[str] = set()
    out: list[LegendSpec] = []
    for legend in legends:
        key = json.dumps(_json_ready(legend), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(legend)
    return tuple(out)


_OUTSIDE_LEGEND_SLOTS = {
    1: ("outside right center",),
    2: ("outside right upper", "outside right lower"),
    3: (
        "outside right upper",
        "outside right center",
        "outside right lower",
    ),
}
_MAX_OUTSIDE_LEGENDS = max(_OUTSIDE_LEGEND_SLOTS)


# One outside legend: its title, handles, and labels.
_LegendBlock = tuple[str | None, list[Any], list[str]]
# A legend block with the handle boxes its markers need, such as the markers
# of a size legend, or None for the theme's default boxes.
_BoxedLegendBlock = tuple[str | None, list[Any], list[str], LegendMarkerBoxes | None]


def _merged_legend_block(blocks: Sequence[_BoxedLegendBlock]) -> _BoxedLegendBlock:
    """Merge blocks into one untitled block whose boxes hold every marker."""
    merged_handles: list[Any] = []
    merged_labels: list[str] = []
    merged_boxes: LegendMarkerBoxes | None = None
    for title, handles, labels, boxes in blocks:
        merged_handles.extend(handles)
        merged_labels.extend(
            f"{title}: {label}" if title else label for label in labels
        )
        if boxes is not None:
            merged_boxes = boxes if merged_boxes is None else merged_boxes.merged(boxes)
    return None, merged_handles, merged_labels, merged_boxes


def _legend_blocks_overlap(figure: Any, legends: Sequence[Any]) -> bool:
    if len(legends) < 2:
        return False
    # Legend extents need the resolved layout, not rendered pixels.
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in legends]
    return any(
        boxes[first].overlaps(boxes[second])
        for first in range(len(boxes))
        for second in range(first + 1, len(boxes))
    )


def _place_legend_blocks(
    figure: Any,
    blocks: Sequence[_LegendBlock],
    *,
    rows: int = LEGEND_SIDE_ENTRIES_PER_COLUMN,
) -> None:
    """Draw outside legend blocks, merging them when separate blocks collide.

    Each legend column holds at most ``rows`` entries.
    """
    _place_boxed_legend_blocks(
        figure,
        [(title, handles, labels, None) for title, handles, labels in blocks],
        rows=rows,
    )


def _place_boxed_legend_blocks(
    figure: Any,
    blocks: Sequence[_BoxedLegendBlock],
    *,
    rows: int = LEGEND_SIDE_ENTRIES_PER_COLUMN,
) -> None:
    """Draw outside legend blocks in the handle boxes their markers need.

    Blocks that collide merge into one legend whose boxes hold the largest
    marker of every block. Each legend column holds at most ``rows`` entries.
    """
    if not blocks:
        return
    placed = list(blocks)
    if len(placed) > _MAX_OUTSIDE_LEGENDS:
        keep = list(placed[: _MAX_OUTSIDE_LEGENDS - 1])
        keep.append(_merged_legend_block(placed[_MAX_OUTSIDE_LEGENDS - 1 :]))
        placed = keep
    legends: list[Any] = []
    for slot, (title, handles, labels, boxes) in zip(
        _OUTSIDE_LEGEND_SLOTS[len(placed)], placed
    ):
        legends.append(
            figure.legend(
                handles=handles,
                labels=labels,
                title=title,
                frameon=False,
                loc=slot,
                ncols=legend_side_columns(len(handles), rows=rows),
                **(boxes.legend_kwargs() if boxes is not None else {}),
            )
        )
    if _legend_blocks_overlap(figure, legends):
        for legend in legends:
            legend.remove()
        title, handles, labels, boxes = _merged_legend_block(placed)
        figure.legend(
            handles=handles,
            labels=labels,
            title=title,
            frameon=False,
            loc="outside right center",
            ncols=legend_side_columns(len(handles), rows=rows),
            **(boxes.legend_kwargs() if boxes is not None else {}),
        )


@dataclass(frozen=True, slots=True)
class _SharedColorbar:
    """A colorbar that the shared legends draw under ``axes``.

    ``under_panel`` is True for one of the colorbars that a child with
    per-panel limits keeps under each of its panels.
    """

    axes: tuple[Any, ...]
    vmin: float
    vmax: float
    scale: ColorScale
    label: str | None
    under_panel: bool


# One element of the shared legends: a key, which elements that draw the same
# legend share, and an outside legend block or a colorbar.
_SharedLegendPart = tuple[tuple[Any, ...], _BoxedLegendBlock | _SharedColorbar]


def _shared_legend_parts(result: PlotResult) -> list[_SharedLegendPart] | None:
    """Return the shared legend elements that redraw each legend of ``result``.

    Returns None when ``result`` reports no legend, or a legend that the
    shared legends cannot draw, such as a line legend or a categorical legend
    whose scale has no order or palette. Such a child keeps the legends that
    it drew.
    """
    if not result.legends:
        return None
    _, mpl = require_matplotlib()
    categorical = [
        scale for scale in result.scales if isinstance(scale, CategoricalScale)
    ]
    continuous = [scale for scale in result.scales if isinstance(scale, ColorScale)]
    sizes = [scale for scale in result.scales if isinstance(scale, SizeScale)]
    categorical_scale_index = 0
    continuous_scale_index = 0
    size_scale_index = 0
    parts: list[_SharedLegendPart] = []
    for legend in result.legends:
        if legend.kind == "categorical" and categorical:
            categorical_scale = categorical[
                min(categorical_scale_index, len(categorical) - 1)
            ]
            categorical_scale_index += 1
            order = tuple(categorical_scale.order or ())
            if not order or categorical_scale.palette is None:
                return None
            categorical_key = (
                "categorical",
                legend.label,
                order,
                tuple(
                    (str(value), categorical_scale.palette[value]) for value in order
                ),
            )
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=categorical_scale.palette[value],
                    markeredgecolor="none",
                    markersize=5,
                )
                for value in order
            ]
            labels = [
                (
                    categorical_scale.labels.get(value, str(value))
                    if categorical_scale.labels is not None
                    else str(value)
                )
                for value in order
            ]
            parts.append((categorical_key, (legend.label, handles, labels, None)))
        elif legend.kind == "colorbar" and continuous:
            color_scale = continuous[min(continuous_scale_index, len(continuous) - 1)]
            continuous_scale_index += 1
            panel_limits = legend.extras.get("panel_limits")
            targets: list[tuple[tuple[Any, ...], Any, Any, bool]]
            if panel_limits:
                # Per-panel limits keep one colorbar under each panel.
                panels = {
                    str(key): ax
                    for key, ax in result.axes.items()
                    if not is_colorbar_axes(ax)
                }
                if any(name not in panels for name in panel_limits):
                    return None
                targets = [
                    ((panels[name],), limits[0], limits[1], True)
                    for name, limits in panel_limits.items()
                ]
            else:
                vmin = legend.extras.get("vmin", color_scale.vmin)
                vmax = legend.extras.get("vmax", color_scale.vmax)
                if vmin is None or vmax is None:
                    return None
                targets = [
                    (
                        tuple(
                            ax
                            for ax in dict.fromkeys(result.axes.values())
                            if not is_colorbar_axes(ax)
                        ),
                        vmin,
                        vmax,
                        False,
                    )
                ]
            for target_axes, vmin, vmax, under_panel in targets:
                continuous_key = (
                    "colorbar",
                    legend.label,
                    color_scale.cmap,
                    float(vmin),
                    float(vmax),
                    color_scale.vcenter,
                    color_scale.scale,
                    id(target_axes[0]) if under_panel else None,
                )
                parts.append(
                    (
                        continuous_key,
                        _SharedColorbar(
                            axes=target_axes,
                            vmin=float(vmin),
                            vmax=float(vmax),
                            scale=color_scale,
                            label=legend.label,
                            under_panel=under_panel,
                        ),
                    )
                )
        elif legend.kind == "size" and sizes:
            size_scale = sizes[min(size_scale_index, len(sizes) - 1)]
            size_scale_index += 1
            domain = legend.extras.get(
                "domain",
                [size_scale.vmin, size_scale.vmax],
            )
            low, high = float(domain[0]), float(domain[1])
            values = np.asarray(
                legend.extras.get("values", np.linspace(low, high, 4)),
                dtype=np.float64,
            )
            if values.ndim != 1 or not values.size:
                return None
            size_key = (
                "size",
                legend.label,
                low,
                high,
                tuple(values.tolist()),
                size_scale.size_min,
                size_scale.size_max,
            )
            # The shared legend draws the child's final areas exactly.
            areas, boxes = size_legend_layout(size_scale, values)
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor="#bdbdbd",
                    markeredgecolor="#666666",
                    markersize=float(np.sqrt(area)),
                )
                for area in areas
            ]
            labels = (
                [f"{value:.0%}" for value in values]
                if low >= 0 and high <= 1
                else [f"{value:g}" for value in values]
            )
            parts.append((size_key, (legend.label, handles, labels, boxes)))
        elif legend.kind == "marker":
            marker_values = list(legend.extras.get("values", ()))
            markers = list(legend.extras.get("markers", ()))
            if not marker_values or len(marker_values) != len(markers):
                return None
            marker_key = (
                "marker",
                legend.label,
                tuple(map(str, marker_values)),
                tuple(map(str, markers)),
            )
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="",
                    markerfacecolor="#9e9e9e",
                    markeredgecolor="#4d4d4d",
                    markersize=5,
                )
                for marker in markers
            ]
            parts.append(
                (
                    marker_key,
                    (
                        legend.label,
                        handles,
                        [str(value) for value in marker_values],
                        None,
                    ),
                )
            )
        else:
            return None
    return parts


def _render_shared_legends(figure: Any, parts: Sequence[_SharedLegendPart]) -> None:
    """Draw each shared legend element once, in order."""
    plt, mpl = require_matplotlib()
    seen: set[tuple[Any, ...]] = set()
    blocks: list[_BoxedLegendBlock] = []
    for key, part in parts:
        if key in seen:
            continue
        seen.add(key)
        if not isinstance(part, _SharedColorbar):
            blocks.append(part)
            continue
        mappable = plt.cm.ScalarMappable(
            cmap=part.scale.cmap or "viridis",
            norm=continuous_norm(
                mpl,
                vmin=part.vmin,
                vmax=part.vmax,
                vcenter=part.scale.vcenter,
                scale=part.scale.scale,
            ),
        )
        colorbar = figure.colorbar(
            mappable,
            ax=list(part.axes),
            location="bottom",
            orientation="horizontal",
            shrink=0.8 if part.under_panel else 0.45,
            fraction=0.04,
            pad=0.04,
        )
        colorbar.set_label(part.label or "")
    _place_boxed_legend_blocks(figure, blocks)


def _child_colorbar_axes(figure: Any, child: PlotResult) -> list[Any]:
    """Return the colorbar axes among the axes of ``child`` or attached to them."""
    own = set(child.axes.values())
    return [
        ax
        for ax in figure.axes
        if is_colorbar_axes(ax)
        and (
            ax in own
            or any(
                parent in own
                for parent in getattr(ax, "_colorbar_info", {}).get("parents", ())
            )
        )
    ]


def _remove_child_legend_artists(figure: Any, children: Sequence[PlotResult]) -> None:
    """Remove the legends and colorbars of ``children`` before shared equivalents.

    A child's legends are those of its axes, and those of the figure when the
    child owns the figure; its colorbars are the colorbar axes among its axes
    or attached to them.
    """
    from matplotlib.legend import Legend

    for child in children:
        for ax in dict.fromkeys(child.axes.values()):
            if is_colorbar_axes(ax):
                continue
            for artist in list(ax.get_children()):
                if isinstance(artist, Legend):
                    artist.remove()
        if child.owns_figure:
            for legend in list(figure.legends):
                legend.remove()
        for ax in _child_colorbar_axes(figure, child):
            figure.delaxes(ax)


def compose_results(
    figure: Any,
    results: Mapping[Hashable, PlotResult] | Sequence[PlotResult],
    *,
    panel_labels: bool | Sequence[str] = True,
    shared_legends: bool = True,
    owns_figure: bool = False,
    theme: str | None = None,
) -> PlotResult:
    """Merge caller-targeted plot results into one inspectable figure result.

    With ``shared_legends=True``, the legends of the children are drawn again
    beside the figure and their colorbars below their axes, once for legends
    that several children share. A child that reports no legend, or one that
    cannot be drawn this way, such as a line legend or a categorical legend
    whose scale has no order or palette, keeps the legends and colorbars it
    drew.
    """
    if isinstance(results, Mapping):
        named_results = list(results.items())
    else:
        named_results = list(enumerate(results))
    if not named_results:
        raise ValueError("results must be non-empty")
    children = [result for _, result in named_results]
    if any(result.figure is not figure for result in children):
        raise ValueError("All child results must use the supplied figure")

    tables: dict[str, pd.DataFrame] = {}
    legends: list[LegendSpec] = []
    scales: list[Any] = []
    for namespace, result in named_results:
        tables.update(
            {
                f"{namespace}:{table_name}": table
                for table_name, table in result.tables.items()
            }
        )
        legends.extend(result.legends)
        scales.extend(result.scales)

    resolved_theme = theme or children[0].theme
    # Each child is one lettered panel; colorbars and strips are not panels.
    label_axes = [
        next(
            (ax for ax in result.axes.values() if not is_colorbar_axes(ax)),
            next(iter(result.axes.values())),
        )
        for result in children
    ]
    with theme_context(resolved_theme):
        # A child whose legends the shared legends cannot all redraw keeps
        # the legends it drew, and adds none to the shared legends.
        shared: list[PlotResult] = []
        parts: list[_SharedLegendPart] = []
        if shared_legends:
            for result in children:
                child_parts = _shared_legend_parts(result)
                if child_parts is not None:
                    shared.append(result)
                    parts.extend(child_parts)
            _remove_child_legend_artists(figure, shared)
        if panel_labels:
            labels = None if panel_labels is True else list(panel_labels)
            label_panels(label_axes, labels=labels)
        if shared_legends:
            _render_shared_legends(figure, parts)
        refresh_layout_point_sizes(figure)
        place_layout_labels(figure)
    remaining_axes = set(figure.axes)
    axes: dict[Hashable, Any] = {
        (namespace, key): ax
        for namespace, result in named_results
        for key, ax in result.axes.items()
        if ax in remaining_axes
    }
    provenance = PlotProvenance(
        cytearc_version=children[0].provenance.cytearc_version,
        n_cells=max(child.provenance.n_cells for child in children),
        renderer="matplotlib",
        notes=("composite",),
        extras={
            "children": {str(name): child.provenance for name, child in named_results},
            "shared_legends": shared_legends,
        },
    )
    return PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=_deduplicate_legend_specs(legends),
        scales=tuple(scales),
        provenance=provenance,
        owns_figure=owns_figure,
        theme=resolved_theme,
    )
