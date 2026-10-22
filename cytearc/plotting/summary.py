"""Dotplot and matrixplot."""

from collections.abc import Mapping, Sequence
import textwrap
from typing import Any, Hashable

import numpy as np
import pandas as pd

from ..features.values import resolve_feature_batch
from ..storage.artifacts import ArtifactRef

from ._contracts import (
    CategoricalScale,
    ColorScale,
    FeatureRef,
    NormalizationSpec,
    PlotProvenance,
    SizeScale,
    StudyDesign,
)
from ._data import (
    _check_feature_count,
    _fetch_cell_column,
    _group_roles,
    _resolve_grouping,
    _summarize_resolved_features,
    coerce_feature_list,
    unmeasured_extras,
)
from ._deps import require_matplotlib
from ._display import resolve_categorical_scale
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._heatmap_utils import (
    annotation_colors,
    annotation_legend_handles,
    draw_annotation_strips,
    normalize_annotations,
    order_heatmap,
    validate_linkage,
)
from ..utils.arrays import sort_categories
from ._style import (
    apply_figure_chrome,
    capped_figsize,
    continuous_norm,
    foreground_color,
    resolve_color_limits,
    scatter_edgecolor,
    size_legend_layout,
    theme_context,
)


def _wrap_tick_labels(values: Sequence[Any], width: int | None) -> list[str]:
    labels = [str(value) for value in values]
    if width is None:
        return labels
    if width < 1:
        raise ValueError("label_wrap must be positive or None")
    return [textwrap.fill(label, width=width) for label in labels]


# Bounds in points on the largest dot of a default size scale, which fills 72%
# of a grid cell. No default dot is wider than 26 pt, however roomy the cell.
_DEFAULT_DOT_DIAMETER_RANGE = (2.5, 26.0)
_SIZE_LEGEND_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _default_dot_size_scale(ax: Any, *, n_x: int, n_y: int) -> SizeScale:
    """Fit default marker areas to the physical dot-grid cells."""
    bounds = ax.get_position()
    width_points = max(float(bounds.width * ax.figure.get_figwidth() * 72), 1.0)
    height_points = max(float(bounds.height * ax.figure.get_figheight() * 72), 1.0)
    slot_points = min(
        width_points / max(n_x, 1),
        height_points / max(n_y, 1),
    )
    maximum_diameter = float(np.clip(0.72 * slot_points, *_DEFAULT_DOT_DIAMETER_RANGE))
    minimum_diameter = float(np.clip(0.18 * maximum_diameter, 1.0, 3.0))
    return SizeScale(
        size_min=minimum_diameter**2,
        size_max=maximum_diameter**2,
    )


def _draw_feature_group_brackets(
    ax: Any,
    row_groups: Sequence[str | None],
    *,
    swap_axes: bool,
) -> int:
    """Bracket runs of feature rows that belong to one feature group."""
    line_color = ax.xaxis.label.get_color()
    grouped_ranges: list[tuple[str, int, int]] = []
    start = 0
    while start < len(row_groups):
        group = row_groups[start]
        end = start
        while end + 1 < len(row_groups) and row_groups[end + 1] == group:
            end += 1
        if group is not None:
            grouped_ranges.append((group, start, end))
        start = end + 1
    left_geometry: tuple[Any, Any, float] | None = None
    if grouped_ranges and not swap_axes:
        from matplotlib.transforms import ScaledTranslation

        ax.figure.draw_without_rendering()
        renderer = ax.figure.canvas.get_renderer()
        axes_width = max(float(ax.get_window_extent(renderer).width), 1.0)
        label_width = max(
            (
                float(tick.get_window_extent(renderer).width)
                for tick in ax.get_yticklabels()
                if tick.get_visible() and tick.get_text()
            ),
            default=0.0,
        )
        bracket_transform = ax.get_yaxis_transform() + ScaledTranslation(
            -(label_width / ax.figure.dpi + 16.0 / 72.0),
            0,
            ax.figure.dpi_scale_trans,
        )
        text_transform = bracket_transform + ScaledTranslation(
            -7.0 / 72.0,
            0,
            ax.figure.dpi_scale_trans,
        )
        left_geometry = (
            bracket_transform,
            text_transform,
            4.0 * ax.figure.dpi / (72.0 * axes_width),
        )
    for label, start, end in grouped_ranges:
        centre = (start + end) / 2
        if swap_axes:
            transform = ax.get_xaxis_transform()
            line = ax.plot(
                [start - 0.35, end + 0.35],
                [1.03, 1.03],
                transform=transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )[0]
            ax.plot(
                [start - 0.35, start - 0.35, end + 0.35, end + 0.35],
                [1.00, 1.03, 1.03, 1.00],
                transform=transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )
            ax.text(
                centre,
                1.07,
                label,
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
        else:
            assert left_geometry is not None
            bracket_transform, text_transform, cap_width = left_geometry
            line = ax.plot(
                [0, 0],
                [start - 0.35, end + 0.35],
                transform=bracket_transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )[0]
            ax.plot(
                [
                    cap_width,
                    0,
                    0,
                    cap_width,
                ],
                [start - 0.35, start - 0.35, end + 0.35, end + 0.35],
                transform=bracket_transform,
                color=line_color,
                linewidth=0.8,
                clip_on=False,
            )
            group_label = ax.text(
                0,
                centre,
                label,
                transform=text_transform,
                ha="right",
                va="center",
                fontsize=7,
                clip_on=False,
            )
            group_label.set_gid("feature-group-label")
        line.set_gid("feature-group-bracket")
    return len(grouped_ranges)


def _standardize_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``zscore``: each feature row's means z-scored across groups.

    Bracket groups are standardized apart, and ``mean`` keeps its raw values.
    """
    out = df.copy()
    rows = out.groupby(["feature", "feature_group"], observed=False, dropna=False)
    means = rows["mean"].transform("mean")
    stds = rows["mean"].transform("std").replace(0, np.nan)
    out["zscore"] = (out["mean"] - means) / stds
    return out


def _feature_group_key(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def _label_order(
    observed: Sequence[str],
    requested: Sequence[Any] | None,
    *,
    name: str,
    noun: str,
) -> list[str]:
    """Order observed labels by a requested sequence, compared as text.

    The request must list every observed label; requested labels that no row
    carries are left out. Without a request, labels sort naturally.
    """
    if requested is None:
        return sort_categories(observed)
    order = [str(value) for value in requested]
    if len(set(order)) != len(order):
        raise ValueError(f"{name} cannot contain duplicates")
    missing = [value for value in observed if value not in order]
    if missing:
        raise ValueError(
            f"{name} is missing observed {noun}: " + ", ".join(missing[:10])
        )
    present = set(observed)
    return [value for value in order if value in present]


def _group_order(
    observed: Sequence[str],
    group_order: Sequence[Any] | None,
    categorical_scale: CategoricalScale | None,
) -> list[str]:
    """Group columns in requested, stored display, or natural order."""
    requested = (
        group_order
        if group_order is not None
        else categorical_scale.order
        if categorical_scale is not None
        else None
    )
    return _label_order(observed, requested, name="group order", noun="groups")


def _group_tick_labels(
    groups: Sequence[str],
    scale: CategoricalScale | None,
) -> list[str]:
    """Display text of each group label, through the scale's category labels.

    Group labels are text, so a scale's labels apply by the text of their
    category.
    """
    if scale is None or scale.labels is None:
        return list(groups)
    by_text = {str(value): str(label) for value, label in scale.labels.items()}
    return [by_text.get(group, group) for group in groups]


def _feature_rows(
    feature_pairs: Sequence[tuple[str | None, str | FeatureRef]],
    resolved_features: Sequence[Any],
    present: set[tuple[str | None, str]],
    feature_order: Sequence[Any] | None,
) -> list[tuple[str | None, str]]:
    """One row per requested feature and bracket group, in input order.

    A feature listed under two groups keeps a row in each. ``feature_order``
    reorders rows by feature label.
    """
    rows = [
        row
        for row in dict.fromkeys(
            (_feature_group_key(group), feature.label)
            for (group, _), feature in zip(
                feature_pairs,
                resolved_features,
                strict=True,
            )
        )
        if row in present
    ]
    if feature_order is None:
        return rows
    label_order = _label_order(
        list(dict.fromkeys(label for _, label in rows)),
        feature_order,
        name="feature_order",
        noun="features",
    )
    return [row for label in label_order for row in rows if row[1] == label]


def _draw_dendrogram(
    ax: Any,
    linkage: np.ndarray,
    *,
    side: str,
    color: str,
) -> Any:
    """Draw a linkage's dendrogram beside a heatmap's leaves.

    ``side`` is ``"top"`` for the column leaves or ``"right"`` for the row
    leaves. The dendrogram's leaves align with the heatmap cells.
    """
    from matplotlib.collections import LineCollection
    from scipy.cluster.hierarchy import dendrogram

    tree = dendrogram(linkage, no_plot=True)
    # SciPy places leaf k at 10 * k + 5.
    leaves = (np.asarray(tree["icoord"], dtype=np.float64) - 5.0) / 10.0
    heights = np.asarray(tree["dcoord"], dtype=np.float64)
    if side == "top":
        inset = ax.inset_axes([0.0, 1.02, 1.0, 0.14])
        segments = np.stack((leaves, heights), axis=-1)
        inset.set_xlim(ax.get_xlim())
        inset.set_ylim(0.0, max(float(heights.max()), 1e-12))
    else:
        inset = ax.inset_axes([1.02, 0.0, 0.14, 1.0])
        segments = np.stack((heights, leaves), axis=-1)
        inset.set_ylim(ax.get_ylim())
        inset.set_xlim(0.0, max(float(heights.max()), 1e-12))
    inset.add_collection(LineCollection(list(segments), colors=color, linewidths=0.8))
    inset.set_axis_off()
    inset.set_gid(f"{side}-dendrogram")
    return inset


def _group_axis_labels(df: pd.DataFrame, roles: tuple[str, ...]) -> pd.Series:
    """Label each row's group from its role-named grouping columns."""
    if len(roles) == 1:
        return df[roles[0]].astype(str)
    return df[list(roles)].astype(str).agg(" | ".join, axis=1)


def _sample_counts(
    store: Any,
    *,
    cell_key: str,
    cell_indices: np.ndarray | None,
    sample_by: str | None,
    study_design: StudyDesign | None,
) -> tuple[int | None, int]:
    sample_key = study_design.sample_by if study_design is not None else sample_by
    if sample_key is None:
        return None, 0
    values = _fetch_cell_column(
        store,
        sample_key,
        cell_key=cell_key,
        cell_idx=cell_indices,
        labels=True,
    ).astype(object)
    valid = pd.notna(values) & (values != "")
    return int(pd.Series(values[valid]).nunique()), int((~valid).sum())


def _feature_assays(
    store: Any,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    from_assay: str | None,
) -> set[str]:
    assays: set[str] = set()
    for _, feature in coerce_feature_list(features):
        assay = (
            feature.assay
            if isinstance(feature, FeatureRef) and feature.assay is not None
            else from_assay or store._defaultAssay
        )
        assays.add(assay)
    return assays


@close_figures_on_error
def dotplot(
    store: Any,
    *,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    group_by: str | tuple[str, ...] | None = None,
    groups: ArtifactRef | None = None,
    cell_key: str = "I",
    from_assay: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    standardize: str = "none",
    color_scale: ColorScale | None = None,
    size_scale: SizeScale | None = None,
    categorical_scale: CategoricalScale | None = None,
    group_order: Sequence[Any] | None = None,
    feature_order: Sequence[str] | None = None,
    swap_axes: bool = False,
    marker_edgecolor: str | None = None,
    marker_linewidth: float = 0.3,
    label_wrap: int | None = None,
    italicize_features: bool = False,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    max_figure_width: float | None = 7.5,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Dotplot of expression by group.

    Color is the mean value in the group. Dot size is the fraction of measured
    cells above ``expression_cutoff``. Pass ``features`` as a list of genes, or
    as a mapping of group name to gene list when you want gene-set brackets.
    Different features of one assay that carry the same explicit
    ``FeatureRef.label`` are pooled into one row. Any other label shared by
    different features raises, including a label shared across assays.

    With ``sample_by`` (or ``study_design.sample_by``), each sample contributes
    equally: CyteArc first summarizes within sample, then averages across samples.
    Without it, every cell contributes equally.
    """
    _, mpl = require_matplotlib()
    color_scale = color_scale or ColorScale(cmap="viridis")
    if color_scale.scale != "linear":
        raise NotImplementedError("dotplot currently supports only linear color scales")
    size_scale_is_explicit = size_scale is not None
    normalization = normalization or NormalizationSpec()
    grouping = _resolve_grouping(
        store,
        group_by=group_by,
        groups=groups,
        cell_key=cell_key,
    )
    group_keys, cell_indices, _group_values, group_missing = grouping
    if marker_linewidth < 0:
        raise ValueError("marker_linewidth must be non-negative")
    if isinstance(group_by, str):
        categorical_scale = resolve_categorical_scale(
            store,
            group_by,
            categorical_scale,
        )
    feature_pairs = coerce_feature_list(features)
    _check_feature_count(feature_pairs)
    resolved_features = resolve_feature_batch(
        store, [feature for _, feature in feature_pairs], from_assay=from_assay
    )
    aggregate, per_sample, unmeasured = _summarize_resolved_features(
        store,
        resolved_features,
        [group for group, _ in feature_pairs],
        grouping,
        sample_by=sample_by,
        study_design=study_design,
        normalization=normalization,
        expression_cutoff=expression_cutoff,
    )
    if standardize == "feature":
        aggregate = _standardize_feature(aggregate)
    elif standardize not in ("none", "feature"):
        raise ValueError("standardize must be 'none' or 'feature'")
    color_values = "zscore" if standardize == "feature" else "mean"

    plot_df = aggregate.copy()
    plot_df["group_label"] = _group_axis_labels(plot_df, _group_roles(group_keys))
    # One row per requested feature and bracket group, in input order, so a
    # feature listed under two groups keeps a row in each.
    plot_df["row"] = list(
        zip(
            (_feature_group_key(value) for value in plot_df["feature_group"]),
            plot_df["feature"],
            strict=True,
        )
    )
    feature_rows = _feature_rows(
        feature_pairs,
        resolved_features,
        set(plot_df["row"]),
        feature_order,
    )
    resolved_feature_order = [label for _, label in feature_rows]
    resolved_group_order = _group_order(
        list(dict.fromkeys(plot_df["group_label"].tolist())),
        group_order,
        categorical_scale,
    )
    row_index = {row: index for index, row in enumerate(feature_rows)}
    plot_df["group_label"] = pd.Categorical(
        plot_df["group_label"], categories=resolved_group_order, ordered=True
    )

    panel_key: Hashable = "dotplot"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        x_count = (
            len(resolved_feature_order) if swap_axes else len(resolved_group_order)
        )
        y_count = (
            len(resolved_group_order) if swap_axes else len(resolved_feature_order)
        )
        resolved_figsize = (
            max(4.6, 0.42 * x_count + 2.6),
            max(3.4, 0.34 * y_count + 2.4),
        )
        resolved_figsize = capped_figsize(
            *resolved_figsize,
            max_width=max_figure_width,
        )
    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        if size_scale is None:
            size_scale = _default_dot_size_scale(
                ax,
                n_x=(
                    len(resolved_feature_order)
                    if swap_axes
                    else len(resolved_group_order)
                ),
                n_y=(
                    len(resolved_group_order)
                    if swap_axes
                    else len(resolved_feature_order)
                ),
            )

        vals = plot_df[color_values].to_numpy(dtype=np.float64)
        vmin, vmax = resolve_color_limits(vals, color_scale)
        norm = continuous_norm(
            mpl,
            vmin=vmin,
            vmax=vmax,
            vcenter=color_scale.vcenter,
        )

        areas = size_scale.areas(plot_df["fraction"].to_numpy(dtype=np.float64))
        x = plot_df["group_label"].cat.codes.to_numpy()
        y = plot_df["row"].map(row_index).to_numpy(dtype=np.int64)

        edgecolor = marker_edgecolor or scatter_edgecolor(theme)
        x_values = x if not swap_axes else y
        y_values = y if not swap_axes else x
        sc = ax.scatter(
            x_values,
            y_values,
            c=vals,
            s=areas,
            cmap=color_scale.cmap or "viridis",
            norm=norm,
            edgecolors=edgecolor,
            linewidths=marker_linewidth,
            rasterized=False,
        )
        if swap_axes:
            ax.set_xticks(range(len(resolved_feature_order)))
            ax.set_xticklabels(
                _wrap_tick_labels(resolved_feature_order, label_wrap),
                rotation=45,
                ha="right",
                fontstyle=("italic" if italicize_features else "normal"),
            )
            ax.set_yticks(range(len(resolved_group_order)))
            ax.set_yticklabels(
                _wrap_tick_labels(
                    _group_tick_labels(resolved_group_order, categorical_scale),
                    label_wrap,
                )
            )
            ax.set_xlim(-0.5, len(resolved_feature_order) - 0.5)
            ax.set_ylim(-0.5, len(resolved_group_order) - 0.5)
            # The first group reads at the top, as in the unswapped plot.
            ax.invert_yaxis()
            ax.set_xlabel("")
            ax.set_ylabel(" / ".join(group_keys))
        else:
            ax.set_xticks(range(len(resolved_group_order)))
            ax.set_xticklabels(
                _wrap_tick_labels(
                    _group_tick_labels(resolved_group_order, categorical_scale),
                    label_wrap,
                ),
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(len(resolved_feature_order)))
            ax.set_yticklabels(
                _wrap_tick_labels(resolved_feature_order, label_wrap),
                fontstyle=("italic" if italicize_features else "normal"),
            )
            ax.set_xlim(-0.5, len(resolved_group_order) - 0.5)
            ax.set_ylim(-0.5, len(resolved_feature_order) - 0.5)
            ax.invert_yaxis()
            ax.set_xlabel(" / ".join(group_keys))
            ax.set_ylabel("")
        bracket_count = _draw_feature_group_brackets(
            ax,
            [group for group, _ in feature_rows],
            swap_axes=swap_axes,
        )
        colorbar_label = (
            "Mean expression"
            if standardize == "none"
            else "Standardized mean expression"
        )
        if show_legend:
            cb = fig.colorbar(
                sc,
                ax=ax,
                location=("bottom" if swap_axes else "top"),
                orientation="horizontal",
                shrink=0.55,
                pad=(0.08 if swap_axes else 0.04),
                fraction=0.05,
            )
            cb.set_label(colorbar_label)
        legend_values = np.asarray(_SIZE_LEGEND_FRACTIONS)
        size_legend = None
        # Keep fraction sizes beside the colorbar, away from x tick labels.
        if show_legend:
            # The legend draws the data's own areas. Its rows are spaced for
            # the largest dot the scale can draw, so the legend keeps its
            # geometry when the default scale is refit to the final panel.
            legend_areas, legend_boxes = size_legend_layout(
                size_scale,
                legend_values,
                max_area=(
                    size_scale.size_max
                    if size_scale_is_explicit
                    else _DEFAULT_DOT_DIAMETER_RANGE[1] ** 2
                ),
            )
            legend_kwargs = {
                "handles": [
                    mpl.lines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markersize=float(np.sqrt(area)),
                        markerfacecolor="#bdbdbd",
                        markeredgecolor=edgecolor,
                        markeredgewidth=marker_linewidth,
                    )
                    for area in legend_areas
                ],
                "labels": [f"{value:.0%}" for value in legend_values],
                "title": "Detected cells",
                "frameon": False,
                "borderaxespad": 0.4,
                "handletextpad": 0.4,
                **legend_boxes.legend_kwargs(),
            }
            if owns:
                size_legend = fig.legend(loc="outside right center", **legend_kwargs)
            else:
                size_legend = ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    **legend_kwargs,
                )
        apply_figure_chrome(fig, theme)
        if not size_scale_is_explicit:
            # The panel's final size needs the layout, not rendered pixels.
            fig.draw_without_rendering()
            size_scale = _default_dot_size_scale(
                ax,
                n_x=(
                    len(resolved_feature_order)
                    if swap_axes
                    else len(resolved_group_order)
                ),
                n_y=(
                    len(resolved_group_order)
                    if swap_axes
                    else len(resolved_feature_order)
                ),
            )
            areas = size_scale.areas(plot_df["fraction"].to_numpy(dtype=np.float64))
            sc.set_sizes(areas)
            if size_legend is not None:
                # A legend draws copies of its handles, so resize the copies.
                for handle, area in zip(
                    size_legend.legend_handles,
                    size_scale.areas(legend_values),
                    strict=True,
                ):
                    handle.set_markersize(float(np.sqrt(area)))

    tables = {"aggregate": aggregate}
    if per_sample is not None:
        tables["per_sample"] = per_sample
    n_samples, dropped_sample_cells = _sample_counts(
        store,
        cell_key=cell_key,
        cell_indices=cell_indices if groups is not None else None,
        sample_by=sample_by,
        study_design=study_design,
    )
    assays = _feature_assays(store, features, from_assay)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(
            LegendSpec(
                kind="colorbar",
                label=colorbar_label,
                extras={"vmin": vmin, "vmax": vmax},
            ),
            LegendSpec(
                kind="size",
                label="Detected cells",
                extras={"domain": [0.0, 1.0], "values": legend_values.tolist()},
            ),
        ),
        scales=(
            color_scale,
            size_scale,
            CategoricalScale(
                order=tuple(resolved_group_order),
                palette=(
                    {
                        str(value): color
                        for value, color in categorical_scale.palette.items()
                    }
                    if categorical_scale is not None
                    and categorical_scale.palette is not None
                    else None
                ),
                labels=(
                    {
                        str(value): label
                        for value, label in categorical_scale.labels.items()
                    }
                    if categorical_scale is not None
                    and categorical_scale.labels is not None
                    else None
                ),
                palette_name=(
                    categorical_scale.palette_name
                    if categorical_scale is not None
                    else "default"
                ),
            ),
        ),
        provenance=PlotProvenance(
            assay=next(iter(assays)) if len(assays) == 1 else None,
            cell_key=cell_key if groups is None else None,
            n_cells=len(cell_indices),
            n_samples=n_samples,
            renderer="matplotlib",
            notes=("dotplot",),
            extras={
                "group_by": list(group_keys) if groups is None else None,
                "groups": groups.to_dict() if groups is not None else None,
                "sample_by": (
                    study_design.sample_by if study_design is not None else sample_by
                ),
                "expression_cutoff": expression_cutoff,
                "dropped_sample_cells": dropped_sample_cells,
                "dropped_group_cells": (
                    0 if group_missing is None else int(group_missing.sum())
                ),
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "color_values": color_values,
                "assays": sorted(assays),
                "feature_group_brackets": bracket_count,
                "swap_axes": swap_axes,
                "group_order": resolved_group_order,
                "feature_order": resolved_feature_order,
                "size_scale_source": (
                    "explicit" if size_scale_is_explicit else "panel"
                ),
                "size_range": [size_scale.size_min, size_scale.size_max],
                **unmeasured_extras(unmeasured),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result


@close_figures_on_error
def matrixplot(
    store: Any,
    *,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    group_by: str | tuple[str, ...] | None = None,
    groups: ArtifactRef | None = None,
    cell_key: str = "I",
    from_assay: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    value: str = "mean",
    standardize: str = "none",
    color_scale: ColorScale | None = None,
    feature_order: Sequence[Any] | None = None,
    group_order: Sequence[Any] | None = None,
    cluster_features: bool = False,
    cluster_groups: bool = False,
    cluster_method: str = "average",
    cluster_metric: str = "euclidean",
    row_annotations: Mapping[
        str,
        Mapping[Any, Any] | Sequence[Any],
    ]
    | None = None,
    column_annotations: Mapping[
        str,
        Mapping[Any, Any] | Sequence[Any],
    ]
    | None = None,
    annotation_scales: Mapping[str, CategoricalScale] | None = None,
    categorical_scale: CategoricalScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Heatmap of mean expression or detection fraction by group.

    Rows follow the feature input order and columns follow the group order of
    :func:`dotplot`: ``categorical_scale`` or a stored display order,
    otherwise natural order. Set explicit orders, compared as text, or enable
    hierarchical clustering independently for either axis; a clustered axis
    draws its dendrogram beside the matrix. Group ticks show
    ``categorical_scale.labels`` or the stored display labels.
    ``value="mean"`` colors by average expression; ``value="fraction"`` colors
    by the share of measured cells above ``expression_cutoff``.
    ``standardize="feature"`` applies only to means. ``sample_by`` has the same
    equal-sample weighting behavior as :func:`dotplot`, and shared feature
    labels follow the same pooling rule.

    As in :func:`dotplot`, a mapping of group names to features draws one row
    per feature and group, so a feature listed under two groups keeps a row in
    each, and brackets each group's rows. The matrix table is then indexed by
    ``feature_group`` and ``feature``. Grouped rows keep their groups, so they
    cannot be clustered.
    """
    _, mpl = require_matplotlib()
    if value not in ("mean", "fraction"):
        raise ValueError("value must be 'mean' or 'fraction'")
    if standardize not in ("none", "feature"):
        raise ValueError("standardize must be 'none' or 'feature'")
    if standardize == "feature" and value != "mean":
        raise ValueError("standardize='feature' applies only to value='mean'")
    validate_linkage(cluster_method, cluster_metric)
    grouped_features = isinstance(features, Mapping)
    if grouped_features and cluster_features:
        raise ValueError(
            "cluster_features would mix the rows of feature groups; pass features "
            "as a list to cluster them, or order grouped rows with feature_order"
        )
    color_scale = color_scale or ColorScale(cmap="viridis")
    if color_scale.scale != "linear":
        raise NotImplementedError(
            "matrixplot currently supports only linear color scales"
        )
    normalization = normalization or NormalizationSpec()
    grouping = _resolve_grouping(
        store,
        group_by=group_by,
        groups=groups,
        cell_key=cell_key,
    )
    group_keys, cell_indices, _group_values, group_missing = grouping

    feature_pairs = coerce_feature_list(features)
    _check_feature_count(feature_pairs)
    resolved_features = resolve_feature_batch(
        store, [feature for _, feature in feature_pairs], from_assay=from_assay
    )
    aggregate, per_sample, unmeasured = _summarize_resolved_features(
        store,
        resolved_features,
        [group for group, _ in feature_pairs],
        grouping,
        sample_by=sample_by,
        study_design=study_design,
        normalization=normalization,
        expression_cutoff=expression_cutoff,
    )
    if standardize == "feature":
        aggregate = _standardize_feature(aggregate)
    color_values = "zscore" if standardize == "feature" else value
    if isinstance(group_by, str):
        categorical_scale = resolve_categorical_scale(
            store,
            group_by,
            categorical_scale,
        )
    plot_df = aggregate.copy()
    plot_df["group_label"] = _group_axis_labels(plot_df, _group_roles(group_keys))
    plot_df["row"] = list(
        zip(
            (_feature_group_key(value) for value in plot_df["feature_group"]),
            plot_df["feature"],
            strict=True,
        )
    )
    feature_rows = _feature_rows(
        feature_pairs,
        resolved_features,
        set(plot_df["row"]),
        feature_order,
    )
    observed_group_order = _group_order(
        list(dict.fromkeys(plot_df["group_label"].tolist())),
        group_order,
        categorical_scale,
    )
    # One matrix row per feature and bracket group, as in dotplot.
    row_positions = {row: index for index, row in enumerate(feature_rows)}
    column_positions = {
        label: index for index, label in enumerate(observed_group_order)
    }
    values = np.full((len(feature_rows), len(observed_group_order)), np.nan)
    values[
        plot_df["row"].map(row_positions).to_numpy(dtype=np.int64),
        plot_df["group_label"].map(column_positions).to_numpy(dtype=np.int64),
    ] = plot_df[color_values].to_numpy(dtype=np.float64)
    mat = pd.DataFrame(
        values,
        index=(
            pd.MultiIndex.from_tuples(feature_rows, names=["feature_group", "feature"])
            if grouped_features
            else pd.Index([label for _, label in feature_rows], name="feature")
        ),
        columns=pd.Index(observed_group_order, name="group"),
    )
    row_annotation_values = normalize_annotations(
        [label for _, label in feature_rows],
        row_annotations,
        axis_name="row",
    ).set_axis(mat.index, axis=0)
    column_annotation_values = normalize_annotations(
        list(mat.columns),
        column_annotations,
        axis_name="column",
    )
    mat, row_linkage, column_linkage = order_heatmap(
        mat,
        row_order=None if feature_order is None else list(mat.index),
        column_order=None if group_order is None else observed_group_order,
        cluster_rows=cluster_features,
        cluster_columns=cluster_groups,
        method=cluster_method,
        metric=cluster_metric,
    )
    # Name the axes by role, as the aggregate table names its columns.
    mat = mat.rename_axis(
        index=["feature_group", "feature"] if grouped_features else "feature",
        columns="group",
    )
    resolved_rows = list(mat.index)
    resolved_feature_order = [
        row[1] if grouped_features else row for row in resolved_rows
    ]
    resolved_group_order = list(mat.columns)
    row_annotation_values = row_annotation_values.reindex(resolved_rows).rename_axis(
        mat.index.names
    )
    column_annotation_values = column_annotation_values.reindex(
        resolved_group_order
    ).rename_axis("group")
    row_colors, row_annotation_scales = annotation_colors(
        row_annotation_values,
        annotation_scales,
    )
    column_colors, column_annotation_scales = annotation_colors(
        column_annotation_values,
        annotation_scales,
    )
    resolved_annotation_scales = row_annotation_scales + column_annotation_scales

    panel_key: Hashable = "matrixplot"
    resolved_figsize = figsize
    if resolved_figsize is None and target is None:
        # Extra height keeps rotated group labels from clipping.
        resolved_figsize = (
            max(4.5, 0.45 * len(resolved_group_order) + 2.2),
            max(4.2, 0.45 * len(resolved_feature_order) + 2.8),
        )
    data = mat.to_numpy(dtype=np.float64)
    vmin, vmax = resolve_color_limits(data, color_scale)
    norm = continuous_norm(
        mpl,
        vmin=vmin,
        vmax=vmax,
        vcenter=color_scale.vcenter,
    )

    with theme_context(theme):
        fig, axes, owns = normalize_axes_target(
            target,
            panel_keys=[panel_key],
            figsize=resolved_figsize,
        )
        ax = axes[panel_key]
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=color_scale.cmap or "viridis",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(resolved_group_order)))
        ax.set_xticklabels(
            _group_tick_labels(resolved_group_order, categorical_scale),
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(resolved_feature_order)))
        ax.set_yticklabels(resolved_feature_order)
        annotation_xlim, annotation_ylim = draw_annotation_strips(
            ax,
            row_colors=row_colors,
            column_colors=column_colors,
            n_rows=len(resolved_feature_order),
            n_columns=len(resolved_group_order),
        )
        ax.set_xlim(annotation_xlim)
        ax.set_ylim(annotation_ylim)
        # Clustered axes show their hierarchy beside the matrix.
        dendrogram_color = foreground_color(theme)
        if column_linkage is not None:
            axes["column_dendrogram"] = _draw_dendrogram(
                ax, column_linkage, side="top", color=dendrogram_color
            )
        if row_linkage is not None:
            axes["row_dendrogram"] = _draw_dendrogram(
                ax, row_linkage, side="right", color=dendrogram_color
            )
        bracket_count = (
            _draw_feature_group_brackets(
                ax,
                [group for group, _ in resolved_rows],
                swap_axes=False,
            )
            if grouped_features
            else 0
        )
        colorbar_label = (
            "Mean expression"
            if value == "mean" and standardize == "none"
            else "Standardized mean expression"
            if value == "mean"
            else "Detected cells"
        )
        if show_legend:
            cb = fig.colorbar(
                im,
                ax=ax,
                location="top",
                orientation="horizontal",
                shrink=0.8,
                fraction=0.06,
                pad=0.04,
            )
            cb.set_label(colorbar_label)
            annotation_handles = annotation_legend_handles(
                mpl,
                list(row_annotation_values.columns)
                + list(column_annotation_values.columns),
                resolved_annotation_scales,
            )
            if annotation_handles:
                legend_kwargs = {
                    "handles": annotation_handles,
                    "title": "Annotations",
                    "frameon": False,
                }
                if owns:
                    fig.legend(loc="outside right center", **legend_kwargs)
                else:
                    ax.legend(
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        borderaxespad=0,
                        **legend_kwargs,
                    )
        apply_figure_chrome(fig, theme)

    tables = {"aggregate": aggregate, "matrix": mat}
    if per_sample is not None:
        tables["per_sample"] = per_sample
    if not row_annotation_values.empty:
        tables["row_annotations"] = row_annotation_values
    if not column_annotation_values.empty:
        tables["column_annotations"] = column_annotation_values
    n_samples, dropped_sample_cells = _sample_counts(
        store,
        cell_key=cell_key,
        cell_indices=cell_indices if groups is not None else None,
        sample_by=sample_by,
        study_design=study_design,
    )
    assays = _feature_assays(store, features, from_assay)

    result = PlotResult(
        figure=fig,
        axes=axes,
        tables=tables,
        legends=(
            LegendSpec(
                kind="colorbar",
                label=colorbar_label,
                extras={"vmin": vmin, "vmax": vmax},
            ),
            *(
                LegendSpec(kind="categorical", label=name)
                for name in list(row_annotation_values.columns)
                + list(column_annotation_values.columns)
            ),
        ),
        scales=(color_scale, *resolved_annotation_scales),
        provenance=PlotProvenance(
            assay=next(iter(assays)) if len(assays) == 1 else None,
            cell_key=cell_key if groups is None else None,
            n_cells=len(cell_indices),
            n_samples=n_samples,
            renderer="matplotlib",
            notes=("matrixplot", value),
            extras={
                "group_by": list(group_keys) if groups is None else None,
                "groups": groups.to_dict() if groups is not None else None,
                "sample_by": (
                    study_design.sample_by if study_design is not None else sample_by
                ),
                "expression_cutoff": expression_cutoff,
                "dropped_sample_cells": dropped_sample_cells,
                "dropped_group_cells": (
                    0 if group_missing is None else int(group_missing.sum())
                ),
                "normalization": {
                    "source": normalization.source,
                    "transform": normalization.transform,
                },
                "color_values": color_values,
                "assays": sorted(assays),
                "feature_order": resolved_feature_order,
                "group_order": resolved_group_order,
                "feature_group_brackets": bracket_count,
                "cluster_features": row_linkage is not None,
                "cluster_groups": column_linkage is not None,
                "cluster_method": cluster_method,
                "cluster_metric": cluster_metric,
                "row_annotations": list(row_annotation_values.columns),
                "column_annotations": list(column_annotation_values.columns),
                **unmeasured_extras(unmeasured),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        result.show()
    return result
