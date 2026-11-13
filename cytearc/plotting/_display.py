"""Resolve stored display metadata into plotting scale contracts."""

from typing import Any

from ..metadata.artifacts import (
    artifact_categories,
    column_category_order,
    column_display,
    validate_display_metadata,
)
from ..storage.refs import ArtifactRef
from ._contracts import CategoricalScale, ColorScale


def stored_display_metadata(store: Any, column: str) -> dict[str, Any] | None:
    """Return validated display metadata for a cell column, when available."""
    frozen_display = getattr(store, "_stored_display_metadata", None)
    if callable(frozen_display):
        display = frozen_display(column)
        return None if display is None else validate_display_metadata(display)
    try:
        root = store.zw
    except AttributeError:
        return None
    return column_display(root, column)


def stored_category_order(store: Any, column: str) -> tuple[Any, ...] | None:
    """Return the categories of a live cell column stored as ordered, in their stored order.

    A run's plot store returns None: a run records the order of a column it
    freezes in the display of the column's field, which
    :func:`stored_display_metadata` reads.
    """
    if callable(getattr(store, "_stored_display_metadata", None)):
        return None
    try:
        root = store.zw
    except AttributeError:
        return None
    return column_category_order(root, column)


def categorical_display_scale(display: dict[str, Any]) -> CategoricalScale:
    """Build the categorical scale a stored display contract describes.

    A display without a missing color or label leaves that field unset.
    """
    categories = display["categories"]
    missing_color = display.get("missing_color")
    missing_label = display.get("missing_label")
    return CategoricalScale(
        order=tuple(category["value"] for category in categories),
        palette={category["value"]: str(category["color"]) for category in categories},
        labels={category["value"]: str(category["label"]) for category in categories},
        missing_color=None if missing_color is None else str(missing_color),
        missing_label=None if missing_label is None else str(missing_label),
    )


def continuous_display_scale(display: dict[str, Any]) -> ColorScale:
    """Build the color scale a stored continuous display contract describes.

    The stored minimum and maximum become fixed limits only when both exist
    and span a range; otherwise limits come from the plotted values.
    """
    minimum = display["minimum"]
    maximum = display["maximum"]
    fixed = minimum is not None and maximum is not None and maximum > minimum
    return ColorScale(
        cmap=str(display["colormap"]),
        vmin=float(minimum) if fixed else None,
        vmax=float(maximum) if fixed else None,
        scale=display["scale"],
    )


def stored_categorical_scale(store: Any, column: str) -> CategoricalScale | None:
    """Return the category order and colors that a cell column stores, or None.

    A stored categorical display sets the order, colors, labels, and missing
    color and label. Without a display, a column stored as ordered keeps its
    categories in their stored order, and generated colors follow that
    order. Any other column, a continuous display included, stores none, so
    plots sort its categories naturally.
    """
    display = stored_display_metadata(store, column)
    return _stored_scale(
        display,
        None if display is not None else stored_category_order(store, column),
    )


def stored_artifact_scale(store: Any, ref: ArtifactRef) -> CategoricalScale | None:
    """Return the category order and colors that a label artifact stores, or None.

    The artifact's value array records them as a cell column does: an H5AD
    import keeps the categories, order, and colors of each clustering it
    imports there. Artifacts are immutable, so a run's plot store reads them
    as a live store does.
    """
    try:
        root = store.zw
    except AttributeError:
        return None
    recorded = artifact_categories(root, ref)
    return _stored_scale(recorded.display, recorded.ordered_levels)


def _stored_scale(
    display: dict[str, Any] | None, order: tuple[Any, ...] | None
) -> CategoricalScale | None:
    """The scale of a stored display, else of an order stored as ordered."""
    if display is not None:
        if display["kind"] != "categorical":
            return None
        return categorical_display_scale(display)
    return None if order is None else CategoricalScale(order=order)


def merge_categorical_scales(
    stored: CategoricalScale | None,
    explicit: CategoricalScale | None,
) -> CategoricalScale | None:
    """Merge a caller's scale over a stored one, field by field.

    A field that ``explicit`` leaves unset keeps the stored value, and
    ``labels`` merge category by category, so renaming one category keeps
    the stored labels of the others. A ``palette_name`` asks for generated
    colors, so the stored palette yields to it while the stored order stays.
    A ``palette`` replaces the stored one whole.
    """
    if stored is None or explicit is None:
        return explicit if stored is None else stored
    keeps_stored_colors = explicit.palette is None and explicit.palette_name is None
    labels = (
        None
        if stored.labels is None and explicit.labels is None
        else {**(stored.labels or {}), **(explicit.labels or {})}
    )
    return CategoricalScale(
        order=stored.order if explicit.order is None else explicit.order,
        palette=stored.palette if keeps_stored_colors else explicit.palette,
        labels=labels,
        missing_color=(
            stored.missing_color
            if explicit.missing_color is None
            else explicit.missing_color
        ),
        missing_label=(
            stored.missing_label
            if explicit.missing_label is None
            else explicit.missing_label
        ),
        palette_name=(
            stored.palette_name
            if explicit.palette_name is None
            else explicit.palette_name
        ),
    )


def resolve_categorical_scale(
    store: Any,
    column: str,
    explicit: CategoricalScale | None,
) -> CategoricalScale | None:
    """Merge an explicit scale over the scale that a cell column stores."""
    return merge_categorical_scales(stored_categorical_scale(store, column), explicit)


def resolve_artifact_scale(
    store: Any,
    ref: ArtifactRef,
    explicit: CategoricalScale | None,
) -> CategoricalScale | None:
    """Merge an explicit scale over the scale that a label artifact stores."""
    return merge_categorical_scales(stored_artifact_scale(store, ref), explicit)
