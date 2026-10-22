"""On-data category labels that stay on their own clusters."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

# A label never moves more than this many label heights from its anchor.
MAX_SHIFT_HEIGHTS = 1.0
# A moved label stays this much nearer its own centre than the halfway point
# to another centre, so it is always strictly nearest its own cluster.
_CENTRE_MARGIN = 0.9
# Room in points around each label for its halo stroke.
_LABEL_PADDING_POINTS = 1.0
_MAX_ITERATIONS = 200


@dataclass(frozen=True, slots=True)
class _LabelLayout:
    """The labels of one panel and the centres of all of its clusters."""

    texts: tuple[Any, ...]
    centres: np.ndarray


_LAYOUT_LABELS: WeakKeyDictionary[Any, _LabelLayout] = WeakKeyDictionary()


def _feasible_fraction(
    anchor: np.ndarray,
    shift: np.ndarray,
    *,
    max_shift: float,
    others: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> float:
    """Largest fraction of ``shift`` that keeps a label within its limits.

    The limits are convex and contain the anchor, which is the label's own
    centre: at most ``max_shift`` away from it, strictly nearer to it than to
    each of ``others``, and inside ``bounds`` (left, bottom, right, top) on
    each axis where the anchor itself is.
    """
    length = float(np.hypot(*shift))
    if length == 0.0:
        return 0.0
    fraction = min(1.0, max_shift / length)
    offsets = others - anchor
    toward = offsets @ shift
    # A point anchor + t * shift is nearer the anchor than the centre at
    # anchor + offset exactly while t < |offset|^2 / (2 shift . offset).
    approaching = toward > 0
    if approaching.any():
        limits = (
            _CENTRE_MARGIN
            * np.einsum("ij,ij->i", offsets[approaching], offsets[approaching])
            / (2.0 * toward[approaching])
        )
        fraction = min(fraction, float(limits.min()))
    for axis, (low, high) in enumerate(
        ((bounds[0], bounds[2]), (bounds[1], bounds[3]))
    ):
        if not low <= anchor[axis] <= high:
            continue
        if shift[axis] > 0:
            fraction = min(fraction, (high - anchor[axis]) / shift[axis])
        elif shift[axis] < 0:
            fraction = min(fraction, (low - anchor[axis]) / shift[axis])
    return max(fraction, 0.0)


def separate_labels(
    anchors: np.ndarray,
    sizes: np.ndarray,
    centres: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    max_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Move overlapping label boxes apart within bounded, cluster-safe limits.

    ``anchors`` holds each label's position at its cluster's centre, and
    ``sizes`` the width and height of its box, in one display unit.
    ``centres`` lists every cluster's centre with the labelled clusters first,
    in the order of ``anchors``. Overlapping boxes push each other apart,
    along the axis that separates them with the smaller move. Each label then
    stays within ``max_shift`` of its anchor (one label height by default),
    strictly nearer its own centre than any other cluster's, and with its box
    inside ``bounds`` (left, bottom, right, top) on each axis where the box
    fits at its anchor. Labels that these limits hold in place may still
    overlap.
    """
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1, 2)
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    n_labels = len(anchors)
    shifts = (
        MAX_SHIFT_HEIGHTS * sizes[:, 1]
        if max_shift is None
        else np.asarray(max_shift, dtype=np.float64)
    )
    positions = anchors.copy()
    if n_labels < 2:
        return positions
    # Ties between labels at one position break by anchor, then by index.
    order = np.arange(n_labels)
    tie_x = np.sign(np.subtract.outer(order, order)).astype(np.float64)
    tie_x = np.where(
        anchors[:, None, 0] != anchors[None, :, 0],
        np.sign(anchors[:, None, 0] - anchors[None, :, 0]),
        tie_x,
    )
    tie_y = np.where(
        anchors[:, None, 1] != anchors[None, :, 1],
        np.sign(anchors[:, None, 1] - anchors[None, :, 1]),
        np.sign(np.subtract.outer(order, order)).astype(np.float64),
    )
    half = sizes / 2.0
    others = [np.delete(centres, index, axis=0) for index in range(n_labels)]
    # Each label's centre stays where its whole box fits inside the bounds.
    label_bounds = [
        (
            bounds[0] + half[index, 0],
            bounds[1] + half[index, 1],
            bounds[2] - half[index, 0],
            bounds[3] - half[index, 1],
        )
        for index in range(n_labels)
    ]
    upper = ~np.tri(n_labels, dtype=bool)
    for _ in range(_MAX_ITERATIONS):
        delta = positions[:, None, :] - positions[None, :, :]
        overlap_x = half[:, None, 0] + half[None, :, 0] - np.abs(delta[..., 0])
        overlap_y = half[:, None, 1] + half[None, :, 1] - np.abs(delta[..., 1])
        overlapping = (overlap_x > 0) & (overlap_y > 0) & upper
        if not overlapping.any():
            break
        direction_x = np.where(delta[..., 0] != 0, np.sign(delta[..., 0]), tie_x)
        direction_y = np.where(delta[..., 1] != 0, np.sign(delta[..., 1]), tie_y)
        along_y = overlap_y <= overlap_x
        push = np.zeros((n_labels, n_labels, 2))
        push[..., 1] = np.where(overlapping & along_y, overlap_y / 2 * direction_y, 0)
        push[..., 0] = np.where(overlapping & ~along_y, overlap_x / 2 * direction_x, 0)
        # Each pair moves its first label by +push and its second by -push.
        candidate = positions + push.sum(axis=1) - push.sum(axis=0)
        moved = False
        for index in range(n_labels):
            shift = candidate[index] - anchors[index]
            fraction = _feasible_fraction(
                anchors[index],
                shift,
                max_shift=float(shifts[index]),
                others=others[index],
                bounds=label_bounds[index],
            )
            position = anchors[index] + fraction * shift
            if not np.allclose(position, positions[index]):
                moved = True
            positions[index] = position
        if not moved:
            break
    return positions


def register_layout_labels(
    ax: Any,
    texts: Sequence[Any],
    centres: np.ndarray,
) -> None:
    """Mark on-data labels to separate once the panel's layout is final.

    ``centres`` lists every drawn cluster's centre in data coordinates, the
    labelled clusters first in the order of ``texts``.
    """
    _LAYOUT_LABELS[ax] = _LabelLayout(
        texts=tuple(texts),
        centres=np.asarray(centres, dtype=np.float64).reshape(-1, 2),
    )


def place_layout_labels(figure: Any) -> None:
    """Separate registered on-data labels in their resolved panels.

    The layout is resolved without rendering. Labels restart from their
    cluster centres, so placing them again after a new layout gives the same
    result as placing them once.
    """
    marked = [
        (ax, layout)
        for ax in figure.axes
        if (layout := _LAYOUT_LABELS.get(ax)) is not None and layout.texts
    ]
    if not marked:
        return
    # Text extents come from the renderer the layout pass used, whatever the
    # backend.
    figure.draw_without_rendering()
    padding = 2.0 * _LABEL_PADDING_POINTS * figure.dpi / 72.0
    for ax, layout in marked:
        n_labels = len(layout.texts)
        for text, centre in zip(layout.texts, layout.centres, strict=False):
            text.set_position((float(centre[0]), float(centre[1])))
        extents = [text.get_window_extent() for text in layout.texts]
        sizes = np.asarray(
            [(extent.width + padding, extent.height + padding) for extent in extents]
        )
        centres = ax.transData.transform(layout.centres)
        box = ax.get_window_extent()
        positions = separate_labels(
            centres[:n_labels],
            sizes,
            centres,
            bounds=(box.x0, box.y0, box.x1, box.y1),
        )
        for text, position in zip(
            layout.texts,
            ax.transData.inverted().transform(positions),
            strict=True,
        ):
            text.set_position((float(position[0]), float(position[1])))
