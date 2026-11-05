"""The figure of a cell phylogeny and the clones cut from it."""

from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument
from ._charts import draw_tree, tree_drawing_bytes
from ._clone_names import clone_names
from ._contracts import CategoricalScale
from ._data import store_method
from ._deps import require_matplotlib
from ._figure import (
    PlotResult,
    close_figures_on_error,
    finish_plot,
    normalize_axes_target,
)
from ._style import (
    CUSTOM_PALETTES,
    MAX_FIGURE_WIDTH_INCHES,
    MAX_NAMED_ITEMS,
    NO_VALUE_COLOR,
    figure_size,
    foreground_color,
    theme_context,
)

__all__ = ["phylogeny"]

# The wedges that a figure draws unless told otherwise.
_MAX_LEAVES = 100
# A clone takes a colour of a palette without greys by its rank among the
# cut's clones, the largest first, so it keeps its colour whatever else is
# drawn; a wedge of several clones is grey, and cells without a clone take
# the missing grey.
_CLONE_COLORS = tuple(
    color
    for color in CUSTOM_PALETTES[20]
    if len({color[1:3], color[3:5], color[5:7]}) > 1
)
_SEVERAL_CLONES_COLOR = "#969696"
_SEVERAL = -2
_NO_CLONE = -1
_PANEL_INCHES = (MAX_FIGURE_WIDTH_INCHES, 5.0)
# Where each row stands: its position, branch, end, middle and extent.
_PLACES = ("x", "start", "end", "y", "low", "high")


def _kinds(nodes: pd.DataFrame) -> np.ndarray:
    """Each row's colour category: its one clone's code, several clones or none."""
    n_clones = nodes["n_clones"].to_numpy(dtype=np.int64)
    codes = nodes["clone"].to_numpy(dtype=np.int64)
    return np.where(n_clones >= 2, _SEVERAL, np.where(n_clones == 1, codes, _NO_CLONE))


def _counted(nodes: pd.DataFrame) -> np.ndarray:
    """Whether each row is a wedge that says how many clones or trees it holds.

    A wedge of several clones or trees says so where it is at least a
    fortieth of the height, so at most 40 do.
    """
    wedge = nodes["wedge"].to_numpy(dtype=bool)
    extent = nodes["high"].to_numpy() - nodes["low"].to_numpy()
    tall = extent >= float(nodes["high"].max()) / MAX_NAMED_ITEMS
    grouped = (nodes["n_clones"].to_numpy() >= 2) | (nodes["n_trees"].to_numpy() > 1)
    return np.asarray(wedge & tall & grouped, dtype=bool)


def _drawing_bytes(nodes: pd.DataFrame) -> int:
    """What drawing the collapsed ``nodes`` holds: their rows, legend entries and texts."""
    kinds = _kinds(nodes)[nodes["wedge"].to_numpy(dtype=bool)]
    return tree_drawing_bytes(
        len(nodes),
        n_entries=len(pd.unique(kinds)),
        n_texts=int(np.count_nonzero(_counted(nodes))),
    )


def _cut_words(collapsed: Any) -> str:
    cut = collapsed.cut
    if cut.time is not None:
        return f"time {cut.time:g} {collapsed.time_unit}"
    return f"depth {cut.depth}"


@close_figures_on_error
def phylogeny(
    store: Any,
    clones: ArtifactRef,
    *,
    tree: str | None = None,
    max_leaves: int = _MAX_LEAVES,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the cell phylogeny that clones were cut from, its clades collapsed to at most ``max_leaves`` wedges.

    ``clones`` are clones cut from a phylogeny by ``ds.clones.phylogeny_clones``,
    whose phylogeny and cut the figure reads from their provenance. Each tree
    starts as one wedge, and the clade that holds the most cells opens until
    ``max_leaves`` wedges are drawn: its child of the most cells is drawn
    alone and its other children as one wedge until they open in turn, so a
    polytomy opens one child at a time. A clade of the cut is one clone and
    never opens, so a cut into fewer clades than ``max_leaves`` draws them
    all. A clade is drawn from its first branching, and leaves that name no
    cell of the phylogeny's selection are left out.

    The trees run from their roots at the left, each wedge as tall as its
    cells and stacked from the top, the most cells first. A wedge of one
    clone takes that clone's colour, by its rank among the cut's clones; a
    wedge of several is grey and says how many it holds where it is tall
    enough, at least a fortieth of the height; and cells whose leaves end
    before the cut, in no clone, take the missing grey. The x axis is time
    for a cut at a time and branchings below the root, single-child nodes
    merged, for a cut at a depth, with the cut as a dashed line. A forest of
    more trees than ``max_leaves`` draws its largest trees and the others as
    one grey wedge that says how many trees it holds; ``tree`` draws one tree
    alone. The legend lists the clones drawn, the largest first, after the
    greys.

    Tables: ``nodes``, one row per node or group drawn, in drawing order:
    ``tree``, ``node`` (its position in its tree, as the phylogeny's
    ``nodes`` table numbers them; -1 for a group of sibling clades or trees),
    ``parent`` (the row it hangs from, -1 for none), ``wedge`` (whether it is
    drawn collapsed), ``n_cells``, ``n_clones``, ``clone`` and ``label`` (its
    one clone's ID and the text drawn, None for none or several),
    ``n_clades`` (the clades that a group holds, 1 otherwise), ``n_trees``,
    ``x`` (its time or depth, a group's earliest clade's), ``start`` (where
    its branch starts), ``end`` (its latest leaf), and ``y``, ``low`` and
    ``high``, its middle and extent in cells from the top.

    Raises:
        TypeError: If ``clones`` is not an ArtifactRef, ``tree`` is not text,
            ``max_leaves`` is not an integer, another argument has the wrong
            type, or the store does not draw phylogenies.
        ValueError: If ``clones`` were not cut from a phylogeny,
            ``max_leaves`` is below 1, or ``tree`` names no tree of the
            phylogeny or one whose leaves name no cell.
        KeyError: If the store does not hold the clones or their phylogeny.
        RuntimeError: If the clones or their phylogeny are incomplete.
        MemoryError: If reading the phylogeny, collapsing it or drawing it
            does not fit the memory budget.
    """
    show_legend = bool_argument(show_legend, "show_legend")
    show = bool_argument(show, "show")
    max_leaves = integer_argument(max_leaves, "max_leaves", minimum=1)
    if not isinstance(clones, ArtifactRef):
        raise TypeError("clones must be an ArtifactRef")
    collapsed = store_method(store, "_phylogeny_figure", "the figure of a phylogeny")(
        clones, tree=tree, max_leaves=max_leaves, drawing_bytes=_drawing_bytes
    )
    names = clone_names(store, clones)
    nodes = collapsed.nodes
    wedge = nodes["wedge"].to_numpy(dtype=bool)
    n_clones = nodes["n_clones"].to_numpy(dtype=np.int64)
    codes = nodes["clone"].to_numpy(dtype=np.int64)
    kinds = _kinds(nodes)
    # Each clone drawn, the largest first, labelled as figures label clones.
    one = n_clones == 1
    drawn_codes, first, at = np.unique(
        codes[one], return_index=True, return_inverse=True
    )
    drawn_labels = np.asarray(
        names.labels(collapsed.clone_ids[one][first].tolist()), dtype=object
    )
    row_labels = np.full(len(nodes), None, dtype=object)
    row_labels[one] = drawn_labels[at]
    greys = [kind for kind in (_SEVERAL, _NO_CLONE) if (kinds[wedge] == kind).any()]
    code_list = drawn_codes.tolist()
    colors = CategoricalScale(
        order=(*greys, *code_list),
        palette={
            _SEVERAL: _SEVERAL_CLONES_COLOR,
            _NO_CLONE: NO_VALUE_COLOR,
            **{code: _CLONE_COLORS[code % len(_CLONE_COLORS)] for code in code_list},
        },
        labels={
            _SEVERAL: f"Several {names.plural}",
            _NO_CLONE: f"No {names.singular}",
            **dict(zip(code_list, drawn_labels.tolist(), strict=True)),
        },
    )
    # A tall wedge of several clones or trees says how many it holds.
    counted = np.flatnonzero(_counted(nodes))
    trees = nodes["n_trees"].to_numpy(dtype=np.int64)
    texts = np.full(len(nodes), "", dtype=object)
    texts[counted] = [
        (f"{trees[row]:,} trees, " if trees[row] > 1 else "")
        + f"{n_clones[row]:,} {names.plural}"
        for row in counted.tolist()
    ]
    # The table of the figure, once, its texts as objects with None where a
    # row has no tree or clone, which pandas 3 would read as its string type.
    table = pd.DataFrame(
        {
            column: (
                pd.Series(values, dtype=object, copy=False)
                if column in ("tree", "clone", "label")
                else values
            )
            for column, values in (
                ("tree", collapsed.tree_names),
                ("node", nodes["node"].to_numpy()),
                ("parent", nodes["parent"].to_numpy()),
                ("wedge", wedge),
                ("n_cells", nodes["n_cells"].to_numpy()),
                ("n_clones", n_clones),
                ("clone", collapsed.clone_ids),
                ("label", row_labels),
                ("n_clades", nodes["n_clades"].to_numpy()),
                ("n_trees", trees),
                *((column, nodes[column].to_numpy()) for column in _PLACES),
            )
        },
        copy=False,
    )
    top = nodes["parent"].to_numpy() < 0
    noun = names.singular if collapsed.n_clones == 1 else names.plural
    title = f"{collapsed.n_clones:,} {noun} cut at {_cut_words(collapsed)}"
    if collapsed.tree is not None:
        title = f"{collapsed.tree}: {title}"
    if figsize is None and target is None:
        figsize, _ = figure_size(1, panel=_PANEL_INCHES)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["tree"], figsize=figsize
        )
        drawn = draw_tree(
            axes["tree"],
            table,
            categories=kinds,
            colors=colors,
            line_color=foreground_color(theme),
            x_label=(
                f"Time ({collapsed.time_unit})"
                if collapsed.cut.time is not None
                else "Branchings below the root"
            ),
            texts=texts,
            cut=(
                collapsed.cut.time
                if collapsed.cut.time is not None
                else float(collapsed.cut.depth)
            ),
            title=title,
            legend_title=names.singular.capitalize(),
            show_legend=show_legend,
        )
        if collapsed.cut.time is None:
            _, mpl = require_matplotlib()
            axes["tree"].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    return finish_plot(
        figure,
        axes,
        owns,
        name="phylogeny",
        theme=theme,
        tables={"nodes": table},
        drawn=[drawn],
        inputs={"clones": collapsed.clones, "phylogeny": collapsed.phylogeny},
        settings={
            "tree": collapsed.tree,
            "max_leaves": max_leaves,
            "time": collapsed.cut.time,
            "depth": collapsed.cut.depth,
        },
        n_cells=int(nodes["n_cells"].to_numpy()[top].sum()),
        show=show,
    )
