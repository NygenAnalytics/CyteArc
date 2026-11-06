"""The figure of a cell phylogeny and the clones cut from it."""

from typing import Any

import numpy as np
import pandas as pd

from ..clones.phylogeny import CollapsedPhylogeny
from ..storage.refs import ArtifactRef
from ..utils.arguments import bool_argument, integer_argument
from ..utils.arrays import sorted_unique
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
_MIXED_COLOR = "#969696"
# A clone takes a colour by its rank among the cut's clones, the largest
# first, so it keeps its colour whatever else is drawn; a wedge whose cells
# are not all in one clone is grey, and a wedge of cells in no clone takes
# the missing grey. The clones' colours are those of the 20-colour palette
# and then of the 102-colour one that stand at least _FROM_GREY from both
# greys and _APART from every colour before them, by their distance in
# CIELAB, so that no clone looks grey or like another: 56 colours, which
# repeat from the 57th clone, past the 40 that a legend lists at most.
_FROM_GREY = 25.0
_APART = 20.0
# sRGB's linear red, green and blue to CIE XYZ, and the D65 white point.
_SRGB_TO_XYZ = np.asarray(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
_D65_WHITE = np.asarray([0.95047, 1.0, 1.08883])


def _lab(color: str) -> np.ndarray:
    """A colour's CIELAB lightness and its a and b coordinates."""
    rgb = np.asarray([int(color[at : at + 2], 16) / 255 for at in (1, 3, 5)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = _SRGB_TO_XYZ @ linear / _D65_WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.asarray([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _clone_colors() -> tuple[str, ...]:
    """The clones' colours by rank: those far from the greys and from every one before."""
    greys = [_lab(NO_VALUE_COLOR), _lab(_MIXED_COLOR)]
    kept: dict[str, np.ndarray] = {}
    for color in (*CUSTOM_PALETTES[20], *CUSTOM_PALETTES[102]):
        point = _lab(color)
        if all(np.linalg.norm(point - grey) >= _FROM_GREY for grey in greys) and all(
            np.linalg.norm(point - other) >= _APART for other in kept.values()
        ):
            kept[color.lower()] = point
    return tuple(kept)


_CLONE_COLORS = _clone_colors()
_MIXED = -2
_NO_CLONE = -1
_PANEL_INCHES = (MAX_FIGURE_WIDTH_INCHES, 5.0)
# Where each row stands: its position, branch, end, middle and extent.
_PLACES = ("x", "start", "end", "y", "low", "high")


def _kinds(collapsed: CollapsedPhylogeny) -> np.ndarray:
    """Each row's colour category: its clone's code where all its cells are in that clone, none where none are in a clone, and mixed otherwise.

    A row of one clone whose other cells end before the cut is mixed, so the
    clone's colour never stands for cells outside it.
    """
    nodes = collapsed.nodes
    n_clones = nodes["n_clones"].to_numpy(dtype=np.int64)
    whole = (n_clones == 1) & (nodes["n_cells"].to_numpy() == collapsed.clone_sizes)
    return np.where(
        whole,
        nodes["clone"].to_numpy(dtype=np.int64),
        np.where(n_clones == 0, _NO_CLONE, _MIXED),
    )


def _tall(nodes: pd.DataFrame) -> np.ndarray:
    """Whether each row is a wedge at least a fortieth of the height, so at most 40 are."""
    wedge = nodes["wedge"].to_numpy(dtype=bool)
    extent = nodes["high"].to_numpy() - nodes["low"].to_numpy()
    return np.asarray(
        wedge & (extent >= float(nodes["high"].max()) / MAX_NAMED_ITEMS), dtype=bool
    )


def _counted(collapsed: CollapsedPhylogeny, kinds: np.ndarray) -> np.ndarray:
    """Whether each row is a tall wedge that says what it holds: a mixed wedge, or a group of trees."""
    trees = collapsed.nodes["n_trees"].to_numpy() > 1
    return np.asarray(_tall(collapsed.nodes) & ((kinds == _MIXED) | trees), dtype=bool)


def _listed(collapsed: CollapsedPhylogeny, kinds: np.ndarray) -> np.ndarray:
    """The categories that the legend lists: the greys drawn, then the clone of each tall wedge, the largest first.

    Where no wedge of one clone is tall enough, the legend lists the largest
    clone drawn, so it lists one at least.
    """
    wedge = collapsed.nodes["wedge"].to_numpy(dtype=bool)
    drawn = kinds[wedge]
    greys = [kind for kind in (_MIXED, _NO_CLONE) if (drawn == kind).any()]
    clones = sorted_unique(kinds[_tall(collapsed.nodes) & (kinds >= 0)])
    if not len(clones) and (drawn >= 0).any():
        clones = np.asarray([drawn[drawn >= 0].min()])
    return np.asarray([*greys, *clones.tolist()], dtype=np.int64)


def _drawing_bytes(collapsed: CollapsedPhylogeny) -> int:
    """What drawing ``collapsed`` holds: its rows, legend entries and texts."""
    kinds = _kinds(collapsed)
    return tree_drawing_bytes(
        len(collapsed.nodes),
        n_entries=len(_listed(collapsed, kinds)),
        n_texts=int(np.count_nonzero(_counted(collapsed, kinds))),
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
    never opens, nor does a leaf that ends before the cut, so where the trees
    hold no more of these than ``max_leaves``, every clone is drawn alone. A
    clade is drawn from its first branching, and leaves that name no cell of
    the phylogeny's selection are left out.

    The trees run from their roots at the left, each wedge as tall as its
    cells and stacked from the top, the most cells first. A wedge whose
    cells are all in one clone takes that clone's colour, by its rank among
    the cut's clones, from 56 colours far from grey and from each other,
    which repeat from the 57th clone; a wedge whose cells all end before the
    cut, in no clone, takes the missing grey; and every other wedge, of
    several clones or of a clone and cells before the cut, is grey, "Mixed"
    in the legend, and says how many clones it holds where it is tall
    enough, at least a fortieth of the height. The x axis is time for a cut at a time and
    branchings below the root, single-child nodes merged, for a cut at a
    depth, with the cut as a dashed line. A forest of more trees than
    ``max_leaves`` draws its largest trees and the others as one grey wedge
    that says how many trees it holds; ``tree`` draws one tree alone. The
    legend lists the greys drawn and then, the largest first, the clones
    whose wedges are at least a fortieth of the height, or the largest clone
    drawn when none is, and its title says how many of the greys and clones
    drawn it lists when it leaves some out.

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
    kinds = _kinds(collapsed)
    # Each clone that a row holds alone, the largest first, labelled as
    # figures label clones.
    one = n_clones == 1
    drawn_codes, first, at = np.unique(
        codes[one], return_index=True, return_inverse=True
    )
    drawn_labels = np.asarray(
        names.labels(collapsed.clone_ids[one][first].tolist()), dtype=object
    )
    row_labels = np.full(len(nodes), None, dtype=object)
    row_labels[one] = drawn_labels[at]
    code_list = drawn_codes.tolist()
    colors = CategoricalScale(
        order=tuple(_listed(collapsed, kinds).tolist()),
        palette={
            _MIXED: _MIXED_COLOR,
            _NO_CLONE: NO_VALUE_COLOR,
            **{code: _CLONE_COLORS[code % len(_CLONE_COLORS)] for code in code_list},
        },
        labels={
            _MIXED: "Mixed",
            _NO_CLONE: f"No {names.singular}",
            **dict(zip(code_list, drawn_labels.tolist(), strict=True)),
        },
    )
    # A tall wedge of mixed cells or of several trees says what it holds.
    counted = np.flatnonzero(_counted(collapsed, kinds))
    trees = nodes["n_trees"].to_numpy(dtype=np.int64)
    texts = np.full(len(nodes), "", dtype=object)
    texts[counted] = [
        ", ".join(
            [
                *([f"{trees[row]:,} trees"] if trees[row] > 1 else []),
                *(
                    [
                        f"{n_clones[row]:,} "
                        + (names.singular if n_clones[row] == 1 else names.plural)
                    ]
                    if kinds[row] == _MIXED
                    else []
                ),
            ]
        )
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
