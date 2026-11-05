"""The figure of a cell phylogeny and the clones cut from it, drawn from a store.

The truth's phylogeny splits below a root and its one child into one clade
per clone, so its figure cut at time 1 draws the truth's clones, whose cells
and times are known from ``tests/clone_truth.py``. Its forest, one tree per
clone, checks the trees drawn alone and together.
"""

import warnings
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.legend import Legend
from matplotlib.patches import PathPatch

from cytearc.plotting._style import CUSTOM_PALETTES, NO_VALUE_COLOR, category_label
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    every_cell,
    import_truth_clones,
    import_truth_phylogeny,
    write_truth_store,
)
from tests.clone_truth import (
    FOUNDER_TIME,
    LEAF_TIME,
    TruthTree,
    clone_truth,
    random_newick,
    truth_forest,
    write_newick,
)
from tests.memory_helpers import audited, check_budget
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()
# The truth's clones, largest first: their labels and cells.
LABELS = [str(clone + 1) for clone in range(TRUTH.n_clones)]
SIZES = [9, 6, 5, 4, 3, 2, 2, 1]
# Clones take the colours of the 20-colour palette without its two greys,
# by rank.
CLONE_COLORS = [
    color for color in CUSTOM_PALETTES[20] if color not in ("#7f7f7f", "#c7c7c7")
]
assert len(CLONE_COLORS) == 18
MIXED = "#969696"


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("phylogeny_figure")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    phylogeny, clones = import_truth_phylogeny(store, TRUTH, folder)
    with warnings.catch_warnings():
        # One cell of the truth ends before depth 2.
        warnings.simplefilter("ignore")
        by_depth = store.clones.phylogeny_clones(phylogeny, depth=2)
    # The truth's forest, one tree per clone, and a ninth tree whose leaves
    # name no cell of the store.
    outside = TruthTree(
        "outside",
        ("root", "d2__AAAA-1", "d2__CCCC-1"),
        (-1, 0, 0),
        (0.0, 1.0, 1.0),
    )
    forest_path = write_newick(
        [*truth_forest(TRUTH), outside], folder / "truth_forest.nwk"
    )
    with pytest.warns(UserWarning, match="match no cell of the store"):
        forest = store.clones.import_phylogeny(
            forest_path, cell_selection=every_cell(store), time_unit="days"
        )
    ids = [str(cell) for cell in TRUTH.cell_ids]
    # Three trees cut at time 0.5: five leaves past the cut, one clone each;
    # a leaf before the cut beside one past it; and two leaves before it.
    mixed_path = write_newick(
        [
            TruthTree("t1", ("r", *ids[:5]), (-1, 0, 0, 0, 0, 0), (0.0,) + (1.0,) * 5),
            TruthTree("t2", ("r", ids[5], ids[6]), (-1, 0, 0), (0.0, 0.2, 1.0)),
            TruthTree("t3", ("r", ids[7], ids[8]), (-1, 0, 0), (0.0, 0.2, 0.2)),
        ],
        folder / "mixed_forest.nwk",
    )
    # A clade of 39 cells and a leaf of one below the root: the leaf's wedge
    # is a fortieth of the height.
    forty_path = write_newick(
        [
            TruthTree(
                "forty",
                ("r", "c", *ids[:39], ids[39]),
                (-1, 0, *[1] * 39, 0),
                (0.0, 1.0, *[2.0] * 39, 2.0),
            )
        ],
        folder / "forty.nwk",
    )
    with warnings.catch_warnings():
        # Cells of the mixed forest end before its cut.
        warnings.simplefilter("ignore")
        mixed, forty = (
            store.clones.phylogeny_clones(
                store.clones.import_phylogeny(
                    path, cell_selection=every_cell(store), time_unit="days"
                ),
                time=0.5,
            )
            for path in (mixed_path, forty_path)
        )
    return SimpleNamespace(
        store=store,
        folder=folder,
        phylogeny=phylogeny,
        clones=clones,
        by_depth=by_depth,
        forest=store.clones.phylogeny_clones(forest, depth=0),
        mixed=mixed,
        forty=forty,
    )


@pytest.fixture
def drawn(truth):
    """Draw a figure of the truth's store, closed when the test ends."""
    results = []

    def draw(clones, **options):
        result = truth.store.plots.phylogeny(clones, show=False, **options)
        results.append(result)
        return result

    yield draw
    for result in results:
        result.close()


def _wedges(result):
    nodes = result.tables["nodes"]
    return nodes[nodes["wedge"]]


def _cut_lines(ax) -> list[float]:
    return [
        float(line.get_xdata()[0])
        for line in ax.get_lines()
        if line.get_linestyle() == "--"
    ]


def _wedge_colors(ax) -> list[str]:
    """The colours of the wedges drawn, one path for each colour."""
    return [
        to_hex(patch.get_facecolor())
        for patch in ax.patches
        if isinstance(patch, PathPatch) and patch.get_fill()
    ]


def test_each_clone_of_the_cut_is_a_wedge_of_its_cells(truth, drawn) -> None:
    result = drawn(truth.clones)
    nodes = result.tables["nodes"]
    wedges = _wedges(result)
    # Every clone is drawn, the most cells first, as a clade of the cut that
    # never opens; ties keep the order of the tree.
    assert wedges["clone"].tolist() == LABELS
    assert wedges["label"].tolist() == LABELS
    assert wedges["n_cells"].tolist() == SIZES
    assert (wedges["n_clones"] == 1).all()
    # Each clade starts at its root's time, 1 and an eighth more per clone,
    # and ends at the leaves.
    assert wedges["x"].tolist() == [1.0 + 0.125 * clone for clone in range(8)]
    assert (wedges["end"] == LEAF_TIME).all()
    # The tree is drawn from the founder, below the root's one child, with
    # the stem from the root at time 0.
    top = nodes.iloc[0]
    assert (top["tree"], top["node"], top["wedge"]) == ("truth_tree", 1, False)
    assert (top["x"], top["start"]) == (FOUNDER_TIME, 0.0)
    assert (wedges["start"] == FOUNDER_TIME).all()
    # Each wedge is as tall as its cells, from the top.
    assert (wedges["high"] - wedges["low"]).tolist() == SIZES
    assert wedges["low"].tolist() == np.cumsum([0, *SIZES[:-1]]).tolist()
    assert result.figure.axes[0].get_title() == "8 clones cut at time 1 days"
    assert result.provenance.extras["clones"] == truth.clones.to_dict()
    assert result.provenance.extras["phylogeny"] == truth.phylogeny.to_dict()
    assert result.provenance.extras["time"] == 1.0
    assert result.provenance.extras["depth"] is None
    assert result.provenance.n_cells == sum(SIZES)


def test_a_clade_of_several_clones_is_grey_and_says_how_many(truth, drawn) -> None:
    result = drawn(truth.clones, max_leaves=5)
    wedges = _wedges(result)
    # The four largest clones, then the other four as one wedge.
    assert wedges["n_clones"].tolist() == [1, 1, 1, 1, 4]
    assert wedges["n_cells"].tolist() == [9, 6, 5, 4, 3 + 2 + 2 + 1]
    group = wedges.iloc[-1]
    assert (group["clone"], group["label"], group["node"], group["n_clades"]) == (
        None,
        None,
        -1,
        4,
    )
    ax = result.figure.axes[0]
    assert "4 clones" in [text.get_text() for text in ax.texts]
    legend = next(artist for artist in ax.get_children() if isinstance(artist, Legend))
    labels = [text.get_text() for text in legend.get_texts()]
    assert legend.get_title().get_text() == "Clone"
    assert labels == ["Mixed", "1", "2", "3", "4"]
    scale = result.scales[0]
    # A clone's colour comes from its rank, the largest first.
    assert [scale.palette[value] for value in scale.order] == [
        MIXED,
        *CLONE_COLORS[:4],
    ]
    assert sorted(_wedge_colors(ax)) == sorted([MIXED, *CLONE_COLORS[:4]])
    # At seven wedges the last holds the two smallest clones, mixed too.
    seven = drawn(truth.clones, max_leaves=7)
    assert _wedges(seven)["n_clones"].tolist() == [1] * 6 + [2]
    assert [text.get_text() for text in seven.figure.axes[0].texts] == ["2 clones"]


def test_a_wedge_takes_a_clones_colour_only_when_all_its_cells_are_in_it(
    truth, drawn
) -> None:
    # The second tree of the mixed forest drawn as one wedge holds its one
    # clone and a cell before the cut: grey, saying it holds one clone.
    whole = drawn(truth.mixed, tree="mixed_forest_2", max_leaves=1)
    (row,) = whole.tables["nodes"].to_dict("records")
    # Every node of the tree has a name, so the clone takes its leaf's.
    assert (row["n_cells"], row["n_clones"], row["label"]) == (
        2,
        1,
        f"mixed_forest_2/{TRUTH.cell_ids[6]}",
    )
    ax = whole.figure.axes[0]
    assert _wedge_colors(ax) == [MIXED]
    assert [text.get_text() for text in ax.texts] == ["1 clone"]
    assert whole.legends[0].label == "Clone"
    assert [
        category_label(whole.scales[0], value) for value in whole.scales[0].order
    ] == ["Mixed"]
    # Opened, the cell before the cut is a point of the missing grey and the
    # clone a wedge of its colour.
    opened = drawn(truth.mixed, tree="mixed_forest_2", max_leaves=10)
    wedges = _wedges(opened)
    assert wedges[["n_cells", "n_clones"]].values.tolist() == [[1, 0], [1, 1]]
    points = opened.figure.axes[0].collections[0]
    assert [to_hex(color) for color in points.get_facecolors()] == [
        NO_VALUE_COLOR,
        CLONE_COLORS[5],
    ]
    # The second and third trees together hold one clone and three cells
    # before the cut: one grey wedge of two trees.
    grouped = drawn(truth.mixed, max_leaves=2)
    nodes = grouped.tables["nodes"]
    assert nodes[["tree", "n_cells", "n_clones", "n_trees"]].values.tolist() == [
        ["mixed_forest_1", 5, 5, 1],
        [None, 4, 1, 2],
    ]
    texts = [text.get_text() for text in grouped.figure.axes[0].texts]
    assert texts == ["5 clones", "2 trees, 1 clone"]
    assert _wedge_colors(grouped.figure.axes[0]) == [MIXED]


def test_a_wedge_a_fortieth_of_the_height_is_listed(truth, drawn) -> None:
    result = drawn(truth.forty)
    wedges = _wedges(result)
    assert wedges["n_cells"].tolist() == [39, 1]
    assert (wedges["high"] - wedges["low"]).tolist() == [39.0, 1.0]
    # The one cell is a fortieth of the forty, tall enough to list.
    assert result.legends[0].label == "Clone"
    assert len(result.scales[0].order) == 2


def test_a_clone_keeps_its_colour_whatever_else_is_drawn(truth, drawn) -> None:
    every = drawn(truth.clones)
    some = drawn(truth.clones, max_leaves=3)
    whole = every.scales[0].palette
    assert [whole[code] for code in range(8)] == CLONE_COLORS[:8]
    for code, color in some.scales[0].palette.items():
        if code >= 0:
            assert whole[code] == color


def test_the_cut_is_a_dashed_line_at_its_time_or_depth(truth, drawn) -> None:
    at_time = drawn(truth.clones)
    at_depth = drawn(truth.by_depth)
    timed, deep = at_time.figure.axes[0], at_depth.figure.axes[0]
    assert _cut_lines(timed) == [1.0]
    assert timed.get_xlabel() == "Time (days)"
    assert _cut_lines(deep) == [2.0]
    assert deep.get_xlabel() == "Branchings below the root"
    assert deep.get_title() == "19 clones cut at depth 2"
    # On an axis of branchings, the root and its one child stand at 0 and
    # the clades of the truth at 1.
    nodes = at_depth.tables["nodes"]
    assert nodes.iloc[0]["x"] == 0.0
    assert set(nodes.loc[nodes["parent"] == 0, "x"]) == {1.0}
    assert all(float(tick).is_integer() for tick in deep.get_xticks())


def test_cells_that_end_before_the_cut_take_the_missing_grey(truth, drawn) -> None:
    result = drawn(truth.by_depth)
    wedges = _wedges(result)
    # One cell of the truth ends one branching below its root, before depth
    # 2: a point at its leaf, in no clone.
    early = wedges[wedges["n_clones"] == 0]
    assert early["n_cells"].tolist() == [1]
    assert early["clone"].isna().all()
    assert (early["x"] == early["end"]).all()
    ax = result.figure.axes[0]
    legend = next(artist for artist in ax.get_children() if isinstance(artist, Legend))
    assert legend.get_texts()[0].get_text() == "No clone"
    assert result.scales[0].palette[result.scales[0].order[0]] == NO_VALUE_COLOR
    points = {
        to_hex(color)
        for collection in ax.collections
        for color in collection.get_facecolors()
    }
    assert NO_VALUE_COLOR in points


def test_one_tree_of_a_forest_is_drawn_alone_by_its_name(truth, drawn) -> None:
    result = drawn(truth.forest, tree="truth_forest_3")
    nodes = result.tables["nodes"]
    # The third clone's tree alone, of five cells, one clone at depth 0.
    assert set(nodes["tree"]) == {"truth_forest_3"}
    assert nodes[["n_cells", "n_clones", "clone", "wedge"]].values.tolist() == [
        [5, 1, "truth_forest_3/3", True]
    ]
    assert result.figure.axes[0].get_title() == "truth_forest_3: 1 clone cut at depth 0"
    assert result.provenance.extras["tree"] == "truth_forest_3"
    with pytest.raises(
        ValueError, match="'truth_forest_1', 'truth_forest_2'.* and 4 more"
    ):
        drawn(truth.forest, tree="truth_forest_10")
    with pytest.raises(TypeError, match="tree must be the name"):
        drawn(truth.forest, tree=3)
    # A tree whose leaves name no cell has nothing to draw.
    with pytest.raises(ValueError, match="so there is nothing to plot$"):
        drawn(truth.forest, tree="truth_forest_9")


def test_a_forest_of_more_trees_than_the_cap_draws_its_smallest_as_one(
    truth, drawn
) -> None:
    result = drawn(truth.forest, max_leaves=3)
    nodes = result.tables["nodes"]
    assert nodes["tree"].tolist() == ["truth_forest_1", "truth_forest_2", None]
    assert nodes["n_trees"].tolist() == [1, 1, 6]
    assert nodes["n_cells"].tolist() == [9, 6, 5 + 4 + 3 + 2 + 2 + 1]
    assert nodes["n_clones"].tolist() == [1, 1, 6]
    texts = [text.get_text() for text in result.figure.axes[0].texts]
    assert texts == ["6 trees, 6 clones"]
    # Cut at depth 0, the cut stands at the root.
    assert _cut_lines(result.figure.axes[0]) == [0.0]
    # Every tree drawn alone fits a larger cap.
    every = drawn(truth.forest, max_leaves=8)
    assert every.tables["nodes"]["tree"].tolist() == [
        f"truth_forest_{tree}" for tree in range(1, 9)
    ]


def test_clones_that_no_cut_made_are_refused(truth, drawn, tmp_path) -> None:
    imported = import_truth_clones(truth.store, TRUTH, tmp_path)
    figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="cut from a phylogeny.*'import_clones'"):
        drawn(imported)
    clonotypes = ArtifactRef(
        scope="datastore", assay=None, kind="clonotypes", artifact_id="1" * 64
    )
    with pytest.raises(ValueError, match="not a 'clonotypes' artifact"):
        drawn(clonotypes)
    with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
        drawn("clones")
    with pytest.raises(ValueError, match="cut from a phylogeny"):
        drawn(truth.phylogeny)
    assert set(plt.get_fignums()) == figures


def test_the_store_refuses_to_collapse_what_is_no_artifact(truth) -> None:
    with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
        truth.store._phylogeny_figure(
            "clones", tree=None, max_leaves=5, drawing_bytes=lambda _nodes: 0
        )


def test_clones_whose_codes_no_longer_fit_their_phylogeny_are_refused(tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group

    store = write_truth_store(tmp_path / "store.zarr", TRUTH)
    _phylogeny, clones = import_truth_phylogeny(store, TRUTH, tmp_path)
    group = artifact_group(store.zw, clones)
    # A clone artifact whose codes lost a cell.
    codes = np.asarray(group["codes"][:])
    group.create_array("codes", data=codes[:-1], overwrite=True)
    with pytest.raises(ValueError, match="selection holds 48"):
        store.plots.phylogeny(clones, show=False)


@pytest.mark.parametrize(
    ("max_leaves", "error", "match"),
    [
        (0, ValueError, "max_leaves must be at least 1"),
        (-2, ValueError, "max_leaves must be at least 1"),
        (2.5, TypeError, "max_leaves must be an integer"),
        ("100", TypeError, "max_leaves must be an integer"),
        (True, TypeError, "max_leaves must be an integer"),
    ],
)
def test_max_leaves_is_a_whole_number_of_at_least_one(
    drawn, truth, max_leaves, error, match
) -> None:
    with pytest.raises(error, match=match):
        drawn(truth.clones, max_leaves=max_leaves)


def _caterpillar(depth: int, ids: list[str]) -> str:
    """Newick text of a caterpillar ``depth`` levels deep: spine node ``s{k}`` holds the next and the leaf ``ids[k]``.

    Spine node k is at time k, the last holds two leaves, and every leaf is
    at time ``depth + 1``. The text is written from the deepest node out, on
    one line.
    """
    parts = ["(" * depth, f"{ids[depth]}:2,{ids[depth - 1]}:2)s{depth - 1}"]
    for level in range(depth - 2, -1, -1):
        parts.append(f":1,{ids[level]}:{depth + 1 - level})s{level}")
    return "".join(parts) + ";"


@pytest.mark.slow
def test_a_caterpillar_30000_levels_deep_is_drawn_without_recursion(tmp_path) -> None:
    depth = 30_000
    ids = [f"cell{index:05d}-1" for index in range(depth + 1)]
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path),
        {"RNA": np.ones((len(ids), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(len(ids), dtype=bool))
    tree = tmp_path / "caterpillar.nwk"
    tree.write_text(_caterpillar(depth, ids))
    phylogeny = store.clones.import_phylogeny(
        tree, cell_selection=every_cell(store), time_unit="days"
    )
    # Cut past the spine: each leaf is a clone of its own.
    clones = store.clones.phylogeny_clones(phylogeny, time=depth + 0.5)
    capped = store.plots.phylogeny(clones, show=False)
    try:
        assert _wedges(capped)["n_cells"].sum() == depth + 1
        assert len(_wedges(capped)) == 100
    finally:
        capped.close()
    # Drawn leaf by leaf, no clone is tall enough to list but the largest.
    # Every leaf drawn alone: the spine opens to its last node.
    every = store.plots.phylogeny(clones, max_leaves=depth + 1, show=False)
    try:
        nodes = every.tables["nodes"]
        assert len(_wedges(every)) == depth + 1
        assert nodes.loc[~nodes["wedge"], "x"].max() == depth - 1
        assert every.legends[0].label == f"Clone (1 of {depth + 1})"
        # The largest clone, the first by rank.
        assert every.scales[0].order == (0,)
    finally:
        every.close()


@pytest.fixture(scope="module")
def random_tree(tmp_path_factory) -> SimpleNamespace:
    """A store of 30,000 cells, 20,000 of them the leaves of a random tree that ends at its root's height."""
    folder = tmp_path_factory.mktemp("phylogeny_figure_memory")
    rng = np.random.default_rng(5)
    ids = [f"cell{index:06d}-1" for index in range(30_000)]
    path = folder / "store.zarr"
    write_count_store(
        str(path),
        {"RNA": np.ones((len(ids), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(len(ids), dtype=bool))
    tree = folder / "tree.nwk"
    tree.write_text(random_newick(ids[:20_000], rng))
    phylogeny = store.clones.import_phylogeny(
        tree, cell_selection=every_cell(store), time_unit="days"
    )
    height = float(store.clones.load_phylogeny(phylogeny).nodes["time"].max())
    # The other 10,000 cells as a forest of two trees, of 6,000 and 4,000.
    forest_path = folder / "forest.nwk"
    forest_path.write_text(
        random_newick(ids[20_000:26_000], rng)
        + "\n"
        + random_newick(ids[26_000:], rng)
        + "\n"
    )
    forest = store.clones.import_phylogeny(
        forest_path, cell_selection=every_cell(store), time_unit="days"
    )
    lowest = float(store.clones.load_phylogeny(forest).trees["latest_leaf_time"].min())
    with warnings.catch_warnings():
        # Leaves of fewer than three branchings end before the depth.
        warnings.simplefilter("ignore")
        by_depth = store.clones.phylogeny_clones(phylogeny, depth=3)
    return SimpleNamespace(
        store=store,
        clones={
            "time": store.clones.phylogeny_clones(phylogeny, time=height / 2),
            "depth": by_depth,
            "tree": store.clones.phylogeny_clones(forest, time=lowest / 2),
        },
    )


def test_the_legend_lists_the_clones_tall_enough_to_see(random_tree) -> None:
    result = random_tree.store.plots.phylogeny(random_tree.clones["time"], show=False)
    try:
        nodes = result.tables["nodes"]
        wedges = nodes[nodes["wedge"]]
        tall = wedges[wedges["high"] - wedges["low"] >= wedges["high"].max() / 40]
        listed = tall[tall["n_clones"] == 1]
        thin = wedges[(wedges["n_clones"] == 1) & ~wedges.index.isin(listed.index)]
        # Some clones are too thin to list, and every grey is listed first.
        assert len(listed) and len(thin)
        greys = ["Mixed"] if (wedges["n_clones"] >= 2).any() else []
        greys += ["No clone"] if (wedges["n_clones"] == 0).any() else []
        legend = next(
            artist
            for artist in result.figure.axes[0].get_children()
            if isinstance(artist, Legend)
        )
        labels = [text.get_text() for text in legend.get_texts()]
        clones = labels[len(greys) :]
        assert labels[: len(greys)] == greys
        # The clones listed, the largest first, which their rank orders.
        assert sorted(clones) == sorted(listed["label"])
        ranks = [result.scales[0].order[len(greys) + at] for at in range(len(clones))]
        assert ranks == sorted(ranks)
        n_drawn = len(greys) + int((wedges["n_clones"] == 1).sum())
        assert legend.get_title().get_text() == f"Clone ({len(labels)} of {n_drawn})"
        assert result.legends[0].label == legend.get_title().get_text()
    finally:
        result.close()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "at_most"),
    # A cut of the one tree at a time or a depth leaves the budget to the
    # nodes' arrays, 1.13 to 1.15 times the traced peak, as measured. One
    # tree of a forest reads the whole forest's arrays, whose reads' charges
    # set the budget of its small collapse, 1.36 times the peak at ten
    # wedges.
    [("time", 1.3), ("depth", 1.3), ("tree", 1.7)],
)
@pytest.mark.parametrize("max_leaves", [10, 1_000])
def test_the_figure_runs_within_the_smallest_budget_it_accepts(
    random_tree, case, at_most, max_leaves
) -> None:
    clones = random_tree.clones[case]
    options = {"tree": "forest_1"} if case == "tree" else {}

    def draw(store) -> None:
        store.plots.phylogeny(
            clones, max_leaves=max_leaves, show=False, **options
        ).close()

    # The kernels compile and Matplotlib loads its fonts before any trace.
    draw(random_tree.store)
    # The budget that runs it bounds its traced peak, one byte less is
    # refused naming it, and it is not far above the peak.
    budget, refusals, _result = check_budget(
        random_tree.store, draw, at_most=at_most, max_refusals=40
    )
    assert refusals
    # Every charge bounds what is held until the next, the copies of one
    # tree's nodes included.
    saved = random_tree.store.memoryBytes
    random_tree.store.memoryBytes = budget
    try:
        _result, overruns = audited(lambda: draw(random_tree.store))
    finally:
        random_tree.store.memoryBytes = saved
    assert not overruns, overruns
