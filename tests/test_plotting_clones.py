"""The clone plots, on clones whose sizes are counted by hand."""

import inspect
import re
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc.clones import diversity as dv
from cytearc.datastore.datastore import DataStore
from cytearc.plotting import (
    clonal_diversity,
    clonal_expansion,
    clonal_rarefaction,
    clone_abundance,
    clone_coupling,
    clone_fate_bias,
    clone_overlap,
    clone_tracking,
)
from cytearc.plotting._groups import labels_by_group, plot_groups, table_groups
from cytearc.plotting._style import MAX_FIGURE_WIDTH_INCHES
from tests.clone_helpers import (
    axis_names,
    names_every_nth_item,
    names_keep_apart,
    every_cell,
    import_truth_clones,
    import_truth_receptors,
    overlap_reference,
    write_truth_store,
)
from tests.clone_truth import CELL_TYPES, N_CELLS, clone_truth, receptor_truth
from tests.storage_helpers import (
    insert_nullable_cell_column,
    open_store,
    write_count_store,
)

pytestmark = pytest.mark.usefixtures("figures_closed")

# Clone a holds c0, c1 and c2, clone b c3, and c4 and c5 have no clone; c0,
# c1 and c3 are in group x, c2 and c4 in group y, and c5 has no group.
CLONES = ["a", "a", "a", "b", "", ""]
GROUPS = ["x", "x", "y", "x", "y", None]


@pytest.fixture(scope="module")
def plotted(tmp_path_factory):
    path = tmp_path_factory.mktemp("clone_plot") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((6, 3), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("lineage", np.asarray(CLONES, dtype=object))
    store.cells.insert("group", np.asarray(GROUPS, dtype=object))
    store.cells.insert("every_cell", np.ones(6, dtype=bool))
    store.cells.insert("first_four", np.arange(6) < 4)
    cells = store.snapshot_cell_selection("every_cell")
    clones = store.clones.from_column("lineage", cell_selection=cells)
    return store, clones


class _NamesOnly:
    """A store that can name and label clones, and nothing else."""

    def __init__(self, store: DataStore) -> None:
        self._clone_figure_nouns = store._clone_figure_nouns
        self._clone_figure_labels = store._clone_figure_labels


def test_without_groups_the_plot_counts_every_cell(plotted) -> None:
    store, clones = plotted

    result = clone_abundance(store, clones, show=False)

    assert result.tables["sizes"].to_dict("list") == {
        "group": ["all cells", "all cells"],
        "clone_size": [1, 3],
        "n_clones": [1, 1],
    }
    # Clones are labeled by their IDs, and every cell is in one group.
    assert result.tables["top"].to_dict("list") == {
        "clone": ["a", "b"],
        "label": ["a", "b"],
        "group": ["all cells", "all cells"],
        "n_cells": [3, 1],
    }
    assert set(result.axes) == {"sizes", "top"}
    assert result.legends == ()
    assert result.provenance.n_cells == 4
    assert result.provenance.extras["clones"] == clones.to_dict()
    assert [tick.get_text() for tick in result.axes["top"].get_yticklabels()] == [
        "a",
        "b",
    ]
    assert result.axes["top"].get_title() == "Largest 2 clones"
    plt.close(result.figure)


def test_shown_figures_that_the_plot_owns_are_closed(plotted) -> None:
    store, clones = plotted
    result = clone_abundance(store, clones, show=True)
    assert result.owns_figure
    assert result.figure.number not in plt.get_fignums()


def test_a_grouping_column_splits_each_clone_by_group(plotted) -> None:
    store, clones = plotted

    result = store.plots.clone_abundance(
        clones, group_by="group", max_clones=1, show=False
    )

    # Within x, clone a holds two cells and b one; within y, a holds one.
    assert result.tables["sizes"].to_dict("list") == {
        "group": ["x", "x", "y"],
        "clone_size": [1, 2, 1],
        "n_clones": [1, 1, 1],
    }
    assert result.tables["top"].to_dict("list") == {
        "clone": ["a", "a"],
        "label": ["a", "a"],
        "group": ["x", "y"],
        "n_cells": [2, 1],
    }
    assert result.legends[0].label == "group"
    assert len(result.axes["sizes"].get_lines()) == 2
    plt.close(result.figure)


def test_a_label_artifact_of_fewer_cells_leaves_the_others_ungrouped(plotted) -> None:
    store, clones = plotted
    labels = store.snapshot_cluster_labels(
        "group", cell_selection=store.snapshot_cell_selection("first_four")
    )

    result = clone_abundance(store, clones, groups=labels, show=False)

    # Of the first four cells, x holds a twice and b once and y holds a once.
    assert result.tables["top"].to_dict("list") == {
        "clone": ["a", "a", "b", "b"],
        "label": ["a", "a", "b", "b"],
        "group": ["x", "y", "x", "y"],
        "n_cells": [2, 1, 1, 0],
    }
    # The legend is titled by the column that the labels copied.
    assert result.legends[0].label == "group"
    plt.close(result.figure)


def test_the_plot_refuses_clones_it_cannot_draw(plotted) -> None:
    store, clones = plotted
    figures = set(plt.get_fignums())
    store.cells.insert("only_c4", np.arange(6) == 4, overwrite=True)
    with pytest.raises(ValueError, match="^max_clones must be at least 1$"):
        clone_abundance(store, clones, max_clones=0, show=False)
    with pytest.raises(TypeError, match="^max_clones must be an integer$"):
        clone_abundance(store, clones, max_clones="3", show=False)
    with pytest.raises(TypeError, match="group_by must be a column name"):
        clone_abundance(store, clones, group_by=3, show=False)
    with pytest.raises(TypeError, match="groups must be an ArtifactRef"):
        clone_abundance(store, clones, groups="group", show=False)
    with pytest.raises(TypeError, match="store does not provide clones.load_clones"):
        clone_abundance(object(), clones, show=False)
    empty = store.clones.from_column(
        "lineage", cell_selection=store.snapshot_cell_selection("only_c4")
    )
    with pytest.raises(ValueError, match="^No cell has a clone, so there is nothing"):
        clone_abundance(store, empty, show=False)
    store.cells.insert(
        "c4_group", np.where(np.arange(6) == 4, "z", None), overwrite=True
    )
    with pytest.raises(ValueError, match="^No cell with a group has a clone"):
        clone_abundance(store, clones, group_by="c4_group", show=False)
    assert set(plt.get_fignums()) == figures


@pytest.mark.parametrize("name", ["clone", "label", "group", "n_cells", "all cells"])
def test_groups_named_like_the_tables_columns_draw(plotted, name) -> None:
    store, clones = plotted
    # c1's group takes the name, beside x: c0 and c3 in x, c1 in the named
    # group, and c2 in x too.
    named = ["x", name, "x", "x", "x", None]
    store.cells.insert("named_group", np.asarray(named), overwrite=True)
    store.cells.insert(
        "neutral_group", np.asarray(["x", "w", "x", "x", "x", None]), overwrite=True
    )
    found = clone_abundance(store, clones, group_by="named_group", show=False)
    neutral = clone_abundance(store, clones, group_by="neutral_group", show=False)
    try:
        for table in ("sizes", "top"):
            expected = neutral.tables[table].replace({"group": {"w": name}})
            assert found.tables[table].columns.tolist() == expected.columns.tolist()
            assert sorted(map(tuple, found.tables[table].values.tolist())) == sorted(
                map(tuple, expected.values.tolist())
            )
        # Clone a holds c0 and c2 in x and c1 in the named group.
        top = found.tables["top"].set_index(["clone", "group"])["n_cells"]
        assert top.loc[("a", name)] == 1 and top.loc[("a", "x")] == 2
        bars = sorted(patch.get_width() for patch in found.axes["top"].patches)
        assert bars == sorted(
            patch.get_width() for patch in neutral.axes["top"].patches
        )
        legend = [text.get_text() for text in found.axes["top"].get_legend().texts]
        assert sorted(legend) == sorted([name, "x"])
    finally:
        found.close()
        neutral.close()


def _grouped(groups) -> tuple[list[int], tuple[str, ...], str | None]:
    return groups.codes.tolist(), groups.names, groups.title


def test_plot_groups_align_labels_to_the_artifacts_cells(plotted) -> None:
    store, clones = plotted
    # Without a grouping, every cell is in one group, and the store is not
    # read.
    ungrouped = plot_groups(object(), clones, group_by=None, groups=None, n_cells=6)
    assert _grouped(ungrouped) == ([0] * 6, ("all cells",), None)
    grouped = plot_groups(store, clones, group_by="group", groups=None, n_cells=6)
    assert _grouped(grouped) == ([0, 0, 1, 0, 1, -1], ("x", "y"), "group")


def test_plot_groups_align_labels_to_clones_of_later_cells(plotted) -> None:
    store, _clones = plotted
    store.cells.insert("first_five", np.arange(6) < 5, overwrite=True)
    store.cells.insert("middle_three", np.isin(np.arange(6), [2, 3, 4]), overwrite=True)
    labels = store.snapshot_cluster_labels(
        "group", cell_selection=store.snapshot_cell_selection("first_five")
    )
    middle = store.clones.from_column(
        "lineage", cell_selection=store.snapshot_cell_selection("middle_three")
    )
    # The clones' cells, c2 to c4, lie in y, x and y; the labels' first
    # cells, c0 and c1, lie in x.
    found = plot_groups(store, middle, group_by=None, groups=labels, n_cells=3)
    assert _grouped(found) == ([1, 0, 1], ("x", "y"), "group")


def test_clones_of_equal_size_list_in_rank_order(plotted) -> None:
    store, _clones = plotted
    # 9 and 10 hold two cells each and rank by number, 9 first, where their
    # text would put 10 first.
    store.cells.insert(
        "tied",
        np.asarray(["10", "10", "9", "9", "x", ""], dtype=object),
        overwrite=True,
    )
    tied = store.clones.from_column(
        "tied", cell_selection=store.snapshot_cell_selection("every_cell")
    )

    result = clone_abundance(store, tied, show=False)

    # Clones are labeled by their IDs, in the same order.
    assert result.tables["top"].to_dict("list") == {
        "clone": ["9", "10", "x"],
        "label": ["9", "10", "x"],
        "group": ["all cells"] * 3,
        "n_cells": [2, 2, 1],
    }
    plt.close(result.figure)


TRUTH = clone_truth()
SAMPLES = ("d2", "d4")
CATEGORIES = (
    "1 cell",
    "2 cells",
    "3 to 5 cells",
    "6 to 20 cells",
    "21 to 100 cells",
    "more than 100 cells",
)


@pytest.fixture(scope="module")
def truth_plot(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("clone_summary_plots")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    clones = import_truth_clones(store, TRUTH, folder)
    cells = every_cell(store)
    return SimpleNamespace(
        store=store,
        clones=clones,
        by_sample=store.clones.expansion(clones, group_by="sample"),
        labels=store.snapshot_cluster_labels("cell_type", cell_selection=cells),
    )


def _in_sample() -> dict[str, int]:
    """The cells with a clone of each sample."""
    return {
        sample: sum(found)
        for sample, found in TRUTH.group_clone_sizes(TRUTH.samples).items()
    }


def test_the_expansion_plot_draws_the_expansions_own_groups(truth_plot) -> None:
    result = truth_plot.store.plots.clonal_expansion(truth_plot.by_sample, show=False)

    table = result.tables["categories"]
    assert table.columns.tolist() == ["group", "category", "n_cells", "fraction"]
    assert table["group"].tolist() == ["d2"] * 6 + ["d4"] * 6
    # Each sample's cells per category, from its clone sizes.
    expected = []
    for sample in ("d2", "d4"):
        cells = dict.fromkeys(CATEGORIES, 0)
        for size in TRUTH.group_clone_sizes(TRUTH.samples)[sample]:
            category = 0 if size == 1 else 1 if size == 2 else 2 if size <= 5 else 3
            cells[CATEGORIES[category]] += size
        expected += list(cells.values())
    assert table["n_cells"].tolist() == expected
    ranks = result.tables["rank_windows"]
    first = ranks[ranks["window"] == "1 to 10"]
    assert first["n_cells"].tolist() == [_in_sample()["d2"], _in_sample()["d4"]]
    assert first["fraction"].tolist() == [1.0, 1.0]
    assert result.provenance.n_cells == 32
    assert result.provenance.extras["grouping"] == "sample"
    assert result.provenance.extras["splits"] == [10, 100, 1000, 10000, 30000, 100000]
    ax = result.axes["categories"]
    assert ax.get_xlabel() == "Fraction of cells"
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["d2", "d4"]
    assert ax.get_legend().get_title().get_text() == "Clone size"
    # Only the windows that hold cells are drawn.
    legend = result.axes["rank_windows"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["1 to 10"]
    plt.close(result.figure)


def test_the_expansion_plot_regroups_cells_by_another_column(truth_plot) -> None:
    store = truth_plot.store
    bounds = (0.1, 0.2, 1)
    names = ("0 to 0.1", "0.1 to 0.2", "0.2 to 1")
    ref = store.clones.expansion(
        truth_plot.clones, group_by="sample", frequency_breakpoints=bounds
    )
    result = clonal_expansion(
        store, ref, group_by="cell_type", normalize=False, splits=(1,), show=False
    )
    # Each cell keeps the category of its clone's share of its sample.
    sizes = TRUTH.cell_clone_sizes(TRUTH.samples)
    totals = _in_sample()
    category = [
        names[next(at for at, bound in enumerate(bounds) if size / totals[s] <= bound)]
        if size
        else None
        for size, s in zip(sizes.tolist(), TRUTH.samples, strict=True)
    ]
    counted = Counter(
        (cell_type, name)
        for cell_type, name in zip(TRUTH.cell_types, category, strict=True)
        if name is not None
    )
    table = result.tables["categories"].set_index(["group", "category"])
    for cell_type in CELL_TYPES:
        for name in names:
            assert table.loc[(cell_type, name), "n_cells"] == counted[(cell_type, name)]
    # The ranks are counted within the plotted groups.
    ranks = result.tables["rank_windows"].set_index(["group", "window"])
    for cell_type, found in TRUTH.group_clone_sizes(TRUTH.cell_types).items():
        assert ranks.loc[(cell_type, "1"), "n_cells"] == found[0]
        assert ranks.loc[(cell_type, "after 1"), "n_cells"] == sum(found[1:])
    assert result.axes["categories"].get_xlabel() == "Cells"
    assert (
        result.axes["categories"].get_legend().get_title().get_text()
        == "Clone frequency"
    )
    assert result.provenance.extras["grouping"] == "cell_type"
    plt.close(result.figure)
    clone_counts = clonal_expansion(
        store, ref, group_by="cell_type", count="clones", show=False
    )
    # Each clone counts once per cell type and category it holds there.
    triples = {
        (cell_type, clone, name)
        for cell_type, clone, name in zip(
            TRUTH.cell_types, TRUTH.clones, category, strict=True
        )
        if name is not None
    }
    per_type = Counter((cell_type, name) for cell_type, _clone, name in triples)
    table = clone_counts.tables["categories"].set_index(["group", "category"])
    assert table.columns.tolist() == ["n_clones", "fraction"]
    for cell_type in CELL_TYPES:
        n_clones = sum(per_type[(cell_type, name)] for name in names)
        for name in names:
            assert (
                table.loc[(cell_type, name), "n_clones"] == per_type[(cell_type, name)]
            )
            assert table.loc[(cell_type, name), "fraction"] == pytest.approx(
                per_type[(cell_type, name)] / n_clones
            )
    assert clone_counts.axes["categories"].get_xlabel() == "Fraction of clones"
    plt.close(clone_counts.figure)


def test_the_expansion_plot_groups_cells_by_labels(truth_plot) -> None:
    store = truth_plot.store
    ref = store.clones.expansion(truth_plot.clones)
    result = clonal_expansion(store, ref, groups=truth_plot.labels, show=False)
    assert result.tables["categories"]["group"].unique().tolist() == sorted(CELL_TYPES)
    assert result.provenance.extras["grouping"] == "cell_type"
    plt.close(result.figure)
    ungrouped = clonal_expansion(store, ref, show=False, show_legend=False)
    assert ungrouped.tables["categories"]["group"].unique().tolist() == ["all cells"]
    assert ungrouped.provenance.extras["grouping"] is None
    assert ungrouped.axes["categories"].get_legend() is None
    plt.close(ungrouped.figure)
    # Shown by default: on Agg the owned figure is closed.
    shown = clonal_expansion(store, ref)
    assert shown.provenance.n_cells == 32
    assert shown.figure.number not in plt.get_fignums()


def test_the_expansion_plot_draws_on_caller_axes(truth_plot) -> None:
    figure, axes = plt.subplots(1, 2)
    result = clonal_expansion(
        truth_plot.store,
        truth_plot.by_sample,
        target={"categories": axes[0], "rank_windows": axes[1]},
        show=False,
    )
    assert result.figure is figure and result.owns_figure is False
    plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"count": "contigs"}, ValueError, "count must be 'cells' or 'clones'"),
        ({"normalize": "yes"}, TypeError, "normalize must be a boolean"),
        ({"splits": (5, 1)}, ValueError, "strictly increasing"),
        ({"groups": "x"}, TypeError, "groups must be an ArtifactRef"),
    ],
)
def test_the_expansion_plot_checks_its_arguments(
    truth_plot, kwargs, error, message
) -> None:
    with pytest.raises(error, match=message):
        clonal_expansion(truth_plot.store, truth_plot.by_sample, show=False, **kwargs)
    with pytest.raises(TypeError, match="store does not provide clones.load_expansion"):
        clonal_expansion(object(), truth_plot.by_sample, show=False)


def test_the_expansion_plot_of_a_grouping_without_labels(truth_plot) -> None:
    store = truth_plot.store
    if "unlabeled" not in store.cells.columns:
        insert_nullable_cell_column(
            store, "unlabeled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
        )
    # No cell has a label, so no group holds a cell to draw.
    with pytest.raises(ValueError, match="^No cell with a group has an expansion"):
        clonal_expansion(store, truth_plot.by_sample, group_by="unlabeled", show=False)


def _joined(*columns: np.ndarray) -> np.ndarray:
    """Each cell's labels in several columns, joined by commas."""
    return np.asarray([", ".join(map(str, row)) for row in zip(*columns)], dtype=object)


def test_figures_group_cells_by_several_columns(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    result = clone_abundance(
        store, clones, group_by=["sample", "cell_type"], max_clones=40, show=False
    )
    # Each clone's cells in each combination of sample and cell type, counted
    # cell by cell.
    found = TRUTH.group_clone_cells(_joined(TRUTH.samples, TRUTH.cell_types))
    sizes = result.tables["sizes"]
    assert sorted(sizes.itertuples(index=False, name=None)) == sorted(
        (group, size, n_clones)
        for group, cells in found.items()
        for size, n_clones in Counter(cells.values()).items()
    )
    top = result.tables["top"]
    groups = top["group"].drop_duplicates().tolist()
    assert groups == sorted(found)
    by_pair = top.set_index(["clone", "group"])["n_cells"]
    for group, cells in found.items():
        for clone, n_cells in cells.items():
            assert by_pair.loc[(str(clone + 1), group)] == n_cells
    assert top["n_cells"].sum() == 32
    assert result.legends[0].label == "sample, cell_type"
    assert result.provenance.extras["grouping"] == "sample, cell_type"
    # The groups are those of the summaries, named and ordered alike.
    diversity = store.clones.diversity(clones, group_by=["sample", "cell_type"])
    assert groups == diversity.indices["group"].tolist()
    result.close()


def test_figures_group_cells_by_columns_and_labels_together(truth_plot) -> None:
    store, clones, labels = truth_plot.store, truth_plot.clones, truth_plot.labels
    abundance = clone_abundance(store, clones, group_by="sample", groups=labels)
    # The label artifact holds each cell's type, so the groups are those of
    # the sample and the type.
    found = TRUTH.group_clone_cells(_joined(TRUTH.samples, TRUTH.cell_types))
    top = abundance.tables["top"]
    assert top["group"].drop_duplicates().tolist() == sorted(found)
    assert abundance.legends[0].label == "sample, cell_type"
    expansion = clonal_expansion(
        store, truth_plot.by_sample, group_by="sample", groups=labels, show=False
    )
    # Each cell keeps its category, and the plotted groups count its cells.
    table = expansion.tables["categories"]
    assert table.groupby("group", sort=False)["n_cells"].sum().to_dict() == {
        group: sum(cells.values()) for group, cells in found.items()
    }
    assert expansion.provenance.extras["grouping"] == "sample, cell_type"
    expansion.close()
    diversity = store.clones.diversity(clones, group_by="sample", groups=labels)
    assert table["group"].unique().tolist() == diversity.indices["group"].tolist()


def test_figures_group_cells_within_the_memory_budget(plotted, monkeypatch) -> None:
    store, clones = plotted
    found = plot_groups(store, clones, group_by=["group"], groups=None, n_cells=6)
    assert _grouped(found) == ([0, 0, 1, 0, 1, -1], ("x", "y"), "group")
    # Figures group cells as the summaries do, within the store's budget.
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="^Grouping the 6 cells of a figure holds"):
        plot_groups(store, clones, group_by="group", groups=None, n_cells=6)
    with pytest.raises(TypeError, match="store does not provide the groups of"):
        plot_groups(object(), clones, group_by="group", groups=None, n_cells=6)
    # No column names no group, and bytes are refused by the store's grouping.
    monkeypatch.undo()
    empty = plot_groups(object(), clones, group_by=[], groups=None, n_cells=6)
    assert _grouped(empty) == ([0] * 6, ("all cells",), None)
    with pytest.raises(TypeError, match="^group_by must be a column name"):
        plot_groups(store, clones, group_by=b"group", groups=None, n_cells=6)
    with pytest.raises(TypeError, match="^artifact must be an ArtifactRef"):
        store._figure_groups("clones", group_by="group", groups=None)


def test_an_empty_grouping_draws_the_expansions_own_groups(truth_plot) -> None:
    store = truth_plot.store
    own = clonal_expansion(store, truth_plot.by_sample, show=False)
    empty = clonal_expansion(store, truth_plot.by_sample, group_by=[], show=False)
    for name, table in own.tables.items():
        pd.testing.assert_frame_equal(empty.tables[name], table)
    assert empty.provenance.extras["grouping"] == "sample"
    own.close()
    empty.close()
    # Bytes name no column, empty or not, and are refused as every grouping
    # refuses them.
    for value in (b"", b"sample"):
        with pytest.raises(TypeError, match="^group_by must be a column name"):
            clonal_expansion(store, truth_plot.by_sample, group_by=value, show=False)


def test_the_expansion_plot_regroups_cells_by_two_columns(truth_plot) -> None:
    store = truth_plot.store
    result = clonal_expansion(
        store, truth_plot.by_sample, group_by=["sample", "cell_type"], show=False
    )
    # Each cell keeps the category of its clone's size in its sample, counted
    # in the group of its sample and type.
    sizes = TRUTH.cell_clone_sizes(TRUTH.samples)
    groups = _joined(TRUTH.samples, TRUTH.cell_types)
    counted = Counter(
        (
            group,
            CATEGORIES[0 if size == 1 else 1 if size == 2 else 2 if size <= 5 else 3],
        )
        for group, size in zip(groups, sizes.tolist(), strict=True)
        if size
    )
    table = result.tables["categories"]
    assert table["group"].unique().tolist() == sorted({group for group, _ in counted})
    for row in table.itertuples(index=False):
        assert row.n_cells == counted[(row.group, row.category)]
    assert result.provenance.extras["grouping"] == "sample, cell_type"
    result.close()


def test_cells_take_their_groups_labels_from_a_records_keys() -> None:
    keys = pd.DataFrame(
        {"sample": ["d2", "d4"], "receptor_type": ["TCR", "TCR"]},
        index=pd.Index(["d2, TCR", "d4, TCR"], name="group"),
    )
    # Each cell takes its group's labels; a cell without a group, or of a
    # group the keys lack, has none, whether the names are text or
    # categories in another order.
    cells = ["d4, TCR", None, "d2, TCR", "d9, TCR"]
    expected = [
        ("sample", ["d4", None, "d2", None]),
        ("receptor_type", ["TCR", None, "TCR", None]),
    ]
    for given in (
        pd.Series(cells, dtype=object),
        pd.Series(pd.Categorical(cells, categories=["d9, TCR", "d4, TCR", "d2, TCR"])),
    ):
        found = labels_by_group(given, keys, ["sample", "receptor_type"])
        assert [(name, values.tolist()) for name, values in found] == expected
    # A key column may take the name of the keys' index, which names the
    # groups, and the cell takes the column's label.
    named = pd.DataFrame(
        {"group": ["a", "b"]}, index=pd.Index(["x", "y"], name="group")
    )
    ((column, labels),) = labels_by_group(pd.Series(["y", "x"]), named, ["group"])
    assert (column, labels.tolist()) == ("group", ["b", "a"])


@pytest.fixture(scope="module")
def truth_clonotypes(truth_plot, tmp_path_factory):
    """The truth's clonotypes, more than nine, on the store of its clones."""
    folder = tmp_path_factory.mktemp("truth_receptors")
    receptors = import_truth_receptors(truth_plot.store, receptor_truth(TRUTH), folder)
    return truth_plot.store.repertoire.clonotypes(receptors, receptor_arms="any")


def test_the_tracking_table_pairs_each_clone_with_its_label(
    truth_plot, truth_clonotypes
) -> None:
    store = truth_plot.store
    tracking = store.clones.track(truth_clonotypes, group_by="sample", top_n=2)
    # Ranks past 9 come before 2 as text, so the clones followed, largest
    # first, are not in the order of their IDs' text.
    assert list(tracking.tracked) != sorted(tracking.tracked)
    result = clone_tracking(store, tracking, show=False)
    table = result.tables["tracking"]
    representatives = store.repertoire.load_clonotypes(truth_clonotypes).clonotypes
    junctions = {
        row.clonotype: row.VDJ_1_junction_aa
        if isinstance(row.VDJ_1_junction_aa, str)
        else row.VJ_1_junction_aa
        for row in representatives.itertuples(index=False)
    }
    labels = {}
    for clone, label in (
        table[["clone", "label"]].drop_duplicates().itertuples(index=False)
    ):
        # Each label is its own clonotype's junction, and its own number where
        # two clonotypes followed share the junction.
        junction, _space, number = label.partition(" ")
        assert junction == junctions[clone], clone
        assert number in ("", clone.removeprefix("clonotype_")), clone
        labels[clone] = label
    assert sorted(labels) == sorted(tracking.tracked)
    # The legend lists the clones followed, largest first, by those labels.
    legend = result.axes["tracking"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        labels[clone] for clone in tracking.tracked
    ]
    result.close()


def test_the_rarefaction_plot_draws_one_panel_per_order(truth_plot) -> None:
    store = truth_plot.store
    result = store.plots.clonal_rarefaction(
        store.clones.rarefaction(truth_plot.clones, group_by="sample"), show=False
    )
    assert list(result.axes) == ["order_0", "order_1", "order_2", "coverage"]
    curves = result.tables["curves"]
    assert curves.columns[:2].tolist() == ["group", "order"]
    assert sorted(curves["group"].unique()) == ["d2", "d4"]
    assert result.provenance.n_cells == 32
    assert result.provenance.extras["grouping"] == "sample"
    assert result.axes["order_1"].get_title() == "Shannon diversity\n(order 1)"
    assert result.axes["coverage"].get_ylabel() == "Sample coverage"
    # One legend of the samples, beside the last panel of the first row.
    legend = result.axes["order_2"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["d2", "d4"]
    assert legend.get_title().get_text() == "sample"
    assert [
        name for name, ax in result.axes.items() if ax.get_legend() is not None
    ] == ["order_2"]
    # The first line of order 0 is d2's curve up to its observed size, which
    # ends at its clones.
    d2 = dv.CloneSizes.from_sizes(TRUTH.group_clone_sizes(TRUTH.samples)["d2"])
    line = result.axes["order_0"].get_lines()[0]
    assert line.get_label() == "d2"
    assert line.get_xdata()[-1] == d2.n
    assert line.get_ydata()[-1] == d2.richness
    plt.close(result.figure)
    by_coverage = clonal_rarefaction(
        store,
        store.clones.rarefaction(truth_plot.clones, orders=(0,)),
        base="coverage",
        show=False,
    )
    assert list(by_coverage.axes) == ["order_0"]
    assert by_coverage.axes["order_0"].get_xlabel() == "Sample coverage"
    assert set(by_coverage.tables["curves"]["group"]) == {"all cells"}
    # One group needs no legend.
    assert by_coverage.axes["order_0"].get_legend() is None
    assert by_coverage.provenance.extras["grouping"] is None
    plt.close(by_coverage.figure)
    # Shown by default: on Agg the owned figure is closed.
    shown = clonal_rarefaction(
        store, store.clones.rarefaction(truth_plot.clones, orders=(2,))
    )
    assert shown.provenance.extras["orders"] == [2]
    assert shown.figure.number not in plt.get_fignums()


def test_the_coverage_panel_draws_the_coverage(truth_plot) -> None:
    store = truth_plot.store
    result = clonal_rarefaction(
        store,
        store.clones.rarefaction(truth_plot.clones, group_by="sample"),
        show=False,
    )
    curves = result.tables["curves"]
    lines = result.axes["coverage"].get_lines()
    # Each sample draws its rarefied part, its extrapolated part and its
    # observed point, of the first order.
    for name, rarefied in zip(("d2", "d4"), lines[::3], strict=True):
        rows = curves[(curves["group"] == name) & (curves["order"] == 0)]
        rows = rows[rows["method"] != "extrapolation"]
        np.testing.assert_allclose(rarefied.get_xdata(), rows["size"])
        np.testing.assert_allclose(rarefied.get_ydata(), rows["coverage"])
        assert rarefied.get_ydata().max() <= 1
    plt.close(result.figure)


def test_the_diversity_plot_draws_each_groups_indices(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    diversity = store.clones.diversity(clones, group_by="sample")
    result = store.plots.clonal_diversity(
        diversity, indices=["n_clones", "gini", "chao1", "d50"], show=False
    )
    try:
        assert list(result.axes) == ["n_clones", "gini", "chao1", "d50"]
        assert result.tables["indices"] is diversity.indices
        # Each panel draws each sample's index, d2 on top, the clones of each
        # sample counted from the truth.
        clones_of = {
            sample: len(found)
            for sample, found in TRUTH.group_clone_sizes(TRUTH.samples).items()
        }
        widths = [bar.get_width() for bar in result.axes["n_clones"].patches]
        assert widths == [clones_of["d2"], clones_of["d4"]]
        for index in ("gini", "chao1", "d50"):
            bars = result.axes[index].patches
            assert [bar.get_width() for bar in bars] == pytest.approx(
                diversity.indices[index].tolist()
            )
        # The first panel of each row names the samples, d2 on top, for the
        # panels beside it.
        for index, named in (
            ("n_clones", ["d2", "d4"]),
            ("gini", []),
            ("chao1", []),
            ("d50", ["d2", "d4"]),
        ):
            ticks = result.axes[index].get_yticklabels()
            assert [tick.get_text() for tick in ticks] == named, index
        assert result.axes["n_clones"].get_title() == "Clones"
        assert result.axes["chao1"].get_title() == "Chao1"
        # One legend of the samples, beside the last panel of the first row,
        # in the colors of the bars.
        legend = result.axes["chao1"].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["d2", "d4"]
        assert result.axes["d50"].get_legend() is None
        assert result.provenance.extras["indices"] == [
            "n_clones",
            "gini",
            "chao1",
            "d50",
        ]
        assert result.provenance.n_cells == 32
    finally:
        result.close()
    one = clonal_diversity(
        _NamesOnly(store), store.clones.diversity(clones), show=False
    )
    try:
        # Without a grouping, one bar per panel and no legend.
        assert one.legends == ()
        assert [len(ax.patches) for ax in one.axes.values()] == [1, 1, 1]
    finally:
        one.close()
    # One index may be named as text.
    gini = clonal_diversity(store, diversity, indices="gini", show=False)
    try:
        assert list(gini.axes) == ["gini"]
        assert gini.provenance.extras["indices"] == ["gini"]
    finally:
        gini.close()


def test_the_diversity_plot_on_caller_axes_names_the_groups_where_a_row_starts(
    truth_plot,
) -> None:
    store = truth_plot.store
    diversity = store.clones.diversity(truth_plot.clones, group_by="sample")
    indices = ["n_clones", "shannon", "inverse_simpson"]

    def named(target) -> list[bool]:
        result = clonal_diversity(
            store, diversity, target=target, show_legend=False, show=False
        )
        try:
            return [bool(result.axes[index].get_yticklabels()) for index in indices]
        finally:
            plt.close(result.figure)

    # Panels side by side name the groups once, on the first; panels one
    # above another each start a row; panels without a grid each name them.
    _, row = plt.subplots(1, 3)
    assert named(dict(zip(indices, row, strict=True))) == [True, False, False]
    _, column = plt.subplots(3, 1)
    assert named(dict(zip(indices, column, strict=True))) == [True, True, True]
    loose = plt.figure()
    panels = [loose.add_axes((0.1 + 0.3 * at, 0.1, 0.2, 0.8)) for at in range(3)]
    assert named(dict(zip(indices, panels, strict=True))) == [True, True, True]
    # The caller's other panels name nothing for these: panels from the
    # second column of a row, or a row's second panel alone, still start one.
    _, wide = plt.subplots(1, 4)
    assert named(dict(zip(indices, wide[1:], strict=True))) == [True, False, False]
    _, grid = plt.subplots(2, 2)
    result = clonal_diversity(
        store,
        diversity,
        indices="shannon",
        target={"shannon": grid[0, 1]},
        show_legend=False,
        show=False,
    )
    try:
        ticks = result.axes["shannon"].get_yticklabels()
        assert [tick.get_text() for tick in ticks] == ["d2", "d4"]
    finally:
        plt.close(result.figure)


def test_a_diversity_panel_of_no_defined_value_says_so(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    diversity = store.clones.diversity(clones, group_by="sample")
    # ACE is undefined for every sample, and Pielou's evenness for d2 alone.
    undefined = replace(
        diversity,
        indices=diversity.indices.assign(
            ace=np.nan, pielou=[np.nan, diversity.indices["pielou"].iloc[1]]
        ),
    )
    result = clonal_diversity(
        store, undefined, indices=["ace", "pielou", "shannon"], show=False
    )
    try:
        assert [text.get_text() for text in result.axes["ace"].texts] == [
            "Not defined for any group"
        ]
        # A panel of some defined values draws them without the message.
        assert len(result.axes["pielou"].texts) == 0
        widths = [bar.get_width() for bar in result.axes["pielou"].patches]
        assert np.isnan(widths[0]) and widths[1] == undefined.indices["pielou"].iloc[1]
        assert len(result.axes["shannon"].texts) == 0
    finally:
        result.close()
    # With no value of any index named, there is nothing to plot.
    with pytest.raises(ValueError, match="^No group has a value of ace, so there"):
        clonal_diversity(store, undefined, indices=["ace"], show=False)


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        (b"shannon", TypeError, "indices must be the name of an index"),
        ([], ValueError, "indices must name at least one index"),
        (["shannon", 3], TypeError, "indices must be the name of an index"),
        (["entropy"], ValueError, "names 'entropy', which ds.clones.diversity does"),
        (["gini", "gini"], ValueError, "indices names an index twice"),
    ],
)
def test_the_diversity_plot_checks_its_indices(
    truth_plot, indices, error, message
) -> None:
    store = truth_plot.store
    diversity = store.clones.diversity(truth_plot.clones)
    with pytest.raises(error, match=message):
        clonal_diversity(store, diversity, indices=indices, show=False)
    with pytest.raises(TypeError, match="^diversity must be the ClonalDiversity"):
        clonal_diversity(store, truth_plot.clones, show=False)


def test_record_groups_need_a_store_that_gives_their_labels(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    grouped = store.clones.rarefaction(clones, group_by="sample")
    # The colors of one column's groups range over the column's labels.
    # They are read only when a figure colors its groups.
    unread = table_groups(object(), grouped)
    assert unread.title == "sample"
    with pytest.raises(TypeError, match="^store does not provide the labels of its"):
        unread.scale(["d2"])
    with pytest.raises(TypeError, match="^store does not provide the labels of its"):
        clonal_rarefaction(_NamesOnly(store), grouped, show=False)
    # A figure that does not color the groups never reads their labels.
    drawn = clone_overlap(
        _NamesOnly(store), store.clones.overlap(clones, group_by="sample"), show=False
    )
    assert drawn.provenance.extras["grouping"] == "sample"
    drawn.close()
    # Labels kept apart that are no vocabulary take colors over the groups
    # drawn, as groups of several columns do.
    kept = SimpleNamespace(
        group_keys=pd.DataFrame(
            {"lineage": ["a", "b"]}, index=pd.Index(["a", "b"], name="group")
        ),
        group_by=(),
        groups=None,
    )
    found = table_groups(object(), kept)
    assert found.colors() is None and found.title == "lineage"


def test_small_multiples_share_the_page_three_to_a_row(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    sizes = {}
    for orders, base in (((0,), "coverage"), ((0,), "size"), ((0, 1), "size")):
        result = clonal_rarefaction(
            store,
            store.clones.rarefaction(clones, orders=orders),
            base=base,
            show=False,
        )
        sizes[len(result.axes)] = tuple(result.figure.get_size_inches())
        result.close()
    # One panel of 4.2 by 3.6 inches; two share the page, each 3.75 wide;
    # three fill a row of the page, each 2.5 wide; each keeps the shape.
    assert sizes == {
        1: pytest.approx((4.2, 3.6)),
        2: pytest.approx((7.5, 3.75 * 3.6 / 4.2)),
        3: pytest.approx((7.5, 2.5 * 3.6 / 4.2)),
    }


def test_the_rarefaction_plot_of_complete_samples(truth_plot) -> None:
    store = truth_plot.store
    expanded = store.select_cells(
        truth_plot.clones, value="clone_size", low=2, keep_bounds=True
    )
    result = clonal_rarefaction(
        store,
        store.clones.rarefaction(truth_plot.clones, cell_selection=expanded),
        base="coverage",
        show=False,
    )
    curves = result.tables["curves"]
    # Without singletons, the observed sample is complete.
    assert curves.loc[curves["method"] == "observed", "coverage"].tolist() == [1.0] * 3
    plt.close(result.figure)


def test_the_rarefaction_plot_draws_on_caller_axes(truth_plot) -> None:
    figure, axes = plt.subplots(1, 2)
    result = clonal_rarefaction(
        truth_plot.store,
        truth_plot.store.clones.rarefaction(truth_plot.clones, orders=(0,)),
        target={"order_0": axes[0], "coverage": axes[1]},
        show_legend=False,
        show=False,
    )
    assert result.figure is figure and result.owns_figure is False
    assert axes[1].get_legend() is None
    plt.close(figure)


def test_the_rarefaction_plot_checks_its_arguments(truth_plot) -> None:
    store = truth_plot.store
    curves = store.clones.rarefaction(truth_plot.clones, orders=(0,))
    with pytest.raises(ValueError, match="base must be 'size' or 'coverage'"):
        clonal_rarefaction(store, curves, base="depth", show=False)
    with pytest.raises(TypeError, match="^rarefaction must be the Rarefaction"):
        clonal_rarefaction(store, truth_plot.clones, show=False)
    # The record holds what the plot draws, so the plot reads nothing more
    # than what figures call its clones.
    drawn = clonal_rarefaction(_NamesOnly(store), curves, show=False)
    assert drawn.tables["curves"] is curves.curves
    drawn.close()
    with pytest.raises(TypeError, match="^store does not provide what figures call"):
        clonal_rarefaction(object(), curves, show=False)


def _documented_errors(plot) -> set[str]:
    """The error classes that a plot's ``Raises:`` section lists."""
    doc = inspect.getdoc(inspect.unwrap(plot)) or ""
    found = re.search(r"^Raises:\n(.*?)(?=^\S|\Z)", doc, re.S | re.M)
    assert found, f"{plot.__name__} documents no errors"
    return set(re.findall(r"^\s+(\w+Error):", found.group(1), re.M))


def test_the_record_plots_document_the_error_of_an_unknown_theme(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    # A plot of a record reads only what figures call its clones from the
    # store, so the theme is the one other name it looks up, and its section
    # lists the class that lookup raises.
    for plot, record in (
        (clonal_rarefaction, store.clones.rarefaction(clones, orders=(0,))),
        (clone_overlap, store.clones.overlap(clones, group_by="sample")),
        (
            clone_coupling,
            store.clones.coupling(clones, group_by="cell_type", n_permutations=20),
        ),
    ):
        with pytest.raises(KeyError, match="Unknown theme 'nope'"):
            plot(_NamesOnly(store), record, theme="nope", show=False)
        assert {"TypeError", "ValueError", "KeyError"} <= _documented_errors(plot)


def test_the_overlap_plot_draws_one_index_between_samples(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    result = store.plots.clone_overlap(
        store.clones.overlap(clones, group_by="sample"), show=False
    )
    matrix = result.tables["matrix"]
    assert matrix.index.tolist() == matrix.columns.tolist() == ["d2", "d4"]
    by_sample = TRUTH.group_clone_cells(TRUTH.samples)
    expected = overlap_reference(by_sample["d2"], by_sample["d4"])["morisita_horn"]
    assert matrix.loc["d2", "d4"] == pytest.approx(expected)
    assert matrix.loc["d4", "d2"] == pytest.approx(expected)
    assert np.diag(matrix).tolist() == [1.0, 1.0]
    assert result.tables["pairs"][["group_1", "group_2"]].values.tolist() == [
        ["d2", "d4"]
    ]
    ax = result.axes["matrix"]
    assert ax.get_title() == "Morisita-Horn index"
    # The pair prints its value on both sides; the diagonal stays blank.
    assert [text.get_text() for text in ax.texts] == [f"{expected:.3g}"] * 2
    assert len(result.figure.axes) == 2
    assert result.legends[0].label == "Morisita-Horn index"
    assert result.legends[0].kind == "colorbar"
    assert result.provenance.n_cells == 32
    assert result.provenance.extras["index"] == "morisita_horn"
    assert result.provenance.extras["groups"] == ["d2", "d4"]
    assert result.provenance.extras["clones"] == clones.to_dict()
    plt.close(result.figure)
    shared = clone_overlap(
        store,
        store.clones.overlap(clones, groups=truth_plot.labels),
        index="shared",
        show_legend=False,
        show=False,
    )
    # The clones of each cell type on the diagonal, those of both elsewhere.
    by_type = TRUTH.group_clone_cells(TRUTH.cell_types)
    names = sorted(by_type)
    assert shared.tables["matrix"].values.tolist() == [
        [len(set(by_type[first]) & set(by_type[second])) for second in names]
        for first in names
    ]
    assert [text.get_text() for text in shared.axes["matrix"].texts] == [
        "5",
        "3",
        "5",
        "4",
        "3",
        "4",
    ]
    assert shared.axes["matrix"].get_title() == "Shared clones"
    assert len(shared.figure.axes) == 1
    plt.close(shared.figure)
    # Shown by default: on Agg the owned figure is closed.
    shown = clone_overlap(
        store, store.clones.overlap(clones, group_by="sample"), index="jaccard"
    )
    assert shown.tables["matrix"].loc["d2", "d4"] == pytest.approx(7 / 8)
    assert shown.figure.number not in plt.get_fignums()


def test_the_overlap_plot_prints_the_values_of_at_most_12_groups(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    crossed = clone_overlap(
        store,
        store.clones.overlap(clones, group_by=["sample", "cell_type"]),
        show=False,
    )
    assert len(crossed.tables["matrix"]) == 6
    assert len(crossed.axes["matrix"].texts) == 30
    plt.close(crossed.figure)
    cell_ids = np.asarray([f"c{at:02d}" for at in range(N_CELLS)], dtype=object)
    store.cells.insert("cell", cell_ids, overwrite=True)
    each = clone_overlap(
        store, store.clones.overlap(clones, group_by="cell"), index="shared", show=False
    )
    matrix = each.tables["matrix"]
    # Each of the 32 cells with a clone is a group of one clone, and two
    # cells share a clone when the truth puts them in one: the clones of 9,
    # 6, 5, 4, 3, 2, 2 and 1 cells give 72 + 30 + 20 + 12 + 6 + 2 + 2 = 144
    # ordered pairs.
    assert matrix.index.tolist() == cell_ids[TRUTH.clones >= 0].tolist()
    assert np.diag(matrix).tolist() == [1.0] * 32
    assert matrix.values.sum() == 32 + 144
    assert len(each.axes["matrix"].texts) == 0
    plt.close(each.figure)


def test_the_overlap_plot_draws_on_caller_axes(truth_plot) -> None:
    figure, ax = plt.subplots()
    result = clone_overlap(
        truth_plot.store,
        truth_plot.store.clones.overlap(truth_plot.clones, group_by="sample"),
        target={"matrix": ax},
        show=False,
    )
    assert result.figure is figure and result.owns_figure is False
    plt.close(figure)


def test_the_overlap_plot_checks_its_arguments(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    overlap = store.clones.overlap(clones, group_by="sample")
    with pytest.raises(ValueError, match="index must be one of"):
        clone_overlap(store, overlap, index="dice", show=False)
    with pytest.raises(TypeError, match="^overlap must be the CloneOverlap"):
        clone_overlap(store, clones, show=False)
    with pytest.raises(TypeError, match="^color_scale must be a ColorScale"):
        clone_overlap(store, overlap, color_scale="viridis", show=False)
    # The record holds what the plot draws, so the plot reads nothing more
    # than what figures call its clones.
    drawn = clone_overlap(_NamesOnly(store), overlap, show=False)
    assert drawn.tables["pairs"] is overlap.pairs
    drawn.close()


def test_the_tracking_plot_draws_one_line_per_clone(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    tracking = store.clones.track(clones, group_by="sample", top_n=2)
    result = store.plots.clone_tracking(tracking, show=False)
    ax = result.axes["tracking"]
    # The two largest clones of each sample: 1 and 4, of 3 cells each, in
    # d2, and 1 and 2 in d4.
    assert result.provenance.extras["tracked"] == ["1", "2", "4"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "1",
        "2",
        "4",
    ]
    assert ax.get_legend().get_title().get_text() == "Clone"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d2", "d4"]
    by_sample = TRUTH.group_clone_cells(TRUTH.samples)
    totals = {sample: sum(found.values()) for sample, found in by_sample.items()}
    for line, clone in zip(ax.get_lines(), (0, 1, 3), strict=True):
        assert line.get_ydata().tolist() == pytest.approx(
            [by_sample[sample].get(clone, 0) / totals[sample] for sample in SAMPLES]
        )
    table = result.tables["tracking"]
    assert table["n_cells"].tolist() == [3, 6, 2, 4, 3, 1]
    # Clones are labeled by their IDs, beside them in the table.
    assert table["clone"].tolist() == ["1", "1", "2", "2", "4", "4"]
    assert table["label"].tolist() == table["clone"].tolist()
    # The record's own table is left as it was.
    assert "label" not in tracking.table.columns
    assert result.provenance.n_cells == 19
    assert result.provenance.extras["order"] == ["d2", "d4"]
    assert ax.get_ylabel() == "Fraction of the group's cells"
    plt.close(result.figure)
    ordered = clone_tracking(
        store,
        store.clones.track(clones, group_by="sample", top_n=1),
        order=["d4", "d2"],
        show_legend=False,
        show=False,
    )
    ax = ordered.axes["tracking"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d4", "d2"]
    assert ax.get_lines()[0].get_ydata().tolist() == pytest.approx([6 / 19, 3 / 13])
    assert ax.get_legend() is None
    plt.close(ordered.figure)
    # Without a grouping, every cell with a clone is one group.
    pooled = clone_tracking(store, store.clones.track(clones, labels=["1"]), show=False)
    ax = pooled.axes["tracking"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["all cells"]
    assert ax.get_lines()[0].get_ydata().tolist() == [9 / 32]
    plt.close(pooled.figure)


def test_the_tracking_plot_places_groups_by_all_their_labels(truth_plot) -> None:
    tracking = truth_plot.store.clones.track(
        truth_plot.clones, group_by="sample", groups=truth_plot.labels, labels=["8"]
    )
    result = clone_tracking(truth_plot.store, tracking, show=False)
    ax = result.axes["tracking"]
    positions = [f"{s}, {t}" for s in SAMPLES for t in sorted(CELL_TYPES)]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == positions
    # Clone 8 holds one cell: an undifferentiated cell of d4, which holds
    # five cells with a clone.
    assert ax.get_lines()[0].get_ydata().tolist() == pytest.approx(
        [0, 0, 0, 0, 0, 1 / 5]
    )
    assert result.tables["tracking"].columns.tolist() == [
        "clone",
        "label",
        "group",
        "n_cells",
        "fraction",
    ]
    assert tracking.group_keys.columns.tolist() == ["sample", "cluster_labels"]
    plt.close(result.figure)


def test_the_tracking_plot_needs_only_the_tracked_clones(truth_plot) -> None:
    result = clone_tracking(
        _NamesOnly(truth_plot.store),
        truth_plot.store.clones.track(truth_plot.clones, group_by="sample", top_n=1),
        show=False,
    )
    legend = result.axes["tracking"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["1"]
    plt.close(result.figure)


def test_the_tracking_plot_checks_its_arguments(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    tracking = store.clones.track(clones, group_by="sample", top_n=1)
    shown = clone_tracking(store, tracking)
    assert shown.provenance.n_cells == 9
    assert shown.figure.number not in plt.get_fignums()
    with pytest.raises(ValueError, match="order must list the groups"):
        clone_tracking(store, tracking, order=["d2"], show=False)
    with pytest.raises(TypeError, match="order must be a sequence"):
        clone_tracking(store, tracking, order="d2", show=False)
    # The order is checked before anything is read.
    with pytest.raises(TypeError, match="order must be a sequence"):
        clone_tracking(object(), clones, order=np.asarray([["d2"]]), show=False)
    ordered = clone_tracking(
        store, tracking, order=np.asarray(["d4", "d2"]), show=False
    )
    ax = ordered.axes["tracking"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d4", "d2"]
    plt.close(ordered.figure)
    with pytest.raises(TypeError, match="^tracking must be the CloneTracking"):
        clone_tracking(store, clones, show=False)


@pytest.mark.parametrize("name", ["label", "clone", "group", "n_cells", "fraction"])
def test_tracking_groups_by_a_column_named_like_the_tables_columns(
    truth_plot, name
) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    store.cells.insert(name, TRUTH.samples.astype(object), overwrite=True)
    # The tracking's labels stay in its keys, so the figure's table gives
    # each clone its label beside them, and the figure places the samples as
    # a column of any other name does.
    tracking = store.clones.track(clones, group_by=name, top_n=1)
    assert tracking.table.columns.tolist() == ["clone", "group", "n_cells", "fraction"]
    neutral = clone_tracking(
        store, store.clones.track(clones, group_by="sample", top_n=1), show=False
    )
    result = clone_tracking(store, tracking, show=False)
    try:
        pd.testing.assert_frame_equal(
            result.tables["tracking"], neutral.tables["tracking"]
        )
        assert result.tables["tracking"].columns.tolist() == [
            "clone",
            "label",
            "group",
            "n_cells",
            "fraction",
        ]
        ax, other = result.axes["tracking"], neutral.axes["tracking"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d2", "d4"]
        assert [line.get_ydata().tolist() for line in ax.get_lines()] == [
            line.get_ydata().tolist() for line in other.get_lines()
        ]
    finally:
        result.close()
        neutral.close()


def test_the_tracking_plot_takes_an_order_of_numbers(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    assert store.cells.get_dtype("time_point") == np.int64
    tracking = store.clones.track(clones, group_by="time_point", top_n=1)
    by_text = clone_tracking(store, tracking, order=["4", "2"], show=False)
    by_number = clone_tracking(store, tracking, order=[4, 2], show=False)
    # Clone 1 holds 6 of the 19 cells of time point 4 and 3 of the 13 of 2.
    for result in (by_text, by_number):
        ax = result.axes["tracking"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["4", "2"]
        assert ax.get_lines()[0].get_ydata().tolist() == pytest.approx([6 / 19, 3 / 13])
        plt.close(result.figure)
    with pytest.raises(ValueError, match="order must list the groups"):
        clone_tracking(store, tracking, order=[4.0, 2.0], show=False)


N_MANY = 200


@pytest.fixture(scope="module")
def many_groups(tmp_path_factory) -> SimpleNamespace:
    """200 cells, cell i in clone i // 2, each with a label of its own."""
    path = tmp_path_factory.mktemp("many_groups") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_MANY, 3), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert(
        "clone", np.asarray([f"k{i // 2:03d}" for i in range(N_MANY)], dtype=object)
    )
    store.cells.insert(
        "cell", np.asarray([f"cell{i:03d}" for i in range(N_MANY)], dtype=object)
    )
    store.cells.insert("half", np.asarray([f"h{i // 100}" for i in range(N_MANY)]))
    store.cells.insert("eighth", np.asarray([f"e{i % 8}" for i in range(N_MANY)]))
    store.cells.insert("every_cell", np.ones(N_MANY, dtype=bool))
    for n_first in (12, 13, 40, 41):
        store.cells.insert(f"first_{n_first}", np.arange(N_MANY) < n_first)
    clones = store.clones.from_column(
        "clone", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    return SimpleNamespace(store=store, clones=clones)


def _overlap_of_first(many_groups, n_first: int, **options):
    """The overlap plot of the first cells, each a group of its own."""
    store = many_groups.store
    overlap = store.clones.overlap(
        many_groups.clones,
        group_by="cell",
        cell_selection=store.snapshot_cell_selection(f"first_{n_first}"),
    )
    return clone_overlap(store, overlap, show=False, **options)


def test_the_overlap_plot_prints_values_of_at_most_12_groups(many_groups) -> None:
    twelve = _overlap_of_first(many_groups, 12)
    ax = twelve.axes["matrix"]
    # Every pair off the diagonal prints its value: 1 for the two cells of a
    # clone, 0 for cells of two clones.
    assert len(ax.texts) == 12 * 11
    shown = {text.get_position(): text for text in ax.texts}
    assert shown[(1, 0)].get_text() == "1" and shown[(2, 0)].get_text() == "0"
    # Dark text on the light color of 1, light text on the dark color of 0.
    assert shown[(1, 0)].get_color() == "black"
    assert shown[(2, 0)].get_color() == "white"
    plt.close(twelve.figure)
    thirteen = _overlap_of_first(many_groups, 13)
    assert len(thirteen.axes["matrix"].texts) == 0
    assert len(thirteen.tables["matrix"]) == 13
    plt.close(thirteen.figure)


@pytest.mark.parametrize("n_first", [12, 40, 41])
def test_the_overlap_plot_names_every_nth_group_that_its_axes_hold(
    many_groups, n_first
) -> None:
    result = _overlap_of_first(many_groups, n_first)
    try:
        result.figure.draw_without_rendering()
        ax = result.axes["matrix"]
        order = result.tables["matrix"].index.tolist()
        assert sorted(order) == [f"cell{i:03d}" for i in range(n_first)]
        for axis in (ax.xaxis, ax.yaxis):
            # Every nth group in the order drawn, every one of 12, at most
            # 40, and none running into the next.
            step, names = axis_names(axis)
            assert step is not None and names == order[::step]
            assert step == 1 if n_first == 12 else len(names) <= 40
            assert names_keep_apart(axis)
    finally:
        plt.close(result.figure)


def test_the_overlap_plot_grows_to_the_width_of_a_page(many_groups) -> None:
    store, clones = many_groups.store, many_groups.clones
    sizes = {}
    for column, n_groups in (("half", 2), ("eighth", 8), ("cell", N_MANY)):
        result = clone_overlap(
            store, store.clones.overlap(clones, group_by=column), show=False
        )
        assert len(result.tables["matrix"]) == n_groups
        sizes[n_groups] = result.figure.get_size_inches().tolist()
        plt.close(result.figure)
    # A side of 0.55 inches per group and 2.5 more, at least 4 and at most
    # 6.3, beside 1.2 inches for the color bar: 7.5 inches at most.
    assert sizes == {
        2: pytest.approx([5.2, 4.0]),
        8: pytest.approx([7.5, 6.3]),
        N_MANY: pytest.approx([7.5, 6.3]),
    }


def _fits_the_figure(result) -> bool:
    """Whether the tracking plot's legend lies within the figure's height."""
    result.figure.canvas.draw()
    legend = result.axes["tracking"].get_legend()
    extent = legend.get_window_extent(result.figure.canvas.get_renderer())
    return extent.height <= result.figure.bbox.height


def test_the_tracking_plot_fits_many_clones_and_groups(many_groups) -> None:
    store, clones = many_groups.store, many_groups.clones
    # Each cell is a group of one clone: 100 clones through 200 groups.
    every = clone_tracking(
        store, store.clones.track(clones, group_by="cell", top_n=1), show=False
    )
    ax = every.axes["tracking"]
    legend = ax.get_legend()
    assert legend.get_title().get_text() == "Clone (80 of 100)"
    assert [text.get_text() for text in legend.get_texts()] == [
        f"k{k:03d}" for k in range(80)
    ]
    assert _fits_the_figure(every)
    assert len(ax.get_lines()) == 100 and len(ax.get_xticks()) == 0
    assert every.figure.get_size_inches().tolist() == pytest.approx([7.5, 4.2])
    plt.close(every.figure)
    # The ten largest clones of each of eight groups: the legend lists all
    # of them, in columns that fit the figure.
    eighths = clone_tracking(
        store, store.clones.track(clones, group_by="eighth", top_n=10), show=False
    )
    legend = eighths.axes["tracking"].get_legend()
    n_tracked = len(eighths.provenance.extras["tracked"])
    assert n_tracked > 20
    assert len(legend.get_texts()) == n_tracked
    assert legend.get_title().get_text() == "Clone"
    assert _fits_the_figure(eighths)
    names = [f"e{k}" for k in range(8)]
    ticks = eighths.axes["tracking"].get_xticklabels()
    assert [tick.get_text() for tick in ticks] == names
    plt.close(eighths.figure)


def test_the_coupling_plot_draws_couplings_by_default(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    coupling = store.clones.coupling(clones, group_by="cell_type", n_permutations=200)
    result = store.plots.clone_coupling(coupling, show=False)
    matrix = result.tables["matrix"]
    order = list(coupling.order)
    assert matrix.index.tolist() == matrix.columns.tolist() == order
    # Each pair's cosine of clone shares, not its z-score, on both sides.
    assert not np.allclose(coupling.pairs["coupling"], coupling.pairs["z_score"])
    for row in coupling.pairs.itertuples(index=False):
        assert matrix.loc[row.group_1, row.group_2] == row.coupling
        assert matrix.loc[row.group_2, row.group_1] == row.coupling
    ax = result.axes["matrix"]
    assert ax.get_title() == "Cosine of clone shares"
    assert result.legends[0].label == "Cosine of clone shares"
    assert result.provenance.extras["value"] == "coupling"
    # Cosines are not negative, so a sequential scale from their least.
    assert ax.images[0].cmap.name == "viridis"
    plt.close(result.figure)
    # A coupling measured without permutations has no z-scores, and its
    # couplings draw by default.
    unpermuted = store.plots.clone_coupling(
        replace(coupling, n_permutations=0), show=False
    )
    assert unpermuted.tables["matrix"].equals(matrix)
    plt.close(unpermuted.figure)


def test_the_coupling_plot_draws_z_scores_in_the_order_of_the_tree(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    coupling = store.clones.coupling(clones, group_by="cell_type", n_permutations=200)
    result = store.plots.clone_coupling(coupling, value="z_score", show=False)
    matrix = result.tables["matrix"]
    order = list(coupling.order)
    assert matrix.index.tolist() == matrix.columns.tolist() == order
    for row in coupling.pairs.itertuples(index=False):
        assert matrix.loc[row.group_1, row.group_2] == row.z_score
        assert matrix.loc[row.group_2, row.group_1] == row.z_score
    assert np.isnan(np.diag(matrix)).all()
    assert result.tables["pairs"] is coupling.pairs
    ax = result.axes["matrix"]
    assert ax.get_title() == "Cosine of clone shares, z-score"
    assert [tick.get_text() for tick in ax.get_yticklabels()] == order
    # Six cells off the diagonal print their z-scores, on a scale centered on 0.
    assert len(ax.texts) == 6
    low, high = ax.images[0].get_clim()
    assert low == -high == -np.abs(coupling.pairs["z_score"]).max()
    # z-scores take a scale centered on 0 even when all lie above it.
    above = replace(
        coupling,
        pairs=coupling.pairs.assign(z_score=np.abs(coupling.pairs["z_score"]) + 1),
    )
    positive = clone_coupling(store, above, value="z_score", show=False)
    low, high = positive.axes["matrix"].images[0].get_clim()
    assert low == -high == -above.pairs["z_score"].max()
    assert positive.axes["matrix"].images[0].cmap.name == "RdBu_r"
    plt.close(positive.figure)
    # The dendrogram above the matrix places its leaves under the columns.
    dendrogram = result.axes["dendrogram"]
    assert dendrogram.get_gid() == "top-dendrogram"
    assert dendrogram.get_xlim() == ax.get_xlim()
    assert result.legends[0].label == "Z-score of the coupling"
    assert result.legends[0].kind == "colorbar"
    assert result.provenance.extras["method"] == "share_cosine"
    assert result.provenance.extras["groups"] == order
    assert result.provenance.n_cells == 32
    plt.close(result.figure)
    values = clone_coupling(store, coupling, value="coupling", show=False)
    assert values.tables["matrix"].loc[order[0], order[1]] == pytest.approx(
        coupling.pairs.set_index(["group_1", "group_2"]).loc[tuple(sorted(order[:2]))][
            "coupling"
        ]
    )
    # Cosines are not negative, so a sequential scale from their least.
    assert values.axes["matrix"].images[0].cmap.name == "viridis"
    assert values.axes["matrix"].get_title() == "Cosine of clone shares"
    plt.close(values.figure)


def test_the_coupling_plot_blanks_values_that_are_not_finite(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    coupling = store.clones.coupling(
        clones, group_by="cell_type", method="normalized_covariance", n_permutations=50
    )
    # Without permutations there are no z-scores to draw.
    with pytest.raises(ValueError, match="without permutations, so it has no"):
        clone_coupling(
            store, replace(coupling, n_permutations=0), value="z_score", show=False
        )
    # A null without spread gives a z-score of NaN, or an infinite one, which
    # stay blank; the finite one is printed on both sides.
    blank = replace(
        coupling, pairs=coupling.pairs.assign(z_score=[np.nan, np.inf, 1.5])
    )
    result = clone_coupling(store, blank, value="z_score", show=False)
    assert [text.get_text() for text in result.axes["matrix"].texts] == ["1.5"] * 2
    assert np.isinf(result.tables["matrix"].to_numpy()).sum() == 2
    plt.close(result.figure)
    # The truth's covariances are all above 0, which a sequential scale
    # draws; covariances below 0 take a scale centered on 0.
    assert coupling.pairs["coupling"].gt(0).all()
    values = clone_coupling(store, coupling, value="coupling", show=False)
    assert values.axes["matrix"].images[0].cmap.name == "viridis"
    plt.close(values.figure)
    below = replace(
        coupling, pairs=coupling.pairs.assign(coupling=-coupling.pairs["coupling"])
    )
    negative = clone_coupling(store, below, value="coupling", show=False)
    low, high = negative.axes["matrix"].images[0].get_clim()
    assert low == -high == -coupling.pairs["coupling"].max()
    plt.close(negative.figure)


def test_the_coupling_plot_checks_its_arguments(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    coupling = store.clones.coupling(clones, group_by="cell_type", n_permutations=0)
    with pytest.raises(ValueError, match="value must be one of"):
        clone_coupling(store, coupling, value="p_value", show=False)
    with pytest.raises(TypeError, match="CloneCoupling"):
        clone_coupling(store, coupling.pairs, show=False)
    with pytest.raises(ValueError, match="does not follow its linkage"):
        clone_coupling(
            store,
            replace(coupling, order=coupling.order[::-1]),
            value="coupling",
            show=False,
        )


def _late_cells(clone: int) -> dict[str, int]:
    """A clone's d4 cells of each type, counted cell by cell."""
    found: dict[str, int] = {}
    for cell_type, own, sample in zip(
        TRUTH.cell_types, TRUTH.clones, TRUTH.samples, strict=True
    ):
        if own == clone and sample == "d4":
            found[str(cell_type)] = found.get(str(cell_type), 0) + 1
    return found


@pytest.fixture(scope="module")
def fates_plot(truth_plot) -> SimpleNamespace:
    store = truth_plot.store
    store.cells.insert("is_d2", TRUTH.samples == "d2", overwrite=True)
    store.cells.insert("is_d4", TRUTH.samples == "d4", overwrite=True)
    bias = store.clones.fate_bias(
        truth_plot.clones,
        group_by="cell_type",
        early=store.snapshot_cell_selection("is_d2"),
        late=store.snapshot_cell_selection("is_d4"),
    )
    return SimpleNamespace(store=store, bias=bias, clones=truth_plot.clones)


def test_the_fates_plot_groups_clones_by_their_dominant_fate(fates_plot) -> None:
    result = fates_plot.store.plots.clone_fate_bias(fates_plot.bias, show=False)
    table = result.tables["clones"]
    types = sorted(CELL_TYPES)
    assert table.columns.tolist() == [
        "clone",
        "label",
        "fate",
        "n_early_cells",
        "n_late_cells",
    ]
    # Each clone's late cells in each fate, zeros included, in the drawn
    # order and then the fates' order.
    late = result.tables["fates"]
    assert late.columns.tolist() == ["clone", "label", "fate", "n_cells"]
    assert late["clone"].tolist() == [
        clone for clone in table["clone"].tolist() for _fate in types
    ]
    assert late["label"].tolist() == [
        label for label in table["label"].tolist() for _fate in types
    ]
    assert late["fate"].astype(str).tolist() == types * len(table)
    tested = sorted({int(c) for c in TRUTH.clones[TRUTH.samples == "d4"] if c >= 0})

    def dominant(clone: int) -> str | None:
        found = _late_cells(clone)
        top = max(found.values())
        winners = [name for name in types if found.get(name, 0) == top]
        return winners[0] if len(winners) == 1 else None

    # By dominant fate in the order of the fates, ties last, then by late
    # cells from the most and by clone.
    def rank(clone: int) -> tuple:
        fate = dominant(clone)
        return (
            types.index(fate) if fate else len(types),
            -sum(_late_cells(clone).values()),
            clone,
        )

    ordered = sorted(tested, key=rank)
    assert table["clone"].tolist() == [str(clone + 1) for clone in ordered]
    # Clones are labeled by their IDs.
    assert table["label"].tolist() == table["clone"].tolist()
    assert [
        name if isinstance(name, str) else None for name in table["fate"].tolist()
    ] == [dominant(clone) for clone in ordered]
    by_fate = late.set_index(["clone", "fate"])["n_cells"]
    for row, clone in zip(table.itertuples(index=False), ordered, strict=True):
        found = _late_cells(clone)
        assert [by_fate.loc[(row.clone, name)] for name in types] == [
            found.get(name, 0) for name in types
        ]
        assert row.n_late_cells == sum(found.values())
        assert row.n_early_cells == int(
            np.count_nonzero((TRUTH.clones == clone) & (TRUTH.samples == "d2"))
        )
    fates = result.axes["fates"]
    assert [tick.get_text() for tick in fates.get_xticklabels()] == table[
        "label"
    ].tolist()
    # One bar per clone and fate, stacked to each clone's late cells.
    heights = np.zeros(len(table))
    for container in fates.containers:
        heights += [bar.get_height() for bar in container]
    np.testing.assert_array_equal(heights, table["n_late_cells"])
    assert [bar.get_height() for bar in result.axes["early"].patches] == table[
        "n_early_cells"
    ].tolist()
    assert result.axes["early"].get_xlim() == fates.get_xlim()
    assert result.provenance.extras["clones"] == fates_plot.clones.to_dict()
    assert result.provenance.extras["bias"] == fates_plot.bias.to_dict()
    assert result.provenance.notes == ("clone_fate_bias",)
    plt.close(result.figure)


def test_the_fates_plot_draws_the_top_clones_and_their_shares(fates_plot) -> None:
    store, bias = fates_plot.store, fates_plot.bias
    shares = clone_fate_bias(store, bias, max_clones=3, normalize=True, show=False)
    table = shares.tables["clones"]
    # The three clones with the most late cells: clones 3, 1 and 2 hold 4,
    # 4 and 3, ties to the larger clone overall.
    late = {
        clone: sum(_late_cells(clone).values())
        for clone in {int(c) for c in TRUTH.clones[TRUTH.samples == "d4"] if c >= 0}
    }
    chosen = sorted(late, key=lambda clone: (-late[clone], clone))[:3]
    assert sorted(table["clone"].astype(int) - 1) == sorted(chosen)
    # Four clones hold one late cell each; the fifth clone drawn is the one
    # of them that is larger overall, the lowest code.
    assert sorted(late.values(), reverse=True) == [6, 4, 3, 2, 1, 1, 1, 1]
    five = clone_fate_bias(store, bias, max_clones=5, show=False)
    assert sorted(five.tables["clones"]["clone"].astype(int) - 1) == [0, 1, 2, 3, 4]
    plt.close(five.figure)
    heights = np.zeros(3)
    for container in shares.axes["fates"].containers:
        heights += [bar.get_height() for bar in container]
    np.testing.assert_allclose(heights, 1.0)
    assert shares.axes["fates"].get_ylabel() == "Fraction of late cells"
    plt.close(shares.figure)
    with pytest.raises(ValueError, match="^max_clones must be at least 1$"):
        clone_fate_bias(store, bias, max_clones=0, show=False)
    with pytest.raises(TypeError, match="normalize must be a boolean"):
        clone_fate_bias(store, bias, normalize="yes", show=False)
    with pytest.raises(TypeError, match="store does not provide clones.load_fate_bias"):
        clone_fate_bias(object(), bias, show=False)


def test_the_coupling_plot_names_at_most_40_groups(many_groups) -> None:
    store = many_groups.store
    for n_first in (40, 41):
        coupling = store.clones.coupling(
            many_groups.clones,
            group_by="cell",
            cell_selection=store.snapshot_cell_selection(f"first_{n_first}"),
            n_permutations=0,
        )
        # Each cell is a group of its own, and the two cells of a clone share
        # all of it.
        result = clone_coupling(store, coupling, value="coupling", show=False)
        matrix = result.tables["matrix"]
        assert matrix.loc["cell000", "cell001"] == pytest.approx(1.0)
        assert matrix.loc["cell000", "cell002"] == 0.0
        ax = result.axes["matrix"]
        # Every nth group in the order drawn, at most 40, apart once laid out.
        result.figure.draw_without_rendering()
        step, names = axis_names(ax.xaxis)
        assert step is not None and names == matrix.index.tolist()[::step]
        assert len(names) <= 40 and names_keep_apart(ax.xaxis)
        assert len(ax.texts) == 0
        plt.close(result.figure)
    # Shown by default: on Agg the owned figure is closed.
    shown = clone_coupling(store, coupling, value="coupling")
    assert shown.figure.number not in plt.get_fignums()
    assert shown.figure.get_size_inches()[0] <= MAX_FIGURE_WIDTH_INCHES


def test_the_fates_plot_names_at_most_40_clones(many_groups) -> None:
    store = many_groups.store
    # The even cell of each clone is early and the odd one late, in one of
    # four fates.
    store.cells.insert("even_cell", np.arange(N_MANY) % 2 == 0, overwrite=True)
    store.cells.insert("odd_cell", np.arange(N_MANY) % 2 == 1, overwrite=True)
    bias = store.clones.fate_bias(
        many_groups.clones,
        group_by="eighth",
        early=store.snapshot_cell_selection("even_cell"),
        late=store.snapshot_cell_selection("odd_cell"),
    )
    # Every nth clone drawn, as many as the axis holds once laid out: of 41
    # clones too, where none was named.
    for max_clones in (40, 41):
        result = clone_fate_bias(store, bias, max_clones=max_clones, show=False)
        assert len(result.tables["clones"]) == max_clones
        result.figure.draw_without_rendering()
        names_every_nth_item(
            result.axes["fates"].xaxis, result.tables["clones"]["label"].tolist()
        )
        plt.close(result.figure)
    shown = clone_fate_bias(store, bias, max_clones=100)
    assert shown.figure.number not in plt.get_fignums()
    assert shown.figure.get_size_inches()[0] <= MAX_FIGURE_WIDTH_INCHES


@pytest.mark.parametrize(
    "name", ["label", "clone", "fate", "n_early_cells", "n_late_cells", "n_cells"]
)
def test_the_fates_plot_draws_fates_named_like_its_tables_columns(
    fates_plot, name
) -> None:
    store = fates_plot.store
    neutral = clone_fate_bias(store, fates_plot.bias, show=False)
    # The monocytes take a fate named like a column of the tables, which the
    # tables hold as a value; the figure and the late cells stay those of the
    # cell types.
    fate = np.where(TRUTH.cell_types == "Monocyte", name, TRUTH.cell_types)
    store.cells.insert("named_fate", fate.astype(object), overwrite=True)
    bias = store.clones.fate_bias(
        fates_plot.clones,
        group_by="named_fate",
        early=store.snapshot_cell_selection("is_d2"),
        late=store.snapshot_cell_selection("is_d4"),
    )
    result = clone_fate_bias(store, bias, show=False)
    try:
        renamed = {"Monocyte": name}
        for key in ("clones", "fates"):
            found, expected = result.tables[key], neutral.tables[key]
            assert found.columns.tolist() == expected.columns.tolist()
            cells = expected.assign(
                fate=expected["fate"].astype(object).replace(renamed)
            )
            assert sorted(
                map(tuple, found.astype(object).fillna("-").values.tolist())
            ) == sorted(map(tuple, cells.astype(object).fillna("-").values.tolist()))
        legend = [text.get_text() for text in result.axes["fates"].get_legend().texts]
        assert sorted(legend) == sorted(renamed.get(t, t) for t in CELL_TYPES)
        heights = sorted(
            bar.get_height()
            for container in result.axes["fates"].containers
            for bar in container
        )
        assert heights == sorted(
            bar.get_height()
            for container in neutral.axes["fates"].containers
            for bar in container
        )
    finally:
        result.close()
        neutral.close()


def test_the_fates_plot_draws_clones_without_a_dominant_fate_last(fates_plot) -> None:
    store = fates_plot.store
    # With the cells of d2 late, clone 3 (code 2) holds one monocyte and one
    # undifferentiated cell there: a tie.
    bias = store.clones.fate_bias(
        fates_plot.clones,
        group_by="cell_type",
        early=store.snapshot_cell_selection("is_d4"),
        late=store.snapshot_cell_selection("is_d2"),
    )
    result = clone_fate_bias(store, bias, show=False)
    table = result.tables["clones"]
    tied = table["fate"].isna().to_numpy()
    assert table.loc[tied, "clone"].tolist() == ["3"]
    # Clones without a dominant fate come after every other clone.
    assert tied[-1] and not tied[:-1].any()
    plt.close(result.figure)


def test_a_regrouped_expansion_of_a_label_artifact_keeps_apart_only_its_kept_labels(
    truth_plot, truth_clonotypes
) -> None:
    store = truth_plot.store
    # Clonotypes measured within the cell types of a label artifact, whose
    # groups' keys are the label artifact's kind and then the receptor type.
    expansion = store.clones.expansion(truth_clonotypes, groups=truth_plot.labels)
    assert store.clones.load_expansion(expansion).group_keys.columns.tolist() == [
        "cluster_labels",
        "receptor_type",
    ]
    result = clonal_expansion(store, expansion, group_by="sample", show=False)
    result.close()
    # Regrouped by sample, each cell keeps its receptor type, which the
    # clonotypes keep apart, and leaves its cell type, which only grouped
    # the expansion; the groups are the samples by receptor type of the cells
    # with a clonotype.
    clones = store.clones.load_clones(truth_clonotypes).cells.loc[TRUTH.cell_ids]
    with_clone = clones["clone"].notna().to_numpy()
    expected = {
        f"{sample}, {kind}"
        for sample, kind in zip(
            TRUTH.samples[with_clone],
            clones["receptor_type"].astype(str).to_numpy()[with_clone],
            strict=True,
        )
    }
    assert set(result.tables["categories"]["group"]) == expected
    assert result.provenance.extras["grouping"] == "sample, receptor_type"


def test_default_figure_sizes_follow_the_clones_and_groups_drawn(
    truth_plot, many_groups, truth_clonotypes
) -> None:
    # The largest clones' panel grows with the clones drawn, not with the
    # rows of its table, one per clone and group: 20 clones in 200 groups.
    store = many_groups.store
    result = clone_abundance(
        store, many_groups.clones, group_by="cell", max_clones=20, show=False
    )
    size = tuple(result.figure.get_size_inches())
    result.close()
    assert len(result.tables["top"]) == 20 * 200
    assert size == pytest.approx((7.5, 6.8))
    # receptor_qc stands its three panels in one column at the page width,
    # however many groups it draws: a label for each of the truth's 48 cells.
    store = truth_plot.store
    store.cells.insert("own", TRUTH.cell_ids.astype(object), overwrite=True)
    receptors = store.artifacts.inspect(truth_clonotypes).input_ref("receptors")
    qc = store.plots.receptor_qc(receptors, group_by="own", show=False)
    size = tuple(qc.figure.get_size_inches())
    qc.close()
    assert qc.tables["receptor_type"]["group"].nunique() == N_CELLS
    assert size == pytest.approx((7.5, 3 * 2.4))


def test_each_summarys_figure_titles_its_named_artifact(truth_plot) -> None:
    from cytearc.plotting import NamedCellArtifact

    store, clones, labels = truth_plot.store, truth_plot.clones, truth_plot.labels
    named = NamedCellArtifact("Cell type", labels)
    store.cells.insert("is_d2", TRUTH.samples == "d2", overwrite=True)
    store.cells.insert("is_d4", TRUTH.samples == "d4", overwrite=True)
    # Each summary groups by the named artifact, and its figure takes the
    # name; the record keeps the artifact alone.
    cases = [
        (
            clonal_rarefaction,
            store.clones.rarefaction(clones, groups=named, orders=(0,)),
        ),
        (clonal_diversity, store.clones.diversity(clones, groups=named)),
        (clone_overlap, store.clones.overlap(clones, groups=named)),
        (clone_tracking, store.clones.track(clones, groups=named, top_n=2)),
        (
            clone_coupling,
            store.clones.coupling(clones, groups=named, n_permutations=20),
        ),
        (
            clone_fate_bias,
            store.clones.fate_bias(
                clones,
                groups=named,
                early=store.snapshot_cell_selection("is_d2"),
                late=store.snapshot_cell_selection("is_d4"),
            ),
        ),
    ]
    for figure, record in cases:
        result = figure(store, record, groups=named, show=False)
        try:
            assert result.provenance.extras["grouping"] == "Cell type", figure
            # Without the name, the figure takes the column the labels copied.
            bare = figure(store, record, show=False)
            assert bare.provenance.extras["grouping"] == "cell_type", figure
            bare.close()
        finally:
            result.close()
