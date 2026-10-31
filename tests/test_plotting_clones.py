"""The clone plots, on clones whose sizes are counted by hand."""

from collections import Counter
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.clones import diversity as dv
from cytearc.datastore.datastore import DataStore
from cytearc.plotting import (
    clonal_expansion,
    clonal_overlap,
    clonal_rarefaction,
    clone_abundance,
    clone_tracking,
)
from cytearc.plotting._groups import cell_groups
from tests.clone_helpers import (
    every_cell,
    import_truth_clones,
    overlap_reference,
    write_truth_store,
)
from tests.clone_truth import CELL_TYPES, N_CELLS, clone_truth
from tests.storage_helpers import insert_nullable_cell_column, write_count_store

pytestmark = pytest.mark.usefixtures("figures_closed")

# Clone a holds c0, c1 and c2, clone b c3, and c4 and c5 have no clone; c0,
# c1 and c3 are in group x, c2 and c4 in group y, and c5 has no group.
CLONES = ["a", "a", "a", "b", "", ""]
GROUPS = ["x", "x", "y", "x", "y", None]


@pytest.fixture(scope="module")
def plotted(tmp_path_factory):
    path = tmp_path_factory.mktemp("clone_plot") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((6, 3), dtype=int)}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    store.cells.insert("lineage", np.asarray(CLONES, dtype=object))
    store.cells.insert("group", np.asarray(GROUPS, dtype=object))
    store.cells.insert("every_cell", np.ones(6, dtype=bool))
    store.cells.insert("first_four", np.arange(6) < 4)
    cells = store.snapshot_cell_selection("every_cell")
    clones = store.clones.from_column("lineage", cell_selection=cells)
    return store, clones


def test_without_groups_the_plot_counts_every_cell(plotted) -> None:
    store, clones = plotted

    result = clone_abundance(store, clones, show=False)

    assert result.tables["sizes"].to_dict("list") == {
        "group": ["all cells", "all cells"],
        "clone_size": [1, 3],
        "n_clones": [1, 1],
    }
    assert result.tables["top"].to_dict("list") == {
        "clone": ["a", "b"],
        "cells": [3, 1],
    }
    assert set(result.axes) == {"sizes", "top"}
    assert result.legends == ()
    assert result.provenance.n_cells == 4
    assert result.provenance.extras["clones"] == clones.to_dict()
    assert [tick.get_text() for tick in result.axes["top"].get_yticklabels()] == [
        "a",
        "b",
    ]
    plt.close(result.figure)


def test_shown_figures_that_the_plot_owns_are_closed(plotted) -> None:
    store, clones = plotted
    result = clone_abundance(store, clones, show=True)
    assert result.owns_figure
    assert result.figure.number not in plt.get_fignums()


def test_a_grouping_column_splits_each_clone_by_group(plotted) -> None:
    store, clones = plotted

    result = store.plots.clone_abundance(clones, group_by="group", top=1, show=False)

    # Within x, clone a holds two cells and b one; within y, a holds one.
    assert result.tables["sizes"].to_dict("list") == {
        "group": ["x", "x", "y"],
        "clone_size": [1, 2, 1],
        "n_clones": [1, 1, 1],
    }
    assert result.tables["top"].to_dict("list") == {"clone": ["a"], "x": [2], "y": [1]}
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
        "clone": ["a", "b"],
        "x": [2, 1],
        "y": [1, 0],
    }
    assert result.legends[0].label == "cluster_labels"
    plt.close(result.figure)


def test_the_plot_refuses_clones_it_cannot_draw(plotted) -> None:
    store, clones = plotted
    figures = set(plt.get_fignums())
    store.cells.insert("only_c4", np.arange(6) == 4, overwrite=True)
    with pytest.raises(ValueError, match="^top must be at least 1$"):
        clone_abundance(store, clones, top=0, show=False)
    with pytest.raises(TypeError, match="^top must be an integer$"):
        clone_abundance(store, clones, top="3", show=False)
    with pytest.raises(ValueError, match="Pass group_by or groups, not both"):
        clone_abundance(store, clones, group_by="group", groups=clones, show=False)
    with pytest.raises(TypeError, match="group_by must be the name"):
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


def test_cell_groups_align_labels_to_the_artifacts_cells(plotted) -> None:
    store, clones = plotted
    assert cell_groups(store, clones, group_by=None, groups=None) == (None, [], None)
    codes, names, name = cell_groups(store, clones, group_by="group", groups=None)
    assert (codes.tolist(), names, name) == ([0, 0, 1, 0, 1, -1], ["x", "y"], "group")


def test_cell_groups_align_labels_to_clones_of_later_cells(plotted) -> None:
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
    codes, names, _name = cell_groups(store, middle, group_by=None, groups=labels)
    assert (codes.tolist(), names) == ([1, 0, 1], ["x", "y"])


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

    assert result.tables["top"].to_dict("list") == {
        "clone": ["9", "10", "x"],
        "cells": [2, 2, 1],
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
    assert table.columns.tolist() == ["group", "category", "cells", "fraction"]
    assert table["group"].tolist() == ["d2"] * 6 + ["d4"] * 6
    # Each sample's cells per category, from its clone sizes.
    expected = []
    for sample in ("d2", "d4"):
        cells = dict.fromkeys(CATEGORIES, 0)
        for size in TRUTH.group_clone_sizes(TRUTH.samples)[sample]:
            category = 0 if size == 1 else 1 if size == 2 else 2 if size <= 5 else 3
            cells[CATEGORIES[category]] += size
        expected += list(cells.values())
    assert table["cells"].tolist() == expected
    ranks = result.tables["rank_windows"]
    first = ranks[ranks["window"] == "1 to 10"]
    assert first["cells"].tolist() == [_in_sample()["d2"], _in_sample()["d4"]]
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
            assert table.loc[(cell_type, name), "cells"] == counted[(cell_type, name)]
    # The ranks are counted within the plotted groups.
    ranks = result.tables["rank_windows"].set_index(["group", "window"])
    for cell_type, found in TRUTH.group_clone_sizes(TRUTH.cell_types).items():
        assert ranks.loc[(cell_type, "1"), "cells"] == found[0]
        assert ranks.loc[(cell_type, "after 1"), "cells"] == sum(found[1:])
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
    assert table.columns.tolist() == ["clones", "fraction"]
    for cell_type in CELL_TYPES:
        n_clones = sum(per_type[(cell_type, name)] for name in names)
        for name in names:
            assert table.loc[(cell_type, name), "clones"] == per_type[(cell_type, name)]
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
    assert result.provenance.extras["grouping"] == "cluster_labels"
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
        ({"group_by": "sample", "groups": "x"}, ValueError, "not both"),
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
    if "unlabelled" not in store.cells.columns:
        insert_nullable_cell_column(
            store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
        )
    result = clonal_expansion(
        store, truth_plot.by_sample, group_by="unlabelled", show=False
    )
    assert result.tables["categories"].empty
    assert result.tables["rank_windows"].empty
    assert result.provenance.n_cells == 0
    plt.close(result.figure)


def test_the_rarefaction_plot_draws_one_panel_per_order(truth_plot) -> None:
    store = truth_plot.store
    result = store.plots.clonal_rarefaction(
        truth_plot.clones, group_by="sample", show=False
    )
    assert list(result.axes) == ["order_0", "order_1", "order_2", "coverage"]
    curves = result.tables["curves"]
    assert curves.columns[:2].tolist() == ["group", "sample"]
    assert sorted(curves["group"].unique()) == ["d2", "d4"]
    assert result.provenance.n_cells == 32
    assert result.provenance.extras["grouping"] == "sample"
    assert result.axes["order_1"].get_title() == "Shannon diversity (order 1)"
    assert result.axes["coverage"].get_ylabel() == "Sample coverage"
    legend = result.axes["coverage"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["d2", "d4"]
    # The first line of order 0 is d2's curve up to its observed size, which
    # ends at its clones.
    d2 = dv.CloneSizes.from_sizes(TRUTH.group_clone_sizes(TRUTH.samples)["d2"])
    line = result.axes["order_0"].get_lines()[0]
    assert line.get_label() == "d2"
    assert line.get_xdata()[-1] == d2.n
    assert line.get_ydata()[-1] == d2.richness
    plt.close(result.figure)
    by_coverage = clonal_rarefaction(
        store, truth_plot.clones, orders=(0,), base="coverage", show=False
    )
    assert list(by_coverage.axes) == ["order_0"]
    assert by_coverage.axes["order_0"].get_xlabel() == "Sample coverage"
    assert set(by_coverage.tables["curves"]["group"]) == {"all cells"}
    # One group needs no legend.
    assert by_coverage.axes["order_0"].get_legend() is None
    assert by_coverage.provenance.extras["grouping"] is None
    plt.close(by_coverage.figure)
    # Shown by default: on Agg the owned figure is closed.
    shown = clonal_rarefaction(store, truth_plot.clones, orders=(2,))
    assert shown.provenance.extras["orders"] == [2]
    assert shown.figure.number not in plt.get_fignums()


def test_the_coverage_panel_draws_the_coverage(truth_plot) -> None:
    result = clonal_rarefaction(
        truth_plot.store, truth_plot.clones, group_by="sample", show=False
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


def test_the_rarefaction_plot_of_complete_samples(truth_plot) -> None:
    store = truth_plot.store
    expanded = store.select_cells(
        truth_plot.clones, value="clone_size", low=2, keep_bounds=True
    )
    result = clonal_rarefaction(
        store, truth_plot.clones, cell_selection=expanded, base="coverage", show=False
    )
    curves = result.tables["curves"]
    # Without singletons, the observed sample is complete.
    assert curves.loc[curves["method"] == "observed", "coverage"].tolist() == [1.0] * 3
    plt.close(result.figure)


def test_the_rarefaction_plot_draws_on_caller_axes(truth_plot) -> None:
    figure, axes = plt.subplots(1, 2)
    result = clonal_rarefaction(
        truth_plot.store,
        truth_plot.clones,
        orders=(0,),
        target={"order_0": axes[0], "coverage": axes[1]},
        show_legend=False,
        show=False,
    )
    assert result.figure is figure and result.owns_figure is False
    assert axes[1].get_legend() is None
    plt.close(figure)


def test_the_rarefaction_plot_checks_its_arguments(truth_plot) -> None:
    with pytest.raises(ValueError, match="base must be 'size' or 'coverage'"):
        clonal_rarefaction(
            truth_plot.store, truth_plot.clones, base="depth", show=False
        )
    with pytest.raises(TypeError, match="store does not provide clones.rarefaction"):
        clonal_rarefaction(object(), truth_plot.clones, show=False)


def test_the_overlap_plot_draws_one_index_between_samples(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    result = store.plots.clonal_overlap(clones, group_by="sample", show=False)
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
    assert result.legends[0].label == "morisita_horn"
    assert result.provenance.n_cells == 32
    assert result.provenance.extras["index"] == "morisita_horn"
    assert result.provenance.extras["groups"] == ["d2", "d4"]
    assert result.provenance.extras["clones"] == clones.to_dict()
    plt.close(result.figure)
    shared = clonal_overlap(
        store,
        clones,
        groups=truth_plot.labels,
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
    shown = clonal_overlap(store, clones, group_by="sample", index="jaccard")
    assert shown.tables["matrix"].loc["d2", "d4"] == pytest.approx(7 / 8)
    assert shown.figure.number not in plt.get_fignums()


def test_the_overlap_plot_prints_the_values_of_at_most_12_groups(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    crossed = clonal_overlap(
        store, clones, group_by=["sample", "cell_type"], show=False
    )
    assert len(crossed.tables["matrix"]) == 6
    assert len(crossed.axes["matrix"].texts) == 30
    plt.close(crossed.figure)
    cell_ids = np.asarray([f"c{at:02d}" for at in range(N_CELLS)], dtype=object)
    store.cells.insert("cell", cell_ids, overwrite=True)
    each = clonal_overlap(store, clones, group_by="cell", index="shared", show=False)
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
    result = clonal_overlap(
        truth_plot.store,
        truth_plot.clones,
        group_by="sample",
        target={"matrix": ax},
        show=False,
    )
    assert result.figure is figure and result.owns_figure is False
    plt.close(figure)


def test_the_overlap_plot_checks_its_arguments(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    with pytest.raises(ValueError, match="index must be one of"):
        clonal_overlap(store, clones, group_by="sample", index="dice", show=False)
    with pytest.raises(ValueError, match="at least two groups"):
        clonal_overlap(store, clones, show=False)
    with pytest.raises(TypeError, match="store does not provide clones.overlap"):
        clonal_overlap(object(), clones, show=False)


def test_the_tracking_plot_draws_one_line_per_clone(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    result = store.plots.clone_tracking(clones, group_by="sample", top=2, show=False)
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
    assert result.tables["cells"]["cells"].tolist() == [3, 6, 2, 4, 3, 1]
    assert result.provenance.n_cells == 19
    assert result.provenance.extras["order"] == ["d2", "d4"]
    assert ax.get_ylabel() == "Fraction of the group's cells"
    plt.close(result.figure)
    ordered = clone_tracking(
        store,
        clones,
        group_by="sample",
        top=1,
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
    pooled = clone_tracking(store, clones, labels=["1"], show=False)
    ax = pooled.axes["tracking"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["all cells"]
    assert ax.get_lines()[0].get_ydata().tolist() == [9 / 32]
    plt.close(pooled.figure)


def test_the_tracking_plot_places_groups_by_all_their_labels(truth_plot) -> None:
    result = clone_tracking(
        truth_plot.store,
        truth_plot.clones,
        group_by="sample",
        groups=truth_plot.labels,
        labels=["8"],
        show=False,
    )
    ax = result.axes["tracking"]
    positions = [f"{s}, {t}" for s in SAMPLES for t in sorted(CELL_TYPES)]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == positions
    # Clone 8 holds one cell: an undifferentiated cell of d4, which holds
    # five cells with a clone.
    assert ax.get_lines()[0].get_ydata().tolist() == pytest.approx(
        [0, 0, 0, 0, 0, 1 / 5]
    )
    assert result.tables["cells"].columns.tolist() == [
        "clone",
        "group",
        "sample",
        "cluster_labels",
        "cells",
        "fraction",
    ]
    plt.close(result.figure)


class _TrackOnly:
    """A store whose clones namespace can follow clones and nothing else."""

    def __init__(self, store: DataStore) -> None:
        self.clones = SimpleNamespace(track=store.clones.track)


def test_the_tracking_plot_needs_only_the_tracked_clones(truth_plot) -> None:
    result = clone_tracking(
        _TrackOnly(truth_plot.store),
        truth_plot.clones,
        group_by="sample",
        top=1,
        show=False,
    )
    legend = result.axes["tracking"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["1"]
    plt.close(result.figure)


def test_the_tracking_plot_checks_its_arguments(truth_plot) -> None:
    store, clones = truth_plot.store, truth_plot.clones
    shown = clone_tracking(store, clones, group_by="sample", top=1)
    assert shown.provenance.n_cells == 9
    assert shown.figure.number not in plt.get_fignums()
    with pytest.raises(ValueError, match="order must list the groups"):
        clone_tracking(store, clones, group_by="sample", order=["d2"], show=False)
    with pytest.raises(TypeError, match="order must be a sequence"):
        clone_tracking(store, clones, group_by="sample", order="d2", show=False)
    # The order is checked before the clones are followed.
    with pytest.raises(TypeError, match="order must be a sequence"):
        clone_tracking(object(), clones, order=np.asarray([["d2"]]), show=False)
    ordered = clone_tracking(
        store, clones, group_by="sample", order=np.asarray(["d4", "d2"]), show=False
    )
    ax = ordered.axes["tracking"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d4", "d2"]
    plt.close(ordered.figure)
    with pytest.raises(ValueError, match="top must be a positive integer"):
        clone_tracking(store, clones, group_by="sample", top=0, show=False)
    with pytest.raises(TypeError, match="store does not provide clones.track"):
        clone_tracking(object(), clones, show=False)
