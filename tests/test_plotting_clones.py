"""The clone abundance plot, on clones whose sizes are counted by hand."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.plotting import clone_abundance
from cytearc.plotting._groups import cell_groups
from tests.storage_helpers import write_count_store

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
