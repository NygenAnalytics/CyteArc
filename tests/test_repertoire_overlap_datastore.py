import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc import plotting
from cytearc.repertoire import OVERLAP_INDICES, ClonotypeOverlap, ClonotypeTracking
from tests.test_repertoire_clonotypes_datastore import CLONAL_ROWS, _receptors

# The default clonotypes of CLONAL_ROWS: clonotype_1 holds the T cells cell0,
# cell1 and cell4, clonotype_2 the B cells cell5 and cell6, clonotype_3 cell3
# and clonotype_4 cell2. cell7 has no receptor.
SAMPLES = np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _clonotypes(tmp_path):
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    store.cells.insert("sample", SAMPLES)
    return store, receptors, store.repertoire.clonotypes(receptors)


def test_overlap_compares_groups_of_one_receptor_type(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    saved = store.artifacts.list(scope="datastore")
    result = store.repertoire.overlap(clonotypes, group_by="sample")
    assert isinstance(result, ClonotypeOverlap)
    assert result.group_by == ("sample",) and result.groups is None
    assert result.within_receptor_type is True and result.cell_selection is None
    sizes = result.group_sizes
    assert sizes.columns.tolist() == [
        "group",
        "sample",
        "receptor_type",
        "cells",
        "clonotypes",
    ]
    assert sizes["group"].tolist() == ["s1, BCR", "s1, TCR", "s2, BCR", "s2, TCR"]
    assert sizes["cells"].tolist() == [1, 3, 1, 2]
    pairs = result.pairs
    assert pairs.columns.tolist() == [
        "group_1",
        "group_2",
        *OVERLAP_INDICES,
        "shared_fraction_1",
        "shared_fraction_2",
    ]
    # B cells are compared with B cells and T cells with T cells: s1 holds
    # clonotype_1 twice and clonotype_4, s2 clonotype_1 and clonotype_3.
    assert pairs[["group_1", "group_2"]].values.tolist() == [
        ["s1, BCR", "s2, BCR"],
        ["s1, TCR", "s2, TCR"],
    ]
    tcr = pairs.iloc[1]
    assert tcr["shared"] == 1 and tcr["jaccard"] == pytest.approx(1 / 3)
    assert tcr["shared_fraction_1"] == pytest.approx(2 / 3)
    assert tcr["shared_fraction_2"] == pytest.approx(1 / 2)
    assert tcr["morisita_horn"] == pytest.approx(2 * 2 / ((5 / 9 + 2 / 4) * 3 * 2))
    assert pairs.iloc[0]["morisita_horn"] == pytest.approx(1.0)
    assert store.artifacts.list(scope="datastore") == saved


def test_overlap_without_receptor_types_and_with_selections(tmp_path) -> None:
    store, receptors, clonotypes = _clonotypes(tmp_path)
    pooled = store.repertoire.overlap(
        clonotypes, group_by="sample", within_receptor_type=False
    )
    assert pooled.pairs[["group_1", "group_2", "shared"]].values.tolist() == [
        ["s1", "s2", 2]
    ]
    t_cells = store.select_cells(receptors, include=["TCR"])
    selected = store.repertoire.overlap(
        clonotypes, group_by="sample", cell_selection=t_cells
    )
    assert selected.cell_selection == t_cells
    assert selected.group_sizes["group"].tolist() == ["s1, TCR", "s2, TCR"]
    store.cells.insert("cluster", np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"]))
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection()
    )
    by_cluster = store.repertoire.overlap(
        clonotypes, groups=labels, within_receptor_type=False
    )
    assert by_cluster.groups == labels
    assert by_cluster.group_sizes["cluster_labels"].tolist() == ["x", "y"]


def test_overlap_needs_two_groups(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    with pytest.raises(ValueError, match="at least two groups"):
        store.repertoire.overlap(clonotypes, within_receptor_type=False)
    with pytest.raises(TypeError, match="group_by must be"):
        store.repertoire.overlap(clonotypes, group_by=3)


def test_track_follows_the_largest_clonotypes_of_each_group(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    tracked = store.repertoire.track(clonotypes, group_by="sample", top=1)
    assert isinstance(tracked, ClonotypeTracking)
    # Each group's largest: clonotype_2 in both B cell groups, clonotype_1 in
    # s1's T cells, and in s2's T cells clonotype_1 over clonotype_3 by rank.
    assert tracked.tracked == ("clonotype_1", "clonotype_2")
    cells = tracked.cells
    assert cells.columns.tolist() == [
        "clonotype",
        "group",
        "sample",
        "receptor_type",
        "cells",
        "fraction",
    ]
    # Each clonotype appears in the groups of its receptor type only.
    assert cells[["clonotype", "group", "cells"]].values.tolist() == [
        ["clonotype_1", "s1, TCR", 2],
        ["clonotype_1", "s2, TCR", 1],
        ["clonotype_2", "s1, BCR", 1],
        ["clonotype_2", "s2, BCR", 1],
    ]
    assert cells["fraction"].tolist() == pytest.approx([2 / 3, 1 / 2, 1, 1])
    named = store.repertoire.track(
        clonotypes,
        group_by="sample",
        labels=["clonotype_3"],
        within_receptor_type=False,
    )
    assert named.tracked == ("clonotype_3",)
    assert named.cells[["group", "cells"]].values.tolist() == [["s1", 0], ["s2", 1]]


def test_groups_whose_labels_join_to_one_name_are_refused(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    # ("a, b", "c") and ("a", "b, c") would both be named "a, b, c".
    store.cells.insert("first", np.asarray(["a, b", "a"] * 4))
    store.cells.insert("second", np.asarray(["c", "b, c"] * 4))
    with pytest.raises(ValueError, match="are both named 'a, b, c"):
        store.repertoire.overlap(
            clonotypes, group_by=["first", "second"], within_receptor_type=False
        )


def test_track_follows_labelled_clonotypes_without_cells(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    # Only clonotype_1's cells are selected; clonotype_3, a T cell singleton,
    # is followed through the T cell groups with no cells.
    selected = store.select_cells(clonotypes, include=["clonotype_1"])
    tracked = store.repertoire.track(
        clonotypes,
        group_by="sample",
        labels=["clonotype_1", "clonotype_3"],
        cell_selection=selected,
    )
    assert tracked.tracked == ("clonotype_1", "clonotype_3")
    rows = tracked.cells[tracked.cells["clonotype"] == "clonotype_3"]
    assert rows[["group", "cells", "fraction"]].values.tolist() == [
        ["s1, TCR", 0, 0.0],
        ["s2, TCR", 0, 0.0],
    ]
    pooled = store.repertoire.track(
        clonotypes,
        group_by="sample",
        labels=["clonotype_2"],
        cell_selection=selected,
        within_receptor_type=False,
    )
    assert pooled.cells[["group", "cells"]].values.tolist() == [["s1", 0], ["s2", 0]]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"top": 0}, ValueError, "top must be a positive integer"),
        ({"top": 1.5}, TypeError, "top must be a positive integer"),
        ({"top": True}, TypeError, "top must be a positive integer"),
        ({"labels": "clonotype_1"}, TypeError, "labels must be a sequence"),
        ({"labels": [1]}, TypeError, "labels must be a sequence"),
        ({"labels": ["clonotype_9"]}, ValueError, "No clonotype is labelled"),
    ],
)
def test_track_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    with pytest.raises(error, match=message):
        store.repertoire.track(clonotypes, group_by="sample", **kwargs)


def test_track_needs_cells_with_a_group(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    store.cells.insert("empty", np.asarray([""] * 8))
    with pytest.raises(ValueError, match="No cell has both a clonotype and a group"):
        store.repertoire.track(clonotypes, group_by="empty")


def test_overlap_plot_draws_the_pairs_of_groups(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    result = store.plots.clonal_overlap(clonotypes, group_by="sample", show=False)
    try:
        matrix = result.tables["matrix"]
        assert matrix.index.tolist() == ["s1, BCR", "s1, TCR", "s2, BCR", "s2, TCR"]
        assert matrix.loc["s1, TCR", "s2, TCR"] == pytest.approx(
            result.tables["pairs"].iloc[1]["morisita_horn"]
        )
        # Pairs of different receptor types stay blank; the diagonal is 1.
        assert np.isnan(matrix.loc["s1, BCR", "s2, TCR"])
        assert matrix.loc["s1, TCR", "s1, TCR"] == 1
        assert result.axes["matrix"].get_title() == "Morisita-Horn index"
        assert len(result.axes["matrix"].texts) == 4
        assert result.provenance.n_cells == 7
    finally:
        result.close()
    shared = plotting.clonal_overlap(
        store,
        clonotypes,
        group_by="sample",
        within_receptor_type=False,
        index="shared",
        show_legend=False,
        show=False,
    )
    try:
        assert shared.tables["matrix"].values.tolist() == [[3, 2], [2, 3]]
        assert [text.get_text() for text in shared.axes["matrix"].texts] == ["2", "2"]
    finally:
        shared.close()
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clonal_overlap(
        store, clonotypes, group_by="sample", index="jaccard"
    )
    assert shown.provenance.extras["index"] == "jaccard"
    with pytest.raises(ValueError, match="index must be one of"):
        plotting.clonal_overlap(store, clonotypes, index="dice", show=False)
    with pytest.raises(TypeError, match="does not provide overlap"):
        plotting.clonal_overlap(object(), clonotypes, show=False)


def test_overlap_plot_leaves_out_values_of_many_groups(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    store.cells.insert("cell", np.asarray([f"c{index:02d}" for index in range(8)]))
    result = plotting.clonal_overlap(
        store, clonotypes, group_by="cell", within_receptor_type=False, show=False
    )
    try:
        assert len(result.tables["matrix"]) == 7
        assert len(result.axes["matrix"].texts) == 42
    finally:
        result.close()


def test_tracking_plot_draws_one_line_per_clonotype(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    result = store.plots.clonotype_tracking(
        clonotypes, group_by="sample", within_receptor_type=False, top=2, show=False
    )
    try:
        ax = result.axes["tracking"]
        # Ties of one cell go to the larger clonotype overall, so both samples'
        # two largest are clonotype_1 and clonotype_2.
        assert len(ax.get_lines()) == len(result.provenance.extras["tracked"]) == 2
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["clonotype_1 (CASSA)", "clonotype_2 (CARW)"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s1", "s2"]
        assert result.tables["cells"]["cells"].tolist() == [2, 1, 1, 1]
    finally:
        result.close()
    ordered = plotting.clonotype_tracking(
        store,
        clonotypes,
        group_by="sample",
        within_receptor_type=False,
        order=["s2", "s1"],
        show_legend=False,
        show=False,
    )
    try:
        ax = ordered.axes["tracking"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s2", "s1"]
        assert ax.get_legend() is None
    finally:
        ordered.close()
    # With receptor types, the T and B cell groups of a sample share its
    # position, which order names by the sample alone.
    by_type = plotting.clonotype_tracking(
        store, clonotypes, group_by="sample", order=["s2", "s1"], show=False
    )
    try:
        ax = by_type.axes["tracking"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s2", "s1"]
        assert all(len(line.get_xdata()) == 2 for line in ax.get_lines())
        assert set(by_type.tables["cells"]["group"]) == {
            "s1, BCR",
            "s1, TCR",
            "s2, BCR",
            "s2, TCR",
        }
    finally:
        by_type.close()
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clonotype_tracking(store, clonotypes, group_by="sample", top=1)
    assert shown.provenance.n_cells == 5
    with pytest.raises(ValueError, match="order must list the groups"):
        plotting.clonotype_tracking(
            store, clonotypes, group_by="sample", order=["s1"], show=False
        )
    with pytest.raises(TypeError, match="order must be a sequence"):
        plotting.clonotype_tracking(
            store, clonotypes, group_by="sample", order="s1", show=False
        )
    with pytest.raises(TypeError, match="does not provide track"):
        plotting.clonotype_tracking(object(), clonotypes, show=False)
