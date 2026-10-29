import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc import plotting
from cytearc.repertoire import ClonalExpansion
from tests.storage_helpers import insert_nullable_cell_column
from tests.test_repertoire_clonotypes_datastore import CLONAL_ROWS, _receptors
from tests.test_repertoire_import import N_CELLS, _store

# The default clonotypes of CLONAL_ROWS: clonotype_1 holds the T cells cell0,
# cell1 and cell4, clonotype_2 the B cells cell5 and cell6, and cell3 and
# cell2 are T cell singletons. cell7 has no receptor.
SAMPLES = np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])
CATEGORIES = (
    "1 cell",
    "2 cells",
    "3 to 5 cells",
    "6 to 20 cells",
    "21 to 100 cells",
    "more than 100 cells",
)


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _clonotypes(tmp_path, rows: list[str] = CLONAL_ROWS):
    store, receptors = _receptors(tmp_path, rows)
    return store, store.repertoire.clonotypes(receptors)


def test_expansion_counts_each_receptor_type_apart_by_default(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(clonotypes)

    assert ref.kind == "clonal_expansion" and ref.scope == "datastore"
    status = store.artifacts.inspect(ref)
    assert status.operation == "measure_clonal_expansion"
    assert status.parameters == {
        "within": [],
        "within_receptor_type": True,
        "breakpoints": [1, 2, 5, 20, 100],
        "frequency_breakpoints": None,
    }
    assert status.input_ref("clonotypes") == clonotypes
    assert status.inputs["within_fingerprint"] is None
    loaded = store.repertoire.load_clonal_expansion(ref)
    assert isinstance(loaded, ClonalExpansion)
    assert loaded.clonotypes == clonotypes
    assert loaded.labels == CATEGORIES
    assert loaded.breakpoints == (1, 2, 5, 20, 100)
    assert loaded.frequency_breakpoints is None
    assert loaded.splits == (10, 100, 1000, 10000, 30000, 100000)
    assert loaded.within == () and loaded.within_receptor_type is True
    cells = loaded.cells
    assert cells.index.tolist() == [f"cell{index}" for index in range(N_CELLS)]
    assert cells.columns.tolist() == [
        "clonotype",
        "category",
        "receptor_type",
        "group_clone_size",
        "group_clone_frequency",
    ]
    assert cells["category"].tolist()[:7] == [
        "3 to 5 cells",
        "3 to 5 cells",
        "1 cell",
        "1 cell",
        "3 to 5 cells",
        "2 cells",
        "2 cells",
    ]
    assert pd.isna(cells.loc["cell7", "category"])
    assert pd.isna(cells.loc["cell7", "receptor_type"])
    assert cells["group_clone_size"].tolist() == [3, 3, 1, 1, 3, 2, 2, 0]
    # The five T cells and the two B cells are counted apart.
    np.testing.assert_allclose(
        cells["group_clone_frequency"],
        [0.6, 0.6, 0.2, 0.2, 0.6, 1.0, 1.0, np.nan],
    )
    table = loaded.categories
    assert table.columns.tolist() == [
        "receptor_type",
        "category",
        "cells",
        "clonotypes",
        "fraction",
    ]
    assert table["receptor_type"].tolist() == ["BCR"] * 6 + ["TCR"] * 6
    assert table["category"].tolist() == list(CATEGORIES) * 2
    assert table["cells"].tolist() == [0, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0]
    assert table["clonotypes"].tolist() == [0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0]
    np.testing.assert_allclose(
        table["fraction"], [0, 1, 0, 0, 0, 0, 0.4, 0, 0.6, 0, 0, 0]
    )
    ranks = loaded.rank_windows
    assert ranks.columns.tolist() == ["receptor_type", "window", "cells", "fraction"]
    assert ranks.query("window == '1 to 10'")["cells"].tolist() == [2, 5]
    assert int(ranks["cells"].sum()) == 7


def test_expansion_counts_cells_within_samples(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    store.cells.insert("sample", SAMPLES)
    ref = store.repertoire.clonal_expansion(clonotypes, within="sample")
    status = store.artifacts.inspect(ref)
    assert status.parameters["within"] == ["sample"]
    assert status.inputs["within_fingerprint"] is not None
    loaded = store.repertoire.load_clonal_expansion(ref, splits=(1,))
    assert loaded.within == ("sample",)
    cells = loaded.cells
    # clonotype_1 holds cell0 and cell4 in s1, and cell1 alone in s2.
    assert cells["group_clone_size"].tolist() == [2, 1, 1, 1, 2, 1, 1, 0]
    np.testing.assert_allclose(
        cells["group_clone_frequency"],
        [2 / 3, 1 / 2, 1 / 3, 1 / 2, 2 / 3, 1.0, 1.0, np.nan],
    )
    assert cells.loc["cell0", "sample"] == "s1"
    assert pd.isna(cells.loc["cell7", "sample"])
    keys = loaded.categories[["sample", "receptor_type"]].drop_duplicates()
    assert keys.values.tolist() == [
        ["s1", "BCR"],
        ["s1", "TCR"],
        ["s2", "BCR"],
        ["s2", "TCR"],
    ]
    ranks = loaded.rank_windows.set_index(["sample", "receptor_type", "window"])
    assert loaded.splits == (1,)
    assert ranks.loc[("s1", "TCR", "1"), "cells"] == 2
    assert ranks.loc[("s1", "TCR", "after 1"), "cells"] == 1
    assert ranks.loc[("s2", "TCR", "after 1"), "fraction"] == 0.5


def test_expansion_can_count_every_cell_together(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(clonotypes, within_receptor_type=False)
    loaded = store.repertoire.load_clonal_expansion(ref)
    assert loaded.cells.columns.tolist() == [
        "clonotype",
        "category",
        "group_clone_size",
        "group_clone_frequency",
    ]
    assert loaded.cells.loc["cell0", "group_clone_frequency"] == 3 / 7
    assert loaded.categories.columns.tolist() == [
        "category",
        "cells",
        "clonotypes",
        "fraction",
    ]
    assert loaded.categories["cells"].tolist() == [2, 2, 3, 0, 0, 0]
    # A column named receptor_type may group the cells when the receptor type
    # does not.
    store.cells.insert("receptor_type", SAMPLES)
    named = store.repertoire.clonal_expansion(
        clonotypes, within="receptor_type", within_receptor_type=False
    )
    assert store.repertoire.load_clonal_expansion(named).within == ("receptor_type",)
    with pytest.raises(ValueError, match="within_receptor_type already adds"):
        store.repertoire.clonal_expansion(clonotypes, within="receptor_type")


def test_frequency_bounds_place_cells_by_their_clone_share(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(
        clonotypes, frequency_breakpoints=(0.25, 0.5, 1)
    )
    status = store.artifacts.inspect(ref)
    assert status.parameters["breakpoints"] is None
    assert status.parameters["frequency_breakpoints"] == [0.25, 0.5, 1.0]
    loaded = store.repertoire.load_clonal_expansion(ref)
    assert loaded.labels == ("0 to 0.25", "0.25 to 0.5", "0.5 to 1")
    assert loaded.breakpoints is None
    assert loaded.frequency_breakpoints == (0.25, 0.5, 1.0)
    assert loaded.cells["category"].tolist()[:7] == [
        "0.5 to 1",
        "0.5 to 1",
        "0 to 0.25",
        "0 to 0.25",
        "0.5 to 1",
        "0.5 to 1",
        "0.5 to 1",
    ]


def test_expansion_is_reused_and_selected_like_labels(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    first = store.repertoire.clonal_expansion(clonotypes)
    assert store.repertoire.clonal_expansion(clonotypes) == first
    assert (
        store.repertoire.clonal_expansion(clonotypes, breakpoints=(1, 2, 5, 20, 100))
        == first
    )
    assert store.repertoire.clonal_expansion(clonotypes, breakpoints=(1, 2)) != first
    assert store.repertoire.clonal_expansion(clonotypes, invalidate_cache=True) != first
    chosen = store.select_cells(first, include=["3 to 5 cells"])
    selected = store.artifacts.load_values(first, cell_selection=chosen)
    assert selected.cell_ids.tolist() == ["cell0", "cell1", "cell4"]
    expanded = store.select_cells(
        first, value="group_clone_size", low=2, keep_bounds=True
    )
    assert store.artifacts.load_values(
        first, value="group_clone_size", cell_selection=expanded
    ).cell_ids.tolist() == ["cell0", "cell1", "cell4", "cell5", "cell6"]
    # Other labels in the within column give another result.
    store.cells.insert("sample", SAMPLES)
    by_sample = store.repertoire.clonal_expansion(clonotypes, within="sample")
    store.cells.insert("sample", SAMPLES[::-1], overwrite=True)
    assert store.repertoire.clonal_expansion(clonotypes, within="sample") != by_sample


def test_within_columns_need_a_label_for_every_cell_with_a_clonotype(
    tmp_path,
) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[7] = True  # cell7 has no clonotype, so it needs no label
    insert_nullable_cell_column(store, "sample", SAMPLES, missing)
    assert store.repertoire.clonal_expansion(clonotypes, within=["sample"])
    missing[0] = True
    insert_nullable_cell_column(store, "batch", SAMPLES, missing)
    with pytest.raises(ValueError, match="needs a label in the within columns"):
        store.repertoire.clonal_expansion(clonotypes, within="batch")


def test_expansion_of_cells_without_clonotypes(tmp_path) -> None:
    rows = [row.replace("true,true,C", "true,false,C") for row in CLONAL_ROWS]
    store, clonotypes = _clonotypes(tmp_path, rows)
    ref = store.repertoire.clonal_expansion(clonotypes)
    loaded = store.repertoire.load_clonal_expansion(ref)
    assert loaded.cells["category"].isna().all()
    assert loaded.categories.empty and loaded.rank_windows.empty
    result = plotting.clonal_expansion(store, ref, show=False)
    try:
        assert result.tables["categories"].empty
        assert result.provenance.n_cells == 0
    finally:
        result.close()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"within_receptor_type": 1}, TypeError, "within_receptor_type must be"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be"),
        ({"within": 3}, TypeError, "within must be"),
        ({"within": ["absent"]}, KeyError, "absent"),
        (
            {"breakpoints": (1, 2), "frequency_breakpoints": (0.5, 1)},
            ValueError,
            "not both",
        ),
        ({"breakpoints": (2, 1)}, ValueError, "strictly increasing"),
        ({"breakpoints": (1.5,)}, TypeError, "whole numbers"),
        ({"frequency_breakpoints": (0.1, 0.5)}, ValueError, "must be 1"),
        ({"frequency_breakpoints": "0.5"}, TypeError, "sequence of numbers"),
    ],
)
def test_expansion_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    with pytest.raises(error, match=message):
        store.repertoire.clonal_expansion(clonotypes, **kwargs)


def test_expansion_refuses_other_artifacts(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    expansion = store.repertoire.clonal_expansion(clonotypes)
    with pytest.raises(ValueError, match="clonotypes artifact"):
        store.repertoire.clonal_expansion(receptors)
    with pytest.raises(TypeError, match="clonotypes must be an ArtifactRef"):
        store.repertoire.clonal_expansion("clonotypes")
    with pytest.raises(ValueError, match="clonal_expansion artifact"):
        store.repertoire.load_clonal_expansion(clonotypes)
    with pytest.raises(TypeError, match="expansion must be an ArtifactRef"):
        store.repertoire.load_clonal_expansion("expansion")
    with pytest.raises(ValueError, match="strictly increasing"):
        store.repertoire.load_clonal_expansion(expansion, splits=(10, 10))


def test_a_read_only_store_refuses_new_expansion(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    saved = store.repertoire.clonal_expansion(clonotypes)
    reader = _store(tmp_path, zarr_mode="r")
    assert reader.repertoire.clonal_expansion(clonotypes) == saved
    with pytest.raises(PermissionError, match="measure_clonal_expansion"):
        reader.repertoire.clonal_expansion(clonotypes, breakpoints=(1,))


def test_expansion_refuses_more_cells_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    saved = store.repertoire.clonal_expansion(clonotypes)
    # Eight cells take about 2,400 bytes to measure and 4,000 to load.
    monkeypatch.setattr(store, "memoryBytes", 3_000)
    with pytest.raises(MemoryError, match="Loading the clonal expansion of 8 cells"):
        store.repertoire.load_clonal_expansion(saved)
    with pytest.raises(MemoryError, match="Loading the clonotypes of 8 cells"):
        store.repertoire.load_clonotypes(clonotypes)
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.clonal_expansion(clonotypes, breakpoints=(1,))
    assert store.artifacts.list(kind="clonal_expansion", scope="datastore") == [saved]


def test_expansion_plot_draws_the_groups_of_the_expansion(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(clonotypes)
    result = store.plots.clonal_expansion(ref, show=False)
    try:
        table = result.tables["categories"]
        assert table.columns.tolist() == ["group", "category", "cells", "fraction"]
        assert table["group"].tolist() == ["BCR"] * 6 + ["TCR"] * 6
        assert table["cells"].tolist() == [0, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0]
        ranks = result.tables["rank_windows"]
        assert ranks.query("window == '1 to 10'")["cells"].tolist() == [2, 5]
        assert ranks.query("window == '1 to 10'")["fraction"].tolist() == [1.0, 1.0]
        assert result.provenance.n_cells == 7
        assert result.provenance.extras["grouping"] == "receptor_type"
        assert result.provenance.extras["splits"] == [
            10,
            100,
            1000,
            10000,
            30000,
            100000,
        ]
        ax = result.axes["categories"]
        assert ax.get_xlabel() == "Fraction of cells"
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["BCR", "TCR"]
        assert ax.get_legend().get_title().get_text() == "Clone size"
        # Only the windows that hold cells are drawn.
        legend = result.axes["rank_windows"].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["1 to 10"]
    finally:
        result.close()


def test_expansion_plot_groups_cells_by_another_column(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(
        clonotypes, frequency_breakpoints=(0.25, 0.5, 1)
    )
    store.cells.insert("sample", SAMPLES)
    result = plotting.clonal_expansion(
        store, ref, group_by="sample", normalize=False, splits=(1,), show=False
    )
    try:
        table = result.tables["categories"].set_index(["group", "category"])
        # The categories stay those of the expansion's groups, the receptor
        # types: s1 holds cell0 and cell4 of clonotype_1, cell2 and cell5.
        assert table.loc[("s1", "0.5 to 1"), "cells"] == 3
        assert table.loc[("s1", "0 to 0.25"), "cells"] == 1
        assert table.loc[("s2", "0.5 to 1"), "cells"] == 2
        ranks = result.tables["rank_windows"].set_index(["group", "window"])
        # The ranks are counted within the plotted groups.
        assert ranks.loc[("s1", "1"), "cells"] == 2
        assert ranks.loc[("s1", "after 1"), "cells"] == 2
        assert result.axes["categories"].get_xlabel() == "Cells"
        assert (
            result.axes["categories"].get_legend().get_title().get_text()
            == "Clone frequency"
        )
        assert result.provenance.extras["grouping"] == "sample"
    finally:
        result.close()
    clonotype_counts = plotting.clonal_expansion(
        store, ref, group_by="sample", count="clonotypes", show=False
    )
    try:
        table = clonotype_counts.tables["categories"].set_index(["group", "category"])
        assert table.columns.tolist() == ["clonotypes", "fraction"]
        assert table.loc[("s1", "0.5 to 1"), "clonotypes"] == 2
        assert table.loc[("s1", "0.5 to 1"), "fraction"] == 2 / 3
        assert clonotype_counts.axes["categories"].get_xlabel() == (
            "Fraction of clonotypes"
        )
    finally:
        clonotype_counts.close()


def test_expansion_plot_groups_cells_by_labels(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    ref = store.repertoire.clonal_expansion(clonotypes, within_receptor_type=False)
    store.cells.insert("cluster", np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"]))
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection()
    )
    result = plotting.clonal_expansion(store, ref, groups=labels, show=False)
    try:
        assert result.tables["categories"]["group"].unique().tolist() == ["x", "y"]
        assert result.provenance.extras["grouping"] == "cluster_labels"
    finally:
        result.close()
    ungrouped = plotting.clonal_expansion(store, ref, show=False, show_legend=False)
    try:
        assert ungrouped.tables["categories"]["group"].unique().tolist() == [
            "all cells"
        ]
        assert ungrouped.provenance.extras["grouping"] is None
        assert ungrouped.axes["categories"].get_legend() is None
    finally:
        ungrouped.close()
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clonal_expansion(store, ref)
    assert shown.provenance.n_cells == 7


def test_expansion_plot_draws_on_caller_axes(tmp_path) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(clonotypes)
    figure, axes = plt.subplots(1, 2)
    try:
        result = plotting.clonal_expansion(
            store,
            ref,
            target={"categories": axes[0], "rank_windows": axes[1]},
            show=False,
        )
        assert result.figure is figure and result.owns_figure is False
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"count": "contigs"}, ValueError, "count must be"),
        ({"normalize": "yes"}, TypeError, "normalize must be a boolean"),
        ({"splits": (5, 1)}, ValueError, "strictly increasing"),
        ({"group_by": "sample", "groups": "x"}, ValueError, "not both"),
    ],
)
def test_expansion_plot_checks_its_arguments(tmp_path, kwargs, error, message) -> None:
    store, clonotypes = _clonotypes(tmp_path)
    ref = store.repertoire.clonal_expansion(clonotypes)
    with pytest.raises(error, match=message):
        plotting.clonal_expansion(store, ref, show=False, **kwargs)
    with pytest.raises(TypeError, match="does not provide clonal expansion"):
        plotting.clonal_expansion(object(), ref, show=False)
