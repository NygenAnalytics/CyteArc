import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc import plotting
from cytearc.repertoire import ClonalDiversity, Rarefaction
from cytearc.repertoire import diversity as dv
from tests.storage_helpers import insert_nullable_cell_column
from tests.test_repertoire_clonotypes_datastore import CLONAL_ROWS, _receptors
from tests.test_repertoire_import import N_CELLS

# The default clonotypes of CLONAL_ROWS: clonotype_1 holds the T cells cell0,
# cell1 and cell4, clonotype_2 the B cells cell5 and cell6, and cell2 and
# cell3 are T cell singletons. cell7 has no receptor.
SAMPLES = np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])
TCR = dv.CloneSizes.from_sizes([3, 1, 1])
BCR = dv.CloneSizes.from_sizes([2])


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
    return store, receptors, store.repertoire.clonotypes(receptors)


def test_diversity_measures_each_receptor_type_apart(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    saved = store.artifacts.list(scope="datastore")
    result = store.repertoire.diversity(clonotypes)
    assert isinstance(result, ClonalDiversity)
    assert result.clonotypes == clonotypes and result.group_by == ()
    assert result.groups is None and result.cell_selection is None
    assert result.within_receptor_type is True
    table = result.indices
    assert table.columns.tolist() == ["receptor_type", *dv.INDEX_COLUMNS]
    assert table["receptor_type"].tolist() == ["BCR", "TCR"]
    for name, clones in (("BCR", BCR), ("TCR", TCR)):
        row = table.set_index("receptor_type").loc[name]
        for column, value in dv.indices(clones).items():
            assert row[column] == pytest.approx(value, nan_ok=True)
    # The smallest doubled size, of the two B cells, and the lowest coverage
    # at doubled sizes, of the T cells.
    assert result.size == 4
    assert result.coverage == pytest.approx(dv.coverage_at(TCR, 10))
    hill = result.hill.set_index(["receptor_type", "order"])
    assert result.hill.columns.tolist() == [
        "receptor_type",
        "order",
        "observed",
        "estimated",
        "at_size",
        "coverage_at_size",
        "method_at_size",
        "at_coverage",
        "size_at_coverage",
        "method_at_coverage",
    ]
    for order in dv.ORDERS:
        row = hill.loc[("TCR", order)]
        assert row["observed"] == pytest.approx(dv.observed_hill(TCR, order))
        assert row["estimated"] == pytest.approx(dv.estimated_hill(TCR, order))
        assert row["at_size"] == pytest.approx(dv.diversity_at(TCR, 4, order)[0])
        assert row["method_at_size"] == "rarefaction"
        assert row["size_at_coverage"] == pytest.approx(10)
        assert row["method_at_coverage"] == "extrapolation"
        assert hill.loc[("BCR", order), "method_at_size"] == "extrapolation"
    # Nothing is saved.
    assert store.artifacts.list(scope="datastore") == saved


def test_diversity_groups_by_columns_labels_and_selections(tmp_path) -> None:
    store, receptors, clonotypes = _clonotypes(tmp_path)
    store.cells.insert("sample", SAMPLES)
    together = store.repertoire.diversity(
        clonotypes, group_by="sample", within_receptor_type=False, size=3, coverage=0.5
    )
    assert together.group_by == ("sample",) and together.size == 3
    assert together.coverage == 0.5
    table = together.indices.set_index("sample")
    # s1 holds two cells of clonotype_1, cell2 and the B cell cell5; s2 one
    # cell of each of three clonotypes.
    assert table.loc["s1", ["cells", "clonotypes", "singletons"]].tolist() == [4, 3, 2]
    assert table.loc["s2", ["cells", "clonotypes", "singletons"]].tolist() == [3, 3, 3]
    store.cells.insert("cluster", np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"]))
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection()
    )
    crossed = store.repertoire.diversity(clonotypes, group_by=["sample"], groups=labels)
    assert crossed.indices.columns[:3].tolist() == [
        "sample",
        "cluster_labels",
        "receptor_type",
    ]
    assert crossed.indices["cells"].sum() == 7
    t_cells = store.select_cells(receptors, include=["TCR"])
    selected = store.repertoire.diversity(clonotypes, cell_selection=t_cells)
    assert selected.cell_selection == t_cells
    assert selected.indices["receptor_type"].tolist() == ["TCR"]
    assert selected.size == 10


def test_diversity_at_full_coverage_is_the_observed_diversity(tmp_path) -> None:
    store, _receptors, clonotypes = _clonotypes(tmp_path)
    # The cells of clonotypes of two or more cells: no group has a singleton,
    # so every coverage, and the default common coverage, is 1.
    expanded = store.select_cells(
        clonotypes, value="clone_size", low=2, keep_bounds=True
    )
    result = store.repertoire.diversity(clonotypes, cell_selection=expanded)
    assert result.coverage == 1.0
    hill = result.hill
    np.testing.assert_allclose(hill["at_coverage"], hill["observed"])
    assert (hill["method_at_coverage"] == "observed").all()
    assert hill.groupby("receptor_type")["size_at_coverage"].first().to_dict() == {
        "BCR": 2.0,
        "TCR": 3.0,
    }
    plot = store.plots.clonal_rarefaction(
        clonotypes, cell_selection=expanded, base="coverage", show=False
    )
    plot.close()


def test_group_columns_may_not_share_a_name_with_the_result(tmp_path) -> None:
    store, receptors, clonotypes = _clonotypes(tmp_path)
    store.cells.insert("coverage", SAMPLES)
    with pytest.raises(ValueError, match="'coverage', which is also a column"):
        store.repertoire.diversity(clonotypes, group_by="coverage")
    with pytest.raises(ValueError, match="'coverage', which is also a column"):
        store.repertoire.rarefaction(clonotypes, group_by="coverage")
    # A name that only other results hold is free.
    store.cells.insert("locus", SAMPLES)
    assert len(store.repertoire.diversity(clonotypes, group_by="locus").indices) == 4
    with pytest.raises(ValueError, match="'locus', which is also a column"):
        store.repertoire.gene_usage(receptors, group_by="locus")


def test_cells_without_a_label_are_left_out(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[[0, 1]] = True
    insert_nullable_cell_column(store, "sample", SAMPLES, missing)
    result = store.repertoire.diversity(
        clonotypes, group_by="sample", within_receptor_type=False
    )
    assert result.indices["cells"].sum() == 5
    insert_nullable_cell_column(store, "batch", SAMPLES, np.ones(N_CELLS, dtype=bool))
    with pytest.raises(ValueError, match="No cell has both a clonotype and a group"):
        store.repertoire.diversity(clonotypes, group_by="batch")
    with pytest.raises(ValueError, match="No cell has both a clonotype and a group"):
        store.repertoire.rarefaction(clonotypes, group_by="batch")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"within_receptor_type": 1}, TypeError, "within_receptor_type must be"),
        ({"group_by": 3}, TypeError, "group_by must be"),
        (
            {"group_by": ["sample", "sample"]},
            ValueError,
            "group_by names a column twice",
        ),
        ({"groups": "clusters"}, TypeError, "groups must be an ArtifactRef"),
        ({"cell_selection": "t"}, TypeError, "cell_selection must be an ArtifactRef"),
        ({"group_by": "receptor_type"}, ValueError, "name one column twice"),
        ({"size": 0}, ValueError, "size must be a positive integer"),
        ({"size": 2.5}, TypeError, "size must be a positive integer"),
        ({"size": True}, TypeError, "size must be a positive integer"),
        ({"coverage": 1}, ValueError, "coverage must be between 0 and 1"),
        ({"coverage": "half"}, TypeError, "coverage must be a number"),
    ],
)
def test_diversity_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    store.cells.insert("sample", SAMPLES)
    store.cells.insert("receptor_type", SAMPLES)
    with pytest.raises(error, match=message):
        store.repertoire.diversity(clonotypes, **kwargs)


def test_diversity_refuses_other_artifacts_and_large_stores(
    tmp_path, monkeypatch
) -> None:
    store, receptors, clonotypes = _clonotypes(tmp_path)
    with pytest.raises(ValueError, match="clonotypes artifact"):
        store.repertoire.diversity(receptors)
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.diversity(clonotypes)


def test_rarefaction_follows_each_groups_default_sizes(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    result = store.repertoire.rarefaction(clonotypes)
    assert isinstance(result, Rarefaction) and result.orders == (0, 1, 2)
    curves = result.curves
    assert curves.columns.tolist() == [
        "receptor_type",
        "order",
        "size",
        "method",
        "diversity",
        "coverage",
    ]
    tcr = curves[(curves["receptor_type"] == "TCR") & (curves["order"] == 1)]
    assert tcr["size"].tolist() == dv.default_sizes(5).tolist()
    expected, methods = dv.hill_curve(TCR, tcr["size"].tolist(), 1)
    np.testing.assert_allclose(tcr["diversity"], expected)
    assert tcr["method"].tolist() == methods
    np.testing.assert_allclose(
        tcr["coverage"], [dv.coverage_at(TCR, m) for m in tcr["size"]]
    )
    fixed = store.repertoire.rarefaction(
        clonotypes, orders=[2], sizes=[4, 2, 4], within_receptor_type=False
    )
    assert fixed.orders == (2,)
    assert fixed.curves.columns.tolist()[:2] == ["order", "size"]
    assert fixed.curves["size"].tolist() == [2, 4]
    short = store.repertoire.rarefaction(clonotypes, endpoint=4, knots=4, orders=(0,))
    assert set(short.curves["size"]) <= {1, 2, 3, 4}


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"orders": 1}, TypeError, "orders must be a sequence"),
        ({"orders": "01"}, TypeError, "orders must be a sequence"),
        ({"orders": []}, ValueError, "distinct orders"),
        ({"orders": [0, 0]}, ValueError, "distinct orders"),
        ({"orders": [3]}, ValueError, "distinct orders"),
        ({"orders": [True]}, ValueError, "distinct orders"),
        ({"sizes": 5}, TypeError, "sizes must be a sequence"),
        ({"sizes": []}, ValueError, "sizes must hold positive integers"),
        ({"sizes": [0]}, ValueError, "sizes must hold positive integers"),
        ({"sizes": [2.5]}, ValueError, "sizes must hold positive integers"),
        ({"endpoint": 0}, ValueError, "endpoint must be"),
        ({"knots": 1}, ValueError, "knots must be"),
    ],
)
def test_rarefaction_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    with pytest.raises(error, match=message):
        store.repertoire.rarefaction(clonotypes, **kwargs)


def test_rarefaction_plot_draws_one_panel_per_order(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    result = store.plots.clonal_rarefaction(clonotypes, show=False)
    try:
        assert list(result.axes) == ["order_0", "order_1", "order_2", "coverage"]
        curves = result.tables["curves"]
        assert curves.columns[:2].tolist() == ["group", "receptor_type"]
        assert sorted(curves["group"].unique()) == ["BCR", "TCR"]
        assert result.provenance.n_cells == 7
        assert result.provenance.extras["grouping"] == "receptor_type"
        assert result.axes["order_1"].get_title() == "Shannon diversity (order 1)"
        assert result.axes["coverage"].get_ylabel() == "Sample coverage"
        legend = result.axes["coverage"].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["BCR", "TCR"]
    finally:
        result.close()
    by_coverage = plotting.clonal_rarefaction(
        store,
        clonotypes,
        within_receptor_type=False,
        orders=(0,),
        base="coverage",
        show=False,
    )
    try:
        assert list(by_coverage.axes) == ["order_0"]
        assert by_coverage.axes["order_0"].get_xlabel() == "Sample coverage"
        assert set(by_coverage.tables["curves"]["group"]) == {"all cells"}
        # One group needs no legend.
        assert by_coverage.axes["order_0"].get_legend() is None
        assert by_coverage.provenance.extras["grouping"] is None
    finally:
        by_coverage.close()
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clonal_rarefaction(store, clonotypes, orders=(2,))
    assert shown.provenance.extras["orders"] == [2]


def test_rarefaction_plot_draws_on_caller_axes(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    figure, axes = plt.subplots(1, 2)
    try:
        result = plotting.clonal_rarefaction(
            store,
            clonotypes,
            orders=(0,),
            target={"order_0": axes[0], "coverage": axes[1]},
            show_legend=False,
            show=False,
        )
        assert result.figure is figure and result.owns_figure is False
        assert axes[1].get_legend() is None
    finally:
        plt.close(figure)


def test_rarefaction_plot_checks_its_arguments(tmp_path) -> None:
    store, _receptors_ref, clonotypes = _clonotypes(tmp_path)
    with pytest.raises(ValueError, match="base must be"):
        plotting.clonal_rarefaction(store, clonotypes, base="depth", show=False)
    with pytest.raises(TypeError, match="does not provide rarefaction"):
        plotting.clonal_rarefaction(object(), clonotypes, show=False)
    frame = pd.DataFrame()
    assert frame.empty
