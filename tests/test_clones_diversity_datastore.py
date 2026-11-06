"""Diversity and rarefaction of the truth's imported clones, by group of cells.

The store and its clones are written once, as a template, and shared by the
tests that only measure, since diversity and rarefaction save nothing; tests
that change cell metadata or the memory budget copy the template.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cytearc.clones import ClonalDiversity, Rarefaction
from cytearc.clones import diversity as dv
from tests.clone_helpers import (
    every_cell,
    import_truth_clones,
    write_truth_store,
)
from tests.clone_truth import CLONE_SIZES, N_CELLS, clone_truth
from cytearc.utils.memory_budget import MemoryBudgetError
from tests.storage_helpers import copy_store, insert_nullable_cell_column

TRUTH = clone_truth()
ALL = dv.CloneSizes.from_sizes(CLONE_SIZES)
HILL_COLUMNS = [
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


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("clone_diversity")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    clones = import_truth_clones(store, TRUTH, folder)
    store.cells.insert("is_d2", TRUTH.samples == "d2")
    return SimpleNamespace(path=folder / "store.zarr", clones=clones)


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    store = copy_store(template.path, tmp_path_factory.mktemp("shared"))
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


@pytest.fixture
def copied(template, tmp_path) -> SimpleNamespace:
    store = copy_store(template.path, tmp_path)
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


def test_diversity_measures_every_cell_together_by_default(shared) -> None:
    store, clones = shared.store, shared.clones
    saved = store.artifacts.list(scope="datastore")
    result = store.clones.diversity(clones)
    assert isinstance(result, ClonalDiversity)
    assert result.clones == clones and result.group_by == ()
    assert result.groups is None and result.cell_selection is None
    table = result.indices
    assert table.columns.tolist() == ["group", *dv.INDEX_COLUMNS]
    assert table["group"].tolist() == ["all cells"]
    # Clones of 9, 6, 5, 4, 3, 2, 2 and 1 cells.
    assert table[
        ["n_cells", "n_clones", "n_singletons", "n_doubletons", "d50"]
    ].values.tolist() == [[32, 8, 1, 2, 3]]
    for column, value in dv.indices(ALL).items():
        assert table[column].iloc[0] == pytest.approx(value, nan_ok=True)
    # The common size doubles the 32 cells, and the common coverage is the
    # coverage there.
    assert result.size == 64
    assert result.coverage == pytest.approx(dv.coverage_at(ALL, 64))
    hill = result.hill
    assert hill.columns.tolist() == ["group", *HILL_COLUMNS]
    for order in dv.ORDERS:
        row = hill.set_index("order").loc[order]
        assert row["observed"] == pytest.approx(dv.observed_hill(ALL, order))
        assert row["estimated"] == pytest.approx(dv.estimated_hill(ALL, order))
        assert row["at_size"] == pytest.approx(dv.diversity_at(ALL, 64, order)[0])
        assert row["method_at_size"] == "extrapolation"
        assert row["size_at_coverage"] == pytest.approx(64)
        assert row["at_coverage"] == pytest.approx(row["at_size"])
    # Order 0 observes the eight clones.
    assert hill.set_index("order").loc[0, "observed"] == 8
    # Nothing is saved.
    assert store.artifacts.list(scope="datastore") == saved


def test_diversity_groups_by_columns_labels_and_selections(shared) -> None:
    store, clones = shared.store, shared.clones
    by_sample = store.clones.diversity(clones, group_by="sample", size=3, coverage=0.5)
    assert by_sample.group_by == ("sample",) and by_sample.size == 3
    assert by_sample.coverage == 0.5
    assert by_sample.indices.columns.tolist() == ["group", *dv.INDEX_COLUMNS]
    assert by_sample.group_keys.to_dict("index") == {
        "d2": {"sample": "d2"},
        "d4": {"sample": "d4"},
    }
    table = by_sample.indices.join(by_sample.group_keys, on="group").set_index("sample")
    for sample, found in TRUTH.group_clone_sizes(TRUTH.samples).items():
        assert table.loc[sample, ["n_cells", "n_clones", "n_singletons"]].tolist() == [
            sum(found),
            len(found),
            found.count(1),
        ]
        shares = np.asarray(found) / sum(found)
        assert table.loc[sample, "shannon"] == pytest.approx(
            -np.sum(shares * np.log(shares))
        )
    labels = store.snapshot_cluster_labels("cell_type", cell_selection=shared.cells)
    crossed = store.clones.diversity(clones, group_by=["sample"], groups=labels)
    assert crossed.groups == labels
    assert crossed.indices.columns.tolist() == ["group", *dv.INDEX_COLUMNS]
    assert crossed.group_keys.columns.tolist() == ["sample", "cluster_labels"]
    assert crossed.indices["n_cells"].sum() == 32
    # Each group is named by its labels joined by commas, in their order.
    names = crossed.group_keys[["sample", "cluster_labels"]].agg(", ".join, axis=1)
    assert crossed.group_keys.index.tolist() == names.tolist()
    assert crossed.indices["group"].tolist() == names.tolist()
    assert crossed.hill["group"].tolist() == [
        name for name in names.tolist() for _ in dv.ORDERS
    ]
    d2 = store.snapshot_cell_selection("is_d2")
    selected = store.clones.diversity(clones, cell_selection=d2)
    assert selected.cell_selection == d2
    d2_sizes = TRUTH.group_clone_sizes(cells=TRUTH.samples == "d2")[0]
    assert selected.indices["n_cells"].tolist() == [sum(d2_sizes)]
    assert selected.indices["n_clones"].tolist() == [len(d2_sizes)]


def test_diversity_at_full_coverage_is_the_observed_diversity(shared) -> None:
    store, clones = shared.store, shared.clones
    # The cells of clones of two or more cells: without singletons every
    # coverage, and the default common coverage, is 1.
    expanded = store.select_cells(clones, value="clone_size", low=2, keep_bounds=True)
    result = store.clones.diversity(clones, cell_selection=expanded)
    assert result.coverage == 1.0
    hill = result.hill
    np.testing.assert_allclose(hill["at_coverage"], hill["observed"])
    assert (hill["method_at_coverage"] == "observed").all()
    assert hill["size_at_coverage"].tolist() == [31.0] * 3


def test_group_columns_may_take_the_names_of_the_results_columns(copied) -> None:
    store, clones = copied.store, copied.clones
    neutral = store.clones.diversity(clones, group_by="sample")
    neutral_curves = store.clones.rarefaction(clones, group_by="sample").curves
    # Columns named like the results' own columns hold the samples' labels and
    # measure what the samples do; their labels stay in the keys.
    for name in ("group", "coverage", "n_clones", "shannon", "order", "diversity"):
        store.cells.insert(name, TRUTH.samples.astype(object), overwrite=True)
        found = store.clones.diversity(clones, group_by=name)
        pd.testing.assert_frame_equal(found.indices, neutral.indices)
        pd.testing.assert_frame_equal(found.hill, neutral.hill)
        curves = store.clones.rarefaction(clones, group_by=name)
        pd.testing.assert_frame_equal(curves.curves, neutral_curves)
        for keys in (found.group_keys, curves.group_keys):
            assert keys.columns.tolist() == [name]
            assert keys.to_dict("index") == {
                "d2": {name: "d2"},
                "d4": {name: "d4"},
            }
        joined = found.indices.join(found.group_keys, on="group", rsuffix="_key")
        label = f"{name}_key" if name in found.indices.columns else name
        assert joined[label].tolist() == ["d2", "d4"]
    # A clones artifact as labels has the key column "clones".
    labelled = store.clones.diversity(clones, groups=clones)
    assert labelled.group_keys.columns.tolist() == ["clones"]


def test_cells_without_a_label_are_left_out(copied) -> None:
    store, clones = copied.store, copied.clones
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[np.flatnonzero(TRUTH.clones >= 0)[:2]] = True
    insert_nullable_cell_column(store, "partial", TRUTH.samples, missing)
    result = store.clones.diversity(clones, group_by="partial")
    assert result.indices["n_cells"].sum() == 30
    insert_nullable_cell_column(
        store, "batch", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    with pytest.raises(ValueError, match="No cell has both a clone and a group"):
        store.clones.diversity(clones, group_by="batch")
    with pytest.raises(ValueError, match="No cell has both a clone and a group"):
        store.clones.rarefaction(clones, group_by="batch")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"group_by": 3}, TypeError, "group_by must be"),
        ({"group_by": ["sample", "sample"]}, ValueError, "names a column twice"),
        ({"group_by": "absent"}, ValueError, "'absent', which is not a cell"),
        ({"groups": "clusters"}, TypeError, "groups must be an ArtifactRef"),
        ({"cell_selection": "t"}, TypeError, "cell_selection must be an ArtifactRef"),
        ({"size": 0}, ValueError, "size must be a positive integer"),
        ({"size": 2.5}, TypeError, "size must be a positive integer"),
        ({"size": True}, TypeError, "size must be a positive integer"),
        ({"coverage": 1}, ValueError, "coverage must be between 0 and 1"),
        ({"coverage": "half"}, TypeError, "coverage must be a number"),
    ],
)
def test_diversity_arguments_are_checked(shared, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        shared.store.clones.diversity(shared.clones, **kwargs)


def test_diversity_refuses_other_artifacts_and_large_stores(
    copied, monkeypatch
) -> None:
    store, clones = copied.store, copied.clones
    with pytest.raises(
        ValueError, match="clones must be a clones or clonotypes artifact, not"
    ):
        store.clones.diversity(copied.cells)
    with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
        store.clones.rarefaction("clones")
    # 100 bytes cannot hold each of the 48 cells' clone, 8 bytes each.
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(
        MemoryBudgetError, match="for each cell's clone, .* Raise mem_budget"
    ):
        store.clones.diversity(clones)


def test_a_label_of_its_own_for_each_cell_makes_groups_of_one_cell(copied) -> None:
    store, clones = copied.store, copied.clones
    # A label of its own for each of the 32 cells with a clone makes 32
    # groups.
    has = TRUTH.clones >= 0
    store.cells.insert("own", np.asarray([f"c{i:02d}" for i in range(N_CELLS)]))
    store.cells.insert("has_clone", has)
    own = store.snapshot_cluster_labels(
        "own", cell_selection=store.snapshot_cell_selection("has_clone")
    )
    result = store.clones.diversity(clones, groups=own)
    assert result.indices["n_cells"].tolist() == [1] * 32


def test_diversity_counts_the_cells_of_each_groups_clones(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    # Two groups: clones of 1, 2, 2 and 5 cells, and two of 3 cells.
    keys = pd.DataFrame(index=pd.Index(["a", "b"], name="group"))
    counts = (
        keys,
        np.array([0, 0, 0, 0, 1, 1]),
        np.arange(6),
        np.array([1, 2, 2, 5, 3, 3]),
        {"cellsWithoutClone": 0, "cellsWithoutGroup": 0, "groups": 2},
    )
    monkeypatch.setattr(store, "_clone_pair_counts", lambda *args, **kwargs: counts)
    indices = store.clones.diversity(clones).indices
    assert indices[["group", "n_cells"]].values.tolist() == [["a", 10], ["b", 6]]


def test_rarefaction_of_many_sizes_refuses_a_small_budget(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    # Two samples at 10,000 sizes: 20,000 rows, which 2 MB cannot hold.
    sizes = np.arange(1, 10_001)
    monkeypatch.setattr(store, "memoryBytes", 2_000_000)
    with pytest.raises(
        MemoryBudgetError, match="or pass fewer sizes or knots"
    ) as refused:
        store.clones.rarefaction(clones, group_by="sample", orders=(0,), sizes=sizes)
    assert refused.value.item == "rarefying 2 groups to 10000 sizes"
    monkeypatch.undo()
    curves = store.clones.rarefaction(
        clones, group_by="sample", orders=(0,), sizes=sizes
    ).curves
    assert len(curves) == 20_000


def test_rarefaction_reaches_each_size_asked_and_refuses_too_many_knots(
    copied,
) -> None:
    store, clones = copied.store, copied.clones
    sizes = np.arange(1, 1_001)
    curves = store.clones.rarefaction(clones, orders=(0,), sizes=sizes).curves
    assert curves["size"].tolist() == sizes.tolist()
    # Knots are counted before any size is made, however many are asked for.
    with pytest.raises(
        MemoryBudgetError, match="for rarefying 1 groups to 1000000000000 sizes"
    ):
        store.clones.rarefaction(clones, knots=10**12)


def test_orders_sizes_endpoint_and_knots_may_be_numpy_values(shared) -> None:
    store, clones = shared.store, shared.clones
    listed = store.clones.rarefaction(clones, orders=[0, 2], sizes=[4, 2, 4])
    arrays = store.clones.rarefaction(
        clones, orders=np.array([0, 2]), sizes=np.array([4, 2, 4])
    )
    assert arrays.orders == (0, 2) and all(
        type(order) is int for order in arrays.orders
    )
    assert arrays.curves.equals(listed.curves)
    numbers = store.clones.rarefaction(
        clones, orders=(0,), endpoint=np.int64(4), knots=np.int64(4)
    )
    assert numbers.curves.equals(
        store.clones.rarefaction(clones, orders=(0,), endpoint=4, knots=4).curves
    )
    by_size = store.clones.diversity(clones, size=np.int64(10))
    assert by_size.size == 10 and type(by_size.size) is int


def test_rarefaction_follows_each_groups_default_sizes(shared) -> None:
    store, clones = shared.store, shared.clones
    result = store.clones.rarefaction(clones, group_by="sample")
    assert isinstance(result, Rarefaction) and result.orders == (0, 1, 2)
    assert result.group_by == ("sample",) and result.clones == clones
    assert result.curves.columns.tolist() == [
        "group",
        "order",
        "size",
        "method",
        "diversity",
        "coverage",
    ]
    assert result.group_keys.columns.tolist() == ["sample"]
    curves = result.curves.join(result.group_keys, on="group")
    d4 = dv.CloneSizes.from_sizes(TRUTH.group_clone_sizes(TRUTH.samples)["d4"])
    rows = curves[(curves["sample"] == "d4") & (curves["order"] == 1)]
    assert rows["size"].tolist() == dv.default_sizes(d4.n).tolist()
    expected, methods = dv.hill_curve(d4, rows["size"].tolist(), 1)
    np.testing.assert_allclose(rows["diversity"], expected)
    assert rows["method"].tolist() == methods
    np.testing.assert_allclose(
        rows["coverage"], [dv.coverage_at(d4, m) for m in rows["size"]]
    )
    # At its observed size, each curve of order 0 counts the sample's clones.
    observed = curves[(curves["method"] == "observed") & (curves["order"] == 0)]
    assert observed.set_index("sample")["diversity"].to_dict() == {
        sample: len(found)
        for sample, found in TRUTH.group_clone_sizes(TRUTH.samples).items()
    }
    fixed = store.clones.rarefaction(clones, orders=[2], sizes=[4, 2, 4])
    assert fixed.orders == (2,)
    assert fixed.curves.columns.tolist()[:3] == ["group", "order", "size"]
    assert set(fixed.curves["group"]) == {"all cells"}
    assert fixed.curves["size"].tolist() == [2, 4]
    short = store.clones.rarefaction(clones, endpoint=4, knots=4, orders=(0,))
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
        ({"orders": np.ones((1, 1), dtype=int)}, TypeError, "orders must be a"),
        ({"sizes": np.ones((1, 1), dtype=int)}, TypeError, "sizes must be a"),
        ({"sizes": [2**53 + 1]}, ValueError, "up to 9007199254740992"),
        ({"endpoint": 2**53 + 1}, ValueError, "up to 9007199254740992"),
    ],
)
def test_rarefaction_arguments_are_checked(shared, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        shared.store.clones.rarefaction(shared.clones, **kwargs)
