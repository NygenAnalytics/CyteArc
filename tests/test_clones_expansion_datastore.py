"""Clonal expansion of the truth's imported clones, against counts taken cell by cell.

The module writes a store of the truth's cells and imports the truth's clones
into it once, as a template. Tests that only measure share one copy, since
each measurement saves an artifact of its own; tests that change cell
metadata, reuse, read-only stores or the memory budget copy the template.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cytearc.clones import ClonalExpansion
from cytearc.clones.expansion import DEFAULT_RANK_SPLITS
from cytearc.utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
)
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    every_cell,
    import_truth_clones,
    write_truth_store,
)
from tests.clone_truth import CLONE_SIZES, N_CELLS, clone_truth
from tests.storage_helpers import copy_store, insert_nullable_cell_column, open_store

TRUTH = clone_truth()
N_IN_CLONES = sum(CLONE_SIZES)
CATEGORIES = (
    "1 cell",
    "2 cells",
    "3 to 5 cells",
    "6 to 20 cells",
    "21 to 100 cells",
    "more than 100 cells",
)
BREAKPOINTS = (1, 2, 5, 20, 100)


def _category(size: int, bounds=BREAKPOINTS) -> int:
    """The first bound that a clone size does not exceed, by hand."""
    return next(
        (index for index, bound in enumerate(bounds) if size <= bound), len(bounds)
    )


def _table_by_hand(sizes: dict[object, list[int]]) -> dict[object, dict[str, list]]:
    """Each group's cells and clones per default category, from its clone sizes."""
    tables = {}
    for group, found in sizes.items():
        cells = [0] * len(CATEGORIES)
        clones = [0] * len(CATEGORIES)
        for size in found:
            cells[_category(size)] += size
            clones[_category(size)] += 1
        tables[group] = {"n_cells": cells, "n_clones": clones}
    return tables


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("clone_expansion")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    clones = import_truth_clones(store, TRUTH, folder)
    return SimpleNamespace(path=folder / "store.zarr", clones=clones)


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    """One writable copy for measurements that save artifacts of their own."""
    store = copy_store(template.path, tmp_path_factory.mktemp("shared"))
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


@pytest.fixture
def copied(template, tmp_path) -> SimpleNamespace:
    store = copy_store(template.path, tmp_path)
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


def test_expansion_counts_every_cell_together_by_default(shared) -> None:
    store, clones = shared.store, shared.clones
    ref = store.clones.expansion(clones)

    assert ref.kind == "clonal_expansion" and ref.scope == "datastore"
    status = store.artifacts.inspect(ref)
    assert status.operation == "measure_clonal_expansion"
    assert status.parameters == {
        "group_by": [],
        "breakpoints": list(BREAKPOINTS),
        "frequency_breakpoints": None,
    }
    assert status.input_ref("clones") == clones
    assert status.input_ref("cell_selection") == shared.cells
    assert status.inputs["groups"] is None
    assert status.inputs["group_assignments_fingerprint"] is None
    loaded = store.clones.load_expansion(ref)
    assert isinstance(loaded, ClonalExpansion)
    assert loaded.clones == clones and loaded.cell_selection == shared.cells
    assert loaded.labels == CATEGORIES
    assert loaded.breakpoints == BREAKPOINTS
    assert loaded.frequency_breakpoints is None
    assert loaded.splits == DEFAULT_RANK_SPLITS
    assert loaded.group_by == () and loaded.groups is None
    cells = loaded.cells
    assert cells.index.tolist() == TRUTH.cell_ids.tolist()
    assert cells.columns.tolist() == [
        "clone",
        "category",
        "group",
        "group_clone_size",
        "group_clone_frequency",
    ]
    has = TRUTH.clones >= 0
    # CoSpar's table numbers the truth's clones from 1, largest first.
    assert cells["clone"].tolist() == [
        str(clone + 1) if clone >= 0 else np.nan for clone in TRUTH.clones.tolist()
    ]
    expected = TRUTH.cell_clone_sizes()
    np.testing.assert_array_equal(cells["group_clone_size"], expected)
    np.testing.assert_allclose(
        cells["group_clone_frequency"], np.where(has, expected / N_IN_CLONES, np.nan)
    )
    assert cells["category"].tolist() == [
        CATEGORIES[_category(size)] if size else np.nan for size in expected.tolist()
    ]
    assert set(cells["group"].dropna()) == {"all cells"}
    assert cells["group"].isna().tolist() == (~has).tolist()
    # Clones of 9 and 6 cells fill "6 to 20 cells", of 5, 4 and 3 "3 to 5
    # cells", the two of 2 "2 cells", and the one of 1 "1 cell".
    table = loaded.categories
    assert table.columns.tolist() == [
        "group",
        "category",
        "n_cells",
        "n_clones",
        "fraction",
    ]
    assert table["category"].tolist() == list(CATEGORIES)
    assert table["n_cells"].tolist() == [1, 4, 12, 15, 0, 0]
    assert table["n_clones"].tolist() == [1, 2, 3, 2, 0, 0]
    np.testing.assert_allclose(table["fraction"], np.array([1, 4, 12, 15, 0, 0]) / 32)
    ranks = loaded.rank_windows
    assert ranks.columns.tolist() == ["group", "window", "n_cells", "fraction"]
    # Eight clones, all among the ten largest.
    assert ranks["n_cells"].tolist() == [N_IN_CLONES, 0, 0, 0, 0, 0, 0]


def test_expansion_counts_cells_within_samples(shared) -> None:
    store = shared.store
    ref = store.clones.expansion(shared.clones, group_by="sample")
    status = store.artifacts.inspect(ref)
    assert status.parameters["group_by"] == ["sample"]
    assert status.inputs["group_assignments_fingerprint"] is not None

    loaded = store.clones.load_expansion(ref, splits=(1,))
    assert loaded.group_by == ("sample",) and loaded.splits == (1,)
    cells = loaded.cells
    assert cells.columns.tolist() == [
        "clone",
        "category",
        "group",
        "group_clone_size",
        "group_clone_frequency",
    ]
    # The groups' labels stay in a table of their own, indexed by the name
    # that the tables hold.
    keys = loaded.group_keys
    assert keys.index.name == "group"
    assert keys.to_dict("index") == {"d2": {"sample": "d2"}, "d4": {"sample": "d4"}}
    expected = TRUTH.cell_clone_sizes(TRUTH.samples)
    np.testing.assert_array_equal(cells["group_clone_size"], expected)
    in_sample = {
        sample: int(np.count_nonzero((TRUTH.samples == sample) & (TRUTH.clones >= 0)))
        for sample in ("d2", "d4")
    }
    np.testing.assert_allclose(
        cells["group_clone_frequency"],
        [
            size / in_sample[sample] if size else np.nan
            for size, sample in zip(expected.tolist(), TRUTH.samples, strict=True)
        ],
    )
    labelled = cells.join(keys, on="group")
    assert [
        None if pd.isna(sample) else sample for sample in labelled["sample"].tolist()
    ] == [
        sample if clone >= 0 else None
        for sample, clone in zip(TRUTH.samples, TRUTH.clones, strict=True)
    ]
    by_hand = _table_by_hand(TRUTH.group_clone_sizes(TRUTH.samples))
    table = loaded.categories
    assert table.columns.tolist() == [
        "group",
        "category",
        "n_cells",
        "n_clones",
        "fraction",
    ]
    table = table.join(keys, on="group")
    for sample in ("d2", "d4"):
        rows = table[table["sample"] == sample]
        assert rows["group"].tolist() == [sample] * len(CATEGORIES)
        assert rows["n_cells"].tolist() == by_hand[sample]["n_cells"]
        assert rows["n_clones"].tolist() == by_hand[sample]["n_clones"]
    # The largest clone of each sample holds the window "1"; the rest follow.
    assert loaded.rank_windows.columns.tolist() == [
        "group",
        "window",
        "n_cells",
        "fraction",
    ]
    ranks = loaded.rank_windows.join(keys, on="group").set_index(["sample", "window"])
    for sample, found in TRUTH.group_clone_sizes(TRUTH.samples).items():
        assert ranks.loc[(sample, "1"), "n_cells"] == found[0]
        assert ranks.loc[(sample, "after 1"), "n_cells"] == sum(found[1:])
        assert ranks.loc[(sample, "1"), "fraction"] == found[0] / sum(found)


def test_expansion_crosses_columns_with_a_label_artifact(shared) -> None:
    store = shared.store
    labels = store.snapshot_cluster_labels("cell_type", cell_selection=shared.cells)
    ref = store.clones.expansion(shared.clones, group_by="sample", groups=labels)
    loaded = store.clones.load_expansion(ref)
    assert loaded.groups == labels
    assert loaded.categories.columns[:2].tolist() == ["group", "category"]
    # The key columns: the group_by column, then the label artifact's kind.
    assert loaded.group_keys.columns.tolist() == ["sample", "cluster_labels"]
    assert loaded.group_keys.index.tolist() == [
        f"{sample}, {cell_type}" for sample, cell_type in loaded.group_keys.values
    ]
    keys = np.asarray(
        [
            f"{sample}, {cell_type}"
            for sample, cell_type in zip(TRUTH.samples, TRUTH.cell_types, strict=True)
        ],
        dtype=object,
    )
    np.testing.assert_array_equal(
        loaded.cells["group_clone_size"], TRUTH.cell_clone_sizes(keys)
    )
    totals = loaded.categories.groupby("group")["n_cells"].sum().to_dict()
    assert totals == {
        group: sum(found) for group, found in TRUTH.group_clone_sizes(keys).items()
    }
    # Groups are named by their labels joined by commas, in natural order.
    assert loaded.categories["group"].drop_duplicates().tolist() == sorted(totals)


def test_expansion_of_a_cell_selection_labels_its_cells(shared) -> None:
    store = shared.store
    d2 = TRUTH.samples == "d2"
    store.cells.insert("is_d2", d2, overwrite=True)
    selection = store.snapshot_cell_selection("is_d2")
    ref = store.clones.expansion(shared.clones, cell_selection=selection)
    assert store.artifacts.inspect(ref).input_ref("cell_selection") == selection
    loaded = store.clones.load_expansion(ref)
    assert loaded.cell_selection == selection
    assert loaded.cells.index.tolist() == TRUTH.cell_ids[d2].tolist()
    # The d2 cells count among themselves.
    np.testing.assert_array_equal(
        loaded.cells["group_clone_size"], TRUTH.cell_clone_sizes(cells=d2)[d2]
    )
    assert store.artifacts.load_values(ref).cell_ids.tolist() == (
        TRUTH.cell_ids[d2].tolist()
    )
    # The clones' own cells give the result of no selection.
    assert store.clones.expansion(shared.clones, cell_selection=shared.cells) == (
        store.clones.expansion(shared.clones)
    )


def test_selected_cells_that_the_clones_do_not_hold_have_no_clone(
    shared, tmp_path
) -> None:
    store = shared.store
    d4 = TRUTH.samples == "d4"
    store.cells.insert("is_d4", d4, overwrite=True)
    d4_clones = import_truth_clones(
        store, TRUTH, tmp_path, cell_selection=store.snapshot_cell_selection("is_d4")
    )
    ref = store.clones.expansion(d4_clones, cell_selection=shared.cells)
    loaded = store.clones.load_expansion(ref)
    assert loaded.cells.index.tolist() == TRUTH.cell_ids.tolist()
    np.testing.assert_array_equal(
        loaded.cells["group_clone_size"], TRUTH.cell_clone_sizes(cells=d4)
    )
    assert loaded.cells.loc[~d4, "clone"].isna().all()
    assert loaded.cells.loc[~d4, "category"].isna().all()
    values = store.artifacts.load_values(ref).values
    assert (np.asarray(values)[~d4] == "").all()


def test_frequency_bounds_place_cells_by_their_clone_share(shared) -> None:
    store = shared.store
    ref = store.clones.expansion(shared.clones, frequency_breakpoints=(0.1, 0.2, 1))
    status = store.artifacts.inspect(ref)
    assert status.parameters["breakpoints"] is None
    assert status.parameters["frequency_breakpoints"] == [0.1, 0.2, 1.0]
    loaded = store.clones.load_expansion(ref)
    assert loaded.labels == ("0 to 0.1", "0.1 to 0.2", "0.2 to 1")
    assert loaded.breakpoints is None
    assert loaded.frequency_breakpoints == (0.1, 0.2, 1.0)
    # Of 32 cells, clones of 3, 2, 2 and 1 cells hold at most a tenth each,
    # clones of 6, 5 and 4 at most a fifth, and the clone of 9 more.
    assert loaded.categories["n_cells"].tolist() == [8, 15, 9]
    assert loaded.categories["n_clones"].tolist() == [4, 3, 1]


def test_expansion_is_reused_and_selected_like_labels(copied) -> None:
    store, clones = copied.store, copied.clones
    first = store.clones.expansion(clones)
    assert store.clones.expansion(clones) == first
    assert store.clones.expansion(clones, breakpoints=BREAKPOINTS) == first
    assert store.clones.expansion(clones, breakpoints=(1, 2)) != first
    assert store.clones.expansion(clones, invalidate_cache=True) != first
    # The cells of the clones of 9 and 6 cells.
    chosen = store.select_cells(first, include=["6 to 20 cells"])
    selected = store.artifacts.load_values(first, cell_selection=chosen)
    assert set(selected.cell_ids.tolist()) == set(
        TRUTH.cell_ids[np.isin(TRUTH.clones, [0, 1])].tolist()
    )
    expanded = store.select_cells(
        first, value="group_clone_size", low=5, keep_bounds=True
    )
    assert set(
        store.artifacts.load_values(first, cell_selection=expanded).cell_ids
    ) == set(TRUTH.cell_ids[np.isin(TRUTH.clones, [0, 1, 2])].tolist())
    # Other labels in a group_by column give another result, also when they
    # make the same groups, and the same labels again reuse it.
    by_sample = store.clones.expansion(clones, group_by="sample")
    renamed = np.where(TRUTH.samples == "d2", "x2", "x4").astype(object)
    store.cells.insert("sample", renamed, overwrite=True)
    assert store.clones.expansion(clones, group_by="sample") != by_sample
    store.cells.insert("sample", TRUTH.samples.astype(object), overwrite=True)
    assert store.clones.expansion(clones, group_by="sample") == by_sample
    store.cells.insert("sample", TRUTH.samples[::-1].astype(object), overwrite=True)
    reversed_samples = store.clones.expansion(clones, group_by="sample")
    assert reversed_samples != by_sample
    # Renamed labels that keep every cell's group give another result too, and
    # the same labels again reuse it.
    renamed = np.where(TRUTH.samples[::-1] == "d2", "day2", "day4").astype(object)
    store.cells.insert("sample", renamed, overwrite=True)
    by_renamed = store.clones.expansion(clones, group_by="sample")
    assert by_renamed not in (by_sample, reversed_samples)
    store.cells.insert("sample", renamed.copy(), overwrite=True)
    assert store.clones.expansion(clones, group_by="sample") == by_renamed


def test_cells_without_a_label_are_counted_in_no_group(copied) -> None:
    store = copied.store
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[np.flatnonzero(TRUTH.clones >= 0)[:5]] = True
    insert_nullable_cell_column(store, "partial", TRUTH.samples, missing)
    loaded = store.clones.load_expansion(
        store.clones.expansion(copied.clones, group_by="partial")
    )
    np.testing.assert_array_equal(
        loaded.cells["group_clone_size"],
        TRUTH.cell_clone_sizes(TRUTH.samples, cells=~missing),
    )
    assert loaded.cells.loc[missing, "category"].isna().all()
    assert loaded.cells.loc[missing, "clone"].notna().all()
    assert int(loaded.categories["n_cells"].sum()) == N_IN_CLONES - 5


def test_expansion_refuses_groups_that_labels_with_commas_name_alike(copied) -> None:
    store = copied.store
    # "x, y" with "z" and "x" with "y, z" both read "x, y, z".
    store.cells.insert("first", np.asarray(["x, y"] * 24 + ["x"] * 24))
    store.cells.insert("second", np.asarray(["z"] * 24 + ["y, z"] * 24))
    with pytest.raises(ValueError, match="remove the commas"):
        store.clones.expansion(copied.clones, group_by=["first", "second"])


def test_group_columns_may_take_the_names_of_the_results_columns(copied) -> None:
    store = copied.store
    neutral = store.clones.load_expansion(
        store.clones.expansion(copied.clones, group_by="sample")
    )
    # Columns named like the expansion's own columns hold the samples' labels,
    # and measure what the samples do; their labels stay in the keys.
    for name in (
        "group",
        "clone",
        "category",
        "n_cells",
        "n_clones",
        "fraction",
        "window",
        "group_clone_size",
    ):
        store.cells.insert(name, TRUTH.samples.astype(object), overwrite=True)
        loaded = store.clones.load_expansion(
            store.clones.expansion(copied.clones, group_by=name)
        )
        for table in ("cells", "categories", "rank_windows"):
            found, expected = getattr(loaded, table), getattr(neutral, table)
            assert found.columns.tolist() == expected.columns.tolist(), (name, table)
            pd.testing.assert_frame_equal(found, expected)
        assert loaded.group_keys.columns.tolist() == [name]
        assert loaded.group_keys[name].tolist() == ["d2", "d4"]
        joined = loaded.categories.join(loaded.group_keys, on="group", rsuffix="_key")
        label = name if name not in loaded.categories.columns else f"{name}_key"
        assert joined[label].tolist() == joined["group"].tolist()
    # A clones artifact as labels has the key column "clones".
    labelled = store.clones.load_expansion(
        store.clones.expansion(copied.clones, groups=copied.clones)
    )
    assert labelled.group_keys.columns.tolist() == ["clones"]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be"),
        ({"group_by": 3}, TypeError, "group_by must be"),
        ({"group_by": ["absent"]}, ValueError, "'absent', which is not a cell"),
        ({"groups": "cluster"}, TypeError, "groups must be an ArtifactRef"),
        ({"cell_selection": "d2"}, TypeError, "cell_selection must be an ArtifactRef"),
        (
            {"breakpoints": (1, 2), "frequency_breakpoints": (0.5, 1)},
            ValueError,
            "not both",
        ),
        ({"breakpoints": (2, 1)}, ValueError, "strictly increasing"),
        ({"breakpoints": (1.5,)}, TypeError, "whole numbers"),
        ({"frequency_breakpoints": (0.1, 0.5)}, ValueError, "must be 1"),
        ({"frequency_breakpoints": "0.5"}, TypeError, "sequence of numbers"),
        ({"breakpoints": np.ones((2, 2), dtype=int)}, TypeError, "sequence of"),
        ({"breakpoints": (1, 10**30)}, ValueError, "at most 9223372036854775807"),
        ({"frequency_breakpoints": (10**400, 1)}, ValueError, "positive and finite"),
    ],
)
def test_expansion_arguments_are_checked(shared, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        shared.store.clones.expansion(shared.clones, **kwargs)


def test_bounds_are_checked_before_the_clones_are_read(shared) -> None:
    unsaved = ArtifactRef("datastore", "clones", "0" * 64)
    with pytest.raises(ValueError, match="strictly increasing"):
        shared.store.clones.expansion(unsaved, breakpoints=(2, 1))
    with pytest.raises(TypeError, match="invalidate_cache must be"):
        shared.store.clones.expansion(unsaved, invalidate_cache=1)


def test_bounds_and_splits_may_be_arrays(shared) -> None:
    store, clones = shared.store, shared.clones
    listed = store.clones.expansion(clones, breakpoints=(1, 2, 5))
    assert store.clones.expansion(clones, breakpoints=np.array([1, 2, 5])) == listed
    loaded = store.clones.load_expansion(listed, splits=np.array([1, 2]))
    assert loaded.splits == (1, 2)
    assert loaded.rank_windows["window"].drop_duplicates().tolist() == [
        "1",
        "2",
        "after 2",
    ]


def test_frequency_bounds_that_print_alike_get_names_of_their_own(shared) -> None:
    store, clones = shared.store, shared.clones
    bounds = (0.1234561, 0.1234562, 0.1234563, 1.0)
    loaded = store.clones.load_expansion(
        store.clones.expansion(clones, frequency_breakpoints=bounds)
    )
    # Six digits print the three bounds alike, so all four get seven.
    assert loaded.labels == (
        "0 to 0.1234561",
        "0.1234561 to 0.1234562",
        "0.1234562 to 0.1234563",
        "0.1234563 to 1",
    )
    # Every clone of the truth holds more than an eighth of the cells.
    assert loaded.categories.groupby("category", sort=False)[
        "n_cells"
    ].sum().to_dict() == {
        "0 to 0.1234561": 1 + 2 + 2 + 3,
        "0.1234561 to 0.1234562": 0,
        "0.1234562 to 0.1234563": 0,
        "0.1234563 to 1": 4 + 5 + 6 + 9,
    }


def test_the_group_keys_order_the_groups_naturally(copied) -> None:
    store = copied.store
    sites = np.where(TRUTH.samples == "d2", "s2", "s10").astype(object)
    store.cells.insert("site", sites)
    loaded = store.clones.load_expansion(
        store.clones.expansion(copied.clones, group_by="site")
    )
    assert loaded.group_keys.index.tolist() == ["s2", "s10"]
    assert loaded.group_keys["site"].tolist() == ["s2", "s10"]
    assert loaded.cells["group"].cat.categories.tolist() == ["s2", "s10"]
    assert loaded.categories["group"].drop_duplicates().tolist() == ["s2", "s10"]
    labelled = loaded.cells.join(loaded.group_keys, on="group")
    assert [None if pd.isna(site) else site for site in labelled["site"].tolist()] == [
        site if clone >= 0 else None
        for site, clone in zip(sites, TRUTH.clones, strict=True)
    ]


def test_expansion_refuses_other_artifacts(shared) -> None:
    store, clones = shared.store, shared.clones
    expansion = store.clones.expansion(clones)
    with pytest.raises(
        ValueError, match="clones must be a clones or clonotypes artifact"
    ):
        store.clones.expansion(shared.cells)
    with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
        store.clones.expansion("clones")
    with pytest.raises(ValueError, match="cell_selection must be a cell_selection"):
        store.clones.expansion(clones, cell_selection=clones)
    with pytest.raises(ValueError, match="clonal_expansion artifact"):
        store.clones.load_expansion(clones)
    with pytest.raises(TypeError, match="expansion must be an ArtifactRef"):
        store.clones.load_expansion("expansion")
    with pytest.raises(ValueError, match="strictly increasing"):
        store.clones.load_expansion(expansion, splits=(10, 10))


def test_a_read_only_store_reuses_saved_expansion(copied) -> None:
    store, clones = copied.store, copied.clones
    saved = store.clones.expansion(clones)
    reader = open_store(Path(store.zw.store.root), zarr_mode="r")
    assert reader.clones.expansion(clones) == saved
    assert reader.clones.load_expansion(saved).categories["n_cells"].sum() == 32
    with pytest.raises(PermissionError, match="measure_clonal_expansion"):
        reader.clones.expansion(clones, breakpoints=(1,))


def test_expansion_refuses_a_small_budget_before_it_writes(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    saved = store.clones.expansion(clones)
    # 100 bytes cannot hold each of the 48 cells' clone, 8 bytes each, and
    # nothing is written.
    listed = store.artifacts.list(kind="clonal_expansion", scope="datastore")
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(
        MemoryBudgetError, match="for each cell's clone, .* Raise mem_budget"
    ):
        store.clones.expansion(clones, breakpoints=(1,))
    assert store.artifacts.list(kind="clonal_expansion", scope="datastore") == listed
    monkeypatch.undo()
    measured = store.clones.expansion(clones, breakpoints=(1,))
    assert measured != saved
    assert store.clones.load_expansion(measured).categories["n_cells"].sum() == 32


def test_reading_the_clones_gives_each_cells_code_and_the_labels(copied) -> None:
    store, clones = copied.store, copied.clones
    for with_ids in (False, True):
        ledger = MemoryLedger(None, "Reading clones")
        source = store._read_clone_input(clones, ledger, with_ids=with_ids)
        assert source.codes.dtype == np.int64
        assert len(source.codes) == N_CELLS
    assert source.clone_ids is not None
    assert source.clone_ids.tolist() == [str(label) for label in range(1, 9)]


def test_expansion_of_some_cells_measures_only_them(copied) -> None:
    store, clones = copied.store, copied.clones
    in_d2 = TRUTH.samples == "d2"
    n_d2 = int(in_d2.sum())
    store.cells.insert("is_d2", in_d2, overwrite=True)
    d2 = store.snapshot_cell_selection("is_d2")
    saved = store.clones.expansion(clones, cell_selection=d2)

    tables = store.clones.load_expansion(saved)
    assert tables.cells.shape[0] == n_d2


def test_expansion_refuses_cells_without_a_clone_and_a_group(copied) -> None:
    store = copied.store
    # Clones of no cell, a column without a label, and labels of the cells
    # without a clone.
    store.cells.insert("no_clone", np.full(N_CELLS, "", dtype=object))
    empty = store.clones.from_column("no_clone", cell_selection=copied.cells)
    insert_nullable_cell_column(
        store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    store.cells.insert("without_clone", TRUTH.clones < 0)
    others = store.snapshot_cluster_labels(
        "sample", cell_selection=store.snapshot_cell_selection("without_clone")
    )
    for clones, options in (
        (empty, {}),
        (empty, {"group_by": "sample"}),
        (copied.clones, {"group_by": "unlabelled"}),
        (copied.clones, {"groups": others}),
    ):
        with pytest.raises(ValueError, match="No cell has both a clone and a group"):
            store.clones.expansion(clones, **options)
    assert store.artifacts.list(kind="clonal_expansion", scope="datastore") == []


def test_labels_that_a_source_keeps_apart_follow_the_measured_cells(shared) -> None:
    """Clonotypes keep receptor types apart; their labels follow the cells measured."""
    from dataclasses import replace

    store = shared.store
    ledger = MemoryLedger(None, "Reading clones")
    source = store._read_clone_input(shared.clones, ledger)
    assert source.kept_apart == () and source.clone_ids is None
    kinds = np.asarray([f"kind{cell % 3}" for cell in range(N_CELLS)], dtype=object)
    labelled = replace(source, kept_apart=(("kind", kinds),))
    # The clones' own cells keep the labels as they are.
    cells, codes, kept = store._measured_clone_cells(labelled, None, ledger)
    assert cells == shared.cells and kept[0][1] is kinds
    d2 = TRUTH.samples == "d2"
    store.cells.insert("is_d2", d2, overwrite=True)
    selection = store.snapshot_cell_selection("is_d2")
    cells, codes, kept = store._measured_clone_cells(labelled, selection, ledger)
    assert cells == selection
    np.testing.assert_array_equal(codes, source.codes[d2])
    ((name, labels),) = kept
    assert name == "kind" and labels.tolist() == kinds[d2].tolist()
    # A measured cell that the clones do not hold has neither clone nor label.
    store.cells.insert("is_d4", ~d2, overwrite=True)
    d4_clones = import_truth_clones(
        store,
        TRUTH,
        Path(store.zw.store.root).parent,
        cell_selection=store.snapshot_cell_selection("is_d4"),
    )
    d4_source = replace(
        store._read_clone_input(d4_clones, ledger),
        kept_apart=(("kind", kinds[~d2]),),
    )
    _cells, codes, kept = store._measured_clone_cells(d4_source, shared.cells, ledger)
    assert (codes[d2] == -1).all()
    assert all(label is None for label in kept[0][1][d2])
    assert kept[0][1][~d2].tolist() == kinds[~d2].tolist()
