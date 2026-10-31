"""Clonal expansion of the truth's imported clones, against counts taken cell by cell.

The module writes a store of the truth's cells and imports the truth's clones
into it once, as a template. Tests that only measure share one copy, since
each measurement saves an artifact of its own; tests that change cell
metadata, reuse, read-only stores or the memory budget copy the template.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cytearc.clones import ClonalExpansion
from cytearc.clones.expansion import DEFAULT_RANK_SPLITS
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    copy_clone_store,
    every_cell,
    import_truth_clones,
    open_clone_store,
    write_truth_store,
)
from tests.clone_truth import CLONE_SIZES, N_CELLS, clone_truth
from tests.storage_helpers import insert_nullable_cell_column

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
        tables[group] = {"cells": cells, "clones": clones}
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
    store = copy_clone_store(template.path, tmp_path_factory.mktemp("shared"))
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


@pytest.fixture
def copied(template, tmp_path) -> SimpleNamespace:
    store = copy_clone_store(template.path, tmp_path)
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
        "cells",
        "clones",
        "fraction",
    ]
    assert table["category"].tolist() == list(CATEGORIES)
    assert table["cells"].tolist() == [1, 4, 12, 15, 0, 0]
    assert table["clones"].tolist() == [1, 2, 3, 2, 0, 0]
    np.testing.assert_allclose(table["fraction"], np.array([1, 4, 12, 15, 0, 0]) / 32)
    ranks = loaded.rank_windows
    assert ranks.columns.tolist() == ["group", "window", "cells", "fraction"]
    # Eight clones, all among the ten largest.
    assert ranks["cells"].tolist() == [N_IN_CLONES, 0, 0, 0, 0, 0, 0]


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
        "sample",
        "group_clone_size",
        "group_clone_frequency",
    ]
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
    assert cells["sample"].tolist() == [
        sample if clone >= 0 else np.nan
        for sample, clone in zip(TRUTH.samples, TRUTH.clones, strict=True)
    ]
    by_hand = _table_by_hand(TRUTH.group_clone_sizes(TRUTH.samples))
    table = loaded.categories
    assert table.columns.tolist() == [
        "group",
        "sample",
        "category",
        "cells",
        "clones",
        "fraction",
    ]
    for sample in ("d2", "d4"):
        rows = table[table["sample"] == sample]
        assert rows["group"].tolist() == [sample] * len(CATEGORIES)
        assert rows["cells"].tolist() == by_hand[sample]["cells"]
        assert rows["clones"].tolist() == by_hand[sample]["clones"]
    # The largest clone of each sample holds the window "1"; the rest follow.
    ranks = loaded.rank_windows.set_index(["sample", "window"])
    for sample, found in TRUTH.group_clone_sizes(TRUTH.samples).items():
        assert ranks.loc[(sample, "1"), "cells"] == found[0]
        assert ranks.loc[(sample, "after 1"), "cells"] == sum(found[1:])
        assert ranks.loc[(sample, "1"), "fraction"] == found[0] / sum(found)


def test_expansion_crosses_columns_with_a_label_artifact(shared) -> None:
    store = shared.store
    labels = store.snapshot_cluster_labels("cell_type", cell_selection=shared.cells)
    ref = store.clones.expansion(shared.clones, group_by="sample", groups=labels)
    loaded = store.clones.load_expansion(ref)
    assert loaded.groups == labels
    assert loaded.categories.columns[:4].tolist() == [
        "group",
        "sample",
        "cluster_labels",
        "category",
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
    totals = loaded.categories.groupby("group")["cells"].sum().to_dict()
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
    assert loaded.categories["cells"].tolist() == [8, 15, 9]
    assert loaded.categories["clones"].tolist() == [4, 3, 1]


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
    assert store.clones.expansion(clones, group_by="sample") != by_sample


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
    assert int(loaded.categories["cells"].sum()) == N_IN_CLONES - 5


def test_expansion_refuses_groups_that_labels_with_commas_name_alike(copied) -> None:
    store = copied.store
    # "x, y" with "z" and "x" with "y, z" both read "x, y, z".
    store.cells.insert("first", np.asarray(["x, y"] * 24 + ["x"] * 24))
    store.cells.insert("second", np.asarray(["z"] * 24 + ["y, z"] * 24))
    with pytest.raises(ValueError, match="remove the commas"):
        store.clones.expansion(copied.clones, group_by=["first", "second"])


def test_group_columns_may_not_share_a_name_with_the_result(copied) -> None:
    store = copied.store
    store.cells.insert("category", TRUTH.samples.astype(object))
    with pytest.raises(ValueError, match="'category', which is also a column"):
        store.clones.expansion(copied.clones, group_by="category")
    # A clones artifact as labels would name a key column after a result column.
    with pytest.raises(ValueError, match="key column 'clones' is also a column"):
        store.clones.expansion(copied.clones, groups=copied.clones)
    # A column that only other summaries hold is free.
    store.cells.insert("coverage", TRUTH.samples.astype(object))
    assert store.clones.expansion(copied.clones, group_by="coverage")


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
        "cells"
    ].sum().to_dict() == {
        "0 to 0.1234561": 1 + 2 + 2 + 3,
        "0.1234561 to 0.1234562": 0,
        "0.1234562 to 0.1234563": 0,
        "0.1234563 to 1": 4 + 5 + 6 + 9,
    }


def test_the_cells_table_orders_the_labels_of_key_columns_naturally(copied) -> None:
    store = copied.store
    sites = np.where(TRUTH.samples == "d2", "s2", "s10").astype(object)
    store.cells.insert("site", sites)
    loaded = store.clones.load_expansion(
        store.clones.expansion(copied.clones, group_by="site")
    )
    assert loaded.cells["site"].cat.categories.tolist() == ["s2", "s10"]
    assert loaded.categories["site"].drop_duplicates().tolist() == ["s2", "s10"]
    assert loaded.cells["site"].astype(object).tolist() == [
        site if clone >= 0 else np.nan
        for site, clone in zip(sites, TRUTH.clones, strict=True)
    ]


def test_expansion_refuses_other_artifacts(shared) -> None:
    store, clones = shared.store, shared.clones
    expansion = store.clones.expansion(clones)
    with pytest.raises(ValueError, match="clones must be a clones artifact"):
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
    reader = open_clone_store(Path(store.zw.store.root), zarr_mode="r")
    assert reader.clones.expansion(clones) == saved
    assert reader.clones.load_expansion(saved).categories["cells"].sum() == 32
    with pytest.raises(PermissionError, match="measure_clonal_expansion"):
        reader.clones.expansion(clones, breakpoints=(1,))


# CoSpar's table labels the truth's eight clones 1 to 8, stored one character
# wide; reading them holds each cell's stored int32 code and its int64 copy,
# and each label's 4 stored bytes three times and 100 more.
CLONES_READ = N_CELLS * (4 + 8) + 8 * (100 + 3 * 4)


def test_expansion_refuses_more_cells_than_the_memory_budget(
    copied, monkeypatch
) -> None:
    store, clones = copied.store, copied.clones
    assert artifact_group(store.zw, clones)["clone_ids"].dtype == np.dtype("<U1")
    saved = store.clones.expansion(clones)
    # Measuring 48 cells in one group takes 48 * 300 bytes and 2,000 for the
    # group.
    monkeypatch.setattr(store, "memoryBytes", 16_399)
    with pytest.raises(MemoryError, match="of 48 cells holds about 16400 bytes"):
        store.clones.expansion(clones, breakpoints=(1,))
    assert store.artifacts.list(kind="clonal_expansion", scope="datastore") == [saved]
    monkeypatch.setattr(store, "memoryBytes", 16_400)
    assert store.clones.expansion(clones, breakpoints=(1,)) != saved
    # Loading takes 48 * 500 bytes for the tables, 2,000 for the group, and
    # the clones read.
    needed = N_CELLS * 500 + 2_000 + CLONES_READ
    assert needed == 27_472
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"48 cells holds about {needed} bytes"):
        store.clones.load_expansion(saved)
    monkeypatch.setattr(store, "memoryBytes", needed)
    assert store.clones.load_expansion(saved).categories["cells"].sum() == 32


def test_reading_the_clones_is_checked_against_the_memory_budget(
    copied, monkeypatch
) -> None:
    store, clones = copied.store, copied.clones
    monkeypatch.setattr(store, "memoryBytes", CLONES_READ - 1)
    with pytest.raises(MemoryError, match=f"48 cells holds about {CLONES_READ} bytes"):
        store._read_clone_input(clones, with_ids=True)
    monkeypatch.setattr(store, "memoryBytes", CLONES_READ)
    source = store._read_clone_input(clones, with_ids=True)
    assert source.clone_ids is not None
    assert source.clone_ids.tolist() == [str(label) for label in range(1, 9)]
    # Without labels, only the codes.
    monkeypatch.setattr(store, "memoryBytes", N_CELLS * 12 - 1)
    with pytest.raises(MemoryError, match="48 cells holds about 576 bytes"):
        store._read_clone_input(clones)
    monkeypatch.setattr(store, "memoryBytes", N_CELLS * 12)
    assert store._read_clone_input(clones).clone_ids is None


def test_expansion_of_some_cells_counts_every_clone_cell_it_reads(
    copied, monkeypatch
) -> None:
    store, clones = copied.store, copied.clones
    in_d2 = TRUTH.samples == "d2"
    n_d2 = int(in_d2.sum())
    store.cells.insert("is_d2", in_d2, overwrite=True)
    d2 = store.snapshot_cell_selection("is_d2")
    saved = store.clones.expansion(clones, cell_selection=d2)
    # Matching the clones' 48 cells to the cells of d2 takes 56 bytes for each
    # clone cell and 32 for each measured cell.
    matched = N_CELLS * 56 + n_d2 * 32
    monkeypatch.setattr(store, "memoryBytes", matched - 1)
    with pytest.raises(
        MemoryError, match=f"48 cells to {n_d2} measured cells holds about {matched}"
    ):
        store.clones.expansion(clones, cell_selection=d2, breakpoints=(1,))
    # Loading their tables also reads and matches every clone cell.
    needed = n_d2 * 500 + 2_000 + CLONES_READ + matched
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"{n_d2} cells holds about {needed} bytes"):
        store.clones.load_expansion(saved)
    monkeypatch.setattr(store, "memoryBytes", needed)
    assert store.clones.load_expansion(saved).cells.shape[0] == n_d2


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
    source = store._read_clone_input(shared.clones)
    assert source.kept_apart == () and source.clone_ids is None
    kinds = np.asarray([f"kind{cell % 3}" for cell in range(N_CELLS)], dtype=object)
    labelled = replace(source, kept_apart=(("kind", kinds),))
    # The clones' own cells keep the labels as they are.
    cells, codes, kept = store._measured_clone_cells(labelled, None)
    assert cells == shared.cells and kept[0][1] is kinds
    d2 = TRUTH.samples == "d2"
    store.cells.insert("is_d2", d2, overwrite=True)
    selection = store.snapshot_cell_selection("is_d2")
    cells, codes, kept = store._measured_clone_cells(labelled, selection)
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
        store._read_clone_input(d4_clones),
        kept_apart=(("kind", kinds[~d2]),),
    )
    _cells, codes, kept = store._measured_clone_cells(d4_source, shared.cells)
    assert (codes[d2] == -1).all()
    assert all(label is None for label in kept[0][1][d2])
    assert kept[0][1][~d2].tolist() == kinds[~d2].tolist()
