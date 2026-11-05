"""Fate coupling and fate bias of the truth's imported clones.

The store and its clones are written once, as a template; tests that save
results, change cell metadata or the memory budget copy it. Expected values
are counted cell by cell from the truth, whose clones keep their rank as
codes, so clone ``k`` is labelled ``str(k + 1)``.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list
from scipy.stats import fisher_exact

from cytearc.clones import COUPLING_METHODS, CloneCoupling, CloneFateBias
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import (
    benjamini_hochberg,
    coupling_reference,
    every_cell,
    import_truth_clones,
    write_truth_store,
)
from cytearc.utils.memory_budget import MemoryBudgetError
from tests.clone_truth import CELL_TYPES, N_CELLS, clone_truth, number_label
from tests.storage_helpers import copy_store, insert_nullable_cell_column, open_store

TRUTH = clone_truth()
TYPES = sorted(CELL_TYPES)
EARLY = TRUTH.samples == "d2"
LATE = TRUTH.samples == "d4"


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("clone_fate")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    clones = import_truth_clones(store, TRUTH, folder)
    store.cells.insert("is_d2", EARLY)
    store.cells.insert("is_d4", LATE)
    store.cells.insert("cell", np.asarray([f"c{at:02d}" for at in range(N_CELLS)]))
    insert_nullable_cell_column(
        store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    return SimpleNamespace(path=folder / "store.zarr", clones=clones)


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    store = copy_store(template.path, tmp_path_factory.mktemp("shared"))
    return SimpleNamespace(
        store=store,
        clones=template.clones,
        early=store.snapshot_cell_selection("is_d2"),
        late=store.snapshot_cell_selection("is_d4"),
    )


@pytest.fixture
def copied(template, tmp_path) -> SimpleNamespace:
    store = copy_store(template.path, tmp_path)
    return SimpleNamespace(
        store=store,
        clones=template.clones,
        early=store.snapshot_cell_selection("is_d2"),
        late=store.snapshot_cell_selection("is_d4"),
    )


@pytest.mark.parametrize("method", COUPLING_METHODS)
def test_coupling_measures_every_pair_of_cell_types(shared, method) -> None:
    store, clones = shared.store, shared.clones
    saved = store.artifacts.list(scope="datastore")
    result = store.clones.coupling(clones, group_by="cell_type", method=method)
    assert isinstance(result, CloneCoupling)
    assert store.artifacts.list(scope="datastore") == saved
    table = TRUTH.group_clone_cells(TRUTH.cell_types)
    assert result.group_sizes.columns.tolist() == ["group", "n_cells", "n_clones"]
    assert result.group_sizes.join(result.group_keys, on="group")[
        ["group", "cell_type", "n_cells", "n_clones"]
    ].values.tolist() == [
        [name, name, sum(table[name].values()), len(table[name])] for name in TYPES
    ]
    assert result.pairs.columns.tolist() == [
        "group_1",
        "group_2",
        "coupling",
        "null_mean",
        "null_sd",
        "z_score",
    ]
    assert result.pairs[["group_1", "group_2"]].values.tolist() == [
        [TYPES[0], TYPES[1]],
        [TYPES[0], TYPES[2]],
        [TYPES[1], TYPES[2]],
    ]
    for row in result.pairs.itertuples(index=False):
        assert row.coupling == pytest.approx(
            coupling_reference(table, row.group_1, row.group_2, method),
            rel=1e-12,
            abs=1e-15,
        )
        assert row.z_score == pytest.approx(
            (row.coupling - row.null_mean) / row.null_sd, rel=1e-12
        )
    names = result.group_sizes["group"].to_numpy(dtype=object)
    assert result.order == tuple(names[leaves_list(result.linkage)].tolist())
    assert sorted(result.order) == TYPES
    assert (result.method, result.n_permutations, result.random_seed) == (
        method,
        1000,
        0,
    )
    assert (result.clones, result.group_by, result.groups) == (
        clones,
        ("cell_type",),
        None,
    )
    assert result.cell_selection is None


def test_the_null_is_seeded_and_does_not_depend_on_threads(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    first = store.clones.coupling(clones, group_by="cell_type", n_permutations=300)
    monkeypatch.setattr(store, "nthreads", 4)
    again = store.clones.coupling(clones, group_by="cell_type", n_permutations=300)
    assert again.pairs.equals(first.pairs)
    other = store.clones.coupling(
        clones, group_by="cell_type", n_permutations=300, random_seed=9
    )
    assert other.pairs["coupling"].equals(first.pairs["coupling"])
    assert not other.pairs["null_mean"].equals(first.pairs["null_mean"])
    assert other.random_seed == 9
    none = store.clones.coupling(clones, group_by="cell_type", n_permutations=0)
    assert none.pairs["coupling"].equals(first.pairs["coupling"])
    assert none.pairs[["null_mean", "null_sd", "z_score"]].isna().all(axis=None)
    # Without permutations there is no null to spread, so its count of pairs
    # without a spread is not measured; with them it counts the pairs whose
    # permutations agree.
    assert none.diagnostics["pairsWithoutNullSpread"] is None
    assert first.diagnostics["pairsWithoutNullSpread"] == int(
        (first.pairs["null_sd"] == 0).sum()
    )


def test_pairs_whose_null_has_no_spread_are_counted(tmp_path) -> None:
    from tests.clone_helpers import write_cells_store

    store = write_cells_store(tmp_path / "store.zarr", [f"c{i}" for i in range(6)])
    store.cells.insert("clone", np.asarray(["a"] * 6, dtype=object))
    store.cells.insert("cell_type", np.asarray(["x"] * 3 + ["y"] * 3, dtype=object))
    clones = store.clones.from_column(
        "clone", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    result = store.clones.coupling(clones, group_by="cell_type", n_permutations=20)
    # One clone holds every cell, so each shuffle of the labels gives the
    # coupling it gives: the null has no spread and the z-score no value.
    assert result.pairs["null_sd"].tolist() == [0.0]
    assert not np.isfinite(result.pairs["z_score"]).any()
    assert result.diagnostics == {
        "cellsWithoutClone": 0,
        "cellsWithoutGroup": 0,
        "groups": 2,
        "pairsWithoutNullSpread": 1,
    }


def test_coupling_keeps_to_a_cell_selection_and_takes_label_artifacts(shared) -> None:
    store, clones = shared.store, shared.clones
    within = store.clones.coupling(
        clones, group_by="cell_type", cell_selection=shared.late, n_permutations=0
    )
    table = TRUTH.group_clone_cells(TRUTH.cell_types, cells=LATE)
    assert within.group_sizes["n_cells"].tolist() == [
        sum(table[name].values()) for name in TYPES
    ]
    for row in within.pairs.itertuples(index=False):
        assert row.coupling == pytest.approx(
            coupling_reference(table, row.group_1, row.group_2, "share_cosine"),
            rel=1e-12,
        )
    assert within.cell_selection == shared.late
    labels = store.snapshot_cluster_labels(
        "cell_type", cell_selection=every_cell(store)
    )
    by_labels = store.clones.coupling(clones, groups=labels, n_permutations=0)
    by_column = store.clones.coupling(clones, group_by="cell_type", n_permutations=0)
    assert by_labels.group_sizes.columns.tolist() == ["group", "n_cells", "n_clones"]
    assert by_labels.group_keys.to_dict("index") == {
        name: {"cluster_labels": name} for name in TYPES
    }
    assert by_labels.pairs.equals(by_column.pairs)
    assert by_labels.groups == labels


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"method": "cosine"}, ValueError, "method must be one of share_cosine"),
        ({"n_permutations": 1}, ValueError, "^n_permutations must be 0"),
        ({"n_permutations": -3}, ValueError, "^n_permutations must be 0"),
        ({"n_permutations": True}, TypeError, "^n_permutations must be"),
        ({"n_permutations": 2.5}, TypeError, "^n_permutations must be"),
        ({"random_seed": -1}, ValueError, "^random_seed must be a non-negative"),
        # A seed of 2**64 would draw the permutations of seed 0.
        ({"random_seed": 2**64}, ValueError, r"below 2\*\*64"),
        ({"random_seed": "0"}, TypeError, "^random_seed must be a non-negative"),
        ({"group_by": "missing"}, ValueError, "not a cell metadata column"),
    ],
)
def test_coupling_arguments_are_checked(shared, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        shared.store.clones.coupling(
            shared.clones, **{"group_by": "cell_type", **kwargs}
        )


def test_coupling_runs_on_no_more_threads_than_numba_has(copied, monkeypatch) -> None:
    import numba

    from cytearc.utils.logging import logger

    store, clones = copied.store, copied.clones
    monkeypatch.setattr(store, "nthreads", 10_000)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        store.clones.coupling(clones, group_by="cell_type", n_permutations=500)
    finally:
        logger.remove(sink)
    (line,) = [message for message in messages if "coupling of" in message]
    threads = min(500, int(numba.config.NUMBA_NUM_THREADS))
    assert f"on {threads} threads" in line, line


def test_coupling_needs_two_groups_with_a_clone(copied) -> None:
    store, clones = copied.store, copied.clones
    with pytest.raises(ValueError, match="No cell has both a clone and a group"):
        store.clones.coupling(clones, group_by="unlabelled")
    with pytest.raises(ValueError, match="at least two groups"):
        store.clones.coupling(clones, group_by="sample", cell_selection=copied.early)
    # A column named like the result's own columns groups the cells as any
    # other, and its labels stay in the keys.
    neutral = store.clones.coupling(clones, group_by="cell_type", n_permutations=0)
    for name in ("group", "n_cells", "n_clones", "coupling", "cells"):
        store.cells.insert(name, TRUTH.cell_types.astype(object), overwrite=True)
        named = store.clones.coupling(clones, group_by=name, n_permutations=0)
        assert named.order == neutral.order
        pd.testing.assert_frame_equal(named.group_sizes, neutral.group_sizes)
        pd.testing.assert_frame_equal(named.pairs, neutral.pairs)
        assert named.group_keys.to_dict("index") == {
            type_: {name: type_} for type_ in TYPES
        }


# Clones that keep labels apart, as clonotypes keep receptor types apart,
# label every cell of a clone alike: here clones 1 to 4 "A" and 5 to 8 "B".
# The cells without a clone are labelled "A" too, as a source may label every
# cell.
LINEAGES = np.where(TRUTH.clones < 4, "A", "B").astype(object)


@pytest.fixture
def kept_apart(copied, monkeypatch) -> SimpleNamespace:
    """A store whose clones keep the labels of ``LINEAGES`` apart."""
    store = copied.store
    read = store._read_clone_input

    def with_lineages(clones, ledger, *, with_ids=False):
        return replace(
            read(clones, ledger, with_ids=with_ids),
            kept_apart=(("lineage", LINEAGES),),
        )

    monkeypatch.setattr(store, "_read_clone_input", with_lineages)
    return copied


def test_coupling_compares_only_groups_of_the_labels_kept_apart(kept_apart) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    result = store.clones.coupling(
        clones, group_by="sample", method="normalized_covariance", n_permutations=200
    )
    assert result.group_sizes["group"].tolist() == ["d2, A", "d2, B", "d4, A", "d4, B"]
    assert result.pairs[["group_1", "group_2"]].values.tolist() == [
        ["d2, A", "d4, A"],
        ["d2, B", "d4, B"],
    ]
    # Each lineage's covariance counts its own clones.
    for row, lineage in zip(result.pairs.itertuples(index=False), "AB", strict=True):
        kept = LINEAGES == lineage
        table = TRUTH.group_clone_cells(TRUTH.samples, cells=kept)
        assert row.coupling == pytest.approx(
            coupling_reference(table, "d2", "d4", "normalized_covariance"), rel=1e-12
        )
    # The lineages join last in the tree.
    assert set(result.order[:2]) in ({"d2, A", "d4, A"}, {"d2, B", "d4, B"})
    store.cells.insert(
        "one_lineage_each", (EARLY & (LINEAGES == "A")) | (LATE & (LINEAGES == "B"))
    )
    with pytest.raises(
        ValueError, match=r"No two groups share their labels in \['lineage'\]"
    ):
        store.clones.coupling(
            clones,
            group_by="sample",
            cell_selection=store.snapshot_cell_selection("one_lineage_each"),
        )


def _late_cells_by_type(clone: int) -> Counter:
    """A clone's late cells of each type, counted cell by cell."""
    return Counter(
        str(cell_type)
        for cell_type, found, late in zip(
            TRUTH.cell_types, TRUTH.clones, LATE, strict=True
        )
        if late and found == clone
    )


def test_fate_bias_follows_each_early_cell_to_its_clones_late_cells(copied) -> None:
    store, clones = copied.store, copied.clones
    bias = store.clones.fate_bias(
        clones, group_by="cell_type", early=copied.early, late=copied.late
    )
    assert bias.kind == "clone_fate_bias"
    loaded = store.clones.load_fate_bias(bias)
    assert isinstance(loaded, CloneFateBias)
    assert (loaded.clones, loaded.early, loaded.late) == (
        clones,
        copied.early,
        copied.late,
    )
    assert (loaded.group_by, loaded.groups) == (("cell_type",), None)
    cells = loaded.cells
    assert cells.index.tolist() == TRUTH.cell_ids[EARLY].tolist()
    assert cells.columns.tolist() == ["clone", "n_late_cells", "fate"]
    # The shares of the fates sit in a table of their own, named by the fates.
    fractions = loaded.fractions
    assert fractions.index.name == "ids"
    assert fractions.index.tolist() == cells.index.tolist()
    assert fractions.columns.tolist() == TYPES
    dominant = []
    for (cell_id, row), clone in zip(
        cells.iterrows(), TRUTH.clones[EARLY], strict=True
    ):
        found = _late_cells_by_type(int(clone)) if clone >= 0 else Counter()
        total = sum(found.values())
        assert row["n_late_cells"] == total, cell_id
        if clone < 0:
            assert row.isna()["clone"]
        else:
            assert row["clone"] == number_label(int(clone))
        if total == 0:
            assert fractions.loc[cell_id].isna().all() and row.isna()["fate"]
            dominant.append("")
            continue
        for name in TYPES:
            assert fractions.loc[cell_id, name] == pytest.approx(
                found[name] / total, rel=1e-15
            )
        top = max(found.values())
        winners = [name for name in TYPES if found[name] == top]
        expected = winners[0] if len(winners) == 1 else None
        assert (row["fate"] if isinstance(row["fate"], str) else None) == expected
        dominant.append(expected or "")
    # The canonical values hold each early cell's dominant fate.
    values = store.artifacts.load_values(bias)
    assert values.values.tolist() == dominant
    assert values.cell_ids.tolist() == TRUTH.cell_ids[EARLY].tolist()


def test_fate_bias_tests_each_clone_against_all_late_cells(shared) -> None:
    store, clones = shared.store, shared.clones
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(
            clones, group_by="cell_type", early=shared.early, late=shared.late
        )
    )
    late_types = Counter(str(name) for name in TRUTH.cell_types[LATE])
    assert loaded.fates.to_dict("list") == {
        "fate": TYPES,
        "n_late_cells": [late_types[name] for name in TYPES],
    }
    # The fates' labels, indexed by the fates' names.
    assert loaded.group_keys.index.name == "fate"
    assert loaded.group_keys.to_dict("index") == {
        name: {"cell_type": name} for name in TYPES
    }
    n_late = int(LATE.sum())
    tested = sorted({int(c) for c in TRUTH.clones[LATE] if c >= 0})
    table = loaded.clone_fates
    assert table["clone"].tolist() == [
        number_label(clone) for clone in tested for _name in TYPES
    ]
    assert table["fate"].tolist() == TYPES * len(tested)
    for fate in TYPES:
        rows = table[table["fate"] == fate]
        p_values = []
        for row, clone in zip(rows.itertuples(index=False), tested, strict=True):
            found = _late_cells_by_type(clone)
            total, x, in_fate = sum(found.values()), found[fate], late_types[fate]
            assert (row.n_cells, row.n_late_cells) == (x, total)
            assert row.n_early_cells == int(
                np.count_nonzero(EARLY & (TRUTH.clones == clone))
            )
            assert row.fraction == pytest.approx(x / total, rel=1e-15)
            # Fisher's exact test of more of the clone's late cells in the
            # fate than among all late cells.
            expected = fisher_exact(
                [[x, total - x], [in_fate - x, n_late - in_fate - (total - x)]],
                alternative="greater",
            ).pvalue
            assert row.p_value == pytest.approx(expected, rel=1e-9)
            p_values.append(row.p_value)
        np.testing.assert_allclose(
            rows["q_value"], benjamini_hochberg(np.asarray(p_values)), rtol=1e-12
        )


def test_an_early_cell_whose_clone_has_no_late_cells_has_no_bias(shared) -> None:
    store, clones = shared.store, shared.clones
    # Clone 8's one cell is late, so as the early cell it has no late cell;
    # the early cells of d4 in clone 8 and the late cells of d2 swap roles.
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(
            clones, group_by="cell_type", early=shared.late, late=shared.early
        )
    )
    alone = loaded.cells.loc[TRUTH.cell_ids[TRUTH.clones == 7]]
    assert alone["n_late_cells"].tolist() == [0]
    shares = loaded.fractions.loc[alone.index, TYPES]
    assert shares.isna().all(axis=None) and alone["fate"].isna().all()
    assert "8" not in set(loaded.clone_fates["clone"].astype(str))


def test_fate_bias_is_reused_and_selected_like_labels(copied) -> None:
    store, clones = copied.store, copied.clones
    options = {"group_by": "cell_type", "early": copied.early, "late": copied.late}
    first = store.clones.fate_bias(clones, **options)
    assert store.clones.fate_bias(clones, **options) == first
    # Measuring again saves another artifact, which later calls reuse.
    rebuilt = store.clones.fate_bias(clones, **options, invalidate_cache=True)
    assert rebuilt != first
    first = rebuilt
    labels = store.artifacts.load_values(first).values
    chosen = store.select_cells(first, include=["Neutrophil"])
    selected = store.artifacts.load_values(first, cell_selection=chosen)
    assert set(selected.cell_ids.tolist()) == set(
        TRUTH.cell_ids[EARLY][labels == "Neutrophil"].tolist()
    )
    # Other labels in the columns of the fates give another result, and the
    # same labels again reuse it.
    renamed = np.char.add(TRUTH.cell_types.astype(str), "s").astype(object)
    store.cells.insert("cell_type", renamed, overwrite=True)
    assert store.clones.fate_bias(clones, **options) != first
    store.cells.insert("cell_type", TRUTH.cell_types.astype(object), overwrite=True)
    assert store.clones.fate_bias(clones, **options) == first


def test_saved_fate_bias_with_damaged_clone_arrays_is_measured_again(copied) -> None:
    store, clones = copied.store, copied.clones
    options = {"group_by": "cell_type", "early": copied.early, "late": copied.late}
    saved = store.clones.fate_bias(clones, **options)
    group = artifact_group(store.zw, saved)
    # One column of p-values instead of one per fate.
    p_values = np.asarray(group["clone_p_values"][:])
    group.create_array("clone_p_values", data=p_values[:, :1], overwrite=True)
    rebuilt = store.clones.fate_bias(clones, **options)
    assert rebuilt != saved
    assert artifact_group(store.zw, rebuilt)["clone_p_values"].shape == p_values.shape


def test_fate_bias_that_an_earlier_version_saved_is_refused(copied) -> None:
    store, clones = copied.store, copied.clones
    options = {"group_by": "cell_type", "early": copied.early, "late": copied.late}
    saved = store.clones.fate_bias(clones, **options)
    # An earlier version stored each early cell's late clone-mates as
    # late_cells.
    group = artifact_group(store.zw, saved)
    group.create_array("late_cells", data=np.asarray(group["n_late_cells"][:]))
    del group["n_late_cells"]
    with pytest.raises(
        ValueError,
        match="^bias was saved by an earlier version of CyteArc, without the arrays "
        "n_late_cells; measure the fate bias again with ds.clones.fate_bias",
    ):
        store.clones.load_fate_bias(saved)
    # The call that the refusal names measures it anew.
    again = store.clones.fate_bias(clones, **options)
    assert again != saved
    assert "n_late_cells" in store.clones.load_fate_bias(again).cells.columns


def test_a_read_only_store_reuses_saved_fate_bias(copied) -> None:
    store, clones = copied.store, copied.clones
    options = {"group_by": "cell_type", "early": copied.early, "late": copied.late}
    saved = store.clones.fate_bias(clones, **options)
    reader = open_store(Path(store.zw.store.root), zarr_mode="r")
    assert reader.clones.fate_bias(clones, **options) == saved
    assert reader.clones.load_fate_bias(saved).cells.shape[0] == int(EARLY.sum())
    with pytest.raises(PermissionError, match="measure_clone_fate_bias"):
        reader.clones.fate_bias(
            clones, **{**options, "group_by": ["cell_type", "sample"]}
        )


def test_fate_bias_names_late_when_none_of_its_cells_has_a_clone(copied) -> None:
    store, clones = copied.store, copied.clones
    # Late cells of the clones' own cells, none of which the clones give a
    # clone, and early cells that hold clones.
    without = (TRUTH.clones < 0) & LATE
    assert without.any()
    store.cells.insert("late_without_clone", without)
    late = store.snapshot_cell_selection("late_without_clone")
    with pytest.raises(
        ValueError,
        match=f"^None of the {int(without.sum())} cells of late has a clone; pass "
        "late cells that the clones give a clone$",
    ):
        store.clones.fate_bias(
            clones, group_by="cell_type", early=copied.early, late=late
        )
    assert store.artifacts.list(kind="clone_fate_bias", scope="datastore") == []


def test_fate_bias_arguments_are_checked(copied) -> None:
    store, clones = copied.store, copied.clones
    early, late = copied.early, copied.late
    with pytest.raises(TypeError, match="early must be an ArtifactRef"):
        store.clones.fate_bias(clones, group_by="cell_type", early="is_d2", late=late)
    with pytest.raises(ValueError, match="late must be a cell_selection artifact"):
        store.clones.fate_bias(clones, group_by="cell_type", early=early, late=clones)
    with pytest.raises(TypeError, match="invalidate_cache must be"):
        store.clones.fate_bias(
            clones, group_by="cell_type", early=early, late=late, invalidate_cache=1
        )
    everyone = every_cell(store)
    with pytest.raises(ValueError, match=f"share {int(LATE.sum())} cells"):
        store.clones.fate_bias(clones, group_by="cell_type", early=everyone, late=late)
    store.cells.insert("no_cell", np.zeros(N_CELLS, dtype=bool))
    none = store.snapshot_cell_selection("no_cell")
    with pytest.raises(ValueError, match="^early selects no cell$"):
        store.clones.fate_bias(clones, group_by="cell_type", early=none, late=late)
    with pytest.raises(ValueError, match="^late selects no cell$"):
        store.clones.fate_bias(clones, group_by="cell_type", early=early, late=none)
    with pytest.raises(ValueError, match="No cell has both a clone and a group"):
        store.clones.fate_bias(clones, group_by="unlabelled", early=early, late=late)
    with pytest.raises(ValueError, match="at least two fates"):
        store.clones.fate_bias(clones, group_by="sample", early=early, late=late)
    assert store.artifacts.list(kind="clone_fate_bias", scope="datastore") == []
    with pytest.raises(TypeError, match="bias must be an ArtifactRef"):
        store.clones.load_fate_bias("bias")
    with pytest.raises(ValueError, match="bias must be a clone_fate_bias artifact"):
        store.clones.load_fate_bias(clones)


@pytest.mark.parametrize(
    "name", ["clone", "n_late_cells", "fate", "n_early_cells", "fraction", "ids"]
)
def test_fates_may_take_the_names_of_the_tables_columns(copied, name) -> None:
    store, clones = copied.store, copied.clones
    early, late = copied.early, copied.late
    neutral = store.clones.load_fate_bias(
        store.clones.fate_bias(clones, group_by="cell_type", early=early, late=late)
    )
    # The monocytes take a fate named like a column of the tables, which holds
    # the shares of the monocytes' fate and changes nothing else.
    renamed = np.where(TRUTH.cell_types == "Monocyte", name, TRUTH.cell_types)
    store.cells.insert("named", renamed.astype(object), overwrite=True)
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(clones, group_by="named", early=early, late=late)
    )
    names = {"Monocyte": name}
    assert loaded.cells.columns.tolist() == ["clone", "n_late_cells", "fate"]
    pd.testing.assert_series_equal(loaded.cells["clone"], neutral.cells["clone"])
    pd.testing.assert_series_equal(
        loaded.cells["n_late_cells"], neutral.cells["n_late_cells"]
    )
    assert loaded.cells["fate"].astype(object).tolist() == [
        names.get(fate, fate) if isinstance(fate, str) else fate
        for fate in neutral.cells["fate"].astype(object).tolist()
    ]
    # The shares are unchanged, under the fates' own names, in the order of
    # the fates.
    order = loaded.fates["fate"].tolist()
    assert sorted(order) == sorted(names.get(fate, fate) for fate in TYPES)
    assert loaded.fractions.columns.tolist() == order
    pd.testing.assert_frame_equal(
        loaded.fractions, neutral.fractions.rename(columns=names)[order]
    )
    assert loaded.group_keys.index.tolist() == order
    assert loaded.group_keys.to_dict("index") == {
        fate: {"named": fate} for fate in order
    }
    by_fate = loaded.clone_fates.assign(fate=loaded.clone_fates["fate"].astype(str))
    expected = neutral.clone_fates.assign(
        fate=neutral.clone_fates["fate"].astype(str).replace(names)
    )
    columns = ["clone", "fate", "n_cells", "p_value"]
    assert sorted(map(tuple, by_fate[columns].astype(str).values.tolist())) == sorted(
        map(tuple, expected[columns].astype(str).values.tolist())
    )
    # Every saved fate bias reads back.
    assert len(
        store.clones.load_fate_bias(
            store.clones.fate_bias(clones, group_by="named", early=early, late=late)
        ).cells
    ) == int(EARLY.sum())


def test_fate_bias_tests_clones_within_the_labels_kept_apart(kept_apart) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(
            clones, group_by="cell_type", early=kept_apart.early, late=kept_apart.late
        )
    )
    # The labels kept apart belong to the clones, so a late cell without a
    # clone has no fate, whatever its source labels it; the fates hold the
    # late cells with a clone, by type and lineage.
    late_with_clone = LATE & (TRUTH.clones >= 0)
    assert int(np.count_nonzero(LATE & (TRUTH.clones < 0))) > 0
    expected = Counter(
        f"{name}, {lineage}"
        for name, lineage in zip(
            TRUTH.cell_types[late_with_clone], LINEAGES[late_with_clone], strict=True
        )
    )
    assert dict(
        zip(loaded.fates["fate"], loaded.fates["n_late_cells"], strict=True)
    ) == (dict(expected))
    assert loaded.fates["fate"].tolist() == sorted(expected)
    assert len(loaded.fates) == 5
    # Each clone is tested only in the fates of its lineage, among the late
    # cells of that lineage with a clone.
    table = loaded.clone_fates
    for row in table.itertuples(index=False):
        lineage = "A" if int(row.clone) <= 4 else "B"
        assert row.fate.endswith(lineage)
        in_lineage = late_with_clone & (LINEAGES == lineage)
        n_late = int(in_lineage.sum())
        fate = row.fate.split(", ")[0]
        in_fate = int((in_lineage & (TRUTH.cell_types == fate)).sum())
        expected = fisher_exact(
            [
                [row.n_cells, row.n_late_cells - row.n_cells],
                [
                    in_fate - row.n_cells,
                    n_late - in_fate - (row.n_late_cells - row.n_cells),
                ],
            ],
            alternative="greater",
        ).pvalue
        assert row.p_value == pytest.approx(expected, rel=1e-9)


def _own_labels(store, mask: np.ndarray):
    """A label artifact of a label of its own, three characters wide, for each cell of ``mask``."""
    store.cells.insert("chosen", mask, overwrite=True)
    own = store.snapshot_cluster_labels(
        "cell", cell_selection=store.snapshot_cell_selection("chosen")
    )
    assert artifact_group(store.zw, own)["values"].dtype == np.dtype("<U3")
    return own


def test_coupling_refuses_a_small_budget_and_batches_its_permutations_alike(
    copied, monkeypatch
) -> None:
    store, clones = copied.store, copied.clones
    # A group of its own for each of the 32 cells with a clone, whose small
    # labels leave the coupling the largest step, which 200 kB cannot hold.
    own = _own_labels(store, TRUTH.clones >= 0)
    monkeypatch.setattr(store, "memoryBytes", 200_000)
    with pytest.raises(
        MemoryBudgetError,
        match=r"for measuring the coupling of 32 groups, 496 pairs, .* or compare "
        "fewer groups",
    ):
        store.clones.coupling(clones, groups=own, n_permutations=50)
    monkeypatch.undo()
    # One thread measures the 50 permutations in batches, four threads in one
    # batch, and both measure the same permutations.
    monkeypatch.setattr(store, "nthreads", 1)
    batched = store.clones.coupling(clones, groups=own, n_permutations=50)
    assert len(batched.pairs) == 496
    monkeypatch.setattr(store, "nthreads", 4)
    whole = store.clones.coupling(clones, groups=own, n_permutations=50)
    assert whole.pairs.equals(batched.pairs)


def test_fate_bias_refuses_a_small_budget_before_it_writes(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    early, late = copied.early, copied.late
    n_early, n_late = int(EARLY.sum()), int(LATE.sum())
    # A fate of its own for each late cell, named by three characters, makes
    # the measurement the largest step, about 72 kB, which 40 kB cannot hold
    # beside the steps before it, about 15 kB at most; nothing is written.
    fates = _own_labels(store, LATE)
    options = {"groups": fates, "early": early, "late": late}
    listed = store.artifacts.list(kind="clone_fate_bias", scope="datastore")
    monkeypatch.setattr(store, "memoryBytes", 40_000)
    with pytest.raises(MemoryBudgetError, match="into fewer fates") as refused:
        store.clones.fate_bias(clones, **options)
    assert refused.value.item == (
        f"measuring the fate bias of {n_early} early cells and 8 clones in {n_late} fates"
    )
    assert store.artifacts.list(kind="clone_fate_bias", scope="datastore") == listed
    monkeypatch.undo()
    bias = store.clones.load_fate_bias(store.clones.fate_bias(clones, **options))
    assert bias.cells.shape == (n_early, 3)
    assert bias.fractions.shape == (n_early, n_late)


def test_the_artifact_stores_its_arrays_by_early_cell_clone_and_fate(shared) -> None:
    store, clones = shared.store, shared.clones
    bias = store.clones.fate_bias(
        clones, group_by="cell_type", early=shared.early, late=shared.late
    )
    group = artifact_group(store.zw, bias)
    n_early = int(EARLY.sum())
    tested = len({int(c) for c in TRUTH.clones[LATE] if c >= 0})
    shapes = {name: group[name].shape for name in group.array_keys()}
    assert {name: shapes[name] for name in ("codes", "fate_codes", "n_late_cells")} == {
        "codes": (n_early,),
        "fate_codes": (n_early,),
        "n_late_cells": (n_early,),
    }
    assert shapes["fractions"] == (n_early, 3)
    assert shapes["clone_codes"] == (tested,)
    assert shapes["clone_fate_cells"] == shapes["clone_p_values"] == (tested, 3)
    assert shapes["fate_cells"] == (3,)
    assert group.attrs["key_columns"] == ["cell_type"]
    assert group.attrs["group_keys"] == [{"cell_type": name} for name in TYPES]
    np.testing.assert_array_equal(
        group["codes"][:], np.where(TRUTH.clones[EARLY] >= 0, TRUTH.clones[EARLY], -1)
    )
