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
import pytest
from scipy.cluster.hierarchy import leaves_list
from scipy.stats import fisher_exact

from cytearc.clones import COUPLING_METHODS, CloneCoupling, CloneFateBias
from cytearc.datastore._operations import clones as clone_operations
from cytearc.metadata.rows import READ_OBJECT_BYTES, array_row_selection_parts
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import (
    benjamini_hochberg,
    copy_clone_store,
    coupling_reference,
    every_cell,
    ids_read_bytes,
    import_truth_clones,
    open_clone_store,
    write_truth_store,
)
from tests.clone_truth import CELL_TYPES, N_CELLS, clone_truth, number_label
from tests.storage_helpers import insert_nullable_cell_column

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
    store = copy_clone_store(template.path, tmp_path_factory.mktemp("shared"))
    return SimpleNamespace(
        store=store,
        clones=template.clones,
        early=store.snapshot_cell_selection("is_d2"),
        late=store.snapshot_cell_selection("is_d4"),
    )


@pytest.fixture
def copied(template, tmp_path) -> SimpleNamespace:
    store = copy_clone_store(template.path, tmp_path)
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
    assert result.group_sizes[
        ["group", "cell_type", "cells", "clones"]
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
    assert (result.method, result.permutations, result.seed) == (method, 1000, 0)
    assert (result.clones, result.group_by, result.groups) == (
        clones,
        ("cell_type",),
        None,
    )
    assert result.cell_selection is None


def test_the_null_is_seeded_and_does_not_depend_on_threads(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    first = store.clones.coupling(clones, group_by="cell_type", permutations=300)
    monkeypatch.setattr(store, "nthreads", 4)
    again = store.clones.coupling(clones, group_by="cell_type", permutations=300)
    assert again.pairs.equals(first.pairs)
    other = store.clones.coupling(
        clones, group_by="cell_type", permutations=300, seed=9
    )
    assert other.pairs["coupling"].equals(first.pairs["coupling"])
    assert not other.pairs["null_mean"].equals(first.pairs["null_mean"])
    assert other.seed == 9
    none = store.clones.coupling(clones, group_by="cell_type", permutations=0)
    assert none.pairs["coupling"].equals(first.pairs["coupling"])
    assert none.pairs[["null_mean", "null_sd", "z_score"]].isna().all(axis=None)


def test_coupling_keeps_to_a_cell_selection_and_takes_label_artifacts(shared) -> None:
    store, clones = shared.store, shared.clones
    within = store.clones.coupling(
        clones, group_by="cell_type", cell_selection=shared.late, permutations=0
    )
    table = TRUTH.group_clone_cells(TRUTH.cell_types, cells=LATE)
    assert within.group_sizes["cells"].tolist() == [
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
    by_labels = store.clones.coupling(clones, groups=labels, permutations=0)
    by_column = store.clones.coupling(clones, group_by="cell_type", permutations=0)
    assert by_labels.group_sizes.columns.tolist() == [
        "group",
        "cluster_labels",
        "cells",
        "clones",
    ]
    assert by_labels.pairs.equals(by_column.pairs)
    assert by_labels.groups == labels


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"method": "cosine"}, ValueError, "method must be one of share_cosine"),
        ({"permutations": 1}, ValueError, "permutations must be 0"),
        ({"permutations": -3}, ValueError, "permutations must be 0"),
        ({"permutations": True}, TypeError, "permutations must be"),
        ({"permutations": 2.5}, TypeError, "permutations must be"),
        ({"seed": -1}, ValueError, "seed must be a non-negative"),
        # A seed of 2**64 would draw the permutations of seed 0.
        ({"seed": 2**64}, ValueError, r"below 2\*\*64"),
        ({"seed": "0"}, TypeError, "seed must be a non-negative"),
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
        store.clones.coupling(clones, group_by="cell_type", permutations=500)
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
    store.cells.insert("cells", TRUTH.cell_types.astype(object))
    with pytest.raises(ValueError, match="'cells', which is also a column"):
        store.clones.coupling(clones, group_by="cells")


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

    def with_lineages(clones, *, with_ids=False):
        return replace(
            read(clones, with_ids=with_ids), kept_apart=(("lineage", LINEAGES),)
        )

    monkeypatch.setattr(store, "_read_clone_input", with_lineages)
    return copied


def test_coupling_compares_only_groups_of_the_labels_kept_apart(kept_apart) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    result = store.clones.coupling(
        clones, group_by="sample", method="normalized_covariance", permutations=200
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
    assert cells.columns.tolist() == ["clone", "late_cells", "fate", *TYPES]
    dominant = []
    for (cell_id, row), clone in zip(
        cells.iterrows(), TRUTH.clones[EARLY], strict=True
    ):
        found = _late_cells_by_type(int(clone)) if clone >= 0 else Counter()
        total = sum(found.values())
        assert row["late_cells"] == total, cell_id
        if clone < 0:
            assert row.isna()["clone"]
        else:
            assert row["clone"] == number_label(int(clone))
        if total == 0:
            assert row[TYPES].isna().all() and row.isna()["fate"]
            dominant.append("")
            continue
        for name in TYPES:
            assert row[name] == pytest.approx(found[name] / total, rel=1e-15)
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
        "cell_type": TYPES,
        "late_cells": [late_types[name] for name in TYPES],
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
            assert (row.cells, row.late_cells) == (x, total)
            assert row.early_cells == int(
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
    assert alone["late_cells"].tolist() == [0]
    assert alone[TYPES].isna().all(axis=None) and alone["fate"].isna().all()
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


def test_a_read_only_store_reuses_saved_fate_bias(copied) -> None:
    store, clones = copied.store, copied.clones
    options = {"group_by": "cell_type", "early": copied.early, "late": copied.late}
    saved = store.clones.fate_bias(clones, **options)
    reader = open_clone_store(Path(store.zw.store.root), zarr_mode="r")
    assert reader.clones.fate_bias(clones, **options) == saved
    assert reader.clones.load_fate_bias(saved).cells.shape[0] == int(EARLY.sum())
    with pytest.raises(PermissionError, match="measure_clone_fate_bias"):
        reader.clones.fate_bias(
            clones, **{**options, "group_by": ["cell_type", "sample"]}
        )


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
    store.cells.insert(
        "named",
        np.where(TRUTH.cell_types == "Monocyte", "clone", "other").astype(object),
    )
    with pytest.raises(ValueError, match="A fate is named 'clone'"):
        store.clones.fate_bias(clones, group_by="named", early=early, late=late)
    store.cells.insert("fraction", TRUTH.cell_types.astype(object))
    with pytest.raises(ValueError, match="'fraction', which is also a column"):
        store.clones.fate_bias(clones, group_by="fraction", early=early, late=late)
    assert store.artifacts.list(kind="clone_fate_bias", scope="datastore") == []
    with pytest.raises(TypeError, match="bias must be an ArtifactRef"):
        store.clones.load_fate_bias("bias")
    with pytest.raises(ValueError, match="bias must be a clone_fate_bias artifact"):
        store.clones.load_fate_bias(clones)


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
    assert dict(zip(loaded.fates["fate"], loaded.fates["late_cells"], strict=True)) == (
        dict(expected)
    )
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
                [row.cells, row.late_cells - row.cells],
                [in_fate - row.cells, n_late - in_fate - (row.late_cells - row.cells)],
            ],
            alternative="greater",
        ).pvalue
        assert row.p_value == pytest.approx(expected, rel=1e-9)


# Reading the truth's clones without labels holds each of its 48 cells'
# stored int32 code and its int64 copy.
CODES_READ = N_CELLS * (4 + 8)


def _own_labels(store, mask: np.ndarray):
    """A label artifact of a label of its own, three characters wide, for each cell of ``mask``."""
    store.cells.insert("chosen", mask, overwrite=True)
    own = store.snapshot_cluster_labels(
        "cell", cell_selection=store.snapshot_cell_selection("chosen")
    )
    assert artifact_group(store.zw, own)["values"].dtype == np.dtype("<U3")
    return own


def test_coupling_refuses_a_budget_below_one_worker(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    # A group of its own for each of the 32 cells with a clone, whose small
    # labels leave the coupling the largest step.
    own = _own_labels(store, TRUTH.clones >= 0)
    # 32 groups and 496 pairs: 250 bytes per pair, 40 per product of two
    # groups, 80 per count of a clone in a group and 12 per cell with a
    # clone; one worker's 4 bytes per cell, 8 per product and 32 per group;
    # one permutation's 8 bytes per pair; and beside them the clones read, 40
    # bytes per count and 2,000 per group.
    needed = (
        250 * 496
        + 40 * 32**2
        + 80 * 32
        + 12 * 32
        + (4 * 32 + 8 * 32**2 + 32 * 32)
        + 8 * 496
        + CODES_READ
        + 40 * 32
        + 2_000 * 32
    )
    assert needed == 247_072
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"496 pairs, holds about {needed} bytes"):
        store.clones.coupling(clones, groups=own)
    monkeypatch.setattr(store, "memoryBytes", needed)
    tight = store.clones.coupling(clones, groups=own, permutations=50)
    assert len(tight.pairs) == 496
    # More threads and more memory measure the same permutations.
    monkeypatch.setattr(store, "nthreads", 4)
    monkeypatch.setattr(store, "memoryBytes", 10**9)
    roomy = store.clones.coupling(clones, groups=own, permutations=50)
    assert roomy.pairs.equals(tight.pairs)


def test_fate_bias_steps_refuse_budgets_below_their_needs(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    early, late = copied.early, copied.late
    n_early, n_late = int(EARLY.sum()), int(LATE.sum())
    # Checking that no cell is both early and late, before anything else:
    # 40 bytes per cell of the two selections, beside the clones read.
    checked = 40 * (n_early + n_late) + CODES_READ
    assert checked == 2_496
    monkeypatch.setattr(store, "memoryBytes", checked - 1)
    with pytest.raises(MemoryError, match=f"late cells holds about {checked} bytes"):
        store.clones.fate_bias(clones, group_by="cell_type", early=early, late=late)
    # Matching the clones' 48 cells to the late cells comes next: 56 bytes per
    # clone cell, 32 per late cell, and the read before, the largest stored
    # chunk of the two selections and the clones' codes, with Zarr's objects.
    # A selection stores a flag for each cell of the store, so its chunk
    # outweighs this store's cells.
    fates = _own_labels(store, LATE)
    options = {"groups": fates, "early": early, "late": late}
    before = max(
        array_row_selection_parts(artifact_group(store.zw, ref)[name])[0]
        for ref, name in (
            (late, "values"),
            (store.artifacts.inspect(clones).input_ref("cell_selection"), "values"),
            (clones, "codes"),
        )
    )
    matched = N_CELLS * 56 + n_late * 32 + before + READ_OBJECT_BYTES
    monkeypatch.setattr(store, "memoryBytes", matched - 1)
    with pytest.raises(
        MemoryError, match=f"48 cells to {n_late} measured cells holds about {matched}"
    ):
        store.clones.fate_bias(clones, **options)
    # Without the read before, a fate of its own for each late cell, named by
    # three characters, makes the measurement the largest step: per early cell
    # 64 bytes, 24 per fraction, 40 to write and 4 per character of a fate's
    # name; per clone with late cells and fate 80 bytes; 1,000 per fate; and
    # 24 per count of a clone in a fate, here one per late cell with a clone.
    monkeypatch.setattr(clone_operations, "previous_read_bytes", lambda *chunks: 0)
    tested = len({int(c) for c in TRUTH.clones[LATE] if c >= 0})
    with_clone = int(np.count_nonzero(TRUTH.clones[LATE] >= 0))
    measured = (
        n_early * (64 + 24 * n_late + 40 + 4 * 3)
        + 80 * tested * n_late
        + 1_000 * n_late
        + 24 * with_clone
        + CODES_READ
    )
    assert (tested, with_clone, measured) == (8, 19, 61_356)
    monkeypatch.setattr(store, "memoryBytes", measured - 1)
    with pytest.raises(
        MemoryError, match=f"8 clones in {n_late} fates holds about {measured}"
    ):
        store.clones.fate_bias(clones, **options)
    assert store.artifacts.list(kind="clone_fate_bias", scope="datastore") == []
    monkeypatch.setattr(store, "memoryBytes", measured)
    saved = store.clones.fate_bias(clones, **options)
    # Loading: per early cell 200 bytes and 24 per fraction, per row of clone
    # and fate 160 bytes and 16 per key column, 2,000 per fate, the clones
    # read with their labels, eight of one stored character, and the early
    # cells' IDs read as text.
    loaded = (
        n_early * (200 + 24 * n_late)
        + tested * n_late * (160 + 16)
        + 2_000 * n_late
        + CODES_READ
        + 8 * (100 + 3 * 4)
    )
    assert loaded == 111_296
    loaded += ids_read_bytes(store, n_early)
    monkeypatch.setattr(store, "memoryBytes", loaded - 1)
    with pytest.raises(
        MemoryError, match=f"fate bias of {n_early} cells holds about {loaded}"
    ):
        store.clones.load_fate_bias(saved)
    monkeypatch.setattr(store, "memoryBytes", loaded)
    assert store.clones.load_fate_bias(saved).cells.shape == (n_early, 3 + n_late)


def test_the_artifact_stores_its_arrays_by_early_cell_clone_and_fate(shared) -> None:
    store, clones = shared.store, shared.clones
    bias = store.clones.fate_bias(
        clones, group_by="cell_type", early=shared.early, late=shared.late
    )
    group = artifact_group(store.zw, bias)
    n_early = int(EARLY.sum())
    tested = len({int(c) for c in TRUTH.clones[LATE] if c >= 0})
    shapes = {name: group[name].shape for name in group.array_keys()}
    assert {name: shapes[name] for name in ("codes", "fate_codes", "late_cells")} == {
        "codes": (n_early,),
        "fate_codes": (n_early,),
        "late_cells": (n_early,),
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
