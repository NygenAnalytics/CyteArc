"""Overlap and tracking of the truth's imported clones, by group of cells.

The store and its clones are written once, as a template, and shared by the
tests that only measure, since overlap and tracking save nothing; tests that
change cell metadata or the memory budget copy the template. Expected values
are counted cell by cell from the truth. The truth's clones keep their rank
as codes, so clone ``k`` is labelled ``str(k + 1)``.
"""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from cytearc.clones import OVERLAP_INDICES, CloneOverlap, CloneTracking
from cytearc.metadata.rows import array_row_selection_parts
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import (
    copy_clone_store,
    every_cell,
    import_truth_clones,
    overlap_reference,
    write_truth_store,
)
from tests.clone_truth import N_CELLS, clone_truth, number_label
from tests.storage_helpers import insert_nullable_cell_column

TRUTH = clone_truth()
SAMPLES = ("d2", "d4")
PAIR_COLUMNS = [
    "group_1",
    "group_2",
    *OVERLAP_INDICES,
    "shared_fraction_1",
    "shared_fraction_2",
]


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("clone_overlap")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    clones = import_truth_clones(store, TRUTH, folder)
    store.cells.insert("is_d2", TRUTH.samples == "d2")
    store.cells.insert("cell", np.asarray([f"c{at:02d}" for at in range(N_CELLS)]))
    return SimpleNamespace(path=folder / "store.zarr", clones=clones)


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    store = copy_clone_store(template.path, tmp_path_factory.mktemp("shared"))
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


@pytest.fixture
def copied(template, tmp_path) -> SimpleNamespace:
    store = copy_clone_store(template.path, tmp_path)
    return SimpleNamespace(store=store, clones=template.clones, cells=every_cell(store))


def _check_pairs(pairs, by_group: dict) -> None:
    """Every pair of groups, in the order of the groups, against the reference."""
    names = list(by_group)
    assert pairs.columns.tolist() == PAIR_COLUMNS
    assert pairs[["group_1", "group_2"]].values.tolist() == [
        [first, second] for first, second in combinations(names, 2)
    ]
    for row in pairs.itertuples(index=False):
        expected = overlap_reference(by_group[row.group_1], by_group[row.group_2])
        for column, value in expected.items():
            assert getattr(row, column) == pytest.approx(value, rel=1e-12)


def test_overlap_compares_the_clones_of_two_samples(shared) -> None:
    store, clones = shared.store, shared.clones
    saved = store.artifacts.list(scope="datastore")
    result = store.clones.overlap(clones, group_by="sample")
    assert isinstance(result, CloneOverlap)
    assert result.clones == clones and result.group_by == ("sample",)
    assert result.groups is None and result.cell_selection is None
    by_sample = TRUTH.group_clone_cells(TRUTH.samples)
    sizes = result.group_sizes
    assert sizes.columns.tolist() == ["group", "sample", "cells", "clones"]
    assert sizes.values.tolist() == [
        [sample, sample, sum(by_sample[sample].values()), len(by_sample[sample])]
        for sample in SAMPLES
    ]
    _check_pairs(result.pairs, {sample: by_sample[sample] for sample in SAMPLES})
    # By hand: d2 holds 13 cells of clones 1 to 7, with 3, 2, 2, 3, 1, 1 and 1
    # cells, and d4 19 cells of clones 1 to 8, with 6, 4, 3, 1, 2, 1, 1 and 1.
    assert {number_label(c): n for c, n in by_sample["d2"].items()} == {
        "1": 3,
        "2": 2,
        "3": 2,
        "4": 3,
        "5": 1,
        "6": 1,
        "7": 1,
    }
    row = result.pairs.iloc[0]
    assert row["shared"] == 7 and row["jaccard"] == pytest.approx(7 / 8)
    assert row["overlap"] == 1.0
    # 2 * 39 * 13 * 19 / (29 * 19**2 + 69 * 13**2), from sum(x y) = 39,
    # sum(x^2) = 29 and sum(y^2) = 69.
    assert row["morisita_horn"] == pytest.approx(19266 / 22130, rel=1e-14)
    assert row["abundance_cosine"] == pytest.approx(39 / np.sqrt(29 * 69))
    # Every cell of d2 is in a clone that d4 holds; d4's one cell of clone 8
    # is not.
    assert row["shared_fraction_1"] == 1.0
    assert row["shared_fraction_2"] == pytest.approx(18 / 19)
    # Nothing is saved.
    assert store.artifacts.list(scope="datastore") == saved


def test_overlap_groups_by_labels_columns_and_selections(shared) -> None:
    store, clones = shared.store, shared.clones
    labels = store.snapshot_cluster_labels("cell_type", cell_selection=shared.cells)
    by_type = store.clones.overlap(clones, groups=labels)
    assert by_type.groups == labels and by_type.group_by == ()
    assert by_type.group_sizes.columns.tolist() == [
        "group",
        "cluster_labels",
        "cells",
        "clones",
    ]
    found = TRUTH.group_clone_cells(TRUTH.cell_types)
    _check_pairs(by_type.pairs, {name: found[name] for name in sorted(found)})
    # Samples by cell types: each group is named by both labels.
    crossed = store.clones.overlap(clones, group_by=["sample", "cell_type"])
    keys = np.asarray(
        [f"{s}, {t}" for s, t in zip(TRUTH.samples, TRUTH.cell_types, strict=True)],
        dtype=object,
    )
    found = TRUTH.group_clone_cells(keys)
    assert crossed.group_sizes["group"].tolist() == sorted(found)
    assert len(crossed.pairs) == 15
    _check_pairs(crossed.pairs, {name: found[name] for name in sorted(found)})
    # Only the cells of d2, compared between cell types.
    d2 = store.snapshot_cell_selection("is_d2")
    selected = store.clones.overlap(clones, group_by="cell_type", cell_selection=d2)
    assert selected.cell_selection == d2
    found = TRUTH.group_clone_cells(TRUTH.cell_types, cells=TRUTH.samples == "d2")
    assert selected.group_sizes["cells"].sum() == 13
    _check_pairs(selected.pairs, {name: found[name] for name in sorted(found)})


def test_overlap_needs_two_groups(shared) -> None:
    store, clones = shared.store, shared.clones
    with pytest.raises(ValueError, match="at least two groups"):
        store.clones.overlap(clones)
    d2 = store.snapshot_cell_selection("is_d2")
    with pytest.raises(ValueError, match="at least two groups"):
        store.clones.overlap(clones, group_by="sample", cell_selection=d2)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"group_by": 3}, TypeError, "group_by must be"),
        ({"group_by": ["sample", "sample"]}, ValueError, "names a column twice"),
        ({"group_by": "absent"}, ValueError, "'absent', which is not a cell"),
        ({"groups": "clusters"}, TypeError, "groups must be an ArtifactRef"),
        ({"cell_selection": "t"}, TypeError, "cell_selection must be an ArtifactRef"),
    ],
)
def test_overlap_arguments_are_checked(shared, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        shared.store.clones.overlap(shared.clones, **kwargs)
    with pytest.raises(error, match=message):
        shared.store.clones.track(shared.clones, **kwargs)


def test_overlap_and_track_refuse_other_artifacts(shared) -> None:
    store = shared.store
    for method in (store.clones.overlap, store.clones.track):
        with pytest.raises(ValueError, match="clones must be a clones artifact, not"):
            method(shared.cells, group_by="sample")
        with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
            method("clones", group_by="sample")


def test_groups_whose_labels_join_to_one_name_are_refused(copied) -> None:
    store, clones = copied.store, copied.clones
    # ("a, b", "c") and ("a", "b, c") would both be named "a, b, c".
    store.cells.insert("first", np.asarray(["a, b", "a"] * (N_CELLS // 2)))
    store.cells.insert("second", np.asarray(["c", "b, c"] * (N_CELLS // 2)))
    with pytest.raises(ValueError, match="are both named 'a, b, c"):
        store.clones.overlap(clones, group_by=["first", "second"])
    with pytest.raises(ValueError, match="are both named 'a, b, c"):
        store.clones.track(clones, group_by=["first", "second"])


def test_group_columns_may_not_share_a_name_with_the_result(copied) -> None:
    store, clones = copied.store, copied.clones
    for column in ("cells", "clones", "fraction", "clone"):
        store.cells.insert(column, TRUTH.samples.astype(object))
    for column in ("cells", "clones"):
        with pytest.raises(ValueError, match=f"'{column}', which is also a column"):
            store.clones.overlap(clones, group_by=column)
    for column in ("cells", "fraction", "clone"):
        with pytest.raises(ValueError, match=f"'{column}', which is also a column"):
            store.clones.track(clones, group_by=column)
    # A clones artifact as labels names the key column "clones", which the
    # overlap's group sizes hold but a tracking table does not.
    with pytest.raises(ValueError, match="key column 'clones' is also a column"):
        store.clones.overlap(clones, groups=clones)
    assert store.clones.track(clones, groups=clones, top=1).tracked
    # Names that only the other result holds are free.
    assert len(store.clones.overlap(clones, group_by="fraction").pairs) == 1
    assert len(store.clones.track(clones, group_by="clones", top=1).cells) == 2


def test_cells_without_a_label_are_left_out(copied) -> None:
    store, clones = copied.store, copied.clones
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[np.flatnonzero(TRUTH.clones >= 0)[:2]] = True
    insert_nullable_cell_column(store, "partial", TRUTH.samples, missing)
    result = store.clones.overlap(clones, group_by="partial")
    assert result.group_sizes["cells"].sum() == 30
    found = TRUTH.group_clone_cells(TRUTH.samples, cells=~missing)
    _check_pairs(result.pairs, {sample: found[sample] for sample in SAMPLES})
    insert_nullable_cell_column(
        store, "batch", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    # No cell has a label, so no group holds a clone.
    for method in (store.clones.overlap, store.clones.track):
        with pytest.raises(ValueError, match="No cell has both a clone and a group"):
            method(clones, group_by="batch")


def _expected_rows(by_group: dict, chosen: list[int]) -> list[list]:
    """Each chosen clone's cells and share of cells in every group, zeros included."""
    rows = []
    for clone in chosen:
        for name, found in by_group.items():
            cells = found.get(clone, 0)
            rows.append([number_label(clone), name, cells, cells / sum(found.values())])
    return rows


def _largest(by_group: dict, top: int) -> list[int]:
    """Each group's ``top`` clones with the most cells, ties to the larger clone."""
    chosen: set[int] = set()
    for found in by_group.values():
        ranked = sorted(found.items(), key=lambda item: (-item[1], item[0]))
        chosen |= {clone for clone, _cells in ranked[:top]}
    return sorted(chosen)


@pytest.mark.parametrize("top", [1, 2, 3, 10])
def test_track_follows_the_largest_clones_of_each_group(shared, top) -> None:
    store, clones = shared.store, shared.clones
    saved = store.artifacts.list(scope="datastore")
    tracked = store.clones.track(clones, group_by="sample", top=top)
    assert isinstance(tracked, CloneTracking)
    assert tracked.clones == clones and tracked.group_by == ("sample",)
    assert tracked.groups is None and tracked.cell_selection is None
    by_sample = TRUTH.group_clone_cells(TRUTH.samples)
    chosen = _largest(by_sample, top)
    assert tracked.tracked == tuple(number_label(clone) for clone in chosen)
    cells = tracked.cells
    assert cells.columns.tolist() == ["clone", "group", "sample", "cells", "fraction"]
    assert (cells["group"] == cells["sample"]).all()
    expected = _expected_rows(by_sample, chosen)
    assert cells[["clone", "group", "cells"]].values.tolist() == [
        row[:3] for row in expected
    ]
    np.testing.assert_allclose(cells["fraction"], [row[3] for row in expected])
    assert store.artifacts.list(scope="datastore") == saved


def test_track_breaks_ties_by_the_larger_clone(shared) -> None:
    # d2's largest clones are 1 and 4, of 3 cells each, and d4's is 1, so
    # the top one of each sample is clone 1 alone.
    tracked = shared.store.clones.track(shared.clones, group_by="sample", top=1)
    assert tracked.tracked == ("1",)
    assert tracked.cells[["group", "cells"]].values.tolist() == [["d2", 3], ["d4", 6]]
    assert tracked.cells["fraction"].tolist() == pytest.approx([3 / 13, 6 / 19])
    two = shared.store.clones.track(shared.clones, group_by="sample", top=2)
    assert two.tracked == ("1", "2", "4")


def test_track_follows_labelled_clones_through_every_group(shared) -> None:
    store, clones = shared.store, shared.clones
    labels = store.snapshot_cluster_labels("cell_type", cell_selection=shared.cells)
    # Clone 8, of one d4 cell, and clone 3, named in any order and twice.
    tracked = store.clones.track(
        clones, group_by="sample", groups=labels, labels=["8", "3", "8"]
    )
    assert tracked.tracked == ("3", "8") and tracked.groups == labels
    keys = np.asarray(
        [f"{s}, {t}" for s, t in zip(TRUTH.samples, TRUTH.cell_types, strict=True)],
        dtype=object,
    )
    by_group = TRUTH.group_clone_cells(keys)
    by_group = {name: by_group[name] for name in sorted(by_group)}
    expected = _expected_rows(by_group, [2, 7])
    assert tracked.cells.columns.tolist() == [
        "clone",
        "group",
        "sample",
        "cluster_labels",
        "cells",
        "fraction",
    ]
    assert tracked.cells[["clone", "group", "cells"]].values.tolist() == [
        row[:3] for row in expected
    ]
    # A clone without cells among the selected cells is followed with none.
    d2 = store.snapshot_cell_selection("is_d2")
    alone = store.clones.track(
        clones, group_by="sample", labels=["8"], cell_selection=d2
    )
    assert alone.cell_selection == d2
    assert alone.cells[["clone", "group", "cells", "fraction"]].values.tolist() == [
        ["8", "d2", 0, 0.0]
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"top": 0}, ValueError, "top must be a positive integer"),
        ({"top": 1.5}, TypeError, "top must be a positive integer"),
        ({"top": True}, TypeError, "top must be a positive integer"),
        ({"labels": "1"}, TypeError, "labels must be a sequence"),
        ({"labels": [1]}, TypeError, "labels must be a sequence"),
        ({"labels": np.asarray([["1"]])}, TypeError, "labels must be a sequence"),
        ({"labels": np.asarray([1])}, TypeError, "labels must be a sequence"),
        ({"labels": []}, ValueError, "labels must name at least one clone"),
        ({"labels": ["9"]}, ValueError, "No clone is labelled"),
    ],
)
def test_track_arguments_are_checked(shared, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        shared.store.clones.track(shared.clones, group_by="sample", **kwargs)


def test_track_takes_labels_as_an_array(shared) -> None:
    store, clones = shared.store, shared.clones
    listed = store.clones.track(clones, group_by="sample", labels=["8", "3"])
    array = store.clones.track(clones, group_by="sample", labels=np.asarray(["8", "3"]))
    assert array.tracked == listed.tracked == ("3", "8")
    assert array.cells.values.tolist() == listed.cells.values.tolist()
    with pytest.raises(ValueError, match="labels must name at least one clone"):
        store.clones.track(clones, labels=np.asarray([], dtype=str))


def _own_labels(store):
    """A label artifact of a label of its own for each of the 32 cells with a clone."""
    store.cells.insert("has_clone", TRUTH.clones >= 0)
    own = store.snapshot_cluster_labels(
        "cell", cell_selection=store.snapshot_cell_selection("has_clone")
    )
    # Its labels are stored 3 characters wide.
    assert artifact_group(store.zw, own)["values"].dtype == np.dtype("<U3")
    return own


# Reading the clones holds each of the 48 cells' stored int32 code and its
# int64 copy, and tracking also reads the eight clones' labels, stored one
# character wide: 4 bytes three times and 100 more each.
READ = 48 * (4 + 8)
READ_WITH_LABELS = READ + 8 * (100 + 3 * 4)


def test_overlap_and_track_refuse_tables_beyond_the_budget(copied, monkeypatch) -> None:
    store, clones = copied.store, copied.clones
    own = _own_labels(store)
    values = artifact_group(store.zw, own)["values"]
    # Grouping the clones' 48 cells takes 48 * (100 + 70) bytes, 32 * (130 +
    # 3 * 12) for the labelled cells and one chunk of their labels, then 500
    # for each of the 32 distinct labels and 500 for each of the 32 groups.
    grouping = 48 * 170 + 32 * 166 + array_row_selection_parts(values)[0]
    grouping += 32 * 500 + 32 * 500
    assert grouping == 47_267
    # Each group holds one clone: their 496 pairs take 250 * 496 + 24 * 32**2
    # bytes, the 32 counts of a clone in a group 40 each, the groups 2,000
    # each, and the clones read are held beside them.
    pairs = 250 * 496 + 24 * 32**2 + 40 * 32 + 32 * 2_000 + READ
    assert pairs == 214_432
    monkeypatch.setattr(store, "memoryBytes", pairs - 1)
    with pytest.raises(
        MemoryError, match=f"gives 496 pairs and holds about {pairs} bytes"
    ):
        store.clones.overlap(clones, groups=own)
    monkeypatch.setattr(store, "memoryBytes", pairs)
    assert len(store.clones.overlap(clones, groups=own).pairs) == 496
    # The 8 clones that are each group's largest, followed through the 32
    # groups, give 256 rows of 100 bytes and 16 for their one key column, a
    # byte for each clone and group, and the counts, groups and clones read
    # as for overlap, with the clones' labels.
    rows = 256 * (100 + 16) + 8 * 32 + 40 * 32 + 32 * 2_000 + READ_WITH_LABELS
    assert rows == 96_704
    monkeypatch.setattr(store, "memoryBytes", rows - 1)
    with pytest.raises(
        MemoryError, match=f"gives 256 rows and holds about {rows} bytes.*smaller"
    ):
        store.clones.track(clones, groups=own, top=1)
    monkeypatch.setattr(store, "memoryBytes", rows)
    assert len(store.clones.track(clones, groups=own, top=1).cells) == 256
    # Grouping is checked first.
    monkeypatch.setattr(store, "memoryBytes", grouping - 1)
    for method in (store.clones.overlap, store.clones.track):
        with pytest.raises(
            MemoryError, match=f"Grouping 48 cells by 1 columns holds about {grouping}"
        ):
            method(clones, groups=own)


# Clones that keep labels apart, as clonotypes keep receptor types apart,
# label every cell of a clone alike: here clones 1 to 4 "A" and 5 to 8 "B".
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


def test_overlap_compares_only_groups_of_the_labels_kept_apart(kept_apart) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    result = store.clones.overlap(clones, group_by="sample")
    assert result.group_sizes.columns.tolist() == [
        "group",
        "sample",
        "lineage",
        "cells",
        "clones",
    ]
    assert result.group_sizes["group"].tolist() == [
        "d2, A",
        "d2, B",
        "d4, A",
        "d4, B",
    ]
    keys = np.asarray(
        [f"{s}, {k}" for s, k in zip(TRUTH.samples, LINEAGES, strict=True)],
        dtype=object,
    )
    found = TRUTH.group_clone_cells(keys)
    # Only groups of one lineage are compared, by first group and then second.
    pairs = result.pairs
    assert pairs[["group_1", "group_2"]].values.tolist() == [
        ["d2, A", "d4, A"],
        ["d2, B", "d4, B"],
    ]
    for row in pairs.itertuples(index=False):
        expected = overlap_reference(found[row.group_1], found[row.group_2])
        for column, value in expected.items():
            assert getattr(row, column) == pytest.approx(value, rel=1e-12)
    # The plot leaves the pairs of two lineages blank.
    plot = store.plots.clonal_overlap(clones, group_by="sample", show=False)
    matrix = plot.tables["matrix"]
    assert np.isnan(matrix.loc["d2, A", "d4, B"]) and np.isnan(
        matrix.loc["d2, A", "d2, B"]
    )
    assert matrix.loc["d2, B", "d4, B"] == pytest.approx(pairs.iloc[1]["morisita_horn"])
    assert len(plot.axes["matrix"].texts) == 4
    plot.close()


def test_track_follows_clones_through_the_groups_of_their_labels(kept_apart) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    tracked = store.clones.track(clones, group_by="sample", labels=["1", "8"])
    # Clone 1 is followed through the groups of lineage A and clone 8 through
    # those of B.
    assert tracked.cells[["clone", "group", "cells"]].values.tolist() == [
        ["1", "d2, A", 3],
        ["1", "d4, A", 6],
        ["8", "d2, B", 0],
        ["8", "d4, B", 1],
    ]
    d2_cells = TRUTH.group_clone_cells(LINEAGES, cells=TRUTH.samples == "d2")
    assert tracked.cells["fraction"].tolist() == pytest.approx(
        [3 / sum(d2_cells["A"].values()), 6 / 14, 0, 1 / 5]
    )
    # Clone 8 has no cell in d2, where it is followed through lineage B alone.
    d2 = store.snapshot_cell_selection("is_d2")
    alone = store.clones.track(
        clones, group_by="sample", labels=["8"], cell_selection=d2
    )
    assert alone.cells[["clone", "group", "cells"]].values.tolist() == [
        ["8", "d2, B", 0]
    ]
    # Without a group of its lineage, a clone has no group to follow.
    store.cells.insert("is_a_in_d2", (TRUTH.samples == "d2") & (LINEAGES == "A"))
    lineage_a = store.snapshot_cell_selection("is_a_in_d2")
    none = store.clones.track(
        clones, group_by="sample", labels=["8"], cell_selection=lineage_a
    )
    assert none.tracked == ("8",) and none.cells.empty
    assert none.cells.columns.tolist() == [
        "clone",
        "group",
        "sample",
        "lineage",
        "cells",
        "fraction",
    ]
    # The empty table keeps the types of a full one.
    full = store.clones.track(clones, group_by="sample", labels=["8"])
    assert none.cells.dtypes.tolist() == full.cells.dtypes.tolist()
    assert none.cells["cells"].dtype == np.int64
    assert none.cells["fraction"].dtype == np.float64
    # The plot places a sample's groups of both lineages at one position.
    plot = store.plots.clone_tracking(clones, group_by="sample", top=1, show=False)
    ax = plot.axes["tracking"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d2", "d4"]
    assert all(len(line.get_xdata()) == 2 for line in ax.get_lines())
    plot.close()


def test_the_budget_counts_only_the_pairs_and_rows_of_one_lineage(
    kept_apart, monkeypatch
) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    own = _own_labels(store)
    # The 24 cells of clones 1 to 4 are groups of lineage A and the 8 cells
    # of clones 5 to 8 groups of B: 276 + 28 pairs, the largest stratum of 24
    # groups, beside the counts, groups and clones read.
    pairs = 250 * (276 + 28) + 24 * 24**2 + 40 * 32 + 32 * 2_000 + READ
    assert pairs == 155_680
    monkeypatch.setattr(store, "memoryBytes", pairs - 1)
    with pytest.raises(
        MemoryError, match=f"gives 304 pairs and holds about {pairs} bytes"
    ):
        store.clones.overlap(clones, groups=own)
    monkeypatch.setattr(store, "memoryBytes", pairs)
    assert len(store.clones.overlap(clones, groups=own).pairs) == 304
    # Clones 1 to 4 are followed through the 24 groups of A and clones 5 to
    # 8 through the 8 of B, in rows with two key columns.
    rows = (4 * 24 + 4 * 8) * (100 + 2 * 16) + 8 * 32 + 40 * 32 + 32 * 2_000
    rows += READ_WITH_LABELS
    assert rows == 83_904
    monkeypatch.setattr(store, "memoryBytes", rows - 1)
    with pytest.raises(
        MemoryError, match=f"gives 128 rows and holds about {rows} bytes"
    ):
        store.clones.track(clones, groups=own, top=1)
    monkeypatch.setattr(store, "memoryBytes", rows)
    assert len(store.clones.track(clones, groups=own, top=1).cells) == 128


def test_overlap_needs_two_groups_of_one_lineage(kept_apart) -> None:
    store, clones = kept_apart.store, kept_apart.clones
    # Lineage A holds clones 1 to 4, of 3 cells of d2 and 6 of d4, and
    # lineage B clones 5 to 8; among the cells of lineage A in d2 and of B in
    # d4, each lineage is one group, and no pair can be compared.
    chosen = ((TRUTH.samples == "d2") & (LINEAGES == "A")) | (
        (TRUTH.samples == "d4") & (LINEAGES == "B")
    )
    store.cells.insert("one_lineage_each", chosen)
    selection = store.snapshot_cell_selection("one_lineage_each")
    with pytest.raises(
        ValueError, match=r"No two groups share their labels in \['lineage'\]"
    ):
        store.clones.overlap(clones, group_by="sample", cell_selection=selection)
