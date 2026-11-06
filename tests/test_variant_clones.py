"""Clones called from the allele frequencies of informative variants.

The synthetic truth fixes which cells share an ancestor; the calls must
recover it, and every summary reads the result as any other clones.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from sklearn.metrics import adjusted_rand_score

from tests.clone_helpers import every_cell
from tests.storage_helpers import copy_store, open_store
from tests.variant_helpers import (
    clone_band_budget,
    write_variant_store,
)
from tests.variant_truth import mito_clone_truth, write_mgatk

TRUTH = mito_clone_truth()


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("variant_clones")
    path = folder / "store.zarr"
    store = write_variant_store(path, TRUTH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            write_mgatk(TRUTH, folder / "final"), cell_selection=every_cell(store)
        )
    store.variants.informative(counts, preset="atac")
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


def _inputs(store):
    counts = store.artifacts.list(kind="variant_counts", scope="datastore")[0]
    selection = store.artifacts.list(kind="variant_selection", scope="datastore")[0]
    return counts, selection


def _called(store, clones) -> np.ndarray:
    # Loading reads the cells' IDs, which decodes a stored chunk of them, so
    # the clones are read within a roomy budget whatever budget called them.
    budget = store.memoryBytes
    store.memoryBytes = max(budget, 1 << 30)
    try:
        cells = store.clones.load_clones(clones).cells
    finally:
        store.memoryBytes = budget
    labels = cells.loc[TRUTH.clones.cell_ids, "clone"].astype(object)
    return labels.where(labels.notna(), "").astype(str).to_numpy()


def test_clones_recover_the_truths_clones(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.frequency_clones(selection)
    called = _called(store, clones)
    labelled = TRUTH.clones.clones >= 0
    assert adjusted_rand_score(TRUTH.clones.clones[labelled], called[labelled]) > 0.9
    # Each true clone is held mostly by one called clone of its own.
    dominant = {}
    for clone in range(TRUTH.clones.n_clones):
        values, sizes = np.unique(
            called[TRUTH.clones.clones == clone], return_counts=True
        )
        dominant[clone] = values[np.argmax(sizes)]
        assert sizes.max() >= 0.9 * (TRUTH.clones.clones == clone).sum()
    assert len(set(dominant.values())) == TRUTH.clones.n_clones
    loaded = store.clones.load_clones(clones)
    assert loaded.source == "call_frequency_clones"
    # Called clones are numbered by size, as every clone that CyteArc calls.
    assert loaded.clones["n_cells"].is_monotonic_decreasing
    assert loaded.clones["clone"].tolist() == [
        f"clone_{rank}" for rank in range(1, len(loaded.clones) + 1)
    ]


def test_cells_without_evidence_get_no_clone_and_are_counted(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.frequency_clones(selection)
    diagnostics = store.artifacts.inspect(clones).diagnostics
    called = _called(store, clones)
    assert diagnostics["cellsScored"] == TRUTH.n_cells
    assert diagnostics["variants"] == 12
    assert (
        diagnostics["cellsWithClone"]
        + diagnostics["cellsWithoutCoverage"]
        + diagnostics["cellsWithoutAlternative"]
        + diagnostics["cellsWithoutNeighbors"]
        == TRUTH.n_cells
    )
    # Cells without a label in the within columns are refused, so no cell is
    # left without a group.
    assert "cellsWithoutGroup" not in diagnostics
    assert diagnostics["cellsWithClone"] == int((called != "").sum())
    # A cell without a clone of its own reads no variant allele, or reads
    # one by error and joins the clone that carries it.
    loaded = store.variants.load_counts(counts)
    variants = {
        (f"chrM:{position}", alternative)
        for position, _reference, alternative in TRUTH.variant_sites()
    }
    shown = {
        str(cell)
        for cell, site, allele in zip(
            loaded.counts["ids"],
            loaded.counts["site"],
            loaded.counts["allele"],
            strict=True,
        )
        if (str(site), str(allele)) in variants
    }
    # A cell read at no informative variant's site has no coverage, and one
    # read there that shows none of them no alternative, counted from the
    # loaded depth and counts.
    sites = {site for site, _alternative in variants}
    covered = {
        str(cell)
        for cell, site, depth in zip(
            loaded.depth["ids"],
            loaded.depth["site"],
            loaded.depth["depth"],
            strict=True,
        )
        if str(site) in sites and depth > 0
    }
    scored = {str(cell) for cell in TRUTH.clones.cell_ids}
    assert diagnostics["cellsWithoutCoverage"] == len(scored - covered)
    assert diagnostics["cellsWithoutAlternative"] == len(covered - shown)
    assert diagnostics["cellsWithoutCoverage"] != diagnostics["cellsWithoutAlternative"]
    # Every cell that shows an informative variant shares it with a clone,
    # so none lacks neighbours, and a cell without a clone shows none.
    assert diagnostics["cellsWithoutNeighbors"] == 0
    for cell, label in zip(TRUTH.clones.cell_ids, called, strict=True):
        if label == "":
            assert str(cell) not in shown
    assert store.artifacts.load(clones).attrs["n_clones"] == len(set(called) - {""})


def test_within_keeps_groups_apart_and_numbers_clones_over_every_group(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.frequency_clones(selection, within="sample")
    loaded = store.clones.load_clones(clones)
    labels = loaded.cells["clone"].astype(object).dropna()
    samples = dict(zip(TRUTH.clones.cell_ids, TRUTH.clones.samples, strict=True))
    # Each clone holds the cells of one sample.
    for label in set(labels):
        assert len({samples[cell] for cell in labels.index[labels == label]}) == 1
    # One ranking over both samples, largest first, labelled clone_1 and on.
    sizes = labels.value_counts()
    ranked = [f"clone_{rank}" for rank in range(1, len(sizes) + 1)]
    assert loaded.clones["clone"].tolist() == ranked
    assert [int(sizes[label]) for label in ranked] == sorted(sizes, reverse=True)
    # The truth's clones span both samples, so each splits in two.
    called = _called(store, clones)
    labelled = TRUTH.clones.clones >= 0
    split = np.asarray(
        [
            f"{sample}:{clone}"
            for sample, clone in zip(
                TRUTH.clones.samples[labelled],
                TRUTH.clones.clones[labelled],
                strict=True,
            )
        ]
    )
    assert adjusted_rand_score(split, called[labelled]) > 0.8


def test_two_within_columns_keep_every_combination_apart(store) -> None:
    counts, selection = _inputs(store)
    # A second column across the samples: each clone holds the cells of one
    # sample and one half, as a single column of both labels would group them.
    halves = np.where(np.arange(TRUTH.n_cells) % 3 == 0, "x", "y").astype(object)
    store.cells.insert("half", halves)
    store.cells.insert(
        "sample_half",
        np.asarray(
            [
                f"{sample}-{half}"
                for sample, half in zip(TRUTH.clones.samples, halves, strict=True)
            ],
            dtype=object,
        ),
    )
    clones = store.variants.frequency_clones(selection, within=["sample", "half"])
    called = _called(store, clones)
    combined = TRUTH.clones.samples.astype(object) + "-" + halves
    for label in set(called) - {""}:
        assert len(set(combined[called == label])) == 1, label
    # The same groups through one column give the same clones and labels.
    single = store.variants.frequency_clones(selection, within="sample_half")
    np.testing.assert_array_equal(_called(store, single), called)
    assert single != clones
    assert store.artifacts.inspect(clones).parameters["within"] == ["sample", "half"]
    # Either column's labels enter the identity.
    store.cells.insert("half", halves[::-1].copy(), overwrite=True)
    assert (
        store.variants.frequency_clones(selection, within=["sample", "half"]) != clones
    )


def test_groups_smaller_than_a_neighbourhood_are_warned_of(store) -> None:
    from cytearc.utils.logging import collect_warnings

    _counts, selection = _inputs(store)
    # The first four cells in a group of their own, and the others in one.
    store.cells.insert(
        "tiny", np.where(np.arange(TRUTH.n_cells) < 4, "few", "many").astype(object)
    )
    expected = (
        "Of the 2 within groups, 1 holds fewer scored cells than n_neighbors=10, "
        "so no neighbourhood there is full: each cell takes every cell of its "
        "group that it resembles as a neighbour, and clones that share a "
        "variant join. Pass a smaller n_neighbors."
    )
    with collect_warnings() as messages:
        clones = store.variants.frequency_clones(selection, within="tiny")
    assert messages == [expected]
    # A reused call warns alike.
    with collect_warnings() as messages:
        assert store.variants.frequency_clones(selection, within="tiny") == clones
    assert messages == [expected]
    # Neighbourhoods of four fit every group, and the samples of over 100
    # cells hold ten.
    for options in ({"within": "tiny", "n_neighbors": 4}, {"within": "sample"}, {}):
        with collect_warnings() as messages:
            store.variants.frequency_clones(selection, **options)
        assert messages == [], options
    # Without within, all the scored cells are one group.
    with collect_warnings() as messages:
        store.variants.frequency_clones(selection, n_neighbors=TRUTH.n_cells + 1)
    assert messages == [
        f"The {TRUTH.n_cells} scored cells are fewer than "
        f"n_neighbors={TRUTH.n_cells + 1}, so no neighbourhood there is full: "
        "each cell takes every cell of its group that it resembles as a "
        "neighbour, and clones that share a variant join. Pass a smaller "
        "n_neighbors."
    ]
    # Statistics of 120 of the counts' cells score only those, which are
    # fewer than 121, though the counts hold 230.
    counts, _selection = _inputs(store)
    store.cells.insert("first_120", np.arange(TRUTH.n_cells) < 120)
    scored = store.variants.informative(
        counts,
        cell_selection=store.snapshot_cell_selection("first_120"),
        preset="atac",
    )
    with collect_warnings() as messages:
        store.variants.frequency_clones(scored, n_neighbors=121)
    assert messages == [
        "The 120 scored cells are fewer than n_neighbors=121, so no "
        "neighbourhood there is full: each cell takes every cell of its group "
        "that it resembles as a neighbour, and clones that share a variant "
        "join. Pass a smaller n_neighbors."
    ]


def test_tied_clones_are_numbered_by_their_first_cell() -> None:
    from cytearc.datastore._operations.variants import _clone_assignment
    from cytearc.variants.frequencies import FrequencyClones

    # Ten scored cells among 13, two groups numbering their own clusters. The
    # clone of group 1 and cluster 2 holds three cells, and three clones tie
    # at two cells: group 1's cluster 1 starts at cell 0, group 0's cluster 1
    # at cell 2 and its cluster 2 at cell 3, an order that differs from the
    # groups' and clusters'. Cell 12 has no clone.
    positions = np.asarray([0, 2, 3, 5, 6, 8, 9, 10, 11, 12])
    called = FrequencyClones(
        clusters=np.asarray([1, 1, 2, 1, 2, 2, 1, 2, 2, 0]),
        groups=np.asarray([1, 0, 0, 1, 0, 1, 0, 1, 1, 0]),
        status=np.zeros(10, dtype=np.int8),
        n_edges=0,
    )
    assignment = _clone_assignment(
        13, positions, groups=called.groups, clusters=called.clusters
    )
    assert assignment.clone_ids.tolist() == ["clone_1", "clone_2", "clone_3", "clone_4"]
    assert assignment.clone_sizes.tolist() == [3, 2, 2, 2]
    assert assignment.codes.tolist() == [1, -1, 2, 3, -1, 1, 3, -1, 0, 2, 0, 0, -1]
    # No clone is called: no label.
    empty = _clone_assignment(
        3,
        np.asarray([0, 2]),
        groups=np.zeros(2, dtype=np.int64),
        clusters=np.zeros(2, dtype=np.int64),
    )
    assert empty.clone_ids.tolist() == [] and empty.codes.tolist() == [-1, -1, -1]


def test_scored_cells_without_a_label_are_refused(store) -> None:
    counts, selection = _inputs(store)
    groups = TRUTH.clones.samples.astype(object).copy()
    groups[:10] = None
    store.cells.insert("donor", groups)
    with pytest.raises(
        ValueError,
        match="no label for 10 cells with counts that the statistics scored",
    ):
        store.variants.frequency_clones(selection, within="donor")
    # A cell outside the selection's statistics needs no label: the
    # statistics of the labelled cells alone call clones within the donors.
    store.cells.insert("labelled", ~pd.isna(groups))
    labelled = store.variants.informative(
        counts, cell_selection=store.snapshot_cell_selection("labelled"), preset="atac"
    )
    called = _called(store, store.variants.frequency_clones(labelled, within="donor"))
    assert (called[:10] == "").all()
    assert (called[10:] != "").any()


def test_clones_are_reused_and_follow_their_arguments(store) -> None:
    counts, selection = _inputs(store)
    first = store.variants.frequency_clones(selection)
    assert store.variants.frequency_clones(selection) == first
    finer = store.variants.frequency_clones(selection, resolution=3.0)
    assert finer != first
    assert len(store.clones.load_clones(finer).clones) >= len(
        store.clones.load_clones(first).clones
    )
    again = store.variants.frequency_clones(selection, invalidate_cache=True)
    assert again != first
    np.testing.assert_array_equal(_called(store, again), _called(store, first))
    # Editing the within column gives new clones.
    store.cells.insert("donor", TRUTH.clones.samples.astype(object))
    grouped = store.variants.frequency_clones(selection, within="donor")
    store.cells.insert(
        "donor",
        np.where(TRUTH.clones.samples == "d2", "a", "b").astype(object),
        overwrite=True,
    )
    assert store.variants.frequency_clones(selection, within="donor") != grouped


def test_clones_do_not_depend_on_threads_or_budget(store, tmp_path) -> None:
    counts, selection = _inputs(store)
    first = _called(store, store.variants.frequency_clones(selection))
    other = open_store(Path(store.zarr_loc), nthreads=4)
    # Bands of 2,000 rows, a few cells each. Zarr writes the clones one file
    # at a time, so the buffer of that write fits beside them.
    other.memoryBytes = clone_band_budget(
        other, counts, 2_000, n_cells=TRUTH.n_cells, n_variants=12
    )
    with zarr.config.set({"async.concurrency": 1}):
        second = _called(
            other,
            other.variants.frequency_clones(selection, invalidate_cache=True),
        )
    np.testing.assert_array_equal(first, second)


def test_clones_of_scattered_cells_do_not_depend_on_the_budget(store) -> None:
    counts, _selection = _inputs(store)
    in_d2 = TRUTH.clones.samples == "d2"
    store.cells.insert("in_d2", in_d2)
    selection = store.variants.informative(
        counts, cell_selection=store.snapshot_cell_selection("in_d2"), preset="atac"
    )
    whole = _called(store, store.variants.frequency_clones(selection))
    assert (whole[~in_d2] == "").all()
    labelled = in_d2 & (TRUTH.clones.clones >= 0)
    assert adjusted_rand_score(TRUTH.clones.clones[labelled], whole[labelled]) > 0.9
    # Bands of 5,000 rows, a few cells each, with cells of d4 between the
    # chosen ones; the rest of the budget holds the graph.
    group = store.artifacts.load(counts)
    rows = np.diff(group["depth_offsets"][:]) + np.diff(group["count_offsets"][:])
    first, last = np.flatnonzero(in_d2)[[0, -1]]
    assert rows[first : last + 1].sum() > 4 * 5_000
    store.memoryBytes = clone_band_budget(
        store, counts, 5_000, n_cells=int(in_d2.sum()), n_variants=12
    )
    banded = store.variants.frequency_clones(selection, invalidate_cache=True)
    np.testing.assert_array_equal(_called(store, banded), whole)


def test_a_read_only_store_reuses_saved_clones_and_refuses_new_ones(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.frequency_clones(selection)
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.variants.frequency_clones(selection) == clones
    with pytest.raises(PermissionError):
        reader.variants.frequency_clones(selection, n_neighbors=5)


def test_clones_check_their_arguments(store, tmp_path) -> None:
    counts, selection = _inputs(store)
    with pytest.raises(ValueError, match="at least 2"):
        store.variants.frequency_clones(selection, n_neighbors=1)
    with pytest.raises(TypeError, match="n_neighbors must be an integer"):
        store.variants.frequency_clones(selection, n_neighbors=2.5)
    with pytest.raises(ValueError, match="finite and positive"):
        store.variants.frequency_clones(selection, resolution=0)
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.variants.frequency_clones(selection, within="mouse")
    with pytest.raises(TypeError, match="within must be"):
        store.variants.frequency_clones(selection, within=3)
    with pytest.raises(TypeError, match="invalidate_cache"):
        store.variants.frequency_clones(selection, invalidate_cache="yes")
    with pytest.raises(ValueError, match="variant_selection artifact"):
        store.variants.frequency_clones(counts)
    nothing = store.variants.informative(counts, min_vmr=1e9)
    with pytest.raises(ValueError, match="selects no informative variant"):
        store.variants.frequency_clones(nothing)
    with pytest.raises(ValueError, match="within names a column twice"):
        store.variants.frequency_clones(selection, within=["sample", "sample"])


def test_frequencies_beyond_the_budget_are_refused(store) -> None:
    counts, selection = _inputs(store)
    store.memoryBytes = 2_000
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.variants.frequency_clones(selection)


def test_variant_clones_feed_the_clone_plots_and_values(store) -> None:
    import matplotlib

    matplotlib.use("Agg")
    counts, selection = _inputs(store)
    clones = store.variants.frequency_clones(selection)
    values = store.artifacts.load_values(clones)
    called = store.clones.load_clones(clones).cells["clone"]
    np.testing.assert_array_equal(
        values.values, called.astype(object).where(called.notna(), "").to_numpy()
    )
    result = store.plots.clone_abundance(clones, show=False)
    assert result.tables["top"]["n_cells"].sum() == int(called.notna().sum())
    result.close()


def test_variant_operations_name_inputs_of_another_kind_missing_or_incomplete(
    store,
) -> None:
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.refs import ArtifactRef

    counts, selection = _inputs(store)
    clones = store.variants.frequency_clones(selection)
    with pytest.raises(ValueError, match="counts must be a variant_counts artifact"):
        store.variants.informative(clones, preset="atac")
    with pytest.raises(ValueError, match="must be a variant_selection artifact"):
        store.variants.load_informative(counts)
    with pytest.raises(ValueError, match="must be a variant_selection artifact"):
        store.variants.frequency_clones(counts)

    def missing(ref: ArtifactRef) -> ArtifactRef:
        return ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)

    for call in (
        lambda: store.variants.informative(missing(counts), preset="atac"),
        lambda: store.variants.load_informative(missing(selection)),
        lambda: store.variants.frequency_clones(missing(selection)),
    ):
        with pytest.raises(KeyError, match="Artifact does not exist"):
            call()
    # An import or a scoring that stopped before it finished leaves an
    # incomplete artifact, which every reader of it refuses.
    readers = {
        counts: (
            lambda: store.variants.informative(
                counts, preset="atac", invalidate_cache=True
            ),
            lambda: store.variants.load_informative(selection),
            lambda: store.variants.frequency_clones(selection, invalidate_cache=True),
        ),
        selection: (
            lambda: store.variants.load_informative(selection),
            lambda: store.variants.frequency_clones(selection, invalidate_cache=True),
        ),
    }
    for ref, calls in readers.items():
        attributes = artifact_group(store.zw, ref).attrs
        attributes["complete"] = False
        try:
            for call in calls:
                with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                    call()
        finally:
            attributes["complete"] = True
    assert len(store.variants.load_informative(selection).variants) > 0
