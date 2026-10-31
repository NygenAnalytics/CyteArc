"""Clones called from the allele frequencies of informative variants.

The synthetic truth fixes which cells share an ancestor; the calls must
recover it, and every summary reads the result as any other clones.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tests.clone_helpers import every_cell, open_clone_store
from tests.variant_helpers import (
    clone_band_budget,
    copy_variant_store,
    write_variant_store,
)
from tests.variant_truth import mito_clone_truth, variant_truth, write_mgatk

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
    return copy_variant_store(template, tmp_path)


def _inputs(store):
    counts = store.artifacts.list(kind="variant_counts", scope="datastore")[0]
    selection = store.artifacts.list(kind="variant_selection", scope="datastore")[0]
    return counts, selection


def _called(store, clones) -> np.ndarray:
    cells = store.clones.load_clones(clones).cells
    labels = cells.loc[TRUTH.clones.cell_ids, "clone"].astype(object)
    return labels.where(labels.notna(), "").astype(str).to_numpy()


def test_clones_recover_the_truths_clones(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.clones(counts, variants=selection)
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
    assert loaded.source == "variant_clones"
    assert loaded.clones["n_cells"].is_monotonic_decreasing


def test_cells_without_evidence_get_no_clone_and_are_counted(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.clones(counts, variants=selection)
    attributes = store.artifacts.load(clones).attrs
    called = _called(store, clones)
    assert attributes["n_cells_scored"] == TRUTH.n_cells
    assert attributes["n_variants"] == 12
    assert (
        attributes["n_cells_clustered"]
        + attributes["n_cells_without_coverage"]
        + attributes["n_cells_without_alternative"]
        + attributes["n_cells_without_neighbors"]
        + attributes["n_cells_without_group"]
        == TRUTH.n_cells
    )
    assert attributes["n_cells_clustered"] == int((called != "").sum())
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
            loaded.counts["cell"],
            loaded.counts["site"],
            loaded.counts["allele"],
            strict=True,
        )
        if (str(site), str(allele)) in variants
    }
    # Every cell that shows an informative variant shares it with a clone,
    # so none lacks neighbours, and a cell without a clone shows none.
    assert attributes["n_cells_without_neighbors"] == 0
    for cell, label in zip(TRUTH.clones.cell_ids, called, strict=True):
        if label == "":
            assert str(cell) not in shown
    assert attributes["n_clones"] == len(set(called) - {""})


def test_within_keeps_groups_apart_and_names_clones_after_them(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.clones(counts, variants=selection, within="sample")
    cells = store.clones.load_clones(clones).cells
    labels = cells["clone"].astype(object).dropna()
    samples = dict(zip(TRUTH.clones.cell_ids, TRUTH.clones.samples, strict=True))
    for cell, label in labels.items():
        group, _cluster = str(label).split(":")
        assert group == samples[cell]
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


def test_cells_without_a_group_get_no_clone(store) -> None:
    counts, selection = _inputs(store)
    groups = TRUTH.clones.samples.astype(object).copy()
    groups[:10] = None
    store.cells.insert("donor", groups)
    clones = store.variants.clones(counts, variants=selection, within="donor")
    called = _called(store, clones)
    assert (called[:10] == "").all()
    assert store.artifacts.load(clones).attrs["n_cells_without_group"] == 10


def test_clones_are_reused_and_follow_their_arguments(store) -> None:
    counts, selection = _inputs(store)
    first = store.variants.clones(counts, variants=selection)
    assert store.variants.clones(counts, variants=selection) == first
    finer = store.variants.clones(counts, variants=selection, resolution=3.0)
    assert finer != first
    assert len(store.clones.load_clones(finer).clones) >= len(
        store.clones.load_clones(first).clones
    )
    again = store.variants.clones(counts, variants=selection, invalidate_cache=True)
    assert again != first
    np.testing.assert_array_equal(_called(store, again), _called(store, first))
    # Editing the within column gives new clones.
    store.cells.insert("donor", TRUTH.clones.samples.astype(object))
    grouped = store.variants.clones(counts, variants=selection, within="donor")
    store.cells.insert(
        "donor",
        np.where(TRUTH.clones.samples == "d2", "a", "b").astype(object),
        overwrite=True,
    )
    assert store.variants.clones(counts, variants=selection, within="donor") != grouped


def test_clones_do_not_depend_on_threads_or_budget(store, tmp_path) -> None:
    counts, selection = _inputs(store)
    first = _called(store, store.variants.clones(counts, variants=selection))
    other = open_clone_store(Path(store.zarr_loc), nthreads=4)
    # Bands of 2,000 rows, a few cells each.
    other.memoryBytes = clone_band_budget(
        other, counts, 2_000, n_cells=TRUTH.n_cells, n_variants=12
    )
    second = _called(
        other,
        other.variants.clones(counts, variants=selection, invalidate_cache=True),
    )
    np.testing.assert_array_equal(first, second)


def test_clones_of_scattered_cells_do_not_depend_on_the_budget(store) -> None:
    counts, _selection = _inputs(store)
    in_d2 = TRUTH.clones.samples == "d2"
    store.cells.insert("in_d2", in_d2)
    selection = store.variants.informative(
        counts, cells=store.snapshot_cell_selection("in_d2"), preset="atac"
    )
    whole = _called(store, store.variants.clones(counts, variants=selection))
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
    banded = store.variants.clones(counts, variants=selection, invalidate_cache=True)
    np.testing.assert_array_equal(_called(store, banded), whole)


def test_a_read_only_store_reuses_saved_clones_and_refuses_new_ones(store) -> None:
    counts, selection = _inputs(store)
    clones = store.variants.clones(counts, variants=selection)
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.variants.clones(counts, variants=selection) == clones
    with pytest.raises(PermissionError):
        reader.variants.clones(counts, variants=selection, n_neighbors=5)


def test_clones_check_their_arguments(store, tmp_path) -> None:
    counts, selection = _inputs(store)
    with pytest.raises(ValueError, match="at least 2"):
        store.variants.clones(counts, variants=selection, n_neighbors=1)
    with pytest.raises(TypeError, match="n_neighbors must be an integer"):
        store.variants.clones(counts, variants=selection, n_neighbors=2.5)
    with pytest.raises(ValueError, match="finite and positive"):
        store.variants.clones(counts, variants=selection, resolution=0)
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.variants.clones(counts, variants=selection, within="mouse")
    with pytest.raises(TypeError, match="within must be"):
        store.variants.clones(counts, variants=selection, within=3)
    with pytest.raises(TypeError, match="invalidate_cache"):
        store.variants.clones(counts, variants=selection, invalidate_cache="yes")
    with pytest.raises(ValueError, match="variant_selection artifact"):
        store.variants.clones(counts, variants=counts)
    nothing = store.variants.informative(counts, min_vmr=1e9)
    with pytest.raises(ValueError, match="selects no informative variant"):
        store.variants.clones(counts, variants=nothing)
    # Statistics of other counts.
    small = variant_truth()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        other = store.variants.import_counts(
            write_mgatk(small, tmp_path / "other"),
            cell_selection=every_cell(store),
        )
    with pytest.raises(ValueError, match="scored other counts"):
        store.variants.clones(other, variants=selection)


def test_frequencies_beyond_the_budget_are_refused(store) -> None:
    counts, selection = _inputs(store)
    store.memoryBytes = 2_000
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.variants.clones(counts, variants=selection)


def test_variant_clones_feed_the_clone_plots_and_values(store) -> None:
    import matplotlib

    matplotlib.use("Agg")
    counts, selection = _inputs(store)
    clones = store.variants.clones(counts, variants=selection)
    values = store.artifacts.load_values(clones)
    called = store.clones.load_clones(clones).cells["clone"]
    np.testing.assert_array_equal(
        values.values, called.astype(object).where(called.notna(), "").to_numpy()
    )
    result = store.plots.clone_abundance(clones, show=False)
    assert result.tables["top"]["cells"].sum() == int(called.notna().sum())
    result.close()
