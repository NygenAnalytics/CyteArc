"""Every clone summary on every clone source.

Each source saves the synthetic truth's clones its own way, in a store of the
truth's cells, and every summary must measure the truth's clones whatever
saved them. The expected values are counted cell by cell from the truth, so
they never come from the code under test. A pull request that adds a clone
source adds a row to ``SOURCES``; one that adds a summary adds a test here.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    coupling_reference,
    import_truth_clones,
    overlap_reference,
    write_truth_store,
)
from tests.clone_truth import N_CELLS, clone_truth, number_label
from tests.storage_helpers import insert_nullable_cell_column

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()
SAMPLES = ("d2", "d4")


def _imported_clones(store: DataStore, folder: Path) -> ArtifactRef:
    return import_truth_clones(store, TRUTH, folder)


#: How each clone source saves the truth's clones, in the order the stack adds
#: them.
SOURCES: dict[str, Callable[[DataStore, Path], ArtifactRef]] = {
    "imported clones": _imported_clones,
}


@pytest.fixture(scope="module", params=list(SOURCES))
def source(request, tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp(request.param.replace(" ", "_"))
    store = write_truth_store(folder / "store.zarr", TRUTH)
    store.cells.insert("is_d2", TRUTH.samples == "d2")
    store.cells.insert("is_d4", TRUTH.samples == "d4")
    insert_nullable_cell_column(
        store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    return SimpleNamespace(store=store, clones=SOURCES[request.param](store, folder))


def _by_sample(cells: np.ndarray | None = None) -> dict[object, list[int]]:
    return TRUTH.group_clone_sizes(TRUTH.samples, cells=cells)


def test_clone_abundance_counts_the_clones_of_each_size(source) -> None:
    result = source.store.plots.clone_abundance(
        source.clones, group_by="sample", show=False
    )
    sizes = result.tables["sizes"]
    for sample in SAMPLES:
        rows = sizes[sizes["group"] == sample]
        found = _by_sample()[sample]
        assert dict(zip(rows["clone_size"], rows["n_clones"], strict=True)) == {
            size: found.count(size) for size in set(found)
        }
    plt.close(result.figure)


def test_expansion_counts_each_clone_within_each_sample(source) -> None:
    store = source.store
    loaded = store.clones.load_expansion(
        store.clones.expansion(source.clones, group_by="sample")
    )
    cells = loaded.cells.loc[TRUTH.cell_ids]
    np.testing.assert_array_equal(
        cells["group_clone_size"], TRUTH.cell_clone_sizes(TRUTH.samples)
    )
    totals = loaded.categories.groupby("sample")["cells"].sum().to_dict()
    assert totals == {sample: sum(_by_sample()[sample]) for sample in SAMPLES}
    clones = loaded.categories.groupby("sample")["clones"].sum().to_dict()
    assert clones == {sample: len(_by_sample()[sample]) for sample in SAMPLES}


def test_diversity_measures_the_clone_sizes_of_each_sample(source) -> None:
    result = source.store.clones.diversity(source.clones, group_by="sample")
    indices = result.indices.set_index("sample")
    for sample in SAMPLES:
        found = np.asarray(_by_sample()[sample])
        shares = found / found.sum()
        row = indices.loc[sample]
        assert row[["cells", "clones", "singletons", "doubletons"]].tolist() == [
            found.sum(),
            len(found),
            np.count_nonzero(found == 1),
            np.count_nonzero(found == 2),
        ]
        assert row["shannon"] == pytest.approx(-np.sum(shares * np.log(shares)))
        assert row["inverse_simpson"] == pytest.approx(1 / np.sum(shares**2))


def test_rarefaction_reaches_each_samples_observed_diversity(source) -> None:
    result = source.store.clones.rarefaction(
        source.clones, group_by="sample", orders=(0, 2)
    )
    observed = result.curves[result.curves["method"] == "observed"]
    for sample in SAMPLES:
        found = np.asarray(_by_sample()[sample])
        shares = found / found.sum()
        rows = observed[observed["sample"] == sample].set_index("order")
        assert rows.loc[0, "size"] == found.sum()
        assert rows.loc[0, "diversity"] == len(found)
        assert rows.loc[2, "diversity"] == pytest.approx(1 / np.sum(shares**2))


def test_overlap_compares_the_clones_that_samples_share(source) -> None:
    result = source.store.clones.overlap(source.clones, group_by="sample")
    by_sample = TRUTH.group_clone_cells(TRUTH.samples)
    assert result.group_sizes[["sample", "cells", "clones"]].values.tolist() == [
        [sample, sum(by_sample[sample].values()), len(by_sample[sample])]
        for sample in SAMPLES
    ]
    (row,) = result.pairs.to_dict("records")
    assert (row["group_1"], row["group_2"]) == SAMPLES
    for column, value in overlap_reference(by_sample["d2"], by_sample["d4"]).items():
        assert row[column] == pytest.approx(value, rel=1e-12)


def test_track_follows_each_samples_largest_clones(source) -> None:
    tracked = source.store.clones.track(source.clones, group_by="sample", top=2)
    by_sample = TRUTH.group_clone_cells(TRUTH.samples)
    # Each sample's two largest clones, ties to the larger clone overall.
    chosen: set[int] = set()
    for found in by_sample.values():
        ranked = sorted(found.items(), key=lambda item: (-item[1], item[0]))
        chosen |= {clone for clone, _cells in ranked[:2]}
    assert tracked.tracked == tuple(number_label(clone) for clone in sorted(chosen))
    assert tracked.cells[["clone", "sample", "cells"]].values.tolist() == [
        [number_label(clone), sample, by_sample[sample].get(clone, 0)]
        for clone in sorted(chosen)
        for sample in SAMPLES
    ]


def test_coupling_couples_the_cell_types_that_share_clones(source) -> None:
    result = source.store.clones.coupling(
        source.clones, group_by="cell_type", method="abundance_cosine"
    )
    by_type = TRUTH.group_clone_cells(TRUTH.cell_types)
    assert result.group_sizes["cells"].tolist() == [
        sum(by_type[name].values()) for name in sorted(by_type)
    ]
    for row in result.pairs.itertuples(index=False):
        assert row.coupling == pytest.approx(
            coupling_reference(by_type, row.group_1, row.group_2, "abundance_cosine"),
            rel=1e-12,
        )
    assert np.isfinite(result.pairs["z_score"]).all()
    assert sorted(result.order) == sorted(by_type)


def test_fate_bias_follows_early_cells_to_their_clones_late_cells(source) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(
            source.clones, group_by="cell_type", early=early, late=late
        )
    )
    in_d4 = TRUTH.group_clone_cells(TRUTH.cell_types, cells=TRUTH.samples == "d4")
    types = sorted(in_d4)
    for (cell_id, row), clone in zip(
        loaded.cells.iterrows(), TRUTH.clones[TRUTH.samples == "d2"], strict=True
    ):
        late_cells = [in_d4[name].get(int(clone), 0) for name in types]
        assert row["late_cells"] == (sum(late_cells) if clone >= 0 else 0), cell_id
        if clone >= 0 and sum(late_cells):
            assert row[types].tolist() == pytest.approx(
                [count / sum(late_cells) for count in late_cells]
            )
    tested = {int(clone) for clone in TRUTH.clones[TRUTH.samples == "d4"] if clone >= 0}
    assert set(loaded.clone_fates["clone"].astype(str)) == {
        number_label(clone) for clone in tested
    }


def test_summaries_keep_to_a_cell_selection(source) -> None:
    store = source.store
    d2 = TRUTH.samples == "d2"
    selection = store.snapshot_cell_selection("is_d2")
    found = _by_sample(cells=d2)["d2"]
    diversity = store.clones.diversity(source.clones, cell_selection=selection)
    assert diversity.indices[["cells", "clones"]].values.tolist() == [
        [sum(found), len(found)]
    ]
    expansion = store.clones.load_expansion(
        store.clones.expansion(source.clones, cell_selection=selection)
    )
    assert expansion.cells.index.tolist() == TRUTH.cell_ids[d2].tolist()
    np.testing.assert_array_equal(
        expansion.cells["group_clone_size"], TRUTH.cell_clone_sizes(cells=d2)[d2]
    )
    by_type = TRUTH.group_clone_cells(TRUTH.cell_types, cells=d2)
    overlap = store.clones.overlap(
        source.clones, group_by="cell_type", cell_selection=selection
    )
    assert overlap.group_sizes["cells"].tolist() == [
        sum(by_type[name].values()) for name in sorted(by_type)
    ]
    tracked = store.clones.track(source.clones, labels=["1"], cell_selection=selection)
    assert tracked.cells[["clone", "cells"]].values.tolist() == [
        ["1", TRUTH.group_clone_cells(cells=d2)[0][0]]
    ]
    coupling = store.clones.coupling(
        source.clones, group_by="cell_type", cell_selection=selection, permutations=0
    )
    assert coupling.group_sizes["cells"].tolist() == [
        sum(by_type[name].values()) for name in sorted(by_type)
    ]


def test_summaries_refuse_a_grouping_that_leaves_out_every_clone(source) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")

    def fate_bias(clones, **options):
        return store.clones.fate_bias(clones, early=early, late=late, **options)

    for summary in (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
        store.clones.overlap,
        store.clones.track,
        store.clones.coupling,
        fate_bias,
    ):
        with pytest.raises(ValueError, match="No cell has both a clone and a group"):
            summary(source.clones, group_by="unlabelled")


def test_summaries_name_missing_and_incomplete_clones(source) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")

    def fate_bias(clones):
        return store.clones.fate_bias(
            clones, group_by="cell_type", early=early, late=late
        )

    summaries = (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
        lambda clones: store.clones.overlap(clones, group_by="sample"),
        store.clones.track,
        lambda clones: store.clones.coupling(
            clones, group_by="cell_type", permutations=0
        ),
        fate_bias,
    )
    # Each saved result, with the loader that reads it.
    saved = {
        store.clones.expansion(source.clones): store.clones.load_expansion,
        fate_bias(source.clones): store.clones.load_fate_bias,
    }
    # Artifacts that the store does not hold raise KeyError.
    for summary in summaries:
        with pytest.raises(KeyError, match="Artifact does not exist"):
            summary(_unsaved(source.clones))
    for result, load in saved.items():
        with pytest.raises(KeyError, match="Artifact does not exist"):
            load(_unsaved(result))
    # An import that stopped before it finished leaves incomplete clones,
    # which raise RuntimeError, also where a saved result reads them.
    loads_of_saved = tuple(
        lambda _ref, result=result, load=load: load(result)
        for result, load in saved.items()
    )
    for ref, readers in (
        (source.clones, (*summaries, *loads_of_saved)),
        *((result, (load,)) for result, load in saved.items()),
    ):
        attributes = artifact_group(store.zw, ref).attrs
        attributes["complete"] = False
        try:
            for read in readers:
                with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                    read(ref)
        finally:
            attributes["complete"] = True
    for result, load in saved.items():
        assert len(load(result).cells) > 0


def _unsaved(ref: ArtifactRef) -> ArtifactRef:
    """A reference of the same kind to an artifact that no store holds."""
    return ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)
