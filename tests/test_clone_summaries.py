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
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import import_truth_clones, write_truth_store
from tests.clone_truth import N_CELLS, clone_truth
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


def test_summaries_refuse_a_grouping_that_leaves_out_every_clone(source) -> None:
    store = source.store
    for summary in (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
    ):
        with pytest.raises(ValueError, match="No cell has both a clone and a group"):
            summary(source.clones, group_by="unlabelled")
