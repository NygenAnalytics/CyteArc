"""Clones called from the combinations of mutant calls that cells carry.

Hand-written GoT tables fix each cell's calls at two targets; the expected
clones follow from the rule written out: cells link where they agree at
every target both were called at, and a cell of its own that agrees with two
clones that disagree is their doublet.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import every_cell, smallest_budget, traced_peak
from tests.variant_helpers import copy_variant_store, write_variant_store
from tests.variant_truth import variant_truth

TRUTH = variant_truth()
IDS = [str(cell) for cell in TRUTH.clones.cell_ids]
# Each cell's wild-type and mutant UMIs at S1 and S2, None where not read:
# A (cells 0, 1) is mutant at S1 alone, B (2, 3) at both, X (4) at S1 with S2
# unread, W (7) at neither, and Z (8) at S2 with S1 unread.
CALLS = {
    0: ((0, 2), (3, 0)),
    1: ((1, 1), (2, 0)),
    2: ((0, 3), (1, 2)),
    3: ((2, 1), (0, 1)),
    4: ((0, 2), None),
    7: ((4, 0), (3, 0)),
    8: (None, (0, 2)),
}


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("genotype_clones") / "store.zarr"
    write_variant_store(path, TRUTH)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_variant_store(template, tmp_path)


def _genotypes(store, folder: Path, calls=CALLS):
    paths = []
    for target, name in enumerate(("S1", "S2")):
        lines = ["BC\twtUMIs\tmutUMIs"] + [
            f"{IDS[cell]}\t{umis[target][0]}\t{umis[target][1]}"
            for cell, umis in calls.items()
            if umis[target] is not None
        ]
        path = folder / f"{name}.tsv"
        path.write_text("\n".join(lines) + "\n")
        paths.append(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(paths, cell_selection=every_cell(store))
    return store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)


def _clones(store, clones) -> dict[str, list[int]]:
    cells = store.clones.load_clones(clones).cells
    found: dict[str, list[int]] = {}
    for position, cell in enumerate(IDS):
        label = cells.loc[cell, "clone"]
        if isinstance(label, str):
            found.setdefault(label, []).append(position)
    return found


def test_cells_link_where_they_agree_and_doublets_are_removed(store, tmp_path) -> None:
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path))
    # Z, unread at S1, joins B, with which it agrees at S2; X agrees with A
    # and B, which disagree at S2, so it is their doublet.
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    cells = store.clones.load_clones(clones).cells
    doublets = [IDS.index(cell) for cell in cells.index[cells["doublet"]]]
    assert doublets == [4]
    attributes = dict(artifact_group(store.zw, clones).attrs)
    assert attributes["n_doublets"] == 1
    assert attributes["n_cells_mutant"] == 6
    assert attributes["n_sites"] == 2
    record = store.artifacts.inspect(clones)
    assert record.operation == "genotype_clones"
    assert record.parameters["similarity"] == 1.0
    assert record.parameters["inclusive"] is True


def test_an_unread_site_is_unknown_rather_than_wild_type(store, tmp_path) -> None:
    # Without X, nothing joins A and B; Z still joins B, which it could not
    # if its unread S1 counted as wild type, where B is mutant.
    calls = {cell: umis for cell, umis in CALLS.items() if cell != 4}
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    # Read as wild type at S1, Z disagrees with B and is a clone of its own.
    calls[8] = ((3, 0), (0, 2))
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
    assert _clones(store, clones) == {
        "clone_1": [0, 1],
        "clone_2": [2, 3],
        "clone_3": [8],
    }


def test_clones_of_chosen_sites_and_within_groups(store, tmp_path) -> None:
    genotypes = _genotypes(store, tmp_path)
    # At S1 alone, A, B and X carry one combination; Z carries none there.
    alone = store.variants.genotype_clones(genotypes, sites=["S1"])
    assert _clones(store, alone) == {"clone_1": [0, 1, 2, 3, 4]}
    donors = np.where(np.arange(len(IDS)) == 4, "d2", "d1").astype(object)
    store.cells.insert("donor", donors)
    within = store.variants.genotype_clones(genotypes, within="donor")
    # X's donor holds no other cell, so it joins none and is no doublet.
    assert _clones(store, within) == {
        "clone_1": [2, 3, 8],
        "clone_2": [0, 1],
        "clone_3": [4],
    }
    assert not store.clones.load_clones(within).cells["doublet"].any()
    assert within != store.variants.genotype_clones(genotypes)


def test_saved_genotype_clones_are_reused_and_called_again(store, tmp_path) -> None:
    genotypes = _genotypes(store, tmp_path)
    first = store.variants.genotype_clones(genotypes)
    assert store.variants.genotype_clones(genotypes) == first
    again = store.variants.genotype_clones(genotypes, invalidate_cache=True)
    assert _clones(store, again) == _clones(store, first)
    del artifact_group(store.zw, first)["doublet"]
    assert store.variants.genotype_clones(genotypes) != first


def test_genotype_clones_refuse_other_artifacts_and_arguments(store, tmp_path) -> None:
    genotypes = _genotypes(store, tmp_path)
    with pytest.raises(ValueError, match="variant_genotypes artifact"):
        store.variants.genotype_clones(every_cell(store))
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.variants.genotype_clones("genotypes")
    with pytest.raises(ValueError, match="no site 'S3'"):
        store.variants.genotype_clones(genotypes, sites=["S3"])
    with pytest.raises(ValueError, match="sites names no site"):
        store.variants.genotype_clones(genotypes, sites=[])
    with pytest.raises(ValueError, match="also a column of the result"):
        store.variants.genotype_clones(genotypes, within="doublet")
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.variants.genotype_clones(genotypes, within="mouse")
    store.cells.insert(
        "half", np.where(np.arange(len(IDS)) == 2, "", "a").astype(object)
    )
    with pytest.raises(ValueError, match="no label for 1 cells with a mutant call"):
        store.variants.genotype_clones(genotypes, within="half")


def test_genotype_clones_refuse_a_budget_too_small_before_writing(
    store, tmp_path
) -> None:
    genotypes = _genotypes(store, tmp_path)
    before = len(store.artifacts.list(kind="clones", scope="datastore"))
    store.memoryBytes = 500
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.variants.genotype_clones(genotypes)
    assert len(store.artifacts.list(kind="clones", scope="datastore")) == before


@pytest.mark.slow
def test_genotype_clones_hold_no_more_than_the_smallest_budget_they_accept(
    store, tmp_path
) -> None:
    genotypes = _genotypes(store, tmp_path)

    def call(budget: int):
        store.memoryBytes = budget
        try:
            return store.variants.genotype_clones(genotypes, invalidate_cache=True)
        finally:
            store.memoryBytes = 1 << 30

    budget = smallest_budget(call, 1 << 28)
    store.memoryBytes = budget
    peak = traced_peak(
        lambda: store.variants.genotype_clones(genotypes, invalidate_cache=True)
    )
    assert peak <= budget
